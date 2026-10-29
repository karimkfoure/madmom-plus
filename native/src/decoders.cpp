/* Native forward HMM and peak picking kernels.
 * The madmom project licence applies. See LICENSE and docs/native.rst.
 */
#include "madmom_decoders.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <vector>

namespace {
int hmm_forward(
    const uint32_t *states, const uint32_t *pointers,
    const double *probabilities, size_t num_states, size_t num_transitions,
    const uint32_t *observation_pointers, const double *densities,
    size_t num_frames, size_t num_densities, double *previous, double *output,
    uint32_t *output_states, bool checked) {
    if (!num_states || !pointers || !previous || !observation_pointers ||
        (num_transitions && (!states || !probabilities)) ||
        (num_frames && (!densities || (!output && !output_states) ||
                        !num_densities)) ||
        num_frames > std::numeric_limits<size_t>::max() / num_states ||
        (num_densities && num_frames >
         std::numeric_limits<size_t>::max() / num_densities) ||
        pointers[0] != 0 || pointers[num_states] != num_transitions) {
        return -1;
    }
    for (size_t state = 0; state < num_states; ++state) {
        if (pointers[state] > pointers[state + 1] ||
            (num_frames && observation_pointers[state] >= num_densities)) {
            return -1;
        }
    }
    for (size_t edge = 0; edge < num_transitions; ++edge) {
        if (states[edge] >= num_states) {
            return -1;
        }
    }
    std::vector<double> scratch(output || !num_frames ? 0 : num_states);
    for (size_t frame = 0; frame < num_frames; ++frame) {
        double *current = output ? output + frame * num_states : scratch.data();
        const double *observation = densities + frame * num_densities;
        double sum = 0.;
        for (size_t state = 0; state < num_states; ++state) {
            double value = 0.;
            for (size_t edge = pointers[state]; edge < pointers[state + 1];
                 ++edge) {
                value += previous[states[edge]] * probabilities[edge];
            }
            current[state] = value * observation[observation_pointers[state]];
            sum += current[state];
        }
        if (checked && (!(sum > 0.) || !std::isfinite(sum))) {
            return -2;
        }
        const double reciprocal = 1. / sum;
        for (size_t state = 0; state < num_states; ++state) {
            current[state] *= reciprocal;
            previous[state] = current[state];
        }
        if (output_states) {
            size_t best = 0;
            for (size_t state = 1; state < num_states; ++state) {
                // NumPy argmax selects the first NaN for an impossible path.
                if (current[state] > current[best] ||
                    (std::isnan(current[state]) &&
                     !std::isnan(current[best]))) {
                    best = state;
                }
            }
            output_states[frame] = static_cast<uint32_t>(best);
        }
    }
    return 0;
}

template <typename T>
ptrdiff_t peak_picking(const T *activations, size_t count, T threshold,
                      size_t pre_avg, size_t post_avg,
                      size_t pre_max, size_t post_max, size_t *indices) {
    if ((count && (!activations || !indices)) ||
        count > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) ||
        pre_avg >= std::numeric_limits<size_t>::max() - post_avg ||
        pre_max >= std::numeric_limits<size_t>::max() - post_max ||
        !std::isfinite(threshold)) {
        return -1;
    }
    if (!count) {
        return 0;
    }
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(activations[i])) {
            return -1;
        }
    }
    const size_t avg_length = pre_avg + post_avg + 1;
    std::vector<T> detections(count);
    double sum = 0.;
    if (avg_length > 1) {
        for (size_t i = 0; i < count && i <= post_avg; ++i) {
            sum += activations[i];
        }
    }
    for (size_t i = 0; i < count; ++i) {
        T average = 0;
        if (avg_length > 1) {
            average = static_cast<T>(sum / avg_length);
        }
        const T cutoff = static_cast<T>(average + threshold);
        detections[i] = activations[i] >= cutoff ? activations[i] : T(0);
        if (avg_length > 1 && i + 1 < count) {
            const double incoming = post_avg < count - i - 1 ?
                static_cast<double>(activations[i + post_avg + 1]) : 0.;
            const double outgoing = i >= pre_avg ?
                static_cast<double>(activations[i - pre_avg]) : 0.;
            sum += incoming - outgoing;
        }
    }

    ptrdiff_t found = 0;
    if (!pre_max && !post_max) {
        for (size_t i = 0; i < count; ++i) {
            if (detections[i] != 0) {
                indices[found++] = i;
            }
        }
        return found;
    }
    std::deque<size_t> maxima;
    size_t next = 0;
    for (size_t i = 0; i < count; ++i) {
        const size_t end = post_max < count - i ? i + post_max : count - 1;
        for (; next <= end; ++next) {
            while (!maxima.empty() &&
                   detections[maxima.back()] <= detections[next]) {
                maxima.pop_back();
            }
            maxima.push_back(next);
        }
        const size_t start = i > pre_max ? i - pre_max : 0;
        while (!maxima.empty() && maxima.front() < start) {
            maxima.pop_front();
        }
        T maximum = detections[maxima.front()];
        if (i < pre_max || post_max >= count - i) {
            maximum = std::max(T(0), maximum);
        }
        if (detections[i] != 0 && detections[i] == maximum) {
            indices[found++] = i;
        }
    }
    return found;
}
}  // namespace

extern "C" int mm_hmm_forward(
    const uint32_t *states, const uint32_t *pointers,
    const double *probabilities, size_t num_states, size_t num_transitions,
    const uint32_t *observation_pointers, const double *densities,
    size_t num_frames, size_t num_densities, double *previous, double *output) {
    try {
        return hmm_forward(states, pointers, probabilities, num_states,
                           num_transitions, observation_pointers, densities,
                           num_frames, num_densities, previous, output, nullptr,
                           false);
    } catch (...) {
        return -1;
    }
}

extern "C" int mm_hmm_forward_checked(
    const uint32_t *states, const uint32_t *pointers,
    const double *probabilities, size_t num_states, size_t num_transitions,
    const uint32_t *observation_pointers, const double *densities,
    size_t num_frames, size_t num_densities, double *previous, double *output) {
    try {
        return hmm_forward(states, pointers, probabilities, num_states,
                           num_transitions, observation_pointers, densities,
                           num_frames, num_densities, previous, output, nullptr,
                           true);
    } catch (...) {
        return -1;
    }
}

extern "C" int mm_hmm_forward_states(
    const uint32_t *states, const uint32_t *pointers,
    const double *probabilities, size_t num_states, size_t num_transitions,
    const uint32_t *observation_pointers, const double *densities,
    size_t num_frames, size_t num_densities, double *previous,
    uint32_t *output_states) {
    try {
        return hmm_forward(states, pointers, probabilities, num_states,
                           num_transitions, observation_pointers, densities,
                           num_frames, num_densities, previous, nullptr,
                           output_states, false);
    } catch (...) {
        return -1;
    }
}

extern "C" ptrdiff_t mm_peak_picking_f32(
    const float *activations, size_t count, float threshold,
    size_t pre_avg, size_t post_avg, size_t pre_max, size_t post_max,
    size_t *indices) {
    try {
        return peak_picking(activations, count, threshold, pre_avg, post_avg,
                            pre_max, post_max, indices);
    } catch (...) {
        return -1;
    }
}

extern "C" ptrdiff_t mm_peak_picking_f64(
    const double *activations, size_t count, double threshold,
    size_t pre_avg, size_t post_avg, size_t pre_max, size_t post_max,
    size_t *indices) {
    try {
        return peak_picking(activations, count, threshold, pre_avg, post_avg,
                            pre_max, post_max, indices);
    } catch (...) {
        return -1;
    }
}

extern "C" int mm_beat_event(double time, double max_bpm, int is_beat,
                              double *last_beat, double *tempo) {
    if (is_beat && time >= *last_beat + 60. / max_bpm) {
        *tempo = 60. / (time - *last_beat);
        *last_beat = time;
        return 1;
    }
    return 0;
}

extern "C" ptrdiff_t mm_beat_events(
    const uint8_t *beat_flags, size_t num_frames, int64_t first_frame,
    double fps, double max_bpm, double *last_beat, double *tempo,
    double *output) {
    if (!last_beat || !tempo || !(fps > 0) || !std::isfinite(fps) ||
        !(max_bpm > 0) || !std::isfinite(max_bpm) ||
        (num_frames && (!beat_flags || !output)) ||
        num_frames > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) ||
        (first_frame > 0 && num_frames > static_cast<size_t>(
            std::numeric_limits<int64_t>::max() - first_frame))) {
        return -1;
    }
    ptrdiff_t count = 0;
    for (size_t frame = 0; frame < num_frames; ++frame) {
        const double time = (static_cast<int64_t>(frame) + first_frame) / fps;
        if (mm_beat_event(time, max_bpm, beat_flags[frame], last_beat, tempo)) {
            output[count++] = time;
        }
    }
    return count;
}
