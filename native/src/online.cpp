/* Native causal frontend, online LSTM ensemble and forward DBN.
 * The madmom project licence applies. See LICENSE and docs/native.rst.
 */
#include "madmom_native.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif

namespace {
thread_local std::string last_error;
const int frame_size = 2048;
const int num_bands = 81;
const int hop_size = 441;
const int hidden_size = 25;
const double pi = 3.14159265358979323846264338327950288;

template <typename T>
std::vector<T> read_values(std::ifstream& stream, size_t count) {
    std::vector<T> values(count);
    stream.read(reinterpret_cast<char*>(values.data()), count * sizeof(T));
    if (!stream) {
        throw std::runtime_error("Truncated native model asset");
    }
    // Model assets use little-endian IEEE-754 values.
    const uint16_t endian = 1;
    if (*reinterpret_cast<const unsigned char*>(&endian) != 1) {
        for (size_t i = 0; i < values.size(); ++i) {
            unsigned char* first = reinterpret_cast<unsigned char*>(&values[i]);
            std::reverse(first, first + sizeof(T));
        }
    }
    return values;
}

template <typename T>
void require_finite(const std::vector<T>& values) {
    for (size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i])) {
            throw std::runtime_error("Non-finite native model value");
        }
    }
}

void check_input(const float* values, size_t count, const char* message) {
    if (!values && count) {
        throw std::runtime_error(message);
    }
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(values[i])) {
            throw std::runtime_error("Non-finite native input");
        }
    }
}

float dot(const float* a, const float* b, int count, bool simd) {
    int i = 0;
    float sum = 0;
    if (simd) {
#if defined(__aarch64__)
        float32x4_t acc = vdupq_n_f32(0);
        for (; i + 4 <= count; i += 4) {
            acc = vaddq_f32(acc, vmulq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
        }
        float lanes[4];
        vst1q_f32(lanes, acc);
        sum = ((lanes[0] + lanes[1]) + lanes[2]) + lanes[3];
#elif defined(__SSE2__)
        __m128 acc = _mm_setzero_ps();
        for (; i + 4 <= count; i += 4) {
            acc = _mm_add_ps(acc, _mm_mul_ps(_mm_loadu_ps(a + i),
                                          _mm_loadu_ps(b + i)));
        }
        float lanes[4];
        _mm_storeu_ps(lanes, acc);
        sum = ((lanes[0] + lanes[1]) + lanes[2]) + lanes[3];
#endif
    }
    for (; i < count; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

float sigmoid(float value) {
    return 1.f / (1.f + std::exp(-value));
}

float fp16_roundtrip(float value) {
    // Binary16 has ten fraction bits, a minimum normal exponent of -14,
    // and a minimum subnormal quantum of 2^-24. Round ties to even.
    const double magnitude = std::abs(static_cast<double>(value));
    if (magnitude == 0) {
        return value;
    }
    if (magnitude > 65504.) {
        throw std::runtime_error("Weight is outside the finite fp16 range");
    }
    int exponent;
    std::frexp(magnitude, &exponent);
    const double quantum = std::ldexp(1., std::max(exponent - 11, -24));
    const double scaled = magnitude / quantum;
    double rounded = std::floor(scaled);
    const double remainder = scaled - rounded;
    if (remainder > .5 || (remainder == .5 && std::fmod(rounded, 2.) != 0)) {
        rounded += 1;
    }
    return std::copysign(static_cast<float>(rounded * quantum), value);
}

void quantize(std::vector<float>& values, size_t columns, int mode) {
    require_finite(values);
    if (mode == 1) {
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = fp16_roundtrip(values[i]);
        }
    } else if (mode == 2) {
        // Symmetric per-row int8 quantization, with deterministic ties away
        // from zero. Decode once; inference retains float32 accumulation.
        for (size_t first = 0; first < values.size(); first += columns) {
            float maximum = 0;
            for (size_t j = first; j < first + columns; ++j) {
                maximum = std::max(maximum, std::abs(values[j]));
            }
            if (maximum == 0) {
                continue;
            }
            const double scale = static_cast<double>(maximum) / 127.;
            for (size_t j = first; j < first + columns; ++j) {
                values[j] = static_cast<float>(std::round(values[j] / scale) * scale);
            }
        }
    }
}

class FFT {
    std::array<size_t, frame_size> reversed;
    std::array<std::complex<double>, frame_size / 2> roots;
    std::array<std::complex<double>, frame_size> buffer;

public:
    FFT() {
        for (size_t i = 0; i < frame_size; ++i) {
            size_t value = i;
            size_t result = 0;
            for (int bit = 0; bit < 11; ++bit) {
                result = (result << 1) | (value & 1);
                value >>= 1;
            }
            reversed[i] = result;
        }
        for (size_t i = 0; i < roots.size(); ++i) {
            const double phase = -2 * pi * i / frame_size;
            roots[i] = std::complex<double>(std::cos(phase), std::sin(phase));
        }
    }

    void magnitudes(const float* frame, const std::vector<double>& window,
                    float* output) {
        for (size_t i = 0; i < frame_size; ++i) {
            buffer[reversed[i]] = std::complex<double>(frame[i] * window[i], 0);
        }
        for (size_t length = 2; length <= frame_size; length *= 2) {
            const size_t half = length / 2;
            const size_t stride = frame_size / length;
            for (size_t first = 0; first < frame_size; first += length) {
                for (size_t j = 0; j < half; ++j) {
                    const std::complex<double> even = buffer[first + j];
                    const std::complex<double> odd =
                        buffer[first + j + half] * roots[j * stride];
                    buffer[first + j] = even + odd;
                    buffer[first + j + half] = even - odd;
                }
            }
        }
        for (size_t i = 0; i < frame_size / 2; ++i) {
            // madmom casts the complex FFT to complex64 before magnitude.
            output[i] = std::abs(std::complex<float>(buffer[i]));
        }
    }
};

struct LSTM {
    int width;
    bool simd;
    std::vector<float> weights, recurrent, bias, peepholes, initial, cell_initial;
    std::array<float, hidden_size> hidden, cell;

    LSTM(std::ifstream& stream, int input_width, int mode, bool use_simd)
        : width(input_width), simd(use_simd),
          weights(read_values<float>(stream, 4 * hidden_size * input_width)),
          recurrent(read_values<float>(stream, 4 * hidden_size * hidden_size)),
          bias(read_values<float>(stream, 4 * hidden_size)),
          peepholes(read_values<float>(stream, 3 * hidden_size)),
          initial(read_values<float>(stream, hidden_size)),
          cell_initial(read_values<float>(stream, hidden_size)) {
        quantize(weights, width, mode);
        quantize(recurrent, hidden_size, mode);
        quantize(peepholes, hidden_size, mode);
        require_finite(bias);
        require_finite(initial);
        require_finite(cell_initial);
        reset();
    }

    void reset() {
        std::copy(initial.begin(), initial.end(), hidden.begin());
        std::copy(cell_initial.begin(), cell_initial.end(), cell.begin());
    }

    void run(const float* features) {
        float gates[4 * hidden_size], rec[4 * hidden_size];
        for (int row = 0; row < 4 * hidden_size; ++row) {
            gates[row] = dot(weights.data() + row * width, features, width,
                             simd) + bias[row];
            rec[row] = dot(recurrent.data() + row * hidden_size,
                           hidden.data(), hidden_size, simd);
        }
        for (int j = 0; j < hidden_size; ++j) {
            const float input = sigmoid((gates[j] + peepholes[j] * cell[j]) +
                                        rec[j]);
            const float forget = sigmoid((gates[50 + j] +
                                          peepholes[50 + j] * cell[j]) +
                                         rec[50 + j]);
            cell[j] = input * std::tanh(gates[75 + j] + rec[75 + j]) +
                      forget * cell[j];
            hidden[j] = sigmoid((gates[25 + j] +
                                 peepholes[25 + j] * cell[j]) + rec[25 + j]) *
                        std::tanh(cell[j]);
        }
    }
};

struct RNN {
    std::vector<LSTM> layers;
    std::vector<float> output, bias;
    bool simd;

    RNN(std::ifstream& stream, int mode, bool use_simd) : simd(use_simd) {
        layers.emplace_back(stream, 162, mode, use_simd);
        layers.emplace_back(stream, 25, mode, use_simd);
        layers.emplace_back(stream, 25, mode, use_simd);
        output = read_values<float>(stream, 25);
        bias = read_values<float>(stream, 1);
        quantize(output, 25, mode);
        require_finite(bias);
    }

    void reset() {
        for (size_t i = 0; i < layers.size(); ++i) {
            layers[i].reset();
        }
    }

    float run(const float* features) {
        for (size_t i = 0; i < layers.size(); ++i) {
            layers[i].run(features);
            features = layers[i].hidden.data();
        }
        return sigmoid(dot(output.data(), features, 25, simd) + bias[0]);
    }
};

struct HMM {
    std::vector<uint32_t> states, pointers, observation;
    std::vector<double> weights, initial, previous, next;
    double observation_lambda, max_bpm, last_beat, tempo;

    explicit HMM(std::ifstream& stream) {
        const std::vector<uint32_t> sizes = read_values<uint32_t>(stream, 2);
        if (sizes[0] == 0 || sizes[0] > 100000 || sizes[1] > 1000000) {
            throw std::runtime_error("Invalid DBN dimensions");
        }
        const std::vector<double> settings = read_values<double>(stream, 2);
        observation_lambda = settings[0];
        max_bpm = settings[1];
        states = read_values<uint32_t>(stream, sizes[1]);
        pointers = read_values<uint32_t>(stream, sizes[0] + 1);
        weights = read_values<double>(stream, sizes[1]);
        observation = read_values<uint32_t>(stream, sizes[0]);
        initial = read_values<double>(stream, sizes[0]);
        require_finite(settings);
        require_finite(weights);
        require_finite(initial);
        if (pointers.front() != 0 || pointers.back() != states.size() ||
            !std::is_sorted(pointers.begin(), pointers.end()) ||
            observation_lambda <= 1 || max_bpm <= 0) {
            throw std::runtime_error("Invalid DBN settings");
        }
        for (size_t i = 0; i < states.size(); ++i) {
            if (states[i] >= initial.size() || weights[i] < 0) {
                throw std::runtime_error("Invalid DBN transition");
            }
        }
        double total = 0;
        for (size_t i = 0; i < initial.size(); ++i) {
            if (observation[i] > 1 || initial[i] < 0) {
                throw std::runtime_error("Invalid DBN state");
            }
            total += initial[i];
        }
        if (std::abs(total - 1) > 1e-6) {
            throw std::runtime_error("Invalid initial DBN distribution");
        }
        next.resize(initial.size());
        reset();
    }

    void reset() {
        previous = initial;
        last_beat = tempo = 0;
    }

    bool run(float activation, double time) {
        // The Python observation model takes float32 logarithms. Its HMM
        // takes float64 exponentials of those rounded log densities.
        const float nonbeat = (1.f - activation) /
                              static_cast<float>(observation_lambda - 1);
        const double density[2] = {
            std::exp(static_cast<double>(std::log(nonbeat))),
            std::exp(static_cast<double>(std::log(activation)))};
        double total = 0;
        size_t best = 0;
        for (size_t i = 0; i < previous.size(); ++i) {
            double value = 0;
            for (size_t j = pointers[i]; j < pointers[i + 1]; ++j) {
                value += previous[states[j]] * weights[j];
            }
            next[i] = value * density[observation[i]];
            total += next[i];
            if (next[i] > next[best]) {
                best = i;
            }
        }
        if (!(total > 0) || !std::isfinite(total)) {
            // Preserve the previous posterior and frame clock. Exact endpoint
            // observations can rule out every path; do not publish NaNs.
            throw std::runtime_error("DBN observation has zero or invalid probability");
        }
        const double inverse = 1 / total;
        for (size_t i = 0; i < next.size(); ++i) {
            next[i] *= inverse;
        }
        previous.swap(next);
        const bool beat = observation[best] == 1 &&
                          time >= last_beat + 60 / max_bpm;
        if (beat) {
            tempo = 60 / (time - last_beat);
            last_beat = time;
        }
        return beat;
    }
};

struct Tracker {
    std::vector<double> window;
    std::array<std::vector<std::pair<int, float> >, num_bands> bank;
    std::vector<RNN> networks;
    std::unique_ptr<HMM> hmm;
    std::deque<std::array<float, num_bands> > history;
    std::deque<float> audio;
    FFT fft;
    int difference_lag;
    size_t offset, produced, reference, frames;

    Tracker(const char* path, int count, int mode, int simd) {
        if (count < 1 || count > 8 || mode < 0 || mode > 2 ||
            simd < 0 || simd > 1) {
            throw std::runtime_error("Invalid native tracker option");
        }
        std::ifstream stream(path, std::ios::binary);
        char magic[8];
        stream.read(magic, sizeof(magic));
        if (!stream || std::memcmp(magic, "MMONL001", sizeof(magic)) != 0) {
            throw std::runtime_error("Invalid native model asset");
        }
        window = read_values<double>(stream, frame_size);
        const std::vector<float> dense =
            read_values<float>(stream, (frame_size / 2) * num_bands);
        require_finite(window);
        require_finite(dense);
        const std::vector<uint32_t> lag = read_values<uint32_t>(stream, 1);
        if (lag[0] < 1 || lag[0] > 10) {
            throw std::runtime_error("Invalid spectral difference lag");
        }
        difference_lag = lag[0];
        for (int k = 0; k < frame_size / 2; ++k) {
            for (int j = 0; j < num_bands; ++j) {
                const float weight = dense[k * num_bands + j];
                if (weight != 0) {
                    bank[j].push_back(std::make_pair(k, weight));
                }
            }
        }
        networks.reserve(count);
        for (int i = 0; i < 8; ++i) {
            RNN network(stream, mode, simd != 0);
            if (i < count) {
                networks.push_back(std::move(network));
            }
        }
        hmm.reset(new HMM(stream));
        if (stream.peek() != std::ifstream::traits_type::eof()) {
            throw std::runtime_error("Trailing bytes in native model asset");
        }
        reset();
    }

    void reset() {
        for (size_t i = 0; i < networks.size(); ++i) {
            networks[i].reset();
        }
        hmm->reset();
        history.clear();
        audio.clear();
        offset = produced = reference = frames = 0;
    }

    void features(const float* frame, float* result) {
        float magnitude[frame_size / 2];
        fft.magnitudes(frame, window, magnitude);
        std::array<float, num_bands> current;
        for (int j = 0; j < num_bands; ++j) {
            float sum = 0;
            for (size_t k = 0; k < bank[j].size(); ++k) {
                sum += magnitude[bank[j][k].first] * bank[j][k].second;
            }
            current[j] = std::log10(1.f + sum);
            result[j] = current[j];
            result[num_bands + j] =
                history.size() >= static_cast<size_t>(difference_lag)
                ? std::max(0.f, current[j] - history.front()[j]) : 0;
        }
        history.push_back(current);
        while (history.size() > static_cast<size_t>(difference_lag)) {
            history.pop_front();
        }
    }

    float network(const float* features) {
        float activation = 0;
        for (size_t i = 0; i < networks.size(); ++i) {
            activation += networks[i].run(features);
        }
        return activation / networks.size();
    }

    void decode(float activation, double* result) {
        const double time = frames / 100.;
        const bool beat = hmm->run(activation, time);
        result[0] = time;
        result[1] = activation;
        result[2] = beat;
        result[3] = hmm->tempo;
        ++frames;
    }

    int push(const float* pcm, size_t count, double* output, size_t capacity) {
        if (count > 44100 || count > std::numeric_limits<size_t>::max() - produced) {
            throw std::runtime_error("Native PCM chunk exceeds 44100 samples");
        }
        check_input(pcm, count, "Missing native PCM buffer");
        const size_t end = produced + count;
        const size_t rows = end > reference ? (end - 1 - reference) / hop_size + 1 : 0;
        if (capacity < rows || (rows && !output)) {
            throw std::runtime_error("Native output buffer is too small");
        }
        if (count == 0) {
            return 0;
        }
        audio.insert(audio.end(), pcm, pcm + count);
        produced = end;
        size_t row = 0;
        while (reference < produced) {
            float frame[frame_size] = {};
            float values[num_bands * 2];
            for (size_t i = 0; i < frame_size; ++i) {
                // Avoid signed arithmetic on the long-running sample clock.
                if (reference + i >= frame_size - 1) {
                    const size_t position = reference + i - (frame_size - 1);
                    frame[i] = audio.at(position - offset);
                }
            }
            features(frame, values);
            decode(network(values), output + 4 * row++);
            reference += hop_size;
            const size_t keep = reference > frame_size - 1
                                ? reference - (frame_size - 1) : 0;
            while (offset < keep && !audio.empty()) {
                audio.pop_front();
                ++offset;
            }
        }
        return static_cast<int>(row);
    }
};

template <typename Function>
int guard(void* handle, Function function) {
    try {
        if (!handle) {
            throw std::runtime_error("Null native tracker handle");
        }
        return function(*static_cast<Tracker*>(handle));
    } catch (const std::exception& exception) {
        last_error = exception.what();
    } catch (...) {
        last_error = "Unknown native tracker error";
    }
    return -1;
}
} // namespace

extern "C" {
const char* mm_online_error(void) {
    return last_error.c_str();
}

void* mm_online_create(const char* path, int count, int mode, int simd) {
    try {
        if (!path) {
            throw std::runtime_error("Missing native model asset path");
        }
        return new Tracker(path, count, mode, simd);
    } catch (const std::exception& exception) {
        last_error = exception.what();
    } catch (...) {
        last_error = "Unknown native tracker error";
    }
    return NULL;
}

void mm_online_destroy(void* handle) {
    delete static_cast<Tracker*>(handle);
}

int mm_online_reset(void* handle) {
    return guard(handle, [](Tracker& tracker) { tracker.reset(); return 0; });
}

int mm_online_features(void* handle, const float* frame, float* output) {
    return guard(handle, [&](Tracker& tracker) {
        check_input(frame, frame_size, "Missing native frontend input");
        if (!output) {
            throw std::runtime_error("Missing native feature output");
        }
        tracker.features(frame, output);
        return 0;
    });
}

int mm_online_network(void* handle, const float* features, float* output) {
    return guard(handle, [&](Tracker& tracker) {
        check_input(features, 2 * num_bands, "Missing native network input");
        if (!output) {
            throw std::runtime_error("Missing native activation output");
        }
        *output = tracker.network(features);
        return 0;
    });
}

int mm_online_decode(void* handle, float activation, double* output) {
    return guard(handle, [&](Tracker& tracker) {
        if (!output || !std::isfinite(activation) || activation < 0 || activation > 1) {
            throw std::runtime_error("Invalid native beat activation or output");
        }
        tracker.decode(activation, output);
        return 0;
    });
}

int mm_online_push(void* handle, const float* pcm, size_t count,
                   double* output, size_t capacity) {
    return guard(handle, [&](Tracker& tracker) {
        return tracker.push(pcm, count, output, capacity);
    });
}

int mm_online_decoder_state(void* handle, double* output, size_t capacity) {
    return guard(handle, [&](Tracker& tracker) {
        const std::vector<double>& state = tracker.hmm->previous;
        if (output) {
            if (capacity < state.size()) {
                throw std::runtime_error("Native DBN state buffer is too small");
            }
            std::copy(state.begin(), state.end(), output);
        }
        return static_cast<int>(state.size());
    });
}

int mm_online_summary(void* handle, double* output) {
    return guard(handle, [&](Tracker& tracker) {
        if (!output) {
            throw std::runtime_error("Missing native DBN summary output");
        }
        const HMM& hmm = *tracker.hmm;
        const std::vector<double>::const_iterator best =
            std::max_element(hmm.previous.begin(), hmm.previous.end());
        output[0] = hmm.tempo;
        output[1] = hmm.last_beat;
        output[2] = *best;
        output[3] = best - hmm.previous.begin();
        return 0;
    });
}
} // extern "C"
