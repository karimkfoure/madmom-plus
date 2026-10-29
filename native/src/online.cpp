/* Native causal frontend, online LSTM ensemble and forward DBN.
 * The madmom project licence applies. See LICENSE and docs/native.rst.
 */
#include "madmom_native.h"
#include "madmom_engine.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
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
    // Pack even and odd real samples into one half-size complex transform.
    // All twiddles and arithmetic remain double precision until madmom's
    // complex64 cast, so the optimization preserves its magnitude convention.
    static const size_t packed_size = frame_size / 2;
    std::array<size_t, packed_size> reversed;
    std::array<std::complex<double>, packed_size> roots;
    std::array<std::complex<double>, packed_size> buffer;

public:
    FFT() {
        for (size_t i = 0; i < packed_size; ++i) {
            size_t value = i;
            size_t result = 0;
            for (int bit = 0; bit < 10; ++bit) {
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
        for (size_t i = 0; i < packed_size; ++i) {
            buffer[reversed[i]] = std::complex<double>(
                frame[2 * i] * window[2 * i],
                frame[2 * i + 1] * window[2 * i + 1]);
        }
        for (size_t length = 2; length <= packed_size; length *= 2) {
            const size_t half = length / 2;
            const size_t stride = frame_size / length;
            for (size_t first = 0; first < packed_size; first += length) {
                for (size_t j = 0; j < half; ++j) {
                    const std::complex<double> even = buffer[first + j];
                    const std::complex<double> odd =
                        buffer[first + j + half] * roots[j * stride];
                    buffer[first + j] = even + odd;
                    buffer[first + j + half] = even - odd;
                }
            }
        }
        output[0] = std::abs(static_cast<float>(buffer[0].real() +
                                                buffer[0].imag()));
        for (size_t i = 1; i < packed_size; ++i) {
            const std::complex<double> paired = std::conj(buffer[packed_size - i]);
            const std::complex<double> even = (buffer[i] + paired) * .5;
            const std::complex<double> odd = (buffer[i] - paired) *
                                             std::complex<double>(0, -.5);
            // madmom casts the complex FFT to complex64 before magnitude.
            output[i] = std::abs(std::complex<float>(even + roots[i] * odd));
        }
    }
};

struct Frontend {
    struct Band {
        size_t first;
        std::vector<float> weights;
    };
    std::vector<double> window;
    std::array<Band, num_bands> bank;
    std::vector<std::array<float, num_bands> > history;
    size_t history_count, history_cursor;
    FFT fft;

    Frontend(const double* fft_window, const float* dense, size_t lag)
        : window(fft_window, fft_window + frame_size), history(lag) {
        if (lag < 1 || lag > 100000) {
            throw std::runtime_error("Invalid spectral difference lag");
        }
        require_finite(window);
        check_input(dense, (frame_size / 2) * num_bands,
                    "Missing native filterbank");
        for (int j = 0; j < num_bands; ++j) {
            size_t first = 0, end = frame_size / 2;
            while (first < end && dense[first * num_bands + j] == 0) {
                ++first;
            }
            while (end > first && dense[(end - 1) * num_bands + j] == 0) {
                --end;
            }
            bank[j].first = first;
            bank[j].weights.reserve(end - first);
            for (size_t k = first; k < end; ++k) {
                bank[j].weights.push_back(dense[k * num_bands + j]);
            }
        }
        reset();
    }

    void reset() {
        history_count = history_cursor = 0;
    }

    void features(const float* frame, float* result) {
        float magnitude[frame_size / 2];
        fft.magnitudes(frame, window, magnitude);
        std::array<float, num_bands>& previous = history[history_cursor];
        for (int j = 0; j < num_bands; ++j) {
            float sum = 0;
            const Band& band = bank[j];
            for (size_t k = 0; k < band.weights.size(); ++k) {
                // Skip interior zeros too, to retain the previous sparse
                // accumulation order for custom non-triangular filterbanks.
                if (band.weights[k] != 0) {
                    sum += magnitude[band.first + k] * band.weights[k];
                }
            }
            const float current = std::log10(1.f + sum);
            result[j] = current;
            result[num_bands + j] = history_count >= history.size()
                ? std::max(0.f, current - previous[j]) : 0;
            previous[j] = current;
        }
        if (history_count < history.size()) {
            ++history_count;
        }
        history_cursor = (history_cursor + 1) % history.size();
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
    std::unique_ptr<Frontend> frontend;
    std::vector<RNN> networks;
    std::unique_ptr<HMM> hmm;
    // Duplicating the ring keeps every causal frame contiguous, including
    // wraparound. Push never allocates or shifts audio samples.
    std::array<float, 2 * frame_size> audio;
    size_t produced, reference, frames;

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
        const std::vector<double> window = read_values<double>(stream, frame_size);
        const std::vector<float> dense =
            read_values<float>(stream, (frame_size / 2) * num_bands);
        const std::vector<uint32_t> lag = read_values<uint32_t>(stream, 1);
        if (lag[0] < 1 || lag[0] > 10) {
            throw std::runtime_error("Invalid spectral difference lag");
        }
        frontend.reset(new Frontend(window.data(), dense.data(), lag[0]));
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
        frontend->reset();
        audio.fill(0);
        produced = reference = frames = 0;
    }

    void features(const float* frame, float* result) {
        frontend->features(frame, result);
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
        size_t row = 0;
        for (size_t i = 0; i < count; ++i) {
            const size_t position = produced % frame_size;
            audio[position] = audio[position + frame_size] = pcm[i];
            ++produced;
            if (reference < produced) {
                float values[num_bands * 2];
                features(audio.data() + (produced % frame_size), values);
                decode(network(values), output + 4 * row++);
                reference += hop_size;
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

// The public tag remains incomplete in the C header.
struct mm_engine {
    std::string asset;
    std::unique_ptr<Tracker> tracker;
    std::vector<mm_engine_frame_v1> queue;
    size_t head, pending;
    uint64_t last_beat_sample;

    explicit mm_engine(const char* path, const mm_engine_config_v1* config)
        : asset(path), head(0), pending(0), last_beat_sample(MM_ENGINE_NO_SAMPLE) {
        configure(config);
    }

    void configure(const mm_engine_config_v1* config) {
        mm_engine_config_v1 defaults = {
            sizeof(mm_engine_config_v1), MM_ENGINE_ABI_V1, 8, 0, 1, 1024, {0, 0}};
        const mm_engine_config_v1& options = config ? *config : defaults;
        if (options.struct_size != sizeof(mm_engine_config_v1) ||
            options.abi_version != MM_ENGINE_ABI_V1 ||
            options.network_count < 1 || options.network_count > 8 ||
            options.weight_mode > 2 || options.use_simd > 1 ||
            options.queue_capacity < 1 || options.queue_capacity > 8192 ||
            options.reserved[0] || options.reserved[1]) {
            throw std::runtime_error("Invalid engine ABI version or configuration");
        }
        std::unique_ptr<Tracker> replacement(new Tracker(
            asset.c_str(), options.network_count, options.weight_mode,
            options.use_simd));
        std::vector<mm_engine_frame_v1> output(options.queue_capacity);
        tracker.swap(replacement);
        queue.swap(output);
        head = pending = 0;
        last_beat_sample = MM_ENGINE_NO_SAMPLE;
    }
};

namespace {
template <typename Handle, typename Function>
int component_guard(Handle* handle, Function function) {
    try {
        if (!handle) {
            throw std::runtime_error("Null native component handle");
        }
        return function(*handle);
    } catch (const std::exception& exception) {
        last_error = exception.what();
    } catch (...) {
        last_error = "Unknown native component error";
    }
    return -1;
}
} // namespace

extern "C" {
void* mm_frontend_create(const double* window, const float* bank, size_t lag) {
    try {
        if (!window || !bank || lag < 1 || lag > 100000) {
            throw std::runtime_error("Invalid native frontend arrays or lag");
        }
        return new Frontend(window, bank, lag);
    } catch (const std::exception& exception) {
        last_error = exception.what();
    } catch (...) {
        last_error = "Unknown native frontend error";
    }
    return NULL;
}

void mm_frontend_destroy(void* handle) {
    delete static_cast<Frontend*>(handle);
}

int mm_frontend_reset(void* handle) {
    return component_guard(static_cast<Frontend*>(handle), [](Frontend& frontend) {
        frontend.reset();
        return 0;
    });
}

int mm_frontend_process(void* handle, const float* frames, size_t count,
                        float* output) {
    return component_guard(static_cast<Frontend*>(handle), [&](Frontend& frontend) {
        if (count > std::numeric_limits<size_t>::max() / frame_size ||
            (count && !output)) {
            throw std::runtime_error("Invalid native frontend frame count or output");
        }
        check_input(frames, count * frame_size, "Missing native frontend frames");
        for (size_t i = 0; i < count; ++i) {
            frontend.features(frames + i * frame_size, output + i * 2 * num_bands);
        }
        return 0;
    });
}

uint32_t mm_engine_version(void) {
    return MM_ENGINE_ABI_V1;
}

const char* mm_engine_error(void) {
    return last_error.c_str();
}

mm_engine* mm_engine_create(const char* asset, const mm_engine_config_v1* config) {
    try {
        if (!asset) {
            throw std::runtime_error("Missing native model asset path");
        }
        return new mm_engine(asset, config);
    } catch (const std::exception& exception) {
        last_error = exception.what();
    } catch (...) {
        last_error = "Unknown native engine error";
    }
    return NULL;
}

int mm_engine_configure(mm_engine* engine, const mm_engine_config_v1* config) {
    return component_guard(engine, [&](mm_engine& value) {
        value.configure(config);
        return 0;
    });
}

void mm_engine_destroy(mm_engine* engine) {
    delete engine;
}

int mm_engine_reset(mm_engine* engine) {
    return component_guard(engine, [](mm_engine& value) {
        value.tracker->reset();
        value.head = value.pending = 0;
        value.last_beat_sample = MM_ENGINE_NO_SAMPLE;
        return 0;
    });
}

int mm_engine_push(mm_engine* engine, const float* pcm, size_t count,
                    size_t* accepted) {
    if (accepted) {
        *accepted = 0;
    }
    return component_guard(engine, [&](mm_engine& value) {
        if (!accepted || count > 44100 ||
            count > std::numeric_limits<size_t>::max() - value.tracker->produced -
                    frame_size - hop_size) {
            throw std::runtime_error("Invalid native input count or accepted output");
        }
        Tracker& tracker = *value.tracker;
        const size_t end = tracker.produced + count;
        const size_t rows = end > tracker.reference
            ? (end - 1 - tracker.reference) / hop_size + 1 : 0;
        if (rows > value.queue.size() - value.pending) {
            return MM_ENGINE_FULL;
        }
        const size_t first_frame = tracker.frames;
        double output[4 * 102];
        const int produced = tracker.push(pcm, count, output, 102);
        for (int i = 0; i < produced; ++i) {
            const size_t position = (value.head + value.pending) % value.queue.size();
            mm_engine_frame_v1& row = value.queue[position];
            row.sample_index = static_cast<uint64_t>(first_frame + i) * hop_size;
            row.activation = static_cast<float>(output[4 * i + 1]);
            row.beat = output[4 * i + 2] != 0;
            if (row.beat) {
                value.last_beat_sample = row.sample_index;
            }
            ++value.pending;
        }
        *accepted = count;
        return MM_ENGINE_OK;
    });
}

int mm_engine_read(mm_engine* engine, mm_engine_frame_v1* output, size_t capacity) {
    return component_guard(engine, [&](mm_engine& value) {
        if (!output) {
            if (capacity) {
                throw std::runtime_error("Missing native engine frame output");
            }
            return static_cast<int>(value.pending);
        }
        const size_t count = std::min(capacity, value.pending);
        for (size_t i = 0; i < count; ++i) {
            output[i] = value.queue[(value.head + i) % value.queue.size()];
        }
        value.head = (value.head + count) % value.queue.size();
        value.pending -= count;
        return static_cast<int>(count);
    });
}

int mm_engine_state(mm_engine* engine, mm_engine_state_v1* output) {
    return component_guard(engine, [&](mm_engine& value) {
        if (!output) {
            throw std::runtime_error("Missing native engine state output");
        }
        const Tracker& tracker = *value.tracker;
        const HMM& hmm = *tracker.hmm;
        const std::vector<double>::const_iterator best =
            std::max_element(hmm.previous.begin(), hmm.previous.end());
        output->sample_index = tracker.frames
            ? static_cast<uint64_t>(tracker.frames - 1) * hop_size : MM_ENGINE_NO_SAMPLE;
        output->received_samples = tracker.produced;
        output->last_beat_sample = value.last_beat_sample;
        output->tempo_bpm = hmm.tempo;
        output->reliability = *best;
        output->state_index = static_cast<uint32_t>(best - hmm.previous.begin());
        output->pending_frames = static_cast<uint32_t>(value.pending);
        return MM_ENGINE_OK;
    });
}

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
