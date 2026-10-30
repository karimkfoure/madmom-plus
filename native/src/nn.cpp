/* Native float32 neural kernels. The madmom project code licence applies.
 * Arithmetic follows madmom/ml/nn/layers.py; this is an original translation.
 */
#include "madmom_nn.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif

namespace {
thread_local bool simd_enabled = true;

class Temporary {
    std::array<float, 1024> local;
    std::vector<float> large;
public:
    explicit Temporary(size_t count) {
        if (count > local.size()) large.resize(count);
    }
    float *data() { return large.empty() ? local.data() : large.data(); }
};

bool dimensions(size_t frames, size_t inputs, size_t outputs) {
    const size_t limit = std::numeric_limits<size_t>::max() / sizeof(float);
    return inputs && outputs && frames <= limit / outputs &&
           frames <= limit / inputs && inputs <= limit / outputs &&
           outputs <= limit / outputs && outputs <= limit / 8;
}

void matvec(const float *data, const float *weights, size_t inputs,
            size_t outputs, float *result) {
    std::fill(result, result + outputs, 0.f);
    for (size_t k = 0; k < inputs; ++k) {
        const float value = data[k];
        const float *row = weights + k * outputs;
        size_t j = 0;
#if defined(__aarch64__)
        if (simd_enabled) {
            const float32x4_t scalar = vdupq_n_f32(value);
            for (; j + 4 <= outputs; j += 4) {
                vst1q_f32(result + j, vaddq_f32(vld1q_f32(result + j),
                    vmulq_f32(scalar, vld1q_f32(row + j))));
            }
        }
#elif defined(__SSE2__)
        if (simd_enabled) {
            const __m128 scalar = _mm_set1_ps(value);
            for (; j + 4 <= outputs; j += 4) {
                _mm_storeu_ps(result + j, _mm_add_ps(_mm_loadu_ps(result + j),
                    _mm_mul_ps(scalar, _mm_loadu_ps(row + j))));
            }
        }
#endif
        for (; j < outputs; ++j) result[j] += value * row[j];
    }
}

float activate(float value, int code) {
    switch (code) {
    case 1: return std::tanh(value);
    case 2: return 1.f / (1.f + std::exp(-value));
    case 3: return value < 0.f ? 0.f : value;
    case 4: return value < 0.f ? std::exp(value) - 1.f : value;
    default: return value;
    }
}

void activation(float *values, size_t count, int code) {
    if (code == 5) {
        const float maximum = *std::max_element(values, values + count);
        float sum = 0.f;
        for (size_t j = 0; j < count; ++j) {
            values[j] = std::exp(values[j] - maximum);
            sum += values[j];
        }
        for (size_t j = 0; j < count; ++j) values[j] /= sum;
    } else {
        for (size_t j = 0; j < count; ++j) values[j] = activate(values[j], code);
    }
}

void gate(const float *data, const float *hidden, const float *cell,
          size_t inputs, size_t outputs, const mm_nn_gate &model,
          float *result, float *scratch) {
    matvec(data, model.weights, inputs, outputs, result);
    matvec(hidden, model.recurrent, outputs, outputs, scratch);
    for (size_t j = 0; j < outputs; ++j) {
        result[j] += model.bias[j];
        if (model.peepholes) result[j] += cell[j] * model.peepholes[j];
        result[j] += scratch[j];
    }
    activation(result, outputs, model.activation);
}
}

extern "C" int mm_nn_dense(const float *data, size_t frames, size_t inputs,
                            size_t outputs, const float *weights,
                            const float *bias, int code, float *result) {
    if (!dimensions(frames, inputs, outputs) || !data || !weights || !bias ||
        !result || code < 0 || code > 5) return -1;
    for (size_t i = 0; i < frames; ++i) {
        float *row = result + i * outputs;
        matvec(data + i * inputs, weights, inputs, outputs, row);
        for (size_t j = 0; j < outputs; ++j) row[j] += bias[j];
        activation(row, outputs, code);
    }
    return 0;
}

extern "C" int mm_nn_recurrent(const float *data, size_t frames, size_t inputs,
                                size_t outputs, const float *weights,
                                const float *bias, const float *recurrent,
                                int code, float *hidden, float *result) {
    if (!dimensions(frames, inputs, outputs) || !data || !weights || !bias ||
        !recurrent || !hidden || !result || code < 0 || code > 5) return -1;
    try {
        Temporary scratch(outputs);
        for (size_t i = 0; i < frames; ++i) {
            float *row = result + i * outputs;
            matvec(data + i * inputs, weights, inputs, outputs, row);
            matvec(hidden, recurrent, outputs, outputs, scratch.data());
            for (size_t j = 0; j < outputs; ++j) {
                row[j] += bias[j];
                row[j] += scratch.data()[j];
            }
            activation(row, outputs, code);
            std::copy(row, row + outputs, hidden);
        }
        return 0;
    } catch (...) { return -1; }
}

extern "C" int mm_nn_lstm(const float *data, size_t frames, size_t inputs,
                           size_t outputs, const mm_nn_gate *gates,
                           int code, float *hidden, float *cell, float *result) {
    if (!dimensions(frames, inputs, outputs) || !data || !gates || !hidden ||
        !cell || !result || code < 0 || code > 5) return -1;
    for (size_t k = 0; k < 4; ++k) {
        if (!gates[k].weights || !gates[k].bias || !gates[k].recurrent ||
            gates[k].activation < 0 || gates[k].activation > 5) return -1;
    }
    try {
        Temporary temporary(6 * outputs);
        float *ig = temporary.data(), *fg = ig + outputs;
        float *candidate = fg + outputs, *og = candidate + outputs;
        float *scratch = og + outputs, *state = scratch + outputs;
        for (size_t i = 0; i < frames; ++i) {
            const float *frame = data + i * inputs;
            gate(frame, hidden, cell, inputs, outputs, gates[0], ig, scratch);
            gate(frame, hidden, cell, inputs, outputs, gates[1], fg, scratch);
            gate(frame, hidden, cell, inputs, outputs, gates[2], candidate, scratch);
            for (size_t j = 0; j < outputs; ++j)
                cell[j] = candidate[j] * ig[j] + cell[j] * fg[j];
            gate(frame, hidden, cell, inputs, outputs, gates[3], og, scratch);
            std::copy(cell, cell + outputs, state);
            activation(state, outputs, code);
            for (size_t j = 0; j < outputs; ++j) {
                hidden[j] = state[j] * og[j];
                result[i * outputs + j] = hidden[j];
            }
        }
        return 0;
    } catch (...) { return -1; }
}

extern "C" int mm_nn_network(const float *data, size_t frames, size_t inputs,
                              const mm_nn_layer *layers, size_t layer_count,
                              float *result) {
    if (!data || !layers || !layer_count || !result) return -1;
    size_t width = inputs, maximum = 0;
    for (size_t i = 0; i < layer_count; ++i) {
        if (layers[i].inputs != width ||
            !dimensions(frames, layers[i].inputs, layers[i].outputs)) return -1;
        maximum = std::max(maximum, layers[i].outputs);
        width = layers[i].outputs;
    }
    try {
        Temporary first(frames * maximum), second(frames * maximum);
        const float *input = data;
        for (size_t i = 0; i < layer_count; ++i) {
            const mm_nn_layer &layer = layers[i];
            float *output = i + 1 == layer_count ? result :
                (i % 2 == 0 ? first.data() : second.data());
            int status = -1;
            if (layer.kind == 0) {
                status = mm_nn_dense(input, frames, layer.inputs, layer.outputs,
                    layer.gates[0].weights, layer.gates[0].bias,
                    layer.activation, output);
            } else if (layer.kind == 1) {
                status = mm_nn_recurrent(input, frames, layer.inputs, layer.outputs,
                    layer.gates[0].weights, layer.gates[0].bias,
                    layer.gates[0].recurrent, layer.activation, layer.hidden,
                    output);
            } else if (layer.kind == 2) {
                status = mm_nn_lstm(input, frames, layer.inputs, layer.outputs,
                    layer.gates, layer.activation, layer.hidden, layer.cell,
                    output);
            }
            if (status) return status;
            input = output;
        }
        return 0;
    } catch (...) { return -1; }
}

extern "C" int mm_nn_network_ex(const float *data, size_t frames, size_t inputs,
                                 const mm_nn_layer *layers, size_t layer_count,
                                 float *result, int use_simd) {
    const bool previous = simd_enabled;
    simd_enabled = use_simd != 0;
    const int status = mm_nn_network(data, frames, inputs, layers, layer_count,
                                     result);
    simd_enabled = previous;
    return status;
}
