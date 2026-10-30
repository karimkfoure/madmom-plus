/* Native kernels for the existing madmom neural layer API.
 * The project code licence applies. No model weights are included here.
 */
#ifndef MADMOM_NN_H
#define MADMOM_NN_H

#include <stddef.h>

#if defined(_WIN32)
#define MM_NN_API __declspec(dllexport)
#else
#define MM_NN_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Internal component ABI. Arrays use row-major float32 storage. Weights have
 * shape (inputs, outputs), recurrent weights have shape (outputs, outputs).
 * Activation codes: 0=linear, 1=tanh, 2=sigmoid, 3=relu, 4=elu, 5=softmax.
 * Kernels return zero on success, -1 for invalid dimensions or allocation.
 * Callers must supply valid, non-overlapping buffers of the documented sizes.
 */
typedef struct mm_nn_gate {
    const float *weights;
    const float *bias;
    const float *recurrent;
    const float *peepholes; /* NULL disables the peephole connection. */
    int activation;
} mm_nn_gate;

MM_NN_API int mm_nn_dense(const float *data, size_t frames, size_t inputs,
                         size_t outputs, const float *weights,
                         const float *bias, int activation, float *result);
MM_NN_API int mm_nn_recurrent(const float *data, size_t frames, size_t inputs,
                             size_t outputs, const float *weights,
                             const float *bias, const float *recurrent,
                             int activation, float *hidden, float *result);
/* Gate order: input, forget, cell, output. Hidden and cell states are updated
 * in place. The output activation acts on cell state before the output gate.
 */
MM_NN_API int mm_nn_lstm(const float *data, size_t frames, size_t inputs,
                        size_t outputs, const mm_nn_gate *gates,
                        int activation, float *hidden, float *cell,
                        float *result);

typedef struct mm_nn_layer {
    int kind; /* 0=dense, 1=recurrent, 2=LSTM. */
    size_t inputs;
    size_t outputs;
    int activation;
    mm_nn_gate gates[4];
    float *hidden;
    float *cell;
} mm_nn_layer;

/* Process a supported sequence in one call. This avoids a Python call between
 * layers during streaming. Each layer retains only its last hidden/cell state.
 */
MM_NN_API int mm_nn_network(const float *data, size_t frames, size_t inputs,
                           const mm_nn_layer *layers, size_t layer_count,
                           float *result);
MM_NN_API int mm_nn_network_ex(const float *data, size_t frames, size_t inputs,
                              const mm_nn_layer *layers, size_t layer_count,
                              float *result, int use_simd);

#ifdef __cplusplus
}
#endif
#endif
