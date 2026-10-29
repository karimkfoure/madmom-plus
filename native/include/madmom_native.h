/* Native implementation of madmom's causal beat tracker.
 * The madmom project licence applies. See LICENSE and docs/native.rst.
 */
#ifndef MADMOM_NATIVE_H
#define MADMOM_NATIVE_H

#include <stddef.h>

#include "madmom_exports.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Audio: mono float32 PCM at 44100 Hz, normally in [-1, 1].
 * Frames: 2048 samples, causal zero padding, 441-sample hop (100 fps).
 * network_count: first 1..8 online LSTMs; 8 preserves the ensemble.
 * weight_mode: 0=float32, 1=fp16 rounded weights, 2=int8 rounded weights.
 * Rounded weights are decoded to float32 at load time. These modes measure
 * quantization error; they do not use packed or integer inference kernels.
 * simd: 0=scalar arithmetic, 1=NEON/SSE2 where the compiler supports it.
 * Creation returns NULL on error. All other fallible calls return -1.
 * Each handle belongs to one thread at a time. Distinct handles are separate.
 */
MM_EXPORT void *mm_online_create(const char *asset, int network_count, int weight_mode,
                       int simd);
MM_EXPORT void mm_online_destroy(void *handle);
MM_EXPORT int mm_online_reset(void *handle);
MM_EXPORT const char *mm_online_error(void);

/* Component calls advance only their respective component state.
 * Buffer sizes are fixed: 2048 input samples, 162 features, one activation.
 */
MM_EXPORT int mm_online_features(void *handle, const float *frame2048, float *features162);
MM_EXPORT int mm_online_network(void *handle, const float *features162, float *activation);
/* If an observation rules out every DBN path, decoding returns -1 and keeps
 * the prior decoder state and frame clock. Reset if a new stream is wanted.
 */
MM_EXPORT int mm_online_decode(void *handle, float activation, double *output4);

/* Push at most 44100 samples. A zero-length push returns zero rows.
 * Output rows: frame reference time (s), activation, beat event (0/1), BPM.
 * capacity is a row count, not a count of doubles. count/441+2 rows always fit.
 * Output is untouched and state does not advance if validation fails.
 * A frame at time t uses samples through t. Its output is available only when
 * that sample has arrived; compute time is not included in reference time.
 */
MM_EXPORT int mm_online_push(void *handle, const float *pcm, size_t count,
                   double *output, size_t capacity);

/* Query the DBN state length with output=NULL, then copy its probabilities.
 * These queries do not advance the tracker. summary has four doubles:
 * tempo BPM, last beat time, maximum posterior, most likely state index.
 */
MM_EXPORT int mm_online_decoder_state(void *handle, double *output, size_t capacity);
MM_EXPORT int mm_online_summary(void *handle, double *output4);


/* The processor frontend uses the same kernel as the streaming engine.
 * window: 2048 float64 values, including any PCM normalization factor.
 * filterbank: contiguous row-major float32 array with shape (1024, 81).
 * difference_lag: positive number of frames, at most 100000.
 * process input/output shapes: (count, 2048) and (count, 162).
 * Frame origin and hop are controlled by the caller. The first lag difference
 * rows after create/reset are zero. Input and output must not overlap.
 * Creation copies the arrays; their lifetimes do not bind the handle.
 */
MM_EXPORT void *mm_frontend_create(const double *window,
                                   const float *filterbank, size_t difference_lag);
MM_EXPORT void mm_frontend_destroy(void *handle);
MM_EXPORT int mm_frontend_reset(void *handle);
/* Replace only the difference history from (count,81) logarithmic rows,
 * oldest first; count must not exceed the configured lag. Leading rows made
 * entirely of +infinity represent upstream missing-history padding and are
 * skipped. Other rows must be finite. Invalid input preserves prior state.
 */
MM_EXPORT int mm_frontend_history(void *handle, const float *log_features,
                                  size_t count);
MM_EXPORT int mm_frontend_process(void *handle, const float *frames,
                                  size_t count, float *output);

#ifdef __cplusplus
}
#endif
#endif
