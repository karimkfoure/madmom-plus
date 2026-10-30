/* Versioned streaming beat engine C ABI.
 * The madmom project licence applies. Exported models retain their licence.
 */
#ifndef MADMOM_ENGINE_H
#define MADMOM_ENGINE_H

#include <stddef.h>
#include <stdint.h>
#include "madmom_exports.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MM_ENGINE_ABI_V1 0x00010000u
#define MM_ENGINE_NO_SAMPLE UINT64_MAX
#define MM_ENGINE_OK 0
#define MM_ENGINE_FULL 1
#define MM_ENGINE_ERROR (-1)

typedef struct mm_engine mm_engine;

/* Set struct_size=sizeof(mm_engine_config_v1) and abi_version=MM_ENGINE_ABI_V1.
 * network_count: 1..8 in released model order; weight_mode: 0=f32, 1=f16-rounded,
 * 2=int8-rounded. Modes 1 and 2 still store and calculate float32 values.
 * use_simd: 0 or 1. queue_capacity: 1..8192 output frames. Reserve fields must
 * be zero. Passing NULL configuration selects 8 float32 networks, SIMD, and
 * 1024 queued frames. Configuration never changes the 44100 Hz mono input,
 * 2048-sample causal window, 441-sample hop, or released DBN defaults.
 */
typedef struct mm_engine_config_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t network_count;
    uint32_t weight_mode;
    uint32_t use_simd;
    uint32_t queue_capacity;
    uint32_t reserved[2];
} mm_engine_config_v1;

/* A frame uses samples through sample_index, measured from the latest reset.
 * An event reports the current observed beat, with no future commitment.
 */
typedef struct mm_engine_frame_v1 {
    uint64_t sample_index;
    float activation;
    uint32_t beat;
} mm_engine_frame_v1;

typedef struct mm_engine_state_v1 {
    uint64_t sample_index;       /* Latest processed frame, or NO_SAMPLE. */
    uint64_t received_samples;   /* Input samples accepted since reset. */
    uint64_t last_beat_sample;   /* Last reported event, or NO_SAMPLE. */
    double tempo_bpm;            /* Zero until an event is reported. */
    double reliability;          /* Maximum raw DBN posterior, uncalibrated. */
    uint32_t state_index;
    uint32_t pending_frames;
} mm_engine_state_v1;

/* Versions encode major in the upper 16 bits and minor in the lower 16 bits.
 * v1 functions and structs remain stable; a new major uses new entry points.
 * Handles belong to one thread at a time. Separate handles are independent.
 * Error text is thread-local and remains valid until the next failed call.
 */
MM_EXPORT uint32_t mm_engine_version(void);
MM_EXPORT const char *mm_engine_error(void);
MM_EXPORT mm_engine *mm_engine_create(const char *model_asset,
                                      const mm_engine_config_v1 *config);
/* Configure reloads the same asset and resets all state. A failed call keeps
 * the old configuration, queued frames, and tracker state.
 */
MM_EXPORT int mm_engine_configure(mm_engine *engine,
                                  const mm_engine_config_v1 *config);
MM_EXPORT void mm_engine_destroy(mm_engine *engine);
MM_EXPORT int mm_engine_reset(mm_engine *engine);
/* Push <=44100 finite float32 PCM samples, normally in [-1,1]. Input is copied
 * into bounded state. accepted must be non-NULL and is set to zero on failure.
 * If all output frames do not fit, return FULL without consuming any input;
 * read frames, then retry. Empty input succeeds without advancing state.
 */
MM_EXPORT int mm_engine_push(mm_engine *engine, const float *pcm, size_t count,
                             size_t *accepted);
/* Read and remove at most capacity frames. Return frame count, or ERROR.
 * NULL output with zero capacity returns the pending frame count, unchanged.
 */
MM_EXPORT int mm_engine_read(mm_engine *engine, mm_engine_frame_v1 *output,
                             size_t capacity);
/* State describes the latest processed frame, independent of queue reads.
 * reset clears clocks, history, recurrence, posterior, events, and queue.
 */
MM_EXPORT int mm_engine_state(mm_engine *engine, mm_engine_state_v1 *output);

#ifdef __cplusplus
}
#endif
#endif
