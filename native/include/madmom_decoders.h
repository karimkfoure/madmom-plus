/* Internal kernels shared by the Python processors and the native engine.
 * The madmom project licence applies. See LICENSE and docs/native.rst.
 */
#ifndef MADMOM_DECODERS_H
#define MADMOM_DECODERS_H

#include <stddef.h>
#include <stdint.h>
#include "madmom_exports.h"

#ifdef __cplusplus
extern "C" {
#endif

/* CSR transitions list the predecessor states of each destination state.
 * Densities and output are contiguous row-major double arrays. previous has
 * num_states values and receives the final normalized row. All arrays must
 * have the dimensions supplied here. Return zero, or -1 for invalid indices
 * or dimensions. Impossible observations retain the original NaN semantics.
 */
MM_EXPORT int mm_hmm_forward(
    const uint32_t *states, const uint32_t *pointers,
    const double *probabilities, size_t num_states, size_t num_transitions,
    const uint32_t *observation_pointers, const double *densities,
    size_t num_frames, size_t num_densities, double *previous, double *output);

/* Reject impossible or non-finite observations with -2 before updating the
 * previous distribution for that frame. Earlier frames remain committed.
 * This lets a one-frame engine call retain its prior state on failure.
 */
MM_EXPORT int mm_hmm_forward_checked(
    const uint32_t *states, const uint32_t *pointers,
    const double *probabilities, size_t num_states, size_t num_transitions,
    const uint32_t *observation_pointers, const double *densities,
    size_t num_frames, size_t num_densities, double *previous, double *output);

/* The same recursion, with only the best state index returned per frame.
 * Working storage is num_states doubles, independent of num_frames.
 */
MM_EXPORT int mm_hmm_forward_states(
    const uint32_t *states, const uint32_t *pointers,
    const double *probabilities, size_t num_states, size_t num_transitions,
    const uint32_t *observation_pointers, const double *densities,
    size_t num_frames, size_t num_densities, double *previous,
    uint32_t *output_states);

/* Accept a selected beat state only after the minimum beat interval. Both
 * functions update last_beat and tempo in place. The scalar function returns
 * 1 for an accepted event, 0 otherwise. The batch function returns the event
 * count, or -1 for invalid dimensions or parameters. Output holds up to
 * num_frames event times in seconds. The first beat is suppressed as in the
 * original online DBN processor.
 */
MM_EXPORT int mm_beat_event(double time, double max_bpm, int is_beat,
                            double *last_beat, double *tempo);
MM_EXPORT ptrdiff_t mm_beat_events(
    const uint8_t *beat_flags, size_t num_frames, int64_t first_frame,
    double fps, double max_bpm, double *last_beat, double *tempo,
    double *output);

/* Pick finite one-dimensional activations with constant-zero boundaries.
 * The moving average and maximum include the current sample. Set both
 * average windows to zero to disable averaging. Output holds up to count
 * indices; the return value is the number of indices, or -1 on failure.
 */
MM_EXPORT ptrdiff_t mm_peak_picking_f32(
    const float *activations, size_t count, float threshold,
    size_t pre_avg, size_t post_avg, size_t pre_max, size_t post_max,
    size_t *indices);
MM_EXPORT ptrdiff_t mm_peak_picking_f64(
    const double *activations, size_t count, double threshold,
    size_t pre_avg, size_t post_avg, size_t pre_max, size_t post_max,
    size_t *indices);

#ifdef __cplusplus
}
#endif
#endif
