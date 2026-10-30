"""Array adapters for the native decoder kernels used by Python processors."""

import ctypes as ct
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _library():
    from . import _native

    library = ct.CDLL(_native.__file__)
    unsigned = ct.POINTER(ct.c_uint32)
    double = ct.POINTER(ct.c_double)
    index = ct.POINTER(ct.c_size_t)
    library.mm_hmm_forward.argtypes = [
        unsigned, unsigned, double, ct.c_size_t, ct.c_size_t,
        unsigned, double, ct.c_size_t, ct.c_size_t, double, double,
    ]
    library.mm_hmm_forward.restype = ct.c_int
    library.mm_hmm_forward_states.argtypes = (
        library.mm_hmm_forward.argtypes[:-1] + [unsigned])
    library.mm_hmm_forward_states.restype = ct.c_int
    library.mm_beat_events.argtypes = [
        ct.POINTER(ct.c_uint8), ct.c_size_t, ct.c_int64, ct.c_double,
        ct.c_double, double, double, double,
    ]
    library.mm_beat_events.restype = ct.c_ssize_t
    for suffix, scalar in (('f32', ct.c_float), ('f64', ct.c_double)):
        function = getattr(library, 'mm_peak_picking_' + suffix)
        function.argtypes = [ct.POINTER(scalar), ct.c_size_t, scalar,
                             ct.c_size_t, ct.c_size_t, ct.c_size_t, ct.c_size_t,
                             index]
        function.restype = ct.c_ssize_t
    return library


@lru_cache(maxsize=2)
def hmm_forward_address(states=False):
    """Return a kernel address for direct calls from the Cython HMM adapter."""
    function = (_library().mm_hmm_forward_states if states else
                _library().mm_hmm_forward)
    return ct.cast(function, ct.c_void_p).value


def _pointer(array, scalar):
    return array.ctypes.data_as(ct.POINTER(scalar))


def forward(transition_model, observation_pointers, densities, previous):
    """Compute normalized forward rows and update the previous state in place."""
    states = np.ascontiguousarray(transition_model.states, dtype=np.uint32)
    pointers = np.ascontiguousarray(transition_model.pointers, dtype=np.uint32)
    probabilities = np.ascontiguousarray(transition_model.probabilities,
                                         dtype=np.float64)
    observation_pointers = np.ascontiguousarray(observation_pointers,
                                               dtype=np.uint32)
    densities = np.ascontiguousarray(densities, dtype=np.float64)
    previous = np.asarray(previous)
    num_states = transition_model.num_states
    if (states.ndim != 1 or probabilities.shape != states.shape or
            pointers.shape != (num_states + 1,) or
            observation_pointers.shape != (num_states,) or
            densities.ndim != 2 or previous.shape != (num_states,) or
            previous.dtype != np.float64 or not previous.flags.c_contiguous or
            not previous.flags.writeable):
        raise ValueError('Invalid forward HMM array dimensions or state buffer')
    result = np.empty((len(densities), num_states), dtype=np.float64)
    status = _library().mm_hmm_forward(
        _pointer(states, ct.c_uint32), _pointer(pointers, ct.c_uint32),
        _pointer(probabilities, ct.c_double), num_states, len(states),
        _pointer(observation_pointers, ct.c_uint32),
        _pointer(densities, ct.c_double), len(densities), densities.shape[1],
        _pointer(previous, ct.c_double), _pointer(result, ct.c_double),
    )
    if status:
        raise ValueError('Invalid forward HMM transition or observation index')
    return result


def peak_picking(activations, threshold, pre_avg, post_avg, pre_max, post_max):
    """Pick finite one-dimensional float activations with zero boundaries."""
    activations = np.ascontiguousarray(activations)
    if activations.ndim != 1 or activations.dtype not in (np.float32, np.float64):
        raise ValueError('Native peak picking requires 1D float32 or float64')
    windows = (pre_avg, post_avg, pre_max, post_max)
    if any(not isinstance(value, (int, np.integer)) or
           not 0 <= value <= np.iinfo(np.intp).max
           for value in windows):
        raise ValueError('Peak picking windows must be nonnegative integers')
    indices = np.empty(len(activations), dtype=np.uintp)
    scalar = ct.c_float if activations.dtype == np.float32 else ct.c_double
    suffix = 'f32' if activations.dtype == np.float32 else 'f64'
    function = getattr(_library(), 'mm_peak_picking_' + suffix)
    count = function(_pointer(activations, scalar), len(activations), threshold,
                     *windows, _pointer(indices, ct.c_size_t))
    if count < 0:
        raise ValueError('Invalid peak picking input or window size')
    return indices[:count].astype(np.intp, copy=False)


def beat_events(beat_flags, first_frame, fps, max_bpm, last_beat, tempo):
    """Apply the shared DBN event interval rule and return events and state."""
    flags = np.ascontiguousarray(beat_flags, dtype=np.uint8)
    if flags.ndim != 1:
        raise ValueError('Beat flags must be one-dimensional')
    if not -(1 << 63) <= first_frame < (1 << 63):
        raise ValueError('Beat frame counter is outside the int64 range')
    output = np.empty(len(flags), dtype=np.float64)
    from . import _native
    count, last, bpm = _native.mm_beat_events(
        flags, first_frame, fps, max_bpm, last_beat, tempo, output)
    return output[:count], last, bpm
