"""Native implementation of the released online beat feature chain."""

import ctypes as ct
from functools import lru_cache
import inspect

import numpy as np

from ..processors import BufferProcessor, SequentialProcessor


@lru_cache(maxsize=1)
def _api():
    from .. import _native

    lib = ct.CDLL(_native.__file__)
    fp, dp, handle = ct.POINTER(ct.c_float), ct.POINTER(ct.c_double), ct.c_void_p
    signatures = {
        'create': ([dp, fp, ct.c_size_t], handle),
        'destroy': ([handle], None),
        'reset': ([handle], ct.c_int),
        'history': ([handle, fp, ct.c_size_t], ct.c_int),
        'process': ([handle, fp, ct.c_size_t, fp], ct.c_int),
    }
    for name, (args, result) in signatures.items():
        function = getattr(lib, 'mm_frontend_' + name)
        function.argtypes, function.restype = args, result
    lib.mm_online_error.restype = ct.c_char_p
    return lib


def _pointer(array, dtype=ct.c_float):
    return array.ctypes.data_as(ct.POINTER(dtype))


class NativeBeatFrontend(SequentialProcessor):
    """Run the beat frontend in C++ while retaining its processor sequence.

    Python retains framing and filter construction, so existing frame origins,
    non-integer hops, and model parameters keep their meaning. The native call
    fuses the FFT, filterbank, logarithm, and spectral difference. At most 64
    frames are copied at a time to bound temporary storage.
    """

    def __init__(self, processors):
        super(NativeBeatFrontend, self).__init__(processors)
        self._handle = None
        self._key = None
        self._native_buffer = None
        self._history = None
        # Fail immediately if a source installation omitted the required core.
        self._lib = _api()

    def _close(self):
        if getattr(self, '_handle', None):
            self._lib.mm_frontend_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self._close()

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ('_handle', '_lib', '_key', '_native_buffer', '_history'):
            state.pop(key, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._handle, self._key, self._native_buffer = None, None, None
        self._history = None
        self._lib = _api()

    def process(self, data, reset=True, **kwargs):
        from .filters import Filterbank
        from .signal import FramedSignalProcessor
        from .spectrogram import (FilteredSpectrogramProcessor,
                                  LogarithmicSpectrogramProcessor,
                                  Spectrogram, SpectrogramDifferenceProcessor,
                                  _diff_frames)
        from .stft import ShortTimeFourierTransformProcessor, fft_frequencies

        # Modified chains and precision modes without a matching native kernel
        # still use their existing processors.
        expected = (FramedSignalProcessor, ShortTimeFourierTransformProcessor,
                    FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor,
                    SpectrogramDifferenceProcessor)
        if (len(self) != 5 or kwargs or
                any(type(processor) is not cls or 'process' in processor.__dict__
                    for processor, cls in zip(self, expected))):
            return super(NativeBeatFrontend, self).process(data, reset=reset,
                                                          **kwargs)
        frames_processor, stft, filt, spec, diff = self
        frames = frames_processor(data)
        if (not reset and diff._buffer is not None and
                diff.diff_frames is not None and
                len(frames) != diff._buffer.buffer_length - diff.diff_frames):
            # Preserve the original buffer's fixed first-call length when a
            # caller changes batch size without resetting the processor.
            return super(NativeBeatFrontend, self).process(data, reset=reset)
        if (frames.shape[1:] != (2048,) or
                frames.signal.dtype not in (np.dtype('int16'), np.dtype('float32')) or
                stft.fft_size not in (None, 2048) or stft.circular_shift or
                stft.include_nyquist or spec.log is not np.log10 or
                spec.mul != 1 or spec.add != 1 or not diff.positive_diffs or
                diff.diff_max_bins is not None or diff.stack_diffs is not np.hstack):
            return super(NativeBeatFrontend, self).process(data, reset=reset)

        if stft.fft_window is None:
            window = stft.window(2048) if callable(stft.window) else stft.window
            window = np.ones(2048) if window is None else np.asarray(window)
            if frames.signal.dtype.kind == 'i':
                window = window / float(np.iinfo(frames.signal.dtype).max)
            stft.fft_window = np.ascontiguousarray(window, dtype=np.float64)
        window = np.ascontiguousarray(stft.fft_window, dtype=np.float64)
        if window.shape != (2048,):
            return super(NativeBeatFrontend, self).process(data, reset=reset)
        bank = filt.filterbank
        if inspect.isclass(bank) and issubclass(bank, Filterbank):
            bank = bank(fft_frequencies(1024, frames.signal.sample_rate),
                        num_bands=filt.num_bands, fmin=filt.fmin, fmax=filt.fmax,
                        fref=filt.fref, norm_filters=filt.norm_filters,
                        unique_filters=filt.unique_filters)
            filt.filterbank = bank
        if np.shape(bank) != (1024, 81):
            return super(NativeBeatFrontend, self).process(data, reset=reset)
        if diff.diff_frames is None:
            diff.diff_frames = _diff_frames(diff.diff_ratio, frames.hop_size,
                                           2048, stft.window)
        lag = diff.diff_frames
        # Byte keys also detect in-place edits to the public window/filterbank.
        key = (window.tobytes(), np.asarray(bank).tobytes(), lag)
        reconfigured = key != self._key
        if reconfigured:
            self._close()
            bank = np.ascontiguousarray(bank, dtype=np.float32)
            self._handle = self._lib.mm_frontend_create(
                _pointer(window, ct.c_double), _pointer(bank), lag)
            if not self._handle:
                raise ValueError(self._lib.mm_online_error().decode())
            self._key = key
        if reset or diff._buffer is None:
            self._lib.mm_frontend_reset(self._handle)
        else:
            history = np.ascontiguousarray(diff._buffer.data[-lag:], dtype=np.float32)
            if (reconfigured or diff._buffer is not self._native_buffer or
                    not np.array_equal(history, self._history)):
                if self._lib.mm_frontend_history(
                        self._handle, _pointer(history), len(history)) < 0:
                    raise ValueError(self._lib.mm_online_error().decode())

        result = np.empty((len(frames), 162), dtype=np.float32)
        block = np.empty((min(64, len(frames)), 2048), dtype=np.float32)
        for start in range(0, len(frames), 64):
            count = min(64, len(frames) - start)
            for index in range(count):
                block[index] = frames[start + index]
            if self._lib.mm_frontend_process(
                    self._handle, _pointer(block), count,
                    _pointer(result[start:start + count])) < 0:
                raise ValueError(self._lib.mm_online_error().decode())
        # Preserve the public difference processor's reset and history behavior.
        if len(result):
            # A Spectrogram view keeps the original difference processor
            # usable when callers change its settings or replace chain stages.
            logged = result[:, :81].view(Spectrogram)
            if reset or diff._buffer is None:
                history = np.full((lag, 81), np.inf, dtype=np.float32)
                initial = np.concatenate((history, logged), axis=0).view(Spectrogram)
                diff._buffer = BufferProcessor(init=initial)
            else:
                diff._buffer(logged)
            self._native_buffer = diff._buffer
            self._history = np.array(diff._buffer.data[-lag:], copy=True)
        return result
