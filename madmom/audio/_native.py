"""Native implementation of the released online beat feature chain."""

from functools import lru_cache
import inspect

import numpy as np

from ..processors import BufferProcessor, SequentialProcessor
from .filters import Filterbank
from .signal import FramedSignalProcessor
from .spectrogram import (FilteredSpectrogramProcessor,
                          LogarithmicSpectrogramProcessor, Spectrogram,
                          SpectrogramDifferenceProcessor, _diff_frames)
from .stft import ShortTimeFourierTransformProcessor, fft_frequencies


_EXPECTED_PROCESSORS = (
    FramedSignalProcessor, ShortTimeFourierTransformProcessor,
    FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor,
    SpectrogramDifferenceProcessor)
_PCM_DTYPES = (np.dtype('int16'), np.dtype('float32'))


@lru_cache(maxsize=1)
def _api():
    from .. import _native
    return _native


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
        self._native_buffer = None
        # Fail immediately if a source installation omitted the required core.
        self._lib = _api()

    def _close(self):
        self._handle = None

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ('_handle', '_lib', '_native_buffer'):
            state.pop(key, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._handle, self._native_buffer = None, None
        self._lib = _api()

    def process(self, data, reset=True, **kwargs):
        # Modified chains and precision modes without a matching native kernel
        # still use their existing processors.
        if (len(self) != 5 or kwargs or
                any(type(processor) is not cls or 'process' in processor.__dict__
                    for processor, cls in zip(self, _EXPECTED_PROCESSORS))):
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
                frames.signal.dtype not in _PCM_DTYPES or
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
        if not window.flags.aligned:
            window = window.copy()
        if window.shape != (2048,):
            return super(NativeBeatFrontend, self).process(data, reset=reset)
        bank = filt.filterbank
        if inspect.isclass(bank) and issubclass(bank, Filterbank):
            bank = bank(fft_frequencies(1024, frames.signal.sample_rate),
                        num_bands=filt.num_bands, fmin=filt.fmin, fmax=filt.fmax,
                        fref=filt.fref, norm_filters=filt.norm_filters,
                        unique_filters=filt.unique_filters)
            filt.filterbank = bank
        if not isinstance(bank, Filterbank) or bank.shape != (1024, 81):
            return super(NativeBeatFrontend, self).process(data, reset=reset)
        if diff.diff_frames is None:
            diff.diff_frames = _diff_frames(diff.diff_ratio, frames.hop_size,
                                           2048, stft.window)
        lag = diff.diff_frames
        # The capsule compares borrowed buffers without copying their bytes.
        # This also detects in-place edits to the exposed window/filterbank.
        bank = np.ascontiguousarray(bank, dtype=np.float32)
        if not bank.flags.aligned:
            bank = bank.copy()
        self._handle = self._lib.mm_frontend_configure(
            self._handle, window, bank, lag)
        clear_history = reset or diff._buffer is None
        history = None if clear_history else np.ascontiguousarray(
            diff._buffer.data[-lag:], dtype=np.float32)
        if history is not None and not history.flags.aligned:
            history = history.copy()

        result = np.empty((len(frames), 162), dtype=np.float32)
        if len(frames) == 1:
            block = np.ascontiguousarray(frames[0], dtype=np.float32)
            if not block.flags.aligned:
                block = block.copy()
            self._lib.mm_frontend_process(
                self._handle, block, 1, result, history, clear_history)
        else:
            block = np.empty((min(64, len(frames)), 2048), dtype=np.float32)
            for start in range(0, len(frames), 64):
                count = min(64, len(frames) - start)
                for index in range(count):
                    block[index] = frames[start + index]
                self._lib.mm_frontend_process(
                    self._handle, block, count, result[start:start + count],
                    history if start == 0 else None,
                    clear_history if start == 0 else False)
        # Preserve the public difference processor's reset and history behavior.
        if len(result):
            # A Spectrogram view keeps the original difference processor
            # usable when callers change its settings or replace chain stages.
            logged = result[:, :81].view(Spectrogram)
            if reset or diff._buffer is None:
                history = np.full((lag, 81), np.inf, dtype=np.float32)
                initial = np.concatenate((history, logged), axis=0).view(Spectrogram)
                diff._buffer = BufferProcessor(init=initial)
            elif (type(diff._buffer) is BufferProcessor and
                  'process' not in diff._buffer.__dict__):
                # Match BufferProcessor's replacement semantics without a
                # general roll: eligibility fixes its length to lag + frames.
                buffered = np.empty_like(diff._buffer.data)
                buffered[:lag] = diff._buffer.data[-lag:]
                buffered[lag:] = logged
                diff._buffer.data = buffered
            else:
                diff._buffer(logged)
            self._native_buffer = diff._buffer
        return result
