"""Check buffer validation and configuration reuse in the internal adapter."""

import numpy as np
import pytest

from madmom import _native
from madmom.audio._native import NativeBeatFrontend
from madmom.audio.signal import FramedSignal, Signal

from .test_native_frontend_compat import processors


def configuration():
    window = np.hanning(2048)
    bank = np.zeros((1024, 81), np.float32)
    bank[np.arange(81), np.arange(81)] = 1
    return window, bank


def test_capsule_reuses_configuration_and_detects_edits():
    window, bank = configuration()
    handle = _native.mm_frontend_configure(None, window, bank, 2)
    assert _native.mm_frontend_configure(handle, window, bank, 2) is handle
    bank[0, 0] *= 2
    changed = _native.mm_frontend_configure(handle, window, bank, 2)
    assert changed is not handle
    window[10] *= 2
    assert _native.mm_frontend_configure(changed, window, bank, 2) is not changed


def test_capsule_rejects_invalid_buffers_and_preserves_state():
    window, bank = configuration()
    handle = _native.mm_frontend_configure(None, window, bank, 2)
    frame = np.zeros(2048, np.float32)
    output = np.empty((1, 162), np.float32)
    with pytest.raises(ValueError):
        _native.mm_frontend_process(handle, frame[:-1], 1, output)
    with pytest.raises(ValueError):
        _native.mm_frontend_process(handle, frame, 1, output[:, :-1])
    with pytest.raises(ValueError):
        _native.mm_frontend_configure(handle, window.astype(np.float32), bank, 2)
    with pytest.raises(ValueError):
        _native.mm_frontend_process(handle, frame, 1, output,
                                    np.full((2, 81), np.nan, np.float32))
    _native.mm_frontend_process(handle, frame, 1, output)
    np.testing.assert_array_equal(output, 0)


def test_unaligned_configuration_and_frames_keep_processor_behavior():
    audio = Signal(np.sin(np.arange(4096, dtype=np.float32) * .03), sample_rate=44100)
    frame = FramedSignal(audio, frame_size=2048, hop_size=441, origin='online')[2]
    reference = NativeBeatFrontend(processors())
    expected = reference(frame)
    adapted = NativeBeatFrontend(processors())
    for processor in (adapted, reference):
        # Initialize the public window and filterbank through normal processing.
        processor(frame)
    for index, attribute in ((1, 'fft_window'), (2, 'filterbank')):
        original = getattr(adapted[index], attribute)
        raw = bytearray(original.nbytes + 1)
        unaligned = np.ndarray(original.shape, dtype=original.dtype, buffer=raw,
                               offset=1).view(type(original))
        unaligned[...] = original
        setattr(adapted[index], attribute, unaligned)
    raw = bytearray(frame.nbytes + 1)
    unaligned_frame = np.ndarray(frame.shape, dtype=frame.dtype, buffer=raw,
                                  offset=1)
    unaligned_frame[...] = frame
    np.testing.assert_array_equal(adapted(Signal(unaligned_frame, sample_rate=44100)), expected)
