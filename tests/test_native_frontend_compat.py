"""Preserve processor customization and history across native frontend calls."""

import pickle
from pathlib import Path
import unittest

import numpy as np

from madmom.audio._native import NativeBeatFrontend
from madmom.audio.signal import FramedSignal, FramedSignalProcessor, Signal
from madmom.audio.spectrogram import (
    FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor,
    SpectrogramDifferenceProcessor,
)
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.processors import SequentialProcessor


class OffsetLogarithm(LogarithmicSpectrogramProcessor):
    def process(self, data, **kwargs):
        return super().process(data, **kwargs) + np.float32(.1)


def processors(streaming=True):
    return (
        FramedSignalProcessor(frame_size=2048, fps=100,
                              origin="stream" if streaming else "online",
                              num_frames=1 if streaming else None),
        ShortTimeFourierTransformProcessor(),
        FilteredSpectrogramProcessor(num_bands=12, fmin=30, fmax=17000,
                                      norm_filters=True),
        LogarithmicSpectrogramProcessor(mul=1, add=1),
        SpectrogramDifferenceProcessor(diff_ratio=.5, positive_diffs=True,
                                        stack_diffs=np.hstack),
    )


class TestNativeFrontendCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.signal = Signal(str(Path(__file__).parent / "data/audio/sample.wav"),
                            sample_rate=44100, num_channels=1, dtype=np.float32)
        cls.frames = FramedSignal(cls.signal, frame_size=2048, hop_size=441,
                                  origin="online")

    @staticmethod
    def frontends(streaming=True):
        # The reference invokes the original component processors directly.
        return (NativeBeatFrontend(processors(streaming)),
                SequentialProcessor(processors(streaming)))

    def assert_same(self, native, reference, frame=22, **kwargs):
        actual = native(self.frames[frame], **kwargs)
        expected = reference(self.frames[frame], **kwargs)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
        return actual

    def test_callable_and_subclass_replacements(self):
        for mode in ("callable", "subclass", "instance_override"):
            with self.subTest(replacement=mode):
                native, reference = self.frontends()
                for chain in (native, reference):
                    original = chain[3]
                    if mode == "callable":
                        chain[3] = lambda data, original=original: original(data)
                    elif mode == "subclass":
                        chain[3] = OffsetLogarithm(mul=1, add=1)
                    else:
                        original.process = lambda data, original=original.process, **kwargs: original(data, **kwargs) + np.float32(.1)
                self.assert_same(native, reference, reset=True)

    def test_fallback_and_native_resume_keep_history(self):
        native, reference = self.frontends()
        self.assert_same(native, reference, frame=20, reset=True)
        for chain in (native, reference):
            chain[3].mul = 2
        self.assert_same(native, reference, frame=21, reset=False)
        for chain in (native, reference):
            chain[3].mul = 1
        self.assert_same(native, reference, frame=22, reset=False)

    def test_keyword_fallback_and_native_resume_keep_history(self):
        native, reference = self.frontends()
        self.assert_same(native, reference, frame=20, reset=True)
        self.assert_same(native, reference, frame=21, reset=False, ignored_argument=True)
        self.assert_same(native, reference, frame=22, reset=False)

    def test_filterbank_and_window_changes_keep_history(self):
        for target in ("filterbank", "window"):
            with self.subTest(target=target):
                native, reference = self.frontends()
                self.assert_same(native, reference, frame=20, reset=True)
                for chain in (native, reference):
                    if target == "filterbank":
                        chain[2].filterbank *= 2
                    else:
                        chain[1].fft_window *= 2
                result = self.assert_same(native, reference, frame=21, reset=False)
                self.assertGreater(result[:, 81:].max(), .1)

    def test_public_buffer_reset_and_edit_keep_history(self):
        for operation in ("reset", "edit", "processor_reset"):
            with self.subTest(operation=operation):
                native, reference = self.frontends()
                self.assert_same(native, reference, frame=20, reset=True)
                self.assert_same(native, reference, frame=21, reset=False)
                for chain in (native, reference):
                    if operation == "reset":
                        chain[4]._buffer.reset()
                    elif operation == "edit":
                        chain[4]._buffer.data[-1] = 0
                    else:
                        chain[4].reset()
                self.assert_same(native, reference, frame=22, reset=False)

    def test_direct_difference_call_keeps_history(self):
        native, reference = self.frontends()
        self.assert_same(native, reference, frame=20, reset=True)
        for chain in (native, reference):
            # Users can call the existing exposed component chain directly.
            logged = SequentialProcessor(chain.processors[:-1])(self.frames[21])
            chain[4](logged, reset=False)
        self.assert_same(native, reference, frame=22, reset=False)

    def test_changed_batch_length_preserves_original_buffer_behavior(self):
        native, reference = self.frontends(streaming=False)
        for first, stop, reset in ((0, 13, True), (13, 14, False),
                                   (14, 27, False), (27, 32, False)):
            chunk = self.signal[first * 441:stop * 441]
            actual = native(chunk, reset=reset)
            expected = reference(chunk, reset=reset)
            self.assertEqual(actual.shape, expected.shape)
            np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
        self.assertEqual(native[4]._buffer.buffer_length,
                         reference[4]._buffer.buffer_length)

    def test_pickle_recreates_native_state_and_resets_difference_history(self):
        native, reference = self.frontends()
        self.assert_same(native, reference, frame=20, reset=True)
        native = pickle.loads(pickle.dumps(native))
        reference = pickle.loads(pickle.dumps(reference))
        self.assert_same(native, reference, frame=21, reset=False)
        self.assert_same(native, reference, frame=22, reset=False)

    def test_invalid_window_shape_uses_original_error_path(self):
        for field in ("window", "fft_window"):
            with self.subTest(field=field):
                native, reference = self.frontends()
                for chain in (native, reference):
                    setattr(chain[1], field, np.ones(1024))
                    with self.assertRaises(ValueError):
                        chain(self.frames[20], reset=True)
