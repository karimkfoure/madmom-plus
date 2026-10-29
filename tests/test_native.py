"""Check the required streaming core against the Python processor API."""

import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from madmom import native
from madmom.audio.signal import FramedSignal, Signal
from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor
from madmom.models import BEATS_LSTM

from . import AUDIO_PATH


class TestNativeAvailability(unittest.TestCase):
    def test_availability_is_boolean(self):
        self.assertIsInstance(native.available(), bool)

    def test_missing_required_core_reports_import_error(self):
        script = """
import importlib.abc
import sys

class WithoutNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'madmom._native':
            raise ImportError('native extension disabled for this test')

sys.meta_path.insert(0, WithoutNative())
from madmom import native
from madmom.features.beats import RNNBeatProcessor
assert native.available() is False
try:
    native.NativeBeatTracker()
except ImportError:
    pass
else:
    raise AssertionError('missing extension must give an actionable error')
try:
    RNNBeatProcessor(online=True, origin='online')(sys.argv[1])
except ImportError:
    pass
else:
    raise AssertionError('online processors must use the required native core')
"""
        result = subprocess.run([sys.executable, '-c', script,
                                 os.path.join(AUDIO_PATH, 'sample.wav')],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class TestNativeBeatTracker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.model_path = os.path.join(cls.directory.name, 'online.bin')
        native.export_model(cls.model_path)
        cls.audio = np.asarray(Signal(os.path.join(AUDIO_PATH, 'sample.wav'),
                                      sample_rate=44100, num_channels=1,
                                      dtype=np.float32))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def tracker(self, **kwargs):
        tracker = native.NativeBeatTracker(model_path=self.model_path,
                                            **kwargs)
        self.addCleanup(tracker.close)
        return tracker

    def assert_activation_parity(self, audio, networks=8, simd=False):
        reference = RNNBeatProcessor(online=True, origin='online', fps=100,
                                     nn_files=BEATS_LSTM[:networks])(audio)
        rows = self.tracker(networks=networks, simd=simd).push(audio)
        self.assertEqual(rows.shape, (len(reference), 4))
        np.testing.assert_array_equal(rows[:, 0], np.arange(len(rows)) / 100.)
        # Different float32 reduction orders affect the recurrent network.
        np.testing.assert_allclose(rows[:, 1], reference, atol=2e-5, rtol=2e-4)
        decoder = DBNBeatTrackingProcessor(fps=100, online=True)
        beats = decoder(reference)
        native_beats = rows[rows[:, 2] != 0, 0]
        self.assertEqual(len(native_beats), len(beats))
        np.testing.assert_allclose(native_beats, beats, atol=0.01, rtol=0)
        return rows

    def test_audio_activation_and_beat_parity(self):
        for filename in ('sample.wav', 'sample2.wav', 'sample_22050.wav'):
            with self.subTest(audio=filename):
                audio = Signal(os.path.join(AUDIO_PATH, filename),
                               sample_rate=44100, num_channels=1,
                               dtype=np.float32)
                self.assert_activation_parity(audio)

    def test_frontend_and_network_stages(self):
        processor = RNNBeatProcessor(online=True, origin='online', fps=100)
        reference_features = processor[0](self.audio)
        reference_activations = processor[1](reference_features)
        tracker = self.tracker()
        frames = FramedSignal(self.audio, frame_size=2048, fps=100,
                              origin='online', sample_rate=44100)
        features = np.vstack([tracker.features(frame) for frame in frames])
        np.testing.assert_allclose(features, reference_features,
                                   atol=2e-6, rtol=2e-5)
        activations = np.array([tracker.network(frame) for frame in features])
        np.testing.assert_allclose(activations, reference_activations,
                                   atol=2e-5, rtol=2e-4)

    def test_decoder_forward_state_and_events(self):
        activations = RNNBeatProcessor(online=True, origin='online',
                                       fps=100)(self.audio)
        reference = DBNBeatTrackingProcessor(fps=100, online=True)
        expected_states = reference.hmm.forward(activations)
        expected_beats = reference(activations)
        tracker = self.tracker()
        rows = []
        for index, activation in enumerate(activations):
            rows.append(tracker.decode(activation))
            if index % 37 == 0 or index == len(activations) - 1:
                # NumPy's vector log and scalar libm logf differ by up to one
                # float32 ULP on this audio (4.77e-7 in log density). The
                # forward recursion accumulates this rounding difference.
                np.testing.assert_allclose(tracker.decoder_state,
                                           expected_states[index],
                                           atol=1e-9, rtol=2e-6)
        rows = np.asarray(rows)
        np.testing.assert_array_equal(rows[rows[:, 2] != 0, 0], expected_beats)
        state = tracker.decoder_state
        self.assertAlmostEqual(state.sum(), 1.)
        summary = tracker.summary
        self.assertEqual(summary.shape, (4,))
        self.assertEqual(summary[3], np.argmax(state))
        self.assertEqual(summary[2], state.max())
        # State queries do not change the decoder.
        np.testing.assert_array_equal(tracker.decoder_state, state)

    def test_chunk_boundaries_and_reset(self):
        tracker = self.tracker()
        expected = tracker.push(self.audio)
        for chunk_size in (1, 137, 441, 4096, 44100):
            with self.subTest(chunk_size=chunk_size):
                tracker.reset()
                rows = [tracker.push(self.audio[start:start + chunk_size])
                        for start in range(0, len(self.audio), chunk_size)]
                np.testing.assert_array_equal(np.vstack(rows), expected)
        tracker.reset()
        np.testing.assert_array_equal(tracker.push(self.audio), expected)
        self.assertEqual(tracker.push(np.empty(0, dtype=np.float32)).shape,
                         (0, 4))

    def test_smaller_ensembles_and_simd(self):
        for count in (1, 4):
            with self.subTest(networks=count):
                self.assert_activation_parity(self.audio, networks=count)
        self.assert_activation_parity(self.audio, simd=True)

    def test_quantized_modes_and_reset(self):
        for precision in ('float16', 'int8'):
            with self.subTest(precision=precision):
                tracker = self.tracker(precision=precision)
                rows = tracker.push(self.audio)
                self.assertEqual(rows.shape[1], 4)
                self.assertTrue(np.isfinite(rows).all())
                self.assertTrue(((rows[:, 1] >= 0) & (rows[:, 1] <= 1)).all())
                tracker.reset()
                np.testing.assert_array_equal(tracker.push(self.audio), rows)

    def test_invalid_inputs_do_not_advance_stream(self):
        tracker = self.tracker()
        for audio in (np.zeros((2, 2)), [np.nan], [np.inf]):
            with self.subTest(audio=repr(audio)):
                with self.assertRaises((ValueError, RuntimeError)):
                    tracker.push(audio)
        np.testing.assert_array_equal(tracker.push(self.audio),
                                       self.tracker().push(self.audio))
        for activation in (-0.1, 1.1, np.nan, np.inf):
            with self.assertRaises((ValueError, RuntimeError)):
                tracker.decode(activation)
        for function, values in ((tracker.features, np.zeros(2047)),
                                 (tracker.network, np.zeros(161)),
                                 (tracker.features, np.full(2048, np.nan)),
                                 (tracker.network, np.full(162, np.inf))):
            with self.assertRaises((ValueError, RuntimeError)):
                function(values)

    def test_invalid_configuration_and_asset(self):
        for kwargs in ({'networks': 0}, {'networks': 9},
                       {'precision': 'unknown'}):
            with self.assertRaises((ValueError, RuntimeError)):
                native.NativeBeatTracker(model_path=self.model_path, **kwargs)
        path = os.path.join(self.directory.name, 'truncated.bin')
        with open(path, 'wb') as output:
            output.write(b'MMONL001')
        with self.assertRaises((ValueError, RuntimeError)):
            native.NativeBeatTracker(model_path=path)

    def test_close_and_context_manager(self):
        with native.NativeBeatTracker(model_path=self.model_path) as tracker:
            self.assertEqual(tracker.push(self.audio[:1]).shape, (1, 4))
        tracker.close()
        with self.assertRaises((ValueError, RuntimeError)):
            tracker.push(self.audio[:1])
