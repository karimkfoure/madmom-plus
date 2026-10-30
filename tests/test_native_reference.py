"""Compare the required native core with frozen madmom 0.16.1 outputs."""

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from madmom.audio.signal import FramedSignal, Signal
from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor
from madmom.features.onsets import OnsetPeakPickingProcessor, peak_picking
from madmom.models import BEATS_LSTM
from madmom.native import NativeBeatTracker, export_model


DATA = Path(__file__).resolve().parent / "data"
REFERENCE = DATA / "native_0161"


class TestUpstreamReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = json.loads((REFERENCE / "metadata.json").read_text())

    @staticmethod
    def signal(clip):
        return Signal(str(DATA / "audio" / clip["audio"]), sample_rate=44100,
                      num_channels=1, dtype=np.float32)

    def test_reference_and_audio_hashes(self):
        self.assertEqual(self.metadata["source"]["commit"],
                         "6545fbb3ecdae0aac336a95ec591f642e0b74f3e")
        for clip in self.metadata["clips"]:
            for path, expected in (
                    (DATA / "audio" / clip["audio"], clip["audio_sha256"]),
                    (REFERENCE / clip["fixture"], clip["fixture_sha256"])):
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)
        for model in BEATS_LSTM:
            name = "madmom/models/beats/2016/" + Path(model).name
            self.assertEqual(hashlib.sha256(Path(model).read_bytes()).hexdigest(),
                             self.metadata["models"]["sha256"][name])
        synthetic = self.metadata["synthetic"]
        self.assertEqual(hashlib.sha256((REFERENCE / synthetic["fixture"]).read_bytes()).hexdigest(),
                         synthetic["fixture_sha256"])

    def test_standard_python_api_frontend_and_ensemble(self):
        for clip in self.metadata["clips"]:
            with self.subTest(audio=clip["audio"]), np.load(REFERENCE / clip["fixture"]) as frozen:
                processor = RNNBeatProcessor(online=True, origin="online", fps=100,
                                             num_threads=1)
                features = processor[0](self.signal(clip), reset=True)
                self.assertEqual(features.shape, frozen["features"].shape)
                self.assertEqual(features.dtype, np.float32)
                np.testing.assert_allclose(features, frozen["features"],
                                           atol=2e-6, rtol=2e-5)
                activations = processor[1](features, reset=True)
                self.assertEqual(activations.shape, frozen["activations"].shape)
                self.assertEqual(activations.dtype, np.float32)
                np.testing.assert_allclose(activations,
                                           frozen["activations"], atol=2e-5, rtol=2e-4)
                # Check each model so ensemble averaging cannot hide a layer error.
                for index, network in enumerate(processor[1][0]):
                    np.testing.assert_allclose(
                        network(frozen["features"], reset=True),
                        frozen["individual_activations"][index], atol=2e-5, rtol=2e-4)

    def test_default_origin_and_frame_streaming(self):
        for clip in self.metadata["clips"]:
            with self.subTest(audio=clip["audio"]), np.load(REFERENCE / clip["fixture"]) as frozen:
                signal = self.signal(clip)
                centered = RNNBeatProcessor(online=True, fps=100, num_threads=1)
                np.testing.assert_allclose(centered(signal), frozen["centered_activations"],
                                           atol=2e-5, rtol=2e-4)
                stream = RNNBeatProcessor(online=True, origin="stream", fps=100,
                                          num_frames=1, num_threads=1)
                frames = FramedSignal(signal, frame_size=2048, hop_size=441,
                                      origin="online")
                activations = []
                for index, frame in enumerate(frames):
                    activation = stream(frame, reset=index == 0)
                    self.assertEqual(activation.shape, (1,))
                    self.assertEqual(activation.dtype, np.float32)
                    activations.append(activation[0])
                actual = np.array(activations)
                np.testing.assert_allclose(actual, frozen["streamed_activations"],
                                           atol=2e-5, rtol=2e-4)

    def test_forward_posteriors_and_beats(self):
        for clip in self.metadata["clips"]:
            with self.subTest(audio=clip["audio"]), np.load(REFERENCE / clip["fixture"]) as frozen:
                decoder = DBNBeatTrackingProcessor(online=True, fps=100)
                forward = decoder.hmm.forward(frozen["activations"], reset=True)
                np.testing.assert_allclose(forward[frozen["forward_indices"]],
                                           frozen["forward"], atol=1e-9, rtol=2e-6)
                np.testing.assert_array_equal(np.argmax(forward, axis=1),
                                              frozen["forward_argmax"])
                np.testing.assert_allclose(np.max(forward, axis=1),
                                           frozen["forward_max"], atol=1e-9, rtol=2e-6)
                np.testing.assert_array_equal(decoder(frozen["activations"], reset=True),
                                              frozen["beats"])
                # Chunking must preserve state and events.
                beats = []
                for first in range(0, len(frozen["activations"]), 13):
                    beats.extend(decoder(frozen["activations"][first:first + 13],
                                         reset=first == 0))
                np.testing.assert_array_equal(beats, frozen["beats"])

    def test_online_peak_picking(self):
        for clip in self.metadata["clips"]:
            with self.subTest(audio=clip["audio"]), np.load(REFERENCE / clip["fixture"]) as frozen:
                picker = OnsetPeakPickingProcessor(online=True, fps=100, threshold=.2,
                                                   pre_avg=.1, pre_max=.03, combine=.05)
                np.testing.assert_array_equal(picker(frozen["activations"], reset=True),
                                              frozen["peaks"])
                events = []
                for first in range(0, len(frozen["activations"]), 13):
                    events.extend(picker(frozen["activations"][first:first + 13],
                                          reset=first == 0))
                np.testing.assert_array_equal(events, frozen["streamed_peaks"])
        with np.load(REFERENCE / "synthetic.npz") as frozen:
            for pre_avg in (0, 3, 8):
                for pre_max in (0, 1, 4):
                    np.testing.assert_array_equal(
                        peak_picking(frozen["activations"], .2, 0, pre_avg, 0, pre_max, 0),
                        frozen["peaks_avg%d_max%d" % (pre_avg, pre_max)])

    def test_native_stream_and_components(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            asset = str(Path(directory) / "online.bin")
            export_model(asset)
            for clip in self.metadata["clips"]:
                with self.subTest(audio=clip["audio"]), np.load(REFERENCE / clip["fixture"]) as frozen:
                    signal = self.signal(clip)
                    with NativeBeatTracker(model_path=asset, simd=True) as tracker:
                        rows = tracker.push(signal)
                        np.testing.assert_allclose(rows[:, 1], frozen["activations"],
                                                   atol=2e-5, rtol=2e-4)
                        np.testing.assert_allclose(rows[rows[:, 2] != 0, 0], frozen["beats"],
                                                   atol=.01, rtol=0)
                        np.testing.assert_array_equal(rows[:, 0], np.arange(len(rows)) / 100.)
                        tracker.reset()
                        pieces = [tracker.push(signal[first:first + 137])
                                  for first in range(0, len(signal), 137)]
                        np.testing.assert_array_equal(np.vstack(pieces), rows)
                        tracker.reset()
                        frames = FramedSignal(signal, frame_size=2048, hop_size=441,
                                              origin="online")
                        features = np.vstack([tracker.features(frame) for frame in frames])
                        np.testing.assert_allclose(features, frozen["features"],
                                                   atol=2e-6, rtol=2e-5)
                        # Use frozen activations to isolate the forward kernel.
                        tracker.reset()
                        selected = set(frozen["forward_indices"].tolist())
                        posteriors = []
                        for index, activation in enumerate(frozen["activations"]):
                            tracker.decode(activation)
                            if index in selected:
                                posteriors.append(tracker.decoder_state)
                        np.testing.assert_allclose(posteriors, frozen["forward"],
                                                   atol=1e-9, rtol=2e-6)
