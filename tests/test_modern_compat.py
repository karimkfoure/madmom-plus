"""Regression tests for supported Python and NumPy versions."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from madmom.audio.signal import Stream
from madmom.features.beats_hmm import (
    BeatStateSpace, RNNBeatTrackingObservationModel,
)


@pytest.mark.parametrize('observations', [0.25, [0.25], np.array([0.25])])
def test_beat_observation_accepts_scalar_and_array(observations):
    """Online scalar input must produce the same densities as batch input."""
    model = RNNBeatTrackingObservationModel(BeatStateSpace(1, 4), 4)
    np.testing.assert_allclose(
        model.log_densities(observations), [[np.log(0.25), np.log(0.25)]])


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_stream_decodes_binary_audio(monkeypatch, dtype):
    """Decode microphone bytes without requiring a physical audio device."""
    samples = np.array([0.25, -0.5], dtype=np.float32)
    audio = Mock()
    audio.open.return_value.read.return_value = samples.tobytes()
    monkeypatch.setitem(sys.modules, 'pyaudio', SimpleNamespace(
        PyAudio=lambda: audio, paFloat32=1, paComplete=2, paContinue=3))
    stream = Stream(sample_rate=8000, num_channels=1, dtype=dtype,
                    frame_size=4, hop_size=2, fps=None)
    try:
        first = next(stream)
        second = next(stream)
        np.testing.assert_array_equal(first, [0, 0, 0.25, -0.5])
        np.testing.assert_array_equal(second, [0.25, -0.5, 0.25, -0.5])
        assert first.dtype == dtype
        assert second.dtype == dtype
        assert first.sample_rate == 8000
        assert first.start == 0
        assert second.start == 2 / 8000
    finally:
        stream.close()
