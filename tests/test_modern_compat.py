"""Regression tests for supported Python and NumPy versions."""

import sys
from importlib.metadata import version
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import madmom
from madmom.audio.signal import Signal, Stream
from madmom.features.beats_hmm import (
    BeatStateSpace, RNNBeatTrackingObservationModel,
)
from madmom.io.audio import _ffmpeg_call


def test_distribution_keeps_import_name():
    """The renamed distribution must expose the same public package."""
    assert version('madmom-plus') == madmom.__version__


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


def test_resampling_keeps_gain_and_channel_filters():
    """Use one filter chain so resampling does not discard other options."""
    signal = Signal(np.zeros(4, dtype=np.int16), sample_rate=44100)
    call = _ffmpeg_call(signal, 'output.wav', fmt='s16le',
                        sample_rate=22050, num_channels=1, channel=1,
                        replaygain_mode='track', replaygain_preamp=2)
    assert call.count('-af') == 1
    assert call[call.index('-af') + 1] == (
        'volume=replaygain=track:replaygain_preamp=2.0,'
        'pan=mono|c0=c1,aresample=internal_sample_fmt=s16p')


@pytest.mark.parametrize('input_dtype, output_format', [
    (np.int16, 'f32le'), (np.float32, 's16le'), (np.int32, 's16le')])
def test_float_resampling_keeps_ffmpeg_precision(input_dtype, output_format):
    """Higher-precision input and output keep FFmpeg's default precision."""
    signal = Signal(np.zeros(4, dtype=input_dtype), sample_rate=44100)
    call = _ffmpeg_call(signal, 'output.wav', fmt=output_format,
                        sample_rate=22050)
    assert '-af' not in call
