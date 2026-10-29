"""Keep stereo downmix gain independent of the audio loader."""

from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from madmom.audio.signal import Signal, remix
from madmom.io.audio import (
    decode_to_memory, load_audio_file, load_ffmpeg_file, load_wave_file,
)

from . import AUDIO_PATH


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int16])
@pytest.mark.parametrize('input_type', ['filename', 'signal', 'file_object'])
def test_stereo_mono_matches_wave_loader(tmp_path, dtype, input_type):
    """Both integer and float output must preserve arithmetic-mean gain."""
    if dtype == np.int16:
        stereo = np.array([[2, 4], [-2, -4], [-2, 4], [30000, 30000]],
                          dtype=dtype)
    else:
        stereo = np.array([[0.25, 0.5], [-0.25, 0.5], [-0.5, -0.75],
                           [0.875, 0.875]], dtype=dtype)
    filename = tmp_path / 'stereo.wav'
    wavfile.write(filename, 8000, stereo)
    reference, _ = load_wave_file(filename, num_channels=1, dtype=dtype)
    if input_type == 'filename':
        source = str(filename)
    elif input_type == 'signal':
        source = Signal(stereo, sample_rate=8000)
    else:
        source = filename.open('rb')
    try:
        actual, sample_rate = load_ffmpeg_file(source, num_channels=1,
                                               dtype=dtype)
    finally:
        if input_type == 'file_object':
            source.close()
    assert actual.dtype == dtype
    assert sample_rate == 8000
    np.testing.assert_array_equal(actual, reference)


def test_dtype_conversion_does_not_change_mono_gain(tmp_path):
    """A float dtype change must not add FFmpeg's stereo downmix gain."""
    stereo = np.array([[0.25, 0.5], [-0.25, 0.5], [0.875, 0.875]],
                      dtype=np.float32)
    filename = tmp_path / 'float-stereo.wav'
    wavfile.write(filename, 8000, stereo)
    wave, _ = load_audio_file(str(filename), num_channels=1, dtype=np.float32)
    converted, _ = load_audio_file(str(filename), num_channels=1,
                                   dtype=np.float64)
    np.testing.assert_array_equal(converted, wave)


@pytest.mark.parametrize('channel', [0, 1])
@pytest.mark.parametrize('num_channels', [None, 1])
def test_explicit_channel_stays_mono(tmp_path, channel, num_channels):
    """An explicit channel must be returned without averaging or reshaping."""
    stereo = np.array([[0.25, 0.5], [-0.25, 0.5], [0.875, 0.125]],
                      dtype=np.float32)
    filename = tmp_path / 'channels.wav'
    wavfile.write(filename, 8000, stereo)
    actual, _ = load_ffmpeg_file(str(filename), channel=channel,
                                 num_channels=num_channels, dtype=np.float32)
    np.testing.assert_array_equal(actual, stereo[:, channel])


def test_mono_mix_preserves_replaygain_and_resampling():
    """Downmix the decoded channels after all requested audio operations."""
    filename = str(Path(AUDIO_PATH) / 'stereo_sample_rg.flac')
    options = dict(sample_rate=22050, replaygain_mode='track', dtype=np.float32)
    stereo, sample_rate = load_ffmpeg_file(filename, num_channels=2, **options)
    mono, mono_rate = load_ffmpeg_file(filename, num_channels=1, **options)
    assert sample_rate == mono_rate == 22050
    np.testing.assert_array_equal(mono, remix(stereo, 1))


def test_multichannel_input_keeps_ffmpeg_mix_policy():
    """The stereo repair must not change the existing surround mix policy."""
    source = Signal(np.array([[0.25, 0.5, 0.125], [-0.25, 0.5, 0.125]],
                             dtype=np.float32), sample_rate=8000)
    expected = np.frombuffer(decode_to_memory(source, num_channels=1),
                             dtype=np.float32)
    actual, _ = load_ffmpeg_file(source, num_channels=1, dtype=np.float32)
    np.testing.assert_array_equal(actual, expected)
