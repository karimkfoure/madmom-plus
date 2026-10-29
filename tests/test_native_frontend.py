"""Check standard online frontend dispatch, state, and serialization."""

import pickle
from unittest import mock

import numpy as np

from madmom.audio.signal import FramedSignal, Signal
from madmom.features.beats import RNNBeatProcessor

from . import AUDIO_PATH


def frontend(**kwargs):
    return RNNBeatProcessor(online=True, origin='online', fps=100,
                            num_threads=1, **kwargs)[0][1][0]


def test_standard_frontend_uses_native_kernel():
    processor = frontend()
    audio = Signal(AUDIO_PATH + '/sample.wav', sample_rate=44100,
                   num_channels=1, dtype=np.float32)
    with mock.patch.object(processor._lib, 'mm_frontend_process',
                           wraps=processor._lib.mm_frontend_process) as kernel:
        result = processor(audio)
    kernel.assert_called()
    assert result.shape == (281, 162)
    assert processor._handle


def test_frontend_stream_reset_and_pickle():
    audio = Signal(AUDIO_PATH + '/sample.wav', sample_rate=44100,
                   num_channels=1, dtype=np.float32)
    reference = frontend()(audio)
    stream = frontend(num_frames=1)
    stream[0].origin = 'stream'
    frames = FramedSignal(audio, frame_size=2048, hop_size=441, origin='online')
    result = np.vstack([stream(frame, reset=index == 0)
                        for index, frame in enumerate(frames)])
    np.testing.assert_array_equal(result, reference)
    stream[4].reset()
    np.testing.assert_array_equal(stream(frames[0], reset=False), reference[:1])
    restored = pickle.loads(pickle.dumps(stream))
    np.testing.assert_array_equal(restored(frames[0]), reference[:1])


def test_integer_pcm_scaling_matches_float_pcm():
    audio = Signal(AUDIO_PATH + '/sample.wav', sample_rate=44100, num_channels=1)
    normalized = Signal(np.asarray(audio, dtype=np.float32) /
                        np.iinfo(audio.dtype).max, sample_rate=44100)
    np.testing.assert_allclose(frontend()(audio), frontend()(normalized),
                               atol=2e-6, rtol=2e-5)
