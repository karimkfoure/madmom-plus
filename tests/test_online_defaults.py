"""Direct online calls must not require command-line-only options."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from madmom.audio.signal import Signal
from madmom.processors import process_online


def test_online_file_omits_device_listing_flag():
    frames = []
    data = Signal(np.arange(8, dtype=np.float32), sample_rate=44100)
    with pytest.warns(UserWarning, match='frame_size'):
        process_online(lambda frame, output: frames.append(frame.copy()),
                       data, None, frame_size=4, hop_size=2, fps=None)
    assert len(frames) == 4
    assert all(frame.shape == (4,) for frame in frames)


def test_online_explicit_device_listing(monkeypatch, capsys):
    audio = Mock()
    audio.get_device_count.return_value = 1
    audio.get_device_info_by_index.return_value = {'index': 0, 'name': 'Input'}
    monkeypatch.setitem(sys.modules, 'pyaudio',
                        SimpleNamespace(PyAudio=lambda: audio))
    with pytest.raises(SystemExit) as exit_info:
        process_online(None, None, None, list_stream_input_device=True)
    assert exit_info.value.code == 0
    assert capsys.readouterr().out == '0: Input\n'
