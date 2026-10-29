"""Keep legacy and current processor pickles usable with current NumPy."""

import io
import pickle
import warnings
from functools import partial

import numpy as np
import pytest

from madmom.processors import Processor


class LegacyDtype:
    """Produce the integer align arguments used by older NumPy pickles."""

    def __init__(self, align):
        self.align = align

    def __reduce__(self):
        return np.dtype, ([('x', 'i1'), ('y', 'i4')], self.align)


class LegacyKeywordDtype(LegacyDtype):
    def __reduce__(self):
        return partial(np.dtype, align=self.align), ([('x', 'i1'), ('y', 'i4')],)


@pytest.mark.parametrize('align', [0, 1])
@pytest.mark.parametrize('protocol', [0, 2, pickle.HIGHEST_PROTOCOL])
@pytest.mark.parametrize('legacy_type', [LegacyDtype, LegacyKeywordDtype])
def test_load_legacy_dtype_without_alignment_warning(align, protocol,
                                                    legacy_type):
    payload = pickle.dumps(legacy_type(align), protocol=protocol)
    dtype_constructor = np.dtype
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        dtype = Processor.load(io.BytesIO(payload))
    assert dtype == np.dtype([('x', 'i1'), ('y', 'i4')], align=bool(align))
    assert dtype.isalignedstruct == bool(align)
    assert np.dtype is dtype_constructor


def test_load_python2_string_uses_latin1():
    # Protocol 2 SHORT_BINSTRING represents a Python 2 str with a high byte.
    assert Processor.load(io.BytesIO(b'\x80\x02U\x01\xff.')) == '\xff'


def test_current_processor_round_trip(tmp_path):
    processor = Processor()
    processor.weights = np.arange(6, dtype=np.float32).reshape(2, 3)
    processor.record_type = np.dtype([('x', 'i1'), ('y', 'i4')], align=True)
    path = str(tmp_path / 'processor.pkl')
    processor.dump(path)
    restored = Processor.load(path)
    assert isinstance(restored, Processor)
    np.testing.assert_array_equal(restored.weights, processor.weights)
    assert restored.weights.dtype == processor.weights.dtype
    assert restored.record_type == processor.record_type
    assert restored.record_type.isalignedstruct


def test_load_preserves_type_errors():
    # Do not retry from the middle of a stream when reconstruction fails.
    with pytest.raises(TypeError):
        Processor.load(io.BytesIO(b'\x80\x02cbuiltins\nlen\n)R.'))
