"""Check native FFT magnitude across finite input scales."""

import numpy as np
import pytest

from madmom.audio._native import NativeBeatFrontend
from madmom.audio.signal import Signal

from .test_native_frontend_compat import processors


@pytest.mark.parametrize('scale', [0., 1e-30, 1e-20, 1., 1e15, 1e30])
def test_fft_magnitude_preserves_small_and_large_finite_inputs(scale):
    from madmom.processors import SequentialProcessor

    signal = Signal(np.asarray(np.sin(np.arange(4410) * .073) * scale,
                               dtype=np.float32), sample_rate=44100)
    native = NativeBeatFrontend(processors(streaming=False))
    reference = SequentialProcessor(processors(streaming=False))
    np.testing.assert_allclose(native(signal), reference(signal),
                               atol=2e-6, rtol=2e-5)
