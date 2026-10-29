"""Check that the public decoder APIs use the shared native kernels."""

from unittest import mock

import numpy as np
import pytest

from madmom import _native_decoders
from madmom.features.onsets import OnsetPeakPickingProcessor, peak_picking
from madmom.ml.hmm import (DiscreteObservationModel, HiddenMarkovModel,
                           ObservationModel)
from tests.test_ml_hmm import (CORRECT_FWD, OBS_PROB, OBS_SEQ, PRIOR,
                               TRANSITIONS, TransitionModel)


def _hmm():
    source, destination, probability = zip(*TRANSITIONS)
    transition = TransitionModel.from_dense(destination, source, probability)
    return HiddenMarkovModel(transition, DiscreteObservationModel(OBS_PROB),
                             PRIOR.copy())


def test_forward_uses_native_kernel_and_preserves_upstream_values():
    model = _hmm()
    with mock.patch.object(_native_decoders, 'hmm_forward_address',
                           wraps=_native_decoders.hmm_forward_address) as kernel:
        result = model.forward(OBS_SEQ)
        kernel.assert_called_once()
    np.testing.assert_allclose(result, CORRECT_FWD, atol=5e-9, rtol=5e-8)
    np.testing.assert_array_equal(model._prev, result[-1])
    model.reset()
    chunks = [model.forward(block, reset=False)
              for block in np.array_split(OBS_SEQ, 7)]
    np.testing.assert_array_equal(np.vstack(chunks), result)
    np.testing.assert_array_equal(model.forward([], reset=False),
                                   np.empty((0, 3)))
    np.testing.assert_array_equal(model._prev, result[-1])


def test_forward_generator_uses_independent_native_state():
    model = _hmm()
    expected = model.forward(OBS_SEQ)
    saved = model._prev.copy()
    for block_size in (None, 1, 5, 128):
        with mock.patch.object(_native_decoders, 'hmm_forward_address',
                               wraps=_native_decoders.hmm_forward_address) as kernel:
            result = np.vstack(list(model.forward_generator(
                OBS_SEQ, block_size=block_size)))
            assert kernel.call_count == len(OBS_SEQ)
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(model._prev, saved)


def test_forward_states_matches_dense_argmax_and_posterior():
    model = _hmm()
    dense = model.forward(OBS_SEQ)
    with mock.patch.object(_native_decoders, 'hmm_forward_address',
                           wraps=_native_decoders.hmm_forward_address) as kernel:
        result = model.forward_states(OBS_SEQ)
        kernel.assert_called_once_with(True)
    assert result.dtype == np.uint32
    np.testing.assert_array_equal(result, dense.argmax(axis=1))
    np.testing.assert_array_equal(model._prev, dense[-1])
    model.reset()
    chunks = [model.forward_states(block, reset=False)
              for block in np.array_split(OBS_SEQ, 7)]
    np.testing.assert_array_equal(np.concatenate(chunks), result)
    np.testing.assert_array_equal(model.forward_states([], reset=False),
                                   np.empty(0, dtype=np.uint32))


def test_invalid_transition_rejected_before_previous_state_changes():
    model = _hmm()
    model.transition_model.states[0] = model.transition_model.num_states
    saved = model._prev.copy()
    with pytest.raises(ValueError, match='transition or observation index'):
        model.forward(OBS_SEQ, reset=False)
    np.testing.assert_array_equal(model._prev, saved)


def test_shared_beat_interval_rule_keeps_first_beat_and_exact_boundary():
    flags = np.ones(8, dtype=np.uint8)
    # At 10 fps and 300 BPM the minimum interval is exactly two frames. Frame
    # zero is suppressed by the original DBN rule. Times retain float division.
    events, last, tempo = _native_decoders.beat_events(
        flags, 0, 10., 300., 0., 0.)
    np.testing.assert_array_equal(events, [0.2, 0.4, 0.7])
    assert last == 0.7
    assert tempo == 60. / (0.7 - 0.4)
    events, last, tempo = _native_decoders.beat_events(
        np.array([1, 0, 1], dtype=np.uint8), 8, 10., 300., last, tempo)
    np.testing.assert_array_equal(events, [1.0])
    assert last == 1.0
    assert tempo == 60. / (1.0 - 0.7)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('windows', [(0, 0, 0, 0), (1, 0, 1, 0),
                                      (10, 0, 10, 0), (3, 4, 1, 2),
                                      (0, 30, 40, 0), (40, 20, 2, 5)])
def test_peaks_match_scipy_for_windows_and_zero_boundaries(dtype, windows):
    rng = np.random.default_rng(391)
    for size in (0, 1, 2, 17, 500):
        values = rng.uniform(-0.2, 1.0, size).astype(dtype)
        # A column uses the unchanged SciPy implementation as an independent
        # reference, with the same time-axis boundaries and dtype rounding.
        expected = peak_picking(values[:, None], 0.13, 0, *windows)[0]
        with mock.patch.object(_native_decoders, 'peak_picking',
                               wraps=_native_decoders.peak_picking) as kernel:
            result = peak_picking(values, 0.13, 0, *windows)
            kernel.assert_called_once()
        np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_peak_plateaus_and_threshold_rounding(dtype):
    threshold = 0.3
    middle = dtype(threshold)
    values = np.array([0, middle, np.nextafter(middle, dtype(0)),
                       middle, middle, 1, 1, 0, -1, 0], dtype=dtype)
    for windows in ((0, 0, 2, 0), (2, 0, 1, 0), (2, 3, 1, 1)):
        np.testing.assert_array_equal(
            peak_picking(values, threshold, 0, *windows),
            peak_picking(values[:, None], threshold, 0, *windows)[0],
        )


def test_online_peak_processor_uses_native_and_preserves_reset():
    values = np.array([0.1, 0.8, 0.7, 0.1, 0.9, 0.1, 0.8])
    processor = OnsetPeakPickingProcessor(online=True, threshold=0.5,
                                          pre_max=0.01, fps=100,
                                          combine=0.01)
    with mock.patch.object(_native_decoders, 'peak_picking',
                           wraps=_native_decoders.peak_picking) as kernel:
        expected = processor(values)
        kernel.assert_called_once()
    np.testing.assert_array_equal(expected, [0.01, 0.04, 0.06])
    np.testing.assert_array_equal(processor(values, reset=True), expected)


class ArrayObservationModel(ObservationModel):
    def densities(self, observations):
        return np.asarray(observations, dtype=np.float64)


def test_forward_state_selection_matches_normalized_dense_rows():
    rng = np.random.default_rng(26029)
    for trial in range(100):
        size = int(rng.integers(1, 17))
        transitions = rng.random((size, size))
        transitions /= transitions.sum(axis=0)
        model = TransitionModel(
            np.tile(np.arange(size, dtype=np.uint32), size),
            np.arange(0, size * size + 1, size, dtype=np.uint32),
            transitions.ravel())
        pointers = rng.integers(0, 3, size=size, dtype=np.uint32)
        prior = rng.random(size)
        prior /= prior.sum()
        observations = rng.random((20, 3))
        if trial % 5 == 1:
            observations[5] = 0  # Impossible paths retain upstream NaNs.
        elif trial % 5 == 2:
            observations -= .5  # Custom densities can change normalization sign.
        elif trial % 5 == 3:
            observations[5, 0] = np.nan
        elif trial % 5 == 4:
            observations[5, 1] = np.inf
        hmm = HiddenMarkovModel(model, ArrayObservationModel(pointers), prior)
        dense = hmm.forward(observations)
        selected = hmm.forward_states(observations)
        np.testing.assert_array_equal(selected, dense.argmax(axis=1))
        np.testing.assert_array_equal(hmm._prev, dense[-1])
    # Equal normalized rows must choose the first state.
    model = TransitionModel(np.arange(3, dtype=np.uint32),
                            np.arange(4, dtype=np.uint32), np.ones(3))
    hmm = HiddenMarkovModel(model, ArrayObservationModel(
        np.zeros(3, dtype=np.uint32)), np.full(3, 1. / 3))
    np.testing.assert_array_equal(hmm.forward_states(np.ones((4, 1))), 0)
