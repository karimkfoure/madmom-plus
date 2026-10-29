# encoding: utf-8
"""Tests for optional meter priors in offline and causal downbeat tracking."""

import pickle
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from madmom.features.downbeats import (DBNDownBeatTrackingProcessor,
                                      OnlineDBNDownBeatTrackingProcessor)


def ambiguous_activations():
    """Return a beat grid with identical beat and downbeat evidence."""
    data = np.full((600, 2), 1e-4)
    data[::25] = [0.45, 0.45]
    return data


class TestDownBeatMeterPriors(unittest.TestCase):

    def make_processor(self, cls, **kwargs):
        parameters = dict(beats_per_bar=[3, 4], fps=50,
                          min_bpm=120, max_bpm=120, num_tempi=5)
        if cls is OnlineDBNDownBeatTrackingProcessor:
            parameters['meter_change_probability'] = 0.
        parameters.update(kwargs)
        return cls(**parameters)

    def test_equal_weights_preserve_exact_default_output(self):
        data = ambiguous_activations()
        for cls in [DBNDownBeatTrackingProcessor,
                    OnlineDBNDownBeatTrackingProcessor]:
            default = self.make_processor(cls)
            expected = default(data)
            for weights in [[1., 1.], np.array([7., 7.]),
                            [1e308, 1e308], [1e-320, 1e-320]]:
                with self.subTest(cls=cls, weights=weights):
                    weighted = self.make_processor(
                        cls, beats_per_bar_weights=weights)
                    assert_array_equal(weighted(data), expected)
                    assert_array_equal(weighted.beats_per_bar_weights,
                                       [0.5, 0.5])
                    if cls is OnlineDBNDownBeatTrackingProcessor:
                        assert_array_equal(
                            weighted.hmm.initial_distribution,
                            default.hmm.initial_distribution)
        # Preserve the original uniform state prior even for three meters,
        # where (1 / count) / states can round differently from
        # 1 / (count * states).
        processor = self.make_processor(OnlineDBNDownBeatTrackingProcessor,
                                        beats_per_bar=[3, 4, 5],
                                        beats_per_bar_weights=[3., 3., 3.])
        expected = np.concatenate([
            np.full(space.num_states, 1. / (3 * space.num_states))
            for space in processor.st.state_spaces])
        assert_array_equal(processor.hmm.initial_distribution, expected)

    def test_ambiguous_meter_obeys_preference(self):
        data = ambiguous_activations()
        for cls in [DBNDownBeatTrackingProcessor,
                    OnlineDBNDownBeatTrackingProcessor]:
            for weights, meter in [([9., 1.], 3), ([1., 9.], 4)]:
                with self.subTest(cls=cls, weights=weights):
                    processor = self.make_processor(
                        cls, beats_per_bar_weights=weights)
                    result = processor(data)
                    if cls is OnlineDBNDownBeatTrackingProcessor:
                        # Equal accent evidence leaves bar phase ambiguous;
                        # test the selected meter, not its unidentifiable
                        # beat-number phase.
                        self.assertEqual(processor.meter, meter)
                    else:
                        self.assertEqual(np.max(result[:, 1]), meter)

    def test_zero_weights_exclude_meters(self):
        data = ambiguous_activations()
        for cls in [DBNDownBeatTrackingProcessor,
                    OnlineDBNDownBeatTrackingProcessor]:
            for weights, meter in [([1., 0.], 3), ([0., 1.], 4)]:
                with self.subTest(cls=cls, weights=weights):
                    parameters = dict(beats_per_bar_weights=weights)
                    if cls is OnlineDBNDownBeatTrackingProcessor:
                        parameters['meter_change_probability'] = 0.25
                    excluded = self.make_processor(cls, **parameters)
                    single = self.make_processor(cls, beats_per_bar=[meter])
                    assert_array_equal(excluded(data), single(data))
                    if cls is OnlineDBNDownBeatTrackingProcessor:
                        self.assertEqual(excluded.st.num_patterns, 1)
                        self.assertEqual(excluded.meter, meter)

    def test_initial_online_probability_matches_meter_mass(self):
        processor = self.make_processor(OnlineDBNDownBeatTrackingProcessor,
                                        beats_per_bar_weights=[2., 6.])
        prior = processor.hmm.initial_distribution
        for pattern, mass in enumerate([0.25, 0.75]):
            assert_allclose(prior[processor.st.state_patterns == pattern].sum(),
                            mass, atol=1e-15)
        processor([1. / 17, 1. / 17])
        self.assertEqual(processor.meter, 4)
        self.assertAlmostEqual(processor.meter_probability, 0.75)
        processor.reset()
        assert_array_equal(processor.hmm._prev, prior)

    def test_weights_do_not_alias_callers_array(self):
        for cls in [DBNDownBeatTrackingProcessor,
                    OnlineDBNDownBeatTrackingProcessor]:
            supplied = np.array([2., 6.])
            processor = self.make_processor(cls,
                                            beats_per_bar_weights=supplied)
            assert_array_equal(supplied, [2., 6.])
            supplied[:] = 0.
            assert_array_equal(processor.beats_per_bar_weights, [0.25, 0.75])

    def test_invalid_vectors_fail_before_filter_or_pool_creation(self):
        vectors = [[], [1.], [1., 2., 3.], [-1., 2.], [0., 0.],
                   [np.nan, 1.], [np.inf, 1.], [-np.inf, 1.],
                   [[1., 2.]], [[1.], [2.]], [[1.], [2., 3.]],
                   1., np.array(1.), ['invalid', 1.], [1j, 1.],
                   np.array([None, 1.], dtype=object)]
        for cls in [DBNDownBeatTrackingProcessor,
                    OnlineDBNDownBeatTrackingProcessor]:
            for weights in vectors:
                with self.subTest(cls=cls, weights=weights):
                    with patch('madmom.features.downbeats.HiddenMarkovModel') as hmm:
                        with patch('multiprocessing.Pool') as pool:
                            with self.assertRaises(ValueError):
                                self.make_processor(
                                    cls, beats_per_bar_weights=weights)
                            hmm.assert_not_called()
                            pool.assert_not_called()

    def test_invalid_offline_replacement_fails_before_decoding(self):
        processor = self.make_processor(DBNDownBeatTrackingProcessor)
        processor.beats_per_bar_weights = np.array([0., 0.])
        with patch.object(processor.hmms[0], 'viterbi') as viterbi:
            with self.assertRaises(ValueError):
                processor(ambiguous_activations())
            viterbi.assert_not_called()

    def test_offline_scores_do_not_require_equal_path_lengths(self):
        data = ambiguous_activations()
        processor = self.make_processor(DBNDownBeatTrackingProcessor,
                                        beats_per_bar_weights=[3., 1.])
        expected = self.make_processor(DBNDownBeatTrackingProcessor,
                                      beats_per_bar=[3])(data)
        with patch.object(processor.hmms[1], 'viterbi',
                          return_value=(np.empty(0, dtype=int), -np.inf)):
            assert_array_equal(processor(data), expected)

    def test_zero_prior_cannot_win_when_model_scores_are_infinite(self):
        data = ambiguous_activations()
        processor = self.make_processor(DBNDownBeatTrackingProcessor,
                                        beats_per_bar_weights=[0., 1.],
                                        threshold=0.)
        paths = [hmm.viterbi(data)[0] for hmm in processor.hmms]
        expected = self.make_processor(DBNDownBeatTrackingProcessor,
                                      beats_per_bar=[4], threshold=0.)(data)
        with patch.object(processor.hmms[0], 'viterbi',
                          return_value=(paths[0], -np.inf)):
            with patch.object(processor.hmms[1], 'viterbi',
                              return_value=(paths[1], -np.inf)):
                assert_array_equal(processor(data), expected)

    def test_online_evidence_can_overcome_a_positive_initial_prior(self):
        data = np.full((1200, 2), 1e-4)
        data[::25, 0] = 0.98
        data[::100, 0] = 1e-4
        data[::100, 1] = 0.98
        processor = self.make_processor(
            OnlineDBNDownBeatTrackingProcessor,
            beats_per_bar_weights=[999., 1.], meter_change_probability=0.01)
        processor(data)
        self.assertEqual(processor.meter, 4)
        self.assertGreater(processor.meter_probability, 0.99)

    def test_legacy_positional_constructor_arguments(self):
        data = ambiguous_activations()
        offline = DBNDownBeatTrackingProcessor(
            [3, 4], 120, 120, 5, 100, 16, 0.05, True, 50)
        online = OnlineDBNDownBeatTrackingProcessor(
            [3, 4], 120, 120, 5, 100, 16, 0., 0.05, 50)
        for processor in [offline, online]:
            assert_array_equal(processor(data),
                               self.make_processor(type(processor))(data))

    def test_serialization_and_reset_preserve_priors(self):
        data = ambiguous_activations()
        for cls in [DBNDownBeatTrackingProcessor,
                    OnlineDBNDownBeatTrackingProcessor]:
            for weights in [[1., 9.], [0., 1.]]:
                with self.subTest(cls=cls, weights=weights):
                    processor = self.make_processor(
                        cls, beats_per_bar_weights=weights)
                    expected = processor(data)
                    restored = pickle.loads(pickle.dumps(processor, protocol=2))
                    assert_array_equal(restored.beats_per_bar_weights,
                                       processor.beats_per_bar_weights)
                    assert_array_equal(restored(data), expected)
                    if cls is OnlineDBNDownBeatTrackingProcessor:
                        restored.reset()
                        assert_array_equal(restored.hmm._prev,
                                           processor.hmm.initial_distribution)
                        assert_array_equal(restored(data, reset=False), expected)

    def test_old_saved_processors_need_no_prior_attributes(self):
        data = ambiguous_activations()
        for cls in [DBNDownBeatTrackingProcessor,
                    OnlineDBNDownBeatTrackingProcessor]:
            processor = self.make_processor(cls)
            expected = processor(data)
            del processor.beats_per_bar_weights
            if cls is OnlineDBNDownBeatTrackingProcessor:
                del processor._active_beats_per_bar
            restored = pickle.loads(pickle.dumps(processor, protocol=2))
            assert_array_equal(restored(data), expected)


if __name__ == '__main__':
    unittest.main()
