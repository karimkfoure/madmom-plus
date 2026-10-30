# encoding: utf-8
"""Tests for causal forward-only bar tracking."""

import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from madmom.features.downbeats import OnlineDBNDownBeatTrackingProcessor
from madmom.ml.hmm import HiddenMarkovModel
from madmom.processors import Processor


def activations(meter, bars=12):
    """Return unambiguous 120 bpm pulses at 50 frames per second."""
    data = np.full((bars * meter * 25, 2), 1e-4)
    indices = np.arange(0, len(data), 25)
    data[indices, 0] = 0.98
    data[indices[::meter], 0] = 1e-4
    data[indices[::meter], 1] = 0.98
    return data


class TestOnlineDBNDownBeatTrackingProcessor(unittest.TestCase):

    def make_processor(self, **kwargs):
        parameters = dict(beats_per_bar=[3, 4], fps=50,
                          min_bpm=110, max_bpm=130, num_tempi=5)
        parameters.update(kwargs)
        return OnlineDBNDownBeatTrackingProcessor(**parameters)

    def test_synthetic_meter_and_beat_numbers(self):
        for meter in [3, 4]:
            with self.subTest(meter=meter):
                processor = self.make_processor()
                self.assertIsInstance(processor, Processor)
                data = activations(meter)
                result = processor(data)
                # Initial meter acquisition may revise the state, but the
                # final eight bars must have the supplied phase and meter.
                settled = result[result[:, 0] >= 4 * meter * 0.5]
                expected = np.column_stack((
                    np.arange(4 * meter, 12 * meter) * 0.5,
                    np.tile(np.arange(1, meter + 1), 8)))
                assert_array_equal(settled, expected)
                self.assertEqual(processor.meter, meter)
                self.assertGreater(processor.meter_probability, 0.9)
                self.assertAlmostEqual(processor.period, 0.5)

    def test_chunks_and_single_frames_match(self):
        data = activations(4, bars=5)
        whole = self.make_processor()(data)
        for chunk_size in [1, 7, 64, 113]:
            with self.subTest(chunk_size=chunk_size):
                processor = self.make_processor()
                rows = []
                for start in range(0, len(data), chunk_size):
                    rows.append(processor(data[start:start + chunk_size],
                                          reset=False))
                    self.assertEqual(processor(
                        np.empty((0, 2)), reset=False).shape, (0, 2))
                assert_array_equal(np.vstack(rows), whole)
                self.assertEqual(processor.counter, len(data))

    def test_prefix_cannot_depend_on_future_input(self):
        data = activations(3, bars=5)
        stop = 237
        processor = self.make_processor()
        prefix = processor(data[:stop])
        altered = data.copy()
        altered[stop:] = [0.1, 0.8]
        for complete in [data, altered]:
            result = self.make_processor()(complete)
            assert_array_equal(result[result[:, 0] < stop / 50.], prefix)

    def test_reset_and_empty(self):
        processor = self.make_processor()
        data = activations(4, bars=3)
        result = processor(data)
        assert_array_equal(processor(data), result)
        processor.reset()
        self.assertEqual(processor.counter, 0)
        self.assertIsNone(processor.meter)
        self.assertTrue(np.isnan(processor.period))
        self.assertEqual(processor(np.empty((0, 2))).shape, (0, 2))
        assert_array_equal(processor(data, reset=False), result)

    def test_online_alias_and_single_frame(self):
        processor = self.make_processor()
        result = processor.process_online([0., 1.])
        assert_array_equal(result, [[0., 1.]])
        self.assertEqual(processor.counter, 1)

    def test_exact_tempo_refractory_uses_frames(self):
        processor = self.make_processor(beats_per_bar=4, min_bpm=120,
                                        max_bpm=120)
        data = np.vstack((np.full((7, 2), 1e-4), activations(4, bars=5)))
        result = processor(data)
        assert_array_equal(result[:, 0], (np.arange(20) * 25 + 7) / 50.)

    def test_boundary_probabilities_and_absence_are_finite(self):
        processor = self.make_processor()
        data = np.tile([[0., 0.], [0., 1.], [1., 0.]], (100, 1))
        result = processor(data)
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertTrue(np.isfinite(processor.period))
        self.assertTrue(0 <= processor.meter_probability <= 1. + 1e-12)
        self.assertEqual(processor(np.zeros((200, 2))).shape, (0, 2))
        self.assertEqual(processor.counter, 200)

    def test_threshold_does_not_stop_clock_or_filter(self):
        data = activations(4, bars=3)
        low = self.make_processor(threshold=0.)
        high = self.make_processor(threshold=1.)
        low(data)
        self.assertEqual(high(data).shape, (0, 2))
        self.assertEqual(high.counter, len(data))
        assert_allclose(high.hmm._prev, low.hmm._prev, rtol=0, atol=0)

    def test_forward_memory_is_bounded_and_viterbi_is_unused(self):
        processor = self.make_processor()
        lengths = []
        forward = processor.hmm.forward

        def record_forward(data, **kwargs):
            lengths.append(len(data))
            return forward(data, **kwargs)

        with patch.object(HiddenMarkovModel, 'viterbi',
                          side_effect=AssertionError('Noncausal decoding')):
            with patch.object(processor.hmm, 'forward',
                              side_effect=record_forward):
                processor(activations(4, bars=4))
        self.assertGreater(len(lengths), 1)
        self.assertLessEqual(max(lengths), 64)

    def test_invalid_activations_do_not_change_stream(self):
        processor = self.make_processor()
        processor(activations(4, bars=1))
        before = processor.hmm._prev.copy()
        counter = processor.counter
        invalid = [np.zeros(3), np.zeros((2, 3)), [[-0.1, 0.2]],
                   [[np.nan, 0.2]], [[np.inf, 0.]], [[0.8, 0.8]],
                   [[1.1, 0.]]]
        for data in invalid:
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    processor(data)
                self.assertEqual(processor.counter, counter)
                assert_array_equal(processor.hmm._prev, before)

    def test_invalid_parameters(self):
        invalid = [dict(beats_per_bar=[]), dict(beats_per_bar=[3, 3]),
                   dict(beats_per_bar=[0, 4]), dict(beats_per_bar=[3.5]),
                   dict(beats_per_bar=[[3, 4]]), dict(fps=None), dict(fps=0),
                   dict(min_bpm=0), dict(max_bpm=100), dict(max_bpm=4000),
                   dict(observation_lambda=1), dict(transition_lambda=-1),
                   dict(threshold=1.1), dict(meter_change_probability=1),
                   dict(num_tempi=0), dict(num_tempi=1.5)]
        for parameters in invalid:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    self.make_processor(**parameters)


if __name__ == '__main__':
    unittest.main()
