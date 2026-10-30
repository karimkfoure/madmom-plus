# encoding: utf-8
"""Tests for bounded online beat decoding."""

import unittest
from unittest.mock import patch

import numpy as np

from madmom.features.beats import (DBNBeatTrackingProcessor,
                                   BeatTrackingProcessor)


class TestOnlineBeatStreaming(unittest.TestCase):

    def setUp(self):
        self.activations = np.random.RandomState(17).uniform(0.001, 0.03, 789)
        self.activations[::50] = 0.9

    def processor(self):
        return DBNBeatTrackingProcessor(fps=100, online=True)

    def test_chunks_frames_and_empty_updates_agree(self):
        processor = self.processor()
        expected = processor(self.activations)
        posterior = processor.hmm._prev.copy()
        for size in [1, 17, 64, 127, 800]:
            with self.subTest(size=size):
                stream = self.processor()
                chunks = []
                for start in range(0, len(self.activations), size):
                    chunks.append(stream(self.activations[start:start + size],
                                         reset=False))
                    counter = stream.counter
                    before = stream.hmm._prev.copy()
                    self.assertEqual(stream([], reset=False).shape, (0,))
                    self.assertEqual(stream.counter, counter)
                    np.testing.assert_array_equal(stream.hmm._prev, before)
                np.testing.assert_array_equal(np.concatenate(chunks), expected)
                np.testing.assert_array_equal(stream.hmm._prev, posterior)
                np.testing.assert_array_equal(stream(self.activations), expected)

    def test_long_input_bounds_forward_storage(self):
        processor = self.processor()
        forward = processor.hmm.forward_states
        lengths = []

        def record(block, **kwargs):
            lengths.append(len(block))
            states = forward(block, **kwargs)
            self.assertEqual(states.shape, (len(block),))
            self.assertEqual(states.dtype, np.uint32)
            return states

        with patch.object(processor.hmm, 'forward_states', side_effect=record), \
                patch.object(processor.hmm, 'forward',
                             side_effect=AssertionError('dense posterior allocation')):
            result = processor(self.activations)
        self.assertGreater(len(result), 10)
        self.assertEqual(lengths, [len(self.activations)])

    def test_prefix_cannot_use_future_activations(self):
        expected = self.processor()(self.activations[:217])
        changed = self.activations.copy()
        changed[217:] = 0.99
        for values in [self.activations, changed]:
            whole = self.processor()(values)
            np.testing.assert_array_equal(whole[whole < 2.17], expected)

    def test_legacy_positional_constructor_arguments(self):
        tempo_estimator = object()
        legacy = BeatTrackingProcessor(.2, None, 100, tempo_estimator)
        self.assertEqual(legacy.look_aside, .2)
        self.assertIsNone(legacy.look_ahead)
        self.assertEqual(legacy.fps, 100)
        self.assertIs(legacy.tempo_estimator, tempo_estimator)
        dbn = DBNBeatTrackingProcessor(55., 215., None, 100, 16,
                                       True, 0, 100, True)
        np.testing.assert_array_equal(dbn(self.activations),
                                      self.processor()(self.activations))


if __name__ == '__main__':
    unittest.main()
