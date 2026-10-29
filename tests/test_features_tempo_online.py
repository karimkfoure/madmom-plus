# encoding: utf-8
"""Regression tests for causal tempo histogram state and window boundaries."""

import unittest

import numpy as np

from madmom.features.tempo import (ACFTempoHistogramProcessor,
                                   CombFilterTempoHistogramProcessor,
                                   DBNTempoHistogramProcessor,
                                   TempoEstimationProcessor)


HISTOGRAM_PROCESSORS = (ACFTempoHistogramProcessor,
                        CombFilterTempoHistogramProcessor,
                        DBNTempoHistogramProcessor)


class TestOnlineTempoHistogramState(unittest.TestCase):

    def setUp(self):
        self.activations = np.random.RandomState(31).uniform(.01, .99, 179)
        self.parameters = dict(min_bpm=40, max_bpm=240, fps=20,
                               hist_buffer=.75, online=True)

    def test_empty_input_preserves_state_and_empty_reset_clears_state(self):
        for processor_class in HISTOGRAM_PROCESSORS:
            with self.subTest(processor=processor_class.__name__):
                processor = processor_class(**self.parameters)
                for empty in ([], np.empty(0)):
                    bins, delays = processor(empty)
                    np.testing.assert_array_equal(bins, np.zeros(len(delays)))
                before, delays = processor(self.activations[:81])
                for empty in ([], np.empty(0)):
                    query, query_delays = processor(empty, reset=False)
                    np.testing.assert_array_equal(query, before)
                    np.testing.assert_array_equal(query_delays, delays)
                continued, _ = processor(self.activations[81:], reset=False)
                reference, _ = processor_class(**self.parameters)(
                    self.activations)
                np.testing.assert_array_equal(continued, reference)
                cleared, _ = processor([], reset=True)
                np.testing.assert_array_equal(cleared, np.zeros(len(delays)))
                restarted, _ = processor(self.activations[81:], reset=False)
                reference, _ = processor_class(**self.parameters)(
                    self.activations[81:])
                np.testing.assert_array_equal(restarted, reference)

    def test_scalar_and_zero_dimensional_input_match_one_frame_arrays(self):
        for processor_class in HISTOGRAM_PROCESSORS:
            with self.subTest(processor=processor_class.__name__):
                array = processor_class(**self.parameters)
                scalar = processor_class(**self.parameters)
                zero_dimensional = processor_class(**self.parameters)
                for value in self.activations:
                    expected, delays = array([value], reset=False)
                    actual, actual_delays = scalar(float(value), reset=False)
                    np.testing.assert_array_equal(actual, expected)
                    np.testing.assert_array_equal(actual_delays, delays)
                    actual, _ = zero_dimensional(np.array(value), reset=False)
                    np.testing.assert_array_equal(actual, expected)

    def test_chunks_longer_than_histogram_buffer_match_framewise_state(self):
        # Every nonempty block below exceeds or crosses the 15-frame buffer.
        boundaries = (0, 37, 38, 90, 179)
        for processor_class in HISTOGRAM_PROCESSORS:
            with self.subTest(processor=processor_class.__name__):
                framewise = processor_class(**self.parameters)
                checkpoints = {}
                for index, value in enumerate(self.activations, 1):
                    histogram = framewise(value, reset=False)
                    if index in boundaries:
                        checkpoints[index] = histogram
                chunked = processor_class(**self.parameters)
                for start, end in zip(boundaries[:-1], boundaries[1:]):
                    actual = chunked(self.activations[start:end], reset=False)
                    for array, expected in zip(actual, checkpoints[end]):
                        np.testing.assert_array_equal(array, expected)
                whole = processor_class(**self.parameters)(self.activations)
                for actual, expected in zip(whole, checkpoints[len(
                        self.activations)]):
                    np.testing.assert_array_equal(actual, expected)
                # Test the hidden filter/posterior state after the final block.
                for value in self.activations[:31]:
                    expected = framewise(value, reset=False)
                    actual = chunked(value, reset=False)
                    np.testing.assert_array_equal(actual[0], expected[0])

    def test_invalid_activation_blocks_do_not_reset_existing_state(self):
        for processor_class in HISTOGRAM_PROCESSORS:
            with self.subTest(processor=processor_class.__name__):
                processor = processor_class(**self.parameters)
                expected, _ = processor(self.activations)
                for invalid in (np.zeros((3, 1)), np.array([np.nan]),
                                np.array([np.inf])):
                    with self.assertRaises(ValueError):
                        processor(invalid, reset=True)
                    actual, _ = processor([], reset=False)
                    np.testing.assert_array_equal(actual, expected)

    def test_reset_is_safe_in_offline_mode(self):
        for processor_class in HISTOGRAM_PROCESSORS:
            with self.subTest(processor=processor_class.__name__):
                parameters = dict(self.parameters, online=False)
                processor = processor_class(**parameters)
                expected = processor(self.activations)
                processor.reset()
                processor.reset()
                actual = processor(self.activations)
                for array, reference in zip(actual, expected):
                    np.testing.assert_array_equal(array, reference)

    def test_positional_constructor_arguments_keep_their_meaning(self):
        acf = ACFTempoHistogramProcessor(40, 240, .75, 20, True)
        comb = CombFilterTempoHistogramProcessor(40, 240, .79, .75, 20, True)
        dbn = DBNTempoHistogramProcessor(40, 240, .75, 20, True)
        for processor_class, positional in zip(
                (ACFTempoHistogramProcessor, CombFilterTempoHistogramProcessor,
                 DBNTempoHistogramProcessor), (acf, comb, dbn)):
            expected = processor_class(**self.parameters)(self.activations)
            actual = positional(self.activations)
            for array, reference in zip(actual, expected):
                np.testing.assert_array_equal(array, reference)

    def test_sub_frame_histogram_buffer_is_rejected(self):
        for processor_class in HISTOGRAM_PROCESSORS:
            with self.subTest(processor=processor_class.__name__):
                for hist_buffer in (0, -.5, .001):
                    with self.assertRaises(ValueError):
                        processor_class(**dict(self.parameters,
                                               hist_buffer=hist_buffer))

    def test_tempo_estimator_accepts_empty_and_scalar_updates(self):
        for processor_class in HISTOGRAM_PROCESSORS:
            with self.subTest(processor=processor_class.__name__):
                histogram = processor_class(**self.parameters)
                estimator = TempoEstimationProcessor(
                    method=None, histogram_processor=histogram, fps=20,
                    online=True)
                np.testing.assert_array_equal(estimator([]), [[np.nan, 0.]])
                estimator(.5, reset=False)
                estimator(np.array(.5), reset=False)
                before = estimator(self.activations, reset=False)
                np.testing.assert_array_equal(estimator([], reset=False),
                                              before)
                np.testing.assert_array_equal(estimator([], reset=True),
                                              [[np.nan, 0.]])


class TestOnlineTempoHistogramWindow(unittest.TestCase):

    def test_acf_and_comb_match_offline_before_window_fills(self):
        activations = np.random.RandomState(22).uniform(0, 1, 97)
        for processor_class in (ACFTempoHistogramProcessor,
                                CombFilterTempoHistogramProcessor):
            with self.subTest(processor=processor_class.__name__):
                parameters = dict(min_bpm=40, max_bpm=240, fps=20,
                                  hist_buffer=5)
                if processor_class is CombFilterTempoHistogramProcessor:
                    # The offline Cython kernel stores alpha as C float.
                    # Use an exact binary value to isolate window semantics.
                    parameters['alpha'] = .5
                offline = processor_class(**parameters)(activations)
                online = processor_class(online=True,
                                         **parameters)(activations)
                np.testing.assert_allclose(online[0], offline[0],
                                           atol=1e-13, rtol=1e-13)
                np.testing.assert_array_equal(online[1], offline[1])

    def test_acf_and_comb_aggregate_recent_outputs_with_past_context(self):
        activations = np.random.RandomState(22).uniform(0, 1, 97)
        for processor_class in (ACFTempoHistogramProcessor,
                                CombFilterTempoHistogramProcessor):
            with self.subTest(processor=processor_class.__name__):
                parameters = dict(min_bpm=40, max_bpm=240, fps=20,
                                  hist_buffer=.75)
                if processor_class is CombFilterTempoHistogramProcessor:
                    parameters['alpha'] = .5
                offline = processor_class(**parameters)
                full, delays = offline(activations)
                earlier, _ = offline(activations[:-15])
                actual, actual_delays = processor_class(
                    online=True, **parameters)(activations)
                np.testing.assert_allclose(actual, full - earlier,
                                           atol=1e-12, rtol=1e-12)
                np.testing.assert_array_equal(actual_delays, delays)

    def test_acf_delays_have_no_one_frame_offset(self):
        parameters = dict(min_bpm=40, max_bpm=240, fps=20,
                          hist_buffer=5)
        for delay in (5, 15, 30):
            with self.subTest(delay=delay):
                activations = np.zeros(40)
                activations[[1, 1 + delay]] = 1
                processor = ACFTempoHistogramProcessor(online=True,
                                                        **parameters)
                bins, delays = processor(activations)
                np.testing.assert_array_equal(bins, (delays == delay) * 1.)

    def test_dbn_histogram_retains_only_latest_frame_votes(self):
        parameters = dict(min_bpm=40, max_bpm=240, fps=20,
                          hist_buffer=.75, online=True)
        activations = np.random.RandomState(9).uniform(.01, .99, 97)
        processor = DBNTempoHistogramProcessor(**parameters)
        bins, delays = processor(activations)
        reference = DBNTempoHistogramProcessor(**parameters)
        posterior = reference.dbn.hmm.forward(activations)
        states = posterior.argmax(axis=1)
        intervals = reference.dbn.st.state_intervals[states[-15:]]
        expected = np.array([(intervals == delay).sum() for delay in delays])
        np.testing.assert_array_equal(bins, expected)
        self.assertEqual(bins.sum(), 15)

    def test_dbn_forward_storage_is_bounded_for_long_chunks(self):
        class ForwardBlockRecorder(object):
            def __init__(self, hmm):
                self.hmm = hmm
                self.block_lengths = []

            def __getattr__(self, name):
                return getattr(self.hmm, name)

            def forward(self, observations, **kwargs):
                self.block_lengths.append(len(observations))
                return self.hmm.forward(observations, **kwargs)

        processor = DBNTempoHistogramProcessor(
            min_bpm=40, max_bpm=240, fps=20, hist_buffer=.75, online=True)
        recorder = ForwardBlockRecorder(processor.dbn.hmm)
        processor.dbn.hmm = recorder
        activations = np.random.RandomState(19).uniform(.01, .99, 1027)
        actual, _ = processor(activations)
        self.assertEqual(sum(recorder.block_lengths), len(activations))
        self.assertLessEqual(max(recorder.block_lengths), 64)
        reference = DBNTempoHistogramProcessor(
            min_bpm=40, max_bpm=240, fps=20, hist_buffer=.75, online=True)
        posterior = reference.dbn.hmm.forward(activations)
        states = posterior.argmax(axis=1)
        intervals = reference.dbn.st.state_intervals[states[-15:]]
        expected = np.array([(intervals == delay).sum()
                             for delay in reference.intervals])
        np.testing.assert_array_equal(actual, expected)
        tail = activations[:17]
        expected = reference.dbn.hmm.forward(tail, reset=False)
        actual = recorder.forward(tail, reset=False)
        np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
