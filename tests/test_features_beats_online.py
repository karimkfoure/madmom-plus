# encoding: utf-8
"""Tests for optional causal beat observations."""

import unittest

import numpy as np

from madmom.features.beats import (DBNBeatTrackingProcessor,
                                   SpectralBeatActivationProcessor)


class TestSpectralBeatActivationProcessor(unittest.TestCase):

    def setUp(self):
        generator = np.random.RandomState(7)
        self.features = generator.uniform(0, 1, (200, 3))
        self.features[:, 1:] *= generator.binomial(1, 0.2, (200, 2))

    def test_disabled_and_zero_onset_are_exact_noops(self):
        processor = SpectralBeatActivationProcessor(strength=0)
        np.testing.assert_array_equal(processor(self.features),
                                      self.features[:, 0])
        features = self.features.copy()
        features[:, 1:] = 0
        processor = SpectralBeatActivationProcessor()
        np.testing.assert_array_equal(processor(features), features[:, 0])

    def test_causal_prefix_and_chunk_equivalence(self):
        expected = SpectralBeatActivationProcessor()(self.features)
        for end in (1, 2, 11, 79, 199):
            actual = SpectralBeatActivationProcessor()(self.features[:end])
            np.testing.assert_array_equal(actual, expected[:end])
        processor = SpectralBeatActivationProcessor()
        chunks = [processor(self.features[:17]),
                  processor(self.features[17:96], reset=False),
                  processor([], reset=False),
                  processor(self.features[96:], reset=False)]
        np.testing.assert_array_equal(np.concatenate(chunks), expected)
        processor.reset()
        framewise = np.concatenate([processor(row, reset=False)
                                    for row in self.features])
        np.testing.assert_array_equal(framewise, expected)

    def test_reset_and_offline_equivalence(self):
        processor = SpectralBeatActivationProcessor()
        expected = processor(self.features)
        np.testing.assert_array_equal(processor(self.features), expected)
        processor(self.features, reset=False)
        np.testing.assert_array_equal(processor(self.features), expected)
        offline = SpectralBeatActivationProcessor(online=False)
        np.testing.assert_array_equal(offline(self.features), expected)
        np.testing.assert_array_equal(offline(self.features), expected)

    def test_constant_band_scale_does_not_change_output(self):
        expected = SpectralBeatActivationProcessor()(self.features)
        scaled = self.features * np.array([1., 700., 0.003])
        np.testing.assert_allclose(SpectralBeatActivationProcessor()(scaled),
                                   expected, atol=1e-15, rtol=1e-15)

    def test_probability_boundaries_and_finite_output(self):
        features = np.array([[0., 1., 0.], [1., 0., 1.],
                             [1e-200, 1., 0.], [0.999999, 0., 1.]])
        result = SpectralBeatActivationProcessor(strength=1000)(features)
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertTrue(np.all((result >= 0) & (result <= 1)))
        np.testing.assert_array_equal(result[:2], [0., 1.])

    def test_low_onsets_win_an_ambiguous_half_period_grid(self):
        # Neural observations slightly favour the high-frequency offbeats.
        # The optional low-frequency prior must preserve the 120 BPM grid.
        features = np.zeros((1200, 3))
        features[:, 0] = 0.01
        features[::50, 0] = 0.65
        features[25::50, 0] = 0.75
        features[::50, 1] = 1.
        features[25::50, 2] = 1.
        adjusted = SpectralBeatActivationProcessor(strength=2)(features)
        self.assertTrue(np.all(adjusted[::50] > features[::50, 0]))
        self.assertTrue(np.all(adjusted[25::50] < features[25::50, 0]))
        baseline = DBNBeatTrackingProcessor(
            min_bpm=119, max_bpm=121, fps=100, online=True)(features[:, 0])
        corrected = DBNBeatTrackingProcessor(
            min_bpm=119, max_bpm=121, fps=100, online=True)(adjusted)
        baseline = baseline[baseline >= 3]
        corrected = corrected[corrected >= 3]
        self.assertGreater(len(baseline), 12)
        self.assertGreater(len(corrected), 12)
        baseline_error = np.abs(((baseline + .25) % .5) - .25)
        corrected_error = np.abs(((corrected + .25) % .5) - .25)
        self.assertGreater(np.median(baseline_error), .2)
        self.assertLess(np.median(corrected_error), .04)

    def test_equal_band_evidence_does_not_claim_phase_information(self):
        features = self.features.copy()
        features[:, 2] = 9 * features[:, 1]
        np.testing.assert_allclose(SpectralBeatActivationProcessor()(features),
                                   features[:, 0], atol=1e-15, rtol=1e-15)

    def test_invalid_parameters(self):
        for parameters in ({'strength': -1}, {'strength': np.inf},
                           {'strength': [1]}, {'fps': 0}, {'fps': np.nan},
                           {'time_constant': 0}, {'time_constant': -1}):
            with self.assertRaises(ValueError):
                SpectralBeatActivationProcessor(**parameters)

    def test_invalid_input_does_not_change_running_state(self):
        processor = SpectralBeatActivationProcessor()
        expected = processor(self.features)
        for invalid in ([.2], np.zeros((3, 2)), np.zeros((1, 3, 1)),
                        [[np.nan, 0, 1]], [[1.1, 0, 1]], [[-.1, 0, 1]],
                        [[.5, -1, 0]], [[.5, 0, np.inf]]):
            with self.assertRaises(ValueError):
                processor(invalid)
        continued = processor(self.features, reset=False)
        reference = SpectralBeatActivationProcessor()
        reference(self.features)
        np.testing.assert_array_equal(continued,
                                      reference(self.features, reset=False))
        self.assertEqual(processor([]).shape, (0,))
        np.testing.assert_array_equal(processor(self.features, reset=False),
                                      expected)


if __name__ == '__main__':
    unittest.main()
