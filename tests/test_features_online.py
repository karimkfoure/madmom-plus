"""Tests for causal pulse fitting and rhythmic reliability."""

import unittest

import numpy as np

from madmom.features.online import BeatPeriodProcessor, RhythmPresenceProcessor


class TestBeatPeriodProcessor(unittest.TestCase):

    def test_fit_jitter_and_missed_beat(self):
        grid = np.arange(20) * .5 + .2
        beats = np.delete(grid, [5, 11])
        beats += .003 * np.sin(np.arange(len(beats)))
        state = BeatPeriodProcessor()(beats)
        self.assertTrue(state.valid)
        self.assertAlmostEqual(state.period, .5, delta=.001)
        self.assertAlmostEqual(state.phase, grid[-1], delta=.003)

    def test_extra_beat_and_outlier(self):
        grid = np.arange(12) * .5
        beats = np.sort(np.append(grid, [2.15, 4.3]))
        state = BeatPeriodProcessor()(beats)
        self.assertTrue(state.valid)
        self.assertAlmostEqual(state.period, .5)
        self.assertAlmostEqual(state.phase, grid[-1])
        self.assertGreater(state.confidence, .8)

    def test_phase_wrap(self):
        beats = np.arange(16) * .5 + np.tile([-.006, .006], 8) + .01
        state = BeatPeriodProcessor()(beats)
        self.assertTrue(state.valid)
        self.assertAlmostEqual(state.period, .5, delta=.001)
        self.assertAlmostEqual(state.phase, 7.51, delta=.008)

    def test_expiry_and_reacquisition(self):
        proc = BeatPeriodProcessor(expiry=2.)
        state = proc([0, .5, 1, 1.5], now=1.6)
        self.assertTrue(state.valid)
        self.assertFalse(proc([], now=3.5, reset=False).valid)
        self.assertFalse(proc([5.], now=5., reset=False).valid)
        self.assertTrue(proc([5.5, 6., 6.5], now=6.5, reset=False).valid)

    def test_outlier_does_not_extend_expiry(self):
        state = BeatPeriodProcessor(expiry=2.)(
            [0, .5, 1, 1.5, 2, 2.5, 2.7], now=2.7)
        self.assertAlmostEqual(state.valid_until, 4.5)

    def test_octave_preference(self):
        beats = np.arange(20) * .25
        raw = BeatPeriodProcessor()(beats)
        preferred = BeatPeriodProcessor(preferred_bpm=(80., 160.),
                                        min_confidence=0)(beats)
        self.assertAlmostEqual(raw.period, .25)
        self.assertAlmostEqual(preferred.period, .5)
        self.assertAlmostEqual(preferred.phase % .5, 0.)
        self.assertTrue(preferred.valid)

    def test_slower_octave_preserves_supported_phase(self):
        beats = np.arange(40) * .25
        parameters = dict(preferred_bpm=(80., 160.))
        proc = BeatPeriodProcessor(**parameters)
        states = [proc([beat], reset=False) for beat in beats]
        valid = [state for state in states if state.valid]
        self.assertGreater(len(valid), 20)
        for state in valid:
            self.assertAlmostEqual(state.period, .5)
            self.assertAlmostEqual(state.phase % .5, 0.)
        for stop in (8, 17, 40):
            self.assertEqual(BeatPeriodProcessor(**parameters)(beats[:stop]),
                             states[stop - 1])
        chunked = BeatPeriodProcessor(**parameters)
        chunked(beats[:9])
        self.assertEqual(chunked(beats[9:], reset=False), states[-1])

    def test_phase_consensus_can_replace_unsupported_prior(self):
        proc = BeatPeriodProcessor(window_size=8)
        first = proc(np.arange(12) * .5)
        self.assertAlmostEqual(first.phase % .5, 0.)
        shifted = proc(6.25 + np.arange(12) * .5, reset=False)
        self.assertTrue(shifted.valid)
        self.assertAlmostEqual(shifted.period, .5)
        self.assertAlmostEqual(shifted.phase % .5, .25)

    def test_tempo_change_adapts_with_bounded_history(self):
        proc = BeatPeriodProcessor(window_size=8)
        proc(np.arange(12) * .5)
        state = proc(5.5 + np.arange(1, 14) * .4, reset=False)
        self.assertTrue(state.valid)
        self.assertAlmostEqual(state.period, .4)
        self.assertEqual(len(proc._beats), 8)

    def test_chunk_prefix_reset_and_empty(self):
        beats = np.arange(20) * .5 + .1
        expected = BeatPeriodProcessor()(beats)
        proc = BeatPeriodProcessor()
        self.assertFalse(proc([]).valid)
        proc(beats[:7])
        actual = proc(beats[7:], reset=False)
        self.assertEqual(actual, expected)
        self.assertEqual(proc(beats), expected)
        for i in range(4, len(beats)):
            prefix = BeatPeriodProcessor()(beats[:i])
            stream = BeatPeriodProcessor()
            for beat in beats[:i]:
                observed = stream([beat], reset=False)
            self.assertEqual(observed, prefix)

    def test_invalid_input_does_not_mutate_state(self):
        proc = BeatPeriodProcessor()
        before = proc([0, .5, 1, 1.5])
        for beats, now in [([np.nan], 2), ([1.5], 2), ([2., 1.9], 2),
                           ([[2]], 2), ([2.1], 2), ([], 1)]:
            with self.assertRaises(ValueError):
                proc(beats, now=now, reset=False)
        self.assertEqual(proc([], now=1.5, reset=False), before)

    def test_invalid_parameters(self):
        for kwargs in [dict(window_size=3), dict(min_intervals=1),
                       dict(expiry=0), dict(min_confidence=2),
                       dict(tolerance=.5), dict(preferred_bpm=(160, 80))]:
            with self.assertRaises(ValueError):
                BeatPeriodProcessor(**kwargs)


class TestRhythmPresenceProcessor(unittest.TestCase):

    def test_periodic_evidence_silence_and_constant(self):
        proc = RhythmPresenceProcessor()
        rhythmic = np.zeros(800)
        rhythmic[::50] = 1.
        scores = proc(rhythmic)
        self.assertTrue(np.all(scores[:210] == 0))
        self.assertGreater(scores[-1], .95)
        self.assertTrue(np.all(proc(np.zeros(800)) == 0))
        self.assertTrue(np.all(proc(np.full(800, .8)) == 0))

    def test_noise_and_weak_evidence(self):
        noise = np.random.RandomState(21).uniform(0, 1, 2000)
        proc = RhythmPresenceProcessor()
        self.assertLess(np.mean(proc(noise)[400:]), .15)
        rhythmic = np.zeros(800)
        rhythmic[::50] = .001
        self.assertLess(proc(rhythmic)[-1], .01)

    def test_chunk_reset_prefix_and_release(self):
        activations = np.zeros(1200)
        activations[:600:50] = 1.
        expected = RhythmPresenceProcessor()(activations)
        proc = RhythmPresenceProcessor()
        actual = np.concatenate([proc(chunk, reset=False) for chunk in
                                 np.array_split(activations, [137, 489, 602])])
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(proc(activations), expected)
        np.testing.assert_array_equal(RhythmPresenceProcessor()(activations[:489]),
                                      expected[:489])
        self.assertEqual(expected[-1], 0.)
        self.assertEqual(proc([], reset=False).shape, (0,))
        self.assertEqual(len(proc._buffer), proc.window_size)

    def test_invalid_parameters_and_input(self):
        for kwargs in [dict(fps=0), dict(window=.5), dict(min_bpm=300),
                       dict(update_interval=0), dict(correlation_floor=1),
                       dict(std_threshold=0)]:
            with self.assertRaises(ValueError):
                RhythmPresenceProcessor(**kwargs)
        for activations in [[np.nan], [-.1], [1.1], [[.5]]]:
            with self.assertRaises(ValueError):
                RhythmPresenceProcessor()(activations)


if __name__ == '__main__':
    unittest.main()
