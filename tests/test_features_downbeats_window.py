# encoding: utf-8
"""Tests for delayed bar evidence on a causal beat stream."""

import unittest
import os
from threading import Event

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from madmom.features.downbeats import (BarPhaseVotingProcessor,
                                      WindowedDownBeatTrackingProcessor)
from madmom.processors import Processor
from madmom.audio.signal import Signal
from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor

from . import AUDIO_PATH


def window_activations(start, stop, meter=4, phase=0, strength=.95,
                       ordinary=.01, change=None):
    """Build mutually exclusive probabilities on a 120 bpm beat grid."""
    fps = 100
    result = np.full((int(round((stop - start) * fps)), 2), 1e-4)
    for index in range(int(np.ceil(start * 2)), int(np.ceil(stop * 2))):
        frame = int(round((index * .5 - start) * fps))
        if not 0 <= frame < len(result):
            continue
        active_meter = meter
        active_phase = phase
        if change is not None and index * .5 >= change[0]:
            active_meter, active_phase = change[1:]
        downbeat = strength if index % active_meter == active_phase else ordinary
        result[frame] = [.99 - downbeat, downbeat]
    return result


def stream_votes(processor, stop=30., meter=4, phase=0, change=None, window=6.):
    """Publish beats first, then submit only completed audio windows."""
    rows = []
    for time in np.arange(0., stop + .25, .5):
        rows.append(processor([time], now=time, reset=False))
        if time >= window and time % 2 == 0:
            processor([], window_activations(time - window, time, meter, phase,
                                             change=change),
                      start=time - window, fps=100, now=time, reset=False)
    return np.vstack(rows)


class TestBarPhaseVotingProcessor(unittest.TestCase):

    def test_synthetic_meter_phase_and_unknown_start(self):
        for meter in (3, 4):
            with self.subTest(meter=meter):
                processor = BarPhaseVotingProcessor()
                self.assertIsInstance(processor, Processor)
                rows = stream_votes(processor, meter=meter, phase=1)
                assert_array_equal(rows[rows[:, 0] < 6, 1], 0)
                settled = rows[rows[:, 0] >= 16]
                expected = (np.rint(settled[:, 0] * 2).astype(int) - 1) % meter + 1
                assert_array_equal(settled[:, 1], expected)
                self.assertEqual(processor.meter, meter)
                self.assertGreater(processor.confidence, .7)
                self.assertAlmostEqual(processor.period, .5)
                self.assertAlmostEqual((30 - processor.phase) / .5 % meter,
                                       (60 - 1) % meter)

    def test_meter_multiple_tie_prefers_shorter_bar(self):
        processor = BarPhaseVotingProcessor(beats_per_bar=(3, 6))
        stream_votes(processor, meter=3, window=8.)
        self.assertEqual(processor.meter, 3)
        # A genuine six-beat bar has one distinct accent, not two equal accents.
        processor = BarPhaseVotingProcessor(beats_per_bar=(3, 6))
        stream_votes(processor, meter=6, window=8.)
        self.assertEqual(processor.meter, 6)

    def test_short_window_cannot_establish_large_meter(self):
        processor = BarPhaseVotingProcessor(beats_per_bar=(6,))
        # Six seconds has twelve grid beats. Excluding boundary peaks leaves
        # fewer than two observations in at least one of six phase classes.
        stream_votes(processor, meter=6, window=6.)
        self.assertIsNone(processor.meter)
        self.assertEqual(processor.confidence, 0)

    def test_larger_meter_can_release_a_multiple_after_song_change(self):
        processor = BarPhaseVotingProcessor(beats_per_bar=(3, 6))
        rows = stream_votes(processor, stop=72, meter=6,
                            change=(40., 3, 0), window=8.)
        old = rows[(rows[:, 0] >= 24) & (rows[:, 0] < 40)]
        new = rows[rows[:, 0] >= 62]
        assert_array_equal(old[:, 1], np.rint(old[:, 0] * 2).astype(int) % 6 + 1)
        assert_array_equal(new[:, 1], np.rint(new[:, 0] * 2).astype(int) % 3 + 1)
        self.assertEqual(processor.meter, 3)

    def test_repeated_integer_gaps_reacquire_a_slower_tempo(self):
        for fill in (False, True):
            with self.subTest(fill=fill):
                processor = BarPhaseVotingProcessor(beats_per_bar=(4,), fill_gaps=fill)
                stream_votes(processor, stop=20)
                rows = []
                for time in np.arange(21., 51.):
                    rows.append(processor([time], now=time, reset=False))
                    if time >= 30 and time % 2 == 0:
                        activation = np.full((1000, 2), 1e-4)
                        for beat in np.arange(time - 10, time):
                            frame = int((beat - time + 10) * 100)
                            downbeat = .95 if (beat - 20) % 4 == 0 else .01
                            activation[frame] = [.99 - downbeat, downbeat]
                        processor([], activation, start=time - 10, now=time, reset=False)
                self.assertAlmostEqual(processor.period, 1.)
                settled = np.vstack(rows)
                settled = settled[settled[:, 0] >= 42]
                assert_array_equal(settled[:, 1], (settled[:, 0] - 20) % 4 + 1)
                self.assertEqual(len(settled), 9)

    def test_meter_recovers_after_change_without_reset(self):
        processor = BarPhaseVotingProcessor()
        rows = stream_votes(processor, stop=54, meter=3, change=(30., 4, 0))
        old = rows[(rows[:, 0] >= 18) & (rows[:, 0] < 30)]
        new = rows[rows[:, 0] >= 46]
        assert_array_equal(old[:, 1], np.rint(old[:, 0] * 2).astype(int) % 3 + 1)
        assert_array_equal(new[:, 1], np.rint(new[:, 0] * 2).astype(int) % 4 + 1)
        self.assertEqual(processor.meter, 4)

    def test_phase_recovers_after_change_without_reset(self):
        processor = BarPhaseVotingProcessor(beats_per_bar=(4,))
        rows = stream_votes(processor, stop=54, meter=4, change=(30., 4, 1))
        settled = rows[rows[:, 0] >= 46]
        expected = (np.rint(settled[:, 0] * 2).astype(int) - 1) % 4 + 1
        assert_array_equal(settled[:, 1], expected)
        self.assertEqual(processor.meter, 4)

    def test_weak_competing_phase_does_not_replace_lock(self):
        processor = BarPhaseVotingProcessor(beats_per_bar=(4,))
        stream_votes(processor, stop=20)
        before = processor.phase_class
        processor(np.arange(20.5, 26.1, .5), now=26, reset=False)
        weak = window_activations(20, 26, phase=1, strength=.12, ordinary=.11)
        processor([], weak, start=20, now=26, reset=False)
        self.assertEqual(processor.phase_class, before)
        self.assertGreater(processor.confidence, 0)

    def test_constant_evidence_cannot_establish_bar(self):
        processor = BarPhaseVotingProcessor()
        processor(np.arange(0., 6., .5), now=6)
        processor([], np.full((600, 2), .1), start=0, now=6, reset=False)
        self.assertIsNone(processor.meter)
        self.assertEqual(processor.confidence, 0)
        assert_array_equal(processor([6.5], now=6.5, reset=False), [[6.5, 0]])

    def test_expired_evidence_releases_lock(self):
        processor = BarPhaseVotingProcessor(evidence_expiry=5.)
        stream_votes(processor, stop=12)
        self.assertIsNotNone(processor.meter)
        self.assertEqual(processor([], now=18, reset=False).shape, (0, 2))
        self.assertIsNone(processor.meter)
        self.assertEqual(processor.confidence, 0)

    def test_consistent_tempo_change_clears_old_bar_votes(self):
        processor = BarPhaseVotingProcessor(beats_per_bar=(4,))
        stream_votes(processor, stop=20)
        rows = processor([20.7, 21.4, 22.1], now=22.1, reset=False)
        self.assertEqual(rows[-1, 1], 0)
        self.assertIsNone(processor.meter)
        self.assertAlmostEqual(processor.period, .7)

    def test_explicit_bar_reset_keeps_beat_clock_and_interval(self):
        processor = BarPhaseVotingProcessor(beats_per_bar=(4,))
        stream_votes(processor, stop=20)
        processor.reset_bar()
        self.assertIsNone(processor.meter)
        self.assertAlmostEqual(processor.period, .5)
        assert_array_equal(processor([20.5], now=20.5, reset=False), [[20.5, 0]])
        with self.assertRaises(ValueError):
            processor([20.], now=20.5, reset=False)

    def test_evidence_never_relabels_emitted_beats(self):
        processor = BarPhaseVotingProcessor(beats_per_bar=(4,))
        before = processor(np.arange(0., 6., .5), now=6)
        assert_array_equal(before[:, 1], 0)
        result = processor([], window_activations(0, 6), start=0,
                           now=6, reset=False)
        self.assertEqual(result.shape, (0, 2))
        assert_array_equal(before[:, 1], 0)
        assert_array_equal(processor([6.5], now=6.5, reset=False), [[6.5, 2]])

    def test_gap_fill_is_optional_bounded_and_preserves_phase(self):
        for fill in (False, True):
            with self.subTest(fill=fill):
                processor = BarPhaseVotingProcessor(beats_per_bar=(4,),
                                                    fill_gaps=fill, max_fill=2)
                stream_votes(processor, stop=20)
                # The beat at 20.5 is missing. Its index must still count.
                rows = processor([21.], now=21, reset=False)
                expected = [[20.5, 2], [21., 3]] if fill else [[21., 3]]
                assert_array_equal(rows, expected)
                # A long interruption must never produce an unbounded grid.
                rows = processor([30.], now=30, reset=False)
                self.assertLessEqual(len(rows), 1)

    def test_gap_fill_requires_bar_evidence(self):
        processor = BarPhaseVotingProcessor(fill_gaps=True)
        processor(np.arange(0., 4.1, .5), now=4)
        assert_array_equal(processor([5.], now=5, reset=False), [[5., 0]])

    def test_reset_returns_to_unknown_state(self):
        processor = BarPhaseVotingProcessor()
        stream_votes(processor, stop=16)
        processor.reset()
        self.assertIsNone(processor.meter)
        self.assertEqual(processor.confidence, 0)
        self.assertTrue(np.isnan(processor.period))
        self.assertTrue(np.isnan(processor.phase))
        assert_array_equal(processor([0.]), [[0., 0.]])

    def test_invalid_input_keeps_existing_state(self):
        processor = BarPhaseVotingProcessor()
        stream_votes(processor, stop=16)
        before = (processor.meter, processor.phase, processor.confidence)
        invalid = [dict(beats=[16.5, 16.4], now=17),
                   dict(beats=[np.nan], now=17),
                   dict(beats=[-1.], now=17),
                   dict(beats=[17.5], now=17),
                   dict(beats=[16.5], now=15),
                   dict(beats=[], activations=np.zeros((3, 3)), now=17),
                   dict(beats=[], activations=[[0., np.nan]], now=17),
                   dict(beats=[], activations=[[.7, .8]], now=17),
                   dict(beats=[], activations=[[0., -.1]], now=17),
                   dict(beats=[], activations=np.zeros((600, 2)),
                        start=16, now=17)]
        for parameters in invalid:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    processor(reset=False, **parameters)
                self.assertEqual((processor.meter, processor.phase,
                                  processor.confidence), before)

    def test_invalid_parameters(self):
        invalid = [dict(beats_per_bar=[]), dict(beats_per_bar=(3, 3)),
                   dict(beats_per_bar=(0, 4)), dict(beats_per_bar=(3.5,)),
                   dict(memory=0), dict(memory=np.nan), dict(min_score=-1),
                   dict(phase_hysteresis=.9), dict(meter_hysteresis=.9),
                   dict(min_beats_per_phase=0), dict(history=0),
                   dict(evidence_expiry=0), dict(peak_radius=-1),
                   dict(max_fill=-1), dict(fill_confidence=1.1)]
        for parameters in invalid:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    BarPhaseVotingProcessor(**parameters)


class FakeBeatProcessor:
    """Emit native-format frame rows without loading neural networks."""

    def __init__(self):
        self.closed = False
        self.reset()

    def reset(self):
        self.samples = 0
        self.next_sample = 0

    def push(self, audio):
        self.samples += len(audio)
        rows = []
        while self.next_sample < self.samples:
            frame = self.next_sample // 441
            rows.append((frame / 100., .8, float(frame % 10 == 0), 600.))
            self.next_sample += 441
        return np.asarray(rows).reshape((-1, 4))

    def close(self):
        self.closed = True


class FakeDownBeatProcessor:

    def __init__(self):
        self.calls = []

    def __call__(self, audio, reset=True):
        if not reset:
            raise AssertionError('Each bidirectional window must start fresh')
        self.calls.append(np.array(audio, copy=True))
        return np.full((int(np.ceil(len(audio) / 441.)), 2), .01)


class TestWindowedDownBeatTrackingProcessor(unittest.TestCase):

    def make_processor(self, **kwargs):
        beat = FakeBeatProcessor()
        downbeat = FakeDownBeatProcessor()
        parameters = dict(window=.06, hop=.02, beat_processor=beat,
                          downbeat_processor=downbeat)
        parameters.update(kwargs)
        processor = WindowedDownBeatTrackingProcessor(**parameters)
        self.addCleanup(processor.close)
        return processor, beat, downbeat

    def test_window_schedule_and_exact_audio_contents(self):
        processor, beat, downbeat = self.make_processor()
        audio = np.linspace(-.1, .1, 4410, dtype=np.float32)
        rows = processor(audio)
        self.assertEqual(len(downbeat.calls), 3)
        for index, actual in enumerate(downbeat.calls):
            first = index * 882
            assert_array_equal(actual, audio[first:first + 2646])
        self.assertEqual(beat.samples, len(audio))
        self.assertEqual(rows.shape, (1, 2))
        assert_array_equal(rows[0], [0., 0.])

    def test_chunk_sizes_and_empty_calls_preserve_schedule(self):
        audio = np.linspace(-.1, .1, 5293, dtype=np.float32)
        reference, _, reference_downbeat = self.make_processor()
        expected = reference(audio)
        for size in (1, 37, 441, 2000):
            with self.subTest(size=size):
                processor, _, downbeat = self.make_processor()
                rows = []
                for first in range(0, len(audio), size):
                    rows.append(processor(audio[first:first + size], reset=False))
                    empty = processor(np.empty(0, dtype=np.float32), reset=False)
                    self.assertEqual(empty.shape, (0, 2))
                assert_array_equal(np.vstack(rows), expected)
                self.assertEqual(len(downbeat.calls), len(reference_downbeat.calls))
                for actual, original in zip(downbeat.calls,
                                            reference_downbeat.calls):
                    assert_array_equal(actual, original)

    def test_prefix_does_not_use_future_audio(self):
        audio = np.linspace(-.1, .1, 7056, dtype=np.float32)
        changed = audio.copy()
        changed[3528:] *= -1
        left, _, left_downbeat = self.make_processor()
        right, _, right_downbeat = self.make_processor()
        left_rows = left(audio)
        right_rows = right(changed)
        assert_array_equal(left_rows[left_rows[:, 0] < .08],
                           right_rows[right_rows[:, 0] < .08])
        for first, second in zip(left_downbeat.calls[:2], right_downbeat.calls[:2]):
            assert_array_equal(first, second)

    def test_acquired_bar_labels_are_independent_of_audio_chunk_size(self):
        def downbeat(audio, reset=True):
            # Sample values encode absolute time, so each crop has the same
            # absolute downbeat phase without retaining model state.
            start = float(audio[0])
            data = np.full((len(audio) // 441, 2), 1e-4)
            for frame in range(len(data)):
                beat = int(round((start + frame / 100.) * 10))
                if frame % 10 == 0:
                    data[frame] = [.01, .95] if beat % 3 == 1 else [.95, .01]
            return data

        audio = np.arange(92610, dtype=np.float32) / 44100.
        reference, _, _ = self.make_processor(window=.8, hop=.2,
                                               beats_per_bar=(3,),
                                               downbeat_processor=downbeat)
        expected = reference(audio)
        self.assertEqual(reference.meter, 3)
        self.assertGreater(np.count_nonzero(expected[:, 1]), 10)
        for size in (37, 441, 2000):
            with self.subTest(size=size):
                processor, _, _ = self.make_processor(window=.8, hop=.2,
                                                       beats_per_bar=(3,),
                                                       downbeat_processor=downbeat)
                rows = [processor(audio[first:first + size], reset=False)
                        for first in range(0, len(audio), size)]
                assert_array_equal(np.vstack(rows), expected)

    def test_reset_restarts_audio_and_window_clocks(self):
        processor, _, downbeat = self.make_processor()
        audio = np.zeros(5293, dtype=np.float32)
        expected = processor(audio)
        calls = len(downbeat.calls)
        assert_array_equal(processor(audio), expected)
        self.assertEqual(len(downbeat.calls), 2 * calls)

    def test_invalid_pcm_does_not_advance_stream(self):
        processor, beat, downbeat = self.make_processor()
        processor(np.zeros(1000, dtype=np.float32))
        before = beat.samples
        for audio in (np.zeros((10, 2), dtype=np.float32),
                      np.zeros(441, dtype=np.int16),
                      np.array([np.nan]), np.array([np.inf])):
            with self.subTest(audio=audio):
                with self.assertRaises(ValueError):
                    processor(audio, reset=False)
                self.assertEqual(beat.samples, before)
                self.assertEqual(len(downbeat.calls), 0)

    def test_close_keeps_caller_owned_beat_processor_open(self):
        processor, beat, _ = self.make_processor()
        processor.close()
        processor.close()
        self.assertFalse(beat.closed)
        with self.assertRaises(RuntimeError):
            processor(np.zeros(441, dtype=np.float32))

    def test_background_worker_never_queues_multiple_windows(self):
        entered = Event()
        release = Event()
        calls = []

        def blocked(audio, reset=True):
            calls.append(len(audio))
            entered.set()
            if not release.wait(5):
                raise AssertionError('Worker did not receive release signal')
            return np.full((len(audio) // 441, 2), .01)

        processor, _, _ = self.make_processor(background=True,
                                              downbeat_processor=blocked)
        try:
            processor(np.zeros(8820, dtype=np.float32))
            self.assertTrue(entered.wait(5))
            self.assertEqual(len(calls), 1)
            self.assertEqual(processor.windows, 1)
            self.assertEqual(processor.skipped_windows, 7)
        finally:
            release.set()
            processor.close()

    def test_reset_discards_completed_evidence_from_old_stream(self):
        entered = Event()
        release = Event()

        def blocked(audio, reset=True):
            entered.set()
            if not release.wait(5):
                raise AssertionError('Worker did not receive release signal')
            data = np.full((len(audio) // 441, 2), 1e-4)
            for index, frame in enumerate(range(0, len(data), 10)):
                data[frame] = ([.01, .95] if index % 4 == 0 else [.95, .01])
            return data

        processor, _, _ = self.make_processor(window=6., hop=2.,
                                              background=True,
                                              downbeat_processor=blocked)
        try:
            processor(np.zeros(264600, dtype=np.float32))
            self.assertTrue(entered.wait(5))
            future = processor._pending[0]
            processor(np.zeros(264600, dtype=np.float32), reset=True)
            self.assertEqual(processor.windows, 0)
            self.assertEqual(processor.skipped_windows, 1)
            release.set()
            future.result(timeout=5)
            processor(np.empty(0, dtype=np.float32), reset=False)
            self.assertIsNone(processor.meter)
            self.assertEqual(processor.confidence, 0)
        finally:
            release.set()
            processor.close()

    def test_background_worker_error_reaches_caller(self):
        def fail(audio, reset=True):
            raise RuntimeError('Inference failed')

        processor, _, _ = self.make_processor(background=True,
                                              downbeat_processor=fail)
        # A fast worker can fail during the initial call or a later poll.
        with self.assertRaisesRegex(RuntimeError, 'Inference failed'):
            processor(np.zeros(2646, dtype=np.float32))
            processor._pending[0].exception(timeout=5)
            processor(np.empty(0, dtype=np.float32), reset=False)

    def test_invalid_window_and_backend_parameters(self):
        for parameters in (dict(window=0), dict(hop=0), dict(window=np.nan),
                           dict(hop=np.inf), dict(window=.061),
                           dict(hop=.023), dict(beat_backend='invalid')):
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    WindowedDownBeatTrackingProcessor(**parameters)


class TestWindowedDownBeatAudio(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.audio = Signal(os.path.join(AUDIO_PATH, 'sample.wav'),
                           sample_rate=44100, num_channels=1, dtype=np.float32)

    def test_python_beats_match_existing_causal_processors(self):
        activations = RNNBeatProcessor(online=True, origin='online', fps=100,
                                       num_threads=1)(self.audio)
        expected = DBNBeatTrackingProcessor(online=True, fps=100)(activations)
        with WindowedDownBeatTrackingProcessor(window=2., hop=1.) as processor:
            rows = processor(self.audio)
            assert_array_equal(rows[:, 0], expected)
            self.assertGreater(processor.windows, 0)
            self.assertTrue(np.all(np.isfinite(rows)))
            self.assertEqual(processor.samples, len(self.audio))

    def test_native_and_python_backends_preserve_same_beat_grid(self):
        results = []
        for backend in ('python', 'native'):
            with WindowedDownBeatTrackingProcessor(window=2., hop=1.,
                                                   beat_backend=backend) as processor:
                results.append(processor(self.audio))
        self.assertEqual(len(results[0]), len(results[1]))
        assert_allclose(results[0][:, 0], results[1][:, 0], atol=.01, rtol=0)

    def test_native_engine_chunks_empty_calls_and_reset_preserve_events(self):
        with WindowedDownBeatTrackingProcessor(
                window=2., hop=1., beat_backend='native') as processor:
            expected = processor(self.audio)
            expected_windows = processor.windows
            for size in (137, 441, 4096):
                with self.subTest(chunk_samples=size):
                    processor.reset()
                    rows = []
                    for first in range(0, len(self.audio), size):
                        rows.append(processor(self.audio[first:first + size],
                                              reset=False))
                        before = processor.samples
                        self.assertEqual(processor([], reset=False).shape, (0, 2))
                        self.assertEqual(processor.samples, before)
                    assert_array_equal(np.concatenate(rows), expected)
                    self.assertEqual(processor.windows, expected_windows)
                    self.assertEqual(processor.samples, len(self.audio))


if __name__ == '__main__':
    unittest.main()
