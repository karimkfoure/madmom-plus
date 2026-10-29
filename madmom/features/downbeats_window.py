# encoding: utf-8
"""Delayed bar evidence on a causal beat grid.

The rolling-window and phase-voting design is informed by BeatTrackerJS by
Mathias Gatti, revision 604869f580fa695aafd9685b94c6c3247ed0ed87:
https://github.com/mathigatti/BeatTrackerJS . This is an independent
implementation using madmom's existing models and processors.
"""

from collections import deque

import numpy as np

from ..processors import Processor


def _finite_scalar(value):
    try:
        return (np.isscalar(value) and np.isrealobj(value) and
                not isinstance(value, (bool, np.bool_)) and
                bool(np.isfinite(value)))
    except (TypeError, ValueError):
        return False


class BarPhaseVotingProcessor(Processor):
    """Assign bar positions to detected beats using recent downbeat evidence.

    Parameters
    ----------
    beats_per_bar : int or sequence of int, optional
        Candidate meters. Smaller meters win near ties between multiples.
    memory : float, optional
        Time constant of the moving average of window evidence [seconds].
    min_score : float, optional
        Minimum contrast above the mean phase activation for meter selection.
    phase_hysteresis, meter_hysteresis : float, optional
        Required score ratios to replace an established phase or meter.
    min_beats_per_phase : int, optional
        Minimum observed beats in every phase class of a window.
    fill_gaps : bool, optional
        Insert missing beats between two observed beats on a stable grid.
        The default preserves the input beat events.
    fill_confidence : float, optional
        Minimum raw phase margin for inserting missing beats.
    max_fill : int, optional
        Maximum missing beats to bridge. Filling never predicts future beats.
    history : float, optional
        Maximum retained beat history [seconds].
    evidence_expiry : float, optional
        Clear bar state after this many seconds without fresh evidence.
    peak_radius : float, optional
        Search radius around each beat in the activation window [seconds].
    tempo_change : float, optional
        Relative tempo change that clears old votes after three consistent
        new intervals. Integer gaps are treated as missing beats first.

    Notes
    -----
    Window scores use mean downbeat activation per beat-index phase class.
    Meter scores compare the best phase with the mean of all phases. The
    moving average uses elapsed evidence time, so its memory does not grow
    with recording length. Overlapping windows are correlated observations.

    ``meter`` is None until acquired. ``phase_class`` is the downbeat beat
    index modulo the selected meter. ``phase`` is an absolute downbeat anchor
    in seconds, derived from the latest beat and median interval. ``period``
    is that interval in seconds. ``bar_position`` starts at zero.
    ``confidence`` is a raw phase margin in [0, 1], not a calibrated probability
    of correctness or music presence. A caller owns any presence policy.
    """

    def __init__(self, beats_per_bar=(3, 4), memory=3., min_score=.15,
                 phase_hysteresis=1.3, meter_hysteresis=1.3,
                 min_beats_per_phase=2, fill_gaps=False, fill_confidence=.6,
                 max_fill=2, history=30., evidence_expiry=12.,
                 peak_radius=.07, tempo_change=.2):
        meters = np.atleast_1d(np.asarray(beats_per_bar, dtype=float))
        if (meters.ndim != 1 or not len(meters) or
                not np.all(np.isfinite(meters)) or np.any(meters < 2) or
                np.any(meters != np.floor(meters)) or
                len(np.unique(meters)) != len(meters)):
            raise ValueError('beats_per_bar must contain distinct integers >= 2.')
        values = (memory, min_score, phase_hysteresis, meter_hysteresis,
                  min_beats_per_phase, fill_confidence, max_fill, history,
                  evidence_expiry, peak_radius, tempo_change)
        if (not all(_finite_scalar(v) for v in values) or
                memory <= 0 or min_score < 0 or phase_hysteresis < 1 or
                meter_hysteresis < 1 or min_beats_per_phase < 1 or
                int(min_beats_per_phase) != min_beats_per_phase or
                not 0 <= fill_confidence <= 1 or max_fill < 0 or
                int(max_fill) != max_fill or history <= 0 or
                evidence_expiry <= 0 or peak_radius < 0 or
                not 0 < tempo_change < 1 or
                not isinstance(fill_gaps, (bool, np.bool_))):
            raise ValueError('Invalid bar phase voting parameters.')
        self.beats_per_bar = tuple(sorted(meters.astype(int)))
        self.memory, self.min_score = float(memory), float(min_score)
        self.phase_hysteresis = float(phase_hysteresis)
        self.meter_hysteresis = float(meter_hysteresis)
        self.min_beats_per_phase = int(min_beats_per_phase)
        self.fill_gaps, self.fill_confidence = bool(fill_gaps), float(fill_confidence)
        self.max_fill = int(max_fill)
        self.history, self.evidence_expiry = float(history), float(evidence_expiry)
        self.peak_radius, self.tempo_change = float(peak_radius), float(tempo_change)
        self.reset()

    def reset(self):
        """Clear beat history, bar votes and the stream clock."""
        self._beats = deque()
        self._intervals = deque(maxlen=8)
        self._changes = deque(maxlen=3)
        self._index = -1
        self._last_beat = -np.inf
        self._now = 0.
        self.period = np.nan
        self._last_evidence = -np.inf
        self._last_window = -np.inf
        self._clear_votes()

    def _clear_votes(self):
        self._votes = {m: np.zeros(m) for m in self.beats_per_bar}
        self._scores = {m: 0. for m in self.beats_per_bar}
        self.meter = None
        self.phase_class = None
        self.confidence = 0.
        self._vote_time = -np.inf

    def reset_bar(self):
        """Clear bar evidence while preserving the beat clock and interval fit.

        Call this at a known recording boundary. No audio classification or
        silence threshold is applied by this processor.
        """
        self._clear_votes()
        self._beats.clear()
        self._last_evidence = -np.inf

    @property
    def bar_position(self):
        """Latest beat position, starting at zero, or NaN before acquisition."""
        if self.meter is None:
            return np.nan
        return (self._index - self.phase_class) % self.meter

    @property
    def phase(self):
        """Absolute latest downbeat anchor [seconds], or NaN if unavailable."""
        if self.meter is None or not np.isfinite(self.period):
            return np.nan
        return self._last_beat - self.bar_position * self.period

    def _expire(self, now):
        if now - self._last_evidence > self.evidence_expiry:
            self._clear_votes()
        while self._beats and self._beats[0][1] < now - self.history:
            self._beats.popleft()

    def _append_beat(self, time):
        rows = []
        steps = 1
        previous_period = self.period
        if np.isfinite(self._last_beat):
            interval = time - self._last_beat
            if interval > self.evidence_expiry:
                self.reset_bar()
                self._intervals.clear()
                self._changes.clear()
                self.period = np.nan
            elif np.isfinite(previous_period) and len(self._intervals) >= 3:
                candidate = int(np.rint(interval / previous_period))
                regular_gap = (1 <= candidate <= self.max_fill + 1 and
                               abs(interval / previous_period - candidate) <= .2)
                if regular_gap:
                    steps = candidate
                if abs(interval / previous_period - 1.) > self.tempo_change:
                    self._changes.append(interval)
                    if (len(self._changes) == 3 and
                            np.ptp(self._changes) <= .1 * np.mean(self._changes)):
                        self.reset_bar()
                        self._intervals.clear()
                        self._intervals.extend(self._changes)
                        self._changes.clear()
                        # Repeated integer gaps can be a new, slower tempo.
                        # Isolated missed detections still keep their indices.
                        steps = 1
                else:
                    self._changes.clear()
            if (steps > 1 and self.fill_gaps and self.meter is not None and
                    self.confidence >= self.fill_confidence and
                    len(self._intervals) >= 3 and
                    np.median(np.abs(np.asarray(self._intervals) - previous_period))
                    <= .1 * previous_period):
                for step in range(1, steps):
                    index = self._index + step
                    rows.append((self._last_beat + step * interval / steps,
                                 (index - self.phase_class) % self.meter + 1))
            self._intervals.append(interval / steps)
            self.period = float(np.median(self._intervals))
        self._index += steps
        self._last_beat = time
        self._beats.append((self._index, time))
        position = (0 if self.meter is None else
                    (self._index - self.phase_class) % self.meter + 1)
        rows.append((time, position))
        return rows

    def _update_evidence(self, activations, start, fps, end, now):
        self._last_window = end
        if now - end > self.evidence_expiry:
            return
        radius = int(np.ceil(self.peak_radius * fps))
        indices, peaks = [], []
        for index, time in self._beats:
            frame = int(np.rint((time - start) * fps))
            if radius <= frame < len(activations) - radius:
                indices.append(index)
                peaks.append(np.max(activations[frame - radius:frame + radius + 1, 1]))
        indices, peaks = np.asarray(indices, dtype=int), np.asarray(peaks)
        elapsed = end - self._vote_time
        decay = np.exp(-elapsed / self.memory)
        self._vote_time = end
        usable = False
        for meter in self.beats_per_bar:
            counts = np.bincount(indices % meter, minlength=meter)
            if np.all(counts >= self.min_beats_per_phase):
                sums = np.bincount(indices % meter, weights=peaks, minlength=meter)
                means = sums / counts
                mean = np.mean(means)
                votes = means / mean if mean > 1e-8 else np.zeros(meter)
                contrast = max(0., float(np.max(votes) - 1.))
                usable = usable or mean > 1e-8
            else:
                votes, contrast = np.zeros(meter), 0.
            self._votes[meter] = decay * self._votes[meter] + (1. - decay) * votes
            self._scores[meter] = decay * self._scores[meter] + (1. - decay) * contrast
        if not usable:
            self.confidence = 0.
            return
        self._last_evidence = end
        best = self.beats_per_bar[0]
        for meter in self.beats_per_bar[1:]:
            if self._scores[meter] > self.meter_hysteresis * self._scores[best]:
                best = meter
        # Equal contrasts cannot distinguish a bar from two identical bars.
        # Permit recovery to a smaller divisor through a narrower tie band;
        # requiring a separated phase and a narrower band avoids chattering.
        preferred_multiple = False
        if self.meter is not None and best < self.meter and self.meter % best == 0:
            ordered = np.sort(self._votes[best])
            preferred_multiple = (
                self._scores[best] * np.sqrt(self.meter_hysteresis) >=
                self._scores[self.meter] and
                ordered[-1] > self.phase_hysteresis * ordered[-2])
        if (self._scores[best] >= self.min_score and
                (self.meter is None or
                 preferred_multiple or
                 self._scores[best] > self.meter_hysteresis * self._scores[self.meter])):
            if best != self.meter:
                self.meter = best
                self.phase_class = int(np.argmax(self._votes[best]))
        if self.meter is None:
            self.confidence = 0.
            return
        votes = self._votes[self.meter]
        best_phase = int(np.argmax(votes))
        if votes[best_phase] > self.phase_hysteresis * votes[self.phase_class]:
            self.phase_class = best_phase
        selected = votes[self.phase_class]
        other = np.max(np.delete(votes, self.phase_class))
        self.confidence = float(max(0., (selected - other) / selected)) if selected > 0 else 0.

    def process(self, beats, activations=None, start=0., fps=100., now=None,
                reset=True, **kwargs):
        """Emit new beats, then apply evidence to subsequent beat labels.

        ``beats`` contains strictly increasing new beat times in seconds.
        Returned rows contain time and beat number starting at one; zero
        means the bar is not acquired. Previously returned rows never change.

        ``activations`` is an optional (frames, 2) array of beat and downbeat
        probabilities from an audio window. Frame zero refers to ``start``
        seconds; ``fps`` is its frame rate. Supply a window only after all its
        audio is available at ``now``. Window end times must increase, although
        windows can overlap. Evidence affects the next call's labels. To
        deliver evidence separately, use an empty beat array.

        ``now`` is the monotonic delivery clock in seconds. It defaults to
        the latest input time. Empty calls with a later clock expire old
        evidence. Use ``reset=False`` to continue a stream.
        """
        beats = np.asarray(beats, dtype=float)
        last = -np.inf if reset else self._last_beat
        previous_now = 0. if reset else self._now
        last_window = -np.inf if reset else self._last_window
        if (beats.ndim != 1 or not np.all(np.isfinite(beats)) or
                np.any(beats < 0) or np.any(np.diff(beats) <= 0) or
                (len(beats) and beats[0] <= last)):
            raise ValueError('beats must be new, finite, increasing nonnegative times.')
        end = -np.inf
        if activations is not None:
            activations = np.asarray(activations, dtype=float)
            if (activations.ndim != 2 or activations.shape[1] != 2 or
                    not len(activations) or not np.all(np.isfinite(activations)) or
                    np.any(activations < 0) or np.any(activations > 1) or
                    np.any(activations.sum(axis=1) > 1. + 1e-6) or
                    not _finite_scalar(start) or start < 0 or
                    not _finite_scalar(fps) or fps <= 0):
                raise ValueError('Expected window probabilities (frames, 2), start >= 0 and fps > 0.')
            end = start + len(activations) / fps
            if end <= last_window:
                raise ValueError('Activation window end times must increase.')
        if now is None:
            now = max(previous_now, end, beats[-1] if len(beats) else 0.)
        if (not _finite_scalar(now) or now < previous_now or now < end - 1e-9 or
                (len(beats) and beats[-1] > now)):
            raise ValueError('now must be finite, monotonic and cover all input.')
        if reset:
            self.reset()
        rows = []
        for beat in beats:
            self._expire(beat)
            rows.extend(self._append_beat(float(beat)))
        self._now = float(now)
        self._expire(now)
        if activations is not None:
            self._update_evidence(activations, float(start), float(fps), end, now)
        return np.asarray(rows, dtype=float).reshape((-1, 2))

    process_online = process


class _PythonBeatTracker:
    """Adapt existing processors to the native sample-clock interface."""

    def __init__(self):
        from .beats import RNNBeatProcessor, DBNBeatTrackingProcessor
        # Each supplied frame already contains only past samples. Framing it
        # from its beginning avoids shifting it a second time.
        self.rnn = RNNBeatProcessor(online=True, origin='stream', num_frames=1,
                                    fps=100, num_threads=1)
        self.decoder = DBNBeatTrackingProcessor(online=True, fps=100)
        self.reset()

    def reset(self):
        self.frame = np.zeros(2048, dtype=np.float32)
        self.samples = self.next_frame = 0
        self._reset = True
        self.decoder.reset()

    def push(self, audio):
        rows = []
        offset = 0
        while offset < len(audio):
            count = min(len(audio) - offset, self.next_frame - self.samples + 1)
            self.frame[:-count] = self.frame[count:]
            self.frame[-count:] = audio[offset:offset + count]
            offset += count
            self.samples += count
            if self.samples > self.next_frame:
                activation = np.atleast_1d(self.rnn(self.frame, reset=self._reset))
                self._reset = False
                beats = self.decoder(activation, reset=False)
                rows.append((self.next_frame / 44100., float(activation[0]),
                             int(len(beats) > 0), self.decoder.tempo))
                self.next_frame += 441
        return np.asarray(rows, dtype=float).reshape((-1, 4))


class WindowedDownBeatTrackingProcessor(Processor):
    """Track causal beats with delayed, rolling-window bar-position evidence.

    Parameters
    ----------
    window, hop : float, optional
        Downbeat audio window length and inference hop [seconds]. Both must
        be positive multiples of 10 ms, and hop must not exceed window.
    beat_backend : {'python', 'native'}, optional
        Beat implementation. The native backend requires its opt-in build.
    background : bool, optional
        Run downbeat inference in one owned background thread. Busy workers
        skip new windows; there is no queued backlog. Results take effect on
        the next audio update after completion. The default is synchronous
        and produces the same result for every input chunk size.
    beat_processor : object, optional
        Caller-owned replacement with push(audio) returning native-style
        (frames, 4) rows and a reset() method.
    downbeat_processor : callable, optional
        Caller-owned replacement for RNNDownBeatProcessor. Called with a
        complete audio window and reset=True; returns (frames, 2) at 100 fps.
    **kwargs : dict
        Options passed to BarPhaseVotingProcessor. History must cover window.

    Notes
    -----
    Input is mono floating point PCM at 44100 Hz, normally in [-1, 1]. Mix
    channels and resample before calling. Integer input is rejected because
    implicit conversion would change the model's amplitude scale.

    The bidirectional downbeat network reads the complete received window;
    it never reads unreceived audio. It does not revise earlier beat events.
    Startup requires a full window and enough observed beats per candidate
    phase. Larger meters and slow tempos can require longer windows.

    Use a context manager or close() when background=True. Each instance
    must be used by one caller thread. Capture, presence policy and transport
    are outside this processor. Models retain their original licence.
    """

    def __init__(self, window=6., hop=2., beat_backend='python',
                 background=False, beat_processor=None, downbeat_processor=None,
                 **kwargs):
        for value in (window, hop):
            if (not _finite_scalar(value) or value <= 0 or
                    not np.isclose(value * 100, np.rint(value * 100), rtol=0, atol=1e-8)):
                raise ValueError('window and hop must be positive multiples of 10 ms.')
        if hop > window or beat_backend not in ('python', 'native'):
            raise ValueError('Require hop <= window and a python or native beat backend.')
        if not isinstance(background, (bool, np.bool_)):
            raise ValueError('background must be a boolean.')
        self.voter = BarPhaseVotingProcessor(**kwargs)
        if self.voter.history < window:
            raise ValueError('Beat history must cover the audio window.')
        self.window, self.hop = float(window), float(hop)
        self.beat_backend, self.background = beat_backend, bool(background)
        self._window_samples = int(round(window * 100)) * 441
        self._hop_samples = int(round(hop * 100)) * 441
        self._owns_beats = beat_processor is None
        if beat_processor is None:
            if beat_backend == 'native':
                from ..native import NativeBeatTracker
                beat_processor = NativeBeatTracker()
            else:
                beat_processor = _PythonBeatTracker()
        self.beat_processor = beat_processor
        if downbeat_processor is None:
            from .downbeats import RNNDownBeatProcessor
            downbeat_processor = RNNDownBeatProcessor(num_threads=1)
        self.downbeat_processor = downbeat_processor
        self._executor = None
        self._pending = None
        self._generation = 0
        self._closed = False
        self.reset()

    def reset(self):
        """Reset audio, beat and vote state; discard pending old bar evidence."""
        if self._closed:
            raise RuntimeError('The windowed tracker is closed.')
        self._generation += 1
        self.beat_processor.reset()
        self.voter.reset()
        self._audio = np.zeros(self._window_samples, dtype=np.float32)
        self._write = 0
        self.samples = 0
        self._next_window = self._window_samples
        self.windows = self.skipped_windows = 0

    @property
    def meter(self):
        return self.voter.meter

    @property
    def confidence(self):
        return self.voter.confidence

    @property
    def period(self):
        return self.voter.period

    @property
    def phase(self):
        return self.voter.phase

    @property
    def bar_position(self):
        return self.voter.bar_position

    def _poll(self, now):
        if self._pending is not None and self._pending[0].done():
            future, start, generation = self._pending
            self._pending = None
            # A reset does not start a concurrent call on the same network.
            if generation == self._generation:
                self.voter([], activations=future.result(), start=start,
                           now=now, reset=False)

    def _window(self, now):
        self._poll(now)
        if self._pending is not None:
            self.skipped_windows += 1
            return
        audio = np.concatenate((self._audio[self._write:], self._audio[:self._write]))
        start = (self.samples - self._window_samples) / 44100.
        self.windows += 1
        if self.background:
            if self._executor is None:
                from concurrent.futures import ThreadPoolExecutor
                self._executor = ThreadPoolExecutor(max_workers=1)
            future = self._executor.submit(self.downbeat_processor, audio, reset=True)
            self._pending = (future, start, self._generation)
        else:
            self.voter([], activations=self.downbeat_processor(audio, reset=True),
                       start=start, now=now, reset=False)

    def process(self, audio, reset=True, **kwargs):
        """Process new PCM and return (time [seconds], bar beat number) rows.

        Beat number zero means unknown bar position. Use reset=False for each
        subsequent chunk. Empty calls poll completed background work without
        advancing audio time. A reset never publishes old worker results.
        """
        if self._closed:
            raise RuntimeError('The windowed tracker is closed.')
        audio = np.asarray(audio)
        if audio.ndim != 1 or audio.dtype.kind != 'f':
            raise ValueError('Expected one-dimensional floating point mono PCM at 44100 Hz.')
        with np.errstate(over='ignore', invalid='ignore'):
            audio = np.asarray(audio, dtype=np.float32)
        if not np.all(np.isfinite(audio)):
            raise ValueError('PCM must contain finite float32 values.')
        if reset:
            self.reset()
        output = []
        offset = 0
        while offset < len(audio):
            now = self.samples / 44100.
            self._poll(now)
            count = min(len(audio) - offset, 441 - self.samples % 441,
                        self._next_window - self.samples)
            block = audio[offset:offset + count]
            # The ring is independent of caller chunk boundaries.
            first = min(count, self._window_samples - self._write)
            self._audio[self._write:self._write + first] = block[:first]
            if first < count:
                self._audio[:count - first] = block[first:]
            self._write = (self._write + count) % self._window_samples
            self.samples += count
            offset += count
            now = self.samples / 44100.
            frames = self.beat_processor.push(block)
            for frame in frames:
                beats = [frame[0]] if frame[2] else []
                output.extend(self.voter(beats, now=float(frame[0]), reset=False))
            self.voter([], now=now, reset=False)
            if self.samples == self._next_window:
                self._window(now)
                self._next_window += self._hop_samples
        self._poll(self.samples / 44100.)
        return np.asarray(output, dtype=float).reshape((-1, 2))

    process_online = process

    def close(self):
        """Wait for the owned worker and release owned beat resources."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=True)
        finally:
            self._pending = None
            if self._owns_beats and hasattr(self.beat_processor, 'close'):
                self.beat_processor.close()

    def __enter__(self):
        if self._closed:
            raise RuntimeError('The windowed tracker is closed.')
        return self

    def __exit__(self, *args):
        self.close()
