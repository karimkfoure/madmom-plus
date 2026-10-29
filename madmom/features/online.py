# encoding: utf-8
"""Causal pulse estimation and optional rhythmic reliability signals."""

from __future__ import absolute_import, division, print_function

from collections import deque, namedtuple

import numpy as np

from ..processors import Processor


PulseState = namedtuple('PulseState', 'period phase confidence valid_until valid')
PulseState.__doc__ = """Immutable pulse estimate.

``period`` is the fitted beat interval in seconds. ``phase`` is a fitted grid
anchor near the most recent accepted beat, also in seconds. ``confidence`` is
an uncalibrated grid-consistency score between zero and one. ``valid_until``
is the latest accepted beat time plus the configured expiry. ``valid`` applies
the confidence threshold, acquisition requirement and caller's current time.
Invalid states can retain a finite estimate for diagnostics. An unavailable
estimate has NaN period and phase, zero confidence and negative-infinite expiry.
"""


class BeatPeriodProcessor(Processor):
    """Fit a bounded, causal pulse grid to detected beat times.

    Parameters
    ----------
    window_size : int, optional
        Maximum number of recent beat times to retain.
    min_intervals : int, optional
        Minimum number of distinct fitted grid intervals before validity.
    expiry : float, optional
        Seconds of validity after the latest accepted beat.
    min_confidence : float, optional
        Minimum grid-consistency score required for validity. Zero disables
        this threshold but does not disable acquisition or expiry.
    preferred_bpm : tuple, optional
        Preferred ``(minimum, maximum)`` tempo. If specified, shift the median
        interval by powers of two into this range when possible. This is a
        user preference, not evidence that one metrical level is correct.
    tolerance : float, optional
        Maximum phase error as a fraction of the candidate period. A 10 ms
        floor prevents quantized beat times from losing all support.

    Notes
    -----
    The median interval initializes a grid. Circular phase consensus rejects
    isolated timing errors. A median of pairwise grid slopes fits its period,
    including integer gaps caused by missed beats. The history is bounded and
    uses past detections only. Stable tempo changes require the history to
    adapt; this processor is intended for pulse publication, not note timing.

    The score measures agreement with a periodic grid. Periodic speech and
    noise can also produce a valid grid. It is not a music-presence probability.
    Existing beat times and detector state are not modified.
    """

    def __init__(self, window_size=16, min_intervals=3, expiry=4.,
                 min_confidence=0.5, preferred_bpm=None, tolerance=0.12):
        if (int(window_size) != window_size or window_size < 4 or
                int(min_intervals) != min_intervals or min_intervals < 2 or
                min_intervals >= window_size):
            raise ValueError('Require window_size >= 4 and '
                             '2 <= min_intervals < window_size.')
        if not np.isfinite(expiry) or expiry <= 0:
            raise ValueError('expiry must be finite and positive.')
        if not np.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
            raise ValueError('min_confidence must be between zero and one.')
        if not np.isfinite(tolerance) or not 0 < tolerance < 0.5:
            raise ValueError('tolerance must be between zero and one half.')
        if preferred_bpm is not None:
            preferred_bpm = np.asarray(preferred_bpm, dtype=float)
            if (preferred_bpm.shape != (2,) or
                    not np.all(np.isfinite(preferred_bpm)) or
                    not 0 < preferred_bpm[0] < preferred_bpm[1]):
                raise ValueError('preferred_bpm must be two increasing '
                                 'positive finite values.')
            preferred_bpm = tuple(preferred_bpm)
        self.window_size = int(window_size)
        self.min_intervals = int(min_intervals)
        self.expiry = float(expiry)
        self.min_confidence = float(min_confidence)
        self.preferred_bpm = preferred_bpm
        self.tolerance = float(tolerance)
        self.reset()

    def reset(self):
        """Clear beat history, the publication clock and the fitted grid."""
        self._beats = deque(maxlen=self.window_size)
        self._last_now = -np.inf
        self._estimate = PulseState(np.nan, np.nan, 0., -np.inf, False)
        self._acquired = False

    def _prefer_period(self, period):
        if self.preferred_bpm is None:
            return period
        low, high = self.preferred_bpm
        bpm = 60. / period
        lower_power = np.ceil(np.log2(low / bpm))
        upper_power = np.floor(np.log2(high / bpm))
        if lower_power <= upper_power:
            power = np.clip(0., lower_power, upper_power)
            return period / (2. ** power)
        return period

    def _fit(self):
        times = np.asarray(self._beats)
        if len(times) < 3:
            self._estimate = PulseState(np.nan, np.nan, 0., -np.inf, False)
            self._acquired = False
            return
        period = self._prefer_period(np.median(np.diff(times)))
        tolerance = max(0.01, self.tolerance * period)
        relative = times - times[-1]
        # Choose a phase with the largest support. Prefer the latest phase on
        # ties, which makes the optional slower octave use the latest parity.
        residuals = ((relative[:, None] - relative[None, :] + period / 2.) %
                     period - period / 2.)
        support = np.sum(np.abs(residuals) <= tolerance, axis=0)
        best = len(times) - 1 - np.argmax(support[::-1])
        inliers = np.abs(residuals[:, best]) <= tolerance
        angles = 2. * np.pi * residuals[inliers, best] / period
        correction = np.angle(np.mean(np.exp(1j * angles))) * period / (2. * np.pi)
        anchor = relative[best] + correction
        indices = np.rint((relative - anchor) / period)
        # Integer grid indices preserve missed beats. Equal indices caused by
        # duplicate detections do not contribute a zero period to the fit.
        accepted = np.flatnonzero(inliers)
        differences = indices[accepted, None] - indices[None, accepted]
        time_differences = relative[accepted, None] - relative[None, accepted]
        pairs = differences > 0
        if not np.any(pairs):
            self._estimate = PulseState(np.nan, np.nan, 0., -np.inf, False)
            self._acquired = False
            return
        period = np.median(time_differences[pairs] / differences[pairs])
        anchor = np.median(relative[inliers] - indices[inliers] * period)
        errors = relative - (anchor + indices * period)
        inliers &= np.abs(errors) <= tolerance
        accepted = np.flatnonzero(inliers)
        if not len(accepted):
            self._estimate = PulseState(np.nan, np.nan, 0., -np.inf, False)
            self._acquired = False
            return
        distinct = len(np.unique(indices[inliers]))
        consistency = np.mean(inliers)
        dispersion = np.median(np.abs(errors[inliers]))
        confidence = consistency * np.exp(-dispersion / tolerance)
        self._acquired = distinct > self.min_intervals
        latest = accepted[-1]
        phase = times[-1] + anchor + indices[latest] * period
        self._estimate = PulseState(float(period), float(phase),
                                    float(confidence),
                                    float(times[latest] + self.expiry), False)

    def process(self, beats, now=None, reset=True, **kwargs):
        """Return the current pulse estimate after new beat detections.

        Parameters
        ----------
        beats : array_like, shape (num_beats,)
            New, strictly increasing beat times in seconds. Each beat must be
            later than all beats supplied in earlier calls. Supply only beats
            available at ``now``; input timestamps alone cannot prove causality.
        now : float, optional
            Current causal clock in seconds. Defaults to the latest input beat
            or the previous clock for an empty input. Supply it explicitly on
            empty calls to expire a stopped stream. It must not move backwards.
        reset : bool, optional
            Clear state first. Use False for successive chunks.

        Returns
        -------
        PulseState
            Fitted grid, consistency and explicit validity at ``now``.
        """
        beats = np.asarray(beats, dtype=float)
        if (beats.ndim != 1 or not np.all(np.isfinite(beats)) or
                np.any(beats < 0) or np.any(np.diff(beats) <= 0)):
            raise ValueError('beats must be finite, nonnegative and '
                             'strictly increasing in one dimension.')
        previous_now = -np.inf if reset else self._last_now
        if now is None:
            now = beats[-1] if len(beats) else max(0., previous_now)
        if (not np.isfinite(now) or now < 0 or now < previous_now or
                (len(beats) and beats[-1] > now)):
            raise ValueError('now must be finite, nonnegative, nondecreasing '
                             'and no earlier than the latest beat.')
        if not reset and len(beats) and self._beats and beats[0] <= self._beats[-1]:
            raise ValueError('new beats must follow previous beat times.')
        if reset:
            self.reset()
        for beat in beats:
            if self._beats and beat - self._beats[-1] > self.expiry:
                self._beats.clear()
            self._beats.append(float(beat))
            self._fit()
        self._last_now = float(now)
        state = self._estimate
        return state._replace(valid=bool(self._acquired and
                              state.confidence >= self.min_confidence and
                              now < state.valid_until))


class RhythmPresenceProcessor(Processor):
    """Estimate causal rhythmic evidence from beat activations.

    The output is an uncalibrated confidence between zero and one for each
    input frame. A normalized autocorrelation searches the configured tempo
    range over a bounded window. Low activation strength or constant input
    reduces confidence. This can support abstention when rhythm evidence is
    weak; periodic speech or noise can pass it. It is not a music classifier.

    Parameters
    ----------
    fps : float, optional
        Activation frame rate.
    min_bpm, max_bpm : float, optional
        Tempo range used to search autocorrelation lags.
    window : float, optional
        History length in seconds. It must hold two minimum-tempo periods.
    update_interval : float, optional
        Seconds between autocorrelation updates. The previous score is held
        between updates. No future frame is used.
    activation_threshold : float, optional
        Peak activation required for full strength confidence.
    std_threshold : float, optional
        Activation standard deviation required for full strength confidence.
    correlation_floor : float, optional
        Correlation at or below this value gives zero confidence.
    """

    def __init__(self, fps=100., min_bpm=55., max_bpm=215., window=4.,
                 update_interval=0.1, activation_threshold=0.1,
                 std_threshold=0.03, correlation_floor=0.2):
        values = (fps, min_bpm, max_bpm, window, update_interval,
                  activation_threshold, std_threshold, correlation_floor)
        if not np.all(np.isfinite(values)):
            raise ValueError('parameters must be finite.')
        if (fps <= 0 or not 0 < min_bpm < max_bpm or window <= 0 or
                update_interval <= 0 or not 0 < activation_threshold <= 1 or
                std_threshold <= 0 or not 0 <= correlation_floor < 1):
            raise ValueError('invalid rhythm evidence parameters.')
        self.fps = float(fps)
        self.min_lag = max(1, int(np.ceil(60. * fps / max_bpm)))
        self.max_lag = int(np.floor(60. * fps / min_bpm))
        self.window_size = int(round(window * fps))
        if self.max_lag < self.min_lag or self.window_size < 2 * self.max_lag:
            raise ValueError('window must hold two minimum-tempo periods '
                             'and the tempo range must contain a frame lag.')
        self.update_frames = max(1, int(round(update_interval * fps)))
        self.activation_threshold = float(activation_threshold)
        self.std_threshold = float(std_threshold)
        self.correlation_floor = float(correlation_floor)
        self.reset()

    def reset(self):
        """Clear history and the last rhythm score."""
        self._buffer = deque(maxlen=self.window_size)
        self._frames = 0
        self._confidence = 0.

    def _update(self):
        if len(self._buffer) < 2 * self.max_lag:
            self._confidence = 0.
            return
        values = np.asarray(self._buffer)
        centered = values - np.mean(values)
        std = np.sqrt(np.mean(centered ** 2))
        if std <= np.finfo(float).eps:
            self._confidence = 0.
            return
        # Zero padding prevents circular wrap. Prefix energies normalize each
        # lag over its actual overlapping samples.
        n_fft = 1 << (2 * len(values) - 1).bit_length()
        spectrum = np.fft.rfft(centered, n=n_fft)
        products = np.fft.irfft(spectrum * spectrum.conjugate(), n=n_fft)
        energies = np.concatenate(([0.], np.cumsum(centered ** 2)))
        lags = np.arange(self.min_lag, self.max_lag + 1)
        norm = np.sqrt(energies[len(values) - lags] *
                       (energies[-1] - energies[lags]))
        correlations = np.divide(products[lags], norm,
                                 out=np.zeros_like(norm), where=norm > 0)
        periodicity = np.clip((np.max(correlations) - self.correlation_floor) /
                              (1. - self.correlation_floor), 0., 1.)
        strength = min(1., np.max(values) / self.activation_threshold,
                       std / self.std_threshold)
        self._confidence = float(periodicity * strength)

    def process(self, activations, reset=True, **kwargs):
        """Return rhythm confidence per frame; use reset=False to continue.

        Activations must be a one-dimensional array with finite values between
        zero and one. Scores have the same length and frame rate as the input.
        The complete result is invariant to chunk boundaries.
        """
        activations = np.asarray(activations, dtype=float)
        if (activations.ndim != 1 or not np.all(np.isfinite(activations)) or
                np.any(activations < 0) or np.any(activations > 1)):
            raise ValueError('activations must be one-dimensional finite '
                             'values between zero and one.')
        if reset:
            self.reset()
        result = np.empty(len(activations))
        for i, value in enumerate(activations):
            self._buffer.append(float(value))
            self._frames += 1
            if self._frames % self.update_frames == 0:
                self._update()
            result[i] = self._confidence
        return result
