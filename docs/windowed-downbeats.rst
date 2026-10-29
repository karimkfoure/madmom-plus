Windowed online downbeat voting
==============================

``WindowedDownBeatTrackingProcessor`` adds delayed bar-position evidence to causal beat detections. It runs the released bidirectional downbeat network on completed audio windows. The network can use future context within a completed window, but it cannot read audio that has not arrived. The processor applies each result to subsequent beats and does not revise past events. This makes bar position available during a stream while preserving the causal beat grid.

This API is opt-in. Existing processors, command-line programs, model files and defaults keep their behavior. The downbeat models retain their existing licence; see ``madmom/models/LICENSE``.

Stream audio
------------

Supply mono floating point PCM at 44100 Hz. Floating point PCM normally has values from -1 to +1. Resample audio and mix channels before each stream. Integer PCM is rejected because converting it without amplitude scaling changes the model input::

    import numpy as np
    from madmom.audio.signal import Signal
    from madmom.features.downbeats import WindowedDownBeatTrackingProcessor

    audio = Signal('tests/data/audio/sample.wav', sample_rate=44100,
                   num_channels=1, dtype=np.float32)
    with WindowedDownBeatTrackingProcessor(window=6., hop=2.) as tracker:
        for first in range(0, len(audio), 441):
            beats = tracker(audio[first:first + 441], reset=False)
            # Each row contains event time [seconds] and bar position.
            # Position zero means that the bar has not been acquired.
            downbeats = beats[beats[:, 1] == 1, 0]

Use ``reset=False`` after the first call to retain audio history, model state and the clock. A call with ``reset=True`` starts a new stream. Empty input does not advance the audio clock. The default Python path and the optional ``beat_backend='native'`` path use the same frame convention: the first frame refers to sample zero, then frames advance by 441 samples. Each frame uses only the samples through its reference sample. There is no tail padding or flush operation.

The first downbeat-network call starts after ``window`` seconds of audio have arrived. Later calls occur every ``hop`` seconds. Both durations must be positive multiples of 0.01 seconds, and ``hop`` must not exceed ``window``. Larger windows give slow tempos and larger meters more complete bars, at a cost in computation and response time. Smaller hops refresh evidence more often but repeat more network computation over overlapping audio. The initial six-second window is not a guaranteed acquisition time: the voter needs enough detected beats in every possible bar position.

The default runs window inference synchronously. This gives repeatable results independent of audio chunk size, but each network call pauses the caller. Set ``background=True`` to use one background thread. The thread runs at most one inference job at a time. It skips scheduled windows while busy and applies a completed result at a later processing call. This bounds pending work instead of accumulating delayed jobs. Computation speed and call timing can therefore change the exact beat on which new evidence takes effect. Keep audio capture separate from this processor when a capture callback must meet a strict deadline.

Use a context manager or call ``close()`` to release the owned beat backend and background worker. Supplied ``beat_processor`` instances remain caller-owned. Each processor instance must be used from one calling thread at a time.

Vote on an existing beat grid
----------------------------

``BarPhaseVotingProcessor`` accepts detected beat times and separately computed downbeat activations. This interface lets a caller select its own causal beat detector or schedule downbeat inference in a separate worker::

    from madmom.features.downbeats import BarPhaseVotingProcessor

    voter = BarPhaseVotingProcessor(beats_per_bar=(3, 4))
    rows = voter([0., .5, 1., 1.5], now=1.5)
    # Later, submit new causal beat detections as they arrive.
    rows = voter([2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5],
                 now=6., reset=False)
    # activations has shape (600, 2), with beat and downbeat columns.
    # voter([], activations, start=0., fps=100., now=6., reset=False)
    # Future detections can now receive acquired bar positions.
    rows = voter([6.5], now=6.5, reset=False)

Beat times must be finite, strictly increasing and no later than ``now``. The clock must not move backwards. Activation windows must arrive in order of increasing end time. Their ``start`` and ``fps`` values place each activation frame on the same clock as the beats. A complete evidence window must be available at ``now``. Submit only model results computed from audio that has arrived; timestamps cannot establish this on their own.

If a call contains both beats and evidence, the voter emits those beats before applying the evidence. Submit evidence with an empty beat array when it must affect the next beat call. Returned arrays have two columns: beat time in seconds and natural bar count from one. Zero means that the bar is unknown.

For each candidate meter, the voter groups downbeat activation peaks by beat index within the bar. It requires ``min_beats_per_phase`` examples in every group. A moving average with time constant ``memory`` seconds retains recent evidence without giving an old track unlimited influence. ``phase_hysteresis`` and ``meter_hysteresis`` require a challenger to exceed the current choice before changing it. Similar scores favor the smaller meter, which limits accidental grouping of two three-beat bars into one six-beat bar. An acquired larger multiple can release its decision when a smaller divisor has similar contrast and a clear phase. This uses a narrower tie band than initial meter selection to limit repeated switching. Candidate meters are a hypothesis set, not a guarantee: missing candidates and ambiguous accents can prevent a correct result.

The published ``meter`` is ``None`` before acquisition. ``phase_class`` identifies the selected beat-index class, and ``phase`` gives a corresponding recent downbeat anchor in seconds. ``period`` is the estimated beat interval in seconds. ``confidence`` is a raw score from phase agreement, between zero and one. It is not a calibrated correctness probability or a music-presence signal. Track changes can replace both meter and phase. Three consistent changed intervals clear accumulated bar evidence, including repeated integer gaps that indicate a slower tempo. A single regular gap still preserves beat indexing. ``evidence_expiry`` releases a stale bar decision.

``fill_gaps=True`` permits a bounded number of missing events between observed beats. It requires an acquired bar, sufficient confidence and agreement with the recent beat period. Inserted events use their estimated beat times and are delivered when a later detection confirms the gap. They are therefore late. The default is ``False`` because startup errors and changes in tempo can look like missing beats. The logical beat index can still advance over a regular gap with filling disabled, so a missing event does not automatically shift bar count.

Reliability and limits
----------------------

Bar voting depends on the causal beat grid. A stable but incorrect grid can produce stable, incorrect bar positions. Overlapping inference windows contain correlated evidence, and short windows can give too few examples at slow tempos. The raw reliability score does not detect silence, speech or music. A caller that needs a presence policy must implement and validate that policy separately.

The implementation and tests use madmom's existing model and processor interfaces. The window-voting design was informed by `BeatTrackerJS <https://github.com/mathigatti/BeatTrackerJS>`_ by Mathias Gatti, snapshot ``604869f580fa695aafd9685b94c6c3247ed0ed87``. The references are its `README <https://github.com/mathigatti/BeatTrackerJS/blob/604869f580fa695aafd9685b94c6c3247ed0ed87/README.md>`_, `migration notes <https://github.com/mathigatti/BeatTrackerJS/blob/604869f580fa695aafd9685b94c6c3247ed0ed87/docs/MIGRATION.md>`_ and `experiment notes <https://github.com/mathigatti/BeatTrackerJS/blob/604869f580fa695aafd9685b94c6c3247ed0ed87/docs/EXPERIMENTS.md>`_. The Python implementation is independent; no JavaScript source or model files from that project are copied. Its reported comparisons use offline madmom as a reference, so they do not establish accuracy against human downbeat annotations.
