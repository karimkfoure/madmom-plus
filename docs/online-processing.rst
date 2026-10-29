Causal pulse processing
========================

The online processors add bar tracking, spectral beat evidence, robust pulse state and an optional rhythm signal. Existing processors, model files and command-line defaults keep their behavior. Each new processor is opt-in.

A causal pipeline uses only audio captured at or before each frame timestamp. A forward decoder cannot remove future information already used by its activation source. For example, the bundled downbeat networks are bidirectional. Their output is suitable for offline analysis, but does not supply causal input to the new bar decoder.

Beat phase and period
----------------------

``SpectralBeatActivationProcessor`` accepts synchronized beat probability, low-frequency onset strength and high-frequency onset strength. Its bounded correction can favor bass evidence over high-frequency subdivisions. This prior can be wrong on syncopated bass, so select its strength on development data. Zero strength returns the original probabilities exactly.

``BeatPeriodProcessor`` fits a pulse grid from recent detected beats. It rejects isolated timing errors, allows integer gaps from missed beats and exposes an expiry time. Its default tempo has no octave preference. A supplied ``preferred_bpm`` range expresses a caller preference; it does not establish the annotated metrical level.

This example uses synthetic causal features at 100 frames per second. For audio input, calculate each onset from a frame that ends at its timestamp. Use running normalization rather than whole-recording normalization.

.. code-block:: python

    import numpy as np
    from madmom.features import (
        BeatPeriodProcessor, DBNBeatTrackingProcessor,
        SpectralBeatActivationProcessor)

    fps = 100
    features = np.zeros((1000, 3))
    features[:, 0] = 0.01
    features[::50, :2] = [0.8, 1.0]
    features[25::50, 2] = 1.0

    observation = SpectralBeatActivationProcessor(fps=fps)
    tracker = DBNBeatTrackingProcessor(fps=fps, online=True)
    period = BeatPeriodProcessor()

    for start in range(0, len(features), 10):
        block = features[start:start + 10]
        probabilities = observation(block, reset=False)
        beats = tracker(probabilities, reset=False)
        now = (start + len(block) - 1) / fps
        state = period(beats, now=now, reset=False)
        if state.valid:
            grid = (state.phase, state.period, state.valid_until)

In a running application, ``now`` must be the current delivery clock in the same time base as beat timestamps. The example has no capture or processing delay. A real caller must include those delays. Use empty updates such as ``period([], now=now, reset=False)`` when detections stop, or check ``valid_until`` directly. Do not keep publishing an expired grid.

Bar position
--------------

``OnlineDBNDownBeatTrackingProcessor`` uses forward filtering over bar position, tempo and allowed meters. Input rows contain mutually exclusive beat and downbeat activations. Returned rows contain current-frame times and beat numbers starting at one. The decoder does not revise earlier events or search for a future peak.

``CausalDownBeatActivationProcessor`` supplies a training-free experiment. It splits a causal beat probability using bass-onset contrast. It preserves the total beat probability, but its bar accents are not calibrated downbeat probabilities. Equal kicks cannot identify bar starts. Strong bass accents can also occur away from the downbeat.

.. code-block:: python

    from madmom.features import (
        CausalDownBeatActivationProcessor,
        OnlineDBNDownBeatTrackingProcessor)

    accent = CausalDownBeatActivationProcessor()
    bars = OnlineDBNDownBeatTrackingProcessor([3, 4], fps=fps)
    bar_activations = accent(features[:, :2])
    events = bars(bar_activations)
    downbeat_times = events[events[:, 1] == 1, 0]

Use ``reset=False`` on both processors for later chunks. Check bar labels separately from beat alignment. A plausible meter posterior does not prove that the first beat of the bar is correct.

Rhythm evidence and abstention
--------------------------------

``RhythmPresenceProcessor`` returns an uncalibrated rhythm score per activation frame. It searches periodic evidence over a bounded history. Silence, constant activations and weak evidence reduce the score. A caller can combine this score with pulse validity to abstain when evidence is weak.

Periodic speech, noise or other non-music can still pass this test. The processor does not replace a music classifier. Its history also introduces acquisition and release delays. Set an application threshold using development music and negative recordings, and measure both coverage loss and false synchronization.

Validation
------------

The new tests check causal prefixes, chunk equivalence, reset behavior, bounded state, probability boundaries, synthetic competing phases, meter acquisition, missed beats and explicit expiry. Synthetic tests establish the implemented behavior. They do not establish accuracy on music. Compare each option with the unchanged forward beat tracker on the same development recordings, using the actual availability time of every published grid. Report paired differences by source recording and the added processing cost.

The period design independently applies ideas discussed by Paulllux and f0k in `Beat This! issue 13 <https://github.com/CPJKU/beat_this/issues/13>`_. The source is a discussion, with no source commit. No attached code or model was adopted. The bar decoder reuses the state and transition models already supplied by madmom; their research references and upstream credit remain in :mod:`madmom.features.beats_hmm`.
