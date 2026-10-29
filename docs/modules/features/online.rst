madmom.features.online
======================

``BeatPeriodProcessor`` adds a causal pulse estimate without changing beat detections. It uses a bounded history, robust period fitting and circular phase consensus. A missed detection can span several grid intervals. The optional tempo range expresses an octave preference; it cannot establish the correct metrical level.

Pass only new detections and set ``now`` to their causal delivery clock. Set ``reset=False`` to retain history. Empty updates with an explicit ``now`` let the estimate expire when detections stop. ``PulseState.phase`` is a grid anchor in seconds; extrapolate other grid times with integer multiples of ``PulseState.period``. Use ``PulseState.valid`` before publishing a pulse.

``RhythmPresenceProcessor`` supplies a separate, optional signal from causal beat activations. Its confidence measures periodic evidence within a configured tempo range. It rejects silence and constant activations. Periodic speech or noise can still receive a high score, so this signal must not be presented as a music classifier or calibrated probability. The score requires two slow-tempo periods to acquire evidence. Its update interval adds a bounded response delay, and its history adds a release tail.

The period design is an independent implementation. Paulllux proposed octave preference and phase-aware grid fitting in `Beat This! issue 13 <https://github.com/CPJKU/beat_this/issues/13>`_; f0k recommended the median of tempo estimates in the discussion. The source is an issue discussion, so no source commit applies. No code or model was copied from the attached scripts. The robust slope fit, causal state, expiry and rhythmic evidence processor were implemented for this module.

.. automodule:: madmom.features.online
    :members:
