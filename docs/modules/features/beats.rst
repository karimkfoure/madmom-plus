madmom.features.beats
=====================

.. automodule:: madmom.features.beats
    :members:

Causal spectral evidence
------------------------

``SpectralBeatActivationProcessor`` adds an optional low-frequency onset prior before beat decoding. It accepts rows of ``[beat_probability, low_onset, high_onset]`` and returns adjusted beat probabilities. Each onset band has a separate running level. The correction increases beat odds for low-frequency onsets and decreases them for high-frequency onsets. This can help when high-frequency subdivisions are stronger than the beat evidence.

All three features must use the same timestamps and captured audio only. For example, calculate positive spectral magnitude differences in 30–200 Hz and 2–10 kHz from frames that end at the current timestamp. Whole-recording normalization would introduce future information. The processor does not calculate these audio features itself.

Pass the adjusted probabilities to ``DBNBeatTrackingProcessor(fps=100, online=True)``. Use ``reset=False`` on both processors after the first chunk. Set ``strength=0`` to return the input beat probabilities exactly. Existing beat processors and command-line defaults retain their behavior.

This is a rhythmic prior, not an instrument classifier. Bass notes can occur between beats, and some music has no clear bass pulse. Select a nonzero strength on development data before use. A synthetic phase test verifies the mechanism, but it does not establish an accuracy gain on music.
