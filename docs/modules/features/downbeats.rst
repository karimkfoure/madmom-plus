madmom.features.downbeats
=========================

.. automodule:: madmom.features.downbeats
    :members:

Causal bar tracking
-------------------

``OnlineDBNDownBeatTrackingProcessor`` applies the forward algorithm to a joint bar-pointer hidden Markov model. It accepts one frame or a chunk of mutually exclusive beat and downbeat probabilities. Each returned row contains the current frame time in seconds and a beat number that starts at one. The filter compares allowed meters and can change meter at a bar boundary.

Use ``reset=False`` after the first chunk to keep the filter state and clock. Chunk size does not change the result. The processor emits each event at the current frame and does not move an event to a future peak. Initial bar labels can be wrong while the filter acquires evidence. ``meter_probability`` measures posterior mass within the selected meter; it is not a calibrated confidence score.

The activation source determines whether the complete pipeline is causal. The bundled ``RNNDownBeatProcessor`` uses bidirectional networks, which read future frames. Passing its output to the online decoder does not make the audio pipeline causal. This release does not include a pretrained causal downbeat network. A caller must supply causal probabilities with the same column meanings, or validate a separate causal feature source.

``CausalDownBeatActivationProcessor`` provides one training-free source of experimental accent evidence. Its input columns contain causal beat probability and non-negative bass-onset salience, such as positive spectral flux below a chosen crossover frequency. The helper compares bass salience with a running mean weighted by beat probability. Its output splits the input beat probability between mutually exclusive beat and downbeat columns. It uses only current and past frames and is invariant to the salience scale. Use ``reset=False`` after the first chunk.

This helper treats strong bass accents as possible bar starts. Bass accents can be syncopated, and equal kicks provide no meter evidence. These activations are not calibrated downbeat probabilities. Evaluate them on the intended music before using their bar labels.

The activation threshold suppresses weak event reports while the filter and clock continue. It does not identify music, speech or noise. A separate presence decision is necessary when a caller must abstain on non-music.
