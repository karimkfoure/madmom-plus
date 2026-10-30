Optional online downbeat processors
===================================

The optional downbeat processors expose bar position and meter estimates while a stream runs. Existing processors, model files and command-line defaults keep their behavior. Use :doc:`windowed-downbeats` for audio input with the bundled models.

.. list-table:: Choose an interface
   :header-rows: 1
   :widths: 35 30 35

   * - Processor
     - Input
     - Output
   * - ``WindowedDownBeatTrackingProcessor``
     - Mono audio at 44100 Hz
     - Causal beat times with delayed bar labels
   * - ``BarPhaseVotingProcessor``
     - Beat times and completed downbeat activation windows
     - Beat times with bar labels; zero means unknown
   * - ``OnlineDBNDownBeatTrackingProcessor``
     - Causal beat/downbeat probabilities
     - Current-frame beat times and bar positions

Forward-only bar decoding
-------------------------

.. code-block:: python

    from madmom.features.downbeats import OnlineDBNDownBeatTrackingProcessor

    tracker = OnlineDBNDownBeatTrackingProcessor([3, 4], fps=100)
    # Each row holds mutually exclusive beat and downbeat probabilities.
    events = tracker(activations)
    # Supply reset=False for every later activation block.

The online decoder uses forward filtering over bar position, tempo and allowed meters. It retains the filter state and bounded internal blocks. Returned events use the current frame time and a beat number starting at one. The filter never moves an event to a future peak or revises an earlier event. ``meter_probability`` is posterior mass within the selected meter; it is not a calibrated correctness probability.

Causal decoding requires causal activations. The bundled ``RNNDownBeatProcessor`` is bidirectional and reads future frames. This release does not supply a pretrained causal downbeat network. The windowed audio processor instead waits for complete windows and applies their evidence to later beats.

Optional meter priors
---------------------

Both downbeat decoders accept ``beats_per_bar_weights`` in the order of ``beats_per_bar``. For example, ``beats_per_bar=[3, 4], beats_per_bar_weights=[1, 3]`` expresses a preference for four beats per bar. Weights must be finite and non-negative, with at least one positive value. Zero excludes a meter. Omitting the option or supplying equal weights preserves default decoding.

.. list-table:: Prior behavior
   :header-rows: 1
   :widths: 30 70

   * - Decoder
     - How it uses the prior
   * - Offline
     - Add the normalized log prior once when selecting a meter from the complete recording.
   * - Online
     - Assign initial meter probability at creation and reset. Later evidence can overcome positive priors; zero-weight meters are absent from the state space.

Configure online weights when creating the processor. Editing its public weight array does not rebuild the state space. Supply a preference only when the caller has evidence about meter.

Validation and limits
---------------------

.. list-table:: Regression coverage
   :header-rows: 1
   :widths: 35 65

   * - Defaults and compatibility
     - Equal weights preserve exact output; positional arguments, saved processors, reset and serialization retain their behavior.
   * - Prior handling
     - Tests check invalid vectors, zero exclusions, ambiguous meter, initial meter mass and evidence that overcomes a positive prior.
   * - Streaming
     - Synthetic tests check meter acquisition, single-frame/chunk equality, causal prefixes, empty input and bounded forward storage.

The installed Python 3.12 wheel passes 1,111 tests and 201 subtests, including all upstream tests, plus 47 module doctests. No upstream assertion or numerical tolerance was weakened.

Synthetic tests establish these behaviors, not accuracy on music. Meter priors express preferences, not proof of the annotated meter. Raw meter probability and bar confidence do not identify music, speech or noise. Applications must select and validate any presence policy separately.

Provenance
----------

The optional offline prior adapts `madmom PR #408 <https://github.com/CPJKU/madmom/pull/408>`_ by declension, commit `fd670e62a0438487f1965a69a5ed6576d291b191 <https://github.com/CPJKU/madmom/commit/fd670e62a0438487f1965a69a5ed6576d291b191>`_. The adaptation preserves positional arguments and handles invalid vectors and unequal path lengths. Online initial priors are new. The online bar decoder reuses madmom's existing bar state and transition models; their research references and upstream credit remain in ``madmom.features.beats_hmm``. The madmom source licence remains in effect.
