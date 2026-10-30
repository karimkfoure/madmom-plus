Windowed downbeat tracking
==========================

``WindowedDownBeatTrackingProcessor`` assigns bar positions to causal beat detections. It runs the bundled bidirectional downbeat network on completed audio windows, then applies that evidence to later beats. It never revises a returned event. Existing processors and command-line defaults do not change.

.. code-block:: text

    mono audio ──► causal beats ──────────► time + bar position
          └──────► completed windows ────► bar-phase votes

Stream audio
------------

.. code-block:: python

    import numpy as np
    from madmom.audio.signal import Signal
    from madmom.features.downbeats import WindowedDownBeatTrackingProcessor

    audio = Signal('song.wav', sample_rate=44100,
                   num_channels=1, dtype=np.float32)
    with WindowedDownBeatTrackingProcessor(window=6., hop=2.) as tracker:
        for first in range(0, len(audio), 441):
            rows = tracker(audio[first:first + 441], reset=False)
            downbeats = rows[rows[:, 1] == 1, 0]

Each row contains event time in seconds and beat number within the bar. Beat number zero means that bar position is unknown. Use ``reset=False`` to continue the stream; ``reset=True`` clears its history. Input must be mono floating point PCM at 44100 Hz. Integer PCM is rejected because an unscaled conversion changes model input amplitude. The first frame refers to sample zero; later frames advance by 441 samples. There is no tail padding or flush operation.

.. list-table:: Main controls
   :header-rows: 1
   :widths: 25 25 50

   * - Option
     - Default
     - Effect
   * - ``window`` / ``hop``
     - 6 s / 2 s
     - Longer windows supply more bar context; shorter hops repeat more model computation. Both must be positive multiples of 10 ms, with hop no larger than window.
   * - ``beat_backend``
     - ``'python'``
     - Select existing Python processor objects or the direct ``'native'`` engine. Both use the same required C++ core.
   * - ``background``
     - ``False``
     - Synchronous inference gives repeatable labels across audio chunk sizes. Background mode uses one worker and skips windows while it is busy.
   * - ``fill_gaps``
     - ``False``
     - Preserve detected beat events. Optional filling inserts only bounded gaps confirmed by a later detection, so inserted events arrive late.
   * - ``beats_per_bar``
     - ``(3, 4)``
     - Set the candidate meters. Missing candidates and ambiguous accents can prevent acquisition.

The first model call needs a complete window. Bar acquisition can take longer because the voter needs examples at every possible bar position. Synchronous model calls pause the caller. In background mode, completion time can change the first beat that receives new evidence. Each instance supports one calling thread; use a context manager or ``close()`` to release owned resources. Supplied beat processors remain caller-owned.

Use a separate beat detector
----------------------------

``BarPhaseVotingProcessor`` accepts detected beat times and downbeat activation windows. This allows callers to schedule inference separately. Beats must be new and strictly increasing; activation window end times must increase. Both use the same time base. Supply evidence only after its complete audio window is available.

A call emits its beat rows before it applies evidence. Submit evidence with an empty beat array when it must affect the next beat call. The voter retains recent phase and meter evidence, requires stronger competing evidence before a change, and clears stale evidence. Smaller meters win near ties between repeated identical bars.

Limits and checks
-----------------

.. list-table:: Regression coverage
   :header-rows: 1
   :widths: 35 65

   * - Check
     - Test behavior
   * - Received audio only
     - Prefix tests reject dependence on later audio; completed-window scheduling uses exact input samples.
   * - Stable event output
     - Chunk sizes, empty calls and reset preserve the beat stream; acquired labels match across synchronous chunk sizes.
   * - Bar adaptation
     - Synthetic cases check phase changes, meter changes, repeated integer gaps and stale evidence.
   * - Worker ownership
     - Tests check bounded background work, reset isolation, error propagation and caller-owned resources.

These checks establish implementation behavior. They do not establish downbeat accuracy on annotated music. A stable wrong beat grid can produce stable wrong bar labels. Overlapping windows contain correlated evidence. ``confidence`` is a raw phase margin, not a calibrated probability or a music-presence decision. Models retain their existing licence in ``madmom/models/LICENSE``.


Measured opt-in cost
--------------------

.. list-table:: AMD Ryzen 9 5900X, complete streaming path
   :header-rows: 1

   * - Configuration
     - CPU seconds / audio second
     - Peak process MiB

   * - Beat tracking only
     - 0.029186
     - 78.29

   * - Beat tracking plus synchronous window voting
     - 0.087136
     - 131.53

Three alternating fresh-process trials use Python 3.12.13, NumPy 2.4.6, SciPy 1.18.1, one pinned core, and one BLAS/OpenMP thread. The public test clips repeat three times (20.552 seconds); chunks contain 441 samples. Windows are 6 seconds with a 2-second hop. CPU excludes construction, loading, reset, and warmup. Memory is the largest process peak across trials and includes setup. All 51 beat times match exactly between configurations. This checks event preservation and processing cost, not bar-label accuracy or callback deadlines. Host activity was uncontrolled.

The `compact trial data <benchmarks/windowed-summary.json>`_ retain versions and event hashes. Reproduce after installing the package::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python tools/benchmark_windowed_downbeats.py --output windowed-cost.json


Provenance
----------

The independent Python design was informed by `BeatTrackerJS <https://github.com/mathigatti/BeatTrackerJS>`_ by Mathias Gatti at commit ``604869f580fa695aafd9685b94c6c3247ed0ed87``: `README <https://github.com/mathigatti/BeatTrackerJS/blob/604869f580fa695aafd9685b94c6c3247ed0ed87/README.md>`_, `migration notes <https://github.com/mathigatti/BeatTrackerJS/blob/604869f580fa695aafd9685b94c6c3247ed0ed87/docs/MIGRATION.md>`_ and `experiment notes <https://github.com/mathigatti/BeatTrackerJS/blob/604869f580fa695aafd9685b94c6c3247ed0ed87/docs/EXPERIMENTS.md>`_. No JavaScript source or model files are copied. Comparisons against offline madmom do not establish accuracy against human annotations.

.. autoclass:: madmom.features.downbeats.BarPhaseVotingProcessor
    :members:

.. autoclass:: madmom.features.downbeats.WindowedDownBeatTrackingProcessor
    :members:
