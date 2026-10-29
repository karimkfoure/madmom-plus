madmom.features.tempo
=====================

.. automodule:: madmom.features.tempo
    :members:

Online histogram state
----------------------

The ACF, comb-filter and DBN histogram processors accept one scalar activation or a one-dimensional activation block in online mode. To continue a stream across calls, pass ``reset=False``. An empty update with ``reset=False`` returns the current histogram and does not advance any filter or DBN state. An empty update with ``reset=True`` clears the state and returns a zero histogram. Calling ``reset()`` on an offline histogram processor is safe.

The ``hist_buffer`` parameter limits the aggregation of histogram contributions, not the input context of the underlying algorithm. ACF adds a product between the current activation and an earlier activation at each candidate lag. It retains products whose later endpoint falls inside the aggregation window, even when the earlier endpoint lies before that window. The comb filter retains its recursive response before it aggregates the recent responses. The DBN retains its forward state and aggregates only the recent winning interval votes. A block longer than the aggregation buffer produces the same final histogram and subsequent state as frame-by-frame processing.

The online tempo DBN evaluates its forward probabilities in blocks of at most 64 frames. This bounds temporary posterior storage for long input blocks while retaining the same forward state and histogram. The aggregation buffer size still comes from ``hist_buffer`` and ``fps``.

For nonnegative activations, ACF and comb-filter online histograms agree with their offline histograms before the aggregation window fills, subject to numerical precision. After the window fills, the correct comparison is the full-prefix offline histogram minus the offline histogram of the prefix before the aggregation window. For the comb filter, both offline calculations must start from the same empty filter state; the offline Cython kernel also stores ``alpha`` at single precision. Recomputing either algorithm only on the last raw input samples resets its context and changes the calculation. The DBN offline Viterbi path uses future evidence, so it is not a reference for the causal forward path.

Regression provenance
---------------------

The fixes and regression tests below were implemented independently from the issue reports. No external patch was copied.

.. list-table:: Upstream issue provenance
    :header-rows: 1
    :widths: 45 20 35

    * - Source
      - Author
      - Commit
    * - `madmom issue #401 <https://github.com/CPJKU/madmom/issues/401>`_
      - SebastianPoell
      - ``null``; issue report, no patch adopted
    * - `madmom issue #378 <https://github.com/CPJKU/madmom/issues/378>`_
      - SebastianPoell
      - ``null``; issue report, no patch adopted

Issue #401 identifies a difference between contribution windows and calculations that restart on a raw input window. The regression tests preserve the existing contribution-window behavior and verify lag alignment, arbitrary input blocks, scalar input, empty updates and reset behavior. Issue #378 reports a buffer error for blocks longer than the aggregation window. The existing ``BufferProcessor`` already truncates such blocks correctly; regression coverage now verifies that the tempo DBN preserves the same histogram and future state as frame-by-frame input. These changes do not establish a tempo-accuracy improvement on music.
