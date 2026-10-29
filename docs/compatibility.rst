Compatibility verification
==========================

madmom-plus 0.17.0.dev1 preserves the ``madmom`` import name, public processors, command-line program names, and upstream model references. The distribution name changes to ``madmom-plus`` so maintenance releases can be identified separately from upstream madmom. Both distributions install the same Python package; use one per environment.

Verified environments
---------------------

The checks below ran on 2026-10-28, before integration of the required native core. They preserve the earlier toolchain and ecosystem evidence. Current core parity, performance, and local-wheel checks are in `native.rst <native.rst>`_ and `native-wheels.rst <native-wheels.rst>`_. These earlier checks used x86-64 Linux with FFmpeg 6.1.1, pytest 9.2.0, and Cython 3.3.0. Each full run included all 811 upstream tests, added compatibility and lifecycle tests, and availability tests for the then-optional backend. No upstream numerical assertion or tolerance was weakened.

.. list-table:: Full installed-package test runs
   :header-rows: 1
   :widths: 15 15 15 15 20

   * - Python
     - NumPy
     - SciPy
     - Result
     - pytest time
   * - 3.11.15
     - 2.4.6
     - 1.17.1
     - 957 passed, 9 skipped
     - 79.99 seconds
   * - 3.12.13
     - 2.5.4
     - 1.18.1
     - 957 passed, 9 skipped
     - 55.76 seconds
   * - 3.13.14
     - 2.5.4
     - 1.18.1
     - 957 passed, 9 skipped
     - 59.98 seconds
   * - 3.11.15
     - 1.26.4
     - 1.13.1
     - 957 passed, 9 skipped
     - 59.19 seconds

The Python 3.12 source archive and wheel built through isolated PEP 517 in 35.57 seconds. That command built the wheel from the source archive, which checks that the archive includes the Cython declaration files. Python 3.11 and 3.13 then installed from the source archive through separate isolated builds. NumPy 2 headers were used for compilation, including the build tested with NumPy 1.26 at runtime.

Tests ran outside the source tree with a copy of the upstream test data. Their expected ``bin`` directory contained links to the installed command-line programs. Package paths were checked before each run to confirm that tests imported the installed wheel. BLAS and OpenMP each used one thread. Timings are wall-clock measurements of this shared host, not performance guarantees.

All 46 installed-module doctests also passed on Python 3.12 in 13.00 seconds. The doctest fixture uses NumPy's legacy 1.25 display format on NumPy 2 to keep array and scalar examples readable across versions. This changes display only. Three examples now show their actual ``float32`` dtype; one FFT-based example has its last displayed decimal corrected. Existing numeric unit-test references and tolerances remain unchanged; the adopted continuity regression tests check corrected behavior with asymmetric tolerances.

In that earlier revision, nine cases required the optional extension and were skipped in default builds. A separate isolated build with ``MADMOM_BUILD_NATIVE=1`` completed in 32.90 seconds. Its installed wheel passed all 11 native tests and 15 subtests in 4.31 seconds, including activation/beat parity, chunk boundaries, reset, invalid input, and resource cleanup. The Python inference path also passed its smoke check. Current builds always include the core and run those tests.

The full runs emitted existing API deprecation and model migration warnings, warnings for deliberately invalid numeric inputs, and Python's warning about forking a process with multiple threads. Those warnings were retained. Python 3.9, 3.10, and versions newer than 3.13 were not tested in this check; the declared dependency floors do not imply that each possible version combination was tested.

The `aarch64 guide <aarch64.rst>`_ records a separate isolated ARM build and 238 passing tests under QEMU. Physical ARM hardware and the full suite on ARM remain unverified.

Model packaging
---------------

The model submodule remains pinned to `CPJKU/madmom_models commit 7e3dc1b0cad499792767074d03c38b194b9b0a79 <https://github.com/CPJKU/madmom_models/commit/7e3dc1b0cad499792767074d03c38b194b9b0a79>`_. Both the source archive and wheel contain all 92 model pickle files. Wheel model files were compared byte for byte with the submodule. All 92 pickle files load in the tested NumPy 1.x and 2.x environments. This includes the two 2018 note models, whose old internal ``dstack`` reference is resolved to NumPy's public function only during unpickling. The model licence and catalogue remain included.

The old packaging rules omitted 20 model files: eight 2013 beat models, eight 2019 TCN beat models, and four 2017 key models. The broader year patterns fix this omission without changing the models or their public references. Builds now report the submodule initialization command if model sources are absent.

Integer resampling compatibility
--------------------------------

The first full run produced 809 passes and two failures in 52.22 seconds. Both failures came from FFmpeg's changed resampling default: the integer signal differed from the old fixture by one sample unit at 3,511 of 61,741 positions, and the CLP chroma result exceeded its existing tolerance.

`Paul B Mahol's FFmpeg change 3bef108fa9302844180aad8632d54e1b7be1f4a3 <https://github.com/FFmpeg/FFmpeg/commit/3bef108fa9302844180aad8632d54e1b7be1f4a3>`_, committed on 2023-09-25, switched sample-rate conversion to floating-point intermediates to improve precision. madmom-plus selects the historical integer intermediate format only when resampling a known signed 16-bit ``Signal`` to signed 16-bit output. This reproduces the complete fixture exactly and preserves the existing chroma result. File decoding and higher-precision input or output keep FFmpeg's defaults. The tradeoff is historical integer rounding for that specific compatibility path.

The option is documented in the `FFmpeg resampler reference <https://ffmpeg.org/ffmpeg-resampler.html>`_. This fix is original maintenance work; no FFmpeg source code was copied.

Ecosystem priorities
--------------------

The `ecosystem catalogue <ecosystem.md>`_ assigns four modernization priorities. Their integration status is recorded below. Source patches retain madmom's inherited BSD source terms; the model terms remain separate. The changes do not package the test suite or ecosystem inventory as runtime assets.

.. list-table:: Adopted or independently matched changes
   :header-rows: 1

   * - Priority
     - Source and provenance
     - Scope and checks
   * - 1
     - PR #548, chadawagner, ``868e004eb3e36f4359d11458e5849f9d7e52b644``
     - The existing independent ``libc.math`` changes match this patch. All supported builds and HMM/CRF tests pass.
   * - 2
     - PR #532, Carl Thomé, ``8bba9971075316210f1c8efd9ad005e5a6486793``
     - The independent packaging fix includes 2019 models plus older omitted files. Installed-wheel tests and the 92-file model comparison validate the complete model set.
   * - 2
     - PR #559, gzivdo, ``e6a2b0e6b7d70071636087af2cb49ad1edd64c8b``
     - Adapt the scoped legacy dtype unpickler. Test positional and keyword integer alignment, protocols 0/2/current, Latin-1 strings, current processor round trips, error propagation, and unchanged global NumPy state. Its setuptools import commit ``9addc37c3aa3e3601c2f2791f702ae86060e0f85`` matches the earlier independent build fix.
   * - 3
     - PR #562, Emil Zimmermann, ``a7ed9984b60495c448146c3a55d837eeb4dbea9d``
     - Adopt the patch and both asymmetric-tolerance tests. All 58 beat evaluation tests pass; equal default tolerances keep their scores.
   * - 4
     - PR #505, Igal Cohen-Hadria, ``0e0a47fb51a00de7066b24b6e559aeb5f26c987a``
     - Adapt per-call owned pools, repair the undefined worker count, and add caller-owned pool injection. Six lifecycle tests cover repeated construction/calls, normal and exceptional cleanup, borrowed pool reuse, serialization, legacy state, and empty input.
   * - 4
     - Issue #471, snownstone; issue #508, fdlm; no source commits
     - Independently fix the omitted online keyword and floating-point stereo downmix. Direct calls, explicit device listing, file/stream/channel parity, ReplayGain, resampling, and unchanged surround policies have regression coverage. No issue code was copied.

Each PR and issue URL, author, and source commit is also recorded in ``CHANGES.rst`` and the corresponding integration commit. Pool ownership follows the scoped design from #505 rather than retaining an owned process pool for the lifetime of a processor. This releases workers reliably but adds pool-startup cost to each parallel decoding call; callers that need reuse can supply their own pool. The existing one-worker default has no pool-startup cost.

Reproduce the checks
--------------------

Initialize the models and create an environment with matching Python headers, a C compiler, and FFmpeg. Then run::

    git submodule update --init --recursive
    python -m pip install --upgrade pip build
    python -m build
    python -m pip install -e '.[test]'
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest tests
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest --import-mode=importlib --doctest-modules madmom

For packaging validation, install the built wheel into a separate environment, copy ``tests`` outside the checkout, and link the installed programs into a sibling ``bin`` directory before running the test command. Check ``madmom.__file__`` to confirm that imports resolve to that environment.
