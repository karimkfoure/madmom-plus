Compatibility verification
==========================

madmom-plus 0.17.0.dev1 preserves the ``madmom`` import name, public processors, command-line program names, and upstream model references. The distribution name changes to ``madmom-plus`` so maintenance releases can be identified separately from upstream madmom. Both distributions install the same Python package; use one per environment.

Verified environments
---------------------

The checks below ran on 2026-10-28. Native tests ran on x86-64 Linux with FFmpeg 6.1.1, pytest 9.2.0, and Cython 3.3.0 for extension builds. Each full run includes all 811 upstream tests and 10 additional regression cases. No upstream unit-test assertion or tolerance was changed.

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
     - 821 passed
     - 50.33 seconds
   * - 3.12.13
     - 2.5.4
     - 1.18.1
     - 821 passed
     - 58.98 seconds
   * - 3.13.14
     - 2.5.4
     - 1.18.1
     - 821 passed
     - 51.86 seconds
   * - 3.11.15
     - 1.26.4
     - 1.13.1
     - 821 passed
     - 56.27 seconds

The Python 3.12 source archive and wheel built through isolated PEP 517 in 34.03 seconds. That command built the wheel from the source archive, which checks that the archive includes the Cython declaration files. Python 3.11 and 3.13 then installed from the source archive through separate isolated builds. NumPy 2 headers were used for compilation, including the build tested with NumPy 1.26 at runtime.

Tests ran outside the source tree with a copy of the upstream test data. Their expected ``bin`` directory contained links to the installed command-line programs. Package paths were checked before each run to confirm that tests imported the installed wheel. BLAS and OpenMP each used one thread. Timings are wall-clock measurements of this shared host, not performance guarantees.

All 46 module doctests also passed on Python 3.12 in 12.32 seconds. The doctest fixture uses NumPy's legacy 1.25 display format on NumPy 2 to keep array and scalar examples readable across versions. This changes display only. Three examples now show their actual ``float32`` dtype; one FFT-based example has its last displayed decimal corrected. Unit-test reference arrays and tolerances remain unchanged.

The full runs emitted existing API deprecation and model migration warnings, warnings for deliberately invalid numeric inputs, and Python's warning about forking a process with multiple threads. Those warnings were retained. Python 3.9, 3.10, and versions newer than 3.13 were not tested in this check; the declared dependency floors do not imply that each possible version combination was tested.

The `aarch64 guide <aarch64.rst>`_ records a separate isolated ARM build and 64 passing tests under QEMU. Physical ARM hardware and the full suite on ARM remain unverified.

Model packaging
---------------

The model submodule remains pinned to `CPJKU/madmom_models commit 7e3dc1b0cad499792767074d03c38b194b9b0a79 <https://github.com/CPJKU/madmom_models/commit/7e3dc1b0cad499792767074d03c38b194b9b0a79>`_. Both the source archive and wheel contain all 92 model pickle files. The ARM wheel's model files were also compared byte for byte with the submodule. The model licence and catalogue remain included.

The old packaging rules omitted 20 model files: eight 2013 beat models, eight 2019 TCN beat models, and four 2017 key models. The broader year patterns fix this omission without changing the models or their public references. Builds now report the submodule initialization command if model sources are absent.

Integer resampling compatibility
--------------------------------

The first full run produced 809 passes and two failures in 52.22 seconds. Both failures came from FFmpeg's changed resampling default: the integer signal differed from the old fixture by one sample unit at 3,511 of 61,741 positions, and the CLP chroma result exceeded its existing tolerance.

`Paul B Mahol's FFmpeg change 3bef108fa9302844180aad8632d54e1b7be1f4a3 <https://github.com/FFmpeg/FFmpeg/commit/3bef108fa9302844180aad8632d54e1b7be1f4a3>`_, committed on 2023-09-25, switched sample-rate conversion to floating-point intermediates to improve precision. madmom-plus selects the historical integer intermediate format only when resampling a known signed 16-bit ``Signal`` to signed 16-bit output. This reproduces the complete fixture exactly and preserves the existing chroma result. File decoding and higher-precision input or output keep FFmpeg's defaults. The tradeoff is historical integer rounding for that specific compatibility path.

The option is documented in the `FFmpeg resampler reference <https://ffmpeg.org/ffmpeg-resampler.html>`_. This fix is original maintenance work; no FFmpeg source code was copied. This release does not adopt an external fork patch.

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
