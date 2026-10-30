Compatibility and provenance
============================

The distribution name changes to ``madmom-plus``. Python imports, processor names, command-line programs, model references, and model bytes stay compatible with upstream madmom. NumPy compatibility work changes reconstruction only while loading a processor; it does not patch global NumPy state.

Validation
----------

The Python 3.12.13 installed wheel passed 968 tests and 28 subtests in 82.33 seconds, plus 46 module doctests in 21.65 seconds. The environment used NumPy 2.4.6, SciPy 1.18.1, Cython 3.3.0, FFmpeg 6.1.1, and one BLAS/OpenMP thread on Linux x86-64. The upstream numerical assertions and tolerances remain unchanged. NumPy 2 doctests use the legacy array display format so display changes do not hide numerical failures.

.. list-table:: Packaging checks
   :header-rows: 1

   * - Check
     - Expected result
   * - Wheel and source archive
     - All 92 model pickle files and their licence
   * - Model bytes
     - Identical to the pinned model submodule
   * - Legacy and current processors
     - Protocol 0/2/current loading; Latin-1 strings; unchanged global dtype constructor
   * - Worker ownership
     - Owned pools close on success and error; caller-owned pools remain open

Owned downbeat worker pools now exist only during each parallel call. This prevents leaked workers but adds startup cost per call. The default remains one worker. Callers can pass their own pool for reuse.

Signed 16-bit ``Signal`` resampling selects FFmpeg's historical integer intermediate format. This retains the upstream audio and chroma references after FFmpeg changed its default. Other precisions and file-decoding paths retain FFmpeg defaults. Stereo floating-point downmix uses arithmetic averaging in both loaders; surround mixing retains its existing policy.

External sources
----------------

Each row records the source URL, author, and exact commit. All source code keeps the inherited BSD terms; models keep their separate licence.

.. list-table:: Adopted or independently matched changes
   :header-rows: 1
   :widths: 16 20 34 30

   * - Source
     - Author
     - Commit
     - Use
   * - `PR #548 <https://github.com/CPJKU/madmom/pull/548>`_
     - Chad Wagner (chadawagner)
     - ``868e004eb3e36f4359d11458e5849f9d7e52b644``
     - Independent matching libc.math fix
   * - `PR #532 <https://github.com/CPJKU/madmom/pull/532>`_
     - Carl Thomé
     - ``8bba9971075316210f1c8efd9ad005e5a6486793``
     - Independent model packaging fix, extended to all years
   * - `PR #559 <https://github.com/CPJKU/madmom/pull/559>`_
     - gzivdo
     - ``e6a2b0e6b7d70071636087af2cb49ad1edd64c8b``; ``9addc37c3aa3e3601c2f2791f702ae86060e0f85``
     - Adapt scoped dtype reconstruction; independent matching setuptools import
   * - `PR #562 <https://github.com/CPJKU/madmom/pull/562>`_
     - Emil Zimmermann
     - ``a7ed9984b60495c448146c3a55d837eeb4dbea9d``
     - Adopt continuity tolerance fix and both regression tests
   * - `PR #505 <https://github.com/CPJKU/madmom/pull/505>`_
     - Igal Cohen-Hadria
     - ``0e0a47fb51a00de7066b24b6e559aeb5f26c987a``
     - Adapt scoped pools; fix worker count; add caller-owned pool support
   * - `FFmpeg change <https://github.com/FFmpeg/FFmpeg/commit/3bef108fa9302844180aad8632d54e1b7be1f4a3>`_
     - Paul B Mahol
     - ``3bef108fa9302844180aad8632d54e1b7be1f4a3``
     - Explain changed resampling; no code copied
   * - `Issue #471 <https://github.com/CPJKU/madmom/issues/471>`_; `issue #508 <https://github.com/CPJKU/madmom/issues/508>`_
     - snownstone; fdlm
     - No source commits
     - Independent fixes for optional online device-list flag and stereo gain

Models remain pinned to `CPJKU/madmom_models <https://github.com/CPJKU/madmom_models/tree/7e3dc1b0cad499792767074d03c38b194b9b0a79>`_, commit ``7e3dc1b0cad499792767074d03c38b194b9b0a79``. Model authors and research references remain in its catalogue.
