Local native-core wheels
========================

The Linux wheels from this host require glibc 2.38 or newer. ``auditwheel`` selects that minimum because the native core refers to ``fmod@GLIBC_2.38`` in the system math library. An older Linux system needs a wheel built against an older system library set, or a source build with its local libraries.

The wheels include the required C++ core and the four existing Cython extensions. The import name remains ``madmom``. All 92 model files must match the pinned source files byte for byte, and their separate model licence remains included.

Build method
------------

The x86-64 build uses CPython 3.11.15. The aarch64 build uses an x86-64 CPython 3.12.13 build environment, aarch64 CPython 3.12.13 headers, and ``aarch64-linux-gnu-gcc`` / ``aarch64-linux-gnu-g++`` 13.3.0. Both build environments use NumPy 2.4.6, Cython 3.3.0, setuptools 84.2.1, and wheel 0.48.0. Runtime checks use NumPy 2.4.6 and SciPy 1.17.1. A separate ARM environment checks NumPy 1.26.4 with SciPy 1.13.1, because the public dependency range includes NumPy 1.x.

The host wheel is built from a source archive through PEP 517. The build extracts that archive to a separate source directory and uses a local ``.build-deps`` link to the build environment's ``site-packages`` directory. This keeps Cython's source filenames relative to the source directory. ``-O3 -DNDEBUG -g0`` keeps optimization enabled and prevents compiler assertion and debug records from storing local header paths. The final checks scan every unpacked wheel entry for private build paths.

To build the ARM wheel, run the helper with a host Python interpreter that has the same major and minor version as the target Python installation::

    python tools/build_core_wheel.py \
      --target-python-prefix "$TARGET_PYTHON_PREFIX" \
      --wheel-dir dist/core-wheels --workers 2

``TARGET_PYTHON_PREFIX`` is the target aarch64 Python installation, including ``include/python3.12/Python.h`` and its matching ``_sysconfigdata`` file. Install the normal build requirements in the host environment first. The helper uses the target headers and extension suffix with the cross compiler, so the output contains ARM machine code. Its build directory is separate from the host build directory.

``auditwheel 6.8.2`` checks shared-library versions and repairs each wheel with its supported manylinux tag. The repair strips symbols. ARM repair uses ``aarch64-linux-gnu-strip`` and searches the ARM system libraries. The final checks read every extension's ELF header to confirm its target architecture.

Verification method
-------------------

Each wheel is installed into a separate virtual environment. Tests run from a copied test directory outside the source package. Imports are checked to confirm that they use the installed wheel. The installed native library must report C application binary interface (ABI) version ``0x00010000``. The checks also compare model bytes and Python source bytes with the build sources.

The x86-64 check runs the complete test suite. The ARM check runs the upstream comb-filter, neural-network, HMM, beat, onset, model-loading, and online-default tests, plus all native-core tests and the stored madmom 0.16.1 parity fixtures. BLAS and OpenMP each use one thread. Build and test processes run with the allowed CPU affinity and at most two build workers.

The ARM interpreter runs under QEMU 8.2.2 with an aarch64 glibc 2.39 system library set and the Ubuntu ARM ``zlib1g`` package ``1:1.3.dfsg-3.1ubuntu2``. These tools are used for validation and are not bundled in the wheel. QEMU checks compilation and execution. Its timings do not measure speed on physical ARM hardware.

Verified results
----------------

The repaired artifacts are in ``dist/core-wheels/repaired``. Artifact filenames, SHA-256 hashes, source hashes, architecture checks, and test results are recorded in the `wheel verification report <benchmarks/native-wheels.json>`_. Both wheels report native C ABI version ``0x00010000``. Each contains five extensions for its target architecture and all 92 unchanged model files. Every Python source file matches the final build source, and the scan of unpacked wheel entries finds no private build paths.

Package file hashes after the final README metadata update match the tested package hashes. Fresh engine checks in all three runtime environments accept 442 samples, return frame sample indices 0 and 441, and verify queue reads, reset, configuration, and destruction. This checks the final artifacts against the versioned C interface as well as the Python API.

.. list-table:: Installed-wheel tests
   :header-rows: 1

   * - Wheel target
     - Runtime NumPy / SciPy
     - Test scope
     - Result
   * - CPython 3.11, manylinux_2_38_x86_64
     - 2.4.6 / 1.17.1
     - Complete suite
     - 1100 passed, 165 subtests
   * - CPython 3.12, manylinux_2_38_aarch64
     - 2.4.6 / 1.17.1
     - ARM scope listed above
     - 285 passed, 40 subtests
   * - Same aarch64 wheel
     - 1.26.4 / 1.13.1
     - Same ARM scope
     - 285 passed, 40 subtests

Platform limits
---------------

Only the listed Linux Python versions are covered by this build. Older manylinux targets need a build with older system libraries. This host has no Docker or Podman installation for that check. Physical aarch64 performance and thermal behavior still need tests on the target hardware.

macOS arm64, macOS x86-64, and Windows wheels are not built or verified. This Linux host has no macOS SDK, Windows build toolchain, or operating-system runtime for those checks. The source build selects MSVC C++17 options on Windows, but that setting alone does not establish Windows compatibility. No GitHub Actions configuration is enabled or changed.
