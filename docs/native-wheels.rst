Linux native-core wheels
========================

Build against old system libraries to support older Linux systems. Changing a wheel tag does not remove newer glibc symbol dependencies. The build and verification tools use digest-pinned OCI root filesystems through PRoot, with QEMU for AArch64 execution on x86-64.

.. list-table:: Historical preserved-source verification
   :header-rows: 1

   * - Python / architecture
     - Build glibc
     - Runtime checks
     - Result
   * - 3.11 / x86-64
     - 2.28
     - AlmaLinux 8 and Debian 12
     - 377 tests and 189 subtests in each
   * - 3.12 / x86-64
     - 2.28
     - AlmaLinux 8; Debian 12
     - 377 tests and 189 subtests; full suite 1147 tests and 217 subtests
   * - 3.12 / AArch64
     - 2.28
     - AlmaLinux 8 and Debian 12 under QEMU
     - 377 tests and 189 subtests in each
   * - Same AArch64 wheel, NumPy 1.26
     - 2.28
     - AlmaLinux 8 under QEMU
     - 35 tests and 10 subtests

These results refer to runtime commit ``20e6bc1`` and package metadata at ``f30b211``, not newly built current artifacts. `Compact records <benchmarks/wheel-summary.json>`_ retain artifact hashes, dependencies, symbol limits, and test results. Each wheel contains five extensions and all 92 unchanged model files. The required-symbol audit found no glibc requirement above 2.27; execution was tested on glibc 2.28 and 2.36. The recorded shared-kernel source hashes match this current source. Rebuild and verify each release from its reviewed source.

The new host build passes the full suite and standalone C API tests on glibc 2.39. Its binaries require glibc 2.38 and GLIBCXX 3.4.29; they are development artifacts and do not satisfy the old-glibc wheel target. The historical old-glibc artifacts remain separate from this host check.

Build
-----

Prepare a source snapshot with initialized models and a digest-pinned platform image. The root filesystem needs its native GCC/G++ compiler and glibc development files. ``PYTHON_PREFIX`` supplies the matching interpreter and headers; ``BUILD_DEPS`` supplies its build dependencies::

    python tools/container_wheels.py --log fetch.log --cpus 0-1 fetch \
      --image "$PINNED_PLATFORM_IMAGE" --destination build/container

    python tools/container_wheels.py --log build.log --cpus 0-1 build \
      --rootfs build/container/rootfs --source "$BUILD_SOURCE" \
      --python-prefix "$PYTHON_PREFIX" --dependencies "$BUILD_DEPS" \
      --python python3.12 --output dist/wheels --workers 2

For AArch64 on x86-64, add ``--qemu /usr/bin/qemu-aarch64-static``. The historical builds used GCC 8.5.0, glibc 2.28, PRoot 5.5.0, and QEMU 8.2.2. PRoot translates filesystem operations with the host kernel; it does not provide kernel namespace isolation. No host system library directories are bound into the guest.

Audit and repair the wheel with ``auditwheel show`` and ``auditwheel repair``. Use an ARM-aware ``strip`` for AArch64. Verify extension architecture, required symbols, every model file, and absence of private build paths before distribution. The alternative ``tools/build_core_wheel.py`` cross-builds with target Python headers; the target glibc still depends on the supplied compiler and sysroot.

Verify
------

Use a separate work directory so tests import the wheel instead of the source tree::

    python tools/verify_container_wheel.py \
      --rootfs build/container/rootfs --expected-id almalinux \
      --expected-glibc 2.28 --python-prefix "$PYTHON_PREFIX" \
      --dependencies "$RUNTIME_DEPS" --wheel "$WHEEL" \
      --workdir build/wheel-check --output wheel-check.json \
      --ffmpeg "$STATIC_FFMPEG" --ffprobe "$STATIC_FFPROBE"

Add ``--test-file tests`` for the full suite. The verifier checks installed-package imports, actual glibc, ABI version ``0x00010000``, sample timestamps, queue reads, reset, and configuration. It deletes copied test audio after use. Static FFmpeg and FFprobe are test dependencies; they are not wheel contents.

QEMU checks compilation and numerical behavior. It does not measure physical ARM performance, thermal behavior, or callback deadlines. macOS and Windows wheels remain unbuilt and unverified. No automated publishing workflow is enabled.
