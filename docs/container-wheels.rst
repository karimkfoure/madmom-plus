Wheels for older Linux systems
==============================

The current Linux wheels are built with GCC 8.5.0 and glibc 2.28 in AlmaLinux 8 root filesystems. ``auditwheel`` finds no required glibc symbol newer than 2.27. The x86-64 wheels have both ``manylinux_2_27`` and ``manylinux_2_28`` tags; the aarch64 wheel has the ``manylinux_2_27`` tag. These artifacts replace the earlier host builds that required glibc 2.38.

The build uses actual old system libraries. It does not change the tag of a wheel built against newer libraries. Execution is checked in both the build baseline, AlmaLinux 8 with glibc 2.28, and Debian 12 Bookworm with glibc 2.36. The current artifact hashes, source hashes, container image digests, test results, and tool provenance are in the `container wheel report <benchmarks/container-wheels.json>`_.

Container method
----------------

This host has no Docker or Podman installation. Its unprivileged user namespace policy rejects UID-map setup in both ``unshare`` and ``bubblewrap``. PRoot 5.5.0 runs digest-pinned OCI root filesystems through filesystem translation with ``ptrace``. The kernel is shared with the host; PRoot does not provide kernel namespace isolation. This method permits the library compatibility checks without a change to host security settings.

``skopeo 1.13.3`` downloads public AlmaLinux 8 minimal and Debian Bookworm slim images with an empty authentication file. ``umoci 0.4.7`` extracts each image in rootless mode. Compressed image-layer caches are deleted after extraction. Every image is pinned by its platform manifest SHA-256 digest, so a moving image tag cannot select a different base during a repeat build.

Each AlmaLinux image installs its own native GCC/G++ 8.5.0 compiler and glibc development files. The ARM compiler executes through QEMU 8.2.2; it produces ARM code against the ARM image's glibc 2.28. Host system libraries are not bound into the guest. Only the source, output directory, Python installation, Python dependencies, and normal device/process filesystems are bound. The bound CPython installations require glibc 2.17; NumPy and SciPy dependencies use compatible manylinux wheels.

PRoot 5.1 from the host distribution produced incorrect ARM file-metadata lookups under QEMU and could not run the RPM package manager correctly. The upstream PRoot 5.5.0 release fixes those operations in this environment. Its binary is checked against the upstream SHA-256 file; its source commit is ``25dc6a3134891f98a79f57ce1c2c1b23ff15cad1``. This tool version is part of the recorded build environment.

Build and verification tools
----------------------------

Initialize the model submodule and prepare a source snapshot before building. The snapshot must contain the same runtime files and tests as the commit under review. Use the scripts below with an existing PRoot runner and OCI tools::

    python tools/container_wheels.py --log fetch.log --cpus 0-1 fetch \
      --image "$PINNED_PLATFORM_IMAGE" --destination build/container

    python tools/container_wheels.py --log build.log --cpus 0-1 build \
      --rootfs build/container/rootfs --source "$BUILD_SOURCE" \
      --python-prefix "$BUILD_PYTHON_PREFIX" --dependencies "$BUILD_DEPS" \
      --python python3.12 --output dist/container-wheels --workers 2

``PINNED_PLATFORM_IMAGE`` is a fully qualified OCI registry name followed by ``@sha256:`` and the platform manifest digest. ``BUILD_PYTHON_PREFIX`` contains the matching Python executable and headers. ``BUILD_DEPS`` is its build environment's ``site-packages`` directory. The container must have its native compiler installed. For aarch64 on an x86-64 host, add ``--qemu /usr/bin/qemu-aarch64-static`` to the build command. Tool executable paths can be supplied when they are installed outside ``PATH``.

The build uses ``-O3 -DNDEBUG -g0`` and C++17 with floating-point contraction disabled. A source-local dependency link keeps Cython's traceback filenames relative. ``auditwheel`` checks symbol requirements and strips the repaired wheels; ARM stripping uses an ARM-aware binary tool. Every extension's ELF machine type and every unpacked wheel entry are checked. The final wheels contain five extensions for their target architecture, all 92 unchanged model files, and no private build paths.

Run the installed-wheel checks in a separate work directory::

    python tools/verify_container_wheel.py \
      --rootfs build/container/rootfs --expected-id almalinux \
      --expected-glibc 2.28 --python-prefix "$BUILD_PYTHON_PREFIX" \
      --dependencies "$RUNTIME_DEPS" --wheel "$WHEEL" \
      --workdir build/wheel-check --output wheel-check.json \
      --ffmpeg "$STATIC_FFMPEG" --ffprobe "$STATIC_FFPROBE"

The default test scope includes neural networks, HMMs, beat/onset/downbeat processors, all native tests, stored madmom 0.16.1 fixtures, windowed native downbeat tracking, and model loading. Add ``--test-file tests`` to run the complete suite. The verifier checks both the loaded C library's version and ``getconf``, requires imports from the extracted wheel, and tests the versioned C engine's sample timestamps, queue reads, reset, configuration, and destruction. It removes copied test audio after the run.

Static FFmpeg and FFprobe 7.0.2, supplied by John Van Sickle, provide the audio test tools in the minimal images. They are test dependencies and are not bundled in the wheels. Their archive and executable hashes, upstream release information, and source URLs are recorded in the report.

Verified results
----------------

The final artifacts are in ``dist/container-wheels/repaired``; the complete source archive is in ``dist/container-wheels``. The runtime source is pinned to ``20e6bc1``. Final package descriptions match the README and change log at ``f30b211``. Hashes of every Python module, extension, and model match the tested package payloads after that metadata update. Fresh checks of the final CPython 3.12 artifacts also pass the C engine smoke test in both distributions.

.. list-table:: Container verification
   :header-rows: 1

   * - Python / architecture
     - Distribution / glibc
     - NumPy / SciPy
     - Scope and result
   * - 3.12.13 / x86-64
     - Bookworm / 2.36
     - 2.4.6 / 1.18.1
     - Complete suite: 1147 tests, 217 subtests passed
   * - 3.12.13 / x86-64
     - AlmaLinux 8 / 2.28
     - 2.4.6 / 1.18.1
     - Focused suite: 377 tests, 189 subtests passed
   * - 3.12.13 / aarch64
     - AlmaLinux 8 / 2.28 and Bookworm / 2.36
     - 2.4.6 / 1.18.1
     - Focused suite in each root: 377 tests, 189 subtests passed
   * - 3.12.13 / aarch64
     - AlmaLinux 8 / 2.28
     - 1.26.4 / 1.13.1
     - Native neural, frontend, buffer, and release-fixture subset: 35 tests, 10 subtests passed
   * - 3.11.15 / x86-64
     - AlmaLinux 8 / 2.28 and Bookworm / 2.36
     - 2.4.6 / 1.17.1
     - Focused suite in each root: 377 tests, 189 subtests passed

The source archive contains all 92 model files, the native neural binding source and header, the two container helpers, and ``SCOPE.md``. Checked source files match the repository, and model bytes match the pinned model submodule. Original source and model licences remain included.

Limits
------

These checks establish compilation and numerical behavior for the listed Python, NumPy, SciPy, architecture, and glibc combinations. Execution checks cover glibc 2.28 and 2.36; the glibc 2.27 tag comes from the required-symbol audit. QEMU results do not measure physical ARM speed, memory pressure, temperature, or callback deadlines. Repeat the host performance protocol on physical aarch64 hardware before making ARM performance claims. macOS and Windows wheels remain unbuilt and unverified.
