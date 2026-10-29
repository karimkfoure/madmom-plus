Linux aarch64 installation
==========================

madmom-plus can build its Cython extensions on 64-bit ARM Linux. The distribution name is ``madmom-plus``; Python code continues to use ``import madmom``. Use a 64-bit Python interpreter and operating system: ``uname -m`` must report ``aarch64`` for the path below.

Install from source
-------------------

Install a C compiler and the development headers for the selected Python version. On Debian or Ubuntu, the corresponding system packages are ``build-essential`` and ``python3-dev``. Install ``ffmpeg`` for compressed audio input and the complete upstream audio tests.

Clone the repository with its pinned model submodule, then install into a virtual environment::

    git clone --recurse-submodules https://github.com/karimkfoure/madmom-plus.git
    cd madmom-plus
    python3 -m venv .venv
    . .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install .
    python -c 'import madmom; print(madmom.__version__)'

PEP 517 build isolation installs the build requirements from ``pyproject.toml``, including NumPy 2 and Cython 3. The build uses the local compiler and the machine's native instruction set. If the Python interpreter was installed separately from the operating system, use headers that match that interpreter.

The model files remain in the upstream Git submodule. For an existing clone, run ``git submodule update --init --recursive`` before installation. The source and model licences remain distinct; see the repository README and ``madmom/models/LICENSE``.

Verification method
-------------------

The aarch64 check uses a target Python interpreter under QEMU user-mode emulation on an x86-64 Linux host. An aarch64 cross compiler compiles the extension modules. The wheel is then installed into the target environment, and tests execute its ARM extension modules under QEMU. This checks both compilation and execution; it does not measure performance on physical ARM hardware.

The target interpreter comes from `Astral python-build-standalone <https://github.com/astral-sh/python-build-standalone>`_, installed with `uv <https://docs.astral.sh/uv/concepts/python-versions/>`_. NumPy and SciPy use their public aarch64 wheels. The emulator uses an aarch64 glibc sysroot and an extracted Ubuntu aarch64 ``zlib1g`` runtime package. These dependencies are test tools and are not bundled with madmom-plus.

Verified build
--------------

The following environment built and installed ``madmom-plus 0.17.0.dev1``:

.. list-table:: Target environment
   :header-rows: 1
   :widths: 35 65

   * - Component
     - Version
   * - Architecture
     - Linux aarch64, little endian, 64 bit
   * - Python
     - CPython 3.12.13
   * - NumPy / SciPy
     - 2.5.4 / 1.18.1
   * - Cython
     - 3.3.0
   * - Compiler
     - aarch64-linux-gnu-gcc 13.3.0
   * - Emulator / glibc sysroot
     - QEMU 8.2.2 / glibc 2.39
   * - Build frontend
     - pip 26.2.2, PEP 517 with build isolation

The final wheel build completed in 86 seconds and produced ``madmom_plus-0.17.0.dev1-cp312-cp312-linux_aarch64.whl``. All four extension modules are AArch64 ELF shared objects and import in the target interpreter. The installed ``DBNBeatTracker --help`` command also succeeds.

All 92 model pickle files in the pinned upstream submodule are present in the wheel and match the source files byte for byte. The wheel also contains the model licence, README and Python model catalogue.

Installed-wheel tests
---------------------

The following upstream and regression tests ran outside the source tree, so imports used the installed wheel::

    python -m pytest -q tests/test_audio_comb_filters.py tests/test_ml_hmm.py tests/test_ml_crf.py tests/test_ml_nn.py tests/test_features_beats.py tests/test_pickle_compat.py tests/test_evaluation_beats.py tests/test_online_defaults.py tests/test_pool_lifecycle.py --durations=5

All 238 tests passed in 71.37 seconds. The run emitted 60 warnings from intentionally invalid evaluation and HMM inputs, logarithms of zero, legacy model migration notices, and existing API deprecations. The tests retained their upstream assertions. They cover comb filters, hidden Markov models, conditional random fields, recurrent neural network layers, online and offline beat processing, temporal convolutional network (TCN) inference, all 92 packaged model pickles, asymmetric evaluation tolerances, direct online calls, and worker-pool cleanup. BLAS and OpenMP each used one thread.

This check verifies one emulated aarch64 environment with Python 3.12. Physical ARM hardware, ARM Python 3.11/3.13, and the complete test suite on ARM remain unverified. Emulation timings measure this test setup and do not predict runtime on an ARM board.
