Installation
============

.. _install_prerequisites:

Requirements
------------

.. list-table::
   :header-rows: 1

   * - Component
     - Requirement
   * - Python
     - 3.9 or newer; validation targets 3.11–3.13
   * - Runtime
     - NumPy >= 1.23.5, SciPy >= 1.13, mido >= 1.2.6
   * - Source build
     - C compiler and Python headers; isolated builds install NumPy 2 and Cython 3
   * - Audio decoding
     - FFmpeg for compressed audio and resampling
   * - Optional
     - PyAudio for microphone input; pyFFTW for supported FFT paths

.. _install_from_source:

Build from source
-----------------

Clone with models, then install into a virtual environment::

    git clone --recurse-submodules https://github.com/karimkfoure/madmom-plus.git
    cd madmom-plus
    python -m venv .venv
    . .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install '.[test]'
    DBNBeatTracker single tests/data/audio/sample.wav

For an existing clone, run ``git submodule update --init --recursive``. Keep the pinned model revision: ``--remote`` selects different model files.

.. _install_from_package:

Install a built wheel
---------------------

Build the source archive and wheel, then install the wheel in a separate environment::

    python -m pip install build
    python -m build
    python -m pip install dist/madmom_plus-*.whl

Source archives contain the models and Cython declarations. Wheels contain the models and executable programs. The Python import remains ``madmom``. Use only one distribution per environment because upstream ``madmom`` uses the same import and program names.

Development checks
------------------

Install an editable build and run the tests with one numerical-library thread::

    python -m pip install -e '.[test]'
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest tests
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest --import-mode=importlib --doctest-modules madmom

For documentation, install ``docs/requirements.txt`` and run ``sphinx-build -b html docs docs/_build/html``.

.. _upgrading:

Update a source installation
----------------------------

After updating the source, run ``git submodule update --init --recursive`` and repeat the editable install. This rebuilds changed extensions. Existing processors and command-line program names remain available.
