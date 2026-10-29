Installation
============

Install the ``madmom-plus`` distribution and continue to use ``import madmom``. The fork retains madmom's Python API, command-line program names, and model references. Use a separate virtual environment because upstream ``madmom`` installs the same package and executable names.

.. _install_prerequisites:

Prerequisites
-------------

Source installation requires Python 3.9 or newer, a C compiler, and matching Python development headers. Runtime dependencies are NumPy 1.23.5 or newer, SciPy 1.13 or newer, and mido 1.2.6 or newer. PEP 517 build isolation installs setuptools 64 or newer, Cython 3 or newer, and NumPy 2 or newer for compilation.

Install FFmpeg for compressed audio input, resampling, and the complete audio test suite. Optional packages are ``pyaudio`` for microphone input and ``pyfftw`` for FFT acceleration. See :doc:`compatibility` for tested versions and :doc:`aarch64` for 64-bit ARM Linux instructions and verification limits.

On Debian or Ubuntu with the system Python, install ``build-essential``, ``python3-dev``, and ``python3-venv``. A separately installed interpreter needs its own matching development headers.

.. _install_from_package:

Install a built package
-----------------------

Until a madmom-plus release is published on PyPI, use :ref:`the source instructions <install_from_source>`. An upstream ``pip install madmom`` release does not include this fork's maintenance changes.

To install a wheel built from this checkout, activate the target virtual environment and pass the wheel filename to pip::

    python -m pip install dist/madmom_plus-*.whl

Use the actual wheel filename and a compatible Python version and platform. Source builds include the pinned upstream models; model licences remain separate from the source licence.

.. _install_from_source:

Install from source
-------------------

Clone the repository with its pinned model submodule, then install into a virtual environment::

    git clone --recurse-submodules https://github.com/karimkfoure/madmom-plus.git
    cd madmom-plus
    python3 -m venv .venv
    . .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install .
    python -c "import madmom; print(madmom.__version__)"

For an existing checkout, initialize its pinned model revision before installation::

    git submodule update --init --recursive

For development and tests, use an editable installation with the test extra::

    python -m pip install -e '.[test]'
    python -m pytest tests
    python -m pytest --import-mode=importlib --doctest-modules madmom

To create a source archive and build its wheel in isolated environments::

    python -m pip install build
    python -m build

.. _upgrading:

Upgrade a source installation
-----------------------------

Pull the updated source, retain its pinned model revision, and reinstall::

    git pull
    git submodule update --init --recursive
    python -m pip install .

Use ``python -m pip install -e '.[test]'`` for an editable development installation. Reinstall after changing Cython source or declaration files so the compiled extensions match the source.
