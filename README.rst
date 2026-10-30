madmom-plus
===========

madmom-plus maintains `madmom <https://github.com/CPJKU/madmom>`_, the music-analysis library by Sebastian Böck and the madmom contributors. It detects beats, downbeats, onsets, tempo, chords, notes, and keys. The fork supports current Python and NumPy while keeping ``import madmom``, existing processors, command-line programs, and model files.

Install from source
-------------------

Use Python 3.11–3.13, FFmpeg, and a C compiler. The model submodule is required::

    git clone --recurse-submodules https://github.com/karimkfoure/madmom-plus.git
    cd madmom-plus
    python -m pip install .
    DBNBeatTracker single audio.wav

The distribution name is ``madmom-plus``. Install only one of ``madmom`` and ``madmom-plus`` in an environment because both provide the same Python package. See `installation <docs/installation.rst>`_ for development setup.

What madmom-plus changes
------------------------

.. list-table::
   :header-rows: 1

   * - Area
     - Change
     - Measured effect
   * - Toolchains
     - NumPy 2 headers, Cython 3, setuptools
     - See `compatibility checks <docs/compatibility.rst>`_
   * - Models
     - Include every model year in wheels
     - 92 model files; 20 previously omitted files restored
   * - Compatibility
     - Scoped legacy unpickling, audio gain, evaluation, worker cleanup
     - Regression tests; model bytes and default parameters retained

Documentation and credit
------------------------

`API and tutorials <https://madmom.readthedocs.io/>`_ · `compatibility and provenance <docs/compatibility.rst>`_ · `release notes <CHANGES.rst>`_ · `citation <docs/citation.rst>`_

The original library comes from the Department of Computational Perception at Johannes Kepler University, Linz, and the Austrian Research Institute for Artificial Intelligence, Vienna. Research references remain in the module documentation and `introduction <docs/introduction.rst>`_. Funding acknowledgements remain in the `documentation index <docs/index.rst>`_.

Source code uses the `BSD licence <LICENSE>`_. Models and data use `CC BY-NC-SA 4.0 <https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode>`_ unless stated otherwise. For commercial use of models, derivatives, or technology that uses them, contact `Gerhard Widmer <http://www.cp.jku.at/people/widmer/>`_. See the model catalogue and licence in ``madmom/models``.
