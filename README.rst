===========
madmom-plus
===========

madmom-plus is a maintained fork of `CPJKU/madmom <https://github.com/CPJKU/madmom>`_, an audio signal processing library for music information retrieval (MIR). It keeps the upstream algorithms, public Python API, command-line programs, trained models, credits, and licences.

The distribution name is ``madmom-plus``. The import name remains ``madmom`` so existing applications and model pickle references keep working. Install one distribution per environment because ``madmom`` and ``madmom-plus`` provide the same Python package and executable names.

This development release, ``0.17.0.dev1``, updates isolated source builds for current Python, NumPy 2, Cython 3, and SciPy. It also fixes scalar online beat input, microphone byte decoding, integer resampling compatibility with current FFmpeg, and missing beat models in wheels. See `CHANGES.rst <CHANGES.rst>`_ for details and `docs/compatibility.rst <docs/compatibility.rst>`_ for tested versions and timings.

The original upstream description and credits follow.

The library is internally used by the Department of Computational Perception,
Johannes Kepler University, Linz, Austria (http://www.cp.jku.at) and the
Austrian Research Institute for Artificial Intelligence (OFAI), Vienna, Austria
(http://www.ofai.at).

Possible acronyms are:

- Madmom Analyzes Digitized Music Of Musicians
- Mostly Audio / Dominantly Music Oriented Modules

It includes reference implementations for some music information retrieval
algorithms, please see the `References`_ section.


Documentation
=============

Documentation of the package can be found online http://madmom.readthedocs.org


License
=======

The package has two licenses, one for source code and one for model/data files.

Source code
-----------

Unless indicated otherwise, all source code files are published under the BSD
license. For details, please see the `LICENSE <LICENSE>`_ file.

Model and data files
--------------------

Unless indicated otherwise, all model and data files are distributed under the
`Creative Commons Attribution-NonCommercial-ShareAlike 4.0
<http://creativecommons.org/licenses/by-nc-sa/4.0/legalcode>`_ license.

If you want to include any of these files (or a variation or modification
thereof) or technology which utilises them in a commercial product, please
contact `Gerhard Widmer <http://www.cp.jku.at/people/widmer/>`_.


Installation
============

Source installation uses PEP 517 build isolation. It installs Cython and NumPy build requirements automatically, then compiles the extension modules. A source checkout must include the pinned model submodule.

Requirements
------------

The package requires Python 3.9 or newer, NumPy 1.23.5 or newer at runtime, SciPy 1.13 or newer, and mido 1.2.6 or newer. The build environment requires setuptools 64 or newer, Cython 3 or newer, and NumPy 2 or newer. A C compiler and matching Python development headers are required for source builds. On Debian or Ubuntu with the system Python, install ``build-essential``, ``python3-dev``, and ``python3-venv``.

Install ``ffmpeg`` for compressed audio input, resampling, and the full audio test suite. Optional packages are ``pyaudio`` for microphone input and ``pyfftw`` for FFT acceleration. The `aarch64 guide <docs/aarch64.rst>`_ gives 64-bit ARM Linux instructions and states what was verified under emulation.

Install from source
-------------------

Create a separate virtual environment so this distribution does not conflict with upstream madmom::

    git clone --recurse-submodules https://github.com/karimkfoure/madmom-plus.git
    cd madmom-plus
    python3 -m venv .venv
    . .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install .
    python -c "import madmom; print(madmom.__version__)"

For an existing checkout, initialize the pinned model revision before installation::

    git submodule update --init --recursive

Use the source instructions until a ``madmom-plus`` release is published on PyPI. An upstream ``madmom`` release has different distribution metadata and does not include these maintenance changes.

Development and tests
---------------------

Install the package in editable mode with the test extra, then run all upstream tests::

    python -m pip install -e '.[test]'
    python -m pytest tests

To include documentation examples::

    python -m pytest --import-mode=importlib --doctest-modules madmom tests

Reinstall after changes to Cython extensions. To build a source archive and then build its wheel in an isolated environment::

    python -m pip install build
    python -m build

Upgrade a source checkout with ``git pull``, then run ``git submodule update --init --recursive`` and reinstall. Keep the pinned submodule revision unless you intend to validate new model data.

Package structure
-----------------

The package has a very simple structure, divided into the following folders:

`/bin <bin>`_
  this folder includes example programs (i.e. executable algorithms)
`/docs <docs>`_
  package documentation
`/madmom <madmom>`_
  the actual Python package
`/madmom/audio <madmom/audio>`_
  low level features (e.g. audio file handling, STFT)
`/madmom/evaluation <madmom/evaluation>`_
  evaluation code
`/madmom/features <madmom/features>`_
  higher level features (e.g. onsets, beats)
`/madmom/ml <madmom/ml>`_
  machine learning stuff (e.g. RNNs, HMMs)
`/madmom/models <../../../madmom_models>`_
  pre-trained model/data files (see the License section)
`/madmom/utils <madmom/utils>`_
  misc stuff (e.g. MIDI and general file handling)
`/tests <tests>`_
  tests

Executable programs
-------------------

The package includes executable programs in the `/bin <bin>`_ folder.
If you installed the package, they were copied to a common place.

All scripts can be run in different modes: in ``single`` file mode to process
a single audio file and write the output to STDOUT or the given output file::

    DBNBeatTracker single [-o OUTFILE] INFILE

If multiple audio files should be processed, the scripts can also be run in
``batch`` mode to write the outputs to files with the given suffix::

    DBNBeatTracker batch [-o OUTPUT_DIR] [-s OUTPUT_SUFFIX] FILES

If no output directory is given, the program writes the output files to the
same location as the audio files.

Some programs can also be run in ``online`` mode, i.e. operate on live audio
signals. This requires `pyaudio <http://people.csail.mit.edu/hubert/pyaudio/>`_
to be installed::

    DBNBeatTracker online [-o OUTFILE] [INFILE]

The ``pickle`` mode can be used to store the used parameters to be able to
exactly reproduce experiments.

Please note that the program itself as well as the modes have help messages::

    DBNBeatTracker -h

    DBNBeatTracker single -h

    DBNBeatTracker batch -h

    DBNBeatTracker online -h

    DBNBeatTracker pickle -h

will give different help messages.


Additional resources
====================

Mailing list
------------

The `mailing list <https://groups.google.com/d/forum/madmom-users>`_ should be
used to get in touch with the developers and other users.

Wiki
----

The wiki can be found here: https://github.com/CPJKU/madmom/wiki

FAQ
---

Frequently asked questions can be found here:
https://github.com/CPJKU/madmom/wiki/FAQ

Citation
========

If you use madmom in your work, please consider citing it:

.. code-block:: latex

   @inproceedings{madmom,
      Title = {{madmom: a new Python Audio and Music Signal Processing Library}},
      Author = {B{\"o}ck, Sebastian and Korzeniowski, Filip and Schl{\"u}ter, Jan and Krebs, Florian and Widmer, Gerhard},
      Booktitle = {Proceedings of the 24th ACM International Conference on
      Multimedia},
      Month = {10},
      Year = {2016},
      Pages = {1174--1178},
      Address = {Amsterdam, The Netherlands},
      Doi = {10.1145/2964284.2973795}
   }

References
==========

.. [1] Florian Eyben, Sebastian Böck, Björn Schuller and Alex Graves,
    *Universal Onset Detection with bidirectional Long Short-Term Memory
    Neural Networks*,
    Proceedings of the 11th International Society for Music Information
    Retrieval Conference (ISMIR), 2010.
.. [2] Sebastian Böck and Markus Schedl,
    *Enhanced Beat Tracking with Context-Aware Neural Networks*,
    Proceedings of the 14th International Conference on Digital Audio Effects
    (DAFx), 2011.
.. [3] Sebastian Böck and Markus Schedl,
    *Polyphonic Piano Note Transcription with Recurrent Neural Networks*,
    Proceedings of the 37th International Conference on Acoustics, Speech and
    Signal Processing (ICASSP), 2012.
.. [4] Sebastian Böck, Andreas Arzt, Florian Krebs and Markus Schedl,
    *Online Real-time Onset Detection with Recurrent Neural Networks*,
    Proceedings of the 15th International Conference on Digital Audio Effects
    (DAFx), 2012.
.. [5] Sebastian Böck, Florian Krebs and Markus Schedl,
    *Evaluating the Online Capabilities of Onset Detection Methods*,
    Proceedings of the 13th International Society for Music Information
    Retrieval Conference (ISMIR), 2012.
.. [6] Sebastian Böck and Gerhard Widmer,
    *Maximum Filter Vibrato Suppression for Onset Detection*,
    Proceedings of the 16th International Conference on Digital Audio Effects
    (DAFx), 2013.
.. [7] Sebastian Böck and Gerhard Widmer,
    *Local Group Delay based Vibrato and Tremolo Suppression for Onset
    Detection*,
    Proceedings of the 13th International Society for Music Information
    Retrieval Conference (ISMIR), 2013.
.. [8] Florian Krebs, Sebastian Böck and Gerhard Widmer,
    *Rhythmic Pattern Modelling for Beat and Downbeat Tracking in Musical
    Audio*,
    Proceedings of the 14th International Society for Music Information
    Retrieval Conference (ISMIR), 2013.
.. [9] Sebastian Böck, Jan Schlüter and Gerhard Widmer,
    *Enhanced Peak Picking for Onset Detection with Recurrent Neural Networks*,
    Proceedings of the 6th International Workshop on Machine Learning and
    Music (MML), 2013.
.. [10] Sebastian Böck, Florian Krebs and Gerhard Widmer,
    *A Multi-Model Approach to Beat Tracking Considering Heterogeneous Music
    Styles*,
    Proceedings of the 15th International Society for Music Information
    Retrieval Conference (ISMIR), 2014.
.. [11] Filip Korzeniowski, Sebastian Böck and Gerhard Widmer,
    *Probabilistic Extraction of Beat Positions from a Beat Activation
    Function*,
    Proceedings of the 15th International Society for Music Information
    Retrieval Conference (ISMIR), 2014.
.. [12] Sebastian Böck, Florian Krebs and Gerhard Widmer,
    *Accurate Tempo Estimation based on Recurrent Neural Networks and
    Resonating Comb Filters*,
    Proceedings of the 16th International Society for Music Information
    Retrieval Conference (ISMIR), 2015.
.. [13] Florian Krebs, Sebastian Böck and Gerhard Widmer,
    *An Efficient State Space Model for Joint Tempo and Meter Tracking*,
    Proceedings of the 16th International Society for Music Information
    Retrieval Conference (ISMIR), 2015.
.. [14] Sebastian Böck, Florian Krebs and Gerhard Widmer,
    *Joint Beat and Downbeat Tracking with Recurrent Neural Networks*,
    Proceedings of the 17th International Society for Music Information
    Retrieval Conference (ISMIR), 2016.
.. [15] Filip Korzeniowski and Gerhard Widmer,
    *Feature Learning for Chord Recognition: The Deep Chroma Extractor*,
    Proceedings of the 17th International Society for Music Information
    Retrieval Conference (ISMIR), 2016.
.. [16] Florian Krebs, Sebastian Böck, Matthias Dorfer and Gerhard Widmer,
    *Downbeat Tracking Using Beat-Synchronous Features and Recurrent Networks*,
    Proceedings of the 17th International Society for Music Information
    Retrieval Conference (ISMIR), 2016.
.. [17] Filip Korzeniowski and Gerhard Widmer,
    *A Fully Convolutional Deep Auditory Model for Musical Chord Recognition*,
    Proceedings of IEEE International Workshop on Machine Learning for Signal
    Processing (MLSP), 2016.
.. [18] Filip Korzeniowski and Gerhard Widmer,
    *Genre-Agnostic Key Classification with Convolutional Neural Networks*,
    Proceedings of the 19th International Society for Music Information
    Retrieval Conference (ISMIR), 2018.
.. [19] Rainer Kelz, Sebastian Böck and Gerhard Widmer,
    *Deep Polyphonic ADSR Piano Note Transcription*,
    Proceedings of the 44th International Conference on Acoustics, Speech and
    Signal Processing (ICASSP), 2019.

Acknowledgements
================

Supported by the European Commission through the `GiantSteps project
<http://www.giantsteps-project.eu>`_ (FP7 grant agreement no. 610591) and the
`Phenicx project <http://phenicx.upf.edu>`_ (FP7 grant agreement no. 601166)
as well as the `Austrian Science Fund (FWF) <https://www.fwf.ac.at>`_ project
Z159.
