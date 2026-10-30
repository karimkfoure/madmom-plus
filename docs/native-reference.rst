madmom 0.16.1 reference outputs
===============================

Stored release outputs make parity tests independent of the current implementation. The tests cannot hide an error shared by two paths in this fork.

.. list-table:: Reference provenance
   :header-rows: 1

   * - Asset
     - Author and source
     - Commit
   * - Numerical algorithms
     - Sebastian Böck and madmom contributors; `madmom v0.16.1 <https://github.com/CPJKU/madmom/tree/6545fbb3ecdae0aac336a95ec591f642e0b74f3e>`_
     - ``6545fbb3ecdae0aac336a95ec591f642e0b74f3e``
   * - Eight online networks
     - Sebastian Böck and contributors; `madmom_models <https://github.com/CPJKU/madmom_models/tree/31fe6ce52affb794c2e3511908fbeccfd68f6925>`_
     - ``31fe6ce52affb794c2e3511908fbeccfd68f6925``
   * - Audio
     - Public ``sample.wav`` and ``sample2.wav`` in the release test data
     - Same release commit

``tests/data/native_0161/metadata.json`` records source, model, audio, and fixture hashes. The online network bytes match model commit ``7e3dc1b0cad499792767074d03c38b194b9b0a79``. The models retain `CC BY-NC-SA 4.0 <https://github.com/CPJKU/madmom_models/blob/31fe6ce52affb794c2e3511908fbeccfd68f6925/LICENSE>`_. Existing test-data and source licences still apply.

.. list-table:: Coverage and tolerances
   :header-rows: 1

   * - Output
     - Coverage
     - Absolute / relative tolerance
   * - Frontend
     - 687 frames; centered, causal, and streaming origins
     - ``2e-6 / 2e-5``
   * - Neural activations
     - Eight individual networks and ensemble
     - ``2e-5 / 2e-4``
   * - Forward HMM
     - State and maximum every frame; full posterior every 37 frames and last frame
     - ``1e-9 / 2e-6``
   * - Python beat and peak times
     - Identical input activations, reset, and chunk boundaries
     - Exact
   * - Complete native audio beat times
     - Both recordings
     - One 10 ms frame

The reference uses Python 3.11.15, NumPy 1.26.4, SciPy 1.13.1, Cython 0.29.37, setuptools 75.8.2, and mido 1.3.3. Runtime aliases restore removed collection and NumPy names; temporary distribution metadata supplies the version query. The numerical source is unchanged. The four original extensions use Cython language level 2. These fixtures preserve the release with this environment, not an unspecified historical platform.

Reproduce with the reference dependencies in a separate environment and local source/model Git objects::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python tools/generate_core_reference.py

The generator removes temporary source, compiled extensions, and audio. Use ``--model-repository`` to select another checkout of the same public models. Test the result in the normal build environment::

    python -m pytest tests/test_native_reference.py

These short recordings do not establish accuracy on a representative music corpus.
