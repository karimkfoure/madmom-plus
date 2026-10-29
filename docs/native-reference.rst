Frozen madmom 0.16.1 reference outputs
====================================

The files in ``tests/data/native_0161`` preserve outputs from the original madmom 0.16.1 implementation. They make native-core parity tests independent of the fork's current implementation. A comparison between two paths in the fork alone can miss a shared error.

Provenance
----------

The numerical source is `CPJKU/madmom, tag v0.16.1 <https://github.com/CPJKU/madmom/tree/6545fbb3ecdae0aac336a95ec591f642e0b74f3e>`_, commit ``6545fbb3ecdae0aac336a95ec591f642e0b74f3e``, by Sebastian Böck and the madmom contributors. The generator archives this Git object into a temporary directory and imports that archive in a separate process. It does not import the fork or modify the original numerical source. It compiles the four original extensions with Cython's Python 2 language level, as required by this release.

The models are `CPJKU/madmom_models <https://github.com/CPJKU/madmom_models/tree/31fe6ce52affb794c2e3511908fbeccfd68f6925>`_, commit ``31fe6ce52affb794c2e3511908fbeccfd68f6925``, by Sebastian Böck and contributors. This is the model submodule commit in madmom 0.16.1. The eight online beat LSTM files have the same bytes as the files at the fork's model commit ``7e3dc1b0cad499792767074d03c38b194b9b0a79``. The model licence remains `CC BY-NC-SA 4.0 <https://github.com/CPJKU/madmom_models/blob/31fe6ce52affb794c2e3511908fbeccfd68f6925/LICENSE>`_. Existing source, model, and test-data licences remain applicable.

``metadata.json`` records the exact commits, authors, URLs, source hashes, model hashes, audio hashes, fixture hashes, environment, and tolerances. ``sample.wav`` and ``sample2.wav`` come from the upstream release's public test data. The generator reads those files from the archived release, not from an external evaluation set.

Reference environment
---------------------

The reference uses Python 3.11.15, NumPy 1.26.4, SciPy 1.13.1, Cython 0.29.37, setuptools 75.8.2, and mido 1.3.3. BLAS, OpenMP, and MKL each use one thread. Runtime compatibility aliases restore ``collections.MutableSequence`` and the removed NumPy names ``np.float``, ``np.int``, ``np.complex``, and ``np.bool`` to their original types. Temporary package metadata supplies the version query in upstream ``madmom.__init__``. These changes let the release run on Python 3.11 without changes to its numerical functions.

These fixtures preserve release algorithms with the stated dependency versions. They do not claim bit-identical results to an unspecified historical NumPy, SciPy, compiler, or CPU.

Coverage and tolerances
-----------------------

The two recordings provide 687 frames at 100 frames per second, with 2048-sample windows and 441-sample hops at 44100 Hz. Each fixture stores the 162 frontend features, eight separate network outputs, and ensemble activations. Separate arrays preserve the default centered origin, causal origin, and frame-at-a-time stream behavior. The tests also verify reset and chunk boundaries.

The DBN fixtures store every selected state and maximum forward probability. They store the full posterior every 37 frames and at the last frame to keep the fixture size small. Beat times come from the online forward decoder. Peak fixtures cover full calls, calls of 13 frames, and synthetic cases with zero values, equal maxima, threshold boundaries, and several past-context windows. A fixed chunk size preserves the observable buffering behavior of the upstream online peak processor.

The frontend uses ``atol=2e-6, rtol=2e-5``. Network activations use ``atol=2e-5, rtol=2e-4`` because float32 reduction order can differ between SIMD and BLAS. DBN posterior checks use ``atol=1e-9, rtol=2e-6`` because float32 logarithm rounding can accumulate in the normalized forward recursion. Standard Python API beat and peak times must match exactly when input activations are identical. End-to-end native beat times allow one 10 ms frame. These tolerances are explicit test limits, not measured errors.

Reproduction
------------

Install the reference dependency versions into a dedicated environment. The source and model Git objects must already exist locally. Then run this command from the repository root with that environment's Python::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python tools/generate_core_reference.py

Use ``--model-repository`` if the model Git objects are in another checkout of the same public repository. The generator writes fixtures under ``tests/data/native_0161`` and removes its temporary source, compiled extensions, and audio copies when it finishes. Run the parity tests in the normal build environment::

    python -m pytest tests/test_native_reference.py

The test recordings are short compatibility examples. They do not establish musical accuracy on a representative corpus or performance on another CPU architecture.
