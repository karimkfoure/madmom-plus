Native music-analysis core
==========================

The C++ core implements the supported online beat path used by the existing Python processors. It is required in every build, like the Cython extensions. Python and the standalone C engine share frontend, neural, forward-HMM, and beat-event arithmetic. There is no backend selection flag.

Build and Python compatibility
------------------------------

A source build needs C and C++17 compilers, Python headers, and the usual isolated build dependencies. Initialize the model submodule before installation::

    git submodule update --init madmom/models
    python -m pip install .

``MADMOM_BUILD_NATIVE`` is no longer required: the extension is always built. A failed core build fails installation. ``setup.py`` selects GCC/Clang or MSVC compiler options. See `native-wheels.rst <native-wheels.rst>`_ for local wheel targets and verification limits. No CI build is enabled.

Existing calls retain their names, arguments, processor sequence, model files, and outputs::

    from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor

    activations = RNNBeatProcessor(online=True, origin='online')('audio.wav')
    beats = DBNBeatTrackingProcessor(online=True, fps=100)(activations)

``online=True`` selects online models and frontend settings. It does not change the existing frame-origin default. Use ``origin='online'`` for causal bulk frames, or ``origin='stream', num_frames=1`` when supplying a frame from a streaming source. The standalone C engine always uses causal frames. The single-frame network result retains its time axis, as in madmom 0.16.1.

The fused frontend supports the released 2048-sample, 81-band chain with int16 or float32 audio. Python retains framing, resampling, and filterbank construction. C++ computes the FFT, magnitudes, filtering, logarithm, and positive spectral difference. It copies at most 64 frames at a time. Changed chain members, unsupported precision, and operations without a native kernel keep their existing implementation. Tests cover configuration changes, history transfer, reset, and serialization.

Native neural kernels support arbitrary float32 dense, recurrent, and LSTM dimensions with supported standard activation functions. They retain the existing model arrays and public state. Model replacement, in-place edits, strided weights, reset, and pickle behavior remain supported. Other layers, dtypes, and custom functions keep their existing implementations because no matching native kernel exists.

The generic C++ HMM kernel performs forward recursion. ``HiddenMarkovModel.forward`` still returns all posterior rows. The additive ``forward_states`` method returns each frame's most likely state with one posterior row of scratch space. The online beat processor uses this method and the same event rule as the C engine. Native peak picking retains the original threshold, zero-boundary filters, and equal-maximum behavior. Offline Viterbi decoding remains in Cython.

FFT and streaming design
-------------------------

The frontend uses a half-size real FFT with precomputed roots and permutation indices. Windowing and FFT arithmetic use double precision; the complex64 cast before magnitude follows upstream. Contiguous filter coefficients, fused logarithm and difference, and fixed audio/history rings reduce repeated work and allocation. A local FFT avoids another runtime dependency and external FFT licensing or binary-distribution constraints. Bulk and frame-at-a-time costs must be measured separately: a faster streaming core does not imply a faster FFT than NumPy in every batch configuration.

The neural kernels vectorize independent outputs with SSE2 on supported x86 builds or NEON on AArch64. Other targets use scalar arithmetic. Supported compiler settings disable fused multiply-add contraction to limit numerical changes. Normal processing retains all eight float32 networks. Fewer networks and rounded weights remain explicit experiments; they can change beat events.

Standalone streaming API
-------------------------

The stable versioned interface is ``native/include/madmom_engine.h``. It provides an opaque handle, create/configure/destroy, sample push, frame and tracker-state reads, reset, and a version query. ``native/README.md`` documents the interface and a C example. Other headers expose component kernels; they are not the versioned application interface.

Build the library independently of Python::

    cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release
    cmake --build build/native --parallel 2

The engine accepts finite mono float32 PCM at 44100 Hz. It uses a 2048-sample Hann window, 441-sample hop, 81 logarithmic bands, and 81 positive differences. Each frame uses samples through its integer sample timestamp and pads missing history with zeros. One sample emits the frame at sample zero. There is no end padding or flush. Chunk boundaries do not change output. Audio timestamps exclude delivery, resampling, scheduling, and computation delay.

The output queue has a fixed capacity. ``MM_ENGINE_FULL`` consumes no samples: read pending frames, then retry. Configuration failure preserves the previous engine. Reset clears audio, difference, recurrent, decoder, event, queue, and clock state. Use each handle from one thread at a time; separate handles are independent. Maximum DBN posterior is exposed as raw reliability. It is not a calibrated music-presence probability; downstream applications own that policy.

Model assets and Python wrapper
--------------------------------

Export installed online models once for use outside Python::

    from madmom.native import export_model
    export_model('online.bin')

The asset retains format magic ``MMONL001`` and the existing little-endian arrays. It contains the window, filterbank, difference lag, eight networks, and default DBN. Export uses trusted installed pickle files; native inference reads the binary asset. The models retain their `CC BY-NC-SA 4.0 licence <https://github.com/CPJKU/madmom_models/blob/7e3dc1b0cad499792767074d03c38b194b9b0a79/LICENSE>`_. Conversion does not change these terms.

``madmom.native.NativeBeatTracker`` remains available. ``push`` returns rows containing seconds, activation, beat event, and tempo. ``features``, ``network``, and ``decode`` expose component calls; ``decoder_state`` and ``summary`` return copies. Reset and context-manager cleanup remain supported. Without a supplied asset, construction exports and removes a temporary asset. The versioned C interface provides integer sample timestamps for application integration.

Parity and verification
------------------------

`native-reference.rst <native-reference.rst>`_ documents stored outputs from unmodified madmom 0.16.1 and its original model revision. Two public recordings cover 687 frames, individual networks, default and causal framing, streaming, forward posteriors, beats, peaks, and reset. Tests do not generate expected values with the current fork.

Frontend tolerance is ``atol=2e-6, rtol=2e-5``. Activation tolerance is ``atol=2e-5, rtol=2e-4``. Posterior tolerance is ``atol=1e-9, rtol=2e-6``. Identical input activations must produce exact Python beat and peak times; the native audio path allows one 10 ms frame. Measured maxima are 2.39e-7 for features, 4.48e-7 for Python-API activations, and 2.39e-7 for direct SIMD activations. Sampled posteriors match exactly with identical observations. Beat times match exactly on both clips. These short clips do not establish accuracy on a representative corpus.

Run Python and independent native checks::

    python -m pytest tests
    python -c "from madmom.native import export_model; export_model('online.bin')"
    cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release \
      -DMADMOM_NATIVE_TEST_ASSET="$PWD/online.bin"
    cmake --build build/native --parallel 2
    ctest --test-dir build/native --output-on-failure

Host measurements
-------------------

``tools/benchmark_core.py`` compares the existing Python APIs with a preserved pre-integration checkout. It measures bulk and frame-at-a-time frontend, network, decoder, and complete paths in separate processes. All CPU runs, Linux resident memory, dependency versions, audio hashes, and source hashes are recorded in ``docs/benchmarks/core-host.json``. ``tools/benchmark_engine.py`` compares the independent C engine with the pre-optimization native library. The README summarizes verified results.

CPU measurements exclude loading, model initialization, export, and scoring. Process memory includes Python and imported dependencies; it is not the isolated native allocation size. CPU frequency and other host work were not controlled. Repeated short recordings are a compatibility workload, not a representative corpus. Earlier optional-backend results remain in ``docs/benchmarks/native-host.json`` and describe the prior implementation.

Physical aarch64 measurements still required
--------------------------------------------

QEMU checks compilation and behavior, not ARM performance. On actual aarch64 hardware, repeat stage and complete-path benchmarks with one pinned core. Record CPU, compiler, enabled SIMD, clock policy, temperature, cooling, dependencies, CPU seconds per audio second, and peak resident memory. Compare NEON activations, posteriors, and events with frozen fixtures. Measure callback wall time and high percentiles, backlog, and deadline misses for the intended block size. Measure capture and resampling separately. Public annotated development data must justify changes to ensemble size, precision, or musical defaults.

Provenance
------------

Reference algorithms are by Sebastian Böck and the madmom contributors: `madmom 0.16.1 <https://github.com/CPJKU/madmom/tree/6545fbb3ecdae0aac336a95ec591f642e0b74f3e>`_, commit ``6545fbb3ecdae0aac336a95ec591f642e0b74f3e``, and the inherited `development source <https://github.com/CPJKU/madmom/tree/27f032e8947204902c675e5e341a3faf5dc86dae>`_, commit ``27f032e8947204902c675e5e341a3faf5dc86dae``. Model authors, URLs, and exact release/current commits are recorded in the reference documentation and fixture metadata. Upstream credit and licences remain intact.

The kernels, FFT optimization, buffer integration, and versioned C interface are original project work. No external native implementation, FFT library, neural runtime, or third-party optimization is bundled. The initial implementation retains Karím Kfoure's authorized contribution from source snapshot ``8fbbb3666f5006a5714954a56c96626d397019f5``. Source code keeps the project code licence; model assets keep their separate licence.
