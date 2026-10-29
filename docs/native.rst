Optional native streaming beat tracker
=====================================

The native backend runs madmom's causal beat frontend, released online LSTM ensemble, and forward DBN in a small C++ library. It reduces Python work per frame and keeps stream state between calls. It is an explicit opt-in API; existing processors, command-line programs, and model files keep their existing behavior.

Build and install
-----------------

Use a C++ compiler that supports C++17 and the GCC or Clang command-line options. Initialize the existing model submodule before a source install::

    git submodule update --init madmom/models
    MADMOM_BUILD_NATIVE=1 python -m pip install .

The environment variable adds ``madmom._native`` to the usual build. Without it, the build keeps the existing Python/Cython backend and needs no additional C++ compiler. Importing ``madmom.native`` works when the optional extension is absent. ``available()`` returns a boolean; creating a native tracker raises an explanatory ``ImportError`` when the extension is absent.

The C++ library has no external runtime dependency beyond the C++ standard library. It includes its own radix-2 FFT, so no FFT or neural inference package is required. Python uses ``ctypes`` to call the extension's public C ABI; these calls release the Python GIL. The binding has no NumPy C ABI dependency. Existing madmom dependencies still apply.

Python API
----------

This example streams normalized mono PCM. Resample and mix channels before calling ``push`` because the native API uses a fixed 44100 Hz sample rate::

    import numpy as np
    from madmom.audio.signal import Signal
    from madmom.native import NativeBeatTracker

    audio = Signal('tests/data/audio/sample.wav', sample_rate=44100,
                   num_channels=1, dtype=np.float32)
    with NativeBeatTracker() as tracker:
        for first in range(0, len(audio), 441):
            rows = tracker.push(audio[first:first + 441])
            beat_times = rows[rows[:, 2] != 0, 0]
            posterior = tracker.decoder_state
            tempo, last_beat, probability, state_index = tracker.summary

``push(audio)`` accepts a finite one-dimensional floating point array. It converts the array to contiguous float32 PCM, normally in the range -1 to +1. Integer PCM is rejected because an implicit conversion would use the wrong amplitude scale. Calls can contain any number of samples, including zero. The Python binding divides long calls into bounded C API chunks. Output has shape ``(frames, 4)`` and dtype float64, with these columns:

* Frame reference time in seconds, starting at zero.
* Beat activation from the float32 network ensemble.
* Beat event, represented by 0 or 1.
* Last estimated tempo in beats per minute, or zero before the first event.

The frontend uses a 2048-sample Hann window, 441-sample hop, 81 logarithmic frequency bands, and 81 positive spectral differences. It uses samples through the frame reference sample and pads the missing past with zeros. One sample is enough to emit the frame at time zero. Chunk boundaries do not change the sample clock or arithmetic. There is no end padding and no flush operation. The last frame has reference index strictly below the number of samples received.

A frame reference time is not its delivery time. For example, a 441-sample call can return the frame at time zero only after that call arrives. Audio capture, resampling, scheduling, computation, and publication add delay. The API does not predict future beat commitments or provide downbeats.

``reset()`` clears audio history, spectral differences, recurrent state, DBN probabilities, and clocks. ``close()`` releases memory and is safe to repeat. A closed tracker rejects processing calls. Use a context manager to release memory reliably. Use each instance from one thread at a time. Different instances have separate state.

``features(frame2048)`` advances only the frontend difference state and returns 162 float32 features. ``network(features162)`` advances only the recurrent networks and returns one activation. ``decode(activation)`` advances only the DBN and its frame clock, and returns one four-column row. These methods support component checks; mixing them with ``push`` requires the caller to keep all component clocks consistent. ``decoder_state`` and ``summary`` return copies and do not advance any state. The posterior maximum is a raw model probability, not a calibrated confidence score.

``decode`` rejects non-finite activations and values outside 0 through 1. Repeated endpoint values can make every DBN path impossible. In that case it raises ``ValueError`` and keeps the previous posterior and frame clock, so the caller can supply a possible observation or reset the tracker.

Model export
------------

Without ``model_path``, construction exports the installed online models to a temporary file, loads them, and deletes that file. To avoid repeated export, create an asset once::

    from madmom.native import export_model, NativeBeatTracker

    export_model('online.bin')
    tracker = NativeBeatTracker(model_path='online.bin')

Export uses the trusted, installed madmom pickle files. Native inference uses the resulting binary asset and does not load pickle files. The asset contains the window, filterbank, difference lag, eight networks, and default DBN. It uses magic ``MMONL001`` and little-endian float32, float64, and uint32 arrays in the order written by ``export_model``. Dimensions are fixed by this format version. A truncated or inconsistent asset is rejected. The initial API supports the released online architecture and default DBN settings; arbitrary networks and decoder settings remain available through the existing Python APIs.

The models keep their `CC BY-NC-SA 4.0 licence <https://github.com/CPJKU/madmom_models/blob/7e3dc1b0cad499792767074d03c38b194b9b0a79/LICENSE>`_. Conversion or reduced precision does not remove these terms. See ``madmom/models/README.rst`` and ``madmom/models/LICENSE`` for attribution and permitted use.

C API
-----

Build the independent library without Python::

    cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release
    cmake --build build/native --parallel 2

The public header is ``native/include/madmom_native.h``. Use a model asset exported by Python, link ``libmadmom_native``, and retain the opaque handle between calls::

    #include <madmom_native.h>

    void *tracker = mm_online_create("online.bin", 8, 0, 0);
    /* Check tracker for NULL and read mm_online_error() on failure. */
    float pcm[441] = {0};
    double rows[3 * 4];
    int count = mm_online_push(tracker, pcm, 441, rows, 3);
    /* Check count for -1. Each successful row has four doubles. */
    mm_online_destroy(tracker);

``mm_online_push`` accepts at most 44100 samples per call. Capacity counts rows; ``sample_count / 441 + 2`` rows always suffice. Check return codes before reading output. Creation returns NULL on error; other fallible calls return -1. ``mm_online_error`` returns the current thread's last error string. The header defines fixed-size component buffers and the query/copy protocol for DBN state. The caller must provide valid buffers of the documented sizes. Handles must not be reused after destruction.

Speed and precision options
---------------------------

``NativeBeatTracker(networks=8, precision='float32', simd=False)`` preserves the full model by default. These options are explicit because reducing the ensemble or changing arithmetic can change the beat sequence:

* ``networks=1`` through ``8`` selects the first N models in ``BEATS_LSTM`` order. It does not select a model by score on the test material.
* ``simd=True`` uses four-lane SSE2 dot products on supported x86 builds or NEON on AArch64. Other targets use scalar code. Different reduction order can change the last bits of activations. Fused multiply-add contraction is disabled to limit this change.
* ``precision='float16'`` rounds matrix and peephole weights to binary16 values at load time. ``precision='int8'`` uses symmetric per-row scaling to -127 through +127. Biases and initial states remain float32. Both modes decode weights back to float32 and use float32 arithmetic and storage. They measure the accuracy effect of weight rounding; they do not provide packed weight storage or accelerated low-precision inference.

A different hop is not an exact speed option. A smaller hop increases the frame count and changes the difference lag and DBN timing. A larger hop also changes the observations. The native format therefore keeps 441 samples and 100 frames per second.

Parity and measured host cost
----------------------------

The tests compare the full frontend and ensemble to ``RNNBeatProcessor(online=True, origin='online', fps=100)`` and the decoder to ``DBNBeatTrackingProcessor(online=True, fps=100)``. Tests cover three upstream audio files, component stages, smaller ensembles, SIMD, chunk boundaries, reset, invalid inputs, and operation without the optional extension. Activation tolerance is ``atol=2e-5, rtol=2e-4``; beat tolerance is one frame (10 ms). On the two benchmark recordings, the measured full-precision maximum activation error is 4.48e-7, and all beat times match exactly. DBN posterior tolerance is ``atol=1e-9, rtol=2e-6``. This allows the last-bit difference between NumPy's vector float32 logarithm and the scalar C++ logarithm. The measured maximum posterior difference on the three test recordings is 5.38e-8 absolute and 1.20e-6 relative. Identical input activations produce identical beat events.

Run the Python parity checks and independent C ABI checks with an exported model::

    python -m pytest tests/test_native.py
    python -c "from madmom.native import export_model; export_model('online.bin')"
    cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release \
      -DMADMOM_NATIVE_TEST_ASSET="$PWD/online.bin"
    cmake --build build/native --parallel 2
    ctest --test-dir build/native --output-on-failure

On the recorded host environment, the full suite has 820 passes and two failures in existing chroma and integer resampling checks. Both failures reproduce on the original ``27f032e`` source without the native backend. Existing tests and their tolerances remain unchanged.

Both default and opt-in native wheels were built from the source archive and passed import and inference checks. The default wheel contains no native extension and builds with the C++ compiler disabled. The native Python checks pass 11 tests and 15 subtests; the independent C ABI test also passes.

The host benchmark uses an AMD Ryzen 9 5900X, one allowed core, one BLAS thread, Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1, and GCC 13.3.0. It warms each path once, then takes the median of five process CPU measurements per recording. The two recordings contain 6.851 seconds of audio in total. The table combines the per-recording median CPU time, divided by total audio duration. Timing includes frontend, networks, DBN, Python calls, and result collection; it excludes loading, export, resampling, and scoring. ``sample.wav`` has only four annotated beats; ``sample2.wav`` has no beat annotation.

.. list-table:: Host cost and test-sample accuracy
   :header-rows: 1
   :widths: 38 24 18 20

   * - Path
     - CPU s / audio s
     - Sample F1 at 70 ms
     - Max activation error
   * - Python causal bulk
     - 0.03841
     - 0.6667
     - Reference
   * - Python frame by frame
     - 0.07768
     - 0.6667
     - 2.39e-7
   * - Native, 8, scalar
     - 0.02882
     - 0.6667
     - 4.48e-7
   * - Native, 8, SIMD
     - 0.02009
     - 0.6667
     - 2.39e-7
   * - Native, 4, SIMD
     - 0.01770
     - 0.6000
     - 0.11054
   * - Native, 2, SIMD
     - 0.01493
     - 0.6000
     - 0.15416
   * - Native, 1, SIMD
     - 0.01411
     - 0.6000
     - 0.25689
   * - Native, 8, fp16 rounded, SIMD
     - 0.02073
     - 0.6667
     - 0.001116
   * - Native, 8, int8 rounded, SIMD
     - 0.02061
     - 0.6667
     - 0.010245

CPU frequency and other host work were not controlled, so these timings describe this run. The full SIMD path is 3.87 times faster than Python frame-by-frame processing on this host. Fewer models reduce cost but change events. Rounded weights do not improve measured runtime. These short recordings do not establish accuracy on a representative music corpus. Keep all eight float32 models for the initial native option; validate SIMD on the target platform before selecting it for deployment.

The individual runs, audio and model hashes, exact beat arrays, and environment are in ``docs/benchmarks/native-host.json``. Reproduce them with the optional extension installed::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python tools/benchmark_native.py --repeats 5 --output native-host.json

On a Raspberry Pi class device, repeat this benchmark on the actual CPU and compiler with one pinned core. Record the board, architecture, clock policy, cooling, temperature, compiler flags, actual SIMD support, peak memory, and process CPU seconds per audio second. Measure complete callback wall time and its high percentiles with real capture and resampling separately. Check backlog and deadline misses for the intended audio block size. Compare scalar and NEON activations, DBN state, and events against the Python reference. Check accuracy on a representative annotated development set before choosing a smaller ensemble or reduced precision. No Raspberry Pi runtime or NEON parity result is claimed here.

Provenance
----------

Karím Kfoure contributed and authorized reuse of the initial native arithmetic from the owner's source snapshot ``8fbbb3666f5006a5714954a56c96626d397019f5``. This contribution adapts it into an independent public library, replaces the external FFT with a local implementation, adds optional bindings and validation, and removes unrelated processing paths. No external native code or ONNX runtime is bundled.

The reference frontend, recurrent layers, and DBN are the existing `CPJKU/madmom source <https://github.com/CPJKU/madmom/tree/27f032e8947204902c675e5e341a3faf5dc86dae>`_ by Sebastian Böck and the madmom contributors, pinned at ``27f032e8947204902c675e5e341a3faf5dc86dae``. Models come from `CPJKU/madmom_models <https://github.com/CPJKU/madmom_models/tree/7e3dc1b0cad499792767074d03c38b194b9b0a79>`_, authored by Sebastian Böck and contributors, pinned at ``7e3dc1b0cad499792767074d03c38b194b9b0a79``. Upstream credit and licences remain in place. The native code uses the project code licence; exported model assets retain the separate model licence.
