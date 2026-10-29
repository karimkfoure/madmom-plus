# Native beat engine

The engine uses the same C++ frontend, neural inference, and forward hidden Markov model kernels as the Python processors. This keeps both interfaces on one implementation. The public interface is [madmom_engine.h](include/madmom_engine.h); the other headers expose internal component kernels.

Build the library without Python:

```sh
cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --parallel 2
cmake --install build/native --prefix build/native-install
```

Export the installed online models once with Python:

```python
from madmom.native import export_model
export_model('online.bin')
```

The exported asset retains the model licence. See [the model licence](../madmom/models/LICENSE). The asset includes the released eight-network online ensemble and default beat decoder. The C interface accepts mono float32 PCM at 44100 Hz, normally from -1 through +1. It performs no resampling or audio capture.

## Version 1 interface

Include `madmom_engine.h` and link `madmom_native`. Check that `mm_engine_version() >> 16` equals `1`. The struct and function definitions for this major version remain stable. Pass `NULL` configuration to select all eight float32 networks, SIMD where supported, and a queue for 1024 frames.

```c
#include <madmom_engine.h>

mm_engine *engine = mm_engine_create("online.bin", NULL);
/* If engine is NULL, read mm_engine_error(). */
float pcm[441] = {0};
size_t accepted = 0;
int status = mm_engine_push(engine, pcm, 441, &accepted);
mm_engine_frame_v1 frames[2];
int count = mm_engine_read(engine, frames, 2);
mm_engine_state_v1 state;
int state_status = mm_engine_state(engine, &state);
/* Check return codes before reading results. */
mm_engine_destroy(engine);
```

Creation returns `NULL` on failure. Other fallible calls return `-1` and provide thread-local error text through `mm_engine_error()`. `mm_engine_push()` can also return `MM_ENGINE_FULL`: the output queue lacks capacity, so no input was consumed. Read pending frames and retry the same input. A push accepts at most 44100 samples. The queue has a fixed capacity from 1 through 8192 frames. A read with `NULL` output and zero capacity reports the pending frame count without removing frames.

Each frame contains an activation, beat event, and integer sample index relative to the latest reset. The causal frame at sample index `n` uses input through sample `n`, with zeros for missing past samples. One sample emits the frame at index zero. Later frames occur every 441 samples. There is no end padding or flush. Chunk boundaries do not change output or timestamps. These timestamps describe the audio; they exclude delivery and computation delay.

The state query reports the latest frame index, accepted sample count, last beat index, tempo, selected DBN state, and maximum posterior probability. `MM_ENGINE_NO_SAMPLE` marks an unavailable frame or beat index. The posterior maximum is a raw tracker reliability signal. It is not a calibrated probability that music or a usable pulse is present.

`mm_engine_reset()` clears audio history, spectral differences, recurrent state, decoder probabilities, clocks, events, and queued frames. `mm_engine_configure()` reloads the same model asset and resets these states. A failed configuration preserves the previous engine. Each handle belongs to one thread at a time; separate handles are independent. Destroying `NULL` is safe. Do not reuse a destroyed handle.

Configuration can reduce the ensemble or round weights for explicit experiments. Reduced precision still uses float32 storage and arithmetic. These settings can change activation values and beat events. Use all eight float32 models when compatibility is required.

## Measured frontend cost

The first native-core report compares commit `daca5b4` with the native implementation at commit `3c7b36a` on an AMD Ryzen 9 5900X. It records frontend cost of 0.00684 CPU seconds per audio second, compared with 0.01101 before, a 1.61 times speed increase. Complete engine cost is 0.02026 CPU seconds per audio second, compared with 0.02399 before, a 1.18 times speed increase. That historical comparison uses the same ctypes component calls in both libraries, all eight float32 networks, SIMD, and 441-sample pushes.

The two upstream WAV files contain 6.851 seconds of audio. Frontend outputs match the previous native implementation exactly. The largest activation difference after shared neural inference is 2.09e-7; frame times and beat events match exactly. These comparisons establish continuity with the previous native code; the separate stored 0.16.1 fixtures establish upstream parity.

The live heap increase after model creation falls from 1,339,232 to 1,320,976 bytes. After a one-second input push, it falls from 1,353,856 to 1,321,232 bytes. Both measurement processes reach 63,508 KiB peak resident memory, including Python, NumPy, SciPy, and audio arrays. Live heap counts exclude allocation peaks that were already freed. These measurements do not show a process peak-memory reduction.

The report uses one pinned CPU, one BLAS thread, one warm run, and the median of seven process CPU measurements. It includes result allocation and copying, and excludes model loading and audio/frame preparation. CPU frequency and unrelated host activity were not controlled. Short test recordings do not establish accuracy on a representative music corpus or predict AArch64 hardware cost. Raw runs, model/audio hashes, source hashes, binary hashes, and memory observations are in [engine-host.json](../docs/benchmarks/engine-host.json).

To reproduce the comparison against the previous native source:

```sh
git show 3c7b36a:native/src/online.cpp > build/native-before.cpp
c++ -std=c++17 -O3 -ffp-contract=off -shared -fPIC -I native/include build/native-before.cpp -o build/libmadmom_before.so
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0 python tools/benchmark_engine.py --before-library build/libmadmom_before.so --after-library build/native/libmadmom_native.so --asset online.bin --output engine-host.json
```

## Further streaming measurements

A later paired comparison uses the exact `NativeBeatTracker.push` Python wrapper and native library from merged commit `38c4b63`, then the runtime at `20e6bc1`. This includes the shared FFT and push-wrapper improvements. All eight float32 networks and SIMD remain enabled. Larger blocks reduce Python call cost while keeping the same audio calculation.

| Input samples per call | Analysis hops | Before CPU s / audio s | After CPU s / audio s | Speed increase |
| --- | --- | --- | --- | --- |
| 441 | 1 | 0.01537 | 0.01048 | 1.47 times |
| 2205 | 5 | 0.01360 | 0.01006 | 1.35 times |
| 4410 | 10 | 0.01333 | 0.00996 | 1.34 times |

Each variant produces exactly the same output for all three block sizes. Before/after beat events match exactly; the maximum activation difference is 1.58e-6. The run uses 34.254 seconds of tiled public test audio, one pinned CPU, one BLAS thread, and three alternating trials with one warm run per trial. Loading and model creation are excluded. Host frequency and unrelated activity are uncontrolled. These short recordings do not establish corpus accuracy or AArch64 hardware cost.

Raw CPU trials, hashes, and comparison results are in [engine-scaling-host.json](../docs/benchmarks/engine-scaling-host.json). Reproduce the comparison with a preserved source tree and its compiled extension:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 4 python tools/benchmark_engine_scaling.py --before-package preserved-checkout --asset online.bin --output engine-scaling.json
```

The [streaming profile report](../docs/benchmarks/stream-profile-host.json) records function calls before and after the changes. It uses elapsed profiler time, which includes profiler overhead; use paired process CPU reports for cost claims. Configuration checks still scan the exposed window and filterbank on each call because users can edit those arrays in place. The speed changes preserve that behavior.

## Arithmetic and tests

The frontend uses a half-size real FFT with independent double-precision SSE2 or AArch64 NEON butterflies, madmom's complex64 cast before magnitude, contiguous filterbank coefficients, fused logarithm and positive spectral difference, and fixed rings for audio and difference history. Magnitude uses a squared norm and square root in the ordinary range, with scaled `hypot` at overflow and underflow limits. It needs no external FFT library and performs no per-frame audio allocation. The shared neural kernels vectorize independent outputs with SSE2 or AArch64 NEON while preserving the input accumulation order. Fused multiply-add contraction is disabled in supported builds to limit numeric differences.

The C test checks queue capacity, sample timestamps, chunk retries, reset, configuration failure, component parity, invalid inputs, and decoder failure recovery. Run it with an exported model:

```sh
cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release -DMADMOM_NATIVE_TEST_ASSET="$PWD/online.bin"
cmake --build build/native --parallel 2
ctest --test-dir build/native --output-on-failure
```

See [the native documentation](../docs/native.rst) and [reference fixture documentation](../docs/native-reference.rst) for stored upstream outputs, tolerances, measurements, and platform limits. The reference algorithms are by Sebastian Böck and the madmom contributors in [CPJKU/madmom](https://github.com/CPJKU/madmom/tree/27f032e8947204902c675e5e341a3faf5dc86dae), commit `27f032e8947204902c675e5e341a3faf5dc86dae`. The FFT and C interface are original project code; no external native implementation is bundled.
