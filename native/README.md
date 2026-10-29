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

## Arithmetic and tests

The frontend uses a half-size real FFT, double precision window/FFT arithmetic, madmom's complex64 cast before magnitude, contiguous filterbank coefficients, fused logarithm and positive spectral difference, and fixed rings for audio and difference history. It needs no external FFT library and performs no per-frame audio allocation. The shared neural kernels vectorize independent outputs with SSE2 or AArch64 NEON while preserving the input accumulation order. Fused multiply-add contraction is disabled in supported builds to limit numeric differences.

The C test checks queue capacity, sample timestamps, chunk retries, reset, configuration failure, component parity, invalid inputs, and decoder failure recovery. Run it with an exported model:

```sh
cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release -DMADMOM_NATIVE_TEST_ASSET="$PWD/online.bin"
cmake --build build/native --parallel 2
ctest --test-dir build/native --output-on-failure
```

See [the native documentation](../docs/native.rst) and [reference fixture documentation](../docs/native-reference.rst) for stored upstream outputs, tolerances, measurements, and platform limits. The reference algorithms are by Sebastian Böck and the madmom contributors in [CPJKU/madmom](https://github.com/CPJKU/madmom/tree/27f032e8947204902c675e5e341a3faf5dc86dae), commit `27f032e8947204902c675e5e341a3faf5dc86dae`. The FFT and C interface are original project code; no external native implementation is bundled.
