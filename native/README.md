# Native beat engine

The Python online beat processors and the C engine use the same C++ frontend, neural networks, and forward hidden Markov model (HMM) decoder. Shared kernels keep the two interfaces numerically consistent. The public C interface is [madmom_engine.h](include/madmom_engine.h).

```sh
cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --parallel 2
cmake --install build/native --prefix build/native-install
```

Export the installed models once:

```python
from madmom.native import export_model
export_model('online.bin')
```

| Input | Output | Default configuration |
| --- | --- | --- |
| Mono float32 PCM, 44100 Hz | Activation, beat event, integer sample timestamp | Eight float32 networks; SIMD where supported |
| Chunks up to 44100 samples | Raw maximum posterior, tempo, tracker state | Fixed queue for 1024 frames |

The engine performs no capture or resampling. Model conversion preserves the separate [CC BY-NC-SA 4.0 model licence](../madmom/models/LICENSE).

```c
#include <madmom_engine.h>

int main(void) {
    mm_engine *engine = mm_engine_create("online.bin", NULL);
    if (!engine) return 1;
    float pcm[441] = {0};
    size_t accepted = 0;
    int status = mm_engine_push(engine, pcm, 441, &accepted);
    mm_engine_frame_v1 frames[2];
    int count = mm_engine_read(engine, frames, 2);
    mm_engine_destroy(engine);
    if (status != 0 || accepted != 441 || count < 0) return 1;
    return 0;
}
```

| Contract | Behavior |
| --- | --- |
| Version | `mm_engine_version() >> 16 == 1`; version 1 structs remain stable |
| Timing | A frame at sample `n` uses samples through `n`; missing history is zero |
| First frame | One sample emits timestamp 0; later frames occur every 441 samples |
| Full queue | `MM_ENGINE_FULL` consumes no input; read queued frames and retry |
| Reset | Clears audio, neural, decoder, queue, event, and clock state |
| Configuration failure | Keeps the previous configuration and state |
| Threads | Use each handle from one thread at a time |
| Reliability | Maximum DBN posterior; not a calibrated music-presence probability |

There is no end padding or flush. Chunk boundaries do not change results. Timestamps exclude delivery and computation delay. Reduced ensemble size and rounded weights are explicit experiments that can change beat events; use eight float32 networks for compatibility.

Run the independent API checks after model export:

```sh
cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release -DMADMOM_NATIVE_TEST_ASSET="$PWD/online.bin"
cmake --build build/native --parallel 2
ctest --test-dir build/native --output-on-failure
```

See [Python compatibility and measurements](../docs/native.rst), [release reference outputs](../docs/native-reference.rst), and [Linux wheel builds](../docs/native-wheels.rst). The algorithms derive from Sebastian Böck and the madmom contributors' [source at `27f032e`](https://github.com/CPJKU/madmom/tree/27f032e8947204902c675e5e341a3faf5dc86dae). The C interface, FFT, and adapters are original project work; no external native runtime is bundled.
