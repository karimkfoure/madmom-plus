Neural call overhead
====================

The buffer adapter reduced median neural ensemble CPU time from 275.67 to 179.11 microseconds per frame on the measured x86_64 host. This is a 1.54-fold speed increase for this stage. The main change removes repeated Python validation and uses one native call for all eight networks. The C++ inference arithmetic stays the same.

The comparison uses 687 stored feature frames from the two public upstream test recordings. Both adapters use the same current native binary, Python 3.11.15, NumPy 2.4.6, and SciPy 1.17.1. The process uses one allowed CPU core and one BLAS thread. Seven timed trials follow one warm trial; trial order alternates. The activation arrays match exactly. The measured scope is the neural call adapter. Frontend cost, full beat tracking cost, memory use, and accuracy on a representative music corpus require separate benchmarks.

The reference adapter, by Karím Kfoure and contributors, comes from the public fork at `commit 38c4b6371049daa24ba537c197908b4d2102bd46 <https://github.com/karimkfoure/madmom-plus/tree/38c4b6371049daa24ba537c197908b4d2102bd46>`_. The benchmark loads that Python adapter from local Git history and selects its sequential ensemble dispatch. Both variants call the current shared C++ inference kernels. This method holds inference arithmetic constant so the result measures adapter changes. Complete release cost is measured by the separate pipeline benchmark.

The raw trials, separate profiles, runtime versions, model hashes, fixture hashes, source hashes, and native binary hash are in ``native-nn-calls.json``. Profile counters include profiler overhead and are collected after the timed trials. Public model and fixture attribution remains in ``tests/data/native_0161/metadata.json`` and the model licence files.

Reproduce this measurement from a built source checkout that contains the reference commit::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      nice -n 10 taskset -c 5 python tools/benchmark_nn_calls.py \
      --repeats 7 --output native-nn-calls.json

Repeat the comparison on aarch64 hardware before using these figures for that architecture. CPU frequency and other host work were not controlled. The complete pipeline benchmark is required to assess the effect on a real application.
