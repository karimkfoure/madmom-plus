#!/usr/bin/env python3
"""Compare the same native component calls in two locally built libraries.

Input audio comes from the upstream public test files. The report contains
hashes and names, but no local paths. Build the earlier library from the
reported revision with the same compiler flags before running this tool.
"""

import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time

import numpy as np
import scipy
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 44100
HOP = 441


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Library:
    def __init__(self, path, asset):
        self.library = ct.CDLL(os.fspath(path))
        signatures = {
            'create': ([ct.c_char_p, ct.c_int, ct.c_int, ct.c_int], ct.c_void_p),
            'destroy': ([ct.c_void_p], None),
            'reset': ([ct.c_void_p], ct.c_int),
            'features': ([ct.c_void_p, ct.c_void_p, ct.c_void_p], ct.c_int),
            'push': ([ct.c_void_p, ct.c_void_p, ct.c_size_t,
                      ct.c_void_p, ct.c_size_t], ct.c_int),
            'error': ([], ct.c_char_p),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.library, 'mm_online_' + name)
            function.argtypes, function.restype = args, result
        self.asset = asset
        self.handle = None

    def create(self):
        self.handle = self.library.mm_online_create(os.fsencode(self.asset), 8, 0, 1)
        if not self.handle:
            raise RuntimeError(self.library.mm_online_error().decode())

    def close(self):
        if self.handle:
            self.library.mm_online_destroy(self.handle)
            self.handle = None

    def check(self, result):
        if result < 0:
            raise RuntimeError(self.library.mm_online_error().decode())
        return result

    def frontend(self, frames):
        self.check(self.library.mm_online_reset(self.handle))
        output = np.empty((len(frames), 162), np.float32)
        for index, frame in enumerate(frames):
            self.check(self.library.mm_online_features(
                self.handle, frame.ctypes.data, output[index].ctypes.data))
        return output

    def engine(self, audio, chunk_size):
        self.check(self.library.mm_online_reset(self.handle))
        output = np.empty((len(audio) // HOP + 2, 4), np.float64)
        scratch = np.empty((chunk_size // HOP + 2, 4), np.float64)
        position = 0
        for start in range(0, len(audio), chunk_size):
            chunk = audio[start:start + chunk_size]
            rows = self.check(self.library.mm_online_push(
                self.handle, chunk.ctypes.data, len(chunk),
                scratch.ctypes.data, len(scratch)))
            output[position:position + rows] = scratch[:rows]
            position += rows
        return output[:position]


def read_audio(path):
    sample_rate, audio = wavfile.read(path)
    if sample_rate != SAMPLE_RATE:
        raise ValueError('The benchmark requires 44100 Hz source audio')
    if audio.ndim == 2:
        # Match the integer channel-mix convention before PCM normalization.
        audio = np.mean(audio, axis=1).astype(audio.dtype)
    if audio.dtype.kind == 'i':
        maximum = np.iinfo(audio.dtype).max
        audio = np.asarray(audio, np.float32) / maximum
    elif audio.dtype.kind == 'f':
        audio = np.asarray(audio, np.float32)
    else:
        raise ValueError('Use signed integer or floating point WAV audio')
    return np.ascontiguousarray(audio)


def measure(function, repeats):
    function()
    timings = []
    for _ in range(repeats):
        start = time.process_time()
        output = function()
        timings.append(time.process_time() - start)
    return output, timings


class Mallinfo(ct.Structure):
    _fields_ = [(name, ct.c_size_t) for name in (
        'arena', 'ordblks', 'smblks', 'hblks', 'hblkhd',
        'usmblks', 'fsmblks', 'uordblks', 'fordblks', 'keepcost')]


try:
    _mallinfo = ct.CDLL(None).mallinfo2
    _mallinfo.argtypes, _mallinfo.restype = [], Mallinfo
except (AttributeError, OSError):
    _mallinfo = None


def live_heap_bytes():
    """Return live glibc allocations, when mallinfo2 is available."""
    if _mallinfo is None:
        return None
    value = _mallinfo()
    return int(value.uordblks + value.hblkhd)


def memory_worker(library_path, asset):
    library = Library(library_path, asset)
    audio = np.zeros(SAMPLE_RATE, np.float32)
    before = live_heap_bytes()
    library.create()
    after_create = live_heap_bytes()
    library.engine(audio, SAMPLE_RATE)
    after_push = live_heap_bytes()
    result = {
        'live_heap_increase_after_create_bytes': (
            after_create - before if before is not None else None),
        'live_heap_increase_after_push_bytes': (
            after_push - before if before is not None else None),
        'process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    library.close()
    return result


def provenance(library, revision):
    result = {'binary_name': Path(library).name, 'binary_sha256': digest(library),
              'source_revision': revision}
    if revision:
        completed = subprocess.run(
            ['git', 'show', revision + ':native/src/online.cpp'], cwd=ROOT,
            capture_output=True, check=True)
        result['online_cpp_sha256'] = hashlib.sha256(completed.stdout).hexdigest()
    return result


def benchmark(args):
    libraries = {name: Library(path, args.asset) for name, path in (
        ('before', args.before_library), ('after', args.after_library))}
    for library in libraries.values():
        library.create()
    compiler = subprocess.run(
        [os.environ.get('CXX', 'c++'), '--version'], capture_output=True,
        text=True, check=True).stdout.splitlines()[0]
    cpu = platform.processor()
    try:
        cpu = next(line.partition(':')[2].strip() for line in
                   Path('/proc/cpuinfo').read_text().splitlines()
                   if line.startswith('model name'))
    except (OSError, StopIteration):
        pass
    report = {
        'schema_version': 1,
        'host': {
            'cpu': cpu, 'machine': platform.machine(), 'system': platform.system(),
            'python': platform.python_version(), 'numpy': np.__version__,
            'scipy': scipy.__version__, 'compiler': compiler,
            'cpu_affinity': sorted(os.sched_getaffinity(0))
            if hasattr(os, 'sched_getaffinity') else None,
            'thread_environment': {name: os.environ.get(name) for name in
                                   ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                    'MKL_NUM_THREADS')},
        },
        'method': {
            'repeats': args.repeats, 'warm_runs': 1, 'chunk_samples': args.chunk_size,
            'networks': 8, 'precision': 'float32', 'simd': True,
            'before_compile_flags': '-std=c++17 -O3 -ffp-contract=off -shared -fPIC',
            'after_build': 'CMake Release; cxx_std_17; -O3 -DNDEBUG -ffp-contract=off; MADMOM_NATIVE_BUILD=1',
            'timing': 'Process CPU seconds; identical ctypes component calls in both libraries; model loading and audio/frame preparation excluded; output allocation and result copies included',
            'frontend': '2048-sample causal frames; 441-sample hop; window, FFT, magnitude, filterbank, log and positive spectral difference; one call per frame',
            'engine': 'Frontend, full ensemble, forward DBN and beat events; bounded sample pushes',
            'memory': 'Separate process per library; live glibc heap increase after create and a 44100-sample push; process peak RSS includes Python, NumPy, SciPy and audio; freed allocation peaks are not captured by live heap counts',
            'limitations': 'Two short upstream recordings; host frequency and unrelated activity uncontrolled; these costs do not predict AArch64 hardware performance or music-corpus accuracy',
        },
        'model_sha256': digest(args.asset),
        'provenance': {
            'before': provenance(args.before_library, args.before_revision),
            'after': provenance(args.after_library, args.after_revision),
            'current_source_sha256': {
                name: digest(ROOT / name) for name in (
                    'native/src/online.cpp', 'native/src/nn.cpp',
                    'native/src/decoders.cpp', 'tools/benchmark_engine.py')},
        },
        'clips': [],
    }
    try:
        for path in args.audio:
            audio = read_audio(path)
            frames = np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(
                np.pad(audio, (2047, 0)), 2048)[::HOP])
            duration = len(audio) / SAMPLE_RATE
            clip = {'audio': Path(path).name, 'audio_sha256': digest(path),
                    'audio_seconds': duration, 'stages': {}}
            for stage in ('frontend', 'engine'):
                outputs, timings = {}, {}
                for name, library in libraries.items():
                    function = (lambda lib=library: lib.frontend(frames)) if stage == 'frontend' else (
                        lambda lib=library: lib.engine(audio, args.chunk_size))
                    outputs[name], times = measure(function, args.repeats)
                    timings[name] = {
                        'cpu_seconds': times, 'cpu_seconds_median': statistics.median(times),
                        'cpu_seconds_per_audio_second': statistics.median(times) / duration,
                    }
                before, after = outputs['before'], outputs['after']
                np.testing.assert_equal(before.shape, after.shape)
                error = np.abs(after - before)
                comparison = {'max_abs_error': float(error.max()),
                              'exact_equal': bool(np.array_equal(before, after))}
                if stage == 'frontend':
                    np.testing.assert_allclose(after, before, atol=2e-6, rtol=2e-5)
                    comparison.update(atol=2e-6, rtol=2e-5)
                else:
                    np.testing.assert_allclose(after[:, 1], before[:, 1], atol=2e-5, rtol=2e-4)
                    np.testing.assert_array_equal(after[:, [0, 2]], before[:, [0, 2]])
                    comparison.update(activation_atol=2e-5, activation_rtol=2e-4,
                                      exact_frame_times_and_events=True,
                                      beat_samples=(np.flatnonzero(after[:, 2]) * HOP).tolist())
                clip['stages'][stage] = {'timings': timings, 'comparison': comparison}
            report['clips'].append(clip)
    finally:
        for library in libraries.values():
            library.close()
    duration = sum(clip['audio_seconds'] for clip in report['clips'])
    report['summary'] = {'audio_seconds': duration}
    for stage in ('frontend', 'engine'):
        rates = {name: sum(clip['stages'][stage]['timings'][name]['cpu_seconds_median']
                           for clip in report['clips']) / duration
                 for name in ('before', 'after')}
        report['summary'][stage] = {
            'cpu_seconds_per_audio_second': rates,
            'speedup': rates['before'] / rates['after'],
        }
    report['memory'] = {}
    for name, path in (('before', args.before_library), ('after', args.after_library)):
        output = subprocess.run(
            [sys.executable, os.fspath(Path(__file__).resolve()), '--memory-worker',
             os.fspath(path), os.fspath(args.asset)], check=True,
            capture_output=True, text=True)
        report['memory'][name] = json.loads(output.stdout)
    return report


def main():
    if len(sys.argv) == 4 and sys.argv[1] == '--memory-worker':
        print(json.dumps(memory_worker(sys.argv[2], sys.argv[3])))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-library', required=True, type=Path)
    parser.add_argument('--after-library', required=True, type=Path)
    parser.add_argument('--asset', required=True, type=Path)
    parser.add_argument('--before-revision', default='3c7b36a')
    parser.add_argument('--after-revision', default=None)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--chunk-size', type=int, default=HOP)
    parser.add_argument('--audio', nargs='+', type=Path, default=[
        ROOT / 'tests/data/audio/sample.wav', ROOT / 'tests/data/audio/sample2.wav'])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or not 1 <= args.chunk_size <= SAMPLE_RATE:
        parser.error('Use positive repeats and a chunk size from 1 through 44100')
    report = benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
