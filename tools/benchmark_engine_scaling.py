#!/usr/bin/env python3
"""Compare NativeBeatTracker.push cost for fixed audio block sizes.

The two variants use their own preserved Python wrapper and shared library.
Loading and model creation are excluded. Reports contain hashes and names,
with no local paths. Run this tool with one allowed CPU and one BLAS thread.
"""

import argparse
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

import numpy as np

from benchmark_engine import Library, ROOT, digest, read_audio


def tracker(package, asset, label):
    source = package / 'madmom/native.py'
    binary = next(package / 'madmom' / ('_native' + suffix)
                  for suffix in importlib.machinery.EXTENSION_SUFFIXES
                  if (package / 'madmom' / ('_native' + suffix)).is_file())
    backend = Library(binary, asset)
    spec = importlib.util.spec_from_file_location('native_' + label, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Only replace extension discovery; all push validation and buffering use
    # the exact preserved wrapper implementation and its own native library.
    module._library = lambda: backend.library
    instance = module.NativeBeatTracker(model_path=asset, networks=8,
                                         precision='float32', simd=True)
    return instance, {'wrapper_sha256': digest(source),
                      'binary_name': binary.name, 'binary_sha256': digest(binary)}


def run(instance, audio, chunk):
    instance.reset()
    pieces = [instance.push(audio[first:first + chunk])
              for first in range(0, len(audio), chunk)]
    return np.concatenate(pieces)


def benchmark(args):
    audio = np.ascontiguousarray(np.tile(np.concatenate([
        read_audio(ROOT / 'tests/data/audio' / name)
        for name in ('sample.wav', 'sample2.wav')]), args.copies))
    duration = len(audio) / 44100.
    instances, provenance = {}, {}
    for label, package in (('before', args.before_package),
                            ('after', args.after_package)):
        instances[label], provenance[label] = tracker(package, args.asset, label)
    report = {
        'schema_version': 1,
        'host': {'cpu': next(line.split(':', 1)[1].strip() for line in
                             Path('/proc/cpuinfo').read_text().splitlines()
                             if line.startswith('model name')),
                 'architecture': platform.machine(),
                 'python': platform.python_version(), 'numpy': np.__version__,
                 'affinity': sorted(os.sched_getaffinity(0)),
                 'thread_environment': {name: os.environ.get(name) for name in
                     ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}},
        'method': {'audio_seconds': duration, 'copies': args.copies,
                   'repeats': args.repeats, 'warmups_per_trial': 1,
                   'alternating_variants': True, 'networks': 8, 'precision': 'float32',
                   'simd': True,
                   'timing': 'Process CPU; includes the exact NativeBeatTracker.push wrapper, native engine, reset and output collection; excludes source loading, audio loading and model creation',
                   'limitations': 'Two short public recordings are tiled for timing. Frequency and unrelated host activity are uncontrolled. No corpus accuracy or AArch64 hardware performance conclusion follows.'},
        'provenance': {'before_revision': args.before_revision,
                       'after_revision': subprocess.check_output(
                           ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                       'libraries': provenance,
                       'model_sha256': digest(args.asset),
                       'script_sha256': digest(Path(__file__)),
                       'audio_sha256': {name: digest(ROOT / 'tests/data/audio' / name)
                           for name in ('sample.wav', 'sample2.wav')}},
        'chunks': [],
    }
    expected = {}
    try:
        for chunk in args.chunks:
            times = {'before': [], 'after': []}
            outputs = {}
            for trial in range(args.repeats):
                order = ('before', 'after') if trial % 2 == 0 else ('after', 'before')
                for label in order:
                    run(instances[label], audio, chunk)
                    start = time.process_time()
                    outputs[label] = run(instances[label], audio, chunk)
                    times[label].append(time.process_time() - start)
            for label in ('before', 'after'):
                if label in expected:
                    np.testing.assert_array_equal(outputs[label], expected[label])
                else:
                    expected[label] = outputs[label].copy()
            before, after = outputs['before'], outputs['after']
            np.testing.assert_array_equal(before[:, 0], after[:, 0])
            np.testing.assert_allclose(after[:, 1], before[:, 1], atol=2e-5, rtol=2e-4)
            rates = {label: statistics.median(values) / duration
                     for label, values in times.items()}
            report['chunks'].append({
                'samples': chunk, 'hops': chunk / 441., 'cpu_seconds': times,
                'cpu_seconds_per_audio_second': rates,
                'speedup': rates['before'] / rates['after'],
                'frames': len(after), 'exact_chunk_invariance_per_variant': True,
                'activation_max_abs_error': float(np.max(np.abs(after[:, 1] - before[:, 1]))),
                'beat_event_indices_equal': bool(np.array_equal(before[:, 2], after[:, 2])),
            })
            print('%d samples: before %.6f, after %.6f CPU s/audio s' % (
                chunk, rates['before'], rates['after']), flush=True)
    finally:
        for instance in instances.values():
            instance.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-package', required=True, type=Path)
    parser.add_argument('--after-package', type=Path, default=ROOT)
    parser.add_argument('--before-revision', default='38c4b63')
    parser.add_argument('--asset', required=True, type=Path)
    parser.add_argument('--chunks', nargs='+', type=int, default=[441, 2205, 4410])
    parser.add_argument('--copies', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if min(args.copies, args.repeats, *args.chunks) < 1:
        parser.error('Copies, repeats and chunk sizes must be positive')
    report = benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
