#!/usr/bin/env python3
"""Measure neural call overhead with the previous and current Python adapters.

Both adapters use the same installed C++ inference binary. The reference comes
from local Git history. Trials alternate in one process to reduce drift. This
test isolates adapter cost; it does not measure audio loading or beat accuracy.
"""

import argparse
import cProfile
import hashlib
import json
import os
from pathlib import Path
import platform
import pstats
import statistics
import subprocess
import sys
import time
import types


ROOT = Path(__file__).resolve().parents[1]


def git(*arguments):
    return subprocess.check_output(['git', *arguments], cwd=ROOT, text=True).strip()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def cpu_name():
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith(('model name', 'Hardware')):
                return line.partition(':')[2].strip()
    except OSError:
        pass
    return platform.processor()


def profile(function):
    profiler = cProfile.Profile()
    profiler.enable()
    function()
    profiler.disable()
    rows = []
    for (filename, line, name), (_, calls, own, cumulative, _) in pstats.Stats(profiler).stats.items():
        try:
            filename = str(Path(filename).relative_to(ROOT))
        except ValueError:
            if not filename.startswith('reference/'):
                filename = Path(filename).name
        rows.append({'file': filename, 'line': line, 'function': name, 'calls': calls,
                     'self_seconds': own, 'cumulative_seconds': cumulative})
    return sorted(rows, key=lambda row: row['self_seconds'], reverse=True)[:20]


def benchmark(arguments):
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import scipy
    from madmom import _native as extension
    from madmom.features.beats import RNNBeatProcessor
    from madmom.ml.nn import NeuralNetworkEnsemble, _native
    from madmom.models import BEATS_LSTM
    from madmom.processors import SequentialProcessor

    reference_revision = git('rev-parse', '--verify', '--end-of-options',
                             arguments.reference_revision + '^{commit}')
    reference_source = git('show', reference_revision + ':madmom/ml/nn/_native.py') + '\n'
    if '.mm_nn_network(' not in reference_source:
        raise ValueError('The reference must contain the previous ctypes neural adapter')
    reference = types.ModuleType('madmom.ml.nn._reference_call_adapter')
    reference.__package__ = 'madmom.ml.nn'
    exec(compile(reference_source, 'reference/madmom/ml/nn/_native.py', 'exec'), reference.__dict__)

    fixture_root = ROOT / 'tests/data/native_0161'
    metadata = json.loads((fixture_root / 'metadata.json').read_text())
    for clip in metadata['clips']:
        if sha256((fixture_root / clip['fixture']).read_bytes()) != clip['fixture_sha256']:
            raise ValueError('Stored feature fixture hash does not match its metadata')
    features = np.concatenate([np.load(fixture_root / clip['fixture'])['features']
                               for clip in metadata['clips']])
    frames = [frame[np.newaxis] for frame in features]
    audio_seconds = sum(clip['samples'] for clip in metadata['clips']) / metadata['settings']['sample_rate']
    ensemble = RNNBeatProcessor(online=True, num_threads=1, fps=100)[1]
    current_adapter = _native.network
    current_process = NeuralNetworkEnsemble.process

    def select(variant):
        _native.network = reference.network if variant == 'before' else current_adapter
        NeuralNetworkEnsemble.process = SequentialProcessor.process if variant == 'before' else current_process

    def run():
        return np.concatenate([ensemble(frame, reset=index == 0)
                               for index, frame in enumerate(frames)])

    measurements = {'before': [], 'after': []}
    outputs = {}
    profiles = {}
    try:
        for trial in range(arguments.repeats + 1):
            # Alternate the order as well as the variant to reduce drift.
            order = ('before', 'after') if trial % 2 == 0 else ('after', 'before')
            for variant in order:
                select(variant)
                start = time.process_time()
                outputs[variant] = run()
                if trial:
                    measurements[variant].append(time.process_time() - start)
        np.testing.assert_array_equal(outputs['after'], outputs['before'])
        for variant in measurements:
            select(variant)
            profiles[variant] = profile(run)
    finally:
        _native.network = current_adapter
        NeuralNetworkEnsemble.process = current_process

    variants = {}
    for variant, values in measurements.items():
        median = statistics.median(values)
        variants[variant] = {'cpu_seconds': values, 'median_cpu_seconds': median,
                             'microseconds_per_hop': median / len(features) * 1e6,
                             'cpu_seconds_per_audio_second': median / audio_seconds,
                             'profile': profiles[variant]}
    sources = ('madmom/ml/nn/_native.py', 'madmom/ml/nn/__init__.py',
               'native/src/nn.cpp', 'native/src/nn_bindings.cpp', 'native/src/bindings.cpp',
               'tools/benchmark_nn_calls.py')
    return {
        'schema_version': 1,
        'host': {'cpu': cpu_name(), 'python': platform.python_version(), 'numpy': np.__version__,
                 'scipy': scipy.__version__, 'architecture': platform.machine(),
                 'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                 'thread_environment': {name: os.environ.get(name) for name in
                                        ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}},
        'settings': {'repeats': arguments.repeats, 'warmups': 1, 'frames': len(features),
                     'networks': len(ensemble[0]), 'audio_seconds': audio_seconds,
                     'scope': 'Concatenate stored public features, reset once per trial, process one frame per call; exclude loading and construction',
                     'comparison': 'Load the reference Python ctypes adapter and inherited sequential ensemble dispatcher; both variants use the current shared C++ inference binary',
                     'profiling': 'Collect separate cProfile diagnostics after timing; profile counters include profiler overhead'},
        'provenance': {'reference_revision': reference_revision, 'current_revision': git('rev-parse', 'HEAD'),
                       'reference_adapter_sha256': sha256(reference_source.encode()),
                       'source_sha256': {name: sha256((ROOT / name).read_bytes()) for name in sources},
                       'binary_sha256': sha256(Path(extension.__file__).read_bytes()),
                       'features_sha256': sha256(features.tobytes()),
                       'fixtures': {clip['fixture']: clip['fixture_sha256'] for clip in metadata['clips']},
                       'model_sha256': {Path(name).name: sha256(Path(name).read_bytes())
                                        for name in BEATS_LSTM}},
        'variants': variants,
        'speedup': variants['before']['median_cpu_seconds'] / variants['after']['median_cpu_seconds'],
        'activation_max_abs_error': float(np.max(np.abs(outputs['after'] - outputs['before']))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-revision', default='38c4b63')
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.repeats < 1:
        parser.error('--repeats must be positive')
    report = benchmark(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'speedup': report['speedup'],
                      'activation_max_abs_error': report['activation_max_abs_error'],
                      'microseconds_per_hop': {variant: result['microseconds_per_hop']
                                                for variant, result in report['variants'].items()}}, indent=2))


if __name__ == '__main__':
    main()
