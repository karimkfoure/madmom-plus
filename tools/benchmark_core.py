#!/usr/bin/env python3
"""Compare the Python processors before and after native core integration.

Each stage runs in a fresh process. CPU measurements exclude construction and
audio loading. Linux RSS measurements include Python and dependencies; they are
process memory measurements, not isolated kernel allocation measurements.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
STAGES = ('frontend', 'network', 'decoder', 'bulk',
          'stream_frontend', 'stream_network', 'stream_decoder', 'stream')
ALL_STAGES = STAGES + ('engine_128', 'engine_441', 'engine_2048')


def worker(args):
    sys.path.insert(0, str(Path(args.package_root).resolve()))
    import numpy as np
    import scipy
    import madmom.ml.hmm
    import madmom.ml.nn
    import madmom.ml.nn.layers
    from madmom.audio.signal import Signal, FramedSignal
    from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor

    def rss():
        return int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')

    def peak_rss():
        # VmHWM belongs to this executable's address space. getrusage maxrss
        # can include a larger pre-exec parent high-water mark on Linux.
        return next(int(line.split()[1]) * 1024 for line in
                    Path('/proc/self/status').read_text().splitlines()
                    if line.startswith('VmHWM:'))

    initial_rss = rss()
    audio = Signal(np.tile(np.concatenate([
        np.asarray(Signal(str(ROOT / 'tests/data/audio' / name),
                          sample_rate=44100, num_channels=1, dtype=np.float32))
        for name in ('sample.wav', 'sample2.wav')]), args.copies), sample_rate=44100)
    chunk_size = int(args.stage.split('_')[1]) if args.stage.startswith('engine_') else None
    if chunk_size:
        from madmom.native import NativeBeatTracker
        engine = NativeBeatTracker()
    else:
        engine = None
    rnn = (RNNBeatProcessor(online=True, origin='online', fps=100, num_threads=1)
           if engine is None else None)
    stream = RNNBeatProcessor(online=True, origin='stream', num_frames=1,
                             fps=100, num_threads=1) if args.stage.startswith('stream') else None
    dbn = DBNBeatTrackingProcessor(online=True, fps=100) if engine is None else None
    ready_rss = rss()
    features = rnn[0](audio) if args.stage in ('network', 'stream_network') else None
    activations = rnn(audio) if args.stage in ('decoder', 'stream_decoder') else None

    def run():
        if engine is not None:
            engine.reset()
            return [engine.push(audio[first:first + chunk_size])
                    for first in range(0, len(audio), chunk_size)]
        if args.stage == 'frontend':
            return rnn[0](audio, reset=True)
        if args.stage == 'network':
            return rnn[1](features, reset=True)
        if args.stage == 'decoder':
            return dbn(activations, reset=True)
        if args.stage == 'bulk':
            acts = rnn(audio, reset=True)
            return dbn(acts, reset=True)
        if args.stage == 'stream_network':
            return [rnn[1](frame[np.newaxis], reset=index == 0)
                    for index, frame in enumerate(features)]
        if args.stage == 'stream_decoder':
            return [dbn(frame[np.newaxis], reset=index == 0)
                    for index, frame in enumerate(activations)]
        events = []
        for index, frame in enumerate(FramedSignal(
                audio, frame_size=2048, hop_size=441, origin='online')):
            if args.stage == 'stream_frontend':
                events.append(stream[0](frame, reset=index == 0))
                continue
            acts = np.atleast_1d(stream(frame, reset=index == 0))
            events.extend(dbn(acts, reset=index == 0))
        return events

    run()
    times = []
    for _ in range(args.repeats):
        start = time.process_time()
        output = run()
        times.append(time.process_time() - start)
        del output
    final_rss, measured_peak = rss(), peak_rss()
    if engine is not None:
        engine.close()
    binaries = {}
    for name, module in (('hmm', madmom.ml.hmm),
                         ('layers', madmom.ml.nn.layers)):
        binaries[name] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    try:
        from madmom import _native
        binaries['core'] = hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest()
    except ImportError:
        binaries['core'] = None
    return {
        'stage': args.stage, 'audio_seconds': len(audio) / 44100,
        'cpu_seconds': times,
        'cpu_seconds_per_audio_second': statistics.median(times) / (len(audio) / 44100),
        'rss_before_creation_bytes': initial_rss,
        'rss_after_creation_bytes': ready_rss,
        'rss_after_measurement_bytes': final_rss,
        'peak_rss_bytes': measured_peak,
        'python': platform.python_version(), 'numpy': np.__version__,
        'scipy': scipy.__version__,
        'binary_sha256': binaries,
        'network_module_sha256': hashlib.sha256(
            Path(madmom.ml.nn.__file__).read_bytes()).hexdigest(),
        'streaming_wrapper_sha256': (hashlib.sha256(
            Path(sys.modules['madmom.native'].__file__).read_bytes()).hexdigest()
            if 'madmom.native' in sys.modules else None),
        'engine_chunk_samples': chunk_size,
        'engine_configuration': ({'networks': 8, 'precision': 'float32', 'simd': False}
                                 if chunk_size else None),
        'push_calls_per_run': ((len(audio) + chunk_size - 1) // chunk_size
                               if chunk_size else None),
        'cpu_microseconds_per_push': (
            statistics.median(times) * 1e6 /
            ((len(audio) + chunk_size - 1) // chunk_size) if chunk_size else None),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path)
    parser.add_argument('--reference-revision', help='Git revision of the preserved checkout')
    parser.add_argument('--package-root', type=Path, default=ROOT)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--copies', type=int, default=10)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--paired', action='store_true',
                        help='Alternate variants in fresh processes for each trial')
    parser.add_argument('--stages', choices=ALL_STAGES, nargs='+', default=STAGES)
    parser.add_argument('--stage', choices=ALL_STAGES)
    args = parser.parse_args()
    if args.stage:
        print(json.dumps(worker(args)))
        return
    if args.reference_root is None or args.output is None:
        parser.error('--reference-root and --output are required')
    if min(args.repeats, args.copies) < 1:
        parser.error('repeats and copies must be positive')
    report = {
        'schema_version': 1,
        'host': {
            'cpu': next(line.split(':', 1)[1].strip() for line in
                        Path('/proc/cpuinfo').read_text().splitlines()
                        if line.startswith('model name')),
            'architecture': platform.machine(),
            'affinity': sorted(os.sched_getaffinity(0)),
            'thread_environment': {key: os.environ.get(key) for key in
                                   ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
        },
        'settings': {'repeats': args.repeats, 'warmups': 1, 'audio_copies': args.copies,
                     'paired_trials': args.paired,
                     'sample_rate': 44100, 'fps': 100,
                     'timing': 'Process CPU; excludes loading and initialization',
                     'memory': 'Fresh Linux process RSS per variant and stage; includes dependencies'},
        'audio_sha256': {name: hashlib.sha256((ROOT / 'tests/data/audio' / name).read_bytes()).hexdigest()
                         for name in ('sample.wav', 'sample2.wav')},
        'variants': {},
        'provenance': {
            'before_revision': args.reference_revision,
            'after_revision': subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        },
    }
    variants = (('before', args.reference_root), ('after', args.package_root))

    def measure(root, stage, repeats):
        result = subprocess.run([
            sys.executable, str(Path(__file__).resolve()), '--stage', stage,
            '--package-root', str(root.resolve()), '--repeats', str(repeats),
            '--copies', str(args.copies)], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr)
        return json.loads(result.stdout)

    if args.paired:
        report['variants'] = {'before': {}, 'after': {}}
        for stage in args.stages:
            trials = {'before': [], 'after': []}
            for trial in range(args.repeats):
                order = variants if trial % 2 == 0 else variants[::-1]
                for name, root in order:
                    trials[name].append(measure(root, stage, 1))
            for name, rows in trials.items():
                result = rows[0].copy()
                result['cpu_seconds'] = [row['cpu_seconds'][0] for row in rows]
                result['cpu_seconds_per_audio_second'] = (
                    statistics.median(result['cpu_seconds']) / result['audio_seconds'])
                result['peak_rss_bytes'] = max(row['peak_rss_bytes'] for row in rows)
                if result.get('push_calls_per_run'):
                    result['cpu_microseconds_per_push'] = (
                        statistics.median(result['cpu_seconds']) * 1e6 /
                        result['push_calls_per_run'])
                result['trials'] = rows
                report['variants'][name][stage] = result
                print('%s %s: %.6f CPU s/audio s, peak %.2f MiB' % (
                    name, stage, result['cpu_seconds_per_audio_second'],
                    result['peak_rss_bytes'] / 2 ** 20), flush=True)
    else:
        for name, root in variants:
            stages = {}
            for stage in args.stages:
                stages[stage] = measure(root, stage, args.repeats)
                print('%s %s: %.6f CPU s/audio s, peak %.2f MiB' % (
                    name, stage, stages[stage]['cpu_seconds_per_audio_second'],
                    stages[stage]['peak_rss_bytes'] / 2 ** 20), flush=True)
            report['variants'][name] = stages
    report['source_sha256'] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for pattern in ('native/src/*.cpp', 'native/include/*.h', 'madmom/audio/_native.py',
                        'madmom/ml/nn/_native.py', 'madmom/ml/nn/__init__.py',
                        'madmom/ml/nn/layers.py', 'madmom/ml/hmm.pyx', 'madmom/features/beats.py',
                        'madmom/_native_decoders.py', 'madmom/features/onsets.py',
                        'madmom/native.py',
                        'setup.py', 'tools/benchmark_core.py')
        for path in ROOT.glob(pattern)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
