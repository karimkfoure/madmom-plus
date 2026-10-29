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
    rnn = RNNBeatProcessor(online=True, origin='online', fps=100, num_threads=1)
    stream = RNNBeatProcessor(online=True, origin='stream', num_frames=1,
                             fps=100, num_threads=1) if args.stage.startswith('stream') else None
    dbn = DBNBeatTrackingProcessor(online=True, fps=100)
    ready_rss = rss()
    features = rnn[0](audio) if args.stage in ('network', 'stream_network') else None
    activations = rnn(audio) if args.stage in ('decoder', 'stream_decoder') else None

    def run():
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
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path)
    parser.add_argument('--reference-revision', help='Git revision of the preserved checkout')
    parser.add_argument('--package-root', type=Path, default=ROOT)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--copies', type=int, default=10)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--stage', choices=STAGES)
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
    for name, root in (('before', args.reference_root), ('after', args.package_root)):
        stages = {}
        for stage in STAGES:
            result = subprocess.run([
                sys.executable, str(Path(__file__).resolve()), '--stage', stage,
                '--package-root', str(root.resolve()), '--repeats', str(args.repeats),
                '--copies', str(args.copies)], capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stderr)
            stages[stage] = json.loads(result.stdout)
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
                        'setup.py', 'tools/benchmark_core.py')
        for path in ROOT.glob(pattern)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
