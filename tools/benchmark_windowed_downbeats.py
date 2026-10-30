#!/usr/bin/env python3
"""Measure optional windowed downbeat cost; no music-accuracy claim."""

import argparse
import hashlib
import importlib.metadata
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


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--audio-dir', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'tests/data/audio')
    parser.add_argument('--copies', type=int, default=3)
    parser.add_argument('--chunk', type=int, default=441)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--worker', choices=('beats', 'windowed'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.copies, args.chunk, args.trials) < 1:
        parser.error('copies, chunk and trials must be positive')
    if not args.worker and args.output is None:
        parser.error('--output is required')
    return args


def worker(args):
    # Import the installed package after its build. Do not add a source checkout.
    import scipy
    from madmom.audio.signal import Signal
    from madmom.features.downbeats_window import (
        WindowedDownBeatTrackingProcessor, _PythonBeatTracker)

    names = ('sample.wav', 'sample2.wav')
    audio = np.tile(np.concatenate([
        Signal(str(args.audio_dir / name), sample_rate=44100,
               num_channels=1, dtype=np.float32) for name in names]), args.copies)
    if args.worker == 'beats':
        tracker = _PythonBeatTracker()
    else:
        tracker = WindowedDownBeatTrackingProcessor(window=6., hop=2., background=False)

    def run():
        rows = []
        for first in range(0, len(audio), args.chunk):
            block = audio[first:first + args.chunk]
            if args.worker == 'beats':
                frames = tracker.push(block)
                rows.append(frames[frames[:, 2] != 0, 0])
            else:
                rows.append(tracker(block, reset=False)[:, 0])
        return np.concatenate(rows)

    try:
        run()  # One complete warmup; model loading and setup are outside timing.
        tracker.reset()
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        events = run()
        cpu_seconds = time.process_time() - cpu_start
        wall_seconds = time.perf_counter() - wall_start
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_mib = peak / (1024. ** 2 if sys.platform == 'darwin' else 1024.)
        return dict(variant=args.worker, audio_seconds=len(audio) / 44100.,
                    cpu_seconds=cpu_seconds, wall_seconds=wall_seconds,
                    cpu_per_audio_second=cpu_seconds / (len(audio) / 44100.),
                    peak_mib=peak_mib, beat_count=len(events),
                    event_times=events.tolist(),
                    versions=dict(python=platform.python_version(), numpy=np.__version__,
                                  scipy=scipy.__version__,
                                  madmom_plus=importlib.metadata.version('madmom-plus')))
    finally:
        if hasattr(tracker, 'close'):
            tracker.close()


def main(args):
    trials = []
    reference = None
    for trial in range(args.trials):
        order = ('beats', 'windowed') if trial % 2 == 0 else ('windowed', 'beats')
        for variant in order:
            print('Trial %d/%d: %s' % (trial + 1, args.trials, variant), file=sys.stderr)
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', variant,
                       '--audio-dir', str(args.audio_dir.resolve()), '--copies', str(args.copies),
                       '--chunk', str(args.chunk)]
            record = json.loads(subprocess.check_output(command, text=True))
            record['trial'] = trial + 1
            events = np.array(record.pop('event_times'), dtype=np.float64)
            if reference is None:
                reference = events
            if not np.array_equal(reference, events):
                raise RuntimeError('Beat event times differ between variants or trials')
            record['events_sha256'] = hashlib.sha256(events.astype('<f8').tobytes()).hexdigest()
            trials.append(record)
    summary = {}
    for variant in ('beats', 'windowed'):
        rows = [row for row in trials if row['variant'] == variant]
        summary[variant] = {key: statistics.median(row[key] for row in rows)
                            for key in ('cpu_seconds', 'cpu_per_audio_second', 'peak_mib')}
    model = platform.processor()
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.exists():
        model = next((line.split(':', 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                      if line.startswith('model name')), model)
    report = dict(schema_version=1, purpose='Optional processing cost; no music-accuracy claim',
                  hardware=dict(cpu=model, architecture=platform.machine(),
                                affinity=sorted(os.sched_getaffinity(0)),
                                blas_threads=os.environ.get('OPENBLAS_NUM_THREADS'),
                                omp_threads=os.environ.get('OMP_NUM_THREADS')),
                  input=dict(files=['sample.wav', 'sample2.wav'], copies=args.copies,
                             sample_rate=44100, chunk_samples=args.chunk,
                             audio_seconds=trials[0]['audio_seconds']),
                  method=dict(fresh_process=True, alternating_order=True,
                              warmup_runs=1, trials_per_variant=args.trials,
                              window_seconds=6., hop_seconds=2., background=False,
                              cpu_excludes='model construction, loading, warmup and reset',
                              peak_includes='process setup, models, audio and warmup'),
                  beat_times_equal=True, beat_count=len(reference), summary=summary, trials=trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    options = arguments()
    if options.worker:
        print(json.dumps(worker(options)))
    else:
        main(options)
