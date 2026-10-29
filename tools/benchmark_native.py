#!/usr/bin/env python3
"""Measure native streaming cost and accuracy on local, annotated audio.

Run from a source checkout with the optional native extension installed. The
default inputs are the two upstream test recordings. No audio is downloaded.
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
import tempfile
import time

import numpy as np
import scipy

from madmom.audio.signal import FramedSignal, Signal
from madmom.evaluation.beats import BeatEvaluation
from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor
from madmom.native import NativeBeatTracker, available, export_model


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 44100
FPS = 100
VARIANTS = {
    "native-8-scalar": {"networks": 8, "precision": "float32", "simd": False},
    "native-8-simd": {"networks": 8, "precision": "float32", "simd": True},
    "native-4-simd": {"networks": 4, "precision": "float32", "simd": True},
    "native-2-simd": {"networks": 2, "precision": "float32", "simd": True},
    "native-1-simd": {"networks": 1, "precision": "float32", "simd": True},
    "native-8-float16-simd": {
        "networks": 8, "precision": "float16", "simd": True,
    },
    "native-8-int8-simd": {
        "networks": 8, "precision": "int8", "simd": True,
    },
}


def _host():
    """Return hardware and software details without host names or paths."""
    cpu = platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith(("model name", "Hardware")):
                cpu = line.partition(":")[2].strip()
                break
    except OSError:
        pass
    compiler = None
    try:
        result = subprocess.run(
            [os.environ.get("CXX", "c++"), "--version"], check=True,
            text=True, capture_output=True, timeout=10,
        )
        compiler = result.stdout.splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    return {
        "cpu": cpu,
        "machine": platform.machine(),
        "system": platform.system(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "compiler_available": compiler,
        "cpu_affinity": affinity,
        "thread_environment": {
            name: os.environ.get(name) for name in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def _provenance():
    """Identify the exact binary and sources without recording local paths."""
    from madmom import _native

    binary = Path(_native.__file__)
    revision = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            text=True, capture_output=True, timeout=10,
        )
        revision = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    sources = (
        "native/src/online.cpp",
        "native/include/madmom_native.h",
        "native/src/bindings.cpp",
        "madmom/native.py",
        "setup.py",
        "tools/benchmark_native.py",
    )
    return {
        "git_revision": revision,
        "native_binary": {
            "filename": binary.name,
            "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        },
        "source_sha256": {
            source: hashlib.sha256((ROOT / source).read_bytes()).hexdigest()
            if (ROOT / source).is_file() else None
            for source in sources
        },
    }


def _rows(activations, beats):
    activations = np.asarray(activations, dtype=np.float32).reshape(-1)
    rows = np.zeros((len(activations), 4), dtype=np.float64)
    rows[:, 0] = np.arange(len(activations)) / FPS
    rows[:, 1] = activations
    indices = np.rint(np.asarray(beats) * FPS).astype(int)
    rows[indices, 2] = 1
    return rows


def _python_runner(streaming):
    """Construct the reference once so model loading is outside timing."""
    rnn = RNNBeatProcessor(
        online=True, origin="stream" if streaming else "online",
        num_frames=1 if streaming else None, fps=FPS, num_threads=1,
    )
    dbn = DBNBeatTrackingProcessor(online=True, fps=FPS)

    def run(signal):
        if not streaming:
            activations = rnn(signal, reset=True)
            return _rows(activations, dbn(activations, reset=True))
        frames = FramedSignal(
            signal, frame_size=2048, hop_size=441, origin="online",
        )
        activations, beats = [], []
        for index, frame in enumerate(frames):
            activation = np.asarray(rnn(frame, reset=index == 0)).reshape(-1)
            if len(activation) != 1:
                raise RuntimeError("Python streaming must emit one activation per frame")
            activations.append(activation[0])
            beats.extend(dbn(activation, reset=index == 0))
        return _rows(activations, beats)

    return run


def _native_runner(tracker, chunk_size):
    def run(signal):
        tracker.reset()
        chunks = [tracker.push(signal[first:first + chunk_size])
                  for first in range(0, len(signal), chunk_size)]
        return np.concatenate(chunks) if chunks else np.empty((0, 4))
    return run


def _measure(run, signal, repeats):
    """Warm once, then retain every process CPU time for review."""
    run(signal)
    cpu_seconds = []
    rows = None
    for _ in range(repeats):
        start = time.process_time()
        rows = run(signal)
        cpu_seconds.append(time.process_time() - start)
    duration = len(signal) / SAMPLE_RATE
    return rows, {
        "cpu_seconds": cpu_seconds,
        "cpu_seconds_median": statistics.median(cpu_seconds),
        "cpu_seconds_per_audio_second": statistics.median(cpu_seconds) / duration,
    }


def _compare(rows, reference, annotations):
    if rows.shape != reference.shape:
        raise RuntimeError("Native and Python frame counts differ: %r != %r" %
                           (rows.shape, reference.shape))
    if not np.array_equal(rows[:, 0], reference[:, 0]):
        raise RuntimeError("Native and Python frame reference times differ")
    difference = rows[:, 1] - reference[:, 1]
    events = rows[rows[:, 2] != 0, 0]
    expected_events = reference[reference[:, 2] != 0, 0]
    same_event_count = len(events) == len(expected_events)
    result = {
        "frames": len(rows),
        "activation_max_abs_error": float(np.max(np.abs(difference))),
        "activation_rmse": float(np.sqrt(np.mean(difference ** 2))),
        "beat_times_seconds": events.tolist(),
        "same_event_count_as_python": same_event_count,
        "beat_max_abs_error_seconds": (
            float(np.max(np.abs(events - expected_events)))
            if same_event_count and len(events) else 0.0 if same_event_count else None
        ),
        "accuracy": None,
    }
    if annotations is not None:
        evaluation = BeatEvaluation(events, annotations, fmeasure_window=0.070)
        result["accuracy"] = {
            "fmeasure_window_seconds": 0.070,
            "fmeasure": float(evaluation.fmeasure),
            "precision": float(evaluation.precision),
            "recall": float(evaluation.recall),
            "annotation_count": int(len(annotations)),
        }
    return result


def benchmark(args):
    if not available():
        raise RuntimeError("Build the optional native extension before this benchmark")
    report = {
        "schema_version": 1,
        "host": _host(),
        "provenance": _provenance(),
        "settings": {
            "sample_rate": SAMPLE_RATE, "fps": FPS,
            "frame_size": 2048, "hop_size": 441,
            "native_chunk_samples": args.chunk_size,
            "repeats": args.repeats,
            "warm_runs": 1,
            "network_selection": "First N released online LSTMs in model order",
            "quantization": "Weights are rounded and decoded to float32 before inference",
            "timing_scope": "Process CPU time includes frontend, networks, DBN, Python calls and result collection; excludes model export/loading, audio loading and scoring",
            "time_reference": "Frames end at their reported reference time; caller delivery and computation delay are excluded from beat times",
            "limitations": "The test recordings do not establish accuracy on a representative music corpus; host cost does not predict Raspberry Pi cost",
        },
        "clips": [],
    }
    bulk_run = _python_runner(streaming=False)
    stream_run = _python_runner(streaming=True)
    with tempfile.TemporaryDirectory(prefix="madmom-native-benchmark-") as folder:
        model = Path(folder) / "model.bin"
        export_model(model)
        report["model_sha256"] = hashlib.sha256(model.read_bytes()).hexdigest()
        trackers = {}
        try:
            for name in args.variants:
                trackers[name] = NativeBeatTracker(model_path=model, **VARIANTS[name])
            for audio_path in args.audio:
                signal = Signal(os.fspath(audio_path), num_channels=1, sample_rate=SAMPLE_RATE,
                                dtype=np.float32)
                if len(signal) == 0:
                    raise ValueError("Audio must contain at least one sample")
                annotation_path = args.annotation_dir / (audio_path.stem + ".beats")
                annotations = np.loadtxt(annotation_path, ndmin=1) if annotation_path.is_file() else None
                clip = {
                    "audio": audio_path.name,
                    "audio_sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
                    "audio_seconds": len(signal) / SAMPLE_RATE,
                    "annotations": annotation_path.name if annotations is not None else None,
                    "variants": {},
                }
                reference, cost = _measure(bulk_run, signal, args.repeats)
                clip["variants"]["python-bulk"] = {
                    **cost, **_compare(reference, reference, annotations),
                }
                rows, cost = _measure(stream_run, signal, args.repeats)
                clip["variants"]["python-stream"] = {
                    **cost, **_compare(rows, reference, annotations),
                }
                for name, tracker in trackers.items():
                    rows, cost = _measure(_native_runner(tracker, args.chunk_size),
                                          signal, args.repeats)
                    clip["variants"][name] = {
                        "configuration": VARIANTS[name],
                        **cost, **_compare(rows, reference, annotations),
                    }
                report["clips"].append(clip)
        finally:
            for tracker in trackers.values():
                tracker.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="*", type=Path,
                        default=[ROOT / "tests/data/audio/sample.wav",
                                 ROOT / "tests/data/audio/sample2.wav"])
    parser.add_argument("--annotation-dir", type=Path,
                        default=ROOT / "tests/data/annotations")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=441)
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS),
                        default=list(VARIANTS))
    parser.add_argument("--output", type=Path,
                        help="Write detailed JSON; without this option write JSON to stdout")
    args = parser.parse_args()
    if args.repeats < 1 or args.chunk_size < 1:
        parser.error("--repeats and --chunk-size must be positive")
    report = benchmark(args)
    document = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(document)
        print("audio | variant | CPU s/audio s | activation max error | F1 at 70 ms")
        for clip in report["clips"]:
            for name, result in clip["variants"].items():
                score = result["accuracy"]
                fmeasure = "n/a" if score is None else "%.4f" % score["fmeasure"]
                print("%s | %s | %.6f | %.8g | %s" % (
                    clip["audio"], name, result["cpu_seconds_per_audio_second"],
                    result["activation_max_abs_error"], fmeasure,
                ))
    else:
        sys.stdout.write(document)


if __name__ == "__main__":
    main()
