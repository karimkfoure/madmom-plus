#!/usr/bin/env python3
"""Freeze outputs from the unmodified madmom 0.16.1 numerical source.

Use Python 3.11, NumPy 1.26.4, SciPy 1.13.1, Cython 0.29.37,
setuptools 75.8.2, and mido 1.3.3. Git objects for the source and models must
already exist locally. Generation does not download files or import the fork.
"""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile


SOURCE_COMMIT = "6545fbb3ecdae0aac336a95ec591f642e0b74f3e"
MODEL_COMMIT = "31fe6ce52affb794c2e3511908fbeccfd68f6925"
ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "madmom/audio/signal.py", "madmom/audio/stft.py",
    "madmom/audio/filters.py", "madmom/audio/spectrogram.py",
    "madmom/features/beats.py", "madmom/features/beats_hmm.py",
    "madmom/features/onsets.py", "madmom/ml/hmm.pyx",
    "madmom/ml/nn/layers.py", "madmom/ml/nn/activations.py",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archive(repository, revision, destination):
    result = subprocess.run(
        ["git", "-C", str(repository), "archive", revision],
        check=True, capture_output=True,
    )
    with tarfile.open(fileobj=io.BytesIO(result.stdout)) as stream:
        # Archives come from the pinned local Git objects, not user input.
        for member in stream.getmembers():
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe Git archive member")
        stream.extractall(destination)


def generate(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="reference-0161-", dir=build) as folder:
        reference = Path(folder)
        archive(args.source_repository, SOURCE_COMMIT, reference)
        archive(args.model_repository, MODEL_COMMIT, reference / "madmom/models")
        # Package metadata lets the original __init__.py query its version.
        metadata = reference / "madmom-0.16.1.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: madmom\nVersion: 0.16.1\n",
            encoding="utf-8",
        )
        setup = reference / "_reference_setup.py"
        setup.write_text(
            "from setuptools import setup, Extension\n"
            "from Cython.Build import cythonize\n"
            "import numpy as np\n"
            "modules = [\n"
            " ('madmom.audio.comb_filters', 'madmom/audio/comb_filters.pyx'),\n"
            " ('madmom.features.beats_crf', 'madmom/features/beats_crf.pyx'),\n"
            " ('madmom.ml.hmm', 'madmom/ml/hmm.pyx'),\n"
            " ('madmom.ml.nn.layers', 'madmom/ml/nn/layers.py')]\n"
            "setup(name='madmom-reference', ext_modules=cythonize([\n"
            " Extension(name, [path], include_dirs=[np.get_include()])\n"
            " for name, path in modules], compiler_directives={'language_level': 2}))\n",
            encoding="utf-8",
        )
        env = os.environ.copy()
        env.update({"PYTHONPATH": str(reference), "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        subprocess.run(
            [sys.executable, str(setup), "build_ext", "--inplace", "--parallel", "2"],
            cwd=reference, env=env, check=True,
        )
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker",
             str(reference), "--output", str(output)],
            cwd=reference, env=env, check=True,
        )


def worker(reference, output):
    # Restore removed API names. These aliases do not alter numerical kernels.
    import collections
    import collections.abc
    import numpy as np
    import scipy
    import Cython

    collections.MutableSequence = collections.abc.MutableSequence
    for name, value in (("float", float), ("int", int), ("complex", complex),
                        ("bool", bool)):
        setattr(np, name, value)

    import madmom
    from madmom.audio.signal import FramedSignal, Signal
    from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor
    from madmom.features.onsets import OnsetPeakPickingProcessor, peak_picking
    from madmom.models import BEATS_LSTM

    if not Path(madmom.__file__).resolve().is_relative_to(reference.resolve()):
        raise RuntimeError("Reference import resolved outside the upstream archive")
    if madmom.__version__ != "0.16.1":
        raise RuntimeError("Wrong upstream version")

    metadata = {
        "schema_version": 1,
        "source": {"url": "https://github.com/CPJKU/madmom",
                   "author": "Sebastian Böck and the madmom contributors",
                   "tag": "v0.16.1", "commit": SOURCE_COMMIT,
                   "sha256": {name: digest(reference / name) for name in SOURCE_FILES}},
        "models": {"url": "https://github.com/CPJKU/madmom_models",
                   "author": "Sebastian Böck and contributors", "commit": MODEL_COMMIT,
                   "licence": "CC BY-NC-SA 4.0",
                   "sha256": {str(Path(name).relative_to(reference)): digest(name)
                              for name in BEATS_LSTM}},
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "scipy": scipy.__version__, "cython": Cython.__version__,
                        "blas_threads": 1},
        "compatibility": [
            "Runtime alias collections.MutableSequence to collections.abc.MutableSequence.",
            "Runtime aliases np.float, np.int, np.complex, and np.bool to the original Python builtin types.",
            "Add distribution metadata for madmom 0.16.1; do not modify upstream source.",
            "Compile the four original extensions with Cython language_level=2.",
        ],
        "settings": {"sample_rate": 44100, "fps": 100, "frame_size": 2048,
                     "hop_size": 441, "origin": "online", "networks": 8},
        "tolerances": {"features": {"atol": 2e-6, "rtol": 2e-5},
                       "activations": {"atol": 2e-5, "rtol": 2e-4},
                       "forward": {"atol": 1e-9, "rtol": 2e-6},
                       "beat_seconds": {"atol": 0.01, "rtol": 0},
                       "peak_seconds": {"atol": 0, "rtol": 0}},
        "clips": [],
    }
    for filename in ("sample.wav", "sample2.wav"):
        audio_path = reference / "tests/data/audio" / filename
        signal = Signal(str(audio_path), sample_rate=44100, num_channels=1,
                        dtype=np.float32)
        rnn = RNNBeatProcessor(online=True, origin="online", fps=100, num_threads=1)
        features = rnn[0](signal, reset=True)
        activations = rnn[1](features, reset=True)
        individual = np.vstack([net(features, reset=True) for net in rnn[1][0]])
        dbn = DBNBeatTrackingProcessor(online=True, fps=100)
        forward = dbn.hmm.forward(activations, reset=True)
        selected = np.unique(np.r_[np.arange(0, len(activations), 37),
                                   len(activations) - 1]).astype(np.int64)
        beats = dbn(activations, reset=True)
        centered = RNNBeatProcessor(online=True, fps=100, num_threads=1)(signal)
        picker = OnsetPeakPickingProcessor(online=True, fps=100, threshold=.2,
                                           pre_avg=.1, pre_max=.03, combine=.05)
        peaks = picker(activations, reset=True)
        # The upstream streaming buffer has observable batch-dependent output.
        # Freeze that behavior for a fixed chunk size as a separate contract.
        streamed_peaks = []
        for first in range(0, len(activations), 13):
            streamed_peaks.extend(picker(activations[first:first + 13],
                                         reset=first == 0))
        frames = FramedSignal(signal, frame_size=2048, hop_size=441, origin="online")
        chunked = RNNBeatProcessor(online=True, origin="stream", fps=100,
                                  num_frames=1, num_threads=1)
        streamed_activations = np.array(
            [chunked(frame, reset=index == 0)[0] for index, frame in enumerate(frames)],
            dtype=np.float32,
        )
        path = output / (Path(filename).stem + ".npz")
        np.savez_compressed(
            path, features=features, activations=activations,
            individual_activations=individual, centered_activations=centered,
            streamed_activations=streamed_activations,
            forward_indices=selected, forward=forward[selected],
            forward_argmax=np.argmax(forward, axis=1), forward_max=np.max(forward, axis=1),
            beats=beats, peaks=peaks, streamed_peaks=np.asarray(streamed_peaks),
        )
        metadata["clips"].append({"audio": filename, "audio_sha256": digest(audio_path),
                                  "samples": len(signal), "frames": len(activations),
                                  "fixture": path.name, "fixture_sha256": digest(path)})

    # Binary thresholds, equal maxima, zeros, and chunk boundaries need stable cases.
    synthetic = np.array([0, .5, .5, .1, .9, .9, 0, .01, 1, 0, .3, .7, 0,
                          .5, 0, 0, .8, .8, .8, 0, .2, .6, .1], dtype=np.float32)
    arrays = {"activations": synthetic}
    for pre_avg in (0, 3, 8):
        for pre_max in (0, 1, 4):
            arrays["peaks_avg%d_max%d" % (pre_avg, pre_max)] = peak_picking(
                synthetic, .2, 0, pre_avg, 0, pre_max, 0)
    path = output / "synthetic.npz"
    np.savez_compressed(path, **arrays)
    metadata["synthetic"] = {"fixture": path.name, "fixture_sha256": digest(path),
                             "threshold": .2, "smooth": 0, "post_avg": 0, "post_max": 0}
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repository", type=Path, default=ROOT)
    parser.add_argument("--model-repository", type=Path, default=ROOT / "madmom/models")
    parser.add_argument("--output", type=Path, default=ROOT / "tests/data/native_0161")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.worker:
        worker(arguments.worker, arguments.output)
    else:
        generate(arguments)
