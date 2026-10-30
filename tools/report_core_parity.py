#!/usr/bin/env python3
"""Report measured errors against the stored madmom 0.16.1 fixtures."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, default=ROOT)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.package_root.resolve()))
    import numpy as np
    import scipy
    from madmom import _native
    from madmom.audio.signal import Signal
    from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
    from madmom.native import NativeBeatTracker

    report = {
        'schema_version': 1, 'runtime_revision': args.revision,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'scipy': scipy.__version__, 'architecture': platform.machine()},
        'native_binary_sha256': hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest(),
        'fixtures': [],
    }
    for name in ('sample', 'sample2'):
        path = ROOT / 'tests/data/native_0161' / (name + '.npz')
        with np.load(path) as reference:
            audio = Signal(str(ROOT / 'tests/data/audio' / (name + '.wav')),
                           sample_rate=44100, num_channels=1, dtype=np.float32)
            processor = RNNBeatProcessor(online=True, origin='online', fps=100,
                                         num_threads=1)
            features = processor[0](audio)
            activations = processor[1](features)
            decoder = DBNBeatTrackingProcessor(online=True, fps=100)
            beats = decoder(activations)
            posterior = decoder.hmm.forward(reference['activations'])
            sampled = posterior[reference['forward_indices']]
            np.testing.assert_allclose(features, reference['features'], atol=2e-6, rtol=2e-5)
            np.testing.assert_allclose(activations, reference['activations'], atol=2e-5, rtol=2e-4)
            np.testing.assert_allclose(sampled, reference['forward'], atol=1e-9, rtol=2e-6)
            np.testing.assert_array_equal(beats, reference['beats'])
            row = {
                'fixture': name, 'fixture_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'frames': len(activations),
                'feature_max_abs_error': float(np.max(np.abs(features - reference['features']))),
                'python_activation_max_abs_error': float(np.max(np.abs(activations - reference['activations']))),
                'sampled_posterior_max_abs_error': float(np.max(np.abs(sampled - reference['forward']))),
                'python_beats_exact': True, 'native': {},
            }
            for simd in (False, True):
                with NativeBeatTracker(simd=simd) as tracker:
                    rows = tracker.push(audio)
                error = float(np.max(np.abs(rows[:, 1] - reference['activations'])))
                np.testing.assert_allclose(rows[:, 1], reference['activations'], atol=2e-5, rtol=2e-4)
                native_beats = rows[rows[:, 2] != 0, 0]
                np.testing.assert_array_equal(native_beats, reference['beats'])
                row['native']['simd' if simd else 'scalar'] = {
                    'activation_max_abs_error': error, 'beats_exact': True,
                }
            report['fixtures'].append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
