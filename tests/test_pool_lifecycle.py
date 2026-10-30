"""Downbeat decoding must release owned pools and preserve borrowed pools."""

import io
import multiprocessing as mp

import numpy as np
import pytest

from madmom.features import downbeats
from madmom.features.downbeats import DBNDownBeatTrackingProcessor


def processor(**kwargs):
    return DBNDownBeatTrackingProcessor(
        [3, 4], min_bpm=90, max_bpm=120, num_tempi=3, fps=10, **kwargs)


def activations():
    data = np.full((30, 2), 0.01, dtype=np.float32)
    data[::5, 0] = 0.8
    data[::15, 0] = 0.01
    data[::15, 1] = 0.9
    return data


def track_pools(monkeypatch):
    create = mp.Pool
    workers = []

    def tracked(*args, **kwargs):
        pool = create(*args, **kwargs)
        workers.extend(pool._pool)
        return pool

    monkeypatch.setattr(mp, 'Pool', tracked)
    return workers


def test_repeated_parallel_calls_release_workers(monkeypatch):
    workers = track_pools(monkeypatch)
    reference = processor()(activations())
    for _ in range(2):
        parallel = processor(num_threads=2)
        assert all(not worker.is_alive() for worker in workers)
        for _ in range(2):
            np.testing.assert_array_equal(parallel(activations()), reference)
            assert all(not worker.is_alive() for worker in workers)
            assert all(worker.exitcode is not None for worker in workers)
    assert len(workers) == 8


def raise_decode_error(_):
    raise RuntimeError('decoding failed')


def test_parallel_exception_releases_workers(monkeypatch):
    workers = track_pools(monkeypatch)
    monkeypatch.setattr(downbeats, '_process_dbn', raise_decode_error)
    with pytest.raises(RuntimeError, match='decoding failed'):
        processor(num_threads=2)(activations())
    assert len(workers) == 2
    assert all(not worker.is_alive() for worker in workers)
    assert all(worker.exitcode is not None for worker in workers)


def test_borrowed_pool_remains_usable():
    reference = processor()(activations())
    with mp.Pool(2) as pool:
        for _ in range(2):
            borrowed = processor(num_threads=2, pool=pool)
            np.testing.assert_array_equal(borrowed(activations()), reference)
        assert pool.map(abs, [-2, 3]) == [2, 3]


def test_parallel_processor_pickle_retains_worker_count():
    original = processor(num_threads=2)
    data = io.BytesIO()
    original.dump(data)
    data.seek(0)
    restored = DBNDownBeatTrackingProcessor.load(data)
    assert restored.num_threads == 2
    np.testing.assert_array_equal(restored(activations()),
                                  processor()(activations()))


def test_legacy_serial_state_has_no_worker_count():
    legacy = processor()
    del legacy.num_threads
    np.testing.assert_array_equal(legacy(activations()),
                                  processor()(activations()))


def test_empty_input_does_not_create_workers(monkeypatch):
    workers = track_pools(monkeypatch)
    np.testing.assert_array_equal(processor(num_threads=2)(np.empty((0, 2))),
                                  np.empty((0, 2)))
    assert workers == []
