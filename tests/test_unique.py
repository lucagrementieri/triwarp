from __future__ import annotations

import numpy as np
import warp as wp

import triwarp as tw


def test_unique_1d(device: str):
    data_np = np.array([0, 1, 20, 3, 1, 3, 10, 20], dtype=np.int32)
    unique_np = np.unique(data_np)

    data_wp = wp.array(data_np, dtype=wp.int32, device=device)
    unique_wp = tw.unique.unique_1d(data_wp)
    assert np.array_equal(unique_wp.numpy(), unique_np)


def test_unique_1d_counts(device: str):
    data_np = np.array([20.0, 10.0, 2.0, 3.0, 1.0, 3.0, 10.0, 20.0], dtype=np.float32)
    unique_np, counts_np = np.unique(data_np, return_counts=True)

    data_wp = wp.array(data_np, dtype=wp.float32, device=device)
    unique_wp, counts_wp = tw.unique.unique_1d(data_wp, return_counts=True)
    assert np.array_equal(unique_wp.numpy(), unique_np)
    assert np.array_equal(counts_wp.numpy(), counts_np)


def test_unique_1d_inverse(device: str):
    data_np = np.array([20, 10, 2, 3, 1, 3, 10, 20], dtype=np.int32)
    unique_np, inverse_np = np.unique(data_np, return_inverse=True)

    data_wp = wp.array(data_np, dtype=wp.int32, device=device)
    unique_wp, inverse_wp = tw.unique.unique_1d(data_wp, return_inverse=True)
    assert np.array_equal(unique_wp.numpy(), unique_np)
    assert np.array_equal(inverse_wp.numpy(), inverse_np)


def test_unique_1d_inverse_counts(device: str):
    data_np = np.array([20, 10, 20, 3, 1, 3, 10, 20], dtype=np.int32)
    unique_np, inverse_np, counts_np = np.unique(data_np, return_inverse=True, return_counts=True)

    data_wp = wp.array(data_np, dtype=wp.int32, device=device)
    unique_wp, inverse_wp, counts_wp = tw.unique.unique_1d(data_wp, return_inverse=True, return_counts=True)
    assert np.array_equal(unique_wp.numpy(), unique_np)
    assert np.array_equal(inverse_wp.numpy(), inverse_np)
    assert np.array_equal(counts_wp.numpy(), counts_np)
