from __future__ import annotations

import numpy as np
import warp as wp
import pytest

import triwarp as tw
from triwarp.kernels.grouping import VEC3_PACK_SHIFT, VEC3_PACK_PRECISION


def _pack_vec3_np(vectors_np: np.ndarray) -> np.ndarray:
    if vectors_np.dtype != np.float32:
        vectors_np = vectors_np.astype(np.float32)
    bits = vectors_np.view(np.uint32)
    ix = bits[:, 0].astype(np.uint64) >> VEC3_PACK_SHIFT.value
    iy = bits[:, 1].astype(np.uint64) >> VEC3_PACK_SHIFT.value
    iz = bits[:, 2].astype(np.uint64) >> VEC3_PACK_SHIFT.value
    return ix | (iy << VEC3_PACK_PRECISION.value) | (iz << (2 * VEC3_PACK_PRECISION.value))


def test_hash_vector_rows(device: str) -> None:
    rng = np.random.default_rng(17)
    n = 256
    vectors_np = rng.standard_normal((n, 3), dtype=np.float64)
    packed_np = _pack_vec3_np(vectors_np)

    vectors_wp = wp.array(vectors_np, dtype=wp.vec3, device=device)
    packed_wp = tw.grouping.hash_vector_rows(vectors_wp)
    packed = packed_wp.numpy()

    assert np.array_equal(packed, packed_np)

    vectors_double_wp = wp.array(vectors_np, dtype=wp.vec3d, device=device)
    with pytest.raises(ValueError, match="data must be a wp.array\\[wp.vec3\\]"):
        _ = tw.grouping.hash_vector_rows(vectors_double_wp)
