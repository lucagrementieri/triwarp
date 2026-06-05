"""Probe: face/vertex gathering via Warp reshape + indexing (no custom kernels)."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp


def _gather_faces_slicing(src_faces: wp.array[wp.int32], face_indices: wp.array[wp.int32]) -> wp.array[wp.int32]:
    k = int(face_indices.shape[0])
    if k == 0:
        return wp.empty(0, dtype=wp.int32, device=src_faces.device)
    gathered = src_faces.reshape((-1, 3))[face_indices]
    out = wp.empty((k, 3), dtype=wp.int32, device=src_faces.device)
    wp.copy(out, gathered)
    return out.reshape((-1,))


def _gather_vertices_slicing(vertices: wp.array[wp.vec3], indices: wp.array[wp.int32]) -> wp.array[wp.vec3]:
    return vertices[indices]


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda:0", marks=pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA not available")),
    ],
)
def test_gather_faces_slicing_matches_numpy(device: str) -> None:
    rng = np.random.default_rng(0)
    n_faces = 20
    faces_np = rng.integers(0, 50, size=n_faces * 3, dtype=np.int32)
    indices_np = rng.integers(0, n_faces, size=8, dtype=np.int32)

    faces_wp = wp.array(faces_np, dtype=wp.int32, device=device)
    indices_wp = wp.array(indices_np, dtype=wp.int32, device=device)

    exp = faces_np.reshape(-1, 3)[indices_np].reshape(-1)
    got = _gather_faces_slicing(faces_wp, indices_wp).numpy()
    assert np.array_equal(got, exp)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda:0", marks=pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA not available")),
    ],
)
def test_gather_vertices_slicing_matches_numpy(device: str) -> None:
    rng = np.random.default_rng(1)
    n_vertices = 30
    vertices_np = rng.random((n_vertices, 3), dtype=np.float32)
    indices_np = rng.integers(0, n_vertices, size=12, dtype=np.int32)

    vertices_wp = wp.array(vertices_np, dtype=wp.vec3, device=device)
    indices_wp = wp.array(indices_np, dtype=wp.int32, device=device)

    exp = vertices_np[indices_np]
    got = _gather_vertices_slicing(vertices_wp, indices_wp).numpy()
    assert np.allclose(got, exp)
