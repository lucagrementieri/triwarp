from __future__ import annotations

import numpy as np
import warp as wp

import triwarp as tw


def _flipped_faces_np(vertices_np: np.ndarray, faces_np: np.ndarray) -> np.ndarray:
    """NumPy reference for libigl ``flipped_triangles``: 2D signed area strictly negative."""
    tri = vertices_np[faces_np]  # (n_faces, 3, 2)
    e0 = tri[:, 1] - tri[:, 0]
    e1 = tri[:, 2] - tri[:, 0]
    signed_area2 = e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0]
    return np.flatnonzero(signed_area2 < 0.0).astype(np.int64)


def _random_2d_mesh(rng: np.random.Generator, n_faces: int):
    """Random 2D triangle soup with mixed orientations as (vertices_2d, flat_faces)."""
    vertices_np = rng.standard_normal((n_faces * 3, 2)).astype(np.float64)
    faces_np = np.arange(n_faces * 3, dtype=np.int64).reshape(n_faces, 3)
    return vertices_np, faces_np


def _to_wp(vertices_np: np.ndarray, faces_np: np.ndarray, device: str):
    vertices_wp = wp.array(
        np.ascontiguousarray(vertices_np, dtype=np.float32), dtype=wp.vec2, device=device
    )
    faces_wp = wp.array(
        np.ascontiguousarray(faces_np.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device
    )
    return vertices_wp, faces_wp


def test_flipped_faces_random_mixed(device):
    rng = np.random.default_rng(0)
    vertices_np, faces_np = _random_2d_mesh(rng, n_faces=64)
    vertices_wp, faces_wp = _to_wp(vertices_np, faces_np, device)

    tri = vertices_np[faces_np]
    e0 = tri[:, 1] - tri[:, 0]
    e1 = tri[:, 2] - tri[:, 0]
    mask_np = (e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0]) < 0.0

    assert np.array_equal(
        tw.parametrization.flipped_faces_mask(vertices_wp, faces_wp).numpy(), mask_np
    )
    assert np.array_equal(
        tw.parametrization.flipped_faces(vertices_wp, faces_wp).numpy(),
        _flipped_faces_np(vertices_np, faces_np),
    )


def test_flipped_faces_mask_index_consistency(device):
    rng = np.random.default_rng(1)
    vertices_np, faces_np = _random_2d_mesh(rng, n_faces=32)
    vertices_wp, faces_wp = _to_wp(vertices_np, faces_np, device)

    mask_wp = tw.parametrization.flipped_faces_mask(vertices_wp, faces_wp)
    assert np.array_equal(
        tw.parametrization.flipped_faces(vertices_wp, faces_wp).numpy(),
        np.flatnonzero(mask_wp.numpy()),
    )


def test_flipped_faces_all_and_none(device):
    # A single CCW (positive-area) triangle: not flipped.
    ccw_np = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    faces_np = np.array([[0, 1, 2]], dtype=np.int64)
    vertices_wp, faces_wp = _to_wp(ccw_np, faces_np, device)
    assert tw.parametrization.flipped_faces(vertices_wp, faces_wp).numpy().size == 0
    assert not tw.parametrization.flipped_faces_mask(vertices_wp, faces_wp).numpy().any()

    # Reverse the winding of every triangle: all flipped.
    cw_faces_np = faces_np[:, ::-1].copy()
    vertices_wp, cw_faces_wp = _to_wp(ccw_np, cw_faces_np, device)
    assert np.array_equal(
        tw.parametrization.flipped_faces(vertices_wp, cw_faces_wp).numpy(),
        np.arange(cw_faces_np.shape[0], dtype=np.int64),
    )
    assert tw.parametrization.flipped_faces_mask(vertices_wp, cw_faces_wp).numpy().all()


def test_flipped_faces_degenerate_not_flagged(device):
    # Collinear (zero-area) triangle: strict "< 0" means it is not flagged.
    collinear_np = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    faces_np = np.array([[0, 1, 2]], dtype=np.int64)
    vertices_wp, faces_wp = _to_wp(collinear_np, faces_np, device)
    assert not tw.parametrization.flipped_faces_mask(vertices_wp, faces_wp).numpy().any()
    assert tw.parametrization.flipped_faces(vertices_wp, faces_wp).numpy().size == 0


def test_flipped_faces_empty_mesh(device):
    vertices_wp = wp.empty(0, dtype=wp.vec2, device=device)
    faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    assert tw.parametrization.flipped_faces_mask(vertices_wp, faces_wp).numpy().size == 0
    assert tw.parametrization.flipped_faces(vertices_wp, faces_wp).numpy().size == 0
