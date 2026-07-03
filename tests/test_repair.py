"""Regression tests for ``triwarp.repair`` against libigl."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

import igl
import triwarp as tw


def _to_wp_mesh(vertices_np: np.ndarray, faces_np: np.ndarray, device: str):
    vertices_wp = wp.array(
        np.ascontiguousarray(vertices_np.astype(np.float32)), dtype=wp.vec3, device=device
    )
    faces_wp = wp.array(
        np.ascontiguousarray(faces_np.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device
    )
    return vertices_wp, faces_wp


def _sort_rows(rows: np.ndarray) -> np.ndarray:
    if rows.size == 0:
        return rows
    return rows[np.lexsort(rows.T[::-1])]


def _resolve_duplicated_faces_ref(faces_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CPU reference mirroring ``igl::resolve_duplicated_faces``."""
    faces = np.asarray(faces_np, dtype=np.int32).reshape((-1, 3))
    n_faces = faces.shape[0]
    if n_faces == 0:
        return faces.copy(), np.empty(0, dtype=np.int32)

    sorted_faces = np.sort(faces, axis=1)
    unique_sorted, inverse = np.unique(sorted_faces, axis=0, return_inverse=True)
    num_unique = unique_sorted.shape[0]

    kept: list[int] = []
    for ui in range(num_unique):
        member = np.flatnonzero(inverse == ui)
        urow = unique_sorted[ui]
        signed_ids: list[int] = []
        count = 0
        for fi in member:
            row = faces[fi]
            consistent = (
                (row[0] == urow[0] and row[1] == urow[1] and row[2] == urow[2])
                or (row[0] == urow[1] and row[1] == urow[2] and row[2] == urow[0])
                or (row[0] == urow[2] and row[1] == urow[0] and row[2] == urow[1])
            )
            signed = int(fi + 1) if consistent else -int(fi + 1)
            signed_ids.append(signed)
            count += 1 if consistent else -1

        if member.size == 1:
            kept.append(int(member[0]))
            continue
        if count == 1:
            for fid in signed_ids:
                if fid > 0:
                    kept.append(fid - 1)
                    break
        elif count == -1:
            for fid in signed_ids:
                if fid < 0:
                    kept.append(-fid - 1)
                    break
        elif count == 0:
            continue
        else:
            raise ValueError(f"non-orientable duplicate face group {ui} with count {count}")

    if len(kept) == 0:
        return np.empty((0, 3), dtype=np.int32), np.empty(0, dtype=np.int32)
    kept_np = np.asarray(kept, dtype=np.int32)
    return faces[kept_np], kept_np


def _assert_duplicate_vertices_match(
    vertices_np: np.ndarray,
    sv_wp: np.ndarray,
    svj_wp: np.ndarray,
    sf_wp: np.ndarray | None = None,
    faces_np: np.ndarray | None = None,
    epsilon: float = 0.0,
) -> None:
    sv_igl, _, svj_igl, sf_igl = (
        igl.remove_duplicate_vertices(vertices_np, faces_np, epsilon)
        if faces_np is not None
        else (*igl.remove_duplicate_vertices(vertices_np, epsilon), None)
    )
    assert np.array_equal(_sort_rows(sv_wp), _sort_rows(sv_igl))
    for i, vertex in enumerate(vertices_np):
        assert np.allclose(sv_wp[svj_wp[i]], vertex, rtol=1e-5, atol=1e-5)
    if sf_wp is not None and faces_np is not None and sf_igl is not None:
        tri_wp = sv_wp[sf_wp.reshape(-1, 3)]
        tri_igl = sv_igl[sf_igl]
        assert np.allclose(np.sort(tri_wp, axis=1), np.sort(tri_igl, axis=1), rtol=1e-5, atol=1e-5)


def test_remove_unreferenced_identity(icosahedron, device: str):
    mesh_tm, mesh_wp = icosahedron
    vertices_np = mesh_tm.vertices
    faces_np = mesh_tm.faces

    nv_wp, nf_wp, remap_wp, inverse_wp = tw.repair.remove_unreferenced_vertices(
        mesh_wp.points, mesh_wp.indices, return_inverse=True
    )

    nv_igl, nf_igl, remap_igl, inverse_igl = igl.remove_unreferenced(
        np.asarray(vertices_np, dtype=np.float64),
        np.asarray(faces_np, dtype=np.int32),
    )

    assert np.allclose(nv_wp.numpy(), nv_igl, rtol=1e-5, atol=1e-5)
    assert np.array_equal(nf_wp.numpy().reshape(-1, 3), nf_igl)
    assert np.array_equal(remap_wp.numpy(), remap_igl.ravel())
    assert np.array_equal(inverse_wp.numpy(), inverse_igl.ravel())


def test_remove_unreferenced_extra_vertices(device: str):
    rng = np.random.default_rng(0)
    vertices_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    faces_np = np.array([[0, 1, 2]], dtype=np.int32)
    extra = rng.normal(size=(4, 3))
    vertices_full_np = np.vstack([vertices_np, extra])

    vertices_wp, faces_wp = _to_wp_mesh(vertices_full_np, faces_np, device)
    nv_wp, nf_wp, remap_wp, inverse_wp = tw.repair.remove_unreferenced_vertices(
        vertices_wp, faces_wp, return_inverse=True
    )

    nv_igl, nf_igl, remap_igl, inverse_igl = igl.remove_unreferenced(
        vertices_full_np, faces_np
    )

    assert np.allclose(nv_wp.numpy(), nv_igl, rtol=1e-5, atol=1e-5)
    assert np.array_equal(nf_wp.numpy().reshape(-1, 3), nf_igl)
    assert np.array_equal(remap_wp.numpy(), remap_igl.ravel())
    assert np.array_equal(inverse_wp.numpy(), inverse_igl.ravel())


def test_remove_unreferenced_sentinel(device: str):
    vertices_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    faces_np = np.array([[0, 1, -1], [0, 2, 1]], dtype=np.int32)
    vertices_wp, faces_wp = _to_wp_mesh(vertices_np, faces_np, device)

    nv_wp, nf_wp, remap_wp = tw.repair.remove_unreferenced_vertices(vertices_wp, faces_wp)

    assert np.allclose(nv_wp.numpy(), vertices_np[[0, 1, 2]], rtol=1e-5, atol=1e-5)
    assert np.array_equal(nf_wp.numpy().reshape(-1, 3), faces_np)
    expected_remap = np.array([0, 1, 2], dtype=np.int32)
    assert np.array_equal(remap_wp.numpy(), expected_remap)


def test_remove_duplicate_vertices_exact(device: str):
    vertices_np = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    vertices_wp = wp.array(
        np.ascontiguousarray(vertices_np.astype(np.float32)), dtype=wp.vec3, device=device
    )

    faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    sv_wp, _, svj_wp, _ = tw.repair.remove_duplicated_vertices(vertices_wp, faces_wp, epsilon=0.0)
    _assert_duplicate_vertices_match(vertices_np, sv_wp.numpy(), svj_wp.numpy())


def test_remove_duplicate_vertices_epsilon(device: str):
    vertices_np = np.array(
        [
            [0.0, 0.0, 0.0],
            [1e-9, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1e-9, 0.0],
        ],
        dtype=np.float64,
    )
    epsilon = 1e-8
    vertices_wp = wp.array(
        np.ascontiguousarray(vertices_np.astype(np.float32)), dtype=wp.vec3, device=device
    )

    faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    sv_wp, _, svj_wp, _ = tw.repair.remove_duplicated_vertices(vertices_wp, faces_wp, epsilon=epsilon)
    _assert_duplicate_vertices_match(vertices_np, sv_wp.numpy(), svj_wp.numpy(), epsilon=epsilon)


def test_remove_duplicate_vertices_faces(device: str):
    vertices_np = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    faces_np = np.array([[0, 1, 3], [2, 1, 3]], dtype=np.int32)
    vertices_wp, faces_wp = _to_wp_mesh(vertices_np, faces_np, device)

    sv_wp, _, svj_wp, sf_wp = tw.repair.remove_duplicated_vertices(vertices_wp, faces_wp, 0.0)
    _assert_duplicate_vertices_match(
        vertices_np,
        sv_wp.numpy(),
        svj_wp.numpy(),
        sf_wp.numpy(),
        faces_np,
    )


def test_resolve_duplicated_faces_cancelling(device: str):
    faces_np = np.array(
        [
            [0, 1, 2],
            [0, 1, 2],
            [0, 2, 1],
            [0, 2, 1],
        ],
        dtype=np.int32,
    )
    faces_wp = wp.array(
        np.ascontiguousarray(faces_np.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device
    )

    f2_wp, j_wp = tw.repair.resolve_duplicated_faces(faces_wp)
    f2_ref, j_ref = _resolve_duplicated_faces_ref(faces_np)

    assert np.array_equal(f2_wp.numpy().reshape(-1, 3), f2_ref)
    assert np.array_equal(j_wp.numpy(), j_ref)


def test_resolve_duplicated_faces_keep_positive(device: str):
    faces_np = np.array(
        [
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 2, 1],
            [0, 2, 1],
        ],
        dtype=np.int32,
    )
    faces_wp = wp.array(
        np.ascontiguousarray(faces_np.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device
    )

    f2_wp, j_wp = tw.repair.resolve_duplicated_faces(faces_wp)
    f2_ref, j_ref = _resolve_duplicated_faces_ref(faces_np)

    assert np.array_equal(f2_wp.numpy().reshape(-1, 3), f2_ref)
    assert np.array_equal(j_wp.numpy(), j_ref)
