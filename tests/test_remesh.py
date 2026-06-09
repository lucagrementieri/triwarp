"""Regression tests for ``triwarp.remesh`` against Trimesh (CPU reference)."""

from __future__ import annotations

import numpy as np
import trimesh as tm
import warp as wp

import triwarp as tw


def test_subdivide(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = icosahedron

    vertices_np = mesh_tm.vertices.astype(np.float32)
    faces_np = mesh_tm.faces.astype(np.int32)

    new_v_tm, new_f_tm = tm.remesh.subdivide(vertices_np.astype(np.float64), faces_np)
    new_v_tm = new_v_tm.astype(np.float32)

    new_v_wp, new_f_wp = tw.remesh.subdivide(mesh_wp.points, mesh_wp.indices)

    new_v_wp_np = new_v_wp.numpy()
    new_f_wp_np = new_f_wp.numpy().reshape(-1, 3)
    new_f_tm_np = new_f_tm.reshape(-1, 3)

    assert new_v_wp_np.shape[0] == new_v_tm.shape[0], (
        f"vertex count mismatch: got {new_v_wp_np.shape[0]}, expected {new_v_tm.shape[0]}"
    )
    assert new_f_wp_np.shape[0] == new_f_tm_np.shape[0], (
        f"face count mismatch: got {new_f_wp_np.shape[0]}, expected {new_f_tm_np.shape[0]}"
    )

    centroids_wp = new_v_wp_np[new_f_wp_np].mean(axis=1)
    centroids_tm = new_v_tm[new_f_tm_np].mean(axis=1)
    order_wp = np.lexsort(centroids_wp.T[::-1])
    order_tm = np.lexsort(centroids_tm.T[::-1])
    assert np.allclose(centroids_wp[order_wp], centroids_tm[order_tm], rtol=1e-5, atol=1e-5), (
        "face centroid sets do not match"
    )


def test_subdivide_empty(device: str) -> None:
    vertices_wp = wp.empty(0, dtype=wp.vec3, device=device)
    faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    new_v_wp, new_f_wp = tw.remesh.subdivide(vertices_wp, faces_wp)
    assert int(new_v_wp.shape[0]) == 0
    assert int(new_f_wp.shape[0]) == 0


def test_subdivide_edge_lengths(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """All edges in the subdivided mesh are at most half the longest original edge."""
    mesh_tm, mesh_wp = icosahedron

    vertices_np = mesh_tm.vertices.astype(np.float32)
    faces_np = mesh_tm.faces.astype(np.int32)

    new_v_wp, new_f_wp = tw.remesh.subdivide(mesh_wp.points, mesh_wp.indices)

    new_v_np = new_v_wp.numpy()
    new_f_np = new_f_wp.numpy().reshape(-1, 3)

    orig_edges = vertices_np[faces_np[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)]
    orig_max_edge = np.linalg.norm(orig_edges[:, 0] - orig_edges[:, 1], axis=1).max()

    new_edges = new_v_np[new_f_np[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)]
    new_max_edge = np.linalg.norm(new_edges[:, 0] - new_edges[:, 1], axis=1).max()

    assert new_max_edge <= orig_max_edge / 2.0 + 1e-5
