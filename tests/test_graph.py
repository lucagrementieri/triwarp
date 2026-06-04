"""Regression tests for ``triwarp.graph`` against Trimesh (CPU reference)."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

import trimesh as tm
import triwarp as tw


def test_faces_to_edges(device: str) -> None:
    rng = np.random.default_rng(42)
    n_faces = 16
    faces_np = rng.integers(0, 100, size=(n_faces, 3), dtype=np.int32)
    edges_np = tm.geometry.faces_to_edges(faces_np)

    faces_wp = wp.array(faces_np.flatten(), dtype=wp.int32, device=device)
    edges_wp = tw.graph.faces_to_edges(faces_wp)
    assert np.array_equal(edges_wp.numpy(), edges_np)


def test_faces_to_edges_sorted(device: str) -> None:
    rng = np.random.default_rng(42)
    n_faces = 16
    faces_np = rng.integers(0, 100, size=(n_faces, 3), dtype=np.int32)
    edges_np = np.sort(tm.geometry.faces_to_edges(faces_np), axis=1)

    faces_wp = wp.array(faces_np.flatten(), dtype=wp.int32, device=device)
    edges_wp = tw.graph.faces_to_edges(faces_wp, sorted=True)
    assert np.array_equal(edges_wp.numpy(), edges_np)


def test_faces_to_edges_empty(device: str) -> None:
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    edges_wp = tw.graph.faces_to_edges(faces_wp)
    assert edges_wp.shape == (0, 2)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
def test_face_adjacency(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    adjacency_tm = mesh_tm.face_adjacency
    adjacency_edges_tm = mesh_tm.face_adjacency_edges
    adjacency_wp, adjacency_edges_wp = tw.graph.face_adjacency(mesh_wp.indices, return_edges=True)

    order_tm = np.lexsort((adjacency_tm[:, 1], adjacency_tm[:, 0]))
    order_wp = np.lexsort((adjacency_wp.numpy()[:, 1], adjacency_wp.numpy()[:, 0]))
    assert np.array_equal(adjacency_wp.numpy()[order_wp], adjacency_tm[order_tm])
    assert np.array_equal(adjacency_edges_wp.numpy()[order_wp], adjacency_edges_tm[order_tm])


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
def test_face_adjacency_unshared(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    adjacency_tm = mesh_tm.face_adjacency
    unshared_tm = mesh_tm.face_adjacency_unshared.astype(np.int32)

    adjacency_wp, adjacency_edges_wp = tw.graph.face_adjacency(mesh_wp.indices, return_edges=True)
    unshared_precomputed_wp = tw.graph.face_adjacency_unshared(
        mesh_wp.indices, face_adjacency=adjacency_wp, face_adjacency_edges=adjacency_edges_wp
    )
    adjacency_wp_np = adjacency_wp.numpy()
    order_tm = np.lexsort((adjacency_tm[:, 1], adjacency_tm[:, 0]))
    order_wp = np.lexsort((adjacency_wp_np[:, 1], adjacency_wp_np[:, 0]))

    assert np.array_equal(unshared_precomputed_wp.numpy()[order_wp], unshared_tm[order_tm])

    unshared_wp = tw.graph.face_adjacency_unshared(mesh_wp.indices)
    order_unshared = np.lexsort((unshared_wp.numpy()[:, 1], unshared_wp.numpy()[:, 0]))
    order_unshared_precomputed = np.lexsort(
        (unshared_precomputed_wp.numpy()[:, 1], unshared_precomputed_wp.numpy()[:, 0])
    )
    assert np.array_equal(
        unshared_wp.numpy()[order_unshared], unshared_precomputed_wp.numpy()[order_unshared_precomputed]
    )


def test_face_adjacency_unshared_empty(device: str) -> None:
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    unshared_wp = tw.graph.face_adjacency_unshared(faces_wp)
    assert unshared_wp.shape == (0, 2)
