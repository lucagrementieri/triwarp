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


def _sort_adjacency_rows(adjacency: np.ndarray) -> np.ndarray:
    order = np.lexsort((adjacency[:, 1], adjacency[:, 0]))
    return adjacency[order]


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
def test_face_adjacency(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    adjacency_tm = mesh_tm.face_adjacency
    adjacency_wp = tw.graph.face_adjacency(mesh_wp.indices)
    assert np.array_equal(_sort_adjacency_rows(adjacency_wp.numpy()), _sort_adjacency_rows(adjacency_tm))
