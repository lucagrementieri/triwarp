"""Regression tests for ``triwarp.graph`` against ``trimesh.geometry`` (CPU reference)."""

from __future__ import annotations

import numpy as np
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


def test_faces_to_edges_empty(device: str) -> None:
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    edges_wp = tw.graph.faces_to_edges(faces_wp)
    assert edges_wp.shape == (0, 2)
