"""Shared mesh format conversions for tests."""

from __future__ import annotations

import numpy as np
import pyvista as pv
import trimesh as tm
import warp as wp


def trimesh_to_warp(mesh: tm.Trimesh, device: str) -> wp.Mesh:
    vertices = wp.array(np.ascontiguousarray(mesh.vertices, dtype=np.float32), dtype=wp.vec3, device=device)
    faces = wp.array(np.ascontiguousarray(mesh.faces.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device)
    return wp.Mesh(points=vertices, indices=faces)


def trimesh_to_pyvista(mesh: tm.Trimesh) -> pv.PolyData:
    faces_np = np.column_stack([
        np.full(mesh.faces.shape[0], 3, dtype=np.int32),
        mesh.faces.astype(np.int32),
    ]).ravel()
    return pv.PolyData(np.ascontiguousarray(mesh.vertices.astype(np.float64)), faces_np)
