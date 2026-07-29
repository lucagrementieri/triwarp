"""Shared mesh format conversions for tests."""

from __future__ import annotations

import numpy as np
import pymeshlab as ml
import pyvista as pv
import trimesh as tm
import warp as wp


def trimesh_to_warp(mesh: tm.Trimesh, device: str) -> wp.Mesh:
    vertices = wp.array(
        np.ascontiguousarray(mesh.vertices, dtype=np.float32), dtype=wp.vec3, device=device
    )
    faces = wp.array(
        np.ascontiguousarray(mesh.faces.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device
    )
    return wp.Mesh(points=vertices, indices=faces)


def trimesh_to_pymeshlab(mesh: tm.Trimesh) -> ml.MeshSet:
    """
    Wrap a ``tm.Trimesh`` in a fresh single-mesh ``pymeshlab.MeshSet``.

    MeshLab wants float64 positions; the ``(n_faces, 3)`` index array goes in as-is. The returned
    MeshSet is **not** reusable across filters: almost every one of them mutates ``current_mesh()``
    in place, so build a new one per comparison rather than threading one through a test.
    """
    meshset = ml.MeshSet()
    meshset.add_mesh(
        ml.Mesh(
            np.ascontiguousarray(mesh.vertices, dtype=np.float64),
            np.ascontiguousarray(mesh.faces, dtype=np.int32),
        )
    )
    return meshset


def warp_to_pymeshlab(vertices_wp: wp.array, faces_wp: wp.array) -> ml.MeshSet:
    """
    Read triwarp's ``(vertices, flat faces)`` pair back into a ``pymeshlab.MeshSet``.

    The face buffer is triwarp's flat one, so it is reshaped to ``(n_faces, 3)`` here; use this when
    the mesh under test is a triwarp *output* rather than one of the ``tests/conftest.py`` fixtures
    (which already carry a ``tm.Trimesh`` for ``trimesh_to_pymeshlab``).
    """
    meshset = ml.MeshSet()
    meshset.add_mesh(
        ml.Mesh(
            np.ascontiguousarray(vertices_wp.numpy(), dtype=np.float64),
            np.ascontiguousarray(faces_wp.numpy().reshape(-1, 3), dtype=np.int32),
        )
    )
    return meshset


def trimesh_to_pyvista(mesh: tm.Trimesh) -> pv.PolyData:
    faces_np = np.column_stack(
        [np.full(mesh.faces.shape[0], 3, dtype=np.int32), mesh.faces.astype(np.int32)]
    ).ravel()
    return pv.PolyData(np.ascontiguousarray(mesh.vertices.astype(np.float64)), faces_np)


def bsr_to_dense(matrix: object, n_vertices: int) -> np.ndarray:
    """
    Densify a ``BsrMatrix``, reading only the entries its offsets actually address.

    ``BsrMatrix.values`` is allocated at the *triplet* count and its ``nnz`` is an upper bound until
    synchronized, so the tail of that buffer is uninitialized scratch. Comparing two matrices'
    ``values`` arrays directly reads that scratch and is flaky by construction; the row offsets are
    the only safe way in.
    """
    offsets = matrix.offsets.numpy()  # type: ignore[attr-defined]
    columns = matrix.columns.numpy()  # type: ignore[attr-defined]
    values = matrix.values.numpy()  # type: ignore[attr-defined]
    dense = np.zeros((n_vertices, n_vertices))
    for row in range(n_vertices):
        for slot in range(offsets[row], offsets[row + 1]):
            dense[row, columns[slot]] = values[slot]
    return dense
