"""Shared mesh format conversions for tests."""

from __future__ import annotations

import numpy as np
import open3d as o3d
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


def wedge_uv_to_pymeshlab(
    vertices_np: np.ndarray, faces_np: np.ndarray, wedge_uv_np: np.ndarray
) -> ml.MeshSet:
    """
    Build a MeshSet carrying a **per-wedge** (per-corner) texture atlas.

    MeshLab stores UVs on face corners rather than on vertices, which is why it has no texcoord
    *index* buffer at all and why its seam predicate compares coordinates. ``w_tex_coords_matrix``
    takes exactly triwarp's per-corner layout, ``(3 * n_faces, 2)`` in ``3 * f + k`` order, so any
    numpy atlas can drive ``compute_selection_by_texture_seams_per_vertex`` without going through a
    file.
    """
    meshset = ml.MeshSet()
    meshset.add_mesh(
        ml.Mesh(
            vertex_matrix=np.ascontiguousarray(vertices_np, dtype=np.float64),
            face_matrix=np.ascontiguousarray(faces_np, dtype=np.int32),
            w_tex_coords_matrix=np.ascontiguousarray(wedge_uv_np, dtype=np.float64),
        )
    )
    return meshset


def points_to_pymeshlab(points_np: np.ndarray, normals_np: np.ndarray | None = None) -> ml.MeshSet:
    """
    Wrap a bare point cloud in a **face-less** single-mesh ``pymeshlab.MeshSet``.

    A handful of MeshLab filters require a mesh with vertices and no faces --
    ``compute_normal_for_point_clouds`` refuses anything else, and the query layer of
    ``compute_scalar_by_distance_from_another_mesh_per_vertex`` wants the sample points on their own
    -- which neither [`trimesh_to_pymeshlab`][tests.conversions.trimesh_to_pymeshlab] nor
    [`warp_to_pymeshlab`][tests.conversions.warp_to_pymeshlab] can build. Same per-filter freshness
    rule as those two: one MeshSet, one filter call.
    """
    meshset = ml.MeshSet()
    vertices = np.ascontiguousarray(points_np, dtype=np.float64)
    if normals_np is None:
        meshset.add_mesh(ml.Mesh(vertices))
    else:
        meshset.add_mesh(
            ml.Mesh(vertices, v_normals_matrix=np.ascontiguousarray(normals_np, dtype=np.float64))
        )
    return meshset


def trimesh_to_open3d(mesh: tm.Trimesh) -> o3d.geometry.TriangleMesh:
    """
    Wrap a ``tm.Trimesh`` in a legacy ``open3d.geometry.TriangleMesh``.

    Open3D wants float64 positions and **int32** faces (``Vector3iVector`` silently misreads a
    wider dtype). Every array goes through ``np.array`` rather than ``np.ascontiguousarray``, which
    is load-bearing: the pybind11 ``Vector3dVector`` cast raises ``ValueError: array is not
    writeable`` on a read-only input, and ``ascontiguousarray`` returns an already-contiguous array
    untouched -- so trimesh's cached ``vertex_normals`` would fail where ``vertices`` succeeded.

    Unlike the pymeshlab helpers this result is safe to reuse across calls *in tests*: the
    per-filter freshness rule that forces ``benchmarks/conftest.py`` to rebuild its MeshSet exists
    because a benchmark applies the same mutating call ten times in a row, which no test does. A
    test that calls a genuinely mutating method twice should still build twice.
    """
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.array(mesh.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.array(mesh.faces, dtype=np.int32)),
    )


def points_to_open3d(
    points_np: np.ndarray, normals_np: np.ndarray | None = None
) -> o3d.geometry.PointCloud:
    """
    Wrap a point cloud, and optionally its normals, in an ``open3d.geometry.PointCloud``.

    ``np.array`` rather than ``np.ascontiguousarray`` for the same writeability reason as
    [`trimesh_to_open3d`][tests.conversions.trimesh_to_open3d].
    """
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.array(points_np, dtype=np.float64))
    if normals_np is not None:
        cloud.normals = o3d.utility.Vector3dVector(np.array(normals_np, dtype=np.float64))
    return cloud


def open3d_to_trimesh(mesh_o3d: o3d.geometry.TriangleMesh) -> tm.Trimesh:
    """Read a legacy open3d mesh back out, unprocessed so the topology survives the round trip."""
    return tm.Trimesh(
        vertices=np.asarray(mesh_o3d.vertices), faces=np.asarray(mesh_o3d.triangles), process=False
    )


def faces_igl(mesh: tm.Trimesh) -> np.ndarray:
    """
    Faces as the ``(n_faces, 3)`` int64 array the libigl bindings expect.

    libigl's Eigen templates are instantiated for 64-bit indices, so handing them trimesh's native
    dtype works by luck rather than contract; several functions crash on int32.
    """
    return mesh.faces.astype(np.int64)


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
