"""Regression tests for ``triwarp.interpolation`` against igl (CPU reference)."""

import igl
import numpy as np
import trimesh as tm
import warp as wp

import triwarp as tw


def test_average_onto_faces(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = half_torus
    rng = np.random.default_rng(0)

    faces_np = np.array(mesh_tm.faces, dtype=np.int64)
    vertex_values_np = rng.uniform(size=mesh_tm.vertices.shape[0])
    face_values_igl = igl.average_onto_faces(faces_np, vertex_values_np)

    vertex_values_wp = wp.array(vertex_values_np, dtype=wp.float32, device=mesh_wp.device)
    face_values_wp = tw.interpolation.average_onto_faces(mesh_wp.indices, vertex_values_wp)
    assert np.allclose(face_values_wp.numpy(), face_values_igl, rtol=1e-5, atol=1e-5)


def test_average_onto_vertices(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = half_torus
    rng = np.random.default_rng(1)

    n_vertices = mesh_tm.vertices.shape[0]
    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int64)
    face_values_np = rng.uniform(size=mesh_tm.faces.shape[0])
    vertex_values_igl = igl.average_onto_vertices(vertices_np, faces_np, face_values_np)

    face_values_wp = wp.array(face_values_np, dtype=wp.float32, device=mesh_wp.device)
    vertex_values_wp = tw.interpolation.average_onto_vertices(
        n_vertices, mesh_wp.indices, face_values_wp
    )
    assert np.allclose(vertex_values_wp.numpy(), vertex_values_igl, rtol=1e-5, atol=1e-5)


def test_average_from_edges_onto_vertices(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = half_torus
    rng = np.random.default_rng(2)

    n_vertices = mesh_tm.vertices.shape[0]
    faces_np = np.array(mesh_tm.faces, dtype=np.int64)
    edges_igl, orientation_igl = igl.orient_halfedges(faces_np)
    edges_igl = np.asarray(edges_igl)
    orientation_igl = np.asarray(orientation_igl)
    n_unique_edges = int(edges_igl.max()) + 1
    edge_values_np = rng.uniform(size=n_unique_edges)
    vertex_values_igl = igl.average_from_edges_onto_vertices(
        faces_np, edges_igl, orientation_igl, edge_values_np
    )

    edges_wp = wp.array(edges_igl.astype(np.int32), dtype=wp.int32, device=mesh_wp.device)
    orientation_wp = wp.array(
        orientation_igl.astype(np.int32), dtype=wp.int32, device=mesh_wp.device
    )
    edge_values_wp = wp.array(edge_values_np, dtype=wp.float32, device=mesh_wp.device)
    vertex_values_wp = tw.interpolation.average_from_edges_onto_vertices(
        n_vertices, mesh_wp.indices, edges_wp, orientation_wp, edge_values_wp
    )
    assert np.allclose(vertex_values_wp.numpy(), vertex_values_igl, rtol=1e-5, atol=1e-5)
