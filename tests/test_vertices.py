"""Regression tests for ``triwarp.vertices`` against Trimesh (CPU reference)."""

import igl
import numpy as np
import trimesh as tm
import warp as wp

import triwarp as tw


def test_mean_vertex_normals(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = half_torus

    n_vertices = mesh_tm.vertices.shape[0]
    face_normals_tm = mesh_tm.face_normals
    vertex_normals_tm = tm.geometry.mean_vertex_normals(n_vertices, mesh_tm.faces, face_normals_tm)

    face_normals_wp = wp.array(face_normals_tm, dtype=wp.vec3, device=mesh_wp.device)
    vertex_normals_wp = tw.vertices.mean_vertex_normals(
        n_vertices, mesh_wp.indices, face_normals_wp
    )
    assert np.allclose(vertex_normals_wp.numpy(), vertex_normals_tm, rtol=1e-5, atol=1e-5)


def test_weighted_vertex_normals(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = half_torus

    n_vertices = mesh_tm.vertices.shape[0]
    face_normals_tm = mesh_tm.face_normals
    face_angles_tm = mesh_tm.face_angles
    vertex_normals_tm = tm.geometry.weighted_vertex_normals(
        n_vertices, mesh_tm.faces, face_normals_tm, face_angles_tm
    )

    face_normals_wp = wp.array(face_normals_tm, dtype=wp.vec3, device=mesh_wp.device)
    face_weights_wp = wp.array(face_angles_tm, dtype=wp.float32, device=mesh_wp.device)
    vertex_normals_wp = tw.vertices.weighted_vertex_normals(
        n_vertices, mesh_wp.indices, face_normals_wp, face_weights_wp
    )
    assert np.allclose(vertex_normals_wp.numpy(), vertex_normals_tm, rtol=1e-5, atol=1e-5)


def test_area_weighted_vertex_normals(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = half_torus

    n_vertices = mesh_tm.vertices.shape[0]
    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int32)
    vertex_normals_igl = igl.per_vertex_normals(
        vertices_np, faces_np, igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA
    )

    vertices_wp = wp.array(mesh_tm.vertices, dtype=wp.vec3, device=mesh_wp.device)
    vertex_normals_wp = tw.vertices.area_weighted_vertex_normals(
        n_vertices, vertices_wp, mesh_wp.indices
    )
    assert np.allclose(vertex_normals_wp.numpy(), vertex_normals_igl, rtol=1e-5, atol=1e-5)


def test_angle_weighted_vertex_normals(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = half_torus

    n_vertices = mesh_tm.vertices.shape[0]
    vertex_normals_tm = tm.geometry.weighted_vertex_normals(
        n_vertices, mesh_tm.faces, mesh_tm.face_normals, mesh_tm.face_angles
    )

    vertices_wp = wp.array(mesh_tm.vertices, dtype=wp.vec3, device=mesh_wp.device)
    vertex_normals_wp = tw.vertices.angle_weighted_vertex_normals(
        n_vertices, vertices_wp, mesh_wp.indices
    )
    assert np.allclose(vertex_normals_wp.numpy(), vertex_normals_tm, rtol=1e-5, atol=1e-5)


def test_vertex_defects(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = half_torus

    n_vertices = mesh_tm.vertices.shape[0]
    face_angles_tm = mesh_tm.face_angles
    vertex_defects_tm = tm.curvature.vertex_defects(mesh_tm)

    face_angles_wp = wp.array(face_angles_tm, dtype=wp.float32, device=mesh_wp.device)
    vertex_defects_wp = tw.vertices.vertex_defects(n_vertices, mesh_wp.indices, face_angles_wp)
    assert np.allclose(vertex_defects_wp.numpy(), vertex_defects_tm, rtol=1e-5, atol=1e-5)
