"""
Regression tests for ``triwarp.vertices`` against ``trimesh.vertices`` (CPU reference).
"""

import numpy as np
import warp as wp

import trimesh as tm
import triwarp as tw


def test_mean_vertex_normals(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = half_torus

    n_vertices = mesh_tm.vertices.shape[0]
    face_normals_tm = mesh_tm.face_normals
    vertex_normals_tm = tm.geometry.mean_vertex_normals(n_vertices, mesh_tm.faces, face_normals_tm)

    face_normals_wp = wp.array(face_normals_tm, dtype=wp.vec3, device=mesh_wp.device)
    vertex_normals_wp = tw.vertices.mean_vertex_normals(n_vertices, mesh_wp.indices, face_normals_wp)
    assert np.allclose(vertex_normals_wp.numpy(), vertex_normals_tm, rtol=1e-5, atol=1e-5)


def test_weighted_vertex_normals(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = half_torus

    n_vertices = mesh_tm.vertices.shape[0]
    face_normals_tm = mesh_tm.face_normals
    face_angles_tm = mesh_tm.face_angles
    vertex_normals_tm = tm.geometry.weighted_vertex_normals(n_vertices, mesh_tm.faces, face_normals_tm, face_angles_tm)

    face_normals_wp = wp.array(face_normals_tm, dtype=wp.vec3, device=mesh_wp.device)
    face_angles_wp = wp.array(face_angles_tm, dtype=wp.float32, device=mesh_wp.device)
    vertex_normals_wp = tw.vertices.weighted_vertex_normals(
        n_vertices, mesh_wp.indices, face_normals_wp, face_angles_wp
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


def test_discrete_gaussian_curvature(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = hemisphere

    face_angles_tm = mesh_tm.face_angles
    points_tm = mesh_tm.vertices[:4]
    radius = 0.1
    gauss_curvature_tm = tm.curvature.discrete_gaussian_curvature_measure(mesh_tm, points_tm, radius)

    points_wp = wp.array(points_tm, dtype=wp.vec3, device=mesh_wp.device)
    vertices_wp = wp.array(mesh_tm.vertices, dtype=wp.vec3, device=mesh_wp.device)
    faces_wp = wp.array(mesh_wp.indices, dtype=wp.int32, device=mesh_wp.device)
    face_angles_wp = wp.array(face_angles_tm, dtype=wp.float32, device=mesh_wp.device)
    gauss_curvature_wp = tw.vertices.discrete_gaussian_curvature(
        points_wp, vertices_wp, faces_wp, face_angles_wp, radius
    )
    assert np.allclose(gauss_curvature_wp.numpy(), gauss_curvature_tm, rtol=1e-5, atol=1e-5)
