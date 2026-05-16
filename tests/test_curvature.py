"""
Regression tests for ``triwarp.curvature`` against ``trimesh.curvature`` (CPU reference).
"""

import numpy as np
import warp as wp

import trimesh as tm
import triwarp as tw


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
