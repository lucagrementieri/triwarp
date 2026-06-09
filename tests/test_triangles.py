"""
Regression tests for ``triwarp.triangles`` against ``trimesh.triangles`` (CPU reference).
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

import trimesh as tm
import triwarp as tw


def test_face_normals_and_areas(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = icosahedron
    normal_wp, area_wp = tw.triangles.face_normals_and_areas(mesh_wp.points, mesh_wp.indices)
    assert np.allclose(normal_wp.numpy(), mesh_tm.face_normals, rtol=1e-5, atol=1e-5)
    assert np.allclose(area_wp.numpy(), mesh_tm.area_faces, rtol=1e-5, atol=1e-5)


def test_angles(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = half_torus
    angles_wp = tw.triangles.face_angles(mesh_wp.points, mesh_wp.indices)
    assert np.allclose(angles_wp.numpy(), mesh_tm.face_angles, rtol=1e-5, atol=1e-5)


def test_nondegenerate(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = hemisphere
    nondegenerate_tm = tm.triangles.nondegenerate(mesh_tm.triangles)
    nondegenerate_wp = tw.triangles.nondegenerate(mesh_wp.points, mesh_wp.indices)
    assert np.array_equal(nondegenerate_wp.numpy().astype(bool), nondegenerate_tm)


def test_barycentric_to_points(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = hemisphere
    barycentric_np = np.random.rand(mesh_tm.triangles.shape[0], 3)
    points_tm = tm.triangles.barycentric_to_points(mesh_tm.triangles, barycentric_np)

    barycentric_wp = wp.array(barycentric_np, dtype=wp.vec3, device=mesh_wp.points.device)
    points_wp = tw.triangles.barycentric_to_points(mesh_wp.points, mesh_wp.indices, barycentric_wp)
    assert np.allclose(points_wp.numpy(), points_tm, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("method", ["cramer", "cross"])
def test_points_to_barycentric(hemisphere: tuple[tm.Trimesh, wp.Mesh], method: str):
    mesh_tm, mesh_wp = hemisphere

    barycentric_np = np.random.rand(mesh_tm.triangles.shape[0], 3)
    points_np = tm.triangles.barycentric_to_points(mesh_tm.triangles, barycentric_np)
    barycentric_tm = tm.triangles.points_to_barycentric(mesh_tm.triangles, points_np, method=method)

    points_wp = wp.array(points_np, dtype=wp.vec3, device=mesh_wp.points.device)
    barycentric_wp = tw.triangles.points_to_barycentric(
        mesh_wp.points, mesh_wp.indices, points_wp, method=method
    )
    assert np.allclose(barycentric_wp.numpy(), barycentric_tm, rtol=1e-5, atol=1e-5)


def test_closest_point(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = hemisphere
    points_np = np.random.rand(mesh_tm.triangles.shape[0], 3)
    closest_points_tm = tm.triangles.closest_point(mesh_tm.triangles, points_np)

    points_wp = wp.array(points_np, dtype=wp.vec3, device=mesh_wp.points.device)
    closest_points_wp = tw.triangles.closest_point(mesh_wp.points, mesh_wp.indices, points_wp)
    assert np.allclose(closest_points_wp.numpy(), closest_points_tm, rtol=1e-5, atol=1e-5)


def test_centroid(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = hemisphere

    centroid_tm = mesh_tm.centroid

    centroid_wp = tw.triangles.centroid(mesh_wp.points, mesh_wp.indices)
    centroid_wp = np.array([centroid_wp.x, centroid_wp.y, centroid_wp.z])
    assert np.allclose(centroid_wp, centroid_tm, rtol=1e-5, atol=1e-5)


def test_centroid_empty(device: str):
    vertices = wp.zeros(1, dtype=wp.vec3, device=device)
    faces = wp.array([], dtype=wp.int32, device=device)
    centroid_wp = tw.triangles.centroid(vertices, faces)
    centroid_wp = np.array([centroid_wp.x, centroid_wp.y, centroid_wp.z])
    assert np.isnan(centroid_wp).all()
