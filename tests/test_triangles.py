"""
Regression tests for ``triwarp.triangles`` against ``trimesh.triangles`` (CPU reference).
"""

from __future__ import annotations

import numpy as np
import warp as wp

import trimesh.triangles as tm
import triwarp.triangles as tw


def _triangle_soup_to_vertices_faces_wp(tri_np: np.ndarray, device):
    """Triangle soup ``(n, 3, 3)`` → indexed mesh on ``device`` (disjoint vertices per face)."""
    tri_np = np.ascontiguousarray(tri_np, dtype=np.float64)
    n = tri_np.shape[0]
    vertices = tri_np.reshape(-1, 3)
    faces = np.arange(n * 3, dtype=np.int32).reshape(n, 3)
    v_wp = wp.array(vertices, dtype=wp.vec3, device=device)
    f_wp = wp.array(faces.reshape(-1), dtype=wp.int32, device=device)
    return v_wp, f_wp


def test_face_areas(device):
    rng = np.random.default_rng(1)
    tri_np = rng.random((48, 3, 3))
    v_wp, f_wp = _triangle_soup_to_vertices_faces_wp(tri_np, device)
    got = tw.face_areas(v_wp, f_wp).numpy()
    exp = tm.area(triangles=tri_np)
    assert np.allclose(got, exp, rtol=1e-5, atol=1e-5)


def test_face_normals(device):
    rng = np.random.default_rng(3)
    tri_np = rng.random((40, 3, 3))
    v_wp, f_wp = _triangle_soup_to_vertices_faces_wp(tri_np, device)
    n_w = tw.face_normals(v_wp, f_wp).numpy()
    n_t, v_t = tm.normals(triangles=tri_np)
    mask = v_t.astype(bool)
    assert np.allclose(n_w[mask], n_t[mask], rtol=1e-5, atol=1e-5)


def test_angles(device):
    rng = np.random.default_rng(5)
    tri_np = rng.random((36, 3, 3))
    v_wp, f_wp = _triangle_soup_to_vertices_faces_wp(tri_np, device)
    got = tw.angles(v_wp, f_wp).numpy()
    exp = tm.angles(tri_np)
    assert np.allclose(got, exp, rtol=1e-5, atol=1e-5)


def test_nondegenerate(device):
    rng = np.random.default_rng(14)
    tri_np = rng.random((26, 3, 3))
    v_wp, f_wp = _triangle_soup_to_vertices_faces_wp(tri_np, device)
    got = tw.nondegenerate(v_wp, f_wp).numpy().astype(bool)
    exp = tm.nondegenerate(triangles=tri_np)
    assert np.array_equal(got, exp)


def test_barycentric_to_points(device):
    rng = np.random.default_rng(15)
    tri_np = rng.random((30, 3, 3))
    b = rng.random((30, 3))
    tri_wp = _triangle_soup_to_vertices_faces_wp(tri_np, device)
    b_wp = wp.array(b, dtype=wp.vec3, device=device)
    got = tw.barycentric_to_points(tri_wp[0], tri_wp[1], b_wp).numpy()
    exp = tm.barycentric_to_points(tri_np, b)
    assert np.allclose(got, exp, rtol=1e-5, atol=1e-5)


def test_points_to_barycentric_cramer(device):
    rng = np.random.default_rng(16)
    tri_np = rng.random((25, 3, 3))
    p = rng.random((25, 3))
    tri_wp = _triangle_soup_to_vertices_faces_wp(tri_np, device)
    p_wp = wp.array(p, dtype=wp.vec3, device=device)
    got = tw.points_to_barycentric(tri_wp[0], tri_wp[1], p_wp, method="cramer").numpy()
    exp = tm.points_to_barycentric(tri_np, p, method="cramer")
    assert np.allclose(got, exp, rtol=1e-5, atol=1e-5)


def test_points_to_barycentric_cross(device):
    rng = np.random.default_rng(17)
    tri_np = rng.random((25, 3, 3))
    p = rng.random((25, 3))
    tri_wp = _triangle_soup_to_vertices_faces_wp(tri_np, device)
    p_wp = wp.array(p, dtype=wp.vec3, device=device)
    got = tw.points_to_barycentric(tri_wp[0], tri_wp[1], p_wp, method="cross").numpy()
    exp = tm.points_to_barycentric(tri_np, p, method="cross")
    assert np.allclose(got, exp, rtol=1e-5, atol=1e-5)


def test_closest_point(device):
    rng = np.random.default_rng(17)
    tri_np = rng.random((33, 3, 3))
    p = rng.random((33, 3))
    v_wp, f_wp = _triangle_soup_to_vertices_faces_wp(tri_np, device)
    p_wp = wp.array(p, dtype=wp.vec3, device=device)
    got = tw.closest_point(v_wp, f_wp, p_wp).numpy()
    exp = tm.closest_point(tri_np, p)
    assert np.allclose(got, exp, rtol=1e-5, atol=1e-5)
