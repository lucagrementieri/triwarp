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


"""
def test_all_coplanar(device):
    tri_c = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [1.0, 2.0, 0.0]],
        ],
        dtype=np.float64,
    )
    tri_wp = triangles_to_wp(tri_c, device)
    assert int(tw.all_coplanar(tri_wp, device=device).numpy()[0]) == int(tm.all_coplanar(tri_c))

    rng = np.random.default_rng(6)
    tri_nc = rng.random((5, 3, 3))
    tri_nc_wp = triangles_to_wp(tri_nc, device)
    assert int(tw.all_coplanar(tri_nc_wp, device=device).numpy()[0]) == int(tm.all_coplanar(tri_nc))


def test_any_coplanar(device):
    tri = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=np.float64,
    )
    tri_wp = triangles_to_wp(tri, device=device)
    assert int(tw.any_coplanar(tri_wp, device=device).numpy()[0]) == int(tm.any_coplanar(tri))


def test_mass_properties(device):
    rng = np.random.default_rng(7)
    tri_np = rng.random((28, 3, 3))
    tri_wp = triangles_to_wp(tri_np, device)
    mp_w = tw.mass_properties(tri_wp, device=device)
    mp_t = tm.mass_properties(tri_np)
    assert abs(mp_w.volume.numpy()[0] - mp_t["volume"]) < 1e-9
    assert abs(mp_w.mass.numpy()[0] - mp_t["mass"]) < 1e-9
    _assert_close(mp_w.center_mass.numpy(), mp_t["center_mass"])
    assert mp_w.inertia is not None
    _assert_close(mp_w.inertia.numpy().reshape(3, 3), mp_t["inertia"])


def test_mass_properties_skip_inertia(device):
    rng = np.random.default_rng(8)
    tri_np = rng.random((10, 3, 3))
    tri_wp = triangles_to_wp(tri_np, device)
    mp_w = tw.mass_properties(tri_wp, skip_inertia=True, device=device)
    mp_t = tm.mass_properties(tri_np, skip_inertia=True)
    assert abs(mp_w.volume.numpy()[0] - mp_t["volume"]) < 1e-9
    assert mp_w.inertia is None


def test_mass_properties_center_mass_override(device):
    rng = np.random.default_rng(9)
    tri_np = rng.random((16, 3, 3))
    tri_wp = triangles_to_wp(tri_np, device)
    cm = np.array([0.1, -0.2, 0.05], dtype=np.float64)
    cm_wp = float3_to_wp(cm, device)
    mp_w = tw.mass_properties(tri_wp, center_mass=cm_wp, device=device)
    mp_t = tm.mass_properties(tri_np, center_mass=cm)
    _assert_close(mp_w.center_mass.numpy(), mp_t["center_mass"])
    assert mp_w.inertia is not None
    _assert_close(mp_w.inertia.numpy().reshape(3, 3), mp_t["inertia"])


def test_windings_aligned_broadcast(device):
    rng = np.random.default_rng(10)
    tri_np = rng.random((20, 3, 3))
    tri_wp = triangles_to_wp(tri_np, device)
    n_cmp = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cmp_wp = wp.array(n_cmp.reshape(1, 3), dtype=wp.vec3d, device=device)
    got = tw.windings_aligned(tri_wp, cmp_wp, device=device).numpy()
    exp = tm.windings_aligned(tri_np, n_cmp)
    assert np.array_equal(got, exp.astype(np.int32))


def test_windings_aligned_per_triangle(device):
    rng = np.random.default_rng(11)
    tri_np = rng.random((18, 3, 3))
    tri_wp = triangles_to_wp(tri_np, device)
    crs = tm.cross(tri_np)
    n_cmp = crs / np.linalg.norm(crs, axis=1, keepdims=True)
    cmp_wp = wp.array(n_cmp, dtype=wp.vec3d, device=device)
    got = tw.windings_aligned(tri_wp, cmp_wp, device=device).numpy()
    exp = tm.windings_aligned(tri_np, n_cmp)
    assert np.array_equal(got, exp.astype(np.int32))


def test_triangle_bounds(device):
    rng = np.random.default_rng(12)
    tri_np = rng.random((15, 3, 3))
    tri_wp = triangles_to_wp(tri_np, device)
    got = tw.triangle_bounds(tri_wp, device=device).numpy().reshape(-1, 6)
    exp = np.column_stack((tri_np.min(axis=1), tri_np.max(axis=1)))
    _assert_close(got, exp)


def test_extents(device):
    rng = np.random.default_rng(13)
    tri_np = rng.random((22, 3, 3))
    tri_wp = triangles_to_wp(tri_np, device)
    got = tw.extents(triangles=tri_wp, device=device).numpy()
    exp = tm.extents(triangles=tri_np)
    _assert_close(got, exp)


def test_nondegenerate(device):
    rng = np.random.default_rng(14)
    tri_np = rng.random((26, 3, 3))
    tri_wp = triangles_to_wp(tri_np, device)
    got = tw.nondegenerate(triangles=tri_wp, device=device).numpy().astype(bool)
    exp = tm.nondegenerate(triangles=tri_np)
    assert np.array_equal(got, exp)







def test_closest_point(device):
    rng = np.random.default_rng(17)
    tri_np = rng.random((33, 3, 3))
    p = rng.random((33, 3))
    tri_wp = triangles_to_wp(tri_np, device)
    p_wp = vec3_points_to_wp(p, device)
    got = tw.closest_point(tri_wp, p_wp, device=device).numpy()
    exp = tm.closest_point(tri_np, p)
    _assert_close(got, exp, rtol=1e-11, atol=1e-11)


def test_to_kwargs_vertices_faces(device):
    rng = np.random.default_rng(18)
    tri_np = rng.random((8, 3, 3))
    n = tri_np.shape[0]
    tri_wp = triangles_to_wp(tri_np, device)
    kw_w = tw.to_kwargs(tri_wp, device=device)
    kw_t = tm.to_kwargs(tri_np)
    _assert_close(kw_w["vertices"].numpy(), kw_t["vertices"])
    assert np.array_equal(kw_w["faces"].numpy().reshape(n, 3), kw_t["faces"])


def test_bounds_tree_not_implemented(device):
    tri_wp = triangles_to_wp(np.random.default_rng(19).random((4, 3, 3)), device)
    with pytest.raises(NotImplementedError):
        tw.bounds_tree(tri_wp, device=device)


def test_points_to_barycentric_cross_raises(device):
    tri_wp = triangles_to_wp(np.random.default_rng(20).random((3, 3, 3)), device)
    p_wp = vec3_points_to_wp(np.random.default_rng(20).random((3, 3)), device)
    with pytest.raises(NotImplementedError):
        tw.points_to_barycentric(tri_wp, p_wp, method="cross", device=device)
"""
