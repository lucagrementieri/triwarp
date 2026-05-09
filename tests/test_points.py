"""
Regression tests for ``triwarp.points.query_ball`` / ``query_ball_count`` against
SciPy ``cKDTree`` (CPU reference).
"""

from __future__ import annotations

import numpy as np
import warp as wp
from scipy.spatial import KDTree

import triwarp as tw


def _vec3(p: np.ndarray) -> wp.vec3:
    return wp.vec3(float(p[0]), float(p[1]), float(p[2]))


"""
def _assert_no_close_pairs(points: np.ndarray, radius: float) -> None:
    if points.shape[0] <= 1:
        return
    pairs = cKDTree(points).query_pairs(radius, output_type="ndarray")
    assert pairs.shape[0] == 0


def test_remove_close_matches_trimesh(device: str):
    rng = np.random.default_rng(0)
    pts = rng.random((80, 3), dtype=np.float32) * 5.0
    radius = np.float32(0.41)

    culled_tm, mask_tm = tm.points.remove_close(pts, float(radius))

    pts_wp = wp.array(np.ascontiguousarray(pts), dtype=wp.vec3, device=device)
    culled_tw, mask_tw = tw.points.remove_close(pts_wp, float(radius))

    assert np.array_equal(mask_tw, mask_tm)
    assert np.allclose(culled_tw, culled_tm, rtol=1e-5, atol=1e-5)
    _assert_no_close_pairs(culled_tw, float(radius))


def test_remove_close_geometric_invariant_random(device: str):
    rng = np.random.default_rng(1)
    pts = rng.random((120, 3), dtype=np.float32) * 4.0
    radius = 0.27

    pts_wp = wp.array(np.ascontiguousarray(pts), dtype=wp.vec3, device=device)
    culled, mask = tw.points.remove_close(pts_wp, radius)

    assert mask.dtype == bool
    assert culled.shape[0] == int(mask.sum())
    assert np.allclose(culled, pts[mask], rtol=1e-5, atol=1e-5)
    _assert_no_close_pairs(culled, radius)


def test_remove_close_empty(device: str):
    pts_wp = wp.empty(0, dtype=wp.vec3, device=device)
    culled, mask = tw.points.remove_close(pts_wp, 1.0)
    assert culled.shape == (0, 3)
    assert mask.shape == (0,)


def test_remove_close_no_pairs(device: str):
    pts = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]],
        dtype=np.float32,
    )
    pts_wp = wp.array(pts, dtype=wp.vec3, device=device)
    culled, mask = tw.points.remove_close(pts_wp, 1.0)
    assert np.all(mask)
    assert np.array_equal(culled, pts)
"""


def test_query_ball_single_matches_scipy(device: str):
    rng = np.random.default_rng(3)
    pts = rng.random((70, 3), dtype=np.float32) * 4.0
    tree = KDTree(pts)
    q = pts[31].copy()
    r = 0.55

    pts_wp = wp.array(np.ascontiguousarray(pts), dtype=wp.vec3, device=device)
    exp = tree.query_ball_point(q, r, return_sorted=False)
    got_idx, got_dist = tw.points.query_ball(pts_wp, _vec3(q), r, return_sorted=False)
    assert set(got_idx.numpy().tolist()) == set(exp)
    exp_d = np.linalg.norm(pts[np.array(sorted(exp), dtype=np.int64)] - q, axis=1)
    order = np.argsort(got_idx.numpy())
    assert np.array_equal(got_idx.numpy()[order], np.sort(np.asarray(exp)))
    assert np.allclose(got_dist.numpy()[order], exp_d, rtol=1e-5, atol=1e-5)

    got_idx_s, got_dist_s = tw.points.query_ball(
        pts_wp, _vec3(q), r, return_sorted=True
    )
    exp_sorted = sorted(exp)
    assert np.array_equal(got_idx_s.numpy(), exp_sorted)
    assert np.allclose(
        got_dist_s.numpy(),
        np.linalg.norm(pts[np.array(exp_sorted, dtype=np.int64)] - q, axis=1),
        rtol=1e-5,
        atol=1e-5,
    )


def test_query_ball_batch_matches_scipy(device: str):
    rng = np.random.default_rng(4)
    pts = rng.random((55, 3), dtype=np.float32) * 3.0
    tree = KDTree(pts)
    x = rng.random((6, 3), dtype=np.float32) * 3.0
    r = 0.4

    pts_wp = wp.array(np.ascontiguousarray(pts), dtype=wp.vec3, device=device)
    x_wp = wp.array(np.ascontiguousarray(x), dtype=wp.vec3, device=device)
    got_idx, got_dist = tw.points.query_ball(pts_wp, x_wp, r, return_sorted=False)
    exp = tree.query_ball_point(x, r)
    assert len(got_idx) == len(exp)
    for i in range(len(exp)):
        g_i = got_idx[i].numpy()
        e_i = np.asarray(exp[i])
        assert set(g_i.tolist()) == set(e_i.tolist())
        order = np.argsort(g_i)
        assert np.allclose(
            got_dist[i].numpy()[order],
            np.linalg.norm(pts[g_i[order]] - x[i], axis=1),
            rtol=1e-5,
            atol=1e-5,
        )


def test_query_ball_count_matches_scipy(device: str):
    rng = np.random.default_rng(5)
    pts = rng.random((40, 3), dtype=np.float32) * 2.0
    tree = KDTree(pts)
    q = pts[5].copy()
    x = rng.random((4, 3), dtype=np.float32) * 2.0
    r = 0.35

    pts_wp = wp.array(np.ascontiguousarray(pts), dtype=wp.vec3, device=device)
    q_wp = wp.array(np.ascontiguousarray(q[None, :]), dtype=wp.vec3, device=device)
    x_wp = wp.array(np.ascontiguousarray(x), dtype=wp.vec3, device=device)

    e1 = tree.query_ball_point(q, r, return_length=True)
    g1 = tw.points.query_ball_count(pts_wp, q_wp, r)
    assert int(g1.numpy()[0]) == int(e1)

    e2 = tree.query_ball_point(x, r, return_length=True)
    g2 = tw.points.query_ball_count(pts_wp, x_wp, r)
    assert np.array_equal(g2.numpy().astype(np.int64), np.asarray(e2, dtype=np.int64))


def test_query_ball_empty_tree(device: str):
    pts_wp = wp.empty(0, dtype=wp.vec3, device=device)
    z = np.zeros(3, dtype=np.float32)
    zi = wp.array(np.zeros((2, 3), dtype=np.float32), dtype=wp.vec3, device=device)

    gi, gd = tw.points.query_ball(pts_wp, _vec3(z), 1.0)
    assert gi.shape == (0,) and gd.shape == (0,)

    gli, gld = tw.points.query_ball(pts_wp, zi, 1.0)
    assert len(gli) == 2 and len(gld) == 2
    assert gli[0].shape == (0,) and gli[1].shape == (0,)
    assert gld[0].shape == (0,) and gld[1].shape == (0,)

    zq = wp.array(np.ascontiguousarray(z[None, :]), dtype=wp.vec3, device=device)
    assert int(tw.points.query_ball_count(pts_wp, zq, r=1.0).numpy()[0]) == 0
    assert np.array_equal(
        tw.points.query_ball_count(pts_wp, zi, r=1.0).numpy(),
        np.zeros(2, dtype=np.int32),
    )
