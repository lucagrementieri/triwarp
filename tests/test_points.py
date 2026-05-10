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


def test_query_ball_single(device: str):
    rng = np.random.default_rng(0)
    points = rng.random((50, 3), dtype=np.float32) * 2.0
    kdtree = KDTree(points)
    query = points[0].copy()
    radius = 0.5

    query_indices_np = np.asarray(
        kdtree.query_ball_point(query, radius, return_sorted=False)
    )
    query_distances_np = np.linalg.norm(points[query_indices_np] - query, axis=1)
    order = np.argsort(query_distances_np)
    query_indices_np = query_indices_np[order]
    query_distances_np = query_distances_np[order]

    points_wp = wp.array(np.ascontiguousarray(points), dtype=wp.vec3, device=device)
    query_wp = wp.vec3(query[0], query[1], query[2])

    query_indices_wp, query_distances_wp = tw.points.query_ball(
        points_wp, query_wp, radius, return_sorted=True
    )

    assert np.array_equal(query_indices_wp.numpy(), query_indices_np)
    assert np.allclose(
        query_distances_wp.numpy(), query_distances_np, rtol=1e-5, atol=1e-5
    )

    query_indices_unsorted_wp, query_distances_unsorted_wp = tw.points.query_ball(
        points_wp, query_wp, radius, return_sorted=False
    )
    assert np.array_equal(
        np.sort(query_indices_unsorted_wp.numpy()), np.sort(query_indices_wp.numpy())
    )
    assert np.allclose(
        np.sort(query_distances_unsorted_wp.numpy()),
        np.sort(query_distances_wp.numpy()),
        rtol=1e-5,
        atol=1e-5,
    )


def test_query_ball_batch(device: str):
    rng = np.random.default_rng(1)
    points = rng.random((50, 3), dtype=np.float32) * 3.0
    kdtree = KDTree(points)
    queries = points[[10, 20, 30]].copy()
    radius = 0.5

    query_indices_np = [
        np.asarray(indices)
        for indices in kdtree.query_ball_point(queries, radius, return_sorted=False)
    ]
    query_distances_np = [
        np.linalg.norm(points[indices] - query, axis=1)
        for indices, query in zip(query_indices_np, queries)
    ]
    orders = [np.argsort(distances) for distances in query_distances_np]
    query_indices_np = [
        indices[order] for indices, order in zip(query_indices_np, orders)
    ]
    query_distances_np = [
        distances[order] for distances, order in zip(query_distances_np, orders)
    ]

    points_wp = wp.array(np.ascontiguousarray(points), dtype=wp.vec3, device=device)
    query_wp = wp.array(np.ascontiguousarray(queries), dtype=wp.vec3, device=device)

    query_indices_wp, query_distances_wp = tw.points.query_ball(
        points_wp, query_wp, radius, return_sorted=True
    )

    for single_query_indices_wp, single_query_indices_np in zip(
        query_indices_wp, query_indices_np
    ):
        assert np.array_equal(single_query_indices_wp.numpy(), single_query_indices_np)
    for single_query_distances_wp, single_query_distances_np in zip(
        query_distances_wp, query_distances_np
    ):
        assert np.allclose(
            single_query_distances_wp.numpy(),
            single_query_distances_np,
            rtol=1e-5,
            atol=1e-5,
        )

    query_indices_unsorted_wp, query_distances_unsorted_wp = tw.points.query_ball(
        points_wp, query_wp, radius, return_sorted=False
    )
    for single_query_indices_unsorted_wp, single_query_indices_wp in zip(
        query_indices_unsorted_wp, query_indices_wp
    ):
        assert np.array_equal(
            np.sort(single_query_indices_unsorted_wp.numpy()),
            np.sort(single_query_indices_wp.numpy()),
        )
    for single_query_distances_unsorted_wp, single_query_distances_wp in zip(
        query_distances_unsorted_wp, query_distances_wp
    ):
        assert np.allclose(
            np.sort(single_query_distances_unsorted_wp.numpy()),
            np.sort(single_query_distances_wp.numpy()),
            rtol=1e-5,
            atol=1e-5,
        )


def test_query_ball_empty(device: str):
    rng = np.random.default_rng(0)
    points = rng.random((10, 3), dtype=np.float32)

    points_wp = wp.array(np.ascontiguousarray(points), dtype=wp.vec3, device=device)
    empty_points_wp = wp.empty(0, dtype=wp.vec3, device=device)
    empty_queries_wp = wp.empty(0, dtype=wp.vec3, device=device)
    query_wp = wp.vec3(points[0][0], points[0][1], points[0][2])
    queries_wp = wp.array(
        np.ascontiguousarray(points[-3:]), dtype=wp.vec3, device=device
    )
    radius = 0.5

    indices, distances = tw.points.query_ball(empty_points_wp, query_wp, radius)
    assert indices.shape == (0,) and distances.shape == (0,)

    indices, distances = tw.points.query_ball(empty_points_wp, queries_wp, radius)
    assert len(indices) == 3 and len(distances) == 3
    for single_indices, single_distances in zip(indices, distances):
        assert single_indices.shape == (0,) and single_distances.shape == (0,)

    indices, distances = tw.points.query_ball(points_wp, empty_queries_wp, radius)
    assert len(indices) == 0 and len(distances) == 0
