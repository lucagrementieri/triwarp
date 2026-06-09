"""
Regression tests for ``triwarp.points`` ball / k-nearest / AABB query APIs
(BVH and HashGrid backends) against SciPy ``KDTree`` or brute-force reference.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import warp as wp
import pytest
from scipy.spatial import KDTree

import triwarp as tw


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
def test_query_ball_single(device: str, backend: Literal["bvh", "hashgrid"]):
    rng = np.random.default_rng(0)
    points = rng.random((50, 3), dtype=np.float32) * 2.0
    kdtree = KDTree(points)
    query = points[0].copy()
    radius = 0.5

    query_indices_np = np.asarray(kdtree.query_ball_point(query, radius, return_sorted=False))
    query_distances_np = np.linalg.norm(points[query_indices_np] - query, axis=1)
    order = np.argsort(query_distances_np)
    query_indices_np = query_indices_np[order]
    query_distances_np = query_distances_np[order]

    points_wp = wp.array(np.ascontiguousarray(points), dtype=wp.vec3, device=device)
    query_wp = wp.vec3(query[0], query[1], query[2])
    query_ball = tw.points.query_bvh_ball if backend == "bvh" else tw.points.query_hashgrid_ball

    query_indices_wp, query_distances_wp = query_ball(points_wp, query_wp, radius, return_sorted=True)

    assert np.array_equal(query_indices_wp.numpy(), query_indices_np)
    assert np.allclose(query_distances_wp.numpy(), query_distances_np, rtol=1e-5, atol=1e-5)

    query_indices_unsorted_wp, query_distances_unsorted_wp = query_ball(
        points_wp, query_wp, radius, return_sorted=False
    )
    assert np.array_equal(np.sort(query_indices_unsorted_wp.numpy()), np.sort(query_indices_wp.numpy()))
    assert np.allclose(
        np.sort(query_distances_unsorted_wp.numpy()),
        np.sort(query_distances_wp.numpy()),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
def test_query_ball_empty_ball(device: str, backend: Literal["bvh", "hashgrid"]):
    rng = np.random.default_rng(0)
    points = rng.random((50, 3), dtype=np.float32) * 2.0
    query = np.array([10.0, 10.0, 10.0], dtype=np.float32)
    radius = 0.5

    points_wp = wp.array(np.ascontiguousarray(points), dtype=wp.vec3, device=device)
    query_wp = wp.vec3(query[0], query[1], query[2])
    query_ball = tw.points.query_bvh_ball if backend == "bvh" else tw.points.query_hashgrid_ball

    query_indices_wp, query_distances_wp = query_ball(points_wp, query_wp, radius, return_sorted=True)
    assert query_indices_wp.shape == (0,) and query_distances_wp.shape == (0,)


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
def test_query_ball_batch(device: str, backend: Literal["bvh", "hashgrid"]):
    rng = np.random.default_rng(1)
    points = rng.random((50, 3), dtype=np.float32) * 3.0
    kdtree = KDTree(points)
    queries = points[[10, 20, 30]].copy()
    radius = 0.5

    query_indices_np = [
        np.asarray(indices) for indices in kdtree.query_ball_point(queries, radius, return_sorted=False)
    ]
    query_distances_np = [
        np.linalg.norm(points[indices] - query, axis=1) for indices, query in zip(query_indices_np, queries)
    ]
    orders = [np.argsort(distances) for distances in query_distances_np]
    query_indices_np = [indices[order] for indices, order in zip(query_indices_np, orders)]
    query_distances_np = [distances[order] for distances, order in zip(query_distances_np, orders)]

    points_wp = wp.array(np.ascontiguousarray(points), dtype=wp.vec3, device=device)
    query_wp = wp.array(np.ascontiguousarray(queries), dtype=wp.vec3, device=device)
    query_ball = tw.points.query_bvh_ball if backend == "bvh" else tw.points.query_hashgrid_ball

    query_indices_wp, query_distances_wp = query_ball(points_wp, query_wp, radius, return_sorted=True)

    for single_query_indices_wp, single_query_indices_np in zip(query_indices_wp, query_indices_np):
        assert np.array_equal(single_query_indices_wp.numpy(), single_query_indices_np)
    for single_query_distances_wp, single_query_distances_np in zip(query_distances_wp, query_distances_np):
        assert np.allclose(
            single_query_distances_wp.numpy(),
            single_query_distances_np,
            rtol=1e-5,
            atol=1e-5,
        )

    query_indices_unsorted_wp, query_distances_unsorted_wp = query_ball(
        points_wp, query_wp, radius, return_sorted=False
    )
    for single_query_indices_unsorted_wp, single_query_indices_wp in zip(query_indices_unsorted_wp, query_indices_wp):
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


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
def test_query_ball_empty(device: str, backend: Literal["bvh", "hashgrid"]):
    rng = np.random.default_rng(0)
    points = rng.random((10, 3), dtype=np.float32)

    points_wp = wp.array(np.ascontiguousarray(points), dtype=wp.vec3, device=device)
    empty_points_wp = wp.empty(0, dtype=wp.vec3, device=device)
    empty_queries_wp = wp.empty(0, dtype=wp.vec3, device=device)
    query_wp = wp.vec3(points[0][0], points[0][1], points[0][2])
    queries_wp = wp.array(np.ascontiguousarray(points[-3:]), dtype=wp.vec3, device=device)
    radius = 0.5
    query_ball = tw.points.query_bvh_ball if backend == "bvh" else tw.points.query_hashgrid_ball

    indices, distances = query_ball(empty_points_wp, query_wp, radius)
    assert indices.shape == (0,) and distances.shape == (0,)

    indices, distances = query_ball(empty_points_wp, queries_wp, radius)
    assert len(indices) == queries_wp.shape[0] and len(distances) == queries_wp.shape[0]
    for single_indices, single_distances in zip(indices, distances):
        assert single_indices.shape == (0,) and single_distances.shape == (0,)

    indices, distances = query_ball(points_wp, empty_queries_wp, radius)
    assert len(indices) == 0 and len(distances) == 0


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
@pytest.mark.parametrize("k", [1, 3, 40])
@pytest.mark.parametrize("max_radius", [math.inf, 0.5, 1.0])
def test_query_nearest_single(device: str, backend: Literal["bvh", "hashgrid"], k: int, max_radius: float):
    rng = np.random.default_rng(0)
    points = rng.random((50, 3), dtype=np.float32) * 4.0
    kdtree = KDTree(points)
    query = points[0].copy()

    points_wp = wp.array(np.ascontiguousarray(points), dtype=wp.vec3, device=device)
    query_wp = wp.vec3(query[0], query[1], query[2])
    query_distances_np, query_indices_np = kdtree.query(query, k=k, distance_upper_bound=max_radius)
    query_indices_np = np.atleast_1d(np.asarray(query_indices_np))
    query_indices_np[query_indices_np == len(points)] = -1
    query_distances_np = np.atleast_1d(np.asarray(query_distances_np))

    query_nearest = tw.points.query_bvh_nearest if backend == "bvh" else tw.points.query_hashgrid_nearest
    query_indices_wp, query_distances_wp = query_nearest(points_wp, query_wp, k=k, max_radius=max_radius)

    assert np.array_equal(query_indices_wp.numpy(), query_indices_np)
    assert np.allclose(query_distances_wp.numpy(), query_distances_np, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
@pytest.mark.parametrize("k", [1, 3, 10])
@pytest.mark.parametrize("max_radius", [math.inf, 0.5, 1.0])
def test_query_nearest_batch(device: str, backend: Literal["bvh", "hashgrid"], k: int, max_radius: float):
    rng = np.random.default_rng(0)
    points = rng.random((50, 3), dtype=np.float32) * 2.0
    kdtree = KDTree(points)
    queries = points[[10, 20, 30]] + 0.5

    points_wp = wp.array(np.ascontiguousarray(points), dtype=wp.vec3, device=device)
    query_wp = wp.array(np.ascontiguousarray(queries), dtype=wp.vec3, device=device)

    query_distances_np, query_indices_np = kdtree.query(queries, k=k, distance_upper_bound=max_radius)
    query_indices_np = np.asarray(query_indices_np)
    query_indices_np[query_indices_np == len(points)] = -1
    query_distances_np = np.asarray(query_distances_np)

    query_nearest = tw.points.query_bvh_nearest if backend == "bvh" else tw.points.query_hashgrid_nearest
    query_indices_wp, query_distances_wp = query_nearest(points_wp, query_wp, k=k, max_radius=max_radius)

    assert np.array_equal(query_indices_wp.numpy(), query_indices_np)
    assert np.allclose(query_distances_wp.numpy(), query_distances_np, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
def test_query_nearest_empty(device: str, backend: Literal["bvh", "hashgrid"]):
    rng = np.random.default_rng(0)
    points = rng.random((10, 3), dtype=np.float32)

    points_wp = wp.array(np.ascontiguousarray(points), dtype=wp.vec3, device=device)
    empty_points_wp = wp.empty(0, dtype=wp.vec3, device=device)
    empty_queries_wp = wp.empty(0, dtype=wp.vec3, device=device)
    query_wp = wp.vec3(points[0][0], points[0][1], points[0][2])
    queries_wp = wp.array(np.ascontiguousarray(points[-3:]), dtype=wp.vec3, device=device)
    k = 2
    query_nearest = tw.points.query_bvh_nearest if backend == "bvh" else tw.points.query_hashgrid_nearest

    indices, distances = query_nearest(empty_points_wp, query_wp, k=k)
    assert np.array_equal(indices.numpy(), -np.ones(k))
    assert np.array_equal(distances.numpy(), np.full(k, np.inf))

    indices, distances = query_nearest(empty_points_wp, queries_wp, k=k)
    assert indices.shape == (queries_wp.shape[0], k)
    assert distances.shape == (queries_wp.shape[0], k)

    indices, distances = query_nearest(points_wp, empty_queries_wp, k=k)
    assert indices.shape == (0, k) and distances.shape == (0, k)


def test_query_bvh_aabb_bounds_with_offsets(device: str) -> None:
    rng = np.random.default_rng(7)
    lower_np = rng.random((8, 3), dtype=np.float32)
    upper_np = lower_np + rng.random((8, 3), dtype=np.float32) * 0.4 + 0.05
    query_lower_np = lower_np[:4].copy()
    query_upper_np = upper_np[:4].copy()

    lower_wp = wp.array(np.ascontiguousarray(lower_np), dtype=wp.vec3, device=device)
    upper_wp = wp.array(np.ascontiguousarray(upper_np), dtype=wp.vec3, device=device)
    query_lower_wp = wp.array(np.ascontiguousarray(query_lower_np), dtype=wp.vec3, device=device)
    query_upper_wp = wp.array(np.ascontiguousarray(query_upper_np), dtype=wp.vec3, device=device)

    bvh = tw.points.bvh_from_bounds(lower_wp, upper_wp)
    indices_wp, offsets_wp, hit_counts_wp = tw.points.query_bvh_aabb_bounds_with_offsets(
        bvh,
        query_lower_wp,
        query_upper_wp,
        max_hits=16,
    )

    indices_np = indices_wp.numpy()
    offsets_np = offsets_wp.numpy()
    hit_counts_np = hit_counts_wp.numpy()
    bounds_np = np.append(offsets_np, indices_np.shape[0])

    for query_idx in range(query_lower_np.shape[0]):
        q_lower = query_lower_np[query_idx]
        q_upper = query_upper_np[query_idx]
        mask_np = np.all(lower_np <= q_upper, axis=1) & np.all(upper_np >= q_lower, axis=1)
        expected_np = np.sort(np.flatnonzero(mask_np).astype(np.int32))
        got_np = np.sort(indices_np[bounds_np[query_idx] : bounds_np[query_idx + 1]])
        assert hit_counts_np[query_idx] == got_np.shape[0]
        assert np.array_equal(got_np, expected_np)
