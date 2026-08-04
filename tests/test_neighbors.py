"""
Regression tests for ``triwarp.neighbors`` ball / k-nearest query APIs.

Against SciPy ``KDTree`` (BVH and HashGrid backends), and ``igl.knn`` as a second exact k-NN.
"""

from __future__ import annotations

import math
from typing import Literal

import igl
import numpy as np
import pytest
import warp as wp
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
    query_ball = (
        tw.neighbors.query_bvh_ball if backend == "bvh" else tw.neighbors.query_hashgrid_ball
    )

    query_indices_wp, query_distances_wp = query_ball(
        points_wp, query_wp, radius, return_sorted=True
    )

    assert np.array_equal(query_indices_wp.numpy(), query_indices_np)
    assert np.allclose(query_distances_wp.numpy(), query_distances_np, rtol=1e-5, atol=1e-5)

    query_indices_unsorted_wp, query_distances_unsorted_wp = query_ball(
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


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
def test_query_ball_empty_ball(device: str, backend: Literal["bvh", "hashgrid"]):
    rng = np.random.default_rng(0)
    points = rng.random((50, 3), dtype=np.float32) * 2.0
    query = np.array([10.0, 10.0, 10.0], dtype=np.float32)
    radius = 0.5

    points_wp = wp.array(np.ascontiguousarray(points), dtype=wp.vec3, device=device)
    query_wp = wp.vec3(query[0], query[1], query[2])
    query_ball = (
        tw.neighbors.query_bvh_ball if backend == "bvh" else tw.neighbors.query_hashgrid_ball
    )

    query_indices_wp, query_distances_wp = query_ball(
        points_wp, query_wp, radius, return_sorted=True
    )
    assert query_indices_wp.shape == (0,)
    assert query_distances_wp.shape == (0,)


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
@pytest.mark.parity("query_bvh_ball", "scipy")
def test_query_ball_batch(device: str, backend: Literal["bvh", "hashgrid"]):
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
        for indices, query in zip(query_indices_np, queries, strict=False)
    ]
    orders = [np.argsort(distances) for distances in query_distances_np]
    query_indices_np = [
        indices[order] for indices, order in zip(query_indices_np, orders, strict=False)
    ]
    query_distances_np = [
        distances[order] for distances, order in zip(query_distances_np, orders, strict=False)
    ]

    points_wp = wp.array(np.ascontiguousarray(points), dtype=wp.vec3, device=device)
    query_wp = wp.array(np.ascontiguousarray(queries), dtype=wp.vec3, device=device)
    query_ball = (
        tw.neighbors.query_bvh_ball if backend == "bvh" else tw.neighbors.query_hashgrid_ball
    )

    query_indices_wp, query_distances_wp = query_ball(
        points_wp, query_wp, radius, return_sorted=True
    )

    for single_query_indices_wp, single_query_indices_np in zip(
        query_indices_wp, query_indices_np, strict=False
    ):
        assert np.array_equal(single_query_indices_wp.numpy(), single_query_indices_np)
    for single_query_distances_wp, single_query_distances_np in zip(
        query_distances_wp, query_distances_np, strict=False
    ):
        assert np.allclose(
            single_query_distances_wp.numpy(), single_query_distances_np, rtol=1e-5, atol=1e-5
        )

    query_indices_unsorted_wp, query_distances_unsorted_wp = query_ball(
        points_wp, query_wp, radius, return_sorted=False
    )
    for single_query_indices_unsorted_wp, single_query_indices_wp in zip(
        query_indices_unsorted_wp, query_indices_wp, strict=False
    ):
        assert np.array_equal(
            np.sort(single_query_indices_unsorted_wp.numpy()),
            np.sort(single_query_indices_wp.numpy()),
        )
    for single_query_distances_unsorted_wp, single_query_distances_wp in zip(
        query_distances_unsorted_wp, query_distances_wp, strict=False
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
    query_ball = (
        tw.neighbors.query_bvh_ball if backend == "bvh" else tw.neighbors.query_hashgrid_ball
    )

    indices, distances = query_ball(empty_points_wp, query_wp, radius)
    assert indices.shape == (0,)
    assert distances.shape == (0,)

    indices, distances = query_ball(empty_points_wp, queries_wp, radius)
    assert len(indices) == queries_wp.shape[0]
    assert len(distances) == queries_wp.shape[0]
    for single_indices, single_distances in zip(indices, distances, strict=False):
        assert single_indices.shape == (0,)
        assert single_distances.shape == (0,)

    indices, distances = query_ball(points_wp, empty_queries_wp, radius)
    assert len(indices) == 0
    assert len(distances) == 0


def test_knn_initial_radius_matches_uniform_density(device: str):
    rng = np.random.default_rng(3)
    points = rng.random((4000, 3), dtype=np.float32)
    points_wp = wp.array(np.ascontiguousarray(points), dtype=wp.vec3, device=device)

    for k in (1, 8):
        radius_wp = tw.neighbors.knn_initial_radius(points_wp, k)
        # Expected count in a ball of that radius under the uniform model that defines it.
        volume_np = float(np.prod(points.max(axis=0) - points.min(axis=0)))
        expected_np = (3.0 * (k / 4000) * volume_np / (4.0 * math.pi)) ** (1.0 / 3.0)
        assert np.isclose(radius_wp, expected_np, rtol=1e-6)
        # It really does find about k neighbours: the median count should be within a factor of a
        # few of k, which is what makes a single scan the common case.
        counts_np = np.asarray(
            KDTree(points).query_ball_point(points[:200], radius_wp, return_length=True)
        )
        assert k <= np.median(counts_np) <= 12 * k + 12


def test_knn_initial_radius_degenerate_clouds(device: str):
    """Flat, collinear, coincident and ``k >= n`` clouds all get a usable (or infinite) radius."""
    rng = np.random.default_rng(4)
    planar_np = rng.random((500, 3), dtype=np.float32)
    planar_np[:, 2] = 0.0
    collinear_np = np.zeros((500, 3), dtype=np.float32)
    collinear_np[:, 0] = np.linspace(0.0, 1.0, 500)
    coincident_np = np.zeros((10, 3), dtype=np.float32)

    def radius_of(points_np: np.ndarray, k: int = 4) -> float:
        return tw.neighbors.knn_initial_radius(
            wp.array(np.ascontiguousarray(points_np), dtype=wp.vec3, device=device), k
        )

    # d = 2: pi r^2 = (k / n) * area
    area_np = float(np.prod(planar_np[:, :2].max(axis=0) - planar_np[:, :2].min(axis=0)))
    assert np.isclose(radius_of(planar_np), math.sqrt((4 / 500) * area_np / math.pi), rtol=1e-6)
    # d = 1: 2 r = (k / n) * length
    assert np.isclose(radius_of(collinear_np), 0.5 * (4 / 500) * 1.0, rtol=1e-5)
    # Degenerate box and k >= n both mean "one complete scan".
    assert radius_of(coincident_np) == math.inf
    assert radius_of(planar_np, k=500) == math.inf
    assert tw.neighbors.knn_initial_radius(wp.empty(0, dtype=wp.vec3, device=device), 1) == math.inf


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
@pytest.mark.parametrize("k", [1, 3, 40])
@pytest.mark.parametrize("max_radius", [math.inf, 0.5, 1.0])
@pytest.mark.parity("query_bvh_nearest_k1", "scipy")
@pytest.mark.parity("query_hashgrid_nearest_k1", "scipy")
@pytest.mark.parity("bvh_from_points", "scipy")
def test_query_nearest_single(
    device: str, backend: Literal["bvh", "hashgrid"], k: int, max_radius: float
):
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

    query_nearest = (
        tw.neighbors.query_bvh_nearest if backend == "bvh" else tw.neighbors.query_hashgrid_nearest
    )
    query_indices_wp, query_distances_wp = query_nearest(
        points_wp, query_wp, k=k, max_radius=max_radius
    )

    assert np.array_equal(query_indices_wp.numpy(), query_indices_np)
    assert np.allclose(query_distances_wp.numpy(), query_distances_np, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
@pytest.mark.parametrize("k", [1, 3, 10])
@pytest.mark.parametrize("max_radius", [math.inf, 0.5, 1.0])
@pytest.mark.parity("query_bvh_nearest_k7", "scipy")
@pytest.mark.parity("query_hashgrid_nearest_k7", "scipy")
def test_query_nearest_batch(
    device: str, backend: Literal["bvh", "hashgrid"], k: int, max_radius: float
):
    rng = np.random.default_rng(0)
    points = rng.random((50, 3), dtype=np.float32) * 2.0
    kdtree = KDTree(points)
    queries = points[[10, 20, 30]] + 0.5

    points_wp = wp.array(np.ascontiguousarray(points), dtype=wp.vec3, device=device)
    query_wp = wp.array(np.ascontiguousarray(queries), dtype=wp.vec3, device=device)

    query_distances_np, query_indices_np = kdtree.query(
        queries, k=k, distance_upper_bound=max_radius
    )
    query_indices_np = np.asarray(query_indices_np)
    query_indices_np[query_indices_np == len(points)] = -1
    query_distances_np = np.asarray(query_distances_np)

    query_nearest = (
        tw.neighbors.query_bvh_nearest if backend == "bvh" else tw.neighbors.query_hashgrid_nearest
    )
    query_indices_wp, query_distances_wp = query_nearest(
        points_wp, query_wp, k=k, max_radius=max_radius
    )

    assert np.array_equal(query_indices_wp.numpy(), query_indices_np)
    assert np.allclose(query_distances_wp.numpy(), query_distances_np, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
@pytest.mark.parametrize("k", [1, 4, 8, 9, 16, 17, 32, 33, 64, 65])
@pytest.mark.parity("query_bvh_nearest_k7", "scipy")
@pytest.mark.parity("query_hashgrid_nearest_k7", "scipy")
@pytest.mark.parity("query_bvh_nearest_k64", "scipy")
def test_query_nearest_row_buckets(device: str, backend: Literal["bvh", "hashgrid"], k: int):
    """
    Class A. Every candidate-row bucket size and one ``k`` past each, against ``KDTree``.

    The row lives in a ``wp.types.vector(length=K)`` register value whose ``K`` is one of
    ``kernels.neighbors.KNN_ROW_BUCKETS``, so a ``k`` that does not equal its bucket keeps ``K``
    neighbours and returns the first ``k``. An off-by-one in that tail is invisible at ``k == K``
    and shows only just above and just below a boundary, which is what this sweep pins. ``k=65``
    is past the largest bucket and exercises the global-memory fallback kernel.

    The cloud is random in a box, so no two points tie in ``float32`` distance from a query and the
    index comparison is exact; ties are covered by
    [`test_query_nearest_ties`][tests.test_neighbors.test_query_nearest_ties].
    """
    rng = np.random.default_rng(11)
    points = rng.random((300, 3), dtype=np.float32) * 5.0
    queries = rng.random((40, 3), dtype=np.float32) * 5.0

    points_wp = wp.array(np.ascontiguousarray(points), dtype=wp.vec3, device=device)
    queries_wp = wp.array(np.ascontiguousarray(queries), dtype=wp.vec3, device=device)
    query_nearest = (
        tw.neighbors.query_bvh_nearest if backend == "bvh" else tw.neighbors.query_hashgrid_nearest
    )
    query_indices_wp, query_distances_wp = query_nearest(points_wp, queries_wp, k=k)
    query_distances_np, query_indices_np = KDTree(points).query(queries, k=k)

    assert np.array_equal(query_indices_wp.numpy(), np.asarray(query_indices_np))
    assert np.allclose(query_distances_wp.numpy(), query_distances_np, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
@pytest.mark.parametrize("k", [1, 7, 64])
@pytest.mark.parity("query_bvh_nearest_k1", "igl")
@pytest.mark.parity("query_bvh_nearest_k7", "igl")
@pytest.mark.parity("query_bvh_nearest_k64", "igl")
@pytest.mark.parity("bvh_from_points", "igl")
def test_query_nearest_matches_igl(device: str, backend: Literal["bvh", "hashgrid"], k: int):
    """
    Class A, against the second exact k-NN. Indices, element-wise, at the three benchmarked ``k``.

    ``igl.knn`` returns ``(n_queries, k)`` ``int64`` neighbour indices sorted by distance -- the
    same layout and the same order as ``KDTree``'s -- so this is a direct comparison and not a set
    one. It is a genuinely independent implementation: an octree walk against triwarp's BVH / hash
    grid and scipy's k-d tree, three different structures for one answer.

    The ``bvh_from_points`` marker rides here for the reason the scipy one does: a structure build
    has no output to compare, so it is validated through the query that consumes it -- and on the
    igl side the octree is quite literally an argument to ``igl.knn``, passed as
    ``*igl.octree(points)[:4]``.

    The cloud is random in a box, so no two points tie in ``float32`` distance from a query and the
    index comparison is exact.
    """
    rng = np.random.default_rng(11)
    points = rng.random((300, 3)) * 5.0
    queries = rng.random((40, 3)) * 5.0

    points_wp = wp.array(
        np.ascontiguousarray(points, dtype=np.float32), dtype=wp.vec3, device=device
    )
    queries_wp = wp.array(
        np.ascontiguousarray(queries, dtype=np.float32), dtype=wp.vec3, device=device
    )
    query_nearest = (
        tw.neighbors.query_bvh_nearest if backend == "bvh" else tw.neighbors.query_hashgrid_nearest
    )
    query_indices_wp, _distances_wp = query_nearest(points_wp, queries_wp, k=k)
    query_indices_igl = igl.knn(queries, points, k, *igl.octree(points)[:4])

    assert np.array_equal(query_indices_wp.numpy().reshape(queries.shape[0], k), query_indices_igl)


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
@pytest.mark.parametrize("k", [8, 32])
def test_query_nearest_ties(device: str, backend: Literal["bvh", "hashgrid"], k: int):
    """
    Class B (reduction: which of two equidistant points wins a slot is not specified).

    On an integer lattice a query has dozens of *exactly* tied neighbours, so the row's insert
    order is exercised rather than assumed — a shift chain that breaks on a run of equal distances
    drops a neighbour and keeps a farther one, which no random cloud reveals. Half the queries sit
    on lattice points (maximal ties), half at cell centres.

    The ``k`` distances must match ``KDTree`` exactly and every returned index must actually sit at
    the distance reported for it; the *identity* of a tied neighbour is not compared, because both
    choices are correct answers.
    """
    axis = np.arange(12, dtype=np.float32)
    lattice = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    rng = np.random.default_rng(12)
    on_lattice = lattice[rng.choice(lattice.shape[0], size=30, replace=False)]
    at_centers = lattice[rng.choice(lattice.shape[0], size=30, replace=False)] + 0.5
    queries = np.ascontiguousarray(np.vstack([on_lattice, at_centers]), dtype=np.float32)

    points_wp = wp.array(np.ascontiguousarray(lattice), dtype=wp.vec3, device=device)
    queries_wp = wp.array(queries, dtype=wp.vec3, device=device)
    query_nearest = (
        tw.neighbors.query_bvh_nearest if backend == "bvh" else tw.neighbors.query_hashgrid_nearest
    )
    query_indices_wp, query_distances_wp = query_nearest(points_wp, queries_wp, k=k)
    query_distances_np, _query_indices_np = KDTree(lattice).query(queries, k=k)

    assert np.allclose(query_distances_wp.numpy(), query_distances_np, rtol=1e-5, atol=1e-5)
    gathered = np.linalg.norm(lattice[query_indices_wp.numpy()] - queries[:, None, :], axis=-1)
    assert np.allclose(gathered, query_distances_wp.numpy(), rtol=1e-5, atol=1e-5)


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
    query_nearest = (
        tw.neighbors.query_bvh_nearest if backend == "bvh" else tw.neighbors.query_hashgrid_nearest
    )

    indices, distances = query_nearest(empty_points_wp, query_wp, k=k)
    assert np.array_equal(indices.numpy(), -np.ones(k))
    assert np.array_equal(distances.numpy(), np.full(k, np.inf))

    indices, distances = query_nearest(empty_points_wp, queries_wp, k=k)
    assert indices.shape == (queries_wp.shape[0], k)
    assert distances.shape == (queries_wp.shape[0], k)

    indices, distances = query_nearest(points_wp, empty_queries_wp, k=k)
    assert indices.shape == (0, k)
    assert distances.shape == (0, k)


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
@pytest.mark.parametrize("k", [1, 3])
def test_query_nearest_initial_radius_invariance(
    device: str, backend: Literal["bvh", "hashgrid"], k: int
):
    """
    ``initial_radius`` is a speed knob: every value must give byte-identical output.

    The three values exercise all three loop paths — the default estimate (usually one certified
    scan), ``inf`` (one forced complete scan), and a radius far too small (repeated geometric
    growth, which is also the case that would double-insert if a scan forgot to reset its row).
    """
    rng = np.random.default_rng(5)
    points = rng.random((400, 3), dtype=np.float32) * 3.0
    queries = rng.random((60, 3), dtype=np.float32) * 3.0
    diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))

    points_wp = wp.array(np.ascontiguousarray(points), dtype=wp.vec3, device=device)
    queries_wp = wp.array(np.ascontiguousarray(queries), dtype=wp.vec3, device=device)
    query_nearest = (
        tw.neighbors.query_bvh_nearest if backend == "bvh" else tw.neighbors.query_hashgrid_nearest
    )

    reference_indices_wp, reference_distances_wp = query_nearest(points_wp, queries_wp, k=k)
    for initial_radius in (math.inf, 1e-6 * diagonal, 0.0):
        indices_wp, distances_wp = query_nearest(
            points_wp, queries_wp, k=k, initial_radius=initial_radius
        )
        assert np.array_equal(indices_wp.numpy(), reference_indices_wp.numpy())
        assert np.array_equal(distances_wp.numpy(), reference_distances_wp.numpy())

    query_distances_np, query_indices_np = KDTree(points).query(queries, k=k)
    assert np.array_equal(reference_indices_wp.numpy(), np.asarray(query_indices_np))
    assert np.allclose(reference_distances_wp.numpy(), query_distances_np, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
def test_query_nearest_clustered(device: str, backend: Literal["bvh", "hashgrid"]):
    """
    Tight clusters in a mostly empty box: the density estimate under-shoots badly on purpose.

    Queries sit in the void between clusters, so the first scans come back empty and the search
    has to grow geometrically. On the hash-grid backend the radius outruns the cell width, which
    is the path that hands off to the exact linear scan.
    """
    rng = np.random.default_rng(6)
    centers = rng.random((6, 3)) * 100.0
    points = np.concatenate([c + rng.normal(scale=0.01, size=(150, 3)) for c in centers])
    points = np.ascontiguousarray(points, dtype=np.float32)
    queries = np.ascontiguousarray(rng.random((80, 3)) * 100.0, dtype=np.float32)

    points_wp = wp.array(points, dtype=wp.vec3, device=device)
    queries_wp = wp.array(queries, dtype=wp.vec3, device=device)
    query_nearest = (
        tw.neighbors.query_bvh_nearest if backend == "bvh" else tw.neighbors.query_hashgrid_nearest
    )

    for k in (1, 5):
        indices_wp, distances_wp = query_nearest(points_wp, queries_wp, k=k)
        query_distances_np, query_indices_np = KDTree(points).query(queries, k=k)
        assert np.array_equal(indices_wp.numpy(), np.asarray(query_indices_np))
        assert np.allclose(distances_wp.numpy(), query_distances_np, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
def test_query_nearest_degenerate_clouds(device: str, backend: Literal["bvh", "hashgrid"]):
    """Coincident, collinear and planar clouds: zero-extent axes must not break the search."""
    rng = np.random.default_rng(7)
    coincident_np = np.full((20, 3), 2.5, dtype=np.float32)
    collinear_np = np.zeros((60, 3), dtype=np.float32)
    collinear_np[:, 0] = np.linspace(-1.0, 1.0, 60)
    planar_np = rng.random((60, 3), dtype=np.float32)
    planar_np[:, 2] = 0.75

    query_nearest = (
        tw.neighbors.query_bvh_nearest if backend == "bvh" else tw.neighbors.query_hashgrid_nearest
    )
    for points in (coincident_np, collinear_np, planar_np):
        # Queries on the cloud and far outside it (the latter has an unbounded complete radius).
        queries = np.ascontiguousarray(np.vstack((points[:5], points[:5] + 40.0)), dtype=np.float32)
        points_wp = wp.array(np.ascontiguousarray(points), dtype=wp.vec3, device=device)
        queries_wp = wp.array(queries, dtype=wp.vec3, device=device)

        indices_wp, distances_wp = query_nearest(points_wp, queries_wp, k=3)
        query_distances_np, _query_indices_np = KDTree(points).query(queries, k=3)
        # Coincident points tie at every slot, so compare the distances (and that the indices are
        # in range and distinct), not the arbitrary index order.
        assert np.allclose(distances_wp.numpy(), query_distances_np, rtol=1e-5, atol=1e-5)
        assert indices_wp.numpy().min() >= 0
        assert all(len(set(row.tolist())) == 3 for row in indices_wp.numpy())


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
def test_query_nearest_prebuilt_index(device: str, backend: Literal["bvh", "hashgrid"]):
    """A hoisted BVH / hash grid gives the same answer as letting the query build its own."""
    rng = np.random.default_rng(8)
    points = rng.random((300, 3), dtype=np.float32) * 2.0
    queries = rng.random((40, 3), dtype=np.float32) * 2.0
    points_wp = wp.array(np.ascontiguousarray(points), dtype=wp.vec3, device=device)
    queries_wp = wp.array(np.ascontiguousarray(queries), dtype=wp.vec3, device=device)

    k = 4
    initial_radius = tw.neighbors.knn_initial_radius(points_wp, k)
    if backend == "bvh":
        query_nearest = tw.neighbors.query_bvh_nearest
        index = {"bvh": tw.neighbors.bvh_from_points(points_wp)}
    else:
        query_nearest = tw.neighbors.query_hashgrid_nearest
        index = {"grid": tw.neighbors.hashgrid_from_points(points_wp, initial_radius)}

    indices_wp, distances_wp = query_nearest(
        points_wp, queries_wp, k=k, initial_radius=initial_radius, **index
    )
    built_indices_wp, built_distances_wp = query_nearest(points_wp, queries_wp, k=k)
    assert np.array_equal(indices_wp.numpy(), built_indices_wp.numpy())
    assert np.array_equal(distances_wp.numpy(), built_distances_wp.numpy())


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
def test_query_nearest_rejects_negative_initial_radius(
    device: str, backend: Literal["bvh", "hashgrid"]
):
    points_wp = wp.array(np.zeros((4, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    query_nearest = (
        tw.neighbors.query_bvh_nearest if backend == "bvh" else tw.neighbors.query_hashgrid_nearest
    )
    with pytest.raises(ValueError, match="initial_radius"):
        query_nearest(points_wp, points_wp, k=1, initial_radius=-1.0)
