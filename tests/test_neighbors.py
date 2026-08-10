"""
Regression tests for ``triwarp.neighbors`` ball / k-nearest query APIs.

Against SciPy ``KDTree`` (BVH and HashGrid backends), and ``igl.knn`` as a second exact k-NN.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from typing import Literal

import igl
import numpy as np
import open3d as o3d
import pytest
import trimesh as tm
import warp as wp
from scipy.spatial import KDTree

import triwarp as tw
from triwarp.kernels import neighbors as kernel_neighbors


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
@pytest.mark.parametrize("k", [1, 7, 64])
@pytest.mark.parity("query_bvh_nearest_k1", "open3d")
@pytest.mark.parity("query_bvh_nearest_k7", "open3d")
@pytest.mark.parity("query_bvh_nearest_k64", "open3d")
def test_query_nearest_matches_open3d(
    device: str, backend: Literal["bvh", "hashgrid"], k: int
) -> None:
    """
    Class A, against the third exact k-NN: ``o3d.core.nns.NearestNeighborSearch.knn_search``.

    Open3D's batched tensor search (not the legacy ``KDTreeFlann`` per-query loop) returns
    ``(n_queries, k)`` indices sorted by distance in ``KDTree``'s layout, plus **squared**
    distances -- the square root is the named transform that makes the distance half class B on
    its own; the index half needs none. The cloud is random in a box, so no two points tie in
    ``float32`` distance from a query and the index comparison is exact.
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
    query_indices_wp, query_distances_wp = query_nearest(points_wp, queries_wp, k=k)

    nns_o3d = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(points))
    assert nns_o3d.knn_index()
    indices_o3d, squared_o3d = nns_o3d.knn_search(o3d.core.Tensor(queries), k)

    assert np.array_equal(
        query_indices_wp.numpy().reshape(queries.shape[0], k), indices_o3d.numpy()
    )
    assert np.allclose(
        query_distances_wp.numpy().reshape(queries.shape[0], k),
        np.sqrt(squared_o3d.numpy()),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
@pytest.mark.parity("query_bvh_ball", "open3d")
def test_query_ball_matches_open3d(device: str, backend: Literal["bvh", "hashgrid"]) -> None:
    """
    Class A on the neighbour sets: ``fixed_radius_search`` against the ``*_with_offsets`` form.

    Open3D returns the same CSR-like ``(indices, distances, offsets)`` triple triwarp's offsets
    form does, with squared distances. Per-query neighbour *sets* are compared (order within a
    radius query is not part of either contract), and the counts vector is compared exactly. The
    cloud is random, so no point sits at exactly the radius and the two libraries' boundary rules
    (Open3D's radius searches are exclusive at exactly ``r``; triwarp's inclusive) cannot differ.
    """
    rng = np.random.default_rng(3)
    points = rng.random((400, 3)) * 3.0
    queries = rng.random((60, 3)) * 3.0
    radius = 0.4

    points_wp = wp.array(
        np.ascontiguousarray(points, dtype=np.float32), dtype=wp.vec3, device=device
    )
    queries_wp = wp.array(
        np.ascontiguousarray(queries, dtype=np.float32), dtype=wp.vec3, device=device
    )
    query_ball_with_offsets = (
        tw.neighbors.query_bvh_ball_with_offsets
        if backend == "bvh"
        else tw.neighbors.query_hashgrid_ball_with_offsets
    )
    neighbors_wp, _distances_wp, offsets_wp = query_ball_with_offsets(points_wp, queries_wp, radius)

    nns_o3d = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(points))
    assert nns_o3d.fixed_radius_index(radius)
    indices_o3d, _squared_o3d, offsets_o3d = nns_o3d.fixed_radius_search(
        o3d.core.Tensor(queries), radius
    )
    indices_o3d = indices_o3d.numpy()
    offsets_o3d = offsets_o3d.numpy()

    neighbors_np = neighbors_wp.numpy()
    starts_np = offsets_wp.numpy()  # per-query slice starts; the last slice ends at the total
    ends_np = np.concatenate([starts_np[1:], [neighbors_np.shape[0]]])
    assert np.array_equal(ends_np - starts_np, np.diff(offsets_o3d))
    for query_index in range(queries.shape[0]):
        set_wp = set(neighbors_np[starts_np[query_index] : ends_np[query_index]])
        set_o3d = set(indices_o3d[offsets_o3d[query_index] : offsets_o3d[query_index + 1]])
        assert set_wp == set_o3d
    assert neighbors_np.shape[0] > 0  # non-vacuous: the radius actually finds neighbours


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
@pytest.mark.parametrize("k", sorted(kernel_neighbors.KNN_ROW_BUCKETS))
def test_query_nearest_ties(device: str, backend: Literal["bvh", "hashgrid"], k: int):
    """
    Class B (reduction: which of two equidistant points wins a slot is not specified).

    On an integer lattice a query has dozens of *exactly* tied neighbours, so the row's insert
    order is exercised rather than assumed — a shift chain that mishandles a run of equal distances
    keeps a farther point and reports a distance ``KDTree`` does not, which no random cloud reveals.
    Half the queries sit on lattice points (maximal ties), half at cell centres. ``k`` runs over
    every ``KNN_ROW_BUCKETS`` size because the register-row carry is written out inline once per
    bucket (see the comment above ``_bvh_nearest_row_kernel``), so each copy meets the lattice here.

    The ``k`` distances must match ``KDTree`` exactly and every returned index must actually sit at
    the distance reported for it; the *identity* of a tied neighbour is not compared, because both
    choices are correct answers — the same thing
    [`query_bvh_nearest`][triwarp.neighbors.query_bvh_nearest] tells callers.
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
    # ``k == 1`` returns length-m 1D arrays on both sides; reshape so one comparison covers all k.
    rows = (queries.shape[0], k)
    query_distances_np = np.reshape(KDTree(lattice).query(queries, k=k)[0], rows)

    if k > 1:
        # Non-vacuity: the fixture must actually put runs of equal float32 distances in the rows,
        # or the insert's tie handling is never exercised and this is a plain smoke test.
        assert np.any(query_distances_np[:, 1:] == query_distances_np[:, :-1])
    assert np.allclose(
        query_distances_wp.numpy().reshape(rows), query_distances_np, rtol=1e-5, atol=1e-5
    )
    gathered = np.linalg.norm(
        lattice[query_indices_wp.numpy().reshape(rows)] - queries[:, None, :], axis=-1
    )
    assert np.allclose(gathered, query_distances_np, rtol=1e-5, atol=1e-5)


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


def _geodesic_ball_neighborhoods_oracle(
    vertices_np: np.ndarray, faces_np: np.ndarray, radius: float, min_count: int = 6
) -> tuple[list[list[int]], np.ndarray]:
    """
    Host NumPy reference (the original ``_geodesic_ball_neighborhoods``) used as the oracle.

    Returns the per-vertex collected lists and the reference-neighbor array.
    """
    n = vertices_np.shape[0]
    adjacency: list[list[int]] = [[] for _ in range(n)]
    seen: set[tuple[int, int]] = set()
    for tri in faces_np:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            a, b = int(a), int(b)
            key = (a, b) if a < b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            adjacency[a].append(b)
            adjacency[b].append(a)
    reference = np.array(
        [min(adjacency[i]) if adjacency[i] else i for i in range(n)], dtype=np.int32
    )
    per_vertex: list[list[int]] = []
    for i in range(n):
        center = vertices_np[i]
        visited = {i}
        queue = deque([i])
        collected: list[int] = []
        extras: list[tuple[float, int]] = []
        while queue:
            current = queue.popleft()
            collected.append(current)
            for neighbor in adjacency[current]:
                if neighbor in visited:
                    continue
                distance = float(np.linalg.norm(center - vertices_np[neighbor]))
                if distance < radius:
                    queue.append(neighbor)
                elif len(collected) < min_count:
                    heapq.heappush(extras, (distance, neighbor))
                visited.add(neighbor)
        while extras and len(collected) < min_count:
            _, cand = heapq.heappop(extras)
            collected.append(cand)
            for neighbor in adjacency[cand]:
                if neighbor in visited:
                    continue
                distance = float(np.linalg.norm(center - vertices_np[neighbor]))
                heapq.heappush(extras, (distance, neighbor))
                visited.add(neighbor)
        per_vertex.append(collected)
    return per_vertex, reference


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus"])
def test_geodesic_ball_neighborhoods(mesh_name: str, request: pytest.FixtureRequest) -> None:
    """On-device geodesic balls match the NumPy/libigl oracle (per-row set equality)."""
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int32)

    vertices_wp = wp.array(vertices_np.astype(np.float32), dtype=wp.vec3, device=mesh_wp.device)
    faces_wp = wp.array(mesh_wp.indices, dtype=wp.int32, device=mesh_wp.device)

    radius = 3.0 * tw.edges.mean_edge_length(vertices_wp, faces_wp)
    neighbor_indices_wp, offsets_wp, reference_wp = tw.neighbors.geodesic_ball(
        vertices_wp, faces_wp, radius
    )
    per_vertex_oracle, reference_oracle = _geodesic_ball_neighborhoods_oracle(
        vertices_np, faces_np, radius
    )

    assert np.array_equal(reference_wp.numpy(), reference_oracle)

    neighbor_indices = neighbor_indices_wp.numpy()
    offsets = offsets_wp.numpy()
    total = neighbor_indices.shape[0]
    n = vertices_np.shape[0]
    for i in range(n):
        start = int(offsets[i])
        end = int(offsets[i + 1]) if i + 1 < n else total
        neighbors_wp = {int(x) for x in neighbor_indices[start:end]}
        assert neighbors_wp == set(per_vertex_oracle[i]), f"vertex {i} neighborhood differs"


def test_geodesic_ball_neighborhoods_overflow_warns() -> None:
    """A neighborhood exceeding the fixed 512 cap clamps (does not crash) and warns."""
    # A subdivided icosphere has > 512 vertices; a radius covering the whole mesh makes every
    # vertex's geodesic ball the entire connected component, exceeding the fixed scratch capacity.
    mesh_tm = tm.creation.icosphere(subdivisions=4)
    assert mesh_tm.vertices.shape[0] > 512
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    vertices_wp = wp.array(
        np.array(mesh_tm.vertices, dtype=np.float32), dtype=wp.vec3, device=device
    )
    faces_wp = wp.array(np.array(mesh_tm.faces, dtype=np.int32).reshape(-1), device=device)
    radius = 100.0 * float(mesh_tm.scale)

    with pytest.warns(UserWarning, match="capacity breaches"):
        neighbor_indices_wp, offsets_wp, _ = tw.neighbors.geodesic_ball(
            vertices_wp, faces_wp, radius
        )
    # Clamped, not crashed: every per-vertex count fits within the fixed capacity.
    counts = np.diff(np.append(offsets_wp.numpy(), neighbor_indices_wp.shape[0]))
    assert counts.max() <= 512
