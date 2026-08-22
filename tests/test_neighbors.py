"""
Regression tests for ``triwarp.neighbors`` ball / k-nearest query APIs.

Against SciPy ``KDTree`` (BVH and HashGrid backends), and ``igl.knn`` as a second exact k-NN.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from collections.abc import Callable
from typing import Literal

import igl
import numpy as np
import open3d as o3d
import pytest
import trimesh as tm
import warp as wp
from meshlib import mrmeshpy as mm
from scipy.spatial import KDTree

import triwarp as tw
from tests.conversions import meshlib_indices_to_numpy, points_to_meshlib, points_to_open3d
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
    """
    Class B: per-query neighbour lists against ``KDTree.query_ball_point``, sorted by distance.

    scipy returns each list unordered, so the named transform sorts it by distance to match
    ``return_sorted=True``; the ``return_sorted=False`` call is then compared against triwarp's
    own sorted answer, since only the *set* is defined there. Both backends, three queries.
    """
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
def test_query_ball_count_matches_scipy_and_the_list_form(
    device: str, backend: Literal["bvh", "hashgrid"]
) -> None:
    """
    Class A: the per-query counts equal ``KDTree.query_ball_point`` lengths, on both backends.

    Also cross-checked against the ``with_offsets`` form of the same query, since the count is
    exactly what that function's offsets differ by -- a count kernel that disagreed with the
    gather it sizes is the failure worth catching. Measured counts on this cloud: 2, 3, 3, 10, 7,
    so no query is empty and none holds the whole cloud.
    """
    rng = np.random.default_rng(1)
    points_np = rng.random((200, 3), dtype=np.float32) * 3.0
    queries_np = points_np[[10, 20, 30, 55, 120]].copy()
    radius = 0.5

    points_wp = wp.array(np.ascontiguousarray(points_np), dtype=wp.vec3, device=device)
    queries_wp = wp.array(np.ascontiguousarray(queries_np), dtype=wp.vec3, device=device)
    query_ball_count = (
        tw.neighbors.query_bvh_ball_count
        if backend == "bvh"
        else tw.neighbors.query_hashgrid_ball_count
    )
    query_ball_with_offsets = (
        tw.neighbors.query_bvh_ball_with_offsets
        if backend == "bvh"
        else tw.neighbors.query_hashgrid_ball_with_offsets
    )

    counts_wp = query_ball_count(points_wp, queries_wp, radius)
    counts_np = np.array(
        [len(indices) for indices in KDTree(points_np).query_ball_point(queries_np, radius)],
        dtype=np.int32,
    )

    assert counts_np.min() > 0
    assert counts_np.max() < points_np.shape[0]
    assert np.array_equal(counts_wp.numpy(), counts_np)

    indices_wp, _distances_wp, offsets_wp = query_ball_with_offsets(
        points_wp, queries_wp, radius, include_total=True
    )
    assert np.array_equal(np.diff(offsets_wp.numpy()), counts_np)
    assert int(indices_wp.shape[0]) == int(counts_np.sum())


@pytest.mark.parametrize("include_total", [False, True])
def test_query_bvh_aabb_with_offsets_matches_a_brute_force_box_overlap(
    device: str, include_total: bool
) -> None:
    """
    Class A: the broad-phase hits are exactly the boxes overlapping the query cube.

    ``bvh_from_bounds`` plus this query is the one pair in the module that indexes *bounds* rather
    than points, and it has no narrow-phase filter -- so the oracle is the full ``lower <= q + h and
    upper >= q - h`` test on all three axes, and the comparison is exact rather than a superset.

    Both offsets forms are covered, and the length-``m`` one is asserted to be the ``m + 1`` form's
    prefix: they are two views of one scan buffer, so a divergence would mean the slicing is wrong
    rather than the query. Measured 8 hits over 6 queries, 3 of which hit at least one box.
    """
    rng = np.random.default_rng(4)
    lower_np = (rng.random((40, 3)) * 2.0).astype(np.float32)
    upper_np = (lower_np + rng.random((40, 3)) * 0.3).astype(np.float32)
    queries_np = (rng.random((6, 3)) * 2.0).astype(np.float32)
    half_extent = 0.25
    n_queries = queries_np.shape[0]

    bvh = tw.neighbors.bvh_from_bounds(
        wp.array(np.ascontiguousarray(lower_np), dtype=wp.vec3, device=device),
        wp.array(np.ascontiguousarray(upper_np), dtype=wp.vec3, device=device),
    )
    queries_wp = wp.array(np.ascontiguousarray(queries_np), dtype=wp.vec3, device=device)
    indices_wp, offsets_wp = tw.neighbors.query_bvh_aabb_with_offsets(
        bvh, queries_wp, half_extent, include_total=include_total
    )
    indices_np = indices_wp.numpy()
    offsets_np = offsets_wp.numpy()

    assert offsets_np.shape == (n_queries + 1 if include_total else n_queries,)
    assert indices_np.size > 0
    if include_total:
        # The trailing element is the flat length, so no caller has to know it separately.
        assert int(offsets_np[-1]) == indices_np.size
        bounds_np = offsets_np
    else:
        bounds_np = np.append(offsets_np, indices_np.size)
        _indices_wp, total_offsets_wp = tw.neighbors.query_bvh_aabb_with_offsets(
            bvh, queries_wp, half_extent, include_total=True
        )
        assert np.array_equal(offsets_np, total_offsets_wp.numpy()[:-1])

    n_matched = 0
    for query_index, query_np in enumerate(queries_np):
        overlapping_np = np.flatnonzero(
            np.all(
                (lower_np <= query_np + half_extent) & (upper_np >= query_np - half_extent), axis=1
            )
        )
        hits_np = indices_np[bounds_np[query_index] : bounds_np[query_index + 1]]
        assert np.array_equal(np.sort(hits_np), overlapping_np)
        n_matched += overlapping_np.size > 0
    # Three of the six queries hit at least one box here; a run where none did would pass vacuously.
    assert n_matched >= 3


@pytest.mark.parametrize("include_total", [False, True])
def test_query_bvh_aabb_with_offsets_degenerate_inputs(device: str, include_total: bool) -> None:
    """
    An empty query set and a query that hits nothing, both honouring ``include_total``.

    These are the two early returns, and each has to produce the *same* offsets shape the general
    path does: a zero-hit query giving the length-``m`` form under ``include_total=True`` would
    break a caller reading the trailing total. Measured ``[0]`` and ``[0, 0]``.
    """
    lower_np = np.zeros((4, 3), dtype=np.float32)
    upper_np = np.full((4, 3), 0.1, dtype=np.float32)
    bvh = tw.neighbors.bvh_from_bounds(
        wp.array(lower_np, dtype=wp.vec3, device=device),
        wp.array(upper_np, dtype=wp.vec3, device=device),
    )

    empty_indices_wp, empty_offsets_wp = tw.neighbors.query_bvh_aabb_with_offsets(
        bvh, wp.empty(0, dtype=wp.vec3, device=device), 0.25, include_total=include_total
    )
    assert empty_indices_wp.shape == (0,)
    assert empty_offsets_wp.shape == ((1,) if include_total else (0,))
    assert np.array_equal(empty_offsets_wp.numpy(), np.zeros(1 if include_total else 0, np.int32))

    far_wp = wp.array(
        np.array([[99.0, 99.0, 99.0]], dtype=np.float32), dtype=wp.vec3, device=device
    )
    miss_indices_wp, miss_offsets_wp = tw.neighbors.query_bvh_aabb_with_offsets(
        bvh, far_wp, 0.25, include_total=include_total
    )
    assert miss_indices_wp.shape == (0,)
    assert np.array_equal(miss_offsets_wp.numpy(), np.zeros(2 if include_total else 1, np.int32))


@pytest.mark.parity("query_bvh_box", "open3d")
def test_query_bvh_box_matches_exact_containment(device: str) -> None:
    """
    Class A: on a point BVH the hits are exactly the points inside each query box.

    A degenerate leaf bound intersects the query box iff the point is in it, so there is no
    broad-phase superset here and the comparison is an equality against two independent exact
    answers: NumPy's ``lower <= p <= upper`` and open3d's
    ``AxisAlignedBoundingBox.get_point_indices_within_bounding_box``. Both share Warp's
    **inclusive** convention, which is the one thing a caller can get wrong here -- measured on Warp
    1.16 and open3d 0.19, a point exactly on a face is inside the box for both -- so the last case
    below constructs one point on the lower face and one on the upper face rather than trusting
    random data to land there.

    Also pins the two offsets forms against each other, as the uniform-cube sibling's test does.
    """
    rng = np.random.default_rng(11)
    points_np = rng.random((400, 3)).astype(np.float32)
    centers_np = rng.random((12, 3)).astype(np.float32)
    lower_np = np.ascontiguousarray(centers_np - 0.12, dtype=np.float32)
    upper_np = np.ascontiguousarray(centers_np + 0.18, dtype=np.float32)
    n_queries = centers_np.shape[0]

    points_wp = wp.array(np.ascontiguousarray(points_np), dtype=wp.vec3, device=device)
    bvh = tw.neighbors.bvh_from_points(points_wp)
    indices_wp, offsets_wp = tw.neighbors.query_bvh_box(
        bvh,
        wp.array(lower_np, dtype=wp.vec3, device=device),
        wp.array(upper_np, dtype=wp.vec3, device=device),
        include_total=True,
    )
    indices_np = indices_wp.numpy()
    offsets_np = offsets_wp.numpy()
    assert offsets_np.shape == (n_queries + 1,)
    assert int(offsets_np[-1]) == indices_np.size

    cloud_o3d = points_to_open3d(points_np.astype(np.float64))
    n_nonempty = 0
    for query_index in range(n_queries):
        box_lower_np, box_upper_np = lower_np[query_index], upper_np[query_index]
        inside_np = np.flatnonzero(
            np.all((points_np >= box_lower_np) & (points_np <= box_upper_np), axis=1)
        )
        hits_np = np.sort(indices_np[offsets_np[query_index] : offsets_np[query_index + 1]])
        assert np.array_equal(hits_np, inside_np)

        box_o3d = o3d.geometry.AxisAlignedBoundingBox(
            box_lower_np.astype(np.float64), box_upper_np.astype(np.float64)
        )
        assert np.array_equal(
            np.sort(np.asarray(box_o3d.get_point_indices_within_bounding_box(cloud_o3d.points))),
            inside_np,
        )

        n_nonempty += inside_np.size > 0
    # 9 of the 12 boxes hold at least one point here; a run where none did would pass vacuously.
    assert n_nonempty >= 9

    # The length-``m`` form is the same scan buffer's prefix, not a separately computed answer.
    _short_indices_wp, short_offsets_wp = tw.neighbors.query_bvh_box(
        bvh,
        wp.array(lower_np, dtype=wp.vec3, device=device),
        wp.array(upper_np, dtype=wp.vec3, device=device),
    )
    assert np.array_equal(short_offsets_wp.numpy(), offsets_np[:-1])

    # Inclusive on both faces, and an inverted box selects nothing.
    face_np = np.array([[0.0, 0.5, 0.5], [1.0, 0.5, 0.5], [0.5, 0.5, 0.5]], dtype=np.float32)
    face_bvh = tw.neighbors.bvh_from_points(wp.array(face_np, dtype=wp.vec3, device=device))
    face_indices_wp, face_offsets_wp = tw.neighbors.query_bvh_box(
        face_bvh,
        wp.array(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], np.float32), wp.vec3, device=device),
        wp.array(np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], np.float32), wp.vec3, device=device),
        include_total=True,
    )
    assert np.array_equal(np.sort(face_indices_wp.numpy()), np.array([0, 1, 2]))
    assert np.array_equal(face_offsets_wp.numpy(), np.array([0, 3, 3]))


@pytest.mark.parity(
    "query_bvh_box",
    "meshlib",
    benchmarked=False,
    reason="findPointsInBox reports every hit through a Python callback, so a timed row would "
    "price the interpreter rather than the search; it is exact as a correctness oracle, and open3d "
    "carries the timed row for this group instead.",
)
def test_query_bvh_box_matches_meshlib(device: str) -> None:
    """
    Class A: the same hit set as ``findPointsInBox``, which is the third exact box query.

    Separate from the open3d comparison because meshlib's is the one that cannot be benchmarked --
    it reports each hit through a ``func_void_from_Id_VertTag_Vector3_float`` callback, so a timed
    row would measure the interpreter. As a correctness oracle it is exact, and it agrees on the
    inclusive-face convention: the second box below has a point on each of its faces.
    """
    rng = np.random.default_rng(17)
    points_np = rng.random((200, 3)).astype(np.float32)
    lower_np = np.array([[0.2, 0.2, 0.2], [0.0, 0.0, 0.0]], dtype=np.float32)
    upper_np = np.array([[0.6, 0.7, 0.55], [1.0, 1.0, 1.0]], dtype=np.float32)

    bvh = tw.neighbors.bvh_from_points(
        wp.array(np.ascontiguousarray(points_np), dtype=wp.vec3, device=device)
    )
    indices_wp, offsets_wp = tw.neighbors.query_bvh_box(
        bvh,
        wp.array(lower_np, dtype=wp.vec3, device=device),
        wp.array(upper_np, dtype=wp.vec3, device=device),
        include_total=True,
    )
    indices_np, offsets_np = indices_wp.numpy(), offsets_wp.numpy()

    def collect_into(sink: list[int]) -> Callable[[int, object], None]:
        """Each hit arrives as ``(VertId, Vector3f)``, so the sink cannot be ``list.append``."""
        return lambda vertex_id, _position: sink.append(int(vertex_id))

    cloud_ml = points_to_meshlib(points_np)
    for query_index in range(lower_np.shape[0]):
        hits_ml: list[int] = []
        mm.findPointsInBox(
            cloud_ml,
            mm.Box3f(
                mm.Vector3f(*lower_np[query_index].tolist()),
                mm.Vector3f(*upper_np[query_index].tolist()),
            ),
            collect_into(hits_ml),
        )
        hits_wp = indices_np[offsets_np[query_index] : offsets_np[query_index + 1]]
        assert np.array_equal(np.sort(np.asarray(hits_ml, dtype=np.int32)), np.sort(hits_wp))
    # The second box holds the whole cloud, so neither side's answer is empty.
    assert int(offsets_np[-1]) - int(offsets_np[-2]) == points_np.shape[0]


def test_query_bvh_box_degenerate_inputs(device: str) -> None:
    """
    An empty query set, a query that hits nothing, and mismatched corner lengths.

    Not a library comparison: these are this wrapper's own early returns and its one guard. The
    offsets shape has to match the general path in both early returns, since a caller reading the
    trailing total would otherwise index past the end.
    """
    points_wp = wp.array(
        np.zeros((4, 3), dtype=np.float32) + np.array([0.0, 0.0, 0.0]), dtype=wp.vec3, device=device
    )
    bvh = tw.neighbors.bvh_from_points(points_wp)
    empty_wp = wp.empty(0, dtype=wp.vec3, device=device)

    for include_total in (False, True):
        indices_wp, offsets_wp = tw.neighbors.query_bvh_box(
            bvh, empty_wp, empty_wp, include_total=include_total
        )
        assert indices_wp.shape == (0,)
        assert offsets_wp.shape == ((1,) if include_total else (0,))

        far_lower_wp = wp.array(np.full((1, 3), 99.0, np.float32), dtype=wp.vec3, device=device)
        far_upper_wp = wp.array(np.full((1, 3), 100.0, np.float32), dtype=wp.vec3, device=device)
        miss_indices_wp, miss_offsets_wp = tw.neighbors.query_bvh_box(
            bvh, far_lower_wp, far_upper_wp, include_total=include_total
        )
        assert miss_indices_wp.shape == (0,)
        assert np.array_equal(
            miss_offsets_wp.numpy(), np.zeros(2 if include_total else 1, np.int32)
        )

    with pytest.raises(ValueError, match="same length"):
        tw.neighbors.query_bvh_box(
            bvh, wp.array(np.zeros((2, 3), np.float32), wp.vec3, device=device), empty_wp
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
    """
    Class A: k-nearest indices and distances against ``KDTree.query``, over k and ``max_radius``.

    Exact rather than set-compared because the clouds are random and tie-free, so the k-th
    neighbour is unambiguous; [`test_query_nearest_ties`] handles the constructed case where it
    is not. The ``max_radius`` axis is what pins the sentinel written for a query with fewer
    than k neighbours in range.
    """
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
    """
    Class A: the batched form, with queries offset off the cloud so some fall short of k.

    The ``+ 0.5`` displacement is deliberate -- querying *at* a data point makes the first
    neighbour trivially itself at distance zero, which hides an off-by-one in the heap.
    """
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


@pytest.mark.parity("nearest_neighbor_distance", "open3d")
def test_nearest_neighbor_distance_matches_open3d(device: str) -> None:
    """
    Class A: element for element against ``PointCloud.compute_nearest_neighbor_distance``.

    The distance to the nearest *other* point, so the self-match in slot 0 of the k-NN table must be
    skipped -- an implementation returning column 0 would report all zeros, which is what the
    strictly-positive assert below excludes. It also pins the strided-column read: column 1 of an
    ``(n, 2)`` table is a non-contiguous view, and the value comparison here is the check that the
    copy out of it is the column and not the buffer's leading entries.

    Also asserted, since no reference has an opinion on it: the mean of this array is a *scale*, so
    scaling the cloud must scale it by the same factor (both ``reconstruction`` call sites divide a
    length by it).
    """
    rng = np.random.default_rng(11)
    points_np = rng.random((600, 3))

    distance_o3d = np.asarray(points_to_open3d(points_np).compute_nearest_neighbor_distance())

    points_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=device)
    distance_wp = tw.neighbors.nearest_neighbor_distance(points_wp)

    assert distance_o3d.min() > 0.0  # non-vacuity: a random cloud has no coincident points
    assert np.allclose(distance_wp.numpy(), distance_o3d, rtol=1e-5, atol=1e-5)

    scaled_wp = wp.array((3.0 * points_np).astype(np.float32), dtype=wp.vec3, device=device)
    scaled_distance_wp = tw.neighbors.nearest_neighbor_distance(scaled_wp)
    assert np.allclose(scaled_distance_wp.numpy(), 3.0 * distance_wp.numpy(), rtol=1e-5, atol=1e-5)


@pytest.mark.parity("nearest_neighbor_distance", "meshlib")
def test_nearest_neighbor_distance_matches_meshlib(device: str) -> None:
    """
    Class B (an index where triwarp returns a length): the same neighbour, exactly.

    ``findNClosestPointsPerPoint(cloud, 1)`` returns the *index* of each point's closest **other**
    point -- self is excluded, so the ``k=2``-and-drop-column-0 transform triwarp's own
    implementation needs has no counterpart here -- and the distance is then taken from the index.
    Against a ``float64`` scipy oracle the reference's own indices reproduce its distances to
    **0.0**, so the only error in the comparison is triwarp's ``float32`` arithmetic.

    **``numNei`` above 1 is a heap, not a sorted list**, which is why the pair is written at 1.
    Measured on this cloud: the ``k`` returned ids are exactly scipy's ``k`` nearest (set equality
    1.0000 at ``k=3``) but only **90.8 %** of the rows come back in decreasing-distance order, and
    the *nearest* is the last entry rather than the first. At ``numNei=2`` the last entry is the
    nearest in 100 % of rows -- a two-element heap is ordered by construction -- so a pair written
    at 2 would pass while resting on an accident of the heap size.
    """
    rng = np.random.default_rng(11)
    points_np = rng.random((600, 3))

    cloud_ml = points_to_meshlib(points_np)
    nearest_ml = meshlib_indices_to_numpy(mm.findNClosestPointsPerPoint(cloud_ml, 1))
    assert nearest_ml.shape == (points_np.shape[0],)
    assert np.all(nearest_ml != np.arange(points_np.shape[0]))  # the closest *other* point
    distance_ml = np.linalg.norm(points_np[nearest_ml] - points_np, axis=1)

    points_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=device)
    distance_wp = tw.neighbors.nearest_neighbor_distance(points_wp)

    assert distance_ml.min() > 0.0  # non-vacuity: a random cloud has no coincident points
    assert np.allclose(distance_wp.numpy(), distance_ml, rtol=1e-5, atol=1e-5)


def test_nearest_neighbor_distance_coincident_and_degenerate(device: str) -> None:
    """
    Not a library comparison: the two cases Open3D answers differently, plus the exact-zero one.

    Two coincident points are at distance zero from each other and both libraries say so, but a
    cloud of fewer than two points has no answer at all -- Open3D reports ``0.0`` there and this
    reports ``inf``, the value its own k-NN uses for a slot it could not fill. The divergence is
    documented rather than papered over, so it is pinned here rather than in the parity test.
    """
    coincident_np = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float32)
    coincident_wp = wp.array(coincident_np, dtype=wp.vec3, device=device)
    distance_wp = tw.neighbors.nearest_neighbor_distance(coincident_wp).numpy()
    assert np.array_equal(distance_wp, np.array([0.0, 0.0, 5.0], dtype=np.float32))

    for n_points in (0, 1):
        sparse_wp = wp.array(
            np.zeros((n_points, 3), dtype=np.float32), dtype=wp.vec3, device=device
        )
        answer_wp = tw.neighbors.nearest_neighbor_distance(sparse_wp).numpy()
        assert answer_wp.shape == (n_points,)
        assert np.all(np.isinf(answer_wp))


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
    """Not a library comparison: a neighborhood past the 512 cap must clamp and warn, not crash."""
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


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
@pytest.mark.parity(
    "query_bvh_ball",
    "meshlib",
    benchmarked=False,
    reason="findPointsInBall reports each neighbour through a Python callback, so a benchmark row "
    "would time 20 000 queries' worth of callback dispatch rather than MeshLib's traversal -- the "
    "same reason section 6 bars a per-vertex Python loop from a benchmark row. There is no batched "
    "radius form: PointsProjector answers k=1 only (and does carry the query_bvh_nearest_k1 row), "
    "and findNClosestPointsPerPoint takes a neighbour count rather than a radius. scipy and open3d "
    "carry the timed rows for this group.",
)
def test_query_ball_matches_meshlib(device: str, backend: Literal["bvh", "hashgrid"]) -> None:
    """
    Class B: ``findPointsInBall`` reports its neighbours through a *callback*, one at a time.

    Two transforms, both forced by the interface. The results arrive by callback rather than as an
    array, so they are accumulated in Python and the answer is a *set* per query -- which is all
    either contract promises for a radius query. And the callback's ``distSq`` is squared, so the
    square root is the second transform, the same one the open3d comparison above makes.

    ``Ball3f`` is likewise built from a centre and a **squared** radius (``radiusSq``), which is the
    field a call written against a plain radius would silently get wrong by a square -- so the
    counts are asserted to be non-trivial rather than merely equal.

    Unlike open3d's ``KDTreeFlann``, MeshLib's ball search is **inclusive at exactly ``r``**, which
    is triwarp's rule too: measured on three points placed at distance exactly 1.0 from the query,
    both return all three. The random cloud here cannot show that, so it is pinned separately in
    the second half of this test.
    """
    rng = np.random.default_rng(3)
    points_np = rng.random((400, 3)) * 3.0
    queries_np = rng.random((60, 3)) * 3.0
    radius = 0.4

    points_wp = wp.array(
        np.ascontiguousarray(points_np, dtype=np.float32), dtype=wp.vec3, device=device
    )
    queries_wp = wp.array(
        np.ascontiguousarray(queries_np, dtype=np.float32), dtype=wp.vec3, device=device
    )
    query_ball = (
        tw.neighbors.query_bvh_ball if backend == "bvh" else tw.neighbors.query_hashgrid_ball
    )
    neighbours_wp, distances_wp = query_ball(points_wp, queries_wp, radius)

    cloud_ml = points_to_meshlib(points_np)
    total_found = 0
    for query_index, query_np in enumerate(queries_np):
        found_ml: list[tuple[int, float]] = []

        def collect(result_ml: object, *_args: object, found=found_ml) -> mm.Processing:
            found.append((int(result_ml.vId), float(result_ml.distSq)))  # type: ignore[attr-defined]
            return mm.Processing.Continue

        ball_ml = mm.Ball3f()
        ball_ml.center = mm.Vector3f(*query_np.tolist())
        ball_ml.radiusSq = radius * radius
        mm.findPointsInBall(cloud_ml, ball_ml, collect)

        indices_ml = np.array([index for index, _distance in found_ml], dtype=np.int32)
        squared_ml = np.array([distance for _index, distance in found_ml])
        order_ml = np.argsort(indices_ml)
        order_wp = np.argsort(neighbours_wp[query_index].numpy())

        assert set(neighbours_wp[query_index].numpy().tolist()) == set(indices_ml.tolist())
        assert np.allclose(
            distances_wp[query_index].numpy()[order_wp],
            np.sqrt(squared_ml[order_ml]),
            rtol=1e-5,
            atol=1e-5,
        )
        total_found += indices_ml.size
    assert total_found > queries_np.shape[0]  # non-vacuity: the balls are not empty

    # The boundary rule, which the random cloud cannot reach: both are inclusive at exactly r.
    tie_np = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.0, 0.0]])
    tie_wp = wp.array(np.ascontiguousarray(tie_np, dtype=np.float32), dtype=wp.vec3, device=device)
    tie_indices_wp, _tie_distances_wp = query_ball(tie_wp, wp.vec3(0.0, 0.0, 0.0), 1.0)
    tie_found: list[int] = []

    def collect_tie(result_ml: object, *_args: object) -> mm.Processing:
        tie_found.append(int(result_ml.vId))  # type: ignore[attr-defined]
        return mm.Processing.Continue

    tie_ball_ml = mm.Ball3f()
    tie_ball_ml.center = mm.Vector3f(0.0, 0.0, 0.0)
    tie_ball_ml.radiusSq = 1.0
    mm.findPointsInBall(points_to_meshlib(tie_np), tie_ball_ml, collect_tie)
    assert sorted(tie_found) == [0, 1, 2, 3]
    assert sorted(tie_indices_wp.numpy().tolist()) == [0, 1, 2, 3]


@pytest.mark.parametrize("k", [1, 7])
@pytest.mark.parity("query_bvh_nearest_k1", "meshlib")
@pytest.mark.parity(
    "query_bvh_nearest_k7",
    "meshlib",
    benchmarked=False,
    reason="findNClosestPointsPerPoint takes no query set -- it answers the cloud against itself, "
    "so at the benchmark's 20 000 displaced queries against 35 947 vertices it would be timing a "
    "different amount of work than every other row in the group. MeshLib's only batched form that "
    "accepts a query cloud is PointsProjector, which is k=1 and carries the "
    "query_bvh_nearest_k1 row; findFewClosestPoints is per query and would time a Python loop. "
    "scipy, igl and open3d carry the timed rows at k=7 and k=64.",
)
def test_query_nearest_matches_meshlib(device: str, k: int) -> None:
    """
    Class B: ``findNClosestPointsPerPoint`` is self-excluding, unordered, and cloud-against-itself.

    Three transforms, and each is a property of the reference rather than a formatting choice. It
    takes only a ``PointCloud`` and a neighbour count -- there is no separate query set -- so the
    comparison is the cloud against itself; it **excludes** each point from its own neighbour list,
    so triwarp is asked for ``k + 1`` and its first column (the point itself, at distance 0) is
    dropped; and its rows are *not* distance-ordered, so the two are compared as sets per row.

    The flat ``Buffer_VertId`` is reshaped to ``(n_points, k)`` -- there is no shape on the returned
    buffer, only ``n * k`` ids in row-major order, so a wrong ``k`` reshapes silently into a
    plausible-looking answer. The self-column assert is what pins that: triwarp's own first column
    must be the identity before anything is dropped.

    This is the batched form deliberately. ``findFewClosestPoints`` is per query and a Python loop
    over it would time the loop, which is the same reason section 6 prefers ``mrmeshnumpy``'s
    batched curvature calls.
    """
    rng = np.random.default_rng(11)
    points_np = rng.random((300, 3)) * 5.0
    points_wp = wp.array(
        np.ascontiguousarray(points_np, dtype=np.float32), dtype=wp.vec3, device=device
    )

    neighbours_ml = np.array(
        [
            int(vertex_ml)
            for vertex_ml in mm.findNClosestPointsPerPoint(points_to_meshlib(points_np), k)
        ]
    ).reshape(points_np.shape[0], k)

    indices_wp, _distances_wp = tw.neighbors.query_bvh_nearest(points_wp, points_wp, k=k + 1)
    indices_np = indices_wp.numpy().reshape(points_np.shape[0], k + 1)

    assert np.array_equal(indices_np[:, 0], np.arange(points_np.shape[0]))  # the self column
    assert neighbours_ml.shape == (points_np.shape[0], k)
    for row_wp, row_ml in zip(indices_np[:, 1:], neighbours_ml, strict=True):
        assert set(row_wp.tolist()) == set(row_ml.tolist())

    if k != 1:
        return
    # The other batched form, and the only one that accepts a *query* cloud: exact at k=1, indices
    # and distances alike, which is what the query_bvh_nearest_k1 benchmark row times. The cloud
    # must be held in a name -- ``setPointCloud`` stores a raw pointer, so a temporary segfaults
    # rather than raising, the same trap ``PointsToMeshProjector`` carries in test_proximity.py.
    queries_np = rng.random((40, 3)) * 5.0
    queries_wp = wp.array(
        np.ascontiguousarray(queries_np, dtype=np.float32), dtype=wp.vec3, device=device
    )
    queries_ml = mm.std_vector_Vector3_float()
    for query_np in queries_np:
        queries_ml.append(mm.Vector3f(*query_np.tolist()))
    cloud_ml = points_to_meshlib(points_np)  # must outlive the projector: it holds a raw pointer
    projector_ml = mm.PointsProjector()
    projector_ml.setPointCloud(cloud_ml)
    projections_ml = mm.std_vector_PointsProjectionResult()
    projector_ml.findProjections(projections_ml, queries_ml, mm.FindProjectionOnPointsSettings())

    nearest_wp, distances_wp = tw.neighbors.query_bvh_nearest(points_wp, queries_wp, k=1)
    assert np.array_equal(
        nearest_wp.numpy(), np.array([int(result.vId) for result in projections_ml])
    )
    assert np.allclose(
        distances_wp.numpy(),
        np.sqrt([result.distSq for result in projections_ml]),
        rtol=1e-5,
        atol=1e-5,
    )
