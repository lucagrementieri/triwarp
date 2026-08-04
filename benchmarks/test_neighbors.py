"""
Benchmarks for ``triwarp.neighbors``: the k-nearest-neighbour queries.

This is the direct gate for the k-NN search radius. Twelve call sites in ``triwarp.distance``, the
point-cloud ICP path in ``triwarp.registration`` and three sites in ``triwarp.reconstruction`` all
bottom out in these two functions, but until this module existed none of them had a measurement
that isolated the query kernel from the surrounding algorithm.

Setup
-----
``points`` is the mesh's own vertices; ``queries`` is a fixed 20 000-point subsample of them pushed
off the surface by 1% of the bbox diagonal along a fixed direction, so no query coincides exactly
with a data point (which would make ``k=1`` trivially certifiable at radius 0) and every query still
has neighbours at a realistic spacing. Both ``k=1`` and ``k=7`` are timed: ``k=1`` is what
``distance.py`` and ICP use, ``k=7`` is ``ball_pivoting``'s seed table.

Axes
----
Nothing here is driven by a mesh property — the meshes are used only as point clouds — so the axes
are the structure's own parameters, and there are three:

* **k**, expressed as separate groups (``_k1`` / ``_k7``) rather than as a parametrize sweep,
  because the two are different call sites rather than two points on a curve: ``k=1`` is what
  ``distance.py`` and ICP use, ``k=7`` is ``ball_pivoting``'s seed table and maintains a sorted
  candidate row. The two land in different generated kernels — the row is register-resident and one
  kernel exists per row-size bucket (``kernels.neighbors.KNN_ROW_BUCKETS``), so ``k=1`` and ``k=7``
  are not the same code. The larger ``k`` values in-repo (30 and 64) are timed by
  [`test_points.py`](test_points.py), where they are what the outlier statistics ask for.
* **build against query**. The k-NN groups above include the build, which is what a caller passing
  raw buffers pays. The ``*_from_points`` groups below time the build *alone*, so subtracting them
  answers the question that matters for ICP: whether its per-iteration index rebuild is the cost or
  a red herring.
* **structure resolution** — ``leaf_size`` for the BVH, ``grid_bins`` for the hash grid, and
  ``radius`` for the ball queries. A leaf too large makes a shallow tree that is cheap to build and
  slow to traverse; too few bins degenerates into a linear scan and too many pays for empty cells.
  Ball-query cost is the *expected neighbour count*, roughly ``density x radius^3``, and the
  two-phase count-then-fill means it is paid twice.

The ``*_with_offsets`` form is used for the ball queries rather than plain ``query_*_ball``: the
latter returns a Python ``list`` of per-query arrays, adding ``O(n_queries)`` of host slicing that
would swamp the kernel. That host cost is real but belongs to a different question.

What is inside the timed callable
---------------------------------
For the k-NN groups, everything the public function does, including the spatial-index build and the
bounds reduction — that is what a caller passing raw buffers actually pays. The index build is small
next to the query (a ``bunny`` BVH build is ~0.2 ms), but hiding it would misreport the API. The
build and ball groups instead pass a prebuilt structure where the signature allows one, so they
isolate the piece they name.

Reference
---------
**scipy** ``spatial.KDTree`` — the same reference ``tests/test_neighbors.py`` validates against, and
the only exact k-NN in the test group with a matching signature (``query(x, k=k)`` returns distances
and indices in the same layout). ``KDTree`` construction is inside the timed region for the same
reason triwarp's index build is. trimesh has no k-NN entry point and libigl's is not exposed in the
Python bindings; open3d's ``KDTreeFlann`` has no batched query (it is a Python loop over
``search_knn_vector_3d``, which would time the interpreter rather than the search), so the
cross-library k-NN comparison lives in
[`test_registration.py`](test_registration.py) and [`test_points.py`](test_points.py) where open3d
drives the whole algorithm from C++.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp
from conftest import BenchCase, skip_larger_than
from scipy.spatial import KDTree

import triwarp as tw

_SEED = 42
_N_QUERIES = 20_000
_OFFSET_FRACTION = 0.01

# BVH leaf sizes and hash-grid bin counts: the structure-resolution knobs.
_LEAF_SIZES = [4, 32]
_GRID_BINS = [32, 256]

# Ball radii as multiples of the mean edge length. Expected neighbours grow ~cubically, so these
# two points are roughly 8x apart in work.
_RADIUS_SCALES = [2.0, 4.0]

_queries_np_cache: dict[str, np.ndarray] = {}
_queries_wp_cache: dict[tuple[str, str], wp.array] = {}
_bvh_cache: dict[tuple[str, str], wp.Bvh] = {}
_kdtree_cache: dict[str, KDTree] = {}


def _queries_np(bench_case: BenchCase) -> np.ndarray:
    """Subsample the mesh vertices at a fixed seed and displace them off the surface."""
    name = bench_case.mesh_name
    if name not in _queries_np_cache:
        rng = np.random.default_rng(_SEED)
        vertices = bench_case.vertices_np
        count = min(_N_QUERIES, vertices.shape[0])
        indices = rng.choice(vertices.shape[0], size=count, replace=False)
        diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
        offset = _OFFSET_FRACTION * diagonal * np.array([1.0, -1.0, 0.5]) / np.sqrt(2.25)
        _queries_np_cache[name] = np.ascontiguousarray(vertices[indices] + offset)
    return _queries_np_cache[name]


def _queries_wp(bench_case: BenchCase) -> wp.array[wp.vec3]:
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _queries_wp_cache:
        _queries_wp_cache[key] = wp.array(
            np.ascontiguousarray(_queries_np(bench_case), dtype=np.float32),
            dtype=wp.vec3,
            device=bench_case.device,
        )
    return _queries_wp_cache[key]


def _bvh(bench_case: BenchCase) -> wp.Bvh:
    """Prebuilt BVH over the cloud -- an *input* for the ball group, timed on its own elsewhere."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _bvh_cache:
        _bvh_cache[key] = tw.neighbors.bvh_from_points(bench_case.vertices_wp)
    return _bvh_cache[key]


def _kdtree(bench_case: BenchCase) -> KDTree:
    """Prebuilt scipy KDTree over the same cloud, for the ball-query comparison."""
    if bench_case.mesh_name not in _kdtree_cache:
        _kdtree_cache[bench_case.mesh_name] = KDTree(bench_case.vertices_np)
    return _kdtree_cache[bench_case.mesh_name]


def _run_scipy(bench_case: BenchCase, k: int) -> None:
    queries_np = _queries_np(bench_case)
    points_np = bench_case.vertices_np
    distances_np, indices_np = bench_case.run(lambda: KDTree(points_np).query(queries_np, k=k))
    assert np.asarray(indices_np).shape[0] == queries_np.shape[0]
    assert np.asarray(distances_np).shape[0] == queries_np.shape[0]


@pytest.mark.benchmark(group="query_bvh_nearest_k1")
@pytest.mark.benchlibs("triwarp", "scipy")
def test_query_bvh_nearest_k1(bench_case: BenchCase) -> None:
    """``k=1`` BVH k-NN — the exact call ICP and the Chamfer family make."""
    skip_larger_than(bench_case, "dragon")
    if bench_case.kind == "triwarp":
        points, queries = bench_case.vertices_wp, _queries_wp(bench_case)
        indices, _distances = bench_case.run(
            lambda: tw.neighbors.query_bvh_nearest(points, queries, k=1)
        )
        assert indices.shape == (queries.shape[0],)
    else:
        _run_scipy(bench_case, 1)


@pytest.mark.benchmark(group="query_hashgrid_nearest_k1")
@pytest.mark.benchlibs("triwarp", "scipy")
def test_query_hashgrid_nearest_k1(bench_case: BenchCase) -> None:
    """``k=1`` hash-grid k-NN — the backend ``distance.py`` picks."""
    skip_larger_than(bench_case, "dragon")
    if bench_case.kind == "triwarp":
        points, queries = bench_case.vertices_wp, _queries_wp(bench_case)
        indices, _distances = bench_case.run(
            lambda: tw.neighbors.query_hashgrid_nearest(points, queries, k=1)
        )
        assert indices.shape == (queries.shape[0],)
    else:
        _run_scipy(bench_case, 1)


@pytest.mark.benchmark(group="query_bvh_nearest_k7")
@pytest.mark.benchlibs("triwarp", "scipy")
def test_query_bvh_nearest_k7(bench_case: BenchCase) -> None:
    """``k=7`` BVH k-NN — ``ball_pivoting``'s seed-candidate table."""
    skip_larger_than(bench_case, "dragon")
    if bench_case.kind == "triwarp":
        points, queries = bench_case.vertices_wp, _queries_wp(bench_case)
        indices, _distances = bench_case.run(
            lambda: tw.neighbors.query_bvh_nearest(points, queries, k=7)
        )
        assert indices.shape == (queries.shape[0], 7)
    else:
        _run_scipy(bench_case, 7)


@pytest.mark.benchmark(group="query_bvh_nearest_k64")
@pytest.mark.benchlibs("triwarp", "scipy")
def test_query_bvh_nearest_k64(bench_case: BenchCase) -> None:
    """
    ``k=64`` BVH k-NN — the largest register-row bucket, and the k axis's far end.

    The candidate row costs ``2 * k`` registers, so this is the last ``k`` the row fits in them; at
    65 the search falls back to the global-memory-row kernel and the cost per neighbour jumps
    (measured 7.8x between the two at this ``k``). The group exists to keep that boundary visible:
    without it the suite's k axis stops at 7 and the regime the row storage governs is unmeasured.
    """
    skip_larger_than(bench_case, "dragon")
    if bench_case.kind == "triwarp":
        points, queries = bench_case.vertices_wp, _queries_wp(bench_case)
        indices, _distances = bench_case.run(
            lambda: tw.neighbors.query_bvh_nearest(points, queries, k=64)
        )
        assert indices.shape == (queries.shape[0], 64)
    else:
        _run_scipy(bench_case, 64)


@pytest.mark.benchmark(group="query_hashgrid_nearest_k7")
@pytest.mark.benchlibs("triwarp", "scipy")
def test_query_hashgrid_nearest_k7(bench_case: BenchCase) -> None:
    """``k=7`` hash-grid k-NN."""
    skip_larger_than(bench_case, "dragon")
    if bench_case.kind == "triwarp":
        points, queries = bench_case.vertices_wp, _queries_wp(bench_case)
        indices, _distances = bench_case.run(
            lambda: tw.neighbors.query_hashgrid_nearest(points, queries, k=7)
        )
        assert indices.shape == (queries.shape[0], 7)
    else:
        _run_scipy(bench_case, 7)


@pytest.mark.benchmark(group="bvh_from_points")
@pytest.mark.benchlibs("triwarp", "scipy")
@pytest.mark.parametrize("leaf_size", _LEAF_SIZES)
def test_bvh_from_points(bench_case: BenchCase, leaf_size: int) -> None:
    """
    Structure build alone: the cost a caller amortizes, or fails to.

    Subtract this from ``query_bvh_nearest_k1`` to get the query in isolation. scipy takes no
    leaf-size parameter, so its two rows are identical by construction and are there as the fixed
    bar; triwarp's own slope is the other half of the ``leaf_size`` trade-off.
    """
    skip_larger_than(bench_case, "dragon", "the scipy reference builds single-threaded")
    if bench_case.kind == "triwarp":
        points = bench_case.vertices_wp
        bvh = bench_case.run(lambda: tw.neighbors.bvh_from_points(points, leaf_size=leaf_size))
        assert bvh is not None
    else:
        points_np = bench_case.vertices_np
        assert bench_case.run(lambda: KDTree(points_np)) is not None


@pytest.mark.benchmark(group="hashgrid_from_points")
@pytest.mark.benchlibs("triwarp")
@pytest.mark.parametrize("grid_bins", _GRID_BINS)
def test_hashgrid_from_points(bench_case: BenchCase, grid_bins: int) -> None:
    """The hash-grid build, swept over its bin count: the other structure's amortization floor."""
    skip_larger_than(bench_case, "dragon")
    points = bench_case.vertices_wp
    radius = _RADIUS_SCALES[0] * bench_case.mean_edge
    grid = bench_case.run(
        lambda: tw.neighbors.hashgrid_from_points(points, radius, grid_bins=grid_bins)
    )
    assert grid is not None


@pytest.mark.benchmark(group="query_bvh_ball")
@pytest.mark.benchlibs("triwarp", "scipy")
@pytest.mark.parametrize("radius_scale", _RADIUS_SCALES)
def test_query_bvh_ball(bench_case: BenchCase, radius_scale: float) -> None:
    """Radius query over a prebuilt BVH: cost is the neighbour count, so ~8x between the radii."""
    skip_larger_than(bench_case, "bunny", "the neighbour count grows cubically with the radius")
    radius = radius_scale * bench_case.mean_edge
    if bench_case.kind == "triwarp":
        points, queries = bench_case.vertices_wp, _queries_wp(bench_case)
        bvh = _bvh(bench_case)
        neighbors, _distances, offsets = bench_case.run(
            lambda: tw.neighbors.query_bvh_ball_with_offsets(points, queries, radius, bvh=bvh)
        )
        assert offsets.shape[0] == int(queries.shape[0])
        assert neighbors.shape[0] >= 0
    else:
        tree, queries_np = _kdtree(bench_case), _queries_np(bench_case)
        found = bench_case.run(lambda: tree.query_ball_point(queries_np, radius))
        assert len(found) == queries_np.shape[0]


@pytest.mark.benchmark(group="query_hashgrid_ball")
@pytest.mark.benchlibs("triwarp")
@pytest.mark.parametrize("grid_bins", _GRID_BINS)
def test_query_hashgrid_ball(bench_case: BenchCase, grid_bins: int) -> None:
    """
    The hash-grid ball query, swept over the bin count at a fixed radius.

    Too few bins and each cell holds enough points that the query degenerates into a linear scan;
    too many and the build pays for cells nothing lands in. Where the optimum sits depends on the
    cloud's density, so this pair is the cheapest way to see which side of it the default is on.
    """
    skip_larger_than(bench_case, "bunny")
    radius = _RADIUS_SCALES[0] * bench_case.mean_edge
    points, queries = bench_case.vertices_wp, _queries_wp(bench_case)
    grid = tw.neighbors.hashgrid_from_points(points, radius, grid_bins=grid_bins)
    neighbors, _distances, offsets = bench_case.run(
        lambda: tw.neighbors.query_hashgrid_ball_with_offsets(points, queries, radius, grid=grid)
    )
    assert offsets.shape[0] == int(queries.shape[0])
    assert neighbors.shape[0] >= 0
