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

What is inside the timed callable
---------------------------------
Everything the public function does, including the spatial-index build and the bounds reduction —
that is what a caller passing raw buffers actually pays. The index build is small next to the query
(a ``bunny`` BVH build is ~0.2 ms), but hiding it would misreport the API.

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

_queries_np_cache: dict[str, np.ndarray] = {}
_queries_wp_cache: dict[tuple[str, str], wp.array] = {}


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
