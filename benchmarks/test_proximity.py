"""
Benchmarks for ``triwarp.proximity`` hot paths.

Covers winding number, AABB bounds, tangent spheres and geodesic-ball queries.
``winding_number`` is O(n_queries x n_faces) even in the tiled variant, so ``lucy`` is skipped;
the pinned serial (``tiled=False``) path is additionally capped at ``bunny`` because one thread
per query walking every face takes minutes beyond that.
"""

from __future__ import annotations

import igl
import numpy as np
import pytest
import warp as wp
from conftest import BenchCase, skip_larger_than

import triwarp as tw

_QUERY_SEED = 42
_N_QUERIES = 10_000

_query_cache: dict[tuple[str, str], wp.array] = {}
_surface_cache: dict[tuple[str, str], wp.array] = {}
_mesh_cache: dict[tuple[str, str], wp.Mesh] = {}


def _query_points_np(bench_case: BenchCase) -> np.ndarray:
    """10k query points: subsampled vertices jittered by 10% of the bbox diagonal."""
    rng = np.random.default_rng(_QUERY_SEED)
    vertices = bench_case.vertices_np
    idx = rng.integers(0, vertices.shape[0], size=_N_QUERIES)
    diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    return vertices[idx] + rng.normal(scale=0.1 * diagonal, size=(_N_QUERIES, 3))


def _query_points_wp(bench_case: BenchCase) -> wp.array[wp.vec3]:
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _query_cache:
        _query_cache[key] = wp.array(
            np.ascontiguousarray(_query_points_np(bench_case), dtype=np.float32),
            dtype=wp.vec3,
            device=bench_case.device,
        )
    return _query_cache[key]


def _surface_points_wp(bench_case: BenchCase) -> wp.array[wp.vec3]:
    """10k on-surface points (subsampled mesh vertices, no jitter) for tangent-sphere queries."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _surface_cache:
        rng = np.random.default_rng(_QUERY_SEED)
        vertices = bench_case.vertices_np
        idx = rng.choice(vertices.shape[0], size=min(_N_QUERIES, vertices.shape[0]), replace=False)
        _surface_cache[key] = wp.array(
            np.ascontiguousarray(vertices[idx], dtype=np.float32),
            dtype=wp.vec3,
            device=bench_case.device,
        )
    return _surface_cache[key]


def _mesh_wp(bench_case: BenchCase) -> wp.Mesh:
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _mesh_cache:
        _mesh_cache[key] = wp.Mesh(points=bench_case.vertices_wp, indices=bench_case.faces_wp)
    return _mesh_cache[key]


@pytest.mark.benchmark(group="winding_number")
@pytest.mark.benchlibs("triwarp", "igl")
def test_winding_number(bench_case: BenchCase) -> None:
    skip_larger_than(bench_case, "happy_buddha", "O(queries x faces): lucy is untenable")
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        points = _query_points_wp(bench_case)
        result = bench_case.run(lambda: tw.proximity.winding_number(vertices, faces, points))
        assert result.shape == (_N_QUERIES,)
    else:  # igl exact generalized winding number
        vertices, faces = bench_case.vertices_np, bench_case.faces_np
        points = _query_points_np(bench_case)
        result = bench_case.run(lambda: igl.winding_number(vertices, faces, points))
        assert result.shape == (_N_QUERIES,)


@pytest.mark.benchmark(group="winding_number_serial")
@pytest.mark.benchlibs("triwarp")
def test_winding_number_serial(bench_case: BenchCase) -> None:
    """Pinned ``tiled=False`` reference path: one thread per query loops over every face."""
    skip_larger_than(bench_case, "bunny", "serial winding takes minutes beyond bunny")
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    points = _query_points_wp(bench_case)
    result = bench_case.run(
        lambda: tw.proximity.winding_number(vertices, faces, points, tiled=False)
    )
    assert result.shape == (_N_QUERIES,)


@pytest.mark.benchmark(group="aabb_bounds")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_aabb_bounds(bench_case: BenchCase) -> None:
    if bench_case.kind == "triwarp":
        vertices = bench_case.vertices_wp
        lower, upper = bench_case.run(lambda: tw.proximity.aabb_bounds(vertices))
        assert lower[0] <= upper[0]
    else:  # what an uncached ``trimesh.Trimesh.bounds`` computes: numpy min/max per axis
        vertices = bench_case.vertices_np
        result = bench_case.run(lambda: np.vstack((vertices.min(axis=0), vertices.max(axis=0))))
        assert result.shape == (2, 3)


@pytest.mark.benchmark(group="max_tangent_sphere_reach")
@pytest.mark.benchlibs("triwarp")
def test_max_tangent_sphere_reach(bench_case: BenchCase) -> None:
    """Exterior tangent spheres: exercises the ``init_sphere_radii`` inf-distance branch."""
    skip_larger_than(bench_case, "dragon")
    mesh = _mesh_wp(bench_case)
    points = _surface_points_wp(bench_case)
    _, radii = bench_case.run(lambda: tw.proximity.max_tangent_sphere(mesh, points, inwards=False))
    assert radii.shape == points.shape


@pytest.mark.benchmark(group="thickness_interior")
@pytest.mark.benchlibs("triwarp")
def test_thickness_interior(bench_case: BenchCase) -> None:
    """Interior thickness: regression guard for the finite-distance fast path."""
    skip_larger_than(bench_case, "dragon")
    mesh = _mesh_wp(bench_case)
    points = _surface_points_wp(bench_case)
    result = bench_case.run(lambda: tw.proximity.thickness(mesh, points))
    assert result.shape == points.shape


@pytest.mark.benchmark(group="query_geodesic_ball")
@pytest.mark.benchlibs("triwarp")
def test_query_geodesic_ball(bench_case: BenchCase) -> None:
    skip_larger_than(bench_case, "happy_buddha")
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    radius = 5.0 * float(tw.edges.mean_edge_length(vertices, faces))
    _, offsets, _ = bench_case.run(
        lambda: tw.proximity.query_geodesic_ball(vertices, faces, radius)
    )
    assert offsets.shape == (vertices.shape[0],)
