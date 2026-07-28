"""
Benchmarks for ``triwarp.proximity`` hot paths.

Covers winding number, signed distance, AABB bounds, tangent spheres and geodesic-ball queries.
``winding_number`` is O(n_queries x n_faces) even in the tiled variant, so ``lucy`` is skipped;
the pinned serial (``tiled=False``) path is additionally capped at ``bunny`` because one thread
per query walking every face takes minutes beyond that.

Read ``winding_number`` and ``signed_distance_on_mesh[winding]`` together: both answer an
inside/outside question from solid angle, but the first accumulates it exactly over every face while
the second lets Warp's BVH traversal approximate it and keeps only the sign. The gap between them is
the cost of needing the winding *value* rather than just its sign.

Only ``aabb_bounds`` has an open3d equivalent (``get_axis_aligned_bounding_box``). Open3D has no
generalized winding number — its inside/outside test is raycasting-based
(``RaycastingScene.compute_occupancy``), a different algorithm answering a coarser question — and no
tangent-sphere, local-thickness or geodesic-ball query at all.
"""

from __future__ import annotations

from typing import Literal

import igl
import numpy as np
import pytest
import warp as wp
from conftest import BenchCase, skip_larger_than

import triwarp as tw

_QUERY_SEED = 42
_N_QUERIES = 10_000

# Query counts for ``winding_number``, the one genuinely O(queries x faces) function here: the
# other half of its product, swept independently of the mesh.
_N_QUERIES_SWEEP = [10_000, 100_000]

_query_cache: dict[tuple[str, str, int], wp.array] = {}
_surface_cache: dict[tuple[str, str], wp.array] = {}
_mesh_cache: dict[tuple[str, str], wp.Mesh] = {}


def _query_points_np(bench_case: BenchCase, count: int = _N_QUERIES) -> np.ndarray:
    """Query points: subsampled vertices jittered by 10% of the bbox diagonal."""
    rng = np.random.default_rng(_QUERY_SEED)
    vertices = bench_case.vertices_np
    idx = rng.integers(0, vertices.shape[0], size=count)
    diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    return vertices[idx] + rng.normal(scale=0.1 * diagonal, size=(count, 3))


def _query_points_wp(bench_case: BenchCase, count: int = _N_QUERIES) -> wp.array[wp.vec3]:
    key = (bench_case.mesh_name, str(bench_case.device), count)
    if key not in _query_cache:
        _query_cache[key] = wp.array(
            np.ascontiguousarray(_query_points_np(bench_case, count), dtype=np.float32),
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
@pytest.mark.parametrize("n_queries", _N_QUERIES_SWEEP)
def test_winding_number(bench_case: BenchCase, n_queries: int) -> None:
    """
    Exact winding number: no BVH, every query sums over every face.

    The one genuinely ``O(queries x faces)`` function in the module, so both sizes are swept --
    the mesh by the registry and the query count here. A 10x step in queries that is not a 10x
    step in time would mean the launch is not saturating the device.
    """
    skip_larger_than(bench_case, "happy_buddha", "O(queries x faces): lucy is untenable")
    if n_queries > _N_QUERIES:
        # 100k queries against dragon is 8.7e10 pair evaluations, and against happy_buddha 1.1e11.
        # The 10x query step is measurable on the medium meshes and the product is what it says.
        skip_larger_than(bench_case, "bunny", "the wide query sweep is only tenable up to bunny")
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        points = _query_points_wp(bench_case, n_queries)
        result = bench_case.run(lambda: tw.proximity.winding_number(vertices, faces, points))
        assert result.shape == (n_queries,)
    else:  # igl exact generalized winding number
        vertices, faces = bench_case.vertices_np, bench_case.faces_np
        points = _query_points_np(bench_case, n_queries)
        result = bench_case.run(lambda: igl.winding_number(vertices, faces, points))
        assert result.shape == (n_queries,)


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


@pytest.mark.benchmark(group="signed_distance_on_mesh")
@pytest.mark.benchlibs("triwarp")
@pytest.mark.parametrize("sign_mode", ["parity", "winding"])
def test_signed_distance_on_mesh(
    bench_case: BenchCase, sign_mode: Literal["parity", "winding"]
) -> None:
    """
    The two sign modes, on the same closest-point query.

    ``"winding"`` walks the BVH accumulating solid angle (Barnes-Hut, ``accuracy=2.0``) instead of
    casting 5 perturbed parity rays, and needs a ``wp.Mesh`` carrying the per-node solid-angle
    expansion — so the ``wp.Mesh`` build inside the timed region differs between the two, which is
    intentional: it is part of what the mode costs. Both include that build because
    ``signed_distance_on_mesh`` constructs its own mesh (it takes vertex/face arrays, not a
    ``wp.Mesh``), so there is no way for a caller to hoist it.
    """
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    points = _query_points_wp(bench_case)
    distance = bench_case.run(
        lambda: tw.proximity.signed_distance_on_mesh(vertices, faces, points, sign_mode=sign_mode)
    )
    assert distance.shape == (_N_QUERIES,)


@pytest.mark.benchmark(group="aabb_bounds")
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d")
def test_aabb_bounds(bench_case: BenchCase) -> None:
    if bench_case.kind == "triwarp":
        vertices = bench_case.vertices_wp
        lower, upper = bench_case.run(lambda: tw.bounds.aabb_bounds(vertices))
        assert lower[0] <= upper[0]
    elif bench_case.kind == "trimesh":
        # what an uncached ``trimesh.Trimesh.bounds`` computes: numpy min/max per axis
        vertices = bench_case.vertices_np
        result = bench_case.run(lambda: np.vstack((vertices.min(axis=0), vertices.max(axis=0))))
        assert result.shape == (2, 3)
    else:  # open3d's own bound reduction over the same vertices
        mesh_o3d = bench_case.mesh_o3d
        box_o3d = bench_case.run(mesh_o3d.get_axis_aligned_bounding_box)
        assert box_o3d.get_min_bound()[0] <= box_o3d.get_max_bound()[0]


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
@pytest.mark.benchaxis("depth")
@pytest.mark.benchlibs("triwarp")
@pytest.mark.parametrize("method", ["ray", "max_sphere"])
def test_thickness_interior(bench_case: BenchCase, method: Literal["ray", "max_sphere"]) -> None:
    """
    Interior thickness by one ray cast against up to 100 shrinking-sphere iterations.

    On the **depth** axis because both methods pay for surface crossings: a ray through
    ``shells_8`` meets 16 of them against ``sphere_med``'s 2, and the sphere method's convergence
    depends on local thickness, which nested shells make small. The two rows are roughly two
    orders of magnitude apart by construction -- the point is whether that ratio holds when the
    geometry stops being convex.
    """
    mesh = _mesh_wp(bench_case)
    points = _surface_points_wp(bench_case)
    result = bench_case.run(lambda: tw.proximity.thickness(mesh, points, method=method))
    assert result.shape == points.shape


@pytest.mark.benchmark(group="query_geodesic_ball")
@pytest.mark.benchlibs("triwarp")
def test_query_geodesic_ball(bench_case: BenchCase) -> None:
    skip_larger_than(bench_case, "happy_buddha")
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    radius = 5.0 * float(tw.edges.mean_edge_length(vertices, faces))
    _, offsets, _ = bench_case.run(lambda: tw.geodesic.geodesic_ball(vertices, faces, radius))
    assert offsets.shape == (vertices.shape[0],)
