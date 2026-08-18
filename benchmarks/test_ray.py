"""
Benchmarks for ``triwarp.ray``: the three ray-cast entry points over the mesh BVH.

Axis: **depth**, the registry's name for the number of times a ray pierces the surface
(``sphere_med`` at 2 against ``shells_8``'s 8 concentric shells at 16). It is the axis that
separates the three groups from each other rather than merely scaling them:

- ``intersects_any`` may stop at the *first* hit, so it should be flat across the axis;
- ``intersects_first`` must find the *nearest* hit, so it cannot stop early and pays the descent;
- ``intersects_location`` does the same traversal and additionally compacts the hits, so it should
  track ``intersects_first`` plus a scan.

A group whose cost follows the crossing count when it should not is the finding here; so is one
that does not when it should.

This file exists because meshlib is the first reference in the suite with a batched ray query --
until it landed, every ray function was timed only indirectly through ``test_proximity.py``'s BVH
groups, which is what ``tests/api_conventions.py`` recorded as the reason ``triwarp/ray.py`` had no
benchmark file.

References
----------
**meshlib**'s ``multiRayMeshIntersect`` answers all three questions from one traversal of the mesh's
AABB tree, so the same call backs all three groups here and the only difference between the rows is
*which outputs are requested*. That is a real property of the reference rather than a shortcut, and
it is also the hazard: ``MultiRayMeshIntersectResult`` starts with every field ``None`` and the call
fills only the fields already holding a container, so a row that forgets to attach one is timing a
query that computes nothing and asserting on ``None``. Each row attaches exactly the outputs its
triwarp counterpart returns, which is what makes the three rows comparable to each other.

Its tree is built lazily and cached on the ``Mesh``, so the mesh is constructed and pre-warmed with
one throwaway query outside the timed callable -- the ``BenchCase.new_mesh_ml`` rule for a query
row, and the same thing triwarp's rows do by holding a ``wp.Mesh``. The ray cloud is likewise an
*input*: filling a ``std_vector_Vector3_float`` is a Python loop over 10 000 ``Vector3f``
constructions, comparable to the query itself, so it is cached per mesh.

**trimesh** has all three functions (``ray.intersects_first`` / ``intersects_any`` /
``intersects_id``) and is deliberately absent: its pure-Python engine takes minutes on these
meshes, and ``tests/test_ray.py`` already holds it as the correctness oracle, which is the useful
half. **open3d**'s ``RaycastingScene`` is Embree and would be a fair fight, but it answers a
different question shape -- ``cast_rays`` returns a dense record of distances, primitive ids and
barycentrics in one call, so it is the ``intersects_location`` row three times over rather than one
row per group.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import warp as wp
from conftest import BenchCase
from meshlib import mrmeshpy as mm

import triwarp as tw

# Ray count. Fixed rather than swept: the crossing axis is the interesting one and a query-count
# sweep here would measure the same launch scaling ``test_proximity.py`` already sweeps.
_N_RAYS = 10_000
_RAY_SEED = 17

_ray_cache: dict[tuple[str, str], tuple[wp.array[wp.vec3], wp.array[wp.vec3]]] = {}
_ray_np_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_ray_ml_cache: dict[str, tuple[mm.std_vector_Vector3_float, mm.std_vector_Vector3_float]] = {}
_mesh_cache: dict[tuple[str, str], wp.Mesh] = {}
_mesh_ml_cache: dict[str, mm.Mesh] = {}


def _rays_np(bench_case: BenchCase) -> tuple[np.ndarray, np.ndarray]:
    """
    Rays fired at the mesh from a sphere around it, aimed near its centre.

    Aimed rather than random so most rays *hit*: a cloud of random directions from outside mostly
    misses, and a group whose rays miss is timing the root-node rejection, not the traversal. The
    jitter on the aim point keeps the hit fraction below one, so both branches are exercised.
    """
    if bench_case.mesh_name not in _ray_np_cache:
        rng = np.random.default_rng(_RAY_SEED)
        vertices_np = bench_case.vertices_np
        centre_np = 0.5 * (vertices_np.min(axis=0) + vertices_np.max(axis=0))
        radius = float(np.linalg.norm(vertices_np.max(axis=0) - vertices_np.min(axis=0)))
        directions_np = rng.normal(size=(_N_RAYS, 3))
        directions_np /= np.linalg.norm(directions_np, axis=1, keepdims=True)
        origins_np = centre_np + radius * directions_np
        targets_np = centre_np + rng.normal(scale=0.25 * radius, size=(_N_RAYS, 3))
        aims_np = targets_np - origins_np
        aims_np /= np.linalg.norm(aims_np, axis=1, keepdims=True)
        _ray_np_cache[bench_case.mesh_name] = (origins_np, aims_np)
    return _ray_np_cache[bench_case.mesh_name]


def _rays_wp(bench_case: BenchCase) -> tuple[wp.array[wp.vec3], wp.array[wp.vec3]]:
    """Upload the ray cloud once per ``(mesh, device)``: it is the benchmark's input."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _ray_cache:
        origins_np, directions_np = _rays_np(bench_case)
        _ray_cache[key] = (
            wp.array(
                np.ascontiguousarray(origins_np, dtype=np.float32),
                dtype=wp.vec3,
                device=bench_case.device,
            ),
            wp.array(
                np.ascontiguousarray(directions_np, dtype=np.float32),
                dtype=wp.vec3,
                device=bench_case.device,
            ),
        )
    return _ray_cache[key]


def _rays_ml(
    bench_case: BenchCase,
) -> tuple[mm.std_vector_Vector3_float, mm.std_vector_Vector3_float]:
    """Build the rays as MeshLib vectors once: the fill is a 10 000-iteration Python loop."""
    if bench_case.mesh_name not in _ray_ml_cache:
        origins_np, directions_np = _rays_np(bench_case)
        origins_ml = mm.std_vector_Vector3_float()
        directions_ml = mm.std_vector_Vector3_float()
        for origin_np, direction_np in zip(origins_np, directions_np, strict=True):
            origins_ml.append(mm.Vector3f(*origin_np.tolist()))
            directions_ml.append(mm.Vector3f(*direction_np.tolist()))
        _ray_ml_cache[bench_case.mesh_name] = (origins_ml, directions_ml)
    return _ray_ml_cache[bench_case.mesh_name]


def _mesh_wp(bench_case: BenchCase) -> wp.Mesh:
    """Build one ``wp.Mesh`` per ``(mesh, device)``: every ``ray`` entry point takes one."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _mesh_cache:
        _mesh_cache[key] = wp.Mesh(points=bench_case.vertices_wp, indices=bench_case.faces_wp)
    return _mesh_cache[key]


def _mesh_part_ml(bench_case: BenchCase) -> mm.MeshPart:
    """
    Build a ``MeshPart``, keeping its mesh alive in a module-level cache.

    The mesh has to outlive the part -- ``MeshPart`` does not own it, and MeshLib's projector family
    has the same rule (see ``test_proximity.py``'s ``closest_point_on_mesh`` row, where handing a
    temporary mesh to ``PointsToMeshProjector`` reads freed memory and segfaults). The cache is the
    reference that keeps it alive; a ``MeshPart`` cannot hold one itself, since the pybind11 object
    has no ``__dict__`` to stash it in. The cache is also what keeps the lazily built AABB tree warm
    across the rows in this file.
    """
    if bench_case.mesh_name not in _mesh_ml_cache:
        _mesh_ml_cache[bench_case.mesh_name] = bench_case.new_mesh_ml()
    return mm.MeshPart(_mesh_ml_cache[bench_case.mesh_name])


def _multi_ray_query_ml(
    bench_case: BenchCase, *, faces: bool = False, points: bool = False, hits: bool = False
) -> Callable[[], mm.MultiRayMeshIntersectResult]:
    """
    Build the timed callable for MeshLib's batched ray query, with the requested outputs attached.

    Every field of ``MultiRayMeshIntersectResult`` defaults to ``None`` and the call fills only the
    ones already holding a container, so which outputs a row asks for is what its timing means --
    and a row that attaches none is timing a query that computes nothing. The mesh, its AABB tree
    and the ray vectors are all resolved here, outside the returned callable, so the callable is the
    traversal alone.
    """
    part_ml = _mesh_part_ml(bench_case)
    origins_ml, directions_ml = _rays_ml(bench_case)

    def query_ml() -> mm.MultiRayMeshIntersectResult:
        result_ml = mm.MultiRayMeshIntersectResult()
        if faces:
            result_ml.isectFaces = mm.std_vector_Id_FaceTag()
        if points:
            result_ml.isectPts = mm.std_vector_Vector3_float()
        if hits:
            result_ml.intersectingRays = mm.BitSet()
        mm.multiRayMeshIntersect(part_ml, origins_ml, directions_ml, result_ml)
        return result_ml

    query_ml()  # pre-warm: the mesh's AABB tree is built lazily on first use
    return query_ml


@pytest.mark.benchmark(group="intersects_first")
@pytest.mark.benchaxis("depth")
@pytest.mark.benchlibs("triwarp", "meshlib")
def test_intersects_first(bench_case: BenchCase) -> None:
    """
    Nearest-hit face per ray: the traversal that cannot stop early.

    meshlib's row requests ``isectFaces`` alone, which is the same dense ``(n_rays,)`` answer
    triwarp returns -- with an invalid ``FaceId`` where triwarp writes ``-1``
    (``tests/test_ray.py``). ``closestIntersect`` defaults to ``True``, which is triwarp's rule, so
    it is left alone here and the ``intersects_any`` row below is what shows the other setting.
    """
    if bench_case.kind == "meshlib":
        first_ml = _multi_ray_query_ml(bench_case, faces=True)
        assert len(bench_case.run(first_ml).isectFaces) == _N_RAYS
        return
    mesh, (origins, directions) = _mesh_wp(bench_case), _rays_wp(bench_case)
    faces_hit = bench_case.run(lambda: tw.ray.intersects_first(mesh, origins, directions))
    assert faces_hit.shape == (_N_RAYS,)


@pytest.mark.benchmark(group="intersects_any")
@pytest.mark.benchaxis("depth")
@pytest.mark.benchlibs("triwarp", "meshlib")
def test_intersects_any(bench_case: BenchCase) -> None:
    """
    Any-hit per ray: the one group here that is allowed to stop at the first triangle it finds.

    Read against ``intersects_first`` on the same rays and the same mesh -- the gap between the two
    groups is what "nearest" costs over "any", and it is the reason both exist. triwarp uses
    ``wp.mesh_query_ray_anyhit``; meshlib's row asks only for ``intersectingRays``, the bitset of
    rays that hit anything. Note that requesting fewer outputs does *not* make MeshLib's traversal
    any-hit -- ``closestIntersect`` governs that, and this row leaves it at its default, so the row
    is an upper bound on the question rather than the matching algorithm.
    """
    if bench_case.kind == "meshlib":
        any_ml = _multi_ray_query_ml(bench_case, hits=True)
        assert 0 < bench_case.run(any_ml).intersectingRays.count() <= _N_RAYS
        return
    mesh, (origins, directions) = _mesh_wp(bench_case), _rays_wp(bench_case)
    hit = bench_case.run(lambda: tw.ray.intersects_any(mesh, origins, directions))
    assert hit.shape == (_N_RAYS,)


@pytest.mark.benchmark(group="intersects_location")
@pytest.mark.benchaxis("depth")
@pytest.mark.benchlibs("triwarp", "meshlib")
def test_intersects_location(bench_case: BenchCase) -> None:
    """
    The hit positions, compacted: ``intersects_first``'s traversal plus a scan and a gather.

    triwarp returns only the rays that hit, as three sparse arrays, so its row carries a prefix sum
    and a compaction the other two groups do not -- which is the cost this group isolates. meshlib
    returns its hits *densely* and asks for ``isectPts`` plus the hit bitset, leaving the selection
    to the caller, so its row is the lower bound of the same work.
    """
    if bench_case.kind == "meshlib":
        location_ml = _multi_ray_query_ml(bench_case, points=True, hits=True)
        result_ml = bench_case.run(location_ml)
        assert len(result_ml.isectPts) == _N_RAYS
        assert 0 < result_ml.intersectingRays.count() <= _N_RAYS
        return
    mesh, (origins, directions) = _mesh_wp(bench_case), _rays_wp(bench_case)
    locations, rays, faces_hit = bench_case.run(
        lambda: tw.ray.intersects_location(mesh, origins, directions)
    )
    assert locations.shape == rays.shape == faces_hit.shape
    assert int(rays.shape[0]) <= _N_RAYS
