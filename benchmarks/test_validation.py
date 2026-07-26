"""
Benchmarks for ``triwarp.validation.is_watertight`` / ``is_volume``.

The trimesh references rebuild the mesh inside the timed callable because trimesh caches
derived properties (a second access would time a dict lookup); note trimesh's
``is_watertight`` is edge-manifold-only, so it is timing context rather than an equivalent
computation (triwarp's check also includes vertex-manifoldness and self-intersection).
``lucy`` is skipped: the self-intersection BVH pass on 28M faces dominates unusably.

**open3d** is the exact equivalent for ``is_watertight``: triwarp's docstring defines itself
against ``open3d.geometry.TriangleMesh.is_watertight`` (edge-manifold without boundary, plus
vertex-manifold and no self-intersection), so this is a like-for-like comparison rather than the
looser trimesh one. Open3D's meshes are not cached-property based, so the shared mesh can be reused
across rounds. There is no open3d ``is_volume``: its closest composition
(``is_watertight() and is_orientable()``) short-circuits on the very first check for these
open scan meshes, so it would time the same work as ``is_watertight`` under a different name.
"""

from __future__ import annotations

import pytest
import trimesh as tm
from conftest import BenchCase, skip_larger_than

import triwarp as tw


@pytest.mark.benchmark(group="is_watertight")
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d")
def test_is_watertight(bench_case: BenchCase) -> None:
    skip_larger_than(bench_case, "happy_buddha")
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        result = bench_case.run(lambda: tw.validation.is_watertight(vertices, faces))
    elif bench_case.kind == "trimesh":
        vertices, faces = bench_case.vertices_np, bench_case.faces_np
        result = bench_case.run(lambda: tm.Trimesh(vertices, faces, process=False).is_watertight)
    else:  # open3d: same definition as triwarp's, and it does not cache the answer
        mesh_o3d = bench_case.mesh_o3d
        result = bench_case.run(mesh_o3d.is_watertight)
    assert result in (True, False)


@pytest.mark.benchmark(group="is_volume")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_is_volume(bench_case: BenchCase) -> None:
    skip_larger_than(bench_case, "happy_buddha")
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        result = bench_case.run(lambda: tw.validation.is_volume(vertices, faces))
    else:
        vertices, faces = bench_case.vertices_np, bench_case.faces_np
        result = bench_case.run(lambda: tm.Trimesh(vertices, faces, process=False).is_volume)
    assert result in (True, False)
