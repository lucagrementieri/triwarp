"""
Benchmarks for ``triwarp.validation.is_watertight`` / ``is_volume``.

The trimesh references rebuild the mesh inside the timed callable because trimesh caches
derived properties (a second access would time a dict lookup); note trimesh's
``is_watertight`` is edge-manifold-only, so it is timing context rather than an equivalent
computation (triwarp's check also includes vertex-manifoldness and self-intersection).
``lucy`` is skipped: the self-intersection BVH pass on 28M faces dominates unusably.
"""

from __future__ import annotations

import pytest
import trimesh as tm
from conftest import BenchCase, skip_larger_than

import triwarp as tw


@pytest.mark.benchmark(group="is_watertight")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_is_watertight(bench_case: BenchCase) -> None:
    skip_larger_than(bench_case, "happy_buddha")
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        result = bench_case.run(lambda: tw.validation.is_watertight(vertices, faces))
    else:
        vertices, faces = bench_case.vertices_np, bench_case.faces_np
        result = bench_case.run(lambda: tm.Trimesh(vertices, faces, process=False).is_watertight)
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
