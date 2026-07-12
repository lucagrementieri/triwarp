"""
Benchmark for ``triwarp.stitching.fill_holes_min_weight`` on real scan meshes.

Scan meshes ship with genuine boundary holes; hole count and size vary per mesh, so the
assertion only checks that faces were added or preserved. The trimesh reference
(``tm.repair.fill_holes``) is a different, much weaker algorithm — timing context only.
"""

from __future__ import annotations

import pytest
import trimesh as tm
from conftest import BenchCase, skip_larger_than

import triwarp as tw


@pytest.mark.benchmark(group="fill_holes_min_weight")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_fill_holes_min_weight(bench_case: BenchCase) -> None:
    skip_larger_than(bench_case, "happy_buddha")
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        result = bench_case.run(lambda: tw.stitching.fill_holes_min_weight(vertices, faces))
        assert result.shape[0] >= faces.shape[0]
    else:  # trimesh mutates in place: rebuild inside the timed callable
        vertices, faces = bench_case.vertices_np, bench_case.faces_np

        def run() -> tm.Trimesh:
            mesh = tm.Trimesh(vertices, faces, process=False)
            tm.repair.fill_holes(mesh)
            return mesh

        result = bench_case.run(run)
        assert result.faces.shape[0] >= faces.shape[0]
