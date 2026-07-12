"""Benchmarks for ``triwarp.triangles``: area-weighted mesh centroid (global reduction)."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import BenchCase

import triwarp as tw


@pytest.mark.benchmark(group="centroid")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_centroid(bench_case: BenchCase) -> None:
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        result = bench_case.run(lambda: tw.triangles.centroid(vertices, faces))
        assert np.isfinite(list(result)).all()
    else:  # numpy reference: the uncached formula behind ``trimesh.Trimesh.centroid``
        vertices, faces = bench_case.vertices_np, bench_case.faces_np

        def run() -> np.ndarray:
            triangles = vertices[faces]
            crosses = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            areas = 0.5 * np.linalg.norm(crosses, axis=1)
            return (triangles.mean(axis=1) * areas[:, None]).sum(axis=0) / areas.sum()

        result = bench_case.run(run)
        assert result.shape == (3,)
