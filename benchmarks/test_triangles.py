"""
Benchmarks for ``triwarp.triangles``: the area-weighted mesh centroid and per-face normals/areas.

Both are per-face arithmetic over the whole mesh with no connectivity structure, so the scan sweep
(pure ``N``) is the right axis for them and every reference is doing the same cross products.

``face_normals_and_areas`` returns both quantities from one cross product, which is what its
consumers in the library want (the heat method, the gradient operators, area-weighted normals). The
references split them up: ``igl.doublearea`` and ``potpourri3d.face_areas`` return areas only, and
``trimesh.triangles.normals`` returns unit normals plus a validity mask. The reference rows
therefore do strictly less work than triwarp's — read this group as a floor for them rather than as
a fair race.
"""

from __future__ import annotations

import igl
import numpy as np
import potpourri3d as pp3d
import pytest
import trimesh as tm
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


@pytest.mark.benchmark(group="face_normals_and_areas")
@pytest.mark.benchlibs("triwarp", "trimesh", "igl", "potpourri3d")
def test_face_normals_and_areas(bench_case: BenchCase) -> None:
    """One cross product per face: the operator prologue every solver in the library pays."""
    n_faces = bench_case.n_faces
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        normals, areas = bench_case.run(
            lambda: tw.triangles.face_normals_and_areas(vertices, faces)
        )
        assert normals.shape == (n_faces,)
        assert areas.shape == (n_faces,)
    elif bench_case.kind == "igl":  # areas only, and doubled
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        areas_igl = bench_case.run(lambda: igl.doublearea(vertices_np, faces_np))
        assert areas_igl.shape == (n_faces,)
    elif bench_case.kind == "potpourri3d":  # areas only, vectorized numpy
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        areas_pp = bench_case.run(lambda: pp3d.face_areas(vertices_np, faces_np))
        assert areas_pp.shape == (n_faces,)
    else:  # trimesh: unit normals plus a validity mask, from the same cross product
        triangles_np = bench_case.vertices_np[bench_case.faces_np]
        normals_tm, valid_tm = bench_case.run(lambda: tm.triangles.normals(triangles_np))
        assert valid_tm.shape == (n_faces,)
        assert normals_tm.shape[1] == 3
