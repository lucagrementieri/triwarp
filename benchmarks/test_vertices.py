"""
Benchmarks for ``triwarp.vertices``: index-count inference and area-weighted vertex normals.

``n_vertices`` runs on every mesh including ``lucy`` (28M faces): before the device-reduce fix
it copies the whole 336 MB face buffer to the host just to take a max.

The normals reference is trimesh's area-weighted ``vertex_normals`` (rebuilt inside the timed
callable — trimesh caches it). ``igl.per_vertex_normals`` was dropped: it segfaults flakily
when invoked late in a session that mixes Warp CUDA/CPU JIT with the other native libraries.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import trimesh as tm
from conftest import BenchCase

import triwarp as tw
import triwarp.typing as twt


@pytest.mark.benchmark(group="n_vertices")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_n_vertices(bench_case: BenchCase) -> None:
    if bench_case.kind == "triwarp":
        faces = cast(twt.Array1dInt32, bench_case.faces_wp)
        result = bench_case.run(lambda: tw.vertices.n_vertices(faces))
        assert result == bench_case.n_vertices
    else:  # numpy reference: what trimesh-style code does on host arrays
        faces = bench_case.faces_np
        result = bench_case.run(lambda: int(faces.max()) + 1)
        assert result == bench_case.n_vertices


@pytest.mark.benchmark(group="area_weighted_vertex_normals")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_area_weighted_vertex_normals(bench_case: BenchCase) -> None:
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        n_vertices = bench_case.n_vertices
        result = bench_case.run(
            lambda: tw.vertices.area_weighted_vertex_normals(n_vertices, vertices, faces)
        )
        assert result.shape == (n_vertices,)
    else:  # trimesh vertex_normals are area-weighted; rebuild inside (cached property)
        vertices, faces = bench_case.vertices_np, bench_case.faces_np
        result = bench_case.run(lambda: tm.Trimesh(vertices, faces, process=False).vertex_normals)
        assert result.shape == (bench_case.n_vertices, 3)
        assert np.isfinite(result).any()
