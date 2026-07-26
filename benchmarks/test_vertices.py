"""
Benchmarks for ``triwarp.vertices``: index-count inference and area-weighted vertex normals.

``n_vertices`` runs on every mesh including ``lucy`` (28M faces): before the device-reduce fix
it copies the whole 336 MB face buffer to the host just to take a max.

The normals references are trimesh's area-weighted ``vertex_normals`` (rebuilt inside the timed
callable — trimesh caches it) and open3d's ``compute_vertex_normals``, which is also area-weighted
and is triwarp's closest analogue; open3d writes the result into the mesh but recomputes on every
call rather than caching, so the timed rounds stay honest with a fresh mesh per round.
``igl.per_vertex_normals`` was dropped: it segfaults flakily when invoked late in a session that
mixes Warp CUDA/CPU JIT with the other native libraries.

``n_vertices`` has no open3d equivalent worth timing: open3d stores the vertex count explicitly, so
``len(mesh.vertices)`` is O(1) and does not measure the max-reduce triwarp performs.
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
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d")
def test_area_weighted_vertex_normals(bench_case: BenchCase) -> None:
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        n_vertices = bench_case.n_vertices
        result = bench_case.run(
            lambda: tw.vertices.area_weighted_vertex_normals(n_vertices, vertices, faces)
        )
        assert result.shape == (n_vertices,)
    elif bench_case.kind == "trimesh":
        # trimesh vertex_normals are area-weighted; rebuild inside (cached property)
        vertices, faces = bench_case.vertices_np, bench_case.faces_np
        result = bench_case.run(lambda: tm.Trimesh(vertices, faces, process=False).vertex_normals)
        assert result.shape == (bench_case.n_vertices, 3)
        assert np.isfinite(result).any()
    else:
        # open3d writes the normals into the mesh, but recomputes them on every call rather than
        # caching (measured: identical cost on the second call), so the shared mesh is reusable.
        mesh_o3d = bench_case.mesh_o3d
        result_o3d = bench_case.run(mesh_o3d.compute_vertex_normals)
        assert np.asarray(result_o3d.vertex_normals).shape == (bench_case.n_vertices, 3)
