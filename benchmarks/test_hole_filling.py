"""
Benchmark for ``triwarp.hole_filling.fill_holes_min_weight`` on real scan meshes.

Scan meshes ship with genuine boundary holes; hole count and size vary per mesh, so the
assertion only checks that faces were added or preserved. The trimesh reference
(``tm.repair.fill_holes``) is a different, much weaker algorithm — timing context only.

**open3d**'s ``fill_holes`` lives on the newer tensor API (``open3d.t.geometry.TriangleMesh``) and
wraps a hole-filling pass over the boundary loops, which is the closest analogue to triwarp's
minimum-weight triangulation. It returns a new tensor mesh, but ``from_legacy`` is a full
conversion, so that conversion is hoisted out of the timed callable and only ``fill_holes`` is
measured.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import trimesh as tm
from conftest import BenchCase, skip_larger_than

import triwarp as tw

if TYPE_CHECKING:
    import open3d as o3d

_tensor_mesh_cache: dict[str, o3d.t.geometry.TriangleMesh] = {}


def _tensor_mesh_o3d(bench_case: BenchCase) -> o3d.t.geometry.TriangleMesh:
    """Tensor-API mesh for ``fill_holes``, converted once per mesh (conversion is not the op)."""
    if bench_case.mesh_name not in _tensor_mesh_cache:
        import open3d as o3d

        _tensor_mesh_cache[bench_case.mesh_name] = o3d.t.geometry.TriangleMesh.from_legacy(
            bench_case.mesh_o3d
        )
    return _tensor_mesh_cache[bench_case.mesh_name]


@pytest.mark.benchmark(group="fill_holes_min_weight")
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d")
def test_fill_holes_min_weight(bench_case: BenchCase) -> None:
    skip_larger_than(bench_case, "happy_buddha")
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        result = bench_case.run(lambda: tw.hole_filling.fill_holes_min_weight(vertices, faces))
        assert result.shape[0] >= faces.shape[0]
    elif bench_case.kind == "trimesh":  # mutates in place: rebuild inside the timed callable
        vertices, faces = bench_case.vertices_np, bench_case.faces_np

        def run() -> tm.Trimesh:
            mesh = tm.Trimesh(vertices, faces, process=False)
            tm.repair.fill_holes(mesh)
            return mesh

        result = bench_case.run(run)
        assert result.faces.shape[0] >= faces.shape[0]
    else:  # open3d tensor API: fill_holes returns a new mesh, so the converted input is reusable
        mesh_t = _tensor_mesh_o3d(bench_case)
        n_faces = bench_case.faces_np.shape[0]
        filled = bench_case.run(mesh_t.fill_holes)
        assert int(filled.triangle["indices"].shape[0]) >= n_faces
