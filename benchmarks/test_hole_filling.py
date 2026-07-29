"""
Benchmarks for ``triwarp.hole_filling``.

Axis: **loops_dp** -- ``rim_short`` (two loops of 512) against ``holes_many`` (512 loops of 3).
The module has the highest non-``N`` sensitivity in the package and this pair separates its two
independent drivers:

* **Loop length**, cubed. ``fill_holes_min_weight`` runs a minimum-weight triangulation DP over a
  ``B x B`` table per loop, filled by ``B - 2`` *sequential* kernel launches and read back to the
  host for the traceback. Total work is ``sum(B_i^3)`` and no batching can remove it: this is the
  real cost the module exists to pay. Two loops of 512 measure **157 ms**.
* **Loop count**, which should cost nothing and used to cost everything. Every loop paid a
  ``.numpy()`` readback, its own forbidden-chord pass over the whole mesh and its own span
  launches, so 512 three-vertex holes -- where the DP itself is one triangle per hole -- ran to
  **376 ms**, *more* than the genuinely expensive ``rim_short``. Batching the DP across loops took
  that to **4.2 ms** (90x) and, because it also runs ``rim_short``'s two rims concurrently, took
  that point from 273 to 157 ms as a side effect.

That is why the axis meshes are *small*: ``rim_short`` is 1 024 faces and ``holes_many`` is 81 408.
Face count is not the variable, and sizing these meshes up would only add DP table entries that the
``B^3`` term already dominates. The two points now differ by 37x in the right direction, which is
what the axis is for -- the inversion is what said the per-loop sequence was the bug.
``fill_holes_fan`` runs on the wider **loops** axis instead, because it has no DP and so can afford
``rim_long``'s 65 536-vertex rims -- it is the floor this module's cost is measured against.

References
----------
**trimesh**'s ``repair.fill_holes`` is a much weaker algorithm (it fans triangles across small
holes and gives up on large ones), so it is timing context rather than an equivalent -- the
assertion only checks that faces were added or preserved.

**open3d**'s ``fill_holes`` lives on the newer tensor API (``open3d.t.geometry.TriangleMesh``) and
wraps a hole-filling pass over the boundary loops, which is the closest analogue to triwarp's
minimum-weight triangulation. It returns a new tensor mesh, but ``from_legacy`` is a full
conversion, so that conversion is hoisted out of the timed callable and only ``fill_holes`` is
measured.

**pymeshlab**'s ``meshing_close_holes`` is a third algorithm again -- an ear-clipping fill with an
optional self-intersection check (``selfintersection=True`` by default, left on) rather than a
minimum-weight DP. One parameter matters and it is a trap: **``maxholesize`` is an edge count with a
default of 30**, so on ``rim_short``'s two 512-edge rims the filter closes *nothing* and returns in
0.66 ms with ``{'closed_holes': 0}``. Lifting it to 10^6 is what makes the axis points comparable at
all, and it moves ``rim_short`` from 0.66 to 64.9 ms while leaving ``holes_many`` at 38.5 (from
36.9, where 30 edges was already enough for a three-edge hole). The dict it returns is what makes
that checkable, and the assertion below reads it.

Note what the axis then says: the reference spends **1.7x** on two 512-edge rims what it spends on
512 three-edge holes, against triwarp's 37x. That is the ``B^3`` term -- MeshLab's ear clipping is
quadratic at worst, so it does not pay it, and its rows are the honest price of *not* computing a
minimum-weight triangulation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import trimesh as tm
from conftest import BenchCase

import triwarp as tw

if TYPE_CHECKING:
    import open3d as o3d

# The DP runs to hundreds of milliseconds a call on both axis points.
_ROUNDS = 3

_tensor_mesh_cache: dict[str, o3d.t.geometry.TriangleMesh] = {}


def _tensor_mesh_o3d(bench_case: BenchCase) -> o3d.t.geometry.TriangleMesh:
    """Convert to the tensor API once per mesh for ``fill_holes`` (conversion is not the op)."""
    if bench_case.mesh_name not in _tensor_mesh_cache:
        import open3d as o3d

        _tensor_mesh_cache[bench_case.mesh_name] = o3d.t.geometry.TriangleMesh.from_legacy(
            bench_case.mesh_o3d
        )
    return _tensor_mesh_cache[bench_case.mesh_name]


@pytest.mark.benchmark(group="fill_holes_fan")
@pytest.mark.benchaxis("loops")
@pytest.mark.benchlibs("triwarp")
def test_fill_holes_fan(bench_case: BenchCase) -> None:
    """The cheapest filler -- one fan per loop, no DP -- so it can take the long-rim axis."""
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    result = bench_case.run(lambda: tw.hole_filling.fill_holes_fan(vertices, faces))
    assert result.shape[0] >= faces.shape[0]


@pytest.mark.benchmark(group="fill_holes_min_weight")
@pytest.mark.benchaxis("loops_dp")
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d", "pymeshlab")
def test_fill_holes_min_weight(bench_case: BenchCase) -> None:
    """The ``B^3`` DP: few long loops against many short ones, at a comparable total boundary."""
    if bench_case.kind == "pymeshlab":
        # ``maxholesize`` is an *edge count* cap, and its default of 30 would silently close nothing
        # on ``rim_short``'s two 512-edge rims -- measured at 0.66 ms for zero holes closed. Lifting
        # it is what makes the two axis points comparable at all, and the returned dict is asserted
        # on so a future default change cannot quietly turn this row back into a no-op.
        statistics_pml = bench_case.run(
            lambda: bench_case.new_meshset_pml().meshing_close_holes(maxholesize=1_000_000),
            rounds=_ROUNDS,
        )
        assert statistics_pml["closed_holes"] > 0
        return
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        result = bench_case.run(
            lambda: tw.hole_filling.fill_holes_min_weight(vertices, faces), rounds=_ROUNDS
        )
        assert result.shape[0] >= faces.shape[0]
    elif bench_case.kind == "trimesh":  # mutates in place: rebuild inside the timed callable
        vertices, faces = bench_case.vertices_np, bench_case.faces_np

        def run() -> tm.Trimesh:
            mesh = tm.Trimesh(vertices, faces, process=False)
            tm.repair.fill_holes(mesh)
            return mesh

        result = bench_case.run(run, rounds=_ROUNDS)
        assert result.faces.shape[0] >= faces.shape[0]
    else:  # open3d tensor API: fill_holes returns a new mesh, so the converted input is reusable
        mesh_t = _tensor_mesh_o3d(bench_case)
        n_faces = bench_case.n_faces
        filled = bench_case.run(mesh_t.fill_holes, rounds=_ROUNDS)
        assert int(filled.triangle["indices"].shape[0]) >= n_faces


@pytest.mark.benchmark(group="fill_holes_min_weight_chords")
@pytest.mark.benchaxis("loops_dp")
@pytest.mark.benchlibs("triwarp")
@pytest.mark.parametrize("resolve_multiple_edges", [False, True], ids=["plain", "chords"])
def test_fill_holes_min_weight_chords(bench_case: BenchCase, resolve_multiple_edges: bool) -> None:
    """
    What the forbidden-chord pass costs on top of the DP.

    ``resolve_multiple_edges=True`` builds a ``B x B`` table of chords that already exist as mesh
    edges and bans them from the triangulation, which is a second quadratic pass per loop. Whether
    that is a rounding error next to the cubic DP or a real fraction of it is only visible against
    the ``plain`` row.
    """
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    result = bench_case.run(
        lambda: tw.hole_filling.fill_holes_min_weight(
            vertices, faces, resolve_multiple_edges=resolve_multiple_edges
        ),
        rounds=_ROUNDS,
    )
    assert result.shape[0] >= faces.shape[0]
