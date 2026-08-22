"""
Benchmarks for ``triwarp.homology``: a basis of non-contractible loops.

Axis: **genus**, which exists for this module and is used by nothing else. Every other mesh in
either registry is genus 0 -- the scan meshes are open surfaces and the feature meshes are spheres,
grids and tubes -- so until ``handles_1`` and ``handles_64`` landed there was nothing here to run
on at all, which is what ``tests/api_conventions.py`` recorded as the reason this file did not
exist. The pair holds the footprint, the tessellation scale and the face count fixed at ~90 000 and
varies only the number of handles, so a spread across the axis attributes to the genus.

What the axis separates is the two halves of the algorithm: building the spanning tree and its
cotree, against *tracing* one walk per generator. ``sphere_med`` at genus 0 is the floor -- the
trees with no loops to trace at all -- and ``handles_64`` is 128 traces on top of the same work.

Measured medians (RTX 5090, ``--device=cuda``)
----------------------------------------------
| mesh | genus | ``homology_generators`` | ``tree_cotree`` | meshlib |
|---|---|---|---|---|
| ``sphere_med`` | 0 | **5.96 ms** | 5.84 ms | 7.38 ms |
| ``handles_1`` | 1 | **5.76 ms** | 5.27 ms | 5.52 ms |
| ``handles_64`` | 64 | **12.07 ms** | 5.98 ms | 5.99 ms |

**This table replaces one that read 51.8 / 50.4 / 65.1 ms in the triwarp column, and both of that
table's findings have gone with it.** The re-measurement is trustworthy because meshlib's column is
unchanged within noise (7.38 / 5.52 / 5.99 against 7.6 / 5.5 / 6.0), which is what says the two
sessions are comparable and the 8.7x is triwarp's own. What earned it is *not* attributed here --
nothing in this module changed, so it came from something shared, and a guess in a benchmark
docstring is worse than the gap.

**The decomposition is no longer the whole cost.** At genus 1 the trees are 5.27 ms of 5.76, but at
genus 64 they are 5.98 of 12.07 -- so 128 loop traces cost 6.1 ms against 0.49 ms for two, and half
the work at high genus is now tracing. That is the reverse of the old reading, and it is the same
6 ms of tracing as before: the trees fell around it.

**And the trees are flat in the genus after all** -- 5.84 / 5.27 / 5.98 ms -- which is what they
should be, since neither BFS knows how many edges will be left over. The old table's 15 % rise was
an artifact of whatever made the trees fifty milliseconds; there is no finding there to chase.

triwarp is now **1.24x faster** than meshlib at genus 0, level with it at genus 1 (1.04x) and
**2.01x behind** at genus 64 -- where the gap is the per-loop tracing, not the decomposition. The
previous claim that this was the module's largest standing gap no longer holds.

References
----------
**meshlib** is the only library in the suite that computes a homology basis. potpourri3d does not
bind geometry-central's homology code and neither trimesh nor libigl has one, which is why
``tests/test_homology.py`` otherwise stands on invariants. Its ``eliminateTunnels`` -- the consumer
of this basis, benchmarked as ``eliminate_tunnels`` in ``benchmarks/test_repair.py`` -- is a no-op on
every input probed, so that group has no meshlib row even though this one does.
``detectBasisTunnels`` returns a vector
of ``EdgeId`` paths -- the same 2 * genus loops, though not the same ones, since a basis is not
unique (measured on a torus: 32 and 18 edges against MeshLib's 72 and 32).

It reads the mesh and allocates its own scratch, so one ``Mesh`` serves every round, and it takes a
``MeshPart``, which does not own the mesh it wraps -- hence the module-level cache rather than a
temporary.
"""

from __future__ import annotations

import pytest
from conftest import BenchCase
from meshlib import mrmeshpy as mm

import triwarp as tw

_mesh_ml_cache: dict[str, mm.Mesh] = {}


def _mesh_ml(bench_case: BenchCase) -> mm.Mesh:
    """Cache one ``meshlib.Mesh`` per mesh: ``detectBasisTunnels`` does not modify it."""
    if bench_case.mesh_name not in _mesh_ml_cache:
        _mesh_ml_cache[bench_case.mesh_name] = bench_case.new_mesh_ml()
    return _mesh_ml_cache[bench_case.mesh_name]


@pytest.mark.benchmark(group="homology_generators")
@pytest.mark.benchaxis("genus")
@pytest.mark.benchlibs("triwarp", "meshlib")
def test_homology_generators(bench_case: BenchCase) -> None:
    """
    The full basis: spanning tree, cotree, then one loop trace per generator.

    Read against [`test_tree_cotree`] below, which stops before the tracing: the gap between the two
    groups is what the loops cost -- 0.49 ms at genus 1 and **6.1 ms** at genus 64, against 5.3-6.0
    for the trees themselves. So at high genus the tracing is half the call, which is the opposite
    of what this group used to say; see the module docstring for why the older numbers are gone.
    """
    expected = {"sphere_med": 0, "handles_1": 2, "handles_64": 128}[bench_case.mesh_name]
    if bench_case.kind == "meshlib":
        mesh_part_ml = mm.MeshPart(_mesh_ml(bench_case))
        tunnels_ml = bench_case.run(lambda: mm.detectBasisTunnels(mesh_part_ml))
        assert len(tunnels_ml) == expected
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    loops = bench_case.run(lambda: tw.homology.homology_generators(vertices, faces))
    assert len(loops) == expected


@pytest.mark.benchmark(group="tree_cotree")
@pytest.mark.benchaxis("genus")
@pytest.mark.benchlibs("triwarp")
def test_tree_cotree(bench_case: BenchCase) -> None:
    """
    The decomposition alone, without tracing a single loop -- and it is where the time goes.

    triwarp-only by construction, not by omission: MeshLib exposes the finished basis and no
    intermediate, so there is nothing to compare a spanning tree against. It is here to attribute
    ``homology_generators``' cost, and it does -- 5.27 of 5.76 ms at genus 1, but only 5.98 of 12.07
    at genus 64.

    It is **flat** along this axis -- 5.84 / 5.27 / 5.98 ms at genus 0 / 1 / 64 -- which is what it
    should be, since neither BFS knows how many edges will be left over. An earlier reading of this
    same group was ten times larger and *not* flat, and the separate group is what makes the
    difference visible: the non-flatness was in the trees, and it went away with the ten times.
    """
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    edges, generators, parents = bench_case.run(lambda: tw.homology.tree_cotree(vertices, faces))
    assert int(edges.shape[0]) > 0
    assert int(parents.shape[0]) == bench_case.n_vertices
    assert (
        int(generators.shape[0])
        == {"sphere_med": 0, "handles_1": 2, "handles_64": 128}[bench_case.mesh_name]
    )
