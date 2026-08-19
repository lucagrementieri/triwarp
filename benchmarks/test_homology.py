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
| ``sphere_med`` | 0 | 51.8 ms | 51.1 ms | 7.6 ms |
| ``handles_1`` | 1 | 50.4 ms | 50.0 ms | 5.5 ms |
| ``handles_64`` | 64 | 65.1 ms | 58.9 ms | 6.0 ms |

Two things that table says, and the second was not what the axis was built to look for.

**The decomposition is the whole cost**, not the tracing: at genus 1 the trees are 50.0 ms of the
50.4 ms total, and even at genus 64 they are 58.9 of 65.1. So 128 loop traces cost about 6 ms while
the two spanning trees cost fifty, and any work on this module belongs there.

**And the trees are not flat in the genus** -- 51.1 -> 50.0 -> 58.9 ms -- which they should be,
since neither BFS knows how many edges will be left over. A 15 % rise from genus 1 to genus 64 at
an unchanged face count is the finding this group exists to surface; ``handles_64``'s extra 5 % of
faces accounts for part of it and not for all of it. meshlib, by contrast, is flat to within noise
(7.6 / 5.5 / 6.0 ms) and **6.9-10.9x faster**, so this is the module's largest standing gap.

References
----------
**meshlib** is the only library in the suite that computes a homology basis. potpourri3d does not
bind geometry-central's homology code and neither trimesh nor libigl has one, which is why
``tests/test_homology.py`` otherwise stands on invariants. ``detectBasisTunnels`` returns a vector
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
    groups is what the loops cost, and it is **small** -- 0.4 ms at genus 1 and 6.2 ms at genus 64,
    against fifty for the trees themselves (see the module docstring).
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
    ``homology_generators``' cost, and it does -- 50.0 of 50.4 ms at genus 1.

    It was expected to be *flat* along this axis, since neither BFS knows how many edges will be
    left over, and it is not: 51.1 / 50.0 / 58.9 ms at genus 0 / 1 / 64. Whatever that 15 % is, it
    is visible only here, which is the reason to keep the group separate from the one above.
    """
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    edges, generators, parents = bench_case.run(lambda: tw.homology.tree_cotree(vertices, faces))
    assert int(edges.shape[0]) > 0
    assert int(parents.shape[0]) == bench_case.n_vertices
    assert (
        int(generators.shape[0])
        == {"sphere_med": 0, "handles_1": 2, "handles_64": 128}[bench_case.mesh_name]
    )
