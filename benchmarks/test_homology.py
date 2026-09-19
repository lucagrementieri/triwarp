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
| mesh | genus | ``homology_generators`` | meshlib |
|---|---|---|---|
| ``sphere_med`` | 0 | **2.66 ms** | 7.38 ms |
| ``handles_1`` | 1 | **2.42 ms** | 5.52 ms |
| ``handles_64`` | 64 | **3.13 ms** | 5.99 ms |

triwarp leads on all three -- 2.8x, 2.3x and 1.9x -- where it was level with meshlib at genus 1 and
2.01x behind at genus 64. The fused rewrite is worth **2.03 / 2.15 / 3.26x** against the previous
implementation on the same box in one interleaved session, and the ratio rises with the genus
because the tracing it replaced was the part that scaled: 128 loops were walked one at a time in
Python over a ``parents`` array read back in full, and they are now two launches and a scan.

**The axis still separates the two halves of the algorithm**, but they no longer separate in the
clock the way they did. The spread across a 64x genus range is 2.42 to 3.13 ms, against 5.76 to
12.07 before, so the decomposition dominates again and the tracing is a few hundred microseconds of
it. There is no longer a ``tree_cotree`` group to read this one against: the decomposition stopped
being a public entry point when it stopped being reachable except through this function, and the
attribution it existed for is in ``triwarp/kernels/homology.py``'s module docstring.

References
----------
**meshlib** is the only library in the suite that computes a homology basis. potpourri3d does not
bind geometry-central's homology code and neither trimesh nor libigl has one, which is why
``tests/test_homology.py`` otherwise stands on invariants. Its ``eliminateTunnels`` -- the consumer
of this basis, benchmarked as ``remove_tunnels`` in ``benchmarks/test_repair.py`` -- is a no-op
on every input probed, so that group has no meshlib row even though this one does.
``detectBasisTunnels`` returns a vector
of ``EdgeId`` paths -- the same 2 * genus loops, though not the same ones, since a basis is not
unique (measured on a torus: 32 and 18 edges against MeshLib's 72 and 32).

It reads the mesh and allocates its own scratch, so one ``Mesh`` serves every round, and it takes a
``MeshPart``, which does not own the mesh it wraps -- hence the module-level cache rather than a
temporary.
"""

from __future__ import annotations

import pytest
from meshlib import mrmeshpy as mm

import triwarp as tw
from conftest import BenchCase

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
    The full basis: spanning tree, cotree, then one loop trace per generator, in one call.

    The genus axis prices the tracing, which is the half that scales with it -- 2.42 ms at genus 1
    against 3.13 at genus 64, for 126 more loops. It used to be 0.49 ms against 6.1, a Python walk
    per generator over a ``parents`` array read back in full; it is now two launches and a scan, so
    the decomposition is the cost again at every genus this axis reaches.
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
