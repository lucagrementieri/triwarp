"""
Benchmarks for ``triwarp.validation``: the topological predicates.

Three axes, one per predicate family, because these functions fail to scale for three unrelated
reasons and a face-count sweep separates none of them.

* **valence** for ``is_vertex_manifold``. It solves a miniature connected-components problem in
  every vertex's one-ring, so its cost is the valence *distribution*, not the vertex count.
* **diameter** for ``face_orientation_bits``. Z2 orientation propagation over the face-adjacency
  graph is depth-bound: one round per level, so a long strip costs O(V) rounds where a blob costs
  O(log V). Measured at **7.7 ms on ``sphere_med`` and 633 ms on ``ribbon_long``** -- 82x at an
  identical vertex count, and the direction reverses against trimesh, which goes 16.8 ms -> 7.8 ms
  on the same pair. This is the clearest slowness path the axis set exposes and it is entirely
  invisible to the scan registry, whose meshes are all compact blobs.
* **overlap** for ``is_watertight`` / ``is_volume``. Both compose an edge-count test with a
  self-intersection test over a BVH, so what matters is collision density, not size.

Measured medians (RTX 5090, ``--device=cuda``)
----------------------------------------------
| group | ``sphere_med`` | perturbed | note |
|---|---|---|---|
| ``is_vertex_manifold`` | 1.9 ms | 2.0 ms (``fan_hub``) | valence is not a hot spot |
| ``face_orientation_bits`` | 7.7 ms | 633 ms (``ribbon_long``) | **82x**, depth-bound |
| ``is_watertight`` | 3.7 ms | 3.5 ms (``tangle_2``) | flat; the references are not |

References
----------
**open3d** is the exact definitional equivalent for ``is_watertight`` -- triwarp's docstring
defines itself against ``open3d.geometry.TriangleMesh.is_watertight`` (edge-manifold without
boundary, plus vertex-manifold and no self-intersection). It is also, measured, **13.6 s on
``sphere_med`` and 3.5 s on ``tangle_2``**: roughly 3 700x slower than triwarp, and *faster on the
harder mesh*, because its self-intersection test is a brute-force scan that early-exits on the
first hit and so pays full price only when the mesh is clean. Both points are worth having on the
record, so open3d runs here at ``rounds=1`` rather than being dropped; that one group is about
35 s of the suite's wall clock and it is the reason for the cap.

**libigl**'s ``is_vertex_manifold`` is the reference for the valence group (89.8 ms on
``sphere_med``, and safe on ``fan_hub`` at 92.8 ms -- unlike ``igl.principal_curvature``, which
takes 110 s on the same mesh; see [`test_curvature.py`](test_curvature.py)).

**trimesh** rebuilds its mesh inside the timed callable because it caches derived properties.
Note its ``is_watertight`` is edge-manifold-only, so it is timing context rather than an
equivalent computation. It has no ``is_vertex_manifold`` in 5.0; ``tm.repair.fix_winding`` is the
closest analogue of the orientation propagation and is timed against it.

There is no open3d ``is_volume``: its closest composition (``is_watertight() and is_orientable()``)
short-circuits on the first check, so it would time ``is_watertight`` under a different name.
"""

from __future__ import annotations

import igl
import numpy as np
import pytest
import trimesh as tm
from conftest import BenchCase

import triwarp as tw

# open3d's is_watertight runs into seconds (see the module docstring); one round is enough to
# record the magnitude without letting it dominate the suite.
_O3D_ROUNDS = 1


@pytest.mark.benchmark(group="is_vertex_manifold")
@pytest.mark.benchaxis("valence")
@pytest.mark.benchlibs("triwarp", "igl")
def test_is_vertex_manifold(bench_case: BenchCase) -> None:
    """
    A connected-components problem per one-ring: driven by the valence distribution.

    Note the two sides return different shapes -- triwarp reduces to a single ``bool`` while
    ``igl.is_vertex_manifold`` hands back the per-vertex mask (triwarp's ``vertex_manifold_mask``
    is the equivalent of that). The work is the same either way; only the final reduction differs.
    """
    if bench_case.kind == "triwarp":
        assert bench_case.run(lambda: tw.validation.is_vertex_manifold(bench_case.faces_wp)) in (
            True,
            False,
        )
    else:
        faces_np = np.ascontiguousarray(bench_case.faces_np, dtype=np.int64)
        mask_igl = np.asarray(bench_case.run(lambda: igl.is_vertex_manifold(faces_np)))
        assert mask_igl.shape == (bench_case.n_vertices,)


@pytest.mark.benchmark(group="face_orientation_bits")
@pytest.mark.benchaxis("diameter")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_face_orientation_bits(bench_case: BenchCase) -> None:
    """Z2 orientation propagation: one round per graph level, so depth is the whole cost."""
    if bench_case.kind == "triwarp":
        faces = bench_case.faces_wp
        _bits, _edges, _seeds, n_components = bench_case.run(
            lambda: tw.validation.face_orientation_bits(faces)
        )
        assert n_components >= 1
    else:  # trimesh's winding fix walks the same adjacency graph; rebuild inside (it mutates)
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np

        def fix_winding_tm() -> tm.Trimesh:
            mesh_tm = tm.Trimesh(vertices_np, faces_np, process=False)
            tm.repair.fix_winding(mesh_tm)
            return mesh_tm

        assert len(bench_case.run(fix_winding_tm).faces) == bench_case.n_faces


@pytest.mark.benchmark(group="is_watertight")
@pytest.mark.benchaxis("overlap")
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d")
def test_is_watertight(bench_case: BenchCase) -> None:
    """Edge counts plus a self-intersection pass: driven by collision density, not size."""
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        result = bench_case.run(lambda: tw.validation.is_watertight(vertices, faces))
    elif bench_case.kind == "trimesh":
        vertices, faces = bench_case.vertices_np, bench_case.faces_np
        result = bench_case.run(lambda: tm.Trimesh(vertices, faces, process=False).is_watertight)
    else:  # open3d: same definition as triwarp's, and it does not cache the answer
        mesh_o3d = bench_case.mesh_o3d
        result = bench_case.run(mesh_o3d.is_watertight, rounds=_O3D_ROUNDS)
    assert result in (True, False)


@pytest.mark.benchmark(group="is_volume")
@pytest.mark.benchaxis("overlap")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_is_volume(bench_case: BenchCase) -> None:
    """``is_watertight`` plus winding plus a signed-volume sign: the priciest predicate."""
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        result = bench_case.run(lambda: tw.validation.is_volume(vertices, faces))
    else:
        vertices, faces = bench_case.vertices_np, bench_case.faces_np
        result = bench_case.run(lambda: tm.Trimesh(vertices, faces, process=False).is_volume)
    assert result in (True, False)
