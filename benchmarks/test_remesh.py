"""
Benchmarks for ``triwarp.remesh``.

Three axes, because the four entry points fail to scale for three different reasons:

* ``subdivide`` on the **scan sweep** -- exactly 4x the faces, one pass, no data dependence. This
  is the module's throughput baseline and the only group here that a face count fully explains.
* ``subdivide_to_size`` on **scale**, sweeping ``max_edge`` -- output size grows as
  ``(L_max / max_edge)^2`` while the *pass count* grows only as ``log2(L_max / max_edge)``, so
  halving the target is roughly four times the output for one extra pass. Edge-length *variance*
  matters too: a single long edge forces another pass over the whole mesh.
* ``flip_to_delaunay`` and ``isotropic_remesh`` on **quality** -- the iterative paths, where the
  number of rounds is set by how far the input is from the fixed point rather than by its size.
  ``saddle_graded`` exists for exactly this: identical connectivity to ``saddle``, worst aspect
  ratio 4 719 against 1.6, so it is badly non-Delaunay and full of slivers while ``saddle`` is
  nearly converged already. A near-Delaunay mesh flips in one or two rounds; a bad one runs all
  ``max_iter`` of them.

Sizing is derived from the mesh's own mean edge length (computed once from the NumPy source so
every library gets the *same* target), which keeps the amount of work proportional to the mesh
rather than to an absolute length that would explode on one mesh and no-op on another.

``flip_to_delaunay`` mutates its face buffer in place, so the timed callable clones it -- the clone
is a single device copy and is negligible against the flip passes it feeds.

References
----------
``subdivide`` has an exact open3d equivalent in ``subdivide_midpoint(1)`` -- same 1:4 midpoint
split, and it returns a new mesh rather than mutating, so the shared mesh is reusable -- and a
trimesh one in ``remesh.subdivide``. ``subdivide_to_size`` has a trimesh counterpart.

Open3D has no ``subdivide_to_size`` (its ``subdivide_midpoint`` takes an iteration count, not an
edge-length target, so it cannot split adaptively), no edge-flip pass, and no isotropic remesher --
``simplify_quadric_decimation`` is decimation, which is the opposite operation. trimesh has neither
of the iterative paths either.

**pymeshlab** is the *only* reference ``isotropic_remesh`` has, and it is an unusually close one:
``meshing_isotropic_explicit_remeshing`` runs the same five stages in the same order (refine,
collapse, edge-swap, Laplacian relax, reproject), each individually switchable, so the comparison is
stage-for-stage rather than algorithm-against-algorithm. Two differences left in place: it preserves
crease edges above ``featuredeg=30`` degrees, which triwarp does not, and its ``checksurfdist``
default rejects any local operation deviating more than 1% of the bbox diagonal from the input. Both
are on, because turning them off would measure a filter no MeshLab user runs.

It also covers ``subdivide_to_size``: ``meshing_surface_subdivision_midpoint(threshold=...)``
refines every edge longer than a length target, which is precisely that operation, and it lands on
the identical output face count (4x at ``0.7 x mean_edge``, 16x at ``0.35 x``). It cannot cover the
uniform ``subdivide`` group, though, and the reason is a hard failure rather than a slow row:
**midpoint subdivision raises** ``PyMeshLabException: Mesh has some not 2 manifold faces,
subdivision surfaces require manifoldness`` on every scan mesh. That is the same boundary the libigl
and potpourri3d references run into (see the benchmarks README hazard table), so the midpoint
reference lives on the ``scale`` axis with ``subdivide_to_size`` instead.

Caps: ``isotropic_remesh`` is capped at ``bunny`` on the scan sweep (it runs ~1.2 s there and ~10x
that on ``dragon``, which would dominate the whole suite); the split paths are capped at ``dragon``
because a 1:4 subdivision of ``happy_buddha`` / ``lucy`` does not fit a sane memory budget. The
``trimesh`` reference for ``subdivide_to_size`` is capped at ``bunny`` on top of that: it takes
~11 s on ``dragon``, for a ratio the smaller meshes already establish, and the pymeshlab one skips
``sphere_large`` for the same reason (5.9 s a call at the finer target).
"""

from __future__ import annotations

import igl
import numpy as np
import pymeshlab as ml
import pytest
import trimesh as tm
import warp as wp
from conftest import BenchCase, skip_larger_than

import triwarp as tw

# Iterations for the full remeshing pipeline. The default is 10; 3 keeps the case under a couple
# of seconds while still exercising the split/collapse/flip/smooth/reproject loop several times.
_REMESH_ITERATIONS = 3

# Split targets as a fraction of the mean edge length. Both are below 1.0 so edges actually split;
# halving the fraction quadruples the output for one extra pass, which is the point of the pair.
_SPLIT_FRACTIONS = [0.7, 0.35]

# The iterative paths run to seconds a call.
_ROUNDS = 3


@pytest.mark.benchmark(group="subdivide")
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d")
def test_subdivide(bench_case: BenchCase) -> None:
    """Exactly 4x the faces in one pass: the module's clean throughput baseline."""
    skip_larger_than(bench_case, "dragon", "a 1:4 subdivision above dragon exceeds memory")
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        _new_vertices, new_faces = bench_case.run(lambda: tw.remesh.subdivide(vertices, faces))
        assert int(new_faces.shape[0]) == 4 * int(faces.shape[0])
    elif bench_case.kind == "trimesh":
        vertices, faces = bench_case.vertices_np, bench_case.faces_np
        _new_vertices, new_faces = bench_case.run(lambda: tm.remesh.subdivide(vertices, faces))
        assert new_faces.shape[0] == 4 * faces.shape[0]
    else:  # open3d midpoint subdivision returns a new mesh, so the shared one is reusable
        mesh_o3d = bench_case.mesh_o3d
        n_faces = bench_case.n_faces
        subdivided = bench_case.run(lambda: mesh_o3d.subdivide_midpoint(number_of_iterations=1))
        assert len(subdivided.triangles) == 4 * n_faces


@pytest.mark.benchmark(group="subdivide_to_size")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "trimesh", "pymeshlab")
@pytest.mark.parametrize("split_fraction", _SPLIT_FRACTIONS)
def test_subdivide_to_size(bench_case: BenchCase, split_fraction: float) -> None:
    """Adaptive splitting to an edge-length target: quadratic in output, logarithmic in passes."""
    max_edge = split_fraction * bench_case.mean_edge
    if bench_case.kind == "pymeshlab":
        if bench_case.mesh_name == "sphere_large":
            pytest.skip("MeshLab's midpoint refinement takes ~6 s at this size and target")
        # ``iterations`` is a pass cap rather than a convergence criterion, so it is set well above
        # the log2 depth the target needs; the surplus passes find nothing to refine.
        bench_case.run(
            lambda: bench_case.new_meshset_pml().meshing_surface_subdivision_midpoint(
                iterations=10, threshold=ml.PureValue(max_edge)
            ),
            rounds=_ROUNDS,
        )
        return
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        _new_vertices, new_faces = bench_case.run(
            lambda: tw.remesh.subdivide_to_size(vertices, faces, max_edge), rounds=_ROUNDS
        )
        assert int(new_faces.shape[0]) >= int(faces.shape[0])
    else:
        if bench_case.mesh_name == "sphere_large":
            pytest.skip("trimesh subdivide_to_size takes tens of seconds at this size")
        vertices, faces = bench_case.vertices_np, bench_case.faces_np
        result = bench_case.run(
            lambda: tm.remesh.subdivide_to_size(vertices, faces, max_edge), rounds=_ROUNDS
        )
        assert result[1].shape[0] >= faces.shape[0]


@pytest.mark.benchmark(group="flip_to_delaunay")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("triwarp")
def test_flip_to_delaunay(bench_case: BenchCase) -> None:
    """Flip rounds until convergence: driven by distance from Delaunay, not by size."""
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp

    # ``flip_to_delaunay`` rewrites the winding in place, so each round needs a fresh buffer;
    # without the clone every round after the first would start from an already-Delaunay mesh.
    def run() -> wp.array[wp.int32]:
        return tw.remesh.flip_to_delaunay(vertices, wp.clone(faces), max_iter=100)

    flipped = bench_case.run(run, rounds=_ROUNDS)
    assert int(flipped.shape[0]) == int(faces.shape[0])


@pytest.mark.benchmark(group="isotropic_remesh")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("triwarp", "pymeshlab")
def test_isotropic_remesh(bench_case: BenchCase) -> None:
    """
    Five stages x ``iterations``, on a nearly-converged mesh and a badly conditioned one.

    MeshLab runs the identical five stages, and the pair reads in opposite directions: **pymeshlab
    339 -> 1 116 ms** across the axis (a 3.3x spread) against **triwarp flat at 123 / 124 ms**, so
    the 2.8x win on ``saddle`` becomes 9.0x on ``saddle_graded``.

    **This is not a like-for-like win, and the timing alone is misleading.** triwarp is flat because
    its loop is a fixed ``iterations`` x five launches whatever the input looks like; MeshLab's
    serial local-operation queue keeps working until the operations stop paying off. Comparing the
    *outputs* at ``iterations=3`` and the identical target length says what the difference buys:

    | | faces | edge / target | zero-area | min angle p1 | aspect p99 / max |
    |---|---|---|---|---|---|
    | ``saddle`` in | 34 848 | 1.00 | 0 | 36.9 deg | 1.64 / 1.66 |
    | ``saddle`` triwarp | 39 100 | 0.93 | 0 | 39.2 deg | 1.58 / 3.5 |
    | ``saddle`` pymeshlab | 34 946 | 0.99 | 0 | 39.6 deg | 1.57 / 2.18 |
    | ``saddle_graded`` in | 34 848 | 1.00 | 0 | 0.013 deg | 4 400 / 4 719 |
    | ``saddle_graded`` tw | 64 867 | 0.73 | 0 | 0.16 deg | 352 / 4 711 |
    | ``saddle_graded`` pml | 31 414 | 0.98 | 0 | 31.9 deg | 1.87 / 3.55 |

    On ``saddle`` triwarp is now at parity (aspect 99th pct 1.58 against 1.57). On the *graded*
    patch it is not: it misses the target length by 27% and leaves a 99th-percentile aspect ratio of
    352 against MeshLab's 1.87, though it no longer makes anything worse than the input.

    Two bugs were found and one fixed by reading this pair against the reference; both were
    pre-existing and invisible to ``tests/test_remesh.py``, which only ever runs the remesher on
    clean closed icospheres:

    * **Fixed** -- the swap stage flipped on the valence objective with only a convexity guard, so
      on a graded mesh it turned slivers into worse slivers and in float32 hit exactly-zero area:
      **2 738 of 84 406 faces**, with the worst aspect ratio reaching 5.7e6. It now also rejects any
      flip that would create a degenerate triangle or worsen the pair's aspect ratio. That is what
      moved the ``saddle`` row to parity and zeroed the degenerate counts above.
    * **Open** -- ``_smooth_pass`` computes the *unweighted* one-ring centroid where this module's
      Notes promise the area-equalizing form. A regular graded grid is valence-perfect (100% of
      interior vertices have valence exactly 6), already Delaunay, *and* a fixed point of the
      unweighted Laplacian, so three of the five stages are blind to its anisotropy by construction
      and only split/collapse act -- reaching a dynamic equilibrium at ~41% short edges (94.8% of
      them collapsible, so it is not the guards). Area-weighting the smoother was measured to
      take the 99th-percentile aspect ratio from **352 to 20** and to improve the icospheres too
      (min angle 45 -> 54 deg), but it makes ``is_watertight`` fail on ``cave_cube`` through a
      self-intersection at *every* step size down to ``lam=0.1``, so it needs a fold guard before it
      can land. Not shipped.
    """
    target = bench_case.mean_edge
    if bench_case.kind == "pymeshlab":
        # ``PureValue`` so both sides get the identical absolute target rather than a percentage of
        # a bounding box the two meshes do not share.
        bench_case.run(
            lambda: bench_case.new_meshset_pml().meshing_isotropic_explicit_remeshing(
                iterations=_REMESH_ITERATIONS, targetlen=ml.PureValue(target)
            ),
            rounds=_ROUNDS,
        )
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    _new_vertices, new_faces = bench_case.run(
        lambda: tw.remesh.isotropic_remesh(
            vertices, faces, target_length=target, iterations=_REMESH_ITERATIONS
        ),
        rounds=_ROUNDS,
    )
    assert int(new_faces.shape[0]) > 0


@pytest.mark.benchmark(group="intrinsic_delaunay")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "igl")
def test_intrinsic_delaunay(bench_case: BenchCase) -> None:
    """
    The flip loop: rounds of predicate, claim, length update and commit until nothing is left.

    On the ``scale`` axis this measures the *no-work* path -- an icosphere is already intrinsically
    Delaunay, so the loop pays one round to discover that and stops -- which is the honest baseline
    for the flag being on by default. The meshes that actually flip are the quad grids, and none of
    them is on this axis; ``tests/test_intrinsic.py`` covers those for correctness.
    """
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        intrinsic_faces, lengths, _ = bench_case.run(
            lambda: tw.remesh.intrinsic_delaunay(vertices, faces)
        )
        assert intrinsic_faces.shape == faces.shape
        assert lengths.shape == (bench_case.n_faces, 3)
    else:  # igl does the flips and the assembly in one call, so its row includes both
        vertices_np = bench_case.vertices_np
        faces_np = np.ascontiguousarray(bench_case.faces_np, dtype=np.int64)
        matrix_igl = bench_case.run(
            lambda: igl.intrinsic_delaunay_cotmatrix(vertices_np, faces_np)[0]
        )
        assert matrix_igl.shape == (bench_case.n_vertices, bench_case.n_vertices)
