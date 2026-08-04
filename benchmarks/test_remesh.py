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

``flip_by_objective`` joins ``flip_to_delaunay`` on the **quality** axis for the same reason: it is
the same engine with a different predicate at the front, so the pair isolates what the predicate
costs from what the flip machinery costs. Both objectives are rows, and they behave differently on
that axis by construction — the planarity one refuses any quad that is not flat, so on a curved
input it converges in one pass, while the curvature one has something to do everywhere. MeshLab's
``meshing_edge_flip_by_planar_optimization`` is the reference for the first, at the same
``pthreshold`` and the same ``planartype``; it rewrites the topology, so that row rebuilds the
MeshSet.

``quadric_decimate`` is the module's **quality**-axis decimator, and the pair with
``cluster_decimate`` is the point: they solve the same problem with opposite structures. Clustering
is three data-independent passes; the quadric method is an iterated greedy loop whose pass count
depends on how contested the rings are, which is exactly what triangle shape changes. All three
serial references are here -- ``igl.decimate``, Open3D's ``simplify_quadric_decimation`` and
MeshLab's ``meshing_decimation_quadric_edge_collapse`` -- because this is the best-referenced port
in the package. Two notes for reading them: the MeshLab filter defaults to ``autoclean=True`` and
deletes unreferenced vertices, so its MeshSet is rebuilt per round; and the *quality* comparison is
in ``tests/test_remesh.py`` rather than here, where triwarp measures a **lower** Hausdorff error
than all three at the same face count (0.0133 against igl's 0.0250 and Open3D's 0.0236 at 512
faces).

**And triwarp is still the slower one on this group**, which is worth stating plainly: it measures
~253 ms on ``saddle_graded`` at ``target_ratio=0.1`` against igl's 80. The reason is the *pass
count*, not the per-pass work -- one hashed-key independent set commits roughly ``candidates / 50``
collapses, so reaching a tenth of the faces takes tens of passes and each pays a full
edge/adjacency/quadric rebuild plus two radix sorts, which a serial queue pays none of. Committing
**several** independent sets against one rebuild closed 1.3-1.8x of that (see ``_QUADRIC_RATIOS``
below for the numbers and for what the *other* candidate lever was measured to be worth), and the
rest of the gap is the same structural trade: the parallel formulation buys quality rather than
speed here.

``cluster_decimate`` is the module's other **scan sweep** group, and the interesting one to read
against ``subdivide``: it is the same shape of work in reverse (bin, remap, dedup -- no data
dependence, no iteration), so if the two do not scale alike something is wrong with one of them.
Both open3d (``simplify_vertex_clustering``) and pymeshlab (``meshing_decimation_clustering``) are
references, and the open3d one is *exact* -- same grid anchor, same cell means, identical face count
-- which is rare enough in this suite to be worth stating. Cell width is a fraction of the mean edge
length so the decimation ratio is comparable across meshes.

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
@pytest.mark.benchlibs("triwarp", "trimesh", "igl", "open3d")
def test_subdivide(bench_case: BenchCase) -> None:
    """
    Exactly 4x the faces in one pass: the module's clean throughput baseline.

    Four libraries, one algorithm -- ``igl.upsample`` is midpoint 1:4 subdivision with triwarp's
    semantics exactly (new vertex per unique edge, original vertices untouched), so this is the
    module's widest reference agreement. All three references differ from triwarp only in the output
    *ordering*, which is what makes the parity comparison a centroid match rather than an array
    compare (``tests/test_remesh.py``).
    """
    skip_larger_than(bench_case, "dragon", "a 1:4 subdivision above dragon exceeds memory")
    if bench_case.kind == "igl":
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        vertices_igl, faces_igl = bench_case.run(lambda: igl.upsample(vertices_np, faces_np))
        assert faces_igl.shape[0] == 4 * bench_case.n_faces
        assert vertices_igl.shape[0] > bench_case.n_vertices
        return
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


@pytest.mark.noparity(
    "pymeshlab",
    reason="D2 the same five stages under a different stopping rule, with the disagreement "
    "measured and tabulated in this function's own docstring: triwarp runs a fixed iterations x "
    "five parallel launches while MeshLab works a serial local-operation queue until the "
    "operations stop paying off, so at iterations=3 and the identical target length the two "
    "return different meshes with no vertex correspondence -- 39 100 faces against 34 946 on "
    "saddle, and on saddle_graded triwarp misses the target edge length by 27% (0.73 of target "
    "against 0.98) and leaves a 99th-percentile "
    "aspect ratio of 352 against 1.87. That gap is the finding this row exists to report, not a "
    "tolerance to widen; the quality statistics themselves are asserted against the *input* in "
    "tests/test_remesh.py, in test_remesh_edge_concentration and "
    "test_remesh_emits_no_degenerate_faces, rather than against MeshLab.",
)
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


# Cluster cell widths as a multiple of the mean edge length. 2x welds most 1-rings into one vertex
# (roughly a 4x face reduction) and 6x is aggressive decimation; below 1x almost nothing merges.
_CLUSTER_FACTORS = [2.0, 6.0]


@pytest.mark.noparity(
    "pymeshlab",
    oracle="open3d",
    reason="D2 a differently anchored grid with a different cell representative, and open3d is "
    "the exact oracle right beside it: meshing_decimation_clustering keeps a per-cell "
    "representative on a grid whose origin is not open3d's min_bound - voxel_size / 2, so at the "
    "same threshold it returns measurably different meshes -- 2 792 faces against triwarp's 2 768 "
    "at a 0.1 cell on icosphere(4), then 656 against 768 at 0.2 and 156 against 252 at 0.4, i.e. "
    "1% to 62% apart and diverging as the cell grows. open3d assigns cells identically and is "
    "asserted to the exact face and vertex count in "
    "tests/test_remesh.py::test_cluster_decimate_matches_open3d, which also pins the grid anchor; "
    "MeshLab cannot be a second oracle for a quantity open3d already fixes exactly.",
)
@pytest.mark.benchmark(group="cluster_decimate")
@pytest.mark.benchlibs("triwarp", "open3d", "pymeshlab")
@pytest.mark.parametrize("cell_factor", _CLUSTER_FACTORS)
def test_cluster_decimate(bench_case: BenchCase, cell_factor: float) -> None:
    """
    Bin, remap, dedup: decimation with no priority queue, against two exact-ish references.

    Its axis spans the wrapper floor and the win: ``bunny_decimated`` sits at the ~340 µs floor
    (see ``test_creation::test_box``) and loses 1.5x, while ``dragon`` wins **76x**. A single point
    from this group is uninterpretable in either direction.
    """
    voxel_size = cell_factor * bench_case.mean_edge
    if bench_case.kind == "pymeshlab":
        # ``meshing_decimation_clustering`` rewrites the topology, so the MeshSet is rebuilt inside
        # the timed callable. Its ``threshold`` is a length, hence ``PureValue`` fed from the same
        # ``voxel_size`` both other rows get rather than a percentage of its own bbox diagonal.
        skip_larger_than(bench_case, "bunny", "MeshLab's clustering is a serial pass over the grid")
        new_meshset_pml = bench_case.new_meshset_pml

        def cluster_pml() -> int:
            meshset_pml = new_meshset_pml()
            meshset_pml.meshing_decimation_clustering(threshold=ml.PureValue(voxel_size))
            return meshset_pml.current_mesh().face_number()

        assert bench_case.run(cluster_pml) >= 0
        return
    if bench_case.kind == "open3d":
        # Returns a new mesh and never touches its input, so the shared mesh is safe across rounds.
        mesh_o3d = bench_case.mesh_o3d
        simplified_o3d = bench_case.run(
            lambda: mesh_o3d.simplify_vertex_clustering(voxel_size=voxel_size)
        )
        assert len(simplified_o3d.triangles) <= bench_case.n_faces
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    _decimated_vertices, decimated_faces = bench_case.run(
        lambda: tw.remesh.cluster_decimate(vertices, faces, voxel_size=voxel_size)
    )
    assert int(decimated_faces.shape[0]) // 3 <= bench_case.n_faces


@pytest.mark.benchmark(group="flip_by_objective")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("triwarp", "pymeshlab")
@pytest.mark.parametrize("objective", ["planarity", "curvature"])
def test_flip_by_objective(bench_case: BenchCase, objective: str) -> None:
    """The Delone engine with another predicate: the same passes, a different candidate set."""
    if bench_case.kind == "pymeshlab":
        if objective == "curvature":
            pytest.skip("MeshLab's curvature flip has no comparable parameterization")
        new_meshset_pml = bench_case.new_meshset_pml

        def flip_pml() -> int:
            meshset_pml = new_meshset_pml()
            meshset_pml.meshing_edge_flip_by_planar_optimization(
                pthreshold=1.0, planartype="area/max side", iterations=1
            )
            return meshset_pml.current_mesh().face_number()

        assert bench_case.run(flip_pml, rounds=_ROUNDS) == bench_case.n_faces
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    flipped = bench_case.run(
        lambda: tw.remesh.flip_by_objective(vertices, faces, objective=objective), rounds=_ROUNDS
    )
    assert int(flipped.shape[0]) == int(faces.shape[0])


# Reduction ratios for the quadric decimator. 0.5 is a mild pass and 0.1 is the ratio MeshLab's own
# dialogue defaults near; the loop count grows as the target falls, which is the shape of this
# group.
#
# That shape is the whole cost: at 0.1 the call is **92 % host** (354 ms wall against 28.5 ms of
# device time on ``saddle_graded``) across 69-93 geometry rebuilds of ~40 wrapper calls each. Two
# things were priced against that. Hoisting the twelve per-pass allocations is worth **3.6 % at
# most** and was not done: a memset that initializes a kernel input cannot be removed by moving
# the allocation. Committing several independent sets per rebuild *is* the lever, and cuts the
# rebuild count by roughly the round count: **1.3-1.8x**, measured back to back
# (``saddle_graded`` at 0.1: 461 -> 253 ms; ``icosphere(4)`` to 2 048 faces: 158 -> 94). Note
# that graph capture is **not** available behind either: the pass body contains a host readback
# that decides the loop's exit, and two data-dependent output shapes.
_QUADRIC_RATIOS = [0.5, 0.1]


@pytest.mark.benchmark(group="quadric_decimate")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("triwarp", "igl", "open3d", "pymeshlab")
@pytest.mark.parametrize("target_ratio", _QUADRIC_RATIOS)
def test_quadric_decimate(bench_case: BenchCase, target_ratio: float) -> None:
    """Greedy quadric collapses to a face budget: batched independent sets against three queues."""
    target_faces = max(4, int(target_ratio * bench_case.n_faces))
    if bench_case.kind == "pymeshlab":
        # ``autoclean=True`` (the default) deletes unreferenced vertices, so this filter is not
        # idempotent even in geometry and the MeshSet has to be rebuilt inside the timed callable.
        new_meshset_pml = bench_case.new_meshset_pml

        def decimate_pml() -> int:
            meshset_pml = new_meshset_pml()
            meshset_pml.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)
            return meshset_pml.current_mesh().face_number()

        assert bench_case.run(decimate_pml, rounds=_ROUNDS) > 0
        return
    if bench_case.kind == "igl":
        vertices_np = bench_case.vertices_np
        faces_np = np.ascontiguousarray(bench_case.faces_np, dtype=np.int64)
        result_igl = bench_case.run(
            lambda: igl.decimate(vertices_np, faces_np, target_faces), rounds=_ROUNDS
        )
        assert np.asarray(result_igl[1]).shape[0] > 0
        return
    if bench_case.kind == "open3d":
        # Returns a new mesh and never touches its input, so the shared mesh is safe across rounds.
        mesh_o3d = bench_case.mesh_o3d
        simplified_o3d = bench_case.run(
            lambda: mesh_o3d.simplify_quadric_decimation(target_number_of_triangles=target_faces),
            rounds=_ROUNDS,
        )
        assert len(simplified_o3d.triangles) > 0
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    _decimated_vertices, decimated_faces = bench_case.run(
        lambda: tw.remesh.quadric_decimate(vertices, faces, target_faces=target_faces),
        rounds=_ROUNDS,
    )
    assert int(decimated_faces.shape[0]) // 3 <= bench_case.n_faces
