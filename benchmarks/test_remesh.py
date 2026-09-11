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

triwarp now has a counterpart to the second of those (``max_deviation``), and it is still **not**
passed here — measured a no-op at MeshLab's own 1% default on both fixtures, byte-identical face
counts, because these patches never drift that far. See this group's ``noparity`` reason for the
numbers. Passing it would buy a per-iteration closest-point pass and no change in output.

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
import pytorch3d.ops as p3d_ops
import trimesh as tm
import warp as wp
from meshlib import mrmeshpy as mm

import triwarp as tw
from conftest import BenchCase, face_bitset_ml, mesh_ml_from_numpy, skip_larger_than

# MeshLib expresses several gates as *absolute* lengths whose defaults assume a unit-scale mesh
# (``SubdivideSettings.maxDeviationAfterFlip`` is 1.0). Passing this instead disables the gate
# rather than letting the fixture's units decide whether it binds.
_UNBOUNDED = 1e30

_mesh_ml_cache: dict[str, mm.Mesh] = {}


def _mesh_ml(bench_case: BenchCase) -> mm.Mesh:
    """
    Cache one ``meshlib.Mesh`` per mesh, for the *non*-mutating rows only.

    Every other meshlib row in this file rebuilds inside its timed callable, because the operation
    rewrites the mesh. ``verticesGridSampling`` is the exception -- it returns a bitset and leaves
    the mesh alone -- so its row may share one, and sharing also keeps the lazily built AABB tree
    warm across rounds.
    """
    if bench_case.mesh_name not in _mesh_ml_cache:
        _mesh_ml_cache[bench_case.mesh_name] = bench_case.new_mesh_ml()
    return _mesh_ml_cache[bench_case.mesh_name]


# Iterations for the full remeshing pipeline. The default is 10; 3 keeps the case under a couple
# of seconds while still exercising the split/collapse/flip/smooth/reproject loop several times.
_REMESH_ITERATIONS = 3

# Split targets as a fraction of the mean edge length. Both are below 1.0 so edges actually split;
# halving the fraction quadruples the output for one extra pass, which is the point of the pair.
_SPLIT_FRACTIONS = [0.7, 0.35]

# The iterative paths run to seconds a call.
_ROUNDS = 3


@pytest.mark.benchmark(group="subdivide")
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d", "pytorch3d")
def test_subdivide(bench_case: BenchCase) -> None:
    """
    Exactly 4x the faces in one pass: the module's clean throughput baseline.

    Both references differ from triwarp only in the output *ordering*, which is what makes the
    parity comparison a centroid match rather than an array compare (``tests/test_remesh.py``).

    !!! warning "``igl.upsample`` is memory-unsafe on the scan meshes and is deliberately absent"
        It is midpoint 1:4 subdivision with triwarp's semantics exactly, so it *should* be the
        widest reference agreement in the module -- but on a scan mesh it corrupts the process heap
        and takes the whole pytest session down with a SIGSEGV, which loses every other row in this
        module because ``--benchmark-json`` is only written at session end.

        Measured, one selection per process, `-p no:randomly`: the igl rows crash **2 of 8** runs
        with no other library in the selection at all, **8 of 8** with the trimesh / open3d /
        triwarp rows present, and per mesh **7 of 8** on ``bunny_decimated``, **7 of 8** on
        ``bunny`` and **0 of 8** on ``dragon``. So it is not an interaction with triwarp (an
        earlier reading of this crash said it was), and it is not a size limit -- the *smallest*
        mesh fails most and the largest never does. Compacting the unreferenced vertices away does
        not help either (``bunny`` compacted crashes 3 of 3 where uncompacted survived), so there
        is nothing to pass it that makes it safe.

        This is the third memory-unsafe binding in this wheel, alongside ``igl.loop``'s
        ``free(): invalid pointer`` and ``igl.in_element``'s ``malloc(): invalid size``; see the
        libigl hazards in ``.claude/CLAUDE.md`` §6. It stays a *tested* reference on the small clean
        ``icosahedron`` fixture, where 1 200 calls across six processes are clean --
        ``tests/test_remesh.py::test_subdivide_matches_igl`` keeps the exact class-B comparison.

    **pytorch3d** is the third reference and the only GPU one. ``SubdivideMeshes()`` is
    constructed *inside* the timed callable on purpose: passing a mesh to its constructor caches
    the subdivision topology and every later call reuses it, which would time a gather rather than
    a subdivision -- and triwarp's row rebuilds everything each call. It agrees with triwarp on
    sorted coordinates at **0.0** (``tests/test_remesh.py::test_subdivide_matches_pytorch3d``), so
    like the other two it differs only in output ordering. The row carries its ``Meshes`` build,
    which is cheap next to a 4x face expansion.
    """
    skip_larger_than(bench_case, "dragon", "a 1:4 subdivision above dragon exceeds memory")
    if bench_case.kind == "pytorch3d":
        mesh_p3d = bench_case.mesh_p3d
        n_faces = bench_case.n_faces
        subdivided_p3d = bench_case.run(lambda: p3d_ops.SubdivideMeshes()(mesh_p3d))
        assert subdivided_p3d.faces_packed().shape[0] == 4 * n_faces
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


@pytest.mark.benchmark(group="subdivide_loop")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "igl", "open3d")
def test_subdivide_loop(bench_case: BenchCase) -> None:
    """
    Loop subdivision: the same 1:4 split as ``subdivide``, with smooth stencils not midpoints.

    Read against ``subdivide`` above, which does identical topology work and returns the same
    output size: the gap between them is the cost of the stencils alone -- two extra atomic
    accumulation passes (per-edge opposite vertices, per-vertex ring sums) and two position
    kernels over grids the midpoint split never visits. The two groups sit on different axes for
    the reason below, so compare per-triangle throughput rather than raw medians.

    ``igl.loop`` and Open3D's ``subdivide_loop`` are the same variant as triwarp -- the two
    agree with each other to 2e-16 on the relocated originals, all three using **Warren's**
    ``beta`` (``3/(8k)``, ``3/16`` at ``k = 3``). ``igl.loop``'s ``number_of_subdivs`` stays at
    1, which is one triwarp call. Both references run into the hundreds of milliseconds here and
    take ``rounds=3``.

    **This group is on the ``scale`` axis, not the scan sweep, because ``igl.loop`` cannot
    survive the scan meshes.** Measured, with no Warp in the process: it aborts with ``free():
    invalid pointer`` on a five-vertex mesh with three faces on one edge, and SIGSEGVs (exit
    139) on ``bunny_decimated``, whose 87 duplicated faces leave it not edge-manifold. On
    ``bunny`` it does return, with **1 113 silent ``NaN`` rows** -- one per unreferenced vertex,
    since ``igl::adjacency_list`` is sized ``F.max() + 1`` where ``igl::loop`` indexes it to
    ``n_verts``. That is exactly the hazard the ``scale`` axis exists for.

    ``trimesh.remesh.subdivide_loop`` gets no row either, for two independent reasons: it uses
    Loop's *original* trigonometric ``beta``, ``(1/k)(5/8 - (3/8 + cos(2 pi/k)/4)^2)``, which
    agrees with Warren's exactly at valence 6 and differs elsewhere (6.79e-3 at the twelve
    valence-5 vertices of ``icosphere(1)`` against 1.2e-16 at the thirty valence-6 ones), so it
    could not be a parity oracle; and it divides by the neighbour count, so its own ``assert
    np.isfinite`` fails on any mesh with an unreferenced vertex.

    triwarp keeps an unreferenced vertex where it is and has no stencil that can divide by zero,
    which is why its row is the one that asserts finiteness.
    """
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        new_vertices, new_faces = bench_case.run(lambda: tw.remesh.subdivide_loop(vertices, faces))
        assert int(new_faces.shape[0]) == 4 * int(faces.shape[0])
        assert np.isfinite(new_vertices.numpy()).all()
    elif bench_case.kind == "igl":
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        vertices_loop_igl, faces_loop_igl = bench_case.run(
            lambda: igl.loop(vertices_np, faces_np), rounds=3
        )
        assert faces_loop_igl.shape[0] == 4 * bench_case.n_faces
        assert np.isfinite(vertices_loop_igl).all()
    else:  # open3d returns a new mesh, so the shared one is reusable
        mesh_o3d = bench_case.mesh_o3d
        n_faces = bench_case.n_faces
        subdivided = bench_case.run(
            lambda: mesh_o3d.subdivide_loop(number_of_iterations=1), rounds=3
        )
        assert len(subdivided.triangles) == 4 * n_faces


@pytest.mark.benchmark(group="subdivide_to_size")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "trimesh", "pymeshlab", "meshlib")
@pytest.mark.parametrize("split_fraction", _SPLIT_FRACTIONS)
def test_subdivide_to_size(bench_case: BenchCase, split_fraction: float) -> None:
    """
    Adaptive splitting to an edge-length target: quadratic in output, logarithmic in passes.

    meshlib's ``subdivideMesh`` takes the same absolute ``maxEdgeLen`` and makes the same guarantee
    -- every edge under the cap -- but it also *flips* as it splits, so it is doing more than the
    other three rows and lands on a slightly different mesh (3 320 faces against triwarp's 3 200 on
    a test-size sphere; ``tests/test_remesh.py``). ``maxEdgeSplits`` is raised from its default of
    1 000, which would stop the reference long before the cap on any registry mesh, and
    ``maxDeviationAfterFlip`` from its default of 1.0 -- an *absolute* length, so leaving it would
    make the flip gate depend on the fixture's units. It mutates in place, so its mesh is rebuilt
    inside the timed callable.
    """
    max_edge = split_fraction * bench_case.mean_edge
    if bench_case.kind == "meshlib":
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np

        def subdivide_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(vertices_np, faces_np)
            settings_ml = mm.SubdivideSettings()
            settings_ml.maxEdgeLen = max_edge
            settings_ml.maxEdgeSplits = 10_000_000
            settings_ml.maxDeviationAfterFlip = _UNBOUNDED
            mm.subdivideMesh(mesh_ml, settings_ml)
            return mesh_ml.topology.numValidFaces()

        assert bench_case.run(subdivide_ml, rounds=_ROUNDS) >= bench_case.n_faces
        return
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


@pytest.mark.benchmark(group="split_edges")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp")
@pytest.mark.parametrize("split_fraction", [0.25, 1.0], ids=["quarter", "all"])
def test_split_edges(bench_case: BenchCase, split_fraction: float) -> None:
    """
    One crack-free split pass over a given edge mask: the primitive ``subdivide_to_size`` iterates.

    Timed separately from that group because the two answer different questions.
    ``subdivide_to_size`` measures the whole *loop* — how many passes a length target needs — and
    hides the per-pass cost inside it; this measures one pass at a known mask density, so a
    regression in the emission templates or in the ``edges_unique`` build shows up here undiluted.
    The ``all`` case is the regular 1-to-4 subdivision and should track the ``subdivide`` group
    closely (same output, one extra edge-table build); the gap between ``quarter`` and ``all`` is
    how much of the pass is fixed edge-table cost rather than emission, and it should be *sublinear*
    in the mask density because the sort runs over every edge either way.

    The mask is built on the host once per case and excluded from the timing, since choosing which
    edges to split is the caller's job and varies per use.
    """
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    unique_edges, inverse = tw.edges.edges_unique(faces)
    n_edges = int(unique_edges.shape[0])
    if split_fraction >= 1.0:
        mask = wp.full(n_edges, True, dtype=wp.bool, device=bench_case.device)
    else:
        rng = np.random.default_rng(20260811)
        mask = wp.array(
            np.ascontiguousarray(rng.random(n_edges) < split_fraction),
            dtype=wp.bool,
            device=bench_case.device,
        )
    new_vertices, new_faces = bench_case.run(
        lambda: tw.remesh.split_edges(
            vertices, faces, mask, unique_edges=unique_edges, inverse=inverse
        ),
        rounds=_ROUNDS,
    )
    assert int(new_faces.shape[0]) >= int(faces.shape[0])
    assert int(new_vertices.shape[0]) >= int(vertices.shape[0])


# Split budgets for the region refiner, as a fraction of the input face count. ``None`` runs to
# convergence; the finite one makes ``max_splits`` bind, which is the only way the budget-truncation
# branch is reached at all.
_REGION_SPLIT_BUDGETS = [None, 0.25]


def _region_half_np(bench_case: BenchCase) -> np.ndarray:
    """Dense face mask covering the half of the mesh below the median face-centroid ``x``."""
    vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
    centroid_x = vertices_np[faces_np, 0].mean(axis=1)
    return np.ascontiguousarray(centroid_x < np.median(centroid_x))


def _region_half(bench_case: BenchCase) -> wp.array[wp.bool]:
    """Face mask covering the half of the mesh below the median face-centroid ``x``."""
    return wp.array(_region_half_np(bench_case), dtype=wp.bool, device=bench_case.device)


@pytest.mark.benchmark(group="subdivide_region_to_size")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "meshlib")
@pytest.mark.parametrize("split_budget", _REGION_SPLIT_BUDGETS)
def test_subdivide_region_to_size(bench_case: BenchCase, split_budget: float | None) -> None:
    """
    The same adaptive split restricted to a face region, with and without a split budget.

    The pair is the point rather than either row alone. ``max_splits=None`` refines the region to
    convergence; a finite budget stops early *and* takes a branch the unbudgeted path never
    reaches, ``_keep_longest_edges``, which ranks the pass's eligible edges and keeps the longest
    ones that fit. Only the budgeted row prices that ranking, and it runs **once per call** rather
    than once per pass: truncating a pass spends the whole remaining budget, so the next pass exits
    at ``remaining <= 0``.

    meshlib is the only reference with the region restriction, and it has the *whole* parameter set:
    ``SubdivideSettings`` carries ``region``, ``maxEdgeLen``, ``maxEdgeSplits``,
    ``maxAngleChangeAfterFlip`` and ``maxDeviationAfterFlip``, which is this function's signature
    argument for argument -- so both rows are given the same target, the same split budget and the
    same unbounded flip gate. trimesh's and MeshLab's refiners take an edge-length target over the
    *whole* mesh with no region restriction and no split cap, which is why the sibling
    ``subdivide_to_size`` group carries them and this one does not.

    It is doing slightly more than triwarp for the same reason its sibling row is: ``subdivideMesh``
    flips as it splits, so it lands on a nearby rather than identical mesh -- measured 3 256 faces
    against triwarp's 3 200 on an ``icosphere(3)`` upper half, and 0.014 % less area because a flip
    moves the surface where a split does not (``tests/test_remesh.py``). Two traps in the settings
    object: ``maxEdgeSplits`` defaults to 1 000, which stops it long before the cap on any registry
    mesh, and ``maintainRegion`` is a ``FaceBitSet`` rather than the bool its name reads as.
    """
    max_edge = 0.35 * bench_case.mean_edge
    max_splits = None if split_budget is None else int(split_budget * bench_case.n_faces)

    if bench_case.kind == "meshlib":
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        region_np = _region_half_np(bench_case)
        budget_ml = 10_000_000 if max_splits is None else max_splits

        def subdivide_region_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(vertices_np, faces_np)
            settings_ml = mm.SubdivideSettings()
            settings_ml.maxEdgeLen = max_edge
            settings_ml.maxEdgeSplits = budget_ml
            settings_ml.maxDeviationAfterFlip = _UNBOUNDED
            settings_ml.region = face_bitset_ml(region_np)
            mm.subdivideMesh(mesh_ml, settings_ml)
            return mesh_ml.topology.numValidFaces()

        assert bench_case.run(subdivide_region_ml, rounds=_ROUNDS) >= bench_case.n_faces
        return

    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    region = _region_half(bench_case)
    _new_vertices, new_faces, new_region = bench_case.run(
        lambda: tw.remesh.subdivide_region_to_size(
            vertices, faces, region, max_edge, max_splits=max_splits
        ),
        rounds=_ROUNDS,
    )
    assert int(new_faces.shape[0]) >= int(faces.shape[0])
    assert int(new_region.shape[0]) == int(new_faces.shape[0]) // 3


@pytest.mark.benchmark(group="refine_region_to_density")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("triwarp")
def test_refine_region_to_density(bench_case: BenchCase) -> None:
    """
    Liepa's density criterion instead of a target edge length, on the axis it exists for.

    The ``quality`` axis is the one that separates the two refiners: ``saddle`` and
    ``saddle_graded`` have identical vertex counts, face counts and connectivity, and differ only in
    how unevenly the triangles are sized. A target-length refiner does the same work on both -- it
    compares every edge against one number -- where this one compares each triangle against its
    *own* neighbourhood, so the graded row is where its per-vertex scale attribute earns its keep
    and the uniform row is the control.

    The cost is one scan pair plus one kernel per pass, and the split is 1 -> 3 at the centroid, so
    unlike edge bisection there is no crack-free template and no agreement with the neighbours --
    which is what makes the criterion cheap in parallel rather than merely correct.

    There is no reference row. pymeshfix performs this refinement (measured: 47 vertices inserted
    into a 24-edge rim's patch) but only as a stage inside ``fill_small_boundaries``, behind a load
    that is 90 % of the round and cannot be hoisted out of it, so a row would price the load;
    ``tests/test_holes.py`` compares the answers instead. pymeshlab's ``refineholeedgelen`` is a
    target *length*, i.e. the other criterion.
    """
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    region = _region_half(bench_case)

    _new_vertices, new_faces, new_region = bench_case.run(
        lambda: tw.remesh.refine_region_to_density(vertices, faces, region), rounds=_ROUNDS
    )
    assert int(new_faces.shape[0]) >= int(faces.shape[0])
    assert int(new_region.shape[0]) == int(new_faces.shape[0]) // 3


@pytest.mark.benchmark(group="flip_to_delaunay")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("triwarp", "meshlib")
def test_flip_to_delaunay(bench_case: BenchCase) -> None:
    """
    Flip rounds until convergence: driven by distance from Delaunay, not by size.

    meshlib's ``makeDeloneEdgeFlips`` is the same empty-circumcircle criterion reached by a serial
    queue where triwarp commits a conflict-free independent set per round, so this pair is the
    clearest parallel-against-serial contrast in the module -- and the two reach the *same*
    fixpoint: after triwarp's pass MeshLib finds zero further flips to make
    (``tests/test_remesh.py``). It mutates, so its mesh is rebuilt inside the timed callable, which
    is the same thing the triwarp row's ``wp.clone`` is doing and for the same reason: without it,
    rounds 2..n would start from an already-Delaunay mesh.
    """
    if bench_case.kind == "meshlib":
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np

        def flip_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(vertices_np, faces_np)
            return mm.makeDeloneEdgeFlips(mesh_ml, mm.DeloneSettings(), 100)

        assert bench_case.run(flip_ml, rounds=_ROUNDS) >= 0
        return
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
    "return different meshes with no vertex correspondence -- 35 568 faces against 34 946 on "
    "saddle, and on saddle_graded 41 604 against 31 414 with triwarp missing the target edge "
    "length by 15% (0.854 of target against 0.983) and leaving a 99th-percentile aspect ratio of "
    "8.60 against 1.87. That gap is the finding this row exists to report, not a tolerance to "
    "widen; the quality statistics themselves are asserted against MeshLib and against the *input* "
    "in tests/test_remesh.py, in test_remesh_edge_concentration and "
    "test_remesh_emits_no_degenerate_faces, rather than against MeshLab. The aspect-ratio half of "
    "that gap used to read 352 against 1.87 and has closed 41x since, in two steps and for one "
    "reason each: the collapse stage gained the fold veto its quadric sibling already had "
    "(352 -> 9.13), and the smooth stage became the area-equalizing relaxation its Notes promise "
    "rather than the plain one-ring centroid (9.13 -> 8.60). The stopping rule is what is left. "
    "Re-measured once before that, after isotropic_remesh gained max_deviation, since an absent "
    "surface-distance gate used to be a listed source of the gap: passing MeshLab's own "
    "checksurfdist default (1% of the bbox diagonal, 0.0288 here) changed nothing at all -- "
    "byte-identical face counts on both fixtures -- because on these patches the remesh never "
    "moves a vertex that far, so the bound does not bind. The parameter is therefore deliberately "
    "NOT passed in this row: it would add a closest-point pass per iteration to the timing while "
    "provably changing no output.",
)
@pytest.mark.benchmark(group="isotropic_remesh")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("triwarp", "pymeshlab", "meshlib")
def test_isotropic_remesh(bench_case: BenchCase) -> None:
    """
    Five stages x ``iterations``, on a nearly-converged mesh and a badly conditioned one.

    MeshLab runs the identical five stages, and the pair reads in opposite directions: **pymeshlab
    343 -> 1 147 ms** across the axis (a 3.3x spread) against **triwarp at 62 / 71 ms**, so the
    5.6x win on ``saddle`` becomes 16x on ``saddle_graded``. MeshLib's own queue is 93 / 71 ms,
    i.e. level with triwarp, and its unchanged column is what certifies the triwarp row against the
    123 / 124 ms this docstring used to carry.

    **This is not a like-for-like win, and the timing alone is misleading.** triwarp is flat because
    its loop is a fixed ``iterations`` x five launches whatever the input looks like; MeshLab's
    serial local-operation queue keeps working until the operations stop paying off. Comparing the
    *outputs* at ``iterations=3`` and the identical target length says what the difference buys:

    | | faces | edge / target | min angle p1 | aspect p99 / max |
    |---|---|---|---|---|
    | ``saddle`` in | 34 848 | 1.00 | 36.9 deg | 1.64 / 1.66 |
    | ``saddle`` triwarp | 35 488 | 0.98 | 38.3 deg | 1.31 / 2.6 |
    | ``saddle`` pymeshlab | 34 946 | 0.99 | 39.6 deg | 1.57 / 2.18 |
    | ``saddle_graded`` in | 34 848 | 1.00 | 0.013 deg | 4 400 / 4 719 |
    | ``saddle_graded`` tw | 40 893 | 0.85 | 0.20 deg | 7 440 / 2.0e6 |
    | ``saddle_graded`` pml | 31 414 | 0.98 | 31.9 deg | 1.87 / 3.55 |

    On ``saddle`` triwarp is now past parity (aspect 99th pct 1.31 against 1.57, worst 2.6 against
    2.18) and within 2 % of the requested length. On the *graded* patch it is not: it misses the
    target by 15 % and leaves a 99th-percentile aspect ratio of 7 440 against MeshLab's 1.87.

    **The two triwarp rows moved together and for one reason, which is worth stating because the
    graded row got worse.** The collapse stage's parallel independent set locked by the *raw* edge
    index, which ``edges_unique`` orders lexicographically and which is therefore spatially
    monotone on any structured mesh -- so the lock had one local minimum and committed **one**
    collapse per pass out of 40 934 candidates (5 vertices removed from 17 689 over five passes,
    against 2 761 hashed, at the same wall clock). ``kernels/remesh.py``'s ``scramble_index`` fixes
    it, which is what took ``saddle`` from 39 100 faces at 0.93 to 35 488 at 0.98. It also unmasked
    the smoother defect below, since closed: with collapse finally committing, the graded patch's
    worst triangles got worse rather than better. That is not the key's doing -- at
    ``tests/test_remesh.py``'s target of half the mean edge the raw key leaves **192**
    float32-degenerate faces on this mesh against the hashed key's **40** -- it is what a collapse
    stage does on anisotropic input when the smoother downstream of it cannot see the anisotropy.

    Two bugs were found and one fixed by reading this pair against the reference; both were
    pre-existing and invisible to ``tests/test_remesh.py``, which only ever runs the remesher on
    clean closed icospheres:

    * **Fixed** -- the swap stage flipped on the valence objective with only a convexity guard, so
      on a graded mesh it turned slivers into worse slivers and in float32 hit exactly-zero area:
      **2 738 of 84 406 faces**, with the worst aspect ratio reaching 5.7e6. It now also rejects any
      flip that would create a degenerate triangle or worsen the pair's aspect ratio. That is what
      moved the ``saddle`` row to parity and zeroed the degenerate counts above.
    * **Closed** -- ``_smooth_pass`` is now the area-equalizing relaxation this module's Notes
      promise, weighting each one-ring neighbour by its barycentric area. It had to be: a regular
      graded grid is valence-perfect (100% of interior vertices have valence exactly 6), already
      Delaunay, *and* a fixed point of the unweighted Laplacian, so with the plain centroid three
      of the five stages were blind to its anisotropy by construction and only split/collapse
      acted. The ``cave_cube`` self-intersection that had blocked this for two rounds was never the
      smoother's: it was the missing collapse fold veto above, and with that in place the weighting
      lands with no veto of its own needed (one is kept anyway, because it measures free). The
      graded row's 99th-percentile aspect ratio against pymeshlab went 352 -> 9.13 on the collapse
      veto and 9.13 -> 8.60 on this, against pymeshlab's 1.87; what is left is the stopping rule,
      which is this group's standing D2 exemption. The whole call is also 1.35x faster, since a
      better-shaped mesh gives split and collapse less to do.
    """
    target = bench_case.mean_edge
    if bench_case.kind == "meshlib":
        # A third stopping rule: MeshLib runs its own local-operation queue to convergence, so like
        # MeshLab's row it is not a fixed-iteration count and the comparison is what the outputs
        # look like rather than what one round costs. ``projectOnOriginalMesh`` is left off, which
        # is its default and matches triwarp: turning it on adds a closest-point pass per vertex.
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np

        def remesh_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(vertices_np, faces_np)
            settings_ml = mm.RemeshSettings()
            settings_ml.targetEdgeLen = target
            mm.remesh(mesh_ml, settings_ml)
            return mesh_ml.topology.numValidFaces()

        assert bench_case.run(remesh_ml, rounds=_ROUNDS) > 0
        return
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
@pytest.mark.benchlibs("triwarp", "open3d", "pymeshlab", "meshlib")
@pytest.mark.parametrize("cell_factor", _CLUSTER_FACTORS)
def test_cluster_decimate(bench_case: BenchCase, cell_factor: float) -> None:
    """
    Bin, remap, dedup: decimation with no priority queue, against two exact-ish references.

    Its axis spans the wrapper floor and the win: ``bunny_decimated`` sits at the ~340 µs floor
    (see ``test_creation::test_box``) and loses 1.5x, while ``dragon`` wins **76x**. A single point
    from this group is uninterpretable in either direction.
    """
    voxel_size = cell_factor * bench_case.mean_edge
    if bench_case.kind == "meshlib":
        # ``verticesGridSampling`` stops one step earlier than the other three rows: it returns the
        # *bitset* of one surviving vertex per occupied cell and never rebuilds the faces, so read
        # it as a lower bound on the group. It leaves the mesh alone, so one mesh serves the rounds.
        mesh_part_ml = mm.MeshPart(_mesh_ml(bench_case))
        sampled_ml = bench_case.run(lambda: mm.verticesGridSampling(mesh_part_ml, voxel_size))
        assert 0 < sampled_ml.count() <= bench_case.n_vertices
        return
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
# device time on ``saddle_graded``) across 69-93 geometry rebuilds of ~40 wrapper calls each, and a
# rebuild costs the same 4.2 ms whether it runs on 32 524 faces or 3 484 — so the count of wrapper
# calls per rebuild is the lever, not the kernels and not the mesh. Three things were priced
# against that. Hoisting the twelve per-pass allocations is worth **3.6 % at most** and was not
# done: a memset that initializes a kernel input cannot be removed by moving the allocation.
# Committing several independent sets per rebuild cuts the rebuild count by roughly the round
# count: **1.3-1.8x**, measured back to back (``saddle_graded`` at 0.1: 461 -> 253 ms;
# ``icosphere(4)`` to 2 048 faces: 158 -> 94). And grouping the pass's edges **once** — the
# candidate scoring and the feature/boundary classification had been grouping the same
# ``3 * n_faces`` rows three times between them — is another **1.31-1.36x** (``saddle`` at 0.1:
# 148 -> 112 ms; ``saddle_graded``: 193 -> 142), which is ``_classify`` alone going 1 059 -> 100 us.
#
# What finally moved it was **capturing the pass**, which the note here used to call unavailable:
# the pass body's host readbacks were replaced by the scans they were reading, every buffer was
# fixed at a bound the mesh cannot exceed, and one graph is then replayed for every pass. A
# further **2.6-4.5x** (``saddle`` at 0.1: 115 -> 37 ms), and it makes triwarp the fastest of the
# five on all four cells where it was 3.1x behind pyvista. See ``remesh._DecimationBuffers``.
_QUADRIC_RATIOS = [0.5, 0.1]


@pytest.mark.benchmark(group="quadric_decimate")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("triwarp", "igl", "open3d", "pymeshlab", "pyvista", "meshlib")
@pytest.mark.parametrize("target_ratio", _QUADRIC_RATIOS)
def test_quadric_decimate(bench_case: BenchCase, target_ratio: float) -> None:
    """
    Greedy quadric collapses to a face budget: batched independent sets against four queues.

    ``PolyData.decimate`` is ``vtkDecimatePro``, and it is the **best** of the four references on
    output quality rather than merely another queue -- 1.14-1.53x less surface deviation than
    triwarp at the same face count on ``icosphere(4)`` (``tests/test_remesh.py`` carries the
    table). It takes a *reduction fraction* where the other four take a face count, so the ratio is
    converted rather than the count passed.
    """
    target_faces = max(4, int(target_ratio * bench_case.n_faces))
    if bench_case.kind == "meshlib":
        # ``decimateMesh`` takes a *deleted*-face budget, not a target, so the count is converted;
        # it lands on the requested face count exactly (``tests/test_remesh.py``). ``packMesh=True``
        # is inside the timed region on purpose -- without it the result cannot be read at all, so
        # it is part of what the operation costs here. It mutates, hence the rebuild per round.
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        n_faces = bench_case.n_faces

        def decimate_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(vertices_np, faces_np)
            settings_ml = mm.DecimateSettings()
            settings_ml.maxDeletedFaces = n_faces - target_faces
            settings_ml.packMesh = True
            mm.decimateMesh(mesh_ml, settings_ml)
            return mesh_ml.topology.numValidFaces()

        assert bench_case.run(decimate_ml, rounds=_ROUNDS) > 0
        return
    if bench_case.kind == "pyvista":
        mesh_pv = bench_case.mesh_pv
        decimated_pv = bench_case.run(lambda: mesh_pv.decimate(1.0 - target_ratio), rounds=_ROUNDS)
        assert decimated_pv.n_faces > 0
        return
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
