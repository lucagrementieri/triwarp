"""
Benchmarks for ``triwarp.holes``.

Axis: **loops_dp** -- ``rim_short`` (two loops of 512) against ``holes_many`` (512 loops of 3).
The module has the highest non-``N`` sensitivity in the package and this pair separates its two
independent drivers:

* **Loop length**, cubed. ``fill_min_weight`` runs a minimum-weight triangulation DP over a
  ``B x B`` table per loop, filled by ``B - 2`` *sequential* kernel launches and read back to the
  host for the traceback. Total work is ``sum(B_i^3)`` and no batching can remove it: this is the
  real cost the module exists to pay. Two loops of 512 measure **20-35 ms**.

  It measured **157 ms** until the per-span launch went from one thread per interval to one *block*
  per interval, its lanes striding the apex loop: the grid was ``n_loops * (max_B - span)`` wide, so
  two long rims put at most ~1 024 threads on a 170-SM part and the whole ``B^3`` term ran at 0.3 %
  of the machine. Same launch count, same DP, byte-identical triangulation (see
  ``fill_dp_span_tiled``). Measured by running this module twice in one session with only the engine
  switched: ``fill_min_weight[rim_short]`` **168.6 -> 35.0 ms (4.8x)**, its two
  ``_chords`` rows **7.6x** and **5.5x**, and ``[holes_many]`` **20.9 -> 7.0 (3.0x)**. An isolated
  in-process timer on the same fixtures reads 173.8 -> 19.7 and 18.4 -> 4.4, so read the ratio, not
  the absolute -- this group's median moves by up to 80 % between runs of the *same* code depending
  on which reference rows share the process.
* **Loop count**, which should cost nothing and used to cost everything. Every loop paid a
  ``.numpy()`` readback, its own forbidden-chord pass over the whole mesh and its own span
  launches, so 512 three-vertex holes -- where the DP itself is one triangle per hole -- ran to
  **376 ms**, *more* than the genuinely expensive ``rim_short``. Batching the DP across loops took
  that to **4.2 ms** (90x) and, because it also runs ``rim_short``'s two rims concurrently, took
  that point from 273 to 157 ms as a side effect.

That is why the axis meshes are *small*: ``rim_short`` is 1 024 faces and ``holes_many`` is 81 408.
Face count is not the variable, and sizing these meshes up would only add DP table entries that the
``B^3`` term already dominates. The two points now differ by ~5x in the right direction, which is
what the axis is for -- the inversion is what said the per-loop sequence was the bug. (The gap was
37x before the per-span launch was widened to a block per interval; that change is worth ~4.8x on
the long rims and ~3.0x on the short ones, so it narrows the axis without inverting it.)
``fill_fan`` runs on the wider **loops** axis instead, because it has no DP and so can afford
``rim_long``'s 65 536-vertex rims -- it is the floor this module's cost is measured against.

One cost the ``loops_dp`` pair cannot see is the default subdivision-target derivation inside
``fill_smooth``: its axis is the raw loop *count*, and 512 loops is too few for it to register
against the DP. ``fill_smooth_target_edge`` runs that one path on the **loops_dense** axis
(512 -> 8 192 loops on the same vertex buffer) instead.

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

import numpy as np
import pytest
import trimesh as tm
import warp as wp
from conftest import BenchCase, mesh_ml_from_numpy
from meshlib import mrmeshpy as mm

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


@pytest.mark.benchmark(group="fill_fan")
@pytest.mark.benchaxis("loops")
@pytest.mark.benchlibs("triwarp")
def test_fill_fan(bench_case: BenchCase) -> None:
    """The cheapest filler -- one fan per loop, no DP -- so it can take the long-rim axis."""
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    result = bench_case.run(lambda: tw.holes.fill_fan(vertices, faces))
    assert result.shape[0] >= faces.shape[0]


@pytest.mark.benchmark(group="fill_cone")
@pytest.mark.benchaxis("loops")
@pytest.mark.benchlibs("triwarp", "meshlib")
def test_fill_cone(bench_case: BenchCase) -> None:
    """
    One centroid vertex and one triangle per boundary edge -- ``fill_fan`` plus an apex.

    Shares ``fill_fan``'s wide ``loops`` axis for the same reason: no DP, so it can afford
    ``rim_long``'s 65 536-vertex rims. meshlib's ``fillHoleTrivially`` is the same operation
    (pinned in tests/test_holes.py::test_fill_cone_matches_meshlib) but fills **one hole per
    call**, so its row includes the Python loop over ``findHoleRepresentiveEdges`` -- which is the
    honest cost of asking MeshLib for the same answer, and is why the ``holes_many`` point is the
    one to read for per-loop overhead rather than the long-rim point. Rebuilt per round because it
    mutates.
    """
    if bench_case.kind == "meshlib":

        def run_ml() -> int:
            mesh_ml = bench_case.new_mesh_ml()
            for edge_ml in mesh_ml.topology.findHoleRepresentiveEdges():
                mm.fillHoleTrivially(mesh_ml, edge_ml)
            return mesh_ml.topology.numValidFaces()

        assert bench_case.run(run_ml, rounds=_ROUNDS) >= bench_case.n_faces
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    new_vertices, new_faces = bench_case.run(
        lambda: tw.holes.fill_cone(vertices, faces), rounds=_ROUNDS
    )
    assert new_faces.shape[0] >= faces.shape[0]
    assert new_vertices.shape[0] >= vertices.shape[0]


@pytest.mark.noparity(
    "trimesh",
    reason="D2 a weaker algorithm for the same task: tm.repair.fill_holes fans triangles across "
    "small holes and gives up on large ones, where fill_min_weight runs the minimum-weight "
    "interval DP, so the two produce different triangulations by design and trimesh has no "
    "minimum-weight answer to compare against. meshlib is the oracle for the DP itself, in "
    "tests/test_holes.py::test_fill_min_weight_matches_meshlib.",
)
@pytest.mark.benchmark(group="fill_min_weight")
@pytest.mark.benchaxis("loops_dp")
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d", "pymeshlab", "meshlib")
def test_fill_min_weight(bench_case: BenchCase) -> None:
    """The ``B^3`` DP: few long loops against many short ones, at a comparable total boundary."""
    if bench_case.kind == "meshlib":
        # The only row here running the *same* algorithm -- a minimum-weight Liepa/Klincsek DP over
        # the same ``plane_normalized`` metric, which is why it is the oracle in
        # tests/test_holes.py::test_fill_min_weight_matches_meshlib and the other three references
        # are timing context. Two settings are load-bearing: ``maxPolygonSubdivisions`` is raised
        # so MeshLib runs the exhaustive search rather than sub-sampling a long rim (the same
        # 1 000 the parity test passes), and the mesh is rebuilt inside the timed callable because
        # ``fillHoles`` mutates it and rounds 2..n would find nothing left to fill.
        def run_ml() -> int:
            mesh_ml = bench_case.new_mesh_ml()
            edges_ml = mesh_ml.topology.findHoleRepresentiveEdges()
            params_ml = mm.FillHoleParams()
            params_ml.maxPolygonSubdivisions = 1000
            params_ml.metric = mm.getPlaneNormalizedFillMetric(mesh_ml, edges_ml[0])
            mm.fillHoles(mesh_ml, edges_ml, params_ml)
            return mesh_ml.topology.numValidFaces()

        assert bench_case.run(run_ml, rounds=_ROUNDS) > bench_case.n_faces
        return
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
        result = bench_case.run(lambda: tw.holes.fill_min_weight(vertices, faces), rounds=_ROUNDS)
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


@pytest.mark.benchmark(group="fill_min_weight_chords")
@pytest.mark.benchaxis("loops_dp")
@pytest.mark.benchlibs("triwarp")
@pytest.mark.parametrize("resolve_multiple_edges", [False, True], ids=["plain", "chords"])
def test_fill_min_weight_chords(bench_case: BenchCase, resolve_multiple_edges: bool) -> None:
    """
    What the forbidden-chord pass costs on top of the DP.

    ``resolve_multiple_edges=True`` builds a ``B x B`` table of chords that already exist as mesh
    edges and bans them from the triangulation, which is a second quadratic pass per loop. Whether
    that is a rounding error next to the cubic DP or a real fraction of it is only visible against
    the ``plain`` row.
    """
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    result = bench_case.run(
        lambda: tw.holes.fill_min_weight(
            vertices, faces, resolve_multiple_edges=resolve_multiple_edges
        ),
        rounds=_ROUNDS,
    )
    assert result.shape[0] >= faces.shape[0]


@pytest.mark.benchmark(group="fill_smooth")
@pytest.mark.benchaxis("loops_dp")
@pytest.mark.benchlibs("triwarp", "meshlib")
@pytest.mark.parametrize("triangulate_only", [True, False], ids=["dp_only", "refined"])
def test_fill_smooth(bench_case: BenchCase, triangulate_only: bool) -> None:
    """
    What refinement and smoothing cost on top of the DP that produced the patch.

    The ``dp_only`` row is ``fill_min_weight`` plus the patch mask, so the gap to ``refined`` is the
    whole refine-and-smooth stage. Parametrizing rather than timing only the full call is what makes
    that stage attributable: on the ``loops_dp`` axis the cubic DP dominates ``rim_short``, so a
    single ``refined`` number cannot say whether a change moved the DP or the smoothing.

    ``max_edge`` is deliberately left at ``None`` so the default target-edge derivation is inside
    the timed callable -- it is the one part of this path whose cost scales with the *mesh* rather
    than with the rims, and leaving it out would hide a regression there.
    """
    if bench_case.kind == "meshlib":
        if triangulate_only:
            pytest.skip("fillHoleNicely has no triangulate-only mode; the refined row is the pair")

        # ``fillHoleNicely`` is the same three stages triwarp's ``fill_smooth`` runs -- DP fill,
        # subdivide the patch, smooth it -- and it is the oracle in
        # tests/test_holes.py::test_fill_smooth_statistics_vs_meshlib, where the comparison is the
        # enclosed volume because the two patches share no vertices. ``maxEdgeLen`` is left at its
        # own default rather than fed from triwarp's derived target, so the two rows refine to
        # different densities: read this as the cost of the *stage*, not as a like-for-like
        # subdivision. Rebuilt per round because it mutates.
        def run_ml() -> int:
            mesh_ml = bench_case.new_mesh_ml()
            settings_ml = mm.FillHoleNicelySettings()
            for edge_ml in mesh_ml.topology.findHoleRepresentiveEdges():
                mm.fillHoleNicely(mesh_ml, edge_ml, settings_ml)
            return mesh_ml.topology.numValidFaces()

        assert bench_case.run(run_ml, rounds=_ROUNDS) > 0
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    result = bench_case.run(
        lambda: tw.holes.fill_smooth(vertices, faces, triangulate_only=triangulate_only),
        rounds=_ROUNDS,
    )
    assert result[1].shape[0] >= faces.shape[0]


def _face_mask_ml(mask_np: np.ndarray) -> mm.FaceBitSet:
    """
    Load a dense face mask into a ``FaceBitSet`` through the packed blocks, not a per-face loop.

    The benchmark-side twin of ``tests.conversions.numpy_to_meshlib_bitset``: ``BitSet.fromBlocks``
    takes ``uint64`` blocks, ``bitorder="little"`` is not NumPy's default and is not optional, and
    ``fromBlocks`` rounds up to whole blocks so the size is trimmed back after.
    """
    packed_np = np.packbits(mask_np, bitorder="little")
    packed_np = np.pad(packed_np, (0, (-packed_np.size) % 8)).view(np.uint64)
    bitset_ml = mm.BitSet.fromBlocks(mm.std_vector_unsigned_long(packed_np.tolist()))
    bitset_ml.resize(int(mask_np.size))
    return mm.FaceBitSet(bitset_ml)


_region_cache: dict[tuple[str, str], tuple] = {}


def _cap_region(bench_case: BenchCase) -> tuple:
    """
    A contiguous face region -- the cap above the mesh's 80th height percentile.

    Contiguity is what makes this a region edit rather than a hole-filling benchmark: a scattered
    mask opens one rim per face, and the DP is cubic in the rim length, so the two shapes are not
    the same measurement at all.
    """
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _region_cache:
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        height_np = vertices_np[faces_np].mean(axis=1)[:, 2]
        mask_np = height_np > np.quantile(height_np, 0.8)
        _region_cache[key] = (wp.array(mask_np, dtype=wp.bool, device=bench_case.device), mask_np)
    return _region_cache[key]


@pytest.mark.benchmark(group="refill_region")
@pytest.mark.benchlibs("triwarp", "meshlib")
@pytest.mark.parametrize("triangulate_only", [True, False], ids=["dp_only", "refined"])
def test_refill_region(bench_case: BenchCase, triangulate_only: bool) -> None:
    """
    Delete a contiguous region and rebuild it: the deletion, the DP, and optionally the refinement.

    Read against ``fill_smooth`` above, whose two rows are the same two stages over the mesh's
    *existing* holes: the difference is the region extraction and the rim classification, which is
    ``delete_region_keep_boundary``'s own group in ``test_selection.py``. Parametrized the same way
    for the same reason -- a single refined number cannot say whether a change moved the DP or the
    smoothing.

    meshlib's ``patchMesh`` is exactly this call and ``tests/test_holes.py`` pins the two to the same
    vertex count, face count and volume in ``triangulateOnly`` mode. It mutates, so it gets a fresh
    mesh per round, and its region bitset is built in ``setup`` -- it is the input.

    First measurement, medians on an RTX 5090, ``bunny`` with a fifth of its faces deleted:
    triwarp-cuda **22.0 ms** against meshlib's **18.6** for ``dp_only`` (1.19x behind) and **58.4**
    against **30.9** refined (1.9x). Both sides are dominated by the rim DP and the refinement here,
    which is why this row is close where ``delete_region_keep_boundary``'s is 8x -- that group
    isolates the extraction, and the extraction is the part triwarp does slowly.
    """
    if bench_case.kind == "meshlib":
        _mask_wp, mask_np = _cap_region(bench_case)
        settings_ml = mm.FillHoleNicelySettings()
        settings_ml.triangulateOnly = triangulate_only

        def run_ml() -> int:
            mesh_ml = bench_case.new_mesh_ml()
            region_ml = _face_mask_ml(mask_np)
            mm.patchMesh(mesh_ml, region_ml, settings_ml)
            return mesh_ml.topology.numValidFaces()

        assert bench_case.run(run_ml, rounds=_ROUNDS) > 0
        return

    mask_wp, _mask_np = _cap_region(bench_case)
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    _out_vertices, out_faces = bench_case.run(
        lambda: tw.holes.refill_region(vertices, faces, mask_wp, triangulate_only=triangulate_only),
        rounds=_ROUNDS,
    )
    assert int(out_faces.shape[0]) > 0


@pytest.mark.benchmark(group="fill_smooth_target_edge")
@pytest.mark.benchaxis("loops_dense")
@pytest.mark.benchlibs("triwarp")
@pytest.mark.parametrize("derive_target", [True, False], ids=["derived", "explicit"])
def test_fill_smooth_target_edge(bench_case: BenchCase, derive_target: bool) -> None:
    """
    What deriving the default subdivision target costs as the loop count grows.

    ``fill_smooth(max_edge=None)`` measures every rim to pick its target edge, and that
    measurement's axis is the loop *count* -- not the rim length, and not the mesh size -- which
    the ``loops_dp`` meshes cap at 512, too few to register against the DP. The ``loops_dense``
    axis holds the vertex buffer fixed and multiplies the loop count by 16, and the gap between
    the ``derived`` and ``explicit`` rows is exactly the derivation. Interleaved A/B at 8 192
    loops on CUDA: within noise (100.3 against 101.7 ms min), against ~0.4 s of per-loop
    ``.numpy()`` readbacks in the form it replaced -- the regression this group exists to catch,
    and one that lands in the *min* (deterministic host cost), where this box's occasional 3x
    clock-state excursions do not. ``explicit`` passes the mesh's mean edge length, which on a
    punched sphere is the value the derivation returns anyway, so the refine stage does identical
    work in both rows.
    """
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    max_edge = None if derive_target else bench_case.mean_edge
    result = bench_case.run(
        lambda: tw.holes.fill_smooth(vertices, faces, max_edge=max_edge), rounds=_ROUNDS
    )
    assert result[1].shape[0] >= faces.shape[0]


# --- Stitching two rims: the same DP over a band rather than a cap -----------------------------

_stitch_cache: dict[tuple[str, str], tuple] = {}
_stitch_np_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _face_slice(bench_case: BenchCase, lo: int, hi: int) -> wp.array[wp.int32]:
    """Contiguous face-index range as a device buffer (untimed setup, so numpy is fine)."""
    return wp.array(np.arange(lo, hi, dtype=np.int32), dtype=wp.int32, device=bench_case.device)


def _submesh(bench_case: BenchCase, lo: int, hi: int) -> tuple:
    """Faces ``[lo, hi)`` of the case mesh as a compact standalone ``(vertices, faces)`` pair."""
    return tw.selection.submesh_from_face_indices(
        bench_case.vertices_wp,
        bench_case.faces_wp,
        _face_slice(bench_case, lo, hi),
        unique_indices=True,
    )


def _stitch_halves(bench_case: BenchCase) -> tuple:
    """
    Cut the open tube into two rings, so each half has exactly one boundary loop to stitch.

    ``_open_cylinder`` emits its lower band of triangles before its upper one, so the first and
    second halves of the face buffer are exactly the two rings -- no adjacency query needed.
    """
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _stitch_cache:
        half = bench_case.n_faces // 2
        _stitch_cache[key] = (
            _submesh(bench_case, 0, half),
            _submesh(bench_case, half, bench_case.n_faces),
        )
    return _stitch_cache[key]


@pytest.mark.benchmark(group="stitch")
@pytest.mark.benchmeshes("rim_short")
@pytest.mark.benchlibs("triwarp")
def test_stitch(bench_case: BenchCase) -> None:
    """Greedy band between two rims: O(La + Lb), the cheap counterpart of the DP below."""
    (va, fa), (vb, fb) = _stitch_halves(bench_case)
    _vertices, faces = bench_case.run(lambda: tw.holes.stitch(va, fa, vb, fb))
    assert int(faces.shape[0]) > 0


def _stitch_pair_np(bench_case: BenchCase) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the same two rings as [`_stitch_halves`] as one NumPy mesh with two disjoint rims.

    MeshLib's ``stitchHoles`` takes a *single* ``Mesh`` holding both holes, so the two halves are
    compacted independently and index-offset into one buffer -- the same construction
    ``tests/test_holes.py::_meshlib_stitch_band`` uses, which is what makes the benchmark and the
    parity test measure the same thing.
    """
    if bench_case.mesh_name not in _stitch_np_cache:
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        half = bench_case.n_faces // 2
        parts = []
        for lo, hi in ((0, half), (half, bench_case.n_faces)):
            used, inverse = np.unique(faces_np[lo:hi], return_inverse=True)
            parts.append((vertices_np[used], inverse.reshape(-1, 3)))
        (va_np, fa_np), (vb_np, fb_np) = parts
        _stitch_np_cache[bench_case.mesh_name] = (
            np.vstack([va_np, vb_np]),
            np.vstack([fa_np, fb_np + len(va_np)]),
        )
    return _stitch_np_cache[bench_case.mesh_name]


@pytest.mark.benchmark(group="stitch_min_weight")
@pytest.mark.benchmeshes("rim_short")
@pytest.mark.benchlibs("triwarp", "meshlib")
def test_stitch_min_weight(bench_case: BenchCase) -> None:
    """
    Grid DP over the two rims: an La x Lb table filled by La + Lb sequential launches.

    meshlib is the only reference in the package that has this operation at all -- ``stitchHoles``
    is a real two-loop minimum-weight stitch where trimesh and pymeshlab have nothing, which is why
    it is also the oracle in tests/test_holes.py::test_stitch_min_weight_matches_meshlib. The
    four-argument overload is used deliberately (see section 6): the two-argument one finds the
    rims itself, and timing that would fold hole detection into the DP.
    """
    if bench_case.kind == "meshlib":
        vertices_np, faces_np = _stitch_pair_np(bench_case)

        def run_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(vertices_np, faces_np)
            edges_ml = mesh_ml.topology.findHoleRepresentiveEdges()
            params_ml = mm.StitchHolesParams()
            params_ml.metric = mm.getComplexStitchMetric(mesh_ml)
            mm.stitchHoles(mesh_ml, edges_ml[0], edges_ml[1], params_ml)
            return mesh_ml.topology.numValidFaces()

        assert bench_case.run(run_ml, rounds=_ROUNDS) > len(faces_np)
        return
    (va, fa), (vb, fb) = _stitch_halves(bench_case)
    up = wp.vec3(0.0, 0.0, 1.0)
    _vertices, faces = bench_case.run(
        lambda: tw.holes.stitch_min_weight(va, fa, vb, fb, up_dir=up), rounds=_ROUNDS
    )
    assert int(faces.shape[0]) > 0
