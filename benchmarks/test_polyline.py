"""
Benchmarks for ``triwarp.polyline``.

The module has 27 public functions over one data shape — an ordered ``(n,)`` array of ``wp.vec3``
— and they fall into four cost classes. One representative of each is timed rather than all 27,
because within a class the kernels differ only in the per-segment expression:

* **Whole-polyline reductions** (``polyline_length``, ``polyline_centroid``, ``polyline_normal``,
  ``polyline_radius``) — one pass over the segments into a scalar. Launch-latency bound at these
  sizes; ``polyline_radius`` is the most expensive of them because it projects every segment onto
  a plane and finds each segment's closest point before reducing.
* **Per-vertex maps** (``polyline_angles``, ``cumulative_arc_length``) — one value per vertex,
  purely local. ``polyline_angles`` is the ``wp.acos`` path.
* **Resampling** (``polyline_upsample``, ``polyline_resample``, ``polyline_downsample``) — an
  output whose length is data-dependent, so these pay a scan plus a host readback of the output
  size before the write pass. The interpolation itself is the ``wp.lerp`` inner loop.
* **Simplification** (``polyline_simplify``) — Ramer-Douglas-Peucker, evaluated
  **level-synchronously**: one round of four ``dim=n`` launches per level of the split tree, driven
  by ``wp.capture_while`` so no round costs a readback. The cost is therefore the tree's *depth*,
  about ``log2(n)`` on a mesh boundary loop. This group is why: it used to be the module's
  deliberate serial outlier — a single-thread stack-based kernel, since Warp forbids recursion —
  and *that* framing is what kept it there. Warp cannot express the recursion, but it can express
  the recursion's **levels**, and the two accept the same points, because breadth-first and
  depth-first evaluation of one split tree differ only in order. Measured 80.47 -> 1.11 ms on
  ``rim_long``, **72x**, with the accepted set identical; ``polyline_simplify``'s Notes carry the
  full table, the two rows that lose, and why the depth is bounded by the accepted count.

``polyline_point_distance`` is timed separately from the rest because it is the only function whose
cost is the product of two sizes (query points x segments) rather than a function of the polyline
alone.

``triangulate_polygon`` is the remaining exception, and the module's worked example of a cost
that is not where it looks. It delegates to ``polyline.polyline_triangulate``, a **parallel**
multi-round ear clipper — every launch in the round loop is ``dim=n``, and a round clips a whole
*independent set* of ears at once. What is serial is the **round count**: a round costs four
``dim=n`` launches whatever it clips, so the only thing that matters is how many ears survive per
round. Since the round loop moved onto the device (``wp.capture_while``, so no readback per round)
that costs 14 µs a round at 64 points, and the group measures **2.2 / 4.5 ms** against 4.5 / 6.9
before — 2.07x and 1.53x.

Read the residual against its attribution rather than against the round loop, because the round loop
is no longer the cost: at 64 points the clip itself is 0.22 ms of the 2.2, and **1.05 ms is the
prologue** — ``polyline_open``'s ``is_closed`` (0.24), ``polyline_normal`` (0.35) and
``polyline_centroid`` (0.25), three reductions that each end in a host readback because their result
is a Python-scope ``wp.vec3``, plus the reflex-count readback. That share is *flat in n*, so it is
the whole gap to trimesh's 0.12 ms at 64 points and none of it at 1 024.

The round *count* is what this group caught first. ``select_independent`` used to rank competing
ear candidates by their raw ring index, which on an alternating star lets ear ``i - 2`` suppress
ear ``i`` for every ``i``, so exactly **one** ear was clipped per round and the loop ran to its
``n`` cap: 6.1 ms at 64 points and **141 ms** at 1 024, growing as ``n^1.12`` (rounds proportional
to ``n`` times a slowly growing per-round cost) rather than the ``O(L^2)`` a serial clipper would
give. Ranking by a bijective hash of the ring index makes it the textbook maximal-independent-set
rule, which retires a constant fraction per round: 16 and 30 rounds, the latter instead of 1 022.

Axis: **polyline** -- longest boundary loop of 268, 528 and 65 536 vertices. Polylines come from
**mesh boundary loops**, not from mesh geometry, and the axis is a loop-length sweep rather than a
face-count one: nothing here reads a face. The scan meshes are excluded on the same grounds --
they are near-closed surfaces whose holes are a handful of vertices each, so they would measure
launch latency and nothing else.

The longest loop of each mesh is gathered into a dense ``wp.vec3`` buffer once per (mesh, device)
and reused across rounds, so the timed region contains only the polyline function itself.

Two groups carry a second sweep, on the parameter that drives them rather than on length:
``polyline_simplify`` on its tolerance (which sets the depth of its split tree, and so its round
count) and
``polyline_point_distance`` on the query count (the other half of its two-size product).

References
----------
**meshlib and pyvista are the module's baselines**, and between them they cover every group here
except the two per-vertex maps. MeshLib's ``Polyline3`` is a complete polyline library --
``totalLength``, ``averageEdgeLength``, ``loopDirArea``, ``splitEdge``,
``findProjectionOnPolyline``, ``subdividePolyline``, ``decimatePolyline``, ``pack`` -- and VTK
reaches the same operations through a single-cell ``PolyData``. An earlier version of this section
said no CPU baseline was registered at all, which was already false of three groups when it was
written.

Two hazards decide every row here, both measured:

* **The single line cell.** ``pv.lines_from_points`` gives one two-point cell *per segment*, and
  every polyline filter then restarts at each of them -- ``compute_arc_length`` reports 0.0638 for
  a 200-point helix whose length is 12.7049, and ``decimate_polyline`` is a **no-op at every
  reduction**. ``_polyline_pv`` builds one cell for that reason; a row built the other way would
  time the right filter on the wrong input and read as a suspiciously fast reference.
* **``pack()`` is mandatory after a decimation, and skipping it is silent.** Measured on a 128-point
  helix at ``maxError=0.1``: ``vertsDeleted`` is **104**, ``points.size()`` is still **128**, and
  ``topology.numValidVerts()`` is **24**. ``totalLength()`` is already correct before packing, so a
  *length* comparison passes unpacked while a *point-count* one silently reads the input's count and
  reads as a no-op. This is CLAUDE.md section 7.6's ``getNumpyFaces``-without-``pack()`` rule, in a
  class that rule does not name.

The three groups that stay triwarp-only, and why it is per-function rather than blanket:

* **``polyline_radius``** -- no reference computes it. ``Polyline3.findCenterFromPoints`` is a
  centroid and ``findMaxProjectionOnPolyline`` projects points *onto* a polyline, which is
  ``polyline_point_distance``'s question and already carries its rows.
* **``polyline_angles``** -- three-point turning angles. MeshLib has ``edgeVector`` only, so a
  reference row would time a Python loop over the segments.
* **``polyline_triangulate``**'s sibling ``triangulate_polygon`` carries trimesh; the ear clipper
  itself carries meshlib and pyvista.

trimesh, libigl and open3d remain unregistered here, and each for its own reason: trimesh models
polylines as ``Path3D`` entities rather than arrays and its only simplification is
``merge_colinear`` (a colinear-run merge, a different algorithm with a different output); libigl's
``igl.upsample`` is *mesh* subdivision and its C++ ``ramer_douglas_peucker`` is not bound; and
open3d's ``LineSet`` stores unordered segments with no ordering, length, resampling or
simplification operation at all.
"""

from __future__ import annotations

import numpy as np
import pytest
import pyvista as pv
import shapely.geometry as sg
import trimesh as tm
import warp as wp
from meshlib import mrmeshpy as mm

import triwarp as tw
from conftest import BenchCase, BenchLibrary, skip_larger_than

# MeshLib expresses several gates as *absolute* lengths whose defaults assume a unit-scale input
# (``DecimatePolylineSettings.maxError`` is 1e-3). Passing this instead disables the gate outright,
# so a row stops for the reason it is given rather than for the fixture's units.
_UNBOUNDED = 1e30

# Resampling step, as a fraction of the mean segment length: < 1 upsamples, > 1 downsamples.
_UPSAMPLE_FRACTION = 0.5
_DOWNSAMPLE_FRACTION = 4.0

# Ramer-Douglas-Peucker tolerances, as a fraction of the polyline's bounding-box diagonal. A
# tighter tolerance keeps more points and so recurses deeper, which on a single-thread kernel is
# the whole cost; the pair is two orders of magnitude apart so the slope is unambiguous.
_SIMPLIFY_FRACTIONS = [1e-3, 1e-1]

# Query-point counts for polyline_point_distance: the second size in its points x segments product,
# swept independently of the polyline length the axis provides.
_N_QUERIES = [1 << 12, 1 << 16]

_polyline_cache: dict[tuple[str, str], wp.array[wp.vec3]] = {}
_query_cache: dict[tuple[str, str, int], wp.array[wp.vec3]] = {}


def _polyline_wp(bench_case: BenchCase) -> wp.array[wp.vec3]:
    """Longest boundary loop of the mesh as a dense ``wp.vec3`` polyline, cached per case."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _polyline_cache:
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        loops = tw.boundary.boundary_loops(vertices, faces)
        if not loops:
            pytest.skip(f"{bench_case.mesh_name} has no boundary loop to use as a polyline")
        longest = max(loops, key=lambda loop: int(loop.shape[0]))
        dense = wp.empty(int(longest.shape[0]), dtype=wp.vec3, device=bench_case.device)
        wp.copy(dense, vertices[longest])
        _polyline_cache[key] = dense
    return _polyline_cache[key]


def _segment_scale(bench_case: BenchCase) -> tuple[float, float]:
    """
    ``(mean_segment_length, bbox_diagonal)`` of the polyline, computed on the host once.

    Through ``_polyline_np`` rather than ``_polyline_wp``, for the reason that helper exists: a
    ``cpu_bound`` reference case has no Warp device, and every resampling row now reads this scale
    on both sides of its branch to give both libraries the same target.
    """
    polyline = _polyline_np(bench_case)
    steps = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    diagonal = float(np.linalg.norm(polyline.max(axis=0) - polyline.min(axis=0)))
    return float(steps.mean()), diagonal


def _query_points_wp(bench_case: BenchCase, count: int) -> wp.array[wp.vec3]:
    """Random query points inside the polyline's bounding box, cached per (case, count)."""
    key = (bench_case.mesh_name, str(bench_case.device), count)
    if key not in _query_cache:
        polyline = _polyline_wp(bench_case).numpy()
        rng = np.random.default_rng(20260726)
        points = rng.uniform(polyline.min(axis=0), polyline.max(axis=0), size=(count, 3))
        _query_cache[key] = wp.array(
            np.ascontiguousarray(points, dtype=np.float32), dtype=wp.vec3, device=bench_case.device
        )
    return _query_cache[key]


_polyline_np_cache: dict[str, np.ndarray] = {}
_query_np_cache: dict[tuple[str, int], np.ndarray] = {}


def _polyline_np(bench_case: BenchCase) -> np.ndarray:
    """
    Build the same polyline as a host array, for the CPU-bound rows.

    ``_polyline_wp`` goes through ``bench_case.vertices_wp``, which needs a Warp device a
    ``cpu_bound`` case does not have -- the situation ``test_points.py``'s pymeshlab cloud is in.
    The extraction runs on the ``cpu`` device instead: it is the benchmark's *input*, so which
    device derives it is immaterial, and the result is cached per mesh.
    """
    if bench_case.mesh_name not in _polyline_np_cache:
        vertices = wp.array(
            np.ascontiguousarray(bench_case.vertices_np, dtype=np.float32),
            dtype=wp.vec3,
            device="cpu",
        )
        faces = wp.array(
            np.ascontiguousarray(bench_case.faces_np.reshape(-1), dtype=np.int32),
            dtype=wp.int32,
            device="cpu",
        )
        loops = tw.boundary.boundary_loops(vertices, faces)
        if not loops:
            pytest.skip(f"{bench_case.mesh_name} has no boundary loop to use as a polyline")
        longest = max(loops, key=lambda loop: int(loop.shape[0]))
        _polyline_np_cache[bench_case.mesh_name] = np.ascontiguousarray(
            bench_case.vertices_np[longest.numpy()], dtype=np.float64
        )
    return _polyline_np_cache[bench_case.mesh_name]


def _query_points_np(bench_case: BenchCase, count: int) -> np.ndarray:
    """Draw the same query cloud as [`_query_points_wp`], on the host and at the same seed."""
    key = (bench_case.mesh_name, count)
    if key not in _query_np_cache:
        polyline_np = _polyline_np(bench_case)
        rng = np.random.default_rng(20260726)
        _query_np_cache[key] = rng.uniform(
            polyline_np.min(axis=0), polyline_np.max(axis=0), size=(count, 3)
        )
    return _query_np_cache[key]


_contour_ml_cache: dict[tuple[str, str], mm.std_vector_Vector3_float] = {}
_polyline_ml_cache: dict[tuple[str, str], mm.Polyline3] = {}


def _contour_ml(bench_case: BenchCase) -> mm.std_vector_Vector3_float:
    """
    Build the benchmark's polyline as a MeshLib contour, cached.

    Filling it is a per-point Python loop over ``Vector3f`` constructions -- comparable to the
    reduction it feeds -- and it is the *input*, so it is built once per polyline like every other
    library's copy in this file.
    """
    key = (bench_case.mesh_name, "contour")
    if key not in _contour_ml_cache:
        contour_ml = mm.std_vector_Vector3_float()
        for point_np in _polyline_np(bench_case):
            contour_ml.append(mm.Vector3f(*point_np.tolist()))
        _contour_ml_cache[key] = contour_ml
    return _contour_ml_cache[key]


def _polyline_ml(bench_case: BenchCase) -> mm.Polyline3:
    """
    Build the same points as a ``Polyline3``, through the **constructor**.

    ``addFromPoints`` binds a raw ``Vector3f*`` plus a count rather than a vector, so the
    single-contour constructor is the usable route. Cached: it builds an AABB tree lazily on first
    query, which the row below pre-warms rather than times.
    """
    key = (bench_case.mesh_name, "polyline")
    if key not in _polyline_ml_cache:
        _polyline_ml_cache[key] = mm.Polyline3(_contour_ml(bench_case))
    return _polyline_ml_cache[key]


_polyline_pv_cache: dict[str, pv.PolyData] = {}


def _polyline_pv(bench_case: BenchCase) -> pv.PolyData:
    """
    Build the benchmark's polyline as a ``PolyData`` holding **one** line cell, cached.

    The single cell is not a detail: ``pv.lines_from_points`` gives one two-point cell per segment
    and every polyline filter then restarts at each of them -- ``compute_arc_length`` reports 0.0638
    for a 200-point helix whose length is 12.7049, so a row built that way would time the right
    filter on the wrong input. Cached like the MeshLib ``Polyline3`` beside it, and for the same
    second reason: ``find_closest_cell`` builds a cell locator lazily on first use, which the row
    below pre-warms rather than times.
    """
    if bench_case.mesh_name not in _polyline_pv_cache:
        points_np = _polyline_np(bench_case)
        _polyline_pv_cache[bench_case.mesh_name] = pv.PolyData(
            points_np, lines=np.hstack([[points_np.shape[0]], np.arange(points_np.shape[0])])
        )
    return _polyline_pv_cache[bench_case.mesh_name]


@pytest.mark.benchmark(group="polyline_length")
@pytest.mark.benchaxis("polyline")
@pytest.mark.benchlibs("triwarp", "meshlib", "pyvista")
def test_polyline_length(bench_case: BenchCase) -> None:
    """
    Summed segment length: the cheapest whole-polyline reduction, launch-latency bound.

    meshlib's ``calcLength`` sums the same segments and returns a **bit-identical** float32
    (``tests/test_polyline.py``), so this pair is a pure host-against-device reading of the same
    arithmetic -- and on a reduction this cheap triwarp's row is its launch latency, which is what
    makes the comparison worth having. The contour is the input and is cached.

    pyvista's ``compute_arc_length`` **does more**: it writes the *cumulative* length at every
    point and the total is its last entry, where both other rows return the scalar directly. So read
    its row as a per-point pass rather than as a reduction -- and it is still cheap, 0.21 / 0.21 /
    0.88 ms across the axis, because VTK walks one line cell.
    """
    if bench_case.kind == "pyvista":
        line_pv = _polyline_pv(bench_case)
        arc_pv = bench_case.run(line_pv.compute_arc_length)
        assert float(np.asarray(arc_pv["arc_length"]).max()) > 0.0
        return
    if bench_case.kind == "meshlib":
        contour_ml = _contour_ml(bench_case)
        assert bench_case.run(lambda: mm.calcLength(contour_ml)) > 0.0
        return
    polyline = _polyline_wp(bench_case)
    length = bench_case.run(lambda: tw.polyline.polyline_length(polyline))
    assert length > 0.0


@pytest.mark.benchmark(group="polyline_normal")
@pytest.mark.benchaxis("polyline")
@pytest.mark.benchlibs("triwarp")
def test_polyline_normal(bench_case: BenchCase) -> None:
    """
    Newell's loop normal: a ``wp.cross`` accumulated over the closing segments.

    triwarp-only, and it is an absence rather than a cost objection -- no registered library
    computes a polyline's Newell normal. meshlib's ``Polyline3`` offers a centroid
    (``findCenterFromPoints``) and a projection but no loop normal, and pyvista's line filters read
    an arc length rather than an orientation.

    **A second row in the whole-polyline-reduction class**, which the module docstring above
    otherwise times through one representative. The rule held while the class really did differ
    "only in the per-segment expression", and it stopped holding: ``polyline_length`` maps
    ``segment_length`` and reduces through ``triwarp.reduce``, so it was always a proper block
    reduction, while this one accumulated one ``wp.atomic_add`` per thread into a single
    ``wp.vec3`` slot -- every thread in the launch contending for one address, and the reduction
    serialized. Measured, converting it to the lane-strided form
    (``kernels/polyline.py::accumulate_newell_normal`` carries the table): **1.32x** at 4 096
    vertices, **19.6x** at 65 536 and **71.8x** at 262 144, with the answer three orders of
    magnitude more accurate against a float64 reference. So the representative had the *good*
    shape and the class member it stood in for did not, which is what a one-row class cannot show.
    ``polyline_centroid`` stays unrepresented: it reaches ``triwarp.reduce`` the way
    ``polyline_length`` does.

    ``rim_long`` is the row that matters here -- 65 536 vertices, the axis's asymptotic point.
    """
    polyline = _polyline_wp(bench_case)
    normal = bench_case.run(lambda: tw.polyline.polyline_normal(polyline))
    assert float(np.linalg.norm(np.asarray(list(normal), dtype=np.float64))) > 0.0


@pytest.mark.benchmark(group="polyline_radius")
@pytest.mark.benchaxis("polyline")
@pytest.mark.benchlibs("triwarp")
def test_polyline_radius(bench_case: BenchCase) -> None:
    """
    Per-segment plane projection and closest-point search, then a reduction.

    triwarp-only, and it is an absence rather than a cost objection: no registered library computes
    a polyline's radius about a centre and normal. ``Polyline3.findCenterFromPoints`` is a centroid
    and ``findMaxProjectionOnPolyline`` projects points *onto* a polyline, which is
    ``polyline_point_distance``'s question and already carries its rows. See the module docstring.
    """
    polyline = _polyline_wp(bench_case)
    radius = bench_case.run(lambda: tw.polyline.polyline_radius(polyline, reduction="min"))
    assert radius >= 0.0


@pytest.mark.benchmark(group="polyline_angles")
@pytest.mark.benchaxis("polyline")
@pytest.mark.benchlibs("triwarp")
def test_polyline_angles(bench_case: BenchCase) -> None:
    """
    Per-vertex turning angle: the ``wp.acos`` path, one angle per point.

    triwarp-only. MeshLib has ``edgeVector`` and nothing above it, so a reference row would time a
    Python loop over the segments rather than MeshLib -- the per-element rule. See the module
    docstring.
    """
    polyline = _polyline_wp(bench_case)
    angles = bench_case.run(lambda: tw.polyline.polyline_angles(polyline))
    assert angles.shape[0] == int(polyline.shape[0])


def _polyline_cpu(bench_case: BenchCase) -> wp.array[wp.vec3]:
    """
    Build the benchmark's polyline as a ``wp.vec3`` array on the **cpu** device, cached per mesh.

    The reference rows for ``polyline_downsample`` and ``polyline_simplify`` are driven by the
    *count* triwarp reaches rather than by their own error parameter (each row says why), so they
    have to call triwarp once to learn it. That call is outside the timed callable and is not the
    measurement, so it runs on the cpu -- a ``cpu_bound`` case has no Warp device of its own.
    """
    key = (bench_case.mesh_name, "cpu-polyline")
    if key not in _polyline_cache:
        _polyline_cache[key] = wp.array(
            np.ascontiguousarray(_polyline_np(bench_case)), dtype=wp.vec3, device="cpu"
        )
    return _polyline_cache[key]


@pytest.mark.benchmark(group="polyline_upsample")
@pytest.mark.benchaxis("polyline")
@pytest.mark.benchlibs("triwarp", "meshlib")
def test_upsample_polyline(bench_case: BenchCase) -> None:
    """
    Arc-length upsampling at half the mean segment length: scan, readback, then a lerp pass.

    meshlib's ``subdividePolyline`` takes the same ``maxEdgeLen`` and makes the same guarantee, but
    it **bisects** where triwarp splits each segment into equal pieces, so at a target that is not
    a power-of-two fraction of the input spacing it overshoots -- measured on a 128-point helix at
    half the spacing, 509 points against triwarp's 254, because one halving leaves 0.0502 against a
    cap of 0.05 and forces a second. Both satisfy the cap; read the row as a cost at a *shared
    post-condition* rather than at a shared output size (``tests/test_polyline.py``).

    It mutates, so the ``Polyline3`` is rebuilt inside the timed callable, and ``maxEdgeSplits`` is
    raised from its default of **1 000** -- the same trap ``SubdivideSettings`` has, and on these
    axes it would stop the reference in the first few percent of the work.
    """
    step = _UPSAMPLE_FRACTION * _segment_scale(bench_case)[0]
    n_points = _polyline_np(bench_case).shape[0]

    if bench_case.kind == "meshlib":
        contour_ml = _contour_ml(bench_case)

        def upsample_ml() -> int:
            polyline_ml = mm.Polyline3(contour_ml)
            settings_ml = mm.PolylineSubdivideSettings()
            settings_ml.maxEdgeLen = step
            settings_ml.maxEdgeSplits = 10_000_000
            mm.subdividePolyline(polyline_ml, settings_ml)
            return polyline_ml.topology.numValidVerts()

        assert bench_case.run(upsample_ml) >= n_points
        return

    polyline = _polyline_wp(bench_case)
    dense = bench_case.run(lambda: tw.polyline.polyline_upsample(polyline, step))
    assert dense.shape[0] >= int(polyline.shape[0])


@pytest.mark.benchmark(group="polyline_downsample")
@pytest.mark.benchaxis("polyline")
@pytest.mark.benchlibs("triwarp", "meshlib")
def test_downsample_polyline(bench_case: BenchCase) -> None:
    """
    Arc-length downsampling at four times the mean segment length.

    meshlib's ``decimatePolyline`` reaches a *count* rather than a step, so its row is given the
    count triwarp's step produces (``maxDeletedVertices``) and its ``maxError`` is opened up so the
    count is what binds -- otherwise the two rows would stop for different reasons and the ratio
    would mean nothing. That makes this the module's clearest structural contrast: an arc-length
    resample is one scan and one gather where a decimator is a priority queue of collapses.

    ``optimizeVertexPos`` is turned **off**. It defaults on and moves each surviving vertex to a
    fitted position, which is work triwarp does not do and which would additionally leave the output
    off the input point set.
    """
    step = _DOWNSAMPLE_FRACTION * _segment_scale(bench_case)[0]

    if bench_case.kind == "meshlib":
        contour_ml = _contour_ml(bench_case)
        cpu_polyline = _polyline_cpu(bench_case)
        n_points = int(cpu_polyline.shape[0])
        n_kept = int(tw.polyline.polyline_downsample(cpu_polyline, step).shape[0])

        def downsample_ml() -> int:
            polyline_ml = mm.Polyline3(contour_ml)
            settings_ml = mm.DecimatePolylineSettings_Vector3f()
            settings_ml.maxDeletedVertices = max(n_points - n_kept, 0)
            settings_ml.maxError = _UNBOUNDED
            settings_ml.optimizeVertexPos = False
            return int(mm.decimatePolyline(polyline_ml, settings_ml).vertsDeleted)

        assert bench_case.run(downsample_ml) >= 0
        return

    polyline = _polyline_wp(bench_case)
    sparse = bench_case.run(lambda: tw.polyline.polyline_downsample(polyline, step))
    assert sparse.shape[0] >= 2


@pytest.mark.benchmark(group="polyline_simplify")
@pytest.mark.benchaxis("polyline")
@pytest.mark.benchlibs("triwarp", "meshlib", "pyvista")
@pytest.mark.parametrize("tolerance_fraction", _SIMPLIFY_FRACTIONS)
def test_simplify_polyline(bench_case: BenchCase, tolerance_fraction: float) -> None:
    """
    Ramer-Douglas-Peucker, one round of four ``dim=n`` launches per level of the split tree.

    This group used to carry the sentence *"the one group in the package where triwarp is expected
    to lose"*, and it is worth leaving a marker where that was: the expectation was load-bearing,
    not descriptive. It rested on Warp forbidding recursion, which is true, and on the conclusion
    that the split is therefore serial, which is not -- a level-synchronous evaluation accepts the
    same points and turned the ``rim_long`` rows from 80.47 ms into 1.11. Read the two ``saddle``
    rows as floor rows now (the graph capture is ~0.16 ms of a ~0.6 ms call), not as the outlier.

    Neither reference is Ramer-Douglas-Peucker, and **neither is driven by triwarp's tolerance**,
    which is the thing to know before reading the ratio: both are given the *reduction* triwarp's
    tolerance produces, so the rows price three ways of removing the same number of points.

    Driving them by their own error parameter was tried and rejected on a measurement.
    ``decimatePolyline``'s ``maxError`` is a collapse cost, **not** a deviation bound: on a 40-point
    random walk at a tolerance of 0.8134 its output sits **1.9187** from the input, 2.4x the number
    it was given, where triwarp's and pyvista's sit at 0.3730. So a tolerance-matched pair would be
    two different amounts of work under one parameter name.

    * **meshlib** ``decimatePolyline`` at ``maxDeletedVertices`` = triwarp's deletion count, with
      ``maxError`` opened up so the count is what binds. ``optimizeVertexPos`` is turned off for the
      reason the ``downsample`` row records; it mutates, so the ``Polyline3`` is rebuilt per round.
    * **pyvista** ``decimate_polyline`` takes a reduction *fraction*, handed the same count. It
      needs the **single-cell** ``PolyData`` ``_polyline_pv`` builds: on a ``lines_from_points``
      polyline (one two-point cell per segment) it is a no-op at every reduction, which is the
      silent-wrong-input hazard that builder exists for.
    """
    tol = tolerance_fraction * _segment_scale(bench_case)[1]

    if bench_case.kind == "meshlib":
        contour_ml = _contour_ml(bench_case)
        cpu_polyline = _polyline_cpu(bench_case)
        n_points = int(cpu_polyline.shape[0])
        n_kept = int(tw.polyline.polyline_simplify(cpu_polyline, tol)[0].shape[0])

        def simplify_ml() -> int:
            polyline_ml = mm.Polyline3(contour_ml)
            settings_ml = mm.DecimatePolylineSettings_Vector3f()
            settings_ml.maxDeletedVertices = max(n_points - n_kept, 0)
            settings_ml.maxError = _UNBOUNDED
            settings_ml.optimizeVertexPos = False
            return int(mm.decimatePolyline(polyline_ml, settings_ml).vertsDeleted)

        assert bench_case.run(simplify_ml) >= 0
        return
    if bench_case.kind == "pyvista":
        line_pv = _polyline_pv(bench_case)
        cpu_polyline = _polyline_cpu(bench_case)
        n_points = int(cpu_polyline.shape[0])
        n_kept = int(tw.polyline.polyline_simplify(cpu_polyline, tol)[0].shape[0])
        reduction = min(max(1.0 - n_kept / n_points, 0.0), 0.999)
        assert bench_case.run(lambda: line_pv.decimate_polyline(reduction)).n_points >= 2
        return

    polyline = _polyline_wp(bench_case)
    simplified, kept = bench_case.run(lambda: tw.polyline.polyline_simplify(polyline, tol))
    assert simplified.shape[0] == kept.shape[0]


@pytest.mark.benchmark(group="polyline_point_distance")
@pytest.mark.benchaxis("polyline")
@pytest.mark.benchlibs("triwarp", "meshlib", "pyvista")
@pytest.mark.parametrize("n_queries", _N_QUERIES)
def test_distance_to_polyline(bench_case: BenchCase, n_queries: int) -> None:
    """
    Brute-force point-to-segment distance: the one case whose cost is points x segments.

    That is the contrast meshlib's row is here for: ``findProjectionOnPolyline`` walks an **AABB
    tree** over the segments, so its cost is ``points x log(segments)`` where triwarp's is the full
    product -- read the gap across the ``polyline`` axis rather than at one point. It has no
    batched form, so the row loops in Python and prices that loop along with the queries; the tree
    is built lazily and is pre-warmed outside the timed callable.

    pyvista's ``find_closest_cell`` is the same query through a ``vtkStaticCellLocator`` and,
    unlike MeshLib's, it is **batched** -- so its row is the honest tree-walk comparison and the
    MeshLib one is a Python loop next to it. It reports the closest *point*, not the distance, so
    the row's output is one array wider than what it is timed against; the distance is a host
    subtraction and is left out deliberately.

    **rim_long is skipped for pyvista**, measured: its locator degrades on a 65 536-segment single
    cell to 4 963.9 ms at 4 096 queries and **104 125 ms** at 65 536, against 24.8 / 382.7 ms on the
    268-segment loop. That is the shape of the axis this group exists to show, and one row of it
    would cost more than the rest of the module put together.
    """
    if bench_case.kind == "pyvista":
        if bench_case.mesh_name == "rim_long":
            pytest.skip("VTK's line locator is 104 s at this loop length; capped at saddle")
        line_pv = _polyline_pv(bench_case)
        queries_np = _query_points_np(bench_case, n_queries)
        line_pv.find_closest_cell(queries_np[:1], return_closest_point=True)  # pre-warm
        _cells_pv, closest_pv = bench_case.run(
            lambda: line_pv.find_closest_cell(queries_np, return_closest_point=True)
        )
        assert np.asarray(closest_pv).shape == (n_queries, 3)
        return
    if bench_case.kind == "meshlib":
        skip_larger_than(bench_case, "bunny", "the query is a per-point Python loop")
        polyline_ml = _polyline_ml(bench_case)
        queries_np = _query_points_np(bench_case, n_queries)
        mm.findProjectionOnPolyline(mm.Vector3f(*queries_np[0].tolist()), polyline_ml)  # pre-warm

        def distances_ml() -> float:
            return sum(
                mm.findProjectionOnPolyline(mm.Vector3f(*point_np.tolist()), polyline_ml).distSq
                for point_np in queries_np
            )

        assert bench_case.run(distances_ml) >= 0.0
        return
    polyline = _polyline_wp(bench_case)
    points = _query_points_wp(bench_case, n_queries)
    distance = bench_case.run(lambda: tw.polyline.polyline_point_distance(points, polyline))
    assert distance.shape[0] == n_queries


# Vertex counts for the triangulation group. A simple polygon of ``n`` vertices always yields
# ``n - 2`` triangles, so the count is the only axis and it is swept directly rather than through a
# mesh: neither library's ear clipping is driven by anything else.
_POLYGON_SIZES = [64, 1024]

_polygon_np_cache: dict[int, np.ndarray] = {}


def _polygon_np(n_vertices: int) -> np.ndarray:
    """Build a non-convex star of ``n_vertices``, cached: a fan would trivialize a convex one."""
    if n_vertices not in _polygon_np_cache:
        angles_np = np.linspace(0.0, 2.0 * np.pi, n_vertices, endpoint=False)
        radii_np = np.where(np.arange(n_vertices) % 2 == 0, 1.0, 0.45)
        _polygon_np_cache[n_vertices] = np.column_stack(
            [radii_np * np.cos(angles_np), radii_np * np.sin(angles_np)]
        )
    return _polygon_np_cache[n_vertices]


@pytest.mark.benchmark(group="polyline_triangulate")
@pytest.mark.benchmeshes("sphere_small")
@pytest.mark.benchlibs("triwarp", "meshlib", "pyvista")
@pytest.mark.parametrize("n_vertices", _POLYGON_SIZES)
def test_triangulate_polyline(bench_case: BenchCase, n_vertices: int) -> None:
    """
    Ear clipping a simple polygon: the one group here whose input is not the mesh's boundary.

    ``benchmeshes("sphere_small")`` pins it to a single case because the mesh is irrelevant -- the
    polygon is generated from ``n_vertices`` alone -- and the sweep is that count, which is the only
    thing either implementation's cost depends on. A **star** rather than a convex ring: a fan over
    vertex 0 triangulates any convex polygon in linear time and would make both rows measure
    nothing.

    meshlib's ``triangulateContours`` takes 2-D **closed** contours (the first point repeated) and
    returns a whole ``Mesh``, so its row carries that construction where triwarp's returns an index
    buffer -- it is doing more, and the two agree on the triangle count and total area
    (``tests/test_polyline.py``). Both build their input contour outside the timed callable.

    pyvista's ``triangulate_contours`` is VTK's ear clipper over the same closed loop, and it too
    returns a whole ``PolyData`` -- but of the loop's **own** points: it introduces zero Steiner
    points, so the count is ``n - 2`` on all three sides. Its line cell must carry the repeated
    first index, the same closing convention MeshLib's contour needs.
    """
    polygon_np = _polygon_np(n_vertices)
    if bench_case.kind == "pyvista":
        indices_np = np.append(np.arange(n_vertices), 0)  # closed: the repeat is required
        loop_pv = pv.PolyData(
            np.column_stack([polygon_np, np.zeros(n_vertices)]),
            lines=np.hstack([[indices_np.size], indices_np]),
        )
        filled_pv = bench_case.run(loop_pv.triangulate_contours)
        assert filled_pv.n_cells == n_vertices - 2
        return
    if bench_case.kind == "meshlib":
        contour_ml = mm.std_vector_Vector2_float()
        for point_np in np.vstack([polygon_np, polygon_np[:1]]):
            contour_ml.append(mm.Vector2f(float(point_np[0]), float(point_np[1])))
        contours_ml = mm.std_vector_std_vector_Vector2_float()
        contours_ml.append(contour_ml)
        mesh_ml = bench_case.run(lambda: mm.triangulateContours(contours_ml))
        assert mesh_ml.topology.numValidFaces() == n_vertices - 2
        return
    points_wp = wp.array(
        np.ascontiguousarray(np.column_stack([polygon_np, np.zeros(n_vertices)]), dtype=np.float32),
        dtype=wp.vec3,
        device=bench_case.device,
    )
    faces = bench_case.run(lambda: tw.polyline.polyline_triangulate(points_wp))
    assert int(faces.shape[0]) == n_vertices - 2


_STAR_RINGS = [64, 1024]


def _star_np(n: int) -> np.ndarray:
    """Non-convex star ring: alternating radii, so ear clipping cannot take the single-fan path."""
    angle_np = 2.0 * np.pi * np.arange(n) / n
    radius_np = np.where(np.arange(n) % 2 == 0, 1.0, 0.45)
    return np.column_stack((radius_np * np.cos(angle_np), radius_np * np.sin(angle_np)))


def _star_wp(n: int, device: str) -> wp.array[wp.vec2]:
    return wp.array(
        np.ascontiguousarray(_star_np(n), dtype=np.float32), dtype=wp.vec2, device=device
    )


@pytest.mark.benchmark(group="triangulate_polygon")
@pytest.mark.benchlibs("triwarp", "trimesh")
@pytest.mark.parametrize("ring_size", _STAR_RINGS)
def test_triangulate_polygon(bench_lib: BenchLibrary, ring_size: int) -> None:
    """
    Ear clipping on a *non-convex* ring: the only group here that reaches the clipper at all.

    ``creation.extrude_polygon`` hands the triangulator a convex ring, which takes the single-fan
    fast path and never reaches the ear loop (its row is in [`test_creation.py`](test_creation.py)).
    A star ring forces it. Each round is fully parallel (four ``dim=n`` launches) and the round loop
    itself runs on device, so what this group measures is **how many rounds the independent-set rule
    needs**: 16 and 30, against 62 and 1 022 before the ranking hash.

    At ``ring_size=64`` that is no longer the dominant term -- the clip is 0.22 ms of a 2.2 ms call
    and the flat plane-fitting prologue is 1.05 -- so read the small point as a floor row and the
    large one as a rounds ratio. No open3d counterpart.
    """
    if bench_lib.kind == "triwarp":
        ring_wp = _star_wp(ring_size, str(bench_lib.device))
        _vertices, faces_wp = bench_lib.run(lambda: tw.polyline.triangulate_polygon(ring_wp))
        assert int(faces_wp.shape[0]) // 3 == ring_size - 2
    else:
        polygon = sg.Polygon(_star_np(ring_size))
        _vertices, faces_tm = bench_lib.run(lambda: tm.creation.triangulate_polygon(polygon))
        assert faces_tm.shape[0] > 0
