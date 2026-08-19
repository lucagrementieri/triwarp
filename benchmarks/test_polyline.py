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
* **Resampling** (``upsample_polyline``, ``resample_polyline``, ``downsample_polyline``) — an
  output whose length is data-dependent, so these pay a scan plus a host readback of the output
  size before the write pass. The interpolation itself is the ``wp.lerp`` inner loop.
* **Simplification** (``simplify_polyline``) — Ramer-Douglas-Peucker, which Warp cannot express in
  parallel (it is a recursive split, and Warp forbids recursion), so it runs as a *single-thread*
  stack-based kernel. This is the deliberate outlier of the module and the only case here whose
  cost is O(n) serial work on one GPU thread; expect it to be slower than everything else by
  orders of magnitude and to be the one function where the CPU would win. Only the *recursion* has
  to be serial, though: the keep-mask initialization was moved out to a parallel ``fill_``, which
  this group measured as neutral (25.59 -> 25.53 ms on ``rim_long``) and so is a structural
  cleanup rather than a win — the recursion dominates by two orders of magnitude.

``distance_to_polyline`` is timed separately from the rest because it is the only function whose
cost is the product of two sizes (query points x segments) rather than a function of the polyline
alone.

``triangulate_polygon`` is the remaining exception, and the module's worked example of a cost
that is not where it looks. It delegates to ``polyline.triangulate_polyline``, a **parallel**
multi-round ear clipper — every launch in the round loop is ``dim=n``, and a round clips a whole
*independent set* of ears at once. What is serial is the **round count**: a round costs four
``dim=n`` launches whatever it clips, so the only thing that matters is how many ears survive per
round. Since the round loop moved onto the device (``wp.capture_while``, so no readback per round)
that costs 14 µs a round at 64 points, and the group measures **2.2 / 4.5 ms** against 4.5 / 6.9
before — 2.07x and 1.53x.

Read the residual against its attribution rather than against the round loop, because the round loop
is no longer the cost: at 64 points the clip itself is 0.22 ms of the 2.2, and **1.05 ms is the
prologue** — ``open_polyline``'s ``is_closed`` (0.24), ``polyline_normal`` (0.35) and
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
``simplify_polyline`` on its tolerance (which sets the recursion depth of a serial algorithm) and
``distance_to_polyline`` on the query count (the other half of its two-size product).

References
----------
**No CPU baseline is registered for this module**, and the reason is per-function rather than
blanket:

* **trimesh** models polylines as ``trimesh.path.Path3D`` entities, not arrays, and its only
  simplification is ``trimesh.path.simplify.merge_colinear`` — a colinear-run merge, a different
  algorithm from Ramer-Douglas-Peucker with a different output, so it is not a parity baseline for
  ``simplify_polyline``. It has no arc-length resampling for 3D polylines
  (``resample_spline`` fits a spline first, which changes the geometry).
* **libigl**'s ``igl.upsample`` is *mesh* subdivision, not polyline resampling; the C++
  ``ramer_douglas_peucker`` that ``simplify_polyline`` is ported from is **not exposed** in the
  Python bindings (only ``upsample`` / ``upsample_matrix`` match the name search).
* **open3d** has no polyline type at all — ``LineSet`` stores unordered segments with no ordering,
  length, resampling or simplification operations.

So these are before/after self-comparisons, which is what the batches touching this module
(``wp.length_sq``, ``wp.lerp``, ``wp.sign``) need.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh as tm
import warp as wp
from conftest import BenchCase, BenchLibrary, skip_larger_than
from meshlib import mrmeshpy as mm

import triwarp as tw

# Resampling step, as a fraction of the mean segment length: < 1 upsamples, > 1 downsamples.
_UPSAMPLE_FRACTION = 0.5
_DOWNSAMPLE_FRACTION = 4.0

# Ramer-Douglas-Peucker tolerances, as a fraction of the polyline's bounding-box diagonal. A
# tighter tolerance keeps more points and so recurses deeper, which on a single-thread kernel is
# the whole cost; the pair is two orders of magnitude apart so the slope is unambiguous.
_SIMPLIFY_FRACTIONS = [1e-3, 1e-1]

# Query-point counts for distance_to_polyline: the second size in its points x segments product,
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
    """``(mean_segment_length, bbox_diagonal)`` of the polyline, computed on the host once."""
    polyline = _polyline_wp(bench_case).numpy()
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


@pytest.mark.benchmark(group="polyline_length")
@pytest.mark.benchaxis("polyline")
@pytest.mark.benchlibs("triwarp", "meshlib")
def test_polyline_length(bench_case: BenchCase) -> None:
    """
    Summed segment length: the cheapest whole-polyline reduction, launch-latency bound.

    meshlib's ``calcLength`` sums the same segments and returns a **bit-identical** float32
    (``tests/test_polyline.py``), so this pair is a pure host-against-device reading of the same
    arithmetic -- and on a reduction this cheap triwarp's row is its launch latency, which is what
    makes the comparison worth having. The contour is the input and is cached.
    """
    if bench_case.kind == "meshlib":
        contour_ml = _contour_ml(bench_case)
        assert bench_case.run(lambda: mm.calcLength(contour_ml)) > 0.0
        return
    polyline = _polyline_wp(bench_case)
    length = bench_case.run(lambda: tw.polyline.polyline_length(polyline))
    assert length > 0.0


@pytest.mark.benchmark(group="polyline_radius")
@pytest.mark.benchaxis("polyline")
@pytest.mark.benchlibs("triwarp")
def test_polyline_radius(bench_case: BenchCase) -> None:
    """Per-segment plane projection and closest-point search, then a reduction."""
    polyline = _polyline_wp(bench_case)
    radius = bench_case.run(lambda: tw.polyline.polyline_radius(polyline, reduction="min"))
    assert radius >= 0.0


@pytest.mark.benchmark(group="polyline_angles")
@pytest.mark.benchaxis("polyline")
@pytest.mark.benchlibs("triwarp")
def test_polyline_angles(bench_case: BenchCase) -> None:
    """Per-vertex turning angle: the ``wp.acos`` path, one angle per point."""
    polyline = _polyline_wp(bench_case)
    angles = bench_case.run(lambda: tw.polyline.polyline_angles(polyline))
    assert angles.shape[0] == int(polyline.shape[0])


@pytest.mark.benchmark(group="upsample_polyline")
@pytest.mark.benchaxis("polyline")
@pytest.mark.benchlibs("triwarp")
def test_upsample_polyline(bench_case: BenchCase) -> None:
    """Arc-length upsampling at half the mean segment length: scan, readback, then a lerp pass."""
    polyline = _polyline_wp(bench_case)
    step = _UPSAMPLE_FRACTION * _segment_scale(bench_case)[0]
    dense = bench_case.run(lambda: tw.polyline.upsample_polyline(polyline, step))
    assert dense.shape[0] >= int(polyline.shape[0])


@pytest.mark.benchmark(group="downsample_polyline")
@pytest.mark.benchaxis("polyline")
@pytest.mark.benchlibs("triwarp")
def test_downsample_polyline(bench_case: BenchCase) -> None:
    """Arc-length downsampling at four times the mean segment length."""
    polyline = _polyline_wp(bench_case)
    step = _DOWNSAMPLE_FRACTION * _segment_scale(bench_case)[0]
    sparse = bench_case.run(lambda: tw.polyline.downsample_polyline(polyline, step))
    assert sparse.shape[0] >= 2


@pytest.mark.benchmark(group="simplify_polyline")
@pytest.mark.benchaxis("polyline")
@pytest.mark.benchlibs("triwarp")
@pytest.mark.parametrize("tolerance_fraction", _SIMPLIFY_FRACTIONS)
def test_simplify_polyline(bench_case: BenchCase, tolerance_fraction: float) -> None:
    """Ramer-Douglas-Peucker on a *single* GPU thread — the module's deliberate serial outlier."""
    polyline = _polyline_wp(bench_case)
    tol = tolerance_fraction * _segment_scale(bench_case)[1]
    simplified, kept = bench_case.run(lambda: tw.polyline.simplify_polyline(polyline, tol))
    assert simplified.shape[0] == kept.shape[0]


@pytest.mark.benchmark(group="distance_to_polyline")
@pytest.mark.benchaxis("polyline")
@pytest.mark.benchlibs("triwarp", "meshlib")
@pytest.mark.parametrize("n_queries", _N_QUERIES)
def test_distance_to_polyline(bench_case: BenchCase, n_queries: int) -> None:
    """
    Brute-force point-to-segment distance: the one case whose cost is points x segments.

    That is the contrast meshlib's row is here for: ``findProjectionOnPolyline`` walks an **AABB
    tree** over the segments, so its cost is ``points x log(segments)`` where triwarp's is the full
    product -- read the gap across the ``polyline`` axis rather than at one point. It has no
    batched form, so the row loops in Python and prices that loop along with the queries; the tree
    is built lazily and is pre-warmed outside the timed callable.
    """
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
    distance = bench_case.run(lambda: tw.polyline.distance_to_polyline(points, polyline))
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


@pytest.mark.benchmark(group="triangulate_polyline")
@pytest.mark.benchmeshes("sphere_small")
@pytest.mark.benchlibs("triwarp", "meshlib")
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
    """
    polygon_np = _polygon_np(n_vertices)
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
    faces = bench_case.run(lambda: tw.polyline.triangulate_polyline(points_wp))
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
    shapely = pytest.importorskip("shapely.geometry")
    if bench_lib.kind == "triwarp":
        ring_wp = _star_wp(ring_size, str(bench_lib.device))
        _vertices, faces_wp = bench_lib.run(lambda: tw.polyline.triangulate_polygon(ring_wp))
        assert int(faces_wp.shape[0]) // 3 == ring_size - 2
    else:
        polygon = shapely.Polygon(_star_np(ring_size))
        _vertices, faces_tm = bench_lib.run(lambda: tm.creation.triangulate_polygon(polygon))
        assert faces_tm.shape[0] > 0
