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
  orders of magnitude and to be the one function where the CPU would win.

``distance_to_polyline`` is timed separately from the rest because it is the only function whose
cost is the product of two sizes (query points x segments) rather than a function of the polyline
alone.

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
import warp as wp
from conftest import BenchCase

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


@pytest.mark.benchmark(group="polyline_length")
@pytest.mark.benchaxis("polyline")
@pytest.mark.benchlibs("triwarp")
def test_polyline_length(bench_case: BenchCase) -> None:
    """Summed segment length: the cheapest whole-polyline reduction, launch-latency bound."""
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
@pytest.mark.benchlibs("triwarp")
@pytest.mark.parametrize("n_queries", _N_QUERIES)
def test_distance_to_polyline(bench_case: BenchCase, n_queries: int) -> None:
    """Brute-force point-to-segment distance: the one case whose cost is points x segments."""
    polyline = _polyline_wp(bench_case)
    points = _query_points_wp(bench_case, n_queries)
    distance = bench_case.run(lambda: tw.polyline.distance_to_polyline(points, polyline))
    assert distance.shape[0] == n_queries
