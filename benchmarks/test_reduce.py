"""
Benchmarks for ``triwarp.reduce``.

This is the module every other one is built on, so what matters here is the *floor*: a whole-array
reduction of a few hundred thousand elements is far too small to saturate a modern GPU, which means
these numbers are dominated by launch latency and — for the reductions that return a Python scalar
— by the device-to-host readback that ends them. That readback is why the full-array variants cannot
be much faster than they are, and why callers inside iterative loops are expected to keep values on
device instead (see the ``check_every`` discussion in ``triwarp/linalg.py``).

Three shapes are timed:

* **Tiled full-array reductions** (``sum``, ``mean``, ``minmax``) — a ``TILE_1D``-wide block
  reduction into an atomic accumulator, then one readback. ``minmax`` produces both extrema in a
  single pass, so timing it next to a bare ``min`` is what justifies its existence.
* **Axis reductions** (``sum(axis=0)``, ``max(axis=1)``) — no readback at all: the result stays on
  device as an array. These are the fair measure of the reduction kernel itself, uncontaminated by
  the host sync, and the row/column split shows the coalescing difference between reducing along
  and across the contiguous axis.
* **Sort-based** (``median``) — the outlier. It radix-sorts a *copy* of the values with
  ``warp.utils.radix_sort_pairs`` and reads the middle element, so it is an O(n log n) full sort
  where every other function here is a single O(n) pass, and it allocates. Expect a large constant
  factor against ``mean``.

``vec3`` overloads (``sum`` / ``mean`` over ``wp.array[wp.vec3]``) are timed alongside the scalar
ones because they are the ones real callers hit — centroids and normal averages — and they exercise
a different tile accumulator (``wp.vec3`` atomics rather than scalar).

Inputs
------
Derived from the registry meshes so the sizes track the rest of the suite: the ``wp.vec3`` vertex
buffer directly, its flattened ``(n_vertices, 3)`` float32 view for the axis cases, and a scalar
per-vertex array for the 1D cases. Buffers are built once per (mesh, device) and reused, so the
timed region holds only the reduction.

References
----------
``reduce`` is an array primitive, not a geometry operation — the same reason
[`test_grouping.py`](test_grouping.py) is triwarp-only. NumPy would be the natural reference but is
not a library *kind* in the harness registry (it is the substrate every CPU baseline is already
built on), and timing a host reduction against a device one measures PCIe and thread count rather
than anything a change to these kernels would move. trimesh, libigl and open3d expose no
array-reduction API at all.

**pymeshlab** is the one exception and lands in the ``median`` group.
``get_scalar_statistics_per_vertex`` reduces a per-vertex scalar attribute to ``{min, max, avg, med,
stddev, variance}`` — so it is genuinely the same work, over the same input (the scalar is seeded
from the vertices' ``z`` with ``compute_scalar_by_function_per_vertex(q='z')``, which reproduces
``_scalars_wp`` exactly). Two things follow:

- **It answers six questions in one call**, so its single number is simultaneously the reference for
  ``median``, ``minmax_scalar``, ``min_scalar`` and a scalar mean — an *upper* bound for each one
  taken alone and a *lower* bound for computing all of them. It appears once, in ``median``, because
  the percentile is the part that needs a sort and therefore the part that dominates both sides.
- **It is read-only** (it returns a dict and touches nothing), so the MeshSet is shared and only the
  attribute seeding sits outside the timed callable.

Everything else here stays a before/after self-comparison, which is what the tile-tail-clamp batch
touching this module needs.
"""

from __future__ import annotations

import numpy as np
import pymeshlab as ml
import pytest
import warp as wp
from conftest import BenchCase

import triwarp as tw
import triwarp.typing as twt

_scalar_cache: dict[tuple[str, str], wp.array[wp.float32]] = {}
_rows_cache: dict[tuple[str, str], twt.Array2dFloat32] = {}


def _scalars_wp(bench_case: BenchCase) -> wp.array[wp.float32]:
    """``(n_vertices,)`` float32 scalars — the vertices' z coordinate."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _scalar_cache:
        _scalar_cache[key] = wp.array(
            np.ascontiguousarray(bench_case.vertices_np[:, 2], dtype=np.float32),
            dtype=wp.float32,
            device=bench_case.device,
        )
    return _scalar_cache[key]


def _rows_wp(bench_case: BenchCase) -> twt.Array2dFloat32:
    """``(n_vertices, 3)`` float32 — the vertex table as a rank-2 array for the axis cases."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _rows_cache:
        _rows_cache[key] = wp.array(
            np.ascontiguousarray(bench_case.vertices_np, dtype=np.float32),
            dtype=wp.float32,
            device=bench_case.device,
        )
    return _rows_cache[key]


@pytest.mark.benchmark(group="sum_scalar")
@pytest.mark.benchlibs("triwarp")
def test_sum_scalar(bench_case: BenchCase) -> None:
    """Tiled float32 sum to a Python scalar: one block reduction plus the host readback."""
    values = _scalars_wp(bench_case)
    total = bench_case.run(lambda: tw.reduce.sum(values))
    assert np.isfinite(total)


@pytest.mark.benchmark(group="sum_vec3")
@pytest.mark.benchlibs("triwarp")
def test_sum_vec3(bench_case: BenchCase) -> None:
    """The ``wp.vec3`` accumulator path — what a centroid actually calls."""
    vertices = bench_case.vertices_wp
    total = bench_case.run(lambda: tw.reduce.sum(vertices))
    assert len(total) == 3


@pytest.mark.benchmark(group="mean_vec3")
@pytest.mark.benchlibs("triwarp")
def test_mean_vec3(bench_case: BenchCase) -> None:
    """``sum`` plus a scalar divide: the delta over ``sum_vec3`` is the normalization."""
    vertices = bench_case.vertices_wp
    centroid = bench_case.run(lambda: tw.reduce.mean(vertices))
    assert len(centroid) == 3


@pytest.mark.benchmark(group="minmax_scalar")
@pytest.mark.benchlibs("triwarp")
def test_minmax_scalar(bench_case: BenchCase) -> None:
    """Both extrema in one pass — compare against ``min_scalar`` for the single-pass saving."""
    values = _scalars_wp(bench_case)
    lo, hi = bench_case.run(lambda: tw.reduce.minmax(values))
    assert lo <= hi


@pytest.mark.benchmark(group="min_scalar")
@pytest.mark.benchlibs("triwarp")
def test_min_scalar(bench_case: BenchCase) -> None:
    """A single extremum, for reference against ``minmax_scalar``."""
    values = _scalars_wp(bench_case)
    lo = bench_case.run(lambda: tw.reduce.min(values))
    assert np.isfinite(lo)


@pytest.mark.benchmark(group="sum_axis0")
@pytest.mark.benchlibs("triwarp")
def test_sum_axis0(bench_case: BenchCase) -> None:
    """Column sums of an ``(n, 3)`` table: device-resident result, no readback."""
    rows = _rows_wp(bench_case)
    sums = bench_case.run(lambda: tw.reduce.sum(rows, axis=0))
    assert sums.shape[0] == 3


@pytest.mark.benchmark(group="max_axis1")
@pytest.mark.benchlibs("triwarp")
def test_max_axis1(bench_case: BenchCase) -> None:
    """Row maxima of the same table: reduces along the contiguous axis instead of across it."""
    rows = _rows_wp(bench_case)
    maxima = bench_case.run(lambda: tw.reduce.max(rows, axis=1))
    assert maxima.shape[0] == bench_case.n_vertices


def _scalar_meshset_pml(bench_case: BenchCase) -> ml.MeshSet:
    """Return the shared MeshSet with the vertices' ``z`` in the vertex scalar attribute."""
    meshset_pml = bench_case.meshset_pml
    meshset_pml.compute_scalar_by_function_per_vertex(q="z")
    return meshset_pml


@pytest.mark.benchmark(group="median")
@pytest.mark.benchlibs("triwarp", "pymeshlab")
def test_median(bench_case: BenchCase) -> None:
    """Radix-sorts a copy and reads the middle: O(n log n) where the rest are one pass."""
    if bench_case.kind == "pymeshlab":  # one call: min, max, avg, med, stddev, variance
        meshset_pml = _scalar_meshset_pml(bench_case)
        statistics_pml = bench_case.run(meshset_pml.get_scalar_statistics_per_vertex)
        assert np.isfinite(statistics_pml["med"])
        return
    values = _scalars_wp(bench_case)
    middle = bench_case.run(lambda: tw.reduce.median(values))
    assert np.isfinite(middle)
