"""
Benchmarks for ``triwarp.reduce``.

This is the module every other one is built on, so what matters here is the *floor*: a whole-array
reduction of a few hundred thousand elements is far too small to saturate a modern GPU, which means
these numbers are dominated by launch latency and — for the reductions that return a Python scalar
— by the device-to-host readback that ends them. That readback is why the full-array variants cannot
be much faster than they are, and why callers inside iterative loops are expected to keep values on
device instead (see the ``check_every`` discussion in ``triwarp/linalg.py``).

Three shapes are timed:

* **Tiled full-array reductions** (``sum``, ``mean``, ``minmax``) — a block reduction into an atomic
  accumulator, then one readback. ``minmax`` produces both extrema in a single pass, so timing it
  next to a bare ``min`` is what justifies its existence.
* **Axis reductions** (``sum(axis=0)``, ``max(axis=1)``) — no readback at all: the result stays on
  device as an array. These are the fair measure of the reduction kernel itself, uncontaminated by
  the host sync, and the row/column split shows the coalescing difference between reducing along
  and across the contiguous axis. Note the two are *not* the same kernel: ``max(axis=1)`` on an
  ``(n, 3)`` table reduces a 3-wide extent and takes the one-thread-per-row serial path, while
  ``sum(axis=0)`` reduces the long extent into 3 outputs and stays tiled. Timing both directions is
  what pins that dispatch, and it is worth 49x in one direction and 89x in the other (see below).
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
``reduce`` is an array primitive, not a geometry operation, so the natural reference is **NumPy**
itself: every group here also times the equivalent host reduction over an already-resident float32
NumPy buffer holding the same values. Read those rows for what they are — the host-side floor, not
a like-for-like kernel race. A device-side reduction that returns a Python scalar pays a flat
launch + readback latency that NumPy never pays, so NumPy *should* win at small sizes and the
question each row answers is *where the crossover sits* and whether the device side stays flat past
it. trimesh, libigl and open3d expose no array-reduction API at all.

What the comparison actually says, so nobody has to re-derive it from the table: **the split is by
return type, not by size or by dtype.** The groups that hand back a device array or a ``wp.vec3``
— both axis groups and both ``vec3`` groups — are ahead of NumPy at every mesh in the registry,
though only narrowly at ``bunny_decimated`` (1.1-1.6x, and ``mean_vec3`` there is close enough that
a single outlier round inverts its *mean* while its min and median stay ahead — read the min) and
by 80-200x at ``lucy``. The groups that hand back a *Python scalar* (``sum_scalar``, ``min_scalar``,
``minmax_scalar``, ``median``) lose to NumPy below roughly half a million elements and win above it:
``min_scalar`` is 60x slower than NumPy on ``bunny_decimated`` and 11x faster on ``lucy``.

That crossover is a host cost, and the split was measured rather than assumed: ~35 µs of launch
marshalling, ~30 µs for the 4-byte readback and ~17 µs to allocate and fill the output buffer —
**~82 µs before a single element is touched**, against NumPy's 1.6 µs for the whole 8k reduction.
Only the ``sync`` term scales with n. So a scalar-returning reduction cannot win at small n no
matter what the kernel does, which is why callers inside iterative loops are expected to keep values
on device instead (see the ``check_every`` discussion in ``triwarp/linalg.py``).

Three kernel defects were found while chasing that floor, all since fixed. None was *exposed* by the
NumPy rows — triwarp was already ahead in those groups — but all three were found by asking the
question these rows invite, namely whether the tiling is earning its keep. Two are the same bug in
different clothes:

- **Tiling below one tile is pure loss.** The tiled axis kernels never reach their ``wp.tile_load``
  branch when the reduced extent is under ``TILE_1D``, so all 64 lanes of every block redundantly
  walked the same 3-element row — a 64-fold read amplification. One thread per output row is **49x**
  faster on a ``(14M, 3)`` table, and the wrapper now dispatches on the reduced extent. Note the
  converse holds too: the same table on ``axis=0`` has 3 outputs, where serial is **89x slower**, so
  the dispatch key is the reduced extent and not the axis.
- **The same thing again, one rank up.** A rank-2 ``axis=None`` reduction tiles ``TILE_2D`` squares,
  which an ``(n, 3)`` vertex table or ``(m, 2)`` edge table clips exactly as above. Flattening a
  contiguous narrow table to the 1-D kernel is **5.95x** on ``(14M, 3)`` and **6.18x** on
  ``(8M, 2)``, versus 1.02x on ``(40k, 128)`` where the tile branch does fire — which is why the
  test is on the trailing extent, not just on contiguity.
- **One atomic per tile does not scale.** The global 1-D reductions ran ~9x off the memory-bandwidth
  floor at 14M elements because one ``atomic_add`` per 64-element block put 219k blocks on a single
  accumulator address. Folding ``TILES_PER_BLOCK_1D`` tiles into a register first is **4.9x** faster
  and lands within 2x of bandwidth.

Those are kernel-time A/Bs, interleaved under one clock state with values verified each round. End
to end the picture is uneven and worth reading carefully: at ``lucy`` ``max_axis1`` gains **36.7x**,
``minmax_global_2d`` **5.5x** and ``sum_scalar`` **2.9x**, but every scalar-returning group at
``bunny``-scale moves by less than +/-10% -- which is *within* the cross-session drift band
CLAUDE.md section 15.7 warns about, so those cells attribute nothing either way. That is the
expected shape: ~82 us of such a call was never the kernel, so no kernel change can move it.

One cross-check is worth keeping, because it is what says the rank-2 fix closed the gap rather than
moved time around: subtract the host floor and ``minmax_global_2d`` at ``lucy`` runs 42.1M elements
at 7.0 ns per thousand, against 7.3 for the rank-1 ``minmax_scalar`` on 14.0M. The two paths agree
on throughput now; before the fix the rank-2 one was 3.7x worse per element.

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

The NumPy rows are the only ones that are *always* comparable: every group computes exactly the
reduction its NumPy call computes, so each pair is a class-A parity claim in
[`tests/test_reduce.py`](../tests/test_reduce.py) with no transform in between.
"""

from __future__ import annotations

import numpy as np
import pymeshlab as ml
import pytest
import pytorch3d.ops.utils as p3d_ops_utils
import torch
import warp as wp

import triwarp as tw
import triwarp.typing as twt
from conftest import BenchCase

_mask_cache: dict[tuple[str, str], wp.array] = {}
_scalar_cache: dict[tuple[str, str], wp.array[wp.float32]] = {}
_rows_cache: dict[tuple[str, str], twt.Array2dFloat32] = {}
_scalar_np_cache: dict[str, np.ndarray] = {}
_rows_np_cache: dict[str, np.ndarray] = {}
_face_cache: dict[tuple[str, str, str], wp.array[wp.float32]] = {}
_face_value_np_cache: dict[str, np.ndarray] = {}
_face_area_np_cache: dict[str, np.ndarray] = {}


def _scalars_np(bench_case: BenchCase) -> np.ndarray:
    """``(n_vertices,)`` float32 host scalars — the same values ``_scalars_wp`` uploads."""
    if bench_case.mesh_name not in _scalar_np_cache:
        _scalar_np_cache[bench_case.mesh_name] = np.ascontiguousarray(
            bench_case.vertices_np[:, 2], dtype=np.float32
        )
    return _scalar_np_cache[bench_case.mesh_name]


def _rows_np(bench_case: BenchCase) -> np.ndarray:
    """``(n_vertices, 3)`` float32 host table — the same values ``_rows_wp`` uploads."""
    if bench_case.mesh_name not in _rows_np_cache:
        _rows_np_cache[bench_case.mesh_name] = np.ascontiguousarray(
            bench_case.vertices_np, dtype=np.float32
        )
    return _rows_np_cache[bench_case.mesh_name]


def _face_values_np(bench_case: BenchCase) -> np.ndarray:
    """``(n_faces,)`` float32 host scalars — the first corner's ``z``, one value per face."""
    if bench_case.mesh_name not in _face_value_np_cache:
        _face_value_np_cache[bench_case.mesh_name] = np.ascontiguousarray(
            bench_case.vertices_np[bench_case.faces_np[:, 0], 2], dtype=np.float32
        )
    return _face_value_np_cache[bench_case.mesh_name]


def _face_areas_np(bench_case: BenchCase) -> np.ndarray:
    """``(n_faces,)`` float32 triangle areas — the integration weights, built once per mesh."""
    if bench_case.mesh_name not in _face_area_np_cache:
        triangles_np = bench_case.vertices_np[bench_case.faces_np]
        crosses_np = np.cross(
            triangles_np[:, 1] - triangles_np[:, 0], triangles_np[:, 2] - triangles_np[:, 0]
        )
        _face_area_np_cache[bench_case.mesh_name] = np.ascontiguousarray(
            0.5 * np.linalg.norm(crosses_np, axis=1), dtype=np.float32
        )
    return _face_area_np_cache[bench_case.mesh_name]


def _face_values_wp(bench_case: BenchCase) -> wp.array[wp.float32]:
    """``(n_faces,)`` float32 per-face scalars on device — the same values as the host copy."""
    key = ("values", bench_case.mesh_name, str(bench_case.device))
    if key not in _face_cache:
        _face_cache[key] = wp.array(
            _face_values_np(bench_case), dtype=wp.float32, device=bench_case.device
        )
    return _face_cache[key]


def _face_areas_wp(bench_case: BenchCase) -> wp.array[wp.float32]:
    """``(n_faces,)`` float32 areas on device — an *input* to the weighted reduction."""
    key = ("areas", bench_case.mesh_name, str(bench_case.device))
    if key not in _face_cache:
        _face_cache[key] = wp.array(
            _face_areas_np(bench_case), dtype=wp.float32, device=bench_case.device
        )
    return _face_cache[key]


def _scalars_wp(bench_case: BenchCase) -> wp.array[wp.float32]:
    """``(n_vertices,)`` float32 scalars — the vertices' z coordinate."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _scalar_cache:
        _scalar_cache[key] = wp.array(
            _scalars_np(bench_case), dtype=wp.float32, device=bench_case.device
        )
    return _scalar_cache[key]


def _rows_wp(bench_case: BenchCase) -> twt.Array2dFloat32:
    """``(n_vertices, 3)`` float32 — the vertex table as a rank-2 array for the axis cases."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _rows_cache:
        _rows_cache[key] = wp.array(
            _rows_np(bench_case), dtype=wp.float32, device=bench_case.device
        )
    return _rows_cache[key]


def _mask_wp(bench_case: BenchCase) -> wp.array[wp.bool]:
    """``(n_vertices,)`` ``wp.bool`` mask — roughly half set, so no predicate short-circuits."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _mask_cache:
        _mask_cache[key] = wp.array(
            _scalars_np(bench_case) > float(np.median(_scalars_np(bench_case))),
            dtype=wp.bool,
            device=bench_case.device,
        )
    return _mask_cache[key]


@pytest.mark.benchmark(group="sum_bool")
@pytest.mark.benchlibs("triwarp", "numpy")
def test_sum_bool(bench_case: BenchCase) -> None:
    """
    Counting a ``wp.bool`` mask, the shape thirteen call sites across the package use.

    Timed separately from ``sum_scalar`` because a mask is **one byte per element** where a float32
    is four, so this is the one reduction whose traffic is set by the input dtype rather than by the
    launch floor, and the only one where a host readback of the whole buffer is a serious rival (a
    bool copy is a quarter the bytes of the float32 one ``sum_scalar`` would need). Callers that
    reduce a mask once per call are the majority; ``registration.icp`` reduces one per iteration.
    """
    if bench_case.kind == "numpy":
        mask_np = _scalars_np(bench_case) > float(np.median(_scalars_np(bench_case)))
        total_np = bench_case.run(lambda: int(mask_np.sum()))
        assert total_np >= 0
        return
    mask = _mask_wp(bench_case)
    total = bench_case.run(lambda: tw.reduce.sum(mask))
    assert total >= 0


@pytest.mark.benchmark(group="any_bool")
@pytest.mark.benchlibs("triwarp", "numpy")
def test_any_bool(bench_case: BenchCase) -> None:
    """
    ``reduce.any`` over a mask — the predicate shape ``validation`` and ``mesh.Trimesh`` use.

    The mask is half set rather than all-``False``, so neither side can answer from the first
    element; a benchmark on an all-``False`` mask measures a different question than the callers
    ask (``is_watertight`` on a watertight mesh is the all-``False`` case, and it is the *cheap*
    one).
    """
    if bench_case.kind == "numpy":
        mask_np = _scalars_np(bench_case) > float(np.median(_scalars_np(bench_case)))
        flag_np = bench_case.run(lambda: bool(mask_np.any()))
        assert isinstance(flag_np, bool)
        return
    mask = _mask_wp(bench_case)
    flag = bench_case.run(lambda: tw.reduce.any(mask))
    assert isinstance(flag, bool)


@pytest.mark.benchmark(group="all_bool")
@pytest.mark.benchlibs("triwarp", "numpy")
def test_all_bool(bench_case: BenchCase) -> None:
    """``reduce.all`` over the same mask — the other half of the predicate pair."""
    if bench_case.kind == "numpy":
        mask_np = _scalars_np(bench_case) > float(np.median(_scalars_np(bench_case)))
        flag_np = bench_case.run(lambda: bool(mask_np.all()))
        assert isinstance(flag_np, bool)
        return
    mask = _mask_wp(bench_case)
    flag = bench_case.run(lambda: tw.reduce.all(mask))
    assert isinstance(flag, bool)


@pytest.mark.benchmark(group="sum_scalar")
@pytest.mark.benchlibs("triwarp", "numpy")
def test_sum_scalar(bench_case: BenchCase) -> None:
    """Tiled float32 sum to a Python scalar: one block reduction plus the host readback."""
    if bench_case.kind == "numpy":
        values_np = _scalars_np(bench_case)
        total_np = bench_case.run(lambda: float(values_np.sum()))
        assert np.isfinite(total_np)
        return
    values = _scalars_wp(bench_case)
    total = bench_case.run(lambda: tw.reduce.sum(values))
    assert np.isfinite(total)


@pytest.mark.benchmark(group="sum_vec3")
@pytest.mark.benchlibs("triwarp", "numpy")
def test_sum_vec3(bench_case: BenchCase) -> None:
    """The ``wp.vec3`` accumulator path — what a centroid actually calls."""
    if bench_case.kind == "numpy":
        rows_np = _rows_np(bench_case)
        total_np = bench_case.run(lambda: rows_np.sum(axis=0))
        assert total_np.shape == (3,)
        return
    vertices = bench_case.vertices_wp
    total = bench_case.run(lambda: tw.reduce.sum(vertices))
    assert len(total) == 3


@pytest.mark.benchmark(group="mean_vec3")
@pytest.mark.benchlibs("triwarp", "numpy")
def test_mean_vec3(bench_case: BenchCase) -> None:
    """``sum`` plus a scalar divide: the delta over ``sum_vec3`` is the normalization."""
    if bench_case.kind == "numpy":
        rows_np = _rows_np(bench_case)
        centroid_np = bench_case.run(lambda: rows_np.mean(axis=0))
        assert centroid_np.shape == (3,)
        return
    vertices = bench_case.vertices_wp
    centroid = bench_case.run(lambda: tw.reduce.mean(vertices))
    assert len(centroid) == 3


@pytest.mark.benchmark(group="weighted_sum")
@pytest.mark.benchlibs("triwarp", "numpy", "pyvista", "pytorch3d")
def test_weighted_sum(bench_case: BenchCase) -> None:
    """
    ``sum(values * weights)`` in one pass: the reduction every surface integral bottoms out in.

    The weights here are the per-face areas and the values a per-face scalar, which is exactly what
    VTK's ``integrate_data`` computes for a cell array -- measured equal to 8 significant digits
    (``tests/test_reduce.py``). Both are *inputs*: the areas are built once per mesh outside the
    timed region, on both sides, so the row measures the reduction and not a cross-product pass.

    The pyvista row does more than the other two by construction -- ``integrate_data`` integrates
    every array on the mesh and returns a one-cell ``UnstructuredGrid`` -- and it is the whole
    reason this group exists rather than folding into ``sum_scalar``: nothing else in the reference
    set exposes a weighted reduction at all.

    **pytorch3d**'s ``ops.utils.wmean`` is the weighted *mean* -- the same reduction plus a division
    by ``sum(weights)`` with an ``eps`` floor -- so its row does one more pass than triwarp's and
    is an upper bound rather than a race. It is the only GPU reference in the module, which is what
    it is here for: the ``numpy`` row is the host floor and this one says what the same reduction
    costs in another device library. Both arrays are inputs and are uploaded outside the timed
    callable, as on the other two rows.
    """
    if bench_case.kind == "pytorch3d":
        values_p3d = torch.as_tensor(
            _face_values_np(bench_case).astype(np.float32), device=bench_case.torch_device
        ).reshape(1, -1, 1)
        weights_p3d = torch.as_tensor(
            _face_areas_np(bench_case).astype(np.float32), device=bench_case.torch_device
        ).reshape(1, -1)
        mean_p3d = bench_case.run(lambda: p3d_ops_utils.wmean(values_p3d, weights_p3d))
        assert bool(torch.isfinite(mean_p3d).all())
        return
    if bench_case.kind == "pyvista":
        mesh_pv = bench_case.mesh_pv
        mesh_pv.cell_data["field"] = _face_values_np(bench_case).astype(np.float64)
        integrated_pv = bench_case.run(mesh_pv.integrate_data)
        assert np.isfinite(np.asarray(integrated_pv.cell_data["field"])[0])
        return
    if bench_case.kind == "numpy":
        values_np, areas_np = _face_values_np(bench_case), _face_areas_np(bench_case)
        total_np = bench_case.run(lambda: float((values_np * areas_np).sum()))
        assert np.isfinite(total_np)
        return
    values, areas = _face_values_wp(bench_case), _face_areas_wp(bench_case)
    total = bench_case.run(lambda: tw.reduce.weighted_sum(values, areas))
    assert np.isfinite(total)


@pytest.mark.benchmark(group="minmax_scalar")
@pytest.mark.benchlibs("triwarp", "numpy")
def test_minmax_scalar(bench_case: BenchCase) -> None:
    """Both extrema in one pass — compare against ``min_scalar`` for the single-pass saving."""
    if bench_case.kind == "numpy":
        values_np = _scalars_np(bench_case)
        lo_np, hi_np = bench_case.run(lambda: (float(values_np.min()), float(values_np.max())))
        assert lo_np <= hi_np
        return
    values = _scalars_wp(bench_case)
    lo, hi = bench_case.run(lambda: tw.reduce.minmax(values))
    assert lo <= hi


@pytest.mark.benchmark(group="min_scalar")
@pytest.mark.benchlibs("triwarp", "numpy")
def test_min_scalar(bench_case: BenchCase) -> None:
    """A single extremum, for reference against ``minmax_scalar``."""
    if bench_case.kind == "numpy":
        values_np = _scalars_np(bench_case)
        lo_np = bench_case.run(lambda: float(values_np.min()))
        assert np.isfinite(lo_np)
        return
    values = _scalars_wp(bench_case)
    lo = bench_case.run(lambda: tw.reduce.min(values))
    assert np.isfinite(lo)


@pytest.mark.benchmark(group="minmax_global_2d")
@pytest.mark.benchlibs("triwarp", "numpy")
def test_minmax_global_2d(bench_case: BenchCase) -> None:
    """
    A rank-2 table reduced to one scalar pair — the shape ``graph`` validates an edge list with.

    Distinct from ``minmax_scalar`` (rank-1) because rank-2 ``axis=None`` dispatches to its own
    ``TILE_2D``-square kernel, and distinct from the axis groups because it ends in a readback.
    The narrow trailing extent is the point: a ``(m, 2)`` edge table or ``(n, 3)`` vertex table
    clips the 8x8 tile to 8x2, so the tile branch is never taken.
    """
    if bench_case.kind == "numpy":
        rows_np = _rows_np(bench_case)
        lo_np, hi_np = bench_case.run(lambda: (float(rows_np.min()), float(rows_np.max())))
        assert lo_np <= hi_np
        return
    rows = _rows_wp(bench_case)
    lo, hi = bench_case.run(lambda: tw.reduce.minmax(rows))
    assert lo <= hi


@pytest.mark.benchmark(group="sum_axis0")
@pytest.mark.benchlibs("triwarp", "numpy")
def test_sum_axis0(bench_case: BenchCase) -> None:
    """Column sums of an ``(n, 3)`` table: device-resident result, no readback."""
    if bench_case.kind == "numpy":
        rows_np = _rows_np(bench_case)
        sums_np = bench_case.run(lambda: rows_np.sum(axis=0))
        assert sums_np.shape == (3,)
        return
    rows = _rows_wp(bench_case)
    sums = bench_case.run(lambda: tw.reduce.sum(rows, axis=0))
    assert sums.shape[0] == 3


@pytest.mark.benchmark(group="max_axis1")
@pytest.mark.benchlibs("triwarp", "numpy")
def test_max_axis1(bench_case: BenchCase) -> None:
    """Row maxima of the same table: reduces along the contiguous axis instead of across it."""
    if bench_case.kind == "numpy":
        rows_np = _rows_np(bench_case)
        maxima_np = bench_case.run(lambda: rows_np.max(axis=1))
        assert maxima_np.shape[0] == bench_case.n_vertices
        return
    rows = _rows_wp(bench_case)
    maxima = bench_case.run(lambda: tw.reduce.max(rows, axis=1))
    assert maxima.shape[0] == bench_case.n_vertices


def _scalar_meshset_pml(bench_case: BenchCase) -> ml.MeshSet:
    """Return the shared MeshSet with the vertices' ``z`` in the vertex scalar attribute."""
    meshset_pml = bench_case.meshset_pml
    meshset_pml.compute_scalar_by_function_per_vertex(q="z")
    return meshset_pml


@pytest.mark.benchmark(group="median")
@pytest.mark.benchlibs("triwarp", "pymeshlab", "numpy")
def test_median(bench_case: BenchCase) -> None:
    """Radix-sorts a copy and reads the middle: O(n log n) where the rest are one pass."""
    if bench_case.kind == "pymeshlab":  # one call: min, max, avg, med, stddev, variance
        meshset_pml = _scalar_meshset_pml(bench_case)
        statistics_pml = bench_case.run(meshset_pml.get_scalar_statistics_per_vertex)
        assert np.isfinite(statistics_pml["med"])
        return
    if bench_case.kind == "numpy":  # introselect partition, not a full sort
        values_np = _scalars_np(bench_case)
        middle_np = bench_case.run(lambda: float(np.median(values_np)))
        assert np.isfinite(middle_np)
        return
    values = _scalars_wp(bench_case)
    middle = bench_case.run(lambda: tw.reduce.median(values))
    assert np.isfinite(middle)
