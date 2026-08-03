"""
Benchmarks for ``triwarp.array``: the packing, sorting and compaction primitives.

Every group here sweeps **segment or selection count**, not mesh size, because that is the axis
these primitives actually respond to. ``pack_1d_arrays`` and ``concatenate`` issue one ``wp.copy``
per input segment on top of a fixed amount of host-side Python, so holding the total element count
fixed and moving only the number of segments separates the two costs: flat means the copies
dominate and the Python loops are noise, rising means the per-segment overhead is the thing to
attack. ``flatnonzero`` sweeps selectivity for the same reason -- the scan is oblivious to it and
only the scatter's output size moves.

``sort_and_argsort`` and ``gather`` are the two primitives on the hot path of nearly every other
module (``grouping.group``, ``adjacency.face_adjacency``, every submesh extraction), so they are
timed on mesh-derived buffers at whatever size the suite is running.

There is no reference library in this file. These are array primitives, not mesh operations:
trimesh, igl, open3d and pymeshlab all operate a level above and expose nothing comparable, and
timing NumPy would compare a host implementation against a device one. ``benchmarks/test_reduce.py``
and ``benchmarks/test_grouping.py`` are triwarp-only for the same reason.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp
from conftest import BenchCase

import triwarp as tw

# Segment counts at a fixed total length: few large pieces against many small ones. The element
# count copied is identical, so only the per-segment cost moves.
_SEGMENT_COUNTS = [4, 256]

# Fraction of the mask that is True. The scan pass is identical either way; only the compaction
# output changes size.
_SELECTIVITIES = [0.5, 0.01]

_segments_cache: dict[tuple[str, str, int], list] = {}
_keys_cache: dict[tuple[str, str], wp.array] = {}
_mask_cache: dict[tuple[str, str, float], wp.array] = {}
_gather_cache: dict[tuple[str, str], tuple] = {}


def _segments(bench_case: BenchCase, n_segments: int) -> list:
    """Split the flat face buffer into ``n_segments`` contiguous 1-D pieces, same total length."""
    key = (bench_case.mesh_name, str(bench_case.device), n_segments)
    if key not in _segments_cache:
        faces_np = bench_case.faces_np.reshape(-1).astype(np.int32)
        pieces = np.array_split(faces_np, n_segments)
        _segments_cache[key] = [
            wp.array(np.ascontiguousarray(piece), dtype=wp.int32, device=bench_case.device)
            for piece in pieces
        ]
    return _segments_cache[key]


def _keys(bench_case: BenchCase) -> wp.array[wp.int32]:
    """Build a shuffled int32 key buffer, one key per face index."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _keys_cache:
        rng = np.random.default_rng(0)
        keys_np = rng.permutation(bench_case.faces_np.size).astype(np.int32)
        _keys_cache[key] = wp.array(keys_np, dtype=wp.int32, device=bench_case.device)
    return _keys_cache[key]


def _mask(bench_case: BenchCase, selectivity: float) -> wp.array[wp.bool]:
    key = (bench_case.mesh_name, str(bench_case.device), selectivity)
    if key not in _mask_cache:
        rng = np.random.default_rng(1)
        mask_np = rng.random(bench_case.faces_np.size) < selectivity
        _mask_cache[key] = wp.array(mask_np, dtype=wp.bool, device=bench_case.device)
    return _mask_cache[key]


def _gather_inputs(bench_case: BenchCase) -> tuple:
    """``(vertices, indices)`` for a half-size vertex gather."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _gather_cache:
        rng = np.random.default_rng(2)
        n = bench_case.n_vertices
        indices_np = rng.integers(0, n, size=max(1, n // 2)).astype(np.int32)
        _gather_cache[key] = (
            bench_case.vertices_wp,
            wp.array(indices_np, dtype=wp.int32, device=bench_case.device),
        )
    return _gather_cache[key]


@pytest.mark.benchmark(group="concatenate_arrays")
@pytest.mark.benchlibs("triwarp")
@pytest.mark.parametrize("n_segments", _SEGMENT_COUNTS, ids=["few", "many"])
def test_concatenate(bench_case: BenchCase, n_segments: int) -> None:
    """One buffer from many, at two segment counts with the total element count held fixed."""
    segments = _segments(bench_case, n_segments)
    flat = bench_case.run(lambda: tw.array.concatenate(segments))
    assert int(flat.shape[0]) == bench_case.faces_np.size


@pytest.mark.benchmark(group="pack_1d_arrays")
@pytest.mark.benchlibs("triwarp")
@pytest.mark.parametrize("n_segments", _SEGMENT_COUNTS, ids=["few", "many"])
def test_pack_1d_arrays(bench_case: BenchCase, n_segments: int) -> None:
    """
    The same packing plus the per-segment offsets, on the same axis as ``concatenate_arrays``.

    Read the two groups together: the offsets are the only difference, so a gap between them at
    ``many`` is the host-side accumulate and the ``wp.array(list)`` transfer, not the copies.
    """
    segments = _segments(bench_case, n_segments)
    flat, offsets = bench_case.run(lambda: tw.array.pack_1d_arrays(segments))
    assert int(flat.shape[0]) == bench_case.faces_np.size
    assert int(offsets.shape[0]) == n_segments


@pytest.mark.benchmark(group="sort_and_argsort")
@pytest.mark.benchlibs("triwarp")
def test_sort_and_argsort(bench_case: BenchCase) -> None:
    """Radix sort plus its permutation: the primitive under ``group`` and every dedup here."""
    keys = _keys(bench_case)
    sorted_keys, order = bench_case.run(lambda: tw.array.sort_and_argsort(keys))
    assert int(sorted_keys.shape[0]) == int(keys.shape[0])
    assert int(order.shape[0]) == int(keys.shape[0])


@pytest.mark.benchmark(group="flatnonzero")
@pytest.mark.benchlibs("triwarp")
@pytest.mark.parametrize("selectivity", _SELECTIVITIES, ids=["half", "sparse"])
def test_flatnonzero(bench_case: BenchCase, selectivity: float) -> None:
    """
    Mask compaction at two selectivities.

    The flag pass and the scan are the same work either way, and only the scatter's output shrinks,
    so these two ids should sit close together. They also pin the cost of the single 4-byte tail
    readback that sizes the output -- the one host synchronisation this primitive cannot avoid.
    """
    mask = _mask(bench_case, selectivity)
    indices = bench_case.run(lambda: tw.array.flatnonzero(mask))
    assert int(indices.shape[0]) > 0


@pytest.mark.benchmark(group="gather")
@pytest.mark.benchlibs("triwarp")
def test_gather(bench_case: BenchCase) -> None:
    """Dense materialization of a fancy-index view: one ``wp.copy`` out of an ``indexedarray``."""
    src, indices = _gather_inputs(bench_case)
    out = bench_case.run(lambda: tw.array.gather(src, indices))
    assert int(out.shape[0]) == int(indices.shape[0])
