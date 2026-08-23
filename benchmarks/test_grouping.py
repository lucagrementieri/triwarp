"""
Benchmarks for ``triwarp.grouping``: fixed-multiplicity grouping and row deduplication.

``group`` runs the ``face_adjacency`` workload on the scan sweep -- interior edges appear exactly
twice, so ``length=2`` groups are the adjacent face pairs, and the cost is one radix sort over
``3F`` keys plus a segment pass.

``unique_rows`` instead sweeps the **duplicate density** at a fixed input length, because that is
the only thing that can move: the sort is oblivious to how many rows collide, so the density only
changes the size of the output compaction. A gap wider than the output-size ratio would mean the
segment scan is not oblivious, which is the thing worth catching.

There is no open3d equivalent for either: grouping and deduplicating rows of an arbitrary id array
is an array primitive, not a mesh operation, and open3d exposes nothing at that level. trimesh's
``grouping.unique_rows`` is the host reference for the second group.
"""

from __future__ import annotations

import igl
import numpy as np
import pytest
import trimesh as tm
import warp as wp

import triwarp as tw
from conftest import BenchCase, skip_larger_than

# Fraction of the rows that are distinct: everything unique, against a tenth as many distinct
# values repeated ten times. The input length is identical, so only the collision density moves.
_UNIQUE_FRACTIONS = [1.0, 0.1]

_rows_cache: dict[tuple[str, str, float], tuple] = {}


def _duplicate_rows(bench_case: BenchCase, unique_fraction: float) -> tuple:
    """Build an ``(n, 3)`` int32 row block whose distinct-row count is ``unique_fraction * n``."""
    key = (bench_case.mesh_name, str(bench_case.device), unique_fraction)
    if key not in _rows_cache:
        faces = bench_case.faces_np
        n_unique = max(1, int(faces.shape[0] * unique_fraction))
        rows_np = np.ascontiguousarray(faces[np.arange(faces.shape[0]) % n_unique], dtype=np.int32)
        rows_wp = wp.array(rows_np.reshape(-1), dtype=wp.int32, device=bench_case.device).reshape(
            rows_np.shape
        )
        _rows_cache[key] = (rows_wp, rows_np)
    return _rows_cache[key]


_inverse_cache: dict[tuple[str, str], wp.array] = {}


def _edge_inverse(bench_case: BenchCase) -> wp.array[wp.int32]:
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _inverse_cache:
        _inverse_cache[key] = tw.edges.edges_unique_inverse(
            bench_case.faces_wp, n_vertices=bench_case.n_vertices
        )
    return _inverse_cache[key]


@pytest.mark.benchmark(group="group")
@pytest.mark.benchlibs("triwarp")
def test_group(bench_case: BenchCase) -> None:
    """Fixed-multiplicity index grouping over the edge inverse: a radix sort plus a segment pass."""
    inverse = _edge_inverse(bench_case)
    pairs = bench_case.run(lambda: tw.grouping.group(inverse, 2))
    assert pairs.shape[1] == 2
    assert pairs.shape[0] > 0


@pytest.mark.benchmark(group="unique_faces")
@pytest.mark.benchlibs("triwarp", "igl")
@pytest.mark.parametrize("duplicate_fraction", [0.0, 0.5], ids=["allunique", "half"])
def test_unique_faces(bench_case: BenchCase, duplicate_fraction: float) -> None:
    """
    Orientation-agnostic face deduplication: sort each row, then dedup.

    The axis is the duplicate density, as in ``unique_rows`` above: at ``half`` the input carries a
    *flipped* copy of half its faces, which is the case that separates an orientation-agnostic dedup
    from a plain row dedup -- the flipped copies must collapse.

    ``igl.unique_simplices`` is the same operation and returns ``(FF, IA, IC)`` where ``IC`` is
    triwarp's inverse. One difference to know before comparing: **igl returns the sorted rows**
    (``FF == sort(F(IA, :), 2)``) where triwarp returns the first occurrence with its original
    winding intact, so the parity comparison sorts triwarp's rows first.
    """
    faces_np = bench_case.faces_np
    if duplicate_fraction:
        n_duplicated = int(duplicate_fraction * faces_np.shape[0])
        faces_np = np.concatenate([faces_np, faces_np[:n_duplicated, ::-1]])
    if bench_case.kind == "igl":
        skip_larger_than(bench_case, "bunny", "the reference sorts and dedups on one core")
        faces_igl = np.ascontiguousarray(faces_np, dtype=np.int64)
        unique_igl = bench_case.run(lambda: igl.unique_simplices(faces_igl))
        assert unique_igl[0].shape[0] == bench_case.n_faces
        return
    faces_wp = wp.array(
        np.ascontiguousarray(faces_np.reshape(-1), dtype=np.int32),
        dtype=wp.int32,
        device=bench_case.device,
    )
    unique_wp = bench_case.run(lambda: tw.grouping.unique_faces(faces_wp))
    assert int(unique_wp.shape[0]) // 3 == bench_case.n_faces


@pytest.mark.benchmark(group="unique_rows")
@pytest.mark.benchmeshes("sphere_med")
@pytest.mark.benchlibs("triwarp", "trimesh")
@pytest.mark.parametrize("unique_fraction", _UNIQUE_FRACTIONS, ids=["allunique", "tenth"])
def test_unique_rows(bench_case: BenchCase, unique_fraction: float) -> None:
    """
    Row deduplication, at two duplicate densities.

    Both radix passes cost the same either way -- the sort does not care how many rows collide --
    so any gap between these two rows is the *output* stage: fewer unique rows means a smaller
    compaction, and a larger gap than that would mean the segment scan is doing more work than it
    needs to. Total input length is held fixed so only the duplicate density varies.
    """
    rows_wp, rows_np = _duplicate_rows(bench_case, unique_fraction)
    if bench_case.kind == "triwarp":
        unique = bench_case.run(lambda: tw.grouping.unique_rows(rows_wp))
        assert int(unique[0].shape[0]) > 0
    else:
        unique_tm = bench_case.run(lambda: tm.grouping.unique_rows(rows_np))
        assert len(unique_tm[0]) > 0
