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

import numpy as np
import pytest
import trimesh as tm
import warp as wp
from conftest import BenchCase

import triwarp as tw

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
