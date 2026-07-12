"""
Benchmark for ``triwarp.grouping.group`` on edge-inverse ids.

This is the ``face_adjacency`` workload: interior edges appear exactly twice, so ``length=2``
groups are the adjacent face pairs.
"""

from __future__ import annotations

import pytest
import warp as wp
from conftest import BenchCase

import triwarp as tw

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
    inverse = _edge_inverse(bench_case)
    pairs = bench_case.run(lambda: tw.grouping.group(inverse, 2))
    assert pairs.shape[1] == 2
    assert pairs.shape[0] > 0
