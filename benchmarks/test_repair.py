"""
Benchmarks for ``triwarp.repair.resolve_duplicated_faces``.

The duplicated input is built untimed: a fixed-seed 10% subset of the faces is appended as
*flipped* copies, so every duplicate group is a cancelling ``(+1, -1)`` pair (kept groups stay
orientable; the flipped pairs are dropped by the signed-count rule).

The pre-fix implementation is O(n_unique x n_faces) host Python, so the baseline only runs on
``bunny_decimated``; raise the cap after the device rewrite lands.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp
from conftest import BenchCase, skip_larger_than

import triwarp as tw

_DUP_SEED = 7

# The device rewrite scales; keep lucy out (the duplicated-input build is host-side numpy).
_LARGEST_MESH = "happy_buddha"

_dup_cache: dict[tuple[str, str], wp.array] = {}


def _faces_with_duplicates_wp(bench_case: BenchCase) -> wp.array[wp.int32]:
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _dup_cache:
        faces = bench_case.faces_np
        # Scan meshes ship with genuine same-orientation duplicates (non-orientable groups that
        # the function rejects by contract) and degenerate faces; start from a clean base so the
        # constructed input is exactly "unique faces + cancelling flipped pairs".
        sorted_rows = np.sort(faces, axis=1)
        _, inverse, counts = np.unique(sorted_rows, axis=0, return_inverse=True, return_counts=True)
        base = faces[counts[inverse] == 1]
        nondegenerate = (
            (base[:, 0] != base[:, 1]) & (base[:, 1] != base[:, 2]) & (base[:, 0] != base[:, 2])
        )
        base = base[nondegenerate]
        rng = np.random.default_rng(_DUP_SEED)
        idx = rng.choice(base.shape[0], size=max(1, base.shape[0] // 10), replace=False)
        flipped = base[idx][:, ::-1]
        combined = np.vstack((base, flipped)).astype(np.int32).reshape(-1)
        _dup_cache[key] = wp.array(
            np.ascontiguousarray(combined), dtype=wp.int32, device=bench_case.device
        )
    return _dup_cache[key]


@pytest.mark.benchmark(group="resolve_duplicated_faces")
@pytest.mark.benchlibs("triwarp")
def test_resolve_duplicated_faces(bench_case: BenchCase) -> None:
    skip_larger_than(bench_case, _LARGEST_MESH, "pre-fix host implementation is O(u*n)")
    faces_dup = _faces_with_duplicates_wp(bench_case)
    resolved, kept = bench_case.run(lambda: tw.repair.resolve_duplicated_faces(faces_dup))
    assert resolved.shape[0] == kept.shape[0] * 3
    assert kept.shape[0] > 0
