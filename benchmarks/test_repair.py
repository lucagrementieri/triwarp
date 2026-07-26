"""
Benchmarks for ``triwarp.repair.resolve_duplicated_faces``.

The duplicated input is built untimed: a fixed-seed 10% subset of the faces is appended as
*flipped* copies, so every duplicate group is a cancelling ``(+1, -1)`` pair (kept groups stay
orientable; the flipped pairs are dropped by the signed-count rule).

The pre-fix implementation is O(n_unique x n_faces) host Python, so the baseline only runs on
``bunny_decimated``; raise the cap after the device rewrite lands.

**open3d** is the reference: ``remove_duplicated_triangles`` solves the same "deduplicate a face
array" problem with a hash set over index triples, against triwarp's sort-based grouping. It is a
comparison of dedup *machinery*, not of results — the semantics differ twice over:

1. open3d keeps one representative of each duplicate group, while triwarp applies a signed-count
   rule that drops cancelling ``(+1, -1)`` pairs outright;
2. open3d's hash is **orientation-sensitive** (measured: it collapses ``[0,1,2]`` against
   ``[0,1,2]`` but not against ``[2,1,0]``), so on this deliberately-flipped input it removes
   nothing and returns the face count unchanged. The hash pass over all ``n`` triples still runs,
   which is the cost being compared; the assertion below only checks the count did not grow.

Open3D mutates the mesh in place and returns ``self``, and the operation is idempotent, so the mesh
is rebuilt inside the timed callable (rounds 2..n would otherwise dedup an already-deduped mesh).
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

_dup_np_cache: dict[str, np.ndarray] = {}
_dup_cache: dict[tuple[str, str], wp.array] = {}


def _faces_with_duplicates_np(bench_case: BenchCase) -> np.ndarray:
    """``(n_faces, 3)`` int32 faces plus a 10% subset re-appended flipped, cached per mesh."""
    if bench_case.mesh_name not in _dup_np_cache:
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
        _dup_np_cache[bench_case.mesh_name] = np.ascontiguousarray(
            np.vstack((base, flipped)), dtype=np.int32
        )
    return _dup_np_cache[bench_case.mesh_name]


def _faces_with_duplicates_wp(bench_case: BenchCase) -> wp.array[wp.int32]:
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _dup_cache:
        combined = _faces_with_duplicates_np(bench_case).reshape(-1)
        _dup_cache[key] = wp.array(
            np.ascontiguousarray(combined), dtype=wp.int32, device=bench_case.device
        )
    return _dup_cache[key]


@pytest.mark.benchmark(group="resolve_duplicated_faces")
@pytest.mark.benchlibs("triwarp", "open3d")
def test_resolve_duplicated_faces(bench_case: BenchCase) -> None:
    skip_larger_than(bench_case, _LARGEST_MESH, "pre-fix host implementation is O(u*n)")
    if bench_case.kind == "triwarp":
        faces_dup = _faces_with_duplicates_wp(bench_case)
        resolved, kept = bench_case.run(lambda: tw.repair.resolve_duplicated_faces(faces_dup))
        assert resolved.shape[0] == kept.shape[0] * 3
        assert kept.shape[0] > 0
    else:
        # open3d dedups in place and is idempotent: build the mesh inside the timed callable.
        import open3d as o3d

        vertices = o3d.utility.Vector3dVector(bench_case.vertices_np)
        faces_dup_np = _faces_with_duplicates_np(bench_case)

        def run() -> o3d.geometry.TriangleMesh:
            mesh = o3d.geometry.TriangleMesh(vertices, o3d.utility.Vector3iVector(faces_dup_np))
            return mesh.remove_duplicated_triangles()

        deduped = bench_case.run(run)
        assert 0 < len(deduped.triangles) <= faces_dup_np.shape[0]
