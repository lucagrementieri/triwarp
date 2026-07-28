"""
Benchmarks for ``triwarp.combine``: assembling meshes from parts and splitting them apart.

Axis: **components** for ``split`` and ``concatenate``, **loops_dp** for the stitching pair.
Neither family scales with face count, and ``split`` is the sharpest example in the package of a
function whose cost lives entirely somewhere else.

``split`` is one labelling pass plus one stable radix sort -- both O(F) and both fast -- followed
by a **host loop calling ``submesh_from_face_indices`` once per component**, each iteration a
handful of allocations and launches. Measured at a fixed 81 920 faces:

| components | triwarp-cuda | trimesh | open3d |
|---|---|---|---|
| 1 | **2.55 ms** | -- | -- |
| 64 | 41.4 ms | -- | -- |
| 1024 | **669 ms** | 203 ms | 284 ms |

262x across the axis, and the sign of the comparison flips: triwarp wins by orders of magnitude
when there is one component and **loses by 3x** when there are a thousand. About 0.64 ms of host
work per returned submesh. Batching that per-component sequence is the single highest-value fix
the axis set has surfaced, and no face-count sweep would have shown it -- the scan registry's
meshes happen to differ in component count by accident (``bunny_decimated`` has 94 scan floaters,
``bunny`` has 1), which is how the effect was originally noticed at all.

``stitch_min_weight`` runs a grid dynamic program over the two rims: an O(La x Lb) table filled by
O(La + Lb) *sequential* anti-diagonal launches, then a host traceback. So it is sized by rim
length, which is what ``loops_dp`` provides -- ``rim_short``'s two 512-vertex rims give a
512 x 512 table, and the meshes are deliberately tiny (1 024 faces) because the face count is
irrelevant to the cost.

References
----------
**trimesh**'s ``Trimesh.split(only_watertight=False)`` and **open3d**'s
``cluster_connected_triangles`` plus a ``select_by_index`` per cluster are the two equivalents of
``split``; both do the same labelling-then-compaction, and the open3d branch's ``np.unique`` over
each cluster's faces is part of what an open3d user pays, exactly as scipy is for the trimesh path.

Neither has an equivalent of ``stitch`` / ``stitch_min_weight``: joining two open meshes along
their boundary loops with a minimum-weight triangulation is not in either API (trimesh's
``util.concatenate`` merges without stitching, and open3d's boolean operations need closed
inputs). Those two groups are before/after self-comparisons.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh as tm
import warp as wp
from conftest import BenchCase

import triwarp as tw

# Copies for the concatenate sweep: the function loops on the host once per input mesh, so the
# input *count* is the driver and the total face count is held roughly fixed between the points.
_CONCAT_COPIES = [8, 512]

# split on a thousand components runs to two thirds of a second.
_ROUNDS = 3

_split_cache: dict[tuple[str, str], tuple] = {}
_parts_cache: dict[tuple[str, str, int], list] = {}
_stitch_cache: dict[tuple[str, str], tuple] = {}


def _split_inputs(bench_case: BenchCase) -> tuple:
    """Return the whole mesh as one soup, on the device (or host) the case needs."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _split_cache:
        if bench_case.kind == "triwarp":
            _split_cache[key] = (bench_case.vertices_wp, bench_case.faces_wp)
        else:
            _split_cache[key] = (bench_case.vertices_np, bench_case.faces_np)
    return _split_cache[key]


@pytest.mark.benchmark(group="split")
@pytest.mark.benchaxis("components")
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d")
def test_split(bench_case: BenchCase) -> None:
    """Label, sort, then one submesh extraction per component: 262x across the axis."""
    expected = {"sphere_med": 1, "parts_64": 64, "parts_1024": 1024}[bench_case.mesh_name]
    if bench_case.kind == "triwarp":
        vertices, faces = _split_inputs(bench_case)
        parts = bench_case.run(lambda: tw.combine.split(vertices, faces), rounds=_ROUNDS)
    elif bench_case.kind == "trimesh":
        vertices_np, faces_np = _split_inputs(bench_case)
        mesh_tm = tm.Trimesh(vertices_np, faces_np, process=False)
        parts = bench_case.run(lambda: mesh_tm.split(only_watertight=False), rounds=_ROUNDS)
    else:
        # ``tw.combine.split`` returns compact per-component ``(vertices, faces)`` submeshes, so the
        # open3d equivalent is ``cluster_connected_triangles`` (the labelling) followed by
        # ``select_by_index`` per cluster (the compaction). ``select_by_index`` takes *vertex*
        # indices, hence the ``np.unique`` over each cluster's faces -- numpy is part of what an
        # open3d user pays here, exactly as scipy is for the trimesh path.
        mesh_o3d = bench_case.mesh_o3d
        faces_i32 = np.ascontiguousarray(bench_case.faces_np, dtype=np.int32)

        def run_o3d() -> list:
            labels_np = np.asarray(mesh_o3d.cluster_connected_triangles()[0])
            return [
                mesh_o3d.select_by_index(np.unique(faces_i32[labels_np == label]))
                for label in range(int(labels_np.max()) + 1)
            ]

        parts = bench_case.run(run_o3d, rounds=_ROUNDS)
    assert len(parts) == expected


def _face_slice(bench_case: BenchCase, lo: int, hi: int) -> wp.array[wp.int32]:
    """Contiguous face-index range as a device buffer (untimed setup, so numpy is fine)."""
    return wp.array(np.arange(lo, hi, dtype=np.int32), dtype=wp.int32, device=bench_case.device)


def _submesh(bench_case: BenchCase, lo: int, hi: int) -> tuple:
    """Faces ``[lo, hi)`` of the case mesh as a compact standalone ``(vertices, faces)`` pair."""
    return tw.selection.submesh_from_face_indices(
        bench_case.vertices_wp,
        bench_case.faces_wp,
        _face_slice(bench_case, lo, hi),
        unique_indices=True,
    )


def _parts(bench_case: BenchCase, copies: int) -> list:
    """Build ``copies`` independent ``(vertices, faces)`` pairs to feed ``concatenate``."""
    key = (bench_case.mesh_name, str(bench_case.device), copies)
    if key not in _parts_cache:
        n_faces = bench_case.n_faces
        stride = max(1, n_faces // copies)
        bounds = [(lo, min(lo + stride, n_faces)) for lo in range(0, n_faces, stride)]
        _parts_cache[key] = [_submesh(bench_case, lo, hi) for lo, hi in bounds if lo < hi]
    return _parts_cache[key]


@pytest.mark.benchmark(group="concatenate")
@pytest.mark.benchlibs("triwarp", "trimesh")
@pytest.mark.benchmeshes("sphere_med")
@pytest.mark.parametrize("copies", _CONCAT_COPIES)
def test_concatenate(bench_case: BenchCase, copies: int) -> None:
    """
    The inverse of ``split``, and the same shape of cost: a host loop over the *input count*.

    Total face count is held at ``sphere_med``'s 81 920 and only the number of pieces changes, so
    any slope here is per-mesh overhead rather than data movement.
    """
    if bench_case.kind == "triwarp":
        pieces = _parts(bench_case, copies)
        vertices, faces = bench_case.run(lambda: tw.combine.concatenate(pieces))
        assert int(faces.shape[0]) // 3 == bench_case.n_faces
        assert int(vertices.shape[0]) > 0
    else:
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        stride = max(1, bench_case.n_faces // copies)
        meshes_tm = [
            tm.Trimesh(vertices_np, faces_np[lo : lo + stride], process=False)
            for lo in range(0, bench_case.n_faces, stride)
        ]
        combined = bench_case.run(lambda: tm.util.concatenate(meshes_tm))
        assert len(combined.faces) == bench_case.n_faces


def _stitch_halves(bench_case: BenchCase) -> tuple:
    """
    Cut the open tube into two rings, so each half has exactly one boundary loop to stitch.

    ``_open_cylinder`` emits its lower band of triangles before its upper one, so the first and
    second halves of the face buffer are exactly the two rings -- no adjacency query needed.
    """
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _stitch_cache:
        half = bench_case.n_faces // 2
        _stitch_cache[key] = (
            _submesh(bench_case, 0, half),
            _submesh(bench_case, half, bench_case.n_faces),
        )
    return _stitch_cache[key]


@pytest.mark.benchmark(group="stitch")
@pytest.mark.benchmeshes("rim_short")
@pytest.mark.benchlibs("triwarp")
def test_stitch(bench_case: BenchCase) -> None:
    """Greedy band between two rims: O(La + Lb), the cheap counterpart of the DP below."""
    (va, fa), (vb, fb) = _stitch_halves(bench_case)
    _vertices, faces = bench_case.run(lambda: tw.combine.stitch(va, fa, vb, fb))
    assert int(faces.shape[0]) > 0


@pytest.mark.benchmark(group="stitch_min_weight")
@pytest.mark.benchmeshes("rim_short")
@pytest.mark.benchlibs("triwarp")
def test_stitch_min_weight(bench_case: BenchCase) -> None:
    """Grid DP over the two rims: an La x Lb table filled by La + Lb sequential launches."""
    (va, fa), (vb, fb) = _stitch_halves(bench_case)
    up = wp.vec3(0.0, 0.0, 1.0)
    _vertices, faces = bench_case.run(
        lambda: tw.combine.stitch_min_weight(va, fa, vb, fb, up_dir=up), rounds=_ROUNDS
    )
    assert int(faces.shape[0]) > 0
