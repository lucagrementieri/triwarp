"""
Benchmarks for ``triwarp.combine``: assembling meshes from parts and splitting them apart.

Axis: **components** for ``split`` and ``concatenate``, **loops_dp** for the stitching pair.
Neither family scales with face count, and this pair is the sharpest example in the package of
functions whose cost lives entirely somewhere else -- in the number of pieces going in or coming
out.

``split`` is one labelling pass plus one stable radix sort -- both O(F) and both fast -- followed
by a **batched compaction of every component at once**. Measured at a fixed 81 920 faces:

| components | triwarp-cuda | trimesh | open3d |
|---|---|---|---|
| 1 | **2.5 ms** | 36.6 ms | 68.1 ms |
| 64 | 3.1 ms | 37.1 ms | 76.4 ms |
| 1024 | **9.4 ms** | 183 ms | 290 ms |

3.7x across the axis and a win at every point (19-31x). It did not start that way: until commit
``f57d3f0`` the compaction was a **host loop calling ``submesh_from_face_indices`` once per
component**, which measured 2.55 / 41.4 / 669 ms -- 262x across the axis, and a 3x *loss* to
trimesh at a thousand components. This group is what surfaced that, and no face-count sweep would
have: the scan registry's meshes happen to differ in component count by accident
(``bunny_decimated`` has 94 scan floaters, ``bunny`` has 1), which is how the effect was originally
noticed at all. The residual slope is what is left of the per-component cost.

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

**pymeshlab**'s ``generate_splitting_by_connected_components`` is the third, and the cheapest to
state: one filter call does both halves and *pushes one new mesh per component* onto the MeshSet, so
the component count is read straight off ``mesh_number()``. It is also the group's sharpest
reference, because it has the same per-component host cost triwarp used to have -- measured **42 /
133 / 1 630 ms** across the axis, a **39x spread** against triwarp's 3.7x. That is the shape the
batched compaction removed, reproduced independently.

Neither trimesh, open3d nor pymeshlab has an equivalent of ``stitch`` / ``stitch_min_weight``:
joining two open meshes along their boundary loops with a minimum-weight triangulation is not in any
of the three APIs (trimesh's ``util.concatenate`` merges without stitching, open3d's boolean
operations need closed inputs, and MeshLab's ``meshing_snap_mismatched_borders`` snaps *coincident*
borders together rather than triangulating a gap between them). Those two groups are before/after
self-comparisons.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh as tm
import warp as wp
from conftest import BenchCase, mesh_ml_from_numpy
from meshlib import mrmeshpy as mm

import triwarp as tw

# Copies for the concatenate sweep: the function still issues one packing copy per input buffer, so
# the input *count* is the driver and the total face count is held roughly fixed between the points.
_CONCAT_COPIES = [8, 512]

# The min-weight stitch DP and split on a thousand components run to tens of milliseconds a call.
_ROUNDS = 3

_split_cache: dict[tuple[str, str], tuple] = {}
_parts_cache: dict[tuple[str, str, int], list] = {}
_stitch_cache: dict[tuple[str, str], tuple] = {}
_stitch_np_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}


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
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d", "pymeshlab")
def test_split(bench_case: BenchCase) -> None:
    """Label, sort, then one batched compaction of every component: 3.7x across the axis."""
    expected = {"sphere_med": 1, "parts_64": 64, "parts_1024": 1024}[bench_case.mesh_name]
    if bench_case.kind == "pymeshlab":
        # One filter, but it *pushes* one new mesh per component onto the MeshSet, so it mutates the
        # set and must be rebuilt per round. The mesh count is asserted on, which is the same
        # component-count check the other three branches make.
        def split_pml() -> int:
            meshset_pml = bench_case.new_meshset_pml()
            meshset_pml.generate_splitting_by_connected_components()
            return meshset_pml.mesh_number() - 1

        assert bench_case.run(split_pml, rounds=_ROUNDS) == expected
        return
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


_parts_ml_cache: dict[tuple[str, int], mm.std_vector_std_shared_ptr_Mesh] = {}


def _parts_ml(bench_case: BenchCase, copies: int) -> mm.std_vector_std_shared_ptr_Mesh:
    """Build the same pieces as a MeshLib mesh vector, cached: the input, not the operation."""
    key = (bench_case.mesh_name, copies)
    if key not in _parts_ml_cache:
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        n_faces = bench_case.n_faces
        stride = max(1, n_faces // copies)
        meshes_ml = mm.std_vector_std_shared_ptr_Mesh()
        for lo in range(0, n_faces, stride):
            block_np = faces_np[lo : min(lo + stride, n_faces)]
            if block_np.shape[0]:
                meshes_ml.append(mesh_ml_from_numpy(vertices_np, block_np))
        _parts_ml_cache[key] = meshes_ml
    return _parts_ml_cache[key]


@pytest.mark.benchmark(group="concatenate")
@pytest.mark.benchlibs("triwarp", "trimesh", "meshlib")
@pytest.mark.benchmeshes("sphere_med")
@pytest.mark.parametrize("copies", _CONCAT_COPIES)
def test_concatenate(bench_case: BenchCase, copies: int) -> None:
    """
    The inverse of ``split``, and the same shape of cost: work proportional to the *input count*.

    Total face count is held at ``sphere_med``'s 81 920 and only the number of pieces changes, so
    any slope here is per-mesh overhead rather than data movement. Measured **0.23 ms at 8 pieces
    and 5.7 ms at 512**, down from 0.39 / 19.0 ms: the index renumbering used to be one ``wp.map``
    per input mesh (~32 us of host-side marshalling each) and is now a single launch over the
    packed buffer. The 25x that remains is two ``wp.copy`` calls per piece -- Warp has no gather
    across separate allocations, so the packing itself cannot be batched.
    """
    if bench_case.kind == "meshlib":
        # ``mergeMeshes`` takes a vector of *shared pointers* and returns a new mesh without
        # touching its inputs, so unlike almost everything else in this library the pieces are
        # cached: they are the input, and rebuilding them per round would time the converter.
        pieces_ml = _parts_ml(bench_case, copies)
        merged_ml = bench_case.run(lambda: mm.mergeMeshes(pieces_ml))
        assert merged_ml.topology.numValidFaces() == bench_case.n_faces
        return
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


def _stitch_pair_np(bench_case: BenchCase) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the same two rings as [`_stitch_halves`] as one NumPy mesh with two disjoint rims.

    MeshLib's ``stitchHoles`` takes a *single* ``Mesh`` holding both holes, so the two halves are
    compacted independently and index-offset into one buffer -- the same construction
    ``tests/test_combine.py::_meshlib_stitch_band`` uses, which is what makes the benchmark and the
    parity test measure the same thing.
    """
    if bench_case.mesh_name not in _stitch_np_cache:
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        half = bench_case.n_faces // 2
        parts = []
        for lo, hi in ((0, half), (half, bench_case.n_faces)):
            used, inverse = np.unique(faces_np[lo:hi], return_inverse=True)
            parts.append((vertices_np[used], inverse.reshape(-1, 3)))
        (va_np, fa_np), (vb_np, fb_np) = parts
        _stitch_np_cache[bench_case.mesh_name] = (
            np.vstack([va_np, vb_np]),
            np.vstack([fa_np, fb_np + len(va_np)]),
        )
    return _stitch_np_cache[bench_case.mesh_name]


@pytest.mark.benchmark(group="stitch_min_weight")
@pytest.mark.benchmeshes("rim_short")
@pytest.mark.benchlibs("triwarp", "meshlib")
def test_stitch_min_weight(bench_case: BenchCase) -> None:
    """
    Grid DP over the two rims: an La x Lb table filled by La + Lb sequential launches.

    meshlib is the only reference in the package that has this operation at all -- ``stitchHoles``
    is a real two-loop minimum-weight stitch where trimesh and pymeshlab have nothing, which is why
    it is also the oracle in tests/test_combine.py::test_stitch_min_weight_matches_meshlib. The
    four-argument overload is used deliberately (see section 6): the two-argument one finds the
    rims itself, and timing that would fold hole detection into the DP.
    """
    if bench_case.kind == "meshlib":
        vertices_np, faces_np = _stitch_pair_np(bench_case)

        def run_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(vertices_np, faces_np)
            edges_ml = mesh_ml.topology.findHoleRepresentiveEdges()
            params_ml = mm.StitchHolesParams()
            params_ml.metric = mm.getComplexStitchMetric(mesh_ml)
            mm.stitchHoles(mesh_ml, edges_ml[0], edges_ml[1], params_ml)
            return mesh_ml.topology.numValidFaces()

        assert bench_case.run(run_ml, rounds=_ROUNDS) > len(faces_np)
        return
    (va, fa), (vb, fb) = _stitch_halves(bench_case)
    up = wp.vec3(0.0, 0.0, 1.0)
    _vertices, faces = bench_case.run(
        lambda: tw.combine.stitch_min_weight(va, fa, vb, fb, up_dir=up), rounds=_ROUNDS
    )
    assert int(faces.shape[0]) > 0
