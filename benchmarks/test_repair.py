"""
Benchmarks for ``triwarp.repair``.

Axis: **defect count** for three of the four groups, and **diameter** for the fourth. Repair
functions are the clearest case in the package of cost that a face count cannot predict -- they
are all "find the broken things and fix them", so the number of broken things is the driver and
the mesh they are embedded in is nearly irrelevant. Every defect group therefore runs on
``sphere_med`` and sweeps the *injected defect count*, holding the mesh fixed:

* ``resolve_duplicated_faces`` -- 0% against 10% cancelling flipped pairs.
* ``remove_non_manifold_faces`` -- 0 against 1 024 extra faces, each one making its three edges
  3-incident. This one also loops: removing faces can create *new* non-manifold edges, so a
  pathological input uses all ``max_iter`` rounds where a clean one exits after the first test.
* ``remove_duplicated_vertices`` -- run on an *unwelded* soup (every face owning its own three
  vertices, the state a freshly loaded STL is in), sweeping ``epsilon``. The two values are two
  code paths rather than two thresholds: positive ``epsilon`` snaps coordinates to
  ``round(v / epsilon)``, while ``0`` buckets by the high bits of the float32 representation --
  a *relative* cell about 2.4e-4 wide, not an equality test.

``make_winding_consistent`` is the exception and gets the **diameter** axis instead: its cost is a
property of the face-adjacency graph, not of the defect count. It used to *propagate* the
orientation bits one level per round and measured **7.4 ms on ``sphere_med`` against 642 ms on
``ribbon_long``** at an identical vertex count -- 87x, while trimesh's equivalent went the other
way, 16.8 ms to 7.8 ms. It now measures **1.0 ms and 0.85 ms**: flat, and faster than trimesh at
both ends. Same finding, and same fix, as ``face_orientation_bits`` in
[`test_validation.py`](test_validation.py), which is the machinery underneath it.

pymeshlab settles what that axis was really measuring: its serial face-to-face visit reads
**53.9 ms on ``sphere_med`` against 33.1 ms on ``ribbon_long``** -- also *faster* on the
high-diameter mesh, the same direction trimesh goes. So graph diameter is not intrinsically
expensive for this problem; it was expensive only for the level-propagating formulation triwarp
used to have.

References
----------
**open3d**'s ``remove_duplicated_triangles`` solves the same "deduplicate a face array" problem
with a hash set over index triples, against triwarp's sort-based grouping. It is a comparison of
dedup *machinery*, not of results -- the semantics differ twice over:

1. open3d keeps one representative of each duplicate group, while triwarp applies a signed-count
   rule that drops cancelling ``(+1, -1)`` pairs outright;
2. open3d's hash is **orientation-sensitive** (measured: it collapses ``[0,1,2]`` against
   ``[0,1,2]`` but not against ``[2,1,0]``), so on this deliberately-flipped input it removes
   nothing and returns the face count unchanged. The hash pass over all ``n`` triples still runs,
   which is the cost being compared; the assertion below only checks the count did not grow.

Open3D mutates in place and the operation is idempotent, so its mesh is rebuilt inside the timed
callable (rounds 2..n would otherwise dedup an already-deduped mesh). ``remove_duplicated_vertices``
is its counterpart for the vertex group. **trimesh**'s ``repair.fix_winding`` is the orientation
reference; it has no non-manifold face removal.

**pymeshlab** is the only library in the set that covers *all four* groups, and it is the first
reference of any kind for ``remove_non_manifold_faces`` -- neither trimesh nor open3d nor libigl
removes non-manifold faces at all:

* ``remove_non_manifold_faces`` -> ``meshing_repair_non_manifold_edges(method='Remove Faces')``.
  The same idea, greedier: for each non-manifold edge MeshLab iteratively deletes the
  *smallest-area* incident face until the edge is 2-manifold, where triwarp drops every face on an
  over-incident edge and re-tests.
* ``resolve_duplicated_faces`` -> ``meshing_remove_duplicate_faces``. Orientation-*insensitive*
  (same vertex set, any order), so unlike open3d's hash it does see the flipped copies -- but it
  keeps one representative rather than cancelling ``(+1, -1)`` pairs.
* ``remove_duplicated_vertices`` -> ``meshing_remove_duplicate_vertices`` and
  ``meshing_merge_close_vertices(threshold=...)``: exactly triwarp's two code paths, exact
  coordinate equality and a tolerance. So this is the one group where the ``epsilon`` sweep maps
  across libraries one-for-one.
* ``make_winding_consistent`` -> ``meshing_re_orient_faces_coherently``. A serial face-to-face
  visit against triwarp's parity union-find, precisely the contrast the ``diameter`` axis exposes.

Every one of them rewrites the topology, so the MeshSet is built inside the timed callable from the
*defect-injected* arrays rather than from the registry mesh, and each row carries that build. It is
a large share at this size: 16.4 ms of the 17.8 ms a clean ``remove_duplicate_faces`` costs, and
38.8 of the 134 ms the unwelded soup's dedup costs (the soup carries 245 760 vertices). Subtract it
before quoting a ratio.
"""

from __future__ import annotations

import numpy as np
import pymeshlab as ml
import pytest
import trimesh as tm
import warp as wp
from conftest import BenchCase

import triwarp as tw

_DUP_SEED = 7

# Fraction of faces re-appended as flipped copies: nothing to do, against a tenth of the mesh.
_DUPLICATE_FRACTIONS = [0.0, 0.1]

# Extra same-orientation face copies, each making its three edges 3-incident.
_NON_MANIFOLD_COUNTS = [0, 1_024]

# Vertex merge tolerances: two different code paths, not two thresholds (see the module docstring).
_MERGE_EPSILONS = [0.0, 1e-6]

_defect_np_cache: dict[tuple[str, float], np.ndarray] = {}
_defect_wp_cache: dict[tuple[str, str, float], wp.array] = {}
_nonmanifold_cache: dict[tuple[str, str, int], wp.array] = {}
_soup_cache: dict[tuple[str, str], tuple] = {}


def _new_meshset_pml(vertices_np: np.ndarray, faces_np: np.ndarray) -> ml.MeshSet:
    """
    Build a fresh MeshSet over *defect-injected* arrays, inside the timed callable.

    ``BenchCase.new_meshset_pml`` builds the registry mesh, which is exactly the mesh these groups
    are not measuring: every one of them needs the duplicated / non-manifold / unwelded variant.
    """
    meshset_pml = ml.MeshSet()
    meshset_pml.add_mesh(
        ml.Mesh(
            np.ascontiguousarray(vertices_np, dtype=np.float64),
            np.ascontiguousarray(faces_np, dtype=np.int32),
        )
    )
    return meshset_pml


def _clean_faces_np(bench_case: BenchCase) -> np.ndarray:
    """Return the mesh faces with any pre-existing duplicate or degenerate triangle removed."""
    faces = bench_case.faces_np
    sorted_rows = np.sort(faces, axis=1)
    _, inverse, counts = np.unique(sorted_rows, axis=0, return_inverse=True, return_counts=True)
    base = faces[counts[inverse] == 1]
    nondegenerate = (
        (base[:, 0] != base[:, 1]) & (base[:, 1] != base[:, 2]) & (base[:, 0] != base[:, 2])
    )
    return np.ascontiguousarray(base[nondegenerate], dtype=np.int32)


def _faces_with_duplicates_np(bench_case: BenchCase, fraction: float) -> np.ndarray:
    """Append a fixed-seed subset of the faces back as *flipped* copies: cancelling pairs."""
    key = (bench_case.mesh_name, fraction)
    if key not in _defect_np_cache:
        base = _clean_faces_np(bench_case)
        if fraction <= 0.0:
            _defect_np_cache[key] = base
        else:
            rng = np.random.default_rng(_DUP_SEED)
            n_dup = max(1, int(base.shape[0] * fraction))
            idx = rng.choice(base.shape[0], size=n_dup, replace=False)
            _defect_np_cache[key] = np.ascontiguousarray(
                np.vstack((base, base[idx][:, ::-1])), dtype=np.int32
            )
    return _defect_np_cache[key]


def _faces_with_duplicates_wp(bench_case: BenchCase, fraction: float) -> wp.array[wp.int32]:
    key = (bench_case.mesh_name, str(bench_case.device), fraction)
    if key not in _defect_wp_cache:
        combined = _faces_with_duplicates_np(bench_case, fraction).reshape(-1)
        _defect_wp_cache[key] = wp.array(
            np.ascontiguousarray(combined), dtype=wp.int32, device=bench_case.device
        )
    return _defect_wp_cache[key]


@pytest.mark.benchmark(group="resolve_duplicated_faces")
@pytest.mark.benchmeshes("sphere_med")
@pytest.mark.benchlibs("triwarp", "open3d", "pymeshlab")
@pytest.mark.parametrize("fraction", _DUPLICATE_FRACTIONS, ids=["clean", "dup10pct"])
def test_resolve_duplicated_faces(bench_case: BenchCase, fraction: float) -> None:
    """Sort-based grouping with a signed-count rule: nothing to do, against a tenth of the mesh."""
    if bench_case.kind == "pymeshlab":
        # Orientation-insensitive, so unlike open3d's hash it does see the flipped copies.
        vertices_np = bench_case.vertices_np
        faces_dup_np = _faces_with_duplicates_np(bench_case, fraction)
        bench_case.run(
            lambda: _new_meshset_pml(vertices_np, faces_dup_np).meshing_remove_duplicate_faces()
        )
        return
    if bench_case.kind == "triwarp":
        faces_dup = _faces_with_duplicates_wp(bench_case, fraction)
        resolved, kept = bench_case.run(lambda: tw.repair.resolve_duplicated_faces(faces_dup))
        assert resolved.shape[0] == kept.shape[0] * 3
        assert kept.shape[0] > 0
    else:
        # open3d dedups in place and is idempotent: build the mesh inside the timed callable.
        import open3d as o3d

        vertices = o3d.utility.Vector3dVector(bench_case.vertices_np)
        faces_dup_np = _faces_with_duplicates_np(bench_case, fraction)

        def run() -> o3d.geometry.TriangleMesh:
            mesh = o3d.geometry.TriangleMesh(vertices, o3d.utility.Vector3iVector(faces_dup_np))
            return mesh.remove_duplicated_triangles()

        deduped = bench_case.run(run)
        assert 0 < len(deduped.triangles) <= faces_dup_np.shape[0]


def _faces_with_non_manifold_np(bench_case: BenchCase, extra: int) -> np.ndarray:
    """Return ``(n, 3)`` faces with ``extra`` same-orientation copies appended: 3-incident edges."""
    base = _clean_faces_np(bench_case)
    if extra > 0:
        rng = np.random.default_rng(_DUP_SEED)
        idx = rng.choice(base.shape[0], size=min(extra, base.shape[0]), replace=False)
        base = np.vstack((base, base[idx]))
    return np.ascontiguousarray(base, dtype=np.int32)


def _faces_with_non_manifold_wp(bench_case: BenchCase, extra: int) -> wp.array[wp.int32]:
    """Append ``extra`` same-orientation face copies, so their edges become 3-incident."""
    key = (bench_case.mesh_name, str(bench_case.device), extra)
    if key not in _nonmanifold_cache:
        _nonmanifold_cache[key] = wp.array(
            np.ascontiguousarray(
                _faces_with_non_manifold_np(bench_case, extra).reshape(-1), dtype=np.int32
            ),
            dtype=wp.int32,
            device=bench_case.device,
        )
    return _nonmanifold_cache[key]


@pytest.mark.benchmark(group="remove_non_manifold_faces")
@pytest.mark.benchmeshes("sphere_med")
@pytest.mark.benchlibs("triwarp", "pymeshlab")
@pytest.mark.parametrize("extra", _NON_MANIFOLD_COUNTS, ids=["clean", "nm1024"])
def test_remove_non_manifold_faces(bench_case: BenchCase, extra: int) -> None:
    """Iterated edge-sort and manifold test: a clean mesh exits in one round, a broken one loops."""
    if bench_case.kind == "pymeshlab":
        # MeshLab deletes the smallest-area incident face per non-manifold edge until the edge is
        # 2-manifold; triwarp drops every face on an over-incident edge and re-tests. Same goal,
        # a greedier rule, and the first reference this group has had.
        vertices_np = bench_case.vertices_np
        faces_nm_np = _faces_with_non_manifold_np(bench_case, extra)
        bench_case.run(
            lambda: _new_meshset_pml(vertices_np, faces_nm_np).meshing_repair_non_manifold_edges(
                method="Remove Faces"
            )
        )
        return
    vertices = bench_case.vertices_wp
    faces = _faces_with_non_manifold_wp(bench_case, extra)
    _kept_vertices, kept_faces = bench_case.run(
        lambda: tw.repair.remove_non_manifold_faces(vertices, faces)
    )
    assert int(kept_faces.shape[0]) > 0


def _soup(bench_case: BenchCase) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """Unweld the mesh so every face owns its three vertices: maximal duplication to collapse."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _soup_cache:
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        soup_np = vertices_np[faces_np].reshape(-1, 3)
        _soup_cache[key] = (
            wp.array(
                np.ascontiguousarray(soup_np, dtype=np.float32),
                dtype=wp.vec3,
                device=bench_case.device,
            ),
            wp.array(
                np.arange(soup_np.shape[0], dtype=np.int32),
                dtype=wp.int32,
                device=bench_case.device,
            ),
        )
    return _soup_cache[key]


@pytest.mark.benchmark(group="remove_duplicated_vertices")
@pytest.mark.benchmeshes("sphere_med")
@pytest.mark.benchlibs("triwarp", "open3d", "pymeshlab")
@pytest.mark.parametrize("epsilon", _MERGE_EPSILONS, ids=["exact", "eps1e-6"])
def test_remove_duplicated_vertices(bench_case: BenchCase, epsilon: float) -> None:
    """Weld an unwelded soup: 245 760 positions down to 40 962, by two different bucketings."""
    if bench_case.kind == "pymeshlab":
        # The one group whose epsilon sweep maps one-for-one: MeshLab has a filter per path.
        soup_np = np.ascontiguousarray(bench_case.vertices_np[bench_case.faces_np].reshape(-1, 3))
        faces_np = np.ascontiguousarray(np.arange(soup_np.shape[0], dtype=np.int32).reshape(-1, 3))

        def weld_pml() -> None:
            meshset_pml = _new_meshset_pml(soup_np, faces_np)
            if epsilon > 0.0:
                meshset_pml.meshing_merge_close_vertices(threshold=ml.PureValue(epsilon))
            else:
                meshset_pml.meshing_remove_duplicate_vertices()

        bench_case.run(weld_pml)
        return
    if bench_case.kind == "triwarp":
        vertices, faces = _soup(bench_case)
        unique_vertices, _unique_indices, _inverse, _faces = bench_case.run(
            lambda: tw.repair.remove_duplicated_vertices(vertices, faces, epsilon)
        )
        assert int(unique_vertices.shape[0]) <= int(vertices.shape[0])
    else:
        import open3d as o3d

        soup_np = np.ascontiguousarray(bench_case.vertices_np[bench_case.faces_np].reshape(-1, 3))
        faces_np = np.ascontiguousarray(np.arange(soup_np.shape[0], dtype=np.int32).reshape(-1, 3))

        def run() -> o3d.geometry.TriangleMesh:
            mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(soup_np), o3d.utility.Vector3iVector(faces_np)
            )
            return mesh.remove_duplicated_vertices()

        welded = bench_case.run(run)
        assert 0 < len(welded.vertices) <= soup_np.shape[0]


@pytest.mark.benchmark(group="make_winding_consistent")
@pytest.mark.benchaxis("diameter")
@pytest.mark.benchlibs("triwarp", "trimesh", "pymeshlab")
def test_make_winding_consistent(bench_case: BenchCase) -> None:
    """Flip mask from the parity union-find, then one relabel pass: flat across the axis."""
    if bench_case.kind == "pymeshlab":  # a serial face-to-face visit, against the union-find
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        bench_case.run(
            lambda: _new_meshset_pml(vertices_np, faces_np).meshing_re_orient_faces_coherently()
        )
        return
    if bench_case.kind == "triwarp":
        faces = bench_case.faces_wp
        oriented = bench_case.run(lambda: tw.repair.make_winding_consistent(faces))
        assert int(oriented.shape[0]) == faces.shape[0]
    else:  # trimesh mutates in place: rebuild inside the timed callable
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np

        def fix_winding_tm() -> tm.Trimesh:
            mesh_tm = tm.Trimesh(vertices_np, faces_np, process=False)
            tm.repair.fix_winding(mesh_tm)
            return mesh_tm

        assert len(bench_case.run(fix_winding_tm).faces) == bench_case.n_faces
