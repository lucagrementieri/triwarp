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

The two *geometric* defect groups sit on the **quality** axis instead of a defect sweep, because
their defects are not injectable: ``bad_face_mask`` and ``remove_t_vertices`` look for thin and
folded triangles, and ``saddle_graded`` already has them by construction (worst aspect ratio 4 719
against ``saddle``'s 1.6). ``bad_face_mask`` is a fixed number of passes over the adjacency whatever
it finds, so it should be flat across that axis; ``remove_t_vertices`` is a flip loop and should
*not* be, since only the graded mesh gives it work to do. That contrast is the point of the pair.

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

import igl
import numpy as np
import pymeshlab as ml
import pytest
import pyvista as pv
import trimesh as tm
import warp as wp
from meshlib import mrmeshpy as mm

import triwarp as tw
from conftest import BenchCase, mesh_ml_from_numpy, skip_larger_than

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


@pytest.mark.noparity(
    "open3d",
    reason="D2 three different dedup rules: triwarp applies a signed-count rule that cancels "
    "(+1, -1) pairs outright, while open3d hashes orientation-*sensitively* and keeps one "
    "representative of each duplicate group -- on this deliberately-flipped input it therefore "
    "removes nothing and returns the face count unchanged. The rows compare dedup machinery, not "
    "results.",
)
@pytest.mark.noparity(
    "pymeshlab",
    reason="D2 as above but the other way round: MeshLab's hash is orientation-*insensitive*, so "
    "it does see the flipped copies, yet it still keeps one representative per group where "
    "triwarp cancels a (+1, -1) pair to nothing. Same input, three different survivor sets by "
    "design; the numpy oracle in tests/test_repair.py covers the signed-count rule itself.",
)
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


@pytest.mark.noparity(
    "pymeshlab",
    reason="D2 the same idea, greedier: for each non-manifold edge MeshLab iteratively deletes the "
    "smallest-area incident face until that edge is 2-manifold, where triwarp drops every face on "
    "an over-incident edge and re-tests. Both leave an edge-manifold mesh but they delete "
    "different faces and different numbers of them, so only the post-condition is shared and "
    "tests/test_repair.py asserts that directly rather than through MeshLab.",
)
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


@pytest.mark.benchmark(group="split_nonmanifold")
@pytest.mark.benchmeshes("sphere_med")
@pytest.mark.benchlibs("triwarp", "igl", "meshlib")
@pytest.mark.parametrize("extra", _NON_MANIFOLD_COUNTS, ids=["clean", "nm1024"])
def test_split_nonmanifold(bench_case: BenchCase, extra: int) -> None:
    """
    Vertex duplication instead of face deletion: the non-lossy repair for the same defect.

    Read against ``remove_non_manifold_faces`` above, which is timed on the identical input: the
    two reach an edge-manifold mesh from opposite directions, one by dropping every face on an
    over-incident edge and the other by splitting vertices apart and keeping all of them. The
    ``clean`` case is the floor -- both must detect that there is nothing to do -- and the
    ``nm1024`` case is the work.

    triwarp's cost is an edge sort plus a connected-components pass over ``3 * n_faces`` corner
    nodes, so it barely moves between the two cases. ``igl.split_nonmanifold`` is sequential by
    construction -- it explodes the mesh to ``3 * n_faces`` singleton vertices and greedily re-
    merges pairs, re-testing manifoldness after each candidate, with the source calling its own
    inner check "Omega(m) and probably O(m log m) or worse" -- and it does move: measured
    standalone at 84 ms clean against 122 ms with the duplicates on an 81 920-face sphere. It
    takes ``rounds=3``.

    The two libraries agree on the split exactly for a bowtie vertex, a same-wound fan of three
    faces on one edge, a flipped face and a boundary, but **not on this group's defect**: for a
    duplicated face igl keeps one arbitrarily chosen pair joined where triwarp splits all copies
    (18 vertices against 15 on an icosahedron with one face duplicated). Both outputs are
    manifold with every face kept; the rows are a cost comparison, and ``tests/test_repair.py``
    carries both the agreement and the divergence.
    """
    if bench_case.kind == "meshlib":
        # The build is inside the timed callable and that is not a converter tax: MeshLib's
        # half-edge topology cannot represent a non-edge-manifold mesh at all, so
        # ``meshFromFacesVerts`` does this group's split while *constructing* -- on the injected
        # defect, which is edge-incidence, ``duplicateMultiHoleVertices`` then reports 0 and the
        # only honest row is both calls together. The ``tests/test_repair.py`` pair shows the two
        # routes reaching the same vertex count.
        vertices_np = bench_case.vertices_np
        faces_nm_np = _faces_with_non_manifold_np(bench_case, extra)

        def split_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(vertices_np, faces_nm_np)
            mm.duplicateMultiHoleVertices(mesh_ml)
            return mesh_ml.topology.numValidVerts()

        assert bench_case.run(split_ml, rounds=3) >= bench_case.n_vertices
        return
    if bench_case.kind == "igl":
        faces_nm_np = np.ascontiguousarray(
            _faces_with_non_manifold_np(bench_case, extra), dtype=np.int64
        )
        faces_igl, source_igl = bench_case.run(lambda: igl.split_nonmanifold(faces_nm_np), rounds=3)
        assert faces_igl.shape[0] == faces_nm_np.shape[0]
        assert source_igl.shape[0] >= bench_case.n_vertices
        return
    vertices = bench_case.vertices_wp
    faces = _faces_with_non_manifold_wp(bench_case, extra)
    split_vertices, split_faces, _source = bench_case.run(
        lambda: tw.repair.split_nonmanifold(vertices, faces)
    )
    assert int(split_faces.shape[0]) == int(faces.shape[0])
    assert int(split_vertices.shape[0]) >= bench_case.n_vertices


@pytest.mark.benchmark(group="remove_unreferenced_vertices")
@pytest.mark.benchlibs("triwarp", "igl", "open3d", "meshlib")
@pytest.mark.parametrize("unreferenced", [0, 1], ids=["clean", "padded"])
def test_remove_unreferenced_vertices(bench_case: BenchCase, unreferenced: int) -> None:
    """
    Compact away vertices no face indexes: a mask, a scan and a gather.

    The ``padded`` id appends a *copy* of the vertex buffer that no face references, so half the
    vertices are dead and the compaction has real work to do; ``clean`` is the identity case, where
    the whole cost is the mask-and-scan that proves there is nothing to remove. Both are on the scan
    sweep because the operation has no other axis: it is one pass over ``V`` plus one over ``3F``.

    ``igl.remove_unreferenced`` returns the same four things in the same order (vertices, faces,
    forward map, inverse map) and is already the oracle in ``tests/test_repair.py``. Note it takes
    ``int32`` faces here while most of the package wants ``int64`` -- passing ``faces_np`` directly
    works because nanobind casts, but the cast is a copy of the face buffer on every call and it is
    inside the timing.

    ``bunny`` itself carries **1 113 unreferenced vertices** of 35 947, so the ``clean`` id is only
    clean in the sense of "nothing added": all the libraries really do drop those.

    open3d's ``remove_unreferenced_vertices`` mutates its mesh in place, so the legacy container is
    rebuilt inside the timed callable (the ``test_repair.py`` rule) -- its row prices the pybind
    ``Vector3dVector`` copy along with the compaction, which is what a caller holding NumPy buffers
    pays. It agrees with triwarp element-wise on both compacted buffers
    (``tests/test_repair.py::test_remove_unreferenced_matches_open3d``).
    """
    vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
    if unreferenced:
        vertices_np = np.ascontiguousarray(np.vstack([vertices_np, vertices_np]))
    if bench_case.kind == "meshlib":
        # ``pack()`` renumbers in place, so the mesh is built inside the timed callable and this
        # row prices the build with it. The ``padded`` id costs meshlib nothing extra either way:
        # the duplicated block is *trailing*, and ``meshFromFacesVerts`` sizes its point buffer by
        # ``F.max() + 1``, so those vertices never reach the mesh. What it does price is the case
        # the other three rows cannot show -- bunny's 1 113 *interior* unreferenced vertices, which
        # arrive as invalid entries and are what pack() compacts away.
        def pack_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(vertices_np, faces_np)
            mesh_ml.pack()
            return mesh_ml.topology.numValidVerts()

        # Not bounded above by the input count: on a non-edge-manifold mesh the builder splits
        # vertices apart before pack() ever runs (8 360 from bunny_decimated's 8 171).
        assert bench_case.run(pack_ml) > 0
        return
    if bench_case.kind == "open3d":
        skip_larger_than(bench_case, "bunny", "the container rebuild dominates past bunny")
        import open3d as o3d

        faces_i32 = np.ascontiguousarray(faces_np, dtype=np.int32)

        def remove_unreferenced_o3d() -> int:
            mesh_o3d = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(vertices_np), o3d.utility.Vector3iVector(faces_i32)
            )
            mesh_o3d.remove_unreferenced_vertices()
            return len(mesh_o3d.vertices)

        assert bench_case.run(remove_unreferenced_o3d) <= vertices_np.shape[0]
        return
    if bench_case.kind == "igl":
        skip_larger_than(bench_case, "bunny", "the reference is a single-threaded scan and gather")
        faces_igl = np.ascontiguousarray(faces_np, dtype=np.int32)
        kept_vertices_igl, kept_faces_igl, _remap_igl, _inverse_igl = bench_case.run(
            lambda: igl.remove_unreferenced(vertices_np, faces_igl)
        )
        assert kept_faces_igl.shape == (bench_case.n_faces, 3)
        assert kept_vertices_igl.shape[0] <= vertices_np.shape[0]
        return
    vertices_wp = wp.array(
        np.ascontiguousarray(vertices_np, dtype=np.float32), dtype=wp.vec3, device=bench_case.device
    )
    faces_wp = bench_case.faces_wp
    kept_vertices, kept_faces, _remap = bench_case.run(
        lambda: tw.repair.remove_unreferenced_vertices(vertices_wp, faces_wp)
    )
    assert int(kept_faces.shape[0]) == int(faces_wp.shape[0])
    assert int(kept_vertices.shape[0]) <= vertices_np.shape[0]


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
@pytest.mark.benchlibs("triwarp", "open3d", "pymeshlab", "pyvista", "meshlib")
@pytest.mark.parametrize("epsilon", _MERGE_EPSILONS, ids=["exact", "eps1e-6"])
def test_remove_duplicated_vertices(bench_case: BenchCase, epsilon: float) -> None:
    """
    Weld an unwelded soup: 245 760 positions down to 40 962, by two different bucketings.

    VTK's ``clean`` takes both paths through one call -- ``tolerance`` with ``absolute=True`` is the
    epsilon form and ``0.0`` the exact one -- and it additionally drops degenerate cells and unused
    points, so its row does a little more than the weld. The soup is rebuilt inside the timed
    callable because ``clean`` returns a new mesh from an input this benchmark does not otherwise
    hold as a ``PolyData``.
    """
    if bench_case.kind == "meshlib":
        # ``uniteCloseVertices`` welds in place and returns the merge count, so the soup mesh is
        # rebuilt inside the timed callable. ``uniteOnlyBd=False`` is the setting that matches
        # triwarp; MeshLib's default of ``True`` would weld only boundary vertices. Its single
        # parameter is a distance, so the exact id is that distance at zero rather than a second
        # code path -- unlike pymeshlab and VTK, whose two ids are two different calls.
        soup_np = np.ascontiguousarray(bench_case.vertices_np[bench_case.faces_np].reshape(-1, 3))
        faces_np = np.ascontiguousarray(np.arange(soup_np.shape[0], dtype=np.int32).reshape(-1, 3))

        def weld_ml() -> int:
            return mm.uniteCloseVertices(mesh_ml_from_numpy(soup_np, faces_np), epsilon, False)

        assert 0 <= bench_case.run(weld_ml) < soup_np.shape[0]
        return
    if bench_case.kind == "pyvista":
        soup_np = np.ascontiguousarray(bench_case.vertices_np[bench_case.faces_np].reshape(-1, 3))
        faces_np = np.ascontiguousarray(np.arange(soup_np.shape[0], dtype=np.int32).reshape(-1, 3))
        soup_pv = pv.PolyData.from_regular_faces(soup_np, faces_np)
        welded_pv = bench_case.run(
            lambda: soup_pv.clean(point_merging=True, tolerance=epsilon, absolute=True)
        )
        assert 0 < welded_pv.n_points <= soup_np.shape[0]
        return
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
@pytest.mark.benchlibs("triwarp", "trimesh", "igl", "pymeshlab", "meshlib")
def test_make_winding_consistent(bench_case: BenchCase) -> None:
    """
    Flip mask from the parity union-find, then one relabel pass: flat across the axis.

    Three references, three traversals, and this is the group where the ``diameter`` axis earns its
    keep: ``igl.bfs_orient`` is a breadth-first walk, MeshLab's
    ``meshing_re_orient_faces_coherently`` a serial face-to-face visit and ``trimesh.repair
    .fix_winding`` a flood fill, so all three grow with graph depth where the union-find does not.

    ``igl.bfs_orient`` returns ``(FF, C)`` -- the reoriented faces and the per-face *component id*,
    not the flip mask; the mask is recovered by comparing ``FF`` to ``F`` (see
    ``tests/test_validation.py``). Unlike the other two it does not mutate an input, so nothing is
    rebuilt inside the callable.
    """
    if bench_case.kind == "meshlib":
        # A fourth traversal, and the only one that also decides the *global* sign: it ray-casts to
        # find which side is outside, so it does strictly more than the flip mask and is expected
        # to cost more. It returns a bitset without touching the mesh, so one mesh serves every
        # round -- but the ray casting builds an AABB tree lazily, so it is pre-warmed outside the
        # timed callable and this row prices the traversal rather than the tree.
        mesh_ml = bench_case.new_mesh_ml()
        mm.findDisorientedFaces(mesh_ml)
        disoriented_ml = bench_case.run(lambda: mm.findDisorientedFaces(mesh_ml))
        assert disoriented_ml.size() <= bench_case.n_faces
        return
    if bench_case.kind == "igl":
        faces_np = np.ascontiguousarray(bench_case.faces_np, dtype=np.int64)
        oriented_igl, _components_igl = bench_case.run(lambda: igl.bfs_orient(faces_np))
        assert oriented_igl.shape == (bench_case.n_faces, 3)
        return
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


@pytest.mark.benchmark(group="make_volume")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_make_volume(bench_case: BenchCase) -> None:
    """
    Orient the whole surface outward: one watertightness predicate, one reduction, one flip pass.

    ``trimesh.repair.fix_inversion`` is the same operation and the same decision -- is the mesh
    closed, and is its signed volume negative -- so the two are directly comparable. It mutates the
    mesh in place and caches the volume on it, so the ``tm.Trimesh`` is rebuilt inside the timed
    callable as the other trimesh rows here do.

    The scan meshes are open, so both sides take the "not watertight, return unchanged" path and
    what this row prices is the *test*, which is the point: the watertightness check dominates a
    call that would otherwise be one reduction and one relabel.
    """
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        oriented = bench_case.run(lambda: tw.repair.make_volume(vertices, faces))
        assert int(oriented.shape[0]) == faces.shape[0]
    else:  # trimesh mutates in place and caches the volume: rebuild inside the timed callable
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np

        def fix_inversion_tm() -> tm.Trimesh:
            mesh_tm = tm.Trimesh(vertices_np, faces_np, process=False)
            tm.repair.fix_inversion(mesh_tm)
            return mesh_tm

        assert len(bench_case.run(fix_inversion_tm).faces) == bench_case.n_faces


@pytest.mark.benchmark(group="bad_face_mask")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("triwarp", "pymeshlab", "meshlib")
def test_bad_face_mask(bench_case: BenchCase) -> None:
    """
    All three defect criteria at once: face quality, adjacency scatter, per-face gate.

    meshlib's ``findOverlappingTris`` covers the **fold** criterion alone -- it is a proximity
    search over the AABB tree for near-coincident triangles with near-antiparallel normals, where
    the other two rows read the dihedral off the face adjacency -- so it is a lower bound on this
    group and a different algorithm for the one clause it shares. ``maxNormalDot`` is the dihedral
    threshold in dot-product form (``cos(radians(160))``); the equality of the two answers, and the
    input class where they part, are in ``tests/test_repair.py``. The tree is lazily built, so the
    mesh is constructed and pre-warmed outside the timed callable.
    """
    n_faces = bench_case.n_faces
    if bench_case.kind == "meshlib":
        settings_ml = mm.FindOverlappingSettings()
        settings_ml.maxNormalDot = float(np.cos(np.radians(160.0)))
        mesh_part_ml = mm.MeshPart(bench_case.new_mesh_ml())
        mm.findOverlappingTris(mesh_part_ml, settings_ml)
        folded_ml = bench_case.run(lambda: mm.findOverlappingTris(mesh_part_ml, settings_ml))
        assert folded_ml.size() <= n_faces
        return
    if bench_case.kind == "pymeshlab":
        # Selection-only, so the geometry survives and the MeshSet is shared.
        meshset_pml = bench_case.meshset_pml
        bench_case.run(
            lambda: meshset_pml.compute_selection_bad_faces(
                usear=True,
                aratio=0.02,
                usenf=True,
                nfratio=60.0,
                select_folded_faces=True,
                folded_faces_angle_threshold=160.0,
            )
        )
        assert meshset_pml.current_mesh().face_selection_array().shape == (n_faces,)
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    bad = bench_case.run(
        lambda: tw.repair.bad_face_mask(
            vertices, faces, min_quality=0.02, max_normal_angle=60.0, max_fold_angle=160.0
        )
    )
    assert bad.shape == (n_faces,)


@pytest.mark.benchmark(group="remove_t_vertices")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("triwarp", "pymeshlab")
def test_remove_t_vertices(bench_case: BenchCase) -> None:
    """A flip loop over the slivers: no work on ``saddle``, real work on ``saddle_graded``."""
    if bench_case.kind == "pymeshlab":
        # Rewrites the topology, so the MeshSet is rebuilt inside the timed callable.
        # ``repeat=True`` is MeshLab's own iterate-to-convergence, which is what triwarp's
        # ``max_iter`` passes are.
        new_meshset_pml = bench_case.new_meshset_pml

        def repair_pml() -> int:
            meshset_pml = new_meshset_pml()
            meshset_pml.meshing_remove_t_vertices(method="Edge Flip", threshold=40.0, repeat=True)
            return meshset_pml.current_mesh().face_number()

        assert bench_case.run(repair_pml, rounds=3) > 0
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    flipped = bench_case.run(lambda: tw.repair.remove_t_vertices(vertices, faces), rounds=3)
    assert int(flipped.shape[0]) == int(faces.shape[0])


@pytest.mark.benchmark(group="remove_degenerate_faces")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("triwarp", "meshlib")
def test_remove_degenerate_faces(bench_case: BenchCase) -> None:
    """
    Find the zero-area triangles and compact them away: an altitude test, a scan and a gather.

    On the ``quality`` axis rather than a defect sweep for the reason ``bad_face_mask`` is: a
    degenerate face cannot be injected without changing what the other rows measure, and
    ``saddle_graded``'s worst aspect ratio of 4 719 is the closest a registry mesh comes to one.
    Both sides should be flat across the axis -- neither's cost depends on how many it finds -- and
    a row that is not is the finding.

    meshlib's ``findDegenerateFaces`` returns the same face set (asserted in
    ``tests/test_repair.py``) and stops there, so it does strictly less than triwarp's row, which
    also rebuilds the mesh without those faces. Its ``criticalAspectRatio`` is left at its
    ``FLT_MAX`` default, the setting under which its criterion is triwarp's.
    """
    if bench_case.kind == "meshlib":
        mesh_part_ml = mm.MeshPart(bench_case.new_mesh_ml())
        degenerate_ml = bench_case.run(lambda: mm.findDegenerateFaces(mesh_part_ml))
        assert degenerate_ml.size() <= bench_case.n_faces
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    kept_vertices, kept_faces = bench_case.run(
        lambda: tw.repair.remove_degenerate_faces(vertices, faces)
    )
    assert int(kept_faces.shape[0]) <= int(faces.shape[0])
    assert int(kept_vertices.shape[0]) <= bench_case.n_vertices


_tangled_cache: dict[tuple[str, str], tuple] = {}


def _tangled(bench_case: BenchCase) -> tuple:
    """
    One mesh that genuinely self-intersects: the mesh concatenated with a shifted copy of itself.

    The registry has no self-intersecting mesh, and a repair benchmark needs damage to repair. Two
    copies overlapping by a third of the diagonal, welded into a single face buffer, is the standard
    way to make one -- the same construction ``test_intersection.py`` uses for its two-mesh rows,
    except merged so the intersection is a *self*-intersection.

    Cached: it is the input, and building it is a concatenate plus a translation.
    """
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _tangled_cache:
        vertices_np = bench_case.vertices_np
        diagonal = float(np.linalg.norm(vertices_np.max(axis=0) - vertices_np.min(axis=0)))
        shifted_np = np.ascontiguousarray(
            vertices_np + np.array([0.35 * diagonal, 0.0, 0.0]), dtype=np.float32
        )
        vertices_wp, faces_wp = bench_case.vertices_wp, bench_case.faces_wp
        shifted_wp = wp.array(shifted_np, dtype=wp.vec3, device=bench_case.device)
        _tangled_cache[key] = tw.combine.concatenate(
            [(vertices_wp, faces_wp), (shifted_wp, faces_wp)]
        )
    return _tangled_cache[key]


@pytest.mark.noparity(
    "meshlib",
    reason="D2 a different algorithm with a measured disagreement: localFixSelfIntersections "
    "subdivides the affected region and relaxes it, where fix_self_intersections cuts the region "
    "out and refills the rim. On the shared 16x16 self-intersecting torus MeshLib's leaves 281 "
    "intersecting faces and triwarp's leaves 0, so the outputs are not comparable and neither is a "
    "reference for the other. No library does the cut-and-refill repair, so the correctness claim "
    "is the contract itself, in tests/test_repair.py::test_fix_self_intersections_local_clears_them.",
)
@pytest.mark.benchmark(group="fix_self_intersections")
@pytest.mark.benchmeshes("sphere_med")
@pytest.mark.benchlibs("triwarp", "meshlib")
@pytest.mark.parametrize("method", ["local", "voxel"])
def test_fix_self_intersections(bench_case: BenchCase, method: str) -> None:
    """
    Repair a genuine self-intersection, by cutting-and-refilling or by rebuilding.

    The two methods are different costs of different kinds and the parametrize is what keeps them
    attributable. ``local`` is a detect-dilate-delete-refill loop whose cost is the *damage*: the
    detector runs on the whole mesh but the DP runs on the rims, so it tracks the intersecting band
    rather than the face count. ``voxel`` is a signed distance field plus a marching pass, so its
    cost is the *lattice* and it does not care what was wrong.

    The input is one mesh overlapping a shifted copy of itself -- a deep interpenetration, which is
    the case the local method only *reduces* rather than clears (152 intersecting faces to 34 on a
    small instance; the function's docstring measures this). The row therefore asserts progress and a
    non-empty answer, not convergence: asserting zero would be asserting something the method does
    not promise on this input class.

    meshlib's two fixers are timed beside it for scale, and the noparity entry says why they are not
    a reference: its local one subdivides and relaxes instead of cutting, and on the shared torus
    fixture the two answers differ by 281 faces to 0.

    First measurement, medians on an RTX 5090 at ``sphere_med`` doubled to 163 840 faces:

    | | triwarp-cuda | meshlib |
    |---|---|---|
    | `voxel` | **18.7 ms** | 117.4 (6.3x) |
    | `local` | 196.3 | **46.0** (4.3x behind) |

    The split is the point. The voxel path is a device field plus a marching pass and wins by the
    margin the ``offset`` groups show; the local path is a host-side loop of detect, dilate, delete,
    DP, refine -- five wrapper chains per pass, three passes -- and loses to a single C++ traversal.
    Anything spent here belongs in the refill chain, not in the detector.
    """
    if bench_case.kind == "meshlib":
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        diagonal = float(np.linalg.norm(vertices_np.max(axis=0) - vertices_np.min(axis=0)))
        tangled_np = np.vstack([vertices_np, vertices_np + np.array([0.35 * diagonal, 0.0, 0.0])])
        tangled_faces_np = np.vstack([faces_np, faces_np + vertices_np.shape[0]])
        voxel = diagonal / 128.0

        def fix_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(tangled_np, tangled_faces_np)
            if method == "voxel":
                mm.fixSelfIntersections(mesh_ml, voxel)
            else:
                mm.localFixSelfIntersections(mesh_ml, mm.SelfIntersections.Settings())
            return mesh_ml.topology.numValidFaces()

        assert bench_case.run(fix_ml, rounds=3) > 0
        return

    vertices, faces = _tangled(bench_case)
    fixed_vertices, fixed_faces = bench_case.run(
        lambda: tw.repair.fix_self_intersections(vertices, faces, method=method), rounds=3
    )
    assert int(fixed_faces.shape[0]) > 0
    assert int(fixed_vertices.shape[0]) > 0


@pytest.mark.benchmark(group="collapse_small_triangles")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("triwarp", "meshlib")
def test_collapse_small_triangles(bench_case: BenchCase) -> None:
    """
    Collapse the sub-threshold triangles to a fixpoint: the one repair group that iterates.

    The ``quality`` axis is the axis this group is *about*: ``saddle`` has nothing under the
    threshold and exits after one pass, while ``saddle_graded``'s fine end gives the loop rounds of
    real work, so the pair separates the detection cost from the collapsing cost. Both rows use the
    same threshold, expressed in each library's own parameter -- triwarp's ``epsilon`` is a relative
    *area* against the squared bounding-box diagonal and MeshLib's ``tinyEdgeLength`` an absolute
    *length* -- so this is a cost comparison at a matched scale rather than at a matched knob.

    meshlib is the only bound reference for this function at all: libigl does not export
    ``collapse_small_triangles`` despite shipping the header. ``resolveMeshDegenerations`` mutates
    and invalidates the mesh's tree, so its mesh is rebuilt inside the timed callable and this row
    carries the build. Both loop, so both take ``rounds=3``.
    """
    diagonal = float(
        np.linalg.norm(bench_case.vertices_np.max(axis=0) - bench_case.vertices_np.min(axis=0))
    )
    epsilon = 1e-5
    if bench_case.kind == "meshlib":
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        settings_ml = mm.ResolveMeshDegenSettings()
        settings_ml.tinyEdgeLength = 1e-2 * diagonal
        settings_ml.maxDeviation = 1e-3 * diagonal

        def resolve_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(vertices_np, faces_np)
            mm.resolveMeshDegenerations(mesh_ml, settings_ml)
            return mesh_ml.topology.numValidFaces()

        assert 0 < bench_case.run(resolve_ml, rounds=3) <= bench_case.n_faces
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    kept_vertices, kept_faces = bench_case.run(
        lambda: tw.repair.collapse_small_triangles(vertices, faces, epsilon), rounds=3
    )
    assert int(kept_faces.shape[0]) <= int(faces.shape[0])
    assert int(kept_vertices.shape[0]) <= bench_case.n_vertices


@pytest.mark.benchmark(group="eliminate_tunnels")
@pytest.mark.benchaxis("genus")
@pytest.mark.benchlibs("triwarp")
def test_eliminate_tunnels(bench_case: BenchCase) -> None:
    """
    Removing every thin handle: homology basis, shorten, cut, fill -- the whole chain in one call.

    On the ``genus`` axis, the only one in either registry with a handle to remove, so the row is
    read against ``benchmarks/test_homology.py`` and ``test_geodesic_walk.py::test_shorten_loop``:
    those two time the first and second stages of this call.

    First measurement, medians on an RTX 5090, with the two upstream groups from the same session:

    | mesh | eliminated | ``eliminate_tunnels`` | basis | shorten | cut + fill |
    |---|---|---|---|---|---|
    | ``handles_1`` | 1 | **14.9 ms** | 5.8 | 1.7 | 7.4 |
    | ``handles_64`` | 1 | **90.8 ms** | 12.1 | 9.1 | 69.6 |

    So the **cut and fill dominate**, and increasingly: half the call at genus 1 and more than three
    quarters at genus 64, where the basis and the shortening together are 21 of 91 ms. That is worth
    stating because the obvious reading of the chain is the reverse -- the basis is the expensive
    thing in its own group and the shortening is the new code -- and neither is where the time goes.
    Note both rows eliminate **one** tunnel: the loops kept per call are vertex-disjoint, so
    ``handles_64``'s 128 overlapping generators still yield one, and its extra cost is the larger
    basis and the longer loops rather than more cutting.

    triwarp-only, and not by omission. MeshLib is the only library that binds the operation and its
    ``eliminateTunnels`` is a **no-op** on every input probed -- unchanged face count and Euler
    characteristic at ``maxTunnelLength`` 4.0 and 1e9, ``maxIters`` 1 to 100, all three
    ``TunnelLoopType`` values, ``buildCoLoops`` off, and through the ``FillHoleNicelySettings``
    overload -- while its own ``detectTunnelFaces`` fires on the same mesh. A row that changes
    nothing is not a comparison, so it is left out rather than timed; see
    ``tests/test_repair.py::test_eliminate_tunnels_drops_the_genus_by_the_count_it_reports``.
    """
    if bench_case.mesh_name == "sphere_med":
        pytest.skip("sphere_med is genus 0: there is no tunnel to eliminate")
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    # Every handle in these fixtures is a real one, so an unbounded length eliminates a maximal
    # disjoint family and times the whole chain rather than the length test.
    cut_vertices, cut_faces, eliminated = bench_case.run(
        lambda: tw.repair.eliminate_tunnels(vertices, faces, 1e9), rounds=3
    )
    assert eliminated >= 1
    assert int(cut_vertices.shape[0]) >= bench_case.n_vertices
    assert tw.measures.euler_characteristic(cut_faces) == (
        tw.measures.euler_characteristic(faces) + 2 * eliminated
    )


@pytest.mark.benchmark(group="eliminate_degree3_vertices")
@pytest.mark.benchlibs("triwarp", "meshlib")
def test_eliminate_degree3_vertices(bench_case: BenchCase) -> None:
    """
    One pass over every vertex, plus a face compaction -- and usually nothing to remove.

    The scan meshes carry few valence-3 interior vertices, so this group mostly times the *search*:
    a ``vertex_one_rings`` build, a candidate map, an independence pass and one readback. That is
    the honest thing to measure, because the search is what a caller pays unconditionally in a
    repair pipeline while the removal is proportional to a defect that may not be there.

    meshlib's row is ``findInnerVertsOfDegree(topology, 3)`` -- the candidate mask, not the removal,
    since ``eliminateDegree3Vertices`` mutates in place and would need a fresh mesh per round while
    finding nothing after the first. So it is the same *search* on both sides; where it is not a
    like-for-like is that meshlib is handed a ``MeshTopology`` built outside its row and triwarp
    builds a halfedge structure inside its own.

    First measurement, medians on an RTX 5090:

    | mesh | triwarp-cuda | meshlib (mask only) |
    |---|---|---|
    | ``bunny`` | 2.47 ms | 0.042 (58.7x) |
    | ``dragon`` | 4.96 ms | 0.336 (14.8x) |
    | ``happy_buddha`` | 6.64 ms | (capped) |

    Attributed on a clean ``icosphere(6)`` at 81 920 faces, where nothing is removed: **0.84 ms for
    one pass**, of which ``vertex_one_rings`` is 0.56 (67 %) and the vertex-count readback 0.09. So
    the floor is the halfedge build, and a scan mesh's several milliseconds are that floor times the
    number of passes -- removing one valence-3 vertex can expose another, so the loop runs until it
    finds none. Moving the vertex compaction out of the loop was tried and is **flat** (2.47 against
    2.36 ms, within noise at three passes); it is kept because it is strictly less work, not because
    it showed up.
    """
    if bench_case.kind == "meshlib":
        mesh_ml = bench_case.new_mesh_ml()
        bits_ml = bench_case.run(lambda: mm.findInnerVertsOfDegree(mesh_ml.topology, 3))
        assert bits_ml.size() >= 0
        return
    if bench_case.mesh_name in {"bunny_decimated", "lucy"}:
        pytest.skip(f"{bench_case.mesh_name} is not edge-manifold, so it has no vertex fans")
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    out_vertices, out_faces, removed = bench_case.run(
        lambda: tw.repair.eliminate_degree3_vertices(vertices, faces), rounds=3
    )
    assert removed >= 0
    assert int(out_faces.shape[0]) <= int(faces.shape[0])
    assert int(out_vertices.shape[0]) <= bench_case.n_vertices


@pytest.mark.benchmark(group="straighten_boundary")
@pytest.mark.benchlibs("triwarp", "meshlib")
def test_straighten_boundary(bench_case: BenchCase) -> None:
    """
    Closing rim notches: per pass, a halfedge build, a rim walk and one candidate test per vertex.

    The cost is dominated by the halfedge build and the per-*vertex* candidate launch, so it tracks
    the mesh rather than the rim -- which is the honest thing to say about it, since a rim is
    ``O(sqrt(n_vertices))`` and the work is not. A rim-indexed candidate pass would fix that and is
    the obvious follow-up; it is not done because the pass count is one by default.

    meshlib's ``straightenBoundary`` is per **rim** and in place, so its row loops over
    ``findHoleRepresentiveEdges`` on a fresh mesh per round, and its topology build is inside the
    timed callable the way ours is. Both take the same two gates by the same definitions and agree
    exactly on a ragged planar rim (``tests/test_repair.py``).

    First measurement, medians on an RTX 5090: **0.70 ms** on ``bunny`` against meshlib's 9.20
    (13.1x ahead) and **1.19 ms** on ``dragon`` against 56.05 (**47.3x**). Both rows carry their own
    structure build, so the ratio is the parallel candidate test against a serial rim walk, and it
    widens with the mesh exactly as that predicts. ``happy_buddha`` reads 1.39 ms.
    """
    if bench_case.kind == "meshlib":

        def straighten_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(bench_case.vertices_np, bench_case.faces_np)
            holes_ml = mesh_ml.topology.findHoleRepresentiveEdges()
            for edge in holes_ml:
                mm.straightenBoundary(mesh_ml, edge, 0.9, 10.0)
            return len(holes_ml)

        assert bench_case.run(straighten_ml, rounds=3) >= 0
        return
    if bench_case.mesh_name in {"bunny_decimated", "lucy"}:
        pytest.skip(f"{bench_case.mesh_name} is not edge-manifold, so the rim cannot be walked")
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    straightened, added = bench_case.run(
        lambda: tw.repair.straighten_boundary(vertices, faces), rounds=3
    )
    assert added >= 0
    assert int(straightened.shape[0]) >= int(faces.shape[0])
