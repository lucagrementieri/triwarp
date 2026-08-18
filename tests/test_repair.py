"""Regression tests for ``triwarp.repair`` against libigl."""

from __future__ import annotations

import igl
import numpy as np
import pymeshlab as ml
import pytest
import trimesh as tm
import trimesh.repair as tm_repair
import warp as wp

import triwarp as tw
from tests.comparisons import canonical_winding, lexsort_rows, same_partition, undirected_edges
from tests.conversions import (
    numpy_to_warp,
    trimesh_to_open3d,
    trimesh_to_pymeshlab,
    trimesh_to_pyvista,
)


def _resolve_duplicated_faces_ref(faces_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CPU reference mirroring ``igl::resolve_duplicated_faces``."""
    faces = np.asarray(faces_np, dtype=np.int32).reshape((-1, 3))
    n_faces = faces.shape[0]
    if n_faces == 0:
        return faces.copy(), np.empty(0, dtype=np.int32)

    sorted_faces = np.sort(faces, axis=1)
    unique_sorted, inverse = np.unique(sorted_faces, axis=0, return_inverse=True)
    num_unique = unique_sorted.shape[0]

    kept: list[int] = []
    for ui in range(num_unique):
        member = np.flatnonzero(inverse == ui)
        urow = unique_sorted[ui]
        signed_ids: list[int] = []
        count = 0
        for fi in member:
            row = faces[fi]
            consistent = (
                (row[0] == urow[0] and row[1] == urow[1] and row[2] == urow[2])
                or (row[0] == urow[1] and row[1] == urow[2] and row[2] == urow[0])
                or (row[0] == urow[2] and row[1] == urow[0] and row[2] == urow[1])
            )
            signed = int(fi + 1) if consistent else -int(fi + 1)
            signed_ids.append(signed)
            count += 1 if consistent else -1

        if member.size == 1:
            kept.append(int(member[0]))
            continue
        if count == 1:
            for fid in signed_ids:
                if fid > 0:
                    kept.append(fid - 1)
                    break
        elif count == -1:
            for fid in signed_ids:
                if fid < 0:
                    kept.append(-fid - 1)
                    break
        elif count == 0:
            continue
        else:
            raise ValueError(f"non-orientable duplicate face group {ui} with count {count}")

    if len(kept) == 0:
        return np.empty((0, 3), dtype=np.int32), np.empty(0, dtype=np.int32)
    kept_np = np.asarray(kept, dtype=np.int32)
    return faces[kept_np], kept_np


def _assert_duplicate_vertices_match(
    vertices_np: np.ndarray,
    sv_wp: np.ndarray,
    svj_wp: np.ndarray,
    sf_wp: np.ndarray | None = None,
    faces_np: np.ndarray | None = None,
    epsilon: float = 0.0,
) -> None:
    sv_igl, _, _svj_igl, sf_igl = (
        igl.remove_duplicate_vertices(vertices_np, faces_np, epsilon)
        if faces_np is not None
        else (*igl.remove_duplicate_vertices(vertices_np, epsilon), None)
    )
    # triwarp keeps float32 vertices where igl works in float64, so the surviving position sets are
    # compared with a tolerance rather than exactly (exact equality happens to hold for coordinates
    # that are exactly representable, but not for a mesh with irrational ones).
    assert np.allclose(lexsort_rows(sv_wp), lexsort_rows(sv_igl), rtol=1e-5, atol=1e-5)
    for i, vertex in enumerate(vertices_np):
        assert np.allclose(sv_wp[svj_wp[i]], vertex, rtol=1e-5, atol=1e-5)
    if sf_wp is not None and faces_np is not None and sf_igl is not None:
        tri_wp = sv_wp[sf_wp.reshape(-1, 3)]
        tri_igl = sv_igl[sf_igl]
        assert np.allclose(np.sort(tri_wp, axis=1), np.sort(tri_igl, axis=1), rtol=1e-5, atol=1e-5)


@pytest.mark.parity("remove_unreferenced_vertices", "igl")
def test_remove_unreferenced_identity(icosahedron, device: str):
    """Class A: all four returns against ``igl.remove_unreferenced``, on a fully-referenced mesh."""
    mesh_tm, mesh_wp = icosahedron
    vertices_np = mesh_tm.vertices
    faces_np = mesh_tm.faces

    nv_wp, nf_wp, remap_wp, inverse_wp = tw.repair.remove_unreferenced_vertices(
        mesh_wp.points, mesh_wp.indices, return_inverse=True
    )

    nv_igl, nf_igl, remap_igl, inverse_igl = igl.remove_unreferenced(
        np.asarray(vertices_np, dtype=np.float64), np.asarray(faces_np, dtype=np.int32)
    )

    assert np.allclose(nv_wp.numpy(), nv_igl, rtol=1e-5, atol=1e-5)
    assert np.array_equal(nf_wp.numpy().reshape(-1, 3), nf_igl)
    assert np.array_equal(remap_wp.numpy(), remap_igl.ravel())
    assert np.array_equal(inverse_wp.numpy(), inverse_igl.ravel())


@pytest.mark.parity("remove_unreferenced_vertices", "igl")
def test_remove_unreferenced_extra_vertices(device: str):
    """
    Class A with four unreferenced vertices appended -- the case the identity test cannot see.

    The forward map and the inverse map are both compared, not just the compacted buffers: a
    compaction that dropped the right vertices but numbered them differently would pass on the first
    two returns alone, and every caller that keeps per-vertex data alongside relies on the maps.
    """
    rng = np.random.default_rng(0)
    vertices_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    faces_np = np.array([[0, 1, 2]], dtype=np.int32)
    extra = rng.normal(size=(4, 3))
    vertices_full_np = np.vstack([vertices_np, extra])

    vertices_wp, faces_wp = numpy_to_warp(vertices_full_np, faces_np, device)
    nv_wp, nf_wp, remap_wp, inverse_wp = tw.repair.remove_unreferenced_vertices(
        vertices_wp, faces_wp, return_inverse=True
    )

    nv_igl, nf_igl, remap_igl, inverse_igl = igl.remove_unreferenced(vertices_full_np, faces_np)

    assert np.allclose(nv_wp.numpy(), nv_igl, rtol=1e-5, atol=1e-5)
    assert np.array_equal(nf_wp.numpy().reshape(-1, 3), nf_igl)
    assert np.array_equal(remap_wp.numpy(), remap_igl.ravel())
    assert np.array_equal(inverse_wp.numpy(), inverse_igl.ravel())


@pytest.mark.parity("remove_unreferenced_vertices", "open3d")
def test_remove_unreferenced_matches_open3d(device: str):
    """
    Class A on both compacted buffers -- open3d keeps first-occurrence order like triwarp does.

    open3d returns no forward or inverse map (the mesh is mutated in place and read back), so
    unlike the igl tests only the vertices and faces are comparable; the maps stay pinned by igl.
    The four appended vertices make the compaction real, and the interleaved layout (an
    unreferenced vertex *between* referenced ones) is what catches a compaction that preserves
    a prefix instead of an order.
    """
    rng = np.random.default_rng(0)
    referenced_np = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float64
    )
    # Interleave dead vertices among live ones rather than appending them.
    vertices_full_np = np.insert(referenced_np, [1, 3, 4, 4], rng.normal(size=(4, 3)), axis=0)
    faces_np = np.array([[0, 2, 4], [2, 5, 4]], dtype=np.int32)

    vertices_wp, faces_wp = numpy_to_warp(vertices_full_np, faces_np, device)
    nv_wp, nf_wp, _remap_wp = tw.repair.remove_unreferenced_vertices(vertices_wp, faces_wp)

    mesh_o3d = trimesh_to_open3d(tm.Trimesh(vertices_full_np, faces_np, process=False))
    mesh_o3d.remove_unreferenced_vertices()
    vertices_o3d = np.asarray(mesh_o3d.vertices)

    assert vertices_o3d.shape[0] == referenced_np.shape[0]  # the reference really compacted
    assert np.allclose(nv_wp.numpy(), vertices_o3d, rtol=1e-5, atol=1e-5)
    assert np.array_equal(nf_wp.numpy().reshape(-1, 3), np.asarray(mesh_o3d.triangles))


def test_remove_unreferenced_sentinel(device: str):
    vertices_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    faces_np = np.array([[0, 1, -1], [0, 2, 1]], dtype=np.int32)
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)

    nv_wp, nf_wp, remap_wp = tw.repair.remove_unreferenced_vertices(vertices_wp, faces_wp)

    assert np.allclose(nv_wp.numpy(), vertices_np[[0, 1, 2]], rtol=1e-5, atol=1e-5)
    assert np.array_equal(nf_wp.numpy().reshape(-1, 3), faces_np)
    expected_remap = np.array([0, 1, 2], dtype=np.int32)
    assert np.array_equal(remap_wp.numpy(), expected_remap)


def test_remove_duplicate_vertices_exact(device: str):
    vertices_np = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    vertices_wp = wp.array(
        np.ascontiguousarray(vertices_np.astype(np.float32)), dtype=wp.vec3, device=device
    )

    faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    sv_wp, _, svj_wp, _ = tw.repair.remove_duplicated_vertices(vertices_wp, faces_wp, epsilon=0.0)
    _assert_duplicate_vertices_match(vertices_np, sv_wp.numpy(), svj_wp.numpy())


def test_remove_duplicate_vertices_epsilon(device: str):
    vertices_np = np.array(
        [[0.0, 0.0, 0.0], [1e-9, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1e-9, 0.0]], dtype=np.float64
    )
    epsilon = 1e-8
    vertices_wp = wp.array(
        np.ascontiguousarray(vertices_np.astype(np.float32)), dtype=wp.vec3, device=device
    )

    faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    sv_wp, _, svj_wp, _ = tw.repair.remove_duplicated_vertices(
        vertices_wp, faces_wp, epsilon=epsilon
    )
    _assert_duplicate_vertices_match(vertices_np, sv_wp.numpy(), svj_wp.numpy(), epsilon=epsilon)


def test_remove_duplicate_vertices_epsilon_negative_coordinates(
    icosahedron: tuple[tm.Trimesh, wp.Mesh], device: str
) -> None:
    """
    Class B (partition): igl decides which vertices merge when the grid indices go negative.

    The fixture is centred at ``(-1, 0, 2)``, so every duplicated vertex has a negative
    coordinate -- the case the row packing cannot take directly and where a wrong offset
    silently merges the wrong pairs.
    """
    # Any mesh spanning the origin rounds to negative grid indices, which the row packing cannot
    # take directly. The fixture is centred at (-1, 0, 2), so every duplicated vertex below has at
    # least one negative coordinate; igl is the oracle for which ones merge.
    mesh_tm, _ = icosahedron
    vertices_np = np.vstack((mesh_tm.vertices, mesh_tm.vertices[:4] + 1e-9)).astype(np.float64)
    assert vertices_np.min() < 0.0
    faces_np = np.asarray(mesh_tm.faces, dtype=np.int32)

    epsilon = 1e-6
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    sv_wp, _, svj_wp, sf_wp = tw.repair.remove_duplicated_vertices(
        vertices_wp, faces_wp, epsilon=epsilon
    )
    assert int(sv_wp.shape[0]) == mesh_tm.vertices.shape[0]
    _assert_duplicate_vertices_match(
        vertices_np, sv_wp.numpy(), svj_wp.numpy(), sf_wp.numpy(), faces_np, epsilon=epsilon
    )


def test_remove_duplicate_vertices_faces(device: str):
    vertices_np = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    faces_np = np.array([[0, 1, 3], [2, 1, 3]], dtype=np.int32)
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)

    sv_wp, _, svj_wp, sf_wp = tw.repair.remove_duplicated_vertices(vertices_wp, faces_wp, 0.0)
    _assert_duplicate_vertices_match(
        vertices_np, sv_wp.numpy(), svj_wp.numpy(), sf_wp.numpy(), faces_np
    )


@pytest.mark.parametrize("epsilon", [1e-3, 1e-2])
def test_duplicate_vertex_inverse_matches_the_documented_quantization(
    device: str, epsilon: float
) -> None:
    """
    Class B (label packing): the equivalence classes are the ``epsilon`` grid cells, as documented.

    ``hash_vector_rows`` snaps each coordinate to a multiple of ``epsilon`` measured from the data's
    own minimum corner, so that is the numpy oracle. Only the *partition* is shared -- triwarp names
    a class by its hash-table slot and ``np.unique`` numbers them in lexicographic order -- hence
    [`same_partition`][tests.comparisons.same_partition]. Measured 34 classes on both sides at both
    tolerances, out of 60 input positions, so the comparison is neither all-merged nor all-distinct.
    """
    rng = np.random.default_rng(9)
    positions_np = rng.integers(0, 4, size=(60, 3)).astype(np.float32) * 0.25
    # Jitter well inside one cell, so a correct implementation merges exactly the seeded collisions.
    positions_np += rng.normal(scale=1e-5, size=positions_np.shape).astype(np.float32)
    positions_wp = wp.array(np.ascontiguousarray(positions_np), dtype=wp.vec3, device=device)

    inverse_np = tw.repair.duplicate_vertex_inverse(positions_wp, epsilon).numpy()

    cells_np = np.rint((positions_np - positions_np.min(axis=0)) / epsilon).astype(np.int64)
    _cells_np, expected_np = np.unique(cells_np, axis=0, return_inverse=True)
    assert 1 < int(expected_np.max()) + 1 < positions_np.shape[0]
    assert int(inverse_np.max()) + 1 == int(expected_np.max()) + 1
    assert same_partition(inverse_np, expected_np)


def test_duplicate_vertex_inverse_at_zero_epsilon_collapses_bitwise_equal_positions(
    device: str,
) -> None:
    """
    The ``epsilon == 0`` branch, on the input its docstring promises it handles reliably.

    A relative bucket is not an equality test, and the Notes say so; what it *does* guarantee is
    collapsing positions that are already bitwise equal, including across ``+0.0`` / ``-0.0``. Three
    classes from five rows here, so neither the merge nor the separation is vacuous.
    """
    positions_np = np.array(
        [[0.0, 0.0, 0.0], [-0.0, -0.0, -0.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [5.0, 5.0, 5.0]],
        dtype=np.float32,
    )
    positions_wp = wp.array(np.ascontiguousarray(positions_np), dtype=wp.vec3, device=device)

    inverse_np = tw.repair.duplicate_vertex_inverse(positions_wp, 0.0).numpy()

    assert int(inverse_np.max()) + 1 == 3
    assert inverse_np[0] == inverse_np[1]
    assert inverse_np[2] == inverse_np[3]
    assert len({int(inverse_np[0]), int(inverse_np[2]), int(inverse_np[4])}) == 3


def test_duplicate_vertex_inverse_is_what_remove_duplicated_vertices_remaps_by(device: str) -> None:
    """
    The reason it is public: the same map ``remove_duplicated_vertices`` used, for other attributes.

    A caller remapping per-vertex colors or UVs needs the map without the deduplicated buffers, so
    what has to hold is that applying it to the faces reproduces that function's own remapped face
    buffer, and that gathering the deduplicated positions by it returns the input to within
    ``epsilon``.
    """
    epsilon = 1e-3
    base_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    positions_np = np.vstack((base_np, base_np + 1e-6, base_np[[0]])).astype(np.float32)
    faces_np = np.array([[0, 1, 2], [3, 4, 5], [6, 1, 2]], dtype=np.int32)
    positions_wp, faces_wp = numpy_to_warp(positions_np, faces_np, device)

    inverse_np = tw.repair.duplicate_vertex_inverse(positions_wp, epsilon).numpy()

    unique_wp, _indices_wp, _inverse_wp, unique_faces_wp = tw.repair.remove_duplicated_vertices(
        positions_wp, faces_wp, epsilon
    )
    assert int(unique_wp.shape[0]) == 3
    assert np.array_equal(unique_faces_wp.numpy(), inverse_np[faces_np.reshape(-1)])
    assert np.abs(unique_wp.numpy()[inverse_np] - positions_np).max() <= epsilon


def test_resolve_duplicated_faces_cancelling(device: str):
    faces_np = np.array([[0, 1, 2], [0, 1, 2], [0, 2, 1], [0, 2, 1]], dtype=np.int32)
    faces_wp = wp.array(
        np.ascontiguousarray(faces_np.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device
    )

    f2_wp, j_wp = tw.repair.resolve_duplicated_faces(faces_wp)
    f2_ref, j_ref = _resolve_duplicated_faces_ref(faces_np)

    assert np.array_equal(f2_wp.numpy().reshape(-1, 3), f2_ref)
    assert np.array_equal(j_wp.numpy(), j_ref)


def test_resolve_duplicated_faces_keep_positive(device: str):
    faces_np = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 2, 1], [0, 2, 1]], dtype=np.int32)
    faces_wp = wp.array(
        np.ascontiguousarray(faces_np.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device
    )

    f2_wp, j_wp = tw.repair.resolve_duplicated_faces(faces_wp)
    f2_ref, j_ref = _resolve_duplicated_faces_ref(faces_np)

    assert np.array_equal(f2_wp.numpy().reshape(-1, 3), f2_ref)
    assert np.array_equal(j_wp.numpy(), j_ref)


def test_resolve_duplicated_faces_random(device: str):
    """Randomized orientable duplicate groups vs the CPU reference (order-insensitive)."""
    rng = np.random.default_rng(17)
    pool = np.stack(np.meshgrid(np.arange(12), np.arange(12, 24), np.arange(24, 36)), -1).reshape(
        -1, 3
    )
    base_np = rng.permutation(pool, axis=0)[:300].astype(np.int32)
    singles = base_np[:100]
    cancelling = np.vstack([base_np[100:180], base_np[100:180][:, ::-1]])
    majority_pos = np.vstack([np.repeat(base_np[180:240], 2, axis=0), base_np[180:240][:, ::-1]])
    faces_np = np.vstack([singles, cancelling, majority_pos])
    faces_np = rng.permutation(faces_np, axis=0)
    faces_wp = wp.array(
        np.ascontiguousarray(faces_np.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device
    )

    f2_wp, j_wp = tw.repair.resolve_duplicated_faces(faces_wp)
    f2_ref, j_ref = _resolve_duplicated_faces_ref(faces_np)

    # The reference emits groups in lexicographic (np.unique) order while the production path
    # follows its hash-sorted unique order; the kept sets must agree exactly.
    assert np.array_equal(np.sort(j_wp.numpy()), np.sort(j_ref))
    resolved_rows = {tuple(row) for row in f2_wp.numpy().reshape(-1, 3).tolist()}
    reference_rows = {tuple(row) for row in f2_ref.tolist()}
    assert resolved_rows == reference_rows


def _faces_2d(faces_wp: wp.array) -> np.ndarray:
    return faces_wp.numpy().reshape(-1, 3)


@pytest.mark.parametrize("epsilon", [0.0, 1e-6])
@pytest.mark.parity("remove_duplicated_vertices", "open3d", "pymeshlab", "pyvista")
def test_remove_duplicated_vertices_matches_open3d_and_pymeshlab(
    device: str, epsilon: float, icosphere_coarse: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Welding an unwelded soup, the one repair group whose ``epsilon`` sweep maps across all three.

    ``benchmarks/test_repair.py`` singles this out: MeshLab has a filter per path
    (``meshing_remove_duplicate_vertices`` for exact, ``meshing_merge_close_vertices`` for a
    tolerance) that line up with triwarp's two code paths one-for-one, and Open3D's
    ``remove_duplicated_vertices`` is the exact path. So unlike its siblings in this module this is
    a straight comparison rather than a semantics negotiation. VTK's ``clean`` is a fourth
    implementation of the exact path and lands on the same 162 (measured 172 -> 162 on a mesh with
    ten duplicated vertices, matching triwarp exactly).

    **``validate_mesh`` is not the oracle here** and that was measured rather than assumed: its
    ``coincident_points`` field reads *empty* on ten exactly duplicated vertices, so a comparison
    against it would report no duplicates to remove. ``clean`` is the dedup reference.

    Class B on the *positions*: all three renumber the survivors differently, so the vertex sets are
    compared after a lexsort. Counting alone would be too weak -- a welder that merged the wrong
    pairs can still land on 162 -- which is why the positions are asserted and the count is only
    the headline. Measured: 960 soup positions collapse to exactly 162 on all three sides at both
    epsilon values, matching the icosphere's true vertex count.
    """
    mesh_tm, _mesh_tm_wp = icosphere_coarse
    soup_np = np.ascontiguousarray(mesh_tm.vertices[mesh_tm.faces].reshape(-1, 3))
    faces_np = np.arange(soup_np.shape[0], dtype=np.int32).reshape(-1, 3)

    vertices_wp = wp.array(soup_np.astype(np.float32), dtype=wp.vec3, device=device)
    faces_wp = wp.array(faces_np.reshape(-1), dtype=wp.int32, device=device)
    unique_wp, _indices_wp, _inverse_wp, _faces_wp = tw.repair.remove_duplicated_vertices(
        vertices_wp, faces_wp, epsilon
    )

    meshset_pml = ml.MeshSet()
    meshset_pml.add_mesh(ml.Mesh(soup_np, faces_np))
    if epsilon > 0.0:
        meshset_pml.meshing_merge_close_vertices(threshold=ml.PureValue(epsilon))
    else:
        meshset_pml.meshing_remove_duplicate_vertices()
    vertices_pml = meshset_pml.current_mesh().vertex_matrix()

    soup_tm = tm.Trimesh(soup_np, faces_np, process=False)
    vertices_o3d = np.asarray(trimesh_to_open3d(soup_tm).remove_duplicated_vertices().vertices)
    cleaned_pv = trimesh_to_pyvista(soup_tm).clean(
        point_merging=True, tolerance=epsilon, absolute=True
    )
    vertices_pv = np.asarray(cleaned_pv.points)

    assert int(unique_wp.shape[0]) == len(mesh_tm.vertices)
    assert vertices_pml.shape[0] == len(mesh_tm.vertices)
    assert vertices_o3d.shape[0] == len(mesh_tm.vertices)
    assert vertices_pv.shape[0] == len(mesh_tm.vertices)

    survivors_wp = lexsort_rows(np.round(unique_wp.numpy().astype(np.float64), 5))
    assert np.allclose(survivors_wp, lexsort_rows(np.round(vertices_pml, 5)), atol=1e-5)
    assert np.allclose(survivors_wp, lexsort_rows(np.round(vertices_o3d, 5)), atol=1e-5)
    assert np.allclose(survivors_wp, lexsort_rows(np.round(vertices_pv, 5)), atol=1e-5)

    # The trap this row is written against: validate_mesh does not see exact duplicates.
    assert len(trimesh_to_pyvista(soup_tm).validate_mesh().coincident_points) == 0


@pytest.mark.parity("make_winding_consistent", "pymeshlab")
def test_make_winding_consistent_matches_pymeshlab(
    device: str, icosphere_coarse: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Orientation repair against MeshLab's, which reaches the identical winding on every face.

    The two algorithms are not the same -- ``benchmarks/test_repair.py`` notes MeshLab does a
    serial face-to-face visit where triwarp solves the Z2 bits with a parity-carrying union-find --
    but on a connected orientable surface the answer is unique up to one global flip, so
    agreement is not
    only possible, it is required.

    Class B twice: ``canonical_winding`` rotates each triangle to start at its lowest index (a
    rotation preserves orientation, so a flipped face still compares unequal) and a lexsort removes
    the face ordering. Measured on an icosphere with every third face reversed, the two agree face
    for face **without** needing the global-flip escape -- both anchor on the first face of the
    component -- so the test asserts the strict form and would catch a flip if one appeared.
    """
    mesh_tm, _mesh_tm_wp = icosphere_coarse
    flipped_np = mesh_tm.faces.copy()
    flipped_np[::3] = flipped_np[::3][:, ::-1]

    faces_wp = wp.array(
        np.ascontiguousarray(flipped_np.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device
    )
    repaired_wp = tw.repair.make_winding_consistent(faces_wp)

    meshset_pml = ml.MeshSet()
    meshset_pml.add_mesh(
        ml.Mesh(
            np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64),
            np.ascontiguousarray(flipped_np, dtype=np.int32),
        )
    )
    meshset_pml.meshing_re_orient_faces_coherently()
    faces_pml = meshset_pml.current_mesh().face_matrix()

    assert np.array_equal(
        lexsort_rows(canonical_winding(repaired_wp.numpy())),
        lexsort_rows(canonical_winding(faces_pml)),
    )
    assert tw.validation.is_winding_consistent(repaired_wp)


@pytest.mark.parity("make_winding_consistent", "igl")
def test_make_winding_consistent_matches_igl(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class B (cyclic winding): the repaired face table against ``igl.bfs_orient``'s ``FF``, exactly.

    No global sign fix is needed and that is the interesting part: both libraries anchor each
    component on its lowest-numbered face, so on a connected mesh the answer is unique rather than
    determined up to a per-component flip. The only transform is
    [`tests.comparisons.canonical_winding`][], because a flip is emitted as a rotation of the
    reversed triangle.

    ``bfs_orient``'s second return is the per-face **component id**, not a flip mask -- the trap
    recorded in ``tests/test_validation.py::test_face_orientation_mask_matches_igl``. Here it is
    asserted to be single-valued, which is what makes "no global sign fix" a claim about the
    anchoring rather than a coincidence of this fixture.
    """
    mesh_tm, mesh_wp = icosahedron
    faces_flipped = mesh_tm.faces.copy()
    faces_flipped[::2] = faces_flipped[::2][:, ::-1]
    _, faces_wp = numpy_to_warp(mesh_tm.vertices, faces_flipped, mesh_wp.device)

    oriented_igl, components_igl = igl.bfs_orient(
        np.ascontiguousarray(faces_flipped, dtype=np.int64)
    )
    assert np.unique(components_igl).shape[0] == 1

    repaired_wp = tw.repair.make_winding_consistent(faces_wp)

    assert np.array_equal(
        canonical_winding(_faces_2d(repaired_wp)), canonical_winding(oriented_igl)
    )


@pytest.mark.parity("make_winding_consistent", "trimesh")
def test_make_winding_consistent_repairs_flipped(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Not a library comparison: an inconsistently-wound mesh must come back consistent.

    The input is asserted inconsistent first, so the repair cannot pass by doing nothing. igl
    supplies the elementwise comparison in [`test_make_winding_consistent_matches_igl`]; the
    impossible case is [`test_make_winding_consistent_on_a_non_orientable_mesh`].
    """
    mesh_tm, mesh_wp = icosahedron
    faces_flipped = mesh_tm.faces.copy()
    faces_flipped[::2] = faces_flipped[::2][:, ::-1]  # reverse winding of half the faces
    _, faces_wp = numpy_to_warp(mesh_tm.vertices, faces_flipped, mesh_wp.device)

    assert tw.validation.is_winding_consistent(faces_wp) is False
    repaired_wp = tw.repair.make_winding_consistent(faces_wp)
    assert tw.validation.is_winding_consistent(repaired_wp) is True

    before = faces_flipped
    after = _faces_2d(repaired_wp)
    # Pure per-face winding operation: corner 0 preserved, same vertex set per face.
    assert np.array_equal(after[:, 0], before[:, 0])
    assert np.array_equal(np.sort(after, axis=1), np.sort(before, axis=1))

    # Reference: trimesh.repair.fix_winding reaches the same consistent state.
    reference_tm = tm.Trimesh(vertices=mesh_tm.vertices, faces=before, process=False)
    tm_repair.fix_winding(reference_tm)
    assert bool(reference_tm.is_winding_consistent) is True
    ours_tm = tm.Trimesh(vertices=mesh_tm.vertices, faces=after, process=False)
    assert bool(ours_tm.is_winding_consistent) is True
    # Single connected component: our winding equals trimesh's up to a global flip.
    same = np.allclose(ours_tm.face_normals, reference_tm.face_normals, atol=1e-5)
    opposite = np.allclose(ours_tm.face_normals, -reference_tm.face_normals, atol=1e-5)
    assert same or opposite


def test_make_winding_consistent_on_a_non_orientable_mesh(
    mobius: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Not a library comparison: a Moebius band admits no consistent winding, so the pass cannot win.

    What it does instead is worth pinning, because the docstring's post-condition does not hold
    here and a caller has to know what it gets: the flood-fill orients everything it reaches and the
    contradiction is confined to a seam — measured 41 of 4 524 edges, under 1%, against 39 before
    the pass, so this is *not* a repair that partially helps. It stays a pure per-face winding
    operation either way.
    """
    mesh_tm, mesh_wp = mobius
    assert tw.validation.is_orientable(mesh_wp.indices) is False

    repaired_wp = tw.repair.make_winding_consistent(mesh_wp.indices)
    assert tw.validation.is_winding_consistent(repaired_wp) is False

    inconsistent_np = ~tw.validation.edge_winding_consistent_mask(repaired_wp).numpy()
    assert 0 < inconsistent_np.sum() < 0.02 * inconsistent_np.shape[0]

    before, after = mesh_wp.indices.numpy().reshape(-1, 3), _faces_2d(repaired_wp)
    assert np.array_equal(after[:, 0], before[:, 0])
    assert np.array_equal(np.sort(after, axis=1), np.sort(before, axis=1))
    assert len(mesh_tm.faces) == after.shape[0]


def test_make_winding_consistent_idempotent(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = icosahedron
    repaired_wp = tw.repair.make_winding_consistent(mesh_wp.indices)
    # Already consistently wound: output identical to input.
    assert np.array_equal(_faces_2d(repaired_wp), _faces_2d(mesh_wp.indices))


@pytest.mark.parity("make_volume", "trimesh")
@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube"])
def test_make_volume_repairs_inversion(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Not a library comparison: an inward-wound closed mesh must come back enclosing positive volume.

    Reversing *every* face leaves the winding consistent, so this is the state
    ``make_winding_consistent`` cannot fix and ``make_volume`` exists for. Asserted false
    before and true after.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    faces_inward = mesh_tm.faces[:, ::-1].copy()  # reverse every face -> inward normals
    vertices_wp, faces_wp = numpy_to_warp(mesh_tm.vertices, faces_inward, mesh_wp.device)

    assert tw.validation.is_volume(vertices_wp, faces_wp) is False
    repaired_wp = tw.repair.make_volume(vertices_wp, faces_wp)
    assert tw.validation.is_volume(vertices_wp, repaired_wp) is True

    # Reference: trimesh.repair.fix_inversion also produces an outward-oriented volume.
    reference_tm = tm.Trimesh(vertices=mesh_tm.vertices, faces=faces_inward, process=False)
    tm_repair.fix_inversion(reference_tm)
    ours_tm = tm.Trimesh(vertices=mesh_tm.vertices, faces=_faces_2d(repaired_wp), process=False)
    assert ours_tm.volume > 0.0
    assert np.allclose(ours_tm.face_normals, reference_tm.face_normals, atol=1e-5)


def test_make_volume_leaves_valid_mesh(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = icosahedron
    repaired_wp = tw.repair.make_volume(mesh_wp.points, mesh_wp.indices)
    # Already outward-oriented: unchanged.
    assert np.array_equal(_faces_2d(repaired_wp), _faces_2d(mesh_wp.indices))


def test_make_normals_outward(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Not a library comparison: both defects at once -- inconsistent winding *and* global inversion.

    The composition is the claim: fixing consistency alone can leave the mesh inverted, and
    fixing orientation alone cannot run on inconsistent input, so only the pair together
    produces a volume.
    """
    mesh_tm, mesh_wp = icosahedron
    faces_bad = mesh_tm.faces.copy()
    faces_bad[::2] = faces_bad[::2][:, ::-1]  # inconsistent winding
    faces_bad = faces_bad[:, ::-1]  # then invert everything
    vertices_wp, faces_wp = numpy_to_warp(mesh_tm.vertices, faces_bad, mesh_wp.device)

    assert tw.validation.is_volume(vertices_wp, faces_wp) is False
    repaired_wp = tw.repair.make_normals_outward(vertices_wp, faces_wp)
    assert tw.validation.is_winding_consistent(repaired_wp) is True
    assert tw.validation.is_volume(vertices_wp, repaired_wp) is True

    # Reference: trimesh.repair.fix_normals. On a closed mesh outward orientation is unique,
    # so per-face normals must agree exactly (index representation may differ).
    reference_tm = tm.Trimesh(vertices=mesh_tm.vertices, faces=faces_bad, process=False)
    tm_repair.fix_normals(reference_tm)
    ours_tm = tm.Trimesh(vertices=mesh_tm.vertices, faces=_faces_2d(repaired_wp), process=False)
    assert np.allclose(ours_tm.face_normals, reference_tm.face_normals, atol=1e-5)


def test_make_volume_multibody(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = icosahedron
    vertices_np = np.asarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.asarray(mesh_tm.faces, dtype=np.int64)
    n_vertices = vertices_np.shape[0]
    n_faces = faces_np.shape[0]

    # Body A: as-is (outward). Body B: translated and inverted (inward normals).
    vertices_two = np.vstack([vertices_np, vertices_np + np.array([10.0, 0.0, 0.0])])
    faces_two = np.vstack([faces_np, faces_np[:, ::-1] + n_vertices])
    vertices_wp, faces_wp = numpy_to_warp(vertices_two, faces_two, mesh_wp.device)

    body_a = slice(0, 3 * n_faces)
    body_b = slice(3 * n_faces, 6 * n_faces)

    def _body_is_volume(flat_faces_wp: wp.array, body: slice) -> bool:
        sub = wp.array(flat_faces_wp.numpy()[body], dtype=wp.int32, device=mesh_wp.device)
        return tw.validation.is_volume(vertices_wp, sub)

    # Initially only body B has inward-facing normals.
    assert _body_is_volume(faces_wp, body_a) is True
    assert _body_is_volume(faces_wp, body_b) is False

    # multibody corrects each connected component independently -> both outward.
    repaired_multi = tw.repair.make_volume(vertices_wp, faces_wp, multibody=True)
    assert _body_is_volume(repaired_multi, body_a) is True
    assert _body_is_volume(repaired_multi, body_b) is True

    # The single-body path flips all-or-nothing and cannot orient two opposing bodies outward.
    repaired_single = tw.repair.make_volume(vertices_wp, faces_wp, multibody=False)
    ok_a = _body_is_volume(repaired_single, body_a)
    ok_b = _body_is_volume(repaired_single, body_b)
    assert not (ok_a and ok_b)


def test_make_repairs_empty_mesh(device: str) -> None:
    vertices_wp = wp.empty(0, dtype=wp.vec3, device=device)
    faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    assert tw.repair.make_winding_consistent(faces_wp).shape[0] == 0
    assert tw.repair.make_volume(vertices_wp, faces_wp).shape[0] == 0
    assert tw.repair.make_normals_outward(vertices_wp, faces_wp).shape[0] == 0


# --- degenerate / small triangle removal ----------------------------------------------------


_MERGE_TOL = 1e-8  # matches triwarp.constants.TOLERANCE_MERGE used by triangles.nondegenerate


def _nondegenerate_ref(vertices_np: np.ndarray, faces_np: np.ndarray) -> np.ndarray:
    """CPU mirror of ``triangles.nondegenerate`` (per-edge altitude vs the merge tolerance)."""
    if faces_np.shape[0] == 0:
        return np.empty(0, dtype=bool)
    v0 = vertices_np[faces_np[:, 0]]
    v1 = vertices_np[faces_np[:, 1]]
    v2 = vertices_np[faces_np[:, 2]]
    e0 = v1 - v0
    e1 = v2 - v0
    area = 0.5 * np.linalg.norm(np.cross(e0, e1), axis=1)
    length_e0 = np.linalg.norm(e0, axis=1)
    length_e1 = np.linalg.norm(e1, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        height_e0 = 2.0 * area / length_e0
        height_e1 = 2.0 * area / length_e1
    return (
        (height_e0 > _MERGE_TOL)
        & (height_e1 > _MERGE_TOL)
        & (length_e0 > _MERGE_TOL)
        & (length_e1 > _MERGE_TOL)
    )


def _uf_find(parent: np.ndarray, x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = int(parent[x])
    return int(x)


def _collapse_small_triangles_ref(
    vertices_np: np.ndarray, faces_np: np.ndarray, epsilon: float
) -> np.ndarray:
    """
    Return surviving faces of the reference collapse as sorted position triples ``(m, 3, 3)``.

    Mirrors ``triwarp.repair.collapse_small_triangles``: each iteration flags faces with doubled
    area below ``2 * epsilon * bbd**2``, merges the two endpoints of each flagged triangle's
    shortest edge (union-find, lowest original index as representative), drops faces that become
    degenerate, and repeats to a fixpoint. Comparing surviving triangles as sorted position triples
    is invariant to vertex reindexing and to the representative chosen among near-coincident
    endpoints.
    """
    vertices = np.asarray(vertices_np, dtype=np.float64)
    faces = np.asarray(faces_np, dtype=np.int64).reshape(-1, 3)
    if faces.shape[0] == 0 or vertices.shape[0] == 0:
        return np.empty((0, 3, 3), dtype=np.float64)

    bbd = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    min_dbl_area = 2.0 * epsilon * bbd * bbd

    while faces.shape[0] > 0:
        v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
        dbl_area = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        small = dbl_area < min_dbl_area
        if not small.any():
            break

        n = vertices.shape[0]
        parent = np.arange(n)
        for fi in np.flatnonzero(small):
            tri = faces[fi]
            edge_sq = [
                float(np.sum((vertices[tri[1]] - vertices[tri[0]]) ** 2)),
                float(np.sum((vertices[tri[2]] - vertices[tri[1]]) ** 2)),
                float(np.sum((vertices[tri[0]] - vertices[tri[2]]) ** 2)),
            ]
            e = int(np.argmin(edge_sq))
            ra, rb = _uf_find(parent, int(tri[e])), _uf_find(parent, int(tri[(e + 1) % 3]))
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        labels = np.array([_uf_find(parent, i) for i in range(n)], dtype=np.int64)
        _, inverse = np.unique(labels, return_inverse=True)
        first = np.full(int(inverse.max()) + 1, n, dtype=np.int64)
        for i in range(n):
            first[inverse[i]] = min(first[inverse[i]], i)
        vertices = vertices[first]
        faces = inverse[faces]
        faces = faces[_nondegenerate_ref(vertices, faces)]

    tris = vertices[faces]
    return np.sort(tris, axis=1)


def _sorted_triangle_positions(vertices_np: np.ndarray, faces_np: np.ndarray) -> np.ndarray:
    if faces_np.shape[0] == 0:
        return np.empty((0, 3, 3), dtype=np.float64)
    return np.sort(vertices_np[faces_np], axis=1)


def _triangle_set_close(a: np.ndarray, b: np.ndarray, atol: float = 1e-4) -> bool:
    """Order-independent equality of two ``(m, 3, 3)`` sorted-triangle sets."""
    if a.shape != b.shape:
        return False
    if a.shape[0] == 0:
        return True
    key = lambda t: np.lexsort(t.reshape(t.shape[0], -1).T[::-1])  # noqa: E731
    return bool(np.allclose(a[key(a)], b[key(b)], atol=atol))


# --------------------------------------------------------------------------------------
# non-manifold input builders, shared by remove_non_manifold_faces and split_nonmanifold
# --------------------------------------------------------------------------------------


def _bowtie_np() -> tuple[np.ndarray, np.ndarray]:
    """Two triangles meeting at a single vertex: edge-manifold, vertex-non-manifold."""
    vertices_np = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, -1.0, 0.0]],
        dtype=np.float32,
    )
    return vertices_np, np.array([[0, 1, 2], [0, 3, 4]], dtype=np.int32)


def _three_faces_on_one_edge_np() -> tuple[np.ndarray, np.ndarray]:
    """Build a fan of three faces on edge ``(0, 1)``, all wound the same way round it."""
    vertices_np = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return vertices_np, np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]], dtype=np.int32)


# --------------------------------------------------------------------------------------
# remove_non_manifold_faces
# --------------------------------------------------------------------------------------


def _remove_non_manifold_faces_np(
    vertices_np: np.ndarray, faces_np: np.ndarray, max_iter: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Drop faces carrying a >2-incident edge, iterating, then compact the vertices (numpy)."""
    faces_np = faces_np.copy()
    for _ in range(max_iter):
        if faces_np.shape[0] == 0:
            break
        # ``undirected_edges`` already orders each row min-first, face-major, three per face.
        edges_np = undirected_edges(faces_np)
        _unique_np, inverse_np, counts_np = np.unique(
            edges_np, axis=0, return_inverse=True, return_counts=True
        )
        bad_np = counts_np[inverse_np].reshape(-1, 3).max(axis=1) > 2
        if not bad_np.any():
            break
        faces_np = faces_np[~bad_np]
    used_np = np.unique(faces_np) if faces_np.size else np.zeros(0, dtype=np.int64)
    remap_np = np.full(vertices_np.shape[0], -1, dtype=np.int64)
    remap_np[used_np] = np.arange(used_np.size)
    kept_np = remap_np[faces_np] if faces_np.size else faces_np.reshape(0, 3)
    return vertices_np[used_np], kept_np


def _cascading_non_manifold_np() -> tuple[np.ndarray, np.ndarray]:
    """Build a three-face fan plus a face hanging off it, so one removal pass is not enough."""
    vertices_np, faces_np = _three_faces_on_one_edge_np()
    vertices_np = np.vstack((vertices_np, [[2.0, 0.0, 0.0]])).astype(np.float32)
    return vertices_np, np.vstack((faces_np, [[1, 2, 5]])).astype(np.int32)


def _icosahedron_plus_a_face_on_an_existing_edge_np() -> tuple[np.ndarray, np.ndarray]:
    """Glue one extra face to an existing edge of a closed mesh: most of it must survive."""
    mesh_tm = tm.creation.icosahedron()
    vertices_np = np.vstack((mesh_tm.vertices, [[3.0, 3.0, 3.0]])).astype(np.float32)
    edge_np = mesh_tm.faces[0][:2]
    extra_np = [[edge_np[0], edge_np[1], mesh_tm.vertices.shape[0]]]
    return vertices_np, np.vstack((mesh_tm.faces, extra_np)).astype(np.int32)


@pytest.mark.parametrize(
    ("mesh_kind", "n_surviving"),
    [("three_faces_on_one_edge", 0), ("cascading", 1), ("icosahedron_plus_a_face", 18)],
)
def test_remove_non_manifold_faces_matches_a_numpy_oracle(
    device: str, mesh_kind: str, n_surviving: int
) -> None:
    """
    Class A: the surviving faces and compacted vertices equal the same iteration run in numpy.

    The expected count is pinned per input because the interesting property is *which* faces go:
    the three-face fan loses all three (every one of them carries the 3-incident edge), the
    cascading input needs a second pass to reach the single survivor a one-pass implementation
    would miss, and the icosahedron keeps 18 of its 21 faces -- the substantive case, since two
    empty answers would compare equal.
    """
    builders = {
        "three_faces_on_one_edge": _three_faces_on_one_edge_np,
        "cascading": _cascading_non_manifold_np,
        "icosahedron_plus_a_face": _icosahedron_plus_a_face_on_an_existing_edge_np,
    }
    vertices_np, faces_np = builders[mesh_kind]()
    faces_wp = wp.array(np.ascontiguousarray(faces_np).ravel(), dtype=wp.int32, device=device)
    assert not tw.validation.is_edge_manifold(faces_wp, allow_boundary_edges=True)

    new_vertices_wp, new_faces_wp = tw.repair.remove_non_manifold_faces(
        wp.array(np.ascontiguousarray(vertices_np), dtype=wp.vec3, device=device), faces_wp
    )
    new_faces_np = new_faces_wp.numpy().reshape(-1, 3)
    expected_vertices_np, expected_faces_np = _remove_non_manifold_faces_np(vertices_np, faces_np)

    assert new_faces_np.shape[0] == n_surviving
    assert np.array_equal(new_faces_np, expected_faces_np)
    assert np.allclose(new_vertices_wp.numpy(), expected_vertices_np, rtol=1e-5, atol=1e-5)
    if n_surviving > 0:
        assert tw.validation.is_edge_manifold(new_faces_wp, allow_boundary_edges=True)


def test_remove_non_manifold_faces_leaves_an_edge_manifold_mesh_alone(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """Not a library comparison: an edge-manifold mesh is returned untouched, buffers included."""
    mesh_tm, mesh_wp = icosahedron

    new_vertices_wp, new_faces_wp = tw.repair.remove_non_manifold_faces(
        mesh_wp.points, mesh_wp.indices
    )

    assert new_vertices_wp.shape[0] == mesh_tm.vertices.shape[0]
    assert new_faces_wp.shape[0] == mesh_tm.faces.size
    # Nothing was rebuilt: the early break hands back the caller's own buffers.
    assert new_vertices_wp.ptr == mesh_wp.points.ptr


# --------------------------------------------------------------------------------------
# split_nonmanifold
# --------------------------------------------------------------------------------------


def _split_nonmanifold_wp(
    vertices_np: np.ndarray, faces_np: np.ndarray, device: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run ``split_nonmanifold`` on NumPy input and bring all three results back."""
    vertices_wp = wp.array(np.ascontiguousarray(vertices_np), dtype=wp.vec3, device=device)
    faces_wp = wp.array(np.ascontiguousarray(faces_np).ravel(), dtype=wp.int32, device=device)
    new_vertices_wp, new_faces_wp, source_wp = tw.repair.split_nonmanifold(vertices_wp, faces_wp)
    return new_vertices_wp.numpy(), new_faces_wp.numpy().reshape(-1, 3), source_wp.numpy()


@pytest.mark.parametrize(
    "mesh_kind", ["manifold", "bowtie", "three_faces_on_one_edge", "flipped_face", "boundary"]
)
@pytest.mark.parity("split_nonmanifold", "igl")
def test_split_nonmanifold_matches_igl(
    mesh_kind: str,
    device: str,
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
    icosphere_coarse: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Class B: the same vertex split as ``igl.split_nonmanifold``, up to which copy gets which index.

    Neither library's vertex numbering is meaningful -- both invent copies -- so what is compared is
    the *partition of corners* the two induce, canonicalised by first occurrence with
    [`same_partition`][tests.comparisons.same_partition]. That is exact rather than tolerant, and it
    is the whole answer: given the partition, the face table and the source map follow.

    The two implementations are unrelated. triwarp takes connected components of a corner graph in
    one parallel pass; ``igl::split_nonmanifold`` explodes the mesh into ``3 * n_faces`` singleton
    vertices and greedily re-merges pairs, re-testing manifoldness after each candidate. They agree
    on every input class here, including the two that a naive implementation gets wrong: a fan of
    three faces round one edge must split into three (not two-plus-one), and a face wound against
    its neighbours must be cut free rather than silently kept.

    ``manifold`` is the identity case and is *not* vacuous coverage -- it is asserted to return the
    input unchanged, which an implementation that split on every edge would fail.
    """
    if mesh_kind == "bowtie":
        vertices_np, faces_np = _bowtie_np()
        expected_new = 6
    elif mesh_kind == "three_faces_on_one_edge":
        vertices_np, faces_np = _three_faces_on_one_edge_np()
        expected_new = 9  # every corner its own vertex: no pair of the three is oppositely wound
    elif mesh_kind == "flipped_face":
        mesh_tm, _ = icosahedron
        vertices_np = mesh_tm.vertices.astype(np.float32)
        faces_np = mesh_tm.faces.astype(np.int32).copy()
        faces_np[-1] = faces_np[-1][::-1]
        expected_new = 15  # the flipped face is cut free of all three neighbours
    elif mesh_kind == "boundary":
        sphere_tm, _sphere_wp = icosphere_coarse
        hemisphere_tm = sphere_tm.slice_plane(
            plane_origin=np.zeros(3), plane_normal=np.array([0.0, 0.0, 1.0]), cap=False
        )
        hemisphere_tm.merge_vertices()
        vertices_np = hemisphere_tm.vertices.astype(np.float32)
        faces_np = hemisphere_tm.faces.astype(np.int32)
        expected_new = int(vertices_np.shape[0])  # a boundary is not a reason to split
    else:
        mesh_tm, _ = icosahedron
        vertices_np = mesh_tm.vertices.astype(np.float32)
        faces_np = mesh_tm.faces.astype(np.int32)
        expected_new = int(vertices_np.shape[0])

    new_vertices_np, new_faces_np, source_np = _split_nonmanifold_wp(vertices_np, faces_np, device)

    faces_igl, source_igl = igl.split_nonmanifold(
        np.ascontiguousarray(faces_np, dtype=np.int64).reshape(-1, 3)
    )
    assert source_igl.shape[0] == expected_new, "the reference produced the expected split"
    assert new_vertices_np.shape[0] == expected_new
    assert same_partition(new_faces_np.ravel(), np.asarray(faces_igl).ravel())

    # The split moves nothing and drops nothing: every copy sits on the vertex it came from.
    assert new_faces_np.shape == faces_np.reshape(-1, 3).shape
    assert np.allclose(new_vertices_np, vertices_np[source_np], atol=1e-6)
    assert np.array_equal(np.sort(source_igl), np.sort(source_np.astype(np.int64)))


@pytest.mark.parametrize(
    "mesh_kind", ["bowtie", "three_faces_on_one_edge", "flipped_face", "duplicated_face"]
)
def test_split_nonmanifold_leaves_a_manifold_mesh(
    mesh_kind: str, device: str, icosahedron: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    The post-condition, on inputs that violate it: edge-manifold out, same face count.

    Checked independently of igl because it is the property callers actually depend on, and because
    it covers ``duplicated_face`` -- the one input class where the two libraries deliberately
    disagree on *how much* to split (see the next test), while both must still satisfy this.
    """
    mesh_tm, _ = icosahedron
    if mesh_kind == "bowtie":
        vertices_np, faces_np = _bowtie_np()
    elif mesh_kind == "three_faces_on_one_edge":
        vertices_np, faces_np = _three_faces_on_one_edge_np()
    else:
        vertices_np = mesh_tm.vertices.astype(np.float32)
        faces_np = mesh_tm.faces.astype(np.int32)
        if mesh_kind == "flipped_face":
            faces_np = faces_np.copy()
            faces_np[-1] = faces_np[-1][::-1]
        else:
            faces_np = np.vstack([faces_np, faces_np[:1]])

    faces_wp = wp.array(np.ascontiguousarray(faces_np).ravel(), dtype=wp.int32, device=device)
    vertices_wp = wp.array(np.ascontiguousarray(vertices_np), dtype=wp.vec3, device=device)
    # Each input violates a *different* precondition, and the guards say which -- a bowtie is
    # edge-manifold with a non-manifold vertex, and a flipped face is manifold but not orientable.
    if mesh_kind in ("three_faces_on_one_edge", "duplicated_face"):
        assert not tw.validation.is_edge_manifold(faces_wp)
    elif mesh_kind == "flipped_face":
        assert not tw.validation.is_winding_consistent(faces_wp)
    else:
        assert not tw.validation.is_vertex_manifold(faces_wp)

    new_vertices_wp, new_faces_wp, _source_wp = tw.repair.split_nonmanifold(vertices_wp, faces_wp)

    assert tw.validation.is_edge_manifold(new_faces_wp)
    assert int(new_faces_wp.shape[0]) == int(faces_wp.shape[0])
    assert int(new_vertices_wp.shape[0]) >= int(vertices_wp.shape[0])
    # Idempotent: a second pass has nothing left to split.
    again_vertices_wp, again_faces_wp, _ = tw.repair.split_nonmanifold(
        new_vertices_wp, new_faces_wp
    )
    assert int(again_vertices_wp.shape[0]) == int(new_vertices_wp.shape[0])
    assert np.array_equal(again_faces_wp.numpy(), new_faces_wp.numpy())


def test_split_nonmanifold_splits_a_duplicated_face_further_than_igl(
    device: str, icosahedron: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Class D exemption, pinned: the one documented divergence from igl, on a duplicated face.

    Where an edge carries one half-edge in one direction and several in the other -- what a
    duplicated face produces -- igl keeps one arbitrarily chosen pair joined while triwarp
    splits every copy. Both answers are manifold with the input's face count; triwarp's is
    order-independent and makes no arbitrary choice.

    Asserted as exact counts in both directions, so a change to either library's rule fails here
    rather than quietly altering the output of a repair function.
    """
    mesh_tm, _ = icosahedron
    vertices_np = mesh_tm.vertices.astype(np.float32)
    faces_np = np.vstack([mesh_tm.faces.astype(np.int32), mesh_tm.faces.astype(np.int32)[:1]])

    new_vertices_np, _new_faces_np, _source_np = _split_nonmanifold_wp(
        vertices_np, faces_np, device
    )
    _faces_igl, source_igl = igl.split_nonmanifold(
        np.ascontiguousarray(faces_np, dtype=np.int64).reshape(-1, 3)
    )

    assert new_vertices_np.shape[0] == 18, "all three copies of each shared vertex split apart"
    assert source_igl.shape[0] == 15, "igl keeps one pair of the three joined"
    # Both answers are legal repairs of the same input: manifold, with every face kept.
    assert _new_faces_np.shape[0] == faces_np.shape[0] == np.asarray(_faces_igl).shape[0]
    faces_wp = wp.array(np.ascontiguousarray(faces_np).ravel(), dtype=wp.int32, device=device)
    vertices_wp = wp.array(np.ascontiguousarray(vertices_np), dtype=wp.vec3, device=device)
    assert tw.validation.is_edge_manifold(tw.repair.split_nonmanifold(vertices_wp, faces_wp)[1])
    # And `resolve_duplicated_faces` is *not* the escape hatch here: libigl's cancellation rules
    # cover a +1/-1 imbalance, so a face duplicated in the *same* orientation makes it raise.
    with pytest.raises(ValueError, match="non-orientable duplicate face group"):
        tw.repair.resolve_duplicated_faces(faces_wp)


def test_split_nonmanifold_empty(device: str) -> None:
    """An empty mesh passes through with an empty source map."""
    vertices_wp = wp.empty(0, dtype=wp.vec3, device=device)
    faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    new_vertices_wp, new_faces_wp, source_wp = tw.repair.split_nonmanifold(vertices_wp, faces_wp)
    assert int(new_vertices_wp.shape[0]) == 0
    assert int(new_faces_wp.shape[0]) == 0
    assert int(source_wp.shape[0]) == 0


def test_remove_degenerate_faces_matches_trimesh(device: str) -> None:
    rng = np.random.default_rng(7)
    # Scale ~1e-2: keeps genuine-face altitudes (~1e-2) far above the absolute merge tolerance
    # (1e-8) while the FMA rounding error of a zero-area face's altitude (~1e-7 * scale) stays far
    # below it, so the geometric degeneracy test agrees on CPU and GPU. This scaling is deliberate:
    # a zero-area face's cross product is exactly 0 on CPU but ~1e-8 on CUDA because FMA fusion
    # (fuse_fp) is left ON for triangle_cross to keep the geometry kernels fast, so degenerate-face
    # tests must stay clear of the tolerance boundary rather than disabling fusion module-wide.
    vertices_np = (0.01 * rng.normal(size=(9, 3))).astype(np.float32)
    # Mix of good faces and injected degeneracies (repeated index and coincident vertices).
    faces_np = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 1, 1], [2, 2, 2]], dtype=np.int32)
    vertices_np[8] = vertices_np[6]  # face [6, 7, 8] has two coincident vertices -> zero area

    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    kept_vertices_wp, kept_faces_wp = tw.repair.remove_degenerate_faces(vertices_wp, faces_wp)

    mesh_tm = tm.Trimesh(vertices=vertices_np.astype(np.float64), faces=faces_np, process=False)
    keep_mask_tm = mesh_tm.nondegenerate_faces(height=_MERGE_TOL)

    kept_ref = _sorted_triangle_positions(vertices_np.astype(np.float64), faces_np[keep_mask_tm])
    kept_wp = _sorted_triangle_positions(
        kept_vertices_wp.numpy().astype(np.float64), kept_faces_wp.numpy().reshape(-1, 3)
    )
    assert _triangle_set_close(kept_wp, kept_ref, atol=1e-5)


def test_remove_degenerate_faces_clean_mesh(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Not a library comparison: a clean mesh must lose no face and no vertex.

    The no-op direction, which a threshold that is slightly too aggressive fails while still
    passing every test that feeds it a genuinely degenerate face.
    """
    mesh_tm, mesh_wp = icosahedron
    kept_vertices_wp, kept_faces_wp = tw.repair.remove_degenerate_faces(
        mesh_wp.points, mesh_wp.indices
    )
    assert kept_faces_wp.shape[0] // 3 == mesh_tm.faces.shape[0]
    assert kept_vertices_wp.shape[0] == mesh_tm.vertices.shape[0]


def test_collapse_small_triangles_noop(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Not a library comparison: nothing is below the threshold, so the mesh comes back untouched.

    ``collapse_small_triangles`` is unbound in the igl wheel (section 6), so there is no
    reference for this function at all -- the numpy oracle in this file covers the collapsing
    case.
    """
    mesh_tm, mesh_wp = icosahedron
    vertices_wp, faces_wp = numpy_to_warp(mesh_tm.vertices, mesh_tm.faces, mesh_wp.device)
    out_vertices_wp, out_faces_wp = tw.repair.collapse_small_triangles(
        vertices_wp, faces_wp, epsilon=1e-9
    )
    # A well-shaped mesh has no sub-threshold triangle: faces and vertices are unchanged.
    assert np.array_equal(out_faces_wp.numpy().reshape(-1, 3), mesh_tm.faces)
    assert np.allclose(out_vertices_wp.numpy(), vertices_wp.numpy(), rtol=1e-6, atol=1e-6)


def test_collapse_small_triangles_removes_sliver(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = icosahedron
    vertices_np = mesh_tm.vertices.astype(np.float32)
    a, b = vertices_np[0], vertices_np[1]
    near_edge = a + 0.5 * (b - a) + 1e-6 * (b - a)  # almost on segment a->b
    vertices_np = np.vstack([vertices_np, near_edge]).astype(np.float32)
    faces_np = np.vstack([mesh_tm.faces, [0, 1, vertices_np.shape[0] - 1]]).astype(np.int32)

    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, mesh_wp.device)
    epsilon = 1e-6
    out_vertices_wp, out_faces_wp = tw.repair.collapse_small_triangles(
        vertices_wp, faces_wp, epsilon=epsilon
    )

    # Sliver is gone.
    assert out_faces_wp.shape[0] // 3 < faces_np.shape[0]
    # Invariant (libigl fixpoint guarantee): no surviving triangle is below threshold.
    bbd = tw.bounds.enclosing_diagonal(out_vertices_wp)
    _, areas_wp = tw.triangles.face_normals_and_areas(out_vertices_wp, out_faces_wp)
    assert (2.0 * areas_wp.numpy()).min() >= epsilon * bbd * bbd * (1.0 - 1e-3)
    # Surviving triangle geometry matches the CPU reference.
    ref = _collapse_small_triangles_ref(vertices_np.astype(np.float64), faces_np, epsilon)
    got = _sorted_triangle_positions(
        out_vertices_wp.numpy().astype(np.float64), out_faces_wp.numpy().reshape(-1, 3)
    )
    assert _triangle_set_close(got, ref, atol=1e-4)


def test_collapse_small_triangles_fan_chain(device: str) -> None:
    # A row of stacked slivers sharing a spine: several become small only after neighbours
    # collapse, exercising the fixpoint loop.
    xs = np.linspace(0.0, 1.0, 6)
    top = np.stack([xs, np.full_like(xs, 1e-5), np.zeros_like(xs)], axis=1)
    bottom = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], axis=1)
    vertices_np = np.vstack([top, bottom]).astype(np.float32)
    n = xs.shape[0]
    faces = []
    for i in range(n - 1):
        faces.append([i, i + 1, n + i])  # thin triangles
        faces.append([i + 1, n + i + 1, n + i])
    # Add one well-shaped triangle far away so the mesh is not entirely collapsed.
    big_triangle = [[5.0, 0.0, 0.0], [6.0, 0.0, 0.0], [5.5, 1.0, 0.0]]
    vertices_np = np.vstack([vertices_np, big_triangle]).astype(np.float32)
    big = vertices_np.shape[0]
    faces.append([big - 3, big - 2, big - 1])
    faces_np = np.asarray(faces, dtype=np.int32)

    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    epsilon = 1e-3
    out_vertices_wp, out_faces_wp = tw.repair.collapse_small_triangles(
        vertices_wp, faces_wp, epsilon=epsilon
    )

    bbd = tw.bounds.enclosing_diagonal(out_vertices_wp)
    _, areas_wp = tw.triangles.face_normals_and_areas(out_vertices_wp, out_faces_wp)
    assert (2.0 * areas_wp.numpy()).min() >= epsilon * bbd * bbd * (1.0 - 1e-3)
    ref = _collapse_small_triangles_ref(vertices_np.astype(np.float64), faces_np, epsilon)
    got = _sorted_triangle_positions(
        out_vertices_wp.numpy().astype(np.float64), out_faces_wp.numpy().reshape(-1, 3)
    )
    assert _triangle_set_close(got, ref, atol=1e-3)


def test_collapse_small_triangles_empty(device: str) -> None:
    vertices_wp = wp.empty(0, dtype=wp.vec3, device=device)
    faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    out_vertices_wp, out_faces_wp = tw.repair.collapse_small_triangles(vertices_wp, faces_wp)
    assert out_vertices_wp.shape[0] == 0
    assert out_faces_wp.shape[0] == 0


# ---------------------------------------------------------------------------
# Geometric defects: bad faces, folds and T-vertices (pymeshlab reference)
# ---------------------------------------------------------------------------


def _t_vertex_patch() -> tuple[np.ndarray, np.ndarray]:
    """
    Two quads stitched at different resolutions, so the left one carries a T-vertex.

    Vertex 4 sits on the interior of the edge ``(1, 2)`` of the right quad's triangulation, which is
    exactly a T-junction: the triangle ``(1, 2, 4)`` is a sliver whose apex is on its own long edge.
    Flipping ``(1, 2)`` to ``(3, 4)`` removes it without moving a vertex.
    """
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 2.0, 0.0],  # 2
            [0.0, 2.0, 0.0],  # 3
            [1.0, 1.0, 0.02],  # 4 -- barely off the (1, 2) edge, and off-plane so a flip is legal
            [2.0, 0.0, 0.0],  # 5
            [2.0, 2.0, 0.0],  # 6
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [[0, 1, 3], [1, 2, 3], [1, 5, 4], [4, 5, 6], [4, 6, 2], [1, 4, 2]], dtype=np.int32
    )
    return vertices, faces


def _folded_patch() -> tuple[np.ndarray, np.ndarray]:
    """
    Build a flat two-triangle quad plus a third triangle folded back on top of it.

    Face 2 shares edge ``(1, 3)`` with face 1 and lies almost in the same plane with the *opposite*
    normal, so the dihedral there is ~179 degrees. Every edge still has at most two faces — a third
    face on the folded edge would make it non-manifold, which ``face_adjacency`` drops entirely and
    which would make this fixture measure nothing.
    """
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.1, 0.9, 0.02],  # the folded apex: back over the quad
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [1, 3, 2], [3, 1, 4]], dtype=np.int32)
    return vertices, faces


def _worst_aspect(vertices_wp, faces_wp) -> float:
    return float(
        tw.triangles.face_quality(vertices_wp, faces_wp, metric="aspect_ratio").numpy().max()
    )


def test_bad_face_mask_flags_the_thin_face(device: str) -> None:
    vertices_np, faces_np = _t_vertex_patch()
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    bad_np = tw.repair.bad_face_mask(vertices_wp, faces_wp, min_quality=0.2).numpy()
    # The sliver (1, 4, 2) is the last face, and it is the only thin one.
    assert bad_np[-1]
    assert bad_np.sum() == 1


def test_bad_face_mask_flags_the_fold(device: str) -> None:
    """Only the *culprit* of a fold is flagged, not the good face on the other side of the edge."""
    vertices_np, faces_np = _folded_patch()
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    folded_np = tw.repair.bad_face_mask(
        vertices_wp, faces_wp, min_quality=None, max_fold_angle=160.0
    ).numpy()
    assert np.array_equal(folded_np.astype(bool), np.array([False, False, True]))


def test_bad_face_mask_flags_the_misoriented_face(device: str) -> None:
    """One face wound the wrong way in a consistent patch reads 180 degrees off the consensus."""
    n = 5
    i_grid, j_grid = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    vertices_np = np.column_stack(
        [i_grid.ravel().astype(np.float64), j_grid.ravel().astype(np.float64), np.zeros(n * n)]
    )
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            faces += [[a, a + 1, a + n + 1], [a, a + n + 1, a + n]]
    faces_np = np.ascontiguousarray(faces, dtype=np.int32)
    # A grid rather than a three-triangle strip: the criterion compares a face against the *sum* of
    # its neighbours' normals, and on a strip the flipped face's own neighbours have only it to
    # agree with, so they would be flagged too.
    target = faces_np.shape[0] // 2
    faces_np[target] = faces_np[target][::-1]

    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    bad_np = tw.repair.bad_face_mask(
        vertices_wp, faces_wp, min_quality=None, max_normal_angle=60.0
    ).numpy()
    assert np.array_equal(np.flatnonzero(bad_np), np.array([target]))


def test_bad_face_mask_all_criteria_disabled(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    bad_np = tw.repair.bad_face_mask(
        mesh_wp.points, mesh_wp.indices, min_quality=None, max_normal_angle=None
    ).numpy()
    assert not bad_np.any()


def test_bad_face_mask_invalid(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    with pytest.raises(ValueError, match="max_fold_angle must be in"):
        tw.repair.bad_face_mask(mesh_wp.points, mesh_wp.indices, max_fold_angle=200.0)
    with pytest.raises(ValueError, match="max_normal_angle must be in"):
        tw.repair.bad_face_mask(mesh_wp.points, mesh_wp.indices, max_normal_angle=0.0)


def test_bad_face_mask_empty(device: str) -> None:
    vertices_wp = wp.zeros(0, dtype=wp.vec3, device=device)
    faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    assert tw.repair.bad_face_mask(vertices_wp, faces_wp).shape == (0,)


def test_remove_folded_faces_drops_the_fold(device: str) -> None:
    vertices_np, faces_np = _folded_patch()
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    kept_vertices_wp, kept_faces_wp = tw.repair.remove_folded_faces(vertices_wp, faces_wp)
    # Only the fold goes, so the flat quad it folded over survives intact.
    assert int(kept_faces_wp.shape[0]) // 3 == 2
    # The folded apex was referenced only by the dropped face, so it is gone too.
    assert int(kept_vertices_wp.shape[0]) == vertices_np.shape[0] - 1
    assert (
        not tw.repair.bad_face_mask(
            kept_vertices_wp, kept_faces_wp, min_quality=None, max_fold_angle=160.0
        )
        .numpy()
        .any()
    )


@pytest.mark.parity("bad_face_mask", "pymeshlab")
def test_remove_folded_faces_matches_pymeshlab_on_which_faces_are_folded(device: str) -> None:
    """
    Class B (compare detection): MeshLab flips the fold where this deletes it.

    ``compute_selection_bad_faces(select_folded_faces=True)`` is the same dihedral criterion at the
    same threshold, and it reports a selection rather than editing the mesh — which makes it the
    oracle for ``bad_face_mask``'s fold gate even though ``meshing_remove_folded_faces`` and
    ``remove_folded_faces`` then do different things with the answer.
    """
    vertices_np, faces_np = _folded_patch()
    meshset_pml = ml.MeshSet()
    meshset_pml.add_mesh(ml.Mesh(vertices_np, np.ascontiguousarray(faces_np, dtype=np.int32)))
    meshset_pml.compute_selection_bad_faces(
        usear=False, usenf=False, select_folded_faces=True, folded_faces_angle_threshold=160.0
    )
    selected_pml = meshset_pml.current_mesh().face_selection_array()

    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    folded_np = tw.repair.bad_face_mask(
        vertices_wp, faces_wp, min_quality=None, max_fold_angle=160.0
    ).numpy()
    assert np.array_equal(folded_np.astype(bool), selected_pml.astype(bool))


def test_remove_folded_faces_leaves_a_clean_mesh_alone(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    _mesh_tm, mesh_wp = icosahedron
    kept_vertices_wp, kept_faces_wp = tw.repair.remove_folded_faces(mesh_wp.points, mesh_wp.indices)
    assert int(kept_faces_wp.shape[0]) == int(mesh_wp.indices.shape[0])
    assert int(kept_vertices_wp.shape[0]) == int(mesh_wp.points.shape[0])


def test_remove_folded_faces_empty(device: str) -> None:
    vertices_wp = wp.zeros(0, dtype=wp.vec3, device=device)
    faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    out_vertices_wp, out_faces_wp = tw.repair.remove_folded_faces(vertices_wp, faces_wp)
    assert int(out_vertices_wp.shape[0]) == 0
    assert int(out_faces_wp.shape[0]) == 0


def test_remove_t_vertices_flips_the_sliver(device: str) -> None:
    """The sliver goes, the face count and the vertices stay, and the patch stays manifold."""
    vertices_np, faces_np = _t_vertex_patch()
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)
    before = _worst_aspect(vertices_wp, faces_wp)
    assert before > 40.0  # the fixture really does carry a T-vertex sliver

    flipped_wp = tw.repair.remove_t_vertices(vertices_wp, faces_wp, threshold=40.0)
    assert _worst_aspect(vertices_wp, flipped_wp) < before
    assert int(flipped_wp.shape[0]) == int(faces_wp.shape[0])
    assert tw.validation.is_winding_consistent(flipped_wp)
    assert tw.validation.is_edge_manifold(flipped_wp)


@pytest.mark.parametrize("mesh_kind", ["t_vertex_patch", "clean_icosphere"])
@pytest.mark.parity("remove_t_vertices", "pymeshlab")
def test_remove_t_vertices_matches_pymeshlab(device: str, mesh_kind: str) -> None:
    """
    Class B: MeshLab's ``Edge Flip`` mode picks the *identical* flips, face for face.

    ``meshing_remove_t_vertices`` rewrites the topology in place rather than returning a face
    buffer, so the transform is reading ``face_matrix()`` back and comparing the two as sorted face
    sets (neither library defines a face or corner order). ``method="Edge Flip"`` selects the
    flip-only mode -- its other modes split edges and would change the vertex count -- and
    ``repeat=True`` is MeshLab's own iterate-to-convergence, which is what triwarp's ``max_iter``
    passes are. Both are what the benchmark passes.

    Two-sided by construction, so neither answer can be reached by a constant: on the T-vertex patch
    both flip the sliver and land on the same 6 faces, **different from the input's 6**; on a clean
    icosphere neither touches anything. Vertex counts are unchanged on both sides in both cases,
    which is the check that MeshLab really took the flip path and not a split.
    """
    if mesh_kind == "t_vertex_patch":
        vertices_np, faces_np = _t_vertex_patch()
        expect_change = True
    else:
        sphere_tm = tm.creation.icosphere(subdivisions=3)
        vertices_np, faces_np = np.asarray(sphere_tm.vertices), np.asarray(sphere_tm.faces)
        expect_change = False
    vertices_wp, faces_wp = numpy_to_warp(vertices_np, faces_np, device)

    meshset_pml = trimesh_to_pymeshlab(tm.Trimesh(vertices_np, faces_np, process=False))
    meshset_pml.meshing_remove_t_vertices(method="Edge Flip", threshold=40.0, repeat=True)
    assert meshset_pml.current_mesh().vertex_number() == vertices_np.shape[0]
    faces_pml = np.asarray(meshset_pml.current_mesh().face_matrix())

    flipped_np = tw.repair.remove_t_vertices(vertices_wp, faces_wp, threshold=40.0).numpy()

    def face_set(faces: np.ndarray) -> np.ndarray:
        sorted_np = np.sort(np.asarray(faces).reshape(-1, 3), axis=1)
        return sorted_np[np.lexsort(sorted_np.T[::-1])]

    assert np.array_equal(face_set(flipped_np), face_set(faces_pml))
    assert (not np.array_equal(face_set(flipped_np), face_set(faces_np))) is expect_change


def test_remove_t_vertices_leaves_a_clean_mesh_alone(
    device: str, icosphere: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """No triangle of an icosphere is anywhere near the threshold, so nothing may move."""
    sphere_tm, _sphere_wp = icosphere
    vertices_wp, faces_wp = numpy_to_warp(
        np.asarray(sphere_tm.vertices), np.asarray(sphere_tm.faces), device
    )
    flipped_wp = tw.repair.remove_t_vertices(vertices_wp, faces_wp)
    assert np.array_equal(flipped_wp.numpy(), faces_wp.numpy())


def test_remove_t_vertices_invalid(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    with pytest.raises(ValueError, match="aspect_threshold must be positive"):
        tw.repair.remove_t_vertices(mesh_wp.points, mesh_wp.indices, threshold=0.0)


def test_remove_t_vertices_empty(device: str) -> None:
    vertices_wp = wp.zeros(0, dtype=wp.vec3, device=device)
    faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    assert int(tw.repair.remove_t_vertices(vertices_wp, faces_wp).shape[0]) == 0
