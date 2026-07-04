"""Regression tests for ``triwarp.repair`` against libigl."""

from __future__ import annotations

import igl
import numpy as np
import pytest
import trimesh as tm
import trimesh.repair as tm_repair
import warp as wp

import triwarp as tw


def _to_wp_mesh(vertices_np: np.ndarray, faces_np: np.ndarray, device: str):
    vertices_wp = wp.array(
        np.ascontiguousarray(vertices_np.astype(np.float32)), dtype=wp.vec3, device=device
    )
    faces_wp = wp.array(
        np.ascontiguousarray(faces_np.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device
    )
    return vertices_wp, faces_wp


def _sort_rows(rows: np.ndarray) -> np.ndarray:
    if rows.size == 0:
        return rows
    return rows[np.lexsort(rows.T[::-1])]


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
    sv_igl, _, svj_igl, sf_igl = (
        igl.remove_duplicate_vertices(vertices_np, faces_np, epsilon)
        if faces_np is not None
        else (*igl.remove_duplicate_vertices(vertices_np, epsilon), None)
    )
    assert np.array_equal(_sort_rows(sv_wp), _sort_rows(sv_igl))
    for i, vertex in enumerate(vertices_np):
        assert np.allclose(sv_wp[svj_wp[i]], vertex, rtol=1e-5, atol=1e-5)
    if sf_wp is not None and faces_np is not None and sf_igl is not None:
        tri_wp = sv_wp[sf_wp.reshape(-1, 3)]
        tri_igl = sv_igl[sf_igl]
        assert np.allclose(np.sort(tri_wp, axis=1), np.sort(tri_igl, axis=1), rtol=1e-5, atol=1e-5)


def test_remove_unreferenced_identity(icosahedron, device: str):
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


def test_remove_unreferenced_extra_vertices(device: str):
    rng = np.random.default_rng(0)
    vertices_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    faces_np = np.array([[0, 1, 2]], dtype=np.int32)
    extra = rng.normal(size=(4, 3))
    vertices_full_np = np.vstack([vertices_np, extra])

    vertices_wp, faces_wp = _to_wp_mesh(vertices_full_np, faces_np, device)
    nv_wp, nf_wp, remap_wp, inverse_wp = tw.repair.remove_unreferenced_vertices(
        vertices_wp, faces_wp, return_inverse=True
    )

    nv_igl, nf_igl, remap_igl, inverse_igl = igl.remove_unreferenced(vertices_full_np, faces_np)

    assert np.allclose(nv_wp.numpy(), nv_igl, rtol=1e-5, atol=1e-5)
    assert np.array_equal(nf_wp.numpy().reshape(-1, 3), nf_igl)
    assert np.array_equal(remap_wp.numpy(), remap_igl.ravel())
    assert np.array_equal(inverse_wp.numpy(), inverse_igl.ravel())


def test_remove_unreferenced_sentinel(device: str):
    vertices_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    faces_np = np.array([[0, 1, -1], [0, 2, 1]], dtype=np.int32)
    vertices_wp, faces_wp = _to_wp_mesh(vertices_np, faces_np, device)

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


def test_remove_duplicate_vertices_faces(device: str):
    vertices_np = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    faces_np = np.array([[0, 1, 3], [2, 1, 3]], dtype=np.int32)
    vertices_wp, faces_wp = _to_wp_mesh(vertices_np, faces_np, device)

    sv_wp, _, svj_wp, sf_wp = tw.repair.remove_duplicated_vertices(vertices_wp, faces_wp, 0.0)
    _assert_duplicate_vertices_match(
        vertices_np, sv_wp.numpy(), svj_wp.numpy(), sf_wp.numpy(), faces_np
    )


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


def _faces_2d(faces_wp: wp.array) -> np.ndarray:
    return faces_wp.numpy().reshape(-1, 3)


def test_make_winding_consistent_repairs_flipped(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = icosahedron
    faces_flipped = mesh_tm.faces.copy()
    faces_flipped[::2] = faces_flipped[::2][:, ::-1]  # reverse winding of half the faces
    _, faces_wp = _to_wp_mesh(mesh_tm.vertices, faces_flipped, mesh_wp.device)

    assert tw.characteristics.is_winding_consistent(faces_wp) is False
    repaired_wp = tw.repair.make_winding_consistent(faces_wp)
    assert tw.characteristics.is_winding_consistent(repaired_wp) is True

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


def test_make_winding_consistent_idempotent(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = icosahedron
    repaired_wp = tw.repair.make_winding_consistent(mesh_wp.indices)
    # Already consistently wound: output identical to input.
    assert np.array_equal(_faces_2d(repaired_wp), _faces_2d(mesh_wp.indices))


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube"])
def test_make_volume_repairs_inversion(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    faces_inward = mesh_tm.faces[:, ::-1].copy()  # reverse every face -> inward normals
    vertices_wp, faces_wp = _to_wp_mesh(mesh_tm.vertices, faces_inward, mesh_wp.device)

    assert tw.characteristics.is_volume(vertices_wp, faces_wp) is False
    repaired_wp = tw.repair.make_volume(vertices_wp, faces_wp)
    assert tw.characteristics.is_volume(vertices_wp, repaired_wp) is True

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


def test_make_normals_consistent(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = icosahedron
    faces_bad = mesh_tm.faces.copy()
    faces_bad[::2] = faces_bad[::2][:, ::-1]  # inconsistent winding
    faces_bad = faces_bad[:, ::-1]  # then invert everything
    vertices_wp, faces_wp = _to_wp_mesh(mesh_tm.vertices, faces_bad, mesh_wp.device)

    assert tw.characteristics.is_volume(vertices_wp, faces_wp) is False
    repaired_wp = tw.repair.make_normals_consistent(vertices_wp, faces_wp)
    assert tw.characteristics.is_winding_consistent(repaired_wp) is True
    assert tw.characteristics.is_volume(vertices_wp, repaired_wp) is True

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
    vertices_wp, faces_wp = _to_wp_mesh(vertices_two, faces_two, mesh_wp.device)

    body_a = slice(0, 3 * n_faces)
    body_b = slice(3 * n_faces, 6 * n_faces)

    def _body_is_volume(flat_faces_wp: wp.array, body: slice) -> bool:
        sub = wp.array(flat_faces_wp.numpy()[body], dtype=wp.int32, device=mesh_wp.device)
        return tw.characteristics.is_volume(vertices_wp, sub)

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
    assert tw.repair.make_normals_consistent(vertices_wp, faces_wp).shape[0] == 0
