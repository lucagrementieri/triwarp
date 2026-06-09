"""Regression tests for ``triwarp.selection`` against Trimesh (CPU reference)."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

import trimesh as tm
import triwarp as tw


def test_submesh_from_face_indices_empty(device: str) -> None:
    vertices_wp = wp.array(np.zeros((4, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    faces_wp = wp.array(np.array([0, 1, 2, 0, 2, 3], dtype=np.int32), dtype=wp.int32, device=device)
    face_indices_wp = wp.empty(0, dtype=wp.int32, device=device)
    submesh_vertices_wp, submesh_faces_wp = tw.selection.submesh_from_face_indices(
        vertices_wp, faces_wp, face_indices_wp
    )
    assert submesh_vertices_wp.shape == (0,)
    assert submesh_faces_wp.shape == (0,)


def test_submesh_from_face_indices_single_face(request: pytest.FixtureRequest) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue("icosahedron")
    face_indices = wp.array([0], dtype=wp.int32, device=mesh_wp.points.device)
    submesh_tm = tm.util.submesh(mesh_tm, [[0]], repair=False, append=False)[0]
    submesh_vertices_wp, submesh_faces_wp = tw.selection.submesh_from_face_indices(
        mesh_wp.points, mesh_wp.indices, face_indices, unique_indices=True
    )
    assert submesh_vertices_wp.shape == (3,)
    assert submesh_faces_wp.shape == (3,)
    assert np.allclose(submesh_vertices_wp.numpy(), submesh_tm.vertices)
    assert np.array_equal(submesh_faces_wp.numpy(), submesh_tm.faces.reshape(-1))


def test_submesh_from_face_indices_duplicated(request: pytest.FixtureRequest) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue("icosahedron")
    face_indices_np = np.array([0, 0, 0, 5, 5, 12, 12], dtype=np.int32)
    face_indices = wp.array(face_indices_np, dtype=wp.int32, device=mesh_wp.points.device)
    submesh_tm = tm.util.submesh(mesh_tm, [face_indices_np], repair=False, append=False)[0]
    submesh_vertices_wp, submesh_faces_wp = tw.selection.submesh_from_face_indices(
        mesh_wp.points, mesh_wp.indices, face_indices
    )
    assert submesh_vertices_wp.shape[0] <= len(np.unique(face_indices_np)) * 3
    assert submesh_faces_wp.shape == (len(face_indices_np) * 3,)
    assert np.allclose(submesh_vertices_wp.numpy(), submesh_tm.vertices)
    assert np.array_equal(submesh_faces_wp.numpy(), submesh_tm.faces.reshape(-1))


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
def test_submesh_from_face_indices_random_faces(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    rng = np.random.default_rng(42)
    n_faces = mesh_tm.faces.shape[0]
    n_select = max(1, n_faces // 3)
    face_indices_np = rng.choice(n_faces, size=n_select, replace=False).astype(np.int32)
    face_indices = wp.array(face_indices_np, dtype=wp.int32, device=mesh_wp.points.device)
    submesh_tm = tm.util.submesh(mesh_tm, [face_indices_np], repair=False, append=False)[0]
    submesh_vertices_wp, submesh_faces_wp = tw.selection.submesh_from_face_indices(
        mesh_wp.points, mesh_wp.indices, face_indices
    )
    assert np.allclose(submesh_vertices_wp.numpy(), submesh_tm.vertices)
    assert np.array_equal(submesh_faces_wp.numpy(), submesh_tm.faces.reshape(-1))


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
def test_submesh_from_face_indices_all_faces(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_faces = mesh_tm.faces.shape[0]
    face_indices_np = np.arange(n_faces, dtype=np.int32)
    face_indices = wp.array(face_indices_np, dtype=wp.int32, device=mesh_wp.points.device)
    submesh_tm = tm.util.submesh(mesh_tm, [face_indices_np], repair=False, append=False)[0]
    submesh_vertices_wp, submesh_faces_wp = tw.selection.submesh_from_face_indices(
        mesh_wp.points, mesh_wp.indices, face_indices, unique_indices=True
    )
    assert submesh_vertices_wp.shape[0] <= mesh_tm.vertices.shape[0]
    assert submesh_faces_wp.shape[0] == n_faces * 3
    assert np.allclose(submesh_vertices_wp.numpy(), submesh_tm.vertices)
    assert np.array_equal(submesh_faces_wp.numpy(), submesh_tm.faces.reshape(-1))


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus"])
def test_submesh_from_face_mask(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    rng = np.random.default_rng(7)
    n_faces = mesh_tm.faces.shape[0]
    face_mask_np = rng.choice([False, True], size=n_faces, replace=True)
    face_mask_np[rng.integers(0, n_faces)] = True
    face_indices_np = np.flatnonzero(face_mask_np).astype(np.int32)

    submesh_tm = tm.util.submesh(mesh_tm, [face_indices_np], repair=False, append=False)[0]
    face_mask = wp.array(face_mask_np, dtype=wp.bool, device=mesh_wp.points.device)
    got_vertices_wp, got_faces_wp = tw.selection.submesh_from_face_mask(mesh_wp.points, mesh_wp.indices, face_mask)
    exp_vertices_wp, exp_faces_wp = tw.selection.submesh_from_face_indices(
        mesh_wp.points,
        mesh_wp.indices,
        wp.array(face_indices_np, dtype=wp.int32, device=mesh_wp.points.device),
        unique_indices=True,
    )
    assert np.allclose(got_vertices_wp.numpy(), exp_vertices_wp.numpy())
    assert np.array_equal(got_faces_wp.numpy(), exp_faces_wp.numpy())
    assert np.allclose(got_vertices_wp.numpy(), submesh_tm.vertices)
    assert np.array_equal(got_faces_wp.numpy(), submesh_tm.faces.reshape(-1))


@pytest.mark.parametrize("face_mode", ["all", "any"])
def test_face_indices_from_vertex_indices(request: pytest.FixtureRequest, face_mode: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue("icosahedron")
    rng = np.random.default_rng(11)
    n_vertices = mesh_tm.vertices.shape[0]
    vertex_indices_np = rng.choice(n_vertices, size=max(1, n_vertices // 4), replace=False).astype(np.int32)
    vertex_indices = wp.array(vertex_indices_np, dtype=wp.int32, device=mesh_wp.points.device)

    face_indices_wp = tw.selection.face_indices_from_vertex_indices(
        mesh_wp.indices, vertex_indices, face_mode=face_mode
    )
    face_indices_ref_np = _face_indices_from_vertex_indices_np(mesh_tm.faces, vertex_indices_np, face_mode=face_mode)
    assert np.array_equal(face_indices_wp.numpy(), face_indices_ref_np)


def test_face_indices_from_vertex_indices_empty() -> None:
    faces_wp = wp.array(np.array([0, 1, 2], dtype=np.int32), dtype=wp.int32, device="cpu")
    vertex_indices_wp = wp.empty(0, dtype=wp.int32, device="cpu")
    face_indices_wp = tw.selection.face_indices_from_vertex_indices(faces_wp, vertex_indices_wp)
    assert face_indices_wp.shape == (0,)


@pytest.mark.parametrize("face_mode", ["all", "any"])
def test_submesh_from_vertex_indices(request: pytest.FixtureRequest, face_mode: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue("half_torus")
    rng = np.random.default_rng(13)
    n_vertices = mesh_tm.vertices.shape[0]
    vertex_indices_np = rng.choice(n_vertices, size=max(3, n_vertices // 5), replace=False).astype(np.int32)
    vertex_indices = wp.array(vertex_indices_np, dtype=wp.int32, device=mesh_wp.points.device)

    face_indices_np = _face_indices_from_vertex_indices_np(mesh_tm.faces, vertex_indices_np, face_mode=face_mode)
    submesh_tm = tm.util.submesh(mesh_tm, [face_indices_np], repair=False, append=False)[0]
    got_vertices_wp, got_faces_wp = tw.selection.submesh_from_vertex_indices(
        mesh_wp.points, mesh_wp.indices, vertex_indices, face_mode=face_mode
    )
    assert np.allclose(got_vertices_wp.numpy(), submesh_tm.vertices)
    assert np.array_equal(got_faces_wp.numpy(), submesh_tm.faces.reshape(-1))


@pytest.mark.parametrize("face_mode", ["all", "any"])
def test_submesh_from_vertex_mask(request: pytest.FixtureRequest, face_mode: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue("hemisphere")
    rng = np.random.default_rng(17)
    n_vertices = mesh_tm.vertices.shape[0]
    vertex_mask_np = np.zeros(n_vertices, dtype=bool)
    selected = rng.choice(n_vertices, size=max(3, n_vertices // 5), replace=False)
    vertex_mask_np[selected] = True
    vertex_mask = wp.array(vertex_mask_np, dtype=wp.bool, device=mesh_wp.points.device)

    got_vertices_wp, got_faces_wp = tw.selection.submesh_from_vertex_mask(
        mesh_wp.points, mesh_wp.indices, vertex_mask, face_mode=face_mode
    )
    exp_vertices_wp, exp_faces_wp = tw.selection.submesh_from_vertex_indices(
        mesh_wp.points,
        mesh_wp.indices,
        wp.array(selected.astype(np.int32), dtype=wp.int32, device=mesh_wp.points.device),
        face_mode=face_mode,
    )
    assert np.allclose(got_vertices_wp.numpy(), exp_vertices_wp.numpy())
    assert np.array_equal(got_faces_wp.numpy(), exp_faces_wp.numpy())


def _face_indices_from_vertex_indices_np(
    faces_np: np.ndarray, vertex_indices_np: np.ndarray, *, face_mode: str
) -> np.ndarray:
    vertex_hit = np.isin(faces_np, vertex_indices_np)
    if face_mode == "all":
        face_hit = np.all(vertex_hit, axis=1)
    else:
        face_hit = np.any(vertex_hit, axis=1)
    return np.flatnonzero(face_hit).astype(np.int32)
