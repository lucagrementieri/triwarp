"""Regression tests for ``triwarp.edges`` against Trimesh (CPU reference)."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh as tm
import trimesh.grouping as tm_grouping
import warp as wp

import triwarp as tw

_MESHES = ["icosahedron", "cave_cube", "hemisphere", "half_torus"]

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _faces_np_to_wp(faces_np: np.ndarray, device: str) -> wp.array:
    return wp.array(faces_np.flatten().astype(np.int32), dtype=wp.int32, device=device)


def _vertices_np_to_wp(vertices_np: np.ndarray, device: str) -> wp.array:
    return wp.array(
        np.ascontiguousarray(vertices_np, dtype=np.float32), dtype=wp.vec3, device=device
    )


# ---------------------------------------------------------------------------
# edges
# ---------------------------------------------------------------------------


def test_edges(device: str) -> None:
    rng = np.random.default_rng(0)
    faces_np = rng.integers(0, 50, size=(20, 3), dtype=np.int32)
    edges_np = tm.geometry.faces_to_edges(faces_np)

    faces_wp = _faces_np_to_wp(faces_np, device)
    edges_wp = tw.edges.faces_to_edges(faces_wp)
    assert np.array_equal(edges_wp.numpy(), edges_np)


def test_edges_sorted(device: str) -> None:
    rng = np.random.default_rng(1)
    faces_np = rng.integers(0, 50, size=(20, 3), dtype=np.int32)
    edges_np = np.sort(tm.geometry.faces_to_edges(faces_np), axis=1)

    faces_wp = _faces_np_to_wp(faces_np, device)
    edges_wp = tw.edges.faces_to_edges(faces_wp, sorted=True)
    assert np.array_equal(edges_wp.numpy(), edges_np)


def test_edges_empty(device: str) -> None:
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    assert tw.edges.faces_to_edges(faces_wp).shape == (0, 2)
    assert tw.edges.faces_to_edges(faces_wp, sorted=True).shape == (0, 2)


# ---------------------------------------------------------------------------
# edges_face
# ---------------------------------------------------------------------------


def test_edges_face(device: str) -> None:
    rng = np.random.default_rng(3)
    n_faces = 24
    faces_np = rng.integers(0, 50, size=(n_faces, 3), dtype=np.int32)

    faces_wp = _faces_np_to_wp(faces_np, device)
    face_idx_wp = tw.edges.edges_face(faces_wp)

    # each face f contributes edges at positions 3*f, 3*f+1, 3*f+2
    expected = np.repeat(np.arange(n_faces, dtype=np.int32), 3)
    assert np.array_equal(face_idx_wp.numpy(), expected)


def test_edges_face_empty(device: str) -> None:
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    assert tw.edges.edges_face(faces_wp).shape == (0,)


# ---------------------------------------------------------------------------
# edges_unique
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
def test_edges_unique(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    unique_idx_tm, _ = tm_grouping.unique_rows(np.sort(mesh_tm.edges, axis=1))
    unique_edges_tm = np.sort(mesh_tm.edges, axis=1)[unique_idx_tm]

    unique_edges_wp, _ = tw.edges.edges_unique(mesh_wp.indices)
    unique_edges_wp_np = unique_edges_wp.numpy()

    order_tm = np.lexsort((unique_edges_tm[:, 1], unique_edges_tm[:, 0]))
    order_wp = np.lexsort((unique_edges_wp_np[:, 1], unique_edges_wp_np[:, 0]))
    assert np.array_equal(unique_edges_wp_np[order_wp], unique_edges_tm[order_tm])


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
def test_edges_unique_inverse(request: pytest.FixtureRequest, mesh_name: str) -> None:
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    unique_edges_wp, inverse_wp = tw.edges.edges_unique(mesh_wp.indices)
    edges_sorted_wp = tw.edges.faces_to_edges(mesh_wp.indices, sorted=True)

    # unique_edges[inverse] must reconstruct edges_sorted
    reconstructed = unique_edges_wp.numpy()[inverse_wp.numpy()]
    assert np.array_equal(reconstructed, edges_sorted_wp.numpy())


def test_edges_unique_empty(device: str) -> None:
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    unique_wp, inverse_wp = tw.edges.edges_unique(faces_wp)
    assert unique_wp.shape == (0, 2)
    assert inverse_wp.shape == (0,)


# ---------------------------------------------------------------------------
# edges_unique_inverse (standalone)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
def test_edges_unique_inverse_standalone(request: pytest.FixtureRequest, mesh_name: str) -> None:
    _, mesh_wp = request.getfixturevalue(mesh_name)

    _unique_edges_wp, inverse_from_unique = tw.edges.edges_unique(mesh_wp.indices)
    inverse_standalone = tw.edges.edges_unique_inverse(mesh_wp.indices)
    assert np.array_equal(inverse_from_unique.numpy(), inverse_standalone.numpy())


# ---------------------------------------------------------------------------
# edges_unique_length
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
def test_edges_unique_length(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    unique_idx_tm, _ = tm_grouping.unique_rows(np.sort(mesh_tm.edges, axis=1))
    unique_edges_tm = np.sort(mesh_tm.edges, axis=1)[unique_idx_tm]
    verts_np = mesh_tm.vertices.astype(np.float32)
    lengths_tm = np.linalg.norm(
        verts_np[unique_edges_tm[:, 1]] - verts_np[unique_edges_tm[:, 0]], axis=1
    )

    vertices_wp = _vertices_np_to_wp(mesh_tm.vertices, mesh_wp.device)
    lengths_wp = tw.edges.edges_unique_length(vertices_wp, mesh_wp.indices)
    lengths_wp_np = lengths_wp.numpy()

    # lengths are unordered — sort both for comparison
    assert np.allclose(np.sort(lengths_wp_np), np.sort(lengths_tm), rtol=1e-4, atol=1e-4)


def test_edges_unique_length_precomputed(device: str) -> None:
    rng = np.random.default_rng(7)
    verts_np = rng.random((30, 3), dtype=np.float32)
    faces_np = rng.integers(0, 30, size=(10, 3), dtype=np.int32)
    faces_wp = _faces_np_to_wp(faces_np, device)
    vertices_wp = _vertices_np_to_wp(verts_np, device)

    unique_edges_wp, _ = tw.edges.edges_unique(faces_wp)
    lengths_via_precomputed = tw.edges.edges_unique_length(
        vertices_wp, faces_wp, unique_edges=unique_edges_wp
    )
    lengths_fresh = tw.edges.edges_unique_length(vertices_wp, faces_wp)
    assert np.allclose(lengths_via_precomputed.numpy(), lengths_fresh.numpy(), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# edges_length
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
def test_edges_length(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    edges_np = mesh_tm.edges
    verts_np = mesh_tm.vertices.astype(np.float32)
    lengths_tm = np.linalg.norm(verts_np[edges_np[:, 1]] - verts_np[edges_np[:, 0]], axis=1)

    vertices_wp = _vertices_np_to_wp(mesh_tm.vertices, mesh_wp.device)
    lengths_wp = tw.edges.edges_length(vertices_wp, mesh_wp.indices)

    assert np.allclose(np.sort(lengths_wp.numpy()), np.sort(lengths_tm), rtol=1e-4, atol=1e-4)


def test_edges_length_precomputed(device: str) -> None:
    rng = np.random.default_rng(8)
    verts_np = rng.random((30, 3), dtype=np.float32)
    faces_np = rng.integers(0, 30, size=(10, 3), dtype=np.int32)
    faces_wp = _faces_np_to_wp(faces_np, device)
    vertices_wp = _vertices_np_to_wp(verts_np, device)

    edges_in_wp = tw.edges.faces_to_edges(faces_wp)
    lengths_via_precomputed = tw.edges.edges_length(vertices_wp, faces_wp, edges_in=edges_in_wp)
    lengths_fresh = tw.edges.edges_length(vertices_wp, faces_wp)
    assert np.allclose(lengths_via_precomputed.numpy(), lengths_fresh.numpy(), rtol=1e-5, atol=1e-5)


def test_edges_length_empty(device: str) -> None:
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    verts_wp = wp.array(np.zeros((0, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    assert tw.edges.edges_length(verts_wp, faces_wp).shape == (0,)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
def test_mean_edge_length(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    verts_np = mesh_tm.vertices.astype(np.float64)
    tri = verts_np[mesh_tm.faces]
    avg_edge_np = float(np.linalg.norm(tri - tri[:, [1, 2, 0]], axis=2).mean())

    vertices_wp = _vertices_np_to_wp(mesh_tm.vertices, mesh_wp.device)
    avg_edge_wp = tw.edges.mean_edge_length(vertices_wp, mesh_wp.indices)

    assert np.allclose(avg_edge_wp, avg_edge_np, rtol=1e-4, atol=1e-4)


def test_mean_edge_length_empty(device: str) -> None:
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    verts_wp = wp.array(np.zeros((0, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    assert tw.edges.mean_edge_length(verts_wp, faces_wp) == 0.0


# --- face_edge_lengths ----------------------------------------------------------------
@pytest.mark.parametrize("mesh_name", _MESHES)
def test_face_edge_lengths_are_the_opposite_edges(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    lengths = tw.edges.face_edge_lengths(mesh_wp.points, mesh_wp.indices).numpy()

    triangles = np.asarray(mesh_tm.vertices)[np.asarray(mesh_tm.faces)]
    expected = np.stack(
        [
            np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
            np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
        ],
        axis=1,
    )
    assert np.allclose(lengths, expected, rtol=1e-5, atol=1e-5)
