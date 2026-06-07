"""Regression tests for ``triwarp.graph`` against Trimesh (CPU reference)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import warp as wp

import trimesh as tm
import triwarp as tw
import triwarp.typing as twt


def test_faces_to_edges(device: str) -> None:
    rng = np.random.default_rng(42)
    n_faces = 16
    faces_np = rng.integers(0, 100, size=(n_faces, 3), dtype=np.int32)
    edges_np = tm.geometry.faces_to_edges(faces_np)

    faces_wp = wp.array(faces_np.flatten(), dtype=wp.int32, device=device)
    edges_wp = tw.graph.faces_to_edges(faces_wp)
    assert np.array_equal(edges_wp.numpy(), edges_np)


def test_faces_to_edges_sorted(device: str) -> None:
    rng = np.random.default_rng(42)
    n_faces = 16
    faces_np = rng.integers(0, 100, size=(n_faces, 3), dtype=np.int32)
    edges_np = np.sort(tm.geometry.faces_to_edges(faces_np), axis=1)

    faces_wp = wp.array(faces_np.flatten(), dtype=wp.int32, device=device)
    edges_wp = tw.graph.faces_to_edges(faces_wp, sorted=True)
    assert np.array_equal(edges_wp.numpy(), edges_np)


def test_faces_to_edges_empty(device: str) -> None:
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    edges_wp = tw.graph.faces_to_edges(faces_wp)
    assert edges_wp.shape == (0, 2)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
def test_face_adjacency(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    adjacency_tm = mesh_tm.face_adjacency
    adjacency_edges_tm = mesh_tm.face_adjacency_edges
    adjacency_wp, adjacency_edges_wp = tw.graph.face_adjacency(mesh_wp.indices, return_edges=True)

    order_tm = np.lexsort((adjacency_tm[:, 1], adjacency_tm[:, 0]))
    order_wp = np.lexsort((adjacency_wp.numpy()[:, 1], adjacency_wp.numpy()[:, 0]))
    assert np.array_equal(adjacency_wp.numpy()[order_wp], adjacency_tm[order_tm])
    assert np.array_equal(adjacency_edges_wp.numpy()[order_wp], adjacency_edges_tm[order_tm])


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
def test_face_adjacency_unshared(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    adjacency_tm = mesh_tm.face_adjacency
    unshared_tm = mesh_tm.face_adjacency_unshared.astype(np.int32)

    adjacency_wp, adjacency_edges_wp = tw.graph.face_adjacency(mesh_wp.indices, return_edges=True)
    unshared_precomputed_wp = tw.graph.face_adjacency_unshared(
        mesh_wp.indices, face_adjacency=adjacency_wp, face_adjacency_edges=adjacency_edges_wp
    )
    adjacency_wp_np = adjacency_wp.numpy()
    order_tm = np.lexsort((adjacency_tm[:, 1], adjacency_tm[:, 0]))
    order_wp = np.lexsort((adjacency_wp_np[:, 1], adjacency_wp_np[:, 0]))

    assert np.array_equal(unshared_precomputed_wp.numpy()[order_wp], unshared_tm[order_tm])

    unshared_wp = tw.graph.face_adjacency_unshared(mesh_wp.indices)
    order_unshared = np.lexsort((unshared_wp.numpy()[:, 1], unshared_wp.numpy()[:, 0]))
    order_unshared_precomputed = np.lexsort(
        (unshared_precomputed_wp.numpy()[:, 1], unshared_precomputed_wp.numpy()[:, 0])
    )
    assert np.array_equal(
        unshared_wp.numpy()[order_unshared], unshared_precomputed_wp.numpy()[order_unshared_precomputed]
    )


def test_is_watertight_icosahedron(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = icosahedron
    edges_wp = tw.graph.faces_to_edges(mesh_wp.indices)
    edges_sorted_wp = tw.graph.faces_to_edges(mesh_wp.indices, sorted=True)

    watertight_wp, winding_wp = tw.graph.is_watertight(edges_wp, edges_sorted_wp)
    watertight_tm, winding_tm = tm.graph.is_watertight(mesh_tm.edges, mesh_tm.edges_sorted)

    assert watertight_wp == watertight_tm
    assert winding_wp == winding_tm
    assert watertight_wp
    assert winding_wp


def test_is_watertight_hemisphere(hemisphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = hemisphere
    edges_wp = tw.graph.faces_to_edges(mesh_wp.indices)
    edges_sorted_wp = tw.graph.faces_to_edges(mesh_wp.indices, sorted=True)

    watertight_wp, winding_wp = tw.graph.is_watertight(edges_wp, edges_sorted_wp)
    watertight_tm, winding_tm = tm.graph.is_watertight(mesh_tm.edges, mesh_tm.edges_sorted)

    assert watertight_wp == watertight_tm
    assert winding_wp == winding_tm
    assert not watertight_wp


def test_is_watertight_icosahedron_reversed_faces(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = icosahedron
    rng = np.random.default_rng(17)
    faces_np = mesh_tm.faces.copy()
    flip_mask = rng.random(len(faces_np)) < 0.5
    faces_np[flip_mask] = faces_np[flip_mask, ::-1]

    faces_wp = wp.array(faces_np.reshape(-1), dtype=wp.int32, device=mesh_wp.points.device)
    edges_wp = tw.graph.faces_to_edges(faces_wp)
    edges_sorted_wp = tw.graph.faces_to_edges(faces_wp, sorted=True)

    watertight_wp, winding_wp = tw.graph.is_watertight(edges_wp, edges_sorted_wp)
    edges_tm = tm.geometry.faces_to_edges(faces_np)
    edges_sorted_tm = np.sort(edges_tm, axis=1)
    watertight_tm, winding_tm = tm.graph.is_watertight(edges_tm, edges_sorted_tm)

    assert watertight_wp == watertight_tm
    assert winding_wp == winding_tm
    assert watertight_wp
    assert not winding_wp


def test_face_adjacency_unshared_empty(device: str) -> None:
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    unshared_wp = tw.graph.face_adjacency_unshared(faces_wp)
    assert unshared_wp.shape == (0, 2)


def test_concatenate_meshes(request: pytest.FixtureRequest) -> None:
    mesh_a_tm, mesh_a_wp = request.getfixturevalue("icosahedron")
    mesh_b_tm, mesh_b_wp = request.getfixturevalue("hemisphere")
    mesh_c_tm, mesh_c_wp = request.getfixturevalue("half_torus")

    concat_tm = tm.util.concatenate([mesh_a_tm, mesh_b_tm, mesh_c_tm])
    concat_vertices_wp, concat_faces_wp = tw.graph.concatenate(
        [
            (mesh_a_wp.points, mesh_a_wp.indices),
            (mesh_b_wp.points, mesh_b_wp.indices),
            (mesh_c_wp.points, mesh_c_wp.indices),
        ]
    )
    assert np.allclose(concat_vertices_wp.numpy(), concat_tm.vertices)
    assert np.array_equal(concat_faces_wp.numpy(), concat_tm.faces.reshape(-1))


def test_concatenate_single_mesh(request: pytest.FixtureRequest) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue("icosahedron")
    concat_vertices_wp, concat_faces_wp = tw.graph.concatenate([(mesh_wp.points, mesh_wp.indices)])
    assert np.allclose(concat_vertices_wp.numpy(), mesh_tm.vertices)
    assert np.array_equal(concat_faces_wp.numpy(), mesh_tm.faces.reshape(-1))


def test_concatenate_empty() -> None:
    vertices_wp, faces_wp = tw.graph.concatenate([])
    assert vertices_wp.shape == (0,)
    assert faces_wp.shape == (0,)


def test_split_meshes(request: pytest.FixtureRequest) -> None:
    mesh_a_tm, mesh_a_wp = request.getfixturevalue("icosahedron")
    mesh_b_tm, mesh_b_wp = request.getfixturevalue("hemisphere")
    mesh_c_tm, mesh_c_wp = request.getfixturevalue("half_torus")

    meshes_wp = [
        (mesh_a_wp.points, mesh_a_wp.indices),
        (mesh_b_wp.points, mesh_b_wp.indices),
        (mesh_c_wp.points, mesh_c_wp.indices),
    ]
    concat_vertices_wp, concat_faces_wp = tw.graph.concatenate(meshes_wp)

    split_wp = tw.graph.split(concat_vertices_wp, concat_faces_wp)
    assert len(split_wp) == 3

    roundtrip_vertices_wp, roundtrip_faces_wp = tw.graph.concatenate(split_wp)
    assert np.allclose(roundtrip_vertices_wp.numpy(), concat_vertices_wp.numpy())
    assert np.array_equal(roundtrip_faces_wp.numpy(), concat_faces_wp.numpy())

    split_wp_sorted = sorted(split_wp, key=lambda mesh: mesh[1].shape[0])
    meshes_tm_sorted = sorted([mesh_a_tm, mesh_b_tm, mesh_c_tm], key=lambda mesh: len(mesh.faces))
    for (vertices_wp, faces_wp), mesh_tm in zip(split_wp_sorted, meshes_tm_sorted, strict=True):
        assert np.allclose(vertices_wp.numpy(), mesh_tm.vertices, rtol=1e-5, atol=1e-5)
        assert np.array_equal(faces_wp.numpy(), mesh_tm.faces.reshape(-1))


def test_split_empty(device: str) -> None:
    vertices_wp = wp.empty(0, dtype=wp.vec3, device=device)
    faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    assert tw.graph.split(vertices_wp, faces_wp) == []


def test_edges_to_csr_roundtrip(device: str) -> None:
    edges_np = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int32)
    node_count = 3
    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)
    adjacency = tw.graph.edges_to_csr(node_count, edges_wp)
    offsets = adjacency.offsets.numpy()  # pyright: ignore[reportAttributeAccessIssue]
    indices = adjacency.columns.numpy()  # pyright: ignore[reportAttributeAccessIssue]

    assert adjacency.nrow == node_count  # pyright: ignore[reportAttributeAccessIssue]
    assert adjacency.ncol == node_count  # pyright: ignore[reportAttributeAccessIssue]
    assert adjacency.block_shape == (1, 1)
    assert offsets[0] == 0
    assert offsets[-1] == len(indices)
    assert offsets.shape[0] == node_count + 1

    neighbors: dict[int, set[int]] = {i: set() for i in range(node_count)}
    for a, b in edges_np:
        neighbors[int(a)].add(int(b))
        neighbors[int(b)].add(int(a))

    for v in range(node_count):
        row = indices[offsets[v] : offsets[v + 1]]
        assert set(row.tolist()) == neighbors[v]


def test_connected_component_labels_random(device: str) -> None:
    rng = np.random.default_rng(7)
    node_count = 64
    n_edges = 200
    edges_np = rng.integers(0, node_count, size=(n_edges, 2), dtype=np.int32)

    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)
    labels_wp = tw.graph.connected_component_labels_from_edges(edges_wp, node_count=node_count)
    labels_np = _scipy_component_labels(edges_np, node_count)

    assert _same_partition(labels_wp.numpy(), labels_np)


def test_connected_component_labels_empty_edges(device: str) -> None:
    node_count = 10
    edges_wp = twt.empty_int32_2d((0, 2), device=device)
    labels_wp = tw.graph.connected_component_labels_from_edges(edges_wp, node_count=node_count)
    labels_exp = _scipy_component_labels(np.empty((0, 2), dtype=np.int32), node_count)
    assert np.array_equal(labels_wp.numpy(), labels_exp)


def test_connected_component_labels_zero_nodes(device: str) -> None:
    edges_wp = twt.empty_int32_2d((0, 2), device=device)
    labels_wp = tw.graph.connected_component_labels_from_edges(edges_wp, node_count=0)
    assert labels_wp.shape == (0,)


def test_connected_component_labels_path_graph(device: str) -> None:
    n = 2048
    edges_np = np.stack([np.arange(n - 1, dtype=np.int32), np.arange(1, n, dtype=np.int32)], axis=1)
    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)
    labels_wp = tw.graph.connected_component_labels_from_edges(edges_wp, node_count=n)
    labels_exp = _scipy_component_labels(edges_np, n)
    assert _same_partition(labels_wp.numpy(), labels_exp)


def test_connected_component_labels_star_graph(device: str) -> None:
    n = 512
    hub = 0
    leaves = np.arange(1, n, dtype=np.int32)
    edges_np = np.stack([np.full(n - 1, hub, dtype=np.int32), leaves], axis=1)
    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)
    labels_wp = tw.graph.connected_component_labels_from_edges(edges_wp, node_count=n)
    labels_exp = _scipy_component_labels(edges_np, n)
    assert _same_partition(labels_wp.numpy(), labels_exp)


def test_face_connected_component_labels(request: pytest.FixtureRequest) -> None:
    mesh_a_tm, mesh_a_wp = request.getfixturevalue("icosahedron")
    mesh_b_tm, mesh_b_wp = request.getfixturevalue("hemisphere")
    mesh_c_tm, mesh_c_wp = request.getfixturevalue("half_torus")

    concat_tm = tm.util.concatenate([mesh_a_tm, mesh_b_tm, mesh_c_tm])
    _, concat_faces_wp = tw.graph.concatenate(
        [
            (mesh_a_wp.points, mesh_a_wp.indices),
            (mesh_b_wp.points, mesh_b_wp.indices),
            (mesh_c_wp.points, mesh_c_wp.indices),
        ]
    )
    face_labels_wp = tw.graph.face_connected_component_labels(concat_faces_wp)
    n_faces = concat_tm.faces.shape[0]
    face_labels_tm = _scipy_component_labels(concat_tm.face_adjacency.astype(np.int32), n_faces)
    assert _same_partition(face_labels_wp.numpy(), face_labels_tm)


def _scipy_component_labels(edges: np.ndarray, node_count: int) -> np.ndarray:
    if node_count == 0:
        return np.array([], dtype=np.int32)
    if len(edges) == 0:
        return np.arange(node_count, dtype=np.int32)
    row = edges[:, 0]
    col = edges[:, 1]
    data = np.ones(len(edges), dtype=np.int8)
    matrix = sp.coo_matrix((data, (row, col)), shape=(node_count, node_count))
    matrix = matrix + matrix.T
    _n_comp, labels = csgraph.connected_components(matrix, directed=False)
    return labels.astype(np.int32)


def _same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    same_a = a[:, None] == a[None, :]
    same_b = b[:, None] == b[None, :]
    return bool(np.array_equal(same_a, same_b))
