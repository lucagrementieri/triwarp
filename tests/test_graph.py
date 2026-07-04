"""Regression tests for ``triwarp.graph`` against Trimesh (CPU reference)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import trimesh as tm
import warp as wp

import triwarp as tw
import triwarp.typing as twt


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
        unshared_wp.numpy()[order_unshared],
        unshared_precomputed_wp.numpy()[order_unshared_precomputed],
    )


def test_face_adjacency_unshared_empty(device: str) -> None:
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    unshared_wp = tw.graph.face_adjacency_unshared(faces_wp)
    assert unshared_wp.shape == (0, 2)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
def test_face_adjacency_angles(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    adjacency_tm = mesh_tm.face_adjacency
    angles_tm = mesh_tm.face_adjacency_angles

    adjacency_wp = tw.graph.face_adjacency(mesh_wp.indices)
    angles_wp = tw.graph.face_adjacency_angles(
        mesh_wp.points, mesh_wp.indices, face_adjacency=adjacency_wp
    )

    adjacency_wp_np = adjacency_wp.numpy()
    angles_wp_np = angles_wp.numpy()
    angles_wp_lookup = {
        (int(row[0]), int(row[1])): float(angles_wp_np[i]) for i, row in enumerate(adjacency_wp_np)
    }
    angles_tm_lookup = {
        (int(row[0]), int(row[1])): float(angles_tm[i]) for i, row in enumerate(adjacency_tm)
    }
    assert angles_wp_lookup.keys() == angles_tm_lookup.keys()
    for key, angle_tm in angles_tm_lookup.items():
        angle_wp = angles_wp_lookup[key]
        assert np.isclose(angle_wp, angle_tm, rtol=1e-4, atol=5e-4)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus"])
def test_face_adjacency_angles_precomputed(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    adjacency_wp = tw.graph.face_adjacency(mesh_wp.indices)
    face_normals_wp, _ = tw.triangles.face_normals_and_areas(mesh_wp.points, mesh_wp.indices)

    angles_all_wp = tw.graph.face_adjacency_angles(
        mesh_wp.points, mesh_wp.indices, face_adjacency=adjacency_wp
    )
    angles_precomputed_wp = tw.graph.face_adjacency_angles(
        mesh_wp.points, mesh_wp.indices, face_adjacency=adjacency_wp, face_normals=face_normals_wp
    )
    assert np.allclose(angles_all_wp.numpy(), angles_precomputed_wp.numpy(), rtol=1e-5, atol=1e-5)

    adjacency_tm = mesh_tm.face_adjacency
    angles_tm = mesh_tm.face_adjacency_angles
    adjacency_wp_np = adjacency_wp.numpy()
    angles_precomputed_np = angles_precomputed_wp.numpy()
    angles_tm_lookup = {
        (int(row[0]), int(row[1])): float(angles_tm[i]) for i, row in enumerate(adjacency_tm)
    }
    angles_precomputed_lookup = {
        (int(row[0]), int(row[1])): float(angles_precomputed_np[i])
        for i, row in enumerate(adjacency_wp_np)
    }
    for key, angle_tm in angles_tm_lookup.items():
        assert np.isclose(angles_precomputed_lookup[key], angle_tm, rtol=1e-4, atol=5e-4)


def test_face_adjacency_angles_empty(device: str) -> None:
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    vertices_wp = wp.empty(0, dtype=wp.vec3, device=device)
    angles_wp = tw.graph.face_adjacency_angles(vertices_wp, faces_wp)
    assert angles_wp.shape == (0,)


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


def test_bfs_random(device: str) -> None:
    rng = np.random.default_rng(7)
    node_count = 48
    pairs = rng.integers(0, node_count, size=(150, 2), dtype=np.int32)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    edges_np = np.unique(np.sort(pairs, axis=1), axis=0).astype(np.int32)
    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)

    for source in (0, node_count // 2, node_count - 1):
        order_wp, parents_wp, distances_wp = tw.graph.bfs_from_edges(
            edges_wp, source, node_count=node_count
        )
        order_np, parents_np, distances_np = _scipy_bfs(edges_np, node_count, source)
        assert np.array_equal(order_wp.numpy(), order_np)
        assert np.array_equal(parents_wp.numpy(), parents_np)
        assert np.array_equal(distances_wp.numpy(), distances_np)


def test_bfs_path_graph(device: str) -> None:
    n = 1024
    edges_np = np.stack([np.arange(n - 1, dtype=np.int32), np.arange(1, n, dtype=np.int32)], axis=1)
    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)

    order_wp, parents_wp, distances_wp = tw.graph.bfs_from_edges(edges_wp, 0, node_count=n)
    assert np.array_equal(order_wp.numpy(), np.arange(n, dtype=np.int32))
    assert np.array_equal(distances_wp.numpy(), np.arange(n, dtype=np.int32))
    parents_exp = np.concatenate([[-1], np.arange(n - 1)]).astype(np.int32)
    assert np.array_equal(parents_wp.numpy(), parents_exp)


def test_bfs_star_graph(device: str) -> None:
    n = 256
    hub = 0
    leaves = np.arange(1, n, dtype=np.int32)
    edges_np = np.stack([np.full(n - 1, hub, dtype=np.int32), leaves], axis=1)
    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)

    for source in (hub, n - 1):
        order_wp, parents_wp, distances_wp = tw.graph.bfs_from_edges(edges_wp, source, node_count=n)
        order_np, parents_np, distances_np = _scipy_bfs(edges_np, n, source)
        assert np.array_equal(order_wp.numpy(), order_np)
        assert np.array_equal(parents_wp.numpy(), parents_np)
        assert np.array_equal(distances_wp.numpy(), distances_np)


def test_bfs_disconnected(device: str) -> None:
    # Two disjoint triangles: {0,1,2} and {3,4,5}.
    edges_np = np.array([[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5]], dtype=np.int32)
    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)
    node_count = 6

    order_wp, parents_wp, distances_wp = tw.graph.bfs_from_edges(edges_wp, 0, node_count=node_count)
    order_np, parents_np, distances_np = _scipy_bfs(edges_np, node_count, 0)

    assert np.array_equal(order_wp.numpy(), order_np)
    assert order_wp.shape[0] == 3  # only the first triangle is reachable
    assert np.array_equal(parents_wp.numpy(), parents_np)
    assert np.array_equal(distances_wp.numpy(), distances_np)
    # Nodes 3,4,5 are unreachable from 0.
    assert np.array_equal(parents_wp.numpy()[3:], np.array([-1, -1, -1], dtype=np.int32))
    assert np.array_equal(distances_wp.numpy()[3:], np.array([-1, -1, -1], dtype=np.int32))


def test_bfs_single_node(device: str) -> None:
    edges_wp = twt.empty_int32_2d((0, 2), device=device)
    order_wp, parents_wp, distances_wp = tw.graph.bfs_from_edges(edges_wp, 0, node_count=1)
    assert np.array_equal(order_wp.numpy(), np.array([0], dtype=np.int32))
    assert np.array_equal(parents_wp.numpy(), np.array([-1], dtype=np.int32))
    assert np.array_equal(distances_wp.numpy(), np.array([0], dtype=np.int32))


def test_bfs_empty_graph(device: str) -> None:
    node_count = 8
    source = 3
    edges_wp = twt.empty_int32_2d((0, 2), device=device)
    order_wp, parents_wp, distances_wp = tw.graph.bfs_from_edges(
        edges_wp, source, node_count=node_count
    )
    assert np.array_equal(order_wp.numpy(), np.array([source], dtype=np.int32))
    parents_exp = np.full(node_count, -1, dtype=np.int32)
    distances_exp = np.full(node_count, -1, dtype=np.int32)
    distances_exp[source] = 0
    assert np.array_equal(parents_wp.numpy(), parents_exp)
    assert np.array_equal(distances_wp.numpy(), distances_exp)


def test_bfs_from_edges_node_count_inference(device: str) -> None:
    edges_np = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)
    order_wp, parents_wp, distances_wp = tw.graph.bfs_from_edges(edges_wp, 0)
    # node_count inferred as max + 1 == 4.
    assert parents_wp.shape[0] == 4
    assert distances_wp.shape[0] == 4
    assert np.array_equal(order_wp.numpy(), np.array([0, 1, 2, 3], dtype=np.int32))


def test_bfs_csr_columns_ascending(device: str) -> None:
    # Locks the precondition that makes serial BFS match scipy's neighbor visitation order.
    edges_np = np.array([[0, 2], [0, 1], [1, 3], [2, 3], [0, 3]], dtype=np.int32)
    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)
    adjacency = tw.graph.edges_to_csr(4, edges_wp)
    offsets = adjacency.offsets.numpy()
    columns = adjacency.columns.numpy()
    for v in range(4):
        row = columns[offsets[v] : offsets[v + 1]]
        assert np.array_equal(row, np.sort(row))


def test_bfs_source_out_of_range(device: str) -> None:
    edges_wp = wp.array(np.array([[0, 1]], dtype=np.int32), dtype=wp.int32, device=device)
    with pytest.raises(ValueError, match="source must be in"):
        tw.graph.bfs_from_edges(edges_wp, source=5, node_count=2)
    with pytest.raises(ValueError, match="source must be in"):
        tw.graph.bfs_from_edges(edges_wp, source=-1, node_count=2)


def test_bfs_from_edges_index_out_of_range(device: str) -> None:
    edges_wp = wp.array(np.array([[0, 9]], dtype=np.int32), dtype=wp.int32, device=device)
    with pytest.raises(ValueError, match="edge indices must lie in"):
        tw.graph.bfs_from_edges(edges_wp, source=0, node_count=4)


def test_bfs_from_edges_negative_node_count(device: str) -> None:
    edges_wp = wp.array(np.array([[0, 1]], dtype=np.int32), dtype=wp.int32, device=device)
    with pytest.raises(ValueError, match="node_count must be non-negative"):
        tw.graph.bfs_from_edges(edges_wp, source=0, node_count=-1)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere", "cave_cube"])
def test_bfs_on_mesh(request: pytest.FixtureRequest, mesh_name: str) -> None:
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    unique_edges_wp, n = _mesh_vertex_edges(mesh_wp)
    edges_np = unique_edges_wp.numpy()

    for source in (0, n // 2, n - 1):
        order_wp, parents_wp, distances_wp = tw.graph.bfs_from_edges(
            unique_edges_wp, source, node_count=n
        )
        order_np, parents_np, distances_np = _scipy_bfs(edges_np, n, source)
        assert np.array_equal(order_wp.numpy(), order_np)
        assert np.array_equal(parents_wp.numpy(), parents_np)
        assert np.array_equal(distances_wp.numpy(), distances_np)


def test_bfs_multi_source_matches_single(request: pytest.FixtureRequest) -> None:
    _mesh_tm, mesh_wp = request.getfixturevalue("icosahedron")
    unique_edges_wp, n = _mesh_vertex_edges(mesh_wp)
    edges_np = unique_edges_wp.numpy()
    device = mesh_wp.device

    sources = [0, n // 3, n - 1]
    sources_wp = wp.array(np.array(sources, dtype=np.int32), dtype=wp.int32, device=device)
    adjacency = tw.graph.edges_to_csr(n, unique_edges_wp)
    neighbors_wp, offsets_wp = tw.graph.bfs_multi_source(adjacency, sources_wp)

    neighbors_np = neighbors_wp.numpy()
    offsets_np = offsets_wp.numpy()
    for k, source in enumerate(sources):
        start = int(offsets_np[k])
        end = int(offsets_np[k + 1]) if k + 1 < len(offsets_np) else len(neighbors_np)
        reachable_wp = set(neighbors_np[start:end].tolist())
        reachable_np = set(_scipy_bfs(edges_np, n, source)[0].tolist())
        assert reachable_wp == reachable_np
        # First entry of each source's slice is the source itself (BFS order).
        assert int(neighbors_np[start]) == source


def test_bfs_multi_source_empty_sources(device: str) -> None:
    edges_wp = wp.array(np.array([[0, 1], [1, 2]], dtype=np.int32), dtype=wp.int32, device=device)
    adjacency = tw.graph.edges_to_csr(3, edges_wp)
    sources_wp = wp.empty(0, dtype=wp.int32, device=device)
    neighbors_wp, offsets_wp = tw.graph.bfs_multi_source(adjacency, sources_wp)
    assert neighbors_wp.shape[0] == 0
    assert offsets_wp.shape[0] == 0


def test_bfs_multi_source_source_out_of_range(device: str) -> None:
    edges_wp = wp.array(np.array([[0, 1], [1, 2]], dtype=np.int32), dtype=wp.int32, device=device)
    adjacency = tw.graph.edges_to_csr(3, edges_wp)
    sources_wp = wp.array(np.array([0, 7], dtype=np.int32), dtype=wp.int32, device=device)
    with pytest.raises(ValueError, match="source indices must lie in"):
        tw.graph.bfs_multi_source(adjacency, sources_wp)


def test_bfs_multi_source_overflow_warns(device: str) -> None:
    # A path longer than the fixed per-source scratch capacity overflows from an endpoint source.
    n = 700  # > kernel_bfs._PER_SOURCE_MAX_NEIGHBORS (512)
    edges_np = np.stack([np.arange(n - 1, dtype=np.int32), np.arange(1, n, dtype=np.int32)], axis=1)
    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)
    adjacency = tw.graph.edges_to_csr(n, edges_wp)
    sources_wp = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=device)
    with pytest.warns(UserWarning, match="capacity breaches"):
        tw.graph.bfs_multi_source(adjacency, sources_wp)


def _scipy_bfs(
    edges_np: np.ndarray, node_count: int, source: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build reference ``(order, parents, distances)`` for :func:`triwarp.graph.bfs` semantics."""
    if len(edges_np) == 0:
        matrix = sp.csr_matrix((node_count, node_count), dtype=np.int8)
    else:
        row = np.concatenate([edges_np[:, 0], edges_np[:, 1]])
        col = np.concatenate([edges_np[:, 1], edges_np[:, 0]])
        data = np.ones(len(row), dtype=np.int8)
        matrix = sp.coo_matrix((data, (row, col)), shape=(node_count, node_count)).tocsr()
    order, pred = csgraph.breadth_first_order(
        matrix, source, directed=False, return_predecessors=True
    )
    dist = csgraph.shortest_path(matrix, directed=False, unweighted=True, indices=source)
    parents = np.where(pred == -9999, -1, pred).astype(np.int32)
    distances = np.where(np.isinf(dist), -1, dist).astype(np.int32)
    return order.astype(np.int32), parents, distances


def _mesh_vertex_edges(mesh_wp: wp.Mesh) -> tuple[wp.array, int]:
    """Return the unique undirected vertex edges and vertex count for a Warp mesh."""
    n = int(mesh_wp.points.shape[0])
    unique_edges, _ = tw.edges.edges_unique(mesh_wp.indices, n_vertices=n)
    return unique_edges, n
