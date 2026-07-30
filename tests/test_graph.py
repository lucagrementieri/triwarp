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
    adjacency_wp, adjacency_edges_wp = tw.adjacency.face_adjacency(
        mesh_wp.indices, return_edges=True
    )

    order_tm = np.lexsort((adjacency_tm[:, 1], adjacency_tm[:, 0]))
    order_wp = np.lexsort((adjacency_wp.numpy()[:, 1], adjacency_wp.numpy()[:, 0]))
    assert np.array_equal(adjacency_wp.numpy()[order_wp], adjacency_tm[order_tm])
    assert np.array_equal(adjacency_edges_wp.numpy()[order_wp], adjacency_edges_tm[order_tm])


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
def test_face_adjacency_unshared(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    adjacency_tm = mesh_tm.face_adjacency
    unshared_tm = mesh_tm.face_adjacency_unshared.astype(np.int32)

    adjacency_wp, adjacency_edges_wp = tw.adjacency.face_adjacency(
        mesh_wp.indices, return_edges=True
    )
    unshared_precomputed_wp = tw.adjacency.face_adjacency_unshared(
        mesh_wp.indices, face_adjacency=adjacency_wp, face_adjacency_edges=adjacency_edges_wp
    )
    adjacency_wp_np = adjacency_wp.numpy()
    order_tm = np.lexsort((adjacency_tm[:, 1], adjacency_tm[:, 0]))
    order_wp = np.lexsort((adjacency_wp_np[:, 1], adjacency_wp_np[:, 0]))

    assert np.array_equal(unshared_precomputed_wp.numpy()[order_wp], unshared_tm[order_tm])

    unshared_wp = tw.adjacency.face_adjacency_unshared(mesh_wp.indices)
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
    unshared_wp = tw.adjacency.face_adjacency_unshared(faces_wp)
    assert unshared_wp.shape == (0, 2)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
def test_face_adjacency_angles(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    adjacency_tm = mesh_tm.face_adjacency
    angles_tm = mesh_tm.face_adjacency_angles

    adjacency_wp = tw.adjacency.face_adjacency(mesh_wp.indices)
    angles_wp = tw.adjacency.face_adjacency_angles(
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
    adjacency_wp = tw.adjacency.face_adjacency(mesh_wp.indices)
    face_normals_wp, _ = tw.triangles.face_normals_and_areas(mesh_wp.points, mesh_wp.indices)

    angles_all_wp = tw.adjacency.face_adjacency_angles(
        mesh_wp.points, mesh_wp.indices, face_adjacency=adjacency_wp
    )
    angles_precomputed_wp = tw.adjacency.face_adjacency_angles(
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
    angles_wp = tw.adjacency.face_adjacency_angles(vertices_wp, faces_wp)
    assert angles_wp.shape == (0,)


@pytest.mark.parity("concatenate", "trimesh")
def test_concatenate_meshes(request: pytest.FixtureRequest) -> None:
    mesh_a_tm, mesh_a_wp = request.getfixturevalue("icosahedron")
    mesh_b_tm, mesh_b_wp = request.getfixturevalue("hemisphere")
    mesh_c_tm, mesh_c_wp = request.getfixturevalue("half_torus")

    concat_tm = tm.util.concatenate([mesh_a_tm, mesh_b_tm, mesh_c_tm])
    concat_vertices_wp, concat_faces_wp = tw.combine.concatenate(
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
    concat_vertices_wp, concat_faces_wp = tw.combine.concatenate(
        [(mesh_wp.points, mesh_wp.indices)]
    )
    assert np.allclose(concat_vertices_wp.numpy(), mesh_tm.vertices)
    assert np.array_equal(concat_faces_wp.numpy(), mesh_tm.faces.reshape(-1))


def test_concatenate_empty() -> None:
    vertices_wp, faces_wp = tw.combine.concatenate([])
    assert vertices_wp.shape == (0,)
    assert faces_wp.shape == (0,)


@pytest.mark.parity("split", "trimesh")
def test_split_meshes(request: pytest.FixtureRequest) -> None:
    mesh_a_tm, mesh_a_wp = request.getfixturevalue("icosahedron")
    mesh_b_tm, mesh_b_wp = request.getfixturevalue("hemisphere")
    mesh_c_tm, mesh_c_wp = request.getfixturevalue("half_torus")

    meshes_wp = [
        (mesh_a_wp.points, mesh_a_wp.indices),
        (mesh_b_wp.points, mesh_b_wp.indices),
        (mesh_c_wp.points, mesh_c_wp.indices),
    ]
    concat_vertices_wp, concat_faces_wp = tw.combine.concatenate(meshes_wp)

    split_wp = tw.combine.split(concat_vertices_wp, concat_faces_wp)
    assert len(split_wp) == 3

    roundtrip_vertices_wp, roundtrip_faces_wp = tw.combine.concatenate(split_wp)
    assert np.allclose(roundtrip_vertices_wp.numpy(), concat_vertices_wp.numpy())
    assert np.array_equal(roundtrip_faces_wp.numpy(), concat_faces_wp.numpy())

    split_wp_sorted = sorted(split_wp, key=lambda mesh: mesh[1].shape[0])
    meshes_tm_sorted = sorted([mesh_a_tm, mesh_b_tm, mesh_c_tm], key=lambda mesh: len(mesh.faces))
    for (vertices_wp, faces_wp), mesh_tm in zip(split_wp_sorted, meshes_tm_sorted, strict=True):
        assert np.allclose(vertices_wp.numpy(), mesh_tm.vertices, rtol=1e-5, atol=1e-5)
        assert np.array_equal(faces_wp.numpy(), mesh_tm.faces.reshape(-1))


def test_split_batched_matches_split(request: pytest.FixtureRequest) -> None:
    """``split`` slices ``split_batched``: the CSR must agree with it slice for slice."""
    meshes_wp = [
        request.getfixturevalue(name) for name in ("icosahedron", "hemisphere", "half_torus")
    ]
    concat_vertices_wp, concat_faces_wp = tw.combine.concatenate(
        [(mesh_wp.points, mesh_wp.indices) for _mesh_tm, mesh_wp in meshes_wp]
    )

    vertices_all_wp, vertex_offsets_wp, faces_all_wp, face_offsets_wp = tw.combine.split_batched(
        concat_vertices_wp, concat_faces_wp
    )
    split_wp = tw.combine.split(concat_vertices_wp, concat_faces_wp)
    assert int(vertex_offsets_wp.shape[0]) == len(split_wp) == 3

    vertex_bounds_np = [*vertex_offsets_wp.numpy().tolist(), int(vertices_all_wp.shape[0])]
    face_bounds_np = [*face_offsets_wp.numpy().tolist(), int(faces_all_wp.shape[0]) // 3]
    for index, (vertices_wp, faces_wp) in enumerate(split_wp):
        v_begin, v_end = vertex_bounds_np[index], vertex_bounds_np[index + 1]
        f_begin, f_end = face_bounds_np[index], face_bounds_np[index + 1]
        assert np.array_equal(vertices_all_wp.numpy()[v_begin:v_end], vertices_wp.numpy())
        assert np.array_equal(faces_all_wp.numpy()[3 * f_begin : 3 * f_end], faces_wp.numpy())

    # ``copy=True`` returns the same data in independent buffers.
    for (view_vertices_wp, view_faces_wp), (copy_vertices_wp, copy_faces_wp) in zip(
        split_wp, tw.combine.split(concat_vertices_wp, concat_faces_wp, copy=True), strict=True
    ):
        assert np.array_equal(view_vertices_wp.numpy(), copy_vertices_wp.numpy())
        assert np.array_equal(view_faces_wp.numpy(), copy_faces_wp.numpy())
        assert copy_vertices_wp.ptr != view_vertices_wp.ptr


def test_split_single_component(request: pytest.FixtureRequest) -> None:
    """The ``k == 1`` fast path must return the same thing as the batched key packing."""
    mesh_tm, mesh_wp = request.getfixturevalue("icosahedron")
    split_wp = tw.combine.split(mesh_wp.points, mesh_wp.indices)
    assert len(split_wp) == 1
    assert np.allclose(split_wp[0][0].numpy(), mesh_tm.vertices, rtol=1e-5, atol=1e-5)
    assert np.array_equal(split_wp[0][1].numpy(), mesh_tm.faces.reshape(-1))


def test_split_empty(device: str) -> None:
    vertices_wp = wp.empty(0, dtype=wp.vec3, device=device)
    faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    assert tw.combine.split(vertices_wp, faces_wp) == []
    vertices_all_wp, vertex_offsets_wp, faces_all_wp, face_offsets_wp = tw.combine.split_batched(
        vertices_wp, faces_wp
    )
    assert vertices_all_wp.shape == (0,)
    assert vertex_offsets_wp.shape == (0,)
    assert faces_all_wp.shape == (0,)
    assert face_offsets_wp.shape == (0,)


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


@pytest.mark.parity("connected_component_labels", "scipy")
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


@pytest.mark.parity("connected_component_labels_depth", "scipy")
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


def test_connected_component_parity_random(device: str) -> None:
    # Signs drawn from a hidden potential, so the constraints are consistent everywhere and the
    # returned parity must reproduce that potential up to a per-component flip.
    rng = np.random.default_rng(11)
    n = 4096
    potential_np = rng.integers(0, 2, size=n).astype(np.int32)
    a_np = rng.integers(0, n, size=12_000).astype(np.int32)
    b_np = rng.integers(0, n, size=12_000).astype(np.int32)
    edges_np = np.stack([a_np, b_np], axis=1)
    signs_np = (potential_np[a_np] ^ potential_np[b_np]).astype(np.int32)

    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)
    signs_wp = wp.array(signs_np, dtype=wp.int32, device=device)
    labels_wp, parity_wp = tw.graph.connected_component_parity_from_edges(edges_wp, signs_wp, n)

    parity_np = parity_wp.numpy()
    assert np.array_equal(parity_np[a_np] ^ parity_np[b_np], signs_np)
    assert _same_partition(labels_wp.numpy(), _scipy_component_labels(edges_np, n))
    # Each component representative anchors its own potential at 0.
    labels_np = labels_wp.numpy()
    assert np.array_equal(parity_np[np.unique(labels_np)], np.zeros(len(np.unique(labels_np))))


def test_connected_component_parity_long_path(device: str) -> None:
    # A path is the worst case for edge-by-edge propagation (one level per node) and the case the
    # union-find is depth-independent on; the potential is then the running XOR of the signs.
    n = 20_001
    rng = np.random.default_rng(5)
    edges_np = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1).astype(np.int32)
    signs_np = rng.integers(0, 2, size=n - 1).astype(np.int32)
    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)
    signs_wp = wp.array(signs_np, dtype=wp.int32, device=device)

    labels_wp, parity_wp = tw.graph.connected_component_parity_from_edges(edges_wp, signs_wp, n)
    parity_exp = np.concatenate([[0], np.cumsum(signs_np) % 2]).astype(np.int32)
    assert np.array_equal(parity_wp.numpy(), parity_exp)
    assert np.array_equal(labels_wp.numpy(), np.zeros(n, dtype=np.int32))


def test_connected_component_parity_contradiction_terminates(device: str) -> None:
    # An odd-signed cycle admits no potential. The contract is best-effort, not an exception: the
    # call must still terminate and label the component, leaving some edge violated.
    n = 1025
    edges_np = np.stack([np.arange(n), (np.arange(n) + 1) % n], axis=1).astype(np.int32)
    signs_np = np.ones(n, dtype=np.int32)
    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)
    signs_wp = wp.array(signs_np, dtype=wp.int32, device=device)

    labels_wp, parity_wp = tw.graph.connected_component_parity_from_edges(edges_wp, signs_wp, n)
    assert np.array_equal(labels_wp.numpy(), np.zeros(n, dtype=np.int32))
    parity_np = parity_wp.numpy()
    assert set(np.unique(parity_np).tolist()) <= {0, 1}
    violated = parity_np[edges_np[:, 0]] ^ parity_np[edges_np[:, 1]] != signs_np
    assert violated.sum() >= 1


def test_connected_component_parity_no_edges(device: str) -> None:
    edges_wp = wp.zeros((0, 2), dtype=wp.int32, device=device)
    signs_wp = wp.zeros(0, dtype=wp.int32, device=device)
    labels_wp, parity_wp = tw.graph.connected_component_parity_from_edges(edges_wp, signs_wp, 7)
    assert np.array_equal(labels_wp.numpy(), np.arange(7, dtype=np.int32))
    assert np.array_equal(parity_wp.numpy(), np.zeros(7, dtype=np.int32))


def test_connected_component_parity_signs_length_mismatch(device: str) -> None:
    edges_wp = wp.zeros((4, 2), dtype=wp.int32, device=device)
    signs_wp = wp.zeros(3, dtype=wp.int32, device=device)
    with pytest.raises(ValueError, match="signs must have length 4"):
        tw.graph.connected_component_parity_from_edges(edges_wp, signs_wp, 4)


def test_face_connected_component_labels(request: pytest.FixtureRequest) -> None:
    mesh_a_tm, mesh_a_wp = request.getfixturevalue("icosahedron")
    mesh_b_tm, mesh_b_wp = request.getfixturevalue("hemisphere")
    mesh_c_tm, mesh_c_wp = request.getfixturevalue("half_torus")

    concat_tm = tm.util.concatenate([mesh_a_tm, mesh_b_tm, mesh_c_tm])
    _, concat_faces_wp = tw.combine.concatenate(
        [
            (mesh_a_wp.points, mesh_a_wp.indices),
            (mesh_b_wp.points, mesh_b_wp.indices),
            (mesh_c_wp.points, mesh_c_wp.indices),
        ]
    )
    face_labels_wp = tw.adjacency.face_connected_component_labels(concat_faces_wp)
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


@pytest.mark.parity("bfs", "scipy")
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


def test_bfs_random_large_frontier(device: str) -> None:
    # Above the serial threshold the frontier-parallel path runs; it must still match scipy's
    # exact discovery order, parents, and distances.
    rng = np.random.default_rng(11)
    node_count = 20_000
    pairs = rng.integers(0, node_count, size=(60_000, 2), dtype=np.int32)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    edges_np = np.unique(np.sort(pairs, axis=1), axis=0).astype(np.int32)
    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)

    for source in (0, node_count // 2):
        order_wp, parents_wp, distances_wp = tw.graph.bfs_from_edges(
            edges_wp, source, node_count=node_count
        )
        order_np, parents_np, distances_np = _scipy_bfs(edges_np, node_count, source)
        assert np.array_equal(order_wp.numpy(), order_np)
        assert np.array_equal(parents_wp.numpy(), parents_np)
        assert np.array_equal(distances_wp.numpy(), distances_np)


def test_bfs_grid_graph_many_levels(device: str) -> None:
    # High-diameter graph above the serial threshold: a 150x150 grid runs ~300 frontier levels,
    # stressing the per-level rank/scan/scatter ordering against scipy across many iterations.
    side = 150
    node_count = side * side
    ids = np.arange(node_count, dtype=np.int32).reshape(side, side)
    horizontal = np.stack([ids[:, :-1].ravel(), ids[:, 1:].ravel()], axis=1)
    vertical = np.stack([ids[:-1, :].ravel(), ids[1:, :].ravel()], axis=1)
    edges_np = np.concatenate([horizontal, vertical]).astype(np.int32)
    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)

    for source in (0, node_count // 2):
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


def test_bfs_long_path_graph(device: str) -> None:
    # A path of 65 536 nodes: one BFS level per node, and a frontier of one throughout. The
    # level-synchronous loop hands over to the serial resume almost immediately here, so this is
    # the case that checks the handoff is order-exact and not just fast.
    n = 65_536
    edges_np = np.stack([np.arange(n - 1, dtype=np.int32), np.arange(1, n, dtype=np.int32)], axis=1)
    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)

    order_wp, parents_wp, distances_wp = tw.graph.bfs_from_edges(edges_wp, 0, node_count=n)
    assert np.array_equal(order_wp.numpy(), np.arange(n, dtype=np.int32))
    assert np.array_equal(distances_wp.numpy(), np.arange(n, dtype=np.int32))
    parents_exp = np.concatenate([[-1], np.arange(n - 1)]).astype(np.int32)
    assert np.array_equal(parents_wp.numpy(), parents_exp)


def test_bfs_wide_then_narrow_matches_scipy(device: str) -> None:
    # A "lollipop": a dense blob whose frontier is wide for a few levels, then a long tail where it
    # is one node across. The traversal therefore runs parallel levels first and escapes to the
    # serial resume part-way through -- the mixed case, where an off-by-one in the handed-over FIFO
    # window would corrupt the discovery order without changing the reachable set.
    rng = np.random.default_rng(19)
    blob, tail = 2_000, 6_000
    blob_edges = np.unique(
        np.sort(rng.integers(0, blob, size=(40_000, 2)).astype(np.int32), axis=1), axis=0
    )
    blob_edges = blob_edges[blob_edges[:, 0] != blob_edges[:, 1]]
    tail_nodes = np.arange(blob - 1, blob + tail, dtype=np.int32)
    tail_edges = np.stack([tail_nodes[:-1], tail_nodes[1:]], axis=1)
    edges_np = np.concatenate([blob_edges, tail_edges]).astype(np.int32)
    n = blob + tail

    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)
    order_wp, parents_wp, distances_wp = tw.graph.bfs_from_edges(edges_wp, 0, node_count=n)
    order_np, parents_np, distances_np = _scipy_bfs(edges_np, n, 0)
    assert np.array_equal(order_wp.numpy(), order_np)
    assert np.array_equal(parents_wp.numpy(), parents_np)
    assert np.array_equal(distances_wp.numpy(), distances_np)


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


def test_bfs_multi_source_large_component(device: str) -> None:
    # A long path exercises what used to be a fixed 512-node scratch cap: the component-based
    # implementation returns the complete reachable set with no truncation warning.
    n = 700
    edges_np = np.stack([np.arange(n - 1, dtype=np.int32), np.arange(1, n, dtype=np.int32)], axis=1)
    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)
    adjacency = tw.graph.edges_to_csr(n, edges_wp)
    sources_wp = wp.array(np.array([5], dtype=np.int32), dtype=wp.int32, device=device)
    neighbors_wp, offsets_wp = tw.graph.bfs_multi_source(adjacency, sources_wp)
    assert offsets_wp.numpy().tolist() == [0]
    neighbors_np = neighbors_wp.numpy()
    assert neighbors_np.shape == (n,)
    assert neighbors_np[0] == 5  # the source leads its own range
    assert np.array_equal(np.sort(neighbors_np), np.arange(n))


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
