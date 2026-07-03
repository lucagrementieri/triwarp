from __future__ import annotations

import igl
import numpy as np
import pytest
import trimesh as tm
import warp as wp

import triwarp as tw

CLOSED_MESHES = ["icosahedron", "cave_cube"]
OPEN_MESHES = ["hemisphere", "half_torus"]
ALL_MESHES = CLOSED_MESHES + OPEN_MESHES


def _faces_igl(mesh_tm: tm.Trimesh) -> np.ndarray:
    """Faces as an ``(n_faces, 3)`` int64 array for the libigl reference functions."""
    return mesh_tm.faces.astype(np.int64)


def _edge_manifold_np(faces_np: np.ndarray, allow_boundary_edges: bool) -> bool:
    """NumPy reference: manifold edge-count check over undirected edges."""
    edges = np.sort(faces_np[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    if allow_boundary_edges:
        return bool((counts <= 2).all())
    return bool((counts == 2).all())


def _edge_manifold_mask_np(faces_np: np.ndarray, allow_boundary_edges: bool) -> np.ndarray:
    """NumPy reference: per-face flag that all three edges are manifold."""
    edges = np.sort(faces_np[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    _, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    edge_ok = counts <= 2 if allow_boundary_edges else counts == 2
    return np.asarray(edge_ok[inverse.reshape(-1)].reshape(-1, 3).all(axis=1))


def _orientable_np(faces_np: np.ndarray) -> bool:
    """NumPy reference: Z2 parity over the face-adjacency graph (BFS with a flip bit)."""
    n_faces = faces_np.shape[0]
    edge_to_faces: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for f, tri in enumerate(faces_np):
        for k in range(3):
            a, b = int(tri[k]), int(tri[(k + 1) % 3])
            forward = 1 if a < b else 0
            key = (min(a, b), max(a, b))
            edge_to_faces.setdefault(key, []).append((f, forward))

    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(n_faces)]
    for shared in edge_to_faces.values():
        if len(shared) != 2:
            continue
        (f0, d0), (f1, d1) = shared
        flip = 1 if d0 == d1 else 0
        adjacency[f0].append((f1, flip))
        adjacency[f1].append((f0, flip))

    orient = [-1] * n_faces
    for seed in range(n_faces):
        if orient[seed] != -1:
            continue
        orient[seed] = 0
        stack = [seed]
        while stack:
            f = stack.pop()
            for nb, flip in adjacency[f]:
                expected = orient[f] ^ flip
                if orient[nb] == -1:
                    orient[nb] = expected
                    stack.append(nb)
                elif orient[nb] != expected:
                    return False
    return True


def _mesh_to_wp(
    vertices_np: np.ndarray, faces_np: np.ndarray, device: str
) -> tuple[wp.array, wp.array]:
    vertices_wp = wp.array(
        np.ascontiguousarray(vertices_np, dtype=np.float32), dtype=wp.vec3, device=device
    )
    faces_wp = wp.array(
        np.ascontiguousarray(np.asarray(faces_np).reshape(-1), dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )
    return vertices_wp, faces_wp


def _mobius_strip(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Triangulated Möbius strip (edge- and vertex-manifold, non-orientable)."""
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    vertices = []
    for t in theta:
        for s in (-0.3, 0.3):
            vertices.append(
                [
                    (1.0 + s * np.cos(t / 2.0)) * np.cos(t),
                    (1.0 + s * np.cos(t / 2.0)) * np.sin(t),
                    s * np.sin(t / 2.0),
                ]
            )
    faces = []
    for i in range(n):
        a0, a1 = 2 * i, 2 * i + 1
        if i < n - 1:
            b0, b1 = 2 * (i + 1), 2 * (i + 1) + 1
        else:
            b0, b1 = 1, 0  # swap rails at the seam -> half twist
        faces.append([a0, a1, b1])
        faces.append([a0, b1, b0])
    return np.array(vertices), np.array(faces)


@pytest.mark.parametrize("mesh_name", ALL_MESHES)
def test_is_edge_manifold_allow_boundary(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    manifold_wp = tw.characteristics.is_edge_manifold(mesh_wp.indices, allow_boundary_edges=True)
    manifold_igl = bool(igl.is_edge_manifold(_faces_igl(mesh_tm))[0])
    manifold_np = _edge_manifold_np(mesh_tm.faces, allow_boundary_edges=True)
    assert manifold_wp == manifold_igl
    assert manifold_wp == manifold_np


@pytest.mark.parametrize("mesh_name", ALL_MESHES)
def test_is_edge_manifold_no_boundary(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    manifold_wp = tw.characteristics.is_edge_manifold(mesh_wp.indices, allow_boundary_edges=False)
    manifold_np = _edge_manifold_np(mesh_tm.faces, allow_boundary_edges=False)
    assert manifold_wp == manifold_np
    # Closed meshes have no boundary edges; open surfaces do.
    assert manifold_wp == (mesh_name in CLOSED_MESHES)


@pytest.mark.parametrize("mesh_name", ALL_MESHES)
def test_is_vertex_manifold(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    manifold_wp = tw.characteristics.is_vertex_manifold(mesh_wp.indices)
    manifold_igl = bool(igl.is_vertex_manifold(_faces_igl(mesh_tm)).all())
    assert manifold_wp == manifold_igl
    assert manifold_wp is True


def test_is_vertex_manifold_bowtie(device: str) -> None:
    # Two triangles sharing only the apex vertex 0 -> non-manifold vertex.
    vertices_np = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    )
    faces_np = np.array([[0, 1, 2], [0, 3, 4]])
    _, faces_wp = _mesh_to_wp(vertices_np, faces_np, device)
    manifold_wp = tw.characteristics.is_vertex_manifold(faces_wp)
    manifold_igl = bool(igl.is_vertex_manifold(faces_np.astype(np.int64)).all())
    assert manifold_wp == manifold_igl
    assert manifold_wp is False
    # The bow-tie is still edge-manifold (each edge used once).
    assert tw.characteristics.is_edge_manifold(faces_wp, allow_boundary_edges=True) is True


@pytest.mark.parametrize("mesh_name", ALL_MESHES)
@pytest.mark.parametrize("allow_boundary_edges", [True, False])
def test_edge_manifold_mask(
    request: pytest.FixtureRequest, mesh_name: str, allow_boundary_edges: bool
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mask_wp = tw.characteristics.edge_manifold_mask(
        mesh_wp.indices, allow_boundary_edges=allow_boundary_edges
    )
    mask_np = _edge_manifold_mask_np(mesh_tm.faces, allow_boundary_edges)
    assert mask_wp.shape[0] == mesh_tm.faces.shape[0]
    assert np.array_equal(mask_wp.numpy(), mask_np)
    # is_edge_manifold is the reduction of the per-face mask.
    assert bool(mask_wp.numpy().all()) == tw.characteristics.is_edge_manifold(
        mesh_wp.indices, allow_boundary_edges=allow_boundary_edges
    )
    if allow_boundary_edges:
        # libigl BF is per-corner; a face is manifold iff all three corners are.
        edge_manifold_bf = igl.is_edge_manifold(_faces_igl(mesh_tm))[1]
        assert np.array_equal(mask_wp.numpy(), edge_manifold_bf.all(axis=1))


def test_edge_manifold_mask_edges_sorted_shortcut(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = icosahedron
    edges_sorted = tw.edges.faces_to_edges(mesh_wp.indices, sorted=True)
    mask_default = tw.characteristics.edge_manifold_mask(mesh_wp.indices)
    mask_shortcut = tw.characteristics.edge_manifold_mask(
        mesh_wp.indices, edges_sorted=edges_sorted
    )
    assert np.array_equal(mask_default.numpy(), mask_shortcut.numpy())


@pytest.mark.parametrize("mesh_name", ALL_MESHES)
def test_vertex_manifold_mask(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mask_wp = tw.characteristics.vertex_manifold_mask(mesh_wp.points, mesh_wp.indices)
    mask_igl = igl.is_vertex_manifold(_faces_igl(mesh_tm))
    assert mask_wp.shape[0] == mesh_tm.vertices.shape[0]
    assert np.array_equal(mask_wp.numpy(), mask_igl)
    # is_vertex_manifold is the reduction of the per-vertex mask.
    assert bool(mask_wp.numpy().all()) == tw.characteristics.is_vertex_manifold(mesh_wp.indices)


def test_vertex_manifold_mask_unreferenced(device: str) -> None:
    # Bow-tie (vertex 0 non-manifold) plus a trailing unreferenced vertex 5.
    vertices_np = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [9.0, 9.0, 9.0],
        ]
    )
    faces_np = np.array([[0, 1, 2], [0, 3, 4]])
    vertices_wp, faces_wp = _mesh_to_wp(vertices_np, faces_np, device)
    mask_wp = tw.characteristics.vertex_manifold_mask(vertices_wp, faces_wp)
    expected = np.array([False, True, True, True, True, False])
    assert mask_wp.shape[0] == vertices_np.shape[0]
    assert np.array_equal(mask_wp.numpy(), expected)


@pytest.mark.parametrize("mesh_name", CLOSED_MESHES)
def test_is_self_intersecting_clean(request: pytest.FixtureRequest, mesh_name: str) -> None:
    _, mesh_wp = request.getfixturevalue(mesh_name)
    assert tw.characteristics.is_self_intersecting(mesh_wp.points, mesh_wp.indices) is False


def test_is_self_intersecting_crossing(device: str) -> None:
    vertices_np = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
            [1.0, -1.0, 0.0],
        ]
    )
    faces_np = np.array([[0, 1, 2], [3, 4, 5]])
    vertices_wp, faces_wp = _mesh_to_wp(vertices_np, faces_np, device)
    assert tw.characteristics.is_self_intersecting(vertices_wp, faces_wp) is True


def test_is_self_intersecting_separated(device: str) -> None:
    vertices_np = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [5.0, 5.0, 5.0],
            [6.0, 5.0, 5.0],
            [5.0, 6.0, 5.0],
        ]
    )
    faces_np = np.array([[0, 1, 2], [3, 4, 5]])
    vertices_wp, faces_wp = _mesh_to_wp(vertices_np, faces_np, device)
    assert tw.characteristics.is_self_intersecting(vertices_wp, faces_wp) is False


@pytest.mark.parametrize("mesh_name", ALL_MESHES)
def test_is_watertight(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    watertight_wp = tw.characteristics.is_watertight(mesh_wp.points, mesh_wp.indices)
    assert watertight_wp == (mesh_name in CLOSED_MESHES)
    # These fixtures are not self-intersecting, so Open3D's composite definition agrees with
    # trimesh's "every edge shared by exactly two faces" check.
    assert watertight_wp == bool(mesh_tm.is_watertight)


@pytest.mark.parametrize("mesh_name", ALL_MESHES)
def test_is_orientable_fixtures(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    orientable_wp = tw.characteristics.is_orientable(mesh_wp.indices)
    orientable_np = _orientable_np(mesh_tm.faces)
    assert orientable_wp == orientable_np
    assert orientable_wp is True


def test_is_orientable_flip_invariance(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, _ = icosahedron
    faces_flipped = mesh_tm.faces.copy()
    faces_flipped[::2] = faces_flipped[::2][:, ::-1]  # reverse winding of half the faces
    faces_wp = wp.array(faces_flipped.reshape(-1).astype(np.int32), dtype=wp.int32)
    # Orientability is flip-invariant even though the winding is now inconsistent.
    assert tw.characteristics.is_orientable(faces_wp) is True
    assert _orientable_np(faces_flipped) is True


def test_is_orientable_mobius(device: str) -> None:
    vertices_np, faces_np = _mobius_strip(12)
    _, faces_wp = _mesh_to_wp(vertices_np, faces_np, device)
    assert tw.characteristics.is_orientable(faces_wp) is False
    assert _orientable_np(faces_np) is False
    # The Möbius strip is still edge- and vertex-manifold.
    assert tw.characteristics.is_edge_manifold(faces_wp, allow_boundary_edges=True) is True
    assert tw.characteristics.is_vertex_manifold(faces_wp) is True


@pytest.mark.parametrize("mesh_name", ALL_MESHES)
def test_euler_characteristic(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    euler_wp = tw.characteristics.euler_characteristic(mesh_wp.indices)
    assert euler_wp == int(mesh_tm.euler_number)


def test_euler_characteristic_icosahedron(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = icosahedron
    assert tw.characteristics.euler_characteristic(mesh_wp.indices) == 2


def test_empty_mesh(device: str) -> None:
    vertices_wp = wp.empty(0, dtype=wp.vec3, device=device)
    faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    assert tw.characteristics.is_edge_manifold(faces_wp, allow_boundary_edges=True) is True
    assert tw.characteristics.is_edge_manifold(faces_wp, allow_boundary_edges=False) is True
    assert tw.characteristics.is_vertex_manifold(faces_wp) is True
    assert tw.characteristics.is_self_intersecting(vertices_wp, faces_wp) is False
    assert tw.characteristics.is_orientable(faces_wp) is True
    assert tw.characteristics.euler_characteristic(faces_wp) == 0
    assert tw.characteristics.edge_manifold_mask(faces_wp).shape[0] == 0
    assert tw.characteristics.vertex_manifold_mask(vertices_wp, faces_wp).shape[0] == 0


def test_vertex_manifold_mask_faces_without_vertices(device: str) -> None:
    # Vertices present but no faces: every vertex is unreferenced -> all False.
    vertices_wp = wp.array(np.zeros((4, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    mask_wp = tw.characteristics.vertex_manifold_mask(vertices_wp, faces_wp)
    assert np.array_equal(mask_wp.numpy(), np.zeros(4, dtype=bool))
