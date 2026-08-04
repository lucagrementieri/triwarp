from __future__ import annotations

import igl
import numpy as np
import pytest
import trimesh as tm
import trimesh.repair as tm_repair
import warp as wp

import triwarp as tw
from tests.comparisons import canonical_winding
from tests.conversions import trimesh_to_open3d, trimesh_to_pymeshlab

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
@pytest.mark.parity("is_vertex_manifold", "pymeshlab")
@pytest.mark.parity("is_watertight", "pymeshlab")
@pytest.mark.parity("is_volume", "pymeshlab")
def test_topological_measures_match_pymeshlab(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Cross-check four independent predicates against MeshLab's one topology report.

    ``get_topological_measures`` returns edge count, boundary-edge count, component count, genus and
    edge-manifoldness together, so a single call checks the whole family at once and -- unlike the
    per-function trimesh and libigl oracles -- checks them *for mutual consistency* as well. That is
    the value here: an inconsistent set (a genus that does not follow from the Euler characteristic,
    say) is a class of bug no single-quantity comparison can see.

    MeshLab reports a genus for open meshes too, so the genus check is written as the full ``chi =
    2C - 2g - L`` identity (components, genus, boundary loops) rather than the closed-surface ``chi
    = 2 - 2g``. That form is what makes it a *joint* constraint tying four independently computed
    quantities together, and it is the one that catches ``cave_cube``: a hollow shell is two
    components, so its Euler characteristic is 4 and the single-component form would be wrong.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    measures_pml = trimesh_to_pymeshlab(mesh_tm).get_topological_measures()

    assert tw.validation.is_edge_manifold(mesh_wp.indices, allow_boundary_edges=True) == bool(
        measures_pml["is_mesh_two_manifold"]
    )

    unique_edges_wp, _ = tw.edges.edges_unique(mesh_wp.indices)
    assert int(unique_edges_wp.shape[0]) == int(measures_pml["edges_number"])

    boundary_edges_wp = tw.boundary.boundary_edges(mesh_wp.points, mesh_wp.indices)
    assert int(boundary_edges_wp.shape[0]) == int(measures_pml["boundary_edges"])

    n_vertices = int(mesh_wp.points.shape[0])
    labels_np = tw.graph.connected_component_labels_from_edges(unique_edges_wp, n_vertices).numpy()
    assert len(np.unique(labels_np)) == int(measures_pml["connected_components_number"])

    assert tw.validation.is_vertex_manifold(mesh_wp.indices) == (
        int(measures_pml["non_two_manifold_vertices"]) == 0
    )

    # ``is_watertight`` follows Open3D, which is two-manifold + no boundary + no self-intersection.
    # MeshLab's topology report covers the first two; the third needs a second filter, so the
    # composition below is what ``benchmarks/test_validation.py`` times as its pymeshlab row.
    selfx_meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    selfx_meshset_pml.compute_selection_by_self_intersections_per_face()
    n_self_intersecting = int(selfx_meshset_pml.current_mesh().face_selection_array().sum())
    watertight_pml = (
        bool(measures_pml["is_mesh_two_manifold"])
        and int(measures_pml["boundary_edges"]) == 0
        and int(measures_pml["non_two_manifold_vertices"]) == 0
        and n_self_intersecting == 0
    )
    assert tw.validation.is_watertight(mesh_wp.points, mesh_wp.indices) == watertight_pml
    # ``is_volume`` adds consistent winding on top, which is true of every fixture here.
    assert tw.validation.is_volume(mesh_wp.points, mesh_wp.indices) == (
        watertight_pml and tw.validation.is_winding_consistent(mesh_wp.indices)
    )

    n_loops = len(tw.boundary.boundary_loops(mesh_wp.points, mesh_wp.indices))
    assert (mesh_name in CLOSED_MESHES) == (n_loops == 0)
    n_components = int(measures_pml["connected_components_number"])
    euler_wp = tw.validation.euler_characteristic(mesh_wp.indices)
    assert euler_wp == 2 * n_components - 2 * int(measures_pml["genus"]) - n_loops


@pytest.mark.parametrize("mesh_name", ALL_MESHES)
def test_euler_characteristic(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    euler_wp = tw.validation.euler_characteristic(mesh_wp.indices)
    assert euler_wp == int(mesh_tm.euler_number)


def test_euler_characteristic_icosahedron(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = icosahedron
    assert tw.validation.euler_characteristic(mesh_wp.indices) == 2


@pytest.mark.parametrize("mesh_name", ALL_MESHES)
@pytest.mark.parity("is_edge_manifold", "igl")
def test_is_edge_manifold_allow_boundary(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A: the reduced verdict against ``igl.is_edge_manifold``'s first return, plus NumPy.

    igl always allows boundary edges, so this is the comparison at triwarp's default
    ``allow_boundary_edges=True``; the ``False`` setting has no igl counterpart and is pinned
    against the NumPy reference in the test below.

    The fixture list spans closed and open meshes, so the assert is not a constant: every fixture
    here is edge-manifold, and the *non*-manifold direction -- where a constant ``True`` would fail
    -- is covered by the deliberately non-manifold inputs in
    [`test_edge_manifold_mask`][tests.test_validation.test_edge_manifold_mask] and the two
    single-edge fan cases further down.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    manifold_wp = tw.validation.is_edge_manifold(mesh_wp.indices, allow_boundary_edges=True)
    manifold_igl = bool(igl.is_edge_manifold(_faces_igl(mesh_tm))[0])
    manifold_np = _edge_manifold_np(mesh_tm.faces, allow_boundary_edges=True)
    assert manifold_wp == manifold_igl
    assert manifold_wp == manifold_np


@pytest.mark.parametrize("mesh_name", ALL_MESHES)
def test_is_edge_manifold_no_boundary(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    manifold_wp = tw.validation.is_edge_manifold(mesh_wp.indices, allow_boundary_edges=False)
    manifold_np = _edge_manifold_np(mesh_tm.faces, allow_boundary_edges=False)
    assert manifold_wp == manifold_np
    # Closed meshes have no boundary edges; open surfaces do.
    assert manifold_wp == (mesh_name in CLOSED_MESHES)


@pytest.mark.parametrize("mesh_name", ALL_MESHES)
@pytest.mark.parametrize("allow_boundary_edges", [True, False])
def test_edge_manifold_mask(
    request: pytest.FixtureRequest, mesh_name: str, allow_boundary_edges: bool
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mask_wp = tw.validation.edge_manifold_mask(
        mesh_wp.indices, allow_boundary_edges=allow_boundary_edges
    )
    mask_np = _edge_manifold_mask_np(mesh_tm.faces, allow_boundary_edges)
    assert mask_wp.shape[0] == mesh_tm.faces.shape[0]
    assert np.array_equal(mask_wp.numpy(), mask_np)
    # is_edge_manifold is the reduction of the per-face mask.
    assert bool(mask_wp.numpy().all()) == tw.validation.is_edge_manifold(
        mesh_wp.indices, allow_boundary_edges=allow_boundary_edges
    )
    if allow_boundary_edges:
        # libigl BF is per-corner; a face is manifold iff all three corners are.
        edge_manifold_bf = igl.is_edge_manifold(_faces_igl(mesh_tm))[1]
        assert np.array_equal(mask_wp.numpy(), edge_manifold_bf.all(axis=1))


def test_edge_manifold_mask_edges_sorted_shortcut(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = icosahedron
    edges_sorted = tw.edges.faces_to_edges(mesh_wp.indices, sorted=True)
    mask_default = tw.validation.edge_manifold_mask(mesh_wp.indices)
    mask_shortcut = tw.validation.edge_manifold_mask(mesh_wp.indices, edges_sorted=edges_sorted)
    assert np.array_equal(mask_default.numpy(), mask_shortcut.numpy())


def test_is_edge_manifold_precomputed_shortcut(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = icosahedron
    edges_sorted = tw.edges.faces_to_edges(mesh_wp.indices, sorted=True)
    n_vertices = tw.vertices.n_vertices(edges_sorted)
    assert tw.validation.is_edge_manifold(
        mesh_wp.indices, edges_sorted=edges_sorted, n_vertices=n_vertices
    ) == tw.validation.is_edge_manifold(mesh_wp.indices)


def test_is_vertex_manifold_precomputed_shortcut(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = icosahedron
    adjacency, adjacency_edges = tw.adjacency.face_adjacency(mesh_wp.indices, return_edges=True)
    assert tw.validation.is_vertex_manifold(
        mesh_wp.indices, face_adjacency=adjacency, face_adjacency_edges=adjacency_edges
    ) == tw.validation.is_vertex_manifold(mesh_wp.indices)
    with pytest.raises(ValueError, match="together"):
        tw.validation.is_vertex_manifold(mesh_wp.indices, face_adjacency=adjacency)


def test_is_watertight_is_volume_precomputed_shortcut(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    _, mesh_wp = icosahedron
    edges = tw.edges.faces_to_edges(mesh_wp.indices)
    edges_sorted = tw.edges.faces_to_edges(mesh_wp.indices, sorted=True)
    assert tw.validation.is_watertight(
        mesh_wp.points, mesh_wp.indices, edges_sorted=edges_sorted
    ) == tw.validation.is_watertight(mesh_wp.points, mesh_wp.indices)
    assert tw.validation.is_volume(
        mesh_wp.points, mesh_wp.indices, edges=edges, edges_sorted=edges_sorted
    ) == tw.validation.is_volume(mesh_wp.points, mesh_wp.indices)


@pytest.mark.parametrize("mesh_name", ALL_MESHES)
@pytest.mark.parity("is_vertex_manifold", "igl")
def test_is_vertex_manifold(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    manifold_wp = tw.validation.is_vertex_manifold(mesh_wp.indices)
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
    manifold_wp = tw.validation.is_vertex_manifold(faces_wp)
    manifold_igl = bool(igl.is_vertex_manifold(faces_np.astype(np.int64)).all())
    assert manifold_wp == manifold_igl
    assert manifold_wp is False
    # The bow-tie is still edge-manifold (each edge used once).
    assert tw.validation.is_edge_manifold(faces_wp, allow_boundary_edges=True) is True


@pytest.mark.parametrize("mesh_name", ALL_MESHES)
def test_vertex_manifold_mask(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mask_wp = tw.validation.vertex_manifold_mask(mesh_wp.points, mesh_wp.indices)
    mask_igl = igl.is_vertex_manifold(_faces_igl(mesh_tm))
    assert mask_wp.shape[0] == mesh_tm.vertices.shape[0]
    assert np.array_equal(mask_wp.numpy(), mask_igl)
    # is_vertex_manifold is the reduction of the per-vertex mask.
    assert bool(mask_wp.numpy().all()) == tw.validation.is_vertex_manifold(mesh_wp.indices)


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
    mask_wp = tw.validation.vertex_manifold_mask(vertices_wp, faces_wp)
    expected = np.array([False, True, True, True, True, False])
    assert mask_wp.shape[0] == vertices_np.shape[0]
    assert np.array_equal(mask_wp.numpy(), expected)


def test_vertex_manifold_mask_faces_without_vertices(device: str) -> None:
    # Vertices present but no faces: every vertex is unreferenced -> all False.
    vertices_wp = wp.array(np.zeros((4, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    mask_wp = tw.validation.vertex_manifold_mask(vertices_wp, faces_wp)
    assert np.array_equal(mask_wp.numpy(), np.zeros(4, dtype=bool))


@pytest.mark.parametrize("mesh_name", CLOSED_MESHES)
def test_is_self_intersecting_clean(request: pytest.FixtureRequest, mesh_name: str) -> None:
    _, mesh_wp = request.getfixturevalue(mesh_name)
    assert tw.validation.is_self_intersecting(mesh_wp) is False


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
    mesh_wp = wp.Mesh(points=vertices_wp, indices=faces_wp)
    assert tw.validation.is_self_intersecting(mesh_wp) is True


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
    mesh_wp = wp.Mesh(points=vertices_wp, indices=faces_wp)
    assert tw.validation.is_self_intersecting(mesh_wp) is False


@pytest.mark.parametrize("mesh_name", ALL_MESHES)
def test_face_self_intersecting_mask_matches_predicate(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mask_wp = tw.validation.face_self_intersecting_mask(mesh_wp.points, mesh_wp.indices)
    assert int(mask_wp.shape[0]) == mesh_tm.faces.shape[0]
    predicate = tw.validation.is_self_intersecting(mesh_wp)
    assert bool(tw.reduce.any(mask_wp)) == predicate


@pytest.mark.parametrize("mesh_name", ALL_MESHES)
def test_is_winding_consistent(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    winding_wp = tw.validation.is_winding_consistent(mesh_wp.indices)
    assert winding_wp == bool(mesh_tm.is_winding_consistent)
    assert winding_wp is True


def test_is_winding_consistent_flipped(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = icosahedron
    faces_flipped = mesh_tm.faces.copy()
    faces_flipped[::2] = faces_flipped[::2][:, ::-1]  # reverse winding of half the faces
    _, faces_wp = _mesh_to_wp(mesh_tm.vertices, faces_flipped, mesh_wp.device)
    mesh_flipped_tm = tm.Trimesh(vertices=mesh_tm.vertices, faces=faces_flipped, process=False)
    winding_wp = tw.validation.is_winding_consistent(faces_wp)
    assert winding_wp == bool(mesh_flipped_tm.is_winding_consistent)
    assert winding_wp is False


@pytest.mark.parametrize("mesh_name", ALL_MESHES)
def test_edge_winding_consistent_mask_matches_predicate(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    _, mesh_wp = request.getfixturevalue(mesh_name)
    mask_wp = tw.validation.edge_winding_consistent_mask(mesh_wp.indices)
    aggregated = bool(tw.reduce.all(mask_wp)) if int(mask_wp.shape[0]) > 0 else True
    assert aggregated == tw.validation.is_winding_consistent(mesh_wp.indices)
    assert aggregated is True


def test_edge_winding_consistent_mask_flags_flipped(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    mesh_tm, mesh_wp = icosahedron
    faces_flipped = mesh_tm.faces.copy()
    faces_flipped[::2] = faces_flipped[::2][:, ::-1]  # reverse winding of half the faces
    _, faces_wp = _mesh_to_wp(mesh_tm.vertices, faces_flipped, mesh_wp.device)
    mask_wp = tw.validation.edge_winding_consistent_mask(faces_wp)
    assert int(mask_wp.shape[0]) > 0
    assert bool(tw.reduce.all(mask_wp)) is False


@pytest.mark.parametrize("mesh_name", ALL_MESHES)
def test_is_orientable_fixtures(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    orientable_wp = tw.validation.is_orientable(mesh_wp.indices)
    orientable_np = _orientable_np(mesh_tm.faces)
    assert orientable_wp == orientable_np
    assert orientable_wp is True


def test_is_orientable_flip_invariance(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, _ = icosahedron
    faces_flipped = mesh_tm.faces.copy()
    faces_flipped[::2] = faces_flipped[::2][:, ::-1]  # reverse winding of half the faces
    faces_wp = wp.array(faces_flipped.reshape(-1).astype(np.int32), dtype=wp.int32)
    # Orientability is flip-invariant even though the winding is now inconsistent.
    assert tw.validation.is_orientable(faces_wp) is True
    assert _orientable_np(faces_flipped) is True


def test_is_orientable_mobius(device: str) -> None:
    vertices_np, faces_np = _mobius_strip(12)
    _, faces_wp = _mesh_to_wp(vertices_np, faces_np, device)
    assert tw.validation.is_orientable(faces_wp) is False
    assert _orientable_np(faces_np) is False
    # The Möbius strip is still edge- and vertex-manifold.
    assert tw.validation.is_edge_manifold(faces_wp, allow_boundary_edges=True) is True
    assert tw.validation.is_vertex_manifold(faces_wp) is True


@pytest.mark.parametrize("mesh_name", ALL_MESHES)
def test_face_orientation_mask_all_false_on_consistent(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mask_wp = tw.validation.face_orientation_mask(mesh_wp.indices)
    assert int(mask_wp.shape[0]) == mesh_tm.faces.shape[0]
    # Fixtures are consistently wound, so no face needs flipping.
    assert bool(tw.reduce.any(mask_wp)) is False


def _triangle_ribbon(n_quads: int) -> tuple[np.ndarray, np.ndarray]:
    """Flat triangle strip: the face-adjacency graph is a path of ``2 * n_quads`` nodes."""
    x = np.arange(n_quads + 1, dtype=np.float64)
    vertices = np.empty((2 * (n_quads + 1), 3))
    vertices[0::2] = np.column_stack((x, np.zeros_like(x), np.zeros_like(x)))
    vertices[1::2] = np.column_stack((x, np.ones_like(x), np.zeros_like(x)))
    i = np.arange(n_quads)
    faces = np.empty((2 * n_quads, 3), dtype=np.int64)
    faces[0::2] = np.column_stack((2 * i, 2 * i + 1, 2 * i + 2))
    faces[1::2] = np.column_stack((2 * i + 1, 2 * i + 3, 2 * i + 2))
    return vertices, faces


@pytest.mark.parity("face_orientation_bits", "igl")
def test_face_orientation_mask_matches_igl(device: str) -> None:
    """
    Class B: igl's flip mask is *derived* from its reoriented face table, not returned.

    ``igl.bfs_orient(F)`` returns ``(FF, C)`` and ``C`` is the per-face **component id** -- all
    zeros on a connected mesh -- which is the trap this test exists to pin: comparing triwarp's bits
    against ``C`` would compare them against a constant and pass for the wrong reason. The named
    transform is therefore *recover the mask*: a face was flipped iff its row in ``FF`` is the
    reversal of its row in ``F``, which the first assert checks is the only possibility (every row
    is either identical or reversed, never a different triangle).

    With that mask recovered the two agree **exactly, with no global sign fix**: both libraries
    anchor each component on its lowest-numbered face, so the answer is unique rather than
    determined up to a per-component flip. The second half of the test pins the same statement one
    level up, on the repaired face table, through
    [`tests.comparisons.canonical_winding`][] -- a flip is emitted as a rotation of the reversed
    triangle, so the rows are compared cyclically.
    """
    vertices_np, faces_np = _triangle_ribbon(1024)
    rng = np.random.default_rng(11)
    scrambled_np = rng.random(faces_np.shape[0]) < 0.5
    flipped_np = faces_np.copy()
    flipped_np[scrambled_np] = flipped_np[scrambled_np][:, ::-1]
    _, faces_wp = _mesh_to_wp(vertices_np, flipped_np, device)

    oriented_igl, components_igl = igl.bfs_orient(np.ascontiguousarray(flipped_np, dtype=np.int64))
    unchanged_igl = (oriented_igl == flipped_np).all(axis=1)
    reversed_igl = (oriented_igl == flipped_np[:, ::-1]).all(axis=1)
    assert bool((unchanged_igl | reversed_igl).all()), "a row is neither kept nor reversed"
    assert np.unique(components_igl).shape[0] == 1, "the ribbon is one component"

    mask_wp = tw.validation.face_orientation_mask(faces_wp).numpy()
    repaired_np = tw.repair.make_winding_consistent(faces_wp).numpy().reshape(-1, 3)

    assert np.array_equal(mask_wp, ~unchanged_igl)
    assert np.array_equal(canonical_winding(repaired_np), canonical_winding(oriented_igl))


@pytest.mark.parity("face_orientation_bits", "trimesh")
def test_face_orientation_mask_long_path(device: str) -> None:
    """
    A ribbon whose face-adjacency graph is a path of 8 192 nodes.

    This is the deep-propagation case: an orientation flood fill needs one round per node here,
    while the parity union-find behind
    [`face_orientation_bits`][triwarp.validation.face_orientation_bits] is depth-independent. The
    answer is unique because the component representative is the smallest face id, so face 0 keeps
    its winding and every other face is determined relative to it.

    Class B against ``trimesh.repair.fix_winding``, the flood fill the benchmark times. Two named
    transforms. The reference *rewrites the mesh* rather than returning bits, so triwarp's bits are
    applied via [`make_winding_consistent`][triwarp.repair.make_winding_consistent] to compare the
    two rewritten face arrays; and a flip is emitted as a rotation of the reversed triangle, so both
    go through [`tests.comparisons.canonical_winding`][] -- which is insensitive to the starting
    corner and still sensitive to the orientation under test. No global sign fix is needed: both
    libraries anchor on the lowest-numbered face of each component, so the answer is unique.
    """
    vertices_np, faces_np = _triangle_ribbon(4096)
    rng = np.random.default_rng(7)
    scrambled_np = rng.random(faces_np.shape[0]) < 0.5
    flipped_np = faces_np.copy()
    flipped_np[scrambled_np] = flipped_np[scrambled_np][:, ::-1]
    _, faces_wp = _mesh_to_wp(vertices_np, flipped_np, device)

    assert tw.validation.is_orientable(faces_wp) is True
    bits_wp, _signed_edges_wp, _signs_wp, _m = tw.validation.face_orientation_bits(faces_wp)
    mask_np = tw.validation.face_orientation_mask(faces_wp).numpy()
    assert np.array_equal(mask_np, bits_wp.numpy().astype(bool))
    assert bool(mask_np[0]) is False
    assert np.array_equal(mask_np, scrambled_np != scrambled_np[0])

    # Applying the mask must reproduce the original strip, up to the global flip face 0 anchors.
    # Compared as cyclic windings: a flip is emitted as a rotation of the reversed triangle.
    repaired_np = tw.repair.make_winding_consistent(faces_wp).numpy().reshape(-1, 3)
    expected_np = faces_np[:, ::-1] if scrambled_np[0] else faces_np
    assert np.array_equal(canonical_winding(repaired_np), canonical_winding(expected_np))
    assert tw.validation.is_winding_consistent(faces_wp) is False

    # trimesh reference: its fix_winding is a flood fill over the same face-adjacency graph, and
    # agrees face for face (both anchor on the first face of the component).
    mesh_tm = tm.Trimesh(vertices_np, flipped_np, process=False)
    tm.repair.fix_winding(mesh_tm)
    assert np.array_equal(canonical_winding(mesh_tm.faces), canonical_winding(repaired_np))
    # ... and so the bits themselves are the faces trimesh chose to flip.
    assert np.array_equal(
        mask_np, np.any(canonical_winding(mesh_tm.faces) != canonical_winding(flipped_np), axis=1)
    )


@pytest.mark.parametrize("mesh_name", ALL_MESHES)
@pytest.mark.parity("is_watertight", "open3d")
def test_is_watertight_matches_open3d(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    The definitional claim: ``is_watertight`` follows Open3D's ``IsWatertight``, not trimesh's.

    Worth a test of its own because the docstring makes that claim explicitly and nothing checked
    it. The two definitions genuinely differ -- trimesh's ``is_watertight`` is edge-manifoldness
    alone, while Open3D additionally requires vertex-manifoldness *and* the absence of
    self-intersections -- so the trimesh row in ``benchmarks/test_validation.py`` is timing context
    rather than an oracle, and open3d is the one that pins the semantics.

    Parametrized over closed and open fixtures so both answers appear; a boolean asserted only over
    watertight meshes would pass on a function that returned ``True`` unconditionally.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    watertight_o3d = trimesh_to_open3d(mesh_tm).is_watertight()

    assert tw.validation.is_watertight(mesh_wp.points, mesh_wp.indices) == watertight_o3d
    assert watertight_o3d == (mesh_name in CLOSED_MESHES)


def test_is_watertight_rejects_self_intersection_like_open3d(device: str) -> None:
    """
    The self-intersection clause, which is the half of Open3D's definition trimesh does not have.

    Two interpenetrating unit boxes are closed, edge-manifold and vertex-manifold, so every test
    above passes them and ``trimesh.is_watertight`` calls them watertight. Open3D does not, and
    neither may triwarp -- this is the only case in the module that separates the two definitions,
    and without it the parametrized test above would be satisfied by an edge-manifold check alone.
    """
    first_tm = tm.creation.box(extents=[1.0, 1.0, 1.0])
    second_tm = tm.creation.box(extents=[1.0, 1.0, 1.0])
    second_tm.apply_translation([0.5, 0.5, 0.5])
    tangled_tm = tm.util.concatenate([first_tm, second_tm])
    tangled_tm.merge_vertices()

    vertices_wp = wp.array(
        np.ascontiguousarray(tangled_tm.vertices, dtype=np.float32), dtype=wp.vec3, device=device
    )
    faces_wp = wp.array(
        np.ascontiguousarray(tangled_tm.faces.reshape(-1), dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )

    assert trimesh_to_open3d(tangled_tm).is_watertight() is False
    assert tw.validation.is_watertight(vertices_wp, faces_wp) is False
    # The clause that does the work: it *is* edge-manifold and closed, which is all trimesh checks.
    assert tw.validation.is_edge_manifold(faces_wp, allow_boundary_edges=False)
    assert bool(tangled_tm.is_watertight) is True


@pytest.mark.parametrize("mesh_name", ALL_MESHES)
@pytest.mark.parity("is_watertight", "trimesh")
def test_is_watertight(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    watertight_wp = tw.validation.is_watertight(mesh_wp.points, mesh_wp.indices)
    assert watertight_wp == (mesh_name in CLOSED_MESHES)
    # These fixtures are not self-intersecting, so Open3D's composite definition agrees with
    # trimesh's "every edge shared by exactly two faces" check.
    assert watertight_wp == bool(mesh_tm.is_watertight)


@pytest.mark.parametrize("mesh_name", ALL_MESHES)
def test_face_watertight_mask_matches_reference(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mask_wp = tw.validation.face_watertight_mask(mesh_wp.indices)
    mask_np = _edge_manifold_mask_np(mesh_tm.faces, allow_boundary_edges=False)
    assert np.array_equal(mask_wp.numpy(), mask_np)
    # Equivalent to edge_manifold_mask with boundary edges disallowed.
    edge_mask_wp = tw.validation.edge_manifold_mask(mesh_wp.indices, allow_boundary_edges=False)
    assert np.array_equal(mask_wp.numpy(), edge_mask_wp.numpy())


@pytest.mark.parametrize("mesh_name", OPEN_MESHES)
def test_face_watertight_mask_broken_faces_reference(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mask_wp = tw.validation.face_watertight_mask(mesh_wp.indices)
    # Faces breaking watertightness are the complement of the mask (trimesh's broken_faces).
    broken_ours = np.flatnonzero(~mask_wp.numpy())
    broken_tm = np.asarray(tm_repair.broken_faces(mesh_tm), dtype=np.int64)
    assert np.array_equal(np.sort(broken_ours), np.sort(broken_tm))


@pytest.mark.parametrize("mesh_name", ALL_MESHES)
@pytest.mark.parity("is_volume", "trimesh")
def test_is_volume(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    volume_wp = tw.validation.is_volume(mesh_wp.points, mesh_wp.indices)
    assert volume_wp == bool(mesh_tm.is_volume)
    assert volume_wp == (mesh_name in CLOSED_MESHES)


def test_is_volume_inward_normals(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = icosahedron
    faces_inward = mesh_tm.faces[:, ::-1].copy()  # reverse every face -> inward-facing normals
    vertices_wp, faces_wp = _mesh_to_wp(mesh_tm.vertices, faces_inward, mesh_wp.device)
    mesh_inward_tm = tm.Trimesh(vertices=mesh_tm.vertices, faces=faces_inward, process=False)
    # Still watertight and winding-consistent, but the enclosed signed volume is negative.
    volume_wp = tw.validation.is_volume(vertices_wp, faces_wp)
    assert volume_wp == bool(mesh_inward_tm.is_volume)
    assert volume_wp is False


def test_empty_mesh(device: str) -> None:
    vertices_wp = wp.empty(0, dtype=wp.vec3, device=device)
    faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    assert tw.validation.is_edge_manifold(faces_wp, allow_boundary_edges=True) is True
    assert tw.validation.is_edge_manifold(faces_wp, allow_boundary_edges=False) is True
    assert tw.validation.is_vertex_manifold(faces_wp) is True
    assert tw.validation.is_orientable(faces_wp) is True
    assert tw.validation.is_winding_consistent(faces_wp) is True
    assert tw.validation.is_volume(vertices_wp, faces_wp) is False
    assert tw.validation.euler_characteristic(faces_wp) == 0
    assert tw.validation.edge_manifold_mask(faces_wp).shape[0] == 0
    assert tw.validation.vertex_manifold_mask(vertices_wp, faces_wp).shape[0] == 0


def test_is_self_intersecting_fewer_than_two_faces(device: str) -> None:
    # A `warp.Mesh` with zero triangles corrupts CUDA state when its BVH is built (a Warp 1.15
    # bug independent of triwarp), so this exercises the n_faces < 2 short-circuit with a
    # single-triangle mesh instead of a fully empty one.
    vertices_np = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces_np = np.array([[0, 1, 2]])
    vertices_wp, faces_wp = _mesh_to_wp(vertices_np, faces_np, device)
    mesh_wp = wp.Mesh(points=vertices_wp, indices=faces_wp)
    assert tw.validation.is_self_intersecting(mesh_wp) is False


def test_new_masks_empty_mesh(device: str) -> None:
    vertices_wp = wp.empty(0, dtype=wp.vec3, device=device)
    faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    assert tw.validation.edge_winding_consistent_mask(faces_wp).shape[0] == 0
    assert tw.validation.face_self_intersecting_mask(vertices_wp, faces_wp).shape[0] == 0
    assert tw.validation.face_watertight_mask(faces_wp).shape[0] == 0
    assert tw.validation.face_orientation_mask(faces_wp).shape[0] == 0
