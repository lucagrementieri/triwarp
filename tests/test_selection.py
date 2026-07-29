"""Regression tests for ``triwarp.selection`` against Trimesh (CPU reference)."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh as tm
import warp as wp

import triwarp as tw
from tests.conversions import trimesh_to_pymeshlab


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
def test_submesh_from_face_indices_random_faces(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
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
def test_submesh_from_face_indices_all_faces(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
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


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube", "half_torus"])
def test_submeshes_from_face_groups_matches_single(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """Every group's slice must equal ``submesh_from_face_indices`` run on that group alone."""
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    device = mesh_wp.points.device
    rng = np.random.default_rng(11)
    n_faces = mesh_tm.faces.shape[0]

    # Four disjoint, non-empty groups of ascending face indices (what ``split`` produces).
    order_np = rng.permutation(n_faces).astype(np.int32)
    cuts_np = np.sort(rng.choice(np.arange(1, n_faces), size=3, replace=False))
    groups_np = [np.sort(part) for part in np.split(order_np, cuts_np)]
    offsets_np = np.cumsum([0, *(len(group) for group in groups_np[:-1])]).astype(np.int32)

    vertices_all_wp, vertex_offsets_wp, faces_all_wp = tw.selection.submeshes_from_face_groups(
        mesh_wp.points,
        mesh_wp.indices,
        wp.array(np.concatenate(groups_np).astype(np.int32), dtype=wp.int32, device=device),
        wp.array(offsets_np, dtype=wp.int32, device=device),
    )
    vertex_bounds_np = [*vertex_offsets_wp.numpy().tolist(), int(vertices_all_wp.shape[0])]
    face_bounds_np = [*offsets_np.tolist(), n_faces]

    for group, v_begin, v_end, f_begin, f_end in zip(
        groups_np,
        vertex_bounds_np[:-1],
        vertex_bounds_np[1:],
        face_bounds_np[:-1],
        face_bounds_np[1:],
        strict=True,
    ):
        single_vertices_wp, single_faces_wp = tw.selection.submesh_from_face_indices(
            mesh_wp.points,
            mesh_wp.indices,
            wp.array(group.astype(np.int32), dtype=wp.int32, device=device),
            unique_indices=True,
        )
        assert np.array_equal(vertices_all_wp.numpy()[v_begin:v_end], single_vertices_wp.numpy())
        assert np.array_equal(
            faces_all_wp.numpy()[3 * f_begin : 3 * f_end], single_faces_wp.numpy()
        )

    # ...and the first group also matches the trimesh reference directly.
    submesh_tm = tm.util.submesh(mesh_tm, [groups_np[0]], repair=False, append=False)[0]
    assert np.allclose(vertices_all_wp.numpy()[: vertex_bounds_np[1]], submesh_tm.vertices)
    assert np.array_equal(
        faces_all_wp.numpy()[: 3 * face_bounds_np[1]], submesh_tm.faces.reshape(-1)
    )


def test_submeshes_from_face_groups_shared_vertex(device: str) -> None:
    """Two bodies touching at one vertex: the shared vertex is duplicated into both groups."""
    # Two triangles meeting only at vertex 2, so face adjacency keeps them separate.
    vertices_np = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 2, 0], [-1, 1, 0]], dtype=np.float32
    )
    faces_np = np.array([0, 1, 2, 2, 3, 4], dtype=np.int32)
    vertices_wp = wp.array(vertices_np, dtype=wp.vec3, device=device)
    faces_wp = wp.array(faces_np, dtype=wp.int32, device=device)

    vertices_all_wp, vertex_offsets_wp, faces_all_wp = tw.selection.submeshes_from_face_groups(
        vertices_wp,
        faces_wp,
        wp.array([0, 1], dtype=wp.int32, device=device),
        wp.array([0, 1], dtype=wp.int32, device=device),
    )
    # 3 + 3 vertices, not 5: vertex 2 belongs to both groups and is emitted in each.
    assert np.array_equal(vertex_offsets_wp.numpy(), [0, 3])
    assert int(vertices_all_wp.shape[0]) == 6
    assert np.array_equal(faces_all_wp.numpy(), [0, 1, 2, 0, 1, 2])
    assert np.array_equal(vertices_all_wp.numpy()[:3], vertices_np[[0, 1, 2]])
    assert np.array_equal(vertices_all_wp.numpy()[3:], vertices_np[[2, 3, 4]])


def test_submeshes_from_face_groups_unreferenced_vertices(device: str) -> None:
    """Vertices no face uses are dropped, exactly as the single-group path drops them."""
    vertices_np = np.array(
        [[0, 0, 0], [9, 9, 9], [1, 0, 0], [0, 1, 0], [8, 8, 8]], dtype=np.float32
    )
    faces_np = np.array([0, 2, 3], dtype=np.int32)
    vertices_wp = wp.array(vertices_np, dtype=wp.vec3, device=device)
    faces_wp = wp.array(faces_np, dtype=wp.int32, device=device)

    vertices_all_wp, vertex_offsets_wp, faces_all_wp = tw.selection.submeshes_from_face_groups(
        vertices_wp,
        faces_wp,
        wp.array([0], dtype=wp.int32, device=device),
        wp.array([0], dtype=wp.int32, device=device),
    )
    assert np.array_equal(vertex_offsets_wp.numpy(), [0])
    assert np.array_equal(vertices_all_wp.numpy(), vertices_np[[0, 2, 3]])
    assert np.array_equal(faces_all_wp.numpy(), [0, 1, 2])


def test_submeshes_from_face_groups_empty(device: str) -> None:
    vertices_wp = wp.array(np.zeros((4, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    faces_wp = wp.array(np.array([0, 1, 2, 0, 2, 3], dtype=np.int32), dtype=wp.int32, device=device)
    empty_wp = wp.empty(0, dtype=wp.int32, device=device)
    vertices_all_wp, vertex_offsets_wp, faces_all_wp = tw.selection.submeshes_from_face_groups(
        vertices_wp, faces_wp, empty_wp, empty_wp
    )
    assert vertices_all_wp.shape == (0,)
    assert vertex_offsets_wp.shape == (0,)
    assert faces_all_wp.shape == (0,)


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
    got_vertices_wp, got_faces_wp = tw.selection.submesh_from_face_mask(
        mesh_wp.points, mesh_wp.indices, face_mask
    )
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
    vertex_indices_np = rng.choice(n_vertices, size=max(1, n_vertices // 4), replace=False).astype(
        np.int32
    )
    vertex_indices = wp.array(vertex_indices_np, dtype=wp.int32, device=mesh_wp.points.device)

    face_indices_wp = tw.selection.face_indices_from_vertex_indices(
        mesh_wp.indices, vertex_indices, face_mode=face_mode
    )
    face_indices_ref_np = _face_indices_from_vertex_indices_np(
        mesh_tm.faces, vertex_indices_np, face_mode=face_mode
    )
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
    vertex_indices_np = rng.choice(n_vertices, size=max(3, n_vertices // 5), replace=False).astype(
        np.int32
    )
    vertex_indices = wp.array(vertex_indices_np, dtype=wp.int32, device=mesh_wp.points.device)

    face_indices_np = _face_indices_from_vertex_indices_np(
        mesh_tm.faces, vertex_indices_np, face_mode=face_mode
    )
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


# ---------------------------------------------------------------------------
# Vertex-selection morphology + region utilities
# ---------------------------------------------------------------------------


def _grid_mesh(n: int = 5):
    xs, ys = np.meshgrid(np.arange(float(n)), np.arange(float(n)))
    vertices = np.stack([xs.ravel(), ys.ravel(), np.zeros(n * n)], axis=1).astype(np.float64)
    faces = []
    for r in range(n - 1):
        for c in range(n - 1):
            a = r * n + c
            faces += [a, a + 1, a + n + 1, a, a + n + 1, a + n]
    return vertices, np.array(faces, dtype=np.int32)


def _graph_distance(faces_np: np.ndarray, n: int, seed: np.ndarray) -> np.ndarray:
    """BFS graph distance from the seed set over the undirected mesh edges (NumPy oracle)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    edges = faces_np.reshape(-1, 3)[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return dijkstra(graph, indices=np.flatnonzero(seed), unweighted=True).min(axis=0)


def test_expand_vertex_mask(device: str):
    vertices_np, faces_np = _grid_mesh(6)
    n = len(vertices_np)
    faces_wp = wp.array(faces_np, dtype=wp.int32, device=device)
    seed = np.zeros(n, dtype=bool)
    seed[len(vertices_np) // 2] = True
    seed_wp = wp.array(seed, dtype=wp.bool, device=device)
    for hops in (1, 2, 3):
        got = tw.selection.expand_vertex_mask(faces_wp, seed_wp, hops).numpy()
        expected = _graph_distance(faces_np, n, seed) <= hops
        assert np.array_equal(got, expected)


def test_expand_vertex_mask_matches_pymeshlab_dilatation(device: str):
    """
    Mask growth against MeshLab's Dilate Selection, which is the only external check it has.

    Neither trimesh nor open3d nor libigl has selection morphology. MeshLab does, but it dilates the
    *face* set, so the composition that lines up with a vertex-mask hop is:

        seed one vertex -> transfer to faces (``inclusive=False``, any selected vertex)
        -> k x Dilate Selection -> read the *vertex* selection back

    which is exactly ``expand_vertex_mask(seed, k)``. Verified to the element on a 9x9 grid for
    ``k = 1..4``. Note ``inclusive=False`` is load-bearing: the default ``True`` selects only faces
    whose *every* vertex is selected, which on a single-vertex seed is no faces at all and clears
    the selection.

    **Erosion does not map the same way** and is deliberately not checked here: MeshLab's Erode
    Selection removes a face when any of its vertices is on the boundary of the selection, so
    reading the vertex selection back gives the vertices of the surviving *faces* -- measured at
    51 / 39 / 25 vertices after 1 / 2 / 3 erosions where ``shrink_vertex_mask`` gives 19 / 7 / 1.
    Different operation, not a discrepancy. ``test_shrink_vertex_mask`` keeps the scipy oracle.
    """
    vertices_np, faces_np = _grid_mesh(9)
    n_vertices = len(vertices_np)
    faces_wp = wp.array(faces_np, dtype=wp.int32, device=device)
    seed_np = np.zeros(n_vertices, dtype=bool)
    centre = n_vertices // 2
    seed_np[centre] = True
    seed_wp = wp.array(seed_np, dtype=wp.bool, device=device)

    mesh_tm = tm.Trimesh(vertices_np, faces_np.reshape(-1, 3), process=False)
    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.compute_selection_by_condition_per_vertex(condselect=f"(vi == {centre})")
    meshset_pml.compute_selection_transfer_vertex_to_face(inclusive=False)

    for hops in (1, 2, 3, 4):
        meshset_pml.apply_selection_dilatation()
        assert np.array_equal(
            tw.selection.expand_vertex_mask(faces_wp, seed_wp, hops).numpy(),
            meshset_pml.current_mesh().vertex_selection_array(),
        )


def test_shrink_vertex_mask(device: str):
    vertices_np, faces_np = _grid_mesh(6)
    n = len(vertices_np)
    faces_wp = wp.array(faces_np, dtype=wp.int32, device=device)
    seed = np.zeros(n, dtype=bool)
    seed[len(vertices_np) // 2] = True
    seed_wp = wp.array(seed, dtype=wp.bool, device=device)
    dilated = tw.selection.expand_vertex_mask(faces_wp, seed_wp, 2)
    shrunk = tw.selection.shrink_vertex_mask(faces_wp, dilated, 1).numpy()
    # shrink = complement of expand of complement: vertex kept iff all 1-ring neighbours dilated.
    dilated_np = dilated.numpy()
    dist = _graph_distance(faces_np, n, ~dilated_np)
    expected = dist > 1
    assert np.array_equal(shrunk, expected)


def test_region_boundary_edges(device: str):
    _, faces_np = _grid_mesh(5)
    n_faces = len(faces_np) // 3
    faces_wp = wp.array(faces_np, dtype=wp.int32, device=device)
    region = np.zeros(n_faces, dtype=bool)
    region[:6] = True  # a contiguous block of faces
    region_wp = wp.array(region, dtype=wp.bool, device=device)

    got = tw.selection.region_boundary_edges(faces_wp, region_wp).numpy()
    got_set = {tuple(sorted(int(x) for x in e)) for e in got}

    # Oracle: undirected edges with exactly two incident faces, exactly one in the region.
    faces = faces_np.reshape(-1, 3)
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for fi, t in enumerate(faces):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edge_faces.setdefault((int(min(a, b)), int(max(a, b))), []).append(fi)
    expected = {
        e for e, fs in edge_faces.items() if len(fs) == 2 and (region[fs[0]] ^ region[fs[1]])
    }
    assert got_set == expected


def test_exclude_fully_selected_components(device: str):
    ico = tm.creation.icosahedron()
    hemi = tm.creation.icosphere(subdivisions=1)
    v_ico = ico.vertices.astype(np.float64)
    v_hemi = hemi.vertices.astype(np.float64) + np.array([5.0, 0.0, 0.0])
    v_wp_ico = wp.array(v_ico, dtype=wp.vec3, device=device)
    f_wp_ico = wp.array(ico.faces.astype(np.int32).reshape(-1), dtype=wp.int32, device=device)
    v_wp_hemi = wp.array(v_hemi, dtype=wp.vec3, device=device)
    f_wp_hemi = wp.array(hemi.faces.astype(np.int32).reshape(-1), dtype=wp.int32, device=device)
    verts, faces = tw.combine.concatenate([(v_wp_ico, f_wp_ico), (v_wp_hemi, f_wp_hemi)])

    n = int(verts.shape[0])
    n_ico = len(v_ico)
    mask = np.zeros(n, dtype=bool)
    mask[:n_ico] = True  # whole icosahedron component
    mask[n_ico : n_ico + 3] = True  # partial hemisphere component
    mask_wp = wp.array(mask, dtype=wp.bool, device=device)

    result = tw.selection.exclude_fully_selected_components(faces, mask_wp, n).numpy()
    # The fully-selected icosahedron component is dropped; the partial hemisphere subset stays.
    assert not result[:n_ico].any()
    assert np.array_equal(result[n_ico : n_ico + 3], np.ones(3, dtype=bool))
