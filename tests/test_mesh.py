"""Regression tests for ``triwarp.mesh.Trimesh`` against Trimesh (CPU reference)."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh as tm
import warp as wp

import triwarp as tw
from tests.comparisons import assert_same_loop_set, lexsort_rows, trimesh_outline_loops
from triwarp.mesh import _TOPOLOGY_KEYS

CLOSED_MESHES = ["icosahedron", "cave_cube"]
OPEN_MESHES = ["hemisphere", "half_torus"]
ALL_MESHES = CLOSED_MESHES + OPEN_MESHES


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def test_construction_flat_and_2d_faces_agree(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    mesh_flat = tw.Trimesh(mesh_wp.points, mesh_wp.indices)
    mesh_2d = tw.Trimesh(mesh_wp.points, mesh_wp.indices.reshape((-1, 3)))
    assert np.array_equal(mesh_flat.faces.numpy(), mesh_2d.faces.numpy())


def test_construction_bad_faces_ndim_raises(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    n_faces = mesh_wp.indices.shape[0] // 3
    bad_faces = wp.zeros((n_faces, 3, 1), dtype=wp.int32, device=mesh_wp.device)
    with pytest.raises(TypeError):
        tw.Trimesh(mesh_wp.points, bad_faces)


def test_construction_bad_2d_faces_shape_raises(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    bad_faces = wp.zeros((mesh_wp.indices.shape[0] // 4, 4), dtype=wp.int32, device=mesh_wp.device)
    with pytest.raises(ValueError, match="shape"):
        tw.Trimesh(mesh_wp.points, bad_faces)


def test_construction_faces_size_not_multiple_of_three_raises(device: str) -> None:
    vertices_wp = wp.zeros(4, dtype=wp.vec3, device=device)
    faces_wp = wp.zeros(4, dtype=wp.int32, device=device)
    with pytest.raises(ValueError, match="multiple of 3"):
        tw.Trimesh(vertices_wp, faces_wp)


def test_from_warp_mesh_seeds_warp_mesh_cache(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    mesh = tw.Trimesh.from_warp_mesh(mesh_wp)
    assert mesh.warp_mesh is mesh_wp
    assert mesh.vertices is mesh_wp.points
    assert mesh.faces is mesh_wp.indices


def test_warp_mesh_raises_for_empty_mesh(device: str) -> None:
    # A warp.Mesh with zero triangles silently corrupts CUDA state (Warp 1.16); warp_mesh must
    # raise instead of building one.
    vertices_wp = wp.empty(0, dtype=wp.vec3, device=device)
    faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    mesh = tw.Trimesh(vertices_wp, faces_wp)
    with pytest.raises(ValueError, match="zero triangles"):
        _ = mesh.warp_mesh


def test_mesh_from_numpy_round_trip(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """Class A: the numpy arrays survive the upload unchanged, positions and indices alike."""
    mesh_tm, _mesh_wp = icosahedron
    mesh = tw.io.mesh_from_numpy(mesh_tm.vertices, mesh_tm.faces, device="cpu")
    assert np.allclose(mesh.vertices.numpy(), mesh_tm.vertices, rtol=1e-5, atol=1e-5)
    assert np.array_equal(mesh.faces.numpy().reshape(-1, 3), mesh_tm.faces)


# ---------------------------------------------------------------------------
# geometry vs trimesh
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", ALL_MESHES)
@pytest.mark.parity("mesh_vertex_normals", "trimesh")
def test_geometry_matches_trimesh(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A on all seven cached properties: same quantities, same order, no transform.

    ``mesh.py`` mirrors ``trimesh.Trimesh``'s property names deliberately (the one allowlisted
    exception to the summary-line rule in section 10), so this is the test that the names mean the
    same thing and not merely that they exist. Run over every fixture, including the
    non-orientable ones, since ``vertex_normals`` and ``vertex_defects`` are where a winding
    assumption would show.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = tw.Trimesh.from_warp_mesh(mesh_wp)

    assert np.allclose(mesh.face_normals.numpy(), mesh_tm.face_normals, rtol=1e-5, atol=1e-5)
    assert np.allclose(mesh.face_areas.numpy(), mesh_tm.area_faces, rtol=1e-5, atol=1e-5)
    assert np.allclose(mesh.area, mesh_tm.area, rtol=1e-5, atol=1e-5)
    assert np.allclose(mesh.face_angles.numpy(), mesh_tm.face_angles, rtol=1e-5, atol=1e-5)
    assert np.allclose(mesh.centroid, mesh_tm.centroid, rtol=1e-5, atol=1e-5)
    assert np.allclose(mesh.vertex_normals.numpy(), mesh_tm.vertex_normals, rtol=1e-5, atol=1e-5)
    assert np.allclose(mesh.vertex_defects.numpy(), mesh_tm.vertex_defects, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# edges / adjacency vs trimesh
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", ALL_MESHES)
def test_edges_match_trimesh(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A, and the row *order* is part of the claim -- ``array_equal``, not a lexsort.

    ``edges`` / ``edges_sorted`` / ``edges_face`` are all indexed by the same halfedge position, so
    a reordering would silently break the correspondence between them even while each stayed a
    correct set. That is why this one is not canonicalized where
    [`test_edges_unique_matches_trimesh`] is.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = tw.Trimesh.from_warp_mesh(mesh_wp)

    assert np.array_equal(mesh.edges.numpy(), mesh_tm.edges)
    assert np.array_equal(mesh.edges_sorted.numpy(), mesh_tm.edges_sorted)
    assert np.array_equal(mesh.edges_face.numpy(), mesh_tm.edges_face)


@pytest.mark.parametrize("mesh_name", ALL_MESHES)
def test_edges_unique_matches_trimesh(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class B (row-set canonicalization): the unique edge *set* matches after a shared lexsort.

    Here the order genuinely is not defined by either library -- triwarp's comes from a parallel
    hash and trimesh's from a serial scan -- so both sides are sorted. The `inverse` is then checked
    against triwarp's own ``edges_sorted`` rather than trimesh's, because it indexes into triwarp's
    row order and that is the invariant the sort throws away.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = tw.Trimesh.from_warp_mesh(mesh_wp)

    unique_wp = mesh.edges_unique.numpy()
    unique_tm = mesh_tm.edges_unique
    assert np.array_equal(lexsort_rows(unique_wp), lexsort_rows(unique_tm))

    # unique_edges[inverse] must reconstruct edges_sorted, independent of row order.
    reconstructed = unique_wp[mesh.edges_unique_inverse.numpy()]
    assert np.array_equal(reconstructed, mesh.edges_sorted.numpy())


@pytest.mark.parametrize("mesh_name", ALL_MESHES)
@pytest.mark.parity("mesh_face_adjacency", "trimesh")
def test_face_adjacency_matches_trimesh(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class B, two named transforms: sort within each pair, then lexsort the rows.

    An adjacency pair is unordered and the list of pairs is unordered, and neither library defines
    either -- so both have to be canonicalized before the sets can be compared elementwise.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = tw.Trimesh.from_warp_mesh(mesh_wp)

    adjacency_wp = mesh.face_adjacency.numpy()
    adjacency_tm = tm.graph.face_adjacency(mesh=mesh_tm)
    assert np.array_equal(lexsort_rows(adjacency_wp), lexsort_rows(np.sort(adjacency_tm, axis=1)))


# ---------------------------------------------------------------------------
# boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", OPEN_MESHES)
def test_boundary_matches_free_functions(request: pytest.FixtureRequest, mesh_name: str) -> None:
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = tw.Trimesh.from_warp_mesh(mesh_wp)

    boundary_edges_ref = tw.boundary.boundary_edges(mesh_wp.points, mesh_wp.indices)
    assert np.array_equal(mesh.boundary_edges.numpy(), boundary_edges_ref.numpy())

    boundary_vertex_indices_ref = tw.boundary.boundary_vertex_indices(
        mesh_wp.points, mesh_wp.indices
    )
    assert np.array_equal(mesh.boundary_vertex_indices.numpy(), boundary_vertex_indices_ref.numpy())

    assert len(mesh.boundary_loops) > 0


@pytest.mark.parametrize("mesh_name", OPEN_MESHES)
@pytest.mark.parity("mesh_boundary_loops", "trimesh")
def test_boundary_loops_matches_trimesh_outline(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B: the cached property against ``Trimesh.outline()``, the property trimesh caches.

    The container-level twin of
    [`tests.test_boundary.test_boundary_loops_matches_trimesh_outline`][], and it exists separately
    because ``benchmarks/test_mesh.py`` times the two as separate groups -- the free function on the
    ``loops`` axis, this one warm against cold. Both named transforms live in
    [`tests.comparisons.trimesh_outline_loops`][]: the entities index the mesh's own vertex array,
    and a closed entity repeats its first point as its last.

    The loop *set* is the claim, not the order: triwarp ranks loops by length and trimesh by
    traversal, and neither fixes a starting point within a loop.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = tw.Trimesh.from_warp_mesh(mesh_wp)

    loops_tm = trimesh_outline_loops(mesh_tm)
    assert len(loops_tm) > 0  # non-vacuous: these fixtures have rims
    assert_same_loop_set([loop.numpy() for loop in mesh.boundary_loops], loops_tm)


def test_boundary_loops_empty_for_closed_mesh(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    mesh = tw.Trimesh.from_warp_mesh(mesh_wp)
    assert mesh.boundary_loops == []
    assert mesh.boundary_vertex_indices.shape == (0,)


# ---------------------------------------------------------------------------
# predicates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", ALL_MESHES)
def test_euler_characteristic_matches_trimesh(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A on an integer: ``V - E + F`` against ``Trimesh.euler_number``, exactly.

    Non-vacuous across the fixture set by construction -- it is 2 on the closed orientable meshes,
    0 on the tori and ``mobius``, and 1 on ``boy_surface`` -- so a function returning a constant
    could not pass this parametrisation.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = tw.Trimesh.from_warp_mesh(mesh_wp)
    assert mesh.euler_characteristic == int(mesh_tm.euler_number)


@pytest.mark.parametrize("mesh_name", ALL_MESHES)
def test_is_winding_consistent_matches_trimesh(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A on a boolean, and parametrized over inputs giving *both* answers.

    Section 6 rules out asserting a predicate on one branch only; the non-orientable fixtures are
    what make this live, since every closed orientable mesh in the suite answers ``True``.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = tw.Trimesh.from_warp_mesh(mesh_wp)
    assert mesh.is_winding_consistent == bool(mesh_tm.is_winding_consistent)


@pytest.mark.parametrize("mesh_name", ALL_MESHES)
def test_is_volume_matches_trimesh(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A on a boolean: the conjunction trimesh calls ``is_volume``, over both answers.

    The open fixtures supply the ``False`` branch, so this is not a one-branch assert.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = tw.Trimesh.from_warp_mesh(mesh_wp)
    assert mesh.is_volume == bool(mesh_tm.is_volume)


@pytest.mark.parametrize("mesh_name", ALL_MESHES)
def test_is_watertight_matches_free_function(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    # triwarp.is_watertight follows Open3D semantics (manifold-closed + no self-intersections),
    # not trimesh's "every edge shared by exactly two faces" — compare against the free
    # function, not mesh_tm.is_watertight. The trimesh comparison is the test below, which
    # decomposes triwarp's answer into trimesh's clause and the one trimesh does not have.
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = tw.Trimesh.from_warp_mesh(mesh_wp)
    expected = tw.validation.is_watertight(mesh_wp.points, mesh_wp.indices)
    assert mesh.is_watertight == expected


@pytest.mark.parametrize("mesh_name", [*ALL_MESHES, "bohemian_dome"])
@pytest.mark.parity("mesh_is_watertight", "trimesh")
def test_is_watertight_decomposes_into_trimesh_clause(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B: ``Trimesh.is_watertight`` is exactly one of the two conjuncts this property tests.

    The named transform is the conjunction itself. trimesh answers "every edge is shared by exactly
    two faces"; triwarp follows Open3D and requires that **and** no self-intersection, so the two
    disagree by construction on a closed surface that passes through itself and comparing them
    directly would look like a bug in triwarp. Supplying the missing conjunct makes it an equality:
    ``mesh.is_watertight == mesh_tm.is_watertight and not is_self_intersecting(...)``, measured
    exact on all eight surfaces probed (the four fixtures here plus ``roman``, ``klein``,
    ``cross_cap`` and ``mobius``).

    That is a stronger claim than an implication, and it is the one worth making: an implication
    (triwarp ⟹ trimesh) would also pass for a property that always answered ``False``.

    Non-vacuous in both directions on this parametrisation -- ``icosahedron`` and ``cave_cube`` are
    watertight both ways, the two open fixtures fail trimesh's clause, and ``bohemian_dome`` is the
    input that separates the two definitions: trimesh ``True``, triwarp ``False``.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = tw.Trimesh.from_warp_mesh(mesh_wp)

    edge_manifold_closed_tm = bool(mesh_tm.is_watertight)
    self_intersecting = tw.validation.is_self_intersecting(mesh_wp)
    assert mesh.is_watertight == (edge_manifold_closed_tm and not self_intersecting)


@pytest.mark.parametrize("mesh_name", CLOSED_MESHES)
def test_is_edge_and_vertex_manifold_closed_meshes(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = tw.Trimesh.from_warp_mesh(mesh_wp)
    assert mesh.is_edge_manifold is True
    assert mesh.is_vertex_manifold is True
    assert mesh.is_self_intersecting is False


# ---------------------------------------------------------------------------
# cache mechanics
# ---------------------------------------------------------------------------


def test_cached_property_identity(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    mesh = tw.Trimesh.from_warp_mesh(mesh_wp)
    assert mesh.face_normals is mesh.face_normals
    assert mesh.edges_unique is mesh.edges_unique


def test_face_normals_and_areas_share_by_product(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    mesh = tw.Trimesh.from_warp_mesh(mesh_wp)
    normals = mesh.face_normals
    assert "face_areas" in mesh._cache
    assert mesh.face_areas is mesh._cache["face_areas"]
    assert mesh.face_normals is normals


def test_edges_unique_shares_inverse_by_product(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    mesh = tw.Trimesh.from_warp_mesh(mesh_wp)
    unique = mesh.edges_unique
    assert "edges_unique_inverse" in mesh._cache
    assert mesh.edges_unique_inverse is mesh._cache["edges_unique_inverse"]
    assert mesh.edges_unique is unique


def test_face_adjacency_shares_edges_by_product(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    mesh = tw.Trimesh.from_warp_mesh(mesh_wp)
    adjacency = mesh.face_adjacency
    assert "face_adjacency_edges" in mesh._cache
    assert mesh.face_adjacency_edges is mesh._cache["face_adjacency_edges"]
    assert mesh.face_adjacency is adjacency


def test_cached_properties_are_frozen(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    mesh = tw.Trimesh.from_warp_mesh(mesh_wp)
    with pytest.raises(AttributeError):
        mesh.face_normals = mesh.face_normals  # type: ignore[misc]


def test_invalidate_clears_cache_and_recomputes(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    mesh = tw.Trimesh.from_warp_mesh(mesh_wp)
    normals_before = mesh.face_normals
    mesh.invalidate()
    assert mesh._cache == {}
    normals_after = mesh.face_normals
    assert normals_after is not normals_before
    assert np.array_equal(normals_after.numpy(), normals_before.numpy())


def test_with_vertices_keeps_topology_drops_geometry(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    _mesh_tm, mesh_wp = icosahedron
    mesh = tw.Trimesh.from_warp_mesh(mesh_wp)
    edges_before = mesh.edges
    _ = mesh.face_normals
    _ = mesh.warp_mesh

    translated_np = mesh.vertices.numpy() + np.array([0.0, 0.0, 1.0], dtype=np.float32)
    new_vertices = wp.array(translated_np, dtype=wp.vec3, device=mesh.device)

    moved = mesh.with_vertices(new_vertices)
    assert moved.edges is edges_before
    assert "face_normals" not in moved._cache
    assert "warp_mesh" not in moved._cache
    assert moved.faces is mesh.faces


@pytest.mark.parametrize("key", sorted(_TOPOLOGY_KEYS))
def test_with_vertices_carries_every_topology_key(
    icosahedron: tuple[tm.Trimesh, wp.Mesh], key: str
) -> None:
    """Each faces-only cached property survives ``with_vertices`` (the ``_TOPOLOGY_KEYS`` rule)."""
    _mesh_tm, mesh_wp = icosahedron
    mesh = tw.Trimesh.from_warp_mesh(mesh_wp)
    before = getattr(mesh, key)
    assert key in mesh._cache

    translated_np = mesh.vertices.numpy() + np.array([0.0, 0.0, 1.0], dtype=np.float32)
    moved = mesh.with_vertices(wp.array(translated_np, dtype=wp.vec3, device=mesh.device))

    assert key in moved._cache, f"{key} was recomputed instead of carried forward"
    assert moved._cache[key] is before


def test_face_adjacency_angles_is_not_a_topology_key(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """``face_adjacency_angles`` reads ``face_normals``, so ``with_vertices`` must drop it."""
    _mesh_tm, mesh_wp = icosahedron
    mesh = tw.Trimesh.from_warp_mesh(mesh_wp)
    _ = mesh.face_adjacency_angles

    translated_np = mesh.vertices.numpy() + np.array([0.0, 0.0, 1.0], dtype=np.float32)
    moved = mesh.with_vertices(wp.array(translated_np, dtype=wp.vec3, device=mesh.device))

    assert "face_adjacency_angles" not in moved._cache


def test_with_vertices_wrong_count_raises(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    mesh = tw.Trimesh.from_warp_mesh(mesh_wp)
    too_few = wp.array(mesh.vertices.numpy()[:-1], dtype=wp.vec3, device=mesh.device)
    with pytest.raises(ValueError, match="vertex count"):
        mesh.with_vertices(too_few)


def test_with_faces_starts_with_empty_cache(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    mesh = tw.Trimesh.from_warp_mesh(mesh_wp)
    _ = mesh.edges

    new_mesh = mesh.with_faces(wp.clone(mesh.faces))
    assert new_mesh._cache == {}
    assert np.array_equal(new_mesh.edges.numpy(), mesh.edges.numpy())


def test_warp_mesh_supports_ray_queries(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    mesh = tw.Trimesh.from_warp_mesh(mesh_wp)

    origins = wp.array([wp.vec3(0.0, 0.0, 2.0)], dtype=wp.vec3, device=mesh.device)
    directions = wp.array([wp.vec3(0.0, 0.0, -1.0)], dtype=wp.vec3, device=mesh.device)

    hits_ref = tw.ray.intersects_any(mesh_wp, origins, directions)
    hits = tw.ray.intersects_any(mesh.warp_mesh, origins, directions)
    assert np.array_equal(hits.numpy(), hits_ref.numpy())
