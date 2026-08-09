"""Regression tests for ``triwarp.mesh.Trimesh`` against Trimesh (CPU reference)."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh as tm
import warp as wp

import triwarp as tw
from triwarp.mesh import _TOPOLOGY_KEYS

CLOSED_MESHES = ["icosahedron", "cave_cube"]
OPEN_MESHES = ["hemisphere", "half_torus"]
ALL_MESHES = CLOSED_MESHES + OPEN_MESHES


def _lexsort_rows(rows: np.ndarray) -> np.ndarray:
    """Sort ``(n, 2)`` rows lexicographically (rows kept intact) for set comparison."""
    order = np.lexsort((rows[:, 1], rows[:, 0]))
    return rows[order]


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
    # A warp.Mesh with zero triangles silently corrupts CUDA state (Warp 1.15); warp_mesh must
    # raise instead of building one.
    vertices_wp = wp.empty(0, dtype=wp.vec3, device=device)
    faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    mesh = tw.Trimesh(vertices_wp, faces_wp)
    with pytest.raises(ValueError, match="zero triangles"):
        _ = mesh.warp_mesh


def test_mesh_from_numpy_round_trip(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
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
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = tw.Trimesh.from_warp_mesh(mesh_wp)

    assert np.array_equal(mesh.edges.numpy(), mesh_tm.edges)
    assert np.array_equal(mesh.edges_sorted.numpy(), mesh_tm.edges_sorted)
    assert np.array_equal(mesh.edges_face.numpy(), mesh_tm.edges_face)


@pytest.mark.parametrize("mesh_name", ALL_MESHES)
def test_edges_unique_matches_trimesh(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = tw.Trimesh.from_warp_mesh(mesh_wp)

    unique_wp = mesh.edges_unique.numpy()
    unique_tm = mesh_tm.edges_unique
    assert np.array_equal(_lexsort_rows(unique_wp), _lexsort_rows(unique_tm))

    # unique_edges[inverse] must reconstruct edges_sorted, independent of row order.
    reconstructed = unique_wp[mesh.edges_unique_inverse.numpy()]
    assert np.array_equal(reconstructed, mesh.edges_sorted.numpy())


@pytest.mark.parametrize("mesh_name", ALL_MESHES)
@pytest.mark.parity("mesh_face_adjacency", "trimesh")
def test_face_adjacency_matches_trimesh(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = tw.Trimesh.from_warp_mesh(mesh_wp)

    adjacency_wp = mesh.face_adjacency.numpy()
    adjacency_tm = tm.graph.face_adjacency(mesh=mesh_tm)
    assert np.array_equal(_lexsort_rows(adjacency_wp), _lexsort_rows(np.sort(adjacency_tm, axis=1)))


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
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = tw.Trimesh.from_warp_mesh(mesh_wp)
    assert mesh.euler_characteristic == int(mesh_tm.euler_number)


@pytest.mark.parametrize("mesh_name", ALL_MESHES)
def test_is_winding_consistent_matches_trimesh(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = tw.Trimesh.from_warp_mesh(mesh_wp)
    assert mesh.is_winding_consistent == bool(mesh_tm.is_winding_consistent)


@pytest.mark.parametrize("mesh_name", ALL_MESHES)
def test_is_volume_matches_trimesh(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = tw.Trimesh.from_warp_mesh(mesh_wp)
    assert mesh.is_volume == bool(mesh_tm.is_volume)


@pytest.mark.parametrize("mesh_name", ALL_MESHES)
def test_is_watertight_matches_free_function(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    # triwarp.is_watertight follows Open3D semantics (manifold-closed + no self-intersections),
    # not trimesh's "every edge shared by exactly two faces" — compare against the free
    # function, not mesh_tm.is_watertight.
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh = tw.Trimesh.from_warp_mesh(mesh_wp)
    expected = tw.validation.is_watertight(mesh_wp.points, mesh_wp.indices)
    assert mesh.is_watertight == expected


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
