"""Regression tests for ``triwarp.remesh`` against Trimesh (CPU reference)."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh as tm
import warp as wp
from scipy.spatial import KDTree

import triwarp as tw
from tests.conversions import trimesh_to_warp


def _undirected_edges(faces_np: np.ndarray) -> np.ndarray:
    """Sorted ``(n_faces * 3, 2)`` undirected edges of a ``(n_faces, 3)`` face array."""
    return np.sort(faces_np[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)


def _max_edge_length(vertices_np: np.ndarray, faces_np: np.ndarray) -> float:
    edges = vertices_np[_undirected_edges(faces_np)]
    return float(np.linalg.norm(edges[:, 0] - edges[:, 1], axis=1).max())


def _surface_area(vertices_np: np.ndarray, faces_np: np.ndarray) -> float:
    tris = vertices_np[faces_np]
    cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    return float(np.linalg.norm(cross, axis=1).sum() * 0.5)


def _signed_volume(vertices_np: np.ndarray, faces_np: np.ndarray) -> float:
    tris = vertices_np[faces_np]
    return float(np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum() / 6.0)


def test_subdivide(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = icosahedron

    vertices_np = mesh_tm.vertices.astype(np.float32)
    faces_np = mesh_tm.faces.astype(np.int32)

    new_v_tm, new_f_tm = tm.remesh.subdivide(vertices_np.astype(np.float64), faces_np)
    new_v_tm = new_v_tm.astype(np.float32)

    new_v_wp, new_f_wp = tw.remesh.subdivide(mesh_wp.points, mesh_wp.indices)

    new_v_wp_np = new_v_wp.numpy()
    new_f_wp_np = new_f_wp.numpy().reshape(-1, 3)
    new_f_tm_np = new_f_tm.reshape(-1, 3)

    assert new_v_wp_np.shape[0] == new_v_tm.shape[0], (
        f"vertex count mismatch: got {new_v_wp_np.shape[0]}, expected {new_v_tm.shape[0]}"
    )
    assert new_f_wp_np.shape[0] == new_f_tm_np.shape[0], (
        f"face count mismatch: got {new_f_wp_np.shape[0]}, expected {new_f_tm_np.shape[0]}"
    )

    centroids_wp = new_v_wp_np[new_f_wp_np].mean(axis=1)
    centroids_tm = new_v_tm[new_f_tm_np].mean(axis=1)
    order_wp = np.lexsort(centroids_wp.T[::-1])
    order_tm = np.lexsort(centroids_tm.T[::-1])
    assert np.allclose(centroids_wp[order_wp], centroids_tm[order_tm], rtol=1e-5, atol=1e-5), (
        "face centroid sets do not match"
    )


def test_subdivide_empty(device: str) -> None:
    vertices_wp = wp.empty(0, dtype=wp.vec3, device=device)
    faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    new_v_wp, new_f_wp = tw.remesh.subdivide(vertices_wp, faces_wp)
    assert int(new_v_wp.shape[0]) == 0
    assert int(new_f_wp.shape[0]) == 0


def test_subdivide_edge_lengths(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """All edges in the subdivided mesh are at most half the longest original edge."""
    mesh_tm, mesh_wp = icosahedron

    vertices_np = mesh_tm.vertices.astype(np.float32)
    faces_np = mesh_tm.faces.astype(np.int32)

    new_v_wp, new_f_wp = tw.remesh.subdivide(mesh_wp.points, mesh_wp.indices)

    new_v_np = new_v_wp.numpy()
    new_f_np = new_f_wp.numpy().reshape(-1, 3)

    orig_max_edge = _max_edge_length(vertices_np, faces_np)
    new_max_edge = _max_edge_length(new_v_np, new_f_np)

    assert new_max_edge <= orig_max_edge / 2.0 + 1e-5


# --------------------------------------------------------------------------------------
# subdivide_to_size
# --------------------------------------------------------------------------------------

_MESH_FIXTURES = ["icosahedron", "half_torus", "cave_cube", "hemisphere"]
_CLOSED_FIXTURES = ["icosahedron", "cave_cube"]


def test_subdivide_to_size_reference_regular(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """A single pass on a regular mesh (all faces 1->4) matches trimesh exactly."""
    mesh_tm, mesh_wp = icosahedron
    faces_np = mesh_tm.faces.astype(np.int32)
    max_edge = 0.6 * _max_edge_length(mesh_tm.vertices.astype(np.float32), faces_np)

    new_v_wp, new_f_wp, index_wp = tw.remesh.subdivide_to_size(
        mesh_wp.points, mesh_wp.indices, max_edge, return_index=True
    )
    new_v_np = new_v_wp.numpy()
    new_f_np = new_f_wp.numpy().reshape(-1, 3)

    ref_v, ref_f, ref_index = tm.remesh.subdivide_to_size(
        mesh_tm.vertices, mesh_tm.faces, max_edge, return_index=True
    )

    assert new_v_np.shape[0] == ref_v.shape[0]
    assert new_f_np.shape[0] == ref_f.shape[0]

    # Map warp vertices onto the trimesh vertex ids (identical set up to fp precision).
    dist_np, wp_to_ref = KDTree(ref_v).query(new_v_np)
    assert dist_np.max() < 1e-4

    faces_mapped = np.sort(wp_to_ref[new_f_np], axis=1)
    faces_ref = np.sort(ref_f, axis=1)
    order_mapped = np.lexsort(faces_mapped.T[::-1])
    order_ref = np.lexsort(faces_ref.T[::-1])
    assert np.array_equal(faces_mapped[order_mapped], faces_ref[order_ref])

    n_in_faces = mesh_tm.faces.shape[0]
    hist_wp = np.bincount(index_wp.numpy(), minlength=n_in_faces)
    hist_ref = np.bincount(ref_index, minlength=n_in_faces)
    assert np.array_equal(hist_wp, hist_ref)


def test_subdivide_to_size_reference_mixed(device: str) -> None:
    """A single pass with mixed 1/2/3-split faces matches trimesh exactly."""
    # A stretched icosahedron gives two distinct edge lengths so that, for an
    # intermediate threshold, faces split on 1, 2, or 3 edges in one pass.
    mesh_tm = tm.creation.icosahedron()
    mesh_tm.vertices = mesh_tm.vertices * np.array([1.0, 1.0, 2.2])
    mesh_wp = trimesh_to_warp(mesh_tm, device)
    max_edge = 1.9

    new_v_wp, new_f_wp, index_wp = tw.remesh.subdivide_to_size(
        mesh_wp.points, mesh_wp.indices, max_edge, max_iter=1, return_index=True
    )
    new_v_np = new_v_wp.numpy()
    new_f_np = new_f_wp.numpy().reshape(-1, 3)

    ref_v, ref_f, ref_index = tm.remesh.subdivide_to_size(
        mesh_tm.vertices, mesh_tm.faces, max_edge, max_iter=1, return_index=True
    )

    assert new_v_np.shape[0] == ref_v.shape[0]
    assert new_f_np.shape[0] == ref_f.shape[0]

    dist_np, wp_to_ref = KDTree(ref_v).query(new_v_np)
    assert dist_np.max() < 1e-4

    faces_mapped = np.sort(wp_to_ref[new_f_np], axis=1)
    faces_ref = np.sort(ref_f, axis=1)
    assert np.array_equal(
        faces_mapped[np.lexsort(faces_mapped.T[::-1])],
        faces_ref[np.lexsort(faces_ref.T[::-1])],
    )

    hist_wp = np.bincount(index_wp.numpy(), minlength=mesh_tm.faces.shape[0])
    hist_ref = np.bincount(ref_index, minlength=mesh_tm.faces.shape[0])
    assert np.array_equal(hist_wp, hist_ref)


@pytest.mark.parametrize("mesh_name", _MESH_FIXTURES)
@pytest.mark.parametrize("frac", [0.75, 0.5, 0.3])
def test_subdivide_to_size_max_edge(mesh_name: str, frac: float, request: pytest.FixtureRequest) -> None:
    """Every edge is at most ``max_edge`` after subdivision (the defining property)."""
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    max_edge = frac * tw.edges.mean_edge_length(mesh_wp.points, mesh_wp.indices)

    new_v_wp, new_f_wp = tw.remesh.subdivide_to_size(mesh_wp.points, mesh_wp.indices, max_edge)
    result_max_edge = _max_edge_length(new_v_wp.numpy(), new_f_wp.numpy().reshape(-1, 3))

    assert result_max_edge <= max_edge + 1e-4


@pytest.mark.parametrize("mesh_name", _MESH_FIXTURES)
def test_subdivide_to_size_noop(mesh_name: str, request: pytest.FixtureRequest) -> None:
    """A threshold above the longest edge returns the mesh unchanged."""
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    faces_np = mesh_tm.faces.astype(np.int32)
    max_edge = 2.0 * _max_edge_length(mesh_wp.points.numpy(), faces_np)

    new_v_wp, new_f_wp = tw.remesh.subdivide_to_size(mesh_wp.points, mesh_wp.indices, max_edge)

    assert np.array_equal(new_v_wp.numpy(), mesh_wp.points.numpy())
    assert np.array_equal(new_f_wp.numpy().reshape(-1, 3), faces_np)


@pytest.mark.parametrize("mesh_name", _CLOSED_FIXTURES)
@pytest.mark.parametrize("frac", [0.5, 0.3])
def test_subdivide_to_size_crack_free(
    mesh_name: str, frac: float, request: pytest.FixtureRequest
) -> None:
    """Closed input stays watertight: every undirected edge is shared by exactly 2 faces."""
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    max_edge = frac * tw.edges.mean_edge_length(mesh_wp.points, mesh_wp.indices)

    new_v_wp, new_f_wp = tw.remesh.subdivide_to_size(mesh_wp.points, mesh_wp.indices, max_edge)
    new_f_np = new_f_wp.numpy().reshape(-1, 3)

    _, counts = np.unique(_undirected_edges(new_f_np), axis=0, return_counts=True)
    assert np.array_equal(counts, np.full(counts.shape, 2)), "T-junctions / cracks introduced"

    # Euler characteristic is preserved (no topology change).
    n_v = new_v_wp.numpy().shape[0]
    n_e = np.unique(_undirected_edges(new_f_np), axis=0).shape[0]
    n_f = new_f_np.shape[0]
    assert n_v - n_e + n_f == mesh_tm.euler_number


@pytest.mark.parametrize("mesh_name", _MESH_FIXTURES)
def test_subdivide_to_size_preserves_surface(
    mesh_name: str, request: pytest.FixtureRequest
) -> None:
    """Midpoints lie on original edges, so surface area (and closed volume) is unchanged."""
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = mesh_wp.points.numpy()
    faces_np = mesh_tm.faces.astype(np.int32)
    max_edge = 0.4 * _max_edge_length(vertices_np, faces_np)

    new_v_wp, new_f_wp = tw.remesh.subdivide_to_size(mesh_wp.points, mesh_wp.indices, max_edge)
    new_v_np = new_v_wp.numpy()
    new_f_np = new_f_wp.numpy().reshape(-1, 3)

    assert np.isclose(
        _surface_area(new_v_np, new_f_np), _surface_area(vertices_np, faces_np), rtol=1e-4
    )
    if mesh_tm.is_watertight:
        assert np.isclose(
            _signed_volume(new_v_np, new_f_np), _signed_volume(vertices_np, faces_np), rtol=1e-4
        )


@pytest.mark.parametrize("mesh_name", _MESH_FIXTURES)
def test_subdivide_to_size_return_index(
    mesh_name: str, request: pytest.FixtureRequest
) -> None:
    """Each output face carries a valid source id and lies inside that source triangle."""
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = mesh_wp.points.numpy()
    faces_np = mesh_tm.faces.astype(np.int32)
    n_in_faces = faces_np.shape[0]
    max_edge = 0.5 * _max_edge_length(vertices_np, faces_np)

    new_v_wp, new_f_wp, index_wp = tw.remesh.subdivide_to_size(
        mesh_wp.points, mesh_wp.indices, max_edge, return_index=True
    )
    new_v_np = new_v_wp.numpy()
    new_f_np = new_f_wp.numpy().reshape(-1, 3)
    index_np = index_wp.numpy()

    assert index_np.shape[0] == new_f_np.shape[0]
    assert index_np.min() >= 0 and index_np.max() < n_in_faces

    # Every output-face centroid lies inside its claimed source triangle.
    centroids = new_v_np[new_f_np].mean(axis=1)
    src = vertices_np[faces_np[index_np]]
    a, b, c = src[:, 0], src[:, 1], src[:, 2]
    v0, v1, v2 = b - a, c - a, centroids - a
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    denom = d00 * d11 - d01 * d01
    bary_v = (d11 * d20 - d01 * d21) / denom
    bary_w = (d00 * d21 - d01 * d20) / denom
    bary_u = 1.0 - bary_v - bary_w
    tol = 1e-3
    assert np.all(bary_u >= -tol) and np.all(bary_v >= -tol) and np.all(bary_w >= -tol)
    assert np.all(bary_u <= 1.0 + tol) and np.all(bary_v <= 1.0 + tol) and np.all(bary_w <= 1.0 + tol)


def test_subdivide_to_size_empty(device: str) -> None:
    vertices_wp = wp.empty(0, dtype=wp.vec3, device=device)
    faces_wp = wp.empty(0, dtype=wp.int32, device=device)

    new_v_wp, new_f_wp, index_wp = tw.remesh.subdivide_to_size(
        vertices_wp, faces_wp, 1.0, return_index=True
    )
    assert int(new_v_wp.shape[0]) == 0
    assert int(new_f_wp.shape[0]) == 0
    assert int(index_wp.shape[0]) == 0


def test_subdivide_to_size_single_triangle(device: str) -> None:
    """A single triangle with one over-long edge splits into two faces (pymesh case)."""
    # Edges: base (0,1) = 2.0, the other two ~1.044. With max_edge = 1.5 only the
    # base edge is over-long, so it splits once into two faces.
    vertices_np = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.3, 0.0]], dtype=np.float32)
    faces_np = np.array([0, 1, 2], dtype=np.int32)
    vertices_wp = wp.array(vertices_np, dtype=wp.vec3, device=device)
    faces_wp = wp.array(faces_np, dtype=wp.int32, device=device)

    new_v_wp, new_f_wp = tw.remesh.subdivide_to_size(vertices_wp, faces_wp, 1.5)
    new_v_np = new_v_wp.numpy()
    new_f_np = new_f_wp.numpy().reshape(-1, 3)

    assert new_v_np.shape[0] == 4  # one midpoint added
    assert new_f_np.shape[0] == 2
    assert _max_edge_length(new_v_np, new_f_np) <= 1.5 + 1e-5
    # midpoint of the base edge (0,1) at (1, 0, 0) is present
    assert np.isclose(new_v_np, np.array([1.0, 0.0, 0.0])).all(axis=1).any()


def test_subdivide_to_size_max_iter_exceeded(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = icosahedron
    small_edge = 0.1 * tw.edges.mean_edge_length(mesh_wp.points, mesh_wp.indices)
    with pytest.raises(ValueError, match="max_iter exceeded"):
        tw.remesh.subdivide_to_size(mesh_wp.points, mesh_wp.indices, small_edge, max_iter=0)
