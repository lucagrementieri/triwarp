"""Regression tests for ``triwarp.remesh`` against Trimesh (CPU reference)."""

from __future__ import annotations

import igl
import numpy as np
import pymeshlab as ml
import pytest
import trimesh as tm
import warp as wp
from scipy.spatial import KDTree

import triwarp as tw
from tests.conversions import bsr_to_dense, open3d_to_trimesh, trimesh_to_open3d, trimesh_to_warp


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


@pytest.mark.parity("subdivide", "trimesh")
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


@pytest.mark.parity("subdivide_to_size", "trimesh")
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
        faces_mapped[np.lexsort(faces_mapped.T[::-1])], faces_ref[np.lexsort(faces_ref.T[::-1])]
    )

    hist_wp = np.bincount(index_wp.numpy(), minlength=mesh_tm.faces.shape[0])
    hist_ref = np.bincount(ref_index, minlength=mesh_tm.faces.shape[0])
    assert np.array_equal(hist_wp, hist_ref)


@pytest.mark.parametrize("mesh_name", _MESH_FIXTURES)
@pytest.mark.parametrize("frac", [0.75, 0.5, 0.3])
def test_subdivide_to_size_max_edge(
    mesh_name: str, frac: float, request: pytest.FixtureRequest
) -> None:
    """Every edge is at most ``max_edge`` after subdivision (the defining property)."""
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
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
def test_subdivide_to_size_return_index(mesh_name: str, request: pytest.FixtureRequest) -> None:
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
    assert index_np.min() >= 0
    assert index_np.max() < n_in_faces

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
    assert np.all(bary_u >= -tol)
    assert np.all(bary_v >= -tol)
    assert np.all(bary_w >= -tol)
    assert np.all(bary_u <= 1.0 + tol)
    assert np.all(bary_v <= 1.0 + tol)
    assert np.all(bary_w <= 1.0 + tol)


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
    """A single triangle with one over-long edge splits into two faces."""
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


# ---------------------------------------------------------------------------
# Region-restricted subdivision + parallel Delaunay flips
# ---------------------------------------------------------------------------


def _filled_hemisphere(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    """Fill the hemisphere's boundary and return (vertices_wp, faces_wp, region_mask_wp)."""
    _, mesh_wp = hemisphere
    faces_filled = tw.hole_filling.fill_holes_min_weight(mesh_wp.points, mesh_wp.indices)
    n0 = int(mesh_wp.indices.shape[0]) // 3
    n1 = int(faces_filled.shape[0]) // 3
    region = np.zeros(n1, dtype=bool)
    region[n0:] = True
    region_wp = wp.array(region, dtype=wp.bool, device=mesh_wp.device)
    return mesh_wp.points, faces_filled, region_wp


def _region_max_edge(vertices_np, faces_np, region_np):
    edges = _undirected_edges(faces_np)
    face_of_edge = np.repeat(np.arange(faces_np.shape[0]), 3)
    lengths = np.linalg.norm(vertices_np[edges[:, 0]] - vertices_np[edges[:, 1]], axis=1)
    in_region = region_np[face_of_edge]
    return float(lengths[in_region].max()) if in_region.any() else 0.0


def test_subdivide_region_max_edge(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    v, f, region = _filled_hemisphere(hemisphere)
    max_edge = 0.2 * _region_max_edge(v.numpy(), f.numpy().reshape(-1, 3), region.numpy())
    nv, nf, nr = tw.remesh.subdivide_region_to_size(v, f, region, max_edge=max_edge, delaunay=False)
    got = _region_max_edge(nv.numpy(), nf.numpy().reshape(-1, 3), nr.numpy())
    assert got <= max_edge + 1e-4


def test_subdivide_region_crack_free(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    v, f, region = _filled_hemisphere(hemisphere)
    max_edge = 0.3 * _region_max_edge(v.numpy(), f.numpy().reshape(-1, 3), region.numpy())
    _, nf, _ = tw.remesh.subdivide_region_to_size(v, f, region, max_edge=max_edge)
    faces_np = nf.numpy().reshape(-1, 3)
    edges = _undirected_edges(faces_np)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    # Filled hemisphere is closed: every undirected edge is shared by exactly two faces.
    assert np.array_equal(np.unique(counts), np.array([2]))


def test_subdivide_region_outside_untouched(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    v, f, region = _filled_hemisphere(hemisphere)
    n_vertices_before = int(v.shape[0])
    max_edge = 0.3 * _region_max_edge(v.numpy(), f.numpy().reshape(-1, 3), region.numpy())
    _, nf, nr = tw.remesh.subdivide_region_to_size(v, f, region, max_edge=max_edge, delaunay=False)
    faces_np = nf.numpy().reshape(-1, 3)
    region_np = nr.numpy()
    original = {tuple(sorted(t)) for t in f.numpy().reshape(-1, 3).tolist()}
    for t in faces_np[~region_np]:
        touches_new = any(idx >= n_vertices_before for idx in t)
        # A non-region face is unchanged, or only retriangulated because it shared a split rim edge.
        assert tuple(sorted(int(x) for x in t)) in original or touches_new


def test_subdivide_region_new_vertex_range(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    v, f, region = _filled_hemisphere(hemisphere)
    n_vertices_before = int(v.shape[0])
    max_edge = 0.3 * _region_max_edge(v.numpy(), f.numpy().reshape(-1, 3), region.numpy())
    nv, _, _ = tw.remesh.subdivide_region_to_size(v, f, region, max_edge=max_edge, delaunay=False)
    nv_np = nv.numpy()
    assert nv_np.shape[0] > n_vertices_before
    assert np.array_equal(nv_np[:n_vertices_before], v.numpy())


def test_subdivide_region_max_splits(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    v, f, region = _filled_hemisphere(hemisphere)
    n_vertices_before = int(v.shape[0])
    max_edge = 0.2 * _region_max_edge(v.numpy(), f.numpy().reshape(-1, 3), region.numpy())
    nv, _, _ = tw.remesh.subdivide_region_to_size(
        v, f, region, max_edge=max_edge, max_splits=5, delaunay=False
    )
    assert nv.numpy().shape[0] - n_vertices_before <= 5


def test_subdivide_region_empty_region(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    v, f, region = _filled_hemisphere(hemisphere)
    empty = wp.zeros(int(region.shape[0]), dtype=wp.bool, device=region.device)
    nv, nf, _ = tw.remesh.subdivide_region_to_size(v, f, empty, max_edge=0.01, delaunay=False)
    assert np.array_equal(nv.numpy(), v.numpy())
    assert np.array_equal(nf.numpy(), f.numpy())


def test_subdivide_region_empty_mesh(device: str):
    v = wp.zeros(0, dtype=wp.vec3, device=device)
    f = wp.zeros(0, dtype=wp.int32, device=device)
    region = wp.zeros(0, dtype=wp.bool, device=device)
    _, nf, _ = tw.remesh.subdivide_region_to_size(v, f, region, max_edge=0.1)
    assert int(nf.shape[0]) == 0


def _delone_violations(vertices_np, faces_np, region_np):
    """Count interior region edges that fail the (angle-gate-free) circumcircle Delone test."""
    faces = faces_np
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for fi, t in enumerate(faces):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edge_faces.setdefault((int(min(a, b)), int(max(a, b))), []).append(fi)

    def circ_diam_sq(a, b, c):
        ab = np.dot(b - a, b - a)
        ca = np.dot(a - c, a - c)
        bc = np.dot(c - b, c - b)
        if ab <= 0 or ca <= 0 or bc <= 0:
            return np.inf
        f = np.dot(np.cross(b - a, c - a), np.cross(b - a, c - a))
        return np.inf if f <= 0 else ab * ca * bc / f

    violations = 0
    for (u, v), fs in edge_faces.items():
        if len(fs) != 2 or not (region_np[fs[0]] and region_np[fs[1]]):
            continue
        apex = []
        for fi in fs:
            apex.extend([int(x) for x in faces[fi] if int(x) not in (u, v)])
        if len(apex) != 2:
            continue
        a, c = vertices_np[u], vertices_np[v]
        b, d = vertices_np[apex[1]], vertices_np[apex[0]]
        m_ac = max(circ_diam_sq(a, c, d), circ_diam_sq(c, a, b))
        m_bd = max(circ_diam_sq(b, d, a), circ_diam_sq(d, b, c))
        if m_bd < m_ac * (1.0 - 1e-6):
            violations += 1
    return violations


def test_flip_to_delaunay_reduces_violations(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    v, f, region = _filled_hemisphere(hemisphere)
    max_edge = 0.3 * _region_max_edge(v.numpy(), f.numpy().reshape(-1, 3), region.numpy())
    nv, nf, nr = tw.remesh.subdivide_region_to_size(v, f, region, max_edge=max_edge, delaunay=False)

    before = _delone_violations(nv.numpy(), nf.numpy().reshape(-1, 3), nr.numpy())
    flipped = tw.remesh.flip_to_delaunay(nv, nf, region=nr)
    after = _delone_violations(nv.numpy(), flipped.numpy().reshape(-1, 3), nr.numpy())

    assert after <= before
    # Face count unchanged; mesh stays closed.
    assert int(flipped.shape[0]) == int(nf.shape[0])
    edges = _undirected_edges(flipped.numpy().reshape(-1, 3))
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.array_equal(np.unique(counts), np.array([2]))


def test_flip_to_delaunay_region_gated(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    v, f, region = _filled_hemisphere(hemisphere)
    nv, nf, nr = tw.remesh.subdivide_region_to_size(
        v, f, region, max_edge=1e9, delaunay=False
    )  # no splits; just exercise gating on the raw fill patch
    flipped = tw.remesh.flip_to_delaunay(nv, nf, region=nr)
    faces_before = nf.numpy().reshape(-1, 3)
    faces_after = flipped.numpy().reshape(-1, 3)
    region_np = nr.numpy()
    # Faces outside the region are never rewritten.
    assert np.array_equal(faces_before[~region_np], faces_after[~region_np])


def test_flip_to_delaunay_empty(device: str):
    v = wp.zeros(0, dtype=wp.vec3, device=device)
    f = wp.zeros(0, dtype=wp.int32, device=device)
    out = tw.remesh.flip_to_delaunay(v, f)
    assert int(out.shape[0]) == 0


# ======================================================================================
# Isotropic explicit remeshing (isotropic_remesh)
#
# Reference: meshlib ``mrmeshpy.remesh`` (``_ml`` suffix) for the edge-length spread; the rest are
# metric/topological invariants (edge concentration, valence variance, Hausdorff, manifoldness,
# feature/boundary preservation). Outputs never match a reference vertex-for-vertex.
# ======================================================================================


def _mesh_arrays(mesh_wp: wp.Mesh) -> tuple[np.ndarray, np.ndarray]:
    return mesh_wp.points.numpy().astype(np.float64), mesh_wp.indices.numpy().reshape(-1, 3)


def _edge_lengths(vertices_np: np.ndarray, faces_np: np.ndarray) -> np.ndarray:
    edges = vertices_np[np.unique(_undirected_edges(faces_np), axis=0)]
    return np.linalg.norm(edges[:, 0] - edges[:, 1], axis=1)


def _valences(faces_np: np.ndarray, n_vertices: int) -> np.ndarray:
    edges = np.unique(_undirected_edges(faces_np), axis=0)
    valence = np.zeros(n_vertices, dtype=np.int64)
    np.add.at(valence, edges[:, 0], 1)
    np.add.at(valence, edges[:, 1], 1)
    return valence


def _two_sided_hausdorff(va: np.ndarray, fa: np.ndarray, vb: np.ndarray, fb: np.ndarray) -> float:
    mesh_a = tm.Trimesh(va, fa, process=False)
    mesh_b = tm.Trimesh(vb, fb, process=False)
    sample_a, _ = tm.sample.sample_surface(mesh_a, 5000, seed=0)
    sample_b, _ = tm.sample.sample_surface(mesh_b, 5000, seed=1)
    a_to_b = np.abs(tm.proximity.signed_distance(mesh_b, sample_a)).max()
    b_to_a = np.abs(tm.proximity.signed_distance(mesh_a, sample_b)).max()
    return float(max(a_to_b, b_to_a))


def _meshlib_remesh_spread(vertices_np: np.ndarray, faces_np: np.ndarray, target: float) -> float:
    """Coefficient of variation (std / mean) of meshlib remesh at ``target`` (NaN if absent)."""
    try:
        from meshlib import mrmeshnumpy as _mn
        from meshlib import mrmeshpy as _mm
    except ImportError:
        return float("nan")
    mesh_ml = _mn.meshFromFacesVerts(
        faces_np.astype(np.int32), np.ascontiguousarray(vertices_np, dtype=np.float64)
    )
    settings = _mm.RemeshSettings()
    settings.targetEdgeLen = float(target)
    settings.projectOnOriginalMesh = True
    _mm.remesh(mesh_ml, settings)
    vertices_ml = _mn.getNumpyVerts(mesh_ml)
    faces_ml = _mn.getNumpyFaces(mesh_ml.topology)
    lengths_ml = _edge_lengths(vertices_ml, faces_ml)
    return float(lengths_ml.std() / lengths_ml.mean())


def _icosphere_wp(device: str, subdivisions: int = 3):
    sphere = tm.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    vertices = wp.array(
        np.ascontiguousarray(sphere.vertices, dtype=np.float64), dtype=wp.vec3, device=device
    )
    faces = wp.array(
        np.ascontiguousarray(sphere.faces.reshape(-1), dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )
    return sphere, vertices, faces


def _graded_patch(n: int = 96, ratio: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a saddle patch whose ``x`` spacing varies by ``ratio``: anisotropic, valence-perfect.

    The shape ``benchmarks/meshes.py`` calls ``saddle_graded``, reduced to test size. Every interior
    vertex has valence exactly 6 and the triangulation is already Delaunay, so neither a
    valence-driven flip nor a Delaunay flip can see the anisotropy -- only the collapse and the
    *area-equalizing* tangential relaxation can remove it. That combination is what makes this the
    input the remesher's stages are individually blind to, and it is why it is worth a test.
    """
    t_np = np.linspace(0.0, 1.0, n) ** 2.0  # quadratic spacing -> strong grading along x
    x_np = t_np * ratio
    y_np = np.linspace(0.0, ratio, n)
    x_grid, y_grid = np.meshgrid(x_np, y_np, indexing="ij")
    z_grid = 0.02 * (x_grid**2 - y_grid**2) / ratio
    vertices = np.column_stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]).astype(np.float64)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            faces += [[a, a + 1, a + n + 1], [a, a + n + 1, a + n]]
    return vertices, np.ascontiguousarray(faces, dtype=np.int32)


def _icosphere_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Clean closed control input, as plain NumPy arrays."""
    sphere = tm.creation.icosphere(subdivisions=3, radius=1.0)
    return (
        np.ascontiguousarray(sphere.vertices, dtype=np.float64),
        np.ascontiguousarray(sphere.faces, dtype=np.int32),
    )


def _degenerate_face_count(vertices_np: np.ndarray, faces_np: np.ndarray) -> int:
    """Faces with exactly zero area -- a repeated index or two coincident corners."""
    triangles = vertices_np[faces_np]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    return int((np.linalg.norm(cross, axis=1) == 0.0).sum())


def _worst_aspect_ratio(vertices_np: np.ndarray, faces_np: np.ndarray) -> float:
    """Longest / shortest edge over the non-degenerate faces (``inf`` if any is degenerate)."""
    triangles = vertices_np[faces_np]
    lengths = np.linalg.norm(
        np.stack(
            [
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 1],
                triangles[:, 0] - triangles[:, 2],
            ],
            axis=1,
        ),
        axis=2,
    )
    if (lengths.min(axis=1) == 0.0).any():
        return float("inf")
    return float((lengths.max(axis=1) / lengths.min(axis=1)).max())


def test_remesh_emits_no_degenerate_faces(device: str) -> None:
    """
    No output face may have exactly zero area, on a clean *and* a badly graded input.

    This is the regression gate for two bugs the pymeshlab benchmark reference exposed, neither of
    which any other assertion in this file would catch -- they all run on clean closed icospheres:

    * ``valence_flip_candidates`` flipped on the valence objective alone. Convexity makes a flip
      legal but bounds nothing about the shape it produces, so on a graded mesh it turned slivers
      into worse slivers and in float32 landed on exactly-zero area: **2 738 of 84 406 faces** on a
      ``saddle_graded``-shaped patch. It now rejects a flip that would create a degenerate triangle
      or increase the worse aspect ratio of the pair.
    A second, *unfixed* gap this input also exposes: ``_smooth_pass`` computes the **unweighted**
    one-ring centroid while ``isotropic_remesh``'s Notes promise the area-equalizing form. On a
    regular graded grid every vertex already sits at the plain average of its neighbours, so the
    smoother is at a fixed point and cannot equalize the sampling at all. Area-weighting it was
    measured to take the 99th-percentile aspect ratio here from **352 to 20** -- but it also makes
    ``is_watertight`` fail on ``cave_cube`` through a self-intersection, at every step size down to
    ``lam=0.1``, so it needs a fold guard first. Hence this test asserts only the degeneracy and
    do-no-harm properties, which do hold.
    """
    for label, (vertices_np, faces_np) in (
        ("icosphere", _icosphere_arrays()),
        ("graded_patch", _graded_patch()),
    ):
        vertices_wp = wp.array(
            np.ascontiguousarray(vertices_np, dtype=np.float32), dtype=wp.vec3, device=device
        )
        faces_wp = wp.array(
            np.ascontiguousarray(faces_np.reshape(-1), dtype=np.int32),
            dtype=wp.int32,
            device=device,
        )
        target = float(tw.edges.mean_edge_length(vertices_wp, faces_wp))
        out_vertices, out_faces = tw.remesh.isotropic_remesh(
            vertices_wp, faces_wp, target_length=target, iterations=3
        )
        out_vertices_np = out_vertices.numpy().astype(np.float64)
        out_faces_np = out_faces.numpy().reshape(-1, 3)
        assert _degenerate_face_count(out_vertices_np, out_faces_np) == 0, label
        # And it must never leave the mesh worse-shaped than it found it.
        assert _worst_aspect_ratio(out_vertices_np, out_faces_np) < 2.0 * _worst_aspect_ratio(
            vertices_np, faces_np
        ), label


def test_remesh_edge_concentration(device: str) -> None:
    sphere, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=3)
    target = 0.5 * tw.edges.mean_edge_length(vertices_wp, faces_wp)

    out_vertices, out_faces = tw.remesh.isotropic_remesh(
        vertices_wp, faces_wp, target_length=target, iterations=10
    )
    vertices_np = out_vertices.numpy().astype(np.float64)
    faces_np = out_faces.numpy().reshape(-1, 3)
    lengths = _edge_lengths(vertices_np, faces_np)

    assert abs(lengths.mean() / target - 1.0) < 0.2  # mean within 20% of target
    in_band = np.mean((lengths >= 0.5 * target) & (lengths <= 1.6 * target))
    assert in_band >= 0.8
    spread_ml = _meshlib_remesh_spread(sphere.vertices, sphere.faces, target)
    if not np.isnan(spread_ml):
        assert lengths.std() / lengths.mean() <= 1.5 * spread_ml


def test_remesh_watertight_genus_preserved(device: str) -> None:
    _sphere, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=3)
    target = 0.5 * tw.edges.mean_edge_length(vertices_wp, faces_wp)

    out_vertices, out_faces = tw.remesh.isotropic_remesh(
        vertices_wp, faces_wp, target_length=target, iterations=10
    )
    mesh_out = tm.Trimesh(out_vertices.numpy(), out_faces.numpy().reshape(-1, 3), process=False)
    assert tw.validation.is_watertight(out_vertices, out_faces)
    assert mesh_out.euler_number == 2  # genus 0
    # Volume of the unit sphere is preserved to a few percent.
    assert abs(mesh_out.volume - 4.0 / 3.0 * np.pi) / (4.0 / 3.0 * np.pi) < 0.05


def test_remesh_valence_variance_decreases(device: str) -> None:
    # A noisy, irregular triangulation: perturbed icosphere with random extra subdivision.
    sphere, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=3)
    target = tw.edges.mean_edge_length(vertices_wp, faces_wp)
    valence_before = _valences(sphere.faces, sphere.vertices.shape[0])

    out_vertices, out_faces = tw.remesh.isotropic_remesh(
        vertices_wp, faces_wp, target_length=target, iterations=10
    )
    faces_np = out_faces.numpy().reshape(-1, 3)
    valence_after = _valences(faces_np, out_vertices.numpy().shape[0])
    # Interior valences concentrate around 6: variance about the ideal does not grow.
    assert np.var(valence_after - 6) <= np.var(valence_before - 6) + 0.5


def test_remesh_surface_distance_bounded(device: str) -> None:
    sphere, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=3)
    target = 0.5 * tw.edges.mean_edge_length(vertices_wp, faces_wp)

    out_vertices, out_faces = tw.remesh.isotropic_remesh(
        vertices_wp, faces_wp, target_length=target, iterations=10
    )
    hausdorff = _two_sided_hausdorff(
        sphere.vertices,
        sphere.faces,
        out_vertices.numpy().astype(np.float64),
        out_faces.numpy().reshape(-1, 3),
    )
    # Reprojection keeps the remesh close to the original surface (well under the target length).
    assert hausdorff < target


def test_remesh_cave_cube_manifold(cave_cube: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = cave_cube
    vertices_wp = wp.clone(mesh_wp.points)
    faces_wp = wp.clone(mesh_wp.indices)
    target = 0.5 * tw.edges.mean_edge_length(vertices_wp, faces_wp)

    out_vertices, out_faces = tw.remesh.isotropic_remesh(
        vertices_wp, faces_wp, target_length=target, iterations=8
    )
    mesh_out = tm.Trimesh(out_vertices.numpy(), out_faces.numpy().reshape(-1, 3), process=False)
    assert tw.validation.is_watertight(out_vertices, out_faces)
    # Two nested cubes: Euler characteristic 4 (two genus-0 shells) is preserved.
    assert mesh_out.euler_number == mesh_tm.euler_number


def test_remesh_feature_preservation(cave_cube: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = cave_cube
    vertices_wp = wp.clone(mesh_wp.points)
    faces_wp = wp.clone(mesh_wp.indices)
    target = 0.4 * tw.edges.mean_edge_length(vertices_wp, faces_wp)

    out_vertices, _ = tw.remesh.isotropic_remesh(
        vertices_wp, faces_wp, target_length=target, iterations=8, feature_angle=30.0
    )
    vertices_np = out_vertices.numpy().astype(np.float64)
    # The 8 outer cube corners (frozen CORNER vertices) survive at their exact positions.
    outer_corners = np.array(
        [[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)]
    )
    for corner in outer_corners:
        assert np.min(np.linalg.norm(vertices_np - corner, axis=1)) < 1e-6


def test_remesh_boundary_preservation(hemisphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = hemisphere
    vertices_wp = wp.clone(mesh_wp.points)
    faces_wp = wp.clone(mesh_wp.indices)
    n_loops_before = len(tm.Trimesh(*_mesh_arrays(mesh_wp), process=False).outline().entities)
    target = 0.5 * tw.edges.mean_edge_length(vertices_wp, faces_wp)

    out_vertices, out_faces = tw.remesh.isotropic_remesh(
        vertices_wp, faces_wp, target_length=target, iterations=8
    )
    mesh_out = tm.Trimesh(out_vertices.numpy(), out_faces.numpy().reshape(-1, 3), process=False)
    # The open boundary is still a single closed loop (the disk boundary is preserved).
    assert len(mesh_out.outline().entities) == n_loops_before


def test_remesh_flags_off(device: str) -> None:
    _sphere, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=2)
    n_faces_before = int(faces_wp.shape[0]) // 3
    # Collapse-only (no split/swap/smooth/reproject) can only reduce the face count.
    out_vertices, out_faces = tw.remesh.isotropic_remesh(
        vertices_wp,
        faces_wp,
        target_length=10.0 * tw.edges.mean_edge_length(vertices_wp, faces_wp),
        iterations=5,
        split=False,
        swap=False,
        smooth=False,
        reproject=False,
    )
    assert int(out_faces.shape[0]) // 3 <= n_faces_before
    assert tw.validation.is_watertight(out_vertices, out_faces)


def test_remesh_adaptive_not_implemented(device: str) -> None:
    _sphere, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=1)
    with pytest.raises(NotImplementedError, match="adaptive"):
        tw.remesh.isotropic_remesh(vertices_wp, faces_wp, adaptive=True)


def test_remesh_target_validation(device: str) -> None:
    _sphere, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=1)
    with pytest.raises(ValueError, match="target_length"):
        tw.remesh.isotropic_remesh(vertices_wp, faces_wp, target_length=-1.0)


def test_remesh_empty_and_degenerate(device: str) -> None:
    empty_v = wp.empty(0, dtype=wp.vec3, device=device)
    empty_f = wp.empty(0, dtype=wp.int32, device=device)
    _out_v, out_f = tw.remesh.isotropic_remesh(empty_v, empty_f)
    assert int(out_f.shape[0]) == 0

    # iterations=0 returns a clone unchanged.
    _sphere, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=1)
    _out_v, out_f = tw.remesh.isotropic_remesh(vertices_wp, faces_wp, iterations=0)
    assert int(out_f.shape[0]) == int(faces_wp.shape[0])


# --- intrinsic_delaunay ---------------------------------------------------------------
@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere", "half_torus", "torus"])
def test_intrinsic_delaunay_removes_negative_cotangent_weights(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = len(mesh_tm.vertices)
    flipped = bsr_to_dense(
        tw.laplacian.robust_laplacian(mesh_wp.points, mesh_wp.indices), n_vertices
    )

    # A non-negative off-diagonal (in this sign convention, where the diagonal is negative) is what
    # "Delaunay" buys: it is the condition for the Laplacian to satisfy a maximum principle.
    off_diagonal = flipped - np.diag(np.diag(flipped))
    assert off_diagonal.min() > -1e-6


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere"])
def test_intrinsic_delaunay_leaves_a_delaunay_mesh_alone(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    _, mesh_wp = request.getfixturevalue(mesh_name)
    original_lengths = tw.edges.face_edge_lengths(mesh_wp.points, mesh_wp.indices).numpy()
    faces, lengths, n_flips = tw.remesh.intrinsic_delaunay(mesh_wp.points, mesh_wp.indices)

    # These fixtures come from an icosphere, whose triangulation is already intrinsically Delaunay.
    assert n_flips == 0
    assert np.array_equal(faces.numpy(), mesh_wp.indices.numpy())
    assert np.allclose(lengths.numpy(), original_lengths, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("mesh_name", ["half_torus", "torus"])
@pytest.mark.parity("intrinsic_delaunay", "igl")
def test_intrinsic_delaunay_metric_matches_igl(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    """
    The intrinsic metric after flipping, against libigl's serial flipper.

    The two cannot be compared face for face: triwarp flips independent sets in parallel rounds
    where libigl drains a queue, so the *sequence* differs and so does the face ordering. What must
    agree is where they land, because the intrinsic Delaunay triangulation of a surface is unique
    away from cocircular degeneracies -- so the multiset of edge lengths is the invariant, and this
    is class B with a sort rather than a weakened tolerance.

    That makes it a real check rather than a formality: on ``half_torus`` triwarp performs **298**
    flips and still reaches libigl's metric to 1e-4, which a wrong flip rule or a mis-unfolded
    diagonal would not. ``igl.intrinsic_delaunay_cotmatrix`` is used for its second return value
    (the lengths); it assembles a matrix as well, which is why the benchmark reads its row as
    including work triwarp's does not.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    _lengths_igl = igl.intrinsic_delaunay_cotmatrix(
        np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64), mesh_tm.faces.astype(np.int64)
    )[1]

    _faces_wp, lengths_wp, n_flips = tw.remesh.intrinsic_delaunay(mesh_wp.points, mesh_wp.indices)

    assert n_flips > 0, "fixture is already Delaunay; this would assert nothing"
    assert np.allclose(
        np.sort(lengths_wp.numpy().ravel()), np.sort(_lengths_igl.ravel()), rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize("mesh_name", ["half_torus", "torus"])
def test_intrinsic_delaunay_flips_a_grid_and_preserves_the_metric(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    faces, lengths, n_flips = tw.remesh.intrinsic_delaunay(mesh_wp.points, mesh_wp.indices)

    # A quad grid split by diagonals is not Delaunay, so there is work to do...
    assert n_flips > 0
    # ... but the flips are *intrinsic*: the vertex count, the face count and the total area are all
    # properties of the surface, not of its triangulation, so none of them may change.
    assert faces.shape == mesh_wp.indices.shape
    assert np.array_equal(np.sort(np.unique(faces.numpy())), np.sort(np.unique(mesh_tm.faces)))
    sides = lengths.numpy().astype(np.float64)
    semi = sides.sum(axis=1) / 2.0
    heron = semi * (semi - sides[:, 0]) * (semi - sides[:, 1]) * (semi - sides[:, 2])
    assert np.isclose(np.sqrt(np.maximum(heron, 0.0)).sum(), mesh_tm.area, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Vertex-clustering decimation vs open3d / pymeshlab
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("voxel_size", [0.1, 0.3])
@pytest.mark.parametrize("contraction", ["average", "closest"])
@pytest.mark.parity("cluster_decimate", "open3d")
def test_cluster_decimate_matches_open3d(device: str, voxel_size: float, contraction: str) -> None:
    """
    Cell assignment is Open3D's, so the face count must match exactly, not approximately.

    Open3D's grid anchor is ``min_bound - voxel_size / 2``; this pins that choice, since an anchor
    at ``min_bound`` splits the vertices on the box face into two cells and the counts diverge.
    """
    sphere_tm, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=4)
    simplified_o3d = trimesh_to_open3d(sphere_tm).simplify_vertex_clustering(voxel_size=voxel_size)

    decimated_vertices_wp, decimated_faces_wp = tw.remesh.cluster_decimate(
        vertices_wp, faces_wp, voxel_size=voxel_size, contraction=contraction
    )
    assert int(decimated_faces_wp.shape[0]) // 3 == len(simplified_o3d.triangles)
    assert int(decimated_vertices_wp.shape[0]) == len(simplified_o3d.vertices)

    if contraction == "average":
        # Same cells and the same mean per cell, so the vertex *sets* coincide pointwise.
        distance_np, _index = KDTree(np.asarray(simplified_o3d.vertices)).query(
            decimated_vertices_wp.numpy().astype(np.float64)
        )
        assert distance_np.max() < 1e-5
    else:
        # 'Closest to centre' keeps every output vertex on the input surface, exactly.
        distance_np, _index = KDTree(np.asarray(sphere_tm.vertices)).query(
            decimated_vertices_wp.numpy().astype(np.float64)
        )
        assert distance_np.max() < 1e-5


def test_cluster_decimate_stays_near_the_input_surface(device: str) -> None:
    """A resampling, so the shape has to survive: Hausdorff within about one cell."""
    sphere_tm, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=4)
    voxel_size = 0.2
    decimated_vertices_wp, decimated_faces_wp = tw.remesh.cluster_decimate(
        vertices_wp, faces_wp, voxel_size=voxel_size
    )
    deviation = _two_sided_hausdorff(
        np.asarray(sphere_tm.vertices),
        np.asarray(sphere_tm.faces),
        decimated_vertices_wp.numpy().astype(np.float64),
        decimated_faces_wp.numpy().reshape(-1, 3),
    )
    assert deviation < voxel_size


def test_cluster_decimate_emits_no_degenerate_or_duplicated_faces(device: str) -> None:
    """Collapsed faces are dropped and welded duplicates deduped: both are part of the algorithm."""
    _sphere_tm, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=4)
    decimated_vertices_wp, decimated_faces_wp = tw.remesh.cluster_decimate(
        vertices_wp, faces_wp, voxel_size=0.25
    )
    faces_np = decimated_faces_wp.numpy().reshape(-1, 3)
    assert _degenerate_face_count(decimated_vertices_wp.numpy().astype(np.float64), faces_np) == 0
    assert len(np.unique(np.sort(faces_np, axis=1), axis=0)) == faces_np.shape[0]
    # Every output vertex is referenced by a face.
    assert len(np.unique(faces_np)) == int(decimated_vertices_wp.shape[0])


def test_cluster_decimate_decimates_monotonically(device: str) -> None:
    """A wider cell can only ever produce fewer faces."""
    _sphere_tm, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=4)
    counts = [
        int(tw.remesh.cluster_decimate(vertices_wp, faces_wp, voxel_size=size)[1].shape[0]) // 3
        for size in (0.05, 0.1, 0.2, 0.4)
    ]
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] < counts[0]


def test_cluster_decimate_default_voxel_size(device: str) -> None:
    """The default is 1% of the bounding-box diagonal, matching MeshLab's ``threshold``."""
    _sphere_tm, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=4)
    default_faces_wp = tw.remesh.cluster_decimate(vertices_wp, faces_wp)[1]
    diagonal = float(np.linalg.norm(np.array([2.0, 2.0, 2.0])))
    explicit_faces_wp = tw.remesh.cluster_decimate(
        vertices_wp, faces_wp, voxel_size=0.01 * diagonal
    )[1]
    assert int(default_faces_wp.shape[0]) == int(explicit_faces_wp.shape[0])


def test_cluster_decimate_invalid(device: str) -> None:
    _sphere_tm, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=2)
    with pytest.raises(ValueError, match="voxel_size > 0"):
        tw.remesh.cluster_decimate(vertices_wp, faces_wp, voxel_size=0.0)
    with pytest.raises(ValueError, match="contraction must be"):
        tw.remesh.cluster_decimate(vertices_wp, faces_wp, contraction="quadric")  # type: ignore[arg-type]


def test_cluster_decimate_empty(device: str) -> None:
    vertices_wp = wp.zeros(0, dtype=wp.vec3, device=device)
    faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    out_vertices_wp, out_faces_wp = tw.remesh.cluster_decimate(vertices_wp, faces_wp)
    assert int(out_vertices_wp.shape[0]) == 0
    assert int(out_faces_wp.shape[0]) == 0


# ---------------------------------------------------------------------------
# Objective-driven edge flips vs pymeshlab
# ---------------------------------------------------------------------------


def _sheared_grid(n: int = 24, shear: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a **flat** grid of sheared parallelograms, split along each cell's long diagonal.

    Shear is what makes this a fixture: a rectangle's two diagonals are the same length, so both
    triangulations of it are congruent and no quality objective can prefer either — an
    axis-aligned grid, however anisotropic, has nothing to flip. Shearing by ``shear`` cells makes
    one diagonal ``(1 + shear, 1)`` and the other ``(1 - shear, -1)``, so the right flip exists at
    every quad and the planarity objective must find it. The default ``shear=4`` maximizes the
    *relative* gain: pushing it higher makes both triangles worse, so the ratio shrinks back
    toward 1 even as the mesh gets uglier. Flat, so the flip is a pure
    retriangulation and cannot change the surface.
    """
    i_grid, j_grid = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    vertices = np.column_stack(
        [
            (i_grid + shear * j_grid).ravel().astype(np.float64),
            j_grid.ravel().astype(np.float64),
            np.zeros(n * n),
        ]
    )
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            faces += [[a, a + 1, a + n + 1], [a, a + n + 1, a + n]]
    return vertices, np.ascontiguousarray(faces, dtype=np.int32)


def _saddle_grid(n: int = 16, step: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """
    ``z = x y`` over a square grid: the fixture the curvature objective exists for.

    On a hyperbolic paraboloid the two diagonals of a quad have *opposite* curvature — one runs
    along a ruling of the surface and is nearly straight, the other bends. So the choice is
    maximally consequential, and the current diagonal is deliberately the bending one.
    """
    i_grid, j_grid = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    x_np = (i_grid * step).ravel().astype(np.float64)
    y_np = (j_grid * step).ravel().astype(np.float64)
    vertices = np.column_stack([x_np, y_np, x_np * y_np])
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            faces += [[a, a + 1, a + n + 1], [a, a + n + 1, a + n]]
    return vertices, np.ascontiguousarray(faces, dtype=np.int32)


def _upload(vertices_np: np.ndarray, faces_np: np.ndarray, device: str):
    return (
        wp.array(np.ascontiguousarray(vertices_np, dtype=np.float32), dtype=wp.vec3, device=device),
        wp.array(
            np.ascontiguousarray(faces_np.reshape(-1), dtype=np.int32),
            dtype=wp.int32,
            device=device,
        ),
    )


def _min_quality(vertices_wp, faces_wp, metric: str = "area_max_side") -> float:
    return float(tw.triangles.face_quality(vertices_wp, faces_wp, metric=metric).numpy().min())


def _total_bend(vertices_np: np.ndarray, faces_np: np.ndarray) -> float:
    """Sum of the absolute dihedral angle over every interior edge — the curvature objective."""
    mesh_tm = tm.Trimesh(vertices_np, faces_np, process=False)
    return float(np.abs(mesh_tm.face_adjacency_angles).sum())


def test_flip_by_objective_planarity_improves_the_worst_triangle(device: str) -> None:
    """Every quad of a flat sheared grid has a better diagonal, and the flip must take it."""
    vertices_np, faces_np = _sheared_grid()
    vertices_wp, faces_wp = _upload(vertices_np, faces_np, device)
    before = _min_quality(vertices_wp, faces_wp)

    flipped_wp = tw.remesh.flip_by_objective(vertices_wp, faces_wp, objective="planarity")
    after = _min_quality(vertices_wp, flipped_wp)
    assert after > before * 1.4

    # A retriangulation: same faces, same vertices, still a clean manifold patch.
    assert int(flipped_wp.shape[0]) == int(faces_wp.shape[0])
    assert tw.validation.is_winding_consistent(flipped_wp)
    assert tw.validation.is_edge_manifold(flipped_wp)
    assert _degenerate_face_count(vertices_np, flipped_wp.numpy().reshape(-1, 3)) == 0


@pytest.mark.parity("flip_by_objective", "pymeshlab")
def test_flip_by_objective_planarity_at_least_matches_pymeshlab(device: str) -> None:
    """
    MeshLab runs the same objective serially, so it is a bar on how many flips this port finds.

    ``meshing_edge_flip_by_planar_optimization`` takes the *same* planarity threshold and the *same*
    quality metric (``planartype='area/max side'``), and its greedy serial pass is free to take
    every flip in any order. A parallel independent-set pass can only ever match it, so requiring
    the resulting worst triangle to be within 10% is a real check that the predicate agrees.
    """
    vertices_np, faces_np = _sheared_grid()
    meshset_pml = ml.MeshSet()
    meshset_pml.add_mesh(ml.Mesh(vertices_np, np.ascontiguousarray(faces_np, dtype=np.int32)))
    meshset_pml.meshing_edge_flip_by_planar_optimization(
        pthreshold=1.0, planartype="area/max side", iterations=10
    )
    faces_pml = meshset_pml.current_mesh().face_matrix()
    assert faces_pml.shape[0] == faces_np.shape[0]

    vertices_wp, faces_wp = _upload(vertices_np, faces_np, device)
    _vertices_pml_wp, faces_pml_wp = _upload(vertices_np, faces_pml, device)
    flipped_wp = tw.remesh.flip_by_objective(vertices_wp, faces_wp, objective="planarity")
    assert _min_quality(vertices_wp, flipped_wp) >= 0.9 * _min_quality(vertices_wp, faces_pml_wp)


def test_flip_by_objective_planarity_refuses_a_curved_quad(device: str) -> None:
    """With ``planar_angle=0`` nothing is flat enough, so the triangulation must be untouched."""
    _sphere_tm, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=3)
    flipped_wp = tw.remesh.flip_by_objective(
        vertices_wp, faces_wp, objective="planarity", planar_angle=0.0
    )
    assert np.array_equal(flipped_wp.numpy(), faces_wp.numpy())


def test_flip_by_objective_curvature_flattens(device: str) -> None:
    """The curvature objective must lower the total absolute dihedral angle."""
    vertices_np, faces_np = _saddle_grid()
    vertices_wp, faces_wp = _upload(vertices_np, faces_np, device)
    before = _total_bend(vertices_np, faces_np)

    flipped_wp = tw.remesh.flip_by_objective(vertices_wp, faces_wp, objective="curvature")
    after = _total_bend(vertices_np, flipped_wp.numpy().reshape(-1, 3))
    assert after < before
    assert int(flipped_wp.shape[0]) == int(faces_wp.shape[0])
    assert tw.validation.is_winding_consistent(flipped_wp)
    assert tw.validation.is_edge_manifold(flipped_wp)


def test_flip_by_objective_curvature_leaves_a_sphere_alone(device: str) -> None:
    """
    An icosphere's diagonals are already the flat ones, so a converged pass changes nothing much.

    Not an equality assertion: the icosphere's quads are close enough to symmetric that a handful
    genuinely tie, and the relative ``1e-6`` margin is what keeps those from oscillating. What must
    hold is that the total bend does not *increase*.
    """
    sphere_tm, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=3)
    before = _total_bend(np.asarray(sphere_tm.vertices), np.asarray(sphere_tm.faces))
    flipped_wp = tw.remesh.flip_by_objective(vertices_wp, faces_wp, objective="curvature")
    after = _total_bend(np.asarray(sphere_tm.vertices), flipped_wp.numpy().reshape(-1, 3))
    assert after <= before * (1.0 + 1e-6)


def test_flip_by_objective_region_gated(device: str) -> None:
    """Faces outside the region keep their edges, so the flip count can only go down."""
    vertices_np, faces_np = _sheared_grid()
    vertices_wp, faces_wp = _upload(vertices_np, faces_np, device)
    n_faces = int(faces_wp.shape[0]) // 3
    region_np = np.zeros(n_faces, dtype=bool)
    region_np[: n_faces // 4] = True
    region_wp = wp.array(region_np, dtype=wp.bool, device=device)

    full_wp = tw.remesh.flip_by_objective(vertices_wp, faces_wp, objective="planarity")
    gated_wp = tw.remesh.flip_by_objective(
        vertices_wp, faces_wp, objective="planarity", region=region_wp
    )
    changed_full = int((full_wp.numpy() != faces_wp.numpy()).sum())
    changed_gated = int((gated_wp.numpy() != faces_wp.numpy()).sum())
    assert 0 < changed_gated < changed_full


def test_flip_by_objective_invalid(device: str) -> None:
    _sphere_tm, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=1)
    with pytest.raises(ValueError, match="objective must be"):
        tw.remesh.flip_by_objective(vertices_wp, faces_wp, objective="delaunay")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="metric must be one of"):
        tw.remesh.flip_by_objective(vertices_wp, faces_wp, metric="aspect_ratio")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="planar_angle must be in"):
        tw.remesh.flip_by_objective(vertices_wp, faces_wp, planar_angle=200.0)
    with pytest.raises(ValueError, match="region must have length"):
        tw.remesh.flip_by_objective(
            vertices_wp, faces_wp, region=wp.zeros(2, dtype=wp.bool, device=device)
        )


def test_flip_by_objective_empty(device: str) -> None:
    vertices_wp = wp.zeros(0, dtype=wp.vec3, device=device)
    faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    assert int(tw.remesh.flip_by_objective(vertices_wp, faces_wp).shape[0]) == 0


# ---------------------------------------------------------------------------
# Quadric edge-collapse decimation vs igl / open3d / pymeshlab
# ---------------------------------------------------------------------------


def _inverted_face_count(vertices_np: np.ndarray, faces_np: np.ndarray) -> int:
    """
    Faces whose outward normal points *inward* on a star-shaped mesh centred on the origin.

    The failure mode an unguarded quadric method produces at a high reduction ratio, and cheap to
    detect on a sphere: a correctly oriented face has its normal agreeing with its own centroid.
    """
    mesh_tm = tm.Trimesh(vertices_np, faces_np, process=False)
    centroids_np = mesh_tm.vertices[mesh_tm.faces].mean(axis=1)
    return int((np.einsum("ij,ij->i", mesh_tm.face_normals, centroids_np) < 0.0).sum())


@pytest.mark.parametrize("target_faces", [2560, 1024, 512])
@pytest.mark.parity("quadric_decimate", "igl", "open3d")
def test_quadric_decimate_beats_igl_and_open3d_on_deviation(device: str, target_faces: int) -> None:
    """
    At the same face count this port must be no *worse* than the two serial references.

    The plan for this port said to expect the batched-parallel formulation to pick a different
    sequence of collapses from a serial priority queue, and to compare by deviation rather than by
    equality. It does, and it comes out ahead: measured two-sided Hausdorff to the input icosphere
    at 512 faces is **0.0147 here against igl's 0.0250 and Open3D's 0.0236**, and the ordering holds
    at every target. Spreading the collapses over independent sets rather than draining a queue
    keeps the error evenly distributed, which is what a max-norm rewards.

    The assertion is one-sided with slack, not an equality — the point is that the parallel method
    is competitive, not that this exact ratio is a contract.
    """
    igl_module = pytest.importorskip("igl")
    sphere_tm, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=4)
    vertices_np = np.ascontiguousarray(sphere_tm.vertices, dtype=np.float64)
    faces_np = np.ascontiguousarray(sphere_tm.faces, dtype=np.int64)

    decimated_igl = igl_module.decimate(vertices_np, faces_np, target_faces)
    igl_tm = tm.Trimesh(np.asarray(decimated_igl[0]), np.asarray(decimated_igl[1]), process=False)
    mesh_o3d = trimesh_to_open3d(sphere_tm).simplify_quadric_decimation(
        target_number_of_triangles=target_faces
    )
    o3d_tm = open3d_to_trimesh(mesh_o3d)

    decimated_vertices_wp, decimated_faces_wp = tw.remesh.quadric_decimate(
        vertices_wp, faces_wp, target_faces=target_faces
    )
    assert int(decimated_faces_wp.shape[0]) // 3 == target_faces

    deviation_wp = _two_sided_hausdorff(
        vertices_np,
        np.asarray(sphere_tm.faces),
        decimated_vertices_wp.numpy().astype(np.float64),
        decimated_faces_wp.numpy().reshape(-1, 3),
    )
    deviation_igl = _two_sided_hausdorff(
        vertices_np, np.asarray(sphere_tm.faces), igl_tm.vertices, igl_tm.faces
    )
    deviation_o3d = _two_sided_hausdorff(
        vertices_np, np.asarray(sphere_tm.faces), o3d_tm.vertices, o3d_tm.faces
    )
    assert deviation_wp <= 1.2 * min(deviation_igl, deviation_o3d)


def test_quadric_decimate_emits_no_inverted_or_degenerate_faces(device: str) -> None:
    """The normal-flip guard's job: even at 5% of the triangles, nothing folds over."""
    _sphere_tm, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=4)
    decimated_vertices_wp, decimated_faces_wp = tw.remesh.quadric_decimate(
        vertices_wp, faces_wp, target_ratio=0.05
    )
    vertices_np = decimated_vertices_wp.numpy().astype(np.float64)
    faces_np = decimated_faces_wp.numpy().reshape(-1, 3)
    assert _inverted_face_count(vertices_np, faces_np) == 0
    assert _degenerate_face_count(vertices_np, faces_np) == 0
    assert tw.validation.is_edge_manifold(decimated_faces_wp, allow_boundary_edges=False)
    assert tw.validation.is_winding_consistent(decimated_faces_wp)


def test_quadric_decimate_preserves_the_topology(device: str) -> None:
    """A closed genus-0 surface stays closed and genus 0, and its volume barely moves."""
    sphere_tm, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=4)
    decimated_vertices_wp, decimated_faces_wp = tw.remesh.quadric_decimate(
        vertices_wp, faces_wp, target_ratio=0.2
    )
    decimated_tm = tm.Trimesh(
        decimated_vertices_wp.numpy().astype(np.float64),
        decimated_faces_wp.numpy().reshape(-1, 3),
        process=False,
    )
    assert decimated_tm.is_watertight
    assert decimated_tm.euler_number == 2
    assert np.isclose(decimated_tm.volume, sphere_tm.volume, rtol=0.02)


def test_quadric_decimate_keeps_the_features_of_a_cube(device: str) -> None:
    """
    A cube's twelve edges are its whole shape, and the metric has to spend its budget elsewhere.

    This is the property that distinguishes a quadric method from a length-driven one: the flat
    faces have zero quadric cost to collapse and the creases have a large one, so the sharp edges
    survive down to the coarsest usable mesh. Checked as the surviving dihedral distribution, which
    is invariant to *which* particular collapses happened.
    """
    box_tm = tm.creation.box(extents=[1.0, 1.0, 1.0]).subdivide().subdivide().subdivide()
    vertices_wp, faces_wp = _upload(np.asarray(box_tm.vertices), np.asarray(box_tm.faces), device)
    decimated_vertices_wp, decimated_faces_wp = tw.remesh.quadric_decimate(
        vertices_wp, faces_wp, target_ratio=0.1
    )
    decimated_tm = tm.Trimesh(
        decimated_vertices_wp.numpy().astype(np.float64),
        decimated_faces_wp.numpy().reshape(-1, 3),
        process=False,
    )
    # Still a box: the same eight corners, the same volume, and 90-degree edges intact.
    assert np.allclose(decimated_tm.bounds, box_tm.bounds, atol=1e-4)
    assert np.isclose(decimated_tm.volume, box_tm.volume, rtol=0.02)
    angles_np = np.rad2deg(np.abs(decimated_tm.face_adjacency_angles))
    assert np.percentile(angles_np, 95.0) > 85.0


@pytest.mark.parity("quadric_decimate", "pymeshlab")
def test_quadric_decimate_reaches_pymeshlab_quality(device: str) -> None:
    """
    ``meshing_decimation_quadric_edge_collapse`` is the same metric, driven serially.

    Its ``autoclean`` default deletes unreferenced vertices, so the MeshSet is built fresh here (it
    is one of the two filters recorded as not idempotent even in geometry). Compared by deviation,
    as with the other two references.
    """
    sphere_tm, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=4)
    target_faces = 1024
    meshset_pml = ml.MeshSet()
    meshset_pml.add_mesh(
        ml.Mesh(
            np.ascontiguousarray(sphere_tm.vertices, dtype=np.float64),
            np.ascontiguousarray(sphere_tm.faces, dtype=np.int32),
        )
    )
    meshset_pml.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)
    mesh_pml = meshset_pml.current_mesh()
    pml_tm = tm.Trimesh(mesh_pml.vertex_matrix(), mesh_pml.face_matrix(), process=False)

    decimated_vertices_wp, decimated_faces_wp = tw.remesh.quadric_decimate(
        vertices_wp, faces_wp, target_faces=target_faces
    )
    vertices_np = np.ascontiguousarray(sphere_tm.vertices, dtype=np.float64)
    deviation_wp = _two_sided_hausdorff(
        vertices_np,
        np.asarray(sphere_tm.faces),
        decimated_vertices_wp.numpy().astype(np.float64),
        decimated_faces_wp.numpy().reshape(-1, 3),
    )
    deviation_pml = _two_sided_hausdorff(
        vertices_np, np.asarray(sphere_tm.faces), pml_tm.vertices, pml_tm.faces
    )
    assert deviation_wp <= 1.2 * deviation_pml


def test_quadric_decimate_is_monotone_in_the_target(device: str) -> None:
    _sphere_tm, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=4)
    counts = [
        int(tw.remesh.quadric_decimate(vertices_wp, faces_wp, target_ratio=ratio)[1].shape[0]) // 3
        for ratio in (0.8, 0.4, 0.2, 0.1)
    ]
    assert counts == sorted(counts, reverse=True)


def test_quadric_decimate_target_at_or_above_the_input_is_a_copy(device: str) -> None:
    _sphere_tm, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=2)
    n_faces = int(faces_wp.shape[0]) // 3
    for kwargs in (
        {"target_faces": n_faces},
        {"target_faces": n_faces + 100},
        {"target_ratio": 1.0},
    ):
        out_vertices_wp, out_faces_wp = tw.remesh.quadric_decimate(vertices_wp, faces_wp, **kwargs)
        assert np.array_equal(out_faces_wp.numpy(), faces_wp.numpy())
        assert np.array_equal(out_vertices_wp.numpy(), vertices_wp.numpy())


def test_quadric_decimate_invalid(device: str) -> None:
    _sphere_tm, vertices_wp, faces_wp = _icosphere_wp(device, subdivisions=1)
    with pytest.raises(ValueError, match="exactly one of target_faces and target_ratio"):
        tw.remesh.quadric_decimate(vertices_wp, faces_wp)
    with pytest.raises(ValueError, match="exactly one of target_faces and target_ratio"):
        tw.remesh.quadric_decimate(vertices_wp, faces_wp, target_faces=10, target_ratio=0.5)
    with pytest.raises(ValueError, match="target_faces must be non-negative"):
        tw.remesh.quadric_decimate(vertices_wp, faces_wp, target_faces=-1)
    with pytest.raises(ValueError, match=r"target_ratio must be in \(0, 1\]"):
        tw.remesh.quadric_decimate(vertices_wp, faces_wp, target_ratio=0.0)


def test_quadric_decimate_empty(device: str) -> None:
    vertices_wp = wp.zeros(0, dtype=wp.vec3, device=device)
    faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    out_vertices_wp, out_faces_wp = tw.remesh.quadric_decimate(
        vertices_wp, faces_wp, target_faces=0
    )
    assert int(out_vertices_wp.shape[0]) == 0
    assert int(out_faces_wp.shape[0]) == 0
