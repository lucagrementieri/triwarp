"""
Tests for point-cloud surface reconstruction.

No trimesh/libigl equivalent exists for this algorithm, so the reference is MeshLib's own
``triangulatePointCloud`` (via the ``meshlib`` Python bindings, ``_mm`` suffix) plus geometric
property checks (watertightness, edge-manifoldness, Euler characteristic, surface closeness).
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh as tm
import warp as wp

import triwarp as tw

_meshlib = pytest.importorskip("meshlib")
from meshlib import mrmeshnumpy as mn  # noqa: E402
from meshlib import mrmeshpy as mm  # noqa: E402


def _meshlib_triangulate(
    points_np: np.ndarray, normals_np: np.ndarray, num_neighbours: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct a reference mesh via MeshLib, returning ``(vertices, faces)`` numpy arrays."""
    cloud_mm = mn.pointCloudFromPoints(
        np.ascontiguousarray(points_np), np.ascontiguousarray(normals_np)
    )
    params_mm = mm.TriangulationParameters()
    params_mm.numNeighbours = num_neighbours
    mesh_mm = mm.triangulatePointCloud(cloud_mm, params_mm)
    return mn.getNumpyVerts(mesh_mm), mn.getNumpyFaces(mesh_mm.topology)


def _sphere_cloud(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    sphere_tm = tm.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    return (sphere_tm.vertices.astype(np.float64), sphere_tm.vertex_normals.astype(np.float64))


def _to_warp(points_np: np.ndarray, normals_np: np.ndarray, device: str):
    points_wp = wp.array(np.ascontiguousarray(points_np), dtype=wp.vec3, device=device)
    normals_wp = wp.array(np.ascontiguousarray(normals_np), dtype=wp.vec3, device=device)
    return points_wp, normals_wp


@pytest.mark.parametrize("subdivisions", [3])
def test_sphere_is_closed_manifold(device: str, subdivisions: int):
    points_np, normals_np = _sphere_cloud(subdivisions)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    vertices_wp, faces_wp = tw.reconstruction.triangulate_point_cloud(
        points_wp, normals_wp, num_neighbours=18
    )
    faces_np = faces_wp.numpy()
    n_points = points_np.shape[0]

    # A closed genus-0 triangulation of n points has exactly 2n - 4 faces (Euler).
    assert faces_np.shape[0] // 3 == 2 * n_points - 4
    assert tw.validation.is_watertight(vertices_wp, faces_wp)
    assert tw.validation.is_edge_manifold(faces_wp)
    assert tw.validation.euler_characteristic(faces_wp) == 2
    # every input point is referenced
    assert np.unique(faces_np).size == n_points

    # reconstructed vertices lie on the unit sphere
    radii = np.linalg.norm(vertices_wp.numpy(), axis=1)
    assert np.allclose(radii, 1.0, rtol=1e-5, atol=1e-5)


def test_torus_is_genus_one(device: str):
    torus_tm = tm.creation.torus(
        major_radius=1.0, minor_radius=0.35, major_sections=48, minor_sections=24
    )
    points_np = torus_tm.vertices.astype(np.float64)
    normals_np = torus_tm.vertex_normals.astype(np.float64)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    vertices_wp, faces_wp = tw.reconstruction.triangulate_point_cloud(
        points_wp, normals_wp, num_neighbours=16
    )
    assert tw.validation.is_watertight(vertices_wp, faces_wp)
    assert tw.validation.is_edge_manifold(faces_wp)
    # genus-1 closed surface: V - E + F = 0
    assert tw.validation.euler_characteristic(faces_wp) == 0


@pytest.mark.parametrize("subdivisions", [3])
def test_matches_meshlib_reference(device: str, subdivisions: int):
    points_np, normals_np = _sphere_cloud(subdivisions)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    _, faces_mm = _meshlib_triangulate(points_np, normals_np, 18)
    _, faces_wp = tw.reconstruction.triangulate_point_cloud(
        points_wp, normals_wp, num_neighbours=18
    )
    n_faces_mm = faces_mm.shape[0]
    n_faces_wp = faces_wp.numpy().shape[0] // 3

    # Same face count as MeshLib on a clean uniform cloud (the greedy fan optimisation reproduces
    # the reference triangulation up to tie-breaking).
    assert abs(n_faces_wp - n_faces_mm) <= max(2, n_faces_mm // 100)


def test_torus_matches_meshlib_reference(device: str):
    torus_tm = tm.creation.torus(
        major_radius=1.0, minor_radius=0.35, major_sections=48, minor_sections=24
    )
    points_np = torus_tm.vertices.astype(np.float64)
    normals_np = torus_tm.vertex_normals.astype(np.float64)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    _, faces_mm = _meshlib_triangulate(points_np, normals_np, 16)
    _, faces_wp = tw.reconstruction.triangulate_point_cloud(
        points_wp, normals_wp, num_neighbours=16
    )
    n_faces_mm = faces_mm.shape[0]
    n_faces_wp = faces_wp.numpy().shape[0] // 3
    assert abs(n_faces_wp - n_faces_mm) <= max(2, n_faces_mm // 100)


def test_open_hemisphere_keeps_single_boundary(device: str):
    sphere_tm = tm.creation.icosphere(subdivisions=4, radius=1.0)
    upper = sphere_tm.vertices[:, 2] >= -1e-9
    points_np = sphere_tm.vertices[upper].astype(np.float64)
    normals_np = sphere_tm.vertex_normals[upper].astype(np.float64)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    vertices_wp, faces_wp = tw.reconstruction.triangulate_point_cloud(
        points_wp, normals_wp, num_neighbours=18
    )
    assert tw.validation.is_edge_manifold(faces_wp)
    # The intended equator rim is a single large boundary loop, not filled and not fragmented.
    loops = tw.boundary.boundary_loops(vertices_wp, faces_wp)
    assert len(loops) == 1


def test_estimated_normals_path_runs(device: str):
    # Exercise the normal-estimation code path (normals=None). Global orientation of PCA normals is
    # a documented best-effort step, so we check the pipeline runs and yields an edge-manifold mesh
    # whose vertices still lie on the sphere -- not full watertightness.
    points_np, _ = _sphere_cloud(3)
    points_wp = wp.array(np.ascontiguousarray(points_np), dtype=wp.vec3, device=device)

    vertices_wp, faces_wp = tw.reconstruction.triangulate_point_cloud(points_wp, num_neighbours=18)
    assert faces_wp.numpy().shape[0] > 0
    assert tw.validation.is_edge_manifold(faces_wp)
    radii = np.linalg.norm(vertices_wp.numpy(), axis=1)
    assert np.allclose(radii, 1.0, rtol=1e-5, atol=1e-5)


def test_hole_filling_seals_small_hole(device: str):
    rng = np.random.default_rng(0)
    points_np, normals_np = _sphere_cloud(4)
    # Remove a small cluster of points to open a genuine boundary hole.
    seed = points_np[rng.integers(points_np.shape[0])]
    keep = np.linalg.norm(points_np - seed, axis=1) > 0.2
    points_np, normals_np = points_np[keep], normals_np[keep]
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    # No filling: the punctured region stays an open boundary.
    vertices_open, faces_open = tw.reconstruction.triangulate_point_cloud(
        points_wp, normals_wp, num_neighbours=18, crit_hole_length=0.0
    )
    assert not tw.validation.is_watertight(vertices_open, faces_open)

    # Large threshold: the hole is sealed into a watertight, manifold mesh.
    vertices_filled, faces_filled = tw.reconstruction.triangulate_point_cloud(
        points_wp, normals_wp, num_neighbours=18, crit_hole_length=10.0
    )
    assert tw.validation.is_edge_manifold(faces_filled)
    assert tw.validation.is_watertight(vertices_filled, faces_filled)


def test_empty_cloud(device: str):
    points_wp = wp.zeros(0, dtype=wp.vec3, device=device)
    vertices_wp, faces_wp = tw.reconstruction.triangulate_point_cloud(points_wp)
    assert int(vertices_wp.shape[0]) == 0
    assert int(faces_wp.shape[0]) == 0


def test_too_few_points(device: str):
    points_wp = wp.array(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64), dtype=wp.vec3, device=device
    )
    _, faces_wp = tw.reconstruction.triangulate_point_cloud(points_wp, num_neighbours=4)
    assert int(faces_wp.shape[0]) == 0


def test_invalid_parameters(device: str):
    points_wp = wp.zeros(4, dtype=wp.vec3, device=device)
    with pytest.raises(ValueError, match="pass at most one of num_neighbours and radius"):
        tw.reconstruction.triangulate_point_cloud(points_wp, num_neighbours=8, radius=1.0)
    with pytest.raises(ValueError, match="max_neighbours must be <="):
        tw.reconstruction.triangulate_point_cloud(
            points_wp, max_neighbours=tw.kernels.reconstruction.MAX_NEIGHBOURS + 1
        )


# ---------------------------------------------------------------------------
# 2D Delaunay triangulation (reference: scipy.spatial.Delaunay)
# ---------------------------------------------------------------------------


def _cross2(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def _edge_set(faces_flat: np.ndarray) -> set[tuple[int, int]]:
    faces_flat = faces_flat.reshape(-1)
    edges: set[tuple[int, int]] = set()
    for i in range(0, len(faces_flat), 3):
        t = faces_flat[i : i + 3]
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edges.add((int(min(a, b)), int(max(a, b))))
    return edges


def _incircle_violations(points_np: np.ndarray, faces_flat: np.ndarray) -> int:
    """Count interior edges whose opposite apex lies inside the adjacent triangle circumcircle."""
    from scipy.spatial import Delaunay  # noqa: F401 — parity handled by caller

    faces = faces_flat.reshape(-1, 3)
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for fi, t in enumerate(faces):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edge_faces.setdefault((int(min(a, b)), int(max(a, b))), []).append(fi)

    def in_circle(a, b, c, d):
        m = np.array(
            [
                [a[0] - d[0], a[1] - d[1], (a[0] - d[0]) ** 2 + (a[1] - d[1]) ** 2],
                [b[0] - d[0], b[1] - d[1], (b[0] - d[0]) ** 2 + (b[1] - d[1]) ** 2],
                [c[0] - d[0], c[1] - d[1], (c[0] - d[0]) ** 2 + (c[1] - d[1]) ** 2],
            ]
        )
        return np.linalg.det(m)

    violations = 0
    for (u, v), fs in edge_faces.items():
        if len(fs) != 2:
            continue
        apex = []
        for fi in fs:
            apex.extend([int(x) for x in faces[fi] if int(x) not in (u, v)])
        if len(apex) != 2:
            continue
        d0, d1 = apex
        a, b, c = points_np[u], points_np[v], points_np[d0]
        if _cross2(b - a, c - a) < 0:
            a, b = b, a
        if in_circle(a, b, c, points_np[d1]) > 1e-9:
            violations += 1
    return violations


def test_delaunay_matches_scipy_random(device: str):
    from scipy.spatial import Delaunay

    rng = np.random.default_rng(42)
    points_np = rng.random((200, 2)).astype(np.float32)
    points_wp = wp.array(points_np, dtype=wp.vec2, device=device)

    faces_wp = tw.reconstruction.delaunay_triangulation(points_wp).numpy()
    faces_sp = Delaunay(points_np.astype(np.float64)).simplices

    assert _edge_set(faces_wp) == _edge_set(faces_sp.reshape(-1))


def test_delaunay_no_violations(device: str):
    rng = np.random.default_rng(7)
    points_np = rng.random((150, 2)).astype(np.float32)
    points_wp = wp.array(points_np, dtype=wp.vec2, device=device)

    faces_wp = tw.reconstruction.delaunay_triangulation(points_wp).numpy().reshape(-1, 3)

    # All triangles counter-clockwise.
    tris = points_np.astype(np.float64)[faces_wp]
    cross = _cross2(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    assert np.all(cross > 0.0)
    assert _incircle_violations(points_np.astype(np.float64), faces_wp) == 0


def test_delaunay_covers_hull(device: str):
    from scipy.spatial import ConvexHull

    rng = np.random.default_rng(3)
    points_np = rng.random((120, 2)).astype(np.float32)
    points_wp = wp.array(points_np, dtype=wp.vec2, device=device)

    faces_wp = tw.reconstruction.delaunay_triangulation(points_wp).numpy().reshape(-1, 3)
    tris = points_np.astype(np.float64)[faces_wp]
    area = float(np.abs(_cross2(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])).sum() * 0.5)
    hull_area = float(ConvexHull(points_np.astype(np.float64)).volume)
    assert np.isclose(area, hull_area, rtol=1e-5, atol=1e-6)


def test_delaunay_cocircular(device: str):
    # Regular 12-gon plus centre: many cocircular quadruples; compare on invariants, not triangles.
    angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    ring = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    points_np = np.vstack([ring, [[0.0, 0.0]]]).astype(np.float32)
    points_wp = wp.array(points_np, dtype=wp.vec2, device=device)

    faces_wp = tw.reconstruction.delaunay_triangulation(points_wp).numpy().reshape(-1, 3)
    assert _incircle_violations(points_np.astype(np.float64), faces_wp) == 0

    from scipy.spatial import ConvexHull

    tris = points_np.astype(np.float64)[faces_wp]
    area = float(np.abs(_cross2(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])).sum() * 0.5)
    assert np.isclose(area, float(ConvexHull(points_np.astype(np.float64)).volume), rtol=1e-5)


def test_delaunay_grid_perturbed(device: str):
    from scipy.spatial import Delaunay

    rng = np.random.default_rng(11)
    grid = np.stack(np.meshgrid(np.arange(8.0), np.arange(8.0)), axis=-1).reshape(-1, 2)
    points_np = (grid + rng.normal(0.0, 0.05, grid.shape)).astype(np.float32)
    points_wp = wp.array(points_np, dtype=wp.vec2, device=device)

    faces_wp = tw.reconstruction.delaunay_triangulation(points_wp).numpy()
    faces_sp = Delaunay(points_np.astype(np.float64)).simplices
    assert _edge_set(faces_wp) == _edge_set(faces_sp.reshape(-1))


def test_delaunay_collinear(device: str):
    points_np = np.array([[float(i), 0.0] for i in range(5)], dtype=np.float32)
    points_wp = wp.array(points_np, dtype=wp.vec2, device=device)
    faces_wp = tw.reconstruction.delaunay_triangulation(points_wp)
    assert int(faces_wp.shape[0]) == 0


def test_delaunay_too_few(device: str):
    points_wp = wp.array(np.zeros((2, 2), dtype=np.float32), dtype=wp.vec2, device=device)
    with pytest.raises(ValueError, match="at least 3 points"):
        tw.reconstruction.delaunay_triangulation(points_wp)
