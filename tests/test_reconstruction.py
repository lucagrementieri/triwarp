"""
Tests for point-cloud surface reconstruction.

No trimesh/libigl equivalent exists for this algorithm, so the reference is MeshLib's own
``triangulatePointCloud`` (via the ``meshlib`` Python bindings, ``_mm`` suffix) plus geometric
property checks (watertightness, edge-manifoldness, Euler characteristic, surface closeness).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import trimesh as tm
import warp as wp

import triwarp as tw
from triwarp.kernels.algorithms import ball_pivoting as kernel_bpa

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


# ======================================================================================
# Screened-Poisson reconstruction (screened_poisson)
#
# References: open3d ``create_from_point_cloud_poisson`` (``_o3d``) and PyMeshLab
# ``generate_surface_reconstruction_screened_poisson`` (``_pml``). Both reconstruct a different
# vertex set than triwarp, so every comparison is metric/topological, never vertex-for-vertex. The
# solve goes through ``warp.optim.linear.cg`` (CUDA-only), so the CPU device is skipped.
# ======================================================================================


def _skip_poisson_on_cpu(device: str) -> None:
    if wp.get_device(device).is_cpu:
        pytest.skip("screened_poisson requires CUDA: warp.optim.linear.cg is NaN on CPU.")


def _torus_cloud(n_major: int = 40, n_minor: int = 20, r_major: float = 1.0, r_minor: float = 0.35):
    u = np.linspace(0.0, 2.0 * np.pi, n_major, endpoint=False)
    v = np.linspace(0.0, 2.0 * np.pi, n_minor, endpoint=False)
    uu, vv = np.meshgrid(u, v)
    uu = uu.ravel()
    vv = vv.ravel()
    cx = np.cos(uu)
    cy = np.sin(uu)
    px = (r_major + r_minor * np.cos(vv)) * cx
    py = (r_major + r_minor * np.cos(vv)) * cy
    pz = r_minor * np.sin(vv)
    points = np.stack([px, py, pz], axis=1)
    nx = np.cos(vv) * cx
    ny = np.cos(vv) * cy
    nz = np.sin(vv)
    normals = np.stack([nx, ny, nz], axis=1)
    return points.astype(np.float64), normals.astype(np.float64)


def _mesh_trimesh(vertices_wp, faces_wp) -> tm.Trimesh:
    return tm.Trimesh(
        vertices=vertices_wp.numpy().astype(np.float64),
        faces=faces_wp.numpy().reshape(-1, 3),
        process=False,
    )


def _symmetric_chamfer(mesh_a: tm.Trimesh, mesh_b: tm.Trimesh, n_samples: int = 4000) -> float:
    """Mean symmetric chamfer distance between two triangle meshes via surface sampling."""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(0)
    sample_a, _ = tm.sample.sample_surface(mesh_a, n_samples, seed=int(rng.integers(1 << 30)))
    sample_b, _ = tm.sample.sample_surface(mesh_b, n_samples, seed=int(rng.integers(1 << 30)))
    tree_a = cKDTree(sample_a)
    tree_b = cKDTree(sample_b)
    a_to_b = tree_b.query(sample_a)[0].mean()
    b_to_a = tree_a.query(sample_b)[0].mean()
    return float(0.5 * (a_to_b + b_to_a))


def _points_to_surface(points_np: np.ndarray, mesh: tm.Trimesh) -> float:
    """Mean distance from a point set to the nearest point on a mesh surface."""
    return float(np.abs(tm.proximity.signed_distance(mesh, points_np)).mean())


def _open3d_poisson(points_np: np.ndarray, normals_np: np.ndarray, depth: int) -> tm.Trimesh:
    o3d = pytest.importorskip("open3d")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points_np, dtype=np.float64))
    pcd.normals = o3d.utility.Vector3dVector(np.ascontiguousarray(normals_np, dtype=np.float64))
    mesh_o3d, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    return tm.Trimesh(
        vertices=np.asarray(mesh_o3d.vertices), faces=np.asarray(mesh_o3d.triangles), process=False
    )


def test_poisson_sphere_watertight_manifold(device: str):
    _skip_poisson_on_cpu(device)
    points_np, normals_np = _sphere_cloud(3)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    vertices_wp, faces_wp = tw.reconstruction.screened_poisson(
        points_wp, normals_wp, depth=6, full_depth=4
    )
    mesh_tw = _mesh_trimesh(vertices_wp, faces_wp)

    assert tw.validation.is_watertight(vertices_wp, faces_wp)
    assert mesh_tw.euler_number == 2  # closed genus-0 surface

    radius_tw = np.linalg.norm(mesh_tw.vertices, axis=1)
    # A depth-6 cube spans ~2.2 across 64 cells => cell ~0.034; recon must hug the unit sphere.
    assert abs(radius_tw.mean() - 1.0) < 0.02
    assert np.abs(radius_tw - 1.0).max() < 0.06


def test_poisson_outward_orientation(device: str):
    _skip_poisson_on_cpu(device)
    points_np, normals_np = _sphere_cloud(3)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    vertices_wp, faces_wp = tw.reconstruction.screened_poisson(
        points_wp, normals_wp, depth=6, full_depth=4
    )
    # Outward normals => positive enclosed volume.
    assert _mesh_trimesh(vertices_wp, faces_wp).volume > 0.0


def test_poisson_torus_genus(device: str):
    _skip_poisson_on_cpu(device)
    points_np, normals_np = _torus_cloud()
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    vertices_wp, faces_wp = tw.reconstruction.screened_poisson(
        points_wp, normals_wp, depth=6, full_depth=4
    )
    mesh_tw = _mesh_trimesh(vertices_wp, faces_wp)
    assert tw.validation.is_watertight(vertices_wp, faces_wp)
    assert mesh_tw.euler_number == 0  # genus-1 torus: V - E + F = 0


def test_poisson_matches_open3d_metric(device: str):
    _skip_poisson_on_cpu(device)
    points_np, normals_np = _sphere_cloud(3)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    vertices_wp, faces_wp = tw.reconstruction.screened_poisson(
        points_wp, normals_wp, depth=6, full_depth=4
    )
    mesh_tw = _mesh_trimesh(vertices_wp, faces_wp)
    mesh_o3d = _open3d_poisson(points_np, normals_np, depth=6)
    # Same iso-surface up to discretization: symmetric chamfer well under a grid cell.
    assert _symmetric_chamfer(mesh_tw, mesh_o3d) < 0.03


def test_poisson_screening_improves_fit(device: str):
    _skip_poisson_on_cpu(device)
    points_np, normals_np = _sphere_cloud(3)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    vertices_screened, faces_screened = tw.reconstruction.screened_poisson(
        points_wp, normals_wp, depth=6, full_depth=4, point_weight=4.0
    )
    vertices_unscreened, faces_unscreened = tw.reconstruction.screened_poisson(
        points_wp, normals_wp, depth=6, full_depth=4, point_weight=0.0
    )
    fit_screened = _points_to_surface(points_np, _mesh_trimesh(vertices_screened, faces_screened))
    fit_unscreened = _points_to_surface(
        points_np, _mesh_trimesh(vertices_unscreened, faces_unscreened)
    )
    # Screening ties the surface to the samples: the fit is at least as good.
    assert fit_screened <= fit_unscreened + 1e-4


def test_poisson_finer_depth_reduces_error(device: str):
    _skip_poisson_on_cpu(device)
    points_np, normals_np = _sphere_cloud(4)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    vertices_coarse, _ = tw.reconstruction.screened_poisson(
        points_wp, normals_wp, depth=5, full_depth=4
    )
    vertices_fine, _ = tw.reconstruction.screened_poisson(
        points_wp, normals_wp, depth=7, full_depth=4
    )
    error_coarse = np.abs(np.linalg.norm(vertices_coarse.numpy(), axis=1) - 1.0).mean()
    error_fine = np.abs(np.linalg.norm(vertices_fine.numpy(), axis=1) - 1.0).mean()
    assert error_fine <= error_coarse


def test_poisson_matches_pymeshlab_metric(device: str):
    _skip_poisson_on_cpu(device)
    pymeshlab = pytest.importorskip("pymeshlab")
    points_np, normals_np = _sphere_cloud(3)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    vertices_wp, faces_wp = tw.reconstruction.screened_poisson(
        points_wp, normals_wp, depth=6, full_depth=4
    )
    mesh_tw = _mesh_trimesh(vertices_wp, faces_wp)

    ms = pymeshlab.MeshSet()
    ms.add_mesh(
        pymeshlab.Mesh(
            vertex_matrix=np.ascontiguousarray(points_np),
            v_normals_matrix=np.ascontiguousarray(normals_np),
        )
    )
    ms.generate_surface_reconstruction_screened_poisson(depth=6)
    mesh_current = ms.current_mesh()
    mesh_pml = tm.Trimesh(
        vertices=mesh_current.vertex_matrix(), faces=mesh_current.face_matrix(), process=False
    )
    assert _symmetric_chamfer(mesh_tw, mesh_pml) < 0.03


def test_poisson_requires_normals_and_valid_params(device: str):
    points_np, normals_np = _sphere_cloud(2)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)
    with pytest.raises(ValueError, match="full_depth"):
        tw.reconstruction.screened_poisson(points_wp, normals_wp, depth=6, full_depth=8)
    with pytest.raises(ValueError, match="full_depth"):
        tw.reconstruction.screened_poisson(points_wp, normals_wp, depth=11)
    with pytest.raises(ValueError, match="scale"):
        tw.reconstruction.screened_poisson(points_wp, normals_wp, scale=0.0)


def test_poisson_too_few_points(device: str):
    _skip_poisson_on_cpu(device)
    points_wp = wp.array(np.zeros((2, 3), dtype=np.float64), dtype=wp.vec3, device=device)
    normals_wp = wp.array(np.ones((2, 3), dtype=np.float64), dtype=wp.vec3, device=device)
    with pytest.raises(ValueError, match="at least 3 points"):
        tw.reconstruction.screened_poisson(points_wp, normals_wp, depth=4, full_depth=3)


def test_poisson_cpu_raises(device: str):
    if not wp.get_device(device).is_cpu:
        pytest.skip("CPU-only guard test.")
    points_np, normals_np = _sphere_cloud(2)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)
    with pytest.raises(NotImplementedError, match="CUDA"):
        tw.reconstruction.screened_poisson(points_wp, normals_wp, depth=4, full_depth=3)


# ======================================================================================
# Screened-Poisson, warp.fem adaptive backend (method="adaptive")
#
# The adaptive Nanogrid + variational assembly reconstructs a different vertex set than the dense
# backend, so comparisons stay metric/topological (dense-vs-fem cross-check is discretization
# agreement, never equality). CUDA-only, like the dense backend.
# ======================================================================================


def test_poisson_adaptive_sphere_watertight_manifold(device: str):
    _skip_poisson_on_cpu(device)
    points_np, normals_np = _sphere_cloud(4)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    vertices_wp, faces_wp = tw.reconstruction.screened_poisson(
        points_wp, normals_wp, depth=6, full_depth=4, method="adaptive"
    )
    mesh_tw = _mesh_trimesh(vertices_wp, faces_wp)

    assert tw.validation.is_watertight(vertices_wp, faces_wp)
    assert mesh_tw.euler_number == 2  # closed genus-0 surface
    assert mesh_tw.volume > 0.0  # outward orientation

    radius_tw = np.linalg.norm(mesh_tw.vertices, axis=1)
    assert abs(radius_tw.mean() - 1.0) < 0.03
    assert np.abs(radius_tw - 1.0).max() < 0.08


def test_poisson_adaptive_torus_genus(device: str):
    _skip_poisson_on_cpu(device)
    points_np, normals_np = _torus_cloud()
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    vertices_wp, faces_wp = tw.reconstruction.screened_poisson(
        points_wp, normals_wp, depth=6, full_depth=4, method="adaptive"
    )
    assert tw.validation.is_watertight(vertices_wp, faces_wp)
    assert _mesh_trimesh(vertices_wp, faces_wp).euler_number == 0  # genus-1 torus


def test_poisson_adaptive_matches_dense(device: str):
    _skip_poisson_on_cpu(device)
    points_np, normals_np = _sphere_cloud(4)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    vertices_dense, faces_dense = tw.reconstruction.screened_poisson(
        points_wp, normals_wp, depth=6, full_depth=4, method="dense"
    )
    vertices_adaptive, faces_adaptive = tw.reconstruction.screened_poisson(
        points_wp, normals_wp, depth=6, full_depth=4, method="adaptive"
    )
    # Same iso-surface on two different grids: symmetric chamfer well within a couple of voxels.
    chamfer = _symmetric_chamfer(
        _mesh_trimesh(vertices_dense, faces_dense), _mesh_trimesh(vertices_adaptive, faces_adaptive)
    )
    assert chamfer < 0.05


def test_poisson_adaptive_screening_improves_fit(device: str):
    _skip_poisson_on_cpu(device)
    points_np, normals_np = _sphere_cloud(4)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    vertices_screened, faces_screened = tw.reconstruction.screened_poisson(
        points_wp, normals_wp, depth=6, full_depth=4, point_weight=4.0, method="adaptive"
    )
    vertices_unscreened, faces_unscreened = tw.reconstruction.screened_poisson(
        points_wp, normals_wp, depth=6, full_depth=4, point_weight=0.0, method="adaptive"
    )
    fit_screened = _points_to_surface(points_np, _mesh_trimesh(vertices_screened, faces_screened))
    fit_unscreened = _points_to_surface(
        points_np, _mesh_trimesh(vertices_unscreened, faces_unscreened)
    )
    assert fit_screened <= fit_unscreened + 1e-4


def test_poisson_adaptive_confidence_runs(device: str):
    _skip_poisson_on_cpu(device)
    points_np, normals_np = _sphere_cloud(4)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    vertices_wp, faces_wp = tw.reconstruction.screened_poisson(
        points_wp, normals_wp, depth=6, full_depth=4, confidence=True, method="adaptive"
    )
    assert tw.validation.is_watertight(vertices_wp, faces_wp)
    assert _mesh_trimesh(vertices_wp, faces_wp).euler_number == 2


def test_poisson_invalid_method(device: str):
    points_np, normals_np = _sphere_cloud(2)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)
    with pytest.raises(ValueError, match="method"):
        tw.reconstruction.screened_poisson(points_wp, normals_wp, method="bogus")


# ======================================================================================
# Ball pivoting (ball_pivoting)
#
# The wave-parallel front is interpolating (output vertices are input points) and edge-manifold
# after cleanup, but is not guaranteed watertight on densely sampled closed surfaces (v1
# limitation), so the tests assert those robust invariants rather than watertightness / Euler.
# open3d BPA (``_o3d``) is used only as a loose face-count sanity reference.
# ======================================================================================


def _edge_multiplicity(faces_np: np.ndarray) -> np.ndarray:
    edges = np.sort(faces_np[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    return np.unique(edges, axis=0, return_counts=True)[1]


def test_ball_pivoting_interpolates_input(device: str):
    points_np, normals_np = _sphere_cloud(3)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    vertices_wp, faces_wp = tw.reconstruction.ball_pivoting(points_wp, normals_wp)
    vertices_np = vertices_wp.numpy().astype(np.float64)
    assert int(faces_wp.shape[0]) > 0

    # Interpolating: every output vertex coincides with an input point.
    from scipy.spatial import cKDTree

    distances = cKDTree(points_np).query(vertices_np)[0]
    assert distances.max() < 1e-6
    # Most input points are incorporated on a well-sampled sphere.
    assert vertices_np.shape[0] >= 0.8 * points_np.shape[0]


def test_ball_pivoting_edge_manifold(device: str):
    points_np, normals_np = _sphere_cloud(3)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    _vertices, faces_wp = tw.reconstruction.ball_pivoting(points_wp, normals_wp)
    # The cleanup tail removes non-manifold faces, so no edge is shared by more than two faces.
    assert _edge_multiplicity(faces_wp.numpy().reshape(-1, 3)).max() <= 2


def test_ball_pivoting_closes_a_dense_sphere(device: str):
    """
    The strongest end-to-end guard available: a uniformly sampled closed surface must close.

    With a persistent front and Border-edge retirement, a subdivided icosphere reconstructs to
    exactly the Euler face count with no boundary edge at all. That single assertion catches both
    directions of failure at once — retiring an edge that could still have succeeded would leave
    holes, and letting colliding fronts triangulate a neighbourhood twice (the artefact the
    front-rebuilt-per-wave design produced, at 24% boundary edges and 3.1 faces per vertex) would
    push the face count well past ``2 v - 4``.
    """
    from scipy.spatial import cKDTree

    points_np, normals_np = _sphere_cloud(3)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    vertices_wp, faces_wp = tw.reconstruction.ball_pivoting(points_wp, normals_wp)
    faces_np = faces_wp.numpy().reshape(-1, 3)
    multiplicity = _edge_multiplicity(faces_np)

    assert int((multiplicity == 1).sum()) == 0  # watertight
    assert int(multiplicity.max()) == 2  # edge-manifold
    n_referenced = len(np.unique(faces_np))
    assert n_referenced == points_np.shape[0]  # every input point used
    assert faces_np.shape[0] == 2 * n_referenced - 4  # Euler, for a closed genus-0 surface

    # And it interpolates: every input point is a vertex of the result.
    assert cKDTree(vertices_wp.numpy().astype(np.float64)).query(points_np)[0].max() < 1e-6


def test_ball_pivoting_grows_the_triangle_budget(device: str):
    """
    A budget far below what the mesh needs must grow, not raise or truncate.

    Growing rehashes the edge table (slot indices move) and rebuilds the front list from it, so
    this also covers that path. The result has to match a run that never had to grow.
    """
    points_np, normals_np = _sphere_cloud(3)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)
    grid = tw.neighbors.hashgrid_from_points(points_wp, 2.0 * 0.2)

    faces_per_budget = []
    for start_budget in (64, 4 * points_np.shape[0] + 16):
        state = tw.reconstruction._BpaState(
            points_wp, normals_wp, grid, 0.2, 0.2, math.cos(math.pi / 2.0), start_budget
        )
        tw.reconstruction._bpa_run(state, 16 * points_np.shape[0])
        counters_np = state.counters.numpy()
        assert counters_np[kernel_bpa.CNT_DONE] == 1
        faces_per_budget.append(int(counters_np[kernel_bpa.CNT_FACE]))
    grown, direct = faces_per_budget
    assert grown > 64  # it really did outgrow the initial allocation
    assert abs(grown - direct) <= 0.01 * direct


def test_ball_pivoting_face_count_near_open3d(device: str):
    points_np, normals_np = _sphere_cloud(3)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    # Auto-guessed radius roughly matches the mean spacing; use it for open3d too.
    _idx, dist = tw.neighbors.query_bvh_nearest(points_wp, points_wp, k=7)
    spacing = float(np.mean(dist.numpy()[:, 1:][np.isfinite(dist.numpy()[:, 1:])]))
    radius = 1.5 * spacing

    _vertices, faces_wp = tw.reconstruction.ball_pivoting(points_wp, normals_wp, radius=radius)
    n_faces_tw = int(faces_wp.shape[0]) // 3

    o3d = pytest.importorskip("open3d")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points_np))
    pcd.normals = o3d.utility.Vector3dVector(np.ascontiguousarray(normals_np))
    mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector([radius, 2.0 * radius])
    )
    n_faces_o3d = np.asarray(mesh_o3d.triangles).shape[0]
    # Same order of magnitude as open3d (both reconstruct ~2n triangles on a closed sphere).
    assert 0.5 * n_faces_o3d <= n_faces_tw <= 2.0 * n_faces_o3d


def test_ball_pivoting_small_radius_leaves_holes(device: str):
    points_np, normals_np = _sphere_cloud(3)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    _idx, dist = tw.neighbors.query_bvh_nearest(points_wp, points_wp, k=2)
    spacing = float(np.mean(dist.numpy()[:, 1][np.isfinite(dist.numpy()[:, 1])]))

    _v_small, faces_small = tw.reconstruction.ball_pivoting(
        points_wp, normals_wp, radius=0.2 * spacing
    )
    _v_ok, faces_ok = tw.reconstruction.ball_pivoting(points_wp, normals_wp, radius=1.5 * spacing)
    # A ball far smaller than the sampling never rests on three points: far fewer (or no) faces.
    assert int(faces_small.shape[0]) < int(faces_ok.shape[0])


def test_ball_pivoting_estimated_normals(device: str):
    points_np, _normals = _sphere_cloud(3)
    points_wp = wp.array(np.ascontiguousarray(points_np), dtype=wp.vec3, device=device)
    # normals=None triggers PCA normal estimation (valid for this star-shaped cloud).
    _vertices, faces_wp = tw.reconstruction.ball_pivoting(points_wp, None)
    assert int(faces_wp.shape[0]) > 0


def test_ball_pivoting_too_few_points(device: str):
    points_wp = wp.array(np.zeros((2, 3), dtype=np.float64), dtype=wp.vec3, device=device)
    normals_wp = wp.array(np.ones((2, 3), dtype=np.float64), dtype=wp.vec3, device=device)
    _vertices, faces_wp = tw.reconstruction.ball_pivoting(points_wp, normals_wp)
    assert int(faces_wp.shape[0]) == 0
