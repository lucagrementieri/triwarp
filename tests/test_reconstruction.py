"""
Tests for point-cloud surface reconstruction.

No trimesh/libigl equivalent exists for this algorithm, so the reference is MeshLib's own
``triangulatePointCloud`` (via the ``meshlib`` Python bindings, ``_mm`` suffix) plus geometric
property checks (watertightness, edge-manifoldness, Euler characteristic, surface closeness).
"""

from __future__ import annotations

import math

import igl
import numpy as np
import open3d as o3d
import pymeshlab as ml
import pytest
import trimesh as tm
import warp as wp

import triwarp as tw
import triwarp.typing as twt
from tests.comparisons import (
    assert_unordered_rows_equal,
    canonical_winding,
    hausdorff_two_sided,
    symmetric_chamfer,
    symmetric_surface_distance,
)
from tests.conversions import (
    open3d_to_trimesh,
    points_to_open3d,
    points_to_pymeshlab,
    points_to_pyvista,
)
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
    assert tw.totals.euler_characteristic(faces_wp) == 2
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
    assert tw.totals.euler_characteristic(faces_wp) == 0


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


def test_holes_seal_small_hole(device: str):
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


@pytest.mark.parity("delaunay_triangulation", "scipy")
def test_delaunay_matches_scipy_random(device: str):
    """Class B: the same triangulation as Qhull's, compared as an undirected edge set."""
    from scipy.spatial import Delaunay

    rng = np.random.default_rng(42)
    points_np = rng.random((200, 2)).astype(np.float32)
    points_wp = wp.array(points_np, dtype=wp.vec2, device=device)

    faces_wp = tw.reconstruction.delaunay_triangulation(points_wp).numpy()
    faces_sp = Delaunay(points_np.astype(np.float64)).simplices

    assert _edge_set(faces_wp) == _edge_set(faces_sp.reshape(-1))


@pytest.mark.parity("delaunay_triangulation", "pyvista")
def test_delaunay_contains_the_pyvista_triangulation(device: str):
    """
    Class B: ``delaunay_2d``'s triangles are a strict **subset** of triwarp's; the gap is hull.

    Both are Delaunay triangulations of the same cloud, so on the interior they agree face for face;
    they differ on how much of the convex hull they keep. Measured on 200 uniform points: pyvista
    returns **374** triangles, every one of them present in triwarp's **384**, so the 10 extra are
    hull slivers ``vtkDelaunay2D`` drops. The comparison is therefore containment plus a bound on
    the excess, not equality -- and stating it that way is the point, since an equality assert would
    fail on a correct implementation.

    The cloud is passed to VTK as ``(x, y, 0)``: ``delaunay_2d`` projects onto the best-fit plane,
    so a genuinely planar input is what keeps its answer comparable to a 2-D triangulator's.
    """
    rng = np.random.default_rng(0)
    points_np = rng.random((200, 2))
    points_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec2, device=device)

    triangulated_pv = points_to_pyvista(np.column_stack([points_np, np.zeros(len(points_np))]))
    triangulated_pv = triangulated_pv.delaunay_2d()
    assert triangulated_pv.is_all_triangles
    faces_pv = {
        tuple(row) for row in np.sort(np.asarray(triangulated_pv.regular_faces), axis=1).tolist()
    }
    assert len(faces_pv) > 300, "the reference triangulated the cloud before it is compared to"

    faces_wp = tw.reconstruction.delaunay_triangulation(points_wp).numpy().reshape(-1, 3)
    faces_tw = {tuple(row) for row in np.sort(faces_wp, axis=1).tolist()}

    assert faces_pv <= faces_tw
    # The excess is hull slivers only: a few triangles, not a different triangulation.
    assert len(faces_tw - faces_pv) < 0.05 * len(faces_tw)


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


def _points_to_surface(points_np: np.ndarray, mesh: tm.Trimesh) -> float:
    """Mean distance from a point set to the nearest point on a mesh surface."""
    return float(np.abs(tm.proximity.signed_distance(mesh, points_np)).mean())


def _open3d_poisson(points_np: np.ndarray, normals_np: np.ndarray, depth: int) -> tm.Trimesh:
    pcd = points_to_open3d(points_np, normals_np)
    mesh_o3d, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    return open3d_to_trimesh(mesh_o3d)


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


@pytest.mark.parity(
    "screened_poisson",
    "open3d",
    benchmarked=False,
    reason="open3d wraps Kazhdan's CPU solver, which was 6 322 s across the open3d and pymeshlab "
    "screened_poisson rows -- 73 % of the whole benchmark suite -- and [dragon-open3d-*] ran 93 "
    "minutes without completing a round. Those rows were removed rather than capped; the "
    "comparison lives here, at a size a correctness test can afford.",
)
def test_poisson_matches_open3d_metric(device: str):
    """
    Class C: mean sample-to-surface distance, there being no vertex correspondence to compare.

    The two solvers march different octrees and return different meshes -- 32 552 faces against
    open3d's 7 976 on this cloud -- so no vertex, face or count agrees and only a surface metric
    can state the claim. Excludes the bug class "the iso-surface is in the wrong place": a global
    radius error, a sign flip on the indicator, a mislocated level set.

    Measured agreement is **0.0043** and the threshold is 0.015, a **3.5x** margin. Mutation
    probe, run against this assert: scaling triwarp's answer by 0.98 makes it read 0.0188 and the
    assert fails, so a 2% radius error on a unit sphere is caught. A mesh against itself scores
    exactly 0.0, which is the whole point of the metric -- see the warning below.

    !!! warning "This assert used to be vacuous, and the reason generalises"
        It was ``symmetric_chamfer(...) < 0.03``, which is sample-to-*sample* and therefore has a
        noise floor of ``0.5 * sqrt(area / n_samples)`` = 0.028 on this fixture. The real
        disagreement, 0.0043, was two hundredths below that floor: the pass margin was 1.08x and
        a mesh compared with *itself* scored 0.0279. Any threshold within a few percent of a
        sampling floor is testing the sampler.
    """
    _skip_poisson_on_cpu(device)
    points_np, normals_np = _sphere_cloud(3)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    vertices_wp, faces_wp = tw.reconstruction.screened_poisson(
        points_wp, normals_wp, depth=6, full_depth=4
    )
    mesh_tw = _mesh_trimesh(vertices_wp, faces_wp)
    # depth=5 on the reference side, not 6: on this 642-point cloud open3d returns the *same*
    # 7 976 faces at both depths (measured) for 0.28 s instead of 0.85 s, so the extra octree
    # level is cost without an answer. Only the reference drops -- triwarp stays at the depth=6
    # its sibling tests above use, where it resolves 32 552 faces in 0.01 s either way.
    mesh_o3d = _open3d_poisson(points_np, normals_np, depth=5)
    mean_distance, _ = symmetric_surface_distance(mesh_tw, mesh_o3d)
    assert mean_distance < 0.015


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


@pytest.mark.parity(
    "screened_poisson",
    "pymeshlab",
    benchmarked=False,
    reason="pymeshlab wraps the same Kazhdan CPU solver open3d does, and the two rows together "
    "were 6 322 s -- 73 % of the whole benchmark suite -- for a reference triwarp already beats "
    "15-25x. Removed from benchmarks/test_reconstruction.py rather than capped; the comparison "
    "lives here instead.",
)
def test_poisson_matches_pymeshlab_metric(device: str):
    """
    Class C: the [`test_poisson_matches_open3d_metric`][] comparison against the other reference.

    Measured agreement is **0.0041** against the same 0.015 threshold, a **3.7x** margin, and the
    same mutation probe reads 0.0189 here and fails; the noise-floor warning on that test applies
    unchanged. Worth having
    both: open3d and pymeshlab agree with each other to 0.0015, a quarter of either one's distance
    to triwarp, so they are not independent enough for one to stand in for the other -- but that
    also means a triwarp regression would have to move past *both* to stay unnoticed.
    """
    _skip_poisson_on_cpu(device)
    points_np, normals_np = _sphere_cloud(3)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    vertices_wp, faces_wp = tw.reconstruction.screened_poisson(
        points_wp, normals_wp, depth=6, full_depth=4
    )
    mesh_tw = _mesh_trimesh(vertices_wp, faces_wp)

    # An oriented point cloud, so this is one of the few places a MeshSet is built from vertices
    # alone rather than through ``conversions.trimesh_to_pymeshlab``.
    meshset_pml = ml.MeshSet()
    meshset_pml.add_mesh(
        ml.Mesh(
            vertex_matrix=np.ascontiguousarray(points_np),
            v_normals_matrix=np.ascontiguousarray(normals_np),
        )
    )
    # depth=5, not 6, for the reason given on ``test_poisson_matches_open3d_metric``: MeshLab
    # returns the same 7 976 faces at both depths on this cloud, in 0.24 s instead of 0.48 s.
    # Do not go below 5 without re-measuring -- the axis is *not* monotonic in either time or
    # resolution, and depth=4 drops to 2 024 faces.
    meshset_pml.generate_surface_reconstruction_screened_poisson(depth=5)
    mesh_current = meshset_pml.current_mesh()
    mesh_pml = tm.Trimesh(
        vertices=mesh_current.vertex_matrix(), faces=mesh_current.face_matrix(), process=False
    )
    mean_distance, _ = symmetric_surface_distance(mesh_tw, mesh_pml)
    assert mean_distance < 0.015


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


def test_poisson_cpu_matches_cuda():
    """Class A: the CPU cascadic solve reconstructs the CUDA surface, vertex for vertex."""
    if not wp.is_cuda_available():
        pytest.skip("needs both devices to compare them")
    points_np, normals_np = _sphere_cloud(2)

    surfaces = {}
    for device in ("cpu", "cuda:0"):
        points_wp, normals_wp = _to_warp(points_np, normals_np, device)
        vertices_wp, faces_wp = tw.reconstruction.screened_poisson(
            points_wp, normals_wp, depth=4, full_depth=3
        )
        surfaces[device] = (vertices_wp.numpy(), faces_wp.numpy())

    assert surfaces["cpu"][1].size > 0
    assert np.array_equal(surfaces["cpu"][1], surfaces["cuda:0"][1])
    assert np.allclose(surfaces["cpu"][0], surfaces["cuda:0"][0], rtol=1e-5, atol=1e-5)


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
    # Same iso-surface on two different grids, so a surface metric rather than a correspondence.
    # Its old ``symmetric_chamfer(...) < 0.05`` sat only 1.8x above that helper's 0.028 sampling
    # floor, which is most of what it was measuring; sample-to-surface has no floor.
    mean_distance, _ = symmetric_surface_distance(
        _mesh_trimesh(vertices_dense, faces_dense), _mesh_trimesh(vertices_adaptive, faces_adaptive)
    )
    assert mean_distance < 0.02


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


def test_ball_pivoting_is_reproducible(device: str):
    """
    Two runs of one build on one cloud must reconstruct the same triangles.

    A wave resolves competing proposals with ``wp.atomic_min`` over ``proposal_key`` — packed from
    the proposal's own source edge — rather than over the ``wp.atomic_add`` slot it was handed, so
    nothing about the answer depends on the order threads reach an atomic. A losing proposal is not
    merely reordered: its front edge is retried a wave later against mutated state, so the
    divergence used to compound into different geometry.

    **The fixture is the load-bearing choice.** Every icosphere is reproducible even *without* the
    fix — a uniformly sampled closed sphere reconstructs to its exact Euler triangulation, so wave
    order has nothing left to decide — and asserting on one would be vacuous. Measured on this
    torus, four runs of the pre-fix implementation give 2 980 / 3 036 / 3 042 / 3 089 faces and
    four different meshes, against a constant 3 269 and one mesh after it.

    Two assertions, covering different halves:

    * the wave loop's own output, compared as a **wound** triangle set — same triangles and same
      winding, which is the property the key buys. Not compared buffer-to-buffer: ``CNT_FACE`` is a
      ``wp.atomic_add``, so the row order is deliberately still arrival-ordered;
    * the public function, compared as an **unoriented** triangle set, because its cleanup tail
      runs [`repair.make_winding_consistent`][triwarp.repair.make_winding_consistent], whose
      arbitrary per-component seed face is still chosen nondeterministically (measured on this same
      fixture). That is a separate defect in that module, and it is why the second assertion sorts
      within the row rather than using ``canonical_winding``.
    """
    torus_tm = tm.creation.torus(
        major_radius=1.0, minor_radius=0.35, major_sections=64, minor_sections=32
    )
    points_wp, normals_wp = _to_warp(torus_tm.vertices, torus_tm.vertex_normals, device)
    n_points = int(points_wp.shape[0])
    # The wrapper's own auto-radius, pinned here so the raw runs below see the identical parameter.
    spacing = tw.neighbors.query_bvh_nearest(points_wp, points_wp, k=7)[1].numpy()[:, 1:]
    radius = 1.5 * float(spacing[np.isfinite(spacing) & (spacing > 0.0)].mean())

    grid = tw.neighbors.hashgrid_from_points(points_wp, radius)
    bvh = tw.neighbors.bvh_from_points(points_wp)
    raw_runs = []
    for _ in range(2):
        state = tw.reconstruction._BpaState(
            points_wp, normals_wp, grid, bvh, radius, 0.2, -1.0, 4 * n_points + 16
        )
        tw.reconstruction._bpa_run(state, 16 * n_points)
        n_faces = int(state.counters.numpy()[kernel_bpa.CNT_FACE])
        raw_runs.append(state.all_faces.numpy()[: n_faces * 3].reshape(-1, 3))

    assert raw_runs[0].shape[0] > 0
    assert_unordered_rows_equal(canonical_winding(raw_runs[0]), canonical_winding(raw_runs[1]))

    _vertices, first_wp = tw.reconstruction.ball_pivoting(points_wp, normals_wp, radius=radius)
    _vertices, second_wp = tw.reconstruction.ball_pivoting(points_wp, normals_wp, radius=radius)
    assert_unordered_rows_equal(
        np.sort(first_wp.numpy().reshape(-1, 3), axis=1),
        np.sort(second_wp.numpy().reshape(-1, 3), axis=1),
    )


def test_ball_pivoting_grows_the_triangle_budget(device: str):
    """
    A budget far below what the mesh needs must grow, not raise or truncate.

    Growing rehashes the edge table (slot indices move) and rebuilds the front list from it, so
    this also covers that path. The result has to match a run that never had to grow.
    """
    points_np, normals_np = _sphere_cloud(3)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)
    grid = tw.neighbors.hashgrid_from_points(points_wp, 2.0 * 0.2)
    bvh = tw.neighbors.bvh_from_points(points_wp)

    faces_per_budget = []
    for start_budget in (64, 4 * points_np.shape[0] + 16):
        state = tw.reconstruction._BpaState(
            points_wp, normals_wp, grid, bvh, 0.2, 0.2, math.cos(math.pi / 2.0), start_budget
        )
        tw.reconstruction._bpa_run(state, 16 * points_np.shape[0])
        counters_np = state.counters.numpy()
        assert counters_np[kernel_bpa.CNT_DONE] == 1
        faces_per_budget.append(int(counters_np[kernel_bpa.CNT_FACE]))
    grown, direct = faces_per_budget
    assert grown > 64  # it really did outgrow the initial allocation
    assert abs(grown - direct) <= 0.01 * direct


@pytest.mark.parity("ball_pivoting", "open3d")
def test_ball_pivoting_face_count_near_open3d(device: str):
    points_np, normals_np = _sphere_cloud(3)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    # Auto-guessed radius roughly matches the mean spacing; use it for open3d too.
    _idx, dist = tw.neighbors.query_bvh_nearest(points_wp, points_wp, k=7)
    spacing = float(np.mean(dist.numpy()[:, 1:][np.isfinite(dist.numpy()[:, 1:])]))
    radius = 1.5 * spacing

    _vertices, faces_wp = tw.reconstruction.ball_pivoting(points_wp, normals_wp, radius=radius)
    n_faces_tw = int(faces_wp.shape[0]) // 3

    pcd = points_to_open3d(points_np, normals_np)
    mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector([radius, 2.0 * radius])
    )
    n_faces_o3d = np.asarray(mesh_o3d.triangles).shape[0]
    # Same order of magnitude as open3d (both reconstruct ~2n triangles on a closed sphere).
    assert 0.5 * n_faces_o3d <= n_faces_tw <= 2.0 * n_faces_o3d


@pytest.mark.parity("ball_pivoting", "pymeshlab")
def test_ball_pivoting_matches_pymeshlab(device: str):
    """
    Class C against VCGlib's original BPA -- but a much tighter one than the open3d row above.

    Two ball-pivoting fronts advance in different orders and produce different triangles, so there
    is no face correspondence to recover. What is comparable is sharp: BPA adds no vertices, so the
    **vertex sets are identical** (Hausdorff **3.8e-08**, class A), and at the same radius the two
    close the same surface -- **1 280 faces against 1 277**, 0.23% apart.

    **The reference's ``clustering`` parameter is load-bearing and its zero is not "off".** At
    ``clustering=0`` the filter returns **0 faces**; 20% is MeshLab's default and the seed-triangle
    spacing floor the algorithm needs. ``benchmarks/test_reconstruction.py`` passed 0 and so timed a
    filter that reconstructed nothing (9.6 ms for no output, against 2.7 ms for the real thing);
    that row now passes the default.

    **Bug class excluded:** a front that closes the surface at the wrong scale or drifts off the
    samples. **Mutation probe** on the chamfer, normalized by the reference's own sampling floor
    (0.0196 at 8 000 samples, essentially the whole signal, so the raw number is not usable):
    measured ratio **0.99**, against **1.47** for a 2% scale of triwarp's output, **2.76** for 5%
    and
    **3.57** for a half-spacing translation. The 1.3 bound therefore sits 1.3x above the measured
    agreement and 1.13x below the smallest probe -- deliberately recorded as the weakest link here,
    because two BPA meshes over one point set cannot differ by more than the sampling density. The
    face-count and vertex-set asserts are the ones with real margin (21x and exact).

    One genuine difference, asserted rather than smoothed over: triwarp's result is watertight and
    MeshLab's is not, because those 3 missing faces are unclosed holes.
    """
    points_np, normals_np = _sphere_cloud(3)
    points_wp, normals_wp = _to_warp(points_np, normals_np, device)

    _index_wp, distance_wp = tw.neighbors.query_bvh_nearest(points_wp, points_wp, k=7)
    neighbor_distance_np = distance_wp.numpy()[:, 1:]
    spacing = float(np.mean(neighbor_distance_np[np.isfinite(neighbor_distance_np)]))
    radius = 1.5 * spacing

    vertices_wp, faces_wp = tw.reconstruction.ball_pivoting(points_wp, normals_wp, radius=radius)
    mesh_wp = tm.Trimesh(
        vertices_wp.numpy().astype(np.float64), faces_wp.numpy().reshape(-1, 3), process=False
    )

    meshset_pml = points_to_pymeshlab(points_np, normals_np)
    meshset_pml.generate_surface_reconstruction_ball_pivoting(
        ballradius=ml.PureValue(radius), clustering=20.0
    )
    mesh_pml = tm.Trimesh(
        np.asarray(meshset_pml.current_mesh().vertex_matrix(), dtype=np.float64),
        np.asarray(meshset_pml.current_mesh().face_matrix()),
        process=False,
    )
    assert mesh_pml.faces.shape[0] > 0, "clustering=0 reconstructs nothing; this must not regress"

    # BPA adds no vertices, so both must be exactly the input samples.
    assert hausdorff_two_sided(mesh_wp.vertices, points_np) < 1e-5
    assert hausdorff_two_sided(mesh_wp.vertices, mesh_pml.vertices) < 1e-5
    assert 0.95 <= mesh_wp.faces.shape[0] / mesh_pml.faces.shape[0] <= 1.05

    floor = symmetric_chamfer(mesh_pml, mesh_pml, n_samples=8000)
    assert symmetric_chamfer(mesh_wp, mesh_pml, n_samples=8000) < 1.3 * floor

    # triwarp closes the surface; MeshLab leaves those three faces as holes.
    assert mesh_wp.is_watertight
    assert not mesh_pml.is_watertight


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


# ---------------------------------------------------------------------------
# Marching cubes and uniform resampling (analytic + pymeshlab / skimage)
# ---------------------------------------------------------------------------


def _sphere_field(resolution: int, radius: float) -> np.ndarray:
    """Build an analytic SDF of a sphere on a ``[-1, 1]`` lattice: the exact-answer fixture."""
    axis_np = np.linspace(-1.0, 1.0, resolution)
    x_np, y_np, z_np = np.meshgrid(axis_np, axis_np, axis_np, indexing="ij")
    return (np.sqrt(x_np**2 + y_np**2 + z_np**2) - radius).astype(np.float32)


def _upload_mesh(mesh_tm: tm.Trimesh, device: str):
    return (
        wp.array(
            np.ascontiguousarray(mesh_tm.vertices, dtype=np.float32), dtype=wp.vec3, device=device
        ),
        wp.array(
            np.ascontiguousarray(mesh_tm.faces.reshape(-1), dtype=np.int32),
            dtype=wp.int32,
            device=device,
        ),
    )


def test_marching_cubes_extracts_an_analytic_sphere(device: str) -> None:
    """Every extracted vertex must land on the sphere the field describes, to grid resolution."""
    radius, resolution = 0.6, 32
    field_wp = wp.array(_sphere_field(resolution, radius), dtype=wp.float32, device=device)
    vertices_wp, faces_wp = tw.reconstruction.marching_cubes(
        twt.as_array3d(field_wp, wp.float32),
        bounds=(wp.vec3(-1.0, -1.0, -1.0), wp.vec3(1.0, 1.0, 1.0)),
    )
    assert int(faces_wp.shape[0]) > 0
    spacing = 2.0 / (resolution - 1)
    radii_np = np.linalg.norm(vertices_wp.numpy(), axis=1)
    assert np.abs(radii_np - radius).max() < spacing


def test_marching_cubes_index_space_by_default(device: str) -> None:
    """Without ``bounds`` the vertices are lattice indices, which is the documented convention."""
    resolution = 24
    field_wp = wp.array(_sphere_field(resolution, 0.6), dtype=wp.float32, device=device)
    vertices_np = tw.reconstruction.marching_cubes(twt.as_array3d(field_wp, wp.float32))[0].numpy()
    assert vertices_np.min() >= 0.0
    assert vertices_np.max() <= float(resolution - 1)
    # Centred field, so the extracted surface is centred on the lattice centre.
    assert np.allclose(vertices_np.mean(axis=0), 0.5 * (resolution - 1), atol=0.5)


def test_marching_cubes_empty_when_the_field_never_crosses(device: str) -> None:
    field_wp = wp.array(np.full((8, 8, 8), 1.0, dtype=np.float32), dtype=wp.float32, device=device)
    _vertices_wp, faces_wp = tw.reconstruction.marching_cubes(twt.as_array3d(field_wp, wp.float32))
    assert int(faces_wp.shape[0]) == 0


def test_marching_cubes_invalid(device: str) -> None:
    thin_wp = wp.array(np.zeros((1, 8, 8), dtype=np.float32), dtype=wp.float32, device=device)
    with pytest.raises(ValueError, match="at least 2 wide"):
        tw.reconstruction.marching_cubes(twt.as_array3d(thin_wp, wp.float32))


@pytest.mark.parametrize("offset", [0.0, 0.2, -0.2])
def test_resample_uniform_offsets_a_sphere(device: str, offset: float) -> None:
    """
    Offset the one shape whose offset surface is known exactly: a sphere by ``d`` gives ``1 + d``.

    Both signs are covered because they are different code paths in spirit — a positive offset needs
    the lattice padded beyond the bounding box (or it clips) and a negative one does not.
    """
    sphere_tm = tm.creation.icosphere(subdivisions=3, radius=1.0)
    vertices_wp, faces_wp = _upload_mesh(sphere_tm, device)
    voxel_size = 0.05

    out_vertices_wp, out_faces_wp = tw.reconstruction.resample_uniform(
        vertices_wp, faces_wp, voxel_size=voxel_size, offset=offset
    )
    radii_np = np.linalg.norm(out_vertices_wp.numpy(), axis=1)
    # The icosphere is *inscribed*, so its own surface sits between ``cos`` of half the face angle
    # and 1; the offset shifts that band without widening it much.
    assert np.abs(radii_np - (1.0 + offset)).max() < 2.0 * voxel_size

    out_tm = tm.Trimesh(
        out_vertices_wp.numpy().astype(np.float64),
        out_faces_wp.numpy().reshape(-1, 3),
        process=False,
    )
    assert out_tm.is_watertight
    assert out_tm.volume > 0.0


def test_resample_uniform_repairs_a_broken_mesh(device: str) -> None:
    """
    The bluntest repair there is: the topology comes from the grid, so the input's cannot leak.

    The input here has duplicated faces, an inverted one and a non-manifold edge — three defects
    that each need their own function in [`triwarp.repair`][triwarp.repair] — and the resampled
    result is a clean watertight sphere regardless.
    """
    sphere_tm = tm.creation.icosphere(subdivisions=3, radius=1.0)
    faces_np = np.asarray(sphere_tm.faces)
    broken_np = np.vstack([faces_np, faces_np[:20], faces_np[30:40][:, ::-1]])
    vertices_wp = wp.array(
        np.ascontiguousarray(sphere_tm.vertices, dtype=np.float32), dtype=wp.vec3, device=device
    )
    faces_wp = wp.array(
        np.ascontiguousarray(broken_np.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device
    )
    assert not tw.validation.is_edge_manifold(faces_wp)

    out_vertices_wp, out_faces_wp = tw.reconstruction.resample_uniform(
        vertices_wp, faces_wp, voxel_size=0.06
    )
    assert tw.validation.is_edge_manifold(out_faces_wp, allow_boundary_edges=False)
    out_tm = tm.Trimesh(
        out_vertices_wp.numpy().astype(np.float64),
        out_faces_wp.numpy().reshape(-1, 3),
        process=False,
    )
    assert out_tm.is_watertight
    assert np.isclose(np.abs(out_tm.volume), 4.0 / 3.0 * np.pi, rtol=0.1)


@pytest.mark.parity("resample_uniform", "pymeshlab")
def test_resample_uniform_matches_pymeshlab(device: str) -> None:
    """
    ``generate_resampled_uniform_mesh`` is the same algorithm at the same absolute cell size.

    One parameter hazard, found by probing: its ``offset`` as a ``PercentageValue`` runs from full
    erosion at ``0%`` to full dilation at ``100%``, so **``PercentageValue(50)`` — its default — is
    the *zero* offset** and ``PercentageValue(0)`` erodes a unit sphere down to radius 0.30. Passing
    ``PureValue(0.0)`` instead means an absolute offset of zero, which is what ``offset=0.0`` is
    here.

    What must agree is the surface: both watertight, both enclosing the sphere's volume, and a
    two-sided Hausdorff distance between them of well under a cell.
    """
    sphere_tm = tm.creation.icosphere(subdivisions=3, radius=1.0)
    voxel_size = 0.06
    meshset_pml = ml.MeshSet()
    meshset_pml.add_mesh(
        ml.Mesh(
            np.ascontiguousarray(sphere_tm.vertices, dtype=np.float64),
            np.ascontiguousarray(sphere_tm.faces, dtype=np.int32),
        )
    )
    meshset_pml.generate_resampled_uniform_mesh(
        cellsize=ml.PureValue(voxel_size), offset=ml.PureValue(0.0)
    )
    mesh_pml = meshset_pml.current_mesh()
    pml_tm = tm.Trimesh(mesh_pml.vertex_matrix(), mesh_pml.face_matrix(), process=False)

    vertices_wp, faces_wp = _upload_mesh(sphere_tm, device)
    out_vertices_wp, out_faces_wp = tw.reconstruction.resample_uniform(
        vertices_wp, faces_wp, voxel_size=voxel_size
    )
    out_tm = tm.Trimesh(
        out_vertices_wp.numpy().astype(np.float64),
        out_faces_wp.numpy().reshape(-1, 3),
        process=False,
    )
    assert out_tm.is_watertight
    assert np.isclose(np.abs(out_tm.volume), np.abs(pml_tm.volume), rtol=0.05)

    # Two-sided Hausdorff between the surfaces, within a cell.
    sample_wp, _face = tm.sample.sample_surface(out_tm, 4000, seed=0)
    sample_pml, _face_pml = tm.sample.sample_surface(pml_tm, 4000, seed=1)
    assert np.abs(tm.proximity.signed_distance(pml_tm, sample_wp)).max() < 2.0 * voxel_size
    assert np.abs(tm.proximity.signed_distance(out_tm, sample_pml)).max() < 2.0 * voxel_size


@pytest.mark.parity("resample_uniform", "igl")
def test_resample_uniform_matches_igl(device: str) -> None:
    """
    Class C (no vertex correspondence): the two isosurfaces coincide to **0.06 of a voxel**.

    ``igl.offset_surface(V, F, isolevel, s, sign_type)`` samples the same signed distance field on a
    grid and marches it. Two named parameter transforms put the two on one lattice: ``isolevel=0``
    is triwarp's zero offset, and ``s`` is a *cell count along the longest axis* rather than a
    length, so it gets ``round(longest_extent / voxel_size)``. The sign mode is ``PSEUDONORMAL``,
    which ``tests/test_proximity.py::test_signed_distance_on_mesh_matches_igl`` establishes agrees
    with triwarp's default to 8e-8 -- the winding modes would scale the field by ``1 - 2w`` and move
    the isosurface.

    No correspondence exists between the outputs (4 186 vertices against triwarp's 5 310 on this
    fixture, since the two march the lattice into different triangle sets), so the comparison is the
    surface: both watertight, enclosed volumes within 5%, and a two-sided Hausdorff distance under a
    quarter of a voxel.

    **Bug class excluded:** a grid anchored differently, or an isolevel or sign convention that
    shifts the surface -- exactly what the plan flagged as the risk for this pair. **Mutation probe,
    measured:** re-running igl at ``isolevel=0.02`` and ``0.05`` moves the one-sided Hausdorff to
    **0.36 and 0.86 voxels** against 0.06 at zero, so the ``0.25``-voxel bound sits 4.2x above the
    measured agreement and fails on a shift of a third of a voxel. That is what makes it a test of
    the anchoring rather than of "both are roughly a sphere".
    """
    sphere_tm = tm.creation.icosphere(subdivisions=3, radius=1.0)
    voxel_size = 0.06
    vertices_np = np.ascontiguousarray(sphere_tm.vertices, dtype=np.float64)
    faces_np = np.ascontiguousarray(sphere_tm.faces, dtype=np.int64)

    extent = float((vertices_np.max(axis=0) - vertices_np.min(axis=0)).max())
    vertices_igl, faces_igl = igl.offset_surface(
        vertices_np,
        faces_np,
        0.0,
        max(2, round(extent / voxel_size)),
        igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL,
    )[:2]
    mesh_igl = tm.Trimesh(vertices_igl, faces_igl, process=False)

    vertices_wp, faces_wp = _upload_mesh(sphere_tm, device)
    out_vertices_wp, out_faces_wp = tw.reconstruction.resample_uniform(
        vertices_wp, faces_wp, voxel_size=voxel_size
    )
    out_tm = tm.Trimesh(
        out_vertices_wp.numpy().astype(np.float64),
        out_faces_wp.numpy().reshape(-1, 3),
        process=False,
    )

    assert mesh_igl.is_watertight
    assert out_tm.is_watertight
    assert np.isclose(np.abs(out_tm.volume), np.abs(mesh_igl.volume), rtol=0.05)

    sample_wp, _face_wp = tm.sample.sample_surface(out_tm, 4000, seed=0)
    sample_igl, _face_igl = tm.sample.sample_surface(mesh_igl, 4000, seed=1)
    assert np.abs(tm.proximity.signed_distance(mesh_igl, sample_wp)).max() < 0.25 * voxel_size
    assert np.abs(tm.proximity.signed_distance(out_tm, sample_igl)).max() < 0.25 * voxel_size


def test_resample_uniform_coarser_is_smaller(device: str) -> None:
    """A wider voxel can only produce fewer triangles, and still a closed surface."""
    sphere_tm = tm.creation.icosphere(subdivisions=3, radius=1.0)
    vertices_wp, faces_wp = _upload_mesh(sphere_tm, device)
    counts = []
    for voxel_size in (0.05, 0.1, 0.2):
        _out_vertices_wp, out_faces_wp = tw.reconstruction.resample_uniform(
            vertices_wp, faces_wp, voxel_size=voxel_size
        )
        counts.append(int(out_faces_wp.shape[0]) // 3)
        assert tw.validation.is_edge_manifold(out_faces_wp, allow_boundary_edges=False)
    assert counts == sorted(counts, reverse=True)


def test_resample_uniform_invalid(device: str) -> None:
    sphere_tm = tm.creation.icosphere(subdivisions=1, radius=1.0)
    vertices_wp, faces_wp = _upload_mesh(sphere_tm, device)
    with pytest.raises(ValueError, match="voxel_size > 0"):
        tw.reconstruction.resample_uniform(vertices_wp, faces_wp, voxel_size=0.0)


def test_resample_uniform_empty(device: str) -> None:
    vertices_wp = wp.zeros(0, dtype=wp.vec3, device=device)
    faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    out_vertices_wp, out_faces_wp = tw.reconstruction.resample_uniform(vertices_wp, faces_wp)
    assert int(out_vertices_wp.shape[0]) == 0
    assert int(out_faces_wp.shape[0]) == 0
