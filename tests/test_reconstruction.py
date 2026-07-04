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
    """Reference reconstruction via MeshLib; returns ``(vertices, faces)`` numpy arrays."""
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
    assert tw.characteristics.is_watertight(vertices_wp, faces_wp)
    assert tw.characteristics.is_edge_manifold(faces_wp)
    assert tw.characteristics.euler_characteristic(faces_wp) == 2
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
    assert tw.characteristics.is_watertight(vertices_wp, faces_wp)
    assert tw.characteristics.is_edge_manifold(faces_wp)
    # genus-1 closed surface: V - E + F = 0
    assert tw.characteristics.euler_characteristic(faces_wp) == 0


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
    assert tw.characteristics.is_edge_manifold(faces_wp)
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
    assert tw.characteristics.is_edge_manifold(faces_wp)
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
    assert not tw.characteristics.is_watertight(vertices_open, faces_open)

    # Large threshold: the hole is sealed into a watertight, manifold mesh.
    vertices_filled, faces_filled = tw.reconstruction.triangulate_point_cloud(
        points_wp, normals_wp, num_neighbours=18, crit_hole_length=10.0
    )
    assert tw.characteristics.is_edge_manifold(faces_filled)
    assert tw.characteristics.is_watertight(vertices_filled, faces_filled)


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
    with pytest.raises(ValueError):
        tw.reconstruction.triangulate_point_cloud(points_wp, num_neighbours=8, radius=1.0)
    with pytest.raises(ValueError):
        tw.reconstruction.triangulate_point_cloud(
            points_wp, max_neighbours=tw.kernels.reconstruction.MAX_NEIGHBOURS + 1
        )
