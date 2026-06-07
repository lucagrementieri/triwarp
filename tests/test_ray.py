"""Regression tests for ``triwarp.ray`` against trimesh."""

from __future__ import annotations

import numpy as np
import trimesh as tm
import warp as wp

import triwarp as tw


def _trimesh_to_warp(mesh: tm.Trimesh, device: str) -> wp.Mesh:
    vertices = wp.array(np.ascontiguousarray(mesh.vertices, dtype=np.float32), dtype=wp.vec3, device=device)
    faces = wp.array(np.ascontiguousarray(mesh.faces.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device)
    return wp.Mesh(points=vertices, indices=faces)


def _contains_points_wp(mesh: wp.Mesh, points_np: np.ndarray) -> np.ndarray:
    points_wp = wp.array(np.ascontiguousarray(points_np, dtype=np.float32), dtype=wp.vec3, device=mesh.device)
    return tw.ray.contains_points(mesh, points_wp).numpy()


def _assert_matches_trimesh(mesh_tm: tm.Trimesh, contains_wp: np.ndarray, points_np: np.ndarray) -> None:
    assert np.array_equal(contains_wp, mesh_tm.contains(points_np))


def _rays_to_warp(
    origins_np: np.ndarray,
    directions_np: np.ndarray,
    device: str,
) -> tuple[wp.array[wp.vec3], wp.array[wp.vec3]]:
    origins_wp = wp.array(np.ascontiguousarray(origins_np, dtype=np.float32), dtype=wp.vec3, device=device)
    directions_wp = wp.array(np.ascontiguousarray(directions_np, dtype=np.float32), dtype=wp.vec3, device=device)
    return origins_wp, directions_wp


def _intersects_first_wp(
    mesh: wp.Mesh,
    origins_np: np.ndarray,
    directions_np: np.ndarray,
) -> np.ndarray:
    origins_wp, directions_wp = _rays_to_warp(origins_np, directions_np, mesh.device)
    return tw.ray.intersects_first(mesh, origins_wp, directions_wp).numpy()


def test_contains_unit_cube(device: str):
    scale = 1.5
    mesh_tm = tm.creation.box(extents=[1.0, 1.0, 1.0])
    mesh_wp = _trimesh_to_warp(mesh_tm, device)

    inside_np = mesh_tm.vertices * (1.0 / scale)
    outside_np = mesh_tm.vertices * scale
    far_np = np.random.default_rng(0).random((30, 3)) * 100.0 + 1.0 + mesh_tm.bounds[1]
    center_np = np.asarray([mesh_tm.center_mass], dtype=np.float32)

    contains_wp = _contains_points_wp(mesh_wp, inside_np)
    assert contains_wp.all()
    _assert_matches_trimesh(mesh_tm, contains_wp, inside_np)

    contains_wp = _contains_points_wp(mesh_wp, outside_np)
    assert not contains_wp.any()
    _assert_matches_trimesh(mesh_tm, contains_wp, outside_np)

    contains_wp = _contains_points_wp(mesh_wp, far_np)
    assert not contains_wp.any()
    _assert_matches_trimesh(mesh_tm, contains_wp, far_np)

    contains_wp = _contains_points_wp(mesh_wp, center_np)
    assert contains_wp.all()
    _assert_matches_trimesh(mesh_tm, contains_wp, center_np)


def test_contains_cavity(device: str):
    mesh_tm = tm.boolean.difference(
        [
            tm.creation.box(extents=[1.0, 1.0, 1.0]),
            tm.creation.box(extents=[0.1, 0.1, 0.1]),
        ]
    )
    mesh_wp = _trimesh_to_warp(mesh_tm, device)
    origin_np = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

    contains_wp = _contains_points_wp(mesh_wp, origin_np)
    assert not contains_wp.any()
    _assert_matches_trimesh(mesh_tm, contains_wp, origin_np)


def test_contains_icosahedron(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = icosahedron
    rng = np.random.default_rng(7)

    inside_np = mesh_tm.center_mass + rng.normal(scale=0.05, size=(50, 3))
    outside_np = rng.random((50, 3)) * 4.0 + mesh_tm.bounds[1]

    contains_wp = _contains_points_wp(mesh_wp, inside_np)
    assert contains_wp.all()
    _assert_matches_trimesh(mesh_tm, contains_wp, inside_np)

    contains_wp = _contains_points_wp(mesh_wp, outside_np)
    assert not contains_wp.any()
    _assert_matches_trimesh(mesh_tm, contains_wp, outside_np)


def test_contains_empty_points(device: str):
    mesh_tm = tm.creation.box(extents=[1.0, 1.0, 1.0])
    mesh_wp = _trimesh_to_warp(mesh_tm, device)
    points_wp = wp.empty(0, dtype=wp.vec3, device=device)
    assert tw.ray.contains_points(mesh_wp, points_wp).numpy().shape == (0,)


def test_intersects_first_unit_cube(device: str):
    mesh_tm = tm.creation.box(extents=[1.0, 1.0, 1.0])
    mesh_wp = _trimesh_to_warp(mesh_tm, device)
    rng = np.random.default_rng(0)
    n = 256
    origins_np = rng.random((n, 3)).astype(np.float32)
    directions_np = np.tile([0.0, 0.0, 1.0], (n, 1)).astype(np.float32)
    origins_np[:, 2] = -5.0

    triangle_wp = _intersects_first_wp(mesh_wp, origins_np, directions_np)
    triangle_tm = mesh_tm.ray.intersects_first(origins_np, directions_np)
    assert np.array_equal(triangle_wp, triangle_tm)


def test_intersects_first_miss(device: str):
    mesh_tm = tm.creation.icosphere()
    mesh_wp = _trimesh_to_warp(mesh_tm, device)
    n = 100
    origins_np = np.random.default_rng(1).random((n, 3)).astype(np.float32)
    directions_np = np.tile([0.0, 1.0, 0.0], (n, 1)).astype(np.float32)
    origins_np[:, 2] = -5.0

    triangle_wp = _intersects_first_wp(mesh_wp, origins_np, directions_np)
    triangle_tm = mesh_tm.ray.intersects_first(origins_np, directions_np)
    assert np.array_equal(triangle_wp, triangle_tm)
    assert (triangle_wp == -1).all()


def test_intersects_first_icosahedron(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = icosahedron
    rng = np.random.default_rng(3)
    n = 200
    origins_np = (rng.random((n, 3)) - 0.5).astype(np.float32) * 5.0
    directions_np = rng.random((n, 3)).astype(np.float32) - 0.5

    triangle_wp = _intersects_first_wp(mesh_wp, origins_np, directions_np)
    triangle_tm = mesh_tm.ray.intersects_first(origins_np, directions_np)
    assert np.array_equal(triangle_wp, triangle_tm)


def test_intersects_first_empty_rays(device: str):
    mesh_tm = tm.creation.box(extents=[1.0, 1.0, 1.0])
    mesh_wp = _trimesh_to_warp(mesh_tm, device)
    origins_wp = wp.empty(0, dtype=wp.vec3, device=device)
    directions_wp = wp.empty(0, dtype=wp.vec3, device=device)
    assert tw.ray.intersects_first(mesh_wp, origins_wp, directions_wp).numpy().shape == (0,)
