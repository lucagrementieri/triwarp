"""Regression tests for ``triwarp.ray`` against trimesh."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh as tm
import warp as wp

import triwarp as tw


def _assert_longest_ray_allclose(distances_wp_np: np.ndarray, distances_tm_np: np.ndarray) -> None:
    assert np.array_equal(np.isinf(distances_wp_np), np.isinf(distances_tm_np))
    finite_tm = ~np.isinf(distances_tm_np)
    if finite_tm.any():
        assert np.allclose(
            distances_wp_np[finite_tm], distances_tm_np[finite_tm], rtol=1e-5, atol=1e-5
        )


def test_contains_points(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = icosahedron
    rng = np.random.default_rng(7)

    inside_np = mesh_tm.center_mass + rng.normal(scale=0.05, size=(50, 3))
    outside_np = rng.random((50, 3)) * 4.0 + mesh_tm.bounds[1]
    far_np = rng.random((30, 3)) * 100.0 + 1.0 + mesh_tm.bounds[1]
    center_np = np.asarray([mesh_tm.center_mass], dtype=np.float32)

    points_wp = wp.array(
        np.ascontiguousarray(inside_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    contains_wp = tw.ray.contains_points(mesh_wp, points_wp).numpy()
    assert contains_wp.all()
    assert np.array_equal(contains_wp, mesh_tm.contains(inside_np))

    points_wp = wp.array(
        np.ascontiguousarray(outside_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    contains_wp = tw.ray.contains_points(mesh_wp, points_wp).numpy()
    assert not contains_wp.any()
    assert np.array_equal(contains_wp, mesh_tm.contains(outside_np))

    points_wp = wp.array(
        np.ascontiguousarray(far_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    contains_wp = tw.ray.contains_points(mesh_wp, points_wp).numpy()
    assert not contains_wp.any()
    assert np.array_equal(contains_wp, mesh_tm.contains(far_np))

    points_wp = wp.array(
        np.ascontiguousarray(center_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    contains_wp = tw.ray.contains_points(mesh_wp, points_wp).numpy()
    assert contains_wp.all()
    assert np.array_equal(contains_wp, mesh_tm.contains(center_np))


def test_contains_cavity(cave_cube: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = cave_cube
    origin_np = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

    points_wp = wp.array(
        np.ascontiguousarray(origin_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    contains_wp = tw.ray.contains_points(mesh_wp, points_wp).numpy()
    assert not contains_wp.any()
    assert np.array_equal(contains_wp, mesh_tm.contains(origin_np))


def test_contains_empty_points(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    _, mesh_wp = icosahedron
    points_wp = wp.empty(0, dtype=wp.vec3, device=mesh_wp.device)
    assert tw.ray.contains_points(mesh_wp, points_wp).numpy().shape == (0,)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere"])
def test_intersects_first(request: pytest.FixtureRequest, mesh_name: str):
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    rng = np.random.default_rng(0)
    n = 256
    origins_np = rng.random((n, 3)).astype(np.float32)
    directions_np = np.tile([0.0, 0.0, 1.0], (n, 1)).astype(np.float32)
    origins_np[:, 2] = mesh_tm.bounds[0, 2] - 5.0

    origins_wp = wp.array(
        np.ascontiguousarray(origins_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    directions_wp = wp.array(
        np.ascontiguousarray(directions_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    triangle_wp = tw.ray.intersects_first(mesh_wp, origins_wp, directions_wp).numpy()
    triangle_tm = mesh_tm.ray.intersects_first(origins_np, directions_np)
    assert np.array_equal(triangle_wp, triangle_tm)


def test_intersects_first_miss(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = icosahedron
    n = 100
    origins_np = np.random.default_rng(1).random((n, 3)).astype(np.float32)
    directions_np = np.tile([0.0, 1.0, 0.0], (n, 1)).astype(np.float32)
    origins_np[:, 2] = mesh_tm.bounds[0, 2] - 5.0

    origins_wp = wp.array(
        np.ascontiguousarray(origins_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    directions_wp = wp.array(
        np.ascontiguousarray(directions_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    triangle_wp = tw.ray.intersects_first(mesh_wp, origins_wp, directions_wp).numpy()
    triangle_tm = mesh_tm.ray.intersects_first(origins_np, directions_np)
    assert np.array_equal(triangle_wp, triangle_tm)
    assert (triangle_wp == -1).all()


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere"])
def test_intersects_any(request: pytest.FixtureRequest, mesh_name: str):
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    rng = np.random.default_rng(0)
    n = 256
    origins_np = rng.random((n, 3)).astype(np.float32)
    directions_np = np.tile([0.0, 0.0, 1.0], (n, 1)).astype(np.float32)
    origins_np[:, 2] = mesh_tm.bounds[0, 2] - 5.0

    origins_wp = wp.array(
        np.ascontiguousarray(origins_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    directions_wp = wp.array(
        np.ascontiguousarray(directions_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    hit_wp = tw.ray.intersects_any(mesh_wp, origins_wp, directions_wp).numpy()
    hit_tm = mesh_tm.ray.intersects_any(origins_np, directions_np)
    assert np.array_equal(hit_wp, hit_tm)


def test_intersects_any_miss(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = icosahedron
    n = 100
    origins_np = np.random.default_rng(1).random((n, 3)).astype(np.float32)
    directions_np = np.tile([0.0, 1.0, 0.0], (n, 1)).astype(np.float32)
    origins_np[:, 2] = mesh_tm.bounds[0, 2] - 5.0

    origins_wp = wp.array(
        np.ascontiguousarray(origins_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    directions_wp = wp.array(
        np.ascontiguousarray(directions_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    hit_wp = tw.ray.intersects_any(mesh_wp, origins_wp, directions_wp).numpy()
    hit_tm = mesh_tm.ray.intersects_any(origins_np, directions_np)
    assert np.array_equal(hit_wp, hit_tm)
    assert not hit_wp.any()


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere"])
def test_intersects_location(request: pytest.FixtureRequest, mesh_name: str):
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    rng = np.random.default_rng(0)
    n = 256
    origins_np = rng.random((n, 3)).astype(np.float32)
    directions_np = np.tile([0.0, 0.0, 1.0], (n, 1)).astype(np.float32)
    origins_np[:, 2] = mesh_tm.bounds[0, 2] - 5.0

    origins_wp = wp.array(
        np.ascontiguousarray(origins_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    directions_wp = wp.array(
        np.ascontiguousarray(directions_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    _loc_wp, ray_wp, tri_wp = tw.ray.intersects_location(mesh_wp, origins_wp, directions_wp)
    tri_tm, ray_tm = mesh_tm.ray.intersects_id(origins_np, directions_np, multiple_hits=False)
    tri_wp_np = tri_wp.numpy()
    ray_wp_np = ray_wp.numpy()
    order_wp = np.lexsort((tri_wp_np, ray_wp_np))
    order_tm = np.lexsort((tri_tm, ray_tm))
    assert np.array_equal(tri_wp_np[order_wp], tri_tm[order_tm])
    assert np.array_equal(ray_wp_np[order_wp], ray_tm[order_tm])


def test_intersects_location_miss(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = icosahedron
    n = 100
    origins_np = np.random.default_rng(1).random((n, 3)).astype(np.float32)
    directions_np = np.tile([0.0, 1.0, 0.0], (n, 1)).astype(np.float32)
    origins_np[:, 2] = mesh_tm.bounds[0, 2] - 5.0

    origins_wp = wp.array(
        np.ascontiguousarray(origins_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    directions_wp = wp.array(
        np.ascontiguousarray(directions_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    loc_wp, ray_wp, tri_wp = tw.ray.intersects_location(mesh_wp, origins_wp, directions_wp)
    assert loc_wp.shape[0] == 0
    assert tri_wp.shape[0] == 0
    assert ray_wp.shape[0] == 0


def test_intersects_location_cave_cube(cave_cube: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = cave_cube
    origins_np = tm.util.grid_linspace(
        mesh_tm.bounds[:, :2] + np.reshape([-0.02, 0.02], (-1, 1)), 100
    )
    origins_np = np.column_stack((origins_np, np.ones(len(origins_np)) * -100.0)).astype(np.float32)
    directions_np = np.ones((len(origins_np), 3), dtype=np.float32) * [0.0, 0.0, 1.0]

    origins_wp = wp.array(
        np.ascontiguousarray(origins_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    directions_wp = wp.array(
        np.ascontiguousarray(directions_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    loc_wp, ray_wp, _tri_wp = tw.ray.intersects_location(mesh_wp, origins_wp, directions_wp)
    loc_tm, ray_tm, _tri_tm = mesh_tm.ray.intersects_location(
        origins_np, directions_np, multiple_hits=False
    )

    ray_wp_np = ray_wp.numpy()
    loc_wp_np = loc_wp.numpy()
    order_wp = np.argsort(ray_wp_np)
    order_tm = np.argsort(ray_tm)
    assert np.array_equal(ray_wp_np[order_wp], ray_tm[order_tm])
    assert np.allclose(loc_wp_np[order_wp], loc_tm[order_tm], rtol=1e-5, atol=1e-4)
    for p, r in zip(loc_wp_np[order_wp], ray_wp_np[order_wp], strict=False):
        assert np.allclose(p[:2], origins_np[r][:2], rtol=1e-5, atol=1e-5)
        assert np.isclose(p[2], mesh_tm.bounds[0, 2], atol=1e-4)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere"])
def test_longest_ray(request: pytest.FixtureRequest, mesh_name: str):
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    rng = np.random.default_rng(3)
    n = 128
    origins_np = (rng.random((n, 3)).astype(np.float32) * 2.0 + mesh_tm.bounds[0]).astype(
        np.float32
    )
    directions_np = rng.normal(size=(n, 3)).astype(np.float32)

    origins_wp = wp.array(
        np.ascontiguousarray(origins_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    directions_wp = wp.array(
        np.ascontiguousarray(directions_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    distances_wp_np = tw.ray.longest_ray(mesh_wp, origins_wp, directions_wp).numpy()
    distances_tm_np = tm.proximity.longest_ray(mesh_tm, origins_np, directions_np)
    _assert_longest_ray_allclose(distances_wp_np, distances_tm_np)


def test_longest_ray_surface_normals(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = icosahedron
    rng = np.random.default_rng(11)
    n = 64
    query_np = rng.random((n, 3)).astype(np.float64)
    closest_np, _distance_np, triangle_id_np = tm.proximity.closest_point(mesh_tm, query_np)
    normals_np = mesh_tm.face_normals[triangle_id_np].astype(np.float32)

    origins_wp = wp.array(
        np.ascontiguousarray(closest_np.astype(np.float32), dtype=np.float32),
        dtype=wp.vec3,
        device=mesh_wp.device,
    )
    directions_wp = wp.array(
        np.ascontiguousarray(normals_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    distances_wp_np = tw.ray.longest_ray(mesh_wp, origins_wp, directions_wp).numpy()
    distances_tm_np = tm.proximity.longest_ray(mesh_tm, closest_np, normals_np)
    _assert_longest_ray_allclose(distances_wp_np, distances_tm_np)


def test_longest_ray_miss(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = icosahedron
    n = 100
    origins_np = np.random.default_rng(1).random((n, 3)).astype(np.float32)
    directions_np = np.tile([0.0, 1.0, 0.0], (n, 1)).astype(np.float32)
    origins_np[:, 2] = mesh_tm.bounds[0, 2] - 5.0

    origins_wp = wp.array(
        np.ascontiguousarray(origins_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    directions_wp = wp.array(
        np.ascontiguousarray(directions_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    distances_wp_np = tw.ray.longest_ray(mesh_wp, origins_wp, directions_wp).numpy()
    distances_tm_np = tm.proximity.longest_ray(mesh_tm, origins_np, directions_np)
    _assert_longest_ray_allclose(distances_wp_np, distances_tm_np)
    assert np.isinf(distances_wp_np).all()


def test_intersects_empty_rays(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    _, mesh_wp = icosahedron
    origins_wp = wp.empty(0, dtype=wp.vec3, device=mesh_wp.device)
    directions_wp = wp.empty(0, dtype=wp.vec3, device=mesh_wp.device)
    assert tw.ray.intersects_first(mesh_wp, origins_wp, directions_wp).numpy().shape == (0,)
    assert tw.ray.intersects_any(mesh_wp, origins_wp, directions_wp).numpy().shape == (0,)
    loc_wp, ray_wp, tri_wp = tw.ray.intersects_location(mesh_wp, origins_wp, directions_wp)
    assert loc_wp.shape[0] == 0
    assert tri_wp.shape[0] == 0
    assert ray_wp.shape[0] == 0
    assert tw.ray.longest_ray(mesh_wp, origins_wp, directions_wp).numpy().shape == (0,)
