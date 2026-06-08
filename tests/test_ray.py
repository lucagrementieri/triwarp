"""Regression tests for ``triwarp.ray`` against trimesh."""

from __future__ import annotations

import numpy as np
import trimesh as tm
import warp as wp

import triwarp as tw


def test_contains_unit_cube(device: str):
    scale = 1.5
    mesh_tm = tm.creation.box(extents=[1.0, 1.0, 1.0])
    vertices = wp.array(np.ascontiguousarray(mesh_tm.vertices, dtype=np.float32), dtype=wp.vec3, device=device)
    faces = wp.array(np.ascontiguousarray(mesh_tm.faces.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device)
    mesh_wp = wp.Mesh(points=vertices, indices=faces)

    inside_np = mesh_tm.vertices * (1.0 / scale)
    outside_np = mesh_tm.vertices * scale
    far_np = np.random.default_rng(0).random((30, 3)) * 100.0 + 1.0 + mesh_tm.bounds[1]
    center_np = np.asarray([mesh_tm.center_mass], dtype=np.float32)

    points_wp = wp.array(np.ascontiguousarray(inside_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device)
    contains_wp = tw.ray.contains_points(mesh_wp, points_wp).numpy()
    assert contains_wp.all()
    assert np.array_equal(contains_wp, mesh_tm.contains(inside_np))

    points_wp = wp.array(np.ascontiguousarray(outside_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device)
    contains_wp = tw.ray.contains_points(mesh_wp, points_wp).numpy()
    assert not contains_wp.any()
    assert np.array_equal(contains_wp, mesh_tm.contains(outside_np))

    points_wp = wp.array(np.ascontiguousarray(far_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device)
    contains_wp = tw.ray.contains_points(mesh_wp, points_wp).numpy()
    assert not contains_wp.any()
    assert np.array_equal(contains_wp, mesh_tm.contains(far_np))

    points_wp = wp.array(np.ascontiguousarray(center_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device)
    contains_wp = tw.ray.contains_points(mesh_wp, points_wp).numpy()
    assert contains_wp.all()
    assert np.array_equal(contains_wp, mesh_tm.contains(center_np))


def test_contains_cavity(device: str):
    mesh_tm = tm.boolean.difference(
        [
            tm.creation.box(extents=[1.0, 1.0, 1.0]),
            tm.creation.box(extents=[0.1, 0.1, 0.1]),
        ]
    )
    vertices = wp.array(np.ascontiguousarray(mesh_tm.vertices, dtype=np.float32), dtype=wp.vec3, device=device)
    faces = wp.array(np.ascontiguousarray(mesh_tm.faces.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device)
    mesh_wp = wp.Mesh(points=vertices, indices=faces)
    origin_np = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

    points_wp = wp.array(np.ascontiguousarray(origin_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device)
    contains_wp = tw.ray.contains_points(mesh_wp, points_wp).numpy()
    assert not contains_wp.any()
    assert np.array_equal(contains_wp, mesh_tm.contains(origin_np))


def test_contains_icosahedron(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = icosahedron
    rng = np.random.default_rng(7)

    inside_np = mesh_tm.center_mass + rng.normal(scale=0.05, size=(50, 3))
    outside_np = rng.random((50, 3)) * 4.0 + mesh_tm.bounds[1]

    points_wp = wp.array(np.ascontiguousarray(inside_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device)
    contains_wp = tw.ray.contains_points(mesh_wp, points_wp).numpy()
    assert contains_wp.all()
    assert np.array_equal(contains_wp, mesh_tm.contains(inside_np))

    points_wp = wp.array(np.ascontiguousarray(outside_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device)
    contains_wp = tw.ray.contains_points(mesh_wp, points_wp).numpy()
    assert not contains_wp.any()
    assert np.array_equal(contains_wp, mesh_tm.contains(outside_np))


def test_contains_empty_points(device: str):
    mesh_tm = tm.creation.box(extents=[1.0, 1.0, 1.0])
    vertices = wp.array(np.ascontiguousarray(mesh_tm.vertices, dtype=np.float32), dtype=wp.vec3, device=device)
    faces = wp.array(np.ascontiguousarray(mesh_tm.faces.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device)
    mesh_wp = wp.Mesh(points=vertices, indices=faces)
    points_wp = wp.empty(0, dtype=wp.vec3, device=device)
    assert tw.ray.contains_points(mesh_wp, points_wp).numpy().shape == (0,)


def test_intersects_first_unit_cube(device: str):
    mesh_tm = tm.creation.box(extents=[1.0, 1.0, 1.0])
    vertices = wp.array(np.ascontiguousarray(mesh_tm.vertices, dtype=np.float32), dtype=wp.vec3, device=device)
    faces = wp.array(np.ascontiguousarray(mesh_tm.faces.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device)
    mesh_wp = wp.Mesh(points=vertices, indices=faces)
    rng = np.random.default_rng(0)
    n = 256
    origins_np = rng.random((n, 3)).astype(np.float32)
    directions_np = np.tile([0.0, 0.0, 1.0], (n, 1)).astype(np.float32)
    origins_np[:, 2] = -5.0

    origins_wp = wp.array(np.ascontiguousarray(origins_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device)
    directions_wp = wp.array(np.ascontiguousarray(directions_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device)
    triangle_wp = tw.ray.intersects_first(mesh_wp, origins_wp, directions_wp).numpy()
    triangle_tm = mesh_tm.ray.intersects_first(origins_np, directions_np)
    assert np.array_equal(triangle_wp, triangle_tm)


def test_intersects_first_miss(device: str):
    mesh_tm = tm.creation.icosphere()
    vertices = wp.array(np.ascontiguousarray(mesh_tm.vertices, dtype=np.float32), dtype=wp.vec3, device=device)
    faces = wp.array(np.ascontiguousarray(mesh_tm.faces.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device)
    mesh_wp = wp.Mesh(points=vertices, indices=faces)
    n = 100
    origins_np = np.random.default_rng(1).random((n, 3)).astype(np.float32)
    directions_np = np.tile([0.0, 1.0, 0.0], (n, 1)).astype(np.float32)
    origins_np[:, 2] = -5.0

    origins_wp = wp.array(np.ascontiguousarray(origins_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device)
    directions_wp = wp.array(np.ascontiguousarray(directions_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device)
    triangle_wp = tw.ray.intersects_first(mesh_wp, origins_wp, directions_wp).numpy()
    triangle_tm = mesh_tm.ray.intersects_first(origins_np, directions_np)
    assert np.array_equal(triangle_wp, triangle_tm)
    assert (triangle_wp == -1).all()


def test_intersects_first_icosahedron(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = icosahedron
    rng = np.random.default_rng(3)
    n = 200
    origins_np = (rng.random((n, 3)) - 0.5).astype(np.float32) * 5.0
    directions_np = rng.random((n, 3)).astype(np.float32) - 0.5

    origins_wp = wp.array(np.ascontiguousarray(origins_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device)
    directions_wp = wp.array(np.ascontiguousarray(directions_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device)
    triangle_wp = tw.ray.intersects_first(mesh_wp, origins_wp, directions_wp).numpy()
    triangle_tm = mesh_tm.ray.intersects_first(origins_np, directions_np)
    assert np.array_equal(triangle_wp, triangle_tm)


def test_intersects_first_empty_rays(device: str):
    mesh_tm = tm.creation.box(extents=[1.0, 1.0, 1.0])
    vertices = wp.array(np.ascontiguousarray(mesh_tm.vertices, dtype=np.float32), dtype=wp.vec3, device=device)
    faces = wp.array(np.ascontiguousarray(mesh_tm.faces.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device)
    mesh_wp = wp.Mesh(points=vertices, indices=faces)
    origins_wp = wp.empty(0, dtype=wp.vec3, device=device)
    directions_wp = wp.empty(0, dtype=wp.vec3, device=device)
    assert tw.ray.intersects_first(mesh_wp, origins_wp, directions_wp).numpy().shape == (0,)
