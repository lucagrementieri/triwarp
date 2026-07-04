"""
Regression tests for ``triwarp.proximity`` ball / k-nearest / AABB query APIs.

Against SciPy ``KDTree`` or brute-force reference (BVH and HashGrid backends).
Closest-on-mesh tests compare against ``trimesh.proximity.closest_point``.
"""

from __future__ import annotations

import math
from typing import Literal

import igl
import numpy as np
import pytest
import trimesh as tm
import trimesh.proximity as tm_proximity
import warp as wp
from scipy.spatial import KDTree

import triwarp as tw
from triwarp.constants import TOLERANCE_MERGE


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
def test_query_ball_single(device: str, backend: Literal["bvh", "hashgrid"]):
    rng = np.random.default_rng(0)
    points = rng.random((50, 3), dtype=np.float32) * 2.0
    kdtree = KDTree(points)
    query = points[0].copy()
    radius = 0.5

    query_indices_np = np.asarray(kdtree.query_ball_point(query, radius, return_sorted=False))
    query_distances_np = np.linalg.norm(points[query_indices_np] - query, axis=1)
    order = np.argsort(query_distances_np)
    query_indices_np = query_indices_np[order]
    query_distances_np = query_distances_np[order]

    points_wp = wp.array(np.ascontiguousarray(points), dtype=wp.vec3, device=device)
    query_wp = wp.vec3(query[0], query[1], query[2])
    query_ball = (
        tw.proximity.query_bvh_ball if backend == "bvh" else tw.proximity.query_hashgrid_ball
    )

    query_indices_wp, query_distances_wp = query_ball(
        points_wp, query_wp, radius, return_sorted=True
    )

    assert np.array_equal(query_indices_wp.numpy(), query_indices_np)
    assert np.allclose(query_distances_wp.numpy(), query_distances_np, rtol=1e-5, atol=1e-5)

    query_indices_unsorted_wp, query_distances_unsorted_wp = query_ball(
        points_wp, query_wp, radius, return_sorted=False
    )
    assert np.array_equal(
        np.sort(query_indices_unsorted_wp.numpy()), np.sort(query_indices_wp.numpy())
    )
    assert np.allclose(
        np.sort(query_distances_unsorted_wp.numpy()),
        np.sort(query_distances_wp.numpy()),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
def test_query_ball_empty_ball(device: str, backend: Literal["bvh", "hashgrid"]):
    rng = np.random.default_rng(0)
    points = rng.random((50, 3), dtype=np.float32) * 2.0
    query = np.array([10.0, 10.0, 10.0], dtype=np.float32)
    radius = 0.5

    points_wp = wp.array(np.ascontiguousarray(points), dtype=wp.vec3, device=device)
    query_wp = wp.vec3(query[0], query[1], query[2])
    query_ball = (
        tw.proximity.query_bvh_ball if backend == "bvh" else tw.proximity.query_hashgrid_ball
    )

    query_indices_wp, query_distances_wp = query_ball(
        points_wp, query_wp, radius, return_sorted=True
    )
    assert query_indices_wp.shape == (0,)
    assert query_distances_wp.shape == (0,)


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
def test_query_ball_batch(device: str, backend: Literal["bvh", "hashgrid"]):
    rng = np.random.default_rng(1)
    points = rng.random((50, 3), dtype=np.float32) * 3.0
    kdtree = KDTree(points)
    queries = points[[10, 20, 30]].copy()
    radius = 0.5

    query_indices_np = [
        np.asarray(indices)
        for indices in kdtree.query_ball_point(queries, radius, return_sorted=False)
    ]
    query_distances_np = [
        np.linalg.norm(points[indices] - query, axis=1)
        for indices, query in zip(query_indices_np, queries, strict=False)
    ]
    orders = [np.argsort(distances) for distances in query_distances_np]
    query_indices_np = [
        indices[order] for indices, order in zip(query_indices_np, orders, strict=False)
    ]
    query_distances_np = [
        distances[order] for distances, order in zip(query_distances_np, orders, strict=False)
    ]

    points_wp = wp.array(np.ascontiguousarray(points), dtype=wp.vec3, device=device)
    query_wp = wp.array(np.ascontiguousarray(queries), dtype=wp.vec3, device=device)
    query_ball = (
        tw.proximity.query_bvh_ball if backend == "bvh" else tw.proximity.query_hashgrid_ball
    )

    query_indices_wp, query_distances_wp = query_ball(
        points_wp, query_wp, radius, return_sorted=True
    )

    for single_query_indices_wp, single_query_indices_np in zip(
        query_indices_wp, query_indices_np, strict=False
    ):
        assert np.array_equal(single_query_indices_wp.numpy(), single_query_indices_np)
    for single_query_distances_wp, single_query_distances_np in zip(
        query_distances_wp, query_distances_np, strict=False
    ):
        assert np.allclose(
            single_query_distances_wp.numpy(), single_query_distances_np, rtol=1e-5, atol=1e-5
        )

    query_indices_unsorted_wp, query_distances_unsorted_wp = query_ball(
        points_wp, query_wp, radius, return_sorted=False
    )
    for single_query_indices_unsorted_wp, single_query_indices_wp in zip(
        query_indices_unsorted_wp, query_indices_wp, strict=False
    ):
        assert np.array_equal(
            np.sort(single_query_indices_unsorted_wp.numpy()),
            np.sort(single_query_indices_wp.numpy()),
        )
    for single_query_distances_unsorted_wp, single_query_distances_wp in zip(
        query_distances_unsorted_wp, query_distances_wp, strict=False
    ):
        assert np.allclose(
            np.sort(single_query_distances_unsorted_wp.numpy()),
            np.sort(single_query_distances_wp.numpy()),
            rtol=1e-5,
            atol=1e-5,
        )


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
def test_query_ball_empty(device: str, backend: Literal["bvh", "hashgrid"]):
    rng = np.random.default_rng(0)
    points = rng.random((10, 3), dtype=np.float32)

    points_wp = wp.array(np.ascontiguousarray(points), dtype=wp.vec3, device=device)
    empty_points_wp = wp.empty(0, dtype=wp.vec3, device=device)
    empty_queries_wp = wp.empty(0, dtype=wp.vec3, device=device)
    query_wp = wp.vec3(points[0][0], points[0][1], points[0][2])
    queries_wp = wp.array(np.ascontiguousarray(points[-3:]), dtype=wp.vec3, device=device)
    radius = 0.5
    query_ball = (
        tw.proximity.query_bvh_ball if backend == "bvh" else tw.proximity.query_hashgrid_ball
    )

    indices, distances = query_ball(empty_points_wp, query_wp, radius)
    assert indices.shape == (0,)
    assert distances.shape == (0,)

    indices, distances = query_ball(empty_points_wp, queries_wp, radius)
    assert len(indices) == queries_wp.shape[0]
    assert len(distances) == queries_wp.shape[0]
    for single_indices, single_distances in zip(indices, distances, strict=False):
        assert single_indices.shape == (0,)
        assert single_distances.shape == (0,)

    indices, distances = query_ball(points_wp, empty_queries_wp, radius)
    assert len(indices) == 0
    assert len(distances) == 0


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
@pytest.mark.parametrize("k", [1, 3, 40])
@pytest.mark.parametrize("max_radius", [math.inf, 0.5, 1.0])
def test_query_nearest_single(
    device: str, backend: Literal["bvh", "hashgrid"], k: int, max_radius: float
):
    rng = np.random.default_rng(0)
    points = rng.random((50, 3), dtype=np.float32) * 4.0
    kdtree = KDTree(points)
    query = points[0].copy()

    points_wp = wp.array(np.ascontiguousarray(points), dtype=wp.vec3, device=device)
    query_wp = wp.vec3(query[0], query[1], query[2])
    query_distances_np, query_indices_np = kdtree.query(query, k=k, distance_upper_bound=max_radius)
    query_indices_np = np.atleast_1d(np.asarray(query_indices_np))
    query_indices_np[query_indices_np == len(points)] = -1
    query_distances_np = np.atleast_1d(np.asarray(query_distances_np))

    query_nearest = (
        tw.proximity.query_bvh_nearest if backend == "bvh" else tw.proximity.query_hashgrid_nearest
    )
    query_indices_wp, query_distances_wp = query_nearest(
        points_wp, query_wp, k=k, max_radius=max_radius
    )

    assert np.array_equal(query_indices_wp.numpy(), query_indices_np)
    assert np.allclose(query_distances_wp.numpy(), query_distances_np, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
@pytest.mark.parametrize("k", [1, 3, 10])
@pytest.mark.parametrize("max_radius", [math.inf, 0.5, 1.0])
def test_query_nearest_batch(
    device: str, backend: Literal["bvh", "hashgrid"], k: int, max_radius: float
):
    rng = np.random.default_rng(0)
    points = rng.random((50, 3), dtype=np.float32) * 2.0
    kdtree = KDTree(points)
    queries = points[[10, 20, 30]] + 0.5

    points_wp = wp.array(np.ascontiguousarray(points), dtype=wp.vec3, device=device)
    query_wp = wp.array(np.ascontiguousarray(queries), dtype=wp.vec3, device=device)

    query_distances_np, query_indices_np = kdtree.query(
        queries, k=k, distance_upper_bound=max_radius
    )
    query_indices_np = np.asarray(query_indices_np)
    query_indices_np[query_indices_np == len(points)] = -1
    query_distances_np = np.asarray(query_distances_np)

    query_nearest = (
        tw.proximity.query_bvh_nearest if backend == "bvh" else tw.proximity.query_hashgrid_nearest
    )
    query_indices_wp, query_distances_wp = query_nearest(
        points_wp, query_wp, k=k, max_radius=max_radius
    )

    assert np.array_equal(query_indices_wp.numpy(), query_indices_np)
    assert np.allclose(query_distances_wp.numpy(), query_distances_np, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("backend", ["bvh", "hashgrid"])
def test_query_nearest_empty(device: str, backend: Literal["bvh", "hashgrid"]):
    rng = np.random.default_rng(0)
    points = rng.random((10, 3), dtype=np.float32)

    points_wp = wp.array(np.ascontiguousarray(points), dtype=wp.vec3, device=device)
    empty_points_wp = wp.empty(0, dtype=wp.vec3, device=device)
    empty_queries_wp = wp.empty(0, dtype=wp.vec3, device=device)
    query_wp = wp.vec3(points[0][0], points[0][1], points[0][2])
    queries_wp = wp.array(np.ascontiguousarray(points[-3:]), dtype=wp.vec3, device=device)
    k = 2
    query_nearest = (
        tw.proximity.query_bvh_nearest if backend == "bvh" else tw.proximity.query_hashgrid_nearest
    )

    indices, distances = query_nearest(empty_points_wp, query_wp, k=k)
    assert np.array_equal(indices.numpy(), -np.ones(k))
    assert np.array_equal(distances.numpy(), np.full(k, np.inf))

    indices, distances = query_nearest(empty_points_wp, queries_wp, k=k)
    assert indices.shape == (queries_wp.shape[0], k)
    assert distances.shape == (queries_wp.shape[0], k)

    indices, distances = query_nearest(points_wp, empty_queries_wp, k=k)
    assert indices.shape == (0, k)
    assert distances.shape == (0, k)


def test_query_mesh_aabb_bounds_with_offsets(device: str) -> None:
    rng = np.random.default_rng(11)
    n_faces = 8
    vertices_np = rng.random((n_faces * 3, 3), dtype=np.float32)
    faces_np = np.arange(n_faces * 3, dtype=np.int32).reshape(n_faces, 3)

    lower_np = np.empty((n_faces, 3), dtype=np.float32)
    upper_np = np.empty((n_faces, 3), dtype=np.float32)
    for face_idx in range(n_faces):
        tri = vertices_np[faces_np[face_idx]]
        lower_np[face_idx] = tri.min(axis=0)
        upper_np[face_idx] = tri.max(axis=0)

    query_lower_np = lower_np[:4].copy()
    query_upper_np = upper_np[:4].copy()

    vertices_wp = wp.array(np.ascontiguousarray(vertices_np), dtype=wp.vec3, device=device)
    faces_wp = wp.array(np.ascontiguousarray(faces_np.reshape(-1)), dtype=wp.int32, device=device)
    mesh = wp.Mesh(points=vertices_wp, indices=faces_wp)
    query_lower_wp = wp.array(np.ascontiguousarray(query_lower_np), dtype=wp.vec3, device=device)
    query_upper_wp = wp.array(np.ascontiguousarray(query_upper_np), dtype=wp.vec3, device=device)

    indices_wp, offsets_wp, hit_counts_wp = tw.proximity.query_mesh_aabb_bounds_with_offsets(
        mesh, query_lower_wp, query_upper_wp, max_hits=16
    )

    indices_np = indices_wp.numpy()
    offsets_np = offsets_wp.numpy()
    hit_counts_np = hit_counts_wp.numpy()
    bounds_np = np.append(offsets_np, indices_np.shape[0])

    for query_idx in range(query_lower_np.shape[0]):
        q_lower = query_lower_np[query_idx]
        q_upper = query_upper_np[query_idx]
        mask_np = np.all(lower_np <= q_upper, axis=1) & np.all(upper_np >= q_lower, axis=1)
        expected_np = np.sort(np.flatnonzero(mask_np).astype(np.int32))
        got_np = np.sort(indices_np[bounds_np[query_idx] : bounds_np[query_idx + 1]])
        assert hit_counts_np[query_idx] == got_np.shape[0]
        assert np.array_equal(got_np, expected_np)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere"])
def test_closest_point_on_mesh_random(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    rng = np.random.default_rng(42)
    points_np = rng.random((200, 3), dtype=np.float64) * 4.0 - 2.0

    closest_tm, distance_tm, _triangle_id_tm = tm.proximity.closest_point(mesh_tm, points_np)

    points_wp = wp.array(
        np.ascontiguousarray(points_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    closest_wp, distance_wp, _triangle_id_wp = tw.proximity.closest_point_on_mesh(
        mesh_wp.points, mesh_wp.indices, points_wp
    )

    assert np.allclose(closest_wp.numpy(), closest_tm, rtol=1e-5, atol=1e-5)
    assert np.allclose(distance_wp.numpy(), distance_tm, rtol=1e-5, atol=1e-5)


def test_closest_point_on_mesh_ambiguous_edge(device: str) -> None:
    mesh_tm = tm.Trimesh(
        vertices=[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        faces=[[0, 1, 2], [0, 1, 3]],
        process=False,
    )
    query_np = np.array([[-0.25 - 1e-9, 0.0, -0.25]], dtype=np.float64)
    closest_tm, distance_tm, _triangle_id_tm = tm.proximity.closest_point(mesh_tm, query_np)

    vertices_wp = wp.array(
        np.ascontiguousarray(mesh_tm.vertices, dtype=np.float32), dtype=wp.vec3, device=device
    )
    faces_wp = wp.array(
        np.ascontiguousarray(mesh_tm.faces.reshape(-1), dtype=np.int32), device=device
    )
    query_wp = wp.array(
        np.ascontiguousarray(query_np, dtype=np.float32), dtype=wp.vec3, device=device
    )
    closest_wp, distance_wp, _triangle_id_wp = tw.proximity.closest_point_on_mesh(
        vertices_wp, faces_wp, query_wp
    )

    assert np.allclose(closest_wp.numpy(), closest_tm, rtol=1e-5, atol=1e-5)
    assert np.allclose(distance_wp.numpy(), distance_tm, rtol=1e-5, atol=1e-5)


def test_closest_point_on_mesh_unreferenced_vertex(device: str) -> None:
    query_np = np.array([[-1.0, -1.0, -1.0]], dtype=np.float64)
    mesh_tm = tm.Trimesh(
        vertices=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-0.5, -0.5, -0.5]],
        faces=[[0, 1, 2]],
        process=False,
    )
    closest_tm, distance_tm, triangle_id_tm = tm.proximity.closest_point(mesh_tm, query_np)

    vertices_wp = wp.array(
        np.ascontiguousarray(mesh_tm.vertices, dtype=np.float32), dtype=wp.vec3, device=device
    )
    faces_wp = wp.array(
        np.ascontiguousarray(mesh_tm.faces.reshape(-1), dtype=np.int32), device=device
    )
    query_wp = wp.array(
        np.ascontiguousarray(query_np, dtype=np.float32), dtype=wp.vec3, device=device
    )
    closest_wp, distance_wp, triangle_id_wp = tw.proximity.closest_point_on_mesh(
        vertices_wp, faces_wp, query_wp
    )

    assert np.allclose(closest_wp.numpy(), closest_tm, rtol=1e-5, atol=1e-5)
    assert np.allclose(distance_wp.numpy(), distance_tm, rtol=1e-5, atol=1e-5)
    assert np.array_equal(triangle_id_wp.numpy(), triangle_id_tm)


def test_closest_point_on_mesh_empty_points(device: str) -> None:
    vertices = wp.array(np.zeros((3, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    faces = wp.array([0, 1, 2], dtype=wp.int32, device=device)
    points = wp.empty(0, dtype=wp.vec3, device=device)
    closest_wp, distance_wp, triangle_id_wp = tw.proximity.closest_point_on_mesh(
        vertices, faces, points
    )
    assert closest_wp.shape == (0,)
    assert distance_wp.shape == (0,)
    assert triangle_id_wp.shape == (0,)


def test_closest_point_on_mesh_empty_faces(device: str) -> None:
    vertices = wp.zeros(1, dtype=wp.vec3, device=device)
    faces = wp.empty(0, dtype=wp.int32, device=device)
    points = wp.array(np.zeros((2, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    closest_wp, distance_wp, triangle_id_wp = tw.proximity.closest_point_on_mesh(
        vertices, faces, points
    )
    assert closest_wp.shape == (2,)
    assert distance_wp.shape == (2,)
    assert triangle_id_wp.shape == (2,)
    assert np.all(np.isnan(closest_wp.numpy()))
    assert np.all(np.isinf(distance_wp.numpy()))
    assert np.all(triangle_id_wp.numpy() == -1)


def test_normals_at_closest_faces(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = icosahedron
    rng = np.random.default_rng(19)
    query_np = rng.random((32, 3)).astype(np.float64)

    query_wp = wp.array(
        np.ascontiguousarray(query_np.astype(np.float32)), dtype=wp.vec3, device=mesh_wp.device
    )
    normals_wp = tw.proximity.normals_at_closest_faces(mesh_wp, query_wp).numpy()

    _, _, triangle_id_wp = tw.proximity.closest_point_on_mesh(
        mesh_wp.points, mesh_wp.indices, query_wp
    )
    all_normals_wp, _ = tw.triangles.face_normals_and_areas(mesh_wp.points, mesh_wp.indices)
    expected_normals_np = all_normals_wp.numpy()[triangle_id_wp.numpy()]
    assert np.allclose(normals_wp, expected_normals_np, rtol=1e-5, atol=1e-5)


def test_normals_at_closest_faces_surface(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = icosahedron
    points_np, face_ids_np = tm.sample.sample_surface(mesh_tm, 24, seed=3)
    expected_normals_np = mesh_tm.face_normals[face_ids_np].astype(np.float32)

    points_wp = wp.array(
        np.ascontiguousarray(points_np.astype(np.float32)), dtype=wp.vec3, device=mesh_wp.device
    )
    normals_wp = tw.proximity.normals_at_closest_faces(mesh_wp, points_wp).numpy()
    assert np.allclose(normals_wp, expected_normals_np, rtol=1e-5, atol=1e-5)


def test_normals_at_closest_faces_empty(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = icosahedron
    points_wp = wp.empty(0, dtype=wp.vec3, device=mesh_wp.device)
    normals_wp = tw.proximity.normals_at_closest_faces(mesh_wp, points_wp)
    assert normals_wp.shape == (0,)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube"])
def test_signed_distance_on_mesh_random(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    rng = np.random.default_rng(42)
    points_np = rng.random((200, 3), dtype=np.float64) * 4.0 - 2.0

    expected_np = -tm_proximity.signed_distance(mesh_tm, points_np)
    points_wp = wp.array(
        np.ascontiguousarray(points_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    signed_wp = tw.proximity.signed_distance_on_mesh(mesh_wp.points, mesh_wp.indices, points_wp)
    assert np.allclose(signed_wp.numpy(), expected_np, rtol=1e-5, atol=1e-5)


def test_signed_distance_on_mesh_sign_direction(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = icosahedron
    outside_np = np.asarray([mesh_tm.bounds[0] + [100.0, 100.0, 100.0]], dtype=np.float32)
    inside_np = np.asarray([mesh_tm.center_mass], dtype=np.float32)
    outside_wp = wp.array(outside_np, dtype=wp.vec3, device=mesh_wp.device)
    inside_wp = wp.array(inside_np, dtype=wp.vec3, device=mesh_wp.device)
    outside_signed_wp = tw.proximity.signed_distance_on_mesh(
        mesh_wp.points, mesh_wp.indices, outside_wp
    )
    inside_signed_wp = tw.proximity.signed_distance_on_mesh(
        mesh_wp.points, mesh_wp.indices, inside_wp
    )
    assert (outside_signed_wp.numpy() > 0.0).all()
    assert (inside_signed_wp.numpy() < 0.0).all()


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube"])
def test_signed_distance_on_mesh_coplanar(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    outside_np = np.asarray([mesh_tm.bounds[0] + [100.0, 0.0, 0.0]], dtype=np.float32)
    outside_wp = wp.array(outside_np, dtype=wp.vec3, device=mesh_wp.device)
    outside_signed_wp = tw.proximity.signed_distance_on_mesh(
        mesh_wp.points, mesh_wp.indices, outside_wp
    )
    assert (outside_signed_wp.numpy() > 0.0).all()


def test_signed_distance_on_mesh_on_surface(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = icosahedron
    surface_np, _face_idx = tm.sample.sample_surface(mesh_tm, 50)
    surface_wp = wp.array(
        np.ascontiguousarray(surface_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    signed_wp = tw.proximity.signed_distance_on_mesh(mesh_wp.points, mesh_wp.indices, surface_wp)
    signed_np = signed_wp.numpy()
    assert (np.abs(signed_np) <= max(TOLERANCE_MERGE, 1e-4)).all()


def test_signed_distance_contains_points_consistency(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    mesh_tm, mesh_wp = icosahedron
    rng = np.random.default_rng(9)
    inside_np = mesh_tm.center_mass + rng.normal(scale=0.05, size=(50, 3))
    points_wp = wp.array(
        np.ascontiguousarray(inside_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    signed_np = tw.proximity.signed_distance_on_mesh(
        mesh_wp.points, mesh_wp.indices, points_wp
    ).numpy()
    contains_np = tw.ray.contains_points(mesh_wp, points_wp).numpy()
    off_surface = np.abs(signed_np) > TOLERANCE_MERGE
    assert np.array_equal(contains_np[off_surface], signed_np[off_surface] < 0.0)


def test_signed_distance_on_mesh_empty_points(device: str) -> None:
    vertices = wp.array(np.zeros((3, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    faces = wp.array([0, 1, 2], dtype=wp.int32, device=device)
    points = wp.empty(0, dtype=wp.vec3, device=device)
    signed_wp = tw.proximity.signed_distance_on_mesh(vertices, faces, points)
    assert signed_wp.shape == (0,)


def test_signed_distance_on_mesh_empty_faces(device: str) -> None:
    vertices = wp.zeros(1, dtype=wp.vec3, device=device)
    faces = wp.empty(0, dtype=wp.int32, device=device)
    points = wp.array(np.zeros((2, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    signed_wp = tw.proximity.signed_distance_on_mesh(vertices, faces, points)
    assert signed_wp.shape == (2,)
    assert np.all(np.isinf(signed_wp.numpy()))


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube", "hemisphere"])
@pytest.mark.parametrize("tiled", [False, True])
def test_winding_number_random(
    request: pytest.FixtureRequest, mesh_name: str, tiled: bool
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    rng = np.random.default_rng(42)
    query_np = rng.random((200, 3), dtype=np.float64) * 4.0 - 2.0
    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int64)

    winding_igl = igl.winding_number(vertices_np, faces_np, query_np)
    query_wp = wp.array(
        np.ascontiguousarray(query_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    winding_wp = tw.proximity.winding_number(
        mesh_wp.points, mesh_wp.indices, query_wp, tiled=tiled
    )
    assert np.allclose(winding_wp.numpy(), winding_igl.ravel(), rtol=1e-5, atol=1e-5)


def test_winding_number_tiled_matches_exact(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    rng = np.random.default_rng(17)
    query_np = rng.random((100, 3), dtype=np.float32) * 2.0 - 1.0
    query_wp = wp.array(
        np.ascontiguousarray(query_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    exact_wp = tw.proximity.winding_number(
        mesh_wp.points, mesh_wp.indices, query_wp, tiled=False
    )
    tiled_wp = tw.proximity.winding_number(
        mesh_wp.points, mesh_wp.indices, query_wp, tiled=True
    )
    assert np.allclose(tiled_wp.numpy(), exact_wp.numpy(), rtol=1e-6, atol=1e-6)


def test_winding_number_inside_outside(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = icosahedron
    outside_np = np.asarray([mesh_tm.bounds[0] + [100.0, 100.0, 100.0]], dtype=np.float32)
    inside_np = np.asarray([mesh_tm.center_mass], dtype=np.float32)
    outside_wp = wp.array(outside_np, dtype=wp.vec3, device=mesh_wp.device)
    inside_wp = wp.array(inside_np, dtype=wp.vec3, device=mesh_wp.device)
    outside_winding_wp = tw.proximity.winding_number(
        mesh_wp.points, mesh_wp.indices, outside_wp
    )
    inside_winding_wp = tw.proximity.winding_number(mesh_wp.points, mesh_wp.indices, inside_wp)
    assert np.allclose(outside_winding_wp.numpy(), 0.0, atol=1e-3)
    assert np.allclose(inside_winding_wp.numpy(), 1.0, atol=1e-3)


def test_winding_number_cave_cube_origin(cave_cube: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = cave_cube
    origin_np = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    origin_wp = wp.array(origin_np, dtype=wp.vec3, device=mesh_wp.device)
    winding_wp = tw.proximity.winding_number(mesh_wp.points, mesh_wp.indices, origin_wp)
    assert np.allclose(winding_wp.numpy(), 0.0, atol=1e-3)


def test_winding_number_empty_points(device: str) -> None:
    vertices = wp.array(np.zeros((3, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    faces = wp.array([0, 1, 2], dtype=wp.int32, device=device)
    points = wp.empty(0, dtype=wp.vec3, device=device)
    winding_wp = tw.proximity.winding_number(vertices, faces, points)
    assert winding_wp.shape == (0,)


def test_winding_number_empty_faces(device: str) -> None:
    vertices = wp.zeros(1, dtype=wp.vec3, device=device)
    faces = wp.empty(0, dtype=wp.int32, device=device)
    points = wp.array(np.zeros((2, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    winding_wp = tw.proximity.winding_number(vertices, faces, points)
    assert winding_wp.shape == (2,)
    assert np.allclose(winding_wp.numpy(), 0.0)


def test_max_tangent_sphere(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = icosahedron
    points_np, face_ids = tm.sample.sample_surface(mesh_tm, 20, seed=42)
    normals_np = mesh_tm.face_normals[face_ids]

    points_wp = wp.array(
        np.ascontiguousarray(points_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    normals_wp = wp.array(
        np.ascontiguousarray(normals_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )

    centers_wp, radii_wp = tw.proximity.max_tangent_sphere(mesh_wp, points_wp, normals=normals_wp)
    centers_tm, radii_tm = tm_proximity.max_tangent_sphere(mesh_tm, points_np, normals=normals_np)

    finite_tm = np.isfinite(radii_tm)
    assert np.array_equal(np.isfinite(radii_wp.numpy()), finite_tm)
    if finite_tm.any():
        assert np.allclose(radii_wp.numpy()[finite_tm], radii_tm[finite_tm], rtol=1e-2, atol=1e-2)
        assert np.allclose(
            centers_wp.numpy()[finite_tm], centers_tm[finite_tm], rtol=1e-2, atol=1e-2
        )


def test_thickness_max_sphere(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = icosahedron
    points_np, face_ids = tm.sample.sample_surface(mesh_tm, 20, seed=7)
    normals_np = mesh_tm.face_normals[face_ids]

    points_wp = wp.array(
        np.ascontiguousarray(points_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    normals_wp = wp.array(
        np.ascontiguousarray(normals_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )

    thickness_wp = tw.proximity.thickness(mesh_wp, points_wp, normals=normals_wp).numpy()
    thickness_tm = tm_proximity.thickness(mesh_tm, points_np, normals=normals_np)

    finite_tm = np.isfinite(thickness_tm)
    assert np.array_equal(np.isfinite(thickness_wp), finite_tm)
    if finite_tm.any():
        assert np.allclose(thickness_wp[finite_tm], thickness_tm[finite_tm], rtol=1e-5, atol=1e-5)


def test_thickness_ray(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = icosahedron
    points_np, face_ids = tm.sample.sample_surface(mesh_tm, 20, seed=13)
    normals_np = mesh_tm.face_normals[face_ids]

    points_wp = wp.array(
        np.ascontiguousarray(points_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    normals_wp = wp.array(
        np.ascontiguousarray(normals_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )

    thickness_wp = tw.proximity.thickness(
        mesh_wp, points_wp, normals=normals_wp, method="ray"
    ).numpy()
    thickness_tm = tm_proximity.thickness(mesh_tm, points_np, normals=normals_np, method="ray")

    finite_tm = np.isfinite(thickness_tm)
    assert np.array_equal(np.isfinite(thickness_wp), finite_tm)
    if finite_tm.any():
        assert np.allclose(thickness_wp[finite_tm], thickness_tm[finite_tm])


def test_max_tangent_sphere_empty(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = icosahedron
    points_wp = wp.empty(0, dtype=wp.vec3, device=mesh_wp.device)
    centers_wp, radii_wp = tw.proximity.max_tangent_sphere(mesh_wp, points_wp)
    assert centers_wp.shape == (0,)
    assert radii_wp.shape == (0,)
