"""
Regression tests for ``triwarp.proximity`` mesh-query APIs.

Mesh AABB queries against a brute-force reference; closest-on-mesh tests compare against
``trimesh.proximity.closest_point``.
"""

from __future__ import annotations

import igl
import numpy as np
import pytest
import trimesh as tm
import trimesh.proximity as tm_proximity
import warp as wp

import triwarp as tw
from triwarp.constants import TOLERANCE_MERGE


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
def test_winding_number_random(request: pytest.FixtureRequest, mesh_name: str, tiled: bool) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    rng = np.random.default_rng(42)
    query_np = rng.random((200, 3), dtype=np.float64) * 4.0 - 2.0
    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int64)

    winding_igl = igl.winding_number(vertices_np, faces_np, query_np)
    query_wp = wp.array(
        np.ascontiguousarray(query_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    winding_wp = tw.proximity.winding_number(mesh_wp.points, mesh_wp.indices, query_wp, tiled=tiled)
    assert np.allclose(winding_wp.numpy(), winding_igl.ravel(), rtol=1e-5, atol=1e-5)


def test_winding_number_tiled_matches_exact(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    rng = np.random.default_rng(17)
    query_np = rng.random((100, 3), dtype=np.float32) * 2.0 - 1.0
    query_wp = wp.array(
        np.ascontiguousarray(query_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    exact_wp = tw.proximity.winding_number(mesh_wp.points, mesh_wp.indices, query_wp, tiled=False)
    tiled_wp = tw.proximity.winding_number(mesh_wp.points, mesh_wp.indices, query_wp, tiled=True)
    assert np.allclose(tiled_wp.numpy(), exact_wp.numpy(), rtol=1e-6, atol=1e-6)


def test_winding_number_inside_outside(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = icosahedron
    outside_np = np.asarray([mesh_tm.bounds[0] + [100.0, 100.0, 100.0]], dtype=np.float32)
    inside_np = np.asarray([mesh_tm.center_mass], dtype=np.float32)
    outside_wp = wp.array(outside_np, dtype=wp.vec3, device=mesh_wp.device)
    inside_wp = wp.array(inside_np, dtype=wp.vec3, device=mesh_wp.device)
    outside_winding_wp = tw.proximity.winding_number(mesh_wp.points, mesh_wp.indices, outside_wp)
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
