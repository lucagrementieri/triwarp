"""Regression tests for ``triwarp.convex`` against Trimesh (CPU reference)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.spatial
import warp as wp

import triwarp as tw


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
def test_face_adjacency_projections(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    adjacency_tm = mesh_tm.face_adjacency
    projections_tm = mesh_tm.face_adjacency_projections

    adjacency_wp, adjacency_edges_wp = tw.adjacency.face_adjacency(
        mesh_wp.indices, return_edges=True
    )
    projections_wp = tw.convex.face_adjacency_projections(
        mesh_wp.points,
        mesh_wp.indices,
        face_adjacency=adjacency_wp,
        face_adjacency_edges=adjacency_edges_wp,
    )

    adjacency_wp_np = adjacency_wp.numpy()
    projections_wp_np = projections_wp.numpy()
    projections_wp_lookup = {
        (int(row[0]), int(row[1])): float(projections_wp_np[i])
        for i, row in enumerate(adjacency_wp_np)
    }
    projections_tm_lookup = {
        (int(row[0]), int(row[1])): float(projections_tm[i]) for i, row in enumerate(adjacency_tm)
    }
    assert projections_wp_lookup.keys() == projections_tm_lookup.keys()
    for key, projection_tm in projections_tm_lookup.items():
        projection_wp = projections_wp_lookup[key]
        assert np.isclose(projection_wp, projection_tm, rtol=1e-4, atol=5e-4)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus"])
def test_face_adjacency_projections_precomputed(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    adjacency_wp, adjacency_edges_wp = tw.adjacency.face_adjacency(
        mesh_wp.indices, return_edges=True
    )
    unshared_wp = tw.adjacency.face_adjacency_unshared(
        mesh_wp.indices, face_adjacency=adjacency_wp, face_adjacency_edges=adjacency_edges_wp
    )
    face_normals_wp, _ = tw.triangles.face_normals_and_areas(mesh_wp.points, mesh_wp.indices)

    projections_all_wp = tw.convex.face_adjacency_projections(
        mesh_wp.points,
        mesh_wp.indices,
        face_adjacency=adjacency_wp,
        face_adjacency_edges=adjacency_edges_wp,
    )
    projections_precomputed_wp = tw.convex.face_adjacency_projections(
        mesh_wp.points,
        mesh_wp.indices,
        face_adjacency=adjacency_wp,
        face_adjacency_edges=adjacency_edges_wp,
        face_adjacency_unshared=unshared_wp,
        face_normals=face_normals_wp,
    )
    assert np.allclose(
        projections_all_wp.numpy(), projections_precomputed_wp.numpy(), rtol=1e-5, atol=1e-5
    )

    adjacency_tm = mesh_tm.face_adjacency
    projections_tm = mesh_tm.face_adjacency_projections
    adjacency_wp_np = adjacency_wp.numpy()
    projections_precomputed_np = projections_precomputed_wp.numpy()
    projections_tm_lookup = {
        (int(row[0]), int(row[1])): float(projections_tm[i]) for i, row in enumerate(adjacency_tm)
    }
    projections_precomputed_lookup = {
        (int(row[0]), int(row[1])): float(projections_precomputed_np[i])
        for i, row in enumerate(adjacency_wp_np)
    }
    for key, projection_tm in projections_tm_lookup.items():
        assert np.isclose(projections_precomputed_lookup[key], projection_tm, rtol=1e-4, atol=5e-4)


def test_face_adjacency_projections_empty(device: str) -> None:
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    vertices_wp = wp.empty(0, dtype=wp.vec3, device=device)
    projections_wp = tw.convex.face_adjacency_projections(vertices_wp, faces_wp)
    assert projections_wp.shape == (0,)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
def test_face_adjacency_convex(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    adjacency_tm = mesh_tm.face_adjacency
    convex_tm = mesh_tm.face_adjacency_convex

    adjacency_wp, adjacency_edges_wp = tw.adjacency.face_adjacency(
        mesh_wp.indices, return_edges=True
    )
    convex_wp = tw.convex.face_adjacency_convex(
        mesh_wp.points,
        mesh_wp.indices,
        face_adjacency=adjacency_wp,
        face_adjacency_edges=adjacency_edges_wp,
    )

    adjacency_wp_np = adjacency_wp.numpy()
    convex_wp_np = convex_wp.numpy()
    convex_wp_lookup = {
        (int(row[0]), int(row[1])): bool(convex_wp_np[i]) for i, row in enumerate(adjacency_wp_np)
    }
    convex_tm_lookup = {
        (int(row[0]), int(row[1])): bool(convex_tm[i]) for i, row in enumerate(adjacency_tm)
    }
    assert convex_wp_lookup.keys() == convex_tm_lookup.keys()
    for key, is_convex_tm in convex_tm_lookup.items():
        assert convex_wp_lookup[key] == is_convex_tm


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus"])
def test_face_adjacency_convex_precomputed(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    adjacency_wp, adjacency_edges_wp = tw.adjacency.face_adjacency(
        mesh_wp.indices, return_edges=True
    )
    unshared_wp = tw.adjacency.face_adjacency_unshared(
        mesh_wp.indices, face_adjacency=adjacency_wp, face_adjacency_edges=adjacency_edges_wp
    )
    face_normals_wp, _ = tw.triangles.face_normals_and_areas(mesh_wp.points, mesh_wp.indices)

    convex_all_wp = tw.convex.face_adjacency_convex(
        mesh_wp.points,
        mesh_wp.indices,
        face_adjacency=adjacency_wp,
        face_adjacency_edges=adjacency_edges_wp,
    )
    convex_precomputed_wp = tw.convex.face_adjacency_convex(
        mesh_wp.points,
        mesh_wp.indices,
        face_adjacency=adjacency_wp,
        face_adjacency_edges=adjacency_edges_wp,
        face_adjacency_unshared=unshared_wp,
        face_normals=face_normals_wp,
    )
    assert np.array_equal(convex_all_wp.numpy(), convex_precomputed_wp.numpy())

    adjacency_tm = mesh_tm.face_adjacency
    convex_tm = mesh_tm.face_adjacency_convex
    adjacency_wp_np = adjacency_wp.numpy()
    convex_precomputed_np = convex_precomputed_wp.numpy()
    convex_tm_lookup = {
        (int(row[0]), int(row[1])): bool(convex_tm[i]) for i, row in enumerate(adjacency_tm)
    }
    convex_precomputed_lookup = {
        (int(row[0]), int(row[1])): bool(convex_precomputed_np[i])
        for i, row in enumerate(adjacency_wp_np)
    }
    for key, is_convex_tm in convex_tm_lookup.items():
        assert convex_precomputed_lookup[key] == is_convex_tm


def test_face_adjacency_convex_empty(device: str) -> None:
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    vertices_wp = wp.empty(0, dtype=wp.vec3, device=device)
    convex_wp = tw.convex.face_adjacency_convex(vertices_wp, faces_wp)
    assert convex_wp.shape == (0,)


def test_fast_convex_set_mask_sound(device: str) -> None:
    rng = np.random.default_rng(0)
    points_np = rng.standard_normal((500, 3)).astype(np.float64)
    points_wp = wp.array(np.ascontiguousarray(points_np), dtype=wp.vec3, device=device)

    mask_wp = tw.convex.fast_convex_set_mask(points_wp, n_directions=256)
    selected = np.flatnonzero(mask_wp.numpy())

    hull_scipy = scipy.spatial.ConvexHull(points_np)
    assert set(selected.tolist()) <= set(hull_scipy.vertices.tolist())


def test_fast_convex_set_mask_scale_invariant(device: str) -> None:
    rng = np.random.default_rng(7)
    points_np = rng.standard_normal((500, 3)).astype(np.float64)
    scaled_np = points_np * 1e4

    points_wp = wp.array(np.ascontiguousarray(points_np), dtype=wp.vec3, device=device)
    scaled_wp = wp.array(np.ascontiguousarray(scaled_np), dtype=wp.vec3, device=device)

    mask_wp = tw.convex.fast_convex_set_mask(points_wp, n_directions=256)
    mask_scaled_wp = tw.convex.fast_convex_set_mask(scaled_wp, n_directions=256)
    assert np.array_equal(mask_wp.numpy(), mask_scaled_wp.numpy())

    hull_scipy = scipy.spatial.ConvexHull(scaled_np)
    selected = np.flatnonzero(mask_scaled_wp.numpy())
    assert set(selected.tolist()) <= set(hull_scipy.vertices.tolist())


def test_fast_convex_set_recall(device: str) -> None:
    rng = np.random.default_rng(2)
    points_np = rng.standard_normal((200, 3)).astype(np.float64)
    points_wp = wp.array(np.ascontiguousarray(points_np), dtype=wp.vec3, device=device)

    mask_wp = tw.convex.fast_convex_set_mask(points_wp, n_directions=4096)
    selected = set(np.flatnonzero(mask_wp.numpy()).tolist())

    hull_scipy = scipy.spatial.ConvexHull(points_np)
    assert selected == set(hull_scipy.vertices.tolist())


def test_fast_convex_set_points(device: str) -> None:
    rng = np.random.default_rng(4)
    points_np = rng.standard_normal((300, 3)).astype(np.float64)
    points_wp = wp.array(np.ascontiguousarray(points_np), dtype=wp.vec3, device=device)

    mask_wp = tw.convex.fast_convex_set_mask(points_wp, n_directions=256)
    subset_wp = tw.convex.fast_convex_set(points_wp, n_directions=256)

    selected = np.flatnonzero(mask_wp.numpy())
    expected_points = points_np[np.sort(selected)]
    assert np.allclose(subset_wp.numpy(), expected_points, rtol=1e-5, atol=1e-5)


def test_fast_convex_set_mask_empty(device: str) -> None:
    points_wp = wp.empty(0, dtype=wp.vec3, device=device)
    mask_wp = tw.convex.fast_convex_set_mask(points_wp)
    assert mask_wp.shape == (0,)
