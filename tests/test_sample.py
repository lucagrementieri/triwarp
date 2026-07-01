"""Regression tests for ``triwarp.sample`` vs ``trimesh.sample`` (CPU reference)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import trimesh as tm
import warp as wp
from scipy.spatial.distance import pdist

import triwarp as tw


def test_sample_surface(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = half_torus
    count = 10_000
    n_faces = int(mesh_tm.faces.shape[0])
    face_idx_tm = tm.sample.sample_surface(mesh_tm, count, seed=0)[1]
    freq_tm = np.bincount(face_idx_tm, minlength=n_faces) / count

    face_idx_wp = tw.sample.sample_surface(mesh_wp.points, mesh_wp.indices, count, seed=0)[1]
    freq_wp = np.bincount(face_idx_wp.numpy(), minlength=n_faces) / count

    freq_expected = mesh_tm.area_faces / mesh_tm.area_faces.sum()
    assert np.allclose(freq_wp, freq_expected, rtol=0.08, atol=0.008)
    assert np.allclose(freq_tm, freq_expected, rtol=0.08, atol=0.008)


def test_sample_surface_with_face_weights(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = icosahedron
    count = 10_000

    weights_np = np.arange(mesh_tm.faces.shape[0], dtype=np.float32)
    face_idx_tm = tm.sample.sample_surface(mesh_tm, count, face_weight=weights_np, seed=0)[1]

    weights_wp = wp.array(weights_np, dtype=wp.float32, device=mesh_wp.points.device)
    face_idx_wp = tw.sample.sample_surface(
        mesh_wp.points, mesh_wp.indices, count, face_weight=weights_wp, seed=0
    )[1]

    n_faces = int(mesh_tm.faces.shape[0])
    freq_wp = np.bincount(face_idx_wp.numpy(), minlength=n_faces) / count
    freq_tm = np.bincount(face_idx_tm, minlength=n_faces) / count

    freq_expected = weights_np / weights_np.sum()
    assert np.allclose(freq_wp, freq_expected, rtol=0.07, atol=0.01)
    assert np.allclose(freq_tm, freq_expected, rtol=0.07, atol=0.01)


def test_sample_surface_poisson_disk_count(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    _, mesh_wp = icosahedron
    count = 100
    pts, fids = tw.sample.sample_surface_poisson_disk(
        mesh_wp.points, mesh_wp.indices, count, seed=0
    )
    assert pts.shape == (count,)
    assert fids.shape == (count,)


def test_sample_surface_poisson_disk_on_surface(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = icosahedron
    count = 100
    pts, _ = tw.sample.sample_surface_poisson_disk(mesh_wp.points, mesh_wp.indices, count, seed=1)
    _, dists, _ = tm.proximity.closest_point(mesh_tm, pts.numpy())
    assert np.all(dists < 1e-4)


def test_sample_surface_poisson_disk_min_distance(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = icosahedron
    count = 100
    init_factor = 5.0
    surface_area = float(mesh_tm.area)
    ratio = 1.0 / init_factor
    r_max = 2.0 * math.sqrt((surface_area / count) / (2.0 * math.sqrt(3.0)))
    r_min = r_max * 0.65 * (1.0 - ratio**1.5)

    points, _ = tw.sample.sample_surface_poisson_disk(
        mesh_wp.points, mesh_wp.indices, count, init_factor=init_factor, seed=2
    )
    min_dist = float(pdist(points.numpy()).min())
    assert min_dist >= r_min * 0.9


def test_sample_surface_poisson_disk_deterministic(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    _, mesh_wp = icosahedron
    points_a, face_indices_a = tw.sample.sample_surface_poisson_disk(
        mesh_wp.points, mesh_wp.indices, 80, seed=7
    )
    points_b, face_indices_b = tw.sample.sample_surface_poisson_disk(
        mesh_wp.points, mesh_wp.indices, 80, seed=7
    )
    assert np.array_equal(points_a.numpy(), points_b.numpy())
    assert np.array_equal(face_indices_a.numpy(), face_indices_b.numpy())


def test_sample_surface_poisson_disk_count_zero(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    _, mesh_wp = icosahedron
    points, face_indices = tw.sample.sample_surface_poisson_disk(mesh_wp.points, mesh_wp.indices, 0)
    assert points.shape == (0,)
    assert face_indices.shape == (0,)


def test_sample_surface_poisson_disk_high_init_factor(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    """Exercise the GPU top-k branch when local maxima exceed excess."""
    _, mesh_wp = icosahedron
    count = 20
    pts, fids = tw.sample.sample_surface_poisson_disk(
        mesh_wp.points, mesh_wp.indices, count, init_factor=20.0, seed=3
    )
    assert pts.shape == (count,)
    assert fids.shape == (count,)


def _blue_noise_radius_for_count(surface_area: float, n: int) -> float:
    return math.sqrt((surface_area * 0.5 / (n * 0.6162910373)) / math.pi)


def test_sample_surface_blue_noise_min_distance(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = icosahedron
    radius = _blue_noise_radius_for_count(float(mesh_tm.area), 80)
    points, _ = tw.sample.sample_surface_blue_noise(
        mesh_wp.points, mesh_wp.indices, radius, seed=2
    )
    points_np = points.numpy().reshape(-1, 3)
    if points_np.shape[0] >= 2:
        min_dist = float(pdist(points_np).min())
        assert min_dist >= radius * 0.99


def test_sample_surface_blue_noise_on_surface(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = icosahedron
    radius = _blue_noise_radius_for_count(float(mesh_tm.area), 50)
    pts, _ = tw.sample.sample_surface_blue_noise(mesh_wp.points, mesh_wp.indices, radius, seed=1)
    _, dists, _ = tm.proximity.closest_point(mesh_tm, pts.numpy())
    assert np.all(dists < 1e-4)


def test_sample_surface_blue_noise_deterministic(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    _, mesh_wp = icosahedron
    radius = _blue_noise_radius_for_count(1.0, 30)
    points_a, face_indices_a = tw.sample.sample_surface_blue_noise(
        mesh_wp.points, mesh_wp.indices, radius, seed=7
    )
    points_b, face_indices_b = tw.sample.sample_surface_blue_noise(
        mesh_wp.points, mesh_wp.indices, radius, seed=7
    )
    assert np.array_equal(points_a.numpy(), points_b.numpy())
    assert np.array_equal(face_indices_a.numpy(), face_indices_b.numpy())


def test_sample_surface_blue_noise_count_order_of_magnitude(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = icosahedron
    surface_area = float(mesh_tm.area)
    expected = 50
    radius = _blue_noise_radius_for_count(surface_area, expected)
    points, _ = tw.sample.sample_surface_blue_noise(
        mesh_wp.points, mesh_wp.indices, radius, seed=0
    )
    n = int(points.shape[0])
    igl_expected = surface_area * (math.pi * math.sqrt(3.0) / 6.0) / (math.pi * radius * radius / 4.0)
    assert 0.5 * igl_expected <= n <= 1.5 * igl_expected


def test_sample_surface_blue_noise_radius_invalid(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    _, mesh_wp = icosahedron
    with pytest.raises(ValueError, match="radius"):
        tw.sample.sample_surface_blue_noise(mesh_wp.points, mesh_wp.indices, 0.0)


def test_sample_surface_blue_noise_empty_faces(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    mesh_wp = icosahedron[1]
    empty_faces = wp.empty(0, dtype=wp.int32, device=mesh_wp.points.device)
    points, face_indices = tw.sample.sample_surface_blue_noise(
        mesh_wp.points, empty_faces, 0.1, seed=0
    )
    assert points.shape == (0,)
    assert face_indices.shape == (0,)


def test_sample_volume_containment(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = icosahedron
    count = 5_000
    points_np = tw.sample.sample_volume(mesh_wp.points, mesh_wp.indices, count, seed=42).numpy()
    assert points_np.shape == (count, 3)
    assert mesh_tm.contains(points_np).all()


def test_sample_volume_uniform(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    # With 20 000 samples the per-axis std-of-mean is ~0.003, so atol=0.05 is safe.
    mesh_tm, mesh_wp = icosahedron
    points_np = tw.sample.sample_volume(mesh_wp.points, mesh_wp.indices, 20_000, seed=0).numpy()
    assert np.allclose(points_np.mean(axis=0), mesh_tm.center_mass, atol=0.05)


def test_sample_volume_deterministic(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    _, mesh_wp = icosahedron
    pts_a = tw.sample.sample_volume(mesh_wp.points, mesh_wp.indices, 200, seed=7).numpy()
    pts_b = tw.sample.sample_volume(mesh_wp.points, mesh_wp.indices, 200, seed=7).numpy()
    assert np.array_equal(pts_a, pts_b)


def test_sample_volume_count_zero(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    _, mesh_wp = icosahedron
    pts = tw.sample.sample_volume(mesh_wp.points, mesh_wp.indices, 0)
    assert pts.shape == (0,)


def test_sample_volume_not_watertight(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    _, mesh_wp = half_torus
    with pytest.raises(ValueError, match="watertight"):
        tw.sample.sample_volume(mesh_wp.points, mesh_wp.indices, 100)
