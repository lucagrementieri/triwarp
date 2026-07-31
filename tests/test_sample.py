"""Regression tests for ``triwarp.sample`` vs ``trimesh.sample`` (CPU reference)."""

from __future__ import annotations

import math

import numpy as np
import pymeshlab as ml
import pytest
import trimesh as tm
import warp as wp
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

import triwarp as tw
from tests.conversions import trimesh_to_open3d, trimesh_to_pymeshlab


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
    points, _ = tw.sample.sample_surface_blue_noise(mesh_wp.points, mesh_wp.indices, radius, seed=2)
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


def test_sample_surface_blue_noise_count_order_of_magnitude(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
):
    mesh_tm, mesh_wp = icosahedron
    surface_area = float(mesh_tm.area)
    expected = 50
    radius = _blue_noise_radius_for_count(surface_area, expected)
    points, _ = tw.sample.sample_surface_blue_noise(mesh_wp.points, mesh_wp.indices, radius, seed=0)
    n = int(points.shape[0])
    igl_expected = (
        surface_area * (math.pi * math.sqrt(3.0) / 6.0) / (math.pi * radius * radius / 4.0)
    )
    assert 0.5 * igl_expected <= n <= 1.5 * igl_expected


def _blue_noise_statistics(
    mesh_tm: tm.Trimesh, points_np: np.ndarray, radius: float, dense_np: np.ndarray
) -> tuple[float, float, int]:
    """
    Return the three radius-relative quantities comparable across blue-noise samplers.

    Returns the closest pair as a multiple of ``radius`` (the Poisson-disk property itself), the
    worst uncovered gap as a multiple of ``radius`` (how space-filling the set is, measured against
    a dense uniform sample of the same surface), and the number of faces carrying at least one
    sample.
    """
    points_np = np.asarray(points_np, dtype=np.float64).reshape(-1, 3)
    _closest, _distance, face_index_np = tm.proximity.closest_point(mesh_tm, points_np)
    return (
        float(pdist(points_np).min()) / radius,
        float(cKDTree(points_np).query(dense_np)[0].max()) / radius,
        int(np.unique(face_index_np).shape[0]),
    )


@pytest.mark.parity("blue_noise", "open3d", "pymeshlab")
def test_sample_surface_blue_noise_matches_open3d_and_pymeshlab(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Class C: three blue-noise samplers, three algorithms, no correspondence between the point sets.

    triwarp does Bridson dart throwing on a background grid, Open3D's ``sample_points_poisson_disk``
    runs Yuksel's sample *elimination* from a dense uniform cloud, and MeshLab's
    ``generate_sampling_poisson_disk`` is Corsini et al.'s hierarchical dart throwing. Nothing about
    the individual samples is shared -- not their count, not their positions, not even their number
    given the same parameter -- so the comparison is on the properties all three claim. MeshLab is
    the only reference that accepts a *radius* (``radius=PureValue(r)`` overrides ``samplenum``), so
    it gets triwarp's own parameter; Open3D takes a count and gets triwarp's output count, exactly
    as the benchmark parametrizes them.

    **Bug class excluded:** a sampler that is not blue noise at all (assert 1) and one that is blue
    noise over only part of the surface (asserts 2 and 3). Both are live failure modes for a
    grid-based dart thrower -- a mis-sized background cell rejects too little, a mis-mapped cell-to-
    face seeding covers too little -- and neither is visible to
    ``test_sample_surface_blue_noise_min_distance``, which tests triwarp against itself.

    **Measured, with both mutation probes.** Two degenerate stand-ins are scored alongside: a
    uniform Monte-Carlo cloud of the *same size* (blue noise's null hypothesis) and one confined to
    4 of the 20 faces.

    | | closest pair / r | worst gap / r | faces hit |
    |---|---|---|---|
    | triwarp | 1.000 | 1.056 | 20 / 20 |
    | MeshLab, same radius | 1.000 | 1.076 | 20 / 20 |
    | Open3D, same count | 0.891 | 1.327 | 20 / 20 |
    | uniform Monte Carlo | **0.008** | 1.763 | 20 / 20 |
    | one patch | **0.005** | **17.680** | **4 / 20** |

    So assert 1 (``>= 0.85``) clears the worst reference by 4.8% and both probes by **>100x** -- it
    is the assert carrying the bug class. Assert 3 (every face hit) is what separates the clumped
    probe, by a factor of 5. Assert 2 (worst gap ``<= 1.4``) is the weakest of the three at a 1.25x
    margin over the Monte-Carlo probe, and it is applied only to triwarp and MeshLab: Open3D's
    elimination sampler genuinely leaves larger gaps on a 20-face mesh (1.327), which is a property
    of its algorithm rather than a disagreement. Sample *counts* at the identical radius are 791
    against MeshLab's 769, 2.9% apart against a 25% bound (8.6x margin).
    """
    mesh_tm, mesh_wp = icosahedron
    radius = _blue_noise_radius_for_count(float(mesh_tm.area), 300)
    dense_np, _face_index = tm.sample.sample_surface(mesh_tm, 20_000, seed=3)

    points_wp, _face_index_wp = tw.sample.sample_surface_blue_noise(
        mesh_wp.points, mesh_wp.indices, radius, seed=11
    )
    n_samples = int(points_wp.shape[0])

    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.generate_sampling_poisson_disk(radius=ml.PureValue(radius))
    points_pml = np.asarray(meshset_pml.current_mesh().vertex_matrix(), dtype=np.float64)

    points_o3d = np.asarray(
        trimesh_to_open3d(mesh_tm).sample_points_poisson_disk(number_of_points=n_samples).points
    )

    closest_wp, gap_wp, faces_wp = _blue_noise_statistics(
        mesh_tm, points_wp.numpy(), radius, dense_np
    )
    closest_pml, gap_pml, faces_pml = _blue_noise_statistics(mesh_tm, points_pml, radius, dense_np)
    closest_o3d, _gap_o3d, faces_o3d = _blue_noise_statistics(mesh_tm, points_o3d, radius, dense_np)

    # 1. The Poisson-disk property, on all three.
    assert min(closest_wp, closest_pml, closest_o3d) >= 0.85
    # 2. Space-filling, against the reference that received the identical radius.
    assert max(gap_wp, gap_pml) <= 1.4
    # 3. Every face reached, on all three.
    assert faces_wp == faces_pml == faces_o3d == mesh_tm.faces.shape[0]
    # 4. And the radius parametrization agrees: the same radius yields the same order of samples.
    assert 0.8 <= n_samples / points_pml.shape[0] <= 1.25


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


def test_sample_fibonacci_sphere_unit(device: str):
    directions_wp = tw.sample.sample_fibonacci_sphere(1000, device=device)
    directions_np = directions_wp.numpy()
    assert directions_np.shape == (1000, 3)
    norms = np.linalg.norm(directions_np, axis=1)
    assert np.allclose(norms, 1.0, rtol=1e-5, atol=1e-5)


def test_sample_fibonacci_sphere_uniform(device: str):
    # A near-uniform covering of the sphere has its centroid essentially at the origin.
    directions_np = tw.sample.sample_fibonacci_sphere(4096, device=device).numpy()
    assert np.allclose(directions_np.mean(axis=0), 0.0, atol=1e-2)


def test_sample_fibonacci_sphere_deterministic(device: str):
    directions_a = tw.sample.sample_fibonacci_sphere(500, device=device).numpy()
    directions_b = tw.sample.sample_fibonacci_sphere(500, device=device).numpy()
    assert np.array_equal(directions_a, directions_b)


def test_sample_fibonacci_sphere_empty(device: str):
    directions_wp = tw.sample.sample_fibonacci_sphere(0, device=device)
    assert directions_wp.shape == (0,)


def test_sample_fibonacci_hemisphere_positive_z(device: str):
    directions_np = tw.sample.sample_fibonacci_hemisphere(1000, device=device).numpy()
    assert directions_np.shape == (1000, 3)
    assert np.all(directions_np[:, 2] > 0.0)
    norms = np.linalg.norm(directions_np, axis=1)
    assert np.allclose(norms, 1.0, rtol=1e-5, atol=1e-5)


def test_sample_fibonacci_hemisphere_empty(device: str):
    directions_wp = tw.sample.sample_fibonacci_hemisphere(0, device=device)
    assert directions_wp.shape == (0,)
