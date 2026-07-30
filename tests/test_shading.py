"""Regression tests for ``triwarp.shading`` against pymeshlab (CPU reference)."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh as tm
import warp as wp

import triwarp as tw
from tests.conversions import trimesh_to_pymeshlab


def _vertex_normals_wp(mesh_wp: wp.Mesh) -> wp.array[wp.vec3]:
    """Smooth outward normals; face normals would make the field piecewise constant per ring."""
    return tw.vertices.area_weighted_vertex_normals(
        int(mesh_wp.points.shape[0]), mesh_wp.points, mesh_wp.indices
    )


@pytest.mark.parametrize("weight", ["cosine", "uniform"])
@pytest.mark.parametrize("mesh_name", ["icosahedron", "torus"])
def test_ambient_occlusion_is_zero_on_a_convex_mesh(
    request: pytest.FixtureRequest, mesh_name: str, weight: str
) -> None:
    """
    No ray leaving a convex closed surface can come back, so the occlusion is *exactly* zero.

    The one analytic check this quantity has, and the reason ``torus`` is here too — it is not
    convex, so it must *not* read zero, which is what makes the icosahedron result meaningful
    rather than a stuck kernel.
    """
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    occlusion_np = tw.shading.ambient_occlusion(
        mesh_wp, mesh_wp.points, normals=_vertex_normals_wp(mesh_wp), n_rays=64, weight=weight
    ).numpy()
    assert (occlusion_np >= 0.0).all()
    assert (occlusion_np <= 1.0).all()
    if mesh_name == "icosahedron":
        assert occlusion_np.max() == 0.0
    else:
        assert occlusion_np.max() > 0.1


def test_ambient_occlusion_finds_the_cavity(cave_cube: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """The inner shell of a hollow cube must be far more occluded than the outer one."""
    mesh_tm, mesh_wp = cave_cube
    occlusion_np = tw.shading.ambient_occlusion(
        mesh_wp, mesh_wp.points, normals=_vertex_normals_wp(mesh_wp), n_rays=128
    ).numpy()
    # The cavity is the inner 0.1-cube; its vertices are the ones near the origin.
    inner_np = np.linalg.norm(np.asarray(mesh_tm.vertices), axis=1) < 0.2
    assert inner_np.any()
    assert (~inner_np).any()
    assert occlusion_np[inner_np].mean() > 0.5
    assert occlusion_np[~inner_np].mean() < 0.1


def test_ambient_occlusion_ranks_like_pymeshlab(torus: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    MeshLab's scalar is the unnormalized *complement*, so the two agree by rank, not by value.

    ``compute_scalar_ambient_occlusion`` sums ``cos`` over the visible directions of its own
    whole-sphere set without dividing, so a fully exposed vertex reads ~``rays / 4`` rather than
    ``1``, and its direction set is not this one. What must hold is that both call the same vertices
    the occluded ones — hence a rank correlation over a torus, whose hole occludes half of its inner
    wall.
    """
    mesh_tm, mesh_wp = torus
    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.compute_scalar_ambient_occlusion(rays=256)
    exposure_pml = meshset_pml.current_mesh().vertex_scalar_array()

    occlusion_np = tw.shading.ambient_occlusion(
        mesh_wp, mesh_wp.points, normals=_vertex_normals_wp(mesh_wp), n_rays=256
    ).numpy()

    # Spearman: rank both sides, then correlate. Exposure is the complement of occlusion, so the
    # correlation must be strongly *negative*.
    rank_pml = np.argsort(np.argsort(exposure_pml))
    rank_wp = np.argsort(np.argsort(occlusion_np))
    assert np.corrcoef(rank_pml, rank_wp)[0, 1] < -0.8


def test_ambient_occlusion_converges_with_more_rays(torus: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """Successive doublings of the ray count must move the field by less and less."""
    _mesh_tm, mesh_wp = torus
    normals_wp = _vertex_normals_wp(mesh_wp)
    fields = [
        tw.shading.ambient_occlusion(
            mesh_wp, mesh_wp.points, normals=normals_wp, n_rays=n_rays
        ).numpy()
        for n_rays in (64, 256, 1024)
    ]
    assert np.abs(fields[2] - fields[1]).mean() < np.abs(fields[1] - fields[0]).mean()


def test_ambient_occlusion_uniform_weight_differs_from_cosine(
    torus: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """The two conventions are genuinely different integrals, not a rescaling of each other."""
    _mesh_tm, mesh_wp = torus
    normals_wp = _vertex_normals_wp(mesh_wp)
    cosine_np = tw.shading.ambient_occlusion(
        mesh_wp, mesh_wp.points, normals=normals_wp, n_rays=256, weight="cosine"
    ).numpy()
    uniform_np = tw.shading.ambient_occlusion(
        mesh_wp, mesh_wp.points, normals=normals_wp, n_rays=256, weight="uniform"
    ).numpy()
    assert not np.allclose(cosine_np, uniform_np, atol=1e-3)
    # Grazing directions are the ones the cosine weight discounts, and in a cavity they are the
    # blocked ones, so uniform weighting reports more occlusion on average.
    assert uniform_np.mean() > cosine_np.mean()


def test_ambient_occlusion_invalid(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    with pytest.raises(ValueError, match="n_rays >= 1"):
        tw.shading.ambient_occlusion(mesh_wp, mesh_wp.points, n_rays=0)
    with pytest.raises(ValueError, match="weight must be"):
        tw.shading.ambient_occlusion(mesh_wp, mesh_wp.points, weight="lambert")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="one entry per point"):
        tw.shading.ambient_occlusion(
            mesh_wp, mesh_wp.points, normals=wp.zeros(2, dtype=wp.vec3, device=mesh_wp.device)
        )


def test_ambient_occlusion_empty(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    points_wp = wp.zeros(0, dtype=wp.vec3, device=mesh_wp.device)
    assert tw.shading.ambient_occlusion(mesh_wp, points_wp).shape == (0,)


def test_volumetric_obscurance_is_zero_on_a_convex_mesh(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    _mesh_tm, mesh_wp = icosahedron
    obscurance_np = tw.shading.volumetric_obscurance(
        mesh_wp, mesh_wp.points, normals=_vertex_normals_wp(mesh_wp), n_rays=64
    ).numpy()
    assert obscurance_np.max() == 0.0


def test_volumetric_obscurance_approaches_ambient_occlusion_as_tau_falls(
    torus: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """Ambient occlusion is the ``tau -> 0`` limit, so the gap must shrink monotonically."""
    _mesh_tm, mesh_wp = torus
    normals_wp = _vertex_normals_wp(mesh_wp)
    occlusion_np = tw.shading.ambient_occlusion(
        mesh_wp, mesh_wp.points, normals=normals_wp, n_rays=128
    ).numpy()
    gaps = [
        np.abs(
            tw.shading.volumetric_obscurance(
                mesh_wp, mesh_wp.points, normals=normals_wp, n_rays=128, tau=tau
            ).numpy()
            - occlusion_np
        ).mean()
        for tau in (1.0, 0.1, 0.01, 1e-4)
    ]
    assert gaps == sorted(gaps, reverse=True)
    assert gaps[-1] < 1e-3


def test_volumetric_obscurance_attenuates_distant_occluders(
    torus: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """A larger ``tau`` discounts every occluder, so the field can only go down."""
    _mesh_tm, mesh_wp = torus
    normals_wp = _vertex_normals_wp(mesh_wp)
    low_np = tw.shading.volumetric_obscurance(
        mesh_wp, mesh_wp.points, normals=normals_wp, n_rays=128, tau=0.1
    ).numpy()
    high_np = tw.shading.volumetric_obscurance(
        mesh_wp, mesh_wp.points, normals=normals_wp, n_rays=128, tau=10.0
    ).numpy()
    assert (high_np <= low_np + 1e-6).all()
    assert high_np.mean() < low_np.mean()


def test_volumetric_obscurance_ranks_like_pymeshlab(torus: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = torus
    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.compute_scalar_by_volumetric_obscurance(rays=256, tau=0.1)
    exposure_pml = meshset_pml.current_mesh().vertex_scalar_array()

    obscurance_np = tw.shading.volumetric_obscurance(
        mesh_wp, mesh_wp.points, normals=_vertex_normals_wp(mesh_wp), n_rays=256, tau=0.1
    ).numpy()
    rank_pml = np.argsort(np.argsort(exposure_pml))
    rank_wp = np.argsort(np.argsort(obscurance_np))
    assert np.corrcoef(rank_pml, rank_wp)[0, 1] < -0.8


def test_volumetric_obscurance_invalid(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    with pytest.raises(ValueError, match="tau must be positive"):
        tw.shading.volumetric_obscurance(mesh_wp, mesh_wp.points, tau=0.0)


def test_sample_fibonacci_cone(device: str) -> None:
    """Every direction inside the cone, and both ends of the range match sphere/hemisphere."""
    half_angle = np.deg2rad(25.0)
    directions_np = tw.sample.sample_fibonacci_cone(512, half_angle, device=device).numpy()
    assert np.allclose(np.linalg.norm(directions_np, axis=1), 1.0, rtol=1e-5, atol=1e-5)
    polar_np = np.arccos(np.clip(directions_np[:, 2], -1.0, 1.0))
    assert polar_np.max() <= half_angle + 1e-6
    # Uniform in solid angle means uniform in z, so the mean z is the midpoint of the band.
    assert np.isclose(directions_np[:, 2].mean(), 0.5 * (1.0 + np.cos(half_angle)), atol=1e-3)

    assert np.allclose(
        tw.sample.sample_fibonacci_cone(64, np.pi / 2.0, device=device).numpy(),
        tw.sample.sample_fibonacci_hemisphere(64, device=device).numpy(),
        rtol=1e-6,
        atol=1e-6,
    )
    assert np.allclose(
        tw.sample.sample_fibonacci_cone(64, np.pi, device=device).numpy(),
        tw.sample.sample_fibonacci_sphere(64, device=device).numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


def test_sample_fibonacci_cone_invalid(device: str) -> None:
    with pytest.raises(ValueError, match=r"half_angle must be in \(0, pi\]"):
        tw.sample.sample_fibonacci_cone(8, 0.0, device=device)
    with pytest.raises(ValueError, match=r"half_angle must be in \(0, pi\]"):
        tw.sample.sample_fibonacci_cone(8, 4.0, device=device)
    assert tw.sample.sample_fibonacci_cone(0, 1.0, device=device).shape == (0,)
