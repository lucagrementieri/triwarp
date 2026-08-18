"""Regression tests for ``triwarp.visibility`` against pymeshlab (CPU reference)."""

from __future__ import annotations

import numpy as np
import pymeshlab as ml
import pytest
import trimesh as tm
import trimesh.proximity as tm_proximity
import warp as wp

import triwarp as tw
from tests.conversions import trimesh_to_pymeshlab, trimesh_to_warp


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
    occlusion_np = tw.visibility.ambient_occlusion(
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
    occlusion_np = tw.visibility.ambient_occlusion(
        mesh_wp, mesh_wp.points, normals=_vertex_normals_wp(mesh_wp), n_rays=128
    ).numpy()
    # The cavity is the inner 0.1-cube; its vertices are the ones near the origin.
    inner_np = np.linalg.norm(np.asarray(mesh_tm.vertices), axis=1) < 0.2
    assert inner_np.any()
    assert (~inner_np).any()
    assert occlusion_np[inner_np].mean() > 0.5
    assert occlusion_np[~inner_np].mean() < 0.1


@pytest.mark.parity("ambient_occlusion", "pymeshlab")
def test_ambient_occlusion_ranks_like_pymeshlab(torus: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class C (rank correlation): MeshLab's scalar is the unnormalized *complement* of this one.

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

    occlusion_np = tw.visibility.ambient_occlusion(
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
        tw.visibility.ambient_occlusion(
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
    cosine_np = tw.visibility.ambient_occlusion(
        mesh_wp, mesh_wp.points, normals=normals_wp, n_rays=256, weight="cosine"
    ).numpy()
    uniform_np = tw.visibility.ambient_occlusion(
        mesh_wp, mesh_wp.points, normals=normals_wp, n_rays=256, weight="uniform"
    ).numpy()
    assert not np.allclose(cosine_np, uniform_np, atol=1e-3)
    # Grazing directions are the ones the cosine weight discounts, and in a cavity they are the
    # blocked ones, so uniform weighting reports more occlusion on average.
    assert uniform_np.mean() > cosine_np.mean()


def test_ambient_occlusion_invalid(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    with pytest.raises(ValueError, match="n_rays >= 1"):
        tw.visibility.ambient_occlusion(mesh_wp, mesh_wp.points, n_rays=0)
    with pytest.raises(ValueError, match="weight must be"):
        tw.visibility.ambient_occlusion(mesh_wp, mesh_wp.points, weight="lambert")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="one entry per point"):
        tw.visibility.ambient_occlusion(
            mesh_wp, mesh_wp.points, normals=wp.zeros(2, dtype=wp.vec3, device=mesh_wp.device)
        )


def test_ambient_occlusion_empty(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    points_wp = wp.zeros(0, dtype=wp.vec3, device=mesh_wp.device)
    assert tw.visibility.ambient_occlusion(mesh_wp, points_wp).shape == (0,)


def test_volumetric_obscurance_is_zero_on_a_convex_mesh(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    _mesh_tm, mesh_wp = icosahedron
    obscurance_np = tw.visibility.volumetric_obscurance(
        mesh_wp, mesh_wp.points, normals=_vertex_normals_wp(mesh_wp), n_rays=64
    ).numpy()
    assert obscurance_np.max() == 0.0


def test_volumetric_obscurance_approaches_ambient_occlusion_as_tau_falls(
    torus: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """Ambient occlusion is the ``tau -> 0`` limit, so the gap must shrink monotonically."""
    _mesh_tm, mesh_wp = torus
    normals_wp = _vertex_normals_wp(mesh_wp)
    occlusion_np = tw.visibility.ambient_occlusion(
        mesh_wp, mesh_wp.points, normals=normals_wp, n_rays=128
    ).numpy()
    gaps = [
        np.abs(
            tw.visibility.volumetric_obscurance(
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
    low_np = tw.visibility.volumetric_obscurance(
        mesh_wp, mesh_wp.points, normals=normals_wp, n_rays=128, tau=0.1
    ).numpy()
    high_np = tw.visibility.volumetric_obscurance(
        mesh_wp, mesh_wp.points, normals=normals_wp, n_rays=128, tau=10.0
    ).numpy()
    assert (high_np <= low_np + 1e-6).all()
    assert high_np.mean() < low_np.mean()


def test_volumetric_obscurance_ranks_like_pymeshlab(torus: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class C (rank correlation), and the sign of the correlation is the claim.

    MeshLab reports *exposure* where triwarp reports *obscurance*, so agreement means a correlation
    below **-0.8**, not above it -- an implementation that returned exposure would pass a
    ``|corr| > 0.8`` bar and fails this one. The torus is the fixture because its hole obscures the
    inner wall and leaves the outer wall exposed, giving the ranks something to disagree about.
    """
    mesh_tm, mesh_wp = torus
    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.compute_scalar_by_volumetric_obscurance(rays=256, tau=0.1)
    exposure_pml = meshset_pml.current_mesh().vertex_scalar_array()

    obscurance_np = tw.visibility.volumetric_obscurance(
        mesh_wp, mesh_wp.points, normals=_vertex_normals_wp(mesh_wp), n_rays=256, tau=0.1
    ).numpy()
    rank_pml = np.argsort(np.argsort(exposure_pml))
    rank_wp = np.argsort(np.argsort(obscurance_np))
    assert np.corrcoef(rank_pml, rank_wp)[0, 1] < -0.8


def test_volumetric_obscurance_invalid(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    with pytest.raises(ValueError, match="tau must be positive"):
        tw.visibility.volumetric_obscurance(mesh_wp, mesh_wp.points, tau=0.0)


# ---------------------------------------------------------------------------
# Shape diameter function (pymeshlab reference; analytic on a sphere)
# ---------------------------------------------------------------------------


def _sphere_wp(device: str, radius: float, subdivisions: int = 3):
    sphere_tm = tm.creation.icosphere(subdivisions=subdivisions, radius=radius)
    vertices_wp = wp.array(
        np.ascontiguousarray(sphere_tm.vertices, dtype=np.float32), dtype=wp.vec3, device=device
    )
    faces_wp = wp.array(
        np.ascontiguousarray(sphere_tm.faces.reshape(-1), dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )
    mesh_wp = wp.Mesh(points=vertices_wp, indices=faces_wp)
    normals_wp = tw.vertices.area_weighted_vertex_normals(
        int(vertices_wp.shape[0]), vertices_wp, faces_wp
    )
    return sphere_tm, mesh_wp, normals_wp


@pytest.mark.parametrize("radius", [1.0, 3.0])
def test_shape_diameter_is_the_diameter_of_a_sphere(device: str, radius: float) -> None:
    """
    A cone through a sphere is bracketed analytically, at any radius — the exact check.

    A ray leaving the surface at angle ``theta`` from the inward normal crosses a chord of exactly
    ``2 R cos(theta)``, so every ray in a cone of half-angle ``alpha`` lands in
    ``[2 R cos(alpha), 2 R]`` and so does any weighted mean of them. Both ends are tight: widening
    the cone lowers the answer by exactly that factor, which is why this is a bracket rather than an
    ``allclose`` against ``2 R``.
    """
    _sphere_tm, mesh_wp, normals_wp = _sphere_wp(device, radius)
    cone_angle = np.deg2rad(5.0)
    diameter_np = tw.visibility.shape_diameter(
        mesh_wp, mesh_wp.points, normals=normals_wp, n_rays=128, cone_angle=cone_angle
    ).numpy()
    assert (diameter_np <= 2.0 * radius * (1.0 + 1e-4)).all()
    assert (diameter_np >= 2.0 * radius * np.cos(cone_angle) * (1.0 - 1e-4)).all()


def test_shape_diameter_reduces_to_thickness(device: str) -> None:
    """One ray down a vanishing cone *is* ``thickness(method="ray")``, to float32."""
    _sphere_tm, mesh_wp, normals_wp = _sphere_wp(device, 1.5)
    diameter_np = tw.visibility.shape_diameter(
        mesh_wp, mesh_wp.points, normals=normals_wp, n_rays=1, cone_angle=1e-4
    ).numpy()
    thickness_np = tw.visibility.thickness(
        mesh_wp, mesh_wp.points, normals=normals_wp, method="ray"
    ).numpy()
    assert np.allclose(diameter_np, thickness_np, rtol=1e-5, atol=1e-5)


def test_shape_diameter_measures_a_slab(device: str) -> None:
    """On a 1 x 1 x 4 box the large faces are 1 apart, and the cone must say so."""
    box_tm = tm.creation.box(extents=[1.0, 1.0, 4.0]).subdivide().subdivide().subdivide()
    vertices_wp = wp.array(
        np.ascontiguousarray(box_tm.vertices, dtype=np.float32), dtype=wp.vec3, device=device
    )
    faces_wp = wp.array(
        np.ascontiguousarray(box_tm.faces.reshape(-1), dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )
    mesh_wp = wp.Mesh(points=vertices_wp, indices=faces_wp)
    normals_wp = tw.vertices.area_weighted_vertex_normals(
        int(vertices_wp.shape[0]), vertices_wp, faces_wp
    )
    diameter_np = tw.visibility.shape_diameter(
        mesh_wp, mesh_wp.points, normals=normals_wp, n_rays=128, cone_angle=np.deg2rad(10.0)
    ).numpy()

    # Vertices strictly *inside* one of the two x-walls: away from the y edges (where the smooth
    # normal is diagonal and the ray crosses the 1.41 diagonal instead) and away from the z ends.
    vertices_np = np.asarray(box_tm.vertices)
    on_wall_np = (
        (np.abs(np.abs(vertices_np[:, 0]) - 0.5) < 1e-6)
        & (np.abs(vertices_np[:, 1]) < 0.5 - 1e-6)
        & (np.abs(vertices_np[:, 2]) < 1.5)
    )
    assert on_wall_np.sum() > 10
    assert np.allclose(diameter_np[on_wall_np], 1.0, rtol=1e-2)


def test_shape_diameter_trimming_rejects_the_escaping_rays(device: str) -> None:
    """
    On a hollow shell the untrimmed mean is dragged out by the rays that cross the whole cavity.

    This is what the outlier rejection is *for*, so it has to be visible: with ``trim`` wide open
    the inner-shell diameters inflate well past the shell's own thickness.
    """
    outer_tm = tm.creation.icosphere(subdivisions=3, radius=1.0)
    inner_tm = tm.creation.icosphere(subdivisions=3, radius=0.8)
    inner_tm.invert()
    shell_tm = tm.util.concatenate([outer_tm, inner_tm])
    vertices_wp = wp.array(
        np.ascontiguousarray(shell_tm.vertices, dtype=np.float32), dtype=wp.vec3, device=device
    )
    faces_wp = wp.array(
        np.ascontiguousarray(shell_tm.faces.reshape(-1), dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )
    mesh_wp = wp.Mesh(points=vertices_wp, indices=faces_wp)
    normals_wp = tw.vertices.area_weighted_vertex_normals(
        int(vertices_wp.shape[0]), vertices_wp, faces_wp
    )
    trimmed_np = tw.visibility.shape_diameter(
        mesh_wp, mesh_wp.points, normals=normals_wp, n_rays=128, trim=1.0
    ).numpy()
    untrimmed_np = tw.visibility.shape_diameter(
        mesh_wp, mesh_wp.points, normals=normals_wp, n_rays=128, trim=100.0
    ).numpy()
    assert trimmed_np.mean() < untrimmed_np.mean()
    # The shell is 0.2 thick; trimming has to keep the outer wall near that, untrimmed does not.
    outer_np = np.linalg.norm(np.asarray(shell_tm.vertices), axis=1) > 0.9
    assert trimmed_np[outer_np].mean() < 0.5
    assert untrimmed_np[outer_np].mean() > trimmed_np[outer_np].mean()


@pytest.mark.parity("shape_diameter", "pymeshlab")
def test_shape_diameter_agrees_with_pymeshlab_on_which_part_is_thinner(device: str) -> None:
    """
    MeshLab's SDF differs from this one by roughly a constant factor, so compare *structure*.

    Its ``cone_amplitude`` parameter is a **no-op** in the 2025.07 build (byte-identical output at
    90 and 120 degrees) and its trimming is not the paper's, so neither a value comparison nor a
    per-vertex rank correlation is available. What both must agree on is the thing the field is
    *for*: on a dumbbell — two radius-1 balls joined by a radius-0.2 bar — the bar is thin and the
    balls are thick, which is the segmentation cue SDF exists to provide.

    Note this is *not* a test that either side reports the bar's diameter as 0.4. A cone of rays
    from a point on a slender bar mostly hits the bar's own walls, so SDF measures the local
    cross-section rather than any global extent — which is exactly why a 1 x 1 x 4 box reads ~1 at
    both ends and is useless as a fixture here.
    """
    ball_left_tm = tm.creation.icosphere(subdivisions=3, radius=1.0)
    ball_right_tm = tm.creation.icosphere(subdivisions=3, radius=1.0)
    ball_right_tm.apply_translation([4.0, 0.0, 0.0])
    bar_tm = tm.creation.cylinder(radius=0.2, height=5.0, sections=24)
    bar_tm.apply_transform(tm.transformations.rotation_matrix(np.pi / 2.0, [0.0, 1.0, 0.0]))
    bar_tm.apply_translation([2.0, 0.0, 0.0])
    # Subdivided after the union: the raw cylinder carries vertices only at its two end caps, which
    # the union buries inside the balls, leaving the bar's *surface* with nothing to measure on.
    dumbbell_tm = tm.boolean.union([ball_left_tm, ball_right_tm, bar_tm]).subdivide_to_size(0.25)

    meshset_pml = ml.MeshSet()
    meshset_pml.add_mesh(
        ml.Mesh(
            np.ascontiguousarray(dumbbell_tm.vertices, dtype=np.float64),
            np.ascontiguousarray(dumbbell_tm.faces, dtype=np.int32),
        )
    )
    meshset_pml.compute_scalar_by_shape_diameter_function_per_vertex(rays=256)
    diameter_pml = meshset_pml.current_mesh().vertex_scalar_array()

    vertices_wp = wp.array(
        np.ascontiguousarray(dumbbell_tm.vertices, dtype=np.float32), dtype=wp.vec3, device=device
    )
    faces_wp = wp.array(
        np.ascontiguousarray(dumbbell_tm.faces.reshape(-1), dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )
    mesh_wp = wp.Mesh(points=vertices_wp, indices=faces_wp)
    normals_wp = tw.vertices.area_weighted_vertex_normals(
        int(vertices_wp.shape[0]), vertices_wp, faces_wp
    )
    diameter_np = tw.visibility.shape_diameter(
        mesh_wp, mesh_wp.points, normals=normals_wp, n_rays=256
    ).numpy()

    vertices_np = np.asarray(dumbbell_tm.vertices)
    bar_np = (np.abs(vertices_np[:, 0] - 2.0) < 0.8) & (
        np.linalg.norm(vertices_np[:, 1:], axis=1) < 0.3
    )
    ball_np = vertices_np[:, 0] < -0.3  # only the left ball reaches there
    assert bar_np.sum() > 10
    assert ball_np.sum() > 10
    # Both must call the bar the thinner part. The margin is loose because MeshLab compresses the
    # contrast: it reads a 0.68 bar-to-ball ratio here where this port reads 0.55.
    for field_np in (diameter_np, diameter_pml):
        assert field_np[bar_np].mean() < 0.8 * field_np[ball_np].mean()


def test_shape_diameter_invalid(device: str) -> None:
    _sphere_tm, mesh_wp, normals_wp = _sphere_wp(device, 1.0, subdivisions=1)
    with pytest.raises(ValueError, match="n_rays >= 1"):
        tw.visibility.shape_diameter(mesh_wp, mesh_wp.points, n_rays=0)
    with pytest.raises(ValueError, match=r"cone_angle must be in \(0, pi / 2\]"):
        tw.visibility.shape_diameter(mesh_wp, mesh_wp.points, cone_angle=2.0)
    with pytest.raises(ValueError, match="trim must be non-negative"):
        tw.visibility.shape_diameter(mesh_wp, mesh_wp.points, trim=-1.0)
    with pytest.raises(ValueError, match="one entry per point"):
        tw.visibility.shape_diameter(
            mesh_wp, mesh_wp.points, normals=wp.zeros(2, dtype=wp.vec3, device=device)
        )
    assert normals_wp.shape[0] > 0


def test_shape_diameter_empty(device: str) -> None:
    _sphere_tm, mesh_wp, _normals_wp = _sphere_wp(device, 1.0, subdivisions=1)
    points_wp = wp.zeros(0, dtype=wp.vec3, device=device)
    assert tw.visibility.shape_diameter(mesh_wp, points_wp).shape == (0,)


# ---------------------------------------------------------------------------
# thickness (trimesh reference)
# ---------------------------------------------------------------------------


def test_thickness_max_sphere(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class A against ``trimesh.proximity.thickness``, including *which* points are infinite.

    The ``isfinite`` mask is compared before the values, so a point where one library finds no
    opposite surface and the other does is a failure rather than a skipped element. Measured 20 of
    20 finite on this fixture, so the guarded ``allclose`` does run -- the guard is there for a
    fixture where it would not, and is not silently making this test vacuous here.
    """
    mesh_tm, mesh_wp = icosahedron
    points_np, face_ids = tm.sample.sample_surface(mesh_tm, 20, seed=7)
    normals_np = mesh_tm.face_normals[face_ids]

    points_wp = wp.array(
        np.ascontiguousarray(points_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    normals_wp = wp.array(
        np.ascontiguousarray(normals_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )

    thickness_wp = tw.visibility.thickness(mesh_wp, points_wp, normals=normals_wp).numpy()
    thickness_tm = tm_proximity.thickness(mesh_tm, points_np, normals=normals_np)

    finite_tm = np.isfinite(thickness_tm)
    assert np.array_equal(np.isfinite(thickness_wp), finite_tm)
    if finite_tm.any():
        assert np.allclose(thickness_wp[finite_tm], thickness_tm[finite_tm], rtol=1e-5, atol=1e-5)


def test_thickness_ray(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class A on the ``method="ray"`` branch, at ``allclose``'s default tolerance.

    A separate algorithm from ``max_sphere`` rather than a tuning of it, so it needs its own
    comparison; 20 of 20 points finite here too.
    """
    mesh_tm, mesh_wp = icosahedron
    points_np, face_ids = tm.sample.sample_surface(mesh_tm, 20, seed=13)
    normals_np = mesh_tm.face_normals[face_ids]

    points_wp = wp.array(
        np.ascontiguousarray(points_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    normals_wp = wp.array(
        np.ascontiguousarray(normals_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )

    thickness_wp = tw.visibility.thickness(
        mesh_wp, points_wp, normals=normals_wp, method="ray"
    ).numpy()
    thickness_tm = tm_proximity.thickness(mesh_tm, points_np, normals=normals_np, method="ray")

    finite_tm = np.isfinite(thickness_tm)
    assert np.array_equal(np.isfinite(thickness_wp), finite_tm)
    if finite_tm.any():
        assert np.allclose(thickness_wp[finite_tm], thickness_tm[finite_tm])


# ---------------------------------------------------------------------------
# max_tangent_sphere (trimesh reference)
# ---------------------------------------------------------------------------


def test_max_tangent_sphere(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class A on both returns, at ``1e-2`` -- the loosest tolerance in this file, and why.

    The sphere's centre slides along the normal as its radius grows, so a small radius disagreement
    displaces the centre by the same amount; both are compared rather than just the radius, since a
    centre off the normal would be a different failure. The tolerance is set by
    ``mesh_query_point``'s own accuracy (section 6 records it as up to 2.1e-5 absolute) amplified by
    that sliding, not by a disagreement about the definition. 20 of 20 points finite.
    """
    mesh_tm, mesh_wp = icosahedron
    points_np, face_ids = tm.sample.sample_surface(mesh_tm, 20, seed=42)
    normals_np = mesh_tm.face_normals[face_ids]

    points_wp = wp.array(
        np.ascontiguousarray(points_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    normals_wp = wp.array(
        np.ascontiguousarray(normals_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )

    centers_wp, radii_wp = tw.visibility.max_tangent_sphere(mesh_wp, points_wp, normals=normals_wp)
    centers_tm, radii_tm = tm_proximity.max_tangent_sphere(mesh_tm, points_np, normals=normals_np)

    finite_tm = np.isfinite(radii_tm)
    assert np.array_equal(np.isfinite(radii_wp.numpy()), finite_tm)
    if finite_tm.any():
        assert np.allclose(radii_wp.numpy()[finite_tm], radii_tm[finite_tm], rtol=1e-2, atol=1e-2)
        assert np.allclose(
            centers_wp.numpy()[finite_tm], centers_tm[finite_tm], rtol=1e-2, atol=1e-2
        )


def test_max_tangent_sphere_empty(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = icosahedron
    points_wp = wp.empty(0, dtype=wp.vec3, device=mesh_wp.device)
    centers_wp, radii_wp = tw.visibility.max_tangent_sphere(mesh_wp, points_wp)
    assert centers_wp.shape == (0,)
    assert radii_wp.shape == (0,)


@pytest.mark.parametrize("kernel_device", ["cpu", "cuda:0"])
def test_max_tangent_sphere_agrees_across_devices(kernel_device: str) -> None:
    """
    Pin the packed support-argmax reduction, which had the same one-lane-per-block CPU bug.

    Query points are pulled *inside* the surface on purpose: with them exactly on it the
    shrinking-sphere iteration is ill-conditioned (the sphere collapses to a 0.0018 radius on a unit
    sphere) and the two devices then differ by 23% from float ordering alone, which would make this
    a test of that degeneracy rather than of the reduction.
    """
    if kernel_device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")

    mesh_tm = tm.creation.icosphere(subdivisions=2)
    mesh_wp = trimesh_to_warp(mesh_tm, kernel_device)
    vertices_np = np.asarray(mesh_tm.vertices)
    points_wp = wp.array(
        np.ascontiguousarray(vertices_np * 0.95), dtype=wp.vec3, device=kernel_device
    )
    normals_wp = wp.array(
        np.ascontiguousarray(np.asarray(mesh_tm.vertex_normals)),
        dtype=wp.vec3,
        device=kernel_device,
    )
    _centers_wp, radii_wp = tw.visibility.max_tangent_sphere(mesh_wp, points_wp, normals=normals_wp)
    # The inscribed tangent sphere of a unit sphere, from just inside it, is the sphere itself.
    assert np.allclose(radii_wp.numpy(), 0.95, rtol=0.1, atol=0.1)
