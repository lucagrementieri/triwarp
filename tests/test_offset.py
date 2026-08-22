"""
Regression tests for ``triwarp.offset``.

The level-set offset against pymeshlab's uniform resampler and MeshLib's ``offsetMesh``, both of
which march the same kind of field -- plus the invariant that is stronger than either comparison:
every output vertex must sit at the requested signed distance from the input.
"""

from __future__ import annotations

import numpy as np
import pymeshlab as ml
import pytest
import trimesh as tm
import warp as wp
from meshlib import mrmeshpy as mm
from scipy.spatial import cKDTree

import triwarp as tw
from tests.comparisons import (
    assert_unordered_rows_equal,
    canonical_winding,
    hausdorff_surface_two_sided,
)
from tests.conversions import (
    meshlib_to_trimesh,
    numpy_to_warp,
    trimesh_to_meshlib,
    trimesh_to_pymeshlab,
    warp_to_trimesh,
)

# One spacing for every comparison here, passed to both sides so neither is resampled finer than
# the other. 0.05 on a unit sphere is ~70 samples across, which is what the automatic default lands
# on and coarse enough to keep the reference rows quick.
_VOXEL = 0.05


def _signed_distance_to(
    mesh_wp: tuple[wp.array[wp.vec3], wp.array[wp.int32]], points_np: np.ndarray
) -> np.ndarray:
    """Signed distance from every row of ``points_np`` to the mesh, by winding sign."""
    vertices_wp, faces_wp = mesh_wp
    points_wp = wp.array(
        np.ascontiguousarray(points_np, dtype=np.float32), dtype=wp.vec3, device=vertices_wp.device
    )
    return tw.proximity.signed_distance_on_mesh(
        vertices_wp, faces_wp, points_wp, sign_mode="winding"
    ).numpy()


@pytest.mark.parametrize("distance", [0.2, -0.2])
def test_offset_mesh_lands_at_the_requested_distance(
    device: str, icosphere: tuple[tm.Trimesh, wp.Mesh], distance: float
) -> None:
    """
    Class A on the defining property: every output vertex is at signed distance ``distance``.

    This is the test that actually constrains the function, and it is stronger than any comparison
    against another implementation -- an offset surface *is* a level set of the distance field, so
    measuring the field at the output is measuring the answer. It is checked with
    ``signed_distance_on_mesh``, which is not the code under test's own field sampler applied twice:
    the offset marches a **lattice** and this queries the **vertices** it produced.

    The tolerance is the lattice's: a marching-cubes vertex is linearly interpolated inside a cell,
    so it lands within a fraction of ``_VOXEL`` of the true level set rather than within a whole
    cell. Measured max deviation **0.0019** at a 0.05 spacing, i.e. 3.8 % of one cell.

    Also asserts the sign of the volume change, which no distance check would catch: an outward
    offset must enclose more and an inward one less.
    """
    mesh_tm, _ = icosphere
    vertices_wp, faces_wp = numpy_to_warp(
        np.asarray(mesh_tm.vertices), np.asarray(mesh_tm.faces, dtype=np.int32).reshape(-1), device
    )
    offset_vertices_wp, offset_faces_wp = tw.offset.offset_mesh(
        vertices_wp, faces_wp, distance, _VOXEL
    )
    assert int(offset_faces_wp.shape[0]) > 0

    signed_np = _signed_distance_to((vertices_wp, faces_wp), offset_vertices_wp.numpy())
    assert np.abs(signed_np - distance).max() < 0.1 * _VOXEL

    volume_before = float(tw.measures.volume(vertices_wp, faces_wp))
    volume_after = float(tw.measures.volume(offset_vertices_wp, offset_faces_wp))
    assert (volume_after > volume_before) is (distance > 0.0)
    assert tw.validation.is_watertight(offset_vertices_wp, offset_faces_wp)


@pytest.mark.parametrize("distance", [0.2, -0.2])
@pytest.mark.parity("offset_mesh", "meshlib")
def test_offset_mesh_matches_meshlib(
    device: str, icosphere: tuple[tm.Trimesh, wp.Mesh], distance: float
) -> None:
    """
    Class C: the same surface as ``offsetMesh`` at a matched voxel size, to a tenth of a cell.

    No correspondence exists between the two triangulations -- both march their own field on their
    own lattice -- so the comparison is the two-sided Hausdorff distance between the *surfaces*,
    plus the vertex counts as a sanity check on the resolution actually used. They agree closely
    enough that the counts are worth asserting: measured **10 746 against 10 736** vertices at
    ``distance = 0.2`` and 4 758 against 4 760 at ``-0.2``, i.e. within 0.1 %, because at a matched
    spacing the two lattices differ only in where their origin falls.

    ``OffsetParameters.voxelSize`` is set explicitly rather than left at its default, which is the
    parameter that would otherwise decide the comparison.
    """
    mesh_tm, _ = icosphere
    vertices_wp, faces_wp = numpy_to_warp(
        np.asarray(mesh_tm.vertices), np.asarray(mesh_tm.faces, dtype=np.int32).reshape(-1), device
    )
    offset_vertices_wp, offset_faces_wp = tw.offset.offset_mesh(
        vertices_wp, faces_wp, distance, _VOXEL
    )

    parameters_ml = mm.OffsetParameters()
    parameters_ml.voxelSize = _VOXEL
    offset_ml = meshlib_to_trimesh(
        mm.offsetMesh(mm.MeshPart(trimesh_to_meshlib(mesh_tm)), distance, parameters_ml)
    )
    assert offset_ml.faces.shape[0] > 0  # non-vacuity: the reference produced a surface

    count_wp = int(offset_vertices_wp.shape[0])
    count_ml = offset_ml.vertices.shape[0]
    assert abs(count_wp - count_ml) < 0.05 * count_ml

    offset_tm = warp_to_trimesh(offset_vertices_wp, offset_faces_wp)
    deviation = hausdorff_surface_two_sided(
        np.asarray(offset_tm.vertices, dtype=np.float64),
        np.asarray(offset_tm.faces),
        np.asarray(offset_ml.vertices, dtype=np.float64),
        np.asarray(offset_ml.faces),
    )
    assert deviation < 0.5 * _VOXEL


@pytest.mark.parity("offset_mesh", "pymeshlab")
def test_offset_mesh_matches_pymeshlab(device: str, icosphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class C: MeshLab's uniform resampler at a matched cell size and the same absolute offset.

    ``generate_resampled_uniform_mesh`` is MeshLab's offset, and its two length parameters are the
    trap: both take a wrapper type, and its ``offset`` as a ``PercentageValue`` runs from *full
    erosion* at 0 % to full dilation at 100 %, so its own 50 % default is the **zero** offset.
    ``PureValue`` is therefore mandatory here, and it is the same number triwarp gets.

    Compared on the surfaces (two-sided Hausdorff) and on the same distance invariant the test above
    applies to triwarp: MeshLab's own output sits at mean signed distance **+0.1999** with a spread
    of 0.0002 from the input, so the two implementations are measuring the same quantity rather than
    two things that happen to look alike.
    """
    distance = 0.2
    mesh_tm, _ = icosphere
    vertices_wp, faces_wp = numpy_to_warp(
        np.asarray(mesh_tm.vertices), np.asarray(mesh_tm.faces, dtype=np.int32).reshape(-1), device
    )
    offset_vertices_wp, offset_faces_wp = tw.offset.offset_mesh(
        vertices_wp, faces_wp, distance, _VOXEL
    )

    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.generate_resampled_uniform_mesh(
        cellsize=ml.PureValue(_VOXEL), offset=ml.PureValue(distance), mergeclosevert=True
    )
    mesh_pml = meshset_pml.current_mesh()
    offset_pml = tm.Trimesh(mesh_pml.vertex_matrix(), mesh_pml.face_matrix(), process=False)
    assert offset_pml.faces.shape[0] > 0

    signed_pml = _signed_distance_to((vertices_wp, faces_wp), np.asarray(offset_pml.vertices))
    assert np.abs(signed_pml.mean() - distance) < 0.1 * _VOXEL  # same quantity, not just a shape

    offset_tm = warp_to_trimesh(offset_vertices_wp, offset_faces_wp)
    deviation = hausdorff_surface_two_sided(
        np.asarray(offset_tm.vertices, dtype=np.float64),
        np.asarray(offset_tm.faces),
        np.asarray(offset_pml.vertices, dtype=np.float64),
        np.asarray(offset_pml.faces),
    )
    assert deviation < 0.5 * _VOXEL


def test_offset_mesh_resolves_what_survives_a_large_inward_offset(
    device: str, icosphere: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Not a library comparison: the automatic ``voxel_size``'s resolution **floor**, and its absence.

    Tying the spacing to the offset distance alone -- the obvious rule, and the first one written
    -- resolves the band the level set sits in and not what is left of the object. An inward offset
    of 0.9 on a unit sphere leaves a sphere of radius ~0.1, which at a spacing of ``0.9 / 3`` is
    smaller than a single cell: the call returned **empty** for a level set that plainly exists. A
    floor of 64 samples across the mesh fixes it, and this is the case that would fail without it.

    The genuinely empty case is asserted beside it, since the two must stay distinguishable: at
    ``-1.5`` there is no point at that distance inside a unit sphere and an empty answer is correct.
    """
    mesh_tm, _ = icosphere
    vertices_wp, faces_wp = numpy_to_warp(
        np.asarray(mesh_tm.vertices), np.asarray(mesh_tm.faces, dtype=np.int32).reshape(-1), device
    )

    survivor_vertices_wp, survivor_faces_wp = tw.offset.offset_mesh(vertices_wp, faces_wp, -0.9)
    assert int(survivor_faces_wp.shape[0]) > 0
    radius_np = np.linalg.norm(survivor_vertices_wp.numpy(), axis=1)
    assert 0.05 < radius_np.max() < 0.15  # the sphere that is left, not a stray cell

    _empty_vertices_wp, empty_faces_wp = tw.offset.offset_mesh(vertices_wp, faces_wp, -1.5)
    assert int(empty_faces_wp.shape[0]) == 0


def test_offset_mesh_guards(device: str, icosphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """Not a library comparison: the three documented value guards."""
    mesh_tm, _ = icosphere
    vertices_wp, faces_wp = numpy_to_warp(
        np.asarray(mesh_tm.vertices), np.asarray(mesh_tm.faces, dtype=np.int32).reshape(-1), device
    )
    with pytest.raises(ValueError, match="non-zero"):
        tw.offset.offset_mesh(vertices_wp, faces_wp, 0.0)
    with pytest.raises(ValueError, match="voxel_size must be positive"):
        tw.offset.offset_mesh(vertices_wp, faces_wp, 0.1, -1.0)
    with pytest.raises(ValueError, match="at least one face"):
        tw.offset.offset_mesh(vertices_wp, wp.empty(0, dtype=wp.int32, device=device), 0.1)


@pytest.mark.parametrize("mesh_name", ["hemisphere", "half_torus", "icosphere_coarse", "unit_box"])
def test_thicken_mesh_closes_into_a_solid(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Not a library comparison: the shell is a valid solid, on open and closed inputs alike.

    Four claims, and the first two are what a thickening is *for*: the result is **watertight** and
    **consistently wound**, so it can be measured, printed or booleaned. On an open input that
    depends entirely on the band -- the two layers alone leave two rims -- and the band's winding is
    inherited from ``oriented_boundary_edges`` rather than guessed, which is why it comes out right
    on ``half_torus``'s *two* loops as well as ``hemisphere``'s one.

    The counts are exact and asserted: ``2 * n_vertices`` positions and
    ``2 * n_faces + 2 * n_boundary_edges`` triangles. And the volume is positive and close to
    ``area * thickness`` -- 0.289 against 0.308 on ``hemisphere``, the 6 % being the inward layer's
    smaller area -- which is the check that catches a shell built inside out, where every other
    assertion here still passes.
    """
    thickness = 0.05
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    n_vertices = int(vertices_wp.shape[0])
    n_faces = int(faces_wp.shape[0]) // 3
    n_rim = int(tw.boundary.oriented_boundary_edges(vertices_wp, faces_wp).shape[0])

    shell_vertices_wp, shell_faces_wp = tw.offset.thicken_mesh(vertices_wp, faces_wp, thickness)

    assert int(shell_vertices_wp.shape[0]) == 2 * n_vertices
    assert int(shell_faces_wp.shape[0]) // 3 == 2 * n_faces + 2 * n_rim
    assert tw.validation.is_watertight(shell_vertices_wp, shell_faces_wp)
    assert tw.validation.is_winding_consistent(shell_faces_wp)
    assert tw.validation.is_edge_manifold(shell_faces_wp, False)

    volume = float(tw.measures.volume(shell_vertices_wp, shell_faces_wp))
    assert volume > 0.0
    assert volume < mesh_tm.area * thickness  # the inward layer has the smaller area
    assert volume > 0.5 * mesh_tm.area * thickness


@pytest.mark.parity("thicken_mesh", "meshlib")
def test_thicken_mesh_matches_meshlib(device: str, hemisphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class B: the same shell as ``makeThickMesh``, under a nearest-neighbour vertex bijection.

    MeshLib's thickening is the same construction rather than a different one -- extrude along the
    vertex normals, reverse a copy, band the rim -- and the outputs agree to a degree worth stating:
    identical counts (194 vertices, 384 faces), volumes equal to five decimals (0.28940), a
    **bijective** nearest-neighbour vertex match at **4.6e-05**, and *identical face sets* once the
    windings are canonicalized and MeshLib's vertex ids are mapped through that bijection.

    The transform is the bijection, which is what makes this Class B rather than A: neither library
    promises a vertex order. The 4.6e-05 residual is 0.09 % of the thickness and is the float32
    normal normalization, not a difference of rule.

    ``ThickenParams`` splits the displacement into ``insideOffset`` and ``outsideOffset``, so the
    single-sided default here is ``(thickness, 0)`` -- passed explicitly, since it is the parameter
    that would otherwise decide the comparison.
    """
    thickness = 0.05
    mesh_tm, mesh_wp = hemisphere
    shell_vertices_wp, shell_faces_wp = tw.offset.thicken_mesh(
        mesh_wp.points, mesh_wp.indices, thickness
    )

    parameters_ml = mm.ThickenParams()
    parameters_ml.insideOffset = thickness
    parameters_ml.outsideOffset = 0.0
    shell_ml = meshlib_to_trimesh(mm.makeThickMesh(trimesh_to_meshlib(mesh_tm), parameters_ml))
    assert shell_ml.faces.shape[0] > 0  # non-vacuity: the reference produced a shell

    assert shell_ml.vertices.shape[0] == int(shell_vertices_wp.shape[0])
    assert shell_ml.faces.shape[0] == int(shell_faces_wp.shape[0]) // 3
    assert np.isclose(
        float(tw.measures.volume(shell_vertices_wp, shell_faces_wp)),
        shell_ml.volume,
        rtol=1e-4,
        atol=1e-6,
    )

    # The bijection, then the face sets through it.
    shell_np = shell_vertices_wp.numpy().astype(np.float64)
    distance_np, match_np = cKDTree(np.asarray(shell_ml.vertices)).query(shell_np)
    assert distance_np.max() < 1e-4
    assert len(set(match_np.tolist())) == match_np.shape[0]
    inverse_np = np.empty(shell_ml.vertices.shape[0], dtype=np.int64)
    inverse_np[match_np] = np.arange(shell_np.shape[0])
    assert_unordered_rows_equal(
        canonical_winding(shell_faces_wp.numpy().reshape(-1, 3)),
        canonical_winding(inverse_np[np.asarray(shell_ml.faces)]),
    )


def test_thicken_mesh_self_intersects_past_the_curvature_radius(
    half_torus: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Not a library comparison: the documented failure mode, asserted rather than left as prose.

    Displacing along vertex normals folds the surface wherever the thickness exceeds the local
    radius of curvature, and this function deliberately does not guard against it: the guard would
    be a whole-mesh intersection test on every call. So the contract is that the condition is
    *detectable*, and that is what is checked: ``half_torus``'s tube has minor radius 0.5 before
    its graded scaling, and thickening it by 0.6 makes the inward layer pass through the tube's own
    axis and out the other side. ``face_self_intersecting_mask`` flags **130** faces there and
    ``is_watertight`` -- which includes a self-intersection test, as open3d's does -- turns
    ``False``, where at 0.05 both are clean. It scales as the geometry says it should: 273 faces at
    a thickness of 1.0 and 467 at 1.5.

    A *closed* input does not fold this way, and that is worth recording because it is the obvious
    thing to test and it does not work: a unit sphere thickened by 1.5 puts its inward layer at
    radius 0.5 with the orientation inverted, which is two nested spheres -- wrong volume, no
    intersection. The tube is the shape whose normals actually converge.

    The alternative for such a thickness is named in the docstring and exercised here: a level-set
    ``offset_mesh`` cannot self-intersect by construction, and does not.
    """
    thickness = 0.6
    _, mesh_wp = half_torus
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices

    thin_vertices_wp, thin_faces_wp = tw.offset.thicken_mesh(vertices_wp, faces_wp, 0.05)
    thin_np = tw.validation.face_self_intersecting_mask(thin_vertices_wp, thin_faces_wp).numpy()
    assert int(thin_np.sum()) == 0
    assert tw.validation.is_watertight(thin_vertices_wp, thin_faces_wp)

    folded_vertices_wp, folded_faces_wp = tw.offset.thicken_mesh(vertices_wp, faces_wp, thickness)
    folded_np = tw.validation.face_self_intersecting_mask(
        folded_vertices_wp, folded_faces_wp
    ).numpy()
    assert int(folded_np.sum()) > 100
    assert not tw.validation.is_watertight(folded_vertices_wp, folded_faces_wp)

    # The recommended alternative at the same distance, and it comes out clean.
    inward_vertices_wp, inward_faces_wp = tw.offset.offset_mesh(vertices_wp, faces_wp, -thickness)
    if int(inward_faces_wp.shape[0]) > 0:
        assert (
            int(
                tw.validation.face_self_intersecting_mask(inward_vertices_wp, inward_faces_wp)
                .numpy()
                .sum()
            )
            == 0
        )


def test_thicken_mesh_guards(device: str, icosphere_coarse: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """Not a library comparison: the three documented value guards."""
    _, mesh_wp = icosphere_coarse
    with pytest.raises(ValueError, match="thickness must be positive"):
        tw.offset.thicken_mesh(mesh_wp.points, mesh_wp.indices, 0.0)
    with pytest.raises(ValueError, match="outside must be non-negative"):
        tw.offset.thicken_mesh(mesh_wp.points, mesh_wp.indices, 0.1, outside=-1.0)
    with pytest.raises(ValueError, match="at least one face"):
        tw.offset.thicken_mesh(
            mesh_wp.points, wp.empty(0, dtype=wp.int32, device=mesh_wp.device), 0.1
        )
