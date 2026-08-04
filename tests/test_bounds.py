"""Regression tests for ``triwarp.bounds`` against trimesh, open3d and libigl."""

from __future__ import annotations

import igl
import numpy as np
import pytest
import warp as wp

import triwarp as tw
from tests.conversions import trimesh_to_open3d

_MESHES = ["icosahedron", "cave_cube", "hemisphere", "half_torus"]


def _bounds_np(lower_wp: wp.vec3, upper_wp: wp.vec3) -> np.ndarray:
    """Stack triwarp's two corner ``wp.vec3`` into the ``(2, 3)`` layout every reference uses."""
    return np.stack(
        [
            np.array([lower_wp.x, lower_wp.y, lower_wp.z]),
            np.array([upper_wp.x, upper_wp.y, upper_wp.z]),
        ]
    )


@pytest.mark.parametrize("mesh_name", _MESHES)
@pytest.mark.parity("aabb_bounds", "trimesh", "open3d", "igl")
def test_aabb_bounds_matches_trimesh_open3d_and_igl(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    The axis-aligned bounding box against all three references: class A on two, class B on igl.

    Trivial to compute and trivial to get subtly wrong -- a reduction that seeds its accumulator at
    zero rather than at +/-inf returns a box clamped to the origin, which is correct for any mesh
    straddling it and wrong for every mesh that does not. Every fixture here is translated away from
    the origin, so that bug would show.

    The benchmark's trimesh row is the uncached ``vstack((v.min(0), v.max(0)))`` formula behind
    ``Trimesh.bounds`` rather than the cached property, and that formula is what is compared here.

    **igl returns the box as geometry, not as two corners.** ``igl.bounding_box(V)`` gives the
    ``2**3 = 8`` corner *vertices* plus the 12 triangles of the box hull, so the named transform is
    to reduce those 8 corners back to a min/max pair. That is a genuinely different output shape for
    the same answer, and reducing it is exact.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    lower_wp, upper_wp = tw.bounds.aabb_bounds(mesh_wp.points)
    bounds_wp = _bounds_np(lower_wp, upper_wp)

    bounds_np = np.vstack((mesh_tm.vertices.min(axis=0), mesh_tm.vertices.max(axis=0)))
    assert np.allclose(bounds_wp, bounds_np, rtol=1e-5, atol=1e-5)

    box_o3d = trimesh_to_open3d(mesh_tm).get_axis_aligned_bounding_box()
    bounds_o3d = np.stack([box_o3d.get_min_bound(), box_o3d.get_max_bound()])
    assert np.allclose(bounds_wp, bounds_o3d, rtol=1e-5, atol=1e-5)

    corners_igl, faces_igl = igl.bounding_box(
        np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    )
    assert corners_igl.shape == (8, 3), "the box comes back as its 8 corner vertices"
    assert faces_igl.shape == (12, 3), "and the 12 triangles of its hull"
    bounds_igl = np.stack([corners_igl.min(axis=0), corners_igl.max(axis=0)])
    assert np.allclose(bounds_wp, bounds_igl, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", _MESHES)
@pytest.mark.parity("aabb_diagonal", "igl")
def test_aabb_diagonal_matches_igl(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A: the bbox diagonal length, against ``igl.bounding_box_diagonal``.

    igl computes the box itself internally where triwarp takes the two corners as arguments, so
    this also checks the composition ``aabb_diagonal(*aabb_bounds(...))`` a caller actually writes
    -- the only path by which triwarp produces this number.

    The fixtures span a cube-like solid and two thin open surfaces, so no single axis dominates the
    answer on all of them: an implementation returning the longest *extent* rather than the diagonal
    would match on none.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    diagonal_igl = igl.bounding_box_diagonal(
        np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    )

    diagonal_wp = tw.bounds.aabb_diagonal(*tw.bounds.aabb_bounds(mesh_wp.points))

    assert np.isclose(diagonal_wp, diagonal_igl, rtol=1e-5, atol=1e-5)
    # Not merely the longest extent: on these fixtures the two differ by more than the tolerance.
    extents_np = mesh_tm.vertices.max(axis=0) - mesh_tm.vertices.min(axis=0)
    assert diagonal_wp > float(extents_np.max()) * (1.0 + 1e-3)


def test_aabb_bounds_single_point(device: str) -> None:
    """One point is a degenerate box: both corners land on it and the diagonal is zero."""
    points_wp = wp.array(
        np.array([[1.5, -2.5, 3.5]], dtype=np.float32), dtype=wp.vec3, device=device
    )
    lower_wp, upper_wp = tw.bounds.aabb_bounds(points_wp)

    assert np.allclose(_bounds_np(lower_wp, upper_wp), [[1.5, -2.5, 3.5]] * 2, rtol=1e-6)
    assert tw.bounds.aabb_diagonal(lower_wp, upper_wp) == 0.0


def test_aabb_union_encloses_both_boxes() -> None:
    """The union takes the component-wise min of the mins and max of the maxes, nothing more."""
    a_min, a_max = wp.vec3(-1.0, 0.0, 2.0), wp.vec3(1.0, 3.0, 4.0)
    b_min, b_max = wp.vec3(0.0, -5.0, 3.0), wp.vec3(2.0, 1.0, 3.5)

    union_min, union_max = tw.bounds.aabb_union(a_min, a_max, b_min, b_max)

    assert np.allclose([union_min.x, union_min.y, union_min.z], [-1.0, -5.0, 2.0])
    assert np.allclose([union_max.x, union_max.y, union_max.z], [2.0, 3.0, 4.0])
    # Idempotent, and each input box is contained in the result.
    for box_min, box_max in ((a_min, a_max), (b_min, b_max)):
        again_min, again_max = tw.bounds.aabb_union(union_min, union_max, box_min, box_max)
        assert np.allclose([again_min.x, again_min.y, again_min.z], [-1.0, -5.0, 2.0])
        assert np.allclose([again_max.x, again_max.y, again_max.z], [2.0, 3.0, 4.0])


def test_aabb_union_matches_a_pooled_reduction(device: str) -> None:
    """
    Unioning two clouds' boxes must equal the box of the concatenated cloud.

    That is the property callers rely on when they accumulate bounds incrementally, and the one
    thing a per-component slip would break while leaving both boxes individually plausible.
    """
    rng = np.random.default_rng(5)
    cloud_a = rng.normal(size=(200, 3)).astype(np.float32) + 4.0
    cloud_b = rng.normal(size=(300, 3)).astype(np.float32) - 2.0

    boxes = [
        tw.bounds.aabb_bounds(wp.array(np.ascontiguousarray(cloud), dtype=wp.vec3, device=device))
        for cloud in (cloud_a, cloud_b)
    ]
    union_min, union_max = tw.bounds.aabb_union(*boxes[0], *boxes[1])

    pooled_min, pooled_max = tw.bounds.aabb_bounds(
        wp.array(np.ascontiguousarray(np.vstack([cloud_a, cloud_b])), dtype=wp.vec3, device=device)
    )
    assert np.allclose(_bounds_np(union_min, union_max), _bounds_np(pooled_min, pooled_max))
