"""Regression tests for ``triwarp.bounds`` against trimesh, open3d and libigl."""

from __future__ import annotations

from typing import Literal

import igl
import numpy as np
import open3d as o3d
import pytest
import pyvista as pv
import trimesh as tm
import warp as wp
from meshlib import mrmeshpy as mm

import triwarp as tw
from tests.conversions import trimesh_to_meshlib, trimesh_to_open3d, trimesh_to_pyvista

_MESHES = ["icosahedron", "cave_cube", "hemisphere", "half_torus"]

_Objective = Literal["volume", "surface_area", "diagonal"]
_OBJECTIVES: list[_Objective] = ["volume", "surface_area", "diagonal"]


def _bounds_np(lower_wp: wp.vec3, upper_wp: wp.vec3) -> np.ndarray:
    """Stack triwarp's two corner ``wp.vec3`` into the ``(2, 3)`` layout every reference uses."""
    return np.stack(
        [
            np.array([lower_wp.x, lower_wp.y, lower_wp.z]),
            np.array([upper_wp.x, upper_wp.y, upper_wp.z]),
        ]
    )


@pytest.mark.parametrize("mesh_name", _MESHES)
@pytest.mark.parity("aabb", "trimesh", "open3d", "igl", "meshlib")
def test_aabb_matches_trimesh_open3d_and_igl(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    The axis-aligned bounding box against all four references: class A on three, class B on igl.

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

    MeshLib's ``computeBoundingBox`` is class A and returns a ``Box3f`` -- ``.min`` / ``.max``, the
    two corners directly. Its ``region`` argument is passed ``None`` for the whole mesh; note that
    it takes the *topology* as well as the points, so on a mesh with unreferenced vertices it would
    box only the referenced ones, where triwarp's takes the point buffer alone. Every fixture here
    references every vertex, so the two coincide.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    lower_wp, upper_wp = tw.bounds.aabb(mesh_wp.points)
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

    mesh_ml = trimesh_to_meshlib(mesh_tm)
    box_ml = mm.computeBoundingBox(mesh_ml.topology, mesh_ml.points, None)
    bounds_ml = np.stack([[*box_ml.min], [*box_ml.max]])
    assert np.allclose(bounds_wp, bounds_ml, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", _MESHES)
@pytest.mark.parity("aabb", "pyvista")
@pytest.mark.parity("enclosing_diagonal", "pyvista")
def test_aabb_and_diagonal_match_pyvista(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A on both, from the same VTK pair: ``DataSet.bounds`` and ``DataSet.length``.

    One named transform, and it is a layout rather than a value: pyvista returns the box
    **interleaved per axis** as ``(xmin, xmax, ymin, ymax, zmin, zmax)`` where every other reference
    in this module returns two corners, so it is reshaped to ``(3, 2)`` and transposed. Reading it
    as two corners without that step gives a plausible-looking box that is wrong on any mesh whose
    extents differ.

    **``DataSet.center`` is not a centroid** and is deliberately not compared here: it is the bbox
    midpoint, so it belongs to this quantity and not to ``totals.surface_centroid``.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mesh_pv = trimesh_to_pyvista(mesh_tm)

    lower_wp, upper_wp = tw.bounds.aabb(mesh_wp.points)
    bounds_pv = np.asarray(mesh_pv.bounds).reshape(3, 2).T
    assert np.allclose(_bounds_np(lower_wp, upper_wp), bounds_pv, rtol=1e-5, atol=1e-5)

    assert np.isclose(
        tw.bounds.enclosing_diagonal(mesh_wp.points), float(mesh_pv.length), rtol=1e-5, atol=1e-5
    )
    # The bbox midpoint, which is what pyvista's ``center`` is -- not the surface centroid.
    assert np.allclose(np.asarray(mesh_pv.center), bounds_pv.mean(axis=0), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", _MESHES)
@pytest.mark.parity("enclosing_diagonal", "igl")
def test_enclosing_diagonal_single_cloud_matches_igl(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A: the bbox diagonal length, against ``igl.bounding_box_diagonal``.

    Both sides compute the box internally from one point set: ``enclosing_diagonal`` with no
    ``other`` is the single-cloud form, and the only path by which triwarp produces this number.

    The fixtures span a cube-like solid and two thin open surfaces, so no single axis dominates the
    answer on all of them: an implementation returning the longest *extent* rather than the diagonal
    would match on none.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    diagonal_igl = igl.bounding_box_diagonal(
        np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    )

    diagonal_wp = tw.bounds.enclosing_diagonal(mesh_wp.points)

    assert np.isclose(diagonal_wp, diagonal_igl, rtol=1e-5, atol=1e-5)
    # Not merely the longest extent: on these fixtures the two differ by more than the tolerance.
    extents_np = mesh_tm.vertices.max(axis=0) - mesh_tm.vertices.min(axis=0)
    assert diagonal_wp > float(extents_np.max()) * (1.0 + 1e-3)


def test_aabb_single_point(device: str) -> None:
    """One point is a degenerate box: both corners land on it and the diagonal is zero."""
    points_wp = wp.array(
        np.array([[1.5, -2.5, 3.5]], dtype=np.float32), dtype=wp.vec3, device=device
    )
    lower_wp, upper_wp = tw.bounds.aabb(points_wp)

    assert np.allclose(_bounds_np(lower_wp, upper_wp), [[1.5, -2.5, 3.5]] * 2, rtol=1e-6)
    assert tw.bounds.enclosing_diagonal(points_wp) == 0.0


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
        tw.bounds.aabb(wp.array(np.ascontiguousarray(cloud), dtype=wp.vec3, device=device))
        for cloud in (cloud_a, cloud_b)
    ]
    union_min, union_max = tw.bounds.aabb_union(*boxes[0], *boxes[1])

    pooled_min, pooled_max = tw.bounds.aabb(
        wp.array(np.ascontiguousarray(np.vstack([cloud_a, cloud_b])), dtype=wp.vec3, device=device)
    )
    assert np.allclose(_bounds_np(union_min, union_max), _bounds_np(pooled_min, pooled_max))


@pytest.mark.parametrize("mesh_name", _MESHES)
@pytest.mark.parity("enclosing_diagonal", "igl")
def test_enclosing_diagonal_matches_igl_on_the_pooled_cloud(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B: equal to ``igl.bounding_box_diagonal`` of the two clouds *stacked*.

    The named transform is the stacking -- igl takes one point set, so the two-set signature maps
    onto it by concatenating, which is exactly the quantity ``enclosing_diagonal`` is defined as.
    Non-vacuous by construction: the queries are pushed outside the mesh's own box along every axis,
    so the union box is strictly larger than either input's and a one-sided implementation
    (measuring ``points`` alone, or ``other`` alone) fails.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.asarray(mesh_tm.vertices, dtype=np.float64)
    rng = np.random.default_rng(11)
    # Offset by the full extent so no query lands inside the mesh's box.
    extent_np = vertices_np.max(axis=0) - vertices_np.min(axis=0)
    queries_np = rng.random((64, 3)) * extent_np + vertices_np.max(axis=0)

    queries_wp = wp.array(
        np.ascontiguousarray(queries_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    diagonal_wp = tw.bounds.enclosing_diagonal(mesh_wp.points, queries_wp)
    diagonal_igl = igl.bounding_box_diagonal(np.vstack([vertices_np, queries_np]))

    assert diagonal_igl > 0.0
    # Strictly larger than either side alone, or the union is not being taken.
    assert diagonal_wp > tw.bounds.enclosing_diagonal(mesh_wp.points) + 1e-4
    assert diagonal_wp > tw.bounds.enclosing_diagonal(queries_wp) + 1e-4
    assert np.allclose(diagonal_wp, diagonal_igl, rtol=1e-5, atol=1e-5)


def test_enclosing_diagonal_ignores_an_empty_second_set(device: str) -> None:
    """``other=None`` and ``other=<empty>`` both measure ``points`` alone."""
    rng = np.random.default_rng(12)
    cloud_wp = wp.array(
        np.ascontiguousarray(rng.normal(size=(128, 3)), dtype=np.float32),
        dtype=wp.vec3,
        device=device,
    )
    empty_wp = wp.empty(0, dtype=wp.vec3, device=device)

    alone = tw.bounds.enclosing_diagonal(cloud_wp)
    assert alone == tw.bounds.enclosing_diagonal(cloud_wp, None)
    assert alone == tw.bounds.enclosing_diagonal(cloud_wp, empty_wp)


def _tilted_cloud(mesh_tm: tm.Trimesh, device: str) -> tuple[np.ndarray, wp.array[wp.vec3]]:
    """
    Stretch a fixture's vertices anisotropically and rotate them off the coordinate axes.

    Every oriented-box comparison below runs on this rather than on the raw fixture, and it is not
    cosmetic: on a shape whose extent is the same in every direction, *all* candidate orientations
    score nearly the same, so an agreement on the achieved objective would hold for an
    implementation that returned any orientation at all. Stretching by ``(1, 2, 3)`` makes the
    minimizer both unique and far from the identity, which is what gives the assertions teeth --
    and the rotation is what stops the axis-aligned box from already being the answer.
    """
    tilt_np = tm.transformations.random_rotation_matrix(np.random.default_rng(7).random(3))[:3, :3]
    points_np = np.ascontiguousarray(mesh_tm.vertices * np.array([1.0, 2.0, 3.0]) @ tilt_np.T)
    points_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=device)
    return points_np, points_wp


def _frame_np(rotation_wp: wp.mat33) -> np.ndarray:
    """Return triwarp's world-to-box frame as ``(3, 3)`` NumPy, its rows still the box axes."""
    return np.array([[rotation_wp[i, j] for j in range(3)] for i in range(3)])


def _achieved_loss(
    points_np: np.ndarray, frame_np: np.ndarray, objective: _Objective = "volume"
) -> float:
    """
    Recompute, in NumPy, the objective a world-to-box frame achieves on ``points_np``.

    Both libraries are scored through this, from the matrix each one returns, so the comparison is
    of the *boxes* rather than of two self-reported numbers.
    """
    sides_np = np.ptp(points_np @ frame_np.T, axis=0)
    if objective == "volume":
        return float(np.prod(sides_np))
    if objective == "surface_area":
        return float(2.0 * (sides_np * np.roll(sides_np, 1)).sum())
    return float(np.square(sides_np).sum())


@pytest.mark.parametrize("mesh_name", _MESHES)
@pytest.mark.parametrize("objective", _OBJECTIVES)
@pytest.mark.parity("oriented_bounding_box", "igl")
def test_oriented_bounding_box_matches_igl(
    request: pytest.FixtureRequest, mesh_name: str, objective: _Objective
) -> None:
    """
    Class A on the achieved objective, for all three of them: the two searches find the same box.

    This is a *search*, and the comparison is only meaningful because both libraries search the
    identical candidate set -- the Super-Fibonacci spiral of [Alexa 2022] with the identity
    appended, which triwarp adopts precisely so that the two are comparable. So this is not "two
    heuristics landed near each other": with the same candidates and the same three objectives the
    argmin is the same element, and the assertion is tight rather than statistical (measured
    agreement 1e-7 relative, asserted at 1e-5).

    Both sides are scored by ``_achieved_loss`` from the matrix they return, not by trusting the
    number each reports, so a library that returned a good loss with the wrong frame would fail.

    ``refine_iterations=0``: igl has no refinement, and the identical-candidate-set argument is
    only about the sampled phase -- refined, triwarp is strictly better than this comparison.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    points_np, points_wp = _tilted_cloud(mesh_tm, mesh_wp.device)
    igl_objective = {
        "volume": igl.ORIENTED_BOUNDING_BOX_MINIMIZE_VOLUME,
        "surface_area": igl.ORIENTED_BOUNDING_BOX_MINIMIZE_SURFACE_AREA,
        "diagonal": igl.ORIENTED_BOUNDING_BOX_MINIMIZE_DIAGONAL_LENGTH,
    }[objective]

    rotation_wp, lower_wp, upper_wp = tw.bounds.oriented_bounding_box(
        points_wp, 512, objective, refine_iterations=0
    )

    # igl applies its matrix on the right of a row vector, so its frame is triwarp's transposed.
    frame_igl = igl.oriented_bounding_box(points_np, 512, igl_objective).T
    frame_wp = _frame_np(rotation_wp)
    loss_wp = _achieved_loss(points_np, frame_wp, objective)
    loss_igl = _achieved_loss(points_np, frame_igl, objective)
    assert np.isclose(loss_wp, loss_igl, rtol=1e-5)

    # The search is worth running on this fixture: the identity is one of the 512 candidates, so a
    # tie with the axis-aligned box would mean the other 511 never improved on it.
    assert loss_wp < _achieved_loss(points_np, np.eye(3), objective) * 0.99

    # And triwarp's own bounds are the extent in its own frame -- the convention the docstring
    # states, and the only part of the answer igl does not return.
    projected_np = points_np @ frame_wp.T
    assert np.allclose(
        _bounds_np(lower_wp, upper_wp),
        np.stack([projected_np.min(0), projected_np.max(0)]),
        atol=1e-5,
    )


@pytest.mark.parametrize("mesh_name", _MESHES)
@pytest.mark.parity("oriented_bounding_box", "igl")
def test_oriented_bounding_box_frame_matches_igl_transposed(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B: the same frame as igl, element-wise, once transposed for the row-vector convention.

    Stronger than the objective comparison above and it pins the one thing that comparison cannot --
    which of the two transpose conventions triwarp returns. Reading it the wrong way round still
    gives an orthonormal matrix and a plausible box, so nothing else here would catch it.

    Element-wise equality of the *frames* is only well posed because the minimizer is unique on a
    stretched, tilted cloud; that is why this does not run on the raw fixtures.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    points_np, points_wp = _tilted_cloud(mesh_tm, mesh_wp.device)

    rotation_wp, _, _ = tw.bounds.oriented_bounding_box(points_wp, 512, refine_iterations=0)

    frame_np = _frame_np(rotation_wp)
    assert np.allclose(frame_np, igl.oriented_bounding_box(points_np, 512).T, atol=1e-5)
    # A proper rotation, not a reflection: the box axes are right-handed.
    assert np.allclose(frame_np @ frame_np.T, np.eye(3), atol=1e-5)
    assert np.isclose(np.linalg.det(frame_np), 1.0, atol=1e-5)


@pytest.mark.parametrize("mesh_name", _MESHES)
@pytest.mark.parity("oriented_bounding_box", "trimesh")
def test_oriented_bounding_box_agrees_with_trimesh_hull_search(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class C: the sampled box and trimesh's hull-face search find the same box within a band.

    A derived scalar -- box volume -- because no correspondence exists between the two frames: the
    two libraries minimize the same quantity over *different* candidate sets, triwarp over a
    low-discrepancy sampling of ``SO(3)`` and trimesh over the orientations flush with a convex-hull
    face, and different orientations can realise the same volume.

    **Neither side bounds the other, and that was measured rather than assumed.** trimesh's answer
    reads like an exact minimum and is not one: with refinement triwarp returns **3.4% less**
    volume on the tilted half torus and 2.1% less on the icosahedron, so the hull-face restriction
    can miss the optimum. Hence a two-sided band rather than an inequality.

    The bug class it excludes is a search that does not search -- a mis-scored objective, candidates
    that fail to cover ``SO(3)``, or a frame paired with extents it did not produce -- all of which
    leave the volume far above a real minimum.

    Margins, measured refined at 32 768 candidates across the four fixtures: ``volume_wp /
    volume_tm`` runs from 0.966 (half_torus) through 0.979 (icosahedron) and 1.0001 (hemisphere) to
    1.0005 (cave_cube -- refinement recovers the cube's exact orientation to 0.05%, the case that
    used to read +5.2%), so the ``[0.88, 1.02]`` band clears the worst reading by 3.5x below and
    40x above. Mutation probe: the axis-aligned box (``rotations=1, refine_iterations=0``, what a
    search that silently scored nothing would return) reads 1.58x to 5.37x trimesh's volume and
    breaks the 1.02 ceiling on every fixture.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    points_np, points_wp = _tilted_cloud(mesh_tm, mesh_wp.device)

    rotation_wp, lower_wp, upper_wp = tw.bounds.oriented_bounding_box(points_wp, 32768)

    volume_wp = float(np.prod(np.ptp(_bounds_np(lower_wp, upper_wp), axis=0)))
    _, extents_tm = tm.bounds.oriented_bounds(tm.PointCloud(points_np))
    volume_tm = float(np.prod(extents_tm))
    assert volume_tm > 0.0, "the reference produced a box before it is compared to"

    assert volume_tm * 0.88 <= volume_wp <= volume_tm * 1.02
    # The reported extents are the box the reported frame achieves, so the volume above is the
    # quantity that was minimized and not an unrelated pair of numbers.
    assert np.isclose(
        volume_wp, _achieved_loss(points_np, _frame_np(rotation_wp), "volume"), rtol=1e-5
    )


@pytest.mark.parametrize("mesh_name", _MESHES)
@pytest.mark.parity("oriented_bounding_box", "pyvista")
def test_oriented_bounding_box_beats_the_pyvista_pca_box(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class C, and the one reference in this module that triwarp's search should *dominate*.

    VTK's ``oriented_bounding_box`` is PCA of the **points** (not of the hull, as open3d's
    ``get_oriented_bounding_box`` is), so it minimizes nothing and there is no correspondence
    between the two frames -- box volume is the derived scalar, as in the trimesh and open3d tests
    above.

    Unlike those two the relation is one-sided by construction: a search over ``SO(3)`` cannot lose
    to a single fixed orientation by more than its own sampling error. Measured ``volume_wp /
    volume_pv`` refined at 32 768 candidates: **1.00001 (icosahedron), 1.00015 (cave_cube), 0.9428
    (hemisphere), 0.8353 (half_torus)** -- so triwarp is up to 17% tighter and never more than 0.02%
    looser, and the ``1.02`` ceiling clears the worst reading by 130x on that side. The floor is a
    sanity bound rather than a tight one: PCA is not arbitrarily bad on these shapes.

    Bug class excluded: a search that does not search. Mutation probe -- the axis-aligned box
    (``rotations=1, refine_iterations=0``, what a scored-nothing search returns) reads **1.65x to
    5.37x** pyvista's volume on these same clouds, breaking the ceiling by 80x to 260x.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    points_np, points_wp = _tilted_cloud(mesh_tm, mesh_wp.device)

    volume_pv = float(pv.PolyData(points_np).oriented_bounding_box(as_composite=False).volume)
    assert volume_pv > 0.0, "the reference produced a box before it is compared to"

    _rotation_wp, lower_wp, upper_wp = tw.bounds.oriented_bounding_box(points_wp, 32768)
    volume_wp = float(np.prod(np.ptp(_bounds_np(lower_wp, upper_wp), axis=0)))
    assert volume_pv * 0.75 <= volume_wp <= volume_pv * 1.02

    # The axis-aligned box is what a search that scored nothing would return, and it fails.
    _rotation_np, lower_np, upper_np = tw.bounds.oriented_bounding_box(
        points_wp, 1, refine_iterations=0
    )
    assert float(np.prod(np.ptp(_bounds_np(lower_np, upper_np), axis=0))) > volume_pv * 1.02


@pytest.mark.parametrize("mesh_name", _MESHES)
@pytest.mark.parity("oriented_bounding_box", "open3d")
def test_oriented_bounding_box_agrees_with_open3d_minimal_box(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class C: the sampled box and open3d's hull-based approximate minimal box, within a band.

    The same derived-scalar comparison as the trimesh test above, against a third independent
    minimizer -- ``get_minimal_oriented_bounding_box``, a convex-hull search like trimesh's. Not
    ``get_oriented_bounding_box``: that one is PCA of the hull and minimizes nothing (measured
    12.9% above triwarp on the tilted half_torus, and *exact* on cave_cube where axis-snapping is
    what PCA happens to do), so a band around it would be a band around an unrelated quantity.

    Neither side bounds the other, measured refined on these four tilted fixtures:
    ``volume_wp / volume_o3d`` runs 0.979 (icosahedron) through 1.0000 (half_torus and hemisphere,
    ties to 4 digits) to 1.0005 (cave_cube -- refinement recovers the cube's exact orientation to
    0.05%; sampled alone this fixture read +5.2%). The ``[0.90, 1.02]`` band clears the worst
    reading by 4.7x below and 40x above. Mutation probe: the axis-aligned box
    (``rotations=1, refine_iterations=0``, what a search that scored nothing returns) reads
    1.58-5.37x open3d's minimal volume on the same fixtures, breaking the 1.02 ceiling on every
    one.

    The bug class excluded is the trimesh test's: a search that does not search, candidates that
    miss ``SO(3)``, or extents decoupled from the reported frame.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    points_np, points_wp = _tilted_cloud(mesh_tm, mesh_wp.device)

    rotation_wp, lower_wp, upper_wp = tw.bounds.oriented_bounding_box(points_wp, 32768)

    volume_wp = float(np.prod(np.ptp(_bounds_np(lower_wp, upper_wp), axis=0)))
    cloud_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_np))
    volume_o3d = cloud_o3d.get_minimal_oriented_bounding_box().volume()
    assert volume_o3d > 0.0, "the reference produced a box before it is compared to"

    assert volume_o3d * 0.90 <= volume_wp <= volume_o3d * 1.02
    assert np.isclose(
        volume_wp, _achieved_loss(points_np, _frame_np(rotation_wp), "volume"), rtol=1e-5
    )


def test_oriented_bounding_box_prefilter_returns_the_identical_box(device: str) -> None:
    """
    Above ``CONVEX_PREFILTER_MIN_POINTS`` the convex-superset prefilter must change nothing.

    The dense cloud (prefiltered internally) and its explicit hull-candidate subset (below the
    threshold, so searched directly) score the same support points, and min/max extents are
    order-independent, so the two boxes must agree to float32 exactness — not merely in volume.
    """
    rng = np.random.default_rng(23)
    n = tw.bounds.CONVEX_PREFILTER_MIN_POINTS + 10_000
    cloud_np = (rng.standard_normal((n, 3)) @ np.diag([3.0, 1.0, 0.5])).astype(np.float32)
    cloud_wp = wp.array(cloud_np, dtype=wp.vec3, device=device)

    kept_wp = tw.array.gather(
        cloud_wp, tw.array.flatnonzero(tw.convex.convex_superset_mask(cloud_wp))
    )
    assert int(kept_wp.shape[0]) < n // 10, "the prefilter must actually discard interior points"

    rotation_dense, lower_dense, upper_dense = tw.bounds.oriented_bounding_box(cloud_wp)
    rotation_kept, lower_kept, upper_kept = tw.bounds.oriented_bounding_box(kept_wp)

    assert np.allclose(_frame_np(rotation_dense), _frame_np(rotation_kept), rtol=0, atol=0)
    assert np.array_equal(_bounds_np(lower_dense, upper_dense), _bounds_np(lower_kept, upper_kept))


def test_oriented_bounding_box_single_rotation_is_the_aabb(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    ``rotations=1`` scores only the identity, so it must reproduce ``aabb`` exactly.

    The identity is deliberately the *last* candidate rather than the first, which is what makes
    this an edge case worth pinning: an off-by-one in the spiral's ``n - 1`` split would drop it and
    return some arbitrary orientation here. ``refine_iterations=0``, because refinement exists to
    *improve* on the sampled answer -- refined ``rotations=1`` legitimately beats the AABB.
    """
    _, mesh_wp = icosahedron
    rotation_wp, lower_wp, upper_wp = tw.bounds.oriented_bounding_box(
        mesh_wp.points, 1, refine_iterations=0
    )

    assert np.array_equal(_frame_np(rotation_wp), np.eye(3))
    assert np.array_equal(
        _bounds_np(lower_wp, upper_wp), _bounds_np(*tw.bounds.aabb(mesh_wp.points))
    )


def test_oriented_bounding_box_never_loses_to_the_aabb(
    half_torus: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Including the identity in the candidate set makes the result monotone in ``rotations``.

    Adding candidates can only lower the minimum, and every count contains the axis-aligned box, so
    the volume sequence must be non-increasing and never above the AABB's. That is a property of the
    candidate construction rather than of any one answer, and it is what lets a caller raise
    ``rotations`` without checking the result got better. ``refine_iterations=0`` for the sequence:
    refinement converges to whichever basin each count's winner lands in, so *refined* volumes are
    not monotone in ``rotations`` -- the refined guarantee is the separate one pinned in
    [`test_oriented_bounding_box_refinement_is_monotone`][tests.test_bounds.test_oriented_bounding_box_refinement_is_monotone].
    """
    mesh_tm, mesh_wp = half_torus
    _, points_wp = _tilted_cloud(mesh_tm, mesh_wp.device)

    volumes = [
        float(
            np.prod(
                np.ptp(
                    _bounds_np(
                        *tw.bounds.oriented_bounding_box(points_wp, n, refine_iterations=0)[1:]
                    ),
                    axis=0,
                )
            )
        )
        for n in (1, 16, 256, 4096)
    ]

    assert volumes == sorted(volumes, reverse=True)
    assert volumes[-1] < volumes[0] * 0.95, "the tilted cloud's box is materially tighter"
    assert np.isclose(
        volumes[0], float(np.prod(np.ptp(_bounds_np(*tw.bounds.aabb(points_wp)), axis=0)))
    )


@pytest.mark.parametrize("mesh_name", _MESHES)
def test_oriented_bounding_box_refinement_is_monotone(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Refinement never loses to sampling, and the refined default ties or beats the sampled 32k box.

    The monotonicity is by construction -- every round's candidate ball includes the identity
    perturbation, so each chain re-scores its own base -- but the construction is exactly the kind
    of thing a refactor breaks silently, so it is pinned per fixture. The second assert is the
    reason the refinement exists: eight rounds on top of the *default* 4 096 candidates reach at
    least the quality of an unrefined 32 768-candidate search (measured: strictly better on every
    fixture here, from 0.3% on the icosahedron to 4.9% on the cube shell).
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    points_np, points_wp = _tilted_cloud(mesh_tm, mesh_wp.device)

    def volume(rotations: int, refine_iterations: int) -> float:
        rotation_wp, lower_wp, upper_wp = tw.bounds.oriented_bounding_box(
            points_wp, rotations, refine_iterations=refine_iterations
        )
        volume_box = float(np.prod(np.ptp(_bounds_np(lower_wp, upper_wp), axis=0)))
        assert np.isclose(
            volume_box, _achieved_loss(points_np, _frame_np(rotation_wp), "volume"), rtol=1e-5
        )
        return volume_box

    sampled = volume(4096, 0)
    refined = volume(4096, 8)
    assert refined <= sampled * (1.0 + 1e-6)
    assert refined <= volume(32768, 0) * (1.0 + 1e-6)


def test_oriented_bounding_box_empty_cloud(device: str) -> None:
    """An empty cloud gives the identity frame and the same inverted box ``aabb`` returns."""
    empty_wp = wp.zeros(0, dtype=wp.vec3, device=device)
    rotation_wp, lower_wp, upper_wp = tw.bounds.oriented_bounding_box(empty_wp)

    assert np.array_equal(_frame_np(rotation_wp), np.eye(3))
    assert np.all(np.isposinf(_bounds_np(lower_wp, upper_wp)[0]))
    assert np.all(np.isneginf(_bounds_np(lower_wp, upper_wp)[1]))


def test_oriented_bounding_box_rejects_bad_arguments(device: str) -> None:
    """All three documented ``ValueError``s are reachable, and none is raised for a valid call."""
    points_wp = wp.array(
        np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32), dtype=wp.vec3, device=device
    )
    with pytest.raises(ValueError, match="rotations must be >= 1"):
        tw.bounds.oriented_bounding_box(points_wp, 0)
    with pytest.raises(ValueError, match="objective must be"):
        tw.bounds.oriented_bounding_box(points_wp, 8, "perimeter")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="refine_iterations must be >= 0"):
        tw.bounds.oriented_bounding_box(points_wp, 8, refine_iterations=-1)
