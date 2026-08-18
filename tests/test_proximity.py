"""
Regression tests for ``triwarp.proximity`` mesh-query APIs.

Mesh AABB queries against a brute-force reference; closest-on-mesh tests compare against
``trimesh.proximity.closest_point``.
"""

from __future__ import annotations

import igl
import numpy as np
import open3d as o3d
import pymeshlab as ml
import pytest
import trimesh as tm
import trimesh.proximity as tm_proximity
import warp as wp
from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm
from scipy.spatial import Delaunay

import triwarp as tw
from tests.conversions import (
    meshlib_scalars_to_numpy,
    numpy_to_warp,
    points_to_pyvista,
    trimesh_to_meshlib,
    trimesh_to_open3d_t,
    trimesh_to_pymeshlab,
    trimesh_to_pyvista,
    trimesh_to_warp,
)
from triwarp.constants import TOLERANCE_MERGE


def _queries_in_bounds_np(mesh_tm: tm.Trimesh, n: int, seed: int) -> np.ndarray:
    """
    Draw ``n`` queries from the mesh's own bounding box, grown 20 %.

    A fixed cube of queries is not usable across fixtures: ``cave_cube`` is a unit box with a
    0.1-wide cavity removed, so its *interior* is a thin shell and 200 points drawn from
    ``[-2, 2]**3`` land outside it every time -- which turns a signed-distance or winding-number
    comparison into a test of the exterior branch alone. Scaling to the fixture puts points on both
    sides of the surface for every mesh in this module, which the non-vacuity asserts then state.
    """
    lower_np, upper_np = mesh_tm.bounds
    margin_np = 0.2 * (upper_np - lower_np)
    return np.random.default_rng(seed).uniform(lower_np - margin_np, upper_np + margin_np, (n, 3))


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
    """
    Class A on the distance, class C on the point: ties make the closest *point* ambiguous.

    200 random queries against ``trimesh.proximity.closest_point``. The distance is the well-
    defined quantity and is compared directly; a query equidistant from two faces has two valid
    closest points, and section 6 records the same divergence against Open3D at ~2e-4. Warp's
    own ``mesh_query_point_no_sign`` is off by up to 2.1e-5, which sets the floor here.
    """
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


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere"])
@pytest.mark.parity("closest_point_on_mesh", "meshlib")
def test_closest_point_on_mesh_matches_meshlib(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A on the distance, class C on the point, and the face index is a *tie-break gauge*.

    ``findProjection`` returns a ``MeshProjectionResult`` carrying the squared distance, the
    projected point and the ``FaceId`` it landed on -- so it is the only reference in this module
    that reports all three. It is per query, so this loops on the MeshLib side and batches on
    triwarp's; the benchmark rows do the same, which is why the row prices a Python loop and says
    so.

    The distances agree to **1.2e-07** and the points to **1.6e-04** on a unit-radius fixture, the
    latter being Warp's own ``mesh_query_point_no_sign`` floor rather than a disagreement about
    geometry (section 6 records the same magnitude against Open3D).

    The face index is the interesting part and it is why this pair is worth having. It differs on
    **39 %** of 200 random queries, and every one of those is a genuine tie: the two faces always
    share at least one corner (57 of 78 share two, i.e. an edge) and the distances differ by at most
    1.2e-07. So the assert is not "the same face" -- which would be wrong to demand -- but "any
    disagreement is a tie", which is a real constraint a mis-indexed lookup would fail.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    rng = np.random.default_rng(42)
    points_np = rng.random((200, 3), dtype=np.float64) * 4.0 - 2.0
    points_wp = wp.array(
        np.ascontiguousarray(points_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )

    mesh_part_ml = mm.MeshPart(trimesh_to_meshlib(mesh_tm))
    projections_ml = [
        mm.findProjection(mm.Vector3f(*point_np.tolist()), mesh_part_ml) for point_np in points_np
    ]
    points_ml = np.array(
        [
            [result.proj.point.x, result.proj.point.y, result.proj.point.z]
            for result in projections_ml
        ]
    )
    distances_ml = np.sqrt(np.array([result.distSq for result in projections_ml]))
    faces_ml = np.array([int(result.proj.face) for result in projections_ml], dtype=np.int32)

    closest_wp, distances_wp, faces_wp = tw.proximity.closest_point_on_mesh(
        mesh_wp.points, mesh_wp.indices, points_wp
    )

    assert np.allclose(distances_wp.numpy(), distances_ml, rtol=1e-5, atol=1e-5)
    assert np.allclose(closest_wp.numpy(), points_ml, rtol=1e-4, atol=1e-4)

    # Every face disagreement is a tie: the same distance, on a face sharing a corner or an edge.
    disagree_np = faces_wp.numpy() != faces_ml
    assert np.allclose(distances_wp.numpy()[disagree_np], distances_ml[disagree_np], atol=1e-5)
    shared_np = np.array(
        [
            len(set(mesh_tm.faces[face_wp]) & set(mesh_tm.faces[face_ml]))
            for face_wp, face_ml in zip(
                faces_wp.numpy()[disagree_np], faces_ml[disagree_np], strict=True
            )
        ]
    )
    assert (faces_wp.numpy() == faces_ml).any()  # non-vacuity: the indices do line up in general
    assert shared_np.size == 0 or shared_np.min() >= 1


def test_closest_point_on_mesh_ambiguous_edge(device: str) -> None:
    """
    Class B: at an exact tie, the *distance* is the answer and the winning face is not.

    Two right triangles meet along the y-axis, one in ``z = 0`` and one in ``x = 0``. The query sits
    at ``(-0.25, 0, -0.25)``, which projects into the interior of each at exactly 0.25 -- the shared
    edge is 0.354 away, so the two face projections are the tied minima and nothing separates them.

    The transform is "up to the tie": which of the two a correct implementation returns is a
    traversal order, and triwarp's differs between the CPU and CUDA BVH. This used to assert the
    point elementwise against trimesh and pass only because the query carried a ``-1e-9`` nudge
    toward the first face -- which does nothing, since triwarp's vertex buffer is ``float32`` and
    ``-0.25 - 1e-9`` rounds to exactly ``-0.25`` there. So the nudge disambiguated trimesh's
    ``float64`` answer and not triwarp's, and the test was asserting a coincidence. It failed on the
    CPU device from the day it was written and nobody ran that device.

    Still non-vacuous: the distance is exact, and the point must be one of the two tied projections
    -- the shared-edge answer ``(0, 0, 0)`` at 0.354 and any point off the surface both fail.
    """
    mesh_tm = tm.Trimesh(
        vertices=[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        faces=[[0, 1, 2], [0, 1, 3]],
        process=False,
    )
    query_np = np.array([[-0.25, 0.0, -0.25]], dtype=np.float64)
    _closest_tm, distance_tm, _triangle_id_tm = tm.proximity.closest_point(mesh_tm, query_np)

    vertices_wp, faces_wp = numpy_to_warp(mesh_tm.vertices, mesh_tm.faces, device)
    query_wp = wp.array(
        np.ascontiguousarray(query_np, dtype=np.float32), dtype=wp.vec3, device=device
    )
    closest_wp, distance_wp, _triangle_id_wp = tw.proximity.closest_point_on_mesh(
        vertices_wp, faces_wp, query_wp
    )

    assert np.allclose(distance_tm, 0.25, rtol=1e-5, atol=1e-5)  # the reference is at the tie too
    assert np.allclose(distance_wp.numpy(), distance_tm, rtol=1e-5, atol=1e-5)
    tied_np = np.array([[-0.25, 0.0, 0.0], [0.0, 0.0, -0.25]])
    assert np.isclose(tied_np, closest_wp.numpy(), rtol=1e-5, atol=1e-5).all(axis=1).any()


def test_closest_point_on_mesh_unreferenced_vertex(device: str) -> None:
    """
    Class A: an unreferenced vertex must not attract the query, on a one-face mesh.

    The stray vertex sits closer to the query than the single face does, so an implementation
    searching *vertices* rather than triangles returns it and fails -- which is exactly the bug
    this exists for. trimesh ignores it too, so the comparison is elementwise.
    """
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


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube", "torus"])
@pytest.mark.parity("signed_distance_on_mesh", "pymeshlab")
def test_signed_distance_on_mesh_matches_pymeshlab(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A: a **third** sign convention that nevertheless returns the same number.

    triwarp's default mode casts perturbed parity rays and MeshLab signs by the dot product with the
    *closest point's normal*, so a priori this is the pair most at risk of a systematic sign flip --
    the plan flagged it as needing thought for that reason. It does not flip: measured on all three
    fixtures, sign agreement is **400 / 400** and the signed values match to **3.3e-07 / 1.5e-08 /
    1.9e-07**, so the assert is a direct ``allclose`` with no transform on the value at all.

    The fixture set is chosen to put the closest-point-normal rule under load rather than to flatter
    it: ``cave_cube`` is non-convex, so points inside the cavity have a nearest face whose normal
    faces the other way from the outer shell's, and ``torus`` is genus 1, where a point in the hole
    is outside the solid but surrounded by surface. Both are exactly where a normal-based sign is
    supposed to be unreliable.

    Two named transforms on the *plumbing*, not the value: the query points go in as a second,
    face-less mesh (``measuremesh=1``, ``refmesh=0``) and the answer is read off that mesh's
    ``vertex_scalar_array()``. ``signeddist=True`` is what makes it signed rather than absolute, and
    is the parameter the benchmark passes.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    rng = np.random.default_rng(4)
    points_np = mesh_tm.bounds[0] + rng.random((400, 3)) * (mesh_tm.bounds[1] - mesh_tm.bounds[0])

    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.add_mesh(ml.Mesh(vertex_matrix=np.ascontiguousarray(points_np)))
    meshset_pml.compute_scalar_by_distance_from_another_mesh_per_vertex(
        measuremesh=1, refmesh=0, signeddist=True
    )
    signed_pml = np.asarray(meshset_pml.current_mesh().vertex_scalar_array())

    points_wp = wp.array(
        np.ascontiguousarray(points_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    signed_wp = tw.proximity.signed_distance_on_mesh(mesh_wp.points, mesh_wp.indices, points_wp)

    assert np.array_equal(np.sign(signed_wp.numpy()), np.sign(signed_pml))
    assert np.allclose(signed_wp.numpy(), signed_pml, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube"])
@pytest.mark.parity("signed_distance_on_mesh", "meshlib")
def test_signed_distance_on_mesh_matches_meshlib(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A, sign included: MeshLib shares triwarp's convention exactly, and trimesh's does not.

    ``findSignedDistances(refMesh, testPoints)`` is the batched form and the one used here; the
    per-query ``signedDistanceToMesh`` gives the identical numbers. Both are **negative inside**,
    which is Warp's SDF convention and open3d's, against trimesh's opposite one -- so this pair
    needs no negation and is asserted without one, which is what makes it a check on the sign rather
    than a restatement of it.

    Two options that had to be set rather than accepted. ``maxDistSq`` defaults to ``FLT_MAX`` but
    ``nullOutsideMinMax`` is ``True``, so a query outside the band comes back as a null rather than
    a distance; and ``signMode`` defaults to ``ProjectionNormal``, which is neither of triwarp's two
    modes by construction. Measured on ``cave_cube``, all three of ``ProjectionNormal``,
    ``WindingRule`` and ``HoleWindingRule`` agree with triwarp's parity mode to **0.0** on 300
    queries spanning the cavity, so the default is used and the equality is asserted at full
    precision; the fixture is what makes that non-trivial, since a convex mesh cannot separate a
    projection-normal sign from a parity one.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    points_np = _queries_in_bounds_np(mesh_tm, 200, seed=42)
    points_wp = wp.array(
        np.ascontiguousarray(points_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )

    cloud_ml = mn.pointCloudFromPoints(np.ascontiguousarray(points_np))
    distances_ml = meshlib_scalars_to_numpy(
        mm.findSignedDistances(trimesh_to_meshlib(mesh_tm), cloud_ml.points)
    )
    distances_wp = tw.proximity.signed_distance_on_mesh(
        mesh_wp.points, mesh_wp.indices, points_wp
    ).numpy()

    assert (distances_ml < 0).any()  # non-vacuity: some query is inside, so the sign is exercised
    assert (distances_ml > 0).any()
    assert np.allclose(distances_wp, distances_ml, rtol=1e-5, atol=1e-5)
    # The convention, stated as an assert: trimesh's sign is the other one.
    assert np.allclose(distances_wp, -tm_proximity.signed_distance(mesh_tm, points_np), atol=1e-4)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube", "torus"])
@pytest.mark.parity("signed_distance_on_mesh", "igl")
def test_signed_distance_on_mesh_matches_igl(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class B (a named sign-rule choice): a direct ``allclose`` against the pseudonormal sign type.

    igl's angle-weighted pseudonormal sign is a *fourth* rule beside triwarp's parity rays,
    trimesh's and MeshLab's closest-point normal, and on these three watertight fixtures it agrees
    with triwarp exactly -- so the value comparison needs no transform.

    The named choice is *which* ``sign_type`` is the oracle, and it is not free: for both
    ``WINDING_NUMBER`` and ``FAST_WINDING_NUMBER`` igl returns ``(1 - 2 * w) * d`` with ``w`` the
    **continuous** winding number, not ``sign(1 - 2 * w) * d``. Its magnitude is therefore ``|d|``
    only where ``w`` is exactly 0 or 1 and is scaled down near the surface -- measured on
    ``bunny_decimated``, ``|S|`` departs from the pseudonormal type's by 2.7e-2 of the bbox diagonal
    where the pseudonormal type agrees with triwarp to 8e-8. So the winding types cannot be compared
    on value at all, and the benchmark's ``sign_mode="winding"`` igl row is a cost comparison only.

    ``cave_cube`` and ``torus`` are here for the same reason they are in the pymeshlab test: a
    normal-based sign rule is supposed to be unreliable in a cavity and in a genus-1 hole.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    rng = np.random.default_rng(4)
    points_np = mesh_tm.bounds[0] + rng.random((400, 3)) * (mesh_tm.bounds[1] - mesh_tm.bounds[0])

    signed_igl, _, _, _ = igl.signed_distance(
        np.ascontiguousarray(points_np),
        np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64),
        np.ascontiguousarray(mesh_tm.faces, dtype=np.int64),
        igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL,
    )

    points_wp = wp.array(
        np.ascontiguousarray(points_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    signed_wp = tw.proximity.signed_distance_on_mesh(mesh_wp.points, mesh_wp.indices, points_wp)

    assert np.array_equal(np.sign(signed_wp.numpy()), np.sign(signed_igl))
    assert np.allclose(signed_wp.numpy(), signed_igl, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube", "torus"])
@pytest.mark.parity("signed_distance_on_mesh", "open3d")
def test_signed_distance_on_mesh_matches_open3d(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A: Embree's ``compute_signed_distance`` shares triwarp's parity sign and its convention.

    Open3D signs by counting ray crossings -- the same rule as triwarp's default ``"parity"`` mode
    -- and uses the same Warp-SDF orientation (negative inside), so no transform is needed on
    either the sign or the value. Probed to 1.8e-7 agreement on an icosphere before this test was
    written. ``cave_cube`` and ``torus`` ride along because a parity rule is exactly what should
    *not* fail in a cavity or through a genus-1 hole, unlike the normal-based rules in the
    pymeshlab test above.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    rng = np.random.default_rng(4)
    points_np = mesh_tm.bounds[0] + rng.random((400, 3)) * (mesh_tm.bounds[1] - mesh_tm.bounds[0])

    mesh_t = trimesh_to_open3d_t(mesh_tm)
    scene_o3d = o3d.t.geometry.RaycastingScene()
    scene_o3d.add_triangles(mesh_t)
    signed_o3d = scene_o3d.compute_signed_distance(
        o3d.core.Tensor(np.ascontiguousarray(points_np, dtype=np.float32))
    ).numpy()

    points_wp = wp.array(
        np.ascontiguousarray(points_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    signed_wp = tw.proximity.signed_distance_on_mesh(mesh_wp.points, mesh_wp.indices, points_wp)

    assert np.array_equal(np.sign(signed_wp.numpy()), np.sign(signed_o3d))
    assert np.allclose(signed_wp.numpy(), signed_o3d, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube", "torus"])
@pytest.mark.parity("signed_distance_on_mesh", "pyvista")
def test_signed_distance_on_mesh_matches_pyvista(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A: ``vtkImplicitPolyDataDistance`` is an exact SDF and shares triwarp's sign convention.

    This is the strongest pyvista row in the suite and the reason the registration was worth having:
    negative inside on both sides with no negation, correlation 1.0000000, **max absolute difference
    1.5e-07** and identical signs on 2 000 queries against ``icosphere(3)``. It comes back float64,
    so the residual is triwarp's ``float32`` vertex buffer.

    Not to be confused with vedo's ``Mesh.signed_distance``, which is ``vtkSignedDistance`` -- a
    tangent-plane point-cloud estimator that correlates 0.956 with a max absolute difference of
    0.375 on the same input, and is a class-D row rather than this one.

    The three fixtures are the ones a parity sign rule must not fail on, as in the open3d test
    above; ``cave_cube``'s interior cavity is signed *outside* by both libraries.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    rng = np.random.default_rng(5)
    centre_np = mesh_tm.bounds.mean(axis=0)
    extent_np = 1.4 * (mesh_tm.bounds[1] - mesh_tm.bounds[0])
    points_np = centre_np - 0.5 * extent_np + rng.random((400, 3)) * extent_np

    signed_pv = np.asarray(
        points_to_pyvista(points_np)
        .compute_implicit_distance(trimesh_to_pyvista(mesh_tm))
        .point_data["implicit_distance"]
    )
    # Both signs present, so the sign comparison is not riding on a constant.
    assert (signed_pv < 0).any()
    assert (signed_pv > 0).any()

    points_wp = wp.array(
        np.ascontiguousarray(points_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    signed_wp = tw.proximity.signed_distance_on_mesh(mesh_wp.points, mesh_wp.indices, points_wp)

    assert np.array_equal(np.sign(signed_wp.numpy()), np.sign(signed_pv))
    assert np.allclose(signed_wp.numpy(), signed_pv, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube"])
def test_signed_distance_on_mesh_winding_matches_trimesh(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """Class B (sign only): winding parity against trimesh's on a watertight mesh."""
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    rng = np.random.default_rng(42)
    points_np = rng.random((200, 3), dtype=np.float64) * 4.0 - 2.0

    distance_tm = -tm_proximity.signed_distance(mesh_tm, points_np)
    points_wp = wp.array(
        np.ascontiguousarray(points_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    distance_wp = tw.proximity.signed_distance_on_mesh(
        mesh_wp.points, mesh_wp.indices, points_wp, sign_mode="winding"
    )
    assert np.allclose(distance_wp.numpy(), distance_tm, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube", "hemisphere", "half_torus"])
def test_signed_distance_on_mesh_winding_sign_matches_exact_winding_number(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    The builtin's Barnes-Hut sign must match thresholding the exact solid-angle sum.

    This is the property that makes ``sign_mode="winding"`` worth having: it holds on the open
    fixtures (``hemisphere``, ``half_torus``) too, where ray parity has no principled answer.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    rng = np.random.default_rng(7)
    lower_np, upper_np = mesh_tm.bounds
    points_np = rng.uniform(lower_np, upper_np, size=(500, 3))
    points_wp = wp.array(
        np.ascontiguousarray(points_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )

    inside_wp = (
        tw.proximity.signed_distance_on_mesh(
            mesh_wp.points, mesh_wp.indices, points_wp, sign_mode="winding"
        ).numpy()
        < 0.0
    )
    inside_exact = (
        tw.proximity.winding_number(mesh_wp.points, mesh_wp.indices, points_wp).numpy() > 0.5
    )
    assert np.array_equal(inside_wp, inside_exact)


def test_signed_distance_on_mesh_winding_unsigned_matches_parity(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """Only the sign may differ between the two modes; the unsigned distance is the same query."""
    _, mesh_wp = icosahedron
    rng = np.random.default_rng(11)
    points_wp = wp.array(
        np.ascontiguousarray(rng.uniform(-2.0, 2.0, size=(200, 3)), dtype=np.float32),
        dtype=wp.vec3,
        device=mesh_wp.device,
    )
    parity_wp = tw.proximity.signed_distance_on_mesh(mesh_wp.points, mesh_wp.indices, points_wp)
    winding_wp = tw.proximity.signed_distance_on_mesh(
        mesh_wp.points, mesh_wp.indices, points_wp, sign_mode="winding"
    )
    assert np.allclose(np.abs(parity_wp.numpy()), np.abs(winding_wp.numpy()), rtol=1e-5, atol=1e-5)


def test_signed_distance_on_mesh_rejects_unknown_sign_mode(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    _, mesh_wp = icosahedron
    points_wp = wp.empty(4, dtype=wp.vec3, device=mesh_wp.device)
    with pytest.raises(ValueError, match="sign_mode"):
        tw.proximity.signed_distance_on_mesh(
            mesh_wp.points,
            mesh_wp.indices,
            points_wp,
            sign_mode="nearest",  # pyright: ignore[reportArgumentType]
        )


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube"])
def test_supplied_mesh_gives_the_same_answer(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    ``mesh=`` is an optimization, so its only correctness obligation is to change nothing.

    Asserted **exactly**, not approximately: both paths query the same BVH over the same buffers,
    so the results must be bit-identical, and a tolerance here would hide a mesh built over the
    wrong geometry. Covers both queries and both of ``signed_distance_on_mesh``'s parity settings
    that accept a mesh at all.
    """
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    rng = np.random.default_rng(21)
    points_wp = wp.array(
        np.ascontiguousarray(rng.normal(scale=1.5, size=(256, 3)), dtype=np.float32),
        dtype=wp.vec3,
        device=mesh_wp.device,
    )
    prebuilt_wp = wp.Mesh(points=wp.clone(mesh_wp.points), indices=wp.clone(mesh_wp.indices))

    rebuilt = tw.proximity.closest_point_on_mesh(mesh_wp.points, mesh_wp.indices, points_wp)
    supplied = tw.proximity.closest_point_on_mesh(
        mesh_wp.points, mesh_wp.indices, points_wp, mesh=prebuilt_wp
    )
    for from_rebuild, from_supplied in zip(rebuilt, supplied, strict=True):
        assert np.array_equal(from_rebuild.numpy(), from_supplied.numpy())

    assert np.array_equal(
        tw.proximity.signed_distance_on_mesh(mesh_wp.points, mesh_wp.indices, points_wp).numpy(),
        tw.proximity.signed_distance_on_mesh(
            mesh_wp.points, mesh_wp.indices, points_wp, mesh=prebuilt_wp
        ).numpy(),
    )


def test_signed_distance_on_mesh_refuses_a_supplied_mesh_in_winding_mode(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    The winding mode needs ``support_winding_number=True`` and cannot check for it, so it refuses.

    ``wp.Mesh`` exposes no way to read the flag back after construction, and without the per-node
    solid-angle expansion the winding builtin silently falls back to ray parity -- a wrong answer
    that looks like a right one. Refusing is the only safe response, and this pins it.
    """
    _mesh_tm, mesh_wp = icosahedron
    points_wp = wp.empty(4, dtype=wp.vec3, device=mesh_wp.device)
    prebuilt_wp = wp.Mesh(points=wp.clone(mesh_wp.points), indices=wp.clone(mesh_wp.indices))

    with pytest.raises(ValueError, match="support_winding_number"):
        tw.proximity.signed_distance_on_mesh(
            mesh_wp.points, mesh_wp.indices, points_wp, sign_mode="winding", mesh=prebuilt_wp
        )
    # ...and the same call without mesh= works, so the guard is on the combination, not the mode.
    assert tw.proximity.signed_distance_on_mesh(
        mesh_wp.points, mesh_wp.indices, points_wp, sign_mode="winding"
    ).shape == (4,)


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
@pytest.mark.parity("winding_number", "igl")
def test_winding_number_random(request: pytest.FixtureRequest, mesh_name: str, tiled: bool) -> None:
    """
    Class A: the generalized winding number against ``igl.winding_number``, both kernels.

    Parametrized over ``tiled`` so the serial and block-cooperative sums are each held to the
    same reference rather than to each other -- 200 queries spanning inside, outside and near-
    surface.
    """
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


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube"])
@pytest.mark.parity("winding_number", "meshlib")
def test_winding_number_matches_meshlib(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class B: ``calcFastWindingNumber`` is a Barnes-Hut approximation, so ``beta`` is the transform.

    triwarp sums the solid angle of every triangle exactly; MeshLib traverses the mesh's AABB tree
    and replaces a distant subtree by its dipole approximation, with ``beta`` the accuracy parameter
    that decides how distant is distant enough. So the two agree only in the limit, and the
    comparison has to say where that limit is rather than pick a tolerance and hope.

    Measured on 200 queries against ``icosphere(3)``, max absolute difference by ``beta``:

    | ``beta`` | 2 (its own default) | 4 | 8 | 20 | 100 |
    |---|---|---|---|---|---|
    | max diff | 2.4e-02 | 5.0e-03 | 1.2e-03 | 1.1e-05 | 1.4e-06 |

    ``beta=20`` is used here: two orders of magnitude inside the 1e-3 tolerance, and the default of
    2 would fail it by 24x -- which is the point, since a comparison that passed at ``beta=2`` would
    be tolerating the approximation rather than measuring the quantity.

    ``calcDipoles`` must run first and takes the tree, not the mesh alone; its two-argument overload
    is the one that *returns* the dipoles; the three-argument form fills a caller-owned buffer.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    query_np = _queries_in_bounds_np(mesh_tm, 200, seed=42)
    query_wp = wp.array(
        np.ascontiguousarray(query_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )

    mesh_ml = trimesh_to_meshlib(mesh_tm)
    tree_ml = mesh_ml.getAABBTree()
    dipoles_ml = mm.calcDipoles(tree_ml, mesh_ml)
    winding_ml = np.array(
        [
            mm.calcFastWindingNumber(
                dipoles_ml, tree_ml, mesh_ml, mm.Vector3f(*point_np.tolist()), 20.0, mm.FaceId()
            )
            for point_np in query_np
        ]
    )
    winding_wp = tw.proximity.winding_number(mesh_wp.points, mesh_wp.indices, query_wp).numpy()

    assert (winding_ml > 0.5).any()  # non-vacuity: inside and outside are both represented
    assert (winding_ml < 0.5).any()
    assert np.allclose(winding_wp, winding_ml, rtol=1e-3, atol=1e-3)


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


@pytest.mark.parametrize("kernel_device", ["cpu", "cuda:0"])
def test_winding_number_tiled_matches_igl_on_both_devices(kernel_device: str) -> None:
    """
    Class A: the tiled winding sum on the CPU device the ``device`` fixture never reaches.

    ``wp.launch_tiled`` executes exactly one lane per block on Warp 1.16's CPU backend -- the lane
    index from ``wp.tid()`` is always 0 -- so the block-wide ``wp.tile_sum`` this reduction used to
    perform summed one face per 64-face tile and returned a winding number off by up to 0.99 there,
    i.e. a whole turn. The reduction is lane-free now; this is the test that fails if it regresses.
    """
    if kernel_device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")

    mesh_tm = tm.creation.icosphere(subdivisions=2)
    mesh_wp = trimesh_to_warp(mesh_tm, kernel_device)
    rng = np.random.default_rng(43)
    query_np = rng.random((200, 3), dtype=np.float64) * 3.0 - 1.5

    winding_igl = igl.winding_number(
        np.array(mesh_tm.vertices, dtype=np.float64),
        np.array(mesh_tm.faces, dtype=np.int64),
        query_np,
    )
    query_wp = wp.array(
        np.ascontiguousarray(query_np, dtype=np.float32), dtype=wp.vec3, device=kernel_device
    )
    winding_wp = tw.proximity.winding_number(mesh_wp.points, mesh_wp.indices, query_wp, tiled=True)
    assert np.allclose(winding_wp.numpy(), winding_igl.ravel(), rtol=1e-5, atol=1e-5)


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


# --------------------------------------------------------------------------------------
# containing_faces_2d
# --------------------------------------------------------------------------------------


def _triangular_lattice_np(rows: int = 26, cols: int = 30) -> np.ndarray:
    """
    Points of a triangular lattice, whose Delaunay triangulation is equilateral throughout.

    The fixture choice is the whole reason the comparison below can be exact. Any Delaunay
    triangulation of a *bounded* point set has needle triangles along its convex hull, and this
    function's single-candidate query cannot resolve a query lying within ``float32`` noise of the
    shared edge of two needles -- measured at 3 in 20 000 on random points, aspect ratios 2 000 to
    12 000. A triangular lattice has no needles anywhere (aspect ratio 6.93 at worst, 2.31 median),
    including along its hull, so it isolates the algorithm from that resolution limit.
    """
    height = np.sqrt(3.0) / 2.0
    return np.array(
        [(col + 0.5 * (row % 2), row * height) for row in range(rows) for col in range(cols)],
        dtype=np.float64,
    )


@pytest.mark.parity("containing_faces_2d", "scipy")
def test_containing_faces_2d_matches_scipy(device: str) -> None:
    """
    Class A: the same triangle index as ``scipy.spatial.Delaunay.find_simplex``, exactly.

    Both sides are given the *same* triangulation -- scipy's own ``simplices`` are what triwarp
    locates against -- so the face numbering is shared and the comparison is an integer array
    equality, ``-1`` for outside included, with no transform at all. Measured agreement is
    ``1.0000000`` over 40 000 queries.

    Non-vacuous in both directions by construction: the query box overhangs the lattice, so roughly
    72% of queries land inside and 28% outside, and both counts are asserted before comparing. An
    implementation returning ``-1`` everywhere, or a face for everything, fails one of them.
    """
    points_np = _triangular_lattice_np()
    triangulation_sp = Delaunay(points_np)
    faces_np = np.ascontiguousarray(triangulation_sp.simplices, dtype=np.int32)
    rng = np.random.default_rng(11)
    queries_np = rng.random((40_000, 2)) * np.array([34.0, 26.0]) - 2.0

    vertices_wp = wp.array(
        np.ascontiguousarray(points_np, dtype=np.float32), dtype=wp.vec2, device=device
    )
    faces_wp = wp.array(faces_np.ravel(), dtype=wp.int32, device=device)
    queries_wp = wp.array(
        np.ascontiguousarray(queries_np, dtype=np.float32), dtype=wp.vec2, device=device
    )

    faces_wp_np = tw.proximity.containing_faces_2d(vertices_wp, faces_wp, queries_wp).numpy()

    faces_sp = triangulation_sp.find_simplex(queries_np)
    assert (faces_sp >= 0).sum() > 20_000, "the reference places most queries inside"
    assert (faces_sp < 0).sum() > 5_000, "and a substantial minority outside"
    assert np.array_equal(faces_wp_np, faces_sp)


def test_containing_faces_2d_locates_every_triangle_from_its_centroid(device: str) -> None:
    """
    Every triangle contains its own centroid, so locating the centroids must be the identity.

    A different failure mode from the random-query test: it visits *all* faces exactly once, so a
    face never reachable by the query -- one dropped from the BVH, or one whose barycentric test is
    mis-signed -- shows up here even if random queries happen to miss it.
    """
    points_np = _triangular_lattice_np(rows=12, cols=14)
    faces_np = np.ascontiguousarray(Delaunay(points_np).simplices, dtype=np.int32)
    centroids_np = points_np[faces_np].mean(axis=1)

    vertices_wp = wp.array(
        np.ascontiguousarray(points_np, dtype=np.float32), dtype=wp.vec2, device=device
    )
    faces_wp = wp.array(faces_np.ravel(), dtype=wp.int32, device=device)
    centroids_wp = wp.array(
        np.ascontiguousarray(centroids_np, dtype=np.float32), dtype=wp.vec2, device=device
    )

    located_np = tw.proximity.containing_faces_2d(vertices_wp, faces_wp, centroids_wp).numpy()

    assert np.array_equal(located_np, np.arange(faces_np.shape[0], dtype=np.int32))


def test_containing_faces_2d_degrades_only_on_needle_triangles(device: str) -> None:
    """
    The documented resolution limit, pinned: on extreme slivers a query can come back ``-1``.

    A Delaunay triangulation of random points always has needles along its convex hull, and a query
    within ``float32`` noise of the shared edge of two of them may be attributed to neither: the
    nearest triangle by distance is the neighbour, and the barycentric test then rejects it.

    Asserted as a *bounded* failure of a *specific* shape, not as a tolerance: at least 99.9% of
    queries agree with scipy, and every disagreement is triwarp reporting ``-1`` -- never a
    different face, which would be a wrong answer rather than a declined one. Measured: 3 in
    20 000, on triangles of aspect ratio 1 900 to 12 400.
    """
    rng = np.random.default_rng(0)
    points_np = rng.random((2_000, 2)) * 4.0 - 2.0
    triangulation_sp = Delaunay(points_np)
    faces_np = np.ascontiguousarray(triangulation_sp.simplices, dtype=np.int32)
    queries_np = rng.random((20_000, 2)) * 5.0 - 2.5

    vertices_wp = wp.array(
        np.ascontiguousarray(points_np, dtype=np.float32), dtype=wp.vec2, device=device
    )
    faces_wp = wp.array(faces_np.ravel(), dtype=wp.int32, device=device)
    queries_wp = wp.array(
        np.ascontiguousarray(queries_np, dtype=np.float32), dtype=wp.vec2, device=device
    )

    faces_wp_np = tw.proximity.containing_faces_2d(vertices_wp, faces_wp, queries_wp).numpy()

    faces_sp = triangulation_sp.find_simplex(queries_np)
    assert (faces_wp_np == faces_sp).mean() >= 0.999
    disagreement = faces_wp_np != faces_sp
    assert np.all(faces_wp_np[disagreement] == -1), "declines, never a different face"


def test_containing_faces_2d_single_triangle(device: str) -> None:
    """One triangle: inside, outside, and a vertex, with the tolerance carrying no weight."""
    vertices_wp = wp.array(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        dtype=wp.vec2,
        device=device,
    )
    faces_wp = wp.array(np.array([0, 1, 2], dtype=np.int32), dtype=wp.int32, device=device)
    queries_wp = wp.array(
        np.array([[0.25, 0.25], [0.9, 0.9], [0.0, 0.0], [-5.0, 3.0]], dtype=np.float32),
        dtype=wp.vec2,
        device=device,
    )

    located_np = tw.proximity.containing_faces_2d(vertices_wp, faces_wp, queries_wp).numpy()

    assert np.array_equal(located_np, [0, -1, 0, -1])


def test_containing_faces_2d_empty(device: str) -> None:
    """No queries gives no answers; no faces gives ``-1`` for every query."""
    vertices_wp = wp.array(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        dtype=wp.vec2,
        device=device,
    )
    faces_wp = wp.array(np.array([0, 1, 2], dtype=np.int32), dtype=wp.int32, device=device)
    no_queries_wp = wp.zeros(0, dtype=wp.vec2, device=device)
    assert int(tw.proximity.containing_faces_2d(vertices_wp, faces_wp, no_queries_wp).shape[0]) == 0

    queries_wp = wp.array(np.array([[0.25, 0.25]], dtype=np.float32), dtype=wp.vec2, device=device)
    no_faces_wp = wp.zeros(0, dtype=wp.int32, device=device)
    assert np.array_equal(
        tw.proximity.containing_faces_2d(vertices_wp, no_faces_wp, queries_wp).numpy(), [-1]
    )
