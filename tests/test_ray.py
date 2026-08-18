"""Regression tests for ``triwarp.ray`` against trimesh."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh as tm
import warp as wp
from meshlib import mrmeshpy as mm

import triwarp as tw
from tests.conversions import points_to_pyvista, trimesh_to_meshlib, trimesh_to_pyvista


def _assert_longest_ray_allclose(distances_wp_np: np.ndarray, distances_tm_np: np.ndarray) -> None:
    assert np.array_equal(np.isinf(distances_wp_np), np.isinf(distances_tm_np))
    finite_tm = ~np.isinf(distances_tm_np)
    if finite_tm.any():
        assert np.allclose(
            distances_wp_np[finite_tm], distances_tm_np[finite_tm], rtol=1e-5, atol=1e-5
        )


def _upward_rays(mesh_tm: tm.Trimesh, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    ``n`` rays starting below the mesh and firing ``+z``, with origins inside its xy footprint.

    Drawing the origins from the unit square instead put every one of them outside the x-extent of
    both the ``icosahedron`` and ``hemisphere`` fixtures -- each translated by ``(-1, 0, 2)`` -- so
    all three ``intersects_*`` comparisons below ran on 256 misses, asserted that two all-miss
    answers agree, and duplicated their own ``_miss`` siblings. Measured after the fix: 208 of 256
    rays hit the icosahedron and 197 the hemisphere. The ``_miss`` tests keep their own
    construction, which misses for a robust reason -- firing ``+y`` from below never leaves the
    plane ``z = min_z - 5`` -- rather than by an accident of where the fixture sits.
    """
    rng = np.random.default_rng(seed)
    origins_np = np.column_stack(
        (
            rng.uniform(mesh_tm.bounds[0, :2], mesh_tm.bounds[1, :2], size=(n, 2)),
            np.full(n, mesh_tm.bounds[0, 2] - 5.0),
        )
    ).astype(np.float32)
    directions_np = np.tile([0.0, 0.0, 1.0], (n, 1)).astype(np.float32)
    return origins_np, directions_np


def test_contains_points(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A: inside/outside against ``Trimesh.contains``, over three deliberately-placed groups.

    Points near the centre, just outside the bounds, and far away -- the third group is what
    catches a ray that runs out of range rather than reporting a miss.
    """
    mesh_tm, mesh_wp = icosahedron
    rng = np.random.default_rng(7)

    inside_np = mesh_tm.center_mass + rng.normal(scale=0.05, size=(50, 3))
    outside_np = rng.random((50, 3)) * 4.0 + mesh_tm.bounds[1]
    far_np = rng.random((30, 3)) * 100.0 + 1.0 + mesh_tm.bounds[1]
    center_np = np.asarray([mesh_tm.center_mass], dtype=np.float32)

    points_wp = wp.array(
        np.ascontiguousarray(inside_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    contains_wp = tw.ray.contains_points(mesh_wp, points_wp).numpy()
    assert contains_wp.all()
    assert np.array_equal(contains_wp, mesh_tm.contains(inside_np))

    points_wp = wp.array(
        np.ascontiguousarray(outside_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    contains_wp = tw.ray.contains_points(mesh_wp, points_wp).numpy()
    assert not contains_wp.any()
    assert np.array_equal(contains_wp, mesh_tm.contains(outside_np))

    points_wp = wp.array(
        np.ascontiguousarray(far_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    contains_wp = tw.ray.contains_points(mesh_wp, points_wp).numpy()
    assert not contains_wp.any()
    assert np.array_equal(contains_wp, mesh_tm.contains(far_np))

    points_wp = wp.array(
        np.ascontiguousarray(center_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    contains_wp = tw.ray.contains_points(mesh_wp, points_wp).numpy()
    assert contains_wp.all()
    assert np.array_equal(contains_wp, mesh_tm.contains(center_np))


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube", "torus"])
@pytest.mark.parity("winding_number", "pyvista")
def test_contains_points_matches_pyvista(request: pytest.FixtureRequest, mesh_name: str):
    """
    Class A: ``select_interior_points`` is the same inside/outside predicate, point for point.

    VTK's answer is a per-point bool in the ``selected_points`` array of the returned cloud (the
    older ``SelectedPoints`` spelling is deprecated in pyvista 0.48), so the only transform is
    reading it out of the right array and casting to ``bool``. Measured **1.000** agreement on
    2 000 uniform queries against ``icosphere(3)``, 604 of them inside.

    The three fixtures are the ones a ray-parity rule should not fail on: a convex solid, a hollow
    shell whose interior is *outside*, and a genus-1 hole. The counts are asserted non-trivial in
    both directions, so a predicate that answered a constant would fail rather than agree -- which
    is why the queries are drawn from a box **grown 40%** about the centre rather than from
    ``mesh_tm.bounds``: on ``cave_cube`` the cavity is 0.1 of a unit box, so a bbox-uniform sample
    lands inside the solid 600 times out of 600 and the comparison would be a constant.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    rng = np.random.default_rng(11)
    centre_np = mesh_tm.bounds.mean(axis=0)
    extent_np = 1.4 * (mesh_tm.bounds[1] - mesh_tm.bounds[0])
    points_np = centre_np - 0.5 * extent_np + rng.random((600, 3)) * extent_np

    selected_pv = points_to_pyvista(points_np).select_interior_points(trimesh_to_pyvista(mesh_tm))
    contains_pv = np.asarray(selected_pv.point_data["selected_points"]).astype(bool)
    assert 0 < int(contains_pv.sum()) < len(points_np)  # both answers present

    points_wp = wp.array(
        np.ascontiguousarray(points_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    assert np.array_equal(tw.ray.contains_points(mesh_wp, points_wp).numpy(), contains_pv)


def test_contains_cavity(cave_cube: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A: the origin sits in ``cave_cube``'s hollow, so it must read *outside*.

    The one case a bounding-box or convex test gets wrong, and the reason this fixture exists.
    trimesh agrees, so the comparison is direct rather than an asserted constant.
    """
    mesh_tm, mesh_wp = cave_cube
    origin_np = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

    points_wp = wp.array(
        np.ascontiguousarray(origin_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    contains_wp = tw.ray.contains_points(mesh_wp, points_wp).numpy()
    assert not contains_wp.any()
    assert np.array_equal(contains_wp, mesh_tm.contains(origin_np))


def test_contains_empty_points(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    _, mesh_wp = icosahedron
    points_wp = wp.empty(0, dtype=wp.vec3, device=mesh_wp.device)
    assert tw.ray.contains_points(mesh_wp, points_wp).numpy().shape == (0,)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere"])
def test_intersects_first(request: pytest.FixtureRequest, mesh_name: str):
    """Class A: the first-hit face index agrees with trimesh's ray engine, ray for ray."""
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    origins_np, directions_np = _upward_rays(mesh_tm, 256, seed=0)

    origins_wp = wp.array(
        np.ascontiguousarray(origins_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    directions_wp = wp.array(
        np.ascontiguousarray(directions_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    triangle_wp = tw.ray.intersects_first(mesh_wp, origins_wp, directions_wp).numpy()
    triangle_tm = mesh_tm.ray.intersects_first(origins_np, directions_np)
    # Without this the test still passes on two all-miss answers; see _upward_rays.
    assert (triangle_tm != -1).any()
    assert np.array_equal(triangle_wp, triangle_tm)


def test_intersects_first_miss(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A on an all-miss answer, which is a claim here rather than a vacuous comparison.

    Firing ``+y`` from below the mesh never leaves the plane ``z = min_z - 5``, so every ray
    misses for a robust reason -- and the ``(triangle == -1).all()`` assert is what makes the
    emptiness the assertion instead of an accident (see ``_upward_rays``).
    """
    mesh_tm, mesh_wp = icosahedron
    n = 100
    origins_np = np.random.default_rng(1).random((n, 3)).astype(np.float32)
    directions_np = np.tile([0.0, 1.0, 0.0], (n, 1)).astype(np.float32)
    origins_np[:, 2] = mesh_tm.bounds[0, 2] - 5.0

    origins_wp = wp.array(
        np.ascontiguousarray(origins_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    directions_wp = wp.array(
        np.ascontiguousarray(directions_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    triangle_wp = tw.ray.intersects_first(mesh_wp, origins_wp, directions_wp).numpy()
    triangle_tm = mesh_tm.ray.intersects_first(origins_np, directions_np)
    assert np.array_equal(triangle_wp, triangle_tm)
    assert (triangle_wp == -1).all()


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere"])
def test_intersects_any(request: pytest.FixtureRequest, mesh_name: str):
    """Class A: the per-ray hit mask agrees with trimesh's, and both answers carry hits."""
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    origins_np, directions_np = _upward_rays(mesh_tm, 256, seed=0)

    origins_wp = wp.array(
        np.ascontiguousarray(origins_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    directions_wp = wp.array(
        np.ascontiguousarray(directions_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    hit_wp = tw.ray.intersects_any(mesh_wp, origins_wp, directions_wp).numpy()
    hit_tm = mesh_tm.ray.intersects_any(origins_np, directions_np)
    # An all-False mask matches an all-False mask; see _upward_rays.
    assert hit_tm.any()
    assert np.array_equal(hit_wp, hit_tm)


def test_intersects_any_miss(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A on the mask form of the same all-miss claim.

    As above: the miss is constructed, and ``not hit.any()`` states it explicitly.
    """
    mesh_tm, mesh_wp = icosahedron
    n = 100
    origins_np = np.random.default_rng(1).random((n, 3)).astype(np.float32)
    directions_np = np.tile([0.0, 1.0, 0.0], (n, 1)).astype(np.float32)
    origins_np[:, 2] = mesh_tm.bounds[0, 2] - 5.0

    origins_wp = wp.array(
        np.ascontiguousarray(origins_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    directions_wp = wp.array(
        np.ascontiguousarray(directions_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    hit_wp = tw.ray.intersects_any(mesh_wp, origins_wp, directions_wp).numpy()
    hit_tm = mesh_tm.ray.intersects_any(origins_np, directions_np)
    assert np.array_equal(hit_wp, hit_tm)
    assert not hit_wp.any()


def _multi_ray_intersect_ml(
    mesh_part_ml: mm.MeshPart, origins_np: np.ndarray, directions_np: np.ndarray
) -> mm.MultiRayMeshIntersectResult:
    """
    Run MeshLib's batched ray query, requesting every output field.

    The fields are **opt-in and silently so**: ``MultiRayMeshIntersectResult`` starts with every
    one of ``isectFaces`` / ``isectPts`` / ``rayDistances`` / ``intersectingRays`` set to ``None``,
    and ``multiRayMeshIntersect`` fills only the ones already holding a container. A result read
    without this step is all ``None`` with no error raised anywhere.
    """
    origins_ml = mm.std_vector_Vector3_float()
    directions_ml = mm.std_vector_Vector3_float()
    for origin_np, direction_np in zip(origins_np, directions_np, strict=True):
        origins_ml.append(mm.Vector3f(*np.asarray(origin_np, dtype=float).tolist()))
        directions_ml.append(mm.Vector3f(*np.asarray(direction_np, dtype=float).tolist()))

    result_ml = mm.MultiRayMeshIntersectResult()
    result_ml.isectFaces = mm.std_vector_Id_FaceTag()
    result_ml.isectPts = mm.std_vector_Vector3_float()
    result_ml.rayDistances = mm.std_vector_float()
    result_ml.intersectingRays = mm.BitSet()
    mm.multiRayMeshIntersect(mesh_part_ml, origins_ml, directions_ml, result_ml)
    return result_ml


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere"])
@pytest.mark.parity("intersects_first", "meshlib")
@pytest.mark.parity("intersects_any", "meshlib")
@pytest.mark.parity("intersects_location", "meshlib")
def test_intersects_match_meshlib(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    Class A on all three: ``multiRayMeshIntersect`` answers every one of them from one traversal.

    The batched call is the pairing rather than the per-ray ``rayMeshIntersect`` (which agrees too,
    on every one of 100 rays) because a Python loop over rays would be measuring the loop -- the
    same reason the curvature comparisons use ``mrmeshnumpy``'s batched forms. Its four outputs map
    one-to-one onto triwarp's three entry points: ``isectFaces`` is ``intersects_first``'s dense
    face array, ``intersectingRays`` is ``intersects_any``'s mask, and ``isectPts`` paired with
    that bitset is ``intersects_location``'s compacted ``(points, rays, faces)`` triple, once that
    triple is sorted by ray -- its compaction emits rows in BVH order, not in ray order, which the
    trimesh comparison below handles with the same sort.

    Two conventions, both named. MeshLib returns the hits **densely** with an invalid ``FaceId``
    where triwarp writes ``-1``, so the transform on the first form is ``FaceId -> int`` with
    invalid mapping to ``-1``; and its location output is dense too, so the class-B half is
    selecting the rows ``intersectingRays`` marks, in ray order, which is the order triwarp's
    compaction already produces.

    Non-vacuous: 208 of 256 rays hit the icosahedron and 197 the hemisphere, so neither the hit
    mask nor the location table is empty or full, and the miss rays check the ``-1`` convention.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    origins_np, directions_np = _upward_rays(mesh_tm, 256, seed=0)
    origins_wp = wp.array(
        np.ascontiguousarray(origins_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    directions_wp = wp.array(
        np.ascontiguousarray(directions_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )

    result_ml = _multi_ray_intersect_ml(
        mm.MeshPart(trimesh_to_meshlib(mesh_tm)), origins_np, directions_np
    )
    hit_ml = np.zeros(origins_np.shape[0], dtype=bool)
    hit_ml[list(result_ml.intersectingRays)] = True
    faces_ml = np.array(
        [
            int(face_ml) if valid else -1
            for face_ml, valid in zip(result_ml.isectFaces, hit_ml, strict=True)
        ],
        dtype=np.int32,
    )
    points_ml = np.array([[point.x, point.y, point.z] for point in result_ml.isectPts])

    assert 0 < hit_ml.sum() < origins_np.shape[0]  # both branches present

    faces_wp = tw.ray.intersects_first(mesh_wp, origins_wp, directions_wp)
    assert np.array_equal(faces_wp.numpy(), faces_ml)

    hit_wp = tw.ray.intersects_any(mesh_wp, origins_wp, directions_wp)
    assert np.array_equal(hit_wp.numpy(), hit_ml)

    locations_wp, rays_wp, triangles_wp = tw.ray.intersects_location(
        mesh_wp, origins_wp, directions_wp
    )
    order_wp = np.argsort(rays_wp.numpy(), kind="stable")
    assert np.array_equal(rays_wp.numpy()[order_wp], np.flatnonzero(hit_ml))
    assert np.array_equal(triangles_wp.numpy()[order_wp], faces_ml[hit_ml])
    assert np.allclose(locations_wp.numpy()[order_wp], points_ml[hit_ml], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere"])
def test_intersects_location(request: pytest.FixtureRequest, mesh_name: str):
    """
    Class B (row order): the ``(ray, face)`` hit pairs equal trimesh's after a shared lexsort.

    triwarp emits one row per hit in BVH order and trimesh in its own, so both sides are ordered
    by ``(ray, face)`` first -- exact on integer rows. The hit *positions* are checked by
    [`test_intersects_location_cave_cube`], which has an analytic answer to compare against.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    origins_np, directions_np = _upward_rays(mesh_tm, 256, seed=0)

    origins_wp = wp.array(
        np.ascontiguousarray(origins_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    directions_wp = wp.array(
        np.ascontiguousarray(directions_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    _loc_wp, ray_wp, tri_wp = tw.ray.intersects_location(mesh_wp, origins_wp, directions_wp)
    tri_tm, ray_tm = mesh_tm.ray.intersects_id(origins_np, directions_np, multiple_hits=False)
    tri_wp_np = tri_wp.numpy()
    ray_wp_np = ray_wp.numpy()
    order_wp = np.lexsort((tri_wp_np, ray_wp_np))
    order_tm = np.lexsort((tri_tm, ray_tm))
    # Two empty hit tables compare equal, which is what this test used to do; see _upward_rays.
    assert tri_tm.size > 0
    assert np.array_equal(tri_wp_np[order_wp], tri_tm[order_tm])
    assert np.array_equal(ray_wp_np[order_wp], ray_tm[order_tm])


def test_intersects_location_miss(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = icosahedron
    n = 100
    origins_np = np.random.default_rng(1).random((n, 3)).astype(np.float32)
    directions_np = np.tile([0.0, 1.0, 0.0], (n, 1)).astype(np.float32)
    origins_np[:, 2] = mesh_tm.bounds[0, 2] - 5.0

    origins_wp = wp.array(
        np.ascontiguousarray(origins_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    directions_wp = wp.array(
        np.ascontiguousarray(directions_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    loc_wp, ray_wp, tri_wp = tw.ray.intersects_location(mesh_wp, origins_wp, directions_wp)
    assert loc_wp.shape[0] == 0
    assert tri_wp.shape[0] == 0
    assert ray_wp.shape[0] == 0


def test_intersects_location_cave_cube(cave_cube: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class B (segment canonicalization): a 100x100 ray grid through the hollow fixture.

    Every ray crosses at least four surfaces here -- outer wall, cavity, cavity, outer wall --
    which is what makes this the multi-hit test; the single-hit path is
    [`test_intersects_first`]. Row order is not defined by either library, so the hits are
    compared as a canonicalized set.
    """
    mesh_tm, mesh_wp = cave_cube
    origins_np = tm.util.grid_linspace(
        mesh_tm.bounds[:, :2] + np.reshape([-0.02, 0.02], (-1, 1)), 100
    )
    origins_np = np.column_stack((origins_np, np.ones(len(origins_np)) * -100.0)).astype(np.float32)
    directions_np = np.ones((len(origins_np), 3), dtype=np.float32) * [0.0, 0.0, 1.0]

    origins_wp = wp.array(
        np.ascontiguousarray(origins_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    directions_wp = wp.array(
        np.ascontiguousarray(directions_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    loc_wp, ray_wp, _tri_wp = tw.ray.intersects_location(mesh_wp, origins_wp, directions_wp)
    loc_tm, ray_tm, _tri_tm = mesh_tm.ray.intersects_location(
        origins_np, directions_np, multiple_hits=False
    )

    ray_wp_np = ray_wp.numpy()
    loc_wp_np = loc_wp.numpy()
    order_wp = np.argsort(ray_wp_np)
    order_tm = np.argsort(ray_tm)
    assert np.array_equal(ray_wp_np[order_wp], ray_tm[order_tm])
    assert np.allclose(loc_wp_np[order_wp], loc_tm[order_tm], rtol=1e-5, atol=1e-4)
    for p, r in zip(loc_wp_np[order_wp], ray_wp_np[order_wp], strict=False):
        assert np.allclose(p[:2], origins_np[r][:2], rtol=1e-5, atol=1e-5)
        assert np.isclose(p[2], mesh_tm.bounds[0, 2], atol=1e-4)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere"])
def test_longest_ray(request: pytest.FixtureRequest, mesh_name: str):
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    rng = np.random.default_rng(3)
    n = 128
    origins_np = (rng.random((n, 3)).astype(np.float32) * 2.0 + mesh_tm.bounds[0]).astype(
        np.float32
    )
    directions_np = rng.normal(size=(n, 3)).astype(np.float32)

    origins_wp = wp.array(
        np.ascontiguousarray(origins_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    directions_wp = wp.array(
        np.ascontiguousarray(directions_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    distances_wp_np = tw.ray.longest_ray(mesh_wp, origins_wp, directions_wp).numpy()
    distances_tm_np = tm.proximity.longest_ray(mesh_tm, origins_np, directions_np)
    _assert_longest_ray_allclose(distances_wp_np, distances_tm_np)


def test_longest_ray_surface_normals(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = icosahedron
    rng = np.random.default_rng(11)
    n = 64
    query_np = rng.random((n, 3)).astype(np.float64)
    closest_np, _distance_np, triangle_id_np = tm.proximity.closest_point(mesh_tm, query_np)
    normals_np = mesh_tm.face_normals[triangle_id_np].astype(np.float32)

    origins_wp = wp.array(
        np.ascontiguousarray(closest_np.astype(np.float32), dtype=np.float32),
        dtype=wp.vec3,
        device=mesh_wp.device,
    )
    directions_wp = wp.array(
        np.ascontiguousarray(normals_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    distances_wp_np = tw.ray.longest_ray(mesh_wp, origins_wp, directions_wp).numpy()
    distances_tm_np = tm.proximity.longest_ray(mesh_tm, closest_np, normals_np)
    _assert_longest_ray_allclose(distances_wp_np, distances_tm_np)


def test_longest_ray_miss(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = icosahedron
    n = 100
    origins_np = np.random.default_rng(1).random((n, 3)).astype(np.float32)
    directions_np = np.tile([0.0, 1.0, 0.0], (n, 1)).astype(np.float32)
    origins_np[:, 2] = mesh_tm.bounds[0, 2] - 5.0

    origins_wp = wp.array(
        np.ascontiguousarray(origins_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    directions_wp = wp.array(
        np.ascontiguousarray(directions_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    distances_wp_np = tw.ray.longest_ray(mesh_wp, origins_wp, directions_wp).numpy()
    distances_tm_np = tm.proximity.longest_ray(mesh_tm, origins_np, directions_np)
    _assert_longest_ray_allclose(distances_wp_np, distances_tm_np)
    assert np.isinf(distances_wp_np).all()


def test_intersects_empty_rays(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    _, mesh_wp = icosahedron
    origins_wp = wp.empty(0, dtype=wp.vec3, device=mesh_wp.device)
    directions_wp = wp.empty(0, dtype=wp.vec3, device=mesh_wp.device)
    assert tw.ray.intersects_first(mesh_wp, origins_wp, directions_wp).numpy().shape == (0,)
    assert tw.ray.intersects_any(mesh_wp, origins_wp, directions_wp).numpy().shape == (0,)
    loc_wp, ray_wp, tri_wp = tw.ray.intersects_location(mesh_wp, origins_wp, directions_wp)
    assert loc_wp.shape[0] == 0
    assert tri_wp.shape[0] == 0
    assert ray_wp.shape[0] == 0
    assert tw.ray.longest_ray(mesh_wp, origins_wp, directions_wp).numpy().shape == (0,)
