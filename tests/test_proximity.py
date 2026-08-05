"""
Regression tests for ``triwarp.proximity`` mesh-query APIs.

Mesh AABB queries against a brute-force reference; closest-on-mesh tests compare against
``trimesh.proximity.closest_point``.
"""

from __future__ import annotations

import igl
import numpy as np
import pymeshlab as ml
import pytest
import trimesh as tm
import trimesh.proximity as tm_proximity
import warp as wp
from scipy.spatial import Delaunay

import triwarp as tw
from tests.conversions import trimesh_to_pymeshlab, trimesh_to_warp
from triwarp.constants import TOLERANCE_MERGE


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


def test_closest_point_on_mesh_ambiguous_edge(device: str) -> None:
    mesh_tm = tm.Trimesh(
        vertices=[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        faces=[[0, 1, 2], [0, 1, 3]],
        process=False,
    )
    query_np = np.array([[-0.25 - 1e-9, 0.0, -0.25]], dtype=np.float64)
    closest_tm, distance_tm, _triangle_id_tm = tm.proximity.closest_point(mesh_tm, query_np)

    vertices_wp = wp.array(
        np.ascontiguousarray(mesh_tm.vertices, dtype=np.float32), dtype=wp.vec3, device=device
    )
    faces_wp = wp.array(
        np.ascontiguousarray(mesh_tm.faces.reshape(-1), dtype=np.int32), device=device
    )
    query_wp = wp.array(
        np.ascontiguousarray(query_np, dtype=np.float32), dtype=wp.vec3, device=device
    )
    closest_wp, distance_wp, _triangle_id_wp = tw.proximity.closest_point_on_mesh(
        vertices_wp, faces_wp, query_wp
    )

    assert np.allclose(closest_wp.numpy(), closest_tm, rtol=1e-5, atol=1e-5)
    assert np.allclose(distance_wp.numpy(), distance_tm, rtol=1e-5, atol=1e-5)


def test_closest_point_on_mesh_unreferenced_vertex(device: str) -> None:
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


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube"])
def test_signed_distance_on_mesh_winding_matches_trimesh(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """On a watertight mesh the winding-number sign must agree with the trimesh reference."""
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
    Pin the tiled winding sum on the CPU device, where the ``device`` fixture never runs it.

    ``wp.launch_tiled`` executes exactly one lane per block on Warp 1.15's CPU backend -- the lane
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
    _centers_wp, radii_wp = tw.proximity.max_tangent_sphere(mesh_wp, points_wp, normals=normals_wp)
    # The inscribed tangent sphere of a unit sphere, from just inside it, is the sphere itself.
    assert np.allclose(radii_wp.numpy(), 0.95, rtol=0.1, atol=0.1)


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


def test_max_tangent_sphere(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = icosahedron
    points_np, face_ids = tm.sample.sample_surface(mesh_tm, 20, seed=42)
    normals_np = mesh_tm.face_normals[face_ids]

    points_wp = wp.array(
        np.ascontiguousarray(points_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    normals_wp = wp.array(
        np.ascontiguousarray(normals_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )

    centers_wp, radii_wp = tw.proximity.max_tangent_sphere(mesh_wp, points_wp, normals=normals_wp)
    centers_tm, radii_tm = tm_proximity.max_tangent_sphere(mesh_tm, points_np, normals=normals_np)

    finite_tm = np.isfinite(radii_tm)
    assert np.array_equal(np.isfinite(radii_wp.numpy()), finite_tm)
    if finite_tm.any():
        assert np.allclose(radii_wp.numpy()[finite_tm], radii_tm[finite_tm], rtol=1e-2, atol=1e-2)
        assert np.allclose(
            centers_wp.numpy()[finite_tm], centers_tm[finite_tm], rtol=1e-2, atol=1e-2
        )


def test_thickness_max_sphere(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = icosahedron
    points_np, face_ids = tm.sample.sample_surface(mesh_tm, 20, seed=7)
    normals_np = mesh_tm.face_normals[face_ids]

    points_wp = wp.array(
        np.ascontiguousarray(points_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    normals_wp = wp.array(
        np.ascontiguousarray(normals_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )

    thickness_wp = tw.proximity.thickness(mesh_wp, points_wp, normals=normals_wp).numpy()
    thickness_tm = tm_proximity.thickness(mesh_tm, points_np, normals=normals_np)

    finite_tm = np.isfinite(thickness_tm)
    assert np.array_equal(np.isfinite(thickness_wp), finite_tm)
    if finite_tm.any():
        assert np.allclose(thickness_wp[finite_tm], thickness_tm[finite_tm], rtol=1e-5, atol=1e-5)


def test_thickness_ray(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = icosahedron
    points_np, face_ids = tm.sample.sample_surface(mesh_tm, 20, seed=13)
    normals_np = mesh_tm.face_normals[face_ids]

    points_wp = wp.array(
        np.ascontiguousarray(points_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    normals_wp = wp.array(
        np.ascontiguousarray(normals_np, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
    )

    thickness_wp = tw.proximity.thickness(
        mesh_wp, points_wp, normals=normals_wp, method="ray"
    ).numpy()
    thickness_tm = tm_proximity.thickness(mesh_tm, points_np, normals=normals_np, method="ray")

    finite_tm = np.isfinite(thickness_tm)
    assert np.array_equal(np.isfinite(thickness_wp), finite_tm)
    if finite_tm.any():
        assert np.allclose(thickness_wp[finite_tm], thickness_tm[finite_tm])


def test_max_tangent_sphere_empty(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = icosahedron
    points_wp = wp.empty(0, dtype=wp.vec3, device=mesh_wp.device)
    centers_wp, radii_wp = tw.proximity.max_tangent_sphere(mesh_wp, points_wp)
    assert centers_wp.shape == (0,)
    assert radii_wp.shape == (0,)


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
    diameter_np = tw.proximity.shape_diameter(
        mesh_wp, mesh_wp.points, normals=normals_wp, n_rays=128, cone_angle=cone_angle
    ).numpy()
    assert (diameter_np <= 2.0 * radius * (1.0 + 1e-4)).all()
    assert (diameter_np >= 2.0 * radius * np.cos(cone_angle) * (1.0 - 1e-4)).all()


def test_shape_diameter_reduces_to_thickness(device: str) -> None:
    """One ray down a vanishing cone *is* ``thickness(method="ray")``, to float32."""
    _sphere_tm, mesh_wp, normals_wp = _sphere_wp(device, 1.5)
    diameter_np = tw.proximity.shape_diameter(
        mesh_wp, mesh_wp.points, normals=normals_wp, n_rays=1, cone_angle=1e-4
    ).numpy()
    thickness_np = tw.proximity.thickness(
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
    diameter_np = tw.proximity.shape_diameter(
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
    trimmed_np = tw.proximity.shape_diameter(
        mesh_wp, mesh_wp.points, normals=normals_wp, n_rays=128, trim=1.0
    ).numpy()
    untrimmed_np = tw.proximity.shape_diameter(
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
    diameter_np = tw.proximity.shape_diameter(
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
        tw.proximity.shape_diameter(mesh_wp, mesh_wp.points, n_rays=0)
    with pytest.raises(ValueError, match=r"cone_angle must be in \(0, pi / 2\]"):
        tw.proximity.shape_diameter(mesh_wp, mesh_wp.points, cone_angle=2.0)
    with pytest.raises(ValueError, match="trim must be non-negative"):
        tw.proximity.shape_diameter(mesh_wp, mesh_wp.points, trim=-1.0)
    with pytest.raises(ValueError, match="one entry per point"):
        tw.proximity.shape_diameter(
            mesh_wp, mesh_wp.points, normals=wp.zeros(2, dtype=wp.vec3, device=device)
        )
    assert normals_wp.shape[0] > 0


def test_shape_diameter_empty(device: str) -> None:
    _sphere_tm, mesh_wp, _normals_wp = _sphere_wp(device, 1.0, subdivisions=1)
    points_wp = wp.zeros(0, dtype=wp.vec3, device=device)
    assert tw.proximity.shape_diameter(mesh_wp, points_wp).shape == (0,)


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
