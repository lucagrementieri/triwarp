"""Regression tests for ``triwarp.intersection`` against Trimesh and potpourri3d."""

from __future__ import annotations

from typing import cast

import igl
import numpy as np
import potpourri3d as pp3d
import pytest
import pyvista as pv
import trimesh as tm
import trimesh.intersections as tm_intersections
import warp as wp
from scipy.spatial import KDTree

import triwarp as tw
from tests.comparisons import hausdorff_two_sided
from tests.conversions import trimesh_to_pyvista, trimesh_to_warp
from triwarp.constants import TOLERANCE_MERGE


def _canonical_segments(lines_np: np.ndarray) -> np.ndarray:
    """Sort segments for order-independent comparison. ``lines_np`` shape ``(m, 2, 3)``."""
    if lines_np.shape[0] == 0:
        return lines_np.reshape(0, 2, 3)
    pairs = []
    for seg in lines_np:
        a, b = np.sort(seg, axis=0)
        pairs.append(np.concatenate([a, b]))
    ordered = np.array(pairs)
    return ordered[
        np.lexsort(
            (
                ordered[:, 3],
                ordered[:, 4],
                ordered[:, 5],
                ordered[:, 0],
                ordered[:, 1],
                ordered[:, 2],
            )
        )
    ].reshape(-1, 2, 3)


def _segments_equal(
    got_np: np.ndarray, exp_np: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-5
) -> bool:
    got = _canonical_segments(got_np.reshape(-1, 2, 3))
    exp = _canonical_segments(exp_np.reshape(-1, 2, 3))
    if got.shape != exp.shape:
        return False
    if got.shape[0] == 0:
        return True
    return bool(np.allclose(got, exp, rtol=rtol, atol=atol))


@pytest.mark.parity("segments_with_plane", "trimesh")
def test_segments_with_plane_axis_aligned(device: str) -> None:
    plane_origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    plane_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    endpoints_np = np.array(
        [
            [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 1.0], [1.0, 0.0, 2.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    endpoints_np = np.transpose(endpoints_np, (1, 0, 2))
    intersections_tm, valid_tm = tm_intersections.plane_lines(
        plane_origin, plane_normal, endpoints_np, line_segments=True
    )

    start_points_wp = wp.array(endpoints_np[0], dtype=wp.vec3, device=device)
    end_points_wp = wp.array(endpoints_np[1], dtype=wp.vec3, device=device)
    intersections_wp, valid_wp = tw.intersection.segments_with_plane(
        start_points_wp,
        end_points_wp,
        wp.vec3(*plane_origin.tolist()),
        wp.vec3(*plane_normal.tolist()),
        line_segments=True,
    )

    assert np.array_equal(valid_wp.numpy(), valid_tm)
    assert np.allclose(
        intersections_wp.numpy()[valid_wp.numpy()], intersections_tm, rtol=1e-5, atol=1e-5
    )


def test_segments_with_plane_parallel(device: str) -> None:
    plane_origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    plane_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    endpoints_np = np.array([[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]], dtype=np.float32)
    endpoints_np = np.transpose(endpoints_np, (1, 0, 2))
    _, valid_tm = tm_intersections.plane_lines(
        plane_origin, plane_normal, endpoints_np, line_segments=True
    )

    start_points_wp = wp.array(endpoints_np[0], dtype=wp.vec3, device=device)
    end_points_wp = wp.array(endpoints_np[1], dtype=wp.vec3, device=device)
    _, valid_wp = tw.intersection.segments_with_plane(
        start_points_wp,
        end_points_wp,
        wp.vec3(*plane_origin.tolist()),
        wp.vec3(*plane_normal.tolist()),
        line_segments=True,
    )
    assert np.array_equal(valid_wp.numpy(), valid_tm)


def test_mesh_with_plane_empty(device: str) -> None:
    vertices_wp = wp.empty(0, dtype=wp.vec3, device=device)
    faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    lines_wp = tw.intersection.mesh_with_plane(
        vertices_wp, faces_wp, wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 0.0, 0.0)
    )
    assert lines_wp.shape == (0, 2)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
@pytest.mark.parity("mesh_with_plane", "trimesh")
def test_mesh_with_plane_axis_planes(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    bounds = mesh_tm.bounds
    mid = 0.5 * (bounds[0] + bounds[1])
    planes = [
        (np.array([0.0, 0.0, 1.0]), mid),
        (np.array([1.0, 0.0, 0.0]), mid),
        (np.array([0.0, 1.0, 0.0]), mid),
    ]

    for plane_normal, plane_origin in planes:
        lines_tm = tm_intersections.mesh_plane(
            mesh=mesh_tm, plane_normal=plane_normal, plane_origin=plane_origin
        )
        lines_wp = tw.intersection.mesh_with_plane(
            mesh_wp.points,
            mesh_wp.indices,
            wp.vec3(*plane_normal.tolist()),
            wp.vec3(*plane_origin.tolist()),
        )
        assert _segments_equal(lines_wp.numpy(), lines_tm)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere"])
def test_mesh_with_plane_tilted_plane(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    axis = tm.unitize(np.array([1.0, 2.0, 0.3], dtype=np.float32))
    angle = np.radians(11)
    base = tm.transformations.rotation_matrix(angle=angle, direction=axis)
    plane_normal = tm.transform_points([[0.0, 0.0, 1.0]], base, translate=False)[0]
    plane_origin = tm.transform_points([mesh_tm.centroid], base)[0]

    lines_tm = tm_intersections.mesh_plane(
        mesh=mesh_tm, plane_normal=plane_normal, plane_origin=plane_origin
    )
    lines_wp = tw.intersection.mesh_with_plane(
        mesh_wp.points,
        mesh_wp.indices,
        wp.vec3(*plane_normal.tolist()),
        wp.vec3(*plane_origin.tolist()),
    )
    assert _segments_equal(lines_wp.numpy(), lines_tm)


def test_mesh_with_plane_return_faces(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = icosahedron
    plane_normal = np.array([0.0, 0.0, 1.0])
    plane_origin = mesh_tm.centroid

    lines_tm, faces_tm = tm_intersections.mesh_plane(
        mesh=mesh_tm, plane_normal=plane_normal, plane_origin=plane_origin, return_faces=True
    )
    lines_wp, faces_wp = tw.intersection.mesh_with_plane(
        mesh_wp.points,
        mesh_wp.indices,
        wp.vec3(*plane_normal.tolist()),
        wp.vec3(*plane_origin.tolist()),
        return_faces=True,
    )

    assert _segments_equal(lines_wp.numpy(), lines_tm)
    assert np.array_equal(np.sort(faces_wp.numpy()), np.sort(faces_tm))


def test_mesh_with_plane_miss_plane(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = icosahedron
    plane_normal = np.array([0.0, 0.0, 1.0])
    plane_origin = mesh_tm.bounds[1] + np.array([0.0, 0.0, 10.0])

    lines_wp = tw.intersection.mesh_with_plane(
        mesh_wp.points,
        mesh_wp.indices,
        wp.vec3(*plane_normal.tolist()),
        wp.vec3(*plane_origin.tolist()),
    )
    assert lines_wp.shape == (0, 2)


def _sliced_meshes_equivalent(
    vertices_a_np: np.ndarray,
    faces_a_np: np.ndarray,
    vertices_b_np: np.ndarray,
    faces_b_np: np.ndarray,
    plane_normal: np.ndarray,
    plane_origin: np.ndarray,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> bool:
    mesh_a_tm = tm.Trimesh(vertices_a_np, faces_a_np, process=False)
    mesh_b_tm = tm.Trimesh(vertices_b_np, faces_b_np, process=False)
    if len(mesh_a_tm.faces) != len(mesh_b_tm.faces):
        return False
    if not np.allclose(mesh_a_tm.bounds, mesh_b_tm.bounds, rtol=rtol, atol=atol):
        return False
    if not np.isclose(mesh_a_tm.area, mesh_b_tm.area, rtol=1e-4, atol=1e-4):
        return False
    dots_b_np = np.dot(plane_normal, (mesh_b_tm.vertices - plane_origin).T)
    return bool(np.min(dots_b_np) >= -max(TOLERANCE_MERGE, 1e-5))


def test_slice_mesh_with_plane_empty(device: str) -> None:
    vertices_wp = wp.empty(0, dtype=wp.vec3, device=device)
    faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    out_vertices_wp, out_faces_wp = tw.intersection.slice_mesh_with_plane(
        vertices_wp, faces_wp, wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 0.0, 0.0)
    )
    assert out_vertices_wp.shape == (0,)
    assert out_faces_wp.shape == (0,)


@pytest.mark.parity("slice_mesh_with_plane", "trimesh")
def test_slice_mesh_with_plane_box_corner() -> None:
    mesh_tm = tm.creation.box()
    plane_origin_np = mesh_tm.bounds[1] - 0.05
    plane_normal_np = mesh_tm.bounds[1]

    vertices_tm, faces_tm, _ = tm_intersections.slice_faces_plane(
        mesh_tm.vertices, mesh_tm.faces, plane_normal_np, plane_origin_np
    )
    mesh_wp = trimesh_to_warp(mesh_tm, "cpu")
    vertices_wp, faces_wp = tw.intersection.slice_mesh_with_plane(
        mesh_wp.points,
        mesh_wp.indices,
        wp.vec3(*plane_normal_np.tolist()),
        wp.vec3(*plane_origin_np.tolist()),
    )
    vertices_wp_np = vertices_wp.numpy()
    faces_wp_np = faces_wp.numpy().reshape(-1, 3)

    assert _sliced_meshes_equivalent(
        vertices_tm, faces_tm, vertices_wp_np, faces_wp_np, plane_normal_np, plane_origin_np
    )
    assert len(faces_tm) == 5


def test_slice_mesh_with_plane_box_top() -> None:
    mesh_tm = tm.creation.box()
    plane_origin_np = mesh_tm.bounds[1] - 0.05
    plane_normal_np = np.array([0.0, 0.0, 1.0])

    vertices_tm, faces_tm, _ = tm_intersections.slice_faces_plane(
        mesh_tm.vertices, mesh_tm.faces, plane_normal_np, plane_origin_np
    )
    mesh_wp = trimesh_to_warp(mesh_tm, "cpu")
    vertices_wp, faces_wp = tw.intersection.slice_mesh_with_plane(
        mesh_wp.points,
        mesh_wp.indices,
        wp.vec3(*plane_normal_np.tolist()),
        wp.vec3(*plane_origin_np.tolist()),
    )
    vertices_wp_np = vertices_wp.numpy()
    faces_wp_np = faces_wp.numpy().reshape(-1, 3)

    assert _sliced_meshes_equivalent(
        vertices_tm, faces_tm, vertices_wp_np, faces_wp_np, plane_normal_np, plane_origin_np
    )
    assert len(faces_tm) == 14


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere"])
def test_slice_mesh_with_plane_axis_planes(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    mid_np = 0.5 * (mesh_tm.bounds[0] + mesh_tm.bounds[1])
    planes = [
        (np.array([0.0, 0.0, 1.0]), mid_np),
        (np.array([1.0, 0.0, 0.0]), mid_np),
        (np.array([0.0, 1.0, 0.0]), mid_np),
    ]

    for plane_normal_np, plane_origin_np in planes:
        vertices_tm, faces_tm, _ = tm_intersections.slice_faces_plane(
            mesh_tm.vertices, mesh_tm.faces, plane_normal_np, plane_origin_np
        )
        vertices_wp, faces_wp = tw.intersection.slice_mesh_with_plane(
            mesh_wp.points,
            mesh_wp.indices,
            wp.vec3(*plane_normal_np.tolist()),
            wp.vec3(*plane_origin_np.tolist()),
        )
        vertices_wp_np = vertices_wp.numpy()
        faces_wp_np = faces_wp.numpy().reshape(-1, 3)
        assert _sliced_meshes_equivalent(
            vertices_tm, faces_tm, vertices_wp_np, faces_wp_np, plane_normal_np, plane_origin_np
        )


def test_slice_mesh_with_plane_tilted_plane(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = icosahedron
    axis_np = tm.unitize(np.array([1.0, 2.0, 0.3], dtype=np.float32))
    angle = np.radians(11)
    base = tm.transformations.rotation_matrix(angle=angle, direction=axis_np)
    plane_normal_np = tm.transform_points([[0.0, 0.0, 1.0]], base, translate=False)[0]
    plane_origin_np = tm.transform_points([mesh_tm.centroid], base)[0]

    vertices_tm, faces_tm, _ = tm_intersections.slice_faces_plane(
        mesh_tm.vertices, mesh_tm.faces, plane_normal_np, plane_origin_np
    )
    vertices_wp, faces_wp = tw.intersection.slice_mesh_with_plane(
        mesh_wp.points,
        mesh_wp.indices,
        wp.vec3(*plane_normal_np.tolist()),
        wp.vec3(*plane_origin_np.tolist()),
    )
    vertices_wp_np = vertices_wp.numpy()
    faces_wp_np = faces_wp.numpy().reshape(-1, 3)
    assert _sliced_meshes_equivalent(
        vertices_tm, faces_tm, vertices_wp_np, faces_wp_np, plane_normal_np, plane_origin_np
    )


def test_slice_mesh_with_plane_on_plane(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = icosahedron
    plane_origin_np = mesh_tm.bounds[1]
    plane_normal_np = np.array([0.0, 0.0, 1.0])

    vertices_tm, faces_tm, _ = tm_intersections.slice_faces_plane(
        mesh_tm.vertices, mesh_tm.faces, plane_normal_np, plane_origin_np
    )
    vertices_wp, faces_wp = tw.intersection.slice_mesh_with_plane(
        mesh_wp.points,
        mesh_wp.indices,
        wp.vec3(*plane_normal_np.tolist()),
        wp.vec3(*plane_origin_np.tolist()),
    )
    vertices_wp_np = vertices_wp.numpy()
    faces_wp_np = faces_wp.numpy().reshape(-1, 3)
    assert len(vertices_tm) == 0
    assert len(faces_tm) == 0
    assert vertices_wp_np.shape[0] == 0
    assert faces_wp_np.shape[0] == 0


# ---------------------------------------------------------------------------
# split_mesh_with_plane
# ---------------------------------------------------------------------------


@pytest.mark.parity("split_mesh_with_plane", "pyvista")
def test_split_mesh_with_plane_matches_pyvista() -> None:
    """
    Class B against ``PolyData.clip(return_clipped=True)``, VTK's both-sides plane clip.

    The named transform is the **side convention**: pyvista's ``clip`` keeps the side the normal
    points *away* from, so its ``kept`` is triwarp's ``~above`` and its ``clipped`` is ``above``
    (measured ``kept z in [-1, 0.1]`` against a plane at ``z = 0.1`` with normal ``+z``). Take the
    names at face value and the two areas are swapped, which the per-side asserts below catch.

    Compared as the *union's* geometry plus the label partition rather than cell-for-cell: VTK
    numbers its output points in its own traversal order and splits the two-crossing quad on its own
    diagonal, so there is no face correspondence to assert. What is asserted is stronger than a
    total: each side's area separately, which pins the partition, and both point sets
    bidirectionally (measured 5.4e-08).
    """
    mesh_tm = tm.creation.icosphere(subdivisions=3, radius=1.0)
    mesh_wp = trimesh_to_warp(mesh_tm, "cpu")
    height = 0.1

    vertices_wp, faces_wp, above_wp = tw.intersection.split_mesh_with_plane(
        mesh_wp.points, mesh_wp.indices, wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 0.0, height)
    )
    kept_pv, clipped_pv = cast(
        "tuple[pv.PolyData, pv.PolyData]",
        trimesh_to_pyvista(mesh_tm).clip(
            normal=(0.0, 0.0, 1.0), origin=(0.0, 0.0, height), return_clipped=True
        ),
    )

    # Anti-vacuity: a plane that missed, or that kept one side only, would pass everything below.
    above_np = above_wp.numpy()
    assert 0 < int(above_np.sum()) < above_np.shape[0]
    assert kept_pv.n_cells > 0
    assert clipped_pv.n_cells > 0

    faces_np = faces_wp.numpy().reshape(-1, 3)
    points_np = vertices_wp.numpy().astype(np.float64)
    assert faces_np.shape[0] == kept_pv.n_cells + clipped_pv.n_cells

    # ``clipped`` is the +normal side, so it pairs with ``above``.
    assert np.isclose(
        tm.Trimesh(points_np, faces_np[above_np], process=False).area, clipped_pv.area, rtol=1e-5
    )
    assert np.isclose(
        tm.Trimesh(points_np, faces_np[~above_np], process=False).area, kept_pv.area, rtol=1e-5
    )

    union_pv = np.vstack([np.asarray(kept_pv.points), np.asarray(clipped_pv.points)])
    assert KDTree(union_pv).query(points_np)[0].max() < 1e-5
    assert KDTree(points_np).query(union_pv)[0].max() < 1e-5


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube", "hemisphere", "half_torus"])
def test_split_mesh_with_plane_refines_without_cracking(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    The invariants that need no reference: crack-free, on-plane, side-pure and area-preserving.

    Every one of these would fail for a per-face cut like
    [`slice_mesh_with_plane`][triwarp.intersection.slice_mesh_with_plane]'s, which is the point of
    the function: a closed input stays closed, the Euler characteristic is unchanged (inserting a
    curve of edges into a triangulation adds equal numbers of vertices, edges and faces), and no
    output face straddles the plane, which is what makes the ``above`` label well defined.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    normal_np = np.array([0.3, -0.5, 1.0])
    normal_np = normal_np / np.linalg.norm(normal_np)
    origin_np = mesh_tm.vertices.mean(axis=0)
    n_vertices_in = int(mesh_wp.points.shape[0])
    closed_in = tw.validation.is_edge_manifold(mesh_wp.indices, allow_boundary_edges=False)

    vertices_wp, faces_wp, above_wp = tw.intersection.split_mesh_with_plane(
        mesh_wp.points, mesh_wp.indices, wp.vec3(*normal_np.tolist()), wp.vec3(*origin_np.tolist())
    )
    points_np = vertices_wp.numpy().astype(np.float64)
    faces_np = faces_wp.numpy().reshape(-1, 3)
    above_np = above_wp.numpy()

    # Anti-vacuity: a plane through the centroid must actually cut.
    assert points_np.shape[0] > n_vertices_in
    assert 0 < int(above_np.sum()) < above_np.shape[0]

    # Every inserted vertex lies on the plane.
    inserted = points_np[n_vertices_in:]
    assert np.abs((inserted - origin_np) @ normal_np).max() < 1e-5

    # No face straddles, so the label is exact rather than a majority vote.
    dots = (points_np[faces_np] - origin_np) @ normal_np
    assert not ((dots > 1e-6).any(axis=1) & (dots < -1e-6).any(axis=1)).any()
    assert (above_np == (dots.max(axis=1) > 1e-6)).all()

    # Crack-free: manifoldness and area survive, and so does the Euler characteristic.
    assert tw.validation.is_edge_manifold(faces_wp, allow_boundary_edges=not closed_in) is True
    if closed_in:
        assert tw.validation.is_edge_manifold(faces_wp, allow_boundary_edges=False) is True
    assert np.isclose(tm.Trimesh(points_np, faces_np, process=False).area, mesh_tm.area, rtol=1e-5)
    assert tw.totals.euler_characteristic(faces_wp) == tw.totals.euler_characteristic(
        mesh_wp.indices
    )


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere"])
def test_split_mesh_with_plane_above_block_is_the_slice(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    The ``above`` submesh is exactly what ``slice_mesh_with_plane`` returns.

    Not triwarp-compared-with-itself for its own sake: the two share no code path — the slice cuts
    per face into three compacted classes, this splits per *edge* and labels afterwards — so
    agreeing on face count and area to ``1e-6`` is a real cross-check of the label convention, which
    is the one thing a caller has to get right. It also pins the documented promise that the two
    agree, including the in-plane-face tie-break.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    normal = wp.vec3(0.0, 0.0, 1.0)
    origin = wp.vec3(*mesh_tm.vertices.mean(axis=0).tolist())

    split_v, split_f, above = tw.intersection.split_mesh_with_plane(
        mesh_wp.points, mesh_wp.indices, normal, origin
    )
    above_v, above_f = tw.selection.submesh_from_face_mask(split_v, split_f, above)
    slice_v, slice_f = tw.intersection.slice_mesh_with_plane(
        mesh_wp.points, mesh_wp.indices, normal, origin
    )

    assert int(above_f.shape[0]) > 0
    assert int(above_f.shape[0]) == int(slice_f.shape[0])
    assert np.isclose(
        tm.Trimesh(
            above_v.numpy().astype(np.float64), above_f.numpy().reshape(-1, 3), process=False
        ).area,
        tm.Trimesh(
            slice_v.numpy().astype(np.float64), slice_f.numpy().reshape(-1, 3), process=False
        ).area,
        rtol=1e-6,
    )


def test_split_mesh_with_plane_through_a_vertex_inserts_nothing_there(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    A vertex already on the plane is used as the crossing rather than duplicated beside it.

    This is what the ``tolerance`` parameter buys, and the assert that bites is the *count*: without
    the strict-opposite-signs test every edge incident to the on-plane vertex would also be
    "crossed" at a point coinciding with it, giving a fan of zero-length edges and degenerate
    faces. Checked by the count of inserted vertices and by the absence of a degenerate face.
    """
    mesh_tm, mesh_wp = icosahedron
    apex = int(np.argmax(mesh_tm.vertices[:, 2]))
    normal_np = np.array([0.0, 0.0, 1.0])
    origin_np = mesh_tm.vertices[apex]
    n_vertices_in = int(mesh_wp.points.shape[0])

    vertices_wp, faces_wp, _ = tw.intersection.split_mesh_with_plane(
        mesh_wp.points, mesh_wp.indices, wp.vec3(*normal_np.tolist()), wp.vec3(*origin_np.tolist())
    )
    points_np = vertices_wp.numpy().astype(np.float64)

    # The apex is the unique highest vertex of an icosahedron, so a plane through it touches the
    # surface at that point alone: nothing is crossed and nothing is inserted.
    assert points_np.shape[0] == n_vertices_in
    assert int(faces_wp.shape[0]) == int(mesh_wp.indices.shape[0])
    # No zero-area face was introduced anywhere.
    assert (
        tm.Trimesh(points_np, faces_wp.numpy().reshape(-1, 3), process=False).area_faces.min() > 0
    )


def test_split_mesh_with_plane_misses_the_mesh(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """A plane clear of the mesh returns it unchanged with a constant label."""
    mesh_tm, mesh_wp = icosahedron
    vertices_wp, faces_wp, above_wp = tw.intersection.split_mesh_with_plane(
        mesh_wp.points,
        mesh_wp.indices,
        wp.vec3(0.0, 0.0, 1.0),
        wp.vec3(0.0, 0.0, float(mesh_tm.bounds[1][2]) + 1.0),
    )
    assert np.array_equal(faces_wp.numpy(), mesh_wp.indices.numpy())
    assert np.allclose(vertices_wp.numpy(), mesh_wp.points.numpy())
    assert not above_wp.numpy().any()


def test_split_mesh_with_plane_empty(device: str) -> None:
    vertices_wp = wp.empty(0, dtype=wp.vec3, device=device)
    faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    out_vertices_wp, out_faces_wp, above_wp = tw.intersection.split_mesh_with_plane(
        vertices_wp, faces_wp, wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 0.0, 0.0)
    )
    assert out_vertices_wp.shape == (0,)
    assert out_faces_wp.shape == (0,)
    assert above_wp.shape == (0,)


# ---------------------------------------------------------------------------
# clip_mesh_with_field
# ---------------------------------------------------------------------------


def _height_field(mesh_tm: tm.Trimesh, device: str) -> wp.array[wp.float32]:
    """Take the z coordinate as a per-vertex ``float32`` field: a horizontal plane's distance."""
    return wp.array(
        np.ascontiguousarray(mesh_tm.vertices[:, 2], dtype=np.float32),
        dtype=wp.float32,
        device=device,
    )


def test_clip_mesh_with_field_empty(device: str) -> None:
    vertices_wp = wp.empty(0, dtype=wp.vec3, device=device)
    faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    values_wp = wp.empty(0, dtype=wp.float32, device=device)
    out_vertices_wp, out_faces_wp = tw.intersection.clip_mesh_with_field(
        vertices_wp, faces_wp, values_wp
    )
    assert out_vertices_wp.shape == (0,)
    assert out_faces_wp.shape == (0,)


@pytest.mark.parity("clip_mesh_with_field", "pyvista")
def test_clip_mesh_with_field_matches_pyvista_clip_scalar() -> None:
    """
    Class A on the kept surface, against ``PolyData.clip_scalar`` over the identical field.

    VTK cuts the same triangles at the same crossings, so the face counts are equal and the
    positions agree as point sets — measured 5.4e-08 on ``icosphere(3)``. The vertex *order* differs
    because each side appends its crossing points in its own traversal order, hence the
    nearest-neighbour comparison rather than an element-wise one.

    ``invert=False`` is not optional and is the trap in this row: ``clip_scalar``'s **default keeps
    the low side** (measured 798 faces below ``z = 0.1`` against 670 above), where triwarp keeps
    ``values >= isovalue``. Take the default and the two answers are different regions of the same
    mesh, which the face-count assert catches only because they happen to differ in size.
    """
    mesh_tm = tm.creation.icosphere(subdivisions=3, radius=1.0)
    mesh_wp = trimesh_to_warp(mesh_tm, "cpu")
    isovalue = 0.1

    clipped_v, clipped_f = tw.intersection.clip_mesh_with_field(
        mesh_wp.points, mesh_wp.indices, _height_field(mesh_tm, "cpu"), isovalue
    )
    mesh_pv = trimesh_to_pyvista(mesh_tm)
    mesh_pv.point_data["height"] = np.ascontiguousarray(mesh_tm.vertices[:, 2])
    clipped_pv = cast(
        pv.PolyData, mesh_pv.clip_scalar(scalars="height", value=isovalue, invert=False)
    )

    # Anti-vacuity: a clip that kept nothing, or everything, would pass the comparisons below.
    assert 0 < clipped_pv.n_faces < len(mesh_tm.faces)
    assert int(clipped_f.shape[0]) // 3 == clipped_pv.n_faces
    points_np = clipped_v.numpy().astype(np.float64)
    points_pv = np.asarray(clipped_pv.points)
    assert KDTree(points_pv).query(points_np)[0].max() < 1e-5
    assert KDTree(points_np).query(points_pv)[0].max() < 1e-5
    assert np.isclose(
        tm.Trimesh(points_np, clipped_f.numpy().reshape(-1, 3), process=False).area,
        clipped_pv.area,
        rtol=1e-5,
    )
    # Nothing below the isovalue survived.
    assert points_np[:, 2].min() >= isovalue - 1e-5


@pytest.mark.parity("clip_mesh_with_field", "pyvista")
def test_clip_mesh_with_field_capped_matches_pyvista_clip_closed_surface() -> None:
    """
    Class A on the enclosed volume, against ``clip_closed_surface`` — VTK's capped plane clip.

    The two cappers triangulate the section differently (a min-weight interval DP here, VTK's own
    there), so the comparison is the *solid* rather than the triangles: measured the same 762 faces
    and the same volume to seven digits on ``icosphere(3)`` at ``z = 0.1``. Watertightness is
    asserted on both sides, which is the property the cap exists to restore and the one a cracked
    section rim would break.
    """
    mesh_tm = tm.creation.icosphere(subdivisions=3, radius=1.0)
    mesh_wp = trimesh_to_warp(mesh_tm, "cpu")
    isovalue = 0.1

    capped_v, capped_f = tw.intersection.clip_mesh_with_field(
        mesh_wp.points, mesh_wp.indices, _height_field(mesh_tm, "cpu"), isovalue, cap=True
    )
    capped_tm = tm.Trimesh(
        capped_v.numpy().astype(np.float64), capped_f.numpy().reshape(-1, 3), process=False
    )
    mesh_pv = trimesh_to_pyvista(mesh_tm)
    closed_pv = cast(
        pv.PolyData,
        mesh_pv.clip_closed_surface(normal=(0.0, 0.0, 1.0), origin=(0.0, 0.0, isovalue)),
    )

    assert closed_pv.n_open_edges == 0
    assert capped_tm.is_watertight
    assert tw.validation.is_edge_manifold(capped_f, allow_boundary_edges=False)
    assert np.isclose(capped_tm.volume, closed_pv.volume, rtol=1e-5)
    # The cap is not free: without it the same clip is open.
    _, uncapped_f = tw.intersection.clip_mesh_with_field(
        mesh_wp.points, mesh_wp.indices, _height_field(mesh_tm, "cpu"), isovalue
    )
    assert int(capped_f.shape[0]) > int(uncapped_f.shape[0])


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere"])
def test_clip_mesh_with_field_reproduces_slice_mesh_with_plane(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    The plane clip *is* this function over the plane's signed distance, so the two must agree.

    No reference: this pins the delegation itself. The one documented difference is the face lying
    in the level set, which the plane resolves from its normal and the field cannot — so the plane
    used here misses every vertex, keeping the two paths comparable face for face.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    plane_origin_np = 0.5 * (mesh_tm.bounds[0] + mesh_tm.bounds[1])
    plane_normal_np = np.array([0.0, 0.0, 1.0])
    field_wp = wp.array(
        np.ascontiguousarray(
            (mesh_tm.vertices - plane_origin_np) @ plane_normal_np, dtype=np.float32
        ),
        dtype=wp.float32,
        device=mesh_wp.device,
    )

    sliced_v, sliced_f = tw.intersection.slice_mesh_with_plane(
        mesh_wp.points,
        mesh_wp.indices,
        wp.vec3(*plane_normal_np.tolist()),
        wp.vec3(*plane_origin_np.tolist()),
    )
    clipped_v, clipped_f = tw.intersection.clip_mesh_with_field(
        mesh_wp.points, mesh_wp.indices, field_wp
    )
    assert int(sliced_f.shape[0]) > 0
    assert np.array_equal(clipped_f.numpy(), sliced_f.numpy())
    assert np.allclose(clipped_v.numpy(), sliced_v.numpy(), rtol=1e-5, atol=1e-5)


def test_clip_mesh_with_field_section_is_the_marching_triangles_curve(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Round trip for the region/level-set pair: the clip's rim is the contour, edge for edge.

    [`marching_triangles`][triwarp.intersection.marching_triangles] returns the level set and this
    returns the region on one side of it, so the boundary of the region has to *be* the level set.
    Compared as total length plus a two-sided point-set distance, because the clip's rim carries one
    vertex per crossing where the contour carries one per segment endpoint.

    The isovalue deliberately misses every vertex, asserted below. Four of the icosahedron's twelve
    sit at exactly the centroid's height, and a contour through a vertex is where the two functions
    legitimately differ: the contour reports a zero-length segment there (documented) while the clip
    has a single rim vertex, so the counts stop matching for a reason that is not a bug.
    """
    mesh_tm, mesh_wp = icosahedron
    isovalue = float(mesh_tm.centroid[2]) + 0.17
    field_np = mesh_tm.vertices[:, 2]
    assert np.abs(field_np - isovalue).min() > 1e-3, "the isovalue must miss every vertex"
    field_wp = wp.array(
        np.ascontiguousarray(field_np, dtype=np.float32), dtype=wp.float32, device=mesh_wp.device
    )

    curves, closed = tw.intersection.marching_triangles(
        mesh_wp.points, mesh_wp.indices, field_wp, isovalue
    )
    assert len(curves) == 1
    contour_np = curves[0].numpy().astype(np.float64)

    clipped_v, clipped_f = tw.intersection.clip_mesh_with_field(
        mesh_wp.points, mesh_wp.indices, field_wp, isovalue
    )
    welded_v, _unique, _inverse, welded_f = tw.repair.remove_duplicated_vertices(
        clipped_v, clipped_f
    )
    rim_edges_np = tw.boundary.boundary_edges(welded_v, welded_f).numpy()
    positions_np = welded_v.numpy().astype(np.float64)
    rim_np = positions_np[np.unique(rim_edges_np)]
    assert rim_np.shape[0] == contour_np.shape[0]
    assert hausdorff_two_sided(rim_np, contour_np) < 1e-5

    rim_segments_np = positions_np[rim_edges_np]
    rim_length = float(np.linalg.norm(rim_segments_np[:, 1] - rim_segments_np[:, 0], axis=1).sum())
    contour_length = _total_length([curve.numpy().astype(np.float64) for curve in curves], closed)
    assert np.isclose(rim_length, contour_length, rtol=1e-4)


def test_clip_mesh_with_field_accepts_a_float64_field(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """A ``float64`` field — what ``heat_geodesic`` returns — clips the same region as its cast."""
    mesh_tm, mesh_wp = icosahedron
    field_np = mesh_tm.vertices[:, 2] - mesh_tm.centroid[2]
    isovalue = 0.05
    clipped_64 = tw.intersection.clip_mesh_with_field(
        mesh_wp.points,
        mesh_wp.indices,
        wp.array(np.ascontiguousarray(field_np), dtype=wp.float64, device=mesh_wp.device),
        isovalue,
    )
    clipped_32 = tw.intersection.clip_mesh_with_field(
        mesh_wp.points,
        mesh_wp.indices,
        wp.array(
            np.ascontiguousarray(field_np, dtype=np.float32),
            dtype=wp.float32,
            device=mesh_wp.device,
        ),
        isovalue,
    )
    assert int(clipped_64[1].shape[0]) > 0
    assert np.array_equal(clipped_64[1].numpy(), clipped_32[1].numpy())
    assert np.allclose(clipped_64[0].numpy(), clipped_32[0].numpy(), rtol=1e-5, atol=1e-5)


def _pyvista_intersection_segments(mesh1_pv: pv.PolyData, mesh2_pv: pv.PolyData) -> np.ndarray:
    intersection_pv = cast(
        pv.PolyData, mesh1_pv.intersection(mesh2_pv, split_first=False, split_second=False)[0]
    )
    if intersection_pv.n_cells == 0:
        return np.empty((0, 2, 3), dtype=np.float64)
    pairs_np = np.reshape(intersection_pv.lines, (-1, 3))[:, 1:]
    return intersection_pv.points[pairs_np]


def _intersection_curves_match(
    got_segments_np: np.ndarray,
    ref_segments_np: np.ndarray,
    *,
    ref_atol: float = 1e-5,
    got_atol: float = 1e-5,
) -> bool:
    """Check both segment sets describe the same intersection curves."""
    got_segments_np = got_segments_np.reshape(-1, 2, 3)
    ref_segments_np = ref_segments_np.reshape(-1, 2, 3)
    if ref_segments_np.shape[0] == 0:
        return got_segments_np.shape[0] == 0
    if got_segments_np.shape[0] == 0:
        return False
    got_pts_np = got_segments_np.reshape(-1, 3)
    ref_pts_np = ref_segments_np.reshape(-1, 3)
    ref_distances_np = KDTree(got_pts_np).query(ref_pts_np, distance_upper_bound=ref_atol)[0]
    got_distances_np = KDTree(ref_pts_np).query(got_pts_np, distance_upper_bound=got_atol)[0]
    return bool(np.all(np.isfinite(ref_distances_np)) and np.all(np.isfinite(got_distances_np)))


def test_mesh_with_mesh_empty(
    icosahedron: tuple[tm.Trimesh, wp.Mesh], cave_cube: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    _, ico_wp = icosahedron
    _, cave_wp = cave_cube

    lines_wp = tw.intersection.mesh_with_mesh(
        ico_wp.points, ico_wp.indices, cave_wp.points, cave_wp.indices
    )
    assert lines_wp.shape == (0, 2)


def test_mesh_with_mesh_icosahedron_cave_cube(
    icosahedron: tuple[tm.Trimesh, wp.Mesh], cave_cube: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    ico_tm, ico_wp = icosahedron
    cave_tm, _ = cave_cube
    cave_at_ico_tm = cave_tm.copy()
    cave_at_ico_tm.apply_translation(ico_tm.centroid)
    cave_wp = trimesh_to_warp(cave_at_ico_tm, ico_wp.device)

    ref_segments_np = _pyvista_intersection_segments(
        trimesh_to_pyvista(ico_tm), trimesh_to_pyvista(cave_at_ico_tm)
    )
    lines_wp = tw.intersection.mesh_with_mesh(
        ico_wp.points, ico_wp.indices, cave_wp.points, cave_wp.indices
    )

    assert lines_wp.shape[0] > 0
    assert _intersection_curves_match(lines_wp.numpy(), ref_segments_np)


# --- marching_triangles: isocontours of a scalar field (potpourri3d reference) ---------
_MESHES = ["icosahedron", "cave_cube", "hemisphere", "half_torus"]


def _curves_pp(
    vertices_np: np.ndarray, faces_np: np.ndarray, values_np: np.ndarray, isovalue: float
) -> tuple[list[np.ndarray], list[bool]]:
    """Decode potpourri3d's barycentric contour output into point arrays and closed flags."""
    edges_pp = np.asarray(pp3d.edges(vertices_np, faces_np))
    curves: list[np.ndarray] = []
    closed: list[bool] = []
    for curve in pp3d.marching_triangles(vertices_np, faces_np, values_np, isovalue):
        points = []
        for element, barycentric in curve:
            if len(barycentric) == 0:  # a vertex
                points.append(vertices_np[element])
            elif len(barycentric) == 1:  # a point along an edge
                start, end = edges_pp[element]
                points.append(
                    (1.0 - barycentric[0]) * vertices_np[start] + barycentric[0] * vertices_np[end]
                )
            else:  # a point inside a face
                weights = np.array(
                    [barycentric[0], barycentric[1], 1.0 - barycentric[0] - barycentric[1]]
                )
                points.append(weights @ vertices_np[faces_np[element]])
        is_closed = len(points) > 2 and np.allclose(points[0], points[-1])
        curves.append(np.array(points[:-1] if is_closed else points))
        closed.append(is_closed)
    return curves, closed


def _total_length(curves: list[np.ndarray], closed: list[bool]) -> float:
    """Sum the curve lengths, counting the closing chord of every closed curve."""
    total = 0.0
    for points, is_closed in zip(curves, closed, strict=True):
        total += float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        if is_closed:
            total += float(np.linalg.norm(points[0] - points[-1]))
    return total


# ---------------------------------------------------------------------------
# marching_triangles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", _MESHES)
@pytest.mark.parametrize("axis", [0, 2])
@pytest.mark.parity("marching_triangles", "potpourri3d")
def test_marching_triangles_matches_potpourri3d(
    request: pytest.FixtureRequest, mesh_name: str, axis: int, device: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.ascontiguousarray(mesh_tm.faces, dtype=np.int32)
    # A coordinate function, contoured a little off centre so the level set misses the vertices: an
    # exact vertex hit is a genuine convention difference (see the dedicated test below).
    values_np = np.ascontiguousarray(vertices_np[:, axis])
    isovalue = float(0.5137 * values_np.min() + 0.4863 * values_np.max())
    values_wp = wp.array(values_np, dtype=wp.float64, device=mesh_wp.device)

    curves_wp, closed_wp = tw.intersection.marching_triangles(
        mesh_wp.points, mesh_wp.indices, values_wp, isovalue, n_vertices=len(vertices_np)
    )
    curves_pp, closed_pp = _curves_pp(vertices_np, faces_np, values_np, isovalue)

    assert len(curves_wp) == len(curves_pp)
    assert sorted(closed_wp) == sorted(closed_pp)
    assert np.isclose(
        _total_length([curve.numpy() for curve in curves_wp], closed_wp),
        _total_length(curves_pp, closed_pp),
        rtol=1e-5,
        atol=1e-5,
    )
    # The level sets must coincide as point sets, not merely in total length.
    bounding_diagonal = float(np.linalg.norm(vertices_np.max(axis=0) - vertices_np.min(axis=0)))
    assert (
        hausdorff_two_sided(
            np.concatenate([curve.numpy() for curve in curves_wp]), np.concatenate(curves_pp)
        )
        < 1e-6 * bounding_diagonal
    )


@pytest.mark.parametrize("mesh_name", _MESHES)
@pytest.mark.parity("marching_triangles", "igl")
@pytest.mark.parity("marching_triangles_curves", "igl")
def test_marching_triangles_matches_igl(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    """
    Class B: ``igl.isolines`` returns a segment **soup**, so the linking must be undone first.

    igl gives ``(points, segments, segment_values)`` with no curve structure at all -- every
    crossing is an independent 2-point segment -- where ``marching_triangles`` returns polylines.
    The named transform is therefore to reduce triwarp's curves to the same soup: each consecutive
    pair of a curve is a segment, plus the closing pair for a closed curve. Two quantities are then
    directly comparable and both are asserted: the total segment length, and the point sets through
    a two-sided Hausdorff distance.

    Segment *count* is not compared, and that is deliberate: a polyline of ``n`` points contributes
    ``n - 1`` segments (``n`` closed), so triwarp's count is derived from its linking while igl's is
    the raw crossing count -- they agree here but the equality is a property of this input rather
    than of the two algorithms, and asserting it would be asserting the wrong thing.

    Both this group and ``marching_triangles_curves`` are covered because the transform is the same
    for one long contour and for a thousand short ones; the linking count is what differs, and it is
    exactly what this comparison steps around.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.ascontiguousarray(mesh_tm.faces, dtype=np.int64)
    values_np = np.ascontiguousarray(vertices_np[:, 2])
    isovalue = float(0.5137 * values_np.min() + 0.4863 * values_np.max())
    values_wp = wp.array(values_np, dtype=wp.float64, device=mesh_wp.device)

    points_igl, segments_igl, _values_igl = igl.isolines(
        vertices_np, faces_np, values_np, np.array([isovalue])
    )
    curves_wp, closed_wp = tw.intersection.marching_triangles(
        mesh_wp.points, mesh_wp.indices, values_wp, isovalue, n_vertices=len(vertices_np)
    )

    assert segments_igl.shape[0] > 0
    length_igl = float(
        np.linalg.norm(
            points_igl[segments_igl[:, 0]] - points_igl[segments_igl[:, 1]], axis=1
        ).sum()
    )
    assert np.isclose(
        _total_length([curve.numpy() for curve in curves_wp], closed_wp),
        length_igl,
        rtol=1e-5,
        atol=1e-5,
    )

    bounding_diagonal = float(np.linalg.norm(vertices_np.max(axis=0) - vertices_np.min(axis=0)))
    assert (
        hausdorff_two_sided(np.concatenate([curve.numpy() for curve in curves_wp]), points_igl)
        < 1e-5 * bounding_diagonal
    )


@pytest.mark.parity("marching_triangles_curves", "potpourri3d")
def test_marching_triangles_many_components_matches_potpourri3d(device: str) -> None:
    # An oscillating field breaks the level set into many small loops, which is what exercises the
    # segment linking rather than the per-face crossing arithmetic.
    mesh_tm = tw.creation.icosphere(subdivisions=3)
    vertices_np = np.ascontiguousarray(mesh_tm[0].numpy(), dtype=np.float64)
    faces_np = np.ascontiguousarray(mesh_tm[1].numpy().reshape(-1, 3), dtype=np.int32)
    values_np = np.ascontiguousarray(
        np.sin(8.0 * vertices_np[:, 0]) * np.cos(8.0 * vertices_np[:, 1])
    )
    isovalue = 0.1370

    vertices_wp = wp.array(vertices_np.astype(np.float32), dtype=wp.vec3, device=device)
    faces_wp = wp.array(faces_np.reshape(-1), dtype=wp.int32, device=device)
    values_wp = wp.array(values_np, dtype=wp.float64, device=device)

    curves_wp, closed_wp = tw.intersection.marching_triangles(
        vertices_wp, faces_wp, values_wp, isovalue, n_vertices=len(vertices_np)
    )
    curves_pp, closed_pp = _curves_pp(vertices_np, faces_np, values_np, isovalue)

    assert len(curves_wp) > 10
    assert len(curves_wp) == len(curves_pp)
    assert all(closed_wp)
    assert np.isclose(
        _total_length([curve.numpy() for curve in curves_wp], closed_wp),
        _total_length(curves_pp, closed_pp),
        rtol=1e-5,
        atol=1e-5,
    )


def test_marching_triangles_open_curve_ends_on_the_boundary(
    hemisphere: tuple[object, wp.Mesh], device: str
) -> None:
    mesh_tm, mesh_wp = hemisphere
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)  # type: ignore[attr-defined]
    # The hemisphere's rim is a single loop, so a level set of x has to run into it.
    values_np = np.ascontiguousarray(vertices_np[:, 0])
    isovalue = float(values_np.mean())
    values_wp = wp.array(values_np, dtype=wp.float64, device=mesh_wp.device)

    curves_wp, closed_wp = tw.intersection.marching_triangles(
        mesh_wp.points, mesh_wp.indices, values_wp, isovalue, n_vertices=len(vertices_np)
    )

    assert not all(closed_wp)
    boundary_vertices = mesh_wp.points.numpy()[
        tw.boundary.boundary_vertex_indices(mesh_wp.points, mesh_wp.indices).numpy()
    ]
    for points, is_closed in zip([curve.numpy() for curve in curves_wp], closed_wp, strict=True):
        if is_closed:
            continue
        # Both ends of an open curve sit on a boundary edge, hence within one edge length of a
        # boundary vertex.
        edge_length = tw.edges.mean_edge_length(mesh_wp.points, mesh_wp.indices)
        for end in (points[0], points[-1]):
            assert np.linalg.norm(boundary_vertices - end, axis=1).min() <= edge_length


def test_marching_triangles_exact_vertex_hit_is_reported_once(device: str) -> None:
    # Two triangles sharing edge (1, 2), with the field vanishing exactly at both shared vertices,
    # so the level set *is* that edge. Counting a value equal to the isovalue as positive leaves the
    # all-positive face with nothing to report and the mixed-sign face with one segment along the
    # shared edge: the curve appears exactly once rather than twice or not at all.
    vertices_wp = wp.array(
        np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        dtype=wp.vec3,
        device=device,
    )
    faces_wp = wp.array(np.array([0, 1, 2, 3, 2, 1], dtype=np.int32), dtype=wp.int32, device=device)
    values_wp = wp.array(np.array([-1.0, 0.0, 0.0, 1.0], dtype=np.float32), device=device)

    curves_wp, closed_wp = tw.intersection.marching_triangles(
        vertices_wp, faces_wp, values_wp, 0.0, n_vertices=4
    )

    assert len(curves_wp) == 1
    assert closed_wp == [False]
    assert curves_wp[0].shape == (2,)
    # The two endpoints are the shared vertices themselves.
    assert np.allclose(
        np.sort(curves_wp[0].numpy(), axis=0),
        np.array([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_marching_triangles_level_set_is_the_piecewise_linear_one(
    icosahedron: tuple[object, wp.Mesh], device: str
) -> None:
    mesh_tm, mesh_wp = icosahedron
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)  # type: ignore[attr-defined]
    faces_np = np.asarray(mesh_tm.faces)  # type: ignore[attr-defined]
    values_np = np.ascontiguousarray(vertices_np[:, 2])
    isovalue = 0.1234
    values_wp = wp.array(values_np, dtype=wp.float64, device=mesh_wp.device)

    curves_wp, _ = tw.intersection.marching_triangles(
        mesh_wp.points, mesh_wp.indices, values_wp, isovalue, n_vertices=len(vertices_np)
    )

    # One segment per face whose vertex values straddle the isovalue, and every segment lands in a
    # curve: the total point count equals the cut-face count for an all-closed level set.
    positive = values_np[faces_np] >= isovalue
    cut_faces = int((~(positive.all(axis=1) | (~positive).all(axis=1))).sum())
    assert sum(int(curve.shape[0]) for curve in curves_wp) == cut_faces


def test_marching_triangles_no_crossing(icosahedron: tuple[object, wp.Mesh], device: str) -> None:
    mesh_tm, mesh_wp = icosahedron
    values_wp = wp.array(
        np.ascontiguousarray(np.asarray(mesh_tm.vertices)[:, 2]),  # type: ignore[attr-defined]
        dtype=wp.float64,
        device=mesh_wp.device,
    )
    assert tw.intersection.marching_triangles(mesh_wp.points, mesh_wp.indices, values_wp, 1e6) == (
        [],
        [],
    )


def test_marching_triangles_empty(device: str) -> None:
    vertices_wp = wp.empty(0, dtype=wp.vec3, device=device)
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    values_wp = wp.empty(0, dtype=wp.float32, device=device)
    assert tw.intersection.marching_triangles(vertices_wp, faces_wp, values_wp) == ([], [])
