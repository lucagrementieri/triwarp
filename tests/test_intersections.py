"""Regression tests for ``triwarp.intersections`` against Trimesh (CPU reference)."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import pyvista as pv
import trimesh as tm
import trimesh.intersections as tm_intersections
import warp as wp
from scipy.spatial import KDTree

import triwarp as tw
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
    intersections_wp, valid_wp = tw.intersections.segments_with_plane(
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
    _, valid_wp = tw.intersections.segments_with_plane(
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
    lines_wp = tw.intersections.mesh_with_plane(
        vertices_wp, faces_wp, wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 0.0, 0.0)
    )
    assert lines_wp.shape == (0, 2)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
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
        lines_wp = tw.intersections.mesh_with_plane(
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
    lines_wp = tw.intersections.mesh_with_plane(
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
    lines_wp, faces_wp = tw.intersections.mesh_with_plane(
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

    lines_wp = tw.intersections.mesh_with_plane(
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
    out_vertices_wp, out_faces_wp = tw.intersections.slice_mesh_with_plane(
        vertices_wp, faces_wp, wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 0.0, 0.0)
    )
    assert out_vertices_wp.shape == (0,)
    assert out_faces_wp.shape == (0,)


def test_slice_mesh_with_plane_box_corner() -> None:
    mesh_tm = tm.creation.box()
    plane_origin_np = mesh_tm.bounds[1] - 0.05
    plane_normal_np = mesh_tm.bounds[1]

    vertices_tm, faces_tm, _ = tm_intersections.slice_faces_plane(
        mesh_tm.vertices, mesh_tm.faces, plane_normal_np, plane_origin_np
    )
    mesh_wp = trimesh_to_warp(mesh_tm, "cpu")
    vertices_wp, faces_wp = tw.intersections.slice_mesh_with_plane(
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
    vertices_wp, faces_wp = tw.intersections.slice_mesh_with_plane(
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
        vertices_wp, faces_wp = tw.intersections.slice_mesh_with_plane(
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
    vertices_wp, faces_wp = tw.intersections.slice_mesh_with_plane(
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


def test_slice_mesh_with_plane_on_plane(icosphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = icosphere
    plane_origin_np = mesh_tm.bounds[1]
    plane_normal_np = np.array([0.0, 0.0, 1.0])

    vertices_tm, faces_tm, _ = tm_intersections.slice_faces_plane(
        mesh_tm.vertices, mesh_tm.faces, plane_normal_np, plane_origin_np
    )
    vertices_wp, faces_wp = tw.intersections.slice_mesh_with_plane(
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

    lines_wp = tw.intersections.mesh_with_mesh(
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
    lines_wp = tw.intersections.mesh_with_mesh(
        ico_wp.points, ico_wp.indices, cave_wp.points, cave_wp.indices
    )

    assert lines_wp.shape[0] > 0
    assert _intersection_curves_match(lines_wp.numpy(), ref_segments_np)
