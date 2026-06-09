"""Regression tests for ``triwarp.intersections`` against Trimesh (CPU reference)."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh as tm
import trimesh.intersections as tm_intersections
import warp as wp

import triwarp as tw


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
        np.lexsort((ordered[:, 3], ordered[:, 4], ordered[:, 5], ordered[:, 0], ordered[:, 1], ordered[:, 2]))
    ].reshape(-1, 2, 3)


def _segments_equal(got_np: np.ndarray, exp_np: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-5) -> bool:
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
    assert np.allclose(intersections_wp.numpy()[valid_wp.numpy()], intersections_tm, rtol=1e-5, atol=1e-5)


def test_segments_with_plane_parallel(device: str) -> None:
    plane_origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    plane_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    endpoints_np = np.array([[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]], dtype=np.float32)
    endpoints_np = np.transpose(endpoints_np, (1, 0, 2))
    _, valid_tm = tm_intersections.plane_lines(plane_origin, plane_normal, endpoints_np, line_segments=True)

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
    lines_wp = tw.intersections.mesh_with_plane(vertices_wp, faces_wp, wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 0.0, 0.0))
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
            mesh=mesh_tm,
            plane_normal=plane_normal,
            plane_origin=plane_origin,
        )
        lines_wp = tw.intersections.mesh_with_plane(
            mesh_wp.points,
            mesh_wp.indices,
            wp.vec3(*map(float, np.asanyarray(plane_normal, dtype=np.float32).reshape(3))),
            wp.vec3(*map(float, np.asanyarray(plane_origin, dtype=np.float32).reshape(3))),
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
        mesh=mesh_tm,
        plane_normal=plane_normal,
        plane_origin=plane_origin,
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
        mesh=mesh_tm,
        plane_normal=plane_normal,
        plane_origin=plane_origin,
        return_faces=True,
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
