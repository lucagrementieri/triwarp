"""
Regression tests for ``triwarp.tracing`` against potpourri3d (CPU reference).

Two things are checked independently of the reference, because they are what "a geodesic" means: the
traced arc length equals the requested one (the direction's tangential magnitude), and every traced
point lies on the surface. Against potpourri3d the arc lengths agree exactly; the *endpoints* agree
only to a fraction of an edge length, because a path crossing a vertex has no unique straightest
continuation and the two libraries resolve that differently (see the module docstring).
"""

from __future__ import annotations

import numpy as np
import potpourri3d as pp3d
import pytest
import trimesh as tm
import warp as wp

import triwarp as tw

_MESHES = ["icosahedron", "cave_cube", "hemisphere", "half_torus"]


def _rays(mesh_tm: tm.Trimesh, n_rays: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Random start vertices and directions, scaled to a few edge lengths."""
    rng = np.random.default_rng(seed)
    start = rng.integers(0, len(mesh_tm.vertices), n_rays).astype(np.int32)
    scale = 3.0 * float(
        np.linalg.norm(
            mesh_tm.vertices[mesh_tm.edges[:, 1]] - mesh_tm.vertices[mesh_tm.edges[:, 0]], axis=1
        ).mean()
    )
    directions = rng.normal(size=(n_rays, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return start, (scale * directions).astype(np.float32)


def _tangential_length(direction: np.ndarray, normal: np.ndarray) -> float:
    return float(np.linalg.norm(direction - np.dot(direction, normal) * normal))


def _path_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


# ---------------------------------------------------------------------------
# trace_geodesic_from_vertex
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", _MESHES)
def test_trace_from_vertex_walks_the_requested_distance(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    start_np, directions_np = _rays(mesh_tm, 24, seed=0)
    frames_wp = tw.tangent_space.vertex_tangent_frames(mesh_wp.points, mesh_wp.indices)
    points_wp, offsets_wp = tw.tracing.trace_geodesic_from_vertex(
        mesh_wp.points,
        mesh_wp.indices,
        wp.array(start_np, dtype=wp.int32, device=mesh_wp.device),
        wp.array(directions_np, dtype=wp.vec3, device=mesh_wp.device),
        frames=frames_wp,
    )

    normals = frames_wp[2].numpy()
    is_boundary = tw.halfedge.vertex_one_rings(mesh_wp.indices, n_vertices=len(mesh_tm.vertices))[
        2
    ].numpy()
    curves = tw.tracing.trace_geodesic_polylines(points_wp, offsets_wp)
    for ray, (start, direction) in enumerate(zip(start_np, directions_np, strict=True)):
        points = curves[ray].numpy()
        requested = _tangential_length(direction.astype(np.float64), normals[start])
        # A ray reaching the boundary stops early, and one leaving a boundary vertex's fan does not
        # start at all, so on an open mesh the requested length is only an upper bound.
        traced = _path_length(points)
        assert traced <= requested * (1.0 + 1e-4) + 1e-6
        if not is_boundary.any():
            assert np.isclose(traced, requested, rtol=1e-4, atol=1e-5)
        # The path starts where it was asked to.
        assert np.allclose(points[0], mesh_tm.vertices[start], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", _MESHES)
def test_trace_from_vertex_stays_on_the_surface(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    start_np, directions_np = _rays(mesh_tm, 24, seed=1)
    points_wp, _ = tw.tracing.trace_geodesic_from_vertex(
        mesh_wp.points,
        mesh_wp.indices,
        wp.array(start_np, dtype=wp.int32, device=mesh_wp.device),
        wp.array(directions_np, dtype=wp.vec3, device=mesh_wp.device),
    )

    # Every traced point must lie on a triangle: an unfolding error would drift off the surface.
    distance_tm = np.abs(
        tm.proximity.signed_distance(mesh_tm, points_wp.numpy().astype(np.float64))
    )
    scale = float(np.linalg.norm(mesh_tm.vertices.max(axis=0) - mesh_tm.vertices.min(axis=0)))
    assert distance_tm.max() < 1e-5 * scale


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube"])
@pytest.mark.parity("trace_geodesic_rays", "potpourri3d")
def test_trace_from_vertex_matches_potpourri3d(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.ascontiguousarray(mesh_tm.faces, dtype=np.int32)
    start_np, directions_np = _rays(mesh_tm, 12, seed=2)

    points_wp, offsets_wp = tw.tracing.trace_geodesic_from_vertex(
        mesh_wp.points,
        mesh_wp.indices,
        wp.array(start_np, dtype=wp.int32, device=mesh_wp.device),
        wp.array(directions_np, dtype=wp.vec3, device=mesh_wp.device),
    )
    curves = tw.tracing.trace_geodesic_polylines(points_wp, offsets_wp)

    tracer_pp = pp3d.GeodesicTracer(vertices_np, faces_np)
    edge_length = float(
        np.linalg.norm(
            mesh_tm.vertices[mesh_tm.edges[:, 1]] - mesh_tm.vertices[mesh_tm.edges[:, 0]], axis=1
        ).mean()
    )
    for ray, (start, direction) in enumerate(zip(start_np, directions_np, strict=True)):
        path_pp = np.asarray(
            tracer_pp.trace_geodesic_from_vertex(int(start), direction.astype(np.float64))
        )
        points = curves[ray].numpy()
        # The arc length is the contract and matches exactly.
        assert np.isclose(_path_length(points), _path_length(path_pp), rtol=1e-4, atol=1e-5)
        # The endpoint only agrees to a fraction of an edge length: the two libraries resolve a
        # vertex crossing differently, and the walk accumulates that over every crossing.
        assert np.linalg.norm(points[-1] - path_pp[-1]) < 0.5 * edge_length


def test_trace_from_vertex_stops_at_the_boundary(
    hemisphere: tuple[object, wp.Mesh], device: str
) -> None:
    mesh_tm, mesh_wp = hemisphere
    # Aim from every boundary vertex along the outward direction with a long reach: each ray must
    # stop at the rim rather than wrap around or leave the surface.
    _, _, is_boundary_wp = tw.halfedge.vertex_one_rings(
        mesh_wp.indices,
        n_vertices=len(mesh_tm.vertices),  # type: ignore[attr-defined]
    )
    boundary = np.flatnonzero(is_boundary_wp.numpy()).astype(np.int32)
    centroid = np.asarray(mesh_tm.vertices).mean(axis=0)  # type: ignore[attr-defined]
    outward = np.asarray(mesh_tm.vertices)[boundary] - centroid  # type: ignore[attr-defined]
    outward *= 100.0 / np.linalg.norm(outward, axis=1, keepdims=True)

    points_wp, offsets_wp = tw.tracing.trace_geodesic_from_vertex(
        mesh_wp.points,
        mesh_wp.indices,
        wp.array(boundary, dtype=wp.int32, device=mesh_wp.device),
        wp.array(outward.astype(np.float32), dtype=wp.vec3, device=mesh_wp.device),
    )

    scale = float(
        np.linalg.norm(np.asarray(mesh_tm.vertices).max(0) - np.asarray(mesh_tm.vertices).min(0))
    )  # type: ignore[attr-defined]
    assert offsets_wp.numpy()[-1] < len(boundary) * 64  # nothing ran to the step cap
    distance_tm = np.abs(
        tm.proximity.signed_distance(mesh_tm, points_wp.numpy().astype(np.float64))
    )
    assert distance_tm.max() < 1e-5 * scale


# ---------------------------------------------------------------------------
# trace_geodesic_from_face
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus"])
def test_trace_from_face_matches_potpourri3d(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.ascontiguousarray(mesh_tm.faces, dtype=np.int32)
    rng = np.random.default_rng(3)
    n_rays = 12
    start_faces = rng.integers(0, len(faces_np), n_rays).astype(np.int32)
    barycentric = np.full((n_rays, 3), 1.0 / 3.0)
    scale = 2.0 * float(
        np.linalg.norm(
            mesh_tm.vertices[mesh_tm.edges[:, 1]] - mesh_tm.vertices[mesh_tm.edges[:, 0]], axis=1
        ).mean()
    )
    directions = rng.normal(size=(n_rays, 3))
    directions *= scale / np.linalg.norm(directions, axis=1, keepdims=True)

    points_wp, offsets_wp = tw.tracing.trace_geodesic_from_face(
        mesh_wp.points,
        mesh_wp.indices,
        wp.array(start_faces, dtype=wp.int32, device=mesh_wp.device),
        wp.array(barycentric.astype(np.float32), dtype=wp.vec3, device=mesh_wp.device),
        wp.array(directions.astype(np.float32), dtype=wp.vec3, device=mesh_wp.device),
    )
    curves = tw.tracing.trace_geodesic_polylines(points_wp, offsets_wp)

    tracer_pp = pp3d.GeodesicTracer(vertices_np, faces_np)
    for ray in range(n_rays):
        path_pp = np.asarray(
            tracer_pp.trace_geodesic_from_face(
                int(start_faces[ray]), barycentric[ray], directions[ray]
            )
        )
        points = curves[ray].numpy()
        assert np.allclose(points[0], path_pp[0], rtol=1e-4, atol=1e-4)
        assert np.isclose(_path_length(points), _path_length(path_pp), rtol=1e-4, atol=1e-5)


def test_trace_from_face_zero_direction_is_a_single_point(
    icosahedron: tuple[object, wp.Mesh], device: str
) -> None:
    _, mesh_wp = icosahedron
    points_wp, offsets_wp = tw.tracing.trace_geodesic_from_face(
        mesh_wp.points,
        mesh_wp.indices,
        wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=mesh_wp.device),
        wp.array(
            np.full((1, 3), 1.0 / 3.0, dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device
        ),
        wp.array(np.zeros((1, 3), dtype=np.float32), dtype=wp.vec3, device=mesh_wp.device),
    )
    assert np.array_equal(offsets_wp.numpy(), np.array([0, 1]))
    assert points_wp.shape == (1,)


def test_trace_empty(device: str) -> None:
    vertices_wp = wp.empty(0, dtype=wp.vec3, device=device)
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    empty_int = wp.empty(0, dtype=wp.int32, device=device)
    empty_vec = wp.empty(0, dtype=wp.vec3, device=device)
    points_wp, offsets_wp = tw.tracing.trace_geodesic_from_vertex(
        vertices_wp, faces_wp, empty_int, empty_vec
    )
    assert points_wp.shape == (0,)
    assert offsets_wp.numpy().tolist() == [0]
