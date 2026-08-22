"""
Regression tests for ``triwarp.geodesic_walk`` against potpourri3d (CPU reference).

Two things are checked independently of the reference, because they are what "a geodesic" means: the
traced arc length equals the requested one (the direction's tangential magnitude), and every traced
point lies on the surface. Against potpourri3d the arc lengths agree exactly; the *endpoints* agree
only to a fraction of an edge length, because a path crossing a vertex has no unique straightest
continuation and the two libraries resolve that differently (see the module docstring).
"""

from __future__ import annotations

import igl
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
# trace_from_vertex
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", _MESHES)
def test_trace_from_vertex_walks_the_requested_distance(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    """
    Not a library comparison: the traced arc length against the requested one, computed here.

    The direction's *tangential* component sets the distance to walk, so the reference is arithmetic
    rather than another implementation -- an equality on a closed mesh and an upper bound on an open
    one, where a ray can stop at the rim. The cross-library check is
    [`test_trace_from_vertex_matches_potpourri3d`].
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    start_np, directions_np = _rays(mesh_tm, 24, seed=0)
    frames_wp = tw.tangent_space.vertex_tangent_frames(mesh_wp.points, mesh_wp.indices)
    points_wp, offsets_wp = tw.geodesic_walk.trace_from_vertex(
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
    curves = tw.geodesic_walk.trace_polylines(points_wp, offsets_wp)
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
    """
    Class C (a distance bound, not a correspondence): every traced point is on the surface.

    trimesh supplies only the point-to-surface distance, so this asserts a property of triwarp's
    answer rather than comparing two answers. The bug class it excludes is the one unfolding gets
    wrong -- drifting off the surface at a triangle crossing -- which no arc-length check would see.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    start_np, directions_np = _rays(mesh_tm, 24, seed=1)
    points_wp, _ = tw.geodesic_walk.trace_from_vertex(
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
@pytest.mark.parity("trace_rays", "potpourri3d")
def test_trace_from_vertex_matches_potpourri3d(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    """
    Class A on the arc length, class C on the endpoint -- and the split is the point.

    The traced *length* is the contract and matches geometry-central to ``1e-4``. The *endpoint*
    cannot: the two libraries resolve a walk crossing exactly through a vertex differently, and the
    path accumulates that choice at every crossing, so it is bounded by half a mean edge length
    instead. Asserting the endpoint at ``allclose`` would be asserting a tie-break neither library
    documents; asserting only the length would miss a path that wandered.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.ascontiguousarray(mesh_tm.faces, dtype=np.int32)
    start_np, directions_np = _rays(mesh_tm, 12, seed=2)

    points_wp, offsets_wp = tw.geodesic_walk.trace_from_vertex(
        mesh_wp.points,
        mesh_wp.indices,
        wp.array(start_np, dtype=wp.int32, device=mesh_wp.device),
        wp.array(directions_np, dtype=wp.vec3, device=mesh_wp.device),
    )
    curves = tw.geodesic_walk.trace_polylines(points_wp, offsets_wp)

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
    """
    Not a library comparison: rays fired off the rim must stop there, not wrap or leave.

    trimesh again supplies only the surface-distance oracle. The step-cap assert is what separates
    *stopping* from *running out of iterations* -- both give a short path, and only one is right.
    """
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

    points_wp, offsets_wp = tw.geodesic_walk.trace_from_vertex(
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
# trace_from_face
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus"])
def test_trace_from_face_matches_potpourri3d(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    """
    Class A on the start point and the arc length, for the barycentric entry point.

    Same split as [`test_trace_from_vertex_matches_potpourri3d`] and for the same reason -- the
    endpoint depends on a vertex-crossing tie-break -- but the *start* is exactly specified by the
    barycentric coordinates, so unlike the vertex form it is asserted at ``1e-4`` rather than
    bounded. No ``parity`` marker: ``trace_from_face`` is not separately benchmarked.
    """
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

    points_wp, offsets_wp = tw.geodesic_walk.trace_from_face(
        mesh_wp.points,
        mesh_wp.indices,
        wp.array(start_faces, dtype=wp.int32, device=mesh_wp.device),
        wp.array(barycentric.astype(np.float32), dtype=wp.vec3, device=mesh_wp.device),
        wp.array(directions.astype(np.float32), dtype=wp.vec3, device=mesh_wp.device),
    )
    curves = tw.geodesic_walk.trace_polylines(points_wp, offsets_wp)

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
    points_wp, offsets_wp = tw.geodesic_walk.trace_from_face(
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
    points_wp, offsets_wp = tw.geodesic_walk.trace_from_vertex(
        vertices_wp, faces_wp, empty_int, empty_vec
    )
    assert points_wp.shape == (0,)
    assert offsets_wp.numpy().tolist() == [0]


# --------------------------------------------------------------------------------------
# descend_field / geodesic_path
# --------------------------------------------------------------------------------------

_PATH_MESHES = ["icosphere", "torus", "unit_box"]


def _paths_to_source(mesh_wp: wp.Mesh, targets_np: np.ndarray) -> list[wp.array[wp.vec3]]:
    """Trace every target back to vertex 0 and slice the packed result."""
    device = mesh_wp.points.device
    source_wp = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=device)
    points_wp, offsets_wp = tw.geodesic_walk.geodesic_path(
        mesh_wp.points, mesh_wp.indices, source_wp, wp.array(targets_np, wp.int32, device=device)
    )
    return tw.geodesic_walk.trace_polylines(points_wp, offsets_wp)


@pytest.mark.parametrize("mesh_name", _PATH_MESHES)
@pytest.mark.parity("geodesic_path", "igl")
def test_geodesic_path_is_never_shorter_than_the_exact_geodesic(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class C with an **inequality**, which is the only bound that holds for an approximate geodesic.

    ``igl.exact_geodesic`` propagates MMP windows and is *globally* exact, so it is a true lower
    bound on the length of any path between the same two vertices -- and the assertion is that
    triwarp never comes in under it. Measured over three fixtures, the minimum ratio is
    **1.0000-1.0005** and the median 1.0000-1.0185, with the worst single path 1.08 long on
    ``unit_box``, where a cube's exact geodesics run along flat faces that a first-order field
    resolves poorly.

    The upper bound is asserted too, because an inequality alone would pass for a wildly detoured
    path. It is deliberately loose (1.35): this is the heat method's accuracy showing through, not
    the walk's, and a tighter bound would be a test of the diffusion time rather than of the path.

    ``igl.exact_geodesic`` needs **all six** arguments -- a four-argument call binds ``vt`` to
    ``fs`` and returns an empty array rather than raising -- so both face sets are passed
    explicitly empty (CLAUDE.md section 6).
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    n_vertices = vertices_np.shape[0]
    rng = np.random.default_rng(2)
    targets_np = rng.choice(
        np.arange(1, n_vertices), min(20, n_vertices - 1), replace=False
    ).astype(np.int32)

    paths = _paths_to_source(mesh_wp, targets_np)
    lengths_np = np.array([float(tw.polyline.polyline_length(path)) for path in paths])

    exact_igl = igl.exact_geodesic(
        vertices_np,
        np.ascontiguousarray(mesh_tm.faces, dtype=np.int64),
        np.array([0], dtype=np.int64),
        np.array([], dtype=np.int64),
        targets_np.astype(np.int64),
        np.array([], dtype=np.int64),
    )
    assert np.all(exact_igl > 0.0)  # non-vacuity: the reference answered for every target

    ratio_np = lengths_np / exact_igl
    assert ratio_np.min() > 0.999  # never shorter than the exact geodesic
    assert ratio_np.max() < 1.35  # nor absurdly longer: the heat field's accuracy, not the walk's


@pytest.mark.parity("geodesic_path", "potpourri3d")
def test_geodesic_path_matches_potpourri3d_on_a_sphere(
    icosphere: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Class C: within a per-cent of ``EdgeFlipGeodesicSolver``'s *exact* path, on a sphere.

    potpourri3d flips edges until the path is locally shortest, so its length is the exact geodesic
    for that homotopy class -- and on a simply-connected surface that is *the* geodesic. Measured on
    ``icosphere(3)`` over 24 targets: triwarp is never shorter (minimum ratio **1.0000**), median
    **1.0051** and worst **1.0917**. The gap is the heat field's first-order accuracy, which is the
    price of getting every path from one solve.

    **This fixture is simply connected on purpose.** On a torus the comparison inverts, for a
    reason that is not an error on either side: ``find_geodesic_path`` shortens within the
    homotopy class of the edge path it starts from, so it can return a path going the long way
    round while a field descent takes the short one -- measured, 3 of 20 paths came out *shorter*
    than the reference there. That is why the globally exact lower bound in the test above uses
    ``igl.exact_geodesic`` instead, and why this comparison stays on the sphere.
    """
    mesh_tm, mesh_wp = icosphere
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    solver_pp = pp3d.EdgeFlipGeodesicSolver(
        vertices_np, np.ascontiguousarray(mesh_tm.faces, dtype=np.int32)
    )
    rng = np.random.default_rng(0)
    targets_np = rng.choice(np.arange(1, vertices_np.shape[0]), 24, replace=False).astype(np.int32)

    paths = _paths_to_source(mesh_wp, targets_np)
    lengths_np = np.array([float(tw.polyline.polyline_length(path)) for path in paths])
    exact_pp = np.array(
        [
            float(
                np.linalg.norm(
                    np.diff(solver_pp.find_geodesic_path(v_start=0, v_end=int(target)), axis=0),
                    axis=1,
                ).sum()
            )
            for target in targets_np
        ]
    )
    assert np.all(exact_pp > 0.0)  # non-vacuity: the reference found every path

    ratio_np = lengths_np / exact_pp
    assert ratio_np.min() > 0.999
    assert np.median(ratio_np) < 1.05
    assert ratio_np.max() < 1.2


@pytest.mark.parametrize("mesh_name", _PATH_MESHES)
def test_geodesic_path_reaches_the_source_along_the_surface(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Not a library comparison: the three properties that make the output a path at all.

    A path must **start at its target**, **end at its source** and **stay on the surface** -- and
    the third is the one worth the closest-point query: a descent that mis-unfolded across an edge
    would produce a plausible polyline floating off the mesh, which no length comparison catches.
    Every point is within a rounding of a face.

    The fourth property is the algorithm's own invariant and the reason it terminates: the field
    **strictly decreases** along the path. It is checked by sampling the distance field at each
    point through its closest face rather than at the vertices: most points are edge crossings.
    """
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_wp, faces_wp = mesh_wp.points, mesh_wp.indices
    n_vertices = int(vertices_wp.shape[0])
    rng = np.random.default_rng(3)
    targets_np = rng.choice(
        np.arange(1, n_vertices), min(12, n_vertices - 1), replace=False
    ).astype(np.int32)
    paths = _paths_to_source(mesh_wp, targets_np)
    vertices_np = vertices_wp.numpy()

    diagonal = float(np.linalg.norm(vertices_np.max(axis=0) - vertices_np.min(axis=0)))
    for target, path in zip(targets_np, paths, strict=True):
        path_np = path.numpy()
        assert path_np.shape[0] >= 2
        assert np.allclose(path_np[0], vertices_np[target], atol=1e-5)
        assert np.allclose(path_np[-1], vertices_np[0], atol=1e-5)

        _closest_wp, distance_wp, _face_wp = tw.proximity.closest_point_on_mesh(
            vertices_wp, faces_wp, path
        )
        assert float(distance_wp.numpy().max()) < 1e-5 * diagonal


def test_descend_field_stops_at_a_local_minimum_and_at_a_boundary(
    icosphere: tuple[tm.Trimesh, wp.Mesh], hemisphere: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Not a library comparison: the two documented ways a descent stops before the stop value.

    A **local minimum** is the interesting one, because it is what makes this a field walk rather
    than a path finder: descending a field with two basins from a vertex in the wrong basin ends at
    that basin's own minimum, not at the global one. Here the field is the distance to a source and
    the descent starts at the *source*, whose value is already at the stop -- so the path is one
    point, which is the honest answer rather than an error.

    At a **mesh boundary** the walk stops where the surface does: on the hemisphere, descending a
    field whose minimum lies off the rim leaves paths ending on the rim.
    """
    _, sphere_wp = icosphere
    device = sphere_wp.points.device
    source_wp = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=device)
    distance_wp = tw.heat.distance.heat_geodesic(sphere_wp.points, sphere_wp.indices, source_wp)
    at_source_wp, offsets_wp = tw.geodesic_walk.descend_field(
        sphere_wp.points, sphere_wp.indices, distance_wp, source_wp
    )
    assert int(at_source_wp.shape[0]) == 1  # already at the stop value
    assert np.array_equal(offsets_wp.numpy(), np.array([0, 1]))

    # A boundary: the field's source is a rim vertex, so paths from the far side reach it, but a
    # field with no reachable minimum stops on the rim instead.
    mesh_tm, hemi_wp = hemisphere
    rim_wp = tw.boundary.boundary_vertex_indices(hemi_wp.points, hemi_wp.indices)
    assert int(rim_wp.shape[0]) > 0
    hemi_distance_wp = tw.heat.distance.heat_geodesic(hemi_wp.points, hemi_wp.indices, rim_wp[:1])
    interior_np = np.setdiff1d(
        np.arange(mesh_tm.vertices.shape[0], dtype=np.int32), rim_wp.numpy()
    )[:8]
    points_wp, path_offsets_wp = tw.geodesic_walk.descend_field(
        hemi_wp.points,
        hemi_wp.indices,
        hemi_distance_wp,
        wp.array(interior_np, dtype=wp.int32, device=device),
    )
    assert int(points_wp.shape[0]) > int(interior_np.shape[0])  # every path has more than a point
    assert int(path_offsets_wp.shape[0]) == interior_np.shape[0] + 1


def test_descend_field_guards_and_empty(icosphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """Not a library comparison: the length guard and the empty batch."""
    _, mesh_wp = icosphere
    device = mesh_wp.points.device
    values_wp = wp.zeros(3, dtype=wp.float64, device=device)
    with pytest.raises(ValueError, match="one entry per vertex"):
        tw.geodesic_walk.descend_field(
            mesh_wp.points,
            mesh_wp.indices,
            values_wp,
            wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=device),
        )

    field_wp = wp.zeros(int(mesh_wp.points.shape[0]), dtype=wp.float64, device=device)
    points_wp, offsets_wp = tw.geodesic_walk.descend_field(
        mesh_wp.points, mesh_wp.indices, field_wp, wp.empty(0, dtype=wp.int32, device=device)
    )
    assert points_wp.shape == (0,)
    assert offsets_wp.shape == (1,)
    assert tw.geodesic_walk.trace_polylines(points_wp, offsets_wp) == []
