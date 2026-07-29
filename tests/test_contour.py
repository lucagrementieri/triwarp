"""
Regression tests for ``triwarp.contour`` against potpourri3d (CPU reference).

potpourri3d returns each curve as *barycentric* points — ``(element_index, coordinates)`` pairs in
its own internal element numbering — so the oracle has to be decoded through ``potpourri3d.edges``
before any geometric comparison. Its closed curves repeat their first point at the end; open ones do
not, which is what identifies them.
"""

from __future__ import annotations

import numpy as np
import potpourri3d as pp3d
import pytest
import warp as wp

import triwarp as tw

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


def _hausdorff(left: np.ndarray, right: np.ndarray) -> float:
    distance = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
    return float(max(distance.min(axis=1).max(), distance.min(axis=0).max()))


# ---------------------------------------------------------------------------
# marching_triangles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", _MESHES)
@pytest.mark.parametrize("axis", [0, 2])
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

    curves_wp, closed_wp = tw.contour.marching_triangles(
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
        _hausdorff(
            np.concatenate([curve.numpy() for curve in curves_wp]), np.concatenate(curves_pp)
        )
        < 1e-6 * bounding_diagonal
    )


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

    curves_wp, closed_wp = tw.contour.marching_triangles(
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

    curves_wp, closed_wp = tw.contour.marching_triangles(
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

    curves_wp, closed_wp = tw.contour.marching_triangles(
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

    curves_wp, _ = tw.contour.marching_triangles(
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
    assert tw.contour.marching_triangles(mesh_wp.points, mesh_wp.indices, values_wp, 1e6) == (
        [],
        [],
    )


def test_marching_triangles_empty(device: str) -> None:
    vertices_wp = wp.empty(0, dtype=wp.vec3, device=device)
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    values_wp = wp.empty(0, dtype=wp.float32, device=device)
    assert tw.contour.marching_triangles(vertices_wp, faces_wp, values_wp) == ([], [])
