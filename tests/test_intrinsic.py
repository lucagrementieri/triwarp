"""
Regression tests for ``triwarp.intrinsic`` against libigl and potpourri3d (CPU references).

Mollification is only interesting on a mesh that needs it, so the fixtures here are joined by a
deliberately degenerate one: a sliver triangle whose edge lengths fail the triangle inequality in
``float32``. The plain cotangent Laplacian returns NaN on it; the robust one must not.

``igl.cotmatrix_intrinsic`` is the reference for the intrinsic assembly itself (same edge lengths
in, same matrix out), and ``potpourri3d.MeshHeatMethodDistanceSolver(use_robust=True)`` for the heat
method's robust path — that flag is potpourri3d's *default*, so it is what its users actually run.
"""

from __future__ import annotations

import igl
import numpy as np
import potpourri3d as pp3d
import pytest
import warp as wp

import triwarp as tw

_MESHES = ["icosahedron", "cave_cube", "hemisphere", "half_torus"]


def _sliver_mesh(device: str) -> tuple[np.ndarray, np.ndarray, wp.array, wp.array]:
    """Build a patch with one near-zero-area triangle, thin enough to break the inequality."""
    vertices_np = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1e-9, 0.0], [0.5, 1.0, 0.0]], dtype=np.float64
    )
    faces_np = np.array([[0, 1, 2], [0, 2, 3], [2, 1, 3]], dtype=np.int32)
    return (
        vertices_np,
        faces_np,
        wp.array(vertices_np.astype(np.float32), dtype=wp.vec3, device=device),
        wp.array(faces_np.reshape(-1), dtype=wp.int32, device=device),
    )


def _dense(matrix: object, n_vertices: int) -> np.ndarray:
    """
    Densify a ``BsrMatrix``, reading only the entries its offsets actually address.

    ``BsrMatrix.values`` is allocated at the *triplet* count and its ``nnz`` is an upper bound until
    synchronized, so the tail of that buffer is uninitialized scratch. Comparing two matrices'
    ``values`` arrays directly reads that scratch and is flaky by construction; the row offsets are
    the only safe way in.
    """
    offsets = matrix.offsets.numpy()  # type: ignore[attr-defined]
    columns = matrix.columns.numpy()  # type: ignore[attr-defined]
    values = matrix.values.numpy()  # type: ignore[attr-defined]
    dense = np.zeros((n_vertices, n_vertices))
    for row in range(n_vertices):
        for slot in range(offsets[row], offsets[row + 1]):
            dense[row, columns[slot]] = values[slot]
    return dense


# ---------------------------------------------------------------------------
# robust_laplacian
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", _MESHES)
def test_robust_laplacian_is_unchanged_on_a_clean_mesh(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = len(mesh_tm.vertices)
    robust = tw.intrinsic.robust_laplacian(
        mesh_wp.points, mesh_wp.indices, use_intrinsic_delaunay=False
    )
    plain = tw.laplacian.cotmatrix(mesh_wp.points, mesh_wp.indices)

    # Mollification adds nothing when every triangle is already non-degenerate, so with the flips
    # turned off the operator must be the ordinary one up to the intrinsic route's rounding.
    assert np.allclose(_dense(robust, n_vertices), _dense(plain, n_vertices), rtol=1e-4, atol=1e-5)


def test_robust_laplacian_is_finite_where_cotmatrix_is_not(device: str) -> None:
    vertices_np, _, vertices_wp, faces_wp = _sliver_mesh(device)
    n_vertices = len(vertices_np)
    plain = tw.laplacian.cotmatrix(vertices_wp, faces_wp)
    robust = tw.intrinsic.robust_laplacian(vertices_wp, faces_wp, use_intrinsic_delaunay=False)

    # The point of the module, in two lines.
    assert not np.isfinite(_dense(plain, n_vertices)).all()
    assert np.isfinite(_dense(robust, n_vertices)).all()


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus"])
def test_robust_laplacian_matches_igl_intrinsic_assembly(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    lengths_wp, delta = tw.intrinsic.mollify_intrinsic(mesh_wp.points, mesh_wp.indices)
    assert delta == 0.0  # these fixtures are clean, so the comparison is against the plain lengths

    # igl takes the same (n_faces, 3) opposite-edge-length table, which pins the column order.
    laplacian_igl = igl.cotmatrix_intrinsic(
        np.ascontiguousarray(lengths_wp.numpy(), dtype=np.float64),
        np.ascontiguousarray(mesh_tm.faces, dtype=np.int64),
    )
    entries_wp = tw.laplacian.cotmatrix_entries_intrinsic(lengths_wp)
    laplacian_wp = tw.laplacian.cotmatrix(mesh_wp.points, mesh_wp.indices, cot_entries=entries_wp)

    dense_igl = np.asarray(laplacian_igl.todense())
    assert np.allclose(_dense(laplacian_wp, len(mesh_tm.vertices)), dense_igl, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere", "half_torus", "torus"])
def test_robust_laplacian_matches_igl_intrinsic_delaunay(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = len(mesh_tm.vertices)

    # With the flips on, this *is* ``igl::intrinsic_delaunay_cotmatrix`` — the strongest oracle in
    # the module, and one that pins the flip's new-edge-length formula and its winding bookkeeping
    # at once. The intrinsic Delaunay triangulation is unique, so the two implementations need not
    # (and do not) perform the same flips in the same order to agree on the matrix.
    laplacian_igl, _, _ = igl.intrinsic_delaunay_cotmatrix(
        np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64),
        np.ascontiguousarray(mesh_tm.faces, dtype=np.int64),
    )
    laplacian_wp = tw.intrinsic.robust_laplacian(mesh_wp.points, mesh_wp.indices)
    assert np.allclose(
        _dense(laplacian_wp, n_vertices), np.asarray(laplacian_igl.todense()), rtol=1e-4, atol=1e-5
    )


# ---------------------------------------------------------------------------
# intrinsic_delaunay
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere", "half_torus", "torus"])
def test_intrinsic_delaunay_removes_negative_cotangent_weights(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = len(mesh_tm.vertices)
    flipped = _dense(tw.intrinsic.robust_laplacian(mesh_wp.points, mesh_wp.indices), n_vertices)

    # A non-negative off-diagonal (in this sign convention, where the diagonal is negative) is what
    # "Delaunay" buys: it is the condition for the Laplacian to satisfy a maximum principle.
    off_diagonal = flipped - np.diag(np.diag(flipped))
    assert off_diagonal.min() > -1e-6


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere"])
def test_intrinsic_delaunay_leaves_a_delaunay_mesh_alone(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    _, mesh_wp = request.getfixturevalue(mesh_name)
    original_lengths = tw.intrinsic.face_edge_lengths(mesh_wp.points, mesh_wp.indices).numpy()
    faces, lengths, n_flips = tw.intrinsic.intrinsic_delaunay(mesh_wp.points, mesh_wp.indices)

    # These fixtures come from an icosphere, whose triangulation is already intrinsically Delaunay.
    assert n_flips == 0
    assert np.array_equal(faces.numpy(), mesh_wp.indices.numpy())
    assert np.allclose(lengths.numpy(), original_lengths, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("mesh_name", ["half_torus", "torus"])
def test_intrinsic_delaunay_flips_a_grid_and_preserves_the_metric(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    faces, lengths, n_flips = tw.intrinsic.intrinsic_delaunay(mesh_wp.points, mesh_wp.indices)

    # A quad grid split by diagonals is not Delaunay, so there is work to do...
    assert n_flips > 0
    # ... but the flips are *intrinsic*: the vertex count, the face count and the total area are all
    # properties of the surface, not of its triangulation, so none of them may change.
    assert faces.shape == mesh_wp.indices.shape
    assert np.array_equal(np.sort(np.unique(faces.numpy())), np.sort(np.unique(mesh_tm.faces)))
    sides = lengths.numpy().astype(np.float64)
    semi = sides.sum(axis=1) / 2.0
    heron = semi * (semi - sides[:, 0]) * (semi - sides[:, 1]) * (semi - sides[:, 2])
    assert np.isclose(np.sqrt(np.maximum(heron, 0.0)).sum(), mesh_tm.area, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# mollify_intrinsic and face_edge_lengths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", _MESHES)
def test_face_edge_lengths_are_the_opposite_edges(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    lengths = tw.intrinsic.face_edge_lengths(mesh_wp.points, mesh_wp.indices).numpy()

    triangles = np.asarray(mesh_tm.vertices)[np.asarray(mesh_tm.faces)]
    expected = np.stack(
        [
            np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
            np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
        ],
        axis=1,
    )
    assert np.allclose(lengths, expected, rtol=1e-5, atol=1e-5)


def test_mollify_intrinsic_is_a_no_op_on_a_clean_mesh(
    icosahedron: tuple[object, wp.Mesh], device: str
) -> None:
    _, mesh_wp = icosahedron
    original = tw.intrinsic.face_edge_lengths(mesh_wp.points, mesh_wp.indices)
    mollified, delta = tw.intrinsic.mollify_intrinsic(mesh_wp.points, mesh_wp.indices)

    assert delta == 0.0
    assert np.array_equal(mollified.numpy(), original.numpy())


def test_mollify_intrinsic_restores_the_triangle_inequality(device: str) -> None:
    _, _, vertices_wp, faces_wp = _sliver_mesh(device)
    original = tw.intrinsic.face_edge_lengths(vertices_wp, faces_wp).numpy()
    mollified, delta = tw.intrinsic.mollify_intrinsic(vertices_wp, faces_wp)

    assert delta > 0.0

    # Every mollified triangle satisfies the inequality strictly; at least one original did not.
    def worst_slack(lengths: np.ndarray) -> np.ndarray:
        a, b, c = lengths[:, 0], lengths[:, 1], lengths[:, 2]
        return np.minimum(np.minimum(a + b - c, b + c - a), c + a - b)

    assert worst_slack(original).min() <= 0.0
    assert worst_slack(mollified.numpy()).min() > 0.0
    # One global constant, added to every length: that is what keeps the operator symmetric. The
    # tolerance is loose because ``delta`` here is ~1e-5 against lengths of ~1, so the sum lands at
    # the edge of float32's resolution.
    assert np.allclose(mollified.numpy() - original, delta, rtol=5e-2, atol=1e-9)


# ---------------------------------------------------------------------------
# heat_geodesic(use_robust=True)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere", "half_torus"])
def test_robust_heat_geodesic_matches_potpourri3d(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    if wp.get_device(device).is_cpu:
        pytest.skip("heat_geodesic needs conjugate gradient, which Warp cannot run on CPU")
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    sources_wp = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=mesh_wp.device)

    distance_wp = tw.heat.distance.heat_geodesic(
        mesh_wp.points, mesh_wp.indices, sources_wp, use_robust=True
    )
    distance_pp = np.asarray(
        pp3d.MeshHeatMethodDistanceSolver(
            np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64),
            np.ascontiguousarray(mesh_tm.faces, dtype=np.int32),
            use_robust=True,
        ).compute_distance(0)
    )

    # potpourri3d's robust path also flips to an intrinsic Delaunay triangulation, which this does
    # not (see the module docstring), so the two agree to the heat method's own accuracy rather than
    # tightly. The comparison is still worth making: it is the configuration potpourri3d ships.
    scale = float(np.linalg.norm(mesh_tm.vertices.max(axis=0) - mesh_tm.vertices.min(axis=0)))
    assert np.abs(distance_wp.numpy() - distance_pp).mean() < 0.1 * scale


def test_robust_heat_geodesic_survives_a_degenerate_triangle(device: str) -> None:
    if wp.get_device(device).is_cpu:
        pytest.skip("heat_geodesic needs conjugate gradient, which Warp cannot run on CPU")
    _, _, vertices_wp, faces_wp = _sliver_mesh(device)
    sources_wp = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=device)

    plain = tw.heat.distance.heat_geodesic(vertices_wp, faces_wp, sources_wp).numpy()
    robust = tw.heat.distance.heat_geodesic(
        vertices_wp, faces_wp, sources_wp, use_robust=True
    ).numpy()

    assert not np.isfinite(plain).all()
    assert np.isfinite(robust).all()
    assert robust[0] == pytest.approx(0.0, abs=1e-6)
