from __future__ import annotations

import igl
import numpy as np
import pytest
import scipy.sparse
import warp as wp

import triwarp as tw


def _skip_on_cpu(device: str) -> None:
    # harmonic / tutte solve the interior system with ``warp.optim.linear.cg``, which returns NaN on
    # the CPU device (NaN) in Warp 1.14.0. The solver raises NotImplementedError there, so skip.
    if wp.get_device(device).is_cpu:
        pytest.skip(
            "harmonic / tutte require CUDA: warp.optim.linear.cg is NaN on CPU in Warp 1.14.0."
        )


def _mesh_numpy(mesh_tm) -> tuple[np.ndarray, np.ndarray]:
    """``(vertices float64 (n, 3), faces int64 (n_faces, 3))`` for the libigl references."""
    return np.asarray(mesh_tm.vertices, dtype=np.float64), np.asarray(mesh_tm.faces, dtype=np.int64)


def _flipped_faces_np(vertices_np: np.ndarray, faces_np: np.ndarray) -> np.ndarray:
    """NumPy reference for libigl ``flipped_triangles``: 2D signed area strictly negative."""
    tri = vertices_np[faces_np]  # (n_faces, 3, 2)
    e0 = tri[:, 1] - tri[:, 0]
    e1 = tri[:, 2] - tri[:, 0]
    signed_area2 = e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0]
    return np.flatnonzero(signed_area2 < 0.0).astype(np.int64)


def _random_2d_mesh(rng: np.random.Generator, n_faces: int):
    """Random 2D triangle soup with mixed orientations as (vertices_2d, flat_faces)."""
    vertices_np = rng.standard_normal((n_faces * 3, 2)).astype(np.float64)
    faces_np = np.arange(n_faces * 3, dtype=np.int64).reshape(n_faces, 3)
    return vertices_np, faces_np


def _to_wp(vertices_np: np.ndarray, faces_np: np.ndarray, device: str):
    vertices_wp = wp.array(
        np.ascontiguousarray(vertices_np, dtype=np.float32), dtype=wp.vec2, device=device
    )
    faces_wp = wp.array(
        np.ascontiguousarray(faces_np.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device
    )
    return vertices_wp, faces_wp


def test_flipped_faces_random_mixed(device):
    rng = np.random.default_rng(0)
    vertices_np, faces_np = _random_2d_mesh(rng, n_faces=64)
    vertices_wp, faces_wp = _to_wp(vertices_np, faces_np, device)

    tri = vertices_np[faces_np]
    e0 = tri[:, 1] - tri[:, 0]
    e1 = tri[:, 2] - tri[:, 0]
    mask_np = (e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0]) < 0.0

    assert np.array_equal(
        tw.parametrization.flipped_faces_mask(vertices_wp, faces_wp).numpy(), mask_np
    )
    assert np.array_equal(
        tw.parametrization.flipped_faces(vertices_wp, faces_wp).numpy(),
        _flipped_faces_np(vertices_np, faces_np),
    )


def test_flipped_faces_mask_index_consistency(device):
    rng = np.random.default_rng(1)
    vertices_np, faces_np = _random_2d_mesh(rng, n_faces=32)
    vertices_wp, faces_wp = _to_wp(vertices_np, faces_np, device)

    mask_wp = tw.parametrization.flipped_faces_mask(vertices_wp, faces_wp)
    assert np.array_equal(
        tw.parametrization.flipped_faces(vertices_wp, faces_wp).numpy(),
        np.flatnonzero(mask_wp.numpy()),
    )


def test_flipped_faces_all_and_none(device):
    # A single CCW (positive-area) triangle: not flipped.
    ccw_np = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    faces_np = np.array([[0, 1, 2]], dtype=np.int64)
    vertices_wp, faces_wp = _to_wp(ccw_np, faces_np, device)
    assert tw.parametrization.flipped_faces(vertices_wp, faces_wp).numpy().size == 0
    assert not tw.parametrization.flipped_faces_mask(vertices_wp, faces_wp).numpy().any()

    # Reverse the winding of every triangle: all flipped.
    cw_faces_np = faces_np[:, ::-1].copy()
    vertices_wp, cw_faces_wp = _to_wp(ccw_np, cw_faces_np, device)
    assert np.array_equal(
        tw.parametrization.flipped_faces(vertices_wp, cw_faces_wp).numpy(),
        np.arange(cw_faces_np.shape[0], dtype=np.int64),
    )
    assert tw.parametrization.flipped_faces_mask(vertices_wp, cw_faces_wp).numpy().all()


def test_flipped_faces_degenerate_not_flagged(device):
    # Collinear (zero-area) triangle: strict "< 0" means it is not flagged.
    collinear_np = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    faces_np = np.array([[0, 1, 2]], dtype=np.int64)
    vertices_wp, faces_wp = _to_wp(collinear_np, faces_np, device)
    assert not tw.parametrization.flipped_faces_mask(vertices_wp, faces_wp).numpy().any()
    assert tw.parametrization.flipped_faces(vertices_wp, faces_wp).numpy().size == 0


def test_flipped_faces_empty_mesh(device):
    vertices_wp = wp.empty(0, dtype=wp.vec2, device=device)
    faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    assert tw.parametrization.flipped_faces_mask(vertices_wp, faces_wp).numpy().size == 0
    assert tw.parametrization.flipped_faces(vertices_wp, faces_wp).numpy().size == 0


@pytest.mark.parametrize("mesh_name", ["hemisphere", "half_torus"])
def test_map_vertices_to_circle_matches_igl(request, device, mesh_name):
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np, _ = _mesh_numpy(mesh_tm)

    boundary_wp = tw.boundary.boundary_loop(mesh_wp.points, mesh_wp.indices)
    boundary_np = boundary_wp.numpy().astype(np.int64)

    circle_wp = tw.parametrization.map_vertices_to_circle(mesh_wp.points, boundary_wp)
    circle_igl = igl.map_vertices_to_circle(vertices_np, boundary_np)

    assert np.allclose(circle_wp.numpy(), circle_igl, rtol=1e-5, atol=1e-5)


def test_uniform_laplacian_matches_igl(device, hemisphere):
    mesh_tm, mesh_wp = hemisphere
    _, faces_np = _mesh_numpy(mesh_tm)
    n_vertices = int(mesh_wp.points.shape[0])

    operator_wp = tw.laplacian.uniform_laplacian(mesh_wp.points, mesh_wp.indices)
    operator_dense = scipy.sparse.csr_matrix(
        (operator_wp.values.numpy(), operator_wp.columns.numpy(), operator_wp.offsets.numpy()),
        shape=(n_vertices, n_vertices),
    ).toarray()

    adjacency_igl = igl.adjacency_matrix(faces_np).toarray().astype(np.float64)
    laplacian_igl = adjacency_igl - np.diag(adjacency_igl.sum(axis=1))

    assert np.allclose(operator_dense, laplacian_igl, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["hemisphere", "half_torus"])
def test_harmonic_matches_igl(request, device, mesh_name):
    _skip_on_cpu(device)
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np, faces_np = _mesh_numpy(mesh_tm)

    boundary_wp = tw.boundary.boundary_loop(mesh_wp.points, mesh_wp.indices)
    boundary_uv_wp = tw.parametrization.map_vertices_to_circle(mesh_wp.points, boundary_wp)
    boundary_np = boundary_wp.numpy().astype(np.int64)
    boundary_uv_np = boundary_uv_wp.numpy().astype(np.float64)

    uv_wp = tw.parametrization.harmonic(
        mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp
    )
    uv_igl = igl.harmonic(vertices_np, faces_np, boundary_np, boundary_uv_np, 1)

    assert np.allclose(uv_wp.numpy(), uv_igl, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("mesh_name", ["hemisphere", "half_torus"])
def test_biharmonic_matches_reference(request, device, mesh_name):
    # k=2 (biharmonic). igl.harmonic's default mass is Voronoi, but triwarp uses the barycentric
    # lumped mass, so compare against a barycentric-mass biharmonic solved directly in SciPy.
    _skip_on_cpu(device)
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np, faces_np = _mesh_numpy(mesh_tm)
    n_vertices = int(mesh_wp.points.shape[0])

    boundary_wp = tw.boundary.boundary_loop(mesh_wp.points, mesh_wp.indices)
    boundary_uv_wp = tw.parametrization.map_vertices_to_circle(mesh_wp.points, boundary_wp)
    boundary_np = boundary_wp.numpy().astype(np.int64)
    boundary_uv_np = boundary_uv_wp.numpy().astype(np.float64)

    uv_wp = tw.parametrization.harmonic(
        mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp, k=2
    )

    laplacian = igl.cotmatrix(vertices_np, faces_np)
    mass_inv = scipy.sparse.diags(
        1.0 / igl.massmatrix(vertices_np, faces_np, igl.MASSMATRIX_TYPE_BARYCENTRIC).diagonal()
    )
    biharmonic = ((-laplacian) @ mass_inv @ (-laplacian)).tocsc()
    interior = np.setdiff1d(np.arange(n_vertices), boundary_np)
    solution = scipy.sparse.linalg.spsolve(
        biharmonic[np.ix_(interior, interior)],
        -(biharmonic[np.ix_(interior, boundary_np)] @ boundary_uv_np),
    )
    uv_ref = np.zeros((n_vertices, 2))
    uv_ref[boundary_np] = boundary_uv_np
    uv_ref[interior] = solution

    assert np.allclose(uv_wp.numpy(), uv_ref, rtol=1e-4, atol=1e-4)


def test_biharmonic_is_deterministic(device, hemisphere):
    # Regression guard for the Warp bsr_mm nondeterminism (issue_report.md): the float64-native
    # biharmonic operator must give the same result across repeated calls. The bug manifested as
    # ~1e22 / NaN corruption, so a tight tolerance (well above conjugate-gradient's ~1e-8 atomic
    # last-ULP jitter) reliably catches a regression.
    _skip_on_cpu(device)
    _, mesh_wp = hemisphere
    boundary_wp = tw.boundary.boundary_loop(mesh_wp.points, mesh_wp.indices)
    boundary_uv_wp = tw.parametrization.map_vertices_to_circle(mesh_wp.points, boundary_wp)
    first = tw.parametrization.harmonic(
        mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp, k=2
    ).numpy()
    for _ in range(5):
        again = tw.parametrization.harmonic(
            mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp, k=2
        ).numpy()
        assert np.allclose(again, first, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["hemisphere", "half_torus"])
def test_tutte_matches_igl_reference(request, device, mesh_name):
    _skip_on_cpu(device)
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    _, faces_np = _mesh_numpy(mesh_tm)
    n_vertices = int(mesh_wp.points.shape[0])

    boundary_wp = tw.boundary.boundary_loop(mesh_wp.points, mesh_wp.indices)
    boundary_uv_wp = tw.parametrization.map_vertices_to_circle(mesh_wp.points, boundary_wp)
    boundary_np = boundary_wp.numpy().astype(np.int64)
    boundary_uv_np = boundary_uv_wp.numpy().astype(np.float64)

    uv_wp = tw.parametrization.tutte(mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp)

    # Reference: uniform Laplacian D - A solved with igl's fixed-value quadratic minimizer, given
    # the identical boundary constraints, so the comparison isolates the interior solve.
    adjacency = igl.adjacency_matrix(faces_np)
    laplacian = (
        scipy.sparse.diags(np.asarray(adjacency.sum(axis=1)).ravel().astype(np.float64)) - adjacency
    ).tocsc()
    if laplacian.shape[0] != n_vertices:
        laplacian.resize((n_vertices, n_vertices))
    uv_igl = np.asarray(
        igl.min_quad_with_fixed(
            laplacian, np.zeros((n_vertices, 2), np.float64), boundary_np, boundary_uv_np
        )
    )

    assert np.allclose(uv_wp.numpy(), uv_igl, rtol=1e-4, atol=1e-4)


def test_tutte_disk_is_fold_free(device, hemisphere):
    # A disk-topology mesh with a convex (circle) boundary yields a bijective, fold-free Tutte map.
    _skip_on_cpu(device)
    _, mesh_wp = hemisphere
    boundary_wp = tw.boundary.boundary_loop(mesh_wp.points, mesh_wp.indices)
    boundary_uv_wp = tw.parametrization.map_vertices_to_circle(mesh_wp.points, boundary_wp)
    uv_wp = tw.parametrization.tutte(mesh_wp.points, mesh_wp.indices, boundary_wp, boundary_uv_wp)
    assert tw.parametrization.flipped_faces(uv_wp, mesh_wp.indices).numpy().size == 0


def test_harmonic_cpu_solve_raises():
    # Single triangle with one interior-free setup forcing a solve: CPU cg is unsupported.
    vertices = wp.array(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
        dtype=wp.vec3,
        device="cpu",
    )
    faces = wp.array(np.array([0, 1, 2, 1, 3, 2], dtype=np.int32), dtype=wp.int32, device="cpu")
    boundary = wp.array(np.array([0, 1, 3], dtype=np.int32), dtype=wp.int32, device="cpu")
    boundary_uv = wp.array(
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        dtype=wp.vec2,
        device="cpu",
    )
    with pytest.raises(NotImplementedError):
        tw.parametrization.harmonic(vertices, faces, boundary, boundary_uv)
