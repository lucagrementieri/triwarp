from __future__ import annotations

import warnings

import numpy as np
import pytest
import trimesh as tm
import warp as wp
import warp.sparse as wps

import triwarp as tw
import triwarp.typing as twt
from tests.conversions import trimesh_to_pymeshlab


def _skip_on_cpu(device: str) -> None:
    # Every solve here goes through ``warp.optim.linear.cg``, which returns NaN on the CPU device
    # in Warp 1.14-1.15; ``triwarp.linalg`` raises NotImplementedError there.
    if wp.get_device(device).is_cpu:
        pytest.skip("warp.optim.linear.cg returns NaN on the CPU device in Warp 1.14-1.15.")


def _spd_system(device: str, n: int = 64, n_rhs: int = 3, seed: int = 11):
    """Build an SPD operator with ``n_rhs`` right-hand sides, plus its NumPy form to solve."""
    rng = np.random.default_rng(seed)
    dense_np = rng.standard_normal((n, n))
    dense_np = dense_np @ dense_np.T + n * np.eye(n)
    rows_np, cols_np = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    matrix_wp = wps.bsr_from_triplets(
        n,
        n,
        wp.array(rows_np.ravel().astype(np.int32), dtype=wp.int32, device=device),
        wp.array(cols_np.ravel().astype(np.int32), dtype=wp.int32, device=device),
        wp.array(np.ascontiguousarray(dense_np.ravel()), dtype=wp.float64, device=device),
    )
    rhs_np = rng.standard_normal((n_rhs, n))
    rhs_wp = wp.array(np.ascontiguousarray(rhs_np), dtype=wp.float64, device=device)
    return matrix_wp, twt.as_array2d_float(rhs_wp, dtype=wp.float64), dense_np, rhs_np


@pytest.mark.parity("min_quad_with_fixed", "pymeshlab")
def test_min_quad_with_fixed_matches_pymeshlab_harmonic_field(
    device: str, icosahedron: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Dirichlet-constrained cotangent solve against MeshLab's Generate Scalar Harmonic Field.

    The only external check ``min_quad_with_fixed`` has: everywhere else it is validated indirectly,
    through ``parametrization.tutte`` against ``igl.min_quad_with_fixed``. MeshLab's harmonic field
    pins exactly two vertices and solves the same cotangent system directly, so pinning the same two
    to 0 and 1 makes the two answers the same field -- measured to 1.4e-8.
    """
    _skip_on_cpu(device)
    mesh_tm, mesh_wp = icosahedron
    vertices_np = np.asarray(mesh_tm.vertices, dtype=np.float64)
    n_vertices = vertices_np.shape[0]

    # The two poles, so the field spans the whole mesh rather than a local patch.
    low, high = int(np.argmin(vertices_np[:, 2])), int(np.argmax(vertices_np[:, 2]))
    fixed_np = np.zeros(n_vertices, dtype=bool)
    fixed_np[[low, high]] = True
    values_np = np.zeros((1, n_vertices), dtype=np.float64)
    values_np[0, high] = 1.0

    operator_wp = tw.laplacian.cotmatrix(mesh_wp.points, mesh_wp.indices, dtype=wp.float64)
    solution_wp, free_map_wp, n_free = tw.linalg.min_quad_with_fixed(
        operator_wp,
        wp.array(fixed_np, dtype=wp.bool, device=device),
        twt.as_array2d_float(
            wp.array(np.ascontiguousarray(values_np), dtype=wp.float64, device=device),
            dtype=wp.float64,
        ),
    )
    assert n_free == n_vertices - 2

    # ``min_quad_with_fixed`` returns the free degrees of freedom only; the scatter back through
    # ``free_map`` is the caller's, as its docstring says.
    field_wp = values_np[0].copy()
    free_np = np.flatnonzero(~fixed_np)
    field_wp[free_np] = solution_wp.numpy()[0][free_map_wp.numpy()[free_np]]

    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.compute_scalar_by_scalar_harmonic_field_per_vertex(
        point1=np.ascontiguousarray(vertices_np[low]),
        point2=np.ascontiguousarray(vertices_np[high]),
        value1=0.0,
        value2=1.0,
        colorize=False,
    )
    assert np.allclose(
        field_wp, meshset_pml.current_mesh().vertex_scalar_array(), rtol=1e-5, atol=1e-5
    )


def test_solve_spd_columns_matches_numpy(device: str) -> None:
    _skip_on_cpu(device)
    matrix_wp, rhs_wp, dense_np, rhs_np = _spd_system(device)
    solution_wp = wp.zeros_like(rhs_wp)
    tw.linalg.solve_spd_columns(
        matrix_wp, rhs_wp, twt.as_array2d_float(solution_wp, dtype=wp.float64)
    )
    solution_np = np.linalg.solve(dense_np, rhs_np.T).T
    assert np.allclose(solution_wp.numpy(), solution_np, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("check_every", [1, 25, 0])
def test_solve_spd_columns_check_every_is_solution_invariant(device: str, check_every: int) -> None:
    # ``check_every`` only changes how often the residual is tested (``0`` tests it on device via
    # ``wp.capture_while``), never the system being solved, so every setting converges to the same
    # answer. It is a performance knob only.
    _skip_on_cpu(device)
    matrix_wp, rhs_wp, dense_np, rhs_np = _spd_system(device)
    solution_wp = wp.zeros_like(rhs_wp)
    tw.linalg.solve_spd_columns(
        matrix_wp,
        rhs_wp,
        twt.as_array2d_float(solution_wp, dtype=wp.float64),
        check_every=check_every,
    )
    solution_np = np.linalg.solve(dense_np, rhs_np.T).T
    assert np.allclose(solution_wp.numpy(), solution_np, rtol=1e-5, atol=1e-5)


def test_spd_column_solver_check_every_reused_across_calls(device: str) -> None:
    # The hoisted functor keeps its ``check_every`` across calls and warm-starts from ``solution``.
    _skip_on_cpu(device)
    matrix_wp, rhs_wp, dense_np, rhs_np = _spd_system(device)
    solution_wp = wp.zeros_like(rhs_wp)
    solver = tw.linalg.spd_column_solver(
        matrix_wp, rhs_wp, twt.as_array2d_float(solution_wp, dtype=wp.float64), check_every=0
    )
    solver()
    solver()
    solution_np = np.linalg.solve(dense_np, rhs_np.T).T
    assert np.allclose(solution_wp.numpy(), solution_np, rtol=1e-5, atol=1e-5)


def test_solve_spd_warns_when_it_runs_out_of_iterations(device: str) -> None:
    """
    A conjugate gradient that stops on ``maxiter`` rather than on ``tol`` says so.

    Without this the returned array is the last iterate and looks exactly like a solution --
    which is how a diverging smoothing pass used to reach a caller silently.
    """
    if wp.get_device(device).is_cpu:
        pytest.skip("warp.optim.linear.cg produces NaN on the CPU device in Warp 1.14-1.15")
    # A 1-D Laplacian: SPD, but Jacobi-preconditioned CG needs O(n) iterations on it, so a budget
    # of two cannot converge. (A *diagonal* system would be solved exactly in one, and a singular
    # one makes the Jacobi preconditioner itself infinite, which CG bails out of instead.)
    n = 64
    rows, cols, values = [], [], []
    for i in range(n):
        rows.append(i), cols.append(i), values.append(2.0)
        if i + 1 < n:
            rows += [i, i + 1]
            cols += [i + 1, i]
            values += [-1.0, -1.0]
    matrix = wps.bsr_from_triplets(
        n,
        n,
        wp.array(np.array(rows, dtype=np.int32), dtype=wp.int32, device=device),
        wp.array(np.array(cols, dtype=np.int32), dtype=wp.int32, device=device),
        wp.array(np.array(values, dtype=np.float64), dtype=wp.float64, device=device),
    )
    rhs = wp.array(np.ones(n, dtype=np.float64), dtype=wp.float64, device=device)
    solution = wp.zeros(n, dtype=wp.float64, device=device)

    with pytest.warns(UserWarning, match="iteration cap"):
        iterations, _, _ = tw.linalg.solve_spd(
            matrix, rhs, solution, tol=1e-14, maxiter=2, name="test_solve_spd"
        )
    assert int(iterations) >= 2


def test_solve_spd_is_quiet_when_it_converges(device: str) -> None:
    """The warning is specific to non-convergence: a well-posed solve emits nothing."""
    if wp.get_device(device).is_cpu:
        pytest.skip("warp.optim.linear.cg produces NaN on the CPU device in Warp 1.14-1.15")
    n = 8
    indices = wp.array(np.arange(n, dtype=np.int32), dtype=wp.int32, device=device)
    values = wp.array(np.full(n, 2.0, dtype=np.float64), dtype=wp.float64, device=device)
    matrix = wps.bsr_from_triplets(n, n, indices, wp.clone(indices), values)
    rhs = wp.array(np.ones(n, dtype=np.float64), dtype=wp.float64, device=device)
    solution = wp.zeros(n, dtype=wp.float64, device=device)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        tw.linalg.solve_spd(matrix, rhs, solution, maxiter=10 * n)
    assert np.allclose(solution.numpy(), np.full(n, 0.5))
