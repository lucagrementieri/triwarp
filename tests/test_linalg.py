from __future__ import annotations

import warnings

import numpy as np
import pytest
import trimesh as tm
import warp as wp
import warp.sparse as wps

import triwarp as tw
import triwarp.typing as twt
from tests.conversions import bsr_to_dense, trimesh_to_pymeshlab


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
    return matrix_wp, twt.as_array2d(rhs_wp, wp.float64), dense_np, rhs_np


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
        twt.as_array2d(
            wp.array(np.ascontiguousarray(values_np), dtype=wp.float64, device=device), wp.float64
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


# --- free_partition / assemble_interior_system ---------------------------------------------


def _pinned_system(device: str, n: int = 12, n_rhs: int = 2, seed: int = 3):
    """Build an SPD operator with three pinned degrees of freedom, plus its NumPy form."""
    matrix_wp, fixed_values_wp, dense_np, values_np = _spd_system(
        device, n=n, n_rhs=n_rhs, seed=seed
    )
    fixed_np = np.zeros(n, dtype=bool)
    fixed_np[[0, 3, 7]] = True
    fixed_wp = wp.array(fixed_np, dtype=wp.bool, device=device)
    return matrix_wp, fixed_wp, fixed_values_wp, dense_np, values_np, fixed_np


def test_free_partition_ranks_the_unpinned_degrees_of_freedom(device: str) -> None:
    """Class A: the map is the rank of each free degree of freedom among the free ones."""
    _matrix_wp, fixed_wp, _values_wp, _dense_np, _rhs_np, fixed_np = _pinned_system(device)

    free_map_wp, n_free = tw.linalg.free_partition(fixed_wp)

    assert n_free == int((~fixed_np).sum())
    free_np = np.flatnonzero(~fixed_np)
    # Only the free entries carry meaning, which is what the docstring promises; the pinned slots
    # hold whatever the compaction left there.
    assert np.array_equal(free_map_wp.numpy()[free_np], np.arange(n_free, dtype=np.int32))


def test_assemble_interior_system_matches_a_numpy_partition(device: str) -> None:
    """
    Class A: ``Q_uu`` and ``-Q_ub bc`` equal the dense row/column partition, and ``nnz`` is exact.

    The CSR-to-CSR extraction is the one that replaced a ``bsr_from_triplets`` round trip, and the
    ``nnz`` assert is the quieter half of that change: ``bsr_from_triplets`` would have left the
    count at the *triplet* count (here ``q.nnz``, 144 against the true 81), so every downstream
    ``bsr_mv`` was dimensioned for the unreduced matrix. Measured on this system, both blocks agree
    with NumPy **exactly** -- the extraction copies entries rather than recomputing them.
    """
    matrix_wp, fixed_wp, fixed_values_wp, dense_np, values_np, fixed_np = _pinned_system(device)
    free_map_wp, n_free = tw.linalg.free_partition(fixed_wp)

    q_uu, rhs_wp = tw.linalg.assemble_interior_system(
        matrix_wp, fixed_wp, free_map_wp, fixed_values_wp, n_free
    )

    free_np = np.flatnonzero(~fixed_np)
    pinned_np = np.flatnonzero(fixed_np)
    assert q_uu.nnz_sync() == n_free * n_free
    assert np.array_equal(bsr_to_dense(q_uu, n_free), dense_np[np.ix_(free_np, free_np)])
    assert np.array_equal(
        rhs_wp.numpy(), -(dense_np[np.ix_(free_np, pinned_np)] @ values_np[:, pinned_np].T).T
    )


def test_solve_spd_columns_matches_numpy(device: str) -> None:
    matrix_wp, rhs_wp, dense_np, rhs_np = _spd_system(device)
    solution_wp = wp.zeros_like(rhs_wp)
    tw.linalg.solve_spd_columns(matrix_wp, rhs_wp, twt.as_array2d(solution_wp, wp.float64))
    solution_np = np.linalg.solve(dense_np, rhs_np.T).T
    assert np.allclose(solution_wp.numpy(), solution_np, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("n", [1, 2, 255, 256, 257, 511, 512, 513])
def test_solve_spd_columns_across_the_reduction_tile_boundary(device: str, n: int) -> None:
    """
    Class A, against ``numpy.linalg.solve``.

    The batched solver pads each column to a whole number of
    ``kernels.algorithms.conjugate_gradient.CG_TILE`` entries so its dot has no ragged block, and
    those pad lanes ride through the dot, the fused Jacobi apply and the ``x`` update -- the last
    of which writes the caller's *unpadded* buffer and so must skip them. How much padding there
    is, and therefore which of those three a mistake shows up in, is decided entirely by
    ``n mod CG_TILE``: a value that passes at 256 proves nothing about 257. These straddle the
    boundary in both directions.
    """
    matrix_wp, rhs_wp, dense_np, rhs_np = _spd_system(device, n=n)
    solution_wp = wp.zeros_like(rhs_wp)
    tw.linalg.solve_spd_columns(matrix_wp, rhs_wp, twt.as_array2d(solution_wp, wp.float64))
    assert np.allclose(
        solution_wp.numpy(), np.linalg.solve(dense_np, rhs_np.T).T, rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("n_rhs", [1, 2, 5])
def test_solve_spd_columns_agrees_across_column_counts(device: str, n_rhs: int) -> None:
    """
    Class A. One column takes ``warp.optim.linear.cg``; more than one takes triwarp's own solver.

    The split is an implementation detail -- a single column has nothing to batch, so Warp already
    reduces it with a tiled tree -- and this is what keeps the two paths answering the same
    question. Without it the batched solver could drift from the reference and only the
    multi-column callers would notice.
    """
    matrix_wp, rhs_wp, dense_np, rhs_np = _spd_system(device, n_rhs=n_rhs)
    solution_wp = wp.zeros_like(rhs_wp)
    tw.linalg.solve_spd_columns(matrix_wp, rhs_wp, twt.as_array2d(solution_wp, wp.float64))
    assert np.allclose(
        solution_wp.numpy(), np.linalg.solve(dense_np, rhs_np.T).T, rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("check_every", [1, 25, 0])
def test_solve_spd_columns_check_every_is_solution_invariant(device: str, check_every: int) -> None:
    # ``check_every`` only changes how often the residual is tested (``0`` tests it on device via
    # ``wp.capture_while``), never the system being solved, so every setting converges to the same
    # answer. It is a performance knob only.
    matrix_wp, rhs_wp, dense_np, rhs_np = _spd_system(device)
    solution_wp = wp.zeros_like(rhs_wp)
    tw.linalg.solve_spd_columns(
        matrix_wp, rhs_wp, twt.as_array2d(solution_wp, wp.float64), check_every=check_every
    )
    solution_np = np.linalg.solve(dense_np, rhs_np.T).T
    assert np.allclose(solution_wp.numpy(), solution_np, rtol=1e-5, atol=1e-5)


def test_spd_column_solver_check_every_reused_across_calls(device: str) -> None:
    # The hoisted functor keeps its ``check_every`` across calls and warm-starts from ``solution``.
    matrix_wp, rhs_wp, dense_np, rhs_np = _spd_system(device)
    solution_wp = wp.zeros_like(rhs_wp)
    solver = tw.linalg.spd_column_solver(
        matrix_wp, rhs_wp, twt.as_array2d(solution_wp, wp.float64), check_every=0
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


# --- replicated_operator ------------------------------------------------------------------


@pytest.mark.parametrize("n_columns", [1, 3])
def test_replicated_operator_applies_the_base_to_each_block(device: str, n_columns: int) -> None:
    """
    Class A: the batched ``matvec`` equals applying the single operator to each contiguous block.

    Nothing is replicated in memory, so what has to be pinned is that the flat layout really is
    ``n_columns`` independent subproblems -- and that ``n_columns == 1`` hands back the unwrapped
    operator, which the docstring calls out as the ``lscm`` path. Measured agreement 1.4e-14.
    """
    n = 12
    matrix_wp, _values_wp, dense_np, _rhs_np = _spd_system(device, n=n, n_rhs=1, seed=5)

    operator = tw.linalg.replicated_operator(matrix_wp, n_columns)

    assert operator.shape == (n_columns * n, n_columns * n)
    # A single column is not batched at all: no ``batch_offsets``, so no per-block residual.
    assert (operator.batch_offsets is None) == (n_columns == 1)

    vector_np = np.random.default_rng(23).standard_normal(n_columns * n)
    vector_wp = wp.array(np.ascontiguousarray(vector_np), dtype=wp.float64, device=device)
    result_wp = wp.zeros(n_columns * n, dtype=wp.float64, device=device)
    operator.matvec(vector_wp, result_wp, result_wp, alpha=1.0, beta=0.0)

    expected_np = np.concatenate(
        [dense_np @ vector_np[column * n : (column + 1) * n] for column in range(n_columns)]
    )
    assert np.allclose(result_wp.numpy(), expected_np, rtol=1e-10, atol=1e-10)
