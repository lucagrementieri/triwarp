from __future__ import annotations

import unittest.mock
import warnings

import igl
import numpy as np
import pytest
import scipy.sparse as sp
import trimesh as tm
import warp as wp
import warp.sparse as wps

import triwarp as tw
import triwarp.typing as twt
from tests.comparisons import assert_nonconstant
from tests.conversions import bsr_to_dense, trimesh_to_pymeshlab


def _spd_system(device: str, n: int = 64, n_rhs: int = 3, seed: int = 11):
    """
    Build an SPD operator with ``n_rhs`` right-hand sides, plus its NumPy form to solve.

    The operator is **dense** -- ``n ** 2`` triplets, 263 169 of them at the ``n = 513`` the
    boundary test reaches -- and a banded rewrite was measured and declined. Once ``conftest.py``
    caps OpenBLAS's thread pool the whole eight-parametrization family of
    [`test_solve_spd_columns_across_the_reduction_tile_boundary`] is dominated by the first launch's
    kernel load, so there is nothing left to win and a banded operator would only make the reference
    solve less obviously right. Before that cap the same family was 9 of the CPU suite's 80 slowest
    rows -- but the cost was OpenBLAS at 48 threads (``np.linalg.solve`` is orders of magnitude
    slower there), not the triplet count.
    """
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


def _grid_laplacian_system(device: str, k: int = 24, n_rhs: int = 3, shift: float = 1e-3, seed=5):
    """
    Build a ``k x k`` five-point Laplacian plus a small shift: sparse, SPD, and it coarsens.

    Assembled here rather than assembled from a mesh so the operator's definiteness and sparsity are
    the test's own, not a cotangent-sign convention's. ``k = 24`` puts it above
    ``linalg``'s dense-coarse-solve threshold, so the hierarchy really has a level.
    """
    index = np.arange(k * k).reshape(k, k)
    rows = [index.ravel()]
    columns = [index.ravel()]
    values = [np.full(k * k, 4.0 + shift)]
    for a, b in (
        (index[:-1, :], index[1:, :]),
        (index[1:, :], index[:-1, :]),
        (index[:, :-1], index[:, 1:]),
        (index[:, 1:], index[:, :-1]),
    ):
        rows.append(a.ravel())
        columns.append(b.ravel())
        values.append(np.full(a.size, -1.0))
    rows_np = np.concatenate(rows).astype(np.int32)
    columns_np = np.concatenate(columns).astype(np.int32)
    values_np = np.concatenate(values)
    matrix_wp = wps.bsr_from_triplets(
        k * k,
        k * k,
        wp.array(rows_np, dtype=wp.int32, device=device),
        wp.array(columns_np, dtype=wp.int32, device=device),
        wp.array(np.ascontiguousarray(values_np), dtype=wp.float64, device=device),
    )
    dense_np = np.zeros((k * k, k * k))
    np.add.at(dense_np, (rows_np, columns_np), values_np)
    rng = np.random.default_rng(seed)
    rhs_np = rng.standard_normal((n_rhs, k * k))
    rhs_wp = wp.array(np.ascontiguousarray(rhs_np), dtype=wp.float64, device=device)
    return matrix_wp, twt.as_array2d(rhs_wp, wp.float64), dense_np, rhs_np


@pytest.mark.parity("min_quad_with_fixed", "pymeshlab", "igl")
def test_min_quad_with_fixed_matches_pymeshlab_harmonic_field(
    device: str, icosahedron: tuple[tm.Trimesh, wp.Mesh]
) -> None:
    """
    Class A twice: MeshLab's harmonic field, and the libigl function this one is named after.

    **libigl binds ``min_quad_with_fixed``**, which makes it the direct comparison and not merely a
    library that solves a similar system: it minimizes ``0.5 x' A x + x' B`` under ``x[known] = Y``,
    so at ``B = 0`` with no equality constraints it is this function's problem exactly. The named
    transform is one sign -- ``igl.cotmatrix`` is negative semi-definite, so ``A`` is ``-L`` --
    plus its ``(n_vertices, 1)`` dense return against triwarp's free-block-only one, which the
    ``free_map`` scatter already resolves for the MeshLab half.

    Two conventions worth knowing before using it: it returns a **plain array** here rather than the
    tuple its C++ signature suggests, and ``Aeq`` / ``Beq`` are not optional -- an empty
    ``csr_matrix((0, n))`` and an ``(0, 1)`` array are what "no equality constraints" looks like.
    ``igl.min_quad_with_fixed_precompute`` / ``_solve`` are bound too, which is what gives the
    benchmark group a real amortized axis on the reference side.

    MeshLab's harmonic field pins exactly two vertices and solves the same cotangent system
    directly, so pinning the same two to 0 and 1 makes the two answers the same field -- measured to
    1.4e-8. igl is given the identical two pins, so all three agree on one field.
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

    # igl: the same problem, with -L for the sign convention and no equality constraints.
    faces_igl = np.ascontiguousarray(mesh_tm.faces, dtype=np.int64)
    field_igl = np.asarray(
        igl.min_quad_with_fixed(
            -igl.cotmatrix(vertices_np, faces_igl),
            np.zeros((n_vertices, 1)),
            np.ascontiguousarray(np.array([low, high], dtype=np.int64)),
            np.array([[0.0], [1.0]]),
            sp.csr_matrix((0, n_vertices)),
            np.zeros((0, 1)),
            True,
        )
    ).ravel()
    # non-vacuity: the reference produced a real field, not a constant
    assert_nonconstant(field_igl, tol=0.5)
    assert np.allclose(field_wp, field_igl, rtol=1e-5, atol=1e-5)


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
    ``bsr_mv`` was dimensioned for the unreduced matrix. The count here is exact.

    ``Q_uu`` is compared with ``np.array_equal`` deliberately: the extraction *copies* entries
    rather than recomputing them, so it is bit-exact on both devices (measured ``0.0`` on each). The
    right-hand side accumulates the pinned-column contributions, and there the devices differ --
    exact on CUDA, one ulp out on the CPU backend (8.9e-16 absolute, 1.1e-16 relative), because the
    row scan sums them in a different order. Asserting equality on that half passed here and failed
    the CPU gate.
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
    assert np.allclose(
        rhs_wp.numpy(),
        -(dense_np[np.ix_(free_np, pinned_np)] @ values_np[:, pinned_np].T).T,
        rtol=1e-14,
        atol=1e-14,
    )


@pytest.mark.parity(
    "solve_spd_columns",
    "numpy",
    benchmarked=False,
    reason="numpy.linalg.solve is a dense LU on the host and this is a batched preconditioned CG "
    "on the device, so a row would compare an O(n^3) factorization with an iterative solve and "
    "report the crossover as a speedup. The *answer* is what is comparable, and a dense solve "
    "is the strongest oracle available for it -- exact up to conditioning.",
)
def test_solve_spd_columns_matches_numpy(device: str) -> None:
    """
    Class A against ``numpy.linalg.solve``, on the default single-column path.

    The reference is a dense LU of the same operator, so this is an exact oracle up to conditioning
    rather than a tolerance dictated by two approximations meeting. The ``1e-5`` bound is the CG
    tolerance's, not the reference's.
    """
    matrix_wp, rhs_wp, dense_np, rhs_np = _spd_system(device)
    solution_wp = wp.zeros_like(rhs_wp)
    tw.linalg.solve_spd_columns(matrix_wp, rhs_wp, twt.as_array2d(solution_wp, wp.float64))
    solution_np = np.linalg.solve(dense_np, rhs_np.T).T
    assert np.allclose(solution_wp.numpy(), solution_np, rtol=1e-5, atol=1e-5)


@pytest.mark.parity(
    "solve_spd_columns",
    "numpy",
    benchmarked=False,
    reason="numpy.linalg.solve is a dense LU on the host and this is a batched preconditioned CG "
    "on the device, so a row would compare an O(n^3) factorization with an iterative solve and "
    "report the crossover as a speedup. The *answer* is what is comparable, and a dense solve "
    "is the strongest oracle available for it -- exact up to conditioning.",
)
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


@pytest.mark.parity(
    "solve_spd_columns",
    "numpy",
    benchmarked=False,
    reason="numpy.linalg.solve is a dense LU on the host and this is a batched preconditioned CG "
    "on the device, so a row would compare an O(n^3) factorization with an iterative solve and "
    "report the crossover as a speedup. The *answer* is what is comparable, and a dense solve "
    "is the strongest oracle available for it -- exact up to conditioning.",
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


def test_solve_spd_columns_two_columns_uses_batched_cg(device: str) -> None:
    """
    Triwarp against triwarp: the dispatch itself, not just the answer it produces.

    Every multi-column solve under ``"diag"`` builds a ``linalg._BatchedCg``, the two-column case
    included. A block conjugate gradient sharing one Krylov subspace across exactly two columns
    was gated in here and removed again (see ``linalg._cg_columns``); a wrong gate would still
    converge to the right answer, so it needs its own assert rather than relying on a value check.
    """
    matrix_wp, rhs_wp, _dense, _rhs = _spd_system(device, n_rhs=2)
    solution_wp = wp.zeros_like(rhs_wp)
    solver = tw.linalg.spd_column_solver(matrix_wp, rhs_wp, twt.as_array2d(solution_wp, wp.float64))
    assert isinstance(solver, tw.linalg._BatchedCg)


@pytest.mark.parametrize("n", [255, 256, 257, 511, 512, 513])
def test_solve_spd_columns_two_columns_across_the_tile_boundary(device: str, n: int) -> None:
    """
    Class A, against ``numpy.linalg.solve``.

    Same boundary ``test_solve_spd_columns_across_the_reduction_tile_boundary`` pins, run again at
    ``n_rhs=2``: the reduction's padded ``stride`` is derived from the column count as well as from
    ``n``, so an even column count exercises a different padding than the ``n_rhs=3`` default every
    other tile-boundary case uses.
    """
    matrix_wp, rhs_wp, dense_np, rhs_np = _spd_system(device, n=n, n_rhs=2)
    solution_wp = wp.zeros_like(rhs_wp)
    tw.linalg.solve_spd_columns(matrix_wp, rhs_wp, twt.as_array2d(solution_wp, wp.float64))
    assert np.allclose(
        solution_wp.numpy(), np.linalg.solve(dense_np, rhs_np.T).T, rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("shift", [1e-3, 1e-7])
def test_two_column_solve_costs_no_more_iterations_than_its_worst_column(
    device: str, shift: float
) -> None:
    """
    Not a library comparison: no reference exposes a solver's iteration count.

    The guard is against a two-column mechanism that *couples* the columns. ``_BatchedCg`` shares
    the launches and leaves each column's Krylov subspace its own, so a two-column solve costs
    exactly the worse of the two single-column solves and can never cost more. A block conjugate
    gradient over a shared subspace does not have that property -- it was measured at 4 774
    iterations against 2 230 on an ill-conditioned system while winning on a well-conditioned one,
    which is why it is gone (``linalg._cg_columns``). The small ``shift`` arm is the
    ill-conditioned member of the pair and is the one that would catch a reintroduction: a
    mechanism validated on the well-conditioned arm alone is exactly the failure this pins.
    """
    matrix_wp, rhs_wp, dense_np, rhs_np = _grid_laplacian_system(device, k=40, n_rhs=2, shift=shift)
    kwargs = {"tol": 1e-10, "maxiter": 40_000, "check_every": 1, "preconditioner": "diag"}

    both_solution = wp.zeros_like(rhs_wp)
    both_iterations, _residual, _tol = tw.linalg._BatchedCg(
        matrix_wp, rhs_wp, twt.as_array2d(both_solution, wp.float64), **kwargs
    )()

    worst_single = 0
    for column in range(2):
        single_rhs = wp.array(
            np.ascontiguousarray(rhs_np[column : column + 1]), dtype=wp.float64, device=device
        )
        single_solution = wp.zeros_like(single_rhs)
        iterations, _residual, _tol = tw.linalg._BatchedCg(
            matrix_wp,
            twt.as_array2d(single_rhs, wp.float64),
            twt.as_array2d(single_solution, wp.float64),
            **kwargs,
        )()
        worst_single = max(worst_single, int(iterations))

    assert np.allclose(
        both_solution.numpy(), np.linalg.solve(dense_np, rhs_np.T).T, rtol=1e-5, atol=1e-5
    )
    assert int(both_iterations) == worst_single, (
        f"a two-column solve took {int(both_iterations)} iterations where its worst column alone "
        f"takes {worst_single}: the columns are no longer independent"
    )


def test_two_identical_columns_do_not_diverge(device: str) -> None:
    """
    Not a library comparison: the degenerate two-column block is the input to pin, not a value.

    Two identical right-hand-side columns are the worst case any two-column mechanism can be
    handed. Under ``_BatchedCg`` they are simply the same solve run twice, so this must return the
    single-column answer in both rows and nothing may go non-finite.
    """
    matrix_wp, _rhs_wp, dense_np, rhs_np = _spd_system(device, n_rhs=1)
    rhs_two_np = np.concatenate([rhs_np, rhs_np], axis=0)
    rhs_two_wp = wp.array(np.ascontiguousarray(rhs_two_np), dtype=wp.float64, device=device)
    solution_wp = wp.zeros((2, dense_np.shape[0]), dtype=wp.float64, device=device)
    tw.linalg.solve_spd_columns(
        matrix_wp, twt.as_array2d(rhs_two_wp, wp.float64), twt.as_array2d(solution_wp, wp.float64)
    )
    solution_np = solution_wp.numpy()
    assert np.all(np.isfinite(solution_np))
    expected_np = np.linalg.solve(dense_np, rhs_np.T).T
    assert np.allclose(solution_np[0], expected_np[0], rtol=1e-4, atol=1e-4)
    assert np.allclose(solution_np[1], expected_np[0], rtol=1e-4, atol=1e-4)


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


@pytest.mark.parity(
    "spd_column_solver_amortized",
    "numpy",
    benchmarked=False,
    reason="same dense-LU-against-iterative-CG mismatch as the solve_spd_columns claims above, and "
    "this group additionally measures *reuse* across calls -- numpy has no hoisted-state form to "
    "amortize, so there is nothing on its side for the amortization axis to time. The answer each "
    "reused call converges to is what is comparable.",
)
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


@pytest.mark.parametrize("check_every", [0, 5])
def test_spd_column_solver_reads_a_rewritten_rhs_on_every_call(
    device: str, check_every: int
) -> None:
    """
    Class A, against ``numpy.linalg.solve``, for a right-hand side rewritten between two calls.

    The state records its device-side loop once and replays it on every later call, and it
    captures ``rhs`` at construction; both are only correct if each call re-reads the buffer. A
    second call against the *same* right-hand side cannot show that -- it is already converged,
    and a replay that ignored the new values would still return the old, correct answer -- so the
    second right-hand side here differs, and the call must run iterations to reach it.
    """
    matrix_wp, rhs_wp, dense_np, rhs_np = _spd_system(device)
    solution_wp = wp.zeros_like(rhs_wp)
    solver = tw.linalg.spd_column_solver(
        matrix_wp, rhs_wp, twt.as_array2d(solution_wp, wp.float64), check_every=check_every
    )
    solver()
    second_np = np.roll(rhs_np, 1, axis=1) - 0.5 * rhs_np
    rhs_wp.assign(np.ascontiguousarray(second_np))
    iterations, _, _ = solver()
    # Device arrays under ``check_every=0``, host scalars otherwise.
    assert int(iterations.numpy()[0] if isinstance(iterations, wp.array) else iterations) > 0
    solution_np = np.linalg.solve(dense_np, second_np.T).T
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
        rows.append(i)
        cols.append(i)
        values.append(2.0)
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


@pytest.mark.parity(
    "multigrid_preconditioner",
    "numpy",
    benchmarked=False,
    reason="nothing installed here binds smoothed-aggregation multigrid -- pyamg is not a "
    "dependency -- so there is no reference *preconditioner* to time. What a preconditioner "
    "cannot do is change the converged answer, and a dense numpy.linalg.solve of the same "
    "operator is the oracle for that; timing it would be the O(n^3)-against-iterative "
    "mismatch the sibling claims record.",
)
def test_multigrid_preconditioner_solves_the_same_system(device: str) -> None:
    """
    Class A, against ``numpy.linalg.solve``: the V-cycle changes the path, not the answer.

    A preconditioner cannot change a converged solution, only how many iterations reach it -- so the
    thing to check is that the multigrid path really does converge to the reference rather than
    stalling somewhere plausible. The Jacobi arm is the one carrying an external oracle (every other
    solver test in this file), and this pins the new path to the same reference.
    """
    matrix_wp, rhs_wp, dense_np, rhs_np = _grid_laplacian_system(device)
    reference_np = np.linalg.solve(dense_np, rhs_np.T).T
    for mode in ("diag", "multigrid"):
        solution_wp = wp.zeros_like(rhs_wp)
        tw.linalg.solve_spd_columns(
            matrix_wp, rhs_wp, twt.as_array2d(solution_wp, wp.float64), preconditioner=mode
        )
        assert np.allclose(solution_wp.numpy(), reference_np, rtol=1e-5, atol=1e-5), mode


def test_multigrid_preconditioner_is_symmetric(device: str) -> None:
    """
    Not a library comparison: no reference here builds a multigrid preconditioner.

    The invariant is the one conjugate gradient actually depends on. ``cg`` is only a valid
    iteration when its preconditioner is symmetric positive definite, and a V-cycle is symmetric
    only if its pre- and post-smoothing are balanced -- drop the post-smoothing to save a mat-vec
    and the solver silently stops being conjugate gradient. So ``<M r1, r2> == <r1, M r2>``, which
    this checks, is what licenses the whole approach; the positive part shows up as convergence in
    the test above. This excludes an unbalanced cycle and a restrictor that is not the
    prolongator's transpose; it does not exclude a hierarchy that is merely a bad one.
    """
    matrix_wp, _rhs, dense_np, _rhs_np = _grid_laplacian_system(device)
    n = dense_np.shape[0]
    operator = tw.linalg.multigrid_preconditioner(matrix_wp)
    rng = np.random.default_rng(3)
    left_np, right_np = rng.standard_normal((2, n))
    left = wp.array(np.ascontiguousarray(left_np), dtype=wp.float64, device=device)
    right = wp.array(np.ascontiguousarray(right_np), dtype=wp.float64, device=device)
    applied_left = wp.zeros(n, dtype=wp.float64, device=device)
    applied_right = wp.zeros(n, dtype=wp.float64, device=device)
    operator.matvec(left, applied_left, applied_left, 1.0, 0.0)
    operator.matvec(right, applied_right, applied_right, 1.0, 0.0)
    cross_a = float(applied_left.numpy() @ right_np)
    cross_b = float(left_np @ applied_right.numpy())
    assert np.isclose(cross_a, cross_b, rtol=1e-9, atol=1e-12), (cross_a, cross_b)
    # Non-vacuity: an operator that returned zero, or the identity, would pass the line above.
    assert abs(cross_a) > 1e-6
    assert not np.allclose(applied_left.numpy(), left_np)


def test_multigrid_preconditioner_needs_fewer_iterations(device: str) -> None:
    """
    Not a library comparison: this is the *reason* the mode exists, stated as an assertion.

    triwarp against triwarp -- the Jacobi arm is the reference implementation and carries the
    oracle. Without this the mode could silently degrade to something that still converges (the
    test above would pass) while costing a hierarchy for nothing. The margin is deliberately loose:
    the measured factor on this operator is far above 2x, and the assertion is only meant to catch
    a hierarchy that has stopped working.
    """
    matrix_wp, rhs_wp, _dense, _rhs_np = _grid_laplacian_system(device)
    counts = {}
    for mode in ("diag", "multigrid"):
        solution_wp = wp.zeros_like(rhs_wp)
        counts[mode] = tw.linalg.solve_spd_columns(
            matrix_wp,
            rhs_wp,
            twt.as_array2d(solution_wp, wp.float64),
            check_every=1,
            preconditioner=mode,
        )[0]
    assert counts["multigrid"] * 2 < counts["diag"], counts


def test_multigrid_preconditioner_auto_matches_the_forced_modes(device: str) -> None:
    """
    Class A, against ``numpy.linalg.solve``, for the third mode.

    ``"auto"`` is a *policy* over the other two, so what has to hold is that whichever branch it
    takes still ends in a converged solve at the caller's own cap -- the bug it is written against
    is returning the probe's unconverged iterate. This operator converges inside the probe, so it
    exercises the branch that never builds a hierarchy; the escalating branch is what
    ``smoothing.smooth_region`` runs on every ill-conditioned region and what its own tests cover.
    """
    matrix_wp, rhs_wp, dense_np, rhs_np = _grid_laplacian_system(device)
    solution_wp = wp.zeros_like(rhs_wp)
    tw.linalg.solve_spd_columns(
        matrix_wp, rhs_wp, twt.as_array2d(solution_wp, wp.float64), preconditioner="auto"
    )
    assert np.allclose(
        solution_wp.numpy(), np.linalg.solve(dense_np, rhs_np.T).T, rtol=1e-5, atol=1e-5
    )


def test_multigrid_preconditioner_auto_converges_past_the_probe(device: str) -> None:
    """
    Not a library comparison: this pins ``"auto"``'s escalating branch, which is the risky one.

    A cap of one iteration forces the probe to end unconverged, so the escalation runs -- and the
    assertion is that the *answer* is the converged one rather than the probe's iterate. That was a
    real defect in the first version of this mode: it returned the capped probe when it decided not
    to escalate, so an ill-conditioned system came back silently wrong by 1.3 in absolute terms.
    """
    matrix_wp, rhs_wp, dense_np, rhs_np = _grid_laplacian_system(device)
    solution_wp = wp.zeros_like(rhs_wp)
    with unittest.mock.patch.object(tw.linalg, "CG_PROBE_ITERATIONS", 1):
        tw.linalg.solve_spd_columns(
            matrix_wp, rhs_wp, twt.as_array2d(solution_wp, wp.float64), preconditioner="auto"
        )
    assert np.allclose(
        solution_wp.numpy(), np.linalg.solve(dense_np, rhs_np.T).T, rtol=1e-5, atol=1e-5
    )


def test_offdiagonal_dominance_matches_a_numpy_reduction(device: str) -> None:
    """
    Not a library comparison: no reference exposes a Gershgorin ratio, so the oracle is the formula.

    ``linalg._offdiagonal_dominance`` is what ``preconditioner="auto"`` gates on, so a wrong answer
    here does not fail anything -- it silently picks the other preconditioner and costs time. The
    row with a **zero** diagonal is the case worth pinning: it is a free vertex no face refers to,
    and it must contribute ``0`` rather than an infinity, or every scan mesh reads ``inf`` and the
    gate degenerates to "always multigrid".
    """
    rows_np = np.array([0, 0, 0, 1, 1, 2, 3, 3], dtype=np.int32)
    columns_np = np.array([0, 1, 2, 1, 0, 2, 0, 2], dtype=np.int32)
    values_np = np.array([2.0, -1.0, -3.0, 4.0, 1.0, 0.5, -1.0, -2.0], dtype=np.float64)
    matrix_wp = wps.bsr_from_triplets(
        4,
        4,
        wp.array(rows_np, dtype=wp.int32, device=device),
        wp.array(columns_np, dtype=wp.int32, device=device),
        wp.array(values_np, dtype=wp.float64, device=device),
    )
    dense_np = np.zeros((4, 4))
    np.add.at(dense_np, (rows_np, columns_np), values_np)
    diagonal_np = np.diag(dense_np)
    off_np = np.abs(dense_np).sum(axis=1) - np.abs(diagonal_np)
    # Row 3 has no diagonal entry at all, so it is excluded rather than divided by zero.
    ratios_np = np.where(
        diagonal_np > 0.0, off_np / np.where(diagonal_np > 0.0, diagonal_np, 1.0), 0.0
    )
    assert diagonal_np[3] == 0.0, "the zero-diagonal row is the point of this fixture"
    assert np.allclose(tw.linalg._offdiagonal_dominance(matrix_wp), ratios_np.max(), rtol=1e-12)


def test_multigrid_preconditioner_auto_gate_takes_both_branches(device: str) -> None:
    """
    Triwarp against triwarp: ``"auto"``'s gate against the forced modes it chooses between.

    The oracle is ``numpy.linalg.solve`` on the small system and forced ``"multigrid"`` on the
    large one, which is too big to densify. What this pins is that the gate is *answer-neutral*:
    it only decides which preconditioner runs, and a converged solve is a converged solve either
    way.

    It also runs every branch of the gate, which no other ``"auto"`` test does. All three matter and
    the middle one is the subtle one:

    * **too small** -- the ``k = 24`` grid Laplacian falls under ``CG_MULTIGRID_MIN_UNKNOWNS`` and
      goes to the probe.
    * **large but not dominant enough** -- at ``k = 90`` it is over
      ``CG_MULTIGRID_LARGE_UNKNOWNS`` and still declined, because a five-point Laplacian's rows
      nearly sum to zero (dominance ~1.0, under ``CG_MULTIGRID_SIZE_FLOOR``). That floor exists
      because the size branch was measured **0.41x** on exactly this shape of operator.
    * **dominant** -- squaring that Laplacian squares its condition number and lifts the dominance
      past ``CG_MULTIGRID_DOMINANCE``, which is the ``harmonic`` at ``k = 2`` case the branch is
      for.
    """
    small_wp, rhs_wp, dense_np, rhs_np = _grid_laplacian_system(device, k=24)
    assert not tw.linalg._wants_multigrid(small_wp), "576 unknowns must fall through to the probe"
    solution_wp = wp.zeros_like(rhs_wp)
    tw.linalg.solve_spd_columns(
        small_wp, rhs_wp, twt.as_array2d(solution_wp, wp.float64), preconditioner="auto"
    )
    assert np.allclose(
        solution_wp.numpy(), np.linalg.solve(dense_np, rhs_np.T).T, rtol=1e-5, atol=1e-5
    )

    large_wp, large_rhs_wp, _dense, _rhs = _grid_laplacian_system(device, k=90, n_rhs=1)
    assert tw.linalg._offdiagonal_dominance(large_wp) < tw.linalg.CG_MULTIGRID_SIZE_FLOOR
    assert not tw.linalg._wants_multigrid(large_wp), (
        "a large but weakly-dominant operator must still be declined"
    )

    squared_wp = wps.bsr_mm(large_wp, large_wp)
    assert tw.linalg._offdiagonal_dominance(squared_wp) > tw.linalg.CG_MULTIGRID_DOMINANCE
    assert tw.linalg._wants_multigrid(squared_wp), "the squared operator must clear the gate"
    answers = {}
    for mode in ("auto", "multigrid"):
        answers[mode] = wp.zeros_like(large_rhs_wp)
        tw.linalg.solve_spd_columns(
            squared_wp, large_rhs_wp, twt.as_array2d(answers[mode], wp.float64), preconditioner=mode
        )
    assert np.allclose(answers["auto"].numpy(), answers["multigrid"].numpy(), rtol=1e-5, atol=1e-5)


def test_spd_column_solver_rejects_the_auto_preconditioner(device: str) -> None:
    # A hoisted state is built to be driven many times, and the probe decides on the first solve --
    # so "auto" has no meaning there and says so rather than quietly picking one of the two.
    matrix_wp, rhs_wp, _dense, _rhs_np = _grid_laplacian_system(device, k=8)
    solution_wp = wp.zeros_like(rhs_wp)
    with pytest.raises(ValueError, match="auto"):
        tw.linalg.spd_column_solver(
            matrix_wp, rhs_wp, twt.as_array2d(solution_wp, wp.float64), preconditioner="auto"
        )


def test_multigrid_preconditioner_falls_back_when_the_operator_does_not_coarsen(
    device: str,
) -> None:
    """
    Not a library comparison: a fallback has no counterpart to compare against.

    A diagonal operator has no off-diagonal graph, so every row is its own aggregate and coarsening
    stalls at the first level. Above the dense-factorization cap there is then no usable hierarchy,
    and the documented behaviour is to hand back a Jacobi preconditioner rather than a cycle whose
    coarse solve is a guess -- which for a diagonal operator is the *exact* inverse, so the solve
    converges in one iteration. That is what makes this case checkable at all.
    """
    n = 1024
    rng = np.random.default_rng(7)
    diagonal_np = rng.uniform(1.0, 4.0, size=n)
    index_np = np.arange(n, dtype=np.int32)
    matrix_wp = wps.bsr_from_triplets(
        n,
        n,
        wp.array(index_np, dtype=wp.int32, device=device),
        wp.array(index_np, dtype=wp.int32, device=device),
        wp.array(np.ascontiguousarray(diagonal_np), dtype=wp.float64, device=device),
    )
    rhs_np = rng.standard_normal((2, n))
    rhs_wp = wp.array(np.ascontiguousarray(rhs_np), dtype=wp.float64, device=device)
    solution_wp = wp.zeros_like(rhs_wp)
    iterations = tw.linalg.solve_spd_columns(
        matrix_wp,
        twt.as_array2d(rhs_wp, wp.float64),
        twt.as_array2d(solution_wp, wp.float64),
        check_every=1,
        preconditioner="multigrid",
    )[0]
    assert iterations == 1, iterations
    assert np.allclose(solution_wp.numpy(), rhs_np / diagonal_np, rtol=1e-8, atol=1e-10)


def _normal_equations_system(
    device: str, *, negative: bool, k: int = 28, n_rhs: int = 3, seed: int = 13
) -> tuple:
    """
    Build ``smoothing.smooth_region``'s least-squares umbrella system on a ``k x k`` grid graph.

    Rows are the free vertices (the grid's interior below its top two rows) plus their first fixed
    ring; a row is ``p_v - sum_d w_vd p_d / sum_d w_vd`` with the fixed terms moved to the right, so
    the free rows alone are ``D^-1 L`` for ``L`` the free block of the weighted graph Laplacian.
    ``negative`` gives one diagonal of every grid square a weight of ``-0.2``, the shape a clamped
    cotangent weight takes on a near-right triangle, which pushes ``D^-1 L``'s spectrum past 2.

    Returns the assembled ``M^T M``, the right-hand side, ``L``, the weight sums, and the dense
    forms of the operator and the right-hand side for a reference solve.
    """
    index = np.arange(k * k).reshape(k, k)
    edges, weights = [], []
    for a, b, w in (
        (index[:-1, :], index[1:, :], 1.0),
        (index[:, :-1], index[:, 1:], 1.0),
        (index[:-1, :-1], index[1:, 1:], -0.2 if negative else 0.5),
    ):
        edges.append(np.stack([a.ravel(), b.ravel()], axis=1))
        weights.append(np.full(a.size, w))
    edges_np, weights_np = np.concatenate(edges), np.concatenate(weights)
    n_nodes = k * k
    w_np = sp.coo_matrix(
        (np.concatenate([weights_np] * 2), (edges_np.ravel("F"), edges_np[:, ::-1].ravel("F"))),
        shape=(n_nodes, n_nodes),
    ).tocsr()
    sums_np = np.asarray(w_np.sum(axis=1)).ravel()
    free_np = np.zeros((k, k), dtype=bool)
    free_np[2:, :] = True
    free_np = free_np.ravel()
    ring_np = ~free_np & (np.asarray(w_np[:, free_np].astype(bool).sum(axis=1)).ravel() > 0)
    free_index = np.flatnonzero(free_np)
    rank = np.full(n_nodes, -1)
    rank[free_index] = np.arange(free_index.size)
    rows_np = np.flatnonzero(free_np | ring_np)
    umbrella = sp.identity(n_nodes, format="csr") - sp.diags(1.0 / sums_np) @ w_np
    m_np = umbrella[rows_np][:, free_index].toarray()
    positions_np = np.random.default_rng(seed).standard_normal((n_nodes, n_rhs))
    positions_np[free_np] = 0.0
    b_np = -(umbrella[rows_np] @ positions_np)
    system_np, rhs_np = m_np.T @ m_np, (m_np.T @ b_np).T
    laplacian_np = (sp.diags(sums_np) - w_np)[free_index][:, free_index].tocoo()

    def upload(matrix_np: sp.coo_matrix) -> wps.BsrMatrix:
        return wps.bsr_from_triplets(
            matrix_np.shape[0],
            matrix_np.shape[1],
            wp.array(matrix_np.row.astype(np.int32), dtype=wp.int32, device=device),
            wp.array(matrix_np.col.astype(np.int32), dtype=wp.int32, device=device),
            wp.array(np.ascontiguousarray(matrix_np.data), dtype=wp.float64, device=device),
        )

    rhs_wp = wp.array(np.ascontiguousarray(rhs_np), dtype=wp.float64, device=device)
    sums_wp = wp.array(np.ascontiguousarray(sums_np[free_index]), dtype=wp.float64, device=device)
    return (
        upload(sp.coo_matrix(system_np)),
        twt.as_array2d(rhs_wp, wp.float64),
        upload(laplacian_np),
        sums_wp,
        system_np,
        rhs_np,
    )


@pytest.mark.parametrize("negative", [False, True], ids=["positive_weights", "negative_weights"])
@pytest.mark.parametrize("n_columns", [1, 3])
def test_squared_laplacian_preconditioner_solves_the_same_system(
    device: str, negative: bool, n_columns: int
) -> None:
    """
    Class A, against ``numpy.linalg.solve``: the preconditioner changes the path, not the answer.

    Parametrized over the weight sign because a negative weight is what takes ``D^-1 L``'s spectrum
    past 2, the end of a fixed Chebyshev interval, and over the column count because this
    preconditioner routes a single column through the batched solver too.
    """
    system_wp, rhs_wp, laplacian_wp, sums_wp, system_np, rhs_np = _normal_equations_system(
        device, negative=negative, n_rhs=n_columns
    )
    reference_np = np.linalg.solve(system_np, rhs_np.T).T
    assert np.ptp(reference_np) > 1e-3  # non-vacuity: a zero solve would pass trivially
    solution_wp = wp.zeros_like(rhs_wp)
    tw.linalg.solve_spd_columns(
        system_wp,
        rhs_wp,
        twt.as_array2d(solution_wp, wp.float64),
        preconditioner=tw.linalg.squared_laplacian_preconditioner(laplacian_wp, sums_wp),
    )
    assert np.allclose(solution_wp.numpy(), reference_np, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("negative", [False, True], ids=["positive_weights", "negative_weights"])
def test_squared_laplacian_preconditioner_needs_far_fewer_iterations(
    device: str, negative: bool
) -> None:
    """
    Not a library comparison: the reason the preconditioner exists, stated as an assertion.

    triwarp against triwarp; the Jacobi arm carries the oracle through the test above. Measured
    19x and 27x fewer iterations than Jacobi. The negative-weight arm is the one that matters: with
    the polynomial's interval ending at a fixed 2 rather than at the operator's Gershgorin bound it
    still converges -- the preconditioner stays positive definite -- but in 340 iterations against
    37, a third of Jacobi's count rather than a twenty-seventh, so only a count catches it.
    """
    system_wp, rhs_wp, laplacian_wp, sums_wp, _system_np, _rhs_np = _normal_equations_system(
        device, negative=negative
    )
    counts = {}
    for name, preconditioner in (
        ("diag", "diag"),
        ("squared", tw.linalg.squared_laplacian_preconditioner(laplacian_wp, sums_wp)),
    ):
        solution_wp = wp.zeros_like(rhs_wp)
        counts[name] = tw.linalg.solve_spd_columns(
            system_wp,
            rhs_wp,
            twt.as_array2d(solution_wp, wp.float64),
            check_every=1,
            preconditioner=preconditioner,
        )[0]
    assert counts["squared"] * 8 < counts["diag"], counts


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
