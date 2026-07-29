"""
Benchmarks for ``triwarp.smoothing``.

Two groups, two axes, chosen because this module has two genuinely different cost regimes and the
switch between them is a keyword argument rather than a mesh property:

* ``filter_mut_dif_laplacian`` on **scale** -- the explicit branch. One sparse mat-vec per
  iteration plus a diffusion-coefficient recomputation, so cost is ``iterations x nnz``: linear,
  predictable, and the reason this group carries the reference comparison. It is also the filter
  whose loop synced a full-array host sum per iteration before the device-mean fix.
* ``filter_laplacian`` on **quality**, sweeping ``implicit_time_integration`` -- the regime change.
  Explicit is the same cheap SpMV loop; **implicit is a full preconditioned CG solve per
  iteration**, whose count depends on the cotangent system's conditioning. Running that on
  ``saddle`` against ``saddle_graded`` (identical connectivity, worst aspect ratio 1.6 against
  4 719) puts the two regimes and the two conditionings in one table, which is where the cost of
  choosing implicit actually becomes visible.

``iterations`` is deliberately not swept. In the explicit branch it is exactly linear by
construction, so a second point measures multiplication; in the implicit branch the interesting
variation is *within* an iteration, which the quality axis already supplies.

The Laplacian operator is precomputed outside the timed callable wherever the signature accepts
one, so the timing isolates the iteration loop from operator assembly (which
[`test_laplacian.py`](test_laplacian.py) covers).

References
----------
**open3d**'s ``filter_smooth_laplacian`` runs the same number of uniform-weight Laplacian
iterations, so it is the reference for the ``novol`` case. It has no volume-constraint variant
(``filter_smooth_taubin`` alternates two Laplacian passes to limit shrinkage, which is a different
scheme), so the ``vol`` case stays triwarp/trimesh only. Open3D returns a new mesh, so the shared
mesh is reusable across rounds. **trimesh** mutates in place and is rebuilt inside the timed
callable.

Neither has an implicit / backward-Euler smoother at all, so the ``quality`` group is a
triwarp-only before/after comparison.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import trimesh as tm
import warp as wp
import warp.sparse as wps
from conftest import BenchCase, skip_larger_than

import triwarp as tw

_ITERATIONS = 10

_operator_cache: dict[tuple[str, str], wps.BsrMatrix[wp.float32]] = {}


def _laplacian_operator(bench_case: BenchCase) -> wps.BsrMatrix[wp.float32]:
    """Assemble the uniform Laplacian once per (mesh, device) -- an *input*, not the operation."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _operator_cache:
        _operator_cache[key] = tw.laplacian.laplacian(bench_case.vertices_wp, bench_case.faces_wp)
    return _operator_cache[key]


@pytest.mark.benchmark(group="filter_mut_dif_laplacian")
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d")
@pytest.mark.parametrize("volume_constraint", [False, True], ids=["novol", "vol"])
def test_filter_mut_dif_laplacian(bench_case: BenchCase, volume_constraint: bool) -> None:
    """The explicit SpMV loop on the scan sweep: linear in iterations, linear in nnz."""
    skip_larger_than(bench_case, "dragon")
    if bench_case.kind == "open3d":
        if volume_constraint:
            pytest.skip("open3d has no volume-constrained Laplacian smoother")
        mesh_o3d = bench_case.mesh_o3d
        smoothed = bench_case.run(
            lambda: mesh_o3d.filter_smooth_laplacian(number_of_iterations=_ITERATIONS)
        )
        assert len(smoothed.vertices) == bench_case.n_vertices
        return
    if bench_case.kind == "triwarp" and bench_case.device == "cpu" and volume_constraint:
        # Native abort inside the volume-constraint path on the CPU device (Warp 1.15);
        # under investigation alongside the device-mean fix.
        pytest.skip("volume-constraint path aborts on the CPU device")
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        operator = _laplacian_operator(bench_case)
        result = bench_case.run(
            lambda: tw.smoothing.filter_mut_dif_laplacian(
                vertices,
                faces,
                iterations=_ITERATIONS,
                volume_constraint=volume_constraint,
                laplacian_operator=operator,
            )
        )
        assert result.shape == vertices.shape
    else:  # trimesh mutates the mesh in place: rebuild it inside the timed callable
        skip_larger_than(bench_case, "bunny", "trimesh CPU loop takes tens of seconds on dragon")
        vertices, faces = bench_case.vertices_np, bench_case.faces_np

        def run() -> tm.Trimesh:
            mesh = tm.Trimesh(vertices, faces, process=False)
            # trimesh's reference normalizes vertex normals via a scipy-sparse divide that
            # hits 1/0 on degenerate (zero-length) normals; the result is finite and
            # unused, so silence the third-party warning rather than let it leak.
            with np.errstate(divide="ignore", invalid="ignore"):
                tm.smoothing.filter_mut_dif_laplacian(
                    mesh, iterations=_ITERATIONS, volume_constraint=volume_constraint
                )
            return mesh

        result = bench_case.run(run)
        assert result.vertices.shape == vertices.shape


@pytest.mark.benchmark(group="filter_laplacian_integration")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("triwarp")
@pytest.mark.parametrize("implicit", [False, True], ids=["explicit", "implicit"])
def test_filter_laplacian_integration(bench_case: BenchCase, implicit: bool) -> None:
    """
    Explicit SpMV against a per-iteration CG solve, on well- and ill-conditioned connectivity.

    Four rows per table: the explicit pair should be flat across the two meshes (an SpMV does not
    care about aspect ratio) and the implicit pair should not (CG does). A flat implicit pair would
    mean the solve is not actually conditioning-bound, which is worth knowing either way.
    """
    assert bench_case.device is not None
    if implicit and wp.get_device(bench_case.device).is_cpu:
        pytest.skip("the implicit branch solves with warp.optim.linear.cg, CUDA-only in Warp 1.15")
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    operator = _laplacian_operator(bench_case)
    result = bench_case.run(
        lambda: tw.smoothing.filter_laplacian(
            vertices,
            faces,
            iterations=_ITERATIONS,
            implicit_time_integration=implicit,
            volume_constraint=False,
            laplacian_operator=operator,
        )
    )
    assert result.shape == vertices.shape


@pytest.mark.benchmark(group="filter_implicit_fairing")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("triwarp")
def test_filter_implicit_fairing(bench_case: BenchCase) -> None:
    """
    Implicit fairing, whose operator is rebuilt every pass rather than reused.

    The distinguishing cost against ``filter_laplacian_integration``'s implicit row: the cotangent
    weights depend on the positions the step moves, so each pass re-assembles the stiffness and mass
    matrices before solving. Assembly is therefore *inside* the timed loop by construction, not a
    setup cost that could be hoisted.

    Both meshes on this axis are open patches, so this measures the default ``pin_boundary=True``:
    the rim is held and each pass is a Dirichlet problem over the interior, which converges however
    many passes are applied. Hence the assertion of *no* non-convergence warning below -- if this
    group ever starts warning, the flow has stopped being well posed.

    ``pin_boundary=False`` is deliberately **not** a row here. The unconstrained flow pulls the rim
    inward until the triangles there collapse, and past two passes the system is effectively
    singular: the conjugate gradient runs to its ``maxiter`` cap and returns its last iterate, so
    timing it measures a failed solve. Measured once, at ``iterations=10``, for the record:

    * ``saddle`` -- 0.54 s pinned against **67 s** free, a 124x gap;
    * ``saddle_graded`` -- 1.74 s pinned against **66 s** free, 38x.

    Those rows also cost nine minutes of suite time to measure divergence at high precision, which
    is not worth having. Finiteness in the free case is itself recent: the collapse used to drive
    ``cot_entries_from_l2``'s division by ``4 * dbl_area`` to ``inf`` and the result came back NaN.
    """
    assert bench_case.device is not None
    if wp.get_device(bench_case.device).is_cpu:
        pytest.skip("implicit fairing solves with warp.optim.linear.cg, CUDA-only in Warp 1.15")
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # a non-converged pass fails the benchmark
        result = bench_case.run(
            lambda: tw.smoothing.filter_implicit_fairing(vertices, faces, iterations=_ITERATIONS),
            rounds=3,
        )
    assert result.shape == vertices.shape
    assert np.isfinite(result.numpy()).all()
