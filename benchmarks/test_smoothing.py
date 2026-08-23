"""
Benchmarks for ``triwarp.smoothing``.

Four groups over two axes, chosen because this module has two genuinely different cost regimes and
the switch between them is a keyword argument rather than a mesh property:

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

* ``filter_taubin`` and ``filter_humphrey`` on **scale** -- the two shrinkage-controlled variants of
  the same SpMV loop. Taubin alternates a shrinking and an inflating pass, Humphrey adds a
  push-back toward the original positions; both cost a small constant multiple of
  ``filter_laplacian``, and both existed unbenchmarked until pymeshlab gave them a reference.

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

Neither has an implicit / backward-Euler smoother at all, so the implicit half of the ``quality``
group is a triwarp-only before/after comparison.

**pymeshlab** carries four of MeshLab's ``apply_coord_*`` smoothers and is what gives this module a
second independent implementation of each explicit scheme:

* ``filter_mut_dif_laplacian`` -> ``apply_coord_laplacian_smoothing_scale_dependent``: the *same*
  scheme, Desbrun et al.'s scale-dependent umbrella, which is the mutual-diffusion filter.
* ``filter_laplacian_integration`` -> ``apply_coord_laplacian_smoothing(cotangentweight=False)``:
  the same uniform-weight explicit loop. MeshLab has no implicit variant, so it appears in the
  ``explicit`` row only.
* ``filter_taubin`` -> ``apply_coord_taubin_smoothing``: the same lambda-mu alternation, at
  MeshLab's ``mu=-0.53`` against triwarp's ``nu=0.5``.
* ``filter_humphrey`` -> ``apply_coord_hc_laplacian_smoothing``: Vollmer et al.'s HC, but **not
  parameter-comparable** -- see below.

Two caveats govern every row:

- **Every one of them mutates the coordinates**, so the MeshSet is rebuilt inside the timed callable
  and the row carries the build. That is a large fraction at these sizes: on ``bunny`` the build is
  16.8 of the 48.2 ms a ten-step Laplacian costs, so **35% of that row is not smoothing**. Subtract
  the build (0.47 us/vertex) before quoting a ratio. - **HC Laplacian exposes no parameters at all**
  -- no step count, no ``alpha``/``beta`` -- so its row is a *single* filter call against triwarp's
  ten iterations, and its output does not match ``filter_humphrey`` at any of the 8 x 11 x 11
  ``(iterations, alpha, beta)`` combinations probed (best max-coordinate deviation 0.019 on a mesh
  carrying 0.016 of noise). MeshLab's HC is a different formulation of Vollmer's scheme, not
  triwarp's with other constants. It is a per-pass cost reference and nothing more: it is
  deliberately **not** used as a test oracle in ``tests/test_smoothing.py``, where trimesh remains
  the only HC check.

``filter_two_step`` is the module's other regime change, and the one on the **quality** axis for a
different reason from ``filter_laplacian``: it is *three* nested loops (outer passes x normal
diffusion x vertex fitting, 3 x 20 x 20 at MeshLab's defaults), so its cost is a fixed 1 200 passes
over the adjacency whatever the mesh, and the axis is there to confirm that triangle shape does not
change it. ``filter_normals`` times the inner half alone, which is what separates the normal
diffusion from the fitting solve. Both have pymeshlab references at the same four parameters;
``apply_coord_two_steps_smoothing`` rewrites the coordinates, so that row carries the MeshSet build
like the other ``apply_coord_*`` rows.

The last group leaves positions alone and runs over a per-vertex **scalar** field:
``filter_scalar_laplacian`` against ``apply_scalar_smoothing_per_vertex``, on the **scale** axis and
a fixed number of SpMV passes. Its old neighbour here, the Lipschitz projection of the same kind of
field, moved to [`test_graph.py`](test_graph.py) as ``shortest_path_envelope`` with the function:
it is a weighted graph relaxation whose pass count is data-dependent, a different cost shape,
and it now sits next to ``bfs``. The filter needs the scalar attribute to exist on the MeshSet,
so this row rebuilds it (the filter mutates the attribute in place).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pymeshlab as ml
import pytest
import trimesh as tm
import warp as wp
import warp.sparse as wps
from meshlib import mrmeshnumpy as mn
from meshlib import mrmeshpy as mm

import triwarp as tw
from conftest import BenchCase, face_bitset_ml, mesh_ml_from_numpy, skip_larger_than

_ITERATIONS = 10

_operator_cache: dict[tuple[str, str], wps.BsrMatrix[wp.float32]] = {}
_scalar_cache: dict[tuple[str, str], wp.array[wp.float32]] = {}


def _laplacian_operator(bench_case: BenchCase) -> wps.BsrMatrix[wp.float32]:
    """Assemble the uniform Laplacian once per (mesh, device) -- an *input*, not the operation."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _operator_cache:
        _operator_cache[key] = tw.laplacian.laplacian(bench_case.vertices_wp, bench_case.faces_wp)
    return _operator_cache[key]


def _skip_pml_beyond_bunny(bench_case: BenchCase) -> None:
    """Cap the pymeshlab smoothers at ``bunny``, the way the trimesh rows are capped."""
    skip_larger_than(bench_case, "bunny", "MeshLab's serial smoothers take seconds beyond bunny")


@pytest.mark.noparity(
    "open3d",
    oracle="trimesh",
    reason="D2 a different diffusion scheme with no mutable coefficient: filter_smooth_laplacian "
    "diffuses at a fixed rate toward the inverse-distance-weighted 1-ring mean, where "
    "filter_mut_dif_laplacian scales each vertex's rate by how much of its Laplacian residual lies "
    "along the normal. Measured 0.0428 max-coordinate deviation on a noisy icosphere(2) of extent "
    "2.02 at 10 iterations. trimesh is the oracle for this group, in "
    "tests/test_smoothing.py::test_filter_mut_dif_laplacian_volume_constraint. Recorded so it is "
    "not re-derived: open3d's filter IS triwarp's filter_laplacian under an inverse-distance "
    "operator, matching it to 6.6e-08 at *one* iteration and diverging to 0.032 by ten only "
    "because open3d re-derives the edge weights from the current positions every pass while "
    "triwarp holds the assembled operator fixed.",
)
@pytest.mark.noparity(
    "pymeshlab",
    oracle="trimesh",
    reason="D2 Desbrun et al.'s scale-dependent umbrella, not Barroqueiro et al.'s mutable "
    "diffusion: apply_coord_laplacian_smoothing_scale_dependent weights the 1-ring by edge length "
    "and exposes no per-vertex rate at all, and its step count is its only parameter. Measured "
    "0.171 max-coordinate deviation on a noisy icosphere(2) of extent 2.02 at 10 iterations -- 4x "
    "further from triwarp than open3d's row is. trimesh is the oracle for this group, in "
    "tests/test_smoothing.py::test_filter_mut_dif_laplacian_volume_constraint.",
)
@pytest.mark.benchmark(group="filter_mut_dif_laplacian")
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d", "pymeshlab")
@pytest.mark.parametrize("volume_constraint", [False, True], ids=["novol", "vol"])
def test_filter_mut_dif_laplacian(bench_case: BenchCase, volume_constraint: bool) -> None:
    """The explicit SpMV loop on the scan sweep: linear in iterations, linear in nnz."""
    skip_larger_than(bench_case, "dragon")
    if bench_case.kind == "pymeshlab":
        if volume_constraint:
            pytest.skip("MeshLab's scale-dependent Laplacian has no volume-constraint variant")
        _skip_pml_beyond_bunny(bench_case)
        # Desbrun et al.'s scale-dependent umbrella: the same scheme, mutating coordinates, so the
        # MeshSet is rebuilt per round and the row carries the build.
        bench_case.run(
            lambda: bench_case.new_meshset_pml().apply_coord_laplacian_smoothing_scale_dependent(
                stepsmoothnum=_ITERATIONS
            )
        )
        return
    if bench_case.kind == "open3d":
        if volume_constraint:
            pytest.skip("open3d has no volume-constrained Laplacian smoother")
        mesh_o3d = bench_case.mesh_o3d
        smoothed = bench_case.run(
            lambda: mesh_o3d.filter_smooth_laplacian(number_of_iterations=_ITERATIONS)
        )
        assert len(smoothed.vertices) == bench_case.n_vertices
        return
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


@pytest.mark.noparity(
    "pyvista",
    oracle="pymeshlab",
    reason="D2 a different algorithm with a measured disagreement: vtkSmoothPolyDataFilter moves "
    "each vertex along its incident edge directions under its own convergence test and feature "
    "handling, not by applying a fixed assembled operator, so it diverges with the iteration count "
    "rather than differing by a tolerance -- measured max coordinate deviation 0.103 / 0.363 / "
    "0.581 at 1 / 5 / 10 iterations on an icosphere(2) at relaxation_factor=1.0 with boundary and "
    "feature smoothing off, which is the closest parameterization to triwarp's lamb=1.0. Even the "
    "single iteration is 1e4 past tolerance, so no mapping of relaxation_factor recovers it. "
    "MeshLab's apply_coord_laplacian_smoothing is this group's oracle -- it applies the same fixed "
    "uniform-weight umbrella -- in tests/test_smoothing.py::"
    "test_filter_laplacian_matches_pymeshlab; trimesh covers the same operator under the "
    "filter_laplacian group.",
)
@pytest.mark.noparity(
    "meshlib",
    oracle="pymeshlab",
    reason="D2 a different algorithm with a measured disagreement: MeshLib's relax is not an "
    "umbrella-Laplacian step at all. Measured on an icosphere(2) at one iteration, its "
    "displacement is 0.33 +- 0.19 of triwarp's per vertex (range 0.008 to 0.54) and the two "
    "displacement *directions* disagree -- mean cosine 0.45, minimum -1.0 -- so it is not a "
    "rescaling of the same step and no value of its force parameter recovers one: the closest pair "
    "over a 5x5 sweep of force against lamb still leaves a max coordinate difference of 4.5e-03. "
    "The row is worth having as a cost comparison against a serial relaxation of the same shape; "
    "MeshLab's apply_coord_laplacian_smoothing stays this group's oracle, in "
    "tests/test_smoothing.py::test_filter_laplacian_matches_pymeshlab. Note relax IS the operator "
    "behind smooth_region -- positionVertsSmoothly, a different function -- which is a class-A "
    "pair in its own group.",
)
@pytest.mark.benchmark(group="filter_laplacian_integration")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("triwarp", "pymeshlab", "pyvista", "meshlib")
@pytest.mark.parametrize("implicit", [False, True], ids=["explicit", "implicit"])
def test_filter_laplacian_integration(bench_case: BenchCase, implicit: bool) -> None:
    """
    Explicit SpMV against a per-iteration CG solve, on well- and ill-conditioned connectivity.

    Four rows per table: the explicit pair should be flat across the two meshes (an SpMV does not
    care about aspect ratio) and the implicit pair should not (CG does). A flat implicit pair would
    mean the solve is not actually conditioning-bound, which is worth knowing either way.

    pymeshlab confirms the explicit half independently -- 19.2 against 19.8 ms across the mesh pair,
    flat to within noise, against triwarp's 0.57 / 0.50 ms (34x and 40x) -- which is the same
    statement its harmonic-field row makes in [`test_linalg.py`](test_linalg.py) about where the
    conditioning cost actually lives.
    """
    if bench_case.kind == "meshlib":
        if implicit:
            pytest.skip("relax is an explicit per-iteration pass: no backward-Euler variant")
        # ``relax`` mutates the mesh and returns a status, so the mesh is rebuilt per round.
        # ``force`` is left at its own default of 0.5: it is not triwarp's ``lamb`` under another
        # name (see the exemption above), so there is no value that would make the rows compare
        # outputs -- the row is a cost comparison at each library's own natural setting.
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        iterations = _ITERATIONS

        def relax_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(vertices_np, faces_np)
            params_ml = mm.MeshRelaxParams()
            params_ml.iterations = iterations
            mm.relax(mesh_ml, params_ml)
            return mesh_ml.topology.numValidVerts()

        assert bench_case.run(relax_ml) > 0
        return
    if bench_case.kind == "pyvista":
        if implicit:
            pytest.skip("vtkSmoothPolyDataFilter is explicit only: no backward-Euler variant")
        mesh_pv = bench_case.mesh_pv
        smoothed_pv = bench_case.run(
            lambda: mesh_pv.smooth(
                n_iter=_ITERATIONS,
                relaxation_factor=1.0,
                boundary_smoothing=False,
                feature_smoothing=False,
            )
        )
        assert smoothed_pv.n_points == bench_case.n_vertices
        return
    if bench_case.kind == "pymeshlab":
        if implicit:
            pytest.skip("MeshLab has no implicit / backward-Euler Laplacian smoother")
        # ``cotangentweight=False`` to match triwarp's uniform-weight operator.
        bench_case.run(
            lambda: bench_case.new_meshset_pml().apply_coord_laplacian_smoothing(
                stepsmoothnum=_ITERATIONS, cotangentweight=False
            )
        )
        return
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


@pytest.mark.noparity(
    "open3d",
    oracle="trimesh",
    reason="D2 the filter_mut_dif_laplacian finding again, on Taubin's scheme: "
    "filter_smooth_taubin alternates lambda/mu passes toward the inverse-distance-weighted 1-ring "
    "mean and re-derives those weights from the current positions every pass, while triwarp holds "
    "one assembled operator fixed. Probed before the row landed: the closest mapping (one o3d "
    "iteration against triwarp's lamb=0.5, nu=0.53, iterations=2 under the inverse-distance "
    "operator) still deviates 3.5e-3 max-coordinate on an icosphere(3) carrying 0.01 noise, and "
    "8.8e-3 under the uniform operator -- 350x past tolerance. Like MeshLab, its iteration count "
    "is in lambda-mu PAIRS. trimesh is the oracle for this group, in "
    "tests/test_smoothing.py::test_filter_taubin.",
)
@pytest.mark.noparity(
    "pyvista",
    oracle="trimesh",
    reason="D2 a different algorithm with a measured disagreement: smooth_taubin is VTK's "
    "windowed-sinc filter (vtkWindowedSincPolyDataFilter), parameterized by a pass_band that it "
    "maps to its own kernel weights rather than by lambda / nu, and it warns 'An optimal offset "
    "for the smoothing filter could not be found' on ordinary input. Measured against triwarp at "
    "pass_band=0.1: max coordinate deviation 0.835 at one iteration -- the whole displacement -- "
    "then 6.5e-03 and 2.6e-02 at 2 and 5, so it is neither close nor consistently off by a "
    "factor. Its iteration count is in lambda-mu PAIRS like MeshLab's. trimesh is the oracle for "
    "this group, in tests/test_smoothing.py::test_filter_taubin.",
)
@pytest.mark.benchmark(group="filter_taubin")
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d", "pymeshlab", "pyvista")
def test_filter_taubin(bench_case: BenchCase) -> None:
    """
    The lambda-nu alternation: two SpMVs per iteration instead of one.

    Against ``filter_laplacian_integration``'s explicit row this measures exactly the second pass,
    so the two rows should sit at a ratio near 2 and nothing else should separate them. All four
    libraries implement Taubin's 1995 scheme; MeshLab's and open3d's inflating step is ``mu=-0.53``
    against triwarp's and trimesh's ``nu=0.53``, which changes the fixed point but not the work per
    pass.

    **MeshLab's ``stepsmoothnum`` and open3d's ``number_of_iterations`` count lambda-mu pairs, not
    half-steps**, where triwarp and trimesh do one half-step per ``iterations`` and alternate. So
    both get ``_ITERATIONS // 2``: passing ``_ITERATIONS`` to both, as this row originally did for
    MeshLab, timed twice the passes. The MeshLab mapping is pinned exactly (5e-08) in
    ``tests/test_smoothing.py::test_filter_taubin_matches_pymeshlab``; open3d's cannot be (see the
    exemption above).
    """
    skip_larger_than(bench_case, "dragon")
    if bench_case.kind == "pyvista":
        mesh_pv = bench_case.mesh_pv
        smoothed_pv = bench_case.run(
            lambda: mesh_pv.smooth_taubin(n_iter=_ITERATIONS // 2, pass_band=0.1)
        )
        assert smoothed_pv.n_points == bench_case.n_vertices
        return
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        operator = _laplacian_operator(bench_case)
        result = bench_case.run(
            lambda: tw.smoothing.filter_taubin(
                vertices, faces, iterations=_ITERATIONS, laplacian_operator=operator
            )
        )
        assert result.shape == vertices.shape
    elif bench_case.kind == "open3d":
        _skip_pml_beyond_bunny(bench_case)
        mesh_o3d = bench_case.mesh_o3d
        smoothed_o3d = bench_case.run(
            lambda: mesh_o3d.filter_smooth_taubin(
                number_of_iterations=_ITERATIONS // 2, lambda_filter=0.5, mu=-0.53
            )
        )
        assert len(smoothed_o3d.vertices) == bench_case.n_vertices
    elif bench_case.kind == "pymeshlab":
        _skip_pml_beyond_bunny(bench_case)
        bench_case.run(
            lambda: bench_case.new_meshset_pml().apply_coord_taubin_smoothing(
                stepsmoothnum=_ITERATIONS // 2
            )
        )
    else:  # trimesh mutates in place: rebuild inside the timed callable
        _skip_pml_beyond_bunny(bench_case)
        vertices, faces = bench_case.vertices_np, bench_case.faces_np

        def run() -> tm.Trimesh:
            mesh = tm.Trimesh(vertices, faces, process=False)
            tm.smoothing.filter_taubin(mesh, iterations=_ITERATIONS)
            return mesh

        assert bench_case.run(run).vertices.shape == vertices.shape


@pytest.mark.noparity(
    "pymeshlab",
    oracle="trimesh",
    reason="D2 measured disagreement: apply_coord_hc_laplacian_smoothing implements the same "
    "Vollmer et al. paper but exposes no parameters at all, and its single pass matches "
    "filter_humphrey at none of the 8 x 11 x 11 (iterations, alpha, beta) combinations probed "
    "(best max-coordinate deviation 0.019 on a mesh carrying 0.016 of noise). It is a different "
    "formulation, not this one with other constants; trimesh is the oracle, in "
    "tests/test_smoothing.py::test_filter_humphrey.",
)
@pytest.mark.benchmark(group="filter_humphrey")
@pytest.mark.benchlibs("triwarp", "trimesh", "pymeshlab")
def test_filter_humphrey(bench_case: BenchCase) -> None:
    """
    HC filtering: a Laplacian pass plus a push-back toward the original positions.

    Read the pymeshlab row as a **per-pass** cost only. MeshLab's HC Laplacian takes no parameters,
    so it is one filter call here against ten triwarp iterations, and the module docstring records
    that its output matches ``filter_humphrey`` at no parameter setting -- it is a different
    formulation of the same paper's scheme. trimesh's is the parameter-comparable reference.
    """
    skip_larger_than(bench_case, "dragon")
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        operator = _laplacian_operator(bench_case)
        result = bench_case.run(
            lambda: tw.smoothing.filter_humphrey(
                vertices, faces, iterations=_ITERATIONS, laplacian_operator=operator
            )
        )
        assert result.shape == vertices.shape
    elif bench_case.kind == "pymeshlab":  # one fixed pass, no step count to match
        _skip_pml_beyond_bunny(bench_case)
        bench_case.run(lambda: bench_case.new_meshset_pml().apply_coord_hc_laplacian_smoothing())
    else:  # trimesh mutates in place: rebuild inside the timed callable
        _skip_pml_beyond_bunny(bench_case)
        vertices, faces = bench_case.vertices_np, bench_case.faces_np

        def run() -> tm.Trimesh:
            mesh = tm.Trimesh(vertices, faces, process=False)
            tm.smoothing.filter_humphrey(mesh, iterations=_ITERATIONS)
            return mesh

        assert bench_case.run(run).vertices.shape == vertices.shape


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
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # a non-converged pass fails the benchmark
        result = bench_case.run(
            lambda: tw.smoothing.filter_implicit_fairing(vertices, faces, iterations=_ITERATIONS),
            rounds=3,
        )
    assert result.shape == vertices.shape
    assert np.isfinite(result.numpy()).all()


def _spike_field_np(bench_case: BenchCase) -> np.ndarray:
    """Build a delta at vertex 0: the steepest field there is, so saturation travels furthest."""
    values_np = np.zeros(bench_case.n_vertices, dtype=np.float64)
    values_np[0] = 10.0
    return values_np


def _scalar_field_wp(bench_case: BenchCase) -> wp.array[wp.float32]:
    """Upload the same spike as a device ``float32`` buffer, cached per (mesh, device)."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _scalar_cache:
        _scalar_cache[key] = wp.array(
            _spike_field_np(bench_case).astype(np.float32),
            dtype=wp.float32,
            device=bench_case.device,
        )
    return _scalar_cache[key]


def _new_scalar_meshset_pml(bench_case: BenchCase) -> ml.MeshSet:
    """Build a fresh MeshSet carrying the spike as its vertex scalar attribute."""
    meshset_pml = ml.MeshSet()
    meshset_pml.add_mesh(
        ml.Mesh(
            bench_case.vertices_np,
            np.ascontiguousarray(bench_case.faces_np, dtype=np.int32),
            v_scalar_array=_spike_field_np(bench_case),
        )
    )
    return meshset_pml


@pytest.mark.benchmark(group="filter_scalar_laplacian")
@pytest.mark.benchlibs("triwarp", "pymeshlab")
def test_filter_scalar_laplacian(bench_case: BenchCase) -> None:
    """Ten diffusion passes over a scalar field, against MeshLab's single full-step pass."""
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "pymeshlab":
        _skip_pml_beyond_bunny(bench_case)

        def smooth_pml() -> int:
            meshset_pml = _new_scalar_meshset_pml(bench_case)
            meshset_pml.apply_scalar_smoothing_per_vertex()
            return meshset_pml.current_mesh().vertex_number()

        assert bench_case.run(smooth_pml) == n_vertices
        return
    values, vertices, faces = (
        _scalar_field_wp(bench_case),
        bench_case.vertices_wp,
        bench_case.faces_wp,
    )
    operator = _laplacian_operator(bench_case)
    smoothed = bench_case.run(
        lambda: tw.smoothing.filter_scalar_laplacian(
            values, vertices, faces, iterations=_ITERATIONS, laplacian_operator=operator
        )
    )
    assert smoothed.shape == (n_vertices,)


# MeshLab's two-step defaults, used verbatim on both sides: 3 outer passes, a 60-degree crease
# threshold, 20 normal-diffusion steps and 20 fitting steps.
_TWO_STEP_OUTER = 3
_TWO_STEP_NORMAL_THRESHOLD = 60.0
_TWO_STEP_NORMAL_STEPS = 20
_TWO_STEP_FIT_STEPS = 20


@pytest.mark.noparity(
    "pymeshlab",
    reason="D3 neither parameter exists on the reference: apply_normal_smoothing_per_face takes no "
    "arguments at all -- no iteration count and no crease threshold -- so the two things "
    "filter_normals is parametrized by cannot be set, and its single unconditional pass is not the "
    "crease-gated 20-pass diffusion under test. It is a per-pass cost reference only, which is why "
    "this row's numbers must be read as time-per-pass rather than compared directly. The gated "
    "diffusion is asserted against MeshLab where MeshLab does expose the parameters, in the "
    "filter_two_step group, whose 3 x 20 x 20 defaults both sides are given.",
)
@pytest.mark.benchmark(group="filter_normals")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("triwarp", "pymeshlab", "meshlib")
def test_filter_normals(bench_case: BenchCase) -> None:
    """
    The crease-gated normal diffusion alone: 20 scatter passes over the face adjacency.

    meshlib's ``denoiseNormals`` is a **different formulation** of the same job -- an L1
    minimization over the face graph regularized by ``gamma``, against triwarp's gated diffusion --
    so the two are not iteration-for-iteration comparable and neither parameter maps to the other.
    What ``tests/test_smoothing.py`` establishes is that both recover the same clean normal field;
    this row prices the two routes to it. Its per-edge weight array is the *input* and is built
    outside the timed callable, but the normals it mutates have to be rebuilt per round -- it edits
    them in place, so rounds 2..n would otherwise denoise an already-denoised field.
    """
    n_faces = bench_case.n_faces
    if bench_case.kind == "meshlib":
        mesh_ml = bench_case.new_mesh_ml()
        weights_ml = mm.UndirectedEdgeScalars()
        weights_ml.resize(mesh_ml.topology.undirectedEdgeSize(), 1.0)

        def denoise_ml() -> mm.FaceNormals:
            normals_ml = mm.computePerFaceNormals(mesh_ml)
            mm.denoiseNormals(mesh_ml, normals_ml, weights_ml, 20.0)
            return normals_ml

        assert bench_case.run(denoise_ml).size() == n_faces
        return
    if bench_case.kind == "pymeshlab":
        # ``apply_normal_smoothing_per_face`` exposes no parameters at all -- no step count, no
        # threshold -- so its row is a *single* pass against triwarp's 20 and is a per-pass
        # reference only. It writes the face normal attribute and leaves the coordinates alone, so
        # the MeshSet is shared.
        meshset_pml = bench_case.meshset_pml
        bench_case.run(meshset_pml.apply_normal_smoothing_per_face)
        assert meshset_pml.current_mesh().face_normal_matrix().shape == (n_faces, 3)
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    normals = bench_case.run(
        lambda: tw.smoothing.filter_normals(
            vertices, faces, iterations=_TWO_STEP_NORMAL_STEPS, threshold=_TWO_STEP_NORMAL_THRESHOLD
        )
    )
    assert normals.shape == (n_faces,)


@pytest.mark.benchmark(group="filter_two_step")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("triwarp", "pymeshlab")
def test_filter_two_step(bench_case: BenchCase) -> None:
    """Normal diffusion plus vertex fitting, at MeshLab's own 3 x 20 x 20 defaults on both sides."""
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "pymeshlab":
        _skip_pml_beyond_bunny(bench_case)
        new_meshset_pml = bench_case.new_meshset_pml  # mutates the coordinates

        def two_step_pml() -> int:
            meshset_pml = new_meshset_pml()
            meshset_pml.apply_coord_two_steps_smoothing(
                stepsmoothnum=_TWO_STEP_OUTER,
                normalthr=_TWO_STEP_NORMAL_THRESHOLD,
                stepnormalnum=_TWO_STEP_NORMAL_STEPS,
                stepfitnum=_TWO_STEP_FIT_STEPS,
            )
            return meshset_pml.current_mesh().vertex_number()

        assert bench_case.run(two_step_pml, rounds=3) == n_vertices
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    smoothed = bench_case.run(
        lambda: tw.smoothing.filter_two_step(
            vertices,
            faces,
            iterations=_TWO_STEP_OUTER,
            threshold=_TWO_STEP_NORMAL_THRESHOLD,
            normal_iterations=_TWO_STEP_NORMAL_STEPS,
            fit_iterations=_TWO_STEP_FIT_STEPS,
        ),
        rounds=3,
    )
    assert smoothed.shape == (n_vertices,)


@pytest.mark.noparity(
    "open3d",
    oracle="pymeshlab",
    reason="D2 a per-vertex factor no parameter can absorb: open3d's filter_sharpen adds "
    "strength * (deg(v) * v - sum of neighbours), the UNNORMALIZED uniform residual, where "
    "triwarp's unsharp mask adds weight * (v - mean of neighbours) through a row-stochastic "
    "operator. Probed before the row landed: the per-vertex displacement ratio o3d/triwarp equals "
    "the vertex degree to 7 significant digits (4.99977-6.00074 on icosphere(3), correlation with "
    "degree 0.9999993), so the two agree only on degree-regular meshes and no strength mapping "
    "fixes an irregular one. pymeshlab is the oracle for this group, in "
    "tests/test_smoothing.py::test_filter_sharpen_matches_pymeshlab.",
)
@pytest.mark.benchmark(group="filter_sharpen")
@pytest.mark.benchlibs("triwarp", "open3d", "pymeshlab")
def test_filter_sharpen(bench_case: BenchCase) -> None:
    """Five Laplacian passes plus one blend: the cheapest thing in the module, on the scan sweep."""
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "open3d":
        _skip_pml_beyond_bunny(bench_case)
        mesh_o3d = bench_case.mesh_o3d
        # Small strength: the degree factor in its update amplifies noise fast (see the exemption).
        sharpened_o3d = bench_case.run(
            lambda: mesh_o3d.filter_sharpen(number_of_iterations=5, strength=0.05)
        )
        assert len(sharpened_o3d.vertices) == n_vertices
        return
    if bench_case.kind == "pymeshlab":
        _skip_pml_beyond_bunny(bench_case)
        new_meshset_pml = bench_case.new_meshset_pml

        def unsharp_pml() -> int:
            meshset_pml = new_meshset_pml()
            meshset_pml.apply_coord_unsharp_mask(weight=0.3, weightorig=1.0, iterations=5)
            return meshset_pml.current_mesh().vertex_number()

        assert bench_case.run(unsharp_pml) == n_vertices
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    operator = _laplacian_operator(bench_case)
    sharpened = bench_case.run(
        lambda: tw.smoothing.filter_sharpen(
            vertices, faces, weight=0.3, iterations=5, laplacian_operator=operator
        )
    )
    assert sharpened.shape == (n_vertices,)


_REGION_FRACTION = 0.25  # free vertices as a fraction of the mesh, by a coordinate cut


def _free_mask_np(bench_case: BenchCase) -> np.ndarray:
    """
    Cut a contiguous free region: the top quarter by z, so its rim is one closed curve.

    A random mask would give the solver a shredded region with an enormous rim and would measure
    something else -- the region *boundary* is what both libraries fold into the right-hand side, so
    its length is the thing to hold fixed across rows.
    """
    z_np = bench_case.vertices_np[:, 2]
    threshold = float(np.quantile(z_np, 1.0 - _REGION_FRACTION))
    return np.ascontiguousarray(z_np > threshold)


@pytest.mark.benchmark(group="smooth_region")
@pytest.mark.benchlibs("triwarp", "meshlib")
def test_smooth_region(bench_case: BenchCase) -> None:
    """
    Solve the umbrella-Laplacian Dirichlet system on a free region: a sparse solve, not a filter.

    Read against the ``filter_*`` groups above, which apply a fixed number of explicit passes: this
    one solves to a fixpoint, so its cost is a linear solve over the free set and is driven by
    that set's size and shape rather than by an iteration count. The free region is the top quarter
    of the mesh by z on both sides, so the two solve the same system.

    meshlib's ``positionVertsSmoothly`` is that same system with the same unit edge weights
    (``EdgeWeights.Unit``, ``VertexMass.Unit``), factorized where triwarp iterates -- the answers
    agree to 1e-4 (``tests/test_smoothing.py``). It mutates the mesh in place and returns nothing,
    so its mesh is rebuilt inside the timed callable and the row carries the build.

    **The two rows land on opposite sides of a solver switch**, which is the thing to know before
    reading a change in either. ``smooth_region`` asks for ``preconditioner="auto"``: Jacobi under a
    2 000-iteration cap, escalating to a multigrid V-cycle only if that has not converged. ``bunny``
    needs 6 541 Jacobi iterations, so it escalates and the row measures **159.7 ms against 225.7
    before the hierarchy existed**; ``bunny_decimated`` needs 1 784, converges inside the cap and is
    unchanged. So a change on one row and not the other is more likely to be the cap than the solver
    -- see ``linalg.CG_PROBE_ITERATIONS``.
    """
    skip_larger_than(
        bench_case,
        "bunny",
        "the Dirichlet solve over a quarter of the mesh runs into tens of seconds past bunny "
        "(15.8 s on dragon, 24.9 s on happy_buddha for one triwarp round)",
    )
    n_vertices = bench_case.n_vertices
    free_np = _free_mask_np(bench_case)
    if bench_case.kind == "meshlib":
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np

        def smooth_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(vertices_np, faces_np)
            mm.positionVertsSmoothly(
                mesh_ml, mn.vertBitSetFromBools(free_np), mm.EdgeWeights.Unit, mm.VertexMass.Unit
            )
            return mesh_ml.topology.numValidVerts()

        assert bench_case.run(smooth_ml, rounds=3) > 0
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    free_wp = wp.array(free_np, dtype=wp.bool, device=bench_case.device)
    smoothed = bench_case.run(
        lambda: tw.smoothing.smooth_region(vertices, faces, free_wp), rounds=3
    )
    assert smoothed.shape == (n_vertices,)


@pytest.mark.benchmark(group="smooth_region_fixed_rim")
@pytest.mark.benchlibs("triwarp", "meshlib")
def test_smooth_region_fixed_rim(bench_case: BenchCase) -> None:
    """
    The same solve with the region rim pinned: the variant hole filling actually calls.

    Read against ``smooth_region`` on the identical region -- the only difference is whether the rim
    is a hard C0 constraint, which changes the system's size rather than its kind, so the two rows
    should track each other closely. A large gap would mean one of the two is not solving what it
    says.

    meshlib's ``positionVertsSmoothlySharpBd`` is the matching variant and takes the region through
    ``PositionVertsSmoothlyParams``; it agrees with triwarp to 1e-5 (``tests/test_smoothing.py``).
    Same in-place mutation, so same rebuild inside the timed callable.
    """
    skip_larger_than(
        bench_case,
        "bunny",
        "the Dirichlet solve over a quarter of the mesh runs into tens of seconds past bunny "
        "(15.8 s on dragon, 24.9 s on happy_buddha for one triwarp round)",
    )
    n_vertices = bench_case.n_vertices
    free_np = _free_mask_np(bench_case)
    if bench_case.kind == "meshlib":
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np

        def smooth_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(vertices_np, faces_np)
            params_ml = mm.PositionVertsSmoothlyParams()
            params_ml.region = mn.vertBitSetFromBools(free_np)
            mm.positionVertsSmoothlySharpBd(mesh_ml, params_ml)
            return mesh_ml.topology.numValidVerts()

        assert bench_case.run(smooth_ml, rounds=3) > 0
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    free_wp = wp.array(free_np, dtype=wp.bool, device=bench_case.device)
    smoothed = bench_case.run(
        lambda: tw.smoothing.smooth_region_fixed_rim(vertices, faces, free_wp), rounds=3
    )
    assert smoothed.shape == (n_vertices,)


@pytest.mark.benchmark(group="inflate")
@pytest.mark.benchlibs("triwarp")
def test_inflate(bench_case: BenchCase) -> None:
    """
    The balloon flow: per pass, a vertex-normal build, one displacement map and one Laplacian pass.

    Read against ``filter_laplacian`` at the same iteration count -- the difference between the two
    groups is what the normals and the displacement cost, and it should be roughly the normal build,
    since the displacement is one ``wp.map`` with the kernel hoisted out of the loop.

    triwarp-only, and not by omission. MeshLib's ``inflate`` is the only reference that has one and
    it cannot be timed here: with every vertex selected -- the operation this performs -- it
    collapses the mesh to a point at every pressure probed, because its implicit solve takes the
    *unselected* vertices as its boundary condition. Given a region it works but solves a different
    problem, dropping the volume below the input before pressure raises it again. Measured numbers
    are in ``tests/test_smoothing.py``.

    First measurement, medians on an RTX 5090 at the default 3 passes: **4.59 ms**
    (``bunny_decimated``), **5.16** (``bunny``), **5.51** (``dragon``), **5.72**
    (``happy_buddha``), **127.1** (``lucy``). Flat from 40k to 1.09M faces, so the three passes are
    launch-bound rather than data-bound at this scale -- and ``lucy``'s 25x jump at a comparable
    face count is the same unexplained outlier ``split_faces_along_field`` records on that mesh.
    """
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    pressure = 0.1 * bench_case.mean_edge
    inflated = bench_case.run(lambda: tw.smoothing.inflate(vertices, faces, pressure), rounds=3)
    assert int(inflated.shape[0]) == bench_case.n_vertices


@pytest.mark.benchmark(group="remove_spikes")
@pytest.mark.benchlibs("triwarp", "meshlib")
def test_remove_spikes(bench_case: BenchCase) -> None:
    """
    Detect and flatten needle vertices: per pass, the corner angles, a defect scatter and one map.

    On a clean mesh this is a *detector* -- one pass finds nothing and the loop stops -- so the row
    is really the cost of asking, which is what a caller pays unconditionally in a repair pipeline.
    That makes it comparable with meshlib's ``removeSpikes`` on the same input, since that also
    finds nothing to do; and it means the row is dominated by ``face_angles`` plus the scatter rather
    than by any displacement.

    meshlib mutates in place, so its mesh is rebuilt per round the way the other ``repair`` rows do.
    Both sides are pinned against each other on a genuinely spiky mesh in
    ``tests/test_smoothing.py``; no scan mesh in this registry has a spike, which is why that test
    needs a generated fixture and this row does not.

    First measurement, medians on an RTX 5090: **15.68 ms** on ``bunny`` against meshlib's 9.62
    (1.63x behind) and **20.87 ms** on ``dragon`` against 60.73 (**2.9x ahead**), with
    ``happy_buddha`` at 22.84 and ``lucy`` at 100.5. The crossover is between 70k and 871k faces,
    which is where a per-vertex angle scatter starts beating a threaded serial pass -- and note
    meshlib's row carries its mesh build while triwarp's carries none, so the small-mesh figure is
    if anything generous to triwarp.
    """
    threshold = 0.5 * math.pi
    if bench_case.kind == "meshlib":

        def remove_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(bench_case.vertices_np, bench_case.faces_np)
            mm.removeSpikes(mesh_ml, 10, threshold)
            return mesh_ml.topology.numValidVerts()

        assert bench_case.run(remove_ml, rounds=3) > 0
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    repaired, flattened = bench_case.run(
        lambda: tw.smoothing.remove_spikes(vertices, faces, threshold), rounds=3
    )
    assert flattened >= 0
    assert int(repaired.shape[0]) == bench_case.n_vertices


@pytest.mark.benchmark(group="equalize_triangle_areas")
@pytest.mark.benchlibs("triwarp", "meshlib")
def test_equalize_triangle_areas(bench_case: BenchCase) -> None:
    """
    A 3x3 float64 solve per vertex per pass, over the incident-face CSR.

    Read against ``filter_laplacian``, which is the same shape of pass over the same 1-ring and does
    a weighted average where this solves: the gap between the two rows is what the area objective
    costs, and it is a fixed per-vertex constant rather than anything that grows.

    ``no_shrinkage`` is deliberately left off. It adds a vertex-normal pass and a 2x2 solve per
    vertex per iteration, which is a second measurement rather than a variation of this one, and the
    two agree with meshlib either way (``tests/test_smoothing.py``, exactly without it and to
    3.6e-07 with).

    meshlib's ``equalizeTriAreas`` is the same solve, threaded across vertices and mutating in
    place, so its mesh is rebuilt per round -- and the positions agree **exactly**
    (``tests/test_smoothing.py``), which is what makes the ratio below a fair one.

    First measurement, medians on an RTX 5090, ten passes:

    | mesh | triwarp-cuda | meshlib |
    |---|---|---|
    | ``bunny`` | 0.481 ms | 14.67 (30.5x) |
    | ``bunny_decimated`` | 0.523 ms | 11.78 (22.5x) |
    | ``dragon`` | 3.05 ms | 122.3 (40.2x) |
    | ``happy_buddha`` | 3.70 ms | (capped) |
    | ``lucy`` | 111.5 ms | (capped) |

    ``bunny`` reads *slower* than the 12x larger ``dragon`` because it is the first mesh in the
    selection and carries the module's compile; read the three large rows against each other, where
    the scaling is clean (3.05 / 3.70 / 111.5 at 0.87M / 1.09M / 28M faces).
    """
    if bench_case.kind == "meshlib":

        def equalize_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(bench_case.vertices_np, bench_case.faces_np)
            params_ml = mm.MeshEqualizeTriAreasParams()
            params_ml.iterations = _ITERATIONS
            mm.equalizeTriAreas(mesh_ml, params_ml)
            return mesh_ml.topology.numValidVerts()

        assert bench_case.run(equalize_ml, rounds=3) > 0
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    relaxed = bench_case.run(
        lambda: tw.smoothing.equalize_triangle_areas(vertices, faces, _ITERATIONS), rounds=3
    )
    assert relaxed.shape == (bench_case.n_vertices,)


@pytest.mark.benchmark(group="relax_keep_volume")
@pytest.mark.benchlibs("triwarp", "meshlib")
def test_relax_keep_volume(bench_case: BenchCase) -> None:
    """
    Two launches per pass over the 1-ring CSR, against ``filter_laplacian``'s one.

    That ratio is the whole cost model: the volume correction is a second pass over the same
    adjacency reading a displacement field instead of positions, so this row should sit near twice
    ``filter_laplacian``'s and anything else means the adjacency build (hoisted out of the loop
    here, as there) has moved.

    meshlib's ``relaxKeepVolume`` is the same two-pass formulation, and the positions agree to
    1.9e-09 (``tests/test_smoothing.py``). It mutates in place, so its mesh is rebuilt per round.

    First measurement, medians on an RTX 5090, ten passes:

    | mesh | triwarp-cuda | meshlib |
    |---|---|---|
    | ``bunny`` | 1.72 ms | 14.53 (8.4x) |
    | ``bunny_decimated`` | 1.80 ms | 12.12 (6.7x) |
    | ``dragon`` | 3.93 ms | 130.4 (33.2x) |
    | ``happy_buddha`` | 4.34 ms | (capped) |
    | ``lucy`` | 119.8 ms | (capped) |

    Against ``equalize_triangle_areas`` on the same passes and meshes -- 0.481 / 0.523 / 3.05 / 3.70
    / 111.5 ms -- the two are within 30 % from ``dragon`` up. That is the cost model working out:
    two cheap ring passes here against one ring pass with a 3x3 float64 solve there, so neither the
    solve nor the extra launch dominates and both rows are bandwidth on the adjacency.
    """
    if bench_case.kind == "meshlib":

        def relax_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(bench_case.vertices_np, bench_case.faces_np)
            params_ml = mm.MeshRelaxParams()
            params_ml.iterations = _ITERATIONS
            mm.relaxKeepVolume(mesh_ml, params_ml)
            return mesh_ml.topology.numValidVerts()

        assert bench_case.run(relax_ml, rounds=3) > 0
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    relaxed = bench_case.run(
        lambda: tw.smoothing.relax_keep_volume(vertices, faces, _ITERATIONS), rounds=3
    )
    assert relaxed.shape == (bench_case.n_vertices,)


@pytest.mark.benchmark(group="relax_approx")
@pytest.mark.benchlibs("triwarp", "meshlib")
def test_relax_approx(bench_case: BenchCase) -> None:
    """
    Neighbourhood fitting, where the neighbourhood build dominates the fit.

    The geodesic balls are built **once** and reused across passes, so this row is one
    ``geodesic_ball`` plus a per-vertex plane fit per pass -- read it against
    ``curvature.principal_curvature``, which builds the same balls and then does a strictly larger
    5x5 solve on each. A change here that does not show there is in the fit; one that shows in both
    is in the ball.

    The radius is 3 % of the bounding-box diagonal, which is the scale at which a ball holds enough
    vertices to fit on every mesh in the registry. It is not scale-free and there is no default:
    meshlib's own ``surfaceDilateRadius`` default of ``0`` is a measured no-op (``tests``), so the
    two rows would otherwise not be timing the same work at all.

    ``fit="planar"`` on both sides. The quadric adds a 6x6 QR per vertex per pass and is a separate
    measurement, not a variation of this one.

    First measurement, medians on an RTX 5090, one pass:

    | mesh | triwarp-cuda | meshlib |
    |---|---|---|
    | ``bunny`` | 6.42 ms | 39.10 (6.1x) |
    | ``bunny_decimated`` | 2.16 ms | 12.98 (6.0x) |
    | ``dragon`` | 202.5 ms | 4 665 (23.0x) |
    | ``happy_buddha`` | 149.1 ms | (capped) |

    **This is the only row in the module whose cost is superlinear**, and the ball is why: 202 ms on
    ``dragon`` against 6.4 on ``bunny`` is 31x for 12x the faces, because a fixed 3 % radius holds
    more vertices as the mesh refines. That is the thing to watch on any change here -- a regression
    in the *fit* would move all four rows together, and one in the ball would move only these two.
    """
    skip_larger_than(
        bench_case,
        "happy_buddha",
        "the geodesic-ball pools are sized per source chunk, and at lucy's 14M vertices the "
        "64 MB flat-buffer allocation fails outright -- an OOM that then corrupts every later row "
        "in the same process, so this is a hard cap rather than a slow row",
    )
    radius = 0.03 * float(
        np.linalg.norm(bench_case.vertices_np.max(0) - bench_case.vertices_np.min(0))
    )
    if bench_case.kind == "meshlib":

        def relax_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(bench_case.vertices_np, bench_case.faces_np)
            params_ml = mm.MeshApproxRelaxParams()
            params_ml.iterations = 1
            params_ml.surfaceDilateRadius = radius
            params_ml.type = mm.RelaxApproxType.Planar
            mm.relaxApprox(mesh_ml, params_ml)
            return mesh_ml.topology.numValidVerts()

        assert bench_case.run(relax_ml, rounds=3) > 0
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    relaxed = bench_case.run(
        lambda: tw.smoothing.relax_approx(vertices, faces, radius, 1), rounds=3
    )
    assert relaxed.shape == (bench_case.n_vertices,)


@pytest.mark.benchmark(group="smooth_region_boundary")
@pytest.mark.benchlibs("triwarp", "meshlib")
def test_smooth_region_boundary(bench_case: BenchCase) -> None:
    """
    A harmonic solve per pass, on the same region as ``smooth_region`` -- but a *scalar* one.

    Read against ``smooth_region``: same region, same cotangent operator, one right-hand side
    instead of three, and the free set is the region's rim band rather than its whole interior. So
    this row should be much the cheaper of the two whenever the region is fat, and the gap closing
    would mean the band had stopped being thin.

    The solve is rebuilt every pass rather than hoisted, and that is not an oversight: the cotangent
    weights are a function of the positions the previous pass moved, so a hoisted operator would
    solve last pass's problem.

    meshlib's ``smoothRegionBoundary`` additionally flips the band's interior edges before each
    solve, which this port does not do -- so its row carries connectivity work triwarp's does not,
    and the two are pinned on the *moved set* and the rim length rather than element-wise
    (``tests/test_smoothing.py``). It mutates in place, so its mesh is rebuilt per round.

    First measurement, medians on an RTX 5090, four passes:

    | mesh | triwarp-cuda | meshlib |
    |---|---|---|
    | ``bunny`` | 15.03 ms | 16.85 (1.12x) |
    | ``bunny_decimated`` | 14.77 ms | 11.89 (**0.81x**) |

    The only **loss** among this pass's new rows, and the reason is visible in the shape: triwarp is
    flat from 16k to 69k faces while meshlib tracks the mesh, so the row is four conjugate-gradient
    solves and their fixed per-call cost rather than anything proportional. Against
    ``smooth_region``'s 159.7 ms on ``bunny`` over the same region it is **10.6x cheaper**, which is
    the band being thin -- exactly what this group was written to check.
    """
    skip_larger_than(
        bench_case,
        "bunny",
        "four harmonic solves over the region rim run into tens of seconds past bunny, the same "
        "wall smooth_region hits on the same region",
    )
    region_np = np.ascontiguousarray(_free_mask_np(bench_case)[bench_case.faces_np].any(axis=1))
    if bench_case.kind == "meshlib":

        def smooth_ml() -> int:
            mesh_ml = mesh_ml_from_numpy(bench_case.vertices_np, bench_case.faces_np)
            region_ml = face_bitset_ml(region_np)
            mm.smoothRegionBoundary(mesh_ml, region_ml, 4)
            return mesh_ml.topology.numValidVerts()

        assert bench_case.run(smooth_ml, rounds=3) > 0
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    region_wp = wp.array(region_np, dtype=wp.bool, device=bench_case.device)
    smoothed = bench_case.run(
        lambda: tw.smoothing.smooth_region_boundary(vertices, faces, region_wp, 4), rounds=3
    )
    assert smoothed.shape == (bench_case.n_vertices,)
