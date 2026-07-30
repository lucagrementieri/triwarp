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

The last two groups leave positions alone and run over a per-vertex **scalar** field:
``filter_scalar_laplacian`` against ``apply_scalar_smoothing_per_vertex`` and
``saturate_scalar_gradient`` against ``apply_scalar_saturation_per_vertex``. Both are on the
**scale** axis, and they are the two ends of a spectrum this module otherwise does not cover:
diffusion is a fixed number of SpMV passes, while saturation is a Bellman-Ford relaxation whose pass
count is the *graph diameter of the violating region*, so it is the one group here whose cost is
genuinely data-dependent. Seeding it from a single spike is the worst case on purpose -- the cap has
to propagate across the whole mesh -- so read that row as an upper bound rather than a typical one.
Both filters need the scalar attribute to exist on the MeshSet, which means those rows rebuild it
(they mutate the attribute, and saturation is not idempotent in it either).
"""

from __future__ import annotations

import warnings

import numpy as np
import pymeshlab as ml
import pytest
import trimesh as tm
import warp as wp
import warp.sparse as wps
from conftest import BenchCase, skip_larger_than

import triwarp as tw

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
@pytest.mark.benchlibs("triwarp", "pymeshlab")
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


@pytest.mark.benchmark(group="filter_taubin")
@pytest.mark.benchlibs("triwarp", "trimesh", "pymeshlab")
def test_filter_taubin(bench_case: BenchCase) -> None:
    """
    The lambda-nu alternation: two SpMVs per iteration instead of one.

    Against ``filter_laplacian_integration``'s explicit row this measures exactly the second pass,
    so the two rows should sit at a ratio near 2 and nothing else should separate them. All three
    libraries implement Taubin's 1995 scheme; MeshLab's inflating step is ``mu=-0.53`` against
    triwarp's and trimesh's ``nu=0.5``, which changes the fixed point but not the work per pass.
    """
    skip_larger_than(bench_case, "dragon")
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        operator = _laplacian_operator(bench_case)
        result = bench_case.run(
            lambda: tw.smoothing.filter_taubin(
                vertices, faces, iterations=_ITERATIONS, laplacian_operator=operator
            )
        )
        assert result.shape == vertices.shape
    elif bench_case.kind == "pymeshlab":
        _skip_pml_beyond_bunny(bench_case)
        bench_case.run(
            lambda: bench_case.new_meshset_pml().apply_coord_taubin_smoothing(
                stepsmoothnum=_ITERATIONS
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


@pytest.mark.benchmark(group="saturate_scalar_gradient")
@pytest.mark.benchlibs("triwarp", "pymeshlab")
def test_saturate_scalar_gradient(bench_case: BenchCase) -> None:
    """
    Lipschitz projection of a scalar field: a relaxation whose pass count is the graph diameter.

    Capped at ``bunny_decimated`` on both sides. The spike seed makes every pass matter, so the
    triwarp row is ``diameter`` launches deep and the MeshLab row is a serial flood over the same
    region -- neither says anything new at larger scale that the two smallest meshes do not.
    """
    skip_larger_than(bench_case, "bunny_decimated", "a spike-seeded relaxation is diameter-deep")
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "pymeshlab":

        def saturate_pml() -> int:
            meshset_pml = _new_scalar_meshset_pml(bench_case)
            meshset_pml.apply_scalar_saturation_per_vertex(gradientthr=1.0)
            return meshset_pml.current_mesh().vertex_number()

        assert bench_case.run(saturate_pml) == n_vertices
        return
    values, vertices, faces = (
        _scalar_field_wp(bench_case),
        bench_case.vertices_wp,
        bench_case.faces_wp,
    )
    saturated = bench_case.run(
        lambda: tw.smoothing.saturate_scalar_gradient(values, vertices, faces, threshold=1.0),
        rounds=3,
    )
    assert saturated.shape == (n_vertices,)


# MeshLab's two-step defaults, used verbatim on both sides: 3 outer passes, a 60-degree crease
# threshold, 20 normal-diffusion steps and 20 fitting steps.
_TWO_STEP_OUTER = 3
_TWO_STEP_NORMAL_THRESHOLD = 60.0
_TWO_STEP_NORMAL_STEPS = 20
_TWO_STEP_FIT_STEPS = 20


@pytest.mark.benchmark(group="filter_normals")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("triwarp", "pymeshlab")
def test_filter_normals(bench_case: BenchCase) -> None:
    """The crease-gated normal diffusion alone: 20 scatter passes over the face adjacency."""
    n_faces = bench_case.n_faces
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


@pytest.mark.benchmark(group="filter_unsharp_mask")
@pytest.mark.benchlibs("triwarp", "pymeshlab")
def test_filter_unsharp_mask(bench_case: BenchCase) -> None:
    """Five Laplacian passes plus one blend: the cheapest thing in the module, on the scan sweep."""
    n_vertices = bench_case.n_vertices
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
        lambda: tw.smoothing.filter_unsharp_mask(
            vertices, faces, weight=0.3, iterations=5, laplacian_operator=operator
        )
    )
    assert sharpened.shape == (n_vertices,)
