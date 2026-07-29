"""
Benchmarks for ``triwarp.vector_heat`` and the connection Laplacian it runs on.

Everything here is conjugate gradient, so **quality** is the axis that matters: ``saddle`` against
``saddle_graded`` holds vertices, faces and connectivity fixed and only worsens the aspect ratio,
which is what sets the iteration count. ``heat_geodesic_conditioning`` already shows 3.4x for the
scalar solve on those two meshes; these groups ask whether the vector solve behaves the same way.
``scale`` is reported alongside for the assembly cost, and ``transport_tangent_vectors`` carries a
``setup=full``/``setup=amortized`` layer -- with the operators hoisted out on both sides, since
``vector_heat_operators`` is a real API path and ``MeshVectorHeatSolver`` is an object -- so the
question "what does a *second* solve on the same mesh cost?" is measured rather than argued.

The three functions cost different numbers of solves, which is most of what separates them:

* ``extend_scalar`` -- two scalar solves.
* ``transport_tangent_vectors`` -- one ``2 x 2``-block vector solve plus an ``extend_scalar``.
* ``log_map`` -- a vector solve, a full ``heat_geodesic`` (two more scalar solves) and a gradient
pass.

The amortized rows price the operator assembly: ``transport_tangent_vectors`` goes
**10.87 -> 7.11 ms** on ``saddle`` when the operators are reused, so a third of a transport call is
assembly. Against the reference the comparison *flips sign* between the two rows -- 14.6x faster at
``full`` (158 ms), 2.4x slower at ``amortized`` (2.94 ms) -- for the reason the ``heat_geodesic``
table shows: a factorization is expensive once and cheap thereafter, conjugate gradient is neither.
Note too that transport barely feels the ``quality`` axis (7.11 against 7.05 ms) where ``log_map``
feels it 2.7x; the difference is the distance field the log map also solves.

Measured on an RTX 5090, ``saddle`` then ``saddle_graded``: ``extend_scalar`` 7.2 / 5.6 ms against
the reference's 38.3 / 38.1; ``transport_tangent_vectors`` 10.4 ms at ``saddle``; ``log_map`` 26.6 /
**72.7 ms** against 182 / 188. That last row is the module's real result: triwarp's vector solve
pays **2.7x** for the worse aspect ratio while the reference's factorization pays nothing -- the
same iterative-versus-direct trade ``heat_geodesic_conditioning`` shows for the scalar. Assembly
alone is measured as ``connection_laplacian`` in
[`test_laplacian.py`](test_laplacian.py) -- 1.11 / 1.23 / 2.82 ms over the scale axis -- since the
operator itself lives in ``triwarp.laplacian``.

References
----------
**potpourri3d** is the only reference for any of this, and its solvers are constructed inside the
timed callable per this suite's convention -- which for ``MeshVectorHeatSolver`` means building a
halfedge mesh and factoring both the cotangent and connection Laplacians. It is constructed with
``use_intrinsic_delaunay=False`` so the discretization matches triwarp's. Note that it cannot run on
``cave_cube`` at all: its right-angle diagonals give zero cotangent weights and geometry-central's
factorization fails there (see ``tests/test_vector_heat.py``). No group here uses that mesh.

**trimesh**, **libigl** and **open3d** have no tangent-space machinery, so nothing else appears
here.
"""

from __future__ import annotations

import numpy as np
import potpourri3d as pp3d
import pytest
import warp as wp
from conftest import BenchCase, skip_larger_than

import triwarp as tw

# Every case is at least two float64 CG solves; the reference also factors two sparse systems.
_ROUNDS = 3
_SOURCES = np.array([0], dtype=np.int32)


def _sources_wp(bench_case: BenchCase) -> wp.array[wp.int32]:
    return wp.array(_SOURCES, dtype=wp.int32, device=bench_case.device)


def _solver_pp(bench_case: BenchCase) -> pp3d.MeshVectorHeatSolver:
    return pp3d.MeshVectorHeatSolver(
        bench_case.vertices_np,
        np.ascontiguousarray(bench_case.faces_np, dtype=np.int32),
        use_intrinsic_delaunay=False,
    )


@pytest.mark.benchmark(group="extend_scalar")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("triwarp", "potpourri3d")
def test_extend_scalar(bench_case: BenchCase) -> None:
    """Two scalar diffusions and a division, on the conditioning axis."""
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        sources = _sources_wp(bench_case)
        values = wp.array(np.array([1.0]), dtype=wp.float64, device=bench_case.device)
        extended = bench_case.run(
            lambda: tw.vector_heat.extend_scalar(vertices, faces, sources, values), rounds=_ROUNDS
        )
        assert extended.shape == (n_vertices,)
    else:
        assert np.asarray(
            bench_case.run(lambda: _solver_pp(bench_case).extend_scalar([0], [1.0]), rounds=_ROUNDS)
        ).shape == (n_vertices,)


def _run_transport(bench_case: BenchCase, *, amortized: bool) -> None:
    """Time one parallel transport, with the operators either rebuilt or reused."""
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        sources = _sources_wp(bench_case)
        vectors = wp.array(
            np.array([[1.0, 0.0]], dtype=np.float32), dtype=wp.vec2, device=bench_case.device
        )
        operators = tw.vector_heat.vector_heat_operators(vertices, faces) if amortized else None
        transported = bench_case.run(
            lambda: tw.vector_heat.transport_tangent_vectors(
                vertices, faces, sources, vectors, operators=operators
            ),
            rounds=_ROUNDS,
        )
        assert transported.shape == (n_vertices,)
    elif amortized:
        # The reference's own split: construct the solver once, then time only its solve.
        solver = _solver_pp(bench_case)
        assert np.asarray(
            bench_case.run(
                lambda: solver.transport_tangent_vectors([0], [[1.0, 0.0]]), rounds=_ROUNDS
            )
        ).shape == (n_vertices, 2)
    else:
        assert np.asarray(
            bench_case.run(
                lambda: _solver_pp(bench_case).transport_tangent_vectors([0], [[1.0, 0.0]]),
                rounds=_ROUNDS,
            )
        ).shape == (n_vertices, 2)


@pytest.mark.benchmark(group="transport_tangent_vectors")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("triwarp", "potpourri3d")
@pytest.mark.parametrize("setup", ["full", "amortized"])
def test_transport_tangent_vectors(bench_case: BenchCase, setup: str) -> None:
    """A 2x2-block vector solve plus a scalar extension for the magnitude."""
    _run_transport(bench_case, amortized=setup == "amortized")


@pytest.mark.benchmark(group="log_map")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("triwarp", "potpourri3d")
def test_log_map(bench_case: BenchCase) -> None:
    """The most expensive of the three: a vector solve, a distance field and a gradient pass."""
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        logarithm = bench_case.run(
            lambda: tw.vector_heat.log_map(vertices, faces, 0), rounds=_ROUNDS
        )
        assert logarithm.shape == (n_vertices,)
    else:
        # geometry-central's own strategy name for the same construction.
        assert np.asarray(
            bench_case.run(
                lambda: _solver_pp(bench_case).compute_log_map(0, "VectorHeat"), rounds=_ROUNDS
            )
        ).shape == (n_vertices, 2)


@pytest.mark.benchmark(group="vector_heat_scale")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "potpourri3d")
def test_transport_tangent_vectors_scale(bench_case: BenchCase) -> None:
    """The same transport over the size sweep, for the assembly-versus-solve split."""
    skip_larger_than(bench_case, "sphere_large")
    _run_transport(bench_case, amortized=False)
