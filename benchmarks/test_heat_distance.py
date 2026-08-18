"""
Benchmarks for ``triwarp.heat.distance.heat_geodesic``.

Two axes, and the second is the interesting one:

* **scale** -- the clean size sweep, 5 120 to 327 680 faces.
* **quality** -- ``saddle`` against ``saddle_graded``: identical vertices, faces and connectivity,
  worst aspect ratio 1.6 against 4 719. Measured at 19.6 ms and 67.0 ms, so **3.4x for a change
  that no face-count registry can express**. The heat method is two conjugate-gradient solves, and
  CG iteration count is a function of the cotangent Laplacian's condition number: bad aspect
  ratios and obtuse angles (which make cotangent weights go negative) inflate it directly. This
  group is the module's real subject.

The method (Crane et al.) is three stages, and the timing is dominated by the two of them that
are sparse linear solves: diffuse heat from the sources for a short time ``t``, normalize the
resulting gradient into a unit field pointing away from the sources, then integrate that field
back into a distance function with a Poisson solve. The per-face gradient normalization in between
is a single cheap pass.

The whole computation runs in ``float64`` (the diffused heat decays exponentially and underflows
``float32``, collapsing the far field), which on a consumer GPU means the solves run at the
device's much lower double-precision rate. That is inherent to the method, not a tuning choice.

``geodesic_ball`` is **not** benchmarked here -- it lives in ``triwarp.neighbors`` and is timed as
``query_geodesic_ball`` in [`test_proximity.py`](test_proximity.py), next to the other neighborhood
queries it belongs with.

References
----------
**libigl** implements the same method (``igl::heat_geodesics``) and now runs on every mesh in both
axes. It previously could not: on every scan mesh ``igl.heat_geodesics_precompute`` raises
``RuntimeError: heat_geodesics: Precomputation failed.`` -- it factors the cotangent and Poisson
systems directly (Cholesky) at precompute time, and those factorizations fail on the scan meshes.
The synthetic meshes are manifold by construction and it succeeds on all of them, measured at
0.10-0.65 s.

That makes the quality axis a direct iterative-versus-direct comparison.

**potpourri3d** (geometry-central) implements the same method a third way, and is constructed with
``use_robust=False`` so all three sides discretize the same triangulation -- its default mollifies
and flips to an intrinsic Delaunay triangulation first, which is more work and a different operator.
It also ships **fast marching** (``MeshFastMarchingDistanceSolver``), a different algorithm for the
same task, timed as its own group; triwarp has no equivalent by design.

**pymeshlab** implements the same method a *fourth* way
(``compute_scalar_by_heat_geodesic_distance_from_selection_per_vertex``), and its own documentation
states the two properties this module is built around: "as this implementation does not use
intrinsic triangulation it is very sensitive to triangulation" -- the same caveat the ``quality``
axis exists to measure -- and "first run takes longer as factorization has to be built", which is
exactly the ``setup=full`` / ``setup=amortized`` split. So it is the only reference whose amortized
row needs no API gymnastics: calling the filter twice on the same MeshSet *is* the amortized path.
Sources go in as a vertex selection (``compute_selection_by_condition_per_vertex(condselect='(vi ==
0)')``, i.e. vertex 0, the same source the other three libraries get).

Measured, it is the third independent confirmation of the ``quality`` axis's point: **71.4 against
71.1 ms** across ``saddle`` / ``saddle_graded`` -- dead flat, like the other two direct solvers --
while triwarp goes 22.4 -> 68.7 ms. Three factorizing implementations all insensitive to a
conditioning change that costs triwarp's CG 3x is about as clear as this suite gets.

It also ships a **second, unrelated** algorithm for the same task --
``compute_scalar_by_geodesic_distance_from_given_point_per_vertex``, a Dijkstra-style front over the
edge graph rather than a PDE solve -- which is timed in the ``fast_marching_distance`` group
alongside potpourri3d's, since that group exists to price the non-PDE alternatives.

**trimesh** has no geodesic distance of any kind (``trimesh.graph`` offers only combinatorial
traversal over the edge graph, not a distance field on the surface). **open3d** has none either --
its legacy geometry module stops at normals and clustering. Neither appears in this module.

Setup, full and amortized
-------------------------
All three libraries split this computation into mesh-dependent setup (operator assembly, and for
the references a factorization) and a per-source solve, and both references advertise that split:
"repeated solves are fast after initial setup". The ``heat_geodesic`` group reports both points:

* ``setup=full`` -- the setup is **inside** the timed callable, which is what a caller computing
  one field pays. Timing only ``heat_geodesics_solve`` here would compare a back-substitution
  against a full iterative solve.
* ``setup=amortized`` -- the setup is hoisted out, so only the solve is timed. On triwarp's side
  that is a real API path (``heat_operators`` passed back through
  ``heat_geodesic(..., operators=...)``), not a benchmark-only shortcut, so the rows compare like
  with like.

The other groups report ``full`` only.

Sources
-------
A single source vertex (index ``0``) for every case. The heat method's cost is essentially
independent of the number of sources -- they only change the right-hand side, not the matrix or
the iteration structure -- so one source keeps the comparison simple and matches how
``igl::heat_geodesics`` is normally driven.
"""

from __future__ import annotations

import igl
import numpy as np
import potpourri3d as pp3d
import pymeshlab as ml
import pytest
import warp as wp
from conftest import BenchCase
from meshlib import mrmeshpy as mm

import triwarp as tw

_SOURCES = np.array([0], dtype=np.int32)

# Two float64 CG solves plus a direct Cholesky on the reference side; both run into hundreds of
# milliseconds at the top of the scale axis.
_ROUNDS = 3

_sources_cache: dict[str, wp.array[wp.int32]] = {}


def _sources_wp(bench_case: BenchCase) -> wp.array[wp.int32]:
    """Return the single-source index buffer on this case's device."""
    key = str(bench_case.device)
    if key not in _sources_cache:
        _sources_cache[key] = wp.array(_SOURCES, dtype=wp.int32, device=bench_case.device)
    return _sources_cache[key]


def _seeded_meshset_pml(bench_case: BenchCase) -> ml.MeshSet:
    """Build a fresh MeshSet with vertex 0 selected -- the single source the other libraries get."""
    meshset_pml = bench_case.new_meshset_pml()
    meshset_pml.compute_selection_by_condition_per_vertex(condselect="(vi == 0)")
    return meshset_pml


def _run_case_pml(bench_case: BenchCase, *, amortized: bool) -> None:
    """
    Time MeshLab's heat method, with its factorization cache either cold or warm.

    The filter caches its factorization on the mesh, so ``amortized=True`` is a *warm* MeshSet --
    the cache primed by one untimed call outside -- and ``amortized=False`` rebuilds the MeshSet per
    round so every round pays the factorization. That is the same distinction the triwarp and
    reference rows draw, expressed in the one API where it needs no special path.
    """
    if amortized:
        warm_pml = _seeded_meshset_pml(bench_case)
        warm_pml.compute_scalar_by_heat_geodesic_distance_from_selection_per_vertex()
        bench_case.run(
            warm_pml.compute_scalar_by_heat_geodesic_distance_from_selection_per_vertex,
            rounds=_ROUNDS,
        )
        assert warm_pml.current_mesh().vertex_scalar_array().shape == (bench_case.n_vertices,)
        return

    def solve_pml() -> None:
        _seeded_meshset_pml(
            bench_case
        ).compute_scalar_by_heat_geodesic_distance_from_selection_per_vertex()

    bench_case.run(solve_pml, rounds=_ROUNDS)


def _run_case(bench_case: BenchCase, *, amortized: bool = False) -> None:
    """
    Time one heat-geodesic field from vertex 0, in triwarp, libigl or potpourri3d.

    With ``amortized=False`` the mesh-dependent setup is inside the timed callable for all three
    libraries; with ``amortized=True`` it is hoisted out and only the solve is timed. See the module
    docstring for why both are reported.
    """
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "pyvista":
        # VTK's geodesic is Dijkstra over mesh *edges*, so it answers a different (and exactly
        # solvable) question -- an upper bound on the geodesic rather than the heat method's
        # smoothed approximation of it, which tests/test_heat_distance.py asserts. It has nothing
        # to amortize: the path search is the whole call.
        if amortized:
            pytest.skip("Dijkstra has no reusable factorization to hoist out of the timed region")
        # ``skip_larger_than`` is a no-op on the synthetic feature meshes, so the cap is by name,
        # as igl's is in the exact-geodesic row below.
        if bench_case.mesh_name == "sphere_large":
            pytest.skip("vtkDijkstra is a serial priority-queue walk: capped at sphere_med")
        mesh_pv = bench_case.mesh_pv
        target = n_vertices - 1
        distance_pv = bench_case.run(
            lambda: float(mesh_pv.geodesic_distance(0, target)), rounds=_ROUNDS
        )
        assert np.isfinite(distance_pv)
        return
    if bench_case.kind == "pymeshlab":
        _run_case_pml(bench_case, amortized=amortized)
        return
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        sources = _sources_wp(bench_case)
        operators = tw.heat.distance.heat_operators(vertices, faces) if amortized else None
        distance = bench_case.run(
            lambda: tw.heat.distance.heat_geodesic(vertices, faces, sources, operators=operators),
            rounds=_ROUNDS,
        )
        assert distance.shape == (n_vertices,)
        return

    vertices_np = bench_case.vertices_np
    faces_np = np.ascontiguousarray(bench_case.faces_np, dtype=np.int32)

    if bench_case.kind == "potpourri3d":
        # ``use_robust=False`` matches triwarp's discretization: potpourri3d otherwise mollifies and
        # flips to an intrinsic Delaunay triangulation first (that path arrives with the plan's P5).
        if amortized:
            solver = pp3d.MeshHeatMethodDistanceSolver(vertices_np, faces_np, use_robust=False)
            solve_pp = lambda: solver.compute_distance(0)  # noqa: E731
        else:

            def solve_pp() -> np.ndarray:
                solver = pp3d.MeshHeatMethodDistanceSolver(vertices_np, faces_np, use_robust=False)
                return np.asarray(solver.compute_distance(0))

        assert np.asarray(bench_case.run(solve_pp, rounds=_ROUNDS)).shape == (n_vertices,)
        return

    if amortized:
        data = igl.HeatGeodesicsData()
        igl.heat_geodesics_precompute(vertices_np, faces_np, data)
        solve_igl = lambda: np.asarray(igl.heat_geodesics_solve(data, _SOURCES))  # noqa: E731
    else:

        def solve_igl() -> np.ndarray:
            data = igl.HeatGeodesicsData()
            igl.heat_geodesics_precompute(vertices_np, faces_np, data)
            return np.asarray(igl.heat_geodesics_solve(data, _SOURCES))

    assert bench_case.run(solve_igl, rounds=_ROUNDS).shape == (n_vertices,)


@pytest.mark.benchmark(group="heat_geodesic")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "igl", "potpourri3d", "pymeshlab", "pyvista")
@pytest.mark.parametrize("setup", ["full", "amortized"])
def test_heat_geodesic(bench_case: BenchCase, setup: str) -> None:
    """
    Two float64 CG solves plus the gradient normalization, over the clean size sweep.

    pyvista's row is the odd one out and is here as the bound rather than as a race: VTK computes
    the exact shortest path along *edges* to **one** target where every other row returns the whole
    field, so it does far less work and answers a different question. It is the class-C oracle for
    this group (``tests/test_heat_distance.py``), which is why it is timed at all.
    """
    _run_case(bench_case, amortized=setup == "amortized")


@pytest.mark.benchmark(group="heat_geodesic_conditioning")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("triwarp", "igl", "potpourri3d", "pymeshlab")
def test_heat_geodesic_conditioning(bench_case: BenchCase) -> None:
    """The same field on the same connectivity, well- and ill-conditioned: 3.4x for triwarp."""
    _run_case(bench_case)


_mesh_ml_cache: dict[str, mm.Mesh] = {}


def _mesh_ml(bench_case: BenchCase) -> mm.Mesh:
    """Cache one ``meshlib.Mesh`` per mesh: the distance call reads it and returns a field."""
    if bench_case.mesh_name not in _mesh_ml_cache:
        _mesh_ml_cache[bench_case.mesh_name] = bench_case.new_mesh_ml()
    return _mesh_ml_cache[bench_case.mesh_name]


@pytest.mark.benchmark(group="fast_marching_distance")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("potpourri3d", "pymeshlab", "igl", "meshlib")
def test_fast_marching_distance(bench_case: BenchCase) -> None:
    """
    The serial single-source geodesics, for scale against the heat solvers on the same meshes.

    triwarp deliberately has no equivalent -- fast marching advances a priority queue one vertex at
    a time and has no parallel formulation -- so it has no row here. It is here to price that
    decision: the non-PDE alternatives for the same task, on the same axis and the same meshes, so
    the numbers can be read next to the ``heat_geodesic`` table. This group therefore contributes no
    ``parity`` pairs by construction (see ``tests/parity.py``): "triwarp agrees" is not a statement
    about a row triwarp does not have.

    Four rows, four serial fronts: potpourri3d's fast marching solves the local Eikonal update per
    triangle, MeshLab's ``compute_scalar_by_geodesic_distance_from_given_point_per_vertex`` advances
    a Dijkstra-style front over the edge graph, **libigl's ``exact_geodesic``** propagates the MMP
    exact windows -- the only one that is exact rather than first-order -- and **meshlib's
    ``computeSurfaceDistances``** is a fourth Eikonal front, the only one of the four that is
    multi-threaded. Its accuracy is measured against triwarp's heat method and against the exact
    great-circle field in ``tests/test_heat_distance.py``: 1.4 % worst deviation against triwarp's
    1.6 %, so this row is a like-for-like cost for a like-for-like answer.

    **igl is capped at ``sphere_small`` with ``rounds=1``, and the numbers say why.** Measured on
    icospheres at one source with every vertex as a target: **57 ms at 2 562 vertices, 839 ms at
    10 242, 20.9 s at 40 962** -- roughly 15-25x per 4x step, so ``sphere_med`` alone would cost
    ~21 s a round and ``sphere_large`` minutes. Window propagation is the price of exactness, and
    that slope is the most useful thing this row records.

    Its call is also the one signature trap in the module: ``exact_geodesic(V, F, VS, FS, VT, FT)``
    needs **all six** arguments. A four-argument call binds ``vt`` to ``FS`` and returns an *empty
    array* rather than raising, so the two face arrays are passed explicitly empty.
    """
    if bench_case.kind == "meshlib":
        # ``startVertices`` is a VertBitSet over the vertex domain -- there is no index-list
        # overload -- and it is the input, so it is built outside the timed callable. The mesh is
        # read-only here and cached; ``maxVertUpdates`` stays at its default of 3, which is the
        # accuracy setting the correctness comparison was measured at.
        mesh_ml = _mesh_ml(bench_case)
        starts_ml = mm.VertBitSet()
        starts_ml.resize(mesh_ml.points.size(), False)
        starts_ml.set(mm.VertId(int(_SOURCES[0])), True)
        distances_ml = bench_case.run(
            lambda: mm.computeSurfaceDistances(mesh_ml, starts_ml), rounds=_ROUNDS
        )
        assert distances_ml.size() == bench_case.n_vertices
        return
    if bench_case.kind == "igl":
        # ``skip_larger_than`` is a no-op on the synthetic feature meshes (they are not on the size
        # ladder), so the cap is by name.
        if bench_case.mesh_name != "sphere_small":
            pytest.skip("MMP window propagation: 839 ms at 10k vertices, 20.9 s at 41k")
        vertices_np = bench_case.vertices_np
        faces_np = np.ascontiguousarray(bench_case.faces_np, dtype=np.int64)
        sources_np = np.array([int(_SOURCES[0])], dtype=np.int64)
        targets_np = np.arange(bench_case.n_vertices, dtype=np.int64)
        no_faces_np = np.array([], dtype=np.int64)
        distance_igl = bench_case.run(
            lambda: igl.exact_geodesic(
                vertices_np, faces_np, sources_np, no_faces_np, targets_np, no_faces_np
            ),
            rounds=1,
        )
        assert distance_igl.shape == (bench_case.n_vertices,)
        return
    if bench_case.kind == "pymeshlab":
        # ``maxdistance=PureValue(0)`` disables the cut-off, so the front covers the whole mesh --
        # the default 50% of the bbox diagonal would stop early and measure less work.
        start_np = np.ascontiguousarray(bench_case.vertices_np[int(_SOURCES[0])], dtype=np.float64)
        bench_case.run(
            lambda: (
                bench_case.new_meshset_pml()
            ).compute_scalar_by_geodesic_distance_from_given_point_per_vertex(
                startpoint=start_np, maxdistance=ml.PureValue(0.0)
            ),
            rounds=_ROUNDS,
        )
        return
    vertices_np = bench_case.vertices_np
    faces_np = np.ascontiguousarray(bench_case.faces_np, dtype=np.int32)
    # One source vertex, as a single-point "curve" of barycentric points.
    sources_pp = [[(int(_SOURCES[0]), [])]]

    def solve_pp() -> np.ndarray:
        solver = pp3d.MeshFastMarchingDistanceSolver(vertices_np, faces_np)
        return np.asarray(solver.compute_distance(sources_pp))

    assert bench_case.run(solve_pp, rounds=_ROUNDS).shape == (bench_case.n_vertices,)
