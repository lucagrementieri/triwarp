"""
Benchmarks for the three heat-diffusion solvers of ``triwarp.heat``.

One module because they are one family and share the property this suite is really measuring: each
is two or three conjugate-gradient solves against a cotangent or connection Laplacian, so each is
sensitive to the *conditioning* of that operator in a way every factorizing reference is not. That
finding turns up three times below -- ``heat_geodesic_conditioning``, ``heat_signed_distance`` on
the quality axis, and ``log_map`` -- and it is the same finding each time.

Geodesic distance (``heat_operators`` / ``heat_geodesic``)
=========================================================
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

Signed distance to curves (``heat_signed_distance``)
====================================================

The axis is the **source curve**, not the mesh: the mesh sets the solve size (fixed here at
``sphere_med``) while the curve sets how much of the surface the source touches. Two points along
it, both built from the mesh itself so the comparison is reproducible:

* ``ring`` -- one vertex's one-ring, six segments. The smallest curve that is closed, edge-connected
  and separating.
* ``band`` -- a full latitude band of one-rings chained together, hundreds of segments.

That contrast asks the question the splat stage raises: is the cost in the *source*, which scales
with the curve, or in the three solves, which do not? On an RTX 5090 the answer is the solves --
**81.0 ms for 6 segments against 71.3 ms for 370** -- the longer curve is if anything *cheaper*,
because a source spread over the surface converges in fewer conjugate-gradient iterations than a
point-like one. The splat itself does not register. potpourri3d runs 1 060 / 1 017 ms over the same
two points, so 13x either way.

The three stages are a vector diffusion on the connection Laplacian (a ``2 x 2``-block CG), a
normalization, and a Poisson solve; ``zero_set`` makes that last one a *constrained* solve through
the machinery behind ``linalg.min_quad_with_fixed``, which is why both level-set modes are timed.
That constraint is not free: **80.6 ms against 17.3 ms** for ``none``, i.e. extracting the free-free
block and solving it costs 4.7x what one unconstrained solve does.

On the ``quality`` axis triwarp goes 40.3 -> 125.8 ms (**3.1x**) while potpourri3d stays at 525 ->
519 ms. Three conjugate-gradient solves feel a bad aspect ratio three times over; a factorization
does not feel it at all. Same finding as ``heat_geodesic_conditioning`` and ``log_map`` -- the third
place in the suite where it turns up.

References
----------
**potpourri3d** is the only reference (``MeshSignedHeatSolver``), and two of its properties shape
its row: the solver is constructed inside the timed callable per this suite's convention (a halfedge
mesh plus two factorizations), and it requires every curve segment to lie within one face, which is
why the curves here are edge paths. **trimesh**, **libigl**, **open3d** and **scipy** have no signed
distance *on a surface* at all — trimesh's ``proximity.signed_distance`` signs against a closed
volume, a different question, already benchmarked in [`test_proximity.py`](test_proximity.py).

Vector transport, scalar extension and the log map
=================================================

The connection Laplacian these run on, and the three solvers built on it.

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
factorization fails there (see ``tests/test_heat_vector.py``). No group here uses that mesh.

**trimesh**, **libigl** and **open3d** have no tangent-space machinery, so nothing else appears
here.
"""

from __future__ import annotations

import itertools

import igl
import numpy as np
import potpourri3d as pp3d
import pymeshlab as ml
import pytest
import warp as wp
from meshlib import mrmeshpy as mm

import triwarp as tw
from conftest import BenchCase, skip_larger_than

# --------------------------------------------------------------------------
# heat_operators / heat_geodesic
# --------------------------------------------------------------------------

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
        operators = tw.heat.heat_operators(vertices, faces) if amortized else None
        distance = bench_case.run(
            lambda: tw.heat.heat_geodesic(vertices, faces, sources, operators=operators),
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


# --------------------------------------------------------------------------
# heat_signed_distance
# --------------------------------------------------------------------------

# Three float64 solves per case on triwarp's side; two factorizations on the reference's.
_ROUNDS = 3

_curve_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}


def _curve(bench_case: BenchCase, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Source curves as a packed vertex buffer plus CSR bounds, cached per (mesh, kind).

    The bounds are not implicit: a one-ring has five or six vertices depending on the centre's
    valence, so slicing a packed buffer at a fixed stride would splice two rings together and give
    segments that are not mesh edges at all -- which the reference rejects outright, and which this
    method would silently splat along a chord.
    """
    key = (bench_case.mesh_name, kind)
    if key not in _curve_cache:
        ring, offsets, is_boundary = (
            array.numpy()
            for array in tw.halfedge.vertex_one_rings(
                bench_case.faces_wp, n_vertices=bench_case.n_vertices
            )
        )
        faces_np = bench_case.faces_np
        interior = np.flatnonzero(~is_boundary)

        def cycle(center: int) -> np.ndarray:
            halfedges = ring[offsets[center] : offsets[center + 1]]
            return np.array([faces_np[h // 3][(h % 3 + 1) % 3] for h in halfedges], dtype=np.int32)

        if kind == "ring":
            cycles = [cycle(int(interior[len(interior) // 2]))]
        else:
            # 64 rings spread over the surface: still edge paths, two orders of magnitude more
            # segments than one ring.
            cycles = [
                cycle(int(center)) for center in interior[:: max(len(interior) // 64, 1)][:64]
            ]
        bounds = np.cumsum([0] + [len(c) for c in cycles]).astype(np.int32)
        _curve_cache[key] = (np.concatenate(cycles), bounds)
    return _curve_cache[key]


@pytest.mark.benchmark(group="heat_signed_distance")
@pytest.mark.benchmeshes("sphere_med")
@pytest.mark.benchlibs("triwarp", "potpourri3d")
@pytest.mark.parametrize("curve_kind", ["ring", "band"])
def test_heat_signed_distance(bench_case: BenchCase, curve_kind: str) -> None:
    """One mesh, a 6-segment curve against a ~370-segment one, to price the source splat."""
    curve_np, bounds_np = _curve(bench_case, curve_kind)
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        curve = wp.array(curve_np, dtype=wp.int32, device=bench_case.device)
        offsets = wp.array(bounds_np, dtype=wp.int32, device=bench_case.device)
        distance = bench_case.run(
            lambda: tw.heat.heat_signed_distance(vertices, faces, curve, offsets), rounds=_ROUNDS
        )
        assert distance.shape == (bench_case.n_vertices,)
    else:
        vertices_np = bench_case.vertices_np
        faces_np = np.ascontiguousarray(bench_case.faces_np, dtype=np.int32)
        curves_pp = [
            [(int(vertex), []) for vertex in curve_np[begin:end]]
            for begin, end in itertools.pairwise(bounds_np)
        ]

        def solve_pp() -> np.ndarray:
            solver = pp3d.MeshSignedHeatSolver(vertices_np, faces_np)
            return np.asarray(solver.compute_distance(curves_pp, level_set_constraint="ZeroSet"))

        assert bench_case.run(solve_pp, rounds=_ROUNDS).shape == (bench_case.n_vertices,)


@pytest.mark.benchmark(group="heat_signed_distance_constraint")
@pytest.mark.benchmeshes("sphere_med")
@pytest.mark.benchlibs("triwarp")
@pytest.mark.parametrize("level_set_constraint", ["zero_set", "none"])
def test_heat_signed_distance_constraint(bench_case: BenchCase, level_set_constraint: str) -> None:
    """Pinning the curve to zero against solving unconstrained and shifting afterwards."""
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    curve = wp.array(_curve(bench_case, "ring")[0], dtype=wp.int32, device=bench_case.device)
    distance = bench_case.run(
        lambda: tw.heat.heat_signed_distance(
            vertices, faces, curve, level_set_constraint=level_set_constraint
        ),
        rounds=_ROUNDS,
    )
    assert distance.shape == (bench_case.n_vertices,)


@pytest.mark.benchmark(group="heat_signed_distance_conditioning")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("triwarp", "potpourri3d")
def test_heat_signed_distance_conditioning(bench_case: BenchCase) -> None:
    """The same curve on well- and ill-conditioned connectivity: three CG solves feel it thrice."""
    test_heat_signed_distance(bench_case, "ring")


# --------------------------------------------------------------------------
# vector_heat_operators / extend_scalar / transport_tangent_vectors / log_map
# --------------------------------------------------------------------------

# Every case is at least two float64 CG solves; the reference also factors two sparse systems.
_ROUNDS = 3


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
            lambda: tw.heat.extend_scalar(vertices, faces, sources, values), rounds=_ROUNDS
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
        operators = tw.heat.vector_heat_operators(vertices, faces) if amortized else None
        transported, resolved = bench_case.run(
            lambda: tw.heat.transport_tangent_vectors(
                vertices, faces, sources, vectors, operators=operators
            ),
            rounds=_ROUNDS,
        )
        assert transported.shape == (n_vertices,)
        assert resolved.shape == (n_vertices,)
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
        logarithm = bench_case.run(lambda: tw.heat.log_map(vertices, faces, 0), rounds=_ROUNDS)
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
