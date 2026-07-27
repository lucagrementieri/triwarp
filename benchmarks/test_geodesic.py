"""
Benchmarks for ``triwarp.geodesic.heat_geodesic``.

The heat method (Crane et al.) is three stages, and the timing is dominated by the two of them
that are sparse linear solves: diffuse heat from the sources for a short time ``t``, normalize the
resulting gradient into a unit field pointing away from the sources, then integrate that field
back into a distance function with a Poisson solve. Both systems are symmetric positive
(semi-)definite and are solved on-device with conjugate gradient, so what this benchmark actually
measures is CG iteration count times sparse-matvec cost — the per-face gradient normalization in
between is a single cheap pass.

The whole computation runs in ``float64`` (the diffused heat decays exponentially and underflows
``float32``, collapsing the far field), which on a consumer GPU means the solves run at the
device's much lower double-precision rate. That is inherent to the method, not a tuning choice.

``geodesic_ball`` — the other public function in this module — is **not** benchmarked here: it is
already timed as ``query_geodesic_ball`` in [`test_proximity.py`](test_proximity.py), next to the
other neighborhood queries it belongs with.

References
----------
**libigl** implements the same method (``igl::heat_geodesics``), but only usably on the *synthetic*
meshes. On every registry scan mesh ``igl.heat_geodesics_precompute`` raises
``RuntimeError: heat_geodesics: Precomputation failed.`` — it factors the cotangent and Poisson
systems directly (Cholesky) at precompute time and those factorizations fail on the scan meshes, the
same failure mode [`test_parametrization.py`](test_parametrization.py) documents for igl's LSCM and
harmonic solvers. triwarp's CG needs no factorization, which is why it has no equivalent cap.

So the igl comparison is drawn in a separate group on the saddle patches (regular, manifold grids),
exactly as [`test_curvature.py`](test_curvature.py) does for ``principal_curvature``. Note the
comparison is generous to igl in one respect and harsh in another: its **precomputation is inside
the timed callable**, because that is where it does the factorization work that triwarp's CG does
per solve — timing only ``heat_geodesics_solve`` would compare a back-substitution against a full
iterative solve. Amortized over many source sets igl's split would favour it.

**trimesh** has no geodesic distance of any kind (``trimesh.graph`` offers only combinatorial
traversal over the edge graph, not a distance field on the surface). **open3d** has none either —
its legacy geometry module stops at normals and clustering. Neither appears in this module.

Sources
-------
A single source vertex (index ``0``) for every case. The heat method's cost is essentially
independent of the number of sources — they only change the right-hand side, not the matrix or the
iteration structure — so one source keeps the comparison simple and matches how
``igl::heat_geodesics`` is normally driven.
"""

from __future__ import annotations

import igl
import numpy as np
import pytest
import warp as wp
from conftest import BenchCase, skip_larger_than

import triwarp as tw

_SOURCES = np.array([0], dtype=np.int32)

_sources_cache: dict[str, wp.array[wp.int32]] = {}


def _sources_wp(bench_case: BenchCase) -> wp.array[wp.int32]:
    """Return the single-source index buffer on this case's device."""
    key = str(bench_case.device)
    if key not in _sources_cache:
        _sources_cache[key] = wp.array(_SOURCES, dtype=wp.int32, device=bench_case.device)
    return _sources_cache[key]


@pytest.mark.benchmark(group="heat_geodesic")
@pytest.mark.benchlibs("triwarp")
def test_heat_geodesic(bench_case: BenchCase) -> None:
    """Two float64 CG solves plus the gradient normalization, from one source vertex."""
    skip_larger_than(bench_case, "happy_buddha", "two float64 CG solves over the whole mesh")
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    sources = _sources_wp(bench_case)
    distance = bench_case.run(lambda: tw.geodesic.heat_geodesic(vertices, faces, sources))
    assert distance.shape == (bench_case.n_vertices,)


@pytest.mark.benchmark(group="heat_geodesic_saddle")
@pytest.mark.benchmeshes("synthetic_saddle_small", "synthetic_saddle")
@pytest.mark.benchlibs("triwarp", "igl")
def test_heat_geodesic_saddle(bench_case: BenchCase) -> None:
    """The same field against ``igl::heat_geodesics`` on the manifold saddle patches."""
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        sources = _sources_wp(bench_case)
        distance = bench_case.run(lambda: tw.geodesic.heat_geodesic(vertices, faces, sources))
        assert distance.shape == (n_vertices,)
    else:  # precompute is inside the timed region -- see the module docstring
        vertices_np = bench_case.vertices_np
        faces_np = np.ascontiguousarray(bench_case.faces_np, dtype=np.int32)

        def solve_igl() -> np.ndarray:
            data = igl.HeatGeodesicsData()
            igl.heat_geodesics_precompute(vertices_np, faces_np, data)
            return np.asarray(igl.heat_geodesics_solve(data, _SOURCES))

        distance_igl = bench_case.run(solve_igl)
        assert distance_igl.shape == (n_vertices,)
