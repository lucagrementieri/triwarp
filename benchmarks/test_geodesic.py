"""
Benchmarks for ``triwarp.geodesic.heat_geodesic``.

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

``geodesic_ball`` -- the other public function in this module -- is **not** benchmarked here: it is
already timed as ``query_geodesic_ball`` in [`test_proximity.py`](test_proximity.py), next to the
other neighborhood queries it belongs with.

References
----------
**libigl** implements the same method (``igl::heat_geodesics``) and now runs on every mesh in both
axes. It previously could not: on every scan mesh ``igl.heat_geodesics_precompute`` raises
``RuntimeError: heat_geodesics: Precomputation failed.`` -- it factors the cotangent and Poisson
systems directly (Cholesky) at precompute time, and those factorizations fail on the scan meshes.
The synthetic meshes are manifold by construction and it succeeds on all of them, measured at
0.10-0.65 s.

That makes the quality axis a direct iterative-versus-direct comparison. Note it is generous to
igl in one respect and harsh in another: its **precomputation is inside the timed callable**,
because that is where it does the factorization work triwarp's CG does per solve -- timing only
``heat_geodesics_solve`` would compare a back-substitution against a full iterative solve.
Amortized over many source sets igl's split would favour it.

**trimesh** has no geodesic distance of any kind (``trimesh.graph`` offers only combinatorial
traversal over the edge graph, not a distance field on the surface). **open3d** has none either --
its legacy geometry module stops at normals and clustering. Neither appears in this module.

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
import pytest
import warp as wp
from conftest import BenchCase

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


def _run_case(bench_case: BenchCase) -> None:
    """Time one heat-geodesic field from vertex 0, triwarp or libigl."""
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        sources = _sources_wp(bench_case)
        distance = bench_case.run(
            lambda: tw.geodesic.heat_geodesic(vertices, faces, sources), rounds=_ROUNDS
        )
        assert distance.shape == (n_vertices,)
    else:  # precompute is inside the timed region -- see the module docstring
        vertices_np = bench_case.vertices_np
        faces_np = np.ascontiguousarray(bench_case.faces_np, dtype=np.int32)

        def solve_igl() -> np.ndarray:
            data = igl.HeatGeodesicsData()
            igl.heat_geodesics_precompute(vertices_np, faces_np, data)
            return np.asarray(igl.heat_geodesics_solve(data, _SOURCES))

        assert bench_case.run(solve_igl, rounds=_ROUNDS).shape == (n_vertices,)


@pytest.mark.benchmark(group="heat_geodesic")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "igl")
def test_heat_geodesic(bench_case: BenchCase) -> None:
    """Two float64 CG solves plus the gradient normalization, over the clean size sweep."""
    _run_case(bench_case)


@pytest.mark.benchmark(group="heat_geodesic_conditioning")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("triwarp", "igl")
def test_heat_geodesic_conditioning(bench_case: BenchCase) -> None:
    """The same field on the same connectivity, well- and ill-conditioned: 3.4x for triwarp."""
    _run_case(bench_case)
