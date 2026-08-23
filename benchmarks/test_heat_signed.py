"""
Benchmarks for ``triwarp.heat.signed.heat_signed_distance``.

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
"""

from __future__ import annotations

import itertools

import numpy as np
import potpourri3d as pp3d
import pytest
import warp as wp

import triwarp as tw
from conftest import BenchCase

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
        offsets, ring, is_boundary = (
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
            lambda: tw.heat.signed.heat_signed_distance(vertices, faces, curve, offsets),
            rounds=_ROUNDS,
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
        lambda: tw.heat.signed.heat_signed_distance(
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
