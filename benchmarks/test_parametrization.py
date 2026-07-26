"""
Benchmarks for ``triwarp.parametrization``.

Times the four solver entry points against their libigl references. All of them need a mesh with a
boundary loop to pin, so the mesh set is chosen explicitly: the ``synthetic_saddle`` patches
(regular grids lifted onto a saddle -- **disk topology**, the domain these solvers are for) plus
``bunny_decimated`` / ``bunny`` as the realistic scanned inputs with small hole loops.

``synthetic_cylinder`` is deliberately **not** used here. It is an annulus, and pinning only its
longer rim leaves ARAP free to fold: triwarp returns 32 767 flipped faces out of 131 072, disagrees
with libigl by 0.35 regardless of CG tolerance (the two land on different local minima of a
non-convex energy), and burns 8 200 CG iterations per solve on the resulting near-singular system.
Timing that measures a pathology, not the algorithm. On the saddle patches triwarp instead agrees
with libigl to ~1e-7 with zero flipped faces.

Setup that is *not* part of the measured operation is precomputed and cached: the boundary loop, its
circle map, and the harmonic warm start ARAP iterates from. What remains inside the timed callable
is what the function itself does — operator assembly plus the conjugate-gradient solve — because
that is what the batched-CG work targets.

``harmonic`` / ``lscm`` / ``arap`` solve with ``warp.optim.linear.cg``, which returns NaN on the
Warp CPU backend (1.14-1.15), so the ``triwarp-cpu`` variant is skipped rather than timed.

**open3d** has no mesh parametrization at all — no harmonic map, no LSCM, no ARAP, and no boundary
circle map — so libigl remains the only reference for this module.

The libigl reference is gated on mesh *conditioning*, not size — measured, not assumed. libigl goes
through a direct LDLT factorization of the cotangent system, which fails outright on the scanned
registry meshes (``RuntimeError: Failed to compute harmonic map`` / ``igl::lscm failed``): they are
not disk topology (``bunny`` has five hole loops, the longest only 80 vertices) and their cotangent
Laplacian is not positive definite on the free set. triwarp's CG converges on exactly the same
input, so those meshes are timed for triwarp only. The comparison is drawn on the saddle patches,
where both solvers are well-conditioned and agree to ~1e-7.
"""

from __future__ import annotations

import igl
import numpy as np
import pytest
import warp as wp
from conftest import BenchCase

import triwarp as tw

_ARAP_ITERATIONS = 10

# Meshes whose cotangent system libigl's direct LDLT can actually factor (see module docstring).
# The scanned registry meshes are not disk topology and make every igl solver here raise.
_IGL_SOLVABLE_MESHES = frozenset({"synthetic_saddle_small", "synthetic_saddle"})

_boundary_cache: dict[tuple[str, str], tuple[wp.array[wp.int32], wp.array[wp.vec2]]] = {}
_warm_start_cache: dict[tuple[str, str], wp.array[wp.vec2]] = {}


def _skip_unsupported(bench_case: BenchCase) -> None:
    """Skip triwarp-on-CPU cases and libigl cases whose cotangent system it cannot factor."""
    if bench_case.kind == "triwarp":
        assert bench_case.device is not None
        if wp.get_device(bench_case.device).is_cpu:
            pytest.skip("warp.optim.linear.cg returns NaN on the CPU device in Warp 1.14-1.15")
    elif bench_case.mesh_name not in _IGL_SOLVABLE_MESHES:
        pytest.skip(f"libigl's direct solver fails on {bench_case.mesh_name} (not disk topology)")


def _boundary(bench_case: BenchCase) -> tuple[wp.array[wp.int32], wp.array[wp.vec2]]:
    """Longest boundary loop and its unit-circle UV, cached per ``(mesh, device)``."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _boundary_cache:
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        loop = tw.boundary.boundary_loop(vertices, faces)
        _boundary_cache[key] = (loop, tw.parametrization.map_vertices_to_circle(vertices, loop))
    return _boundary_cache[key]


def _warm_start(bench_case: BenchCase) -> wp.array[wp.vec2]:
    """Harmonic UV used as the ARAP initial guess, cached per ``(mesh, device)``."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _warm_start_cache:
        loop, loop_uv = _boundary(bench_case)
        _warm_start_cache[key] = tw.parametrization.harmonic(
            bench_case.vertices_wp, bench_case.faces_wp, loop, loop_uv
        )
    return _warm_start_cache[key]


@pytest.mark.benchmark(group="map_vertices_to_circle")
@pytest.mark.benchlibs("triwarp", "igl")
@pytest.mark.benchmeshes("synthetic_saddle_small", "synthetic_saddle", "bunny_decimated", "bunny")
def test_map_vertices_to_circle(bench_case: BenchCase) -> None:
    if bench_case.kind == "triwarp":
        vertices = bench_case.vertices_wp
        loop, _loop_uv = _boundary(bench_case)
        circle = bench_case.run(lambda: tw.parametrization.map_vertices_to_circle(vertices, loop))
        assert int(circle.shape[0]) == int(loop.shape[0])
    else:
        vertices_np = bench_case.vertices_np
        loop_np = igl.boundary_loop(bench_case.faces_np.astype(np.int64))
        circle_igl = bench_case.run(lambda: igl.map_vertices_to_circle(vertices_np, loop_np))
        assert circle_igl.shape[0] == loop_np.shape[0]


@pytest.mark.benchmark(group="harmonic")
@pytest.mark.benchlibs("triwarp", "igl")
@pytest.mark.benchmeshes("synthetic_saddle_small", "synthetic_saddle", "bunny_decimated", "bunny")
def test_harmonic(bench_case: BenchCase) -> None:
    _skip_unsupported(bench_case)
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        loop, loop_uv = _boundary(bench_case)
        uv = bench_case.run(lambda: tw.parametrization.harmonic(vertices, faces, loop, loop_uv))
        assert int(uv.shape[0]) == int(vertices.shape[0])
    else:
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np.astype(np.int64)
        loop_np = igl.boundary_loop(faces_np)
        circle_np = igl.map_vertices_to_circle(vertices_np, loop_np)
        uv_igl = bench_case.run(lambda: igl.harmonic(vertices_np, faces_np, loop_np, circle_np, 1))
        assert uv_igl.shape[0] == vertices_np.shape[0]


@pytest.mark.benchmark(group="lscm")
@pytest.mark.benchlibs("triwarp", "igl")
@pytest.mark.benchmeshes("synthetic_saddle_small", "synthetic_saddle", "bunny_decimated", "bunny")
def test_lscm(bench_case: BenchCase) -> None:
    _skip_unsupported(bench_case)
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        loop, _loop_uv = _boundary(bench_case)
        # LSCM needs only enough pins to kill the similarity freedom: two opposite loop vertices.
        loop_np = loop.numpy()
        pins_np = np.array([loop_np[0], loop_np[len(loop_np) // 2]], dtype=np.int32)
        pins = wp.array(pins_np, dtype=wp.int32, device=bench_case.device)
        pins_uv = wp.array(
            np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            dtype=wp.vec2,
            device=bench_case.device,
        )
        uv = bench_case.run(lambda: tw.parametrization.lscm(vertices, faces, pins, pins_uv))
        assert int(uv.shape[0]) == int(vertices.shape[0])
    else:
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np.astype(np.int64)
        loop_np = igl.boundary_loop(faces_np)
        pins_np = np.array([loop_np[0], loop_np[len(loop_np) // 2]], dtype=np.int64)
        pins_uv_np = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        uv_igl, _hessian = bench_case.run(
            lambda: igl.lscm(vertices_np, faces_np, pins_np, pins_uv_np)
        )
        assert uv_igl.shape[0] == vertices_np.shape[0]


@pytest.mark.benchmark(group="arap")
@pytest.mark.benchlibs("triwarp", "igl")
@pytest.mark.benchmeshes("synthetic_saddle_small", "synthetic_saddle", "bunny_decimated", "bunny")
def test_arap(bench_case: BenchCase) -> None:
    _skip_unsupported(bench_case)
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        loop, loop_uv = _boundary(bench_case)
        uv_init = _warm_start(bench_case)
        uv = bench_case.run(
            lambda: tw.parametrization.arap(
                vertices, faces, loop, loop_uv, uv_init, max_iterations=_ARAP_ITERATIONS
            )
        )
        assert int(uv.shape[0]) == int(vertices.shape[0])
    else:
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np.astype(np.int64)
        loop_np = igl.boundary_loop(faces_np)
        circle_np = igl.map_vertices_to_circle(vertices_np, loop_np)
        uv_init_np = np.ascontiguousarray(
            igl.harmonic(vertices_np, faces_np, loop_np, circle_np, 1)
        )

        # triwarp's ``arap`` rebuilds its operator on every call, so the igl side includes
        # ``arap_precomputation`` for a like-for-like comparison rather than solve-only.
        def run() -> np.ndarray:
            data = igl.ARAPData()
            data.max_iter = _ARAP_ITERATIONS
            igl.arap_precomputation(vertices_np, faces_np, 2, loop_np.astype(np.int32), data)
            return igl.arap_solve(circle_np, data, uv_init_np)

        uv_igl = bench_case.run(run)
        assert uv_igl.shape[0] == vertices_np.shape[0]
