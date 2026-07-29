"""
Benchmarks for ``triwarp.curvature``.

Axis: **scale**, plus a **radius** sweep on every group. Both matter, and the second one matters
more: all three functions gather a neighborhood before they compute anything, so their cost is
``V * k_bar`` where ``k_bar`` is the average neighborhood size, and ``k_bar`` grows with the
*square* of the radius on a surface. Doubling the radius is roughly four times the work at an
unchanged vertex count, which is why the radius is swept explicitly rather than pinned to one
value that would hide the exponent.

Three functions, three cost profiles:

* ``principal_curvature`` -- a per-vertex 5x5 float64 least-squares quadric fit over a geodesic-ball
  neighborhood. The neighborhood collection ([`geodesic_ball`][triwarp.neighbors.geodesic_ball],
  timed on its own in [`test_proximity.py`](test_proximity.py)) is part of the call, so the number
  here is "ball + fit"; the fit is what a change to the 5x5 solver moves.
* ``discrete_gaussian_curvature`` -- a hash-grid ball query plus a segmented scatter-sum of vertex
  defects. Cheap, memory-bound, and dominated by the neighbor query.
* ``discrete_mean_curvature`` -- the same query shape over *edges* rather than vertices, so it also
  pays face adjacency and the per-edge dihedral angles.

Why the clean spheres rather than the scan meshes
-------------------------------------------------
This module used to run ``principal_curvature`` on the scan registry for triwarp and draw the
libigl comparison separately on two small saddle patches, because ``igl.principal_curvature``
**segfaults** on every scan mesh -- they have non-manifold vertices and libigl's vertex-ring walk
assumes manifoldness, so it is a hard crash that takes the pytest process with it rather than an
exception that can be caught.

The ``scale`` axis is manifold by construction, so that restriction is gone: libigl runs on all
three points and the comparison is drawn on the same meshes triwarp is measured on. Measured on
``sphere_med`` at radius 5, ``igl.principal_curvature`` takes 0.48 s.

**Not** on the ``valence`` axis, though. On ``fan_hub`` the same call takes **110 seconds** --
libigl's per-vertex ring walk is quadratic in valence, and a 40 960-valence apex is exactly the
input it cannot survive. The guard below is explicit rather than implied by the axis, because the
failure mode is a wall-clock blowout that looks like a hang.

References
----------
**libigl**: ``igl.principal_curvature(..., useKring=False)`` is the sphere-neighborhood variant
triwarp reproduces (the ``useKring=True`` default collects a combinatorial k-ring instead, a
different neighborhood and therefore a different amount of work), timed with the same radius
multiplier.

**trimesh** is the reference for the two Cohen-Steiner / Morvan measures --
``discrete_gaussian_curvature_measure`` and ``discrete_mean_curvature_measure``, given the same
query points and the same radius.

**open3d** has no curvature estimation at all: no quadric fit, and no normal-cycle curvature
measure. Vertex angle defects are also not exposed (``open3d.geometry`` stops at normals and
areas), so there is nothing to compare against for any of the three.

What is inside the timed callable
---------------------------------
Everything the public function does. For the Gaussian measure that includes the vertex-defect
recomputation, so the trimesh reference builds its ``tm.Trimesh`` inside the timed region too --
otherwise the cached ``vertex_defects`` / ``face_adjacency`` / ``kdtree`` properties would make
rounds 2..n measure only the ball query. The precomputed inputs are the ones the triwarp signature
asks the caller for: the ``(n_faces, 3)`` face-angle table for the Gaussian measure, and the query
points (the mesh's own vertices) for both measures.
"""

from __future__ import annotations

import igl
import numpy as np
import pytest
import trimesh as tm
from conftest import BenchCase

import triwarp as tw
import triwarp.typing as twt

# Sphere radius of the quadric fit, as a multiple of the average edge length. libigl defaults to 5;
# the pair brackets it so the quadratic growth in neighborhood size is visible.
_QUADRIC_RADII = [3, 8]

# Radius of the normal-cycle curvature measures, as a multiple of the mean edge length.
_MEASURE_RADII = [2.0, 4.0]

# The quadric fit runs into hundreds of milliseconds a call at the wide radius; ten rounds of that
# on three meshes and two libraries would dominate the suite.
_HEAVY_ROUNDS = 3

_face_angles_cache: dict[tuple[str, str], twt.Array2dFloat32] = {}


def _face_angles(bench_case: BenchCase) -> twt.Array2dFloat32:
    """``(n_faces, 3)`` interior angles -- an *input* of ``discrete_gaussian_curvature``."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _face_angles_cache:
        _face_angles_cache[key] = tw.triangles.face_angles(
            bench_case.vertices_wp, bench_case.faces_wp
        )
    return _face_angles_cache[key]


@pytest.mark.benchmark(group="principal_curvature")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "igl")
@pytest.mark.parametrize("radius", _QUADRIC_RADII)
def test_principal_curvature(bench_case: BenchCase, radius: int) -> None:
    """Per-vertex quadric fit over a geodesic ball: the 5x5 solve in bulk, at two radii."""
    if bench_case.kind == "igl" and bench_case.mesh_name == "sphere_large":
        pytest.skip("igl.principal_curvature is ~2 s a call at this size; capped at sphere_med")
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        *_, pv1, _pv2 = bench_case.run(
            lambda: tw.curvature.principal_curvature(vertices, faces, radius=radius),
            rounds=_HEAVY_ROUNDS,
        )
        assert pv1.shape == (n_vertices,)
    else:
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        *_, pv1_igl, _pv2, _bad = bench_case.run(
            lambda: igl.principal_curvature(vertices_np, faces_np, radius=radius, useKring=False),
            rounds=_HEAVY_ROUNDS,
        )
        assert pv1_igl.shape == (n_vertices,)


@pytest.mark.benchmark(group="discrete_gaussian_curvature")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "trimesh")
@pytest.mark.parametrize("radius_scale", _MEASURE_RADII)
def test_discrete_gaussian_curvature(bench_case: BenchCase, radius_scale: float) -> None:
    """Summed vertex defects inside a ball around every vertex (Cohen-Steiner / Morvan)."""
    n_vertices = bench_case.n_vertices
    radius = radius_scale * bench_case.mean_edge
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        face_angles = _face_angles(bench_case)
        curvature = bench_case.run(
            lambda: tw.curvature.discrete_gaussian_curvature(
                vertices, vertices, faces, face_angles, radius
            )
        )
        assert curvature.shape == (n_vertices,)
    else:  # rebuild inside: vertex_defects and kdtree are cached Trimesh properties
        if bench_case.mesh_name != "sphere_small":
            # Measured: 40 s a call on ``sphere_med`` -- one cKDTree ball query per point over
            # 40 962 points. Two rows of that were 95% of this module's wall clock, for a ratio the
            # 2 562-point mesh already establishes at three orders of magnitude.
            pytest.skip("trimesh queries one cKDTree ball per point; capped at sphere_small")
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        curvature_tm = bench_case.run(
            lambda: tm.curvature.discrete_gaussian_curvature_measure(
                tm.Trimesh(vertices_np, faces_np, process=False), vertices_np, radius
            ),
            rounds=_HEAVY_ROUNDS,
        )
        assert curvature_tm.shape == (n_vertices,)


@pytest.mark.benchmark(group="discrete_mean_curvature")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "trimesh")
@pytest.mark.parametrize("radius_scale", _MEASURE_RADII)
def test_discrete_mean_curvature(bench_case: BenchCase, radius_scale: float) -> None:
    """Summed edge dihedral angles inside a ball around every vertex: adjacency plus the query."""
    n_vertices = bench_case.n_vertices
    radius = radius_scale * bench_case.mean_edge
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        curvature = bench_case.run(
            lambda: tw.curvature.discrete_mean_curvature(vertices, vertices, faces, radius)
        )
        assert curvature.shape == (n_vertices,)
    else:  # rebuild inside: face_adjacency and kdtree are cached Trimesh properties
        if bench_case.mesh_name != "sphere_small":
            # Measured: 40 s a call on ``sphere_med`` -- one cKDTree ball query per point over
            # 40 962 points. Two rows of that were 95% of this module's wall clock, for a ratio the
            # 2 562-point mesh already establishes at three orders of magnitude.
            pytest.skip("trimesh queries one cKDTree ball per point; capped at sphere_small")
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        curvature_tm = bench_case.run(
            lambda: tm.curvature.discrete_mean_curvature_measure(
                tm.Trimesh(vertices_np, faces_np, process=False), vertices_np, radius
            ),
            rounds=_HEAVY_ROUNDS,
        )
        assert np.asarray(curvature_tm).shape == (n_vertices,)
