"""
Benchmarks for ``triwarp.curvature``.

Three functions, three very different cost profiles:

* ``principal_curvature`` — a per-vertex 5x5 float64 least-squares quadric fit over a geodesic-ball
  neighborhood. The neighborhood collection ([`geodesic_ball`][triwarp.geodesic.geodesic_ball],
  timed on its own in [`test_proximity.py`](test_proximity.py)) is part of the call, so the number
  here is "ball + fit"; the fit is what a change to the 5x5 solver moves.
* ``discrete_gaussian_curvature`` — a hash-grid ball query plus a segmented scatter-sum of vertex
  defects. Cheap, memory-bound, and dominated by the neighbor query.
* ``discrete_mean_curvature`` — the same query shape over *edges* rather than vertices, so it also
  pays face adjacency and the per-edge dihedral angles.

References
----------
**libigl** is the reference for ``principal_curvature``: ``igl.principal_curvature(..., useKring
=False)`` is the sphere-neighborhood variant triwarp reproduces (the ``useKring=True`` default
collects a combinatorial k-ring instead, a different neighborhood and therefore a different amount
of work). It is timed with the same ``radius=5`` multiplier — but **only on the synthetic saddle
patches**, in a separate group. On every registry scan mesh libigl **segfaults**, taking the whole
pytest process with it: ``igl.is_vertex_manifold`` reports non-manifold vertices on all of them
(``bunny_decimated`` included, and compacting away its 25 unreferenced vertices does not help), and
libigl's vertex-triangle ring walk assumes manifoldness. It is a hard crash, not an exception, so it
cannot be caught and retried — the only safe option is to keep igl out of the registry-mesh group.
The saddle patches are regular grids and vertex-manifold, so the comparison is drawn there.

**trimesh** is the reference for the two Cohen-Steiner / Morvan measures —
``discrete_gaussian_curvature_measure`` and ``discrete_mean_curvature_measure``, given the same
query points and the same radius.

**open3d** has no curvature estimation at all: no quadric fit, and no normal-cycle curvature
measure. Vertex angle defects are also not exposed (``open3d.geometry`` stops at normals and
areas), so there is nothing to compare against for any of the three.

Caps
----
* triwarp's ``principal_curvature`` is capped at ``happy_buddha``, the same cap
  [`geodesic_ball`][triwarp.geodesic.geodesic_ball] uses in
  [`test_proximity.py`](test_proximity.py) — the per-source BFS scratch scales with the vertex
  count.
* trimesh's measures issue a per-point ``cKDTree.query_ball_point`` and are capped at ``bunny``.

What is inside the timed callable
--------------------------------
Everything the public function does. For the Gaussian measure that includes the vertex-defect
recomputation, so the trimesh reference builds its ``tm.Trimesh`` inside the timed region too —
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
from conftest import BenchCase, skip_larger_than

import triwarp as tw
import triwarp.typing as twt

# Sphere radius of the quadric fit, as a multiple of the average edge length (libigl's default).
_QUADRIC_RADIUS = 5

# Radius of the normal-cycle curvature measures, as a multiple of the mean edge length.
_MEASURE_RADIUS_FRACTION = 2.0

_face_angles_cache: dict[tuple[str, str], twt.Array2dFloat32] = {}


def _face_angles(bench_case: BenchCase) -> twt.Array2dFloat32:
    """``(n_faces, 3)`` interior angles — an *input* of ``discrete_gaussian_curvature``."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _face_angles_cache:
        _face_angles_cache[key] = tw.triangles.face_angles(
            bench_case.vertices_wp, bench_case.faces_wp
        )
    return _face_angles_cache[key]


def _measure_radius(bench_case: BenchCase) -> float:
    """Ball radius shared by triwarp and trimesh, derived from the shared NumPy source."""
    return _MEASURE_RADIUS_FRACTION * bench_case.mean_edge


@pytest.mark.benchmark(group="principal_curvature")
@pytest.mark.benchlibs("triwarp")
def test_principal_curvature(bench_case: BenchCase) -> None:
    """Per-vertex quadric fit over a geodesic ball on the scan meshes: the 5x5 solve in bulk."""
    skip_larger_than(bench_case, "happy_buddha", "geodesic-ball scratch scales with vertices")
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    *_, pv1, _pv2 = bench_case.run(
        lambda: tw.curvature.principal_curvature(vertices, faces, radius=_QUADRIC_RADIUS)
    )
    assert pv1.shape == (bench_case.n_vertices,)


@pytest.mark.benchmark(group="principal_curvature_saddle")
@pytest.mark.benchmeshes("synthetic_saddle_small", "synthetic_saddle")
@pytest.mark.benchlibs("triwarp", "igl")
def test_principal_curvature_saddle(bench_case: BenchCase) -> None:
    """The same fit against libigl, on the vertex-manifold saddle patches (see module docstring)."""
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        *_, pv1, _pv2 = bench_case.run(
            lambda: tw.curvature.principal_curvature(vertices, faces, radius=_QUADRIC_RADIUS)
        )
        assert pv1.shape == (n_vertices,)
    else:
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        *_, pv1_igl, _pv2, _bad = bench_case.run(
            lambda: igl.principal_curvature(
                vertices_np, faces_np, radius=_QUADRIC_RADIUS, useKring=False
            )
        )
        assert pv1_igl.shape == (n_vertices,)


@pytest.mark.benchmark(group="discrete_gaussian_curvature")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_discrete_gaussian_curvature(bench_case: BenchCase) -> None:
    """Summed vertex defects inside a ball around every vertex (Cohen-Steiner / Morvan)."""
    n_vertices = bench_case.n_vertices
    radius = _measure_radius(bench_case)
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
        skip_larger_than(bench_case, "bunny", "trimesh queries one cKDTree ball per point")
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        curvature_tm = bench_case.run(
            lambda: tm.curvature.discrete_gaussian_curvature_measure(
                tm.Trimesh(vertices_np, faces_np, process=False), vertices_np, radius
            )
        )
        assert curvature_tm.shape == (n_vertices,)


@pytest.mark.benchmark(group="discrete_mean_curvature")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_discrete_mean_curvature(bench_case: BenchCase) -> None:
    """Summed edge dihedral angles inside a ball around every vertex: adjacency plus the query."""
    n_vertices = bench_case.n_vertices
    radius = _measure_radius(bench_case)
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        curvature = bench_case.run(
            lambda: tw.curvature.discrete_mean_curvature(vertices, vertices, faces, radius)
        )
        assert curvature.shape == (n_vertices,)
    else:  # rebuild inside: face_adjacency and kdtree are cached Trimesh properties
        skip_larger_than(bench_case, "bunny", "trimesh queries one cKDTree ball per point")
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        curvature_tm = bench_case.run(
            lambda: tm.curvature.discrete_mean_curvature_measure(
                tm.Trimesh(vertices_np, faces_np, process=False), vertices_np, radius
            )
        )
        assert np.asarray(curvature_tm).shape == (n_vertices,)
