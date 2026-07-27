"""
Benchmarks for ``triwarp.convex``.

Two unrelated cost shapes share this module:

* **Local convexity** (``face_adjacency_projections``, ``face_adjacency_convex``) — one thread per
  adjacent face pair projects the neighbour's unshared vertex onto the first face's plane. The
  arithmetic is trivial; the cost is almost entirely the *adjacency construction* that precedes it
  when the optional ``face_adjacency`` arguments are not supplied. Both are timed the way a caller
  who has nothing precomputed would call them, so these numbers are "adjacency + projection" and it
  is the adjacency that dominates. ``face_adjacency_convex`` is ``face_adjacency_projections``
  plus a threshold, so the delta between the two groups is the comparison pass alone.
* **Approximate hull** (``fast_convex_set_mask``, ``fast_convex_set``) — a tiled block reduction:
  for each of ``n_directions`` Fibonacci hemisphere directions, a ``TILE_1D``-wide ``wp.tile_max`` /
  ``wp.tile_min`` sweep over every point finds the two support extrema, then a second pass marks
  them. Cost is ``n_points * n_directions``, so this is the module's compute-bound case and the one
  the tile-tail clamp sits in. ``fast_convex_set`` is the mask plus a ``flatnonzero`` and a gather,
  so its delta over the mask is the compaction cost.

References
----------
**trimesh** is a genuine baseline for ``face_adjacency_convex``: ``Trimesh.face_adjacency_convex``
computes the same predicate. It is a *cached property*, so the ``Trimesh`` is rebuilt inside the
timed callable — otherwise rounds 2..n would return a memoized array and measure nothing. That
rebuild also pays trimesh's own ``face_adjacency`` construction, which is the honest comparison
since the triwarp side builds its adjacency inside the timed region too.

For the hull, **the baselines compute a different (and stronger) result**, and that is the point of
the comparison rather than a flaw in it. ``trimesh.Trimesh.convex_hull`` and
``open3d.geometry.TriangleMesh.compute_convex_hull`` both run **qhull**, producing the exact hull as
a *mesh* — full connectivity, exact vertex set. ``fast_convex_set`` produces only an approximate
*vertex subset* (a support sweep over finitely many directions, which misses hull vertices whose
normal cone no sampled direction enters). So this is not a parity comparison: it is the
quantification of what the approximation buys, which is the reason the function exists. The
module docstring in ``triwarp/convex.py`` documents the accuracy side of that trade;
this benchmark is the cost side. ``scipy.spatial.ConvexHull`` — the reference the
docstring cross-links — is the same qhull algorithm as both registered baselines, so it would add
a third timing of the same thing and is not registered separately.

**libigl** has no convex-hull or local-convexity binding in the Python package, so igl is absent
from every case here.

Points
------
The hull cases take the mesh's own vertices as the point cloud, so they scale with the registry
mesh sizes. ``n_directions`` stays at the default 128 — the cost is exactly linear in it, so a sweep
would add cases without adding information.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh as tm
from conftest import BenchCase, skip_larger_than

import triwarp as tw


@pytest.mark.benchmark(group="face_adjacency_projections")
@pytest.mark.benchlibs("triwarp")
def test_face_adjacency_projections(bench_case: BenchCase) -> None:
    """Unshared-vertex plane projections per adjacent face pair, adjacency built inside."""
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    projections = bench_case.run(lambda: tw.convex.face_adjacency_projections(vertices, faces))
    assert projections.shape[0] >= 0


@pytest.mark.benchmark(group="face_adjacency_convex")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_face_adjacency_convex(bench_case: BenchCase) -> None:
    """Locally-convex adjacent face pairs: the projection plus a tolerance threshold."""
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        convex = bench_case.run(lambda: tw.convex.face_adjacency_convex(vertices, faces))
        assert convex.shape[0] >= 0
    else:  # rebuild inside: face_adjacency_convex is a cached Trimesh property
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        convex_tm = bench_case.run(
            lambda: tm.Trimesh(vertices_np, faces_np, process=False).face_adjacency_convex
        )
        assert convex_tm.dtype == bool


@pytest.mark.benchmark(group="fast_convex_set_mask")
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d")
def test_fast_convex_set_mask(bench_case: BenchCase) -> None:
    """Tiled support sweep over 128 directions vs exact qhull (see the module docstring)."""
    if bench_case.kind == "triwarp":
        points = bench_case.vertices_wp
        mask = bench_case.run(lambda: tw.convex.fast_convex_set_mask(points))
        assert mask.shape[0] == bench_case.n_vertices
    elif bench_case.kind == "trimesh":
        skip_larger_than(bench_case, "dragon", "qhull is single-threaded on the host")
        points_np = bench_case.vertices_np
        hull_tm = bench_case.run(lambda: tm.points.PointCloud(points_np).convex_hull)
        assert hull_tm.vertices.shape[1] == 3
    else:
        skip_larger_than(bench_case, "dragon", "qhull is single-threaded on the host")
        mesh_o3d = bench_case.mesh_o3d
        hull_o3d, _indices = bench_case.run(lambda: mesh_o3d.compute_convex_hull())
        assert np.asarray(hull_o3d.vertices).shape[1] == 3


@pytest.mark.benchmark(group="fast_convex_set")
@pytest.mark.benchlibs("triwarp")
def test_fast_convex_set(bench_case: BenchCase) -> None:
    """The mask plus ``flatnonzero`` and a gather: isolates the compaction cost."""
    points = bench_case.vertices_wp
    selected = bench_case.run(lambda: tw.convex.fast_convex_set(points))
    assert selected.shape[0] <= bench_case.n_vertices
