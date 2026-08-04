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
* **Approximate hull** (``convex_subset_mask``, ``convex_subset``) — for each of ``n_directions``
  Fibonacci hemisphere directions, one thread per strided slice of the cloud reduces the support
  function and one atomic per thread combines the slices, then a second pass marks the extrema. Cost
  is ``n_points * n_directions``, so this is the module's compute-bound case. ``convex_subset`` is
  the mask plus a ``flatnonzero`` and a gather, so its delta over the mask is the compaction cost.
  (This sweep used to be a ``TILE_1D``-wide ``wp.tile_max`` / ``wp.tile_min`` block reduction. It is
  lane-free now because ``wp.launch_tiled`` runs exactly one lane per block on Warp 1.15's CPU
  backend, which made every tiled formulation silently wrong there; the replacement also measured
  1.0-2.7x *faster* on CUDA, the gap widening with ``n_points * n_directions``.)
* **Conservative hull prefilter** (``convex_superset_mask``) — the same support sweep over an
  icosphere's directions, then one pass testing every point against the ``20 * 4 ** subdivisions``
  tetrahedra spanned by the resulting shell. Its second half is the new cost shape: a loop whose
  every thread reads the same tetrahedron's face planes at the same time, so the plane table is
  broadcast out of cache and the arithmetic dominates.

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
a *mesh* — full connectivity, exact vertex set. ``convex_subset`` produces only an approximate
*vertex subset* (a support sweep over finitely many directions, which misses hull vertices whose
normal cone no sampled direction enters). So this is not a parity comparison: it is the
quantification of what the approximation buys, which is the reason the function exists.
``convex_subset_mask``'s own docstring documents the accuracy side of that trade -- including the
measured recall per ``n_directions`` and the normal-cone sizes that explain it; this benchmark is
the cost side.

**scipy** is registered for ``convex_superset_mask`` only, and there it is a genuine parity row
rather than a bar. That filter exists to run *before* an exact hull, so ``scipy.spatial.ConvexHull``
on the same cloud is exactly the cost it has to be cheap against, and its output provably contains
that hull's vertex set (asserted in ``tests/test_convex.py``). The ratio is the number that decides
whether the prefilter is worth running: on ``dragon`` it measured 2.4 ms at ``subdivisions=1`` and
7.6 ms at 3, against 145 ms for the hull itself — 60x and 19x — and the gap widens with the point
count, because the filter is linear where qhull is not. For the two approximate-hull groups scipy
would only be a third timing of the qhull already covered by trimesh and Open3D, so it stays out of
those.

**pymeshlab**'s ``generate_convex_hull`` is qhull a third time, so it adds no new algorithm -- what
it adds is a *second* wrapper cost around the same computation, which is the only way to tell
whether trimesh's number is qhull or trimesh. It pushes the hull onto the MeshSet as a new mesh, so
the set is rebuilt per round, and it is capped at ``dragon`` with the other two for the same reason.

**libigl** has no convex-hull or local-convexity binding in the Python package, so igl is absent
from every case here.

Points
------
The hull cases take the mesh's own vertices as the point cloud, so they scale with the registry
mesh sizes. ``n_directions`` and ``subdivisions`` are each swept over two values spanning their
useful range; both costs are close to linear in the direction count, so two points fix the line.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.spatial
import trimesh as tm
from conftest import BenchCase, skip_larger_than

import triwarp as tw

# Direction counts for the support sweep. Cost is exactly ``points x n_directions`` -- the only
# knob in the module, and the accuracy/speed trade against exact qhull.
_N_DIRECTIONS = [32, 256]

# Icosphere refinement levels for the conservative filter: 42 directions / 80 tetrahedra at 1, and
# 642 / 1280 at 3. Both halves of its cost scale with this, so it spans the useful range.
_SUBDIVISIONS = [1, 3]


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


@pytest.mark.benchmark(group="convex_subset_mask")
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d", "pymeshlab")
@pytest.mark.parametrize("n_directions", _N_DIRECTIONS)
def test_convex_subset_mask(bench_case: BenchCase, n_directions: int) -> None:
    """
    Tiled support sweep vs exact qhull (see the module docstring).

    Cost is ``points x n_directions`` with no topology involved, so the direction count is the
    axis. The references are exact and take no such parameter, so their two rows are identical by
    construction -- they are there as the fixed bar the approximation is trading accuracy against.
    """
    if bench_case.kind == "pymeshlab":  # qhull again, through MeshLab's own wrapper
        skip_larger_than(bench_case, "dragon", "qhull is single-threaded on the host")
        bench_case.run(lambda: bench_case.new_meshset_pml().generate_convex_hull())
        return
    if bench_case.kind == "triwarp":
        points = bench_case.vertices_wp
        mask = bench_case.run(
            lambda: tw.convex.convex_subset_mask(points, n_directions=n_directions)
        )
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


@pytest.mark.benchmark(group="convex_subset")
@pytest.mark.benchlibs("triwarp")
def test_convex_subset(bench_case: BenchCase) -> None:
    """The mask plus ``flatnonzero`` and a gather: isolates the compaction cost."""
    points = bench_case.vertices_wp
    selected = bench_case.run(lambda: tw.convex.convex_subset(points))
    assert selected.shape[0] <= bench_case.n_vertices


@pytest.mark.benchmark(group="convex_superset_mask")
@pytest.mark.benchlibs("triwarp", "scipy")
@pytest.mark.parametrize("subdivisions", _SUBDIVISIONS)
def test_convex_superset_mask(bench_case: BenchCase, subdivisions: int) -> None:
    """
    The conservative prefilter against the exact hull it prefilters for.

    ``scipy`` is the right row here, unlike in the two groups above where the qhull wrappers are
    only a fixed accuracy bar: this filter's *purpose* is to run before an exact hull, so the
    question the benchmark has to answer is whether it is cheap relative to the hull it feeds. Both
    rows are timed on the same cloud and produce comparable results (the filter's output contains
    the hull's vertex set, which ``tests/test_convex.py`` asserts), so this one *is* a parity
    comparison.

    ``subdivisions`` is the axis because it drives both halves of the cost -- the direction count of
    the support sweep and the tetrahedron count of the interior test -- and is the knob that trades
    selectivity for time.
    """
    if bench_case.kind == "triwarp":
        points = bench_case.vertices_wp
        mask = bench_case.run(
            lambda: tw.convex.convex_superset_mask(points, subdivisions=subdivisions)
        )
        assert mask.shape[0] == bench_case.n_vertices
    else:
        skip_larger_than(bench_case, "dragon", "qhull is single-threaded on the host")
        points_np = bench_case.vertices_np
        hull_np = bench_case.run(lambda: scipy.spatial.ConvexHull(points_np))
        assert hull_np.vertices.shape[0] >= 4
