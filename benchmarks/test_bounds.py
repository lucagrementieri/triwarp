"""
Benchmarks for ``triwarp.bounds``: the axis-aligned box, its diagonal, and the oriented box.

Axis: the **scan sweep**, because that is the only thing the axis-aligned groups can be driven by --
a component-wise min/max over ``V`` has no topology and no parameters, so vertex count is the whole
story. ``oriented_bounding_box`` has a second axis of its own, the candidate count, and it is held
*fixed* at ``_ROTATIONS`` here so the sweep still reads as a sweep; see that group's docstring.

This module exists for one reason beyond coverage: ``aabb`` is on the hot path of *every*
k-NN query (``neighbors.query_*`` calls it to size the hash grid), and it is the clearest example in
the suite of a row that is **host-latency-bound at the small end and bandwidth-bound at the large
end**. Below roughly ``10 ** 3`` vertices it reports the ~340 µs wrapper floor rather than the
reduction (the same floor ``test_creation::test_box`` measures), so a NumPy ``min``/``max`` pair
wins by orders of magnitude there and loses at ``lucy``. The row is here for the crossover; neither
endpoint means anything on its own. That is also why ``bounds.aabb`` does not use the generic
[`reduce.minmax`][triwarp.reduce.minmax] path -- it writes both corners into one six-element buffer
so the host pays a single readback.

References
----------
**trimesh** has no ``bounds`` function, only the cached ``Trimesh.bounds`` property, so the row runs
the uncached formula behind it (``vstack((v.min(0), v.max(0)))``) rather than timing a cache lookup.

**open3d**'s ``get_axis_aligned_bounding_box`` is its own reduction over the same vertices and
returns an object with ``get_min_bound`` / ``get_max_bound``.

**libigl** answers both groups, and in both cases it returns *more* than the number asked for:
``igl.bounding_box(V)`` builds the box as geometry -- 8 corner vertices and the 12 triangles of its
hull -- so its row includes constructing a mesh triwarp never materialises, and
``igl.bounding_box_diagonal(V)`` computes the box internally, exactly as
[`enclosing_diagonal`][triwarp.bounds.enclosing_diagonal] does on triwarp's side. Read both as
upper bounds.

For the **oriented** box, the references answer it a different way and the rows say which:
``igl.oriented_bounding_box`` searches the *same* global candidate set triwarp's first phase does
(Super-Fibonacci over ``SO(3)``, identity appended) but has no refinement phase, so triwarp's row
carries ~3-5 ms of trust-region rounds igl's does not -- rounds that buy the quality the
``tests/test_bounds.py`` bands pin (the sampled-only phase is reachable with
``refine_iterations=0`` and measured 0.45 ms on a 36k cloud back to back against ~2.4 refined).
``trimesh.bounds.oriented_bounds`` and open3d's ``get_minimal_oriented_bounding_box`` are the
*other* algorithm family -- convex hull, then the minimal box flush with each hull face -- so their
rows price a hull triwarp never builds and their cost does not scale with the candidate count at
all. Read them as the cost of that approach rather than as the same work done slower: on ``bunny``
trimesh reads 98.4 ms against igl's 60.6 at 4 096 candidates, even though the hull collapses 35 947
points to 1 564, because the caliper search then runs over all 3 124 hull faces serially.

One measured warning about igl's row, in the spirit of the ``igl.octree`` finding in
``test_neighbors``: ``igl.oriented_bounding_box`` is threaded (``igl::parallel_for`` over the
candidates) and its cost depends on the state of that thread pool. The same call at the same count
measured 203 ms standalone against 60.6 ms in-harness, a 3.4x spread, so quote the in-harness
median and nothing else.

``aabb_union`` gets no row: it is four ``wp.vec3`` component-wise minima at Python scope with no
device work at all, so a row would time the interpreter. It is covered in ``tests/test_bounds.py``.
"""

from __future__ import annotations

import igl
import numpy as np
import pytest
import trimesh as tm
import warp as wp
from meshlib import mrmeshpy as mm

import triwarp as tw
from conftest import BenchCase

# Candidate orientations scored by every ``oriented_bounding_box`` row, triwarp's and igl's alike.
# Fixed and shared: the two libraries search the identical candidate set, so a row that let them
# score different counts would compare quality against cost. It is also triwarp's own default.
_ROTATIONS = 4096

# Query points for the ``enclosing_diagonal`` row. Fixed count and seed: the cost is two box
# reductions and is flat in the second cloud's size at any realistic count, so sweeping it would
# add a second axis measuring nothing.
_N_QUERIES = 10_000
_QUERY_SEED = 7

_query_cache: dict[tuple[str, str], wp.array[wp.vec3]] = {}


def _queries_np(bench_case: BenchCase) -> np.ndarray:
    """Query points outside the mesh's own box, so the union is strictly larger than either side."""
    rng = np.random.default_rng(_QUERY_SEED)
    vertices = bench_case.vertices_np
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    return rng.random((_N_QUERIES, 3)) * extent + vertices.max(axis=0)


def _queries_wp(bench_case: BenchCase) -> wp.array[wp.vec3]:
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _query_cache:
        _query_cache[key] = wp.array(
            np.ascontiguousarray(_queries_np(bench_case), dtype=np.float32),
            dtype=wp.vec3,
            device=bench_case.device,
        )
    return _query_cache[key]


@pytest.mark.benchmark(group="aabb")
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d", "igl", "pyvista", "meshlib")
def test_aabb(bench_case: BenchCase) -> None:
    """
    A min/max reduce over the vertices, and the suite's clearest host-latency floor.

    VTK caches nothing but returns the box interleaved per axis (``xmin, xmax, ymin, ...``), which
    is a layout the test decodes and this row does not care about.

    meshlib's ``computeBoundingBox`` returns the two corners directly and is the only row here that
    is **multi-threaded**, which on a reduce this cheap mostly prices its own fork-join -- read it
    against ``triwarp-cuda``, not ``triwarp-cpu``.
    """
    if bench_case.kind == "meshlib":
        mesh_ml = bench_case.new_mesh_ml()
        box_ml = bench_case.run(
            lambda: mm.computeBoundingBox(mesh_ml.topology, mesh_ml.points, None)
        )
        assert box_ml.min.x <= box_ml.max.x
        return
    if bench_case.kind == "pyvista":
        mesh_pv = bench_case.mesh_pv
        bounds_pv = bench_case.run(lambda: np.asarray(mesh_pv.bounds))
        assert bounds_pv.shape == (6,)
        return
    if bench_case.kind == "triwarp":
        vertices = bench_case.vertices_wp
        lower, upper = bench_case.run(lambda: tw.bounds.aabb(vertices))
        assert lower[0] <= upper[0]
    elif bench_case.kind == "trimesh":
        # what an uncached ``trimesh.Trimesh.bounds`` computes: numpy min/max per axis
        vertices = bench_case.vertices_np
        result = bench_case.run(lambda: np.vstack((vertices.min(axis=0), vertices.max(axis=0))))
        assert result.shape == (2, 3)
    elif bench_case.kind == "igl":
        # Returns the box as geometry: 8 corner vertices plus the 12 triangles of its hull.
        vertices_np = bench_case.vertices_np
        corners_igl, faces_igl = bench_case.run(lambda: igl.bounding_box(vertices_np))
        assert corners_igl.shape == (8, 3)
        assert faces_igl.shape == (12, 3)
    else:  # open3d's own bound reduction over the same vertices
        mesh_o3d = bench_case.mesh_o3d
        box_o3d = bench_case.run(mesh_o3d.get_axis_aligned_bounding_box)
        assert box_o3d.get_min_bound()[0] <= box_o3d.get_max_bound()[0]


@pytest.mark.benchmark(group="enclosing_diagonal")
@pytest.mark.benchlibs("triwarp", "igl", "pyvista")
def test_enclosing_diagonal(bench_case: BenchCase) -> None:
    """
    The default search radius every mesh query derives, over the mesh *and* the query points.

    Two clouds rather than one, so against the ``aabb`` group above this is two box
    reductions and therefore two host readbacks; the row answers whether the default costs twice
    the single-cloud reduction or whether the second readback disappears into the first launch's
    latency. Every ``max_dist=None`` call in ``proximity``, ``ray``,
    ``visibility`` and ``registration`` pays exactly this.

    igl's ``bounding_box_diagonal`` takes one point set, so its side is fed the stacked cloud --
    which makes its row also price the ``vstack`` a caller would need, and that copy is the point:
    triwarp never materializes the union.
    """
    if bench_case.kind == "pyvista":
        # ``DataSet.length`` is the *single*-cloud diagonal -- VTK has no two-cloud form -- so this
        # row is the one-box half of what the other two compute. Read it as a floor.
        mesh_pv = bench_case.mesh_pv
        assert bench_case.run(lambda: float(mesh_pv.length)) > 0.0
        return
    if bench_case.kind == "triwarp":
        vertices, queries = bench_case.vertices_wp, _queries_wp(bench_case)
        diagonal = bench_case.run(lambda: tw.bounds.enclosing_diagonal(vertices, queries))
        assert diagonal > 0.0
        return
    vertices_np, queries_np = bench_case.vertices_np, _queries_np(bench_case)
    diagonal_igl = bench_case.run(
        lambda: igl.bounding_box_diagonal(np.vstack([vertices_np, queries_np]))
    )
    assert diagonal_igl > 0.0


@pytest.mark.benchmark(group="oriented_bounding_box")
@pytest.mark.benchlibs("triwarp", "igl", "trimesh", "open3d", "pyvista")
def test_oriented_bounding_box(bench_case: BenchCase) -> None:
    """
    The sampled-plus-refined minimum-volume box: ``_ROTATIONS`` global frames, then eight rounds.

    Cost is ``rotations * n_vertices`` point transforms for the global phase plus eight 512-frame
    refinement rounds of device-side frame generation, extent scoring and one table readback
    (~0.24 ms per round; measured back to back on a 36k cloud, 0.45 ms sampled against ~2.4 ms
    refined). The row
    times the *default*, refinement included, because that is what a caller gets -- and what the
    quality bands in ``tests/test_bounds.py`` are measured against; igl walks the same global
    candidates over a CPU ``parallel_for`` with no refinement phase.

    The CPU references run into hundreds of milliseconds on ``bunny`` and take ``rounds=3`` for it,
    the same allowance the other second-scale rows in the suite use.

    **pyvista's row is PCA of the points**, a single fixed orientation rather than a search -- so it
    is the floor of this group by construction and the only reference here that triwarp should beat
    on *quality* as well (measured 0.84x its volume on a tilted half_torus).

    open3d times ``get_minimal_oriented_bounding_box`` -- its hull-based approximate minimizer,
    the same algorithm family as trimesh's row -- not ``get_oriented_bounding_box``, whose PCA box
    does not minimize anything (measured 12.9% above triwarp's volume on the tilted half_torus
    where the minimal box sits within 2%). Like both other references it is insensitive to
    ``_ROTATIONS`` by construction.
    """
    if bench_case.kind == "pyvista":
        # PCA of the points: one fixed orientation, so it is a floor and not a minimizer (measured
        # up to 19% above triwarp's volume in tests/test_bounds.py).
        mesh_pv = bench_case.mesh_pv
        box_pv = bench_case.run(lambda: mesh_pv.oriented_bounding_box(as_composite=False), rounds=3)
        assert box_pv.volume > 0.0
        return
    if bench_case.kind == "open3d":
        import open3d as o3d

        vertices_np = bench_case.vertices_np

        def minimal_box_o3d() -> object:
            cloud_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices_np))
            return cloud_o3d.get_minimal_oriented_bounding_box()

        assert bench_case.run(minimal_box_o3d, rounds=3).volume() > 0.0
        return
    if bench_case.kind == "triwarp":
        vertices = bench_case.vertices_wp
        rotation, lower, upper = bench_case.run(
            lambda: tw.bounds.oriented_bounding_box(vertices, _ROTATIONS)
        )
        assert lower[0] <= upper[0]
        assert np.isfinite(rotation[0, 0])
    elif bench_case.kind == "igl":
        # The same candidate set at the same count; igl returns the frame alone, transposed relative
        # to triwarp's because it multiplies row vectors on the right.
        vertices_np = bench_case.vertices_np
        frame_igl = bench_case.run(
            lambda: igl.oriented_bounding_box(vertices_np, _ROTATIONS), rounds=3
        )
        assert frame_igl.shape == (3, 3)
    else:
        # A different algorithm, not a slower one: convex hull, then the best hull-face-flush box.
        # Insensitive to ``_ROTATIONS`` by construction, which is why the row does not take it.
        vertices_np = bench_case.vertices_np
        _, extents_tm = bench_case.run(
            lambda: tm.bounds.oriented_bounds(tm.PointCloud(vertices_np)), rounds=3
        )
        assert extents_tm.shape == (3,)
