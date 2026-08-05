"""
Benchmarks for ``triwarp.bounds``: the axis-aligned box, its diagonal, and the oriented box.

Axis: the **scan sweep**, because that is the only thing the axis-aligned groups can be driven by --
a component-wise min/max over ``V`` has no topology and no parameters, so vertex count is the whole
story. ``oriented_bounding_box`` has a second axis of its own, the candidate count, and it is held
*fixed* at ``_ROTATIONS`` here so the sweep still reads as a sweep; see that group's docstring.

This module exists for one reason beyond coverage: ``aabb_bounds`` is on the hot path of *every*
k-NN query (``neighbors.query_*`` calls it to size the hash grid), and it is the clearest example in
the suite of a row that is **host-latency-bound at the small end and bandwidth-bound at the large
end**. Below roughly ``10 ** 3`` vertices it reports the ~340 µs wrapper floor rather than the
reduction (the same floor ``test_creation::test_box`` measures), so a NumPy ``min``/``max`` pair
wins by orders of magnitude there and loses at ``lucy``. The row is here for the crossover; neither
endpoint means anything on its own. That is also why ``bounds.aabb_bounds`` does not use the generic
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
``igl.bounding_box_diagonal(V)`` recomputes the box internally where
[`aabb_diagonal`][triwarp.bounds.aabb_diagonal] takes the two corners it is given. Read both as
upper bounds, and read the ``aabb_diagonal`` pair as what a caller pays *with* and *without* the box
already in hand: triwarp's row includes its own ``aabb_bounds`` call for exactly that reason, since
there is no other way to produce the number.

For the **oriented** box, both references answer it a different way and the rows say which:
``igl.oriented_bounding_box`` searches the *same* candidate set triwarp does (Super-Fibonacci over
``SO(3)``, identity appended) so it is a like-for-like comparison at an identical ``rotations``,
while ``trimesh.bounds.oriented_bounds`` is a different algorithm -- convex hull, then the minimal
box flush with each hull face -- so its row prices a hull triwarp never builds and its cost does not
scale with the candidate count at all. Read the trimesh row as the cost of the *other* approach
rather than as the same work done slower: on ``bunny`` it is the *slower* of the two references
(98.4 ms against igl's 60.6 at 4 096 candidates, triwarp 0.61) even though the hull collapses 35 947
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
from conftest import BenchCase

import triwarp as tw

# Candidate orientations scored by every ``oriented_bounding_box`` row, triwarp's and igl's alike.
# Fixed and shared: the two libraries search the identical candidate set, so a row that let them
# score different counts would compare quality against cost. It is also triwarp's own default.
_ROTATIONS = 4096


@pytest.mark.benchmark(group="aabb_bounds")
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d", "igl")
def test_aabb_bounds(bench_case: BenchCase) -> None:
    """A min/max reduce over the vertices, and the suite's clearest host-latency floor."""
    if bench_case.kind == "triwarp":
        vertices = bench_case.vertices_wp
        lower, upper = bench_case.run(lambda: tw.bounds.aabb_bounds(vertices))
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


@pytest.mark.benchmark(group="aabb_diagonal")
@pytest.mark.benchlibs("triwarp", "igl")
def test_aabb_diagonal(bench_case: BenchCase) -> None:
    """
    The bbox diagonal length, which on triwarp's side *is* the box reduction plus arithmetic.

    ``aabb_diagonal`` takes the two corners, so the timed callable composes it with ``aabb_bounds``
    -- the only path by which a caller obtains this number, and what makes the row comparable to
    igl's, which computes the box internally too. The gap against ``aabb_bounds`` above is therefore
    the ``sqrt`` and three subtractions, i.e. it should be nil; if it is not, the composition is
    paying a second readback.
    """
    if bench_case.kind == "triwarp":
        vertices = bench_case.vertices_wp
        diagonal = bench_case.run(lambda: tw.bounds.aabb_diagonal(*tw.bounds.aabb_bounds(vertices)))
        assert diagonal >= 0.0
        return
    vertices_np = bench_case.vertices_np
    diagonal_igl = bench_case.run(lambda: igl.bounding_box_diagonal(vertices_np))
    assert diagonal_igl >= 0.0


@pytest.mark.benchmark(group="oriented_bounding_box")
@pytest.mark.benchlibs("triwarp", "igl", "trimesh")
def test_oriented_bounding_box(bench_case: BenchCase) -> None:
    """
    The sampled minimum-volume box: ``_ROTATIONS`` candidate frames, each scored by an extent.

    Cost is ``rotations * n_vertices`` point transforms, so this is the one group here whose work is
    not a single pass over the vertices -- at ``_ROTATIONS`` it is 4 096 of them. triwarp scores
    every candidate in parallel and finishes the objective and the ``argmin`` on the host over a
    ``(rotations, 6)`` table; igl walks the same candidates over a CPU ``parallel_for``.

    Both CPU references run into hundreds of milliseconds on ``bunny`` and take ``rounds=3`` for it,
    the same allowance the other second-scale rows in the suite use.
    """
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
