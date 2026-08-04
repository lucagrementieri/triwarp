"""
Benchmarks for ``triwarp.bounds``: the axis-aligned box reduction and its diagonal.

Axis: the **scan sweep**, because that is the only thing these can be driven by -- a component-wise
min/max over ``V`` has no topology and no parameters, so triangle count is the whole story.

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

``aabb_union`` gets no row: it is four ``wp.vec3`` component-wise minima at Python scope with no
device work at all, so a row would time the interpreter. It is covered in ``tests/test_bounds.py``.
"""

from __future__ import annotations

import igl
import numpy as np
import pytest
from conftest import BenchCase

import triwarp as tw


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
