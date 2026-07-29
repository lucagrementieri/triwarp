"""
Benchmarks for ``triwarp.intrinsic``: what robustness costs.

One question, asked on the scan sweep because these are flat per-face passes: how much more does
``intrinsic.robust_laplacian`` cost than ``laplacian.cotmatrix``? The difference is an edge-length
table, a max-reduce, a ``wp.map`` and **two host readbacks** -- the readbacks being the part
that does not shrink with mesh size, so this group really measures where they stop mattering.

``mollify_intrinsic`` is timed on its own so the readbacks are visible separately from the assembly
they feed, and ``intrinsic_delaunay`` on its own because its cost is a *loop*: every round rebuilds
face adjacency, marks candidates, claims a conflict-free set, updates lengths, commits, and reads
one int back to decide whether to go again. On an already-Delaunay mesh that is one round; on a quad
grid it is many.

Measured on an RTX 5090: ``mollify_intrinsic`` 0.28 ms on both ``bunny_decimated`` and
``bunny`` -- flat across a 6x face-count range, which is the two readbacks and nothing else -- and
``robust_laplacian`` 0.63 / 0.72 ms against ``igl.cotmatrix_intrinsic``'s 5.94 / 15.8 ms, so 9.5x
and 22x. Robustness costs about half a millisecond, most of it fixed.

``intrinsic_delaunay`` on the already-Delaunay scale axis runs **1.15 / 1.20 / 2.18 ms** against
igl's 7.2 / 58.9 / 260 ms (5.5x / 49x / 120x). Read igl's row with the caveat in the References
below -- it assembles the matrix in the same call -- but the shape holds: one round of the parallel
loop discovering there is nothing to do costs about a millisecond, and that is the price of leaving
the flag on by default.

References
----------
**libigl** is the reference for the intrinsic assembly: ``igl.cotmatrix_intrinsic`` takes the same
``(n_faces, 3)`` opposite-edge-length table and returns the same matrix, so its row measures the
same work from the same inputs. It has no mollification of its own; its
``igl.intrinsic_delaunay_cotmatrix`` is the reference for the *flips* instead, and appears in the
``intrinsic_delaunay`` group above.

**potpourri3d** exposes mollification only *inside*
``MeshHeatMethodDistanceSolver(use_robust=True)``, never as a standalone call, so it appears in
[`test_geodesic.py`](test_geodesic.py) rather than here. **trimesh**, **open3d** and **scipy** have
no equivalent.
"""

from __future__ import annotations

import igl
import numpy as np
import pytest
from conftest import BenchCase

import triwarp as tw


@pytest.mark.benchmark(group="robust_laplacian")
@pytest.mark.benchlibs("triwarp", "igl")
def test_robust_laplacian(bench_case: BenchCase) -> None:
    """Mollified edge lengths plus the intrinsic cotangent assembly."""
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        matrix = bench_case.run(lambda: tw.intrinsic.robust_laplacian(vertices, faces))
        assert matrix.nrow == n_vertices
    else:  # igl's intrinsic overload, from the same length table (built on the host with numpy)
        triangles_np = bench_case.vertices_np[bench_case.faces_np]
        lengths_np = np.ascontiguousarray(
            np.stack(
                [
                    np.linalg.norm(triangles_np[:, 2] - triangles_np[:, 1], axis=1),
                    np.linalg.norm(triangles_np[:, 0] - triangles_np[:, 2], axis=1),
                    np.linalg.norm(triangles_np[:, 1] - triangles_np[:, 0], axis=1),
                ],
                axis=1,
            ),
            dtype=np.float64,
        )
        faces_np = np.ascontiguousarray(bench_case.faces_np, dtype=np.int64)
        matrix_igl = bench_case.run(lambda: igl.cotmatrix_intrinsic(lengths_np, faces_np))
        assert matrix_igl.shape == (n_vertices, n_vertices)


@pytest.mark.benchmark(group="mollify_intrinsic")
@pytest.mark.benchlibs("triwarp")
def test_mollify_intrinsic(bench_case: BenchCase) -> None:
    """The length table, the max-reduce and the two host readbacks it needs."""
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    lengths, _ = bench_case.run(lambda: tw.intrinsic.mollify_intrinsic(vertices, faces))
    assert lengths.shape == (bench_case.n_faces, 3)


@pytest.mark.benchmark(group="intrinsic_delaunay")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "igl")
def test_intrinsic_delaunay(bench_case: BenchCase) -> None:
    """
    The flip loop: rounds of predicate, claim, length update and commit until nothing is left.

    On the ``scale`` axis this measures the *no-work* path -- an icosphere is already intrinsically
    Delaunay, so the loop pays one round to discover that and stops -- which is the honest baseline
    for the flag being on by default. The meshes that actually flip are the quad grids, and none of
    them is on this axis; ``tests/test_intrinsic.py`` covers those for correctness.
    """
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        intrinsic_faces, lengths, _ = bench_case.run(
            lambda: tw.intrinsic.intrinsic_delaunay(vertices, faces)
        )
        assert intrinsic_faces.shape == faces.shape
        assert lengths.shape == (bench_case.n_faces, 3)
    else:  # igl does the flips and the assembly in one call, so its row includes both
        vertices_np = bench_case.vertices_np
        faces_np = np.ascontiguousarray(bench_case.faces_np, dtype=np.int64)
        matrix_igl = bench_case.run(
            lambda: igl.intrinsic_delaunay_cotmatrix(vertices_np, faces_np)[0]
        )
        assert matrix_igl.shape == (bench_case.n_vertices, bench_case.n_vertices)


@pytest.mark.benchmark(group="face_edge_lengths")
@pytest.mark.benchlibs("triwarp")
def test_face_edge_lengths(bench_case: BenchCase) -> None:
    """The table alone: one pass, three lengths per face, no reduction."""
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    lengths = bench_case.run(lambda: tw.intrinsic.face_edge_lengths(vertices, faces))
    assert lengths.shape == (bench_case.n_faces, 3)
