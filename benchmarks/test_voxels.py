"""
Benchmarks for ``triwarp.voxels``.

Every group here is really a race between **one hash table and one NanoVDB grid**. open3d keeps a
``std::unordered_map<Eigen::Vector3i>`` and walks it on one core; triwarp keeps a NanoVDB index grid
whose build deduplicates as it goes and whose membership query is an ``O(1)`` probe from a kernel.
So the interesting reading is not any single row but the *slope*: how each side responds to the
cubic axis when the cell shrinks.

References
----------
**open3d** is the direct counterpart for four of the groups and computes the same answer in each:
``create_from_triangle_mesh_within_bounds`` runs Moller's tri-box test per candidate cell (in
``float64``, over a slightly wider candidate window that cannot change the accept set),
``create_from_point_cloud`` bins with the same ``min_bound - voxel_size / 2`` anchor,
``voxel_down_sample`` averages the same cells, and ``check_if_included`` answers the same membership
question. All four are asserted element-wise in ``tests/test_voxels.py``.

**trimesh** covers the morphology and the box meshing. ``voxel.morphology.binary_dilation`` and
``fill_holes`` are ``scipy.ndimage`` on a dense array, so those two rows are really "sparse grid
against dense ndimage" and their axis is the one that separates them: the dense side pays for the
whole bounding box, the sparse side only for the occupied cells. ``voxel.ops.multibox`` builds
``12 n`` triangles on the host with no corner sharing.

**libigl** covers the two lattice functions, ``grid`` and ``unique_sparse_voxel_corners``. Both are
single-core C++ over the same integer arithmetic, so they read as a bandwidth comparison.

``trimesh.voxel.creation.voxelize_subdivide`` is timed on the ``voxelize_mesh`` axis but is
**not** a parity reference — it is a different algorithm (subdivide every triangle until its edges
are under half a pitch, then round each vertex to a cell), and its result neither contains nor is
contained in the exact accept set. open3d is the oracle for that group.

What has no group, and why
--------------------------
``cells``, ``from_cells``, ``grid_transform``, ``cell_indices``, ``cell_centers``,
``occupancy_at_cells``, ``erode``, ``surface_voxels``, ``to_dense``, ``from_dense`` and
``to_field`` are each one launch over the voxel set with no allocation of their
own, so a row would time the ~340 µs wrapper floor rather than the operation. Most are measured
anyway through a group that calls them: ``fill_holes`` and ``fill_orthographic`` both go through
``to_dense`` / ``from_dense``, ``surface_voxels`` runs the same probe kernel as ``erode``, and
``cells`` is on the critical path of every group below.

``surface_voxels`` has no reference row either: ``trimesh.voxel.morphology.surface`` exists, but it
is the ``erode`` complement on a dense array and would measure the same thing the ``dilate`` row
already does.
"""

from __future__ import annotations

import igl
import numpy as np
import pytest
import trimesh as tm
import warp as wp
from conftest import BenchCase, BenchLibrary, skip_larger_than

import triwarp as tw

# Cell widths as a fraction of the bounding-box diagonal. The pair is a slope check: 1/256 is 64x
# the cells of 1/64, and the whole point of the flattened (triangle, cell) work-item design is that
# the cost tracks the *accepted* cell count rather than the triangle count. Mirrors
# ``benchmarks/test_reconstruction.py``'s ``_RESAMPLE_CELL_FRACTIONS`` pair.
_CELL_DIVISORS = [64, 256]

# One cell width for the groups whose axis is the mesh rather than the pitch.
_MORPHOLOGY_DIVISOR = 96

# A voxelization at 1/256 of the diagonal and a dense morphology pass over its bounding box are
# both seconds a call on the CPU references.
_HEAVY_ROUNDS = 3


def _voxel_size(bench_case: BenchCase, divisor: int) -> float:
    """Absolute cell width from the mesh's own bbox diagonal, identical on every row."""
    vertices_np = bench_case.vertices_np
    return float(np.linalg.norm(vertices_np.max(axis=0) - vertices_np.min(axis=0))) / divisor


@pytest.mark.noparity(
    "trimesh",
    oracle="open3d",
    reason="D2 a different algorithm with a measured disagreement: voxelize_subdivide splits every "
    "triangle until its edges are under pitch/2 and then rounds each resulting vertex to a cell, "
    "so it accepts a cell iff some *sample* lands in it. That set neither contains nor is "
    "contained in the exact tri-box accept set -- it misses cells a triangle merely clips and adds "
    "cells no part of the triangle reaches. open3d runs the identical 13-axis test and is asserted "
    "cell-for-cell in tests/test_voxels.py::test_voxelize_mesh_matches_open3d.",
)
@pytest.mark.benchmark(group="voxelize_mesh")
@pytest.mark.benchlibs("triwarp", "open3d", "trimesh")
@pytest.mark.parametrize("divisor", _CELL_DIVISORS)
def test_voxelize_mesh(bench_case: BenchCase, divisor: int) -> None:
    """
    The **cubic** group: one thread per (triangle, candidate cell) pair against two serial loops.

    This is the group that would catch a regression to a thread-per-triangle launch. Per-triangle
    candidate windows span orders of magnitude on any real mesh, so a thread-per-triangle kernel is
    load-imbalanced by that same factor while its total work is unchanged — the symptom would be
    the 1/256 point rising far faster than the accepted cell count does.

    open3d gets the identical absolute cell width and the identical ``min_bound``, so both sides
    lay down the same lattice; ``create_from_triangle_mesh_within_bounds`` is used rather than
    ``create_from_triangle_mesh`` precisely so the bounds are the caller's on both sides.
    """
    voxel_size = _voxel_size(bench_case, divisor)
    origin = bench_case.vertices_np.min(axis=0) - 0.5 * voxel_size

    if bench_case.kind == "open3d":
        import open3d as o3d

        skip_larger_than(bench_case, "bunny", "open3d walks the candidate cube on one core")
        mesh_o3d = bench_case.mesh_o3d
        upper = origin + np.ceil((bench_case.vertices_np.max(axis=0) - origin) / voxel_size) * (
            voxel_size
        )
        grid_o3d = bench_case.run(
            lambda: o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
                mesh_o3d, voxel_size=voxel_size, min_bound=origin, max_bound=upper
            ),
            rounds=_HEAVY_ROUNDS,
        )
        assert len(grid_o3d.get_voxels()) > 0
        return
    if bench_case.kind == "trimesh":
        skip_larger_than(
            bench_case, "bunny_decimated", "voxelize_subdivide subdivides the whole mesh in Python"
        )
        mesh_tm = tm.Trimesh(bench_case.vertices_np, bench_case.faces_np, process=False)
        grid_tm = bench_case.run(
            lambda: tm.voxel.creation.voxelize_subdivide(mesh_tm, pitch=voxel_size),
            rounds=_HEAVY_ROUNDS,
        )
        assert grid_tm.filled_count > 0
        return

    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    origin_wp = wp.vec3(*origin.tolist())
    grid = bench_case.run(
        lambda: tw.voxels.voxelize_mesh(vertices, faces, voxel_size, origin=origin_wp),
        rounds=_HEAVY_ROUNDS,
    )
    assert int(grid.get_active_stats().voxel_count) > 0


@pytest.mark.benchmark(group="voxelize_points")
@pytest.mark.benchlibs("triwarp", "open3d")
def test_voxelize_points(bench_case: BenchCase) -> None:
    """
    Pure ``N``: bin a cloud and deduplicate, with the pitch pinned so only the point count moves.

    This is where the grid build itself is regression-guarded — measured at 0.64 ms per million
    cells on an RTX 5090 against 1.51 ms for the equivalent ``grouping.unique_rows`` dedup, which
    is the measurement the whole module's data structure choice rests on.
    """
    voxel_size = _voxel_size(bench_case, _MORPHOLOGY_DIVISOR)
    if bench_case.kind == "open3d":
        import open3d as o3d

        skip_larger_than(bench_case, "bunny", "open3d bins into a hash map on one core")
        cloud_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(bench_case.vertices_np))
        grid_o3d = bench_case.run(
            lambda: o3d.geometry.VoxelGrid.create_from_point_cloud(cloud_o3d, voxel_size=voxel_size)
        )
        assert len(grid_o3d.get_voxels()) > 0
        return

    points = bench_case.vertices_wp
    grid = bench_case.run(lambda: tw.voxels.voxelize_points(points, voxel_size))
    assert int(grid.get_active_stats().voxel_count) > 0


@pytest.mark.benchmark(group="voxel_down_sample")
@pytest.mark.benchlibs("triwarp", "open3d")
@pytest.mark.parametrize("divisor", _CELL_DIVISORS)
def test_voxel_down_sample(bench_case: BenchCase, divisor: int) -> None:
    """
    The inverse map plus a deterministic segment reduce, against open3d's hash-map accumulation.

    triwarp pays a radix sort over the point count that open3d does not, in exchange for a bitwise
    reproducible mean; the pitch axis is what shows whether that sort or the grid build dominates.
    A finer pitch means more voxels and shorter segments, so the sort's share rises while the sort
    itself — one pass over the point count — does not move at all.
    """
    voxel_size = _voxel_size(bench_case, divisor)
    if bench_case.kind == "open3d":
        import open3d as o3d

        skip_larger_than(bench_case, "bunny", "open3d accumulates into a hash map on one core")
        cloud_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(bench_case.vertices_np))
        pooled_o3d = bench_case.run(lambda: cloud_o3d.voxel_down_sample(voxel_size))
        assert len(pooled_o3d.points) > 0
        return

    points = bench_case.vertices_wp
    pooled = bench_case.run(lambda: tw.voxels.voxel_down_sample(points, voxel_size))
    assert int(pooled.shape[0]) > 0


# Query counts for the membership group. The grid is fixed, so this axis is purely "how does the
# cost of a lookup scale" -- one serial hash probe per query on the reference, one kernel thread
# per query here.
_QUERY_COUNTS = [10_000, 1_000_000]


@pytest.mark.benchmark(group="occupancy_at_points")
@pytest.mark.benchlibs("triwarp", "open3d")
@pytest.mark.parametrize("n_queries", _QUERY_COUNTS)
def test_occupancy_at_points(bench_case: BenchCase, n_queries: int) -> None:
    """
    Membership against a pinned grid: ``check_if_included`` is one hash lookup per query, serially.

    The queries are the mesh's own vertices, tiled with a deterministic jitter so that the answer
    is a genuine mix of hits and misses rather than all-true — a query set entirely inside the grid
    would let a constant answer keep pace.
    """
    voxel_size = _voxel_size(bench_case, _MORPHOLOGY_DIVISOR)
    vertices_np = bench_case.vertices_np
    rng = np.random.default_rng(4)
    repeats = int(np.ceil(n_queries / vertices_np.shape[0]))
    queries_np = np.ascontiguousarray(
        (
            np.tile(vertices_np, (repeats, 1))
            + rng.normal(0.0, 2.0 * voxel_size, (repeats * vertices_np.shape[0], 3))
        )[:n_queries]
    )

    if bench_case.kind == "open3d":
        import open3d as o3d

        skip_larger_than(bench_case, "bunny", "one serial hash lookup per query")
        cloud_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices_np))
        grid_o3d = o3d.geometry.VoxelGrid.create_from_point_cloud(cloud_o3d, voxel_size=voxel_size)
        queries_o3d = o3d.utility.Vector3dVector(queries_np)
        inside_o3d = bench_case.run(lambda: grid_o3d.check_if_included(queries_o3d))
        assert len(inside_o3d) == n_queries
        return

    points = bench_case.vertices_wp
    grid = tw.voxels.voxelize_points(points, voxel_size)
    queries = wp.array(queries_np.astype(np.float32), dtype=wp.vec3, device=bench_case.device)
    mask = bench_case.run(lambda: tw.voxels.occupancy_at_points(grid, queries))
    assert int(mask.shape[0]) == n_queries


@pytest.mark.benchmark(group="dilate")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_dilate(bench_case: BenchCase) -> None:
    """
    Grow the set by one shell: a ``(k + 1) * n_voxels`` candidate buffer against dense ndimage.

    The candidate buffer is what to watch here — it is 27 ``vec3`` per voxel at
    ``connectivity=26``, 324 MB at a million voxels, and the reason 6 is the default. trimesh's
    reference is ``scipy.ndimage.binary_dilation`` over the *dense* bounding box, so the two sides
    scale with different quantities and the gap widens with sparsity.
    """
    voxel_size = _voxel_size(bench_case, _MORPHOLOGY_DIVISOR)
    if bench_case.kind == "trimesh":
        skip_larger_than(bench_case, "bunny", "ndimage dilates the whole dense bounding box")
        mesh_tm = tm.Trimesh(bench_case.vertices_np, bench_case.faces_np, process=False)
        grid_tm = tm.voxel.creation.voxelize_subdivide(mesh_tm, pitch=voxel_size)
        encoding_tm = grid_tm.encoding
        dilated_tm = bench_case.run(
            lambda: tm.voxel.morphology.binary_dilation(encoding_tm), rounds=_HEAVY_ROUNDS
        )
        assert dilated_tm.sum > 0
        return

    grid = tw.voxels.voxelize_mesh(bench_case.vertices_wp, bench_case.faces_wp, voxel_size)
    dilated = bench_case.run(lambda: tw.voxels.dilate(grid))
    assert int(dilated.get_active_stats().voxel_count) > 0


@pytest.mark.benchmark(group="fill_holes")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_fill_holes(bench_case: BenchCase) -> None:
    """
    The one group whose cost is *not* the grid: it is the empty complement, which grows cubically.

    triwarp labels the empty cells' 6-connected components with an ECL-CC union-find driven from an
    implicit stencil — three backward probes per cell, one launch, zero edge memory — where
    ``scipy.ndimage.binary_fill_holes`` runs a serial binary propagation over the same dense box.
    A regression to an explicit edge list would show up here first, as memory before time.
    """
    voxel_size = _voxel_size(bench_case, _MORPHOLOGY_DIVISOR)
    if bench_case.kind == "trimesh":
        skip_larger_than(bench_case, "bunny", "binary_fill_holes iterates over the dense box")
        mesh_tm = tm.Trimesh(bench_case.vertices_np, bench_case.faces_np, process=False)
        grid_tm = tm.voxel.creation.voxelize_subdivide(mesh_tm, pitch=voxel_size)
        encoding_tm = grid_tm.encoding
        filled_tm = bench_case.run(
            lambda: tm.voxel.morphology.fill_holes(encoding_tm), rounds=_HEAVY_ROUNDS
        )
        assert filled_tm.sum > 0
        return

    grid = tw.voxels.voxelize_mesh(bench_case.vertices_wp, bench_case.faces_wp, voxel_size)
    filled = bench_case.run(lambda: tw.voxels.fill_holes(grid), rounds=_HEAVY_ROUNDS)
    assert int(filled.get_active_stats().voxel_count) > 0


@pytest.mark.benchmark(group="fill_orthographic")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_fill_orthographic(bench_case: BenchCase) -> None:
    """
    The cheap fill: three axis sweeps and an intersection, against three dense NumPy reductions.

    Both sides work on the dense occupancy box, so unlike ``fill_holes`` this pair really is the
    same shape of work on the same data and reads as a straight parallel-versus-serial comparison.
    trimesh's version builds three ``cumsum``-style masks with NumPy, so it is the fastest of its
    morphology functions and the hardest of them to beat.
    """
    voxel_size = _voxel_size(bench_case, _MORPHOLOGY_DIVISOR)
    if bench_case.kind == "trimesh":
        skip_larger_than(bench_case, "bunny", "three dense NumPy passes over the bounding box")
        mesh_tm = tm.Trimesh(bench_case.vertices_np, bench_case.faces_np, process=False)
        occupancy_np = tm.voxel.creation.voxelize_subdivide(mesh_tm, pitch=voxel_size).matrix
        filled_tm = bench_case.run(
            lambda: tm.voxel.ops.fill_orthographic(occupancy_np), rounds=_HEAVY_ROUNDS
        )
        assert filled_tm.sum() > 0
        return

    grid = tw.voxels.voxelize_mesh(bench_case.vertices_wp, bench_case.faces_wp, voxel_size)
    filled = bench_case.run(lambda: tw.voxels.fill_orthographic(grid), rounds=_HEAVY_ROUNDS)
    assert int(filled.get_active_stats().voxel_count) > 0


@pytest.mark.benchmark(group="to_boxes")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_to_boxes(bench_case: BenchCase) -> None:
    """
    Mesh the voxel set as cubes: shared nanogrid corners against ``12 n`` unshared triangles.

    ``multibox`` tiles a template cube per centre and never welds, so it always emits ``8 n``
    vertices; triwarp's corners come deduplicated out of ``warp.fem``'s vertex grid, which is a
    smaller output *and* skips a welding pass. Both rows emit every face (``cull_internal=False``)
    so the comparison is like for like.
    """
    voxel_size = _voxel_size(bench_case, _MORPHOLOGY_DIVISOR)
    if bench_case.kind == "trimesh":
        skip_larger_than(bench_case, "bunny", "multibox tiles a template cube per voxel in Python")
        mesh_tm = tm.Trimesh(bench_case.vertices_np, bench_case.faces_np, process=False)
        centers_np = tm.voxel.creation.voxelize_subdivide(mesh_tm, pitch=voxel_size).points
        boxes_tm = bench_case.run(
            lambda: tm.voxel.ops.multibox(centers_np, pitch=voxel_size), rounds=_HEAVY_ROUNDS
        )
        assert boxes_tm.faces.shape[0] > 0
        return

    grid = tw.voxels.voxelize_mesh(bench_case.vertices_wp, bench_case.faces_wp, voxel_size)
    _vertices, faces = bench_case.run(lambda: tw.voxels.to_boxes(grid, cull_internal=False))
    assert int(faces.shape[0]) > 0


@pytest.mark.benchmark(group="voxel_corners")
@pytest.mark.benchlibs("triwarp", "igl")
def test_voxel_corners(bench_case: BenchCase) -> None:
    """
    The deduplicated corner lattice, from ``warp.fem``'s vertex grid against igl's own hash.

    igl's ``unique_sparse_voxel_corners`` builds all ``8 n`` candidate subscripts, packs each into
    an ``int64`` code and runs ``igl::unique`` over them. triwarp builds none of that: the nanogrid
    geometry derives its vertex grid from the cell grid, and the per-cell indices are eight ``O(1)``
    probes. The geometry construction (~1.2 ms at 179k voxels) is the fixed cost on this row.
    """
    voxel_size = _voxel_size(bench_case, _MORPHOLOGY_DIVISOR)
    if bench_case.kind == "igl":
        skip_larger_than(bench_case, "bunny", "igl hashes 8 n candidate subscripts on one core")
        mesh_tm = tm.Trimesh(bench_case.vertices_np, bench_case.faces_np, process=False)
        grid_tm = tm.voxel.creation.voxelize_subdivide(mesh_tm, pitch=voxel_size)
        cells_np = np.ascontiguousarray(grid_tm.sparse_indices.astype(np.int64))
        depth = int(np.ceil(np.log2(cells_np.max() + 2)))
        origin_np = np.zeros(3)
        corners_igl = bench_case.run(
            lambda: igl.unique_sparse_voxel_corners(
                origin_np, voxel_size * (1 << depth), depth, cells_np
            )[0],
            rounds=_HEAVY_ROUNDS,
        )
        assert corners_igl.shape[0] > 0
        return

    grid = tw.voxels.voxelize_mesh(bench_case.vertices_wp, bench_case.faces_wp, voxel_size)
    corners, _cell_corners = bench_case.run(lambda: tw.voxels.voxel_corners(grid))
    assert int(corners.shape[0]) > 0


# Lattice resolutions for the mesh-free group. 256 cubed is 16.7 M samples, the size at which an
# SDF round trip is actually run.
_LATTICE_RESOLUTIONS = [64, 256]


@pytest.mark.benchmark(group="grid_points")
@pytest.mark.benchlibs("triwarp", "igl")
@pytest.mark.parametrize("resolution", _LATTICE_RESOLUTIONS)
def test_grid_points(bench_lib: BenchLibrary, resolution: int) -> None:
    """
    Lattice generation and nothing else, so this row reads as a memory-bandwidth floor.

    One of the few mesh-free groups in the suite (``bench_lib`` rather than ``bench_case``): the
    axis is resolution, and there is no input to speak of. ``igl.grid`` fills the same
    ``resolution ** 3`` by 3 array with a triple loop.
    """
    shape = (resolution, resolution, resolution)
    if bench_lib.kind == "igl":
        if resolution > 128:
            pytest.skip("igl fills 16.7 M rows with a serial triple loop")
        res_np = np.array(shape)
        lattice_igl = bench_lib.run(lambda: igl.grid(res_np), rounds=_HEAVY_ROUNDS)
        assert lattice_igl.shape[0] == resolution**3
        return

    device = bench_lib.device
    bounds = (wp.vec3(0.0, 0.0, 0.0), wp.vec3(1.0, 1.0, 1.0))
    lattice = bench_lib.run(lambda: tw.voxels.grid_points(shape, bounds=bounds, device=device))
    assert int(lattice.shape[0]) == resolution**3
