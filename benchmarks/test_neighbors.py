"""
Benchmarks for ``triwarp.neighbors``: the k-nearest-neighbour queries.

This is the direct gate for the k-NN search radius. Twelve call sites in ``triwarp.distance``, the
point-cloud ICP path in ``triwarp.registration`` and three sites in ``triwarp.reconstruction`` all
bottom out in these two functions, but until this module existed none of them had a measurement
that isolated the query kernel from the surrounding algorithm.

Setup
-----
``points`` is the mesh's own vertices; ``queries`` is a fixed 20 000-point subsample of them pushed
off the surface by 1% of the bbox diagonal along a fixed direction, so no query coincides exactly
with a data point (which would make ``k=1`` trivially certifiable at radius 0) and every query still
has neighbours at a realistic spacing. Both ``k=1`` and ``k=7`` are timed: ``k=1`` is what
``distance.py`` and ICP use, ``k=7`` is ``ball_pivoting``'s seed table.

Axes
----
Nothing here is driven by a mesh property — the meshes are used only as point clouds — so the axes
are the structure's own parameters, and there are three:

* **k**, expressed as separate groups (``_k1`` / ``_k7``) rather than as a parametrize sweep,
  because the two are different call sites rather than two points on a curve: ``k=1`` is what
  ``distance.py`` and ICP use, ``k=7`` is ``ball_pivoting``'s seed table and maintains a sorted
  candidate row. The two land in different generated kernels — the row is register-resident and one
  kernel exists per row-size bucket (``kernels.neighbors.KNN_ROW_BUCKETS``), so ``k=1`` and ``k=7``
  are not the same code. The larger ``k`` values in-repo (30 and 64) are timed by
  [`test_points.py`](test_points.py), where they are what the outlier statistics ask for.
* **build against query**. The k-NN groups above include the build, which is what a caller passing
  raw buffers pays. The ``*_from_points`` groups below time the build *alone*, so subtracting them
  answers the question that matters for ICP: whether its per-iteration index rebuild is the cost or
  a red herring.
* **structure resolution** — ``leaf_size`` for the BVH, ``grid_bins`` for the hash grid, and
  ``radius`` for the ball queries. A leaf too large makes a shallow tree that is cheap to build and
  slow to traverse; too few bins degenerates into a linear scan and too many pays for empty cells.
  Ball-query cost is the *expected neighbour count*, roughly ``density x radius^3``, and the
  two-phase count-then-fill means it is paid twice.

The ``*_with_offsets`` form is used for the ball queries rather than plain ``query_*_ball``: the
latter returns a Python ``list`` of per-query arrays, adding ``O(n_queries)`` of host slicing that
would swamp the kernel. That host cost is real but belongs to a different question.

What is inside the timed callable
---------------------------------
For the k-NN groups, everything the public function does, including the spatial-index build and the
bounds reduction — that is what a caller passing raw buffers actually pays. The index build is small
next to the query (a ``bunny`` BVH build is ~0.2 ms), but hiding it would misreport the API. The
build and ball groups instead pass a prebuilt structure where the signature allows one, so they
isolate the piece they name.

Reference
---------
**scipy** ``spatial.KDTree`` — the same reference ``tests/test_neighbors.py`` validates against, and
the only exact k-NN in the test group with a matching signature (``query(x, k=k)`` returns distances
and indices in the same layout). ``KDTree`` construction is inside the timed region for the same
reason triwarp's index build is.

**libigl** ``igl.knn`` is the second exact k-NN, and it *is* bound — an earlier version of this
docstring claimed it was not, which was wrong. It returns ``(n_queries, k)`` ``int64`` indices
sorted by distance, byte-identical to ``KDTree``'s on a tie-free cloud, so it is a genuine
independent implementation rather than a shape-compatible stand-in. Three things shape its rows:

* **Seven positional arguments, and the octree is built over the *data* cloud**:
  ``igl.knn(queries, points, k, *igl.octree(points)[:4])``. The build goes inside the timed
  callable, like scipy's ``KDTree`` and triwarp's own, and is *also* timed alone in
  ``bvh_from_points`` — where ``igl.octree`` is the only structure build the reference side of that
  group has ever had.
* **``igl.octree``'s cost is mostly Python objects, and its variance is entirely the garbage
  collector.** It returns its per-cell point lists as a **Python ``list`` of 151 233 nested lists**
  on ``bunny``'s 35 947 points (54 851 of them non-empty), so the binding allocates ~150k list
  objects per call. Measured back to back: **40.6-44.6 ms with ``gc`` disabled against 90.6-176.9 ms
  with it enabled** — a tight distribution becomes a 2-4x spread, which is why this row's ``Min``
  and ``Median`` differ by 5x where every other row in the module agrees to within 20%. Read the igl
  rows' *medians*, not their minima, and read the build-included k-NN numbers as pricing the binding
  as much as the search. It is the same hazard as the ``*_lists`` variants elsewhere in the suite,
  except here it lands on the only structure igl exposes for k-NN, so there is no array form to
  switch to.
* **It is capped at ``bunny``**, one step below scipy's ``dragon``, because that object churn grows
  with the cell count. In-harness medians on ``bunny`` at 20 000 queries: **250 / 340 ms** at ``k``
  = 1 / 64 build-included (minima 122 / 211), against triwarp's 0.65 / 2.65 ms. ``k > n_points``
  silently returns ``n_points`` columns rather than raising, which no row here hits but a future one
  might.

**open3d** ``o3d.core.nns.NearestNeighborSearch`` is the third exact k-NN, and it is **batched** --
an earlier version of this docstring pointed at the legacy ``KDTreeFlann``, whose only query is a
Python loop over ``search_knn_vector_3d`` (measured 62 ms against 10 ms for the batched
``knn_search`` at 20 000 queries over a 36k cloud, i.e. the loop times the interpreter). Probed
before these rows landed: ``knn_search`` returns indices byte-identical to ``KDTree``'s on a
tie-free cloud, distances come back **squared** (as FLANN's do), and ``fixed_radius_search``
returns a CSR-like ``(indices, distances, offsets)`` triple whose per-query counts matched
triwarp's ball counts exactly on a 36k random cloud. The index build (``knn_index`` /
``fixed_radius_index``) sits inside the timed callable for the k-NN groups, exactly like scipy's
``KDTree`` and triwarp's own build; the ball group passes a prebuilt index, like scipy's cached
tree. Note ``fixed_radius_index(radius)`` bakes the radius into the structure, so each radius
point pays its own build. trimesh has no k-NN entry point.

``nearest_neighbor_distance`` is the one group here whose queries are the cloud *itself* rather than
the fixed 20 000-point subsample, because that is how its only callers use it -- once over the whole
input, before ``reconstruction`` can pick a ball radius or an octree depth. Its open3d counterpart,
``compute_nearest_neighbor_distance``, builds a ``KDTreeFlann`` and then searches point by point in
C++, so it is serial but not a Python loop. Medians on the ``scale`` axis (CUDA against one core):
**0.391 / 0.606 / 1.20 ms** against **0.547 / 12.3 / 49.8 ms** at 2 562 / 40 962 / 163 842 points --
1.4x, 20x, 41x, the ratio growing because triwarp's is nearly flat over that range.
"""

from __future__ import annotations

import igl
import numpy as np
import pytest
import warp as wp
from conftest import BenchCase, skip_larger_than
from meshlib import mrmeshpy as mm
from scipy.spatial import KDTree

import triwarp as tw

_SEED = 42
_N_QUERIES = 20_000
_OFFSET_FRACTION = 0.01

# BVH leaf sizes and hash-grid bin counts: the structure-resolution knobs.
_LEAF_SIZES = [4, 32]
_GRID_BINS = [32, 256]

# Ball radii as multiples of the mean edge length. Expected neighbours grow ~cubically, so these
# two points are roughly 8x apart in work.
_RADIUS_SCALES = [2.0, 4.0]

_queries_np_cache: dict[str, np.ndarray] = {}
_queries_wp_cache: dict[tuple[str, str], wp.array] = {}
_bvh_cache: dict[tuple[str, str], wp.Bvh] = {}
_kdtree_cache: dict[str, KDTree] = {}
_pcd_o3d_cache: dict[str, object] = {}
_cloud_ml_cache: dict[str, mm.PointCloud] = {}
_queries_ml_cache: dict[str, mm.std_vector_Vector3_float] = {}


def _queries_np(bench_case: BenchCase) -> np.ndarray:
    """Subsample the mesh vertices at a fixed seed and displace them off the surface."""
    name = bench_case.mesh_name
    if name not in _queries_np_cache:
        rng = np.random.default_rng(_SEED)
        vertices = bench_case.vertices_np
        count = min(_N_QUERIES, vertices.shape[0])
        indices = rng.choice(vertices.shape[0], size=count, replace=False)
        diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
        offset = _OFFSET_FRACTION * diagonal * np.array([1.0, -1.0, 0.5]) / np.sqrt(2.25)
        _queries_np_cache[name] = np.ascontiguousarray(vertices[indices] + offset)
    return _queries_np_cache[name]


def _queries_wp(bench_case: BenchCase) -> wp.array[wp.vec3]:
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _queries_wp_cache:
        _queries_wp_cache[key] = wp.array(
            np.ascontiguousarray(_queries_np(bench_case), dtype=np.float32),
            dtype=wp.vec3,
            device=bench_case.device,
        )
    return _queries_wp_cache[key]


def _cloud_ml(bench_case: BenchCase) -> mm.PointCloud:
    """
    Wrap the mesh vertices in a ``PointCloud``, cached per mesh so the reference outlives its users.

    ``PointsProjector.setPointCloud`` stores a **raw pointer**: handing it a temporary cloud leaves
    it reading freed memory and segfaults on a cloud this size rather than raising, which is the
    same rule ``PointsToMeshProjector`` follows in ``test_proximity.py``. The cache is that
    reference, and it keeps the lazily built tree warm as well.
    """
    if bench_case.mesh_name not in _cloud_ml_cache:
        from meshlib import mrmeshnumpy as mn

        _cloud_ml_cache[bench_case.mesh_name] = mn.pointCloudFromPoints(
            np.ascontiguousarray(bench_case.vertices_np, dtype=np.float64)
        )
    return _cloud_ml_cache[bench_case.mesh_name]


def _queries_ml(bench_case: BenchCase) -> mm.std_vector_Vector3_float:
    """Build the query cloud as a MeshLib vector, cached: the fill is a per-point Python loop."""
    if bench_case.mesh_name not in _queries_ml_cache:
        queries_ml = mm.std_vector_Vector3_float()
        for query_np in _queries_np(bench_case):
            queries_ml.append(mm.Vector3f(*query_np.tolist()))
        _queries_ml_cache[bench_case.mesh_name] = queries_ml
    return _queries_ml_cache[bench_case.mesh_name]


def _bvh(bench_case: BenchCase) -> wp.Bvh:
    """Prebuilt BVH over the cloud -- an *input* for the ball group, timed on its own elsewhere."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _bvh_cache:
        _bvh_cache[key] = tw.neighbors.bvh_from_points(bench_case.vertices_wp)
    return _bvh_cache[key]


def _pcd_o3d(bench_case: BenchCase):
    """Open3D cloud over the same vertices, once per mesh: the pure query calls do not mutate it."""
    import open3d as o3d

    if bench_case.mesh_name not in _pcd_o3d_cache:
        _pcd_o3d_cache[bench_case.mesh_name] = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(bench_case.vertices_np)
        )
    return _pcd_o3d_cache[bench_case.mesh_name]


def _kdtree(bench_case: BenchCase) -> KDTree:
    """Prebuilt scipy KDTree over the same cloud, for the ball-query comparison."""
    if bench_case.mesh_name not in _kdtree_cache:
        _kdtree_cache[bench_case.mesh_name] = KDTree(bench_case.vertices_np)
    return _kdtree_cache[bench_case.mesh_name]


def _run_scipy(bench_case: BenchCase, k: int) -> None:
    queries_np = _queries_np(bench_case)
    points_np = bench_case.vertices_np
    distances_np, indices_np = bench_case.run(lambda: KDTree(points_np).query(queries_np, k=k))
    assert np.asarray(indices_np).shape[0] == queries_np.shape[0]
    assert np.asarray(distances_np).shape[0] == queries_np.shape[0]


def _run_igl_knn(bench_case: BenchCase, k: int) -> None:
    """``igl.knn`` with the octree build inside the timed region, as scipy's ``KDTree`` is."""
    skip_larger_than(bench_case, "bunny", "igl.octree allocates one Python list per octree cell")
    queries_np, points_np = _queries_np(bench_case), bench_case.vertices_np
    indices_igl = bench_case.run(
        lambda: igl.knn(queries_np, points_np, k, *igl.octree(points_np)[:4])
    )
    assert indices_igl.shape == (queries_np.shape[0], k)


def _run_o3d_nns_knn(bench_case: BenchCase, k: int) -> None:
    """Batched ``o3d.core.nns.knn_search`` with the index build inside the timed region."""
    import open3d as o3d

    queries_np, points_np = _queries_np(bench_case), bench_case.vertices_np
    queries_t = o3d.core.Tensor(queries_np)
    points_t = o3d.core.Tensor(points_np)

    def knn_o3d() -> tuple:
        nns = o3d.core.nns.NearestNeighborSearch(points_t)
        nns.knn_index()
        return nns.knn_search(queries_t, k)

    indices_o3d, _squared_o3d = bench_case.run(knn_o3d)
    assert indices_o3d.shape == (queries_np.shape[0], k)


@pytest.mark.benchmark(group="query_bvh_nearest_k1")
@pytest.mark.benchlibs("triwarp", "scipy", "igl", "open3d", "meshlib")
def test_query_bvh_nearest_k1(bench_case: BenchCase) -> None:
    """
    ``k=1`` BVH k-NN — the exact call ICP and the Chamfer family make.

    meshlib's ``PointsProjector`` is the only batched form it has that takes a *query* cloud, and
    it answers ``k=1`` only -- which is why meshlib appears in this group and not in the ``k7`` or
    ``k64`` ones (``tests/test_neighbors.py`` records that as a ``benchmarked=False`` claim). It is
    exact against triwarp on both indices and distances. The projector and its tree are built
    outside the timed callable, so the row prices the queries; the point cloud is held in a name
    because ``setPointCloud`` stores a raw pointer and a temporary segfaults.
    """
    skip_larger_than(bench_case, "dragon")
    if bench_case.kind == "meshlib":
        cloud_ml = _cloud_ml(bench_case)  # cached: the projector does not own it
        queries_ml = _queries_ml(bench_case)
        projector_ml = mm.PointsProjector()
        projector_ml.setPointCloud(cloud_ml)
        settings_ml = mm.FindProjectionOnPointsSettings()

        def nearest_ml() -> mm.std_vector_PointsProjectionResult:
            results_ml = mm.std_vector_PointsProjectionResult()
            projector_ml.findProjections(results_ml, queries_ml, settings_ml)
            return results_ml

        nearest_ml()  # pre-warm the lazily built tree
        assert len(bench_case.run(nearest_ml)) == _queries_np(bench_case).shape[0]
        return
    if bench_case.kind == "triwarp":
        points, queries = bench_case.vertices_wp, _queries_wp(bench_case)
        indices, _distances = bench_case.run(
            lambda: tw.neighbors.query_bvh_nearest(points, queries, k=1)
        )
        assert indices.shape == (queries.shape[0],)
    elif bench_case.kind == "igl":
        _run_igl_knn(bench_case, 1)
    elif bench_case.kind == "open3d":
        _run_o3d_nns_knn(bench_case, 1)
    else:
        _run_scipy(bench_case, 1)


@pytest.mark.benchmark(group="query_hashgrid_nearest_k1")
@pytest.mark.benchlibs("triwarp", "scipy")
def test_query_hashgrid_nearest_k1(bench_case: BenchCase) -> None:
    """``k=1`` hash-grid k-NN — the backend ``distance.py`` picks."""
    skip_larger_than(bench_case, "dragon")
    if bench_case.kind == "triwarp":
        points, queries = bench_case.vertices_wp, _queries_wp(bench_case)
        indices, _distances = bench_case.run(
            lambda: tw.neighbors.query_hashgrid_nearest(points, queries, k=1)
        )
        assert indices.shape == (queries.shape[0],)
    else:
        _run_scipy(bench_case, 1)


@pytest.mark.benchmark(group="query_bvh_nearest_k7")
@pytest.mark.benchlibs("triwarp", "scipy", "igl", "open3d")
def test_query_bvh_nearest_k7(bench_case: BenchCase) -> None:
    """``k=7`` BVH k-NN — ``ball_pivoting``'s seed-candidate table."""
    skip_larger_than(bench_case, "dragon")
    if bench_case.kind == "triwarp":
        points, queries = bench_case.vertices_wp, _queries_wp(bench_case)
        indices, _distances = bench_case.run(
            lambda: tw.neighbors.query_bvh_nearest(points, queries, k=7)
        )
        assert indices.shape == (queries.shape[0], 7)
    elif bench_case.kind == "igl":
        _run_igl_knn(bench_case, 7)
    elif bench_case.kind == "open3d":
        _run_o3d_nns_knn(bench_case, 7)
    else:
        _run_scipy(bench_case, 7)


@pytest.mark.benchmark(group="query_bvh_nearest_k64")
@pytest.mark.benchlibs("triwarp", "scipy", "igl", "open3d")
def test_query_bvh_nearest_k64(bench_case: BenchCase) -> None:
    """
    ``k=64`` BVH k-NN — the largest register-row bucket, and the k axis's far end.

    The candidate row costs ``2 * k`` registers, so this is the last ``k`` the row fits in them; at
    65 the search falls back to the global-memory-row kernel and the cost per neighbour jumps
    (measured 7.8x between the two at this ``k``). The group exists to keep that boundary visible:
    without it the suite's k axis stops at 7 and the regime the row storage governs is unmeasured.
    """
    skip_larger_than(bench_case, "dragon")
    if bench_case.kind == "triwarp":
        points, queries = bench_case.vertices_wp, _queries_wp(bench_case)
        indices, _distances = bench_case.run(
            lambda: tw.neighbors.query_bvh_nearest(points, queries, k=64)
        )
        assert indices.shape == (queries.shape[0], 64)
    elif bench_case.kind == "igl":
        _run_igl_knn(bench_case, 64)
    elif bench_case.kind == "open3d":
        _run_o3d_nns_knn(bench_case, 64)
    else:
        _run_scipy(bench_case, 64)


@pytest.mark.benchmark(group="query_hashgrid_nearest_k7")
@pytest.mark.benchlibs("triwarp", "scipy")
def test_query_hashgrid_nearest_k7(bench_case: BenchCase) -> None:
    """``k=7`` hash-grid k-NN."""
    skip_larger_than(bench_case, "dragon")
    if bench_case.kind == "triwarp":
        points, queries = bench_case.vertices_wp, _queries_wp(bench_case)
        indices, _distances = bench_case.run(
            lambda: tw.neighbors.query_hashgrid_nearest(points, queries, k=7)
        )
        assert indices.shape == (queries.shape[0], 7)
    else:
        _run_scipy(bench_case, 7)


@pytest.mark.benchmark(group="bvh_from_points")
@pytest.mark.benchlibs("triwarp", "scipy", "igl")
@pytest.mark.parametrize("leaf_size", _LEAF_SIZES)
def test_bvh_from_points(bench_case: BenchCase, leaf_size: int) -> None:
    """
    Structure build alone: the cost a caller amortizes, or fails to.

    Subtract this from ``query_bvh_nearest_k1`` to get the query in isolation. Neither reference
    takes a leaf-size parameter, so each one's two rows are identical by construction and are there
    as fixed bars; triwarp's own slope is the other half of the ``leaf_size`` trade-off.

    ``igl.octree`` is the structure ``igl.knn`` consumes, so this row is also what the k-NN groups'
    build-included numbers carry: ~198 ms of ``bunny``'s 250 ms at ``k=1``, against triwarp's
    0.22 ms — and most of that 198 is the 151 233 Python lists it returns rather than the tree, see
    the module docstring.
    """
    skip_larger_than(bench_case, "dragon", "the scipy reference builds single-threaded")
    if bench_case.kind == "triwarp":
        points = bench_case.vertices_wp
        bvh = bench_case.run(lambda: tw.neighbors.bvh_from_points(points, leaf_size=leaf_size))
        assert bvh is not None
    elif bench_case.kind == "igl":
        skip_larger_than(bench_case, "bunny", "igl.octree is superlinear in the point count")
        points_np = bench_case.vertices_np
        point_indices_igl, _, _, _ = bench_case.run(lambda: igl.octree(points_np))[:4]
        assert len(point_indices_igl) > 0
    else:
        points_np = bench_case.vertices_np
        assert bench_case.run(lambda: KDTree(points_np)) is not None


@pytest.mark.benchmark(group="hashgrid_from_points")
@pytest.mark.benchlibs("triwarp")
@pytest.mark.parametrize("grid_bins", _GRID_BINS)
def test_hashgrid_from_points(bench_case: BenchCase, grid_bins: int) -> None:
    """The hash-grid build, swept over its bin count: the other structure's amortization floor."""
    skip_larger_than(bench_case, "dragon")
    points = bench_case.vertices_wp
    radius = _RADIUS_SCALES[0] * bench_case.mean_edge
    grid = bench_case.run(
        lambda: tw.neighbors.hashgrid_from_points(points, radius, grid_bins=grid_bins)
    )
    assert grid is not None


@pytest.mark.benchmark(group="query_bvh_ball")
@pytest.mark.benchlibs("triwarp", "scipy", "open3d")
@pytest.mark.parametrize("radius_scale", _RADIUS_SCALES)
def test_query_bvh_ball(bench_case: BenchCase, radius_scale: float) -> None:
    """
    Radius query over a prebuilt BVH: cost is the neighbour count, so ~8x between the radii.

    All three structures are prebuilt: triwarp's BVH, scipy's cached ``KDTree``, and open3d's
    ``fixed_radius_index`` -- the last per radius, because that index bakes the radius in.
    ``fixed_radius_search`` returns a CSR-like triple, which is exactly the ``*_with_offsets``
    layout triwarp's row times, so neither side pays per-query host slicing.
    """
    skip_larger_than(bench_case, "bunny", "the neighbour count grows cubically with the radius")
    radius = radius_scale * bench_case.mean_edge
    if bench_case.kind == "triwarp":
        points, queries = bench_case.vertices_wp, _queries_wp(bench_case)
        bvh = _bvh(bench_case)
        neighbors, _distances, offsets = bench_case.run(
            lambda: tw.neighbors.query_bvh_ball_with_offsets(points, queries, radius, bvh=bvh)
        )
        assert offsets.shape[0] == int(queries.shape[0])
        assert neighbors.shape[0] >= 0
    elif bench_case.kind == "open3d":
        import open3d as o3d

        queries_t = o3d.core.Tensor(_queries_np(bench_case))
        nns = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(bench_case.vertices_np))
        assert nns.fixed_radius_index(radius)
        _indices_o3d, _squared_o3d, offsets_o3d = bench_case.run(
            lambda: nns.fixed_radius_search(queries_t, radius)
        )
        assert offsets_o3d.shape[0] == queries_t.shape[0] + 1
    else:
        tree, queries_np = _kdtree(bench_case), _queries_np(bench_case)
        found = bench_case.run(lambda: tree.query_ball_point(queries_np, radius))
        assert len(found) == queries_np.shape[0]


@pytest.mark.benchmark(group="query_hashgrid_ball")
@pytest.mark.benchlibs("triwarp")
@pytest.mark.parametrize("grid_bins", _GRID_BINS)
def test_query_hashgrid_ball(bench_case: BenchCase, grid_bins: int) -> None:
    """
    The hash-grid ball query, swept over the bin count at a fixed radius.

    Too few bins and each cell holds enough points that the query degenerates into a linear scan;
    too many and the build pays for cells nothing lands in. Where the optimum sits depends on the
    cloud's density, so this pair is the cheapest way to see which side of it the default is on.
    """
    skip_larger_than(bench_case, "bunny")
    radius = _RADIUS_SCALES[0] * bench_case.mean_edge
    points, queries = bench_case.vertices_wp, _queries_wp(bench_case)
    grid = tw.neighbors.hashgrid_from_points(points, radius, grid_bins=grid_bins)
    neighbors, _distances, offsets = bench_case.run(
        lambda: tw.neighbors.query_hashgrid_ball_with_offsets(points, queries, radius, grid=grid)
    )
    assert offsets.shape[0] == int(queries.shape[0])
    assert neighbors.shape[0] >= 0


@pytest.mark.benchmark(group="query_geodesic_ball")
@pytest.mark.benchlibs("triwarp")
def test_query_geodesic_ball(bench_case: BenchCase) -> None:
    """
    The mesh-graph ball: a BFS that enqueues a neighbour only when it is inside the radius.

    The one query here that is not a spatial one, and the reason it is not: a Euclidean
    hash-grid query at the same radius would also return vertices across a fold of the surface
    -- the opposite wall of a torus tube -- which is what corrupts the quadric fit in
    ``curvature.principal_curvature``, its only caller. So this row prices the surface-aware
    alternative to the ball groups above rather than a variant of them, and no reference
    library exposes it to time against.
    """
    skip_larger_than(bench_case, "happy_buddha")
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    radius = 5.0 * float(tw.edges.mean_edge_length(vertices, faces))
    _, offsets, _ = bench_case.run(lambda: tw.neighbors.geodesic_ball(vertices, faces, radius))
    assert offsets.shape == (vertices.shape[0],)


@pytest.mark.benchmark(group="nearest_neighbor_distance")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "open3d")
def test_nearest_neighbor_distance(bench_case: BenchCase) -> None:
    """
    The cloud's own spacing: a ``k=2`` self-query, keeping the second column.

    A *self*-query, so unlike every group above the query count is the cloud size rather than
    ``_N_QUERIES`` -- which is the point, since this is what ``reconstruction`` calls once on the
    whole input before it can pick a radius or an octree depth. Everything past the k-NN search is a
    strided column copy, so the ratio against ``query_bvh_nearest_k1`` prices that tail.

    open3d's ``compute_nearest_neighbor_distance`` builds a ``KDTreeFlann`` and then searches one
    point at a time in C++ -- so it is serial, but not the Python-per-query loop the legacy tree
    would be from this side, and it is the same quantity to 9e-09.
    """
    if bench_case.kind == "open3d":
        cloud = _pcd_o3d(bench_case)
        distance_o3d = bench_case.run(cloud.compute_nearest_neighbor_distance)
        assert len(distance_o3d) == bench_case.n_vertices
        return
    points = bench_case.vertices_wp
    distance = bench_case.run(lambda: tw.neighbors.nearest_neighbor_distance(points))
    assert distance.shape == (bench_case.n_vertices,)
