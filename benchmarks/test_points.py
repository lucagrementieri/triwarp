"""
Benchmarks for ``triwarp.points``.

The point cloud is a registry mesh's own vertices — deterministic, and it scales with the mesh. The
functions here fall into two groups:

* **Tiled reductions** — ``fit_line`` and ``fit_plane`` accumulate a 3x3 scatter matrix with
  ``outer_sum_chunk`` and finish with a single-thread ``wp.svd3``. Both end in a host readback of
  the resulting axis / normal, so the numbers include one synchronisation by construction.
* **Element-wise maps and sorts** — ``point_plane_distance`` and ``vector_angle`` are one
  ``wp.map`` each (the cheapest thing in this module, so they are the most sensitive to launch
  overhead), and ``radial_sort`` is a key kernel plus a radix sort.

``estimate_normals`` is timed twice, because the public function takes the neighbour table as an
*input*:

* ``estimate_normals`` — the PCA kernel alone, on a cached ``(n, k)`` table. This is the
  measurement to read for a change to the covariance / SVD kernel.
* ``estimate_normals_knn`` — [`query_nearest`][triwarp.neighbors.query_nearest] plus the
  kernel, which is the whole of what open3d's ``estimate_normals`` does (it builds a ``KDTreeFlann``
  internally on every call), so it is the only fair cross-library comparison.

References
----------
**trimesh** covers everything except normal estimation: ``trimesh.points.point_plane_distance``,
``trimesh.points.major_axis``, ``trimesh.points.plane_fit``, ``trimesh.points.radial_sort`` and
``trimesh.geometry.vector_angle`` are the exact functions triwarp's are ports of.

**open3d** is the reference for ``estimate_normals``: ``PointCloud.estimate_normals`` with
``KDTreeSearchParamKNN`` is the same k-nearest PCA estimator (its ``FastEigen3x3`` picks the
smallest-eigenvalue eigenvector, as triwarp's ``wp.svd3`` path does).

**pymeshlab** covers three: ``compute_normal_for_point_clouds(k=)`` is the same k-nearest PCA
estimator at the same ``k`` (and it exists for exactly this case -- a dataset with no faces -- so
its input is a *face-less* MeshSet); ``compute_matrix_by_fitting_to_plane`` is the ``fit_plane``
counterpart, reporting the fitted normal and the average fitting error. That one has a
precondition: it raises ``Cannot compute rotation: there is no selection`` unless something is
selected, so ``set_selection_all`` runs first, untimed -- it is how the filter is told "fit all the
points", not part of the fit. It also builds a rotation matrix onto a target plane, which triwarp
does not, so its row is an upper bound. Third, ``compute_selection_point_cloud_outliers`` is the
LoOP score behind ``outlier_probability``, at the same ``knearest``.

The outlier groups are timed the same way as ``estimate_normals_knn``: the neighbour table is an
*input* of triwarp's functions and is built inside the timed callable, because both references build
their own k-d tree per call and there would otherwise be nothing to compare. ``open3d`` covers the
statistical variant (``remove_statistical_outlier``) and the radius one (``remove_radius_outlier``),
both of which additionally *copy* the surviving points into a new cloud -- triwarp returns a mask,
so those rows are upper bounds.

Three groups have no triwarp-side neighbour table to hoist, because they take the cloud directly:
``radius_outlier_mask``, ``point_duplicate_mask`` and ``farthest_point_sample``. Medians on
``sphere_med`` (40 962 points, RTX 5090, CUDA against open3d's one core):

| group | triwarp | open3d | ratio |
|---|---|---|---|
| ``radius_outlier_mask`` at 2 / 4 mean edges | 0.233 / 0.361 ms | 14.6 / 19.2 ms | 63x / 53x |
| ``point_duplicate_mask`` | 1.40 ms | 6.18 ms | 4.4x (15x on ``sphere_large``) |
| ``farthest_point_sample`` at 64 / 1024 | 1.92 / 32.3 ms | 2.94 / 41.9 ms | 1.5x / 1.3x |

The last row is the one to read carefully, and it is stale by two rewrites. Those numbers are from
when the greedy loop was ``Theta(count)`` *launches* over the whole cloud -- ~2 000 launches of
marshalling and essentially no kernel time at ``count = 1024``. Capturing one round and replaying it
took the row to 21.1 / 4.32 ms (``sphere_med`` / ``sphere_small``), and the whole sweep is now **one
persistent block** (``kernels/points.py::farthest_point_sample_block``): 9.5-10.1 / 1.11 ms, so
``sphere_small`` at ``count = 1024`` reads 1.11 against open3d's 2.28 and wins where the
launch-bound
form lost 4x. The count is still the axis -- it is the number of dependent rounds -- but a round now
costs about a microsecond of block time rather than a launch.

MeshLab has nothing for ``fit_line`` / ``major_axis``, ``point_plane_distance``, ``vector_angle`` or
``radial_sort``: those are array primitives rather than filters.

**libigl** has no equivalent for anything in this module — it is a mesh library, and its
point-cloud entry points (``igl.fit_plane`` does not exist in the Python bindings) are not exposed.

Caps
----
Both ``estimate_normals`` groups are capped at ``bunny``. The neighbour table is what costs: at
``k = 30`` triwarp's k-NN kernel is the dominant term (see
[`test_neighbors.py`](test_neighbors.py), which times that kernel on its own), and open3d's serial
``KDTreeFlann`` scales the same way.

``fit_line``'s **trimesh** case is capped at ``bunny`` as well, and for a different reason:
``trimesh.points.major_axis`` calls ``numpy.linalg.svd`` on the ``(n, 3)`` point matrix with the
default ``full_matrices=True``, so it materializes the full ``(n, n)`` left-singular matrix — 1.39
**TiB** for dragon's 437,645 points, which raises ``numpy._core._exceptions._ArrayMemoryError``. The
triwarp side never forms that matrix (it accumulates a 3x3 Gram matrix and calls ``wp.svd3`` on it),
so it runs the full registry. ``fit_plane`` needs no such cap — ``trimesh.points.plane_fit`` does
not take the full-matrices path.

Everything else in the module runs the full registry.

Approximate hull
----------------
``convex_subset_mask`` / ``convex_subset`` / ``convex_superset_mask`` moved here with the ``convex``
module's point-cloud half. For each of ``n_directions`` Fibonacci hemisphere directions, one thread
per strided slice of the cloud reduces the support function and one atomic per thread combines the
slices, then a second pass marks the extrema. Cost is ``n_points * n_directions``, so this is the
compute-bound case in the file. ``convex_subset`` is the mask plus a ``flatnonzero`` and a gather,
so its delta over the mask is the compaction cost. (This sweep used to be a ``TILE_1D``-wide
``wp.tile_max`` / ``wp.tile_min`` block reduction. It is lane-free now because ``wp.launch_tiled``
runs exactly one lane per block on Warp 1.17's CPU backend, which made every tiled formulation
silently wrong there; the replacement also measured 1.0-2.7x *faster* on CUDA, the gap widening
with ``n_points * n_directions``.) ``convex_superset_mask`` adds a second cost shape: after the same
support sweep over an icosphere's directions, one pass tests every point against the
``20 * 4 ** subdivisions`` tetrahedra spanned by the resulting shell, and there every thread reads
the same tetrahedron's face planes at the same time, so the plane table is broadcast out of cache
and the arithmetic dominates.

**The baselines compute a different (and stronger) result**, and that is the point of the comparison
rather than a flaw in it. ``trimesh.Trimesh.convex_hull`` and
``open3d.geometry.TriangleMesh.compute_convex_hull`` both run **qhull**, producing the exact hull as
a *mesh* -- full connectivity, exact vertex set. ``convex_subset`` produces only an approximate
*vertex subset* (a support sweep over finitely many directions, which misses hull vertices whose
normal cone no sampled direction enters). So this is not a parity comparison: it is the
quantification of what the approximation buys, which is the reason the function exists.
``convex_subset_mask``'s own docstring documents the accuracy side of that trade -- including the
measured recall per ``n_directions`` and the normal-cone sizes that explain it; this is the cost
side. The argument is about the *operation*, so ``convex_subset`` and ``convex_subset_mask`` take
the same three qhull rows.

**scipy** is registered for ``convex_superset_mask`` only, and there it is a genuine parity row
rather than a bar. That filter exists to run *before* an exact hull, so ``scipy.spatial.ConvexHull``
on the same cloud is exactly the cost it has to be cheap against, and its output provably contains
that hull's vertex set (asserted in ``tests/test_points.py``). The ratio is the number that decides
whether the prefilter is worth running: on ``dragon`` it measured 2.4 ms at ``subdivisions=1`` and
7.6 ms at 3, against 145 ms for the hull itself -- 60x and 19x -- and the gap widens with the point
count, because the filter is linear where qhull is not. For the two approximate-hull groups scipy
would only be a third timing of the qhull already covered by trimesh and Open3D, so it stays out of
those. **pymeshlab**'s ``generate_convex_hull`` is qhull a third time, so it adds no new algorithm
-- what it adds is a *second* wrapper cost around the same computation, which is the only way to
tell whether trimesh's number is qhull or trimesh. **libigl** has no convex-hull binding in the
Python package, so igl is absent from all three.

The hull cases take the mesh's own vertices as the point cloud, so they scale with the registry mesh
sizes. ``n_directions`` and ``subdivisions`` are each swept over two values spanning their useful
range; both costs are close to linear in the direction count, so two points fix the line.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pymeshlab as ml
import pytest
import pyvista as pv
import scipy.spatial
import trimesh as tm
import warp as wp
from meshlib import mrmeshpy as mm

import triwarp as tw
import triwarp.typing as twt
from conftest import BenchCase, skip_larger_than

# Neighbour count for the PCA normal estimate (open3d's own default for KDTreeSearchParamKNN).
_KNN = 30

# Neighbourhood widths to sweep: the per-point 3x3 PCA is linear in k, and the k-NN table build
# that feeds it is not, so the pair separates the estimator's cost from its input's. The table is
# cached (``_neighbor_table``), so this sweep times the estimator *only* — the query's own k axis is
# ``query_nearest_bvh_k1`` / ``_k7`` / ``_k64`` in ``test_neighbors.py``.
_KNN_SWEEP = [8, 64]

# The trimesh references here are single-threaded host passes and one of them is far worse than
# single-threaded: ``tm.points.fit_line`` measured **22 s a call** on ``bunny``'s 35 947 points
# against 8 s for the whole 11-round case on ``bunny_decimated``'s 8 171. That one reference was 91%
# of this module's wall clock, so every host branch is capped at the smallest scan mesh -- the ratio
# against triwarp is four orders of magnitude and needs no larger input to establish.
_HOST_CAP_REASON = "host reference is a single-threaded pass; capped at bunny_decimated"

# Radius sweep for ``radius_outlier_mask``, in mean edge lengths. The ball count is linear in how
# many points each ball holds, so a 2x radius is ~8x the point tests -- the two points bracket where
# the hash grid stops paying for the extra cells.
_RADIUS_SCALES = [2.0, 4.0]

# Density floor for the same group, held fixed so the sweep is the radius alone. Open3D's own
# ``nb_points`` semantics: a point survives when it has strictly more than this many neighbours,
# itself included.
_MIN_NEIGHBORS = 6

# Sample counts for ``farthest_point_sample``. The greedy sweep is Theta(count) dependent rounds of
# one persistent block, so this axis is the round count and the step between the two is 16x.
_SAMPLE_COUNTS = [64, 1024]

# Fixed plane / sort axis, deliberately not axis-aligned so no branch is skipped.
_PLANE_NORMAL = np.array([0.3, -0.6, 0.74])

_neighbors_cache: dict[tuple[str, str, int], twt.Array2dInt32] = {}
_normals_wp_cache: dict[tuple[str, str], wp.array[wp.vec3]] = {}
_pcd_cache: dict[str, o3d.geometry.PointCloud] = {}
_cloud_pml_cache: dict[str, ml.MeshSet] = {}


def _neighbor_table(bench_case: BenchCase, k: int = _KNN) -> twt.Array2dInt32:
    """``(n, k)`` k-nearest table over the cloud itself — an *input* of ``estimate_normals``."""
    key = (bench_case.mesh_name, str(bench_case.device), k)
    if key not in _neighbors_cache:
        points = bench_case.vertices_wp
        _neighbors_cache[key] = tw.neighbors.query_nearest(points, points, k=k, backend="bvh")[0]
    return _neighbors_cache[key]


def _unit_normals_wp(bench_case: BenchCase) -> wp.array[wp.vec3]:
    """Area-weighted unit vertex normals: the left operand of the ``vector_angle`` pairs."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _normals_wp_cache:
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        _normals_wp_cache[key] = tw.vertices.vertex_normals(vertices, faces)
    return _normals_wp_cache[key]


def _unit_directions_np(bench_case: BenchCase) -> np.ndarray:
    """Build unit vectors from the cloud centroid to each point (right operand of the pairs)."""
    vertices = bench_case.vertices_np
    directions = vertices - vertices.mean(axis=0)
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def _pcd(bench_case: BenchCase) -> o3d.geometry.PointCloud:
    """Open3D point cloud over the same vertices, built once per mesh (setup, not timed)."""
    name = bench_case.mesh_name
    if name not in _pcd_cache:
        _pcd_cache[name] = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(bench_case.vertices_np)
        )
    return _pcd_cache[name]


@pytest.mark.benchmark(group="point_plane_distance")
@pytest.mark.benchlibs("triwarp", "trimesh", "pyvista")
def test_point_plane_distance(bench_case: BenchCase) -> None:
    """
    Signed point-to-plane distance of every point: a single ``wp.map`` over the cloud.

    The thinnest kernel in the module, so it is the module's floor row: below roughly ``10 ** 3``
    points it reports the ~340 µs wrapper floor rather than the map (see
    ``test_creation::test_box``), and being a ~50 µs GPU row it also has the widest run-to-run
    spread in the suite -- measured at 46x on unchanged code across two processes. Only read it as
    part of its axis.

    pyvista's ``compute_implicit_distance`` evaluates VTK's plane implicit function over the cloud,
    which is the same signed dot product -- measured 1.27e-07 from the exact float64 answer
    against triwarp's 2.11e-07, i.e. both at their own storage precision
    (``tests/test_points.py``). It needs
    a ``pv.Plane`` **large enough to span the cloud**: the implicit function is unbounded but the
    plane object carries an extent, and it is also the reason this row builds the plane outside the
    timed callable -- it is the query's parameter, not its input.
    """
    n_points = bench_case.n_vertices
    if bench_case.kind == "pyvista":
        skip_larger_than(bench_case, "bunny_decimated", _HOST_CAP_REASON)
        cloud_pv = pv.PolyData(bench_case.vertices_np)
        extent = 4.0 * float(
            np.linalg.norm(bench_case.vertices_np.max(axis=0) - bench_case.vertices_np.min(axis=0))
        )
        plane_pv = pv.Plane(
            center=(0.0, 0.0, 0.0), direction=_PLANE_NORMAL.tolist(), i_size=extent, j_size=extent
        )
        distances_pv = bench_case.run(lambda: cloud_pv.compute_implicit_distance(plane_pv))
        assert distances_pv.point_data["implicit_distance"].shape[0] == n_points
        return
    if bench_case.kind == "triwarp":
        points = bench_case.vertices_wp
        normal = wp.vec3(*_PLANE_NORMAL.tolist())
        distances = bench_case.run(lambda: tw.points.point_plane_distance(points, normal))
        assert distances.shape == (n_points,)
    else:
        skip_larger_than(bench_case, "bunny_decimated", _HOST_CAP_REASON)
        points_np = bench_case.vertices_np
        distances_tm = bench_case.run(
            lambda: tm.points.point_plane_distance(points_np, _PLANE_NORMAL)
        )
        assert distances_tm.shape == (n_points,)


@pytest.mark.noparity(
    "pyvista",
    oracle="trimesh",
    reason="D2 a different estimator with a measured disagreement: pv.fit_line_to_points returns "
    "the first *principal* axis of the centred covariance, while fit_line is a faithful port of "
    "trimesh.points.major_axis -- normalize(S @ V) over the SVD of the *uncentered* point matrix, "
    "a singular-value-weighted sum of all three right singular vectors. Measured |dot| between the "
    "two of 0.802 on an aspect-3:1:0.05 cloud offset from the origin, where pyvista agrees with "
    "the leading eigenvector to 1.0000000 and triwarp does not; on a 1000:1 needle the two "
    "definitions coincide to 3e-6, which is why the difference is easy to miss. The principal "
    "frame is a *separate* triwarp function -- points.principal_axes, whose own group carries the "
    "pyvista row -- so this is a naming coincidence rather than two implementations of one "
    "quantity. trimesh is the oracle here, in tests/test_points.py::test_fit_line, and the "
    "distinction is pinned in test_principal_axes_is_not_fit_line.",
)
@pytest.mark.benchmark(group="fit_line")
@pytest.mark.benchlibs("triwarp", "trimesh", "pyvista")
def test_fit_line(bench_case: BenchCase) -> None:
    """
    Major axis from the uncentred Gram matrix: tiled outer-product sum plus one ``svd3``.

    pyvista's row computes a *different* axis -- see the exemption above -- and additionally returns
    the fitted segment as a ``PolyData`` rather than a direction, so read it as the cost of "fit a
    line to this cloud" in VTK rather than as the same arithmetic.
    """
    if bench_case.kind == "pyvista":
        skip_larger_than(bench_case, "bunny_decimated", _HOST_CAP_REASON)
        points_np = bench_case.vertices_np
        line_pv = bench_case.run(lambda: pv.fit_line_to_points(points_np))
        assert np.asarray(line_pv.points).shape[1] == 3
        return
    if bench_case.kind == "triwarp":
        points = bench_case.vertices_wp
        axis = bench_case.run(lambda: tw.points.fit_line(points))
        assert len(axis) == 3
    else:
        skip_larger_than(bench_case, "bunny_decimated", _HOST_CAP_REASON)
        # major_axis SVDs with full_matrices=True -> an (n, n) allocation; see the module docstring.
        skip_larger_than(bench_case, "bunny", "trimesh's major_axis allocates an (n, n) SVD matrix")
        points_np = bench_case.vertices_np
        axis_tm = bench_case.run(lambda: tm.points.major_axis(points_np))
        assert axis_tm.shape == (3,)


_cloud_ml_cache: dict[str, mm.PointCloud] = {}
_points_ml_cache: dict[str, mm.std_vector_Vector3_float] = {}


def _cloud_ml(bench_case: BenchCase) -> mm.PointCloud:
    """
    Wrap the vertices in a ``meshlib.PointCloud``, cached per mesh.

    Cached because it is the *input*, and because MeshLib's projector-style objects keep a raw
    pointer to the cloud they are given -- a temporary is a segfault, not an exception (see
    ``test_proximity.py``'s ``closest_point_on_mesh`` row).
    """
    if bench_case.mesh_name not in _cloud_ml_cache:
        from meshlib import mrmeshnumpy as mn

        _cloud_ml_cache[bench_case.mesh_name] = mn.pointCloudFromPoints(
            np.ascontiguousarray(bench_case.vertices_np, dtype=np.float64)
        )
    return _cloud_ml_cache[bench_case.mesh_name]


def _points_ml(bench_case: BenchCase) -> mm.std_vector_Vector3_float:
    """Build the vertices as a MeshLib vector, cached: the fill is a per-point Python loop."""
    if bench_case.mesh_name not in _points_ml_cache:
        points_ml = mm.std_vector_Vector3_float()
        for point_np in bench_case.vertices_np:
            points_ml.append(mm.Vector3f(*point_np.tolist()))
        _points_ml_cache[bench_case.mesh_name] = points_ml
    return _points_ml_cache[bench_case.mesh_name]


@pytest.mark.benchmark(group="half_space_mask")
@pytest.mark.benchlibs("triwarp", "meshlib", "pyvista")
def test_half_space_mask(bench_case: BenchCase) -> None:
    """
    Select every point on one side of a plane: the same ``wp.map`` shape as the distance row.

    Read next to ``point_plane_distance``, which is the same pass writing a ``float32`` instead of a
    ``bool`` -- so the pair prices the output width and nothing else, and both sit on this module's
    wrapper floor rather than on any real arithmetic.

    meshlib's ``findHalfSpacePoints`` is the one query in family H that needs no callback: it takes
    the whole cloud and returns a ``VertBitSet``, so this row compares two batched calls rather than
    pricing an interpreter loop. Its plane is ``dot(n, x) = d`` where triwarp takes a normal and a
    point on the plane, and the packed bitset is a 64x narrower write than a ``wp.bool`` array --
    both worth knowing before reading the ratio.

    First measurement: at ``bunny_decimated`` meshlib is **23.8 us** against triwarp-cuda's
    **69.7 us**, i.e. the GPU row is 2.9x *behind* -- which is this module's wrapper floor rather
    than the map, exactly as ``point_plane_distance`` warns. At ``dragon`` triwarp-cuda is 66 us for
    a cloud 60x larger, so the floor is the whole story below ~10^5 points and the axis is the only
    honest way to read either row.

    pyvista reaches the mask through the same ``compute_implicit_distance`` the
    ``point_plane_distance`` row times, plus one host threshold -- so read the two pyvista rows as
    that threshold's cost, which is the same thing this pair measures on triwarp's side. The
    threshold is inside the timed callable for that reason.
    """
    n_points = bench_case.n_vertices
    if bench_case.kind == "pyvista":
        skip_larger_than(bench_case, "bunny_decimated", _HOST_CAP_REASON)
        cloud_pv = pv.PolyData(bench_case.vertices_np)
        extent = 4.0 * float(
            np.linalg.norm(bench_case.vertices_np.max(axis=0) - bench_case.vertices_np.min(axis=0))
        )
        plane_pv = pv.Plane(
            center=(0.0, 0.0, 0.0), direction=_PLANE_NORMAL.tolist(), i_size=extent, j_size=extent
        )
        mask_pv = bench_case.run(
            lambda: (
                np.asarray(cloud_pv.compute_implicit_distance(plane_pv)["implicit_distance"]) > 0.0
            )
        )
        assert mask_pv.shape[0] == n_points
        return
    if bench_case.kind == "meshlib":
        skip_larger_than(bench_case, "bunny_decimated", _HOST_CAP_REASON)
        cloud_ml = _cloud_ml(bench_case)  # held in a name, per the helper's docstring
        plane_ml = mm.Plane3f(mm.Vector3f(*_PLANE_NORMAL.tolist()), 0.0)
        mask_ml = bench_case.run(lambda: mm.findHalfSpacePoints(cloud_ml, plane_ml))
        assert mask_ml.size() <= n_points
        return
    points = bench_case.vertices_wp
    normal = wp.vec3(*_PLANE_NORMAL.tolist())
    mask = bench_case.run(lambda: tw.points.half_space_mask(points, normal))
    assert mask.shape == (n_points,)


@pytest.mark.benchmark(group="principal_axes")
@pytest.mark.benchlibs("triwarp", "pyvista", "meshlib")
def test_principal_axes(bench_case: BenchCase) -> None:
    """
    Principal frame: centroid reduction, centred scatter, then one 3x3 SVD.

    meshlib splits the same work across two calls -- ``accumulatePoints`` builds the scatter matrix
    and ``getCenteredCovarianceEigen`` decomposes it -- and only the pair is comparable, so both are
    inside the timed callable. The ``std_vector_Vector3_float`` the accumulator consumes is *not*:
    filling it is a per-point Python loop, which is the input rather than the fit.
    """
    if bench_case.kind == "meshlib":
        skip_larger_than(bench_case, "bunny_decimated", _HOST_CAP_REASON)
        points_ml = _points_ml(bench_case)

        def principal_axes_ml() -> mm.Vector3d:
            accumulator_ml = mm.PointAccumulator()
            mm.accumulatePoints(accumulator_ml, points_ml)
            centroid_ml, eigenvectors_ml = mm.Vector3d(), mm.Matrix3d()
            eigenvalues_ml = mm.Vector3d()
            accumulator_ml.getCenteredCovarianceEigen(centroid_ml, eigenvectors_ml, eigenvalues_ml)
            return eigenvalues_ml

        assert bench_case.run(principal_axes_ml).z > 0.0
        return
    if bench_case.kind == "triwarp":
        points = bench_case.vertices_wp
        rotation, eigenvalues, centroid = bench_case.run(lambda: tw.points.principal_axes(points))
        assert len(rotation) == 3
        assert len(eigenvalues) == 3
        assert len(centroid) == 3
    else:
        skip_larger_than(bench_case, "bunny_decimated", _HOST_CAP_REASON)
        points_np = bench_case.vertices_np
        axes_pv = bench_case.run(lambda: pv.principal_axes(points_np))
        assert axes_pv.shape == (3, 3)


@pytest.mark.benchmark(group="fit_plane")
@pytest.mark.benchlibs("triwarp", "trimesh", "pymeshlab", "pyvista", "meshlib")
def test_fit_plane(bench_case: BenchCase) -> None:
    """
    Least-squares plane: centroid reduction, centred covariance, then the smallest-sigma axis.

    pyvista's ``fit_plane_to_points(return_meta=True)`` returns the normal, a centre and a whole
    ``PolyData`` of the plane itself, so its row carries that geometry as well as the fit -- and its
    centre is *not* the centroid, which is why only the normal is compared
    (``tests/test_points.py``).
    """
    if bench_case.kind == "meshlib":
        # The same accumulator as the principal_axes row, read through getBestPlanef instead: it is
        # the cheaper half, since the plane needs no eigen-decomposition of its own.
        skip_larger_than(bench_case, "bunny_decimated", _HOST_CAP_REASON)
        points_ml = _points_ml(bench_case)

        def fit_plane_ml() -> mm.Plane3f:
            accumulator_ml = mm.PointAccumulator()
            mm.accumulatePoints(accumulator_ml, points_ml)
            return accumulator_ml.getBestPlanef()

        assert abs(bench_case.run(fit_plane_ml).n.length() - 1.0) < 1e-5
        return
    if bench_case.kind == "pyvista":
        skip_larger_than(bench_case, "bunny_decimated", _HOST_CAP_REASON)
        points_np = bench_case.vertices_np
        _plane_pv, _centre_pv, normal_pv = bench_case.run(
            lambda: pv.fit_plane_to_points(points_np, return_meta=True)
        )
        assert np.asarray(normal_pv).shape == (3,)
        return
    if bench_case.kind == "pymeshlab":
        # ``compute_matrix_by_fitting_to_plane`` raises ``Cannot compute rotation: there is no
        # selection`` unless something is selected, so ``set_selection_all`` runs first (untimed --
        # it is how the filter is told "fit all the points", not part of the fit). It returns the
        # fitted normal and the average fitting error, which is triwarp's answer plus a residual;
        # the rotation matrix it also builds is the part triwarp does not do.
        meshset_pml = bench_case.meshset_pml
        meshset_pml.set_selection_all()
        result_pml = bench_case.run(
            lambda: meshset_pml.compute_matrix_by_fitting_to_plane(targetplane="XY plane")
        )
        assert result_pml["fitting_plane_normal"].shape == (3,)
        return
    if bench_case.kind == "triwarp":
        points = bench_case.vertices_wp
        normal, centroid = bench_case.run(lambda: tw.points.fit_plane(points))
        assert len(centroid) == 3
        assert len(normal) == 3
    else:
        skip_larger_than(bench_case, "bunny_decimated", _HOST_CAP_REASON)
        points_np = bench_case.vertices_np
        centroid_tm, normal_tm = bench_case.run(lambda: tm.points.plane_fit(points_np))
        assert centroid_tm.shape == (3,)
        assert normal_tm.shape == (3,)


@pytest.mark.benchmark(group="vector_angle")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_vector_angle(bench_case: BenchCase) -> None:
    """Unsigned angle between paired unit vectors: the ``acos``-of-dot map."""
    n_points = bench_case.n_vertices
    if bench_case.kind == "triwarp":
        normals = _unit_normals_wp(bench_case)
        directions = wp.array(
            np.ascontiguousarray(_unit_directions_np(bench_case), dtype=np.float32),
            dtype=wp.vec3,
            device=bench_case.device,
        )
        angles = bench_case.run(lambda: tw.points.vector_angle(normals, directions))
        assert angles.shape == (n_points,)
    else:
        skip_larger_than(bench_case, "bunny_decimated", _HOST_CAP_REASON)
        mesh_tm = tm.Trimesh(bench_case.vertices_np, bench_case.faces_np, process=False)
        pairs_np = np.stack(
            (np.array(mesh_tm.vertex_normals), _unit_directions_np(bench_case)), axis=1
        )
        angles_tm = bench_case.run(lambda: tm.geometry.vector_angle(pairs_np))
        assert angles_tm.shape == (n_points,)


@pytest.mark.benchmark(group="radial_sort")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_radial_sort(bench_case: BenchCase) -> None:
    """Order points by angle about an axis: the key kernel plus a radix sort."""
    n_points = bench_case.n_vertices
    origin_np = bench_case.vertices_np.mean(axis=0)
    if bench_case.kind == "triwarp":
        points = bench_case.vertices_wp
        origin = wp.vec3(*origin_np.tolist())
        normal = wp.vec3(*_PLANE_NORMAL.tolist())
        ordered = bench_case.run(lambda: tw.points.radial_sort(points, origin, normal))
        assert ordered.shape == (n_points,)
    else:
        skip_larger_than(bench_case, "bunny_decimated", _HOST_CAP_REASON)
        points_np = bench_case.vertices_np
        ordered_tm = bench_case.run(
            lambda: tm.points.radial_sort(points_np, origin=origin_np, normal=_PLANE_NORMAL)
        )
        assert ordered_tm.shape == (n_points, 3)


@pytest.mark.benchmark(group="estimate_normals")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp")
@pytest.mark.parametrize("k", _KNN_SWEEP)
def test_estimate_normals(bench_case: BenchCase, k: int) -> None:
    """
    PCA normals from a cached neighbour table: the covariance accumulation and ``svd3``.

    The table is an *input*, so this group is linear in ``k`` and nothing else. Comparing its slope
    against ``estimate_normals_knn`` below separates the estimator from the search that feeds it --
    which matters, because the search is where the time actually goes.
    """
    points = bench_case.vertices_wp
    neighbors = _neighbor_table(bench_case, k)
    normals = bench_case.run(lambda: tw.points.estimate_normals(points, neighbors))
    assert normals.shape == (bench_case.n_vertices,)


def _cloud_meshset_pml(bench_case: BenchCase) -> ml.MeshSet:
    """
    Build the same vertices as a *face-less* pymeshlab mesh, once per mesh.

    ``compute_normal_for_point_clouds`` exists precisely for datasets with no faces, and writing the
    vertex normals leaves the positions alone, so this is cached rather than rebuilt per round.
    """
    if bench_case.mesh_name not in _cloud_pml_cache:
        meshset_pml = ml.MeshSet()
        meshset_pml.add_mesh(
            ml.Mesh(vertex_matrix=np.ascontiguousarray(bench_case.vertices_np, dtype=np.float64))
        )
        _cloud_pml_cache[bench_case.mesh_name] = meshset_pml
    return _cloud_pml_cache[bench_case.mesh_name]


@pytest.mark.benchmark(group="estimate_normals_knn")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "open3d", "pymeshlab", "meshlib")
def test_estimate_normals_knn(bench_case: BenchCase) -> None:
    """
    Neighbour search plus PCA -- what open3d's ``estimate_normals`` does in one call.

    meshlib's ``makeUnorientedNormals`` searches by **radius** where the other three take a
    neighbour count, so its row is given the radius that holds ``_KNN`` points at this cloud's
    density (2 mean spacings) rather than a count; the two see the same neighbourhood only where the
    cloud is uniform, which is what the agreement in ``tests/test_points.py`` is measured on. It is
    also the only row here that returns a *new* array rather than writing into the cloud, so nothing
    is mutated and one cloud serves every round.
    """
    if bench_case.mesh_name == "sphere_large":
        pytest.skip("open3d searches one point at a time; capped at sphere_med")
    if bench_case.kind == "meshlib":
        cloud_ml = _cloud_ml(bench_case)
        radius = 2.0 * bench_case.mean_edge
        normals_ml = bench_case.run(lambda: mm.makeUnorientedNormals(cloud_ml, radius))
        assert normals_ml.size() == bench_case.n_vertices
        return
    if bench_case.kind == "pymeshlab":
        # The same search-plus-PCA in one call, at the same ``k``; ``smoothiter=0`` keeps it to that
        # and leaves out the orientation propagation triwarp does not do either. A *point-cloud*
        # MeshSet: the filter is for datasets with no faces.
        cloud_pml = _cloud_meshset_pml(bench_case)
        bench_case.run(lambda: cloud_pml.compute_normal_for_point_clouds(k=_KNN, smoothiter=0))
        assert cloud_pml.current_mesh().vertex_normal_matrix().shape == (bench_case.n_vertices, 3)
        return
    if bench_case.kind == "triwarp":
        points = bench_case.vertices_wp
        normals = bench_case.run(
            lambda: tw.points.estimate_normals(
                points, tw.neighbors.query_nearest(points, points, k=_KNN, backend="bvh")[0]
            )
        )
        assert normals.shape == (bench_case.n_vertices,)
    else:
        cloud = _pcd(bench_case)
        search = o3d.geometry.KDTreeSearchParamKNN(knn=_KNN)
        bench_case.run(lambda: cloud.estimate_normals(search_param=search))
        assert np.asarray(cloud.normals).shape == (bench_case.n_vertices, 3)


@pytest.mark.benchmark(group="outlier_probability")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "pymeshlab")
def test_outlier_probability(bench_case: BenchCase) -> None:
    """LoOP scores: the k-NN table plus four passes over it, against the filter they came from."""
    if bench_case.mesh_name == "sphere_large":
        pytest.skip("MeshLab's k-d tree searches one point at a time; capped at sphere_med")
    if bench_case.kind == "pymeshlab":
        # Selection-only, so the geometry is untouched and the MeshSet is shared; the filter still
        # rebuilds its k-d tree every call, which is why triwarp's row builds its table in-callable.
        cloud_pml = _cloud_meshset_pml(bench_case)
        bench_case.run(
            lambda: cloud_pml.compute_selection_point_cloud_outliers(
                propthreshold=0.8, knearest=_KNN
            )
        )
        assert cloud_pml.current_mesh().vertex_selection_array().shape == (bench_case.n_vertices,)
        return
    points = bench_case.vertices_wp
    probability = bench_case.run(
        lambda: tw.points.outlier_probability(
            *tw.neighbors.query_nearest(points, points, k=_KNN, backend="bvh")
        )
    )
    assert probability.shape == (bench_case.n_vertices,)


@pytest.mark.benchmark(group="statistical_outlier_mask")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "open3d")
def test_statistical_outlier_mask(bench_case: BenchCase) -> None:
    """One global threshold on the mean neighbour distance — open3d's own outlier criterion."""
    if bench_case.mesh_name == "sphere_large":
        pytest.skip("open3d searches one point at a time; capped at sphere_med")
    if bench_case.kind == "triwarp":
        points = bench_case.vertices_wp
        mask = bench_case.run(
            lambda: tw.points.statistical_outlier_mask(
                tw.neighbors.query_nearest(points, points, k=_KNN, backend="bvh")[1]
            )
        )
        assert mask.shape == (bench_case.n_vertices,)
    else:
        # ``remove_statistical_outlier`` also materializes the kept subset, which triwarp does not.
        cloud = _pcd(bench_case)
        _kept, keep_indices = bench_case.run(
            lambda: cloud.remove_statistical_outlier(nb_neighbors=_KNN, std_ratio=2.0)
        )
        assert len(keep_indices) <= bench_case.n_vertices


@pytest.mark.benchmark(group="radius_outlier_mask")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "open3d")
@pytest.mark.parametrize("radius_scale", _RADIUS_SCALES)
def test_radius_outlier_mask(bench_case: BenchCase, radius_scale: float) -> None:
    """
    A ball count plus one threshold map, swept over the radius rather than the mesh.

    The radius is the whole cost model: the count is linear in how many points each ball holds, so a
    2x radius is ~8x the point tests, and where the hash grid stops paying for itself is what the
    two points bracket. Nothing after the count scales -- the threshold is one ``wp.map``.

    open3d's ``remove_radius_outlier`` builds its own ``KDTreeFlann`` per call and *materializes*
    the kept subset, so its row is an upper bound on the same question; it is also
    **nondeterministic** (a shared tree across an OpenMP loop), which is why the comparison in
    ``tests/test_points.py`` goes through that tree serially instead.
    """
    if bench_case.mesh_name == "sphere_large":
        pytest.skip("open3d searches one point at a time; capped at sphere_med")
    radius = radius_scale * bench_case.mean_edge
    if bench_case.kind == "triwarp":
        points = bench_case.vertices_wp
        mask = bench_case.run(lambda: tw.points.radius_outlier_mask(points, radius, _MIN_NEIGHBORS))
        assert mask.shape == (bench_case.n_vertices,)
    else:
        cloud = _pcd(bench_case)
        _kept, keep_indices = bench_case.run(
            lambda: cloud.remove_radius_outlier(nb_points=_MIN_NEIGHBORS, radius=radius)
        )
        assert len(keep_indices) <= bench_case.n_vertices


@pytest.mark.benchmark(group="point_duplicate_mask")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "open3d", "meshlib")
def test_point_duplicate_mask(bench_case: BenchCase) -> None:
    """
    Exact positional dedup: two ``unique_1d`` rounds over 64-bit keys, then a first-occurrence pass.

    The most expensive of the module's four masks and the only one that is not a single map -- two
    open-addressing hash builds and two sorts of the unique values, so it is the row to read for a
    change to ``grouping``'s uniqueness machinery from the point-cloud side.

    open3d's ``remove_duplicated_points`` answers the same question with an
    ``unordered_map<Vector3d>`` on one core and copies the survivors out; the mask here is compared
    against its survivor list in ``tests/test_points.py``. A registry mesh has *no* duplicated
    vertices, so all three rows do their full work and none takes an early exit.

    meshlib's ``findSmallestCloseVertices(cloud, 0.0)`` is the third implementation and the only
    multi-threaded one: it returns each point's smallest-indexed coincident neighbour, which is
    this mask under ``map != index`` (``tests/test_points.py``). It goes through the cloud's point
    tree rather than a hash, so unlike the other two rows its cost is a *search* -- which is why
    the tree is dropped per round the way the ``nearest_neighbor_distance`` row drops it, rather
    than being reused across rounds and pricing the query alone.

    First measurement, medians on an RTX 5090: at ``sphere_large`` triwarp-cuda **2.57 ms** against
    meshlib's **7.43** and open3d's **30.0**, but at ``sphere_small`` triwarp is **1.47 ms** where
    both references are under 0.33 -- the two hash builds and two sorts have a fixed cost the
    reference searches do not, so this group's ratio *inverts* below a few thousand points. That is
    the shape to watch here rather than the large-cloud number.
    """
    if bench_case.kind == "meshlib":
        cloud_ml = _cloud_ml(bench_case)  # held in a name: MeshLib's trees point into it

        def uncached_cloud_ml() -> mm.PointCloud:
            """Drop the lazily built point tree, which the search below would otherwise reuse."""
            cloud_ml.invalidateCaches()
            return cloud_ml

        representative_ml = bench_case.run(
            lambda cloud: mm.findSmallestCloseVertices(cloud, 0.0), setup=uncached_cloud_ml
        )
        assert representative_ml.size() == bench_case.n_vertices
        return
    if bench_case.kind == "open3d":
        # The dedup mutates nothing -- it returns a fresh cloud -- so one cached input cloud is
        # valid for every round, unlike the ``remove_*`` methods that rewrite in place.
        cloud = _pcd(bench_case)
        deduplicated_o3d = bench_case.run(cloud.remove_duplicated_points)
        assert len(deduplicated_o3d.points) <= bench_case.n_vertices
        return
    points = bench_case.vertices_wp
    mask = bench_case.run(lambda: tw.points.point_duplicate_mask(points))
    assert mask.shape == (bench_case.n_vertices,)


@pytest.mark.benchmark(group="farthest_point_sample")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "open3d")
@pytest.mark.parametrize("count", _SAMPLE_COUNTS)
def test_farthest_point_sample(bench_case: BenchCase, count: int) -> None:
    """
    The greedy maximin subsample, swept over the sample count -- which is the round count.

    Inherently sequential in ``count``: each round folds one selected point into the running
    distances and takes a global arg-max, so the work is ``Theta(count)`` rounds over the whole
    cloud however small the sample is -- run as one persistent block with a ``wp.tile_max`` per
    round, not as one launch per round. That makes this the one group in the module whose slope is
    set by a *parameter* rather than by the mesh, and the two counts here are a 16x step in it.

    open3d's ``FarthestPointDownSample`` runs the identical loop serially in C++ on one core, so
    the comparison is device parallelism against a tighter inner loop; it also copies the selected
    points out where triwarp returns indices.
    """
    if bench_case.mesh_name == "sphere_large":
        pytest.skip("open3d's greedy loop is serial over the whole cloud; capped at sphere_med")
    if bench_case.kind == "triwarp":
        points = bench_case.vertices_wp
        indices = bench_case.run(lambda: tw.points.farthest_point_sample(points, count))
        assert indices.shape == (count,)
    else:
        cloud = _pcd(bench_case)
        sampled_o3d = bench_case.run(lambda: cloud.farthest_point_down_sample(count))
        assert len(sampled_o3d.points) == count


# Direction counts for the support sweep. Cost is exactly ``points x n_directions`` -- the only
# knob in the module, and the accuracy/speed trade against exact qhull.
_N_DIRECTIONS = [32, 256]

# Icosphere refinement levels for the conservative filter: 42 directions / 80 tetrahedra at 1, and
# 642 / 1280 at 3. Both halves of its cost scale with this, so it spans the useful range.
_SUBDIVISIONS = [1, 3]


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
            lambda: tw.points.convex_subset_mask(points, n_directions=n_directions)
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
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d", "pymeshlab")
def test_convex_subset(bench_case: BenchCase) -> None:
    """
    The mask plus ``flatnonzero`` and a gather: isolates the compaction cost.

    The same three qhull wrappers ``convex_subset_mask`` times, because the module docstring's
    argument -- that the exact hull is the accuracy bar the approximation trades against -- is about
    the *operation*, not about which of its two entry points is called. Both return the hull's
    vertices; only the container differs, and a qhull wrapper returns positions rather than a mask,
    so if anything this is the closer shape of the two. The rows carried only on the mask group for
    as long as they did because that group was written first.

    No ``n_directions`` axis here (unlike the mask group): the compaction this group isolates does
    not scale with the direction count, so one row per library is the whole comparison.
    """
    if bench_case.kind == "triwarp":
        points = bench_case.vertices_wp
        selected = bench_case.run(lambda: tw.points.convex_subset(points))
        assert selected.shape[0] <= bench_case.n_vertices
        return
    skip_larger_than(bench_case, "dragon", "qhull is single-threaded on the host")
    if bench_case.kind == "pymeshlab":  # qhull again, through MeshLab's own wrapper
        bench_case.run(lambda: bench_case.new_meshset_pml().generate_convex_hull())
    elif bench_case.kind == "trimesh":
        points_np = bench_case.vertices_np
        hull_tm = bench_case.run(lambda: tm.points.PointCloud(points_np).convex_hull)
        assert hull_tm.vertices.shape[1] == 3
    else:
        mesh_o3d = bench_case.mesh_o3d
        hull_o3d, _indices = bench_case.run(lambda: mesh_o3d.compute_convex_hull())
        assert np.asarray(hull_o3d.vertices).shape[1] == 3


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
            lambda: tw.points.convex_superset_mask(points, subdivisions=subdivisions)
        )
        assert mask.shape[0] == bench_case.n_vertices
    else:
        skip_larger_than(bench_case, "dragon", "qhull is single-threaded on the host")
        points_np = bench_case.vertices_np
        hull_np = bench_case.run(lambda: scipy.spatial.ConvexHull(points_np))
        assert hull_np.vertices.shape[0] >= 4
