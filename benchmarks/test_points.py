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
* ``estimate_normals_knn`` — [`query_bvh_nearest`][triwarp.neighbors.query_bvh_nearest] plus the
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
their own k-d tree per call and there would otherwise be nothing to compare. ``open3d`` covers only
the statistical variant (``remove_statistical_outlier``), which additionally *copies* the surviving
points into a new cloud -- triwarp returns a mask, so that row is an upper bound.

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
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pymeshlab as ml
import pytest
import pyvista as pv
import trimesh as tm
import warp as wp
from conftest import BenchCase, skip_larger_than

import triwarp as tw
import triwarp.typing as twt

# Neighbour count for the PCA normal estimate (open3d's own default for KDTreeSearchParamKNN).
_KNN = 30

# Neighbourhood widths to sweep: the per-point 3x3 PCA is linear in k, and the k-NN table build
# that feeds it is not, so the pair separates the estimator's cost from its input's. The table is
# cached (``_neighbor_table``), so this sweep times the estimator *only* — the query's own k axis is
# ``query_bvh_nearest_k1`` / ``_k7`` / ``_k64`` in ``test_neighbors.py``.
_KNN_SWEEP = [8, 64]

# The trimesh references here are single-threaded host passes and one of them is far worse than
# single-threaded: ``tm.points.fit_line`` measured **22 s a call** on ``bunny``'s 35 947 points
# against 8 s for the whole 11-round case on ``bunny_decimated``'s 8 171. That one reference was 91%
# of this module's wall clock, so every host branch is capped at the smallest scan mesh -- the ratio
# against triwarp is four orders of magnitude and needs no larger input to establish.
_HOST_CAP_REASON = "host reference is a single-threaded pass; capped at bunny_decimated"

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
        _neighbors_cache[key] = tw.neighbors.query_bvh_nearest(points, points, k=k)[0]
    return _neighbors_cache[key]


def _unit_normals_wp(bench_case: BenchCase) -> wp.array[wp.vec3]:
    """Area-weighted unit vertex normals: the left operand of the ``vector_angle`` pairs."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _normals_wp_cache:
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        _normals_wp_cache[key] = tw.vertices.area_weighted_vertex_normals(
            bench_case.n_vertices, vertices, faces
        )
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
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_point_plane_distance(bench_case: BenchCase) -> None:
    """
    Signed point-to-plane distance of every point: a single ``wp.map`` over the cloud.

    The thinnest kernel in the module, so it is the module's floor row: below roughly ``10 ** 3``
    points it reports the ~340 µs wrapper floor rather than the map (see
    ``test_creation::test_box``), and being a ~50 µs GPU row it also has the widest run-to-run
    spread in the suite -- measured at 46x on unchanged code across two processes. Only read it as
    part of its axis.
    """
    n_points = bench_case.n_vertices
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


@pytest.mark.benchmark(group="fit_line")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_fit_line(bench_case: BenchCase) -> None:
    """Major axis from the uncentred Gram matrix: tiled outer-product sum plus one ``svd3``."""
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


@pytest.mark.benchmark(group="principal_axes")
@pytest.mark.benchlibs("triwarp", "pyvista")
def test_principal_axes(bench_case: BenchCase) -> None:
    """Principal frame: centroid reduction, centred scatter, then one 3x3 SVD."""
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
@pytest.mark.benchlibs("triwarp", "trimesh", "pymeshlab")
def test_fit_plane(bench_case: BenchCase) -> None:
    """Least-squares plane: centroid reduction, centred covariance, then the smallest-sigma axis."""
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
        centroid, normal = bench_case.run(lambda: tw.points.fit_plane(points))
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
@pytest.mark.benchlibs("triwarp", "open3d", "pymeshlab")
def test_estimate_normals_knn(bench_case: BenchCase) -> None:
    """Neighbour search plus PCA — what open3d's ``estimate_normals`` does in one call."""
    if bench_case.mesh_name == "sphere_large":
        pytest.skip("open3d searches one point at a time; capped at sphere_med")
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
                points, tw.neighbors.query_bvh_nearest(points, points, k=_KNN)[0]
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
            *tw.neighbors.query_bvh_nearest(points, points, k=_KNN)
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
                tw.neighbors.query_bvh_nearest(points, points, k=_KNN)[1]
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
