"""
Benchmarks for ``triwarp.triangles``: per-face arithmetic, one triangle at a time.

Nothing here reads connectivity, so the scan sweep (pure ``N``) is the right axis throughout and
every reference is doing the same cross products. The whole-mesh reductions these feed --
the centroid, the volume and the moments -- moved to [`test_totals.py`](test_totals.py) with their
module.

``face_normals_and_areas`` returns both quantities from one cross product, which is what its
consumers in the library want (the heat method, the gradient operators, area-weighted normals). The
references split them up: ``igl.doublearea`` and ``potpourri3d.face_areas`` return areas only, and
``trimesh.triangles.normals`` returns unit normals plus a validity mask. The reference rows
therefore do strictly less work than triwarp's — read this group as a floor for them rather than as
a fair race.

**pymeshlab** splits them the other way: ``compute_normal_per_face`` is normals only, so it belongs
with the other partial references above. It is geometry-preserving, so it shares the MeshSet.

**pyvista** (VTK 9.6) is the one reference that answers both halves, but in *two* filters --
``compute_normals`` for the unit cell normal and ``compute_cell_sizes`` for the area -- so its row
times both calls rather than half the work. It is the only reference here whose normals come back
``float32``; the areas are ``float64``.

``face_quality`` runs on the **quality** axis rather than the scan sweep: it is the quantity that
axis is *defined* by (``saddle`` and ``saddle_graded`` share connectivity and differ only in
triangle shape), so measuring it there says whether reading the measure costs anything once the
triangles get bad. It should not — every metric is branch-free arithmetic on three edge vectors —
and that flatness is the point of the row. ``compute_scalar_by_aspect_ratio_per_face`` is the exact
filter the four VCG metrics were ported from and is geometry-preserving, so it shares the MeshSet;
``igl`` needs two calls (``circumradius`` and ``inradius``) to build the same ratio.
"""

from __future__ import annotations

import igl
import numpy as np
import potpourri3d as pp3d
import pytest
import trimesh as tm
import warp as wp
from conftest import BenchCase
from meshlib import mrmeshpy as mm

import triwarp as tw

_barycentre_cache: dict[tuple[str, str], wp.array] = {}


def _barycentres_wp(bench_case: BenchCase) -> wp.array[wp.vec3]:
    """One query point per triangle -- its own barycentre -- as an *input*, not part of the work."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _barycentre_cache:
        centres_np = bench_case.vertices_np[bench_case.faces_np].mean(axis=1)
        _barycentre_cache[key] = wp.array(
            np.ascontiguousarray(centres_np, dtype=np.float32),
            dtype=wp.vec3,
            device=bench_case.device,
        )
    return _barycentre_cache[key]


@pytest.mark.benchmark(group="face_normals_and_areas")
@pytest.mark.benchlibs(
    "triwarp", "trimesh", "igl", "open3d", "potpourri3d", "pymeshlab", "pyvista", "meshlib"
)
def test_face_normals_and_areas(bench_case: BenchCase) -> None:
    """
    One cross product per face: the operator prologue every solver in the library pays.

    The meshlib row is ``computePerFaceNormals`` alone -- normals, not areas -- for the same reason
    pymeshlab's is: its area entry point ``dblArea`` is **per face**, so batching it would mean a
    Python loop over the face buffer, and that row would time the loop rather than MeshLib (see
    section 6). Both halves are still compared for correctness, in
    tests/test_triangles.py::test_per_face_quantities_match_meshlib. Pure, so the mesh is built once
    outside the timed callable.
    """
    n_faces = bench_case.n_faces
    if bench_case.kind == "meshlib":
        mesh_ml = bench_case.new_mesh_ml()
        normals_ml = bench_case.run(lambda: mm.computePerFaceNormals(mesh_ml))
        assert normals_ml.size() == n_faces
        return
    if bench_case.kind == "pyvista":  # both halves, in two filters -- see the module docstring
        mesh_pv = bench_case.mesh_pv
        areas_pv = bench_case.run(
            lambda: (
                mesh_pv.compute_normals(
                    cell_normals=True,
                    point_normals=False,
                    consistent_normals=False,
                    auto_orient_normals=False,
                    split_vertices=False,
                ).cell_data["Normals"],
                mesh_pv.compute_cell_sizes(length=False, area=True, volume=False).cell_data["Area"],
            )
        )[1]
        assert np.asarray(areas_pv).shape == (n_faces,)
        return
    if bench_case.kind == "pymeshlab":  # normals only; the area total is in get_geometric_measures
        meshset_pml = bench_case.meshset_pml
        bench_case.run(meshset_pml.compute_normal_per_face)
        assert meshset_pml.current_mesh().face_normal_matrix().shape == (n_faces, 3)
        return
    if bench_case.kind == "open3d":  # unit normals only; recomputed unconditionally per call
        mesh_o3d = bench_case.mesh_o3d
        bench_case.run(mesh_o3d.compute_triangle_normals)
        assert np.asarray(mesh_o3d.triangle_normals).shape == (n_faces, 3)
        return
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        normals, areas = bench_case.run(
            lambda: tw.triangles.face_normals_and_areas(vertices, faces)
        )
        assert normals.shape == (n_faces,)
        assert areas.shape == (n_faces,)
    elif bench_case.kind == "igl":  # areas only, and doubled
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        areas_igl = bench_case.run(lambda: igl.doublearea(vertices_np, faces_np))
        assert areas_igl.shape == (n_faces,)
    elif bench_case.kind == "potpourri3d":  # areas only, vectorized numpy
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        areas_pp = bench_case.run(lambda: pp3d.face_areas(vertices_np, faces_np))
        assert areas_pp.shape == (n_faces,)
    else:  # trimesh: unit normals plus a validity mask, from the same cross product
        triangles_np = bench_case.vertices_np[bench_case.faces_np]
        normals_tm, valid_tm = bench_case.run(lambda: tm.triangles.normals(triangles_np))
        assert valid_tm.shape == (n_faces,)
        assert normals_tm.shape[1] == 3


@pytest.mark.benchmark(group="face_angles")
@pytest.mark.benchlibs("triwarp", "trimesh", "igl", "pyvista")
def test_face_angles(bench_case: BenchCase) -> None:
    """
    The three interior angles per face, and the group with the widest margin in the module.

    They are the input to ``vertex_defects`` and to the angle-weighted normals.

    Nothing has to be matched up here -- ``igl.internal_angles``, ``trimesh``'s ``face_angles``
    property and triwarp all return ``(n_faces, 3)`` angles aligned with the corners
    ``(i0, i1, i2)``, agreeing element-wise with no transform (verified in
    ``tests/test_triangles.py``). trimesh rebuilds its ``tm.Trimesh`` inside the callable because
    ``face_angles`` is a cached property; a shared mesh would time the cache lookup.

    **pyvista does strictly less**: ``cell_quality`` gives the per-face *extremes* only, so its row
    reads as a floor -- two reductions of the three angles rather than the three angles.
    """
    if bench_case.kind == "pyvista":
        mesh_pv = bench_case.mesh_pv
        quality_pv = bench_case.run(lambda: mesh_pv.cell_quality(["min_angle", "max_angle"]))
        assert np.asarray(quality_pv.cell_data["min_angle"]).shape == (bench_case.n_faces,)
        return
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        angles = bench_case.run(lambda: tw.triangles.face_angles(vertices, faces))
        assert angles.shape == (bench_case.n_faces, 3)
    elif bench_case.kind == "igl":
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        angles_igl = bench_case.run(lambda: igl.internal_angles(vertices_np, faces_np))
        assert angles_igl.shape == (bench_case.n_faces, 3)
    else:
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        angles_tm = bench_case.run(
            lambda: tm.Trimesh(vertices_np, faces_np, process=False).face_angles
        )
        assert angles_tm.shape == (bench_case.n_faces, 3)


@pytest.mark.benchmark(group="face_quality")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("triwarp", "igl", "pymeshlab", "pyvista")
def test_face_quality(bench_case: BenchCase) -> None:
    """
    Per-face shape measure, on the axis it defines: bad triangles must not cost more.

    pyvista's ``cell_quality`` is VTK's Verdict library, whose measure names invert against
    triwarp's: its ``radius_ratio`` is triwarp's ``aspect_ratio`` (and triwarp's ``radius_ratio`` is
    its reciprocal), which ``tests/test_triangles.py`` decodes in full. One measure is requested, so
    the row prices the same single ratio per face the other three do.
    """
    n_faces = bench_case.n_faces
    if bench_case.kind == "pyvista":
        mesh_pv = bench_case.mesh_pv
        quality_pv = bench_case.run(lambda: mesh_pv.cell_quality("radius_ratio"))
        assert np.asarray(quality_pv.cell_data["radius_ratio"]).shape == (n_faces,)
    elif bench_case.kind == "pymeshlab":
        meshset_pml = bench_case.meshset_pml
        bench_case.run(
            lambda: meshset_pml.compute_scalar_by_aspect_ratio_per_face(
                metric="inradius/circumradius"
            )
        )
        assert meshset_pml.current_mesh().face_scalar_array().shape == (n_faces,)
    elif bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        quality = bench_case.run(
            lambda: tw.triangles.face_quality(vertices, faces, metric="radius_ratio")
        )
        assert quality.shape == (n_faces,)
    else:  # igl: the same ratio, but as two separate passes over the faces
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        ratio_igl = bench_case.run(
            lambda: (
                np.asarray(igl.inradius(vertices_np, faces_np))
                / np.asarray(igl.circumradius(vertices_np, faces_np)[0])
            )
        )
        assert ratio_igl.shape == (n_faces,)


@pytest.mark.benchmark(group="points_to_barycentric")
@pytest.mark.benchlibs("triwarp", "trimesh", "igl")
@pytest.mark.parametrize("method", ["cramer", "cross"])
def test_points_to_barycentric(bench_case: BenchCase, method: str) -> None:
    """
    One point per triangle, back to barycentric coordinates: the module's other soup operation.

    triwarp's two ``method`` settings are two formulations of the same solve -- Cramer's rule on the
    2x2 system against a ratio of cross products -- and they should not differ measurably, which is
    what the pair checks. trimesh exposes the same choice and gets both ids; ``igl`` has one
    formulation, so its two rows are identical by construction and sit there as the fixed bar (the
    same convention as scipy's leaf-size rows in [`test_neighbors.py`](test_neighbors.py)).

    The query points are the face barycentres, so every one lies in its triangle's plane: this
    measures the in-plane solve rather than a projection.
    """
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        points = _barycentres_wp(bench_case)
        barycentric = bench_case.run(
            lambda: tw.triangles.points_to_barycentric(vertices, faces, points, method=method)
        )
        assert barycentric.shape == (bench_case.n_faces,)
        return
    triangles_np = bench_case.vertices_np[bench_case.faces_np]
    points_np = triangles_np.mean(axis=1)
    if bench_case.kind == "igl":
        barycentric_igl = bench_case.run(
            lambda: igl.barycentric_coordinates(
                np.ascontiguousarray(points_np),
                np.ascontiguousarray(triangles_np[:, 0]),
                np.ascontiguousarray(triangles_np[:, 1]),
                np.ascontiguousarray(triangles_np[:, 2]),
            )
        )
        assert barycentric_igl.shape == (bench_case.n_faces, 3)
        return
    barycentric_tm = bench_case.run(
        lambda: tm.triangles.points_to_barycentric(triangles_np, points_np, method=method)
    )
    assert barycentric_tm.shape == (bench_case.n_faces, 3)


@pytest.mark.benchmark(group="face_centroids")
@pytest.mark.benchlibs("triwarp", "igl", "pyvista")
def test_face_centroids(bench_case: BenchCase) -> None:
    """
    One barycentre per face: a ``3F`` gather and a divide, the module's cheapest kernel.

    It shares the scan sweep with ``face_normals_and_areas`` for a reason -- both are pure per-face
    arithmetic with no connectivity -- so the pair prices a cross product against a mean.
    ``igl.barycenter`` computes the identical quantity, and so does VTK's ``cell_centers`` -- which
    additionally builds a whole ``PolyData`` of vertex cells around the answer, so read its row as
    the cost of the container as much as of the mean.
    """
    if bench_case.kind == "pyvista":
        mesh_pv = bench_case.mesh_pv
        centres_pv = bench_case.run(lambda: mesh_pv.cell_centers())
        assert np.asarray(centres_pv.points).shape == (bench_case.n_faces, 3)
        return
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        centroids = bench_case.run(lambda: tw.triangles.face_centroids(vertices, faces))
        assert centroids.shape == (bench_case.n_faces,)
        return
    vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
    centroids_igl = bench_case.run(lambda: igl.barycenter(vertices_np, faces_np))
    assert centroids_igl.shape == (bench_case.n_faces, 3)
