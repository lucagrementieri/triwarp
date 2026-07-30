"""
Benchmarks for ``triwarp.triangles``: the area-weighted mesh centroid and per-face normals/areas.

Both are per-face arithmetic over the whole mesh with no connectivity structure, so the scan sweep
(pure ``N``) is the right axis for them and every reference is doing the same cross products.

``face_normals_and_areas`` returns both quantities from one cross product, which is what its
consumers in the library want (the heat method, the gradient operators, area-weighted normals). The
references split them up: ``igl.doublearea`` and ``potpourri3d.face_areas`` return areas only, and
``trimesh.triangles.normals`` returns unit normals plus a validity mask. The reference rows
therefore do strictly less work than triwarp's — read this group as a floor for them rather than as
a fair race.

**pymeshlab** splits them the other way, and its two rows are opposite kinds of bound.
``compute_normal_per_face`` is normals only, so it belongs with the other partial references above.
``get_geometric_measures`` is the ``centroid`` reference and does *more*: one read-only call returns
``shell_barycenter`` (the area-weighted centroid triwarp computes), ``barycenter`` (the plain vertex
mean), the surface area, the mesh volume, the average edge length and the inertia tensor. So that
row is an **upper** bound on the centroid alone — and the same number appears as the
``mean_edge_length`` reference in [`test_edges.py`](test_edges.py), which is worth knowing before
reading either as a per-quantity cost. Both are geometry-preserving, so they share the MeshSet.

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
from conftest import BenchCase, skip_larger_than

import triwarp as tw


@pytest.mark.benchmark(group="centroid")
@pytest.mark.benchlibs("triwarp", "trimesh", "pymeshlab")
def test_centroid(bench_case: BenchCase) -> None:
    if bench_case.kind == "pymeshlab":
        # ``get_geometric_measures``' ``barycenter`` is the vertex mean and ``shell_barycenter`` the
        # area-weighted centroid triwarp computes; the call returns both plus the area, volume and
        # inertia tensor, so it is an upper bound rather than an equivalent. Capped at ``bunny``:
        # it costs 1.12 s a call on ``dragon``, for a ratio the two medium meshes already establish.
        skip_larger_than(bench_case, "bunny", "get_geometric_measures is 1.12 s a call on dragon")
        meshset_pml = bench_case.meshset_pml
        assert bench_case.run(meshset_pml.get_geometric_measures)["shell_barycenter"].shape == (3,)
        return
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        result = bench_case.run(lambda: tw.triangles.centroid(vertices, faces))
        assert np.isfinite(list(result)).all()
    else:  # numpy reference: the uncached formula behind ``trimesh.Trimesh.centroid``
        vertices, faces = bench_case.vertices_np, bench_case.faces_np

        def run() -> np.ndarray:
            triangles = vertices[faces]
            crosses = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            areas = 0.5 * np.linalg.norm(crosses, axis=1)
            return (triangles.mean(axis=1) * areas[:, None]).sum(axis=0) / areas.sum()

        result = bench_case.run(run)
        assert result.shape == (3,)


@pytest.mark.benchmark(group="face_normals_and_areas")
@pytest.mark.benchlibs("triwarp", "trimesh", "igl", "potpourri3d", "pymeshlab")
def test_face_normals_and_areas(bench_case: BenchCase) -> None:
    """One cross product per face: the operator prologue every solver in the library pays."""
    n_faces = bench_case.n_faces
    if bench_case.kind == "pymeshlab":  # normals only; the area total is in get_geometric_measures
        meshset_pml = bench_case.meshset_pml
        bench_case.run(meshset_pml.compute_normal_per_face)
        assert meshset_pml.current_mesh().face_normal_matrix().shape == (n_faces, 3)
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


@pytest.mark.benchmark(group="face_quality")
@pytest.mark.benchaxis("quality")
@pytest.mark.benchlibs("triwarp", "igl", "pymeshlab")
def test_face_quality(bench_case: BenchCase) -> None:
    """Per-face shape measure, on the axis it defines: bad triangles must not cost more."""
    n_faces = bench_case.n_faces
    if bench_case.kind == "pymeshlab":
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
