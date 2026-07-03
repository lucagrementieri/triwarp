"""
Benchmarks for ``triwarp.edges`` vs trimesh and libigl (igl), on real scan meshes.

Mirrors the correctness tests in ``tests/test_edges.py`` but times each function instead of
asserting equality. Every test is parametrised over ``(mesh_name, library)`` by the harness in
``conftest.py`` and receives a ``bench_case`` bundling the inputs and a GPU-safe ``run``; the
``benchlibs`` marker declares which reference libraries implement an equivalent of the function.

The triwarp ``edges_unique*`` calls pass ``n_vertices=`` (known from the mesh) so the API's
internal ``.numpy().max()`` host sync does not dominate the GPU measurement.
"""

from __future__ import annotations

import igl
import numpy as np
import pytest
import trimesh as tm

import triwarp as tw


@pytest.mark.benchmark(group="faces_to_edges")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_faces_to_edges(bench_case) -> None:
    if bench_case.kind == "triwarp":
        faces = bench_case.faces_wp
        result = bench_case.run(lambda: tw.edges.faces_to_edges(faces))
        assert result.shape == (faces.shape[0], 2)
    else:  # trimesh
        faces = bench_case.faces_np
        result = bench_case.run(lambda: tm.geometry.faces_to_edges(faces))
        assert result.shape == (faces.shape[0] * 3, 2)


@pytest.mark.benchmark(group="faces_to_edges_sorted")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_faces_to_edges_sorted(bench_case) -> None:
    if bench_case.kind == "triwarp":
        faces = bench_case.faces_wp
        result = bench_case.run(lambda: tw.edges.faces_to_edges(faces, sorted=True))
        assert result.shape == (faces.shape[0], 2)
    else:  # trimesh
        faces = bench_case.faces_np
        result = bench_case.run(lambda: np.sort(tm.geometry.faces_to_edges(faces), axis=1))
        assert result.shape == (faces.shape[0] * 3, 2)


@pytest.mark.benchmark(group="edges_face")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_edges_face(bench_case) -> None:
    if bench_case.kind == "triwarp":
        faces = bench_case.faces_wp
        result = bench_case.run(lambda: tw.edges.edges_face(faces))
        assert result.shape == (faces.shape[0],)
    else:  # trimesh: Trimesh.edges_face == repeat(arange(n_faces), 3)
        n_faces = bench_case.faces_np.shape[0]
        result = bench_case.run(lambda: np.repeat(np.arange(n_faces, dtype=np.int64), 3))
        assert result.shape == (n_faces * 3,)


@pytest.mark.benchmark(group="edges_unique")
@pytest.mark.benchlibs("triwarp", "trimesh", "igl")
def test_edges_unique(bench_case) -> None:
    if bench_case.kind == "triwarp":
        faces, nv = bench_case.faces_wp, bench_case.n_vertices
        unique_edges, _ = bench_case.run(lambda: tw.edges.edges_unique(faces, n_vertices=nv))
        assert unique_edges.shape[1] == 2
    elif bench_case.kind == "trimesh":
        faces = bench_case.faces_np

        def run():
            edges_sorted = np.sort(tm.geometry.faces_to_edges(faces), axis=1)
            unique_idx, _ = tm.grouping.unique_rows(edges_sorted)
            return edges_sorted[unique_idx]

        assert run().shape[1] == 2
        bench_case.run(run)
    else:  # igl.unique_edge_map -> (E, uE, EMAP, uEC, uEE); uE is the unique undirected edges
        faces = bench_case.faces_np
        result = bench_case.run(lambda: igl.unique_edge_map(faces)[1])
        assert result.shape[1] == 2


@pytest.mark.benchmark(group="edges_unique_inverse")
@pytest.mark.benchlibs("triwarp", "trimesh", "igl")
def test_edges_unique_inverse(bench_case) -> None:
    if bench_case.kind == "triwarp":
        faces, nv = bench_case.faces_wp, bench_case.n_vertices
        result = bench_case.run(lambda: tw.edges.edges_unique_inverse(faces, n_vertices=nv))
        assert result.shape == (faces.shape[0],)
    elif bench_case.kind == "trimesh":
        faces = bench_case.faces_np

        def run():
            edges_sorted = np.sort(tm.geometry.faces_to_edges(faces), axis=1)
            _, inverse = tm.grouping.unique_rows(edges_sorted)
            return inverse

        bench_case.run(run)
    else:  # igl.unique_edge_map -> EMAP (index 2) maps each directed edge to its unique edge
        faces = bench_case.faces_np
        bench_case.run(lambda: igl.unique_edge_map(faces)[2])


@pytest.mark.benchmark(group="edges_unique_length")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_edges_unique_length(bench_case) -> None:
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        nv = bench_case.n_vertices
        result = bench_case.run(
            lambda: tw.edges.edges_unique_length(vertices, faces, n_vertices=nv)
        )
        assert result.ndim == 1
    else:  # trimesh: unique undirected edges then Euclidean norm
        vertices, faces = bench_case.vertices_np, bench_case.faces_np

        def run():
            edges_sorted = np.sort(tm.geometry.faces_to_edges(faces), axis=1)
            unique_idx, _ = tm.grouping.unique_rows(edges_sorted)
            unique = edges_sorted[unique_idx]
            return np.linalg.norm(vertices[unique[:, 1]] - vertices[unique[:, 0]], axis=1)

        bench_case.run(run)


@pytest.mark.benchmark(group="edges_length")
@pytest.mark.benchlibs("triwarp", "trimesh", "igl")
def test_edges_length(bench_case) -> None:
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        result = bench_case.run(lambda: tw.edges.edges_length(vertices, faces))
        assert result.shape == (faces.shape[0],)
    elif bench_case.kind == "trimesh":
        vertices, faces = bench_case.vertices_np, bench_case.faces_np

        def run():
            edges = np.asarray(tm.geometry.faces_to_edges(faces))
            return np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1)

        bench_case.run(run)
    else:  # igl.edge_lengths -> (#F, 3) per-face directed edge lengths
        vertices, faces = bench_case.vertices_np, bench_case.faces_np
        result = bench_case.run(lambda: igl.edge_lengths(vertices, faces))
        assert result.shape == (faces.shape[0], 3)


@pytest.mark.benchmark(group="mean_edge_length")
@pytest.mark.benchlibs("triwarp", "trimesh", "igl")
def test_mean_edge_length(bench_case) -> None:
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        result = bench_case.run(lambda: tw.edges.mean_edge_length(vertices, faces))
        assert result >= 0.0
    elif bench_case.kind == "trimesh":  # mean of all per-face edge norms (test reference)
        vertices, faces = bench_case.vertices_np, bench_case.faces_np

        def run():
            tri = vertices[faces]
            return float(np.linalg.norm(tri - tri[:, [1, 2, 0]], axis=2).mean())

        bench_case.run(run)
    else:  # igl.avg_edge_length
        vertices, faces = bench_case.vertices_np, bench_case.faces_np
        bench_case.run(lambda: float(igl.avg_edge_length(vertices, faces)))
