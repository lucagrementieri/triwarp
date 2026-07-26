"""
Benchmarks for ``triwarp.graph``: connected components, single/multi-source BFS and ``split``.

The vertex-adjacency CSR matrix is prebuilt (untimed, cached per mesh/device) so the timings
isolate the graph algorithms. The scipy BFS reference sits in the ``trimesh`` library slot:
``scipy.sparse.csgraph.breadth_first_order`` is the exact-order oracle the triwarp ``bfs``
docstring promises to match (and the backend trimesh itself uses for graph traversals).

Only ``split`` has an open3d equivalent. ``connected_component_labels`` and both ``bfs`` variants
take an abstract CSR adjacency matrix, and open3d exposes no graph-traversal API over one — its
connectivity work is mesh-bound (``cluster_connected_triangles``), which is what ``split`` uses.

``split`` cost is dominated by the **component count**, not the mesh size, and the two registry
meshes differ enormously there: ``bunny`` is a single component while ``bunny_decimated`` has 94
(scan floaters). At ``_SPLIT_COPIES = 64`` that is 64 versus 6 016 returned submeshes, and triwarp
takes 78 ms for the former but 3.8 s for the latter — ~0.64 ms of host work per submesh, against
0.29 ms for open3d and 0.23 ms for trimesh. So triwarp wins by 34-120x when there are few
components and loses by 2-3x per component when there are many: the per-component allocation and
launch sequence in ``tw.combine.split``, not the labelling, is the thing to batch.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import warp as wp
import warp.sparse as wps
from conftest import BenchCase, skip_larger_than
from scipy.sparse.csgraph import breadth_first_order

import triwarp as tw

_N_SOURCES = 128
_SOURCE_SEED = 3
_SPLIT_COPIES = 64

_adjacency_cache: dict[tuple[str, str], wps.BsrMatrix] = {}
_scipy_cache: dict[str, sp.csr_matrix] = {}
_split_cache: dict[tuple[str, str], tuple] = {}


def _adjacency(bench_case: BenchCase) -> wps.BsrMatrix:
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _adjacency_cache:
        unique_edges, _ = tw.edges.edges_unique(
            bench_case.faces_wp, n_vertices=bench_case.n_vertices
        )
        _adjacency_cache[key] = tw.graph.edges_to_csr(bench_case.n_vertices, unique_edges)
    return _adjacency_cache[key]


def _scipy_graph(bench_case: BenchCase) -> sp.csr_matrix:
    if bench_case.mesh_name not in _scipy_cache:
        faces = bench_case.faces_np
        edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
        n = bench_case.n_vertices
        data = np.ones(edges.shape[0], dtype=np.float32)
        graph = sp.coo_matrix((data, (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
        _scipy_cache[bench_case.mesh_name] = graph
    return _scipy_cache[bench_case.mesh_name]


@pytest.mark.benchmark(group="connected_component_labels")
@pytest.mark.benchlibs("triwarp")
def test_connected_component_labels(bench_case: BenchCase) -> None:
    skip_larger_than(bench_case, "happy_buddha")
    adjacency = _adjacency(bench_case)
    labels = bench_case.run(lambda: tw.graph.connected_component_labels(adjacency))
    assert labels.shape == (bench_case.n_vertices,)


@pytest.mark.benchmark(group="bfs")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_bfs(bench_case: BenchCase) -> None:
    skip_larger_than(bench_case, "happy_buddha")
    if bench_case.kind == "triwarp":
        adjacency = _adjacency(bench_case)
        order, _, _ = bench_case.run(lambda: tw.graph.bfs(adjacency, 0))
        assert order.shape[0] >= 1
    else:  # scipy exact-order oracle (see module docstring)
        graph = _scipy_graph(bench_case)
        order, _ = bench_case.run(
            lambda: breadth_first_order(graph, 0, directed=False, return_predecessors=True)
        )
        assert order.shape[0] >= 1


@pytest.mark.benchmark(group="bfs_multi_source")
@pytest.mark.benchlibs("triwarp")
def test_bfs_multi_source(bench_case: BenchCase) -> None:
    skip_larger_than(bench_case, "happy_buddha")
    adjacency = _adjacency(bench_case)
    rng = np.random.default_rng(_SOURCE_SEED)
    sources = wp.array(
        rng.integers(0, bench_case.n_vertices, size=_N_SOURCES).astype(np.int32),
        dtype=wp.int32,
        device=bench_case.device,
    )
    neighbors, offsets = bench_case.run(lambda: tw.graph.bfs_multi_source(adjacency, sources))
    assert offsets.shape == (_N_SOURCES,)
    assert neighbors.shape[0] >= _N_SOURCES


def _multi_component_np(bench_case: BenchCase) -> tuple[np.ndarray, np.ndarray]:
    """64 translated copies of the mesh as one (vertices, faces) soup, built untimed."""
    vertices, faces = bench_case.vertices_np, bench_case.faces_np
    n_vertices = vertices.shape[0]
    diagonal = vertices.max(axis=0) - vertices.min(axis=0)
    all_vertices = [
        vertices + np.array([1.5 * diagonal[0] * i, 0.0, 0.0]) for i in range(_SPLIT_COPIES)
    ]
    all_faces = [faces + n_vertices * i for i in range(_SPLIT_COPIES)]
    return np.vstack(all_vertices), np.vstack(all_faces)


def _split_inputs(bench_case: BenchCase) -> tuple:
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _split_cache:
        vertices_np, faces_np = _multi_component_np(bench_case)
        if bench_case.kind == "triwarp":
            vertices = wp.array(
                np.ascontiguousarray(vertices_np, dtype=np.float32),
                dtype=wp.vec3,
                device=bench_case.device,
            )
            faces = wp.array(
                np.ascontiguousarray(faces_np.reshape(-1), dtype=np.int32),
                dtype=wp.int32,
                device=bench_case.device,
            )
            _split_cache[key] = (vertices, faces)
        else:
            _split_cache[key] = (vertices_np, faces_np)
    return _split_cache[key]


@pytest.mark.benchmark(group="split")
@pytest.mark.benchlibs("triwarp", "trimesh", "open3d")
def test_split(bench_case: BenchCase) -> None:
    skip_larger_than(bench_case, "bunny")
    if bench_case.kind == "triwarp":
        vertices, faces = _split_inputs(bench_case)
        parts = bench_case.run(lambda: tw.combine.split(vertices, faces))
    elif bench_case.kind == "trimesh":
        import trimesh as tm

        vertices_np, faces_np = _split_inputs(bench_case)
        mesh = tm.Trimesh(vertices_np, faces_np, process=False)

        def run() -> list:
            return mesh.split(only_watertight=False)

        parts = bench_case.run(run)
    else:
        # ``tw.combine.split`` returns compact per-component ``(vertices, faces)`` submeshes, so the
        # open3d equivalent is ``cluster_connected_triangles`` (the labelling) followed by
        # ``select_by_index`` per cluster (the compaction). ``select_by_index`` takes *vertex*
        # indices, hence the ``np.unique`` over each cluster's faces — numpy is part of what an
        # open3d user pays here, exactly as scipy is for the trimesh path.
        import open3d as o3d

        vertices_np, faces_np = _split_inputs(bench_case)
        mesh_o3d = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices_np),
            o3d.utility.Vector3iVector(np.ascontiguousarray(faces_np, dtype=np.int32)),
        )
        faces_i32 = np.ascontiguousarray(faces_np, dtype=np.int32)

        def run_o3d() -> list:
            labels_np = np.asarray(mesh_o3d.cluster_connected_triangles()[0])
            return [
                mesh_o3d.select_by_index(np.unique(faces_i32[labels_np == label]))
                for label in range(int(labels_np.max()) + 1)
            ]

        parts = bench_case.run(run_o3d)
    # Scan meshes contain floater components, so each copy contributes its own component count.
    assert len(parts) >= _SPLIT_COPIES
    assert len(parts) % _SPLIT_COPIES == 0
