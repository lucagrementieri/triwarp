"""
Benchmarks for ``triwarp.graph``: connected components and single/multi-source BFS.

Two axes, matching the two ways a graph algorithm gets slow:

* **components** for ``connected_component_labels``. ECL-CC is three launches whatever the graph
  looks like, but the pointer-jumping hook/flatten converges at a rate set by component structure,
  so 1 / 64 / 1024 components at a fixed 81 920 faces is the shape that would expose a regression.
  The same two labellings are *also* run on the **diameter** axis, where they must read flat --
  that is a separate claim from the component sweep and it is load-bearing elsewhere in the
  package, so it is regression-covered rather than measured once (see
  ``test_connected_component_labels_depth``).
* **diameter** for ``bfs``. This is the one that hurts, and the only group in the suite that still
  loses to a CPU reference after being worked on. The traversal is level-synchronous under
  ``wp.capture_while`` -- **one iteration per BFS level**, seven launches at a grid size CUDA
  graphs bake in -- so a graph's *depth* sets the launch count and its size does not. It measured
  **5.06 ms on ``sphere_med`` (diameter ~130) against 367 ms on ``ribbon_long`` (diameter
  20 481)** at an identical 40 962 vertices: 73x, from a property no face count records. Bounding
  the serial block scan to the frontier and handing narrow frontiers to a serial resume brings that
  to **4.0 ms and 23 ms**, a 5.8x spread -- but scipy does the ribbon in 0.68 ms, so this group
  stays a loss by 34x and is kept as the open item it is. Closing it needs a single-block
  ``launch_tiled(dim=[1])`` traversal engine, which is a different program.

The vertex-adjacency CSR matrix is prebuilt (untimed, cached per mesh/device) so the timings
isolate the graph algorithms from the edge sort that produces them.

``combine.split`` used to live here because it is the other component-count-driven function in the
package. It now sits in [`test_combine.py`](test_combine.py) with the rest of ``triwarp.combine``,
on the same ``components`` axis.

References
----------
The scipy references sit in the ``scipy`` library slot: ``scipy.sparse.csgraph
.breadth_first_order`` is the exact-order oracle the triwarp ``bfs`` docstring promises to match
(and the backend trimesh itself uses for graph traversals), and ``connected_components`` is its
labelling counterpart.

Neither **trimesh** nor **open3d** appears: both functions take an abstract CSR adjacency matrix,
and open3d exposes no graph-traversal API over one -- its connectivity work is mesh-bound
(``cluster_connected_triangles``), which is what ``split`` uses over in ``test_combine``.

**pymeshlab** appears in ``connected_component_labels`` only, and with a caveat: it has no filter
that returns a label array. The closest thing that runs the component pass *without* also splitting
or deleting anything is ``compute_selection_by_small_disconnected_components_per_face`` at
``nbfaceratio=0.0`` -- it labels every component and then thresholds against a fraction of the
largest one, selecting nothing. So the row is "label everything, then one threshold pass", against
triwarp's "label everything". It is also mesh-bound rather than CSR-bound, which is why it cannot
appear in the ``bfs`` groups at all; the labelling is the only graph work MeshLab exposes.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import warp as wp
import warp.sparse as wps
from conftest import BenchCase
from scipy.sparse.csgraph import breadth_first_order

import triwarp as tw

# Source counts for the multi-source traversal. The output is a packed CSR of every reachable set,
# so on a single-component mesh its size is sources x V and the pair shows that directly.
_N_SOURCES = [16, 256]
_SOURCE_SEED = 3

# The long-diameter BFS runs into hundreds of milliseconds a call.
_ROUNDS = 3

_adjacency_cache: dict[tuple[str, str], wps.BsrMatrix] = {}
_scipy_cache: dict[str, sp.csr_matrix] = {}


def _adjacency(bench_case: BenchCase) -> wps.BsrMatrix:
    """Vertex adjacency as a warp BSR matrix -- an *input*, built once per (mesh, device)."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _adjacency_cache:
        unique_edges, _ = tw.edges.edges_unique(
            bench_case.faces_wp, n_vertices=bench_case.n_vertices
        )
        _adjacency_cache[key] = tw.graph.edges_to_csr(bench_case.n_vertices, unique_edges)
    return _adjacency_cache[key]


def _scipy_graph(bench_case: BenchCase) -> sp.csr_matrix:
    """Build the same adjacency as a scipy CSR matrix, for the exact-order BFS oracle."""
    if bench_case.mesh_name not in _scipy_cache:
        faces = bench_case.faces_np
        edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
        n = bench_case.n_vertices
        data = np.ones(edges.shape[0], dtype=np.float32)
        graph = sp.coo_matrix((data, (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
        _scipy_cache[bench_case.mesh_name] = graph
    return _scipy_cache[bench_case.mesh_name]


@pytest.mark.benchmark(group="connected_component_labels")
@pytest.mark.benchaxis("components")
@pytest.mark.benchlibs("triwarp", "scipy", "pymeshlab")
def test_connected_component_labels(bench_case: BenchCase) -> None:
    """ECL-CC hook and flatten, over 1 / 64 / 1024 components at a fixed face count."""
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "pymeshlab":
        # MeshLab has no filter that hands back a label array; the closest thing that *runs* the
        # component pass without also splitting the mesh is the small-component face selection,
        # which labels every component and then thresholds against a fraction of the largest one.
        # ``nbfaceratio=0.0`` selects nothing, so the timing is the labelling pass and the
        # threshold, not a deletion.
        meshset_pml = bench_case.meshset_pml
        bench_case.run(
            lambda: meshset_pml.compute_selection_by_small_disconnected_components_per_face(
                nbfaceratio=0.0
            )
        )
        assert meshset_pml.current_mesh().face_selection_array().shape == (bench_case.n_faces,)
        return
    if bench_case.kind == "triwarp":
        adjacency = _adjacency(bench_case)
        labels = bench_case.run(lambda: tw.graph.connected_component_labels(adjacency))
        assert labels.shape == (n_vertices,)
    else:
        graph = _scipy_graph(bench_case)
        n_labelled, labels_np = bench_case.run(lambda: sp.csgraph.connected_components(graph))
        assert n_labelled >= 1
        assert labels_np.shape == (n_vertices,)


@pytest.mark.benchmark(group="connected_component_labels_depth")
@pytest.mark.benchaxis("diameter")
@pytest.mark.benchlibs("triwarp", "scipy")
def test_connected_component_labels_depth(bench_case: BenchCase) -> None:
    """
    The depth-robustness gate: ECL-CC on a graph of diameter 130 against one of diameter 20 481.

    Pointer jumping has no level loop, so this pair should read **flat** -- measured at 0.08 ms on
    both. That is not a self-evident property (every level-synchronous traversal in the package
    fails it, which is what the ``bfs`` group below shows), and it is the premise the parity
    union-find in ``validation.face_orientation_bits`` rests on. A slope appearing here is the
    regression that would invalidate it.
    """
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "triwarp":
        adjacency = _adjacency(bench_case)
        labels = bench_case.run(lambda: tw.graph.connected_component_labels(adjacency))
        assert labels.shape == (n_vertices,)
    else:
        graph = _scipy_graph(bench_case)
        n_labelled, labels_np = bench_case.run(lambda: sp.csgraph.connected_components(graph))
        assert n_labelled == 1
        assert labels_np.shape == (n_vertices,)


@pytest.mark.benchmark(group="face_connected_component_labels_depth")
@pytest.mark.benchaxis("diameter")
@pytest.mark.benchlibs("triwarp")
def test_face_connected_component_labels_depth(bench_case: BenchCase) -> None:
    """
    Same gate one level up, over the face-adjacency graph.

    This is the build-plus-ECL-CC pair ``validation.face_orientation_bits`` actually calls.

    Measured at 1.33 ms (sphere) against 1.26 ms (ribbon) -- the edge sort dominates and neither
    half of it is depth-sensitive. No scipy counterpart: the comparison there would be against a
    face-adjacency matrix triwarp has to build anyway, which is the part being measured.
    """
    labels = bench_case.run(
        lambda: tw.adjacency.face_connected_component_labels(bench_case.faces_wp)
    )
    assert labels.shape == (bench_case.n_faces,)


@pytest.mark.benchmark(group="bfs")
@pytest.mark.benchaxis("diameter")
@pytest.mark.benchlibs("triwarp", "scipy")
def test_bfs(bench_case: BenchCase) -> None:
    """Level-synchronous frontier BFS with a serial escape once the frontier narrows: 5.8x here."""
    if bench_case.kind == "triwarp":
        adjacency = _adjacency(bench_case)
        order, _, _ = bench_case.run(lambda: tw.graph.bfs(adjacency, 0), rounds=_ROUNDS)
        assert order.shape[0] >= 1
    else:  # scipy exact-order oracle (see the module docstring)
        graph = _scipy_graph(bench_case)
        order, _ = bench_case.run(
            lambda: breadth_first_order(graph, 0, directed=False, return_predecessors=True),
            rounds=_ROUNDS,
        )
        assert order.shape[0] >= 1


@pytest.mark.benchmark(group="bfs_multi_source")
@pytest.mark.benchaxis("components")
@pytest.mark.benchlibs("triwarp")
@pytest.mark.parametrize("n_sources", _N_SOURCES)
def test_bfs_multi_source(bench_case: BenchCase, n_sources: int) -> None:
    """
    Packed reachable sets from many sources at once.

    On the component axis this is a coverage sweep as much as a timing one: on ``sphere_med`` every
    source reaches all 40 962 vertices, so the output is ``sources x V``, while on ``parts_1024``
    each source is trapped in its own 42-vertex sphere. Same call, output sizes three orders of
    magnitude apart.
    """
    adjacency = _adjacency(bench_case)
    rng = np.random.default_rng(_SOURCE_SEED)
    sources = wp.array(
        rng.integers(0, bench_case.n_vertices, size=n_sources).astype(np.int32),
        dtype=wp.int32,
        device=bench_case.device,
    )
    neighbors, offsets = bench_case.run(
        lambda: tw.graph.bfs_multi_source(adjacency, sources), rounds=_ROUNDS
    )
    assert offsets.shape == (n_sources,)
    assert neighbors.shape[0] >= n_sources
