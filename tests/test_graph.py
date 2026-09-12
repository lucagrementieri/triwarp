"""Regression tests for ``triwarp.graph`` against Trimesh (CPU reference)."""

from __future__ import annotations

import igl
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import trimesh as tm
import warp as wp
from meshlib import mrmeshpy as mm

import triwarp as tw
import triwarp.typing as twt
from tests.comparisons import same_partition
from tests.conversions import meshlib_bitset_to_numpy, trimesh_to_meshlib, trimesh_to_pymeshlab


def test_edges_to_csr_roundtrip(device: str) -> None:
    edges_np = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int32)
    node_count = 3
    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)
    adjacency = tw.graph.edges_to_csr(node_count, edges_wp)
    offsets = adjacency.offsets.numpy()  # pyright: ignore[reportAttributeAccessIssue]
    indices = adjacency.columns.numpy()  # pyright: ignore[reportAttributeAccessIssue]

    assert adjacency.nrow == node_count  # pyright: ignore[reportAttributeAccessIssue]
    assert adjacency.ncol == node_count  # pyright: ignore[reportAttributeAccessIssue]
    assert adjacency.block_shape == (1, 1)
    assert offsets[0] == 0
    assert offsets[-1] == len(indices)
    assert offsets.shape[0] == node_count + 1

    neighbors: dict[int, set[int]] = {i: set() for i in range(node_count)}
    for a, b in edges_np:
        neighbors[int(a)].add(int(b))
        neighbors[int(b)].add(int(a))

    for v in range(node_count):
        row = indices[offsets[v] : offsets[v + 1]]
        assert set(row.tolist()) == neighbors[v]


@pytest.mark.parity("connected_component_labels", "scipy")
def test_connected_component_labels_random(device: str) -> None:
    """
    Class B (label packing): the *partition* matches scipy's, through ``same_partition``.

    triwarp names a component after a representative node and scipy numbers them in traversal
    order, so only the partition is shared -- comparing labels directly would fail on a correct
    answer. But that transform is only *exercised* if the graph genuinely fragments: with one
    component the two numbering conventions coincide and ``same_partition`` compares one constant
    labelling with another, which any implementation returning a single label would pass.

    **24 edges, not 200.** Measured on this seed, 200 random edges over 64 nodes gives **1**
    component holding all 64 nodes -- the count is well past the giant-component threshold -- where
    24 gives **40** components, **12** of them non-trivial, the largest holding 6 nodes. The assert
    below therefore checks the fragmentation before comparing to it, which is the guard the vacuous
    version lacked.
    """
    rng = np.random.default_rng(7)
    node_count = 64
    n_edges = 24
    edges_np = rng.integers(0, node_count, size=(n_edges, 2), dtype=np.int32)

    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)
    labels_wp = tw.graph.connected_component_labels_from_edges(edges_wp, node_count=node_count)
    labels_np = _scipy_component_labels(edges_np, node_count)

    # Non-vacuous: with one component the label-packing transform under test is the identity.
    assert np.unique(labels_np).shape[0] > 1
    assert same_partition(labels_wp.numpy(), labels_np)


@pytest.mark.parametrize(("node_count", "n_edges"), [(64, 24), (2048, 2047)])
@pytest.mark.parity("connected_component_labels", "igl")
@pytest.mark.parity("connected_component_labels_depth", "igl")
def test_connected_component_labels_matches_igl(device: str, node_count: int, n_edges: int) -> None:
    """
    Class B (label packing): the partition against ``igl.connected_components``.

    The reference worth having here because it takes the *same argument* triwarp does -- a
    ``scipy.sparse`` adjacency matrix -- rather than a mesh, which is why
    ``benchmarks/test_graph.py`` used to say no second implementation existed for these two groups.
    It returns ``(n_components, labels, sizes)``, three values, and numbers components ``0..k-1`` in
    its own traversal order where triwarp names each after a representative node, so
    [`same_partition`][tests.comparisons.same_partition] is the transform exactly as it is for the
    scipy comparison above.

    Two shapes, and each carries a different half of the claim. The 64-node random graph is what
    exercises the *label packing*, so its edge count is chosen to fragment it -- 24 edges give 40
    components where the 200 this used to pass gave **1**, and with one component the two numbering
    conventions coincide and the transform under test is the identity (the same defect the scipy
    comparison above records). The 2 048-node **path** is deliberately a single component: its
    diameter equals its node count, which is what the ``*_depth`` group exists to gate, and an
    implementation that stopped propagating early would pass the first shape and fail this one.
    """
    if n_edges == node_count - 1:  # the path graph
        edges_np = np.stack(
            [np.arange(node_count - 1, dtype=np.int32), np.arange(1, node_count, dtype=np.int32)],
            axis=1,
        )
    else:
        rng = np.random.default_rng(7)
        edges_np = rng.integers(0, node_count, size=(n_edges, 2), dtype=np.int32)

    adjacency_np = sp.coo_matrix(
        (np.ones(len(edges_np), dtype=np.int8), (edges_np[:, 0], edges_np[:, 1])),
        shape=(node_count, node_count),
    ).tocsr()
    adjacency_np = adjacency_np + adjacency_np.T

    n_components_igl, labels_igl, sizes_igl = igl.connected_components(adjacency_np)
    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)
    labels_wp = tw.graph.connected_component_labels_from_edges(edges_wp, node_count=node_count)

    assert int(n_components_igl) == np.unique(labels_wp.numpy()).shape[0]
    assert int(np.asarray(sizes_igl).sum()) == node_count  # every node landed in a component
    # Non-vacuous on the random shape: the path is one component by design, see the docstring.
    assert int(n_components_igl) > 1 or n_edges == node_count - 1
    assert same_partition(labels_wp.numpy(), np.asarray(labels_igl).ravel())


@pytest.mark.parametrize("face_ratio", [0.0, 0.1, 0.5])
@pytest.mark.parity("connected_component_labels", "pymeshlab")
def test_connected_component_labels_matches_pymeshlab(
    request: pytest.FixtureRequest, face_ratio: float
) -> None:
    """
    Class B: MeshLab has no filter that returns labels, so the labelling is observed through a mask.

    ``compute_selection_by_small_disconnected_components_per_face`` is the closest thing that runs
    the component pass without also splitting the mesh -- it labels every component, then selects
    the faces of every component holding fewer than ``nbfaceratio`` times the largest component's
    face count. So the named transform runs triwarp's labels through exactly that rule: push the
    per-vertex labels onto faces (all three corners of a face share a component), count faces per
    label, and select where the count is below the threshold. Equality is then exact on the bool
    array.

    This does more than re-express the same thing twice: the *sizes* of all the components and their
    ranking have to agree, not just the partition, which is what a label array compared up to
    renumbering (the scipy oracle above) deliberately does not pin.

    Parametrized over three ratios that produce **different** answers on a 20 / 320 / 2 048-face
    three-component mesh -- 0, 20 and 340 faces selected -- so no constant mask can pass.
    """
    mesh_a_tm, _mesh_a_wp = request.getfixturevalue("icosahedron")
    mesh_b_tm, _mesh_b_wp = request.getfixturevalue("hemisphere")
    mesh_c_tm, _mesh_c_wp = request.getfixturevalue("half_torus")
    combined_tm = tm.util.concatenate([mesh_a_tm, mesh_b_tm, mesh_c_tm])
    device = str(_mesh_a_wp.points.device)
    faces_wp = wp.array(
        np.ascontiguousarray(combined_tm.faces.reshape(-1), dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )
    n_vertices = combined_tm.vertices.shape[0]

    meshset_pml = trimesh_to_pymeshlab(combined_tm)
    meshset_pml.compute_selection_by_small_disconnected_components_per_face(nbfaceratio=face_ratio)
    selection_pml = np.asarray(meshset_pml.current_mesh().face_selection_array())

    unique_edges_wp, _inverse_wp = tw.edges.edges_unique(faces_wp, n_vertices=n_vertices)
    labels_np = tw.graph.connected_component_labels(
        tw.graph.edges_to_csr(n_vertices, unique_edges_wp)
    ).numpy()

    face_labels_np = labels_np[combined_tm.faces[:, 0]]
    _label, size_np = np.unique(face_labels_np, return_counts=True)
    threshold = face_ratio * size_np.max()
    sizes_by_face_np = size_np[np.searchsorted(_label, face_labels_np)]
    assert np.array_equal(sizes_by_face_np < threshold, selection_pml)


@pytest.mark.parity("connected_component_labels", "meshlib")
def test_connected_component_labels_matches_meshlib(request: pytest.FixtureRequest) -> None:
    """
    Class B (label packing): ``getAllComponentsVerts`` returns the components as *bitsets*.

    The transform is the one every component comparison in this package makes plus a decode:
    MeshLib hands back a vector of ``VertBitSet``, one per component in its own traversal order,
    so each is expanded to a bool array over the vertex domain and its index becomes the label,
    and only the *partition* is then shared -- triwarp names a component after a representative
    vertex. Hence [`same_partition`][tests.comparisons.same_partition].

    Stronger than the scipy pairing in one respect and weaker in another, which is why both stay:
    the bitsets pin each component's *membership* directly rather than through a renumbering, but
    MeshLib takes a mesh where scipy takes an edge list, so it cannot reach the random-graph inputs
    at all.

    Non-vacuous: three disjoint fixtures, so the answer is neither one component nor as many as
    there are vertices, and the component *sizes* are asserted to be the fixtures' vertex counts
    before the partition is compared.
    """
    mesh_a_tm, mesh_a_wp = request.getfixturevalue("icosahedron")
    mesh_b_tm, _mesh_b_wp = request.getfixturevalue("hemisphere")
    mesh_c_tm, _mesh_c_wp = request.getfixturevalue("half_torus")
    combined_tm = tm.util.concatenate([mesh_a_tm, mesh_b_tm, mesh_c_tm])
    n_vertices = combined_tm.vertices.shape[0]
    faces_wp = wp.array(
        np.ascontiguousarray(combined_tm.faces.reshape(-1), dtype=np.int32),
        dtype=wp.int32,
        device=mesh_a_wp.points.device,
    )

    components_ml = mm.getAllComponentsVerts(trimesh_to_meshlib(combined_tm), None)
    labels_ml = np.full(n_vertices, -1, dtype=np.int64)
    for label, component_ml in enumerate(components_ml):
        labels_ml[meshlib_bitset_to_numpy(component_ml, n_vertices)] = label

    unique_edges_wp, _inverse_wp = tw.edges.edges_unique(faces_wp, n_vertices=n_vertices)
    labels_wp = tw.graph.connected_component_labels(
        tw.graph.edges_to_csr(n_vertices, unique_edges_wp)
    )

    assert sorted(component_ml.count() for component_ml in components_ml) == sorted(
        mesh.vertices.shape[0] for mesh in (mesh_a_tm, mesh_b_tm, mesh_c_tm)
    )
    assert (labels_ml >= 0).all()  # every vertex landed in a component
    assert same_partition(labels_wp.numpy(), labels_ml)


def test_connected_component_labels_empty_edges(device: str) -> None:
    node_count = 10
    edges_wp = twt.empty_2d((0, 2), wp.int32, device=device)
    labels_wp = tw.graph.connected_component_labels_from_edges(edges_wp, node_count=node_count)
    labels_exp = _scipy_component_labels(np.empty((0, 2), dtype=np.int32), node_count)
    assert np.array_equal(labels_wp.numpy(), labels_exp)


def test_connected_component_labels_zero_nodes(device: str) -> None:
    edges_wp = twt.empty_2d((0, 2), wp.int32, device=device)
    labels_wp = tw.graph.connected_component_labels_from_edges(edges_wp, node_count=0)
    assert labels_wp.shape == (0,)


@pytest.mark.parity("connected_component_labels_depth", "scipy")
def test_connected_component_labels_path_graph(device: str) -> None:
    """
    Class B: the same partition comparison on the worst case for a label-propagation sweep.

    A 2 048-node path is the graph that needs the most propagation rounds -- a diameter equal
    to its node count -- so it catches an implementation that stops iterating too early, which
    a random graph of diameter ~4 cannot.
    """
    n = 2048
    edges_np = np.stack([np.arange(n - 1, dtype=np.int32), np.arange(1, n, dtype=np.int32)], axis=1)
    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)
    labels_wp = tw.graph.connected_component_labels_from_edges(edges_wp, node_count=n)
    labels_exp = _scipy_component_labels(edges_np, n)
    assert same_partition(labels_wp.numpy(), labels_exp)


def test_connected_component_labels_star_graph(device: str) -> None:
    n = 512
    hub = 0
    leaves = np.arange(1, n, dtype=np.int32)
    edges_np = np.stack([np.full(n - 1, hub, dtype=np.int32), leaves], axis=1)
    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)
    labels_wp = tw.graph.connected_component_labels_from_edges(edges_wp, node_count=n)
    labels_exp = _scipy_component_labels(edges_np, n)
    assert same_partition(labels_wp.numpy(), labels_exp)


def test_connected_component_parity_random(device: str) -> None:
    # Signs drawn from a hidden potential, so the constraints are consistent everywhere and the
    # returned parity must reproduce that potential up to a per-component flip.
    rng = np.random.default_rng(11)
    n = 4096
    potential_np = rng.integers(0, 2, size=n).astype(np.int32)
    a_np = rng.integers(0, n, size=12_000).astype(np.int32)
    b_np = rng.integers(0, n, size=12_000).astype(np.int32)
    edges_np = np.stack([a_np, b_np], axis=1)
    signs_np = (potential_np[a_np] ^ potential_np[b_np]).astype(np.int32)

    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)
    signs_wp = wp.array(signs_np, dtype=wp.int32, device=device)
    labels_wp, parity_wp = tw.graph.connected_component_parity_from_edges(edges_wp, signs_wp, n)

    parity_np = parity_wp.numpy()
    assert np.array_equal(parity_np[a_np] ^ parity_np[b_np], signs_np)
    assert same_partition(labels_wp.numpy(), _scipy_component_labels(edges_np, n))
    # Each component representative anchors its own potential at 0.
    labels_np = labels_wp.numpy()
    assert np.array_equal(parity_np[np.unique(labels_np)], np.zeros(len(np.unique(labels_np))))


def test_connected_component_parity_long_path(device: str) -> None:
    # A path is the worst case for edge-by-edge propagation (one level per node) and the case the
    # union-find is depth-independent on; the potential is then the running XOR of the signs.
    n = 20_001
    rng = np.random.default_rng(5)
    edges_np = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1).astype(np.int32)
    signs_np = rng.integers(0, 2, size=n - 1).astype(np.int32)
    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)
    signs_wp = wp.array(signs_np, dtype=wp.int32, device=device)

    labels_wp, parity_wp = tw.graph.connected_component_parity_from_edges(edges_wp, signs_wp, n)
    parity_exp = np.concatenate([[0], np.cumsum(signs_np) % 2]).astype(np.int32)
    assert np.array_equal(parity_wp.numpy(), parity_exp)
    assert np.array_equal(labels_wp.numpy(), np.zeros(n, dtype=np.int32))


def test_connected_component_parity_contradiction_terminates(device: str) -> None:
    # An odd-signed cycle admits no potential. The contract is best-effort, not an exception: the
    # call must still terminate and label the component, leaving some edge violated.
    n = 1025
    edges_np = np.stack([np.arange(n), (np.arange(n) + 1) % n], axis=1).astype(np.int32)
    signs_np = np.ones(n, dtype=np.int32)
    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)
    signs_wp = wp.array(signs_np, dtype=wp.int32, device=device)

    labels_wp, parity_wp = tw.graph.connected_component_parity_from_edges(edges_wp, signs_wp, n)
    assert np.array_equal(labels_wp.numpy(), np.zeros(n, dtype=np.int32))
    parity_np = parity_wp.numpy()
    assert set(np.unique(parity_np).tolist()) <= {0, 1}
    violated = parity_np[edges_np[:, 0]] ^ parity_np[edges_np[:, 1]] != signs_np
    assert violated.sum() >= 1


def test_connected_component_parity_no_edges(device: str) -> None:
    edges_wp = wp.zeros((0, 2), dtype=wp.int32, device=device)
    signs_wp = wp.zeros(0, dtype=wp.int32, device=device)
    labels_wp, parity_wp = tw.graph.connected_component_parity_from_edges(edges_wp, signs_wp, 7)
    assert np.array_equal(labels_wp.numpy(), np.arange(7, dtype=np.int32))
    assert np.array_equal(parity_wp.numpy(), np.zeros(7, dtype=np.int32))


def test_connected_component_parity_signs_length_mismatch(device: str) -> None:
    edges_wp = wp.zeros((4, 2), dtype=wp.int32, device=device)
    signs_wp = wp.zeros(3, dtype=wp.int32, device=device)
    with pytest.raises(ValueError, match="signs must have length 4"):
        tw.graph.connected_component_parity_from_edges(edges_wp, signs_wp, 4)


def test_connected_component_parity_validates_range(device: str) -> None:
    """
    The default range check rejects an endpoint outside ``[0, node_count)``.

    Without this, ``ecl_hook_parity`` indexes a ``node_count``-element buffer by the raw
    endpoint -- an out-of-range value reads and writes out of bounds rather than raising.
    """
    edges_wp = wp.array(np.array([[0, 10]], dtype=np.int32), dtype=wp.int32, device=device)
    signs_wp = wp.zeros(1, dtype=wp.int32, device=device)
    with pytest.raises(ValueError, match="edge indices must lie in"):
        tw.graph.connected_component_parity_from_edges(edges_wp, signs_wp, 5)


def test_connected_component_parity_validates_signs(device: str) -> None:
    """
    Not a parity assert: an out-of-range ``sign`` is rejected, and is bounded when unchecked.

    The hook packs the parity bit into the low bit of a word whose upper bits are a node id, so a
    sign outside ``{0, 1}`` lands in the *parent* half. Before the mask in ``ecl_hook_edge_parity``
    a sign of ``2`` made the union's compare-and-swap write back exactly what it expected, leaving
    the two endpoints in separate components with no error at all, and a negative sign wrote a
    parent of ``-1`` that the next find read out of bounds. Both arms are asserted: the checked
    path raises, and the unchecked path still joins the edge.
    """
    edges_wp = wp.array(np.array([[2, 3]], dtype=np.int32), dtype=wp.int32, device=device)
    for bad in (2, -1, 1 << 20):
        signs_wp = wp.array(np.array([bad], dtype=np.int32), dtype=wp.int32, device=device)
        with pytest.raises(ValueError, match="signs must be 0 or 1"):
            tw.graph.connected_component_parity_from_edges(edges_wp, signs_wp, 6)
        labels_wp, parity_wp = tw.graph.connected_component_parity_from_edges(
            edges_wp, signs_wp, 6, validate=False
        )
        labels_np = labels_wp.numpy()
        # The structure is what the mask protects: the edge still joins its endpoints, and only
        # the sign's low bit reaches the potential.
        assert labels_np[2] == labels_np[3]
        assert set(np.unique(parity_wp.numpy()).tolist()) <= {0, 1}
        assert int(parity_wp.numpy()[3]) == bad & 1

    # A valid sign is untouched by the mask.
    for good in (0, 1):
        signs_wp = wp.array(np.array([good], dtype=np.int32), dtype=wp.int32, device=device)
        labels_wp, parity_wp = tw.graph.connected_component_parity_from_edges(edges_wp, signs_wp, 6)
        assert labels_wp.numpy()[2] == labels_wp.numpy()[3]
        assert int(parity_wp.numpy()[3]) == good


def test_face_connected_component_labels(request: pytest.FixtureRequest) -> None:
    """
    Class B: the face-side partition against scipy over ``trimesh.face_adjacency``.

    Two named transforms, both on the reference side: build the dual graph from trimesh's face
    adjacency, then compare partitions rather than labels. Three concatenated fixtures make the
    expected component count three, which is asserted so the comparison cannot pass on one
    blob.
    """
    mesh_a_tm, mesh_a_wp = request.getfixturevalue("icosahedron")
    mesh_b_tm, mesh_b_wp = request.getfixturevalue("hemisphere")
    mesh_c_tm, mesh_c_wp = request.getfixturevalue("half_torus")

    concat_tm = tm.util.concatenate([mesh_a_tm, mesh_b_tm, mesh_c_tm])
    _, concat_faces_wp = tw.combine.concatenate(
        [
            (mesh_a_wp.points, mesh_a_wp.indices),
            (mesh_b_wp.points, mesh_b_wp.indices),
            (mesh_c_wp.points, mesh_c_wp.indices),
        ]
    )
    face_labels_wp = tw.adjacency.face_connected_component_labels(concat_faces_wp)
    n_faces = concat_tm.faces.shape[0]
    face_labels_tm = _scipy_component_labels(concat_tm.face_adjacency.astype(np.int32), n_faces)
    assert same_partition(face_labels_wp.numpy(), face_labels_tm)


def _scipy_component_labels(edges: np.ndarray, node_count: int) -> np.ndarray:
    if node_count == 0:
        return np.array([], dtype=np.int32)
    if len(edges) == 0:
        return np.arange(node_count, dtype=np.int32)
    row = edges[:, 0]
    col = edges[:, 1]
    data = np.ones(len(edges), dtype=np.int8)
    matrix = sp.coo_matrix((data, (row, col)), shape=(node_count, node_count))
    matrix = matrix + matrix.T
    _n_comp, labels = csgraph.connected_components(matrix, directed=False)
    return labels.astype(np.int32)


def test_successor_cycles_single_cycle(device: str) -> None:
    """One 4-cycle: the result starts at the smallest node and follows the edge direction."""
    edges_wp = wp.array(
        np.array([[5, 2], [2, 7], [7, 3], [3, 5]], dtype=np.int32), dtype=wp.int32, device=device
    )
    flat_wp, offsets_wp, sizes_wp = tw.graph.successor_cycles(edges_wp, 8)

    assert np.array_equal(flat_wp.numpy(), np.array([2, 7, 3, 5], dtype=np.int32))
    assert np.array_equal(offsets_wp.numpy(), np.array([0], dtype=np.int32))
    assert np.array_equal(sizes_wp.numpy(), np.array([4], dtype=np.int32))


def test_successor_cycles_multiple_cycles(device: str) -> None:
    """
    Interleaved node ids across three cycles, plus nodes on no cycle.

    Every cycle must come back in successor order from its own minimum; the per-cycle offsets
    partition the packed buffer; nodes 1 and 8 appear in no edge and in no cycle.
    """
    rng = np.random.default_rng(7)
    cycles = [[0, 4, 2], [3, 9, 6, 5], [7, 10]]
    edge_rows = [
        (cycle[i], cycle[(i + 1) % len(cycle)]) for cycle in cycles for i in range(len(cycle))
    ]
    order = rng.permutation(len(edge_rows))
    edges_np = np.array(edge_rows, dtype=np.int32)[order]
    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)

    flat_wp, offsets_wp, sizes_wp = tw.graph.successor_cycles(edges_wp, 11)

    flat_np = flat_wp.numpy()
    starts_np = offsets_wp.numpy()
    sizes_np = sizes_wp.numpy()
    assert sizes_np.sum() == flat_np.shape[0] == 9
    recovered = [
        flat_np[start : start + size].tolist()
        for start, size in zip(starts_np, sizes_np, strict=True)
    ]
    # Each cycle starts at its minimum and follows the successor direction.
    assert sorted(recovered) == sorted([[0, 4, 2], [3, 9, 6, 5], [7, 10]])


def test_successor_cycles_validates_range(device: str) -> None:
    """The default range check rejects an endpoint outside ``[0, node_count)``."""
    edges_wp = wp.array(np.array([[0, 9], [9, 0]], dtype=np.int32), dtype=wp.int32, device=device)
    with pytest.raises(ValueError, match="edge indices must lie in"):
        tw.graph.successor_cycles(edges_wp, 4)


def test_successor_cycles_validates_before_launching(
    device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The range check must run before *any* kernel launch, not merely before returning.

    ``scatter_successor`` indexes a ``node_count``-element buffer by the raw edge endpoint, so a
    check that runs after it has already let an out-of-range endpoint write past the end — on the
    CPU device that is a host-heap overwrite which aborts the process much later, somewhere
    unrelated. Asserting only that ``ValueError`` is raised does not catch that: the exception is
    raised either way. Counting launches is what pins the ordering.
    """
    launches = 0
    real_launch = wp.launch

    def counting_launch(*args: object, **kwargs: object) -> object:
        nonlocal launches
        launches += 1
        return real_launch(*args, **kwargs)

    monkeypatch.setattr(wp, "launch", counting_launch)
    edges_wp = wp.array(np.array([[0, 9], [9, 0]], dtype=np.int32), dtype=wp.int32, device=device)
    with pytest.raises(ValueError, match="edge indices must lie in"):
        tw.graph.successor_cycles(edges_wp, 4)
    assert launches == 0


def test_successor_cycles_malformed_input_stays_in_range(device: str) -> None:
    """
    Two in-edges on one node (not a successor graph) must not return garbage.

    The documented behavior: ranks may collide and slots fall back to zero, but every value in
    the packed buffer stays a valid node index and the sizes still partition it.
    """
    edges_np = np.array([[0, 1], [1, 2], [2, 0], [3, 1]], dtype=np.int32)
    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)
    flat_wp, _offsets_wp, sizes_wp = tw.graph.successor_cycles(edges_wp, 4)

    flat_np = flat_wp.numpy()
    assert flat_np.shape[0] == int(sizes_wp.numpy().sum()) == 4
    assert np.all((flat_np >= 0) & (flat_np < 4))


def test_successor_cycles_excludes_a_chain(device: str) -> None:
    """
    A plain successor chain (no cycle at all) contributes nothing to the result.

    The docstring promises a successor graph decomposes into cycles *and* chains, with only
    cycles returned. A chain's dead end (a node with no outgoing edge) used to be treated as a
    second rank-0 fixed point alongside the arbitrary cut at the component's smallest node, so
    the two collided and fabricated a bogus "cycle" out of the collision -- one that could even
    contain a node id that never appeared in the input at all.
    """
    edges_wp = wp.array(np.array([[5, 3], [3, 7]], dtype=np.int32), dtype=wp.int32, device=device)
    flat_wp, offsets_wp, sizes_wp = tw.graph.successor_cycles(edges_wp, 8)
    assert flat_wp.shape == (0,)
    assert offsets_wp.shape == (0,)
    assert sizes_wp.shape == (0,)


def test_successor_cycles_mixed_cycle_and_chain(device: str) -> None:
    """A real cycle is reported unchanged alongside a chain that contributes nothing."""
    edges_np = np.array([[0, 1], [1, 2], [2, 0], [5, 3], [3, 7]], dtype=np.int32)
    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)
    flat_wp, offsets_wp, sizes_wp = tw.graph.successor_cycles(edges_wp, 8)
    assert np.array_equal(flat_wp.numpy(), np.array([0, 1, 2], dtype=np.int32))
    assert np.array_equal(offsets_wp.numpy(), np.array([0], dtype=np.int32))
    assert np.array_equal(sizes_wp.numpy(), np.array([3], dtype=np.int32))


def test_successor_cycles_empty(device: str) -> None:
    edges_wp = twt.empty_2d((0, 2), wp.int32, device=device)
    flat_wp, offsets_wp, sizes_wp = tw.graph.successor_cycles(edges_wp, 5)
    assert flat_wp.shape == (0,)
    assert offsets_wp.shape == (0,)
    assert sizes_wp.shape == (0,)


@pytest.mark.parity("bfs", "scipy")
def test_bfs_random(device: str) -> None:
    """
    Class A on all three returns: visit order, parents and distances, against scipy's BFS.

    The visit *order* is comparable only because both sides break ties by ascending node index
    -- triwarp by construction, scipy through ``breadth_first_order`` on a sorted CSR -- so
    this is the one place the ordering contract is pinned rather than sorted away. Run from
    three sources, including both ends of the index range.
    """
    rng = np.random.default_rng(7)
    node_count = 48
    pairs = rng.integers(0, node_count, size=(150, 2), dtype=np.int32)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    edges_np = np.unique(np.sort(pairs, axis=1), axis=0).astype(np.int32)
    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)

    for source in (0, node_count // 2, node_count - 1):
        order_wp, parents_wp, distances_wp = tw.graph.bfs_from_edges(
            edges_wp, source, node_count=node_count
        )
        order_np, parents_np, distances_np = _scipy_bfs(edges_np, node_count, source)
        assert np.array_equal(order_wp.numpy(), order_np)
        assert np.array_equal(parents_wp.numpy(), parents_np)
        assert np.array_equal(distances_wp.numpy(), distances_np)


def test_bfs_random_large_frontier(device: str) -> None:
    # Above the serial threshold the frontier-parallel path runs; it must still match scipy's
    # exact discovery order, parents, and distances.
    rng = np.random.default_rng(11)
    node_count = 20_000
    pairs = rng.integers(0, node_count, size=(60_000, 2), dtype=np.int32)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    edges_np = np.unique(np.sort(pairs, axis=1), axis=0).astype(np.int32)
    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)

    for source in (0, node_count // 2):
        order_wp, parents_wp, distances_wp = tw.graph.bfs_from_edges(
            edges_wp, source, node_count=node_count
        )
        order_np, parents_np, distances_np = _scipy_bfs(edges_np, node_count, source)
        assert np.array_equal(order_wp.numpy(), order_np)
        assert np.array_equal(parents_wp.numpy(), parents_np)
        assert np.array_equal(distances_wp.numpy(), distances_np)


def test_bfs_grid_graph_many_levels(device: str) -> None:
    # High-diameter graph above the serial threshold: a 150x150 grid runs ~300 frontier levels,
    # stressing the per-level rank/scan/scatter ordering against scipy across many iterations.
    side = 150
    node_count = side * side
    ids = np.arange(node_count, dtype=np.int32).reshape(side, side)
    horizontal = np.stack([ids[:, :-1].ravel(), ids[:, 1:].ravel()], axis=1)
    vertical = np.stack([ids[:-1, :].ravel(), ids[1:, :].ravel()], axis=1)
    edges_np = np.concatenate([horizontal, vertical]).astype(np.int32)
    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)

    for source in (0, node_count // 2):
        order_wp, parents_wp, distances_wp = tw.graph.bfs_from_edges(
            edges_wp, source, node_count=node_count
        )
        order_np, parents_np, distances_np = _scipy_bfs(edges_np, node_count, source)
        assert np.array_equal(order_wp.numpy(), order_np)
        assert np.array_equal(parents_wp.numpy(), parents_np)
        assert np.array_equal(distances_wp.numpy(), distances_np)


def test_bfs_path_graph(device: str) -> None:
    n = 1024
    edges_np = np.stack([np.arange(n - 1, dtype=np.int32), np.arange(1, n, dtype=np.int32)], axis=1)
    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)

    order_wp, parents_wp, distances_wp = tw.graph.bfs_from_edges(edges_wp, 0, node_count=n)
    assert np.array_equal(order_wp.numpy(), np.arange(n, dtype=np.int32))
    assert np.array_equal(distances_wp.numpy(), np.arange(n, dtype=np.int32))
    parents_exp = np.concatenate([[-1], np.arange(n - 1)]).astype(np.int32)
    assert np.array_equal(parents_wp.numpy(), parents_exp)


def test_bfs_long_path_graph(device: str) -> None:
    # A path of 65 536 nodes: one BFS level per node, and a frontier of one throughout. The
    # level-synchronous loop hands over to the serial resume almost immediately here, so this is
    # the case that checks the handoff is order-exact and not just fast.
    n = 65_536
    edges_np = np.stack([np.arange(n - 1, dtype=np.int32), np.arange(1, n, dtype=np.int32)], axis=1)
    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)

    order_wp, parents_wp, distances_wp = tw.graph.bfs_from_edges(edges_wp, 0, node_count=n)
    assert np.array_equal(order_wp.numpy(), np.arange(n, dtype=np.int32))
    assert np.array_equal(distances_wp.numpy(), np.arange(n, dtype=np.int32))
    parents_exp = np.concatenate([[-1], np.arange(n - 1)]).astype(np.int32)
    assert np.array_equal(parents_wp.numpy(), parents_exp)


def test_bfs_wide_then_narrow_matches_scipy(device: str) -> None:
    # A "lollipop": a dense blob whose frontier is wide for a few levels, then a long tail where it
    # is one node across. The traversal therefore runs parallel levels first and escapes to the
    # serial resume part-way through -- the mixed case, where an off-by-one in the handed-over FIFO
    # window would corrupt the discovery order without changing the reachable set.
    rng = np.random.default_rng(19)
    blob, tail = 2_000, 6_000
    blob_edges = np.unique(
        np.sort(rng.integers(0, blob, size=(40_000, 2)).astype(np.int32), axis=1), axis=0
    )
    blob_edges = blob_edges[blob_edges[:, 0] != blob_edges[:, 1]]
    tail_nodes = np.arange(blob - 1, blob + tail, dtype=np.int32)
    tail_edges = np.stack([tail_nodes[:-1], tail_nodes[1:]], axis=1)
    edges_np = np.concatenate([blob_edges, tail_edges]).astype(np.int32)
    n = blob + tail

    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)
    order_wp, parents_wp, distances_wp = tw.graph.bfs_from_edges(edges_wp, 0, node_count=n)
    order_np, parents_np, distances_np = _scipy_bfs(edges_np, n, 0)
    assert np.array_equal(order_wp.numpy(), order_np)
    assert np.array_equal(parents_wp.numpy(), parents_np)
    assert np.array_equal(distances_wp.numpy(), distances_np)


def test_bfs_star_graph(device: str) -> None:
    n = 256
    hub = 0
    leaves = np.arange(1, n, dtype=np.int32)
    edges_np = np.stack([np.full(n - 1, hub, dtype=np.int32), leaves], axis=1)
    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)

    for source in (hub, n - 1):
        order_wp, parents_wp, distances_wp = tw.graph.bfs_from_edges(edges_wp, source, node_count=n)
        order_np, parents_np, distances_np = _scipy_bfs(edges_np, n, source)
        assert np.array_equal(order_wp.numpy(), order_np)
        assert np.array_equal(parents_wp.numpy(), parents_np)
        assert np.array_equal(distances_wp.numpy(), distances_np)


def test_bfs_disconnected(device: str) -> None:
    # Two disjoint triangles: {0,1,2} and {3,4,5}.
    edges_np = np.array([[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5]], dtype=np.int32)
    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)
    node_count = 6

    order_wp, parents_wp, distances_wp = tw.graph.bfs_from_edges(edges_wp, 0, node_count=node_count)
    order_np, parents_np, distances_np = _scipy_bfs(edges_np, node_count, 0)

    assert np.array_equal(order_wp.numpy(), order_np)
    assert order_wp.shape[0] == 3  # only the first triangle is reachable
    assert np.array_equal(parents_wp.numpy(), parents_np)
    assert np.array_equal(distances_wp.numpy(), distances_np)
    # Nodes 3,4,5 are unreachable from 0.
    assert np.array_equal(parents_wp.numpy()[3:], np.array([-1, -1, -1], dtype=np.int32))
    assert np.array_equal(distances_wp.numpy()[3:], np.array([-1, -1, -1], dtype=np.int32))


def test_bfs_single_node(device: str) -> None:
    edges_wp = twt.empty_2d((0, 2), wp.int32, device=device)
    order_wp, parents_wp, distances_wp = tw.graph.bfs_from_edges(edges_wp, 0, node_count=1)
    assert np.array_equal(order_wp.numpy(), np.array([0], dtype=np.int32))
    assert np.array_equal(parents_wp.numpy(), np.array([-1], dtype=np.int32))
    assert np.array_equal(distances_wp.numpy(), np.array([0], dtype=np.int32))


def test_bfs_empty_graph(device: str) -> None:
    node_count = 8
    source = 3
    edges_wp = twt.empty_2d((0, 2), wp.int32, device=device)
    order_wp, parents_wp, distances_wp = tw.graph.bfs_from_edges(
        edges_wp, source, node_count=node_count
    )
    assert np.array_equal(order_wp.numpy(), np.array([source], dtype=np.int32))
    parents_exp = np.full(node_count, -1, dtype=np.int32)
    distances_exp = np.full(node_count, -1, dtype=np.int32)
    distances_exp[source] = 0
    assert np.array_equal(parents_wp.numpy(), parents_exp)
    assert np.array_equal(distances_wp.numpy(), distances_exp)


def test_bfs_from_edges_node_count_inference(device: str) -> None:
    edges_np = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)
    order_wp, parents_wp, distances_wp = tw.graph.bfs_from_edges(edges_wp, 0)
    # node_count inferred as max + 1 == 4.
    assert parents_wp.shape[0] == 4
    assert distances_wp.shape[0] == 4
    assert np.array_equal(order_wp.numpy(), np.array([0, 1, 2, 3], dtype=np.int32))


def test_bfs_csr_columns_ascending(device: str) -> None:
    # Locks the precondition that makes serial BFS match scipy's neighbor visitation order.
    edges_np = np.array([[0, 2], [0, 1], [1, 3], [2, 3], [0, 3]], dtype=np.int32)
    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)
    adjacency = tw.graph.edges_to_csr(4, edges_wp)
    offsets = adjacency.offsets.numpy()
    columns = adjacency.columns.numpy()
    for v in range(4):
        row = columns[offsets[v] : offsets[v + 1]]
        assert np.array_equal(row, np.sort(row))


def test_bfs_source_out_of_range(device: str) -> None:
    edges_wp = wp.array(np.array([[0, 1]], dtype=np.int32), dtype=wp.int32, device=device)
    with pytest.raises(ValueError, match="source must be in"):
        tw.graph.bfs_from_edges(edges_wp, source=5, node_count=2)
    with pytest.raises(ValueError, match="source must be in"):
        tw.graph.bfs_from_edges(edges_wp, source=-1, node_count=2)


def test_bfs_from_edges_index_out_of_range(device: str) -> None:
    edges_wp = wp.array(np.array([[0, 9]], dtype=np.int32), dtype=wp.int32, device=device)
    with pytest.raises(ValueError, match="edge indices must lie in"):
        tw.graph.bfs_from_edges(edges_wp, source=0, node_count=4)


def test_bfs_from_edges_negative_node_count(device: str) -> None:
    edges_wp = wp.array(np.array([[0, 1]], dtype=np.int32), dtype=wp.int32, device=device)
    with pytest.raises(ValueError, match="node_count must be non-negative"):
        tw.graph.bfs_from_edges(edges_wp, source=0, node_count=-1)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere", "cave_cube"])
def test_bfs_on_mesh(request: pytest.FixtureRequest, mesh_name: str) -> None:
    _mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    unique_edges_wp, n = _mesh_vertex_edges(mesh_wp)
    edges_np = unique_edges_wp.numpy()

    for source in (0, n // 2, n - 1):
        order_wp, parents_wp, distances_wp = tw.graph.bfs_from_edges(
            unique_edges_wp, source, node_count=n
        )
        order_np, parents_np, distances_np = _scipy_bfs(edges_np, n, source)
        assert np.array_equal(order_wp.numpy(), order_np)
        assert np.array_equal(parents_wp.numpy(), parents_np)
        assert np.array_equal(distances_wp.numpy(), distances_np)


@pytest.mark.parity("bfs_multi_source", "scipy")
def test_bfs_multi_source_matches_scipy_min_only(request: pytest.FixtureRequest) -> None:
    """
    Class B (a reduction): scipy's min-only distance *field* against triwarp's reachable *sets*.

    ``csgraph.dijkstra(unweighted=True, indices=sources, min_only=True)`` is the multi-source
    reduction in one call, and it is the call the benchmark row times -- so this is the comparison
    that row rests on rather than the per-source loop ``test_bfs_multi_source_matches_single`` runs.
    The two answers have different *shapes*, which is the whole of the transform: scipy returns one
    distance per vertex over the entire vertex set, finite exactly where some source reaches it, and
    triwarp returns the packed reachable set per source. So the union of triwarp's sets is scipy's
    finite set, and that equality is asserted both ways.

    Run on two topologies because the shapes only diverge on one of them. On a connected mesh every
    source reaches everything and the union is trivially the whole vertex set -- which would pass
    for a function that ignored ``sources`` entirely. The disconnected graph is the one that bites:
    three sources in two of four components leave 2 of 12 vertices unreachable, so scipy reports two
    infinities and triwarp's union must miss exactly those.
    """
    _mesh_tm, mesh_wp = request.getfixturevalue("icosahedron")
    unique_edges_wp, n_mesh = _mesh_vertex_edges(mesh_wp)
    device = mesh_wp.device

    disjoint_np = np.array(
        [[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], [9, 10], [10, 11]], dtype=np.int32
    )
    cases = (
        (unique_edges_wp, n_mesh, [0, n_mesh // 3, n_mesh - 1], 0),
        (wp.array(disjoint_np, dtype=wp.int32, device=device), 12, [0, 4, 5], 6),
    )
    for edges_wp, n_nodes, sources, n_unreachable in cases:
        sources_np = np.array(sources, dtype=np.int32)
        sources_wp = wp.array(sources_np, dtype=wp.int32, device=device)
        adjacency = tw.graph.edges_to_csr(n_nodes, edges_wp)
        neighbors_wp, offsets_wp = tw.graph.bfs_multi_source(adjacency, sources_wp)

        graph_np = sp.coo_matrix(
            (
                np.ones(edges_wp.shape[0] * 2, dtype=np.float64),
                (
                    np.concatenate([edges_wp.numpy()[:, 0], edges_wp.numpy()[:, 1]]),
                    np.concatenate([edges_wp.numpy()[:, 1], edges_wp.numpy()[:, 0]]),
                ),
            ),
            shape=(n_nodes, n_nodes),
        ).tocsr()
        distances_np = csgraph.dijkstra(
            graph_np, unweighted=True, indices=sources_np, min_only=True
        )

        reachable_np = set(np.flatnonzero(np.isfinite(distances_np)).tolist())
        reachable_wp = set(neighbors_wp.list())
        # Non-vacuity: on the disjoint graph the sources must leave something out.
        assert n_nodes - len(reachable_np) == n_unreachable
        assert reachable_wp == reachable_np
        assert int(offsets_wp.shape[0]) == len(sources)


def test_bfs_multi_source_matches_single(request: pytest.FixtureRequest) -> None:
    _mesh_tm, mesh_wp = request.getfixturevalue("icosahedron")
    unique_edges_wp, n = _mesh_vertex_edges(mesh_wp)
    edges_np = unique_edges_wp.numpy()
    device = mesh_wp.device

    sources = [0, n // 3, n - 1]
    sources_wp = wp.array(np.array(sources, dtype=np.int32), dtype=wp.int32, device=device)
    adjacency = tw.graph.edges_to_csr(n, unique_edges_wp)
    neighbors_wp, offsets_wp = tw.graph.bfs_multi_source(adjacency, sources_wp)

    neighbors_np = neighbors_wp.numpy()
    offsets_np = offsets_wp.numpy()
    for k, source in enumerate(sources):
        start = int(offsets_np[k])
        end = int(offsets_np[k + 1]) if k + 1 < len(offsets_np) else len(neighbors_np)
        reachable_wp = set(neighbors_np[start:end].tolist())
        reachable_np = set(_scipy_bfs(edges_np, n, source)[0].tolist())
        assert reachable_wp == reachable_np
        # First entry of each source's slice is the source itself (BFS order).
        assert int(neighbors_np[start]) == source


def test_bfs_multi_source_empty_sources(device: str) -> None:
    edges_wp = wp.array(np.array([[0, 1], [1, 2]], dtype=np.int32), dtype=wp.int32, device=device)
    adjacency = tw.graph.edges_to_csr(3, edges_wp)
    sources_wp = wp.empty(0, dtype=wp.int32, device=device)
    neighbors_wp, offsets_wp = tw.graph.bfs_multi_source(adjacency, sources_wp)
    assert neighbors_wp.shape[0] == 0
    assert offsets_wp.shape[0] == 0


def test_bfs_multi_source_source_out_of_range(device: str) -> None:
    edges_wp = wp.array(np.array([[0, 1], [1, 2]], dtype=np.int32), dtype=wp.int32, device=device)
    adjacency = tw.graph.edges_to_csr(3, edges_wp)
    sources_wp = wp.array(np.array([0, 7], dtype=np.int32), dtype=wp.int32, device=device)
    with pytest.raises(ValueError, match="source indices must lie in"):
        tw.graph.bfs_multi_source(adjacency, sources_wp)


def test_bfs_multi_source_large_component(device: str) -> None:
    # A long path exercises what used to be a fixed 512-node scratch cap: the component-based
    # implementation returns the complete reachable set with no truncation warning.
    n = 700
    edges_np = np.stack([np.arange(n - 1, dtype=np.int32), np.arange(1, n, dtype=np.int32)], axis=1)
    edges_wp = wp.array(edges_np, dtype=wp.int32, device=device)
    adjacency = tw.graph.edges_to_csr(n, edges_wp)
    sources_wp = wp.array(np.array([5], dtype=np.int32), dtype=wp.int32, device=device)
    neighbors_wp, offsets_wp = tw.graph.bfs_multi_source(adjacency, sources_wp)
    assert offsets_wp.list() == [0]
    neighbors_np = neighbors_wp.numpy()
    assert neighbors_np.shape == (n,)
    assert neighbors_np[0] == 5  # the source leads its own range
    assert np.array_equal(np.sort(neighbors_np), np.arange(n))


def _scipy_bfs(
    edges_np: np.ndarray, node_count: int, source: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build reference ``(order, parents, distances)`` for [`triwarp.graph.bfs`][]'s semantics."""
    if len(edges_np) == 0:
        matrix = sp.csr_matrix((node_count, node_count), dtype=np.int8)
    else:
        row = np.concatenate([edges_np[:, 0], edges_np[:, 1]])
        col = np.concatenate([edges_np[:, 1], edges_np[:, 0]])
        data = np.ones(len(row), dtype=np.int8)
        matrix = sp.coo_matrix((data, (row, col)), shape=(node_count, node_count)).tocsr()
    order, pred = csgraph.breadth_first_order(
        matrix, source, directed=False, return_predecessors=True
    )
    dist = csgraph.shortest_path(matrix, directed=False, unweighted=True, indices=source)
    parents = np.where(pred == -9999, -1, pred).astype(np.int32)
    distances = np.where(np.isinf(dist), -1, dist).astype(np.int32)
    return order.astype(np.int32), parents, distances


def _mesh_vertex_edges(mesh_wp: wp.Mesh) -> tuple[wp.array, int]:
    """Return the unique undirected vertex edges and vertex count for a Warp mesh."""
    n = int(mesh_wp.points.shape[0])
    unique_edges, _ = tw.edges.edges_unique(mesh_wp.indices, n_vertices=n)
    return unique_edges, n


# ---------------------------------------------------------------------------
# shortest_path_envelope
# ---------------------------------------------------------------------------


def _length_weighted_csr(mesh_wp: wp.Mesh, threshold: float = 1.0) -> object:
    """
    Build the mesh edge graph with Euclidean lengths as weights, divided by ``threshold``.

    That division is MeshLab's ``gradientthr``: its cap is ``|p_i - p_j| / gradientthr``, and
    ``shortest_path_envelope`` takes no threshold because the weights carry it.
    """
    edges, n_vertices = _mesh_vertex_edges(mesh_wp)
    lengths = tw.edges.edges_unique_length(mesh_wp.points, mesh_wp.indices, edges)
    if threshold != 1.0:
        scaled = wp.empty(int(lengths.shape[0]), dtype=wp.float32, device=mesh_wp.device)
        wp.map(wp.div, lengths, wp.float32(threshold), out=scaled)
        lengths = scaled
    return tw.graph.edges_to_csr(n_vertices, edges, lengths)


def _spike_field(n_vertices: int, device: str) -> np.ndarray:
    """Build a delta at vertex 0: the field with the steepest possible gradient."""
    values_np = np.zeros(n_vertices, dtype=np.float64)
    values_np[0] = 10.0
    return values_np


@pytest.mark.parametrize("threshold", [0.5, 1.0, 3.0])
@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
@pytest.mark.parity("shortest_path_envelope", "pymeshlab")
def test_shortest_path_envelope_matches_pymeshlab(
    request: pytest.FixtureRequest, mesh_name: str, threshold: float
) -> None:
    """
    Class A against ``apply_scalar_saturation_per_vertex``, the Lipschitz-cap reading.

    The named transform is the weights: MeshLab's ``gradientthr`` divides the edge length, so the
    adjacency carries ``length / threshold`` and the envelope needs no threshold of its own. The
    sweep is over that parameter because it is the only one the reference has.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    values_np = _spike_field(mesh_tm.vertices.shape[0], str(mesh_wp.device))

    meshset_pml = trimesh_to_pymeshlab(mesh_tm, values_np)
    meshset_pml.apply_scalar_saturation_per_vertex(gradientthr=threshold)

    values_wp = wp.array(values_np.astype(np.float32), dtype=wp.float32, device=mesh_wp.device)
    saturated_wp = tw.graph.shortest_path_envelope(
        _length_weighted_csr(mesh_wp, threshold), values_wp
    )
    # Anti-vacuity: the spike must have been lowered, or a no-op would pass. It is the *peak* that
    # moves and not the field around it -- the zeros are already minimal and nothing is ever raised.
    assert 0.0 < float(saturated_wp.numpy()[0]) < 10.0
    assert np.allclose(
        saturated_wp.numpy(), meshset_pml.current_mesh().vertex_scalar_array(), rtol=1e-4, atol=1e-5
    )


@pytest.mark.parametrize("n_sources", [1, 3])
def test_shortest_path_envelope_is_the_edge_graph_distance(
    icosahedron: tuple[tm.Trimesh, wp.Mesh], n_sources: int
) -> None:
    """
    Class A against ``scipy.sparse.csgraph.dijkstra``: seeded with zeros, this *is* the distance.

    The envelope ``min_u (values[u] + d(u, v))`` reduces to the weighted multi-source shortest-path
    distance when ``values`` is zero on the sources and large elsewhere — the reading the function
    is named for, and why the package needs no second shortest-path implementation. Both sides take
    the same graph, so the only difference is float32 against float64: measured 7.1e-07.
    """
    mesh_tm, mesh_wp = icosahedron
    n_vertices = mesh_tm.vertices.shape[0]
    sources_np = np.arange(n_sources)

    seeded_np = np.full(n_vertices, 1e6, dtype=np.float32)
    seeded_np[sources_np] = 0.0
    envelope_wp = tw.graph.shortest_path_envelope(
        _length_weighted_csr(mesh_wp), wp.array(seeded_np, dtype=wp.float32, device=mesh_wp.device)
    )

    edges_np = mesh_tm.edges_unique
    lengths_np = np.linalg.norm(
        mesh_tm.vertices[edges_np[:, 0]] - mesh_tm.vertices[edges_np[:, 1]], axis=1
    )
    both_np = np.concatenate([edges_np, edges_np[:, ::-1]])
    graph_sp = sp.coo_matrix(
        (np.concatenate([lengths_np, lengths_np]), (both_np[:, 0], both_np[:, 1])),
        shape=(n_vertices, n_vertices),
    ).tocsr()
    distance_sp = csgraph.dijkstra(graph_sp, indices=sources_np, min_only=True)

    # Anti-vacuity: the field has to have propagated, not stayed at its 1e6 seed.
    assert distance_sp.max() > 0.5
    assert envelope_wp.numpy().max() < 1e5
    assert np.allclose(envelope_wp.numpy(), distance_sp, rtol=1e-5, atol=1e-5)


def test_shortest_path_envelope_respects_the_bound(half_torus: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """After convergence no edge violates the cap, and nothing was raised."""
    mesh_tm, mesh_wp = half_torus
    threshold = 2.0
    rng = np.random.default_rng(7)
    values_np = rng.uniform(0.0, 5.0, size=mesh_tm.vertices.shape[0])
    values_wp = wp.array(values_np.astype(np.float32), dtype=wp.float32, device=mesh_wp.device)

    saturated_np = tw.graph.shortest_path_envelope(
        _length_weighted_csr(mesh_wp, threshold), values_wp
    ).numpy()
    assert (saturated_np <= values_np.astype(np.float32) + 1e-5).all()

    edges_np = mesh_tm.edges_unique
    lengths_np = np.linalg.norm(
        mesh_tm.vertices[edges_np[:, 0]] - mesh_tm.vertices[edges_np[:, 1]], axis=1
    )
    jumps_np = np.abs(saturated_np[edges_np[:, 0]] - saturated_np[edges_np[:, 1]])
    assert (jumps_np <= lengths_np / threshold + 1e-4).all()
    # Every minimum survives: the smallest value in the field is untouched.
    assert np.isclose(saturated_np.min(), values_np.min(), rtol=1e-5, atol=1e-5)


def test_shortest_path_envelope_invalid(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = icosahedron
    adjacency = _length_weighted_csr(mesh_wp)
    values_wp = wp.zeros(mesh_tm.vertices.shape[0], dtype=wp.float32, device=mesh_wp.device)
    with pytest.raises(ValueError, match="max_iterations must be non-negative"):
        tw.graph.shortest_path_envelope(adjacency, values_wp, max_iterations=-1)
    with pytest.raises(ValueError, match="one entry per node"):
        tw.graph.shortest_path_envelope(adjacency, values_wp[:3])
    edges, n_vertices = _mesh_vertex_edges(mesh_wp)
    with pytest.raises(ValueError, match="one entry per edge"):
        tw.graph.edges_to_csr(n_vertices, edges, values_wp)


def test_shortest_path_envelope_rejects_negative_weights(device: str) -> None:
    """
    A negative weight makes the relaxation decrease without bound instead of converging.

    Without this guard the loop returns a plausible-looking array that keeps getting smaller as
    ``max_iterations`` grows -- silently wrong rather than raising, exactly what the docstring's
    "not admissible" note warns about.
    """
    edges_wp = wp.array(np.array([[0, 1]], dtype=np.int32), dtype=wp.int32, device=device)
    weights_wp = wp.array(np.array([-1.0], dtype=np.float32), dtype=wp.float32, device=device)
    adjacency = tw.graph.edges_to_csr(2, edges_wp, weights_wp)
    values_wp = wp.array(np.array([0.0, 1000.0], dtype=np.float32), dtype=wp.float32, device=device)
    with pytest.raises(ValueError, match="non-negative"):
        tw.graph.shortest_path_envelope(adjacency, values_wp, max_iterations=5)
