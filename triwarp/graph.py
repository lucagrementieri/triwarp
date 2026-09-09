"""Generic graph algorithms on sparse adjacency matrices and edge lists (mesh-agnostic)."""

from __future__ import annotations

import math

import warp as wp
import warp.sparse as wps

import triwarp as tw
import triwarp.typing as twt
from triwarp._device import read_scalar, require_same_device
from triwarp.array import arange
from triwarp.constants import INT32_MAX
from triwarp.kernels import graph as kernel_graph
from triwarp.kernels import scatter as kernel_scatter
from triwarp.kernels.algorithms import bfs as kernel_bfs
from triwarp.kernels.algorithms import connected_components as kernel_connected_components

# Frontier width at which the level-synchronous BFS hands the rest of the traversal to one serial
# thread. A level costs a fixed amount of launch overhead whatever its frontier, while the serial
# walk costs a small amount per node, so the two break even once the frontier is narrow.
#
# Frontier width, not node count, decides this: node count alone cannot separate a small graph
# from a large one whose frontier still narrows early (a long thin ribbon, average degree 4.0,
# from a sphere, average degree 6.0). Frontier width is observable and is what actually decides.
_BFS_ESCAPE_FRONTIER = 32


def edges_to_csr(
    node_count: int, edges: twt.Array2dInt32, weights: wp.array[wp.float32] | None = None
) -> wps.BsrMatrix[wp.float32]:
    """
    Undirected adjacency as a 1x1-block ``warp.sparse.BsrMatrix`` (CSR form).

    Each undirected edge ``(a, b)`` contributes directed entries ``(a, b)`` and ``(b, a)``.

    Parameters
    ----------
    node_count
        Number of vertices ``0 .. node_count - 1``.
    edges
        ``(m, 2)`` ``wp.int32`` edge rows on the target device.
    weights
        Length-``m`` edge weights, one per undirected edge, written into both of its directed
        entries. Defaults to unit weights, which is what the unweighted traversals want; the
        weighted relaxation in
        [`shortest_path_envelope`][triwarp.graph.shortest_path_envelope] measures paths in whatever
        this carries — mesh edge lengths from
        [`edges_unique_length`][triwarp.edges.edges_unique_length] for a geometric distance.

    Returns
    -------
    warp.sparse.BsrMatrix
        Square ``(node_count, node_count)`` adjacency. Duplicate directed pairs from repeated input
        edges are merged (values **summed**), so a repeated weighted edge doubles its weight —
        deduplicate with [`edges_unique`][triwarp.edges.edges_unique] first when that matters.

    Raises
    ------
    ValueError
        If ``weights`` is given and does not have one entry per edge row.
    RuntimeError
        If ``edges`` and ``weights`` are not all on one device.

    See Also
    --------
    [`bfs`][triwarp.graph.bfs]
    [`shortest_path_envelope`][triwarp.graph.shortest_path_envelope]
    """
    require_same_device(edges=edges, weights=weights)
    device = edges.device
    m = int(edges.shape[0])
    if weights is not None and int(weights.shape[0]) != m:
        raise ValueError(
            f"weights must have one entry per edge, got {weights.shape[0]} for {m} edges"
        )

    n_entries = 2 * m
    rows = wp.empty(n_entries, dtype=wp.int32, device=device)
    cols = wp.empty(n_entries, dtype=wp.int32, device=device)
    if m > 0:
        wp.launch(kernel_graph.edges_to_adjacency, dim=m, inputs=[edges, rows, cols], device=device)
    if weights is None:
        data = wp.ones(n_entries, dtype=wp.float32, device=device)
    else:
        data = wp.empty(n_entries, dtype=wp.float32, device=device)
        if m > 0:
            wp.launch(
                kernel_graph.duplicate_edge_weights, dim=m, inputs=[weights, data], device=device
            )
    return wps.bsr_from_triplets(
        node_count, node_count, rows, cols, data, prune_numerical_zeros=False
    )


def connected_component_labels(adjacency: wps.BsrMatrix[wp.Scalar]) -> wp.array[wp.int32]:
    """
    Per-node connected-component labels from a sparse adjacency matrix.

    Uses ECL-CC (init, single-pass CAS hooking with in-kernel retry, intermediate
    pointer jumping) on the CSR structure of ``adjacency`` (1x1 BSR blocks): one
    init launch, one hook launch, one flatten launch — no host-side convergence
    loop. Each label is the smallest node id in its component (hooks always point
    the larger root at the smaller one); values are not necessarily contiguous in
    ``0 .. k-1`` (compare partitions, not raw ids).

    Parameters
    ----------
    adjacency
        Square undirected adjacency in 1x1-block ``warp.sparse.BsrMatrix`` form.
        Each nonzero ``(i, j)`` denotes an edge between nodes ``i`` and ``j``; for
        undirected graphs both ``(i, j)`` and ``(j, i)`` should be present.

    Returns
    -------
    wp.array[wp.int32]
        Length ``adjacency.nrow`` on ``adjacency.device``. Isolated nodes (empty rows)
        receive distinct labels. When ``nnz == 0``, ``labels[i] == i``.

    Raises
    ------
    ValueError
        If ``adjacency`` is not square or does not use 1x1 blocks.

    See Also
    --------
    [`connected_component_labels_from_edges`][triwarp.graph.connected_component_labels_from_edges]
    [`edges_to_csr`][triwarp.graph.edges_to_csr]
    [`face_connected_component_labels`][triwarp.adjacency.face_connected_component_labels]
    """
    node_count, offsets, indices = _validate_square_csr(adjacency)

    device = adjacency.device
    if node_count <= 1:
        return wp.zeros(node_count, dtype=wp.int32, device=device)
    if adjacency.nnz == 0:
        return arange(node_count, device=device)

    labels = wp.empty(node_count, dtype=wp.int32, device=device)
    parents = wp.empty(node_count, dtype=wp.int32, device=device)

    wp.launch(
        kernel_connected_components.ecl_init_parent,
        dim=node_count,
        inputs=[offsets, indices, parents],
        device=device,
    )
    wp.launch(
        kernel_connected_components.ecl_hook,
        dim=node_count,
        inputs=[offsets, indices, parents],
        device=device,
    )
    wp.launch(
        kernel_connected_components.ecl_flatten,
        dim=node_count,
        inputs=[parents, labels],
        device=device,
    )
    return labels


def connected_component_labels_from_edges(
    edges: twt.Array2dInt32, node_count: int | None = None, *, validate: bool = True
) -> wp.array[wp.int32]:
    """
    Per-node connected-component labels from an undirected edge list.

    Builds a CSR adjacency via [`edges_to_csr`][triwarp.graph.edges_to_csr] and delegates to
    [`connected_component_labels`][triwarp.graph.connected_component_labels].

    Parameters
    ----------
    edges
        ``(m, 2)`` ``wp.int32`` edge list. Each row ``(a, b)`` connects nodes ``a``
        and ``b`` (undirected; order does not matter).
    node_count
        Number of nodes ``0 .. node_count - 1``. When ``None``, inferred as
        ``max(edges) + 1`` if ``m > 0``, else ``0``.
    validate
        When ``False``, skip the range check on ``edges`` and its host readback. Requires
        ``node_count``; see the warning below. Follows the same convention as
        [`group_int_rows`][triwarp.grouping.group_int_rows].

    Returns
    -------
    wp.array[wp.int32]
        Length ``node_count`` on ``edges.device``.

    Raises
    ------
    TypeError
        If ``edges`` is not a rank-2 ``int32`` array.
    ValueError
        If ``edges`` is not ``(m, 2)``, an endpoint is outside ``[0, node_count)``,
        or ``node_count`` is negative.

    Warning
    -------
    !!! warning "``validate=False`` trades a guard for a synchronization"
        The range check copies the whole ``(m, 2)`` edge buffer to the host, so it costs a device
        synchronization on a path that otherwise has none. Pass ``validate=False`` **only** when
        the caller produced ``edges`` itself and knows the bound holds — an adjacency list from
        [`face_adjacency`][triwarp.adjacency.face_adjacency], say. With an out-of-range index the
        unchecked path reads **and writes** out of bounds rather than raising: the scatter kernels
        downstream index a ``node_count``-element buffer by the raw endpoint. On a CUDA device that
        lands in device memory; on the **CPU** device a Warp array is host heap, so it overwrites
        glibc's allocator metadata and aborts the process later, somewhere unrelated.

    See Also
    --------
    [`connected_component_labels`][triwarp.graph.connected_component_labels]
    [`connected_component_parity_from_edges`][triwarp.graph.connected_component_parity_from_edges]
    [`face_connected_component_labels`][triwarp.adjacency.face_connected_component_labels]
    [`trimesh.graph.connected_component_labels`][]
    """
    node_count = _validate_edge_list(edges, node_count, validate=validate)

    # With no edges every node is its own component, which ``arange`` gives directly -- the
    # CSR build and traversal below would reach the same answer the long way.
    if int(edges.shape[0]) == 0:
        return arange(node_count, device=edges.device)

    adjacency = edges_to_csr(node_count, edges)
    return connected_component_labels(adjacency)


def connected_component_parity_from_edges(
    edges: twt.Array2dInt32, signs: wp.array[wp.int32], node_count: int, *, validate: bool = True
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Component labels plus a Z2 potential satisfying the per-edge parity constraints.

    Each edge ``(a, b)`` carries a sign in ``{0, 1}`` demanding ``parity[a] ^ parity[b] == sign``.
    On a component where those constraints are consistent this determines ``parity`` uniquely once
    the component representative is pinned to ``0``, and this function returns exactly that — the
    discrete analog of a potential function, and the reason it can replace an iterative flood fill.

    Extends ECL-CC by packing ``(parent, parity-to-parent)`` into a single ``int32`` word, so the
    union-find still hooks with one ``wp.atomic_cas``: **three launches and no host
    synchronization** whatever the graph's diameter, where propagating the bits edge by edge costs
    one launch per graph level.

    Parameters
    ----------
    edges
        ``(m, 2)`` ``wp.int32`` undirected edge list; each row ``(a, b)`` constrains ``a`` and
        ``b``. Endpoints must lie in ``[0, node_count)``. Self-loops are ignored.
    signs
        Length-``m`` ``wp.int32`` parity constraint per edge, ``0`` (equal) or ``1`` (opposite).
    node_count
        Number of nodes ``0 .. node_count - 1``.
    validate
        When ``False``, skip the range check on ``edges`` and the device synchronization it
        costs. Follows the same convention as
        [`connected_component_labels_from_edges`][triwarp.graph.connected_component_labels_from_edges]
        — see the warning there.

    Returns
    -------
    labels : wp.array[wp.int32]
        Length ``node_count``; the smallest node id in each component, as in
        [`connected_component_labels`][triwarp.graph.connected_component_labels]. Isolated nodes
        label themselves.
    parity : wp.array[wp.int32]
        Length ``node_count`` of ``0`` / ``1`` bits, ``0`` at every component representative.

    Raises
    ------
    TypeError
        If ``edges`` is not a rank-2 ``int32`` array.
    ValueError
        If ``edges`` is not ``(m, 2)``, ``signs`` is not length ``m``, ``node_count`` is
        negative, or (with ``validate``) an endpoint is out of range.
    RuntimeError
        If ``edges`` and ``signs`` are not all on one device.

    Warning
    -------
    !!! warning "``validate=False`` trades a guard for a synchronization"
        With an out-of-range endpoint the unchecked path reads **and writes** out of bounds
        rather than raising: ``ecl_hook_parity`` indexes a ``node_count``-element buffer by the
        raw endpoint. On a CUDA device that lands in device memory; on the **CPU** device a Warp
        array is host heap, so it overwrites glibc's allocator metadata and aborts the process
        later, somewhere unrelated. Pass ``validate=False`` only when the caller produced
        ``edges`` itself and knows the bound holds.

    Notes
    -----
    A component whose constraints are **contradictory** (an odd-signed cycle — a Möbius band, in
    the orientation application) admits no potential at all. No error is raised: the constraints
    along whichever spanning tree the union-find happened to build are satisfied and the remaining
    edges are left violated, so a caller that needs to know must re-test the edges against the
    returned ``parity``. This is the same best-effort contract a flood fill gives.

    See Also
    --------
    [`connected_component_labels_from_edges`][triwarp.graph.connected_component_labels_from_edges]
    [`face_orientation_bits`][triwarp.validation.face_orientation_bits]
    """
    require_same_device(edges=edges, signs=signs)
    twt.ensure_edge_pairs(edges, "edges")
    m = int(edges.shape[0])
    if int(signs.shape[0]) != m:
        raise ValueError(f"signs must have length {m} to match edges, got {int(signs.shape[0])}")
    if node_count < 0:
        raise ValueError(f"node_count must be non-negative, got {node_count}")
    if validate and m > 0:
        # One 8-byte read of both bounds, not a copy of the whole edge buffer -- see
        # ``_validate_edge_list``, whose range check this mirrors for a signed edge list.
        lowest, highest = tw.reduce.minmax(edges)
        if lowest < 0 or highest >= node_count:
            raise ValueError(
                f"edge indices must lie in [0, {node_count}), got min={lowest} max={highest}"
            )

    device = edges.device
    labels = wp.empty(node_count, dtype=wp.int32, device=device)
    parity = wp.zeros(node_count, dtype=wp.int32, device=device)
    if node_count == 0:
        return labels, parity

    words = wp.empty(node_count, dtype=wp.int32, device=device)
    wp.launch(
        kernel_connected_components.ecl_init_parent_parity,
        dim=node_count,
        inputs=[words],
        device=device,
    )
    if m > 0:
        wp.launch(
            kernel_connected_components.ecl_hook_parity,
            dim=m,
            inputs=[edges, signs, words],
            device=device,
        )
    wp.launch(
        kernel_connected_components.ecl_flatten_parity,
        dim=node_count,
        inputs=[words, labels, parity],
        device=device,
    )
    return labels, parity


def successor_cycles(
    edges: twt.Array2dInt32, node_count: int, *, validate: bool = True
) -> tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Every cycle of a successor graph in traversal order, packed flat plus per-cycle offsets.

    The edges of a **successor graph** — one in which each node has at most one outgoing edge —
    decompose into node-disjoint cycles and chains. This orders every node along its cycle,
    following the edge direction from the cycle's smallest node index, with no per-cycle Python
    and no per-cycle allocation. The ranking is pointer-jumping (Wyllie's list ranking), so the
    work is ``O(k log L)`` over ``k`` cycle nodes with longest cycle ``L`` rather than the
    quadratic per-node successor walk. A mesh boundary's oriented edges are the motivating input
    (see [`boundary_loops_batched`][triwarp.boundary.boundary_loops_batched]).

    Parameters
    ----------
    edges
        ``(m, 2)`` ``wp.int32`` directed edges; row ``(a, b)`` makes ``b`` the successor of
        ``a``. At most one out-edge per node. Nodes appearing in no edge belong to no cycle and
        do not appear in the result.
    node_count
        Number of nodes ``0 .. node_count - 1``.
    validate
        When ``False``, skip the range check on ``edges`` and the device synchronization it
        costs; forwarded to [`connected_component_labels_from_edges`]
        [triwarp.graph.connected_component_labels_from_edges] — see the warning there.

    Returns
    -------
    flat_cycles : wp.array[wp.int32]
        Concatenated ordered node indices of every cycle, on ``edges.device``. Each cycle starts
        at its smallest node index and follows the edge direction.
    offsets : wp.array[wp.int32]
        Length-``n_cycles`` exclusive prefix sum of the cycle sizes: cycle ``i`` occupies
        ``flat_cycles[offsets[i] : offsets[i] + cycle_sizes[i]]``. Not a total-terminated CSR
        array — the last cycle ends at ``flat_cycles.shape[0]``.
    cycle_sizes : wp.array[wp.int32]
        Length-``n_cycles`` node count per cycle.

    Raises
    ------
    TypeError
        If ``edges`` is not a rank-2 ``int32`` array.
    ValueError
        If ``edges`` is not ``(m, 2)``, ``node_count`` is negative, or (with ``validate``) an
        endpoint is out of range.

    Notes
    -----
    On malformed input — a node with several in-edges, so two chains merge — the cycle ranks can
    collide. Colliding nodes overwrite one slot and leave another at ``0``, which is a valid node
    index, so the result stays in-range rather than returning uninitialized garbage; it is the
    caller's job to pass a true successor graph if exact cycles are required.

    See Also
    --------
    [`connected_component_labels_from_edges`][triwarp.graph.connected_component_labels_from_edges]
    [`boundary_loops_batched`][triwarp.boundary.boundary_loops_batched]
    """
    # The range check must run BEFORE ``scatter_successor`` below, not be deferred to the
    # ``connected_component_labels_from_edges`` call: that kernel indexes ``next_node`` by the raw
    # edge endpoints, so an out-of-range endpoint writes past a ``node_count``-element buffer. On
    # the CPU device that is a host-heap overwrite, silent at the point of the write and surfacing
    # later as a glibc abort somewhere unrelated.
    node_count = _validate_edge_list(edges, node_count, validate=validate)

    device = edges.device
    m = int(edges.shape[0])
    if m == 0 or node_count == 0:
        # Three *distinct* empty allocations, so callers may write into them independently.
        return tuple(wp.empty(0, dtype=wp.int32, device=device) for _ in range(3))

    next_node = wp.full(node_count, -1, dtype=wp.int32, device=device)
    wp.launch(kernel_graph.scatter_successor, dim=m, inputs=[edges, next_node], device=device)

    # Already range-checked above, so the downstream call skips the second reduction and host sync.
    labels = connected_component_labels_from_edges(edges, node_count=node_count, validate=False)

    cycle_nodes = tw.grouping.unique_1d(edges.flatten())
    n_nodes = int(cycle_nodes.shape[0])

    label_min = wp.full(node_count, node_count, dtype=wp.int32, device=device)
    label_count = wp.zeros(node_count, dtype=wp.int32, device=device)
    is_chain = wp.zeros(node_count, dtype=wp.int32, device=device)
    wp.launch(
        kernel_graph.scatter_cycle_min_and_count,
        dim=n_nodes,
        inputs=[cycle_nodes, next_node, labels, label_min, label_count, is_chain],
        device=device,
    )

    # A successor graph decomposes into node-disjoint cycles *and* chains (see the docstring). A
    # chain component contains a node with no outgoing edge (``next_node[v] < 0``), which
    # ``init_rank_arrays`` below treats as a fixed point identical to a genuine cycle's cut at
    # ``label_min`` -- two fixed points in one component collide onto rank slot 0 and fabricate a
    # bogus "cycle" out of whatever the collision leaves there, including node ids that never
    # appeared in the input. Excluding a chain's nodes here keeps that collision scoped to the
    # malformed input the Notes above already describe (an in-degree collision), rather than
    # firing on an ordinary chain.
    keep_mask = wp.empty(n_nodes, dtype=wp.int32, device=device)
    wp.launch(
        kernel_graph.chain_node_mask,
        dim=n_nodes,
        inputs=[cycle_nodes, labels, is_chain, keep_mask],
        device=device,
    )
    cycle_nodes = tw.array.gather(cycle_nodes, tw.array.flatnonzero(keep_mask))
    n_nodes = int(cycle_nodes.shape[0])
    if n_nodes == 0:
        return tuple(wp.empty(0, dtype=wp.int32, device=device) for _ in range(3))

    # Pointer-jumping list ranking (Wyllie): O(log L) rounds of pointer doubling replace the
    # per-node successor walk, whose total work was quadratic in the cycle length.
    successor = wp.empty(node_count, dtype=wp.int32, device=device)
    steps = wp.empty(node_count, dtype=wp.int32, device=device)
    successor_next = wp.empty(node_count, dtype=wp.int32, device=device)
    steps_next = wp.empty(node_count, dtype=wp.int32, device=device)
    wp.launch(
        kernel_graph.init_rank_arrays,
        dim=n_nodes,
        inputs=[cycle_nodes, next_node, labels, label_min, successor, steps],
        device=device,
    )
    rounds = max(1, math.ceil(math.log2(max(n_nodes, 2))))
    for _ in range(rounds):
        wp.launch(
            kernel_graph.jump_rank,
            dim=n_nodes,
            inputs=[cycle_nodes, successor, steps, successor_next, steps_next],
            device=device,
        )
        successor, successor_next = successor_next, successor
        steps, steps_next = steps_next, steps

    position = wp.empty(n_nodes, dtype=wp.int32, device=device)
    wp.launch(
        kernel_graph.finalize_rank_positions,
        dim=n_nodes,
        inputs=[cycle_nodes, labels, label_count, steps, position],
        device=device,
    )

    node_labels = tw.array.gather(labels, cycle_nodes)
    unique_labels, cycle_index = tw.grouping.unique_1d(node_labels, return_inverse=True)
    n_cycles = int(unique_labels.shape[0])

    cycle_sizes = tw.array.gather(label_count, unique_labels)
    offsets = wp.empty(n_cycles, dtype=wp.int32, device=device)
    wp.utils.array_scan(cycle_sizes, out_array=offsets, inclusive=False)

    # Zero-initialised (not wp.empty): colliding ranks on malformed input (see Notes) can leave
    # slots unwritten by scatter_cycle_slot, and zero is a valid node index.
    flat_cycles = wp.zeros(n_nodes, dtype=wp.int32, device=device)
    wp.launch(
        kernel_graph.scatter_cycle_slot,
        dim=n_nodes,
        inputs=[cycle_nodes, cycle_index, position, offsets, flat_cycles],
        device=device,
    )

    return flat_cycles, offsets, cycle_sizes


def bfs(
    adjacency: wps.BsrMatrix[wp.Scalar], source: int
) -> tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Single-source breadth-first search over a sparse CSR adjacency matrix.

    The discovery order, parent tree, and distances match
    [`scipy.sparse.csgraph.breadth_first_order`][] exactly when the adjacency columns are sorted
    ascending per row (as produced by [`edges_to_csr`][triwarp.graph.edges_to_csr]). This mirrors
    ``igl::bfs``, additionally returning the BFS level of each node.

    Two engines, chosen by the *observed frontier width* rather than by any property of the graph
    known up front. The traversal starts level-synchronous and parallel, and hands over to a single
    serial thread as soon as its frontier is both narrow and no longer growing: a level costs the
    same four fixed-size launches whatever it carries, so once the frontier is a handful of nodes
    the serial walk is cheaper per node than the launches are per level. On a graph that stays wide
    the handover never fires; on one whose frontier is narrow from the start (a path) it fires
    almost immediately. Order-exactness survives it by construction — the parallel path builds the
    same explicit FIFO the serial one drains, so the serial engine just continues from where the
    queue got to.

    Parameters
    ----------
    adjacency
        Square undirected adjacency in 1x1-block ``warp.sparse.BsrMatrix`` form. Each nonzero
        ``(i, j)`` denotes an edge between nodes ``i`` and ``j``; for undirected graphs both
        ``(i, j)`` and ``(j, i)`` should be present (as from
        [`edges_to_csr`][triwarp.graph.edges_to_csr]).
    source
        Start node, in ``[0, node_count)``.

    Returns
    -------
    order
        ``wp.array[wp.int32]`` of the reachable nodes in BFS discovery order; length equals the
        number of nodes reachable from ``source`` (matches scipy's ``node_array``).
    parents
        Length ``node_count`` on ``adjacency.device``. ``parents[i]`` is the predecessor of ``i``
        in the BFS tree; ``-1`` for ``source`` and for unreachable nodes (scipy uses ``-9999``).
    distances
        Length ``node_count``. ``distances[i]`` is the BFS level (hop count) of ``i`` from
        ``source``; ``-1`` for unreachable nodes.

    Raises
    ------
    ValueError
        If ``adjacency`` is not square, does not use 1x1 blocks, or ``source`` is out of range.

    See Also
    --------
    [`bfs_from_edges`][triwarp.graph.bfs_from_edges]
    [`bfs_multi_source`][triwarp.graph.bfs_multi_source]
    [`connected_component_labels`][triwarp.graph.connected_component_labels]
    [`scipy.sparse.csgraph.breadth_first_order`][]

    Notes
    -----
    !!! note "A long, narrow graph (e.g. a thin ribbon) hits a ceiling here"

        Once the frontier narrows, the traversal hands the rest of the walk to a single serial
        thread, so a graph that stays narrow for most of its diameter costs close to one thread's
        full pointer-chasing walk over it. This is a genuine memory-throughput limit rather than a
        tuning gap: the per-node cost is dominated by dependent loads with no independent work left
        to hide behind them, and a cooperative block-synchronized rewrite is *slower* here because
        a narrow frontier's per-level barriers cost more than the work they protect. A host (CPU)
        fallback is not used either, since it would mean copying the whole CSR structure and the
        result back across the bus, a different contract from the one this function has. So on a
        very long, narrow graph this function will not beat
        ``scipy.sparse.csgraph.breadth_first_order``, whose single-threaded walk has no such
        transfer cost.

        On a moderately wide frontier the parallel engine's per-level cost is dominated by its
        fixed launch overhead rather than by graph size, so narrowing the grid is not a useful
        lever there; the remaining launches per level are already at the minimum the algorithm's
        synchronization points allow.
    """
    node_count, offsets, columns = _validate_square_csr(adjacency)
    if source < 0 or source >= node_count:
        raise ValueError(f"source must be in [0, {node_count}), got {source}")

    device = adjacency.device

    parents = wp.full(node_count, -1, dtype=wp.int32, device=device)
    distances = wp.full(node_count, -1, dtype=wp.int32, device=device)
    order_buffer = wp.empty(node_count, dtype=wp.int32, device=device)

    reached = wp.zeros(1, dtype=wp.int32, device=device)
    if device.is_cpu:
        # The Warp CPU backend executes kernels serially anyway, so the frontier loop buys nothing
        # there at any size and the single-thread traversal is trivially order-exact.
        wp.launch(
            kernel_bfs.single_source_bfs_kernel,
            dim=1,
            inputs=[wp.int32(source), offsets, columns, order_buffer, parents, distances, reached],
            device=device,
        )
        return wp.clone(order_buffer[: int(read_scalar(reached, 0))]), parents, distances

    # Level-synchronous frontier BFS that reproduces scipy's FIFO discovery order exactly,
    # sort-free: unvisited neighbors are claimed with the parent dequeue rank via atomic_min
    # (first-dequeued parent wins, matching scipy's predecessor), per-rank owned counts are
    # scanned into rank-major segment offsets, and each rank scatters its owned nodes in
    # ascending CSR column order — (rank, ascending node id), the same order the former int64
    # claim-key radix sort produced. The whole loop runs on device via ``wp.capture_while``
    # (kernels launch at fixed dim=node_count and early-exit on the device-side frontier size),
    # so the only host sync is the final window readback; when conditional CUDA graphs are
    # unavailable, ``capture_while`` itself falls back to direct execution with one pinned
    # 4-byte condition readback per level.
    #
    # The loop also *stops early* once the frontier narrows (``_BFS_ESCAPE_FRONTIER``) and hands
    # its half-built FIFO to the serial kernel above, which is what keeps a long-diameter graph
    # from paying four fixed-size launches for a two-node frontier, tens of thousands of times.
    # ``int(...)`` because ``BFS_SCAN_BLOCK`` is a typed ``wp.int32`` constant, and ``//`` on a
    # ``wp.int32`` at host scope raises ``TypeError`` rather than dividing.
    scan_block = int(kernel_bfs.BFS_SCAN_BLOCK)
    n_blocks = (node_count + scan_block - 1) // scan_block
    padded = n_blocks * scan_block
    claim_rank = wp.full(node_count, INT32_MAX, dtype=wp.int32, device=device)
    offsets_scan = wp.empty(padded, dtype=wp.int32, device=device)
    block_sums = wp.empty(n_blocks, dtype=wp.int32, device=device)
    # state = [frontier start, frontier end, level to emit, loop condition,
    #          emit start, emit end, emit level] -- the last three are the window
    # ``bfs_scatter_claims`` works on, snapshotted by ``bfs_scan_and_advance`` before it advances
    # the live one. See ``kernels/algorithms/bfs.py`` for why the update runs before the scatter.
    state = wp.zeros(kernel_bfs.BFS_STATE_SIZE, dtype=wp.int32, device=device)
    state.assign([0, 1, 1, 1, 0, 1, 1])
    wp.launch(
        kernel_bfs.bfs_seed,
        dim=1,
        inputs=[wp.int32(source), order_buffer, distances],
        device=device,
    )

    def bfs_level_body() -> None:
        wp.launch(
            kernel_bfs.bfs_expand_claim,
            dim=node_count,
            inputs=[offsets, columns, order_buffer, state, distances, claim_rank],
            device=device,
        )
        # Count and scan share a launch (see the kernel), and the scan is a capture-safe
        # fixed-buffer one: wp.utils.array_scan allocates temp storage internally, which a
        # conditional graph body rejects ("Conditional body graph contains an unsupported
        # operation (memory allocation)"), so ``bfs_count_and_scan`` avoids it with a fixed buffer.
        wp.launch_tiled(
            kernel_bfs.bfs_count_and_scan,
            dim=[n_blocks],
            inputs=[
                offsets,
                columns,
                order_buffer,
                state,
                distances,
                claim_rank,
                offsets_scan,
                block_sums,
            ],
            block_dim=scan_block,
            device=device,
        )
        wp.launch(
            kernel_bfs.bfs_scan_and_advance,
            dim=1,
            inputs=[offsets_scan, wp.int32(_BFS_ESCAPE_FRONTIER), block_sums, state],
            device=device,
        )
        wp.launch(
            kernel_bfs.bfs_scatter_claims,
            dim=node_count,
            inputs=[
                offsets,
                columns,
                state,
                claim_rank,
                offsets_scan,
                block_sums,
                order_buffer,
                parents,
                distances,
            ],
            device=device,
        )

    # CUDA only: ``bfs_count_and_scan`` builds its tile with ``wp.tile``, which fills lane 0 alone
    # on the CPU backend. On CPU the level loop is skipped entirely and the serial kernel below
    # walks from the seed — the same kernel the escape path already hands off to, so the answer is
    # identical rather than degraded, and one CPU core doing the walk is the faster engine there
    # anyway.
    if device.is_cuda:
        condition = state[3:4]
        if wp.is_conditional_graph_supported():
            with wp.ScopedCapture(device) as capture:
                wp.capture_while(condition, bfs_level_body)
            wp.capture_launch(capture.graph)
        else:
            wp.capture_while(condition, bfs_level_body)

    # One readback of the FIFO window tells both things there are to know: an empty window means
    # the traversal ran out of frontier, a non-empty one means it escaped and the serial kernel
    # takes over from exactly there.
    window = state[:2].numpy()
    head, tail = int(window[0]), int(window[1])
    if head < tail:
        wp.launch(
            kernel_bfs.resume_bfs_kernel,
            dim=1,
            inputs=[state, offsets, columns, order_buffer, parents, distances, reached],
            device=device,
        )
        tail = int(read_scalar(reached, 0))
    return wp.clone(order_buffer[:tail]), parents, distances


def bfs_from_edges(
    edges: twt.Array2dInt32, source: int, node_count: int | None = None
) -> tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Single-source BFS from an undirected edge list.

    Builds a CSR adjacency via [`edges_to_csr`][triwarp.graph.edges_to_csr] and delegates
    to [`bfs`][triwarp.graph.bfs].

    Parameters
    ----------
    edges
        ``(m, 2)`` ``wp.int32`` edge list. Each row ``(a, b)`` connects nodes ``a`` and ``b``
        (undirected; order does not matter).
    source
        Start node, in ``[0, node_count)``.
    node_count
        Number of nodes ``0 .. node_count - 1``. When ``None``, inferred as ``max(edges) + 1`` if
        ``m > 0``, else ``0``.

    Returns
    -------
    tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32]]
        ``(order, parents, distances)`` as in [`bfs`][triwarp.graph.bfs], on ``edges.device``.

    Raises
    ------
    TypeError
        If ``edges`` is not a rank-2 ``int32`` array.
    ValueError
        If ``edges`` is not ``(m, 2)``, an endpoint is outside ``[0, node_count)``, ``node_count``
        is negative, or ``source`` is out of range.

    See Also
    --------
    [`bfs`][triwarp.graph.bfs]
    [`connected_component_labels_from_edges`][triwarp.graph.connected_component_labels_from_edges]
    """
    node_count = _validate_edge_list(edges, node_count, validate=True)

    adjacency = edges_to_csr(node_count, edges)
    return bfs(adjacency, source)


def bfs_multi_source(
    adjacency: wps.BsrMatrix[wp.Scalar], sources: wp.array[wp.int32]
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Reachable sets for many sources, packed as a CSR buffer.

    The reachable set of an unbounded traversal is the source's connected component, so this
    labels components once
    ([`connected_component_labels`][triwarp.graph.connected_component_labels])
    and emits each source's component from one label-sorted node array. Source ``sources[k]``
    owns ``neighbors[offsets[k] : offsets[k + 1]]``, listed with the source itself first and
    the remaining nodes in ascending index order. There is no reachable-set capacity limit.
    For BFS discovery order, parents, and distances of a single source, use
    [`bfs`][triwarp.graph.bfs].

    Parameters
    ----------
    adjacency
        Square undirected adjacency in 1x1-block ``warp.sparse.BsrMatrix`` form.
    sources
        Length-``k`` ``wp.int32`` start nodes, each in ``[0, node_count)``.

    Returns
    -------
    neighbors
        Flat ``wp.array[wp.int32]`` of reachable nodes for all sources, concatenated in source
        order (CSR column buffer).
    offsets
        Length-``k`` exclusive prefix sum of per-source counts (CSR starts); source ``k`` owns
        ``neighbors[offsets[k] : offsets[k + 1]]`` with ``offsets[k_total]`` implied as the total.

    Raises
    ------
    ValueError
        If ``adjacency`` is not square, does not use 1x1 blocks, or a source is out of range.
    RuntimeError
        If ``adjacency`` and ``sources`` are not on the same device.

    See Also
    --------
    [`bfs`][triwarp.graph.bfs]
    [`geodesic_ball`][triwarp.neighbors.geodesic_ball]
    """
    require_same_device(adjacency=adjacency, sources=sources)
    # This one traverses through ``bfs`` rather than the CSR buffers directly, so it wants only
    # the validation and the node count.
    node_count, _, _ = _validate_square_csr(adjacency)

    device = adjacency.device
    k = int(sources.shape[0])
    if k == 0:
        empty = wp.empty(0, dtype=wp.int32, device=device)
        return empty, wp.empty(0, dtype=wp.int32, device=device)

    # Unlike the two edge-buffer range checks in this module, this one stays on the host:
    # ``sources`` is ``k`` seeds, not a mesh-sized buffer, so a plain host-side check is cheaper
    # than a device reduction launch at any realistic ``k``.
    sources_np = sources.numpy()
    if sources_np.min() < 0 or int(sources_np.max()) >= node_count:
        raise ValueError(
            f"source indices must lie in [0, {node_count}), "
            f"got min={sources_np.min()} max={sources_np.max()}"
        )

    # The reachable set of an unbounded BFS is exactly the source's connected component, so a
    # single component labeling plus one stable key sort replaces the per-source traversals —
    # with no per-thread scratch and no reachable-set capacity cap.
    labels = connected_component_labels(adjacency)
    n = int(node_count)
    # This is ``sort_and_argsort``'s body, not a call to it: that helper takes an already-built
    # length-``n`` key array and copies it into its own ``2n`` scratch, where the keys here can be
    # written directly into the scratch's first half by the packing kernel below, at n's cost in
    # ``wp.int64`` one ``wp.copy`` cheaper.
    keys_buffer = wp.empty(2 * n, dtype=wp.int64, device=device)
    node_ids = tw.array.sort_pair_indices(n, -1, device)
    wp.launch(
        kernel_graph.pack_label_node_keys,
        dim=n,
        inputs=[labels, wp.int64(n), keys_buffer],
        device=device,
    )
    wp.utils.radix_sort_pairs(keys_buffer, node_ids, count=n)
    sorted_keys = wp.clone(keys_buffer[:n])
    sorted_nodes = wp.clone(node_ids[:n])
    node_rank = wp.empty(n, dtype=wp.int32, device=device)
    wp.launch(kernel_scatter.scatter_index, dim=n, inputs=[sorted_nodes, node_rank], device=device)

    segment_start = wp.empty(k, dtype=wp.int32, device=device)
    counts = wp.empty(k, dtype=wp.int32, device=device)
    wp.launch(
        kernel_graph.component_segment_bounds,
        dim=k,
        inputs=[sources, labels, sorted_keys, wp.int64(n), segment_start, counts],
        device=device,
    )
    # Host readback: only the device knows the total, and it sizes the neighbour buffer. One scan
    # and one 4-byte read give both -- reconstructing it as ``offsets[k - 1] + counts[k - 1]`` cost
    # two separate readbacks, so two full device synchronizations, for the same number.
    offsets, total = tw.array.counts_to_offsets(counts)

    neighbors = wp.empty(total, dtype=wp.int32, device=device)
    wp.launch(
        kernel_graph.emit_component_neighbors,
        dim=total,
        inputs=[sources, sorted_nodes, node_rank, segment_start, offsets, neighbors],
        device=device,
    )
    return neighbors, offsets


def shortest_path_envelope(
    adjacency: wps.BsrMatrix[wp.float32], values: wp.array[wp.float32], max_iterations: int = 0
) -> wp.array[wp.float32]:
    """
    Lower every node's value onto the shortest-path envelope ``min_u (values[u] + d(u, v))``.

    Two readings of one relaxation, and both are worth knowing because they are the same call:

    * **A shortest-path distance — Dijkstra's answer.** Seed ``values`` with ``0`` on the source
      nodes and a number larger than any reachable distance elsewhere, and the result is the
      weighted multi-source shortest-path distance to the nearest source, ``d`` being the sum of
      ``adjacency``'s values along the path, agreeing with
      [`scipy.sparse.csgraph.dijkstra`][] up to float32 precision.
    * **A Lipschitz cap.** Applied to an arbitrary field it enforces
      ``values[i] <= values[j] + w(i, j)`` on every edge by lowering values only, so every local
      minimum of the input survives untouched and only peaks that rise too steeply out of them are
      shaved down. That is MeshLab's ``apply_scalar_saturation_per_vertex`` (VCG
      ``UpdateQuality::VertexSaturate``) and the standard way to make a raw scalar usable as a
      **sizing field**: an adaptive remesher fed an ungraded target-length field produces a band of
      bad triangles where the field jumps, and this is the projection that removes the jump while
      respecting the field's small values.

    MeshLab's ``gradientthr`` is not a parameter here because it is a property of the *graph*: its
    cap is ``|p_i - p_j| / gradientthr``, so dividing the edge lengths by it when building
    ``adjacency`` reproduces it exactly, and the same weights then serve any other slope.

    Parameters
    ----------
    adjacency
        Square undirected adjacency in 1x1-block ``warp.sparse.BsrMatrix`` form whose **values are
        the edge weights**, as [`edges_to_csr`][triwarp.graph.edges_to_csr] builds with its
        ``weights`` argument. Negative weights are not admissible: the iteration would not
        terminate at the envelope.
    values
        Length-``node_count`` ``wp.float32`` initial labels. Not modified.
    max_iterations
        Cap on relaxation passes. Each pass propagates one edge further, so the number needed is the
        graph diameter of the region that violates the bound. ``0`` (the default) means
        ``node_count``, which can never be exceeded.

    Returns
    -------
    wp.array[wp.float32]
        Length-``node_count`` envelope on ``values.device``.

    Raises
    ------
    ValueError
        If ``adjacency`` is not square with 1x1 blocks, if ``max_iterations`` is negative, if
        ``values`` is not length ``node_count``, or if any weight in ``adjacency`` is negative.
    RuntimeError
        If ``adjacency`` and ``values`` are not on the same device.

    Examples
    --------
    Geodesic-ish distance from vertex 0 along the mesh's edges — the mesh recipe for both readings,
    since the weights are what make the envelope geometric:

    ```python
    n_vertices = int(v.shape[0])
    edges = tw.edges.edges_unique(f, n_vertices=n_vertices)[0]
    lengths = tw.edges.edges_unique_length(v, f, edges)
    adjacency = tw.graph.edges_to_csr(n_vertices, edges, lengths)
    seed = wp.full(n_vertices, 1.0e6, dtype=wp.float32, device=v.device)
    wp.copy(seed[:1], wp.zeros(1, dtype=wp.float32, device=v.device))
    print(float(tw.reduce.max(tw.graph.shortest_path_envelope(adjacency, seed))))
    ```

    Notes
    -----
    **The answer is Dijkstra's; the method is Bellman-Ford.** A priority queue is inherently
    serial — it processes one node per pop — so this relaxes *every* node against its neighbours in
    parallel and repeats until nothing improves. Shortest-path distances are unique, so the two
    agree on the result; what differs is the cost model — ``O(diameter)`` launches over the whole
    CSR here against ``O(E log V)`` sequential work there. The name is the result, per this
    package's naming rule, not the algorithm.

    One relaxation kernel per pass, so a pass is a pure function of the previous labels and the
    answer does not depend on thread interleaving. The pass count is data-dependent; on CUDA the
    whole pass loop runs as one device-side conditional graph (``wp.capture_while``, as
    [`bfs`][triwarp.graph.bfs] already does), so the convergence check costs no host readback at
    all rather than the one-per-pass a naive early exit would need — see the ``linalg`` note on
    ``check_every`` for why that per-pass sync would otherwise be the expensive part. The CPU
    backend, which has no conditional-graph capture, still checks with a plain readback per pass.

    For distance *across* a surface rather than along its edges — shorter, and what "geodesic"
    usually means — use [`heat_geodesic`][triwarp.heat.heat_geodesic]. The edge-graph
    distance is an upper bound on it.

    See Also
    --------
    [`edges_to_csr`][triwarp.graph.edges_to_csr]
    [`bfs`][triwarp.graph.bfs]
    [`heat_geodesic`][triwarp.heat.heat_geodesic]
    [`triwarp.remesh.isotropic_remesh`][triwarp.remesh.isotropic_remesh]
    [`scipy.sparse.csgraph.dijkstra`][]
    """
    require_same_device(adjacency=adjacency, values=values)
    node_count, offsets, columns = _validate_square_csr(adjacency)
    if max_iterations < 0:
        raise ValueError(f"max_iterations must be non-negative, got {max_iterations}")
    if int(values.shape[0]) != node_count:
        raise ValueError(
            f"values must have one entry per node, got {values.shape[0]} for {node_count} nodes"
        )

    device = wp.get_device(values.device)
    labels = wp.clone(values)
    if node_count == 0:
        return labels

    weights = adjacency.values  # pyright: ignore[reportAttributeAccessIssue]
    if int(weights.shape[0]) > 0 and float(tw.reduce.min(weights)) < 0.0:
        raise ValueError("adjacency weights must be non-negative for the envelope to converge")

    max_pass_count = max_iterations or node_count
    relaxed = wp.empty(node_count, dtype=wp.float32, device=device)
    changed = wp.zeros(1, dtype=wp.int32, device=device)

    if not device.is_cuda:
        # No conditional-graph capture on the CPU backend (see `bfs`'s identical device split);
        # the plain per-pass loop below, with its one 4-byte readback per pass, is already the
        # cheapest thing a CPU launch can do here.
        for _ in range(max_pass_count):
            changed.zero_()
            wp.launch(
                kernel_graph.shortest_path_envelope_pass,
                dim=node_count,
                inputs=[offsets, columns, weights, labels, relaxed, changed],
                device=device,
            )
            labels, relaxed = relaxed, labels
            if int(read_scalar(changed, 0)) == 0:
                break
        return labels

    # CUDA: the whole pass loop runs on-device via ``wp.capture_while`` (as ``bfs`` does), so the
    # only host sync in the common case is none at all -- each pass's convergence check and
    # iteration cap are folded into ``envelope_advance_and_check``, which runs after the relax
    # kernel and the label copy below.
    #
    # Buffer *swapping* (the CPU path's ``labels, relaxed = relaxed, labels``) cannot be captured:
    # a conditional graph replays the exact pointers its body recorded the one time it was traced,
    # so a Python-level rebind between iterations has no effect on the device-side loop -- every
    # replayed pass would keep reading and writing the same two buffers in the same direction.
    # ``wp.copy`` moves this pass's answer into ``labels`` in place instead, which is itself just a
    # device memcpy and captures fine (no allocation, no host sync, unlike ``wp.utils.array_scan``).
    counter = wp.zeros(1, dtype=wp.int32, device=device)
    condition = wp.ones(1, dtype=wp.int32, device=device)
    max_pass_count_i32 = wp.int32(max_pass_count)

    def envelope_pass_body() -> None:
        wp.launch(
            kernel_graph.shortest_path_envelope_pass,
            dim=node_count,
            inputs=[offsets, columns, weights, labels, relaxed, changed],
            device=device,
        )
        wp.copy(labels, relaxed)
        wp.launch(
            kernel_graph.envelope_advance_and_check,
            dim=1,
            inputs=[max_pass_count_i32, changed, counter, condition],
            device=device,
        )

    if wp.is_conditional_graph_supported():
        with wp.ScopedCapture(device) as capture:
            wp.capture_while(condition, envelope_pass_body)
        wp.capture_launch(capture.graph)
    else:
        wp.capture_while(condition, envelope_pass_body)
    return labels


# --- private helpers ---------------------------------------------------------------------


def _validate_square_csr(
    adjacency: wps.BsrMatrix[wp.Scalar],
) -> tuple[int, wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Check a CSR adjacency is square with scalar blocks, and unpack what the traversals need.

    The shared entry check of every function here that takes a prebuilt adjacency. The
    ``pyright: ignore`` comments live here rather than at each call site: Warp's stub omits
    ``BsrMatrix.offsets`` / ``.columns``.
    """
    node_count = adjacency.nrow  # pyright: ignore[reportAttributeAccessIssue]
    ncol = adjacency.ncol  # pyright: ignore[reportAttributeAccessIssue]
    if ncol != node_count:
        raise ValueError(f"adjacency must be square, got shape ({node_count}, {ncol})")
    if adjacency.block_shape != (1, 1):
        raise ValueError(f"adjacency must use 1x1 blocks, got block_shape {adjacency.block_shape}")
    offsets = adjacency.offsets  # pyright: ignore[reportAttributeAccessIssue]
    columns = adjacency.columns  # pyright: ignore[reportAttributeAccessIssue]
    return int(node_count), offsets, columns


def _validate_edge_list(edges: twt.Array2dInt32, node_count: int | None, *, validate: bool) -> int:
    """
    Check an ``(m, 2)`` edge list and resolve its node count.

    The shared entry check of every function here that takes an edge list instead of a prebuilt
    adjacency. When ``node_count`` is ``None`` it is inferred from the edges; when it is supplied
    it is checked for sign and, under ``validate``, the edge indices are checked to fall inside it.

    ``validate=False`` skips only the range check, which is one
    [`minmax`][triwarp.reduce.minmax] plus a host synchronization -- pass it from a caller that
    forwards the same edges to another checked entry point, so the reduction is not paid twice.
    """
    twt.ensure_edge_pairs(edges, "edges")

    if node_count is None:
        return int(tw.array.index_bound(edges))
    if node_count < 0:
        raise ValueError(f"node_count must be non-negative, got {node_count}")
    if validate and int(edges.shape[0]) > 0:
        # One 8-byte read of both bounds, not a copy of the whole edge buffer.
        lowest, highest = tw.reduce.minmax(edges)
        if lowest < 0 or highest >= node_count:
            raise ValueError(
                f"edge indices must lie in [0, {node_count}), got min={lowest} max={highest}"
            )
    return node_count
