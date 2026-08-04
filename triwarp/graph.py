"""Generic graph algorithms on sparse adjacency matrices and edge lists (mesh-agnostic)."""

from __future__ import annotations

import warp as wp
import warp.sparse as wps

import triwarp as tw
import triwarp.typing as twt
from triwarp.array import init_range
from triwarp.constants import INT32_MAX
from triwarp.kernels import graph as kernel_graph
from triwarp.kernels.algorithms import bfs as kernel_bfs
from triwarp.kernels.algorithms import connected_components as kernel_connected_components

# Frontier width at which the level-synchronous BFS hands the rest of the traversal to one serial
# thread. A level costs a fixed ~13 us of launch overhead whatever its frontier, while the serial
# walk costs ~0.5 us a node (measured, RTX 5090), so the two break even around 26 nodes wide.
#
# This replaces a ``node_count < 16384`` guard, which was the wrong predicate: its own comment named
# the failure mode as "small **or path-like**" but a node count only detects "small", and no static
# function of ``(node_count, nnz)`` separates a ribbon (average degree 4.0) from a sphere (6.0).
# Frontier width is observable and is the quantity that actually decides it.
_BFS_ESCAPE_FRONTIER = 32


def edges_to_csr(node_count: int, edges: twt.Array2dInt32) -> wps.BsrMatrix[wp.float32]:
    """
    Undirected adjacency as a 1x1-block ``warp.sparse.BsrMatrix`` (CSR form).

    Each undirected edge ``(a, b)`` contributes directed entries ``(a, b)`` and ``(b, a)``.

    Parameters
    ----------
    node_count
        Number of vertices ``0 .. node_count - 1``.
    edges
        ``(m, 2)`` ``wp.int32`` edge rows on the target device.

    Returns
    -------
    warp.sparse.BsrMatrix
        Square ``(node_count, node_count)`` adjacency with unit block values. Duplicate
        directed pairs from repeated input edges are merged (values summed).
    """
    device = edges.device
    m = int(edges.shape[0])

    n_entries = 2 * m
    rows = wp.empty(n_entries, dtype=wp.int32, device=device)
    cols = wp.empty(n_entries, dtype=wp.int32, device=device)
    if m > 0:
        wp.launch(kernel_graph.edges_to_adjacency, dim=m, inputs=[edges, rows, cols], device=device)
    data = wp.ones(n_entries, dtype=wp.float32, device=device)
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
    node_count = adjacency.nrow  # pyright: ignore[reportAttributeAccessIssue]
    ncol = adjacency.ncol  # pyright: ignore[reportAttributeAccessIssue]
    if ncol != node_count:
        raise ValueError(f"adjacency must be square, got shape ({node_count}, {ncol})")
    if adjacency.block_shape != (1, 1):
        raise ValueError(f"adjacency must use 1x1 blocks, got block_shape {adjacency.block_shape}")

    device = adjacency.device
    if node_count <= 1:
        return wp.zeros(node_count, dtype=wp.int32, device=device)
    if adjacency.nnz == 0:
        return init_range(node_count, device)

    offsets = adjacency.offsets  # pyright: ignore[reportAttributeAccessIssue]
    indices = adjacency.columns  # pyright: ignore[reportAttributeAccessIssue]

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
        unchecked path reads out of bounds rather than raising.

    See Also
    --------
    [`connected_component_labels`][triwarp.graph.connected_component_labels]
    [`connected_component_parity_from_edges`]
    [triwarp.graph.connected_component_parity_from_edges]
    [`face_connected_component_labels`][triwarp.adjacency.face_connected_component_labels]
    [`trimesh.graph.connected_component_labels`][]
    """
    twt.ensure_ndim(edges, 2, dtype=wp.int32)
    if int(edges.shape[1]) != 2:
        raise ValueError(f"edges must have shape (m, 2), got {edges.shape}")

    device = edges.device
    m = int(edges.shape[0])

    if node_count is None:
        node_count = tw.vertices.n_vertices(edges)
    elif node_count < 0:
        raise ValueError(f"node_count must be non-negative, got {node_count}")
    elif m == 0:
        return init_range(node_count, device)
    elif validate:
        edges_np = edges.numpy()
        if edges_np.min() < 0 or int(edges_np.max()) >= node_count:
            raise ValueError(
                f"edge indices must lie in [0, {node_count}), "
                f"got min={edges_np.min()} max={edges_np.max()}"
            )

    adjacency = edges_to_csr(node_count, edges)
    return connected_component_labels(adjacency)


def connected_component_parity_from_edges(
    edges: twt.Array2dInt32, signs: wp.array[wp.int32], node_count: int
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
    ValueError
        If ``edges`` is not ``(m, 2)``, ``signs`` is not length ``m``, or ``node_count`` is
        negative.

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
    twt.ensure_ndim(edges, 2, dtype=wp.int32)
    if int(edges.shape[1]) != 2:
        raise ValueError(f"edges must have shape (m, 2), got {edges.shape}")
    m = int(edges.shape[0])
    if int(signs.shape[0]) != m:
        raise ValueError(f"signs must have length {m} to match edges, got {int(signs.shape[0])}")
    if node_count < 0:
        raise ValueError(f"node_count must be non-negative, got {node_count}")

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


def bfs(
    adjacency: wps.BsrMatrix[wp.Scalar], source: int
) -> tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Single-source breadth-first search over a sparse CSR adjacency matrix.

    The discovery order, parent tree, and distances match
    [`scipy.sparse.csgraph.breadth_first_order`][] exactly when the adjacency columns are sorted
    ascending per row (as produced by [`edges_to_csr`][triwarp.graph.edges_to_csr]). This mirrors
    ``igl::bfs`` (`reference/libigl/include/igl/bfs.cpp`), additionally returning the BFS level of
    each node.

    Two engines, chosen by the *observed frontier width* rather than by any property of the graph
    known up front. The traversal starts level-synchronous and parallel, and hands over to a single
    serial thread as soon as its frontier is both narrow and no longer growing: a level costs the
    same seven fixed-size launches whatever it carries, so once the frontier is a handful of nodes
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
    !!! note "A long-diameter graph is at its ceiling here, and the ceiling is measured"

        On a two-wide ribbon of 40 962 vertices (diameter 20 480) the serial engine takes almost
        the whole traversal and runs **23.3 ms against scipy's 0.74 ms**. That is not a defect in
        the serial kernel, and three ways of attacking it were measured and all failed:

        - **It is memory-op *throughput* per thread, not a latency chain.** 481 ns per node against
          524 ns for fifteen *independent* loads issued from one thread on the same device (one
          dependent L2 load is 118 ns), so there is no stall left to hide. One- and two-deep
          software pipelining of the ``order -> offsets`` half measured **1.05x and 1.01x**, and
          batching the neighbours' ``dist`` loads through a register vector was a **loss**
          (21.0 against 19.7 ms).
        - **More threads cannot pay for their own synchronization.** A single-block cooperative
          rewrite — level-synchronous, order-exact by construction, verified byte-identical in
          ``order`` / ``parents`` / ``distances`` at 4, 8, 16 and 32 lanes — runs **39.9-44.1 ms, a
          2.2x loss**. A ribbon's frontier is ~2 nodes, so the block barriers *are* the cost:
          ``wp.tile_sum(wp.tile(x))[0]`` is 126 ns and ``wp.tile_scan_exclusive`` 353 ns, and a
          correct round needs one of each plus a second barrier — ~600 ns of synchronization against
          the serial engine's 962 ns for the whole level. Over 20 480 levels that is a ~12 ms floor
          on the synchronization alone, so no barrier arrangement reaches even 2x.
        - **The host is not available.** One CPU core does this in well under a millisecond, but
          Warp 1.15's CPU backend corrupts the process heap on this stack, so a host fallback would
          trade a slow row for a random crash.

        The parallel engine is far worse on this shape (seven fixed-size launches per level,
        ~410 ms even under conditional-graph capture), which is why the handover exists at all.
        Single-source BFS on a path graph has two-way parallelism, and one GPU thread
        pointer-chasing is ~30x slower than one CPU core doing the same, so **triwarp will not beat
        scipy on this axis.** The
        narrower ``sphere_med`` gap (2.2x) is a *different* problem: there the captured parallel
        engine launches all seven kernels at ``dim=node_count`` for a 130-wide frontier, and Warp
        cannot take a device-side launch dimension, so the lever there is fusing the seven kernels
        into two or three.
    """
    node_count = adjacency.nrow  # pyright: ignore[reportAttributeAccessIssue]
    ncol = adjacency.ncol  # pyright: ignore[reportAttributeAccessIssue]
    if ncol != node_count:
        raise ValueError(f"adjacency must be square, got shape ({node_count}, {ncol})")
    if adjacency.block_shape != (1, 1):
        raise ValueError(f"adjacency must use 1x1 blocks, got block_shape {adjacency.block_shape}")
    if source < 0 or source >= node_count:
        raise ValueError(f"source must be in [0, {node_count}), got {source}")

    device = adjacency.device
    offsets = adjacency.offsets  # pyright: ignore[reportAttributeAccessIssue]
    columns = adjacency.columns  # pyright: ignore[reportAttributeAccessIssue]

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
        return wp.clone(order_buffer[: int(reached.numpy()[0])]), parents, distances

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
    # from paying seven fixed-size launches for a two-node frontier, tens of thousands of times.
    scan_block = kernel_bfs.BFS_SCAN_BLOCK
    n_blocks = (node_count + scan_block - 1) // scan_block
    padded = n_blocks * scan_block
    claim_rank = wp.full(node_count, INT32_MAX, dtype=wp.int32, device=device)
    counts = wp.empty(padded, dtype=wp.int32, device=device)
    offsets_scan = wp.empty(padded, dtype=wp.int32, device=device)
    block_sums = wp.empty(n_blocks, dtype=wp.int32, device=device)
    # state = [frontier start, frontier end, level to emit, loop condition].
    state = wp.array([0, 1, 1, 1], dtype=wp.int32, device=device)
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
        wp.launch(
            kernel_bfs.bfs_count_claims,
            dim=node_count,
            inputs=[offsets, columns, order_buffer, state, distances, claim_rank, counts],
            device=device,
        )
        # Capture-safe fixed-buffer inclusive scan (wp.utils.array_scan allocates temp storage
        # internally, which conditional graph bodies reject).
        wp.launch_tiled(
            kernel_bfs.bfs_scan_blocks,
            dim=[n_blocks],
            inputs=[counts, offsets_scan, block_sums],
            block_dim=scan_block,
            device=device,
        )
        wp.launch(kernel_bfs.bfs_scan_block_sums, dim=1, inputs=[state, block_sums], device=device)
        wp.launch(
            kernel_bfs.bfs_add_block_offsets,
            dim=node_count,
            inputs=[block_sums, offsets_scan],
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
                order_buffer,
                parents,
                distances,
            ],
            device=device,
        )
        wp.launch(
            kernel_bfs.bfs_update_state,
            dim=1,
            inputs=[offsets_scan, wp.int32(_BFS_ESCAPE_FRONTIER), state],
            device=device,
        )

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
        tail = int(reached.numpy()[0])
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
    ValueError
        If ``edges`` is not ``(m, 2)``, an endpoint is outside ``[0, node_count)``, ``node_count``
        is negative, or ``source`` is out of range.

    See Also
    --------
    [`bfs`][triwarp.graph.bfs]
    [`connected_component_labels_from_edges`][triwarp.graph.connected_component_labels_from_edges]
    """
    twt.ensure_ndim(edges, 2, dtype=wp.int32)
    if int(edges.shape[1]) != 2:
        raise ValueError(f"edges must have shape (m, 2), got {edges.shape}")

    m = int(edges.shape[0])
    if node_count is None:
        node_count = tw.vertices.n_vertices(edges)
    else:
        if node_count < 0:
            raise ValueError(f"node_count must be non-negative, got {node_count}")
        if m > 0:
            edges_np = edges.numpy()
            if edges_np.min() < 0 or int(edges_np.max()) >= node_count:
                raise ValueError(
                    f"edge indices must lie in [0, {node_count}), "
                    f"got min={edges_np.min()} max={edges_np.max()}"
                )

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

    See Also
    --------
    [`bfs`][triwarp.graph.bfs]
    [`geodesic_ball`][triwarp.neighbors.geodesic_ball]
    """
    node_count = adjacency.nrow  # pyright: ignore[reportAttributeAccessIssue]
    ncol = adjacency.ncol  # pyright: ignore[reportAttributeAccessIssue]
    if ncol != node_count:
        raise ValueError(f"adjacency must be square, got shape ({node_count}, {ncol})")
    if adjacency.block_shape != (1, 1):
        raise ValueError(f"adjacency must use 1x1 blocks, got block_shape {adjacency.block_shape}")

    device = adjacency.device
    k = int(sources.shape[0])
    if k == 0:
        empty = wp.empty(0, dtype=wp.int32, device=device)
        return empty, wp.empty(0, dtype=wp.int32, device=device)

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
    keys_buffer = wp.empty(2 * n, dtype=wp.int64, device=device)
    node_ids = tw.array.init_sort_pair_indices(n, -1, device)
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
    wp.launch(
        kernel_graph.scatter_sorted_positions,
        dim=n,
        inputs=[sorted_nodes, node_rank],
        device=device,
    )

    segment_start = wp.empty(k, dtype=wp.int32, device=device)
    counts = wp.empty(k, dtype=wp.int32, device=device)
    wp.launch(
        kernel_graph.component_segment_bounds,
        dim=k,
        inputs=[sources, labels, sorted_keys, wp.int64(n), segment_start, counts],
        device=device,
    )
    offsets = wp.empty(k, dtype=wp.int32, device=device)
    wp.utils.array_scan(counts, out_array=offsets, inclusive=False)
    total = int(offsets[k - 1 :].numpy()[0]) + int(counts[k - 1 :].numpy()[0])

    neighbors = wp.empty(total, dtype=wp.int32, device=device)
    wp.launch(
        kernel_graph.emit_component_neighbors,
        dim=total,
        inputs=[sources, sorted_nodes, node_rank, segment_start, offsets, neighbors],
        device=device,
    )
    return neighbors, offsets
