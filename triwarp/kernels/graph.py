import warp as wp

from triwarp.kernels import array as kernel_array


@wp.func
def write_adjacency_pair(
    edges: wp.array2d[wp.int32],
    e: wp.int32,
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
) -> None:
    # One undirected edge becomes two directed triplet slots: ``2e`` is ``(a, b)`` and ``2e + 1``
    # is ``(b, a)``. Both kernels below emit in exactly this layout, which is what lets the
    # weighted one duplicate the weight into the matching slots.
    a = edges[e, 0]
    b = edges[e, 1]
    base = e * 2
    out_rows[base] = a
    out_cols[base] = b
    out_rows[base + 1] = b
    out_cols[base + 1] = a


@wp.kernel
def edges_to_adjacency(
    edges: wp.array2d[wp.int32], out_rows: wp.array[wp.int32], out_cols: wp.array[wp.int32]
) -> None:
    # The unweighted path, whose values are a ``wp.ones`` fill rather than a launch.
    write_adjacency_pair(edges, wp.int32(wp.tid()), out_rows, out_cols)


@wp.kernel
def edges_to_adjacency_weighted(
    edges: wp.array2d[wp.int32],
    weights: wp.array[wp.float32],
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_values: wp.array[wp.float32],
) -> None:
    # The weighted path: the structure and the two duplicated values in one launch.
    e = wp.int32(wp.tid())
    write_adjacency_pair(edges, e, out_rows, out_cols)
    out_values[e * 2] = weights[e]
    out_values[e * 2 + 1] = weights[e]


@wp.kernel
def scatter_neighbor_lists(
    edges: wp.array2d[wp.int32],
    offsets: wp.array[wp.int32],
    cursor: wp.array[wp.int32],
    out_neighbors: wp.array[wp.int32],
) -> None:
    # Both directed entries of each undirected edge, into the CSR row the counting pass sized --
    # the payload half of ``graph.edges_to_neighbor_lists``. The same ``offsets[row] +
    # wp.atomic_add(cursor, row, 1)`` counting-sort fill as
    # [`scatter_vertex_faces`][triwarp.kernels.adjacency.scatter_vertex_faces]; what differs is the
    # payload, which is the *other endpoint* here and the owning face there, so the two are
    # siblings rather than one kernel.
    #
    # ``cursor`` is zeroed per-node scratch, not an output: each node's slots are handed out by the
    # atomic, so the column order within a row is thread order and **not sorted** -- nor stable,
    # so two runs return different permutations. That is deliberate and is what this build costs
    # less than ``bsr_from_triplets`` for. ``graph.edges_to_neighbor_lists(sort_rows=True)`` pins
    # it with one ``array.sort_segments`` launch, which makes the buffer identical to
    # ``graph.edges_to_csr``'s; that is what ``neighbors.geodesic_ball`` asks for, because it emits
    # its BFS queue in visit order.
    e = wp.int32(wp.tid())
    a = edges[e, 0]
    b = edges[e, 1]
    out_neighbors[offsets[a] + wp.atomic_add(cursor, a, 1)] = b
    out_neighbors[offsets[b] + wp.atomic_add(cursor, b, 1)] = a


@wp.kernel
def shortest_path_envelope_pass(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    weights: wp.array[wp.float32],
    labels: wp.array[wp.float32],
    out_labels: wp.array[wp.float32],
    out_state: wp.array[wp.int32],
) -> None:
    # One Bellman-Ford relaxation of ``q_i <= q_j + w(i, j)``, one thread per node, pulling from the
    # neighbours' labels of the *previous* round -- so the pass is a pure function of ``labels`` and
    # the result does not depend on how the threads interleave. Labels only ever go down, so the
    # iteration is monotone and converges in at most (graph diameter) passes.
    # (VCG ``UpdateQuality::VertexSaturate`` is this loop.)
    #
    # **Two passes per round, written into each other's buffer, would remove the wrapper's
    # ``wp.copy``** -- it is a whole device pass over the node array and worth a real fraction of
    # that call -- and it is **not portable**: the two devices then disagree. Measured with the
    # unrolled body, the CPU device relaxed strictly fewer nodes per round than CUDA at every cap,
    # because the recorded body did not replay as two passes per round there; even caps agreed
    # exactly. A ``wp.capture_while`` body is not guaranteed to execute as an indivisible unit
    # across devices, so a loop whose *result buffer* depends on the body running whole cannot rely
    # on it. A Python-level ping-pong cannot help either: the body is recorded once and replayed,
    # so rebinding the names would only take effect at record time.
    #
    # ``out_state`` is the shared round-loop word of ``kernels/array.py``; raising
    # ``LOOP_PROGRESS`` is this pass's "something moved", which the closing ``array.loop_advance``
    # turns into the next round's condition and clears. A plain store, not an atomic: the slot only
    # ever goes 0 -> 1 within a round, so every writer writes the same value.
    i = wp.int32(wp.tid())
    best = labels[i]
    for k in range(offsets[i], offsets[i + 1]):
        relaxed = labels[columns[k]] + weights[k]
        if relaxed < best:
            best = relaxed
    out_labels[i] = best
    if best < labels[i]:
        out_state[kernel_array.LOOP_PROGRESS] = 1


@wp.kernel
def scatter_successor(directed_edges: wp.array2d[wp.int32], out_next: wp.array[wp.int32]) -> None:
    tid = wp.int32(wp.tid())
    out_next[directed_edges[tid, 0]] = directed_edges[tid, 1]


@wp.kernel
def scatter_cycle_min_and_count(
    cycle_nodes: wp.array[wp.int32],
    next_node: wp.array[wp.int32],
    labels: wp.array[wp.int32],
    out_label_min: wp.array[wp.int32],
    out_label_count: wp.array[wp.int32],
    out_is_chain: wp.array[wp.int32],
) -> None:
    tid = wp.int32(wp.tid())
    v = cycle_nodes[tid]
    label = labels[v]
    wp.atomic_min(out_label_min, label, v)
    wp.atomic_add(out_label_count, label, wp.int32(1))
    # A node with no outgoing edge is a chain terminus, not a cycle: mark the whole
    # (undirected-connectivity) component so its nodes can be excluded before ranking.
    if next_node[v] < 0:
        wp.atomic_max(out_is_chain, label, wp.int32(1))


@wp.kernel
def chain_node_mask(
    cycle_nodes: wp.array[wp.int32],
    labels: wp.array[wp.int32],
    is_chain: wp.array[wp.int32],
    out_keep: wp.array[wp.int32],
) -> None:
    tid = wp.int32(wp.tid())
    v = cycle_nodes[tid]
    out_keep[tid] = wp.where(is_chain[labels[v]] != 0, wp.int32(0), wp.int32(1))


@wp.kernel
def init_rank_arrays(
    cycle_nodes: wp.array[wp.int32],
    next_node: wp.array[wp.int32],
    labels: wp.array[wp.int32],
    label_min: wp.array[wp.int32],
    out_successor: wp.array[wp.int32],
    out_steps: wp.array[wp.int32],
) -> None:
    # Pointer-jumping list ranking, step 1: cut each cycle at its canonical start (the smallest
    # node index, label_min) so cycles become chains ending in a fixed point (successor ==
    # self, steps == 0). Broken chains (-1 sentinel where a node has no out-edge) also terminate
    # at a fixed point, so the whole ranking finishes in a fixed round count on any input.
    tid = wp.int32(wp.tid())
    v = cycle_nodes[tid]
    start = label_min[labels[v]]
    nxt = next_node[v]
    if v == start or nxt < 0:
        out_successor[v] = v
        out_steps[v] = wp.int32(0)
    else:
        out_successor[v] = nxt
        out_steps[v] = wp.int32(1)


@wp.kernel
def jump_rank(
    cycle_nodes: wp.array[wp.int32],
    successor_in: wp.array[wp.int32],
    steps_in: wp.array[wp.int32],
    out_successor: wp.array[wp.int32],
    out_steps: wp.array[wp.int32],
) -> None:
    # Pointer doubling (Wyllie): after k rounds each node knows its 2^k-th successor and the
    # exact hop count to it; the fixed point at the cycle start contributes zero, so steps
    # converges to the hop distance to the start in ceil(log2(chain length)) rounds.
    tid = wp.int32(wp.tid())
    v = cycle_nodes[tid]
    s = successor_in[v]
    out_steps[v] = steps_in[v] + steps_in[s]
    out_successor[v] = successor_in[s]


@wp.kernel
def finalize_rank_positions(
    cycle_nodes: wp.array[wp.int32],
    labels: wp.array[wp.int32],
    label_count: wp.array[wp.int32],
    steps: wp.array[wp.int32],
    out_position: wp.array[wp.int32],
) -> None:
    # position = (cycle_length - hops to start) mod cycle_length; the positive modulo keeps
    # malformed chains (steps beyond cycle_length when in-edges collide) in range.
    tid = wp.int32(wp.tid())
    v = cycle_nodes[tid]
    cycle_length = label_count[labels[v]]
    out_position[tid] = kernel_array.wrap_index(cycle_length - steps[v], cycle_length)


@wp.kernel
def scatter_cycle_slot(
    cycle_nodes: wp.array[wp.int32],
    cycle_index: wp.array[wp.int32],
    position: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    out_cycles: wp.array[wp.int32],
) -> None:
    tid = wp.int32(wp.tid())
    out_cycles[offsets[cycle_index[tid]] + position[tid]] = cycle_nodes[tid]
