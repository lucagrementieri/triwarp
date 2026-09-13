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
def shortest_path_envelope_pass(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    weights: wp.array[wp.float32],
    labels: wp.array[wp.float32],
    out_labels: wp.array[wp.float32],
    out_changed: wp.array[wp.int32],
) -> None:
    # One Bellman-Ford relaxation of ``q_i <= q_j + w(i, j)``, one thread per node, pulling from the
    # neighbours' labels of the *previous* round -- so the pass is a pure function of ``labels`` and
    # the result does not depend on how the threads interleave. Labels only ever go down, so the
    # iteration is monotone and converges in at most (graph diameter) passes; ``out_changed`` is the
    # host's early-exit signal. (VCG ``UpdateQuality::VertexSaturate`` is this loop.)
    i = wp.int32(wp.tid())
    best = labels[i]
    for k in range(offsets[i], offsets[i + 1]):
        relaxed = labels[columns[k]] + weights[k]
        if relaxed < best:
            best = relaxed
    out_labels[i] = best
    if best < labels[i]:
        out_changed[0] = 1


@wp.kernel
def envelope_advance_and_check(
    max_iterations: wp.int32,
    passes: wp.int32,
    out_changed: wp.array[wp.int32],
    out_counter: wp.array[wp.int32],
    out_condition: wp.array[wp.int32],
) -> None:
    # Runs once per *round* of ``passes`` relaxation passes -- the loop body runs two, writing each
    # into the other's buffer so neither has to be copied back. Advances the iteration count by
    # that many, decides whether ``shortest_path_envelope``'s captured
    # ``wp.capture_while`` loop should run another pass, and resets ``out_changed`` for the next
    # round to write into. ``out_changed`` only ever goes 0 -> 1 in a pass, so reading it once per
    # round is the OR over that round's passes -- which is what "did anything move" has to mean.
    # It is read here as this round's own answer and reset in the same launch for the next -- the
    # same in-place shape as ``bfs``'s ``out_order``. dim=1, so no thread
    # index -- this is host-side bookkeeping moved onto the device so the whole loop can run as one
    # conditional graph with no per-pass readback.
    out_counter[0] += passes
    out_condition[0] = wp.where(out_changed[0] != 0 and out_counter[0] < max_iterations, 1, 0)
    out_changed[0] = 0


@wp.kernel
def pack_label_node_keys(
    labels: wp.array[wp.int32], node_count: wp.int64, out_keys: wp.array[wp.int64]
) -> None:
    # Composite sort key label * n + node: sorting groups nodes by component with node ids
    # ascending inside each component (keys are strictly increasing within a label).
    i = wp.int32(wp.tid())
    out_keys[i] = wp.int64(labels[i]) * node_count + wp.int64(i)


@wp.kernel
def component_segment_bounds(
    sources: wp.array[wp.int32],
    labels: wp.array[wp.int32],
    sorted_keys: wp.array[wp.int64],
    node_count: wp.int64,
    out_segment_start: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
) -> None:
    # The label-L segment of the sorted key array is exactly
    # [lower_bound(L*n), lower_bound((L+1)*n)).
    q = wp.int32(wp.tid())
    label = wp.int64(labels[sources[q]])
    lo = kernel_array.binary_search_index_left(sorted_keys, label * node_count)
    hi = kernel_array.binary_search_index_left(sorted_keys, (label + wp.int64(1)) * node_count)
    out_segment_start[q] = lo
    out_counts[q] = hi - lo


@wp.kernel
def emit_component_neighbors(
    sources: wp.array[wp.int32],
    sorted_nodes: wp.array[wp.int32],
    node_rank: wp.array[wp.int32],
    segment_start: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    out_neighbors: wp.array[wp.int32],
) -> None:
    # One thread per output slot: slot 0 of each source's range holds the source itself, the
    # rest list its component's nodes in ascending order (the source's own position skipped).
    t = wp.int32(wp.tid())
    q = kernel_array.binary_search_index(offsets, t) - 1
    rel = t - offsets[q]
    source = sources[q]
    if rel == 0:
        out_neighbors[t] = source
        return
    base = segment_start[q]
    source_position = node_rank[source] - base
    idx = rel - 1
    if idx >= source_position:
        idx = rel
    out_neighbors[t] = sorted_nodes[base + idx]


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
