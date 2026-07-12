import warp as wp

from triwarp.kernels.array import wrap_index


@wp.kernel
def scatter_successor(directed_edges: wp.array2d[wp.int32], out_next: wp.array[wp.int32]) -> None:
    tid = int(wp.tid())
    out_next[directed_edges[tid, 0]] = directed_edges[tid, 1]


@wp.kernel
def scatter_loop_min_and_count(
    boundary_vertices: wp.array[wp.int32],
    labels: wp.array[wp.int32],
    out_label_min: wp.array[wp.int32],
    out_label_count: wp.array[wp.int32],
) -> None:
    tid = int(wp.tid())
    v = boundary_vertices[tid]
    label = labels[v]
    wp.atomic_min(out_label_min, label, v)
    wp.atomic_add(out_label_count, label, wp.int32(1))


@wp.kernel
def init_rank_arrays(
    boundary_vertices: wp.array[wp.int32],
    next_vertex: wp.array[wp.int32],
    labels: wp.array[wp.int32],
    label_min: wp.array[wp.int32],
    out_successor: wp.array[wp.int32],
    out_steps: wp.array[wp.int32],
) -> None:
    # Pointer-jumping list ranking, step 1: cut each loop at its canonical start (the smallest
    # vertex index, label_min) so cycles become chains ending in a fixed point (successor ==
    # self, steps == 0). Broken chains (-1 sentinel on non-manifold boundaries) also terminate
    # at a fixed point, so the whole ranking finishes in a fixed round count on any input.
    tid = int(wp.tid())
    v = boundary_vertices[tid]
    start = label_min[labels[v]]
    nxt = next_vertex[v]
    if v == start or nxt < 0:
        out_successor[v] = v
        out_steps[v] = wp.int32(0)
    else:
        out_successor[v] = nxt
        out_steps[v] = wp.int32(1)


@wp.kernel
def jump_rank(
    boundary_vertices: wp.array[wp.int32],
    successor_in: wp.array[wp.int32],
    steps_in: wp.array[wp.int32],
    out_successor: wp.array[wp.int32],
    out_steps: wp.array[wp.int32],
) -> None:
    # Pointer doubling (Wyllie): after k rounds each vertex knows its 2^k-th successor and the
    # exact hop count to it; the fixed point at the loop start contributes zero, so steps
    # converges to the hop distance to the start in ceil(log2(chain length)) rounds.
    tid = int(wp.tid())
    v = boundary_vertices[tid]
    s = successor_in[v]
    out_steps[v] = steps_in[v] + steps_in[s]
    out_successor[v] = successor_in[s]


@wp.kernel
def finalize_rank_positions(
    boundary_vertices: wp.array[wp.int32],
    labels: wp.array[wp.int32],
    label_count: wp.array[wp.int32],
    steps: wp.array[wp.int32],
    out_position: wp.array[wp.int32],
) -> None:
    # position = (loop_length - hops to start) mod loop_length; the positive modulo keeps
    # malformed chains (steps beyond loop_length on non-manifold boundaries) in range.
    tid = int(wp.tid())
    v = boundary_vertices[tid]
    loop_length = label_count[labels[v]]
    out_position[tid] = wrap_index(loop_length - steps[v], loop_length)


@wp.kernel
def scatter_loop_slot(
    boundary_vertices: wp.array[wp.int32],
    loop_index: wp.array[wp.int32],
    position: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    out_loops: wp.array[wp.int32],
) -> None:
    tid = int(wp.tid())
    out_loops[offsets[loop_index[tid]] + position[tid]] = boundary_vertices[tid]


@wp.kernel
def find_ears(
    edge_boundary: wp.array[wp.bool],
    out_ear: wp.array[wp.int32],
    out_ear_opp: wp.array[wp.int32],
    out_count: wp.array[wp.int32],
) -> None:
    f = int(wp.tid())
    base = f * 3
    b0 = edge_boundary[base]
    b1 = edge_boundary[base + 1]
    b2 = edge_boundary[base + 2]
    n = wp.int32(b0) + wp.int32(b1) + wp.int32(b2)
    if n == 2:
        slot = wp.atomic_add(out_count, 0, 1)
        out_ear[slot] = wp.int32(f)
        if not b0:
            out_ear_opp[slot] = wp.int32(0)
        elif not b1:
            out_ear_opp[slot] = wp.int32(1)
        else:
            out_ear_opp[slot] = wp.int32(2)
