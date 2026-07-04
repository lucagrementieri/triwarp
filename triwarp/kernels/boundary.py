import warp as wp


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
def rank_loop_positions(
    boundary_vertices: wp.array[wp.int32],
    next_vertex: wp.array[wp.int32],
    labels: wp.array[wp.int32],
    label_min: wp.array[wp.int32],
    label_count: wp.array[wp.int32],
    out_position: wp.array[wp.int32],
) -> None:
    tid = int(wp.tid())
    v = boundary_vertices[tid]
    label = labels[v]
    start = label_min[label]
    loop_length = label_count[label]
    current = v
    steps = wp.int32(0)
    # Follow the successor chain back to the loop's start vertex. The guards make the walk
    # terminating on any input: a well-formed loop reaches `start` in fewer than `loop_length`
    # hops, while a broken successor chain (non-manifold boundary: a `-1` sentinel or a sub-cycle
    # not containing `start`) stops at the bound instead of spinning forever on the device.
    while current != start and current >= 0 and steps < loop_length:
        current = next_vertex[current]
        steps += 1
    out_position[tid] = (loop_length - steps) % loop_length


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
