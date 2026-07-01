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
    while current != start:
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
