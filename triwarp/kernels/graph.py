import warp as wp

from triwarp.kernels import array as kernel_array


@wp.kernel
def edges_to_adjacency(
    edges: wp.array2d[wp.int32], out_rows: wp.array[wp.int32], out_cols: wp.array[wp.int32]
) -> None:
    tid = int(wp.tid())
    a = edges[tid, 0]
    b = edges[tid, 1]
    base = tid * 2
    out_rows[base] = a
    out_cols[base] = b
    out_rows[base + 1] = b
    out_cols[base + 1] = a


@wp.func
def unshared_vertex(
    v0: wp.int32, v1: wp.int32, v2: wp.int32, e0: wp.int32, e1: wp.int32
) -> wp.int32:
    result = wp.int32(-1)
    count = wp.int32(0)
    if v0 != e0 and v0 != e1:
        result = v0
        count = count + wp.int32(1)
    if v1 != e0 and v1 != e1:
        if count == wp.int32(0):
            result = v1
        count = count + wp.int32(1)
    if v2 != e0 and v2 != e1:
        if count == wp.int32(0):
            result = v2
        count = count + wp.int32(1)
    if count != wp.int32(1):
        return wp.int32(-1)
    return result


@wp.kernel
def face_adjacency_unshared(
    faces: wp.array[wp.int32],
    face_adjacency: wp.array2d[wp.int32],
    face_adjacency_edges: wp.array2d[wp.int32],
    out_unshared: wp.array2d[wp.int32],
) -> None:
    tid = int(wp.tid())
    f0 = face_adjacency[tid, 0] * 3
    f1 = face_adjacency[tid, 1] * 3
    e0 = face_adjacency_edges[tid, 0]
    e1 = face_adjacency_edges[tid, 1]
    out_unshared[tid, 0] = unshared_vertex(faces[f0 + 0], faces[f0 + 1], faces[f0 + 2], e0, e1)
    out_unshared[tid, 1] = unshared_vertex(faces[f1 + 0], faces[f1 + 1], faces[f1 + 2], e0, e1)


@wp.kernel
def face_adjacency_angles(
    face_normals: wp.array[wp.vec3],
    face_adjacency: wp.array2d[wp.int32],
    out_angles: wp.array[wp.float32],
) -> None:
    tid = int(wp.tid())
    normal_a = face_normals[face_adjacency[tid, 0]]
    normal_b = face_normals[face_adjacency[tid, 1]]
    out_angles[tid] = kernel_array.vector_angle_vec(normal_a, normal_b)


@wp.kernel
def mark_label_starts(sorted_labels: wp.array[wp.int32], out_is_start: wp.array[wp.bool]) -> None:
    # Segment boundaries of a label-sorted array: position 0 and every label change.
    i = int(wp.tid())
    out_is_start[i] = i == 0 or sorted_labels[i] != sorted_labels[i - 1]


@wp.kernel
def pack_label_node_keys(
    labels: wp.array[wp.int32], node_count: wp.int64, out_keys: wp.array[wp.int64]
) -> None:
    # Composite sort key label * n + node: sorting groups nodes by component with node ids
    # ascending inside each component (keys are strictly increasing within a label).
    i = int(wp.tid())
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
    q = int(wp.tid())
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
    t = int(wp.tid())
    q = kernel_array.binary_search_index(offsets, wp.int32(t)) - 1
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
def scatter_sorted_positions(
    sorted_nodes: wp.array[wp.int32], out_rank: wp.array[wp.int32]
) -> None:
    i = int(wp.tid())
    out_rank[sorted_nodes[i]] = i
