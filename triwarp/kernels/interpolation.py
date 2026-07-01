import warp as wp


@wp.kernel
def average_onto_faces(
    faces: wp.array[wp.int32],
    vertex_values: wp.array[wp.float32],
    out_face_values: wp.array[wp.float32],
) -> None:
    f = int(wp.tid())
    v0 = faces[f * 3 + 0]
    v1 = faces[f * 3 + 1]
    v2 = faces[f * 3 + 2]
    out_face_values[f] = (
        vertex_values[v0] + vertex_values[v1] + vertex_values[v2]
    ) / wp.float32(3.0)


@wp.kernel
def scatter_face_values_sum_and_valence(
    faces: wp.array[wp.int32],
    face_values: wp.array[wp.float32],
    out_sum: wp.array[wp.float32],
    out_valence: wp.array[wp.float32],
) -> None:
    f = int(wp.tid())
    value = face_values[f]
    for j in range(3):
        vertex_index = faces[f * 3 + j]
        wp.atomic_add(out_sum, vertex_index, value)
        wp.atomic_add(out_valence, vertex_index, wp.float32(1.0))


@wp.kernel
def scatter_edges_sum_and_valence(
    faces: wp.array[wp.int32],
    edges: wp.array2d[wp.int32],
    edges_orientation: wp.array2d[wp.int32],
    edge_values: wp.array[wp.float32],
    out_sum: wp.array[wp.float32],
    out_valence: wp.array[wp.float32],
) -> None:
    f = int(wp.tid())
    for j in range(3):
        if edges_orientation[f, j] < 0:
            continue
        e = edges[f, j]
        vi = faces[f * 3 + (j + 1) % 3]
        vj = faces[f * 3 + (j + 2) % 3]
        value = edge_values[e]
        wp.atomic_add(out_sum, vi, value)
        wp.atomic_add(out_sum, vj, value)
        wp.atomic_add(out_valence, vi, wp.float32(1.0))
        wp.atomic_add(out_valence, vj, wp.float32(1.0))
