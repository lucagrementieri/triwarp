import warp as wp


@wp.kernel
def faces_to_edges(
    faces: wp.array[wp.int32],
    out_edges: wp.array2d[wp.int32],
) -> None:
    tid = int(wp.tid())
    f = tid * 3
    i0 = faces[f + 0]
    i1 = faces[f + 1]
    i2 = faces[f + 2]
    out_edges[f, 0] = i0
    out_edges[f, 1] = i1
    out_edges[f + 1, 0] = i1
    out_edges[f + 1, 1] = i2
    out_edges[f + 2, 0] = i2
    out_edges[f + 2, 1] = i0


@wp.kernel
def faces_to_edges_sorted(
    faces: wp.array[wp.int32],
    out_edges: wp.array2d[wp.int32],
) -> None:
    tid = int(wp.tid())
    f = tid * 3
    i0 = faces[f + 0]
    i1 = faces[f + 1]
    i2 = faces[f + 2]
    out_edges[f, 0] = wp.min(i0, i1)
    out_edges[f, 1] = wp.max(i0, i1)
    out_edges[f + 1, 0] = wp.min(i1, i2)
    out_edges[f + 1, 1] = wp.max(i1, i2)
    out_edges[f + 2, 0] = wp.min(i2, i0)
    out_edges[f + 2, 1] = wp.max(i2, i0)


@wp.func
def unshared_vertex(v0: wp.int32, v1: wp.int32, v2: wp.int32, e0: wp.int32, e1: wp.int32) -> wp.int32:
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
