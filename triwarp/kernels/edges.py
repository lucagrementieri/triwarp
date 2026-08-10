import warp as wp


@wp.func
def _write_edge(
    out_edges: wp.array2d[wp.int32], row: wp.int32, a: wp.int32, b: wp.int32, sort: wp.bool
):
    if sort:
        out_edges[row, 0] = wp.min(a, b)
        out_edges[row, 1] = wp.max(a, b)
    else:
        out_edges[row, 0] = a
        out_edges[row, 1] = b


@wp.kernel
def faces_to_edges(
    faces: wp.array[wp.int32], sort: wp.bool, out_edges: wp.array2d[wp.int32]
) -> None:
    # Three directed edges per face; ``sort`` puts the smaller vertex index first per row.
    tid = int(wp.tid())
    f = tid * 3
    i0 = faces[f + 0]
    i1 = faces[f + 1]
    i2 = faces[f + 2]
    _write_edge(out_edges, wp.int32(f), i0, i1, sort)
    _write_edge(out_edges, wp.int32(f + 1), i1, i2, sort)
    _write_edge(out_edges, wp.int32(f + 2), i2, i0, sort)


@wp.kernel
def edge_lengths(
    vertices: wp.array[wp.vec3], edges: wp.array2d[wp.int32], out_lengths: wp.array[wp.float32]
) -> None:
    i = int(wp.tid())
    out_lengths[i] = wp.length(vertices[edges[i, 1]] - vertices[edges[i, 0]])


@wp.kernel
def face_edge_lengths(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], out_lengths: wp.array2d[wp.float32]
) -> None:
    # Column ``e`` is the edge *opposite* corner ``e``, the igl intrinsic convention that
    # ``laplacian.cotmatrix_entries_intrinsic`` reads.
    f = int(wp.tid())
    v0 = vertices[faces[f * 3 + 0]]
    v1 = vertices[faces[f * 3 + 1]]
    v2 = vertices[faces[f * 3 + 2]]
    out_lengths[f, 0] = wp.length(v2 - v1)
    out_lengths[f, 1] = wp.length(v0 - v2)
    out_lengths[f, 2] = wp.length(v1 - v0)
