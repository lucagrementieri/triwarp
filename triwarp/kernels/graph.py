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
