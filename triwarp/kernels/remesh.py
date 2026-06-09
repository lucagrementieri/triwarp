import warp as wp


@wp.kernel
def scatter_first_occurrence(
    inverse: wp.array[wp.int32], out_first_occurrence: wp.array[wp.int32]
) -> None:
    i = int(wp.tid())
    k = inverse[i]
    out_first_occurrence[k] = i


@wp.kernel
def compute_midpoints(
    vertices: wp.array[wp.vec3],
    edges: wp.array2d[wp.int32],
    first_occurrence: wp.array[wp.int32],
    out_midpoints: wp.array[wp.vec3],
) -> None:
    k = int(wp.tid())
    i = first_occurrence[k]
    v0 = vertices[edges[i, 0]]
    v1 = vertices[edges[i, 1]]
    out_midpoints[k] = (v0 + v1) * wp.float32(0.5)


@wp.kernel
def build_mid_idx(
    inverse: wp.array[wp.int32], vertex_offset: wp.int32, out_mid_idx: wp.array2d[wp.int32]
) -> None:
    f = int(wp.tid())
    out_mid_idx[f, 0] = inverse[f * 3 + 0] + vertex_offset
    out_mid_idx[f, 1] = inverse[f * 3 + 1] + vertex_offset
    out_mid_idx[f, 2] = inverse[f * 3 + 2] + vertex_offset


@wp.kernel
def subdivide_faces(
    faces: wp.array[wp.int32], mid_idx: wp.array2d[wp.int32], out_faces: wp.array[wp.int32]
) -> None:
    f = int(wp.tid())
    v0 = faces[f * 3 + 0]
    v1 = faces[f * 3 + 1]
    v2 = faces[f * 3 + 2]
    m0 = mid_idx[f, 0]
    m1 = mid_idx[f, 1]
    m2 = mid_idx[f, 2]
    base = f * 12
    # (v0, m0, m2)
    out_faces[base + 0] = v0
    out_faces[base + 1] = m0
    out_faces[base + 2] = m2
    # (m0, v1, m1)
    out_faces[base + 3] = m0
    out_faces[base + 4] = v1
    out_faces[base + 5] = m1
    # (m2, m1, v2)
    out_faces[base + 6] = m2
    out_faces[base + 7] = m1
    out_faces[base + 8] = v2
    # (m0, m1, m2)
    out_faces[base + 9] = m0
    out_faces[base + 10] = m1
    out_faces[base + 11] = m2
