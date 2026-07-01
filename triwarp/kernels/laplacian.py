import warp as wp


@wp.func
def squared_edge_lengths(
    v0: wp.vec3, v1: wp.vec3, v2: wp.vec3
) -> tuple[wp.float32, wp.float32, wp.float32]:
    l2_0 = wp.dot(v1 - v2, v1 - v2)
    l2_1 = wp.dot(v2 - v0, v2 - v0)
    l2_2 = wp.dot(v0 - v1, v0 - v1)
    return l2_0, l2_1, l2_2


@wp.func
def sort_three_lengths(
    l0: wp.float32, l1: wp.float32, l2: wp.float32
) -> tuple[wp.float32, wp.float32, wp.float32]:
    if l0 > l1:
        tmp = l0
        l0 = l1
        l1 = tmp
    if l1 > l2:
        tmp = l1
        l1 = l2
        l2 = tmp
    if l0 > l1:
        tmp = l0
        l0 = l1
        l1 = tmp
    return l0, l1, l2


@wp.func
def doublearea_from_lengths(l0: wp.float32, l1: wp.float32, l2: wp.float32) -> wp.float32:
    l0, l1, l2 = sort_three_lengths(l0, l1, l2)
    arg = (l0 + (l1 + l2)) * (l2 - (l0 - l1)) * (l2 + (l0 - l1)) * (l0 + (l1 - l2))
    if arg < wp.float32(0.0):
        arg = wp.float32(0.0)
    dbl_area = wp.float32(0.5) * wp.sqrt(arg)
    if dbl_area != dbl_area:
        return wp.float32(0.0)
    return dbl_area


@wp.func
def cot_entries_from_l2(
    l2_0: wp.float32, l2_1: wp.float32, l2_2: wp.float32, dbl_area: wp.float32
) -> tuple[wp.float32, wp.float32, wp.float32]:
    inv_denom = wp.float32(4.0) * dbl_area
    c0 = (l2_1 + l2_2 - l2_0) / inv_denom
    c1 = (l2_2 + l2_0 - l2_1) / inv_denom
    c2 = (l2_0 + l2_1 - l2_2) / inv_denom
    return c0, c1, c2


@wp.func
def cot_entries_from_edge_lengths(
    l0: wp.float32, l1: wp.float32, l2: wp.float32
) -> tuple[wp.float32, wp.float32, wp.float32]:
    l2_0 = l0 * l0
    l2_1 = l1 * l1
    l2_2 = l2 * l2
    dbl_area = doublearea_from_lengths(l0, l1, l2)
    return cot_entries_from_l2(l2_0, l2_1, l2_2, dbl_area)


@wp.kernel
def cotmatrix_entries(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    out_cot: wp.array2d[wp.float32],
) -> None:
    f = int(wp.tid())
    i0 = faces[f * 3 + 0]
    i1 = faces[f * 3 + 1]
    i2 = faces[f * 3 + 2]
    v0 = vertices[i0]
    v1 = vertices[i1]
    v2 = vertices[i2]
    l2_0, l2_1, l2_2 = squared_edge_lengths(v0, v1, v2)
    l0 = wp.sqrt(l2_0)
    l1 = wp.sqrt(l2_1)
    l2 = wp.sqrt(l2_2)
    dbl_area = doublearea_from_lengths(l0, l1, l2)
    c0, c1, c2 = cot_entries_from_l2(l2_0, l2_1, l2_2, dbl_area)
    out_cot[f, 0] = c0
    out_cot[f, 1] = c1
    out_cot[f, 2] = c2


@wp.kernel
def cotmatrix_entries_intrinsic(
    edge_lengths: wp.array2d[wp.float32],
    out_cot: wp.array2d[wp.float32],
) -> None:
    f = int(wp.tid())
    l0 = edge_lengths[f, 0]
    l1 = edge_lengths[f, 1]
    l2 = edge_lengths[f, 2]
    c0, c1, c2 = cot_entries_from_edge_lengths(l0, l1, l2)
    out_cot[f, 0] = c0
    out_cot[f, 1] = c1
    out_cot[f, 2] = c2


@wp.kernel
def cotmatrix_triplets(
    faces: wp.array[wp.int32],
    cot_entries: wp.array2d[wp.float32],
    out_rows: wp.array[wp.int32],
    out_cols: wp.array[wp.int32],
    out_vals: wp.array[wp.float32],
) -> None:
    f = int(wp.tid())
    for e in range(3):
        c0 = (e + 1) % 3
        c1 = (e + 2) % 3
        source = faces[f * 3 + c0]
        dest = faces[f * 3 + c1]
        w = cot_entries[f, e]
        base = f * 12 + e * 4
        out_rows[base + 0] = source
        out_cols[base + 0] = dest
        out_vals[base + 0] = w
        out_rows[base + 1] = dest
        out_cols[base + 1] = source
        out_vals[base + 1] = w
        out_rows[base + 2] = source
        out_cols[base + 2] = source
        out_vals[base + 2] = -w
        out_rows[base + 3] = dest
        out_cols[base + 3] = dest
        out_vals[base + 3] = -w
