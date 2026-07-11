import warp as wp


@wp.func
def edge_midpoint(
    vertices: wp.array[wp.vec3], unique_edges: wp.array2d[wp.int32], e: wp.int32
) -> wp.vec3:
    v0 = vertices[unique_edges[e, 0]]
    v1 = vertices[unique_edges[e, 1]]
    return (v0 + v1) * wp.float32(0.5)


@wp.kernel
def compute_midpoints(
    vertices: wp.array[wp.vec3],
    unique_edges: wp.array2d[wp.int32],
    out_midpoints: wp.array[wp.vec3],
) -> None:
    k = int(wp.tid())
    out_midpoints[k] = edge_midpoint(vertices, unique_edges, wp.int32(k))


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


@wp.func
def is_long_edge(length: wp.float32, max_edge: wp.float32) -> wp.bool:
    return length > max_edge


@wp.kernel
def build_midpoint_index(
    long_mask: wp.array[wp.bool],
    offsets: wp.array[wp.int32],
    vertex_offset: wp.int32,
    out_midpoint_idx: wp.array[wp.int32],
) -> None:
    e = int(wp.tid())
    if long_mask[e]:
        out_midpoint_idx[e] = vertex_offset + offsets[e]
    else:
        out_midpoint_idx[e] = wp.int32(-1)


@wp.kernel
def fill_edge_midpoints(
    vertices: wp.array[wp.vec3],
    unique_edges: wp.array2d[wp.int32],
    long_mask: wp.array[wp.bool],
    offsets: wp.array[wp.int32],
    out_mid: wp.array[wp.vec3],
) -> None:
    e = int(wp.tid())
    if long_mask[e]:
        out_mid[offsets[e]] = edge_midpoint(vertices, unique_edges, wp.int32(e))


@wp.kernel
def build_face_mid(
    inverse: wp.array[wp.int32],
    midpoint_idx: wp.array[wp.int32],
    out_face_mid: wp.array2d[wp.int32],
) -> None:
    f = int(wp.tid())
    out_face_mid[f, 0] = midpoint_idx[inverse[f * 3 + 0]]
    out_face_mid[f, 1] = midpoint_idx[inverse[f * 3 + 1]]
    out_face_mid[f, 2] = midpoint_idx[inverse[f * 3 + 2]]


@wp.func
def _write_tri(
    out_faces: wp.array2d[wp.int32],
    out_valid: wp.array[wp.bool],
    out_slot_index: wp.array[wp.int32],
    slot: wp.int32,
    tri: wp.vec3i,
    valid: wp.bool,
    src: wp.int32,
) -> None:
    out_faces[slot, 0] = tri[0]
    out_faces[slot, 1] = tri[1]
    out_faces[slot, 2] = tri[2]
    out_valid[slot] = valid
    out_slot_index[slot] = src


@wp.kernel
def emit_size_faces(
    faces: wp.array[wp.int32],
    face_mid: wp.array2d[wp.int32],
    vertices: wp.array[wp.vec3],
    index_in: wp.array[wp.int32],
    out_faces: wp.array2d[wp.int32],
    out_valid: wp.array[wp.bool],
    out_slot_index: wp.array[wp.int32],
) -> None:
    f = int(wp.tid())
    src = index_in[f]

    fv = wp.vec3i(faces[f * 3 + 0], faces[f * 3 + 1], faces[f * 3 + 2])
    mv = wp.vec3i(face_mid[f, 0], face_mid[f, 1], face_mid[f, 2])

    s0 = wp.int32(0)
    s1 = wp.int32(0)
    s2 = wp.int32(0)
    if mv[0] >= 0:
        s0 = 1
    if mv[1] >= 0:
        s1 = 1
    if mv[2] >= 0:
        s2 = 1
    count = s0 + s1 + s2

    # Four output triangle slots; unused slots are marked invalid.
    t0 = wp.vec3i(0, 0, 0)
    t1 = wp.vec3i(0, 0, 0)
    t2 = wp.vec3i(0, 0, 0)
    t3 = wp.vec3i(0, 0, 0)
    n0 = False
    n1 = False
    n2 = False
    n3 = False

    if count == 0:
        # No split edges: the face passes through unchanged.
        t0 = fv
        n0 = True
    elif count == 1:
        # Rotate so the split edge is (a, b); fan its midpoint p to the
        # opposite corner c as [a, p, c], [p, b, c].
        j = wp.int32(0)
        if s1 == 1:
            j = 1
        if s2 == 1:
            j = 2
        a = fv[j]
        b = fv[(j + 1) % 3]
        c = fv[(j + 2) % 3]
        p = mv[j]
        t0 = wp.vec3i(a, p, c)
        n0 = True
        t1 = wp.vec3i(p, b, c)
        n1 = True
    elif count == 2:
        # Rotate so the unsplit edge is (c, a); emit corner triangle [p, b, q]
        # plus the quad (a, p, q, c) cut along its shorter diagonal.
        u = wp.int32(0)
        if s1 == 0:
            u = 1
        if s2 == 0:
            u = 2
        j = (u + 1) % 3
        a = fv[j]
        b = fv[(j + 1) % 3]
        c = fv[(j + 2) % 3]
        p = mv[j]
        q = mv[(j + 1) % 3]
        t0 = wp.vec3i(p, b, q)
        n0 = True
        d_aq = wp.length_sq(vertices[a] - vertices[q])
        d_pc = wp.length_sq(vertices[p] - vertices[c])
        if d_aq <= d_pc:
            t1 = wp.vec3i(a, p, q)
            t2 = wp.vec3i(a, q, c)
        else:
            t1 = wp.vec3i(a, p, c)
            t2 = wp.vec3i(p, q, c)
        n1 = True
        n2 = True
    else:
        # Three split edges: the regular 1 -> 4 split (matches subdivide).
        m0 = mv[0]
        m1 = mv[1]
        m2 = mv[2]
        t0 = wp.vec3i(fv[0], m0, m2)
        t1 = wp.vec3i(m0, fv[1], m1)
        t2 = wp.vec3i(m2, m1, fv[2])
        t3 = wp.vec3i(m0, m1, m2)
        n0 = True
        n1 = True
        n2 = True
        n3 = True

    base = f * 4
    _write_tri(out_faces, out_valid, out_slot_index, base + 0, t0, n0, src)
    _write_tri(out_faces, out_valid, out_slot_index, base + 1, t1, n1, src)
    _write_tri(out_faces, out_valid, out_slot_index, base + 2, t2, n2, src)
    _write_tri(out_faces, out_valid, out_slot_index, base + 3, t3, n3, src)
