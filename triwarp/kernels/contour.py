import warp as wp


@wp.func
def crossing_point(
    value_from: wp.Float, value_to: wp.Float, point_from: wp.vec3, point_to: wp.vec3
) -> wp.vec3:
    # Linear crossing of the zero level set along one edge. The caller only passes edges whose
    # endpoints straddle zero with the ``>= 0`` convention below, so the denominator is never zero.
    t = wp.float32(value_from / (value_from - value_to))
    return point_from + t * (point_to - point_from)


@wp.kernel
def marching_triangles_segments(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    values: wp.array[wp.Float],
    edge_ids: wp.array[wp.int32],
    out_valid: wp.array[wp.bool],
    out_segments: wp.array2d[wp.vec3],
    out_edges: wp.array2d[wp.int32],
) -> None:
    # One thread per face. ``values`` is the field with the isovalue already subtracted, and a value
    # of exactly zero counts as positive, so every cut face has exactly one vertex alone in sign and
    # yields exactly one segment: the two edges incident to that vertex are the crossed ones.
    f = int(wp.tid())
    i0 = faces[f * 3 + 0]
    i1 = faces[f * 3 + 1]
    i2 = faces[f * 3 + 2]
    d0 = values[i0]
    d1 = values[i1]
    d2 = values[i2]
    p0 = d0 >= values.dtype(0.0)
    p1 = d1 >= values.dtype(0.0)
    p2 = d2 >= values.dtype(0.0)

    if p0 == p1 and p1 == p2:
        out_valid[f] = False
        return

    # Local index of the vertex whose sign differs from the other two.
    lone = wp.int32(0)
    if p1 != p0 and p1 != p2:
        lone = wp.int32(1)
    elif p2 != p0 and p2 != p1:
        lone = wp.int32(2)
    next_index = (lone + wp.int32(1)) % wp.int32(3)
    prev_index = (lone + wp.int32(2)) % wp.int32(3)

    vertex_lone = faces[f * 3 + lone]
    vertex_next = faces[f * 3 + next_index]
    vertex_prev = faces[f * 3 + prev_index]
    value_lone = values[vertex_lone]

    point_next = crossing_point(
        value_lone, values[vertex_next], vertices[vertex_lone], vertices[vertex_next]
    )
    point_prev = crossing_point(
        value_lone, values[vertex_prev], vertices[vertex_lone], vertices[vertex_prev]
    )
    # Halfedge ``3f + k`` spans local corners ``k`` and ``k + 1``, so the edge from ``lone`` to the
    # next corner is halfedge ``lone`` and the edge from the previous corner to ``lone`` is halfedge
    # ``prev_index``; their unique-edge ids are what stitches segments into curves.
    edge_next = edge_ids[f * 3 + lone]
    edge_prev = edge_ids[f * 3 + prev_index]

    # Orient the segment so the region where the field exceeds the isovalue lies to its left, with
    # the face normal as up. That makes the crossing shared by two faces an outgoing endpoint of one
    # and an incoming endpoint of the other, which is what lets the curves be linked at all.
    out_valid[f] = True
    lone_is_positive = p0
    if lone == wp.int32(1):
        lone_is_positive = p1
    elif lone == wp.int32(2):
        lone_is_positive = p2

    if lone_is_positive:
        out_segments[f, 0] = point_next
        out_segments[f, 1] = point_prev
        out_edges[f, 0] = edge_next
        out_edges[f, 1] = edge_prev
    else:
        out_segments[f, 0] = point_prev
        out_segments[f, 1] = point_next
        out_edges[f, 0] = edge_prev
        out_edges[f, 1] = edge_next
