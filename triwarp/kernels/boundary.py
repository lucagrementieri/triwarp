import warp as wp


@wp.kernel
def find_ears(
    edge_boundary: wp.array[wp.bool],
    out_ear: wp.array[wp.int32],
    out_ear_opp: wp.array[wp.int32],
    out_count: wp.array[wp.int32],
) -> None:
    f = wp.int32(wp.tid())
    base = f * 3
    b0 = edge_boundary[base]
    b1 = edge_boundary[base + 1]
    b2 = edge_boundary[base + 2]
    n = wp.int32(b0) + wp.int32(b1) + wp.int32(b2)
    if n == 2:
        slot = wp.atomic_add(out_count, 0, 1)
        out_ear[slot] = f
        if not b0:
            out_ear_opp[slot] = wp.int32(0)
        elif not b1:
            out_ear_opp[slot] = wp.int32(1)
        else:
            out_ear_opp[slot] = wp.int32(2)


@wp.kernel
def scatter_boundary_neighbors(
    boundary_edges: wp.array2d[wp.int32],
    slot_count: wp.array[wp.int32],
    out_neighbors: wp.array2d[wp.int32],
) -> None:
    # Fill each boundary vertex's two neighbour slots from the *undirected* boundary edges. Every
    # boundary vertex of a manifold boundary has exactly two, so the atomic counter never exceeds
    # 2 -- a third increment would mean a pinch point and is dropped rather than corrupting memory.
    e = wp.int32(wp.tid())
    a = boundary_edges[e, 0]
    b = boundary_edges[e, 1]
    slot_a = wp.atomic_add(slot_count, a, 1)
    if slot_a < 2:
        out_neighbors[a, slot_a] = b
    slot_b = wp.atomic_add(slot_count, b, 1)
    if slot_b < 2:
        out_neighbors[b, slot_b] = a


@wp.kernel
def sort_boundary_neighbor_slots(neighbors: wp.array2d[wp.int32]) -> None:
    # Order each vertex's two slots ascending, so the dart numbering below does not depend on the
    # order the atomics happened to run in. This is what makes the non-orientable answer
    # reproducible: with no face winding to follow, slot 0 is the smaller neighbour by definition.
    v = wp.int32(wp.tid())
    first = neighbors[v, 0]
    second = neighbors[v, 1]
    if first >= 0 and second >= 0 and second < first:
        neighbors[v, 0] = second
        neighbors[v, 1] = first


@wp.kernel
def build_dart_successors(
    boundary_vertices: wp.array[wp.int32],
    neighbors: wp.array2d[wp.int32],
    out_edges: wp.array2d[wp.int32],
) -> None:
    # One successor edge per *dart*, where dart ``2 * v + s`` means "at vertex v, arrived from
    # neighbour slot s". Its successor leaves by the other slot: the next vertex is
    # ``w = neighbors[v, 1 - s]``, and the arriving slot at w is whichever of w's two slots holds
    # v. Every dart therefore has exactly one out-edge -- which is the property the directed
    # boundary edges lose on a non-orientable surface, and the whole reason this path exists.
    i, s = wp.tid()
    v = boundary_vertices[i]
    w = neighbors[v, 1 - s]
    arriving = wp.int32(0)
    if neighbors[w, 0] != v:
        arriving = wp.int32(1)
    out_edges[2 * i + s, 0] = 2 * v + s
    out_edges[2 * i + s, 1] = 2 * w + arriving


@wp.kernel
def count_boundary_degrees(
    directed_edges: wp.array2d[wp.int32], out_degrees: wp.array2d[wp.int32]
) -> None:
    # Column 0: how many boundary edges *leave* each vertex. Column 1: how many touch it at all.
    # One out-edge and two incidences is the well-behaved case. Two out-edges is the seam of a
    # non-orientable surface, where ``succ[tail] = head`` silently drops an edge. Four incidences
    # is a pinch point, where two loops meet and no 2-regular walk exists at all -- the two are
    # different defects and only the first one has a better answer available.
    e = wp.int32(wp.tid())
    tail = directed_edges[e, 0]
    head = directed_edges[e, 1]
    wp.atomic_add(out_degrees, tail, 0, 1)
    wp.atomic_add(out_degrees, tail, 1, 1)
    wp.atomic_add(out_degrees, head, 1, 1)


@wp.kernel
def flag_boundary_degree_defects(
    degrees: wp.array2d[wp.int32], out_flags: wp.array[wp.int32]
) -> None:
    # Reduce the per-vertex degrees to two bits, so the caller pays one 8-byte readback rather than
    # two full max-reductions: slot 0 is "some vertex has two out-edges" (a non-orientable seam)
    # and slot 1 is "some vertex has more than two incidences" (a pinch point).
    v = wp.int32(wp.tid())
    if degrees[v, 0] > 1:
        out_flags[0] = 1
    if degrees[v, 1] > 2:
        out_flags[1] = 1
