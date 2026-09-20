import warp as wp

from triwarp.kernels.array import loop_next_slot, pack_ranked_key


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
    directed_edges: wp.array2d[wp.int32],
    out_degrees: wp.array2d[wp.int32],
    out_flags: wp.array[wp.int32],
) -> None:
    # Column 0: how many boundary edges *leave* each vertex. Column 1: how many touch it at all.
    # One out-edge and two incidences is the well-behaved case. Two out-edges is the seam of a
    # non-orientable surface, where ``succ[tail] = head`` silently drops an edge. Four incidences
    # is a pinch point, where two loops meet and no 2-regular walk exists at all -- the two are
    # different defects and only the first one has a better answer available.
    #
    # The two defect bits are stamped here rather than by a second pass over the degree table:
    # ``wp.atomic_add`` returns the value the slot held *before* the increment, so the thread that
    # pushes a vertex past the threshold is the one that knows it. That retires a launch whose
    # ``dim`` was the whole vertex count for an answer that is two bits -- boundary vertices are
    # the only ones whose degrees are ever non-zero, and there are ``2 * n_edges`` of those.
    # Slot 0 is "some vertex has two out-edges", slot 1 is "some vertex has more than two
    # incidences", and the caller reads both in one 8-byte transfer.
    e = wp.int32(wp.tid())
    tail = directed_edges[e, 0]
    head = directed_edges[e, 1]
    if wp.atomic_add(out_degrees, tail, 0, 1) >= 1:
        out_flags[0] = 1
    if wp.atomic_add(out_degrees, tail, 1, 1) >= 2:
        out_flags[1] = 1
    if wp.atomic_add(out_degrees, head, 1, 1) >= 2:
        out_flags[1] = 1


@wp.kernel
def loop_perimeters(
    flat_loops: wp.array[wp.int32],
    loop_id: wp.array[wp.int32],
    loop_starts: wp.array[wp.int32],
    loop_sizes: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    out_perimeter: wp.array[wp.float32],
) -> None:
    # Segmented ``polyline_length(closed=True)``: the arc length of every loop in one launch, so
    # ``preserve_largest_hole`` costs one readback instead of two per loop.
    t = wp.int32(wp.tid())
    ell = loop_id[t]
    a = vertices[flat_loops[t]]
    c = vertices[flat_loops[loop_next_slot(loop_id, loop_starts, loop_sizes, t)]]
    wp.atomic_add(out_perimeter, ell, wp.length(c - a))


@wp.kernel
def loop_directed_areas(
    flat_loops: wp.array[wp.int32],
    loop_id: wp.array[wp.int32],
    loop_starts: wp.array[wp.int32],
    loop_sizes: wp.array[wp.int32],
    vertices: wp.array[wp.vec3],
    out_directed_area: wp.array[wp.vec3],
) -> None:
    # Half the sum of ``p_i x p_{i+1}`` around the loop: the directed area vector, whose norm is the
    # area of the planar polygon spanning the loop and whose direction is that polygon's normal.
    # Origin-independent because the cross products of a *closed* ring cancel the shift, so no
    # centroid pass is needed -- and accumulated per segment in one launch, like the perimeter.
    t = wp.int32(wp.tid())
    ell = loop_id[t]
    a = vertices[flat_loops[t]]
    c = vertices[flat_loops[loop_next_slot(loop_id, loop_starts, loop_sizes, t)]]
    wp.atomic_add(out_directed_area, ell, wp.float32(0.5) * wp.cross(a, c))


@wp.kernel
def longest_loop_key(
    loop_starts: wp.array[wp.int32], loop_sizes: wp.array[wp.int32], out_best: wp.array[wp.int64]
) -> None:
    # The longest packed loop, as one ``wp.atomic_max`` over ``pack_ranked_key``. The low half
    # carries the loop's *start* rather than its index, which is what lets a single readback of
    # this key give the caller both halves of the answer -- a second read of ``loop_starts`` at
    # the winning index would otherwise cost as much again as the reduction. Starts increase with
    # the loop index, so "lowest start on a tie" is "lowest index on a tie" and the packer's
    # tie-break is the one a host-side first-maximum scan would have produced.
    #
    # Against unpacking the loops and scanning them on the host, output identical: a wash at 7
    # rims -- where the reduction's launch costs about what the handful of array views it removes
    # did -- 1.8x on the whole public call at 384, and 4.1x on a mesh with several thousand. The
    # win grows with the rim count because the host form was linear in it and this is not.
    ell = wp.int32(wp.tid())
    wp.atomic_max(out_best, 0, pack_ranked_key(loop_sizes[ell], loop_starts[ell]))
