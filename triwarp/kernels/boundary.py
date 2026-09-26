import warp as wp

from triwarp.kernels.adjacency import edge_endpoints
from triwarp.kernels.array import loop_rim_edge, pack_directed_key, pack_ranked_key, scanned_count
from triwarp.kernels.grouping import sorted_run_of_length
from triwarp.kernels.halfedge import halfedge_endpoints, next_boundary_halfedge

# Every boundary query here is one radix sort of the halfedges' undirected edge keys, carrying each
# halfedge's index as the payload: a boundary edge is then a run of exactly one key, and its
# payload names the halfedge -- which is row ``h`` of ``edges.faces_to_edges``, so the edge's
# endpoints (sorted or directed) are read straight off ``faces`` and no ``(3F, 2)`` edge table is
# ever built. The kernels below write the sort's input and read its verdict.


@wp.kernel
def table_edge_keys_and_order(
    edges_sorted: wp.array2d[wp.int32],
    base: wp.uint64,
    out_keys: wp.array[wp.uint64],
    out_order: wp.array[wp.int32],
) -> None:
    # The same sort input from a caller's precomputed ``(3F, 2)`` sorted edge table: row ``h`` is
    # already ``[min, max]``, so it packs as it stands, exactly as ``grouping.pack_indices`` would.
    h = wp.int32(wp.tid())
    out_keys[h] = pack_directed_key(edges_sorted[h, 0], edges_sorted[h, 1], base)
    out_order[h] = h


@wp.func
def boundary_halfedge_pair(
    faces: wp.array[wp.int32], table: wp.array2d[wp.int32], sort_pair: wp.bool, h: wp.int32
) -> tuple[wp.int32, wp.int32]:
    # Row ``h`` of the edge table the caller holds, or of the one ``faces`` implies when ``table``
    # is ``None`` (a null descriptor, whose ``shape[0]`` reads 0): halfedge ``h``'s directed pair,
    # ascending when ``sort_pair``.
    if table.shape[0] > 0:
        return table[h, 0], table[h, 1]
    if sort_pair:
        return edge_endpoints(faces, h)
    return halfedge_endpoints(faces, h)


@wp.kernel
def mark_boundary_runs(
    sorted_keys: wp.array[wp.uint64],
    order: wp.array[wp.int32],
    n: wp.int32,
    faces: wp.array[wp.int32],
    table: wp.array2d[wp.int32],
    out_degrees: wp.array2d[wp.int32],
    out_flags: wp.array[wp.int32],
) -> None:
    # 1 where a sorted position holds a key occurring exactly once -- a boundary edge -- as the
    # ``int32`` flag ``wp.utils.array_scan`` then scans in place, over ``out_flags``' first ``n``
    # entries. With ``out_degrees`` (else ``None``), ``boundary_loops_batched``'s degree census of
    # the directed rows (see ``boundary_halfedge_pair``) rides in the same launch, its two defect
    # bits stamped into the zeroed ``out_flags[n]`` and ``out_flags[n + 1]`` -- the scan does not
    # reach them, so the total and both bits come back in one readback.
    #
    # Degree column 0: how many boundary edges *leave* each vertex. Column 1: how many touch it at
    # all. One out-edge and two incidences is the well-behaved case. Two out-edges is the seam of a
    # non-orientable surface, where ``succ[tail] = head`` silently drops an edge (bit ``n``). Four
    # incidences is a pinch point, where two loops meet and no 2-regular walk over vertices exists
    # at all (bit ``n + 1``) -- the two are different defects with different walks: the seam's is
    # undirected, the pinch's is over halfedges (``boundary_halfedge_successors``).
    #
    # The bits are stamped here rather than by a second pass over the degree table:
    # ``wp.atomic_add`` returns the value the slot held *before* the increment, so the thread that
    # pushes a vertex past the threshold is the one that knows it. Boundary vertices are the only
    # ones whose degrees are ever non-zero, so no pass over the vertices is needed.
    i = wp.int32(wp.tid())
    boundary = sorted_run_of_length(sorted_keys, n, i, 1)
    out_flags[i] = wp.where(boundary, wp.int32(1), wp.int32(0))
    if boundary and out_degrees.shape[0] > 0:
        tail, head = boundary_halfedge_pair(faces, table, False, order[i])
        if wp.atomic_add(out_degrees, tail, 0, 1) >= 1:
            out_flags[n] = 1
        if wp.atomic_add(out_degrees, tail, 1, 1) >= 2:
            out_flags[n + 1] = 1
        if wp.atomic_add(out_degrees, head, 1, 1) >= 2:
            out_flags[n + 1] = 1


@wp.kernel
def emit_boundary_edges(
    inclusive: wp.array[wp.int32],
    order: wp.array[wp.int32],
    faces: wp.array[wp.int32],
    table: wp.array2d[wp.int32],
    sort_pair: wp.bool,
    out_rows: wp.array[wp.int32],
    out_edges: wp.array2d[wp.int32],
) -> None:
    # One thread per sorted position; ``inclusive`` is ``mark_boundary_runs``' flags scanned in
    # place, so a boundary edge is where the scan steps and its rank is the step's start. Rows come
    # out in ascending key order -- the order ``grouping.group`` emitted them in. Both outputs are
    # optional (``None``, read as ``shape[0] == 0``): the halfedge index, and its edge row (see
    # ``boundary_halfedge_pair``).
    i = wp.int32(wp.tid())
    g, count = scanned_count(inclusive, i)
    if count == 0:
        return
    h = order[i]
    if out_rows.shape[0] > 0:
        out_rows[g] = h
    if out_edges.shape[0] > 0:
        a, b = boundary_halfedge_pair(faces, table, sort_pair, h)
        out_edges[g, 0] = a
        out_edges[g, 1] = b


@wp.kernel
def mark_boundary_vertices(
    sorted_keys: wp.array[wp.uint64],
    order: wp.array[wp.int32],
    n: wp.int32,
    faces: wp.array[wp.int32],
    table: wp.array2d[wp.int32],
    out_flags: wp.array[wp.int32],
) -> None:
    # Flag both endpoints of every boundary edge in a zeroed per-vertex ``int32`` array, which the
    # caller scans in place: the sorted unique boundary vertices then come out of the scan's steps
    # with no edge list and no ``unique_1d``. Concurrent writers all store 1, so no atomic.
    i = wp.int32(wp.tid())
    if not sorted_run_of_length(sorted_keys, n, i, 1):
        return
    a, b = boundary_halfedge_pair(faces, table, False, order[i])
    out_flags[a] = 1
    out_flags[b] = 1


@wp.kernel
def boundary_halfedge_mask(
    sorted_keys: wp.array[wp.uint64],
    order: wp.array[wp.int32],
    n: wp.int32,
    out_mask: wp.array[wp.bool],
) -> None:
    # Per halfedge: is its edge a boundary edge? ``order`` is a permutation of ``0 .. n - 1``, so
    # every entry of ``out_mask`` is written exactly once and it needs no zero fill -- and there is
    # no scan and no readback, since the mask's size is known.
    i = wp.int32(wp.tid())
    out_mask[order[i]] = sorted_run_of_length(sorted_keys, n, i, 1)


@wp.func
def ear_interior_corner(edge_boundary: wp.array[wp.bool], f: wp.int32) -> wp.int32:
    # The local index of face ``f``'s one interior edge when exactly two of its three edges are
    # boundary edges -- an ear -- else ``-1``.
    b0 = edge_boundary[3 * f]
    b1 = edge_boundary[3 * f + 1]
    b2 = edge_boundary[3 * f + 2]
    if wp.int32(b0) + wp.int32(b1) + wp.int32(b2) != 2:
        return -1
    if not b0:
        return 0
    if not b1:
        return 1
    return 2


@wp.kernel
def mark_ears(edge_boundary: wp.array[wp.bool], out_flags: wp.array[wp.int32]) -> None:
    # 0/1 per face: is it an ear? The ``int32`` flag the caller scans in place.
    f = wp.int32(wp.tid())
    out_flags[f] = wp.where(ear_interior_corner(edge_boundary, f) >= 0, wp.int32(1), wp.int32(0))


@wp.kernel
def emit_ears(
    edge_boundary: wp.array[wp.bool],
    inclusive: wp.array[wp.int32],
    out_ear: wp.array[wp.int32],
    out_ear_opp: wp.array[wp.int32],
) -> None:
    # Where ``mark_ears``' in-place scan steps, write the ear and its interior corner at its rank:
    # ascending face order on both devices, with no atomic cursor and no trim copy.
    f = wp.int32(wp.tid())
    slot, flag = scanned_count(inclusive, f)
    if flag != 0:
        out_ear[slot] = f
        out_ear_opp[slot] = ear_interior_corner(edge_boundary, f)


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
def boundary_halfedge_successors(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    boundary_halfedges: wp.array[wp.int32],
    out_edges: wp.array2d[wp.int32],
) -> None:
    # The boundary as a successor graph over *halfedges*: row ``(h, next)`` for each boundary
    # halfedge, where ``next`` is the boundary halfedge leaving ``h``'s tip in ``h``'s own sector.
    # Over vertices a pinch point has two successors; over halfedges every node has exactly one,
    # which is what ``successor_cycles`` needs. A fan that does not close (a malformed twin
    # table) maps to a self-loop, so the graph stays in range and the walk stays bounded.
    i = wp.int32(wp.tid())
    h = boundary_halfedges[i]
    following = next_boundary_halfedge(faces, twins, h)
    out_edges[i, 0] = h
    out_edges[i, 1] = wp.where(following >= 0, following, h)


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
    ell, a, c = loop_rim_edge(flat_loops, loop_id, loop_starts, loop_sizes, vertices, t)
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
    ell, a, c = loop_rim_edge(flat_loops, loop_id, loop_starts, loop_sizes, vertices, t)
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
