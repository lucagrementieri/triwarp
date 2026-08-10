import warp as wp

from triwarp.constants import INT32_MAX_CONSTANT


@wp.func
def halfedge_prev(h: wp.int32) -> wp.int32:
    # Halfedge ``3*f + k`` runs ``faces[3f+k] -> faces[3f+(k+1)%3]``, so the face-local cycle is
    # pure index arithmetic: no stored ``prev`` pointer.
    k = h % wp.int32(3)
    if k == wp.int32(0):
        return h + wp.int32(2)
    return h - wp.int32(1)


@wp.func
def halfedge_destination(faces: wp.array[wp.int32], h: wp.int32) -> wp.int32:
    # Halfedge ``3*f + k`` ends at the next corner of its face; ``faces[h]`` is its origin.
    k = h % wp.int32(3)
    if k == wp.int32(2):
        return faces[h - wp.int32(2)]
    return faces[h + wp.int32(1)]


@wp.kernel
def pair_sorted_halfedges(
    sorted_keys: wp.array[wp.uint64],
    order: wp.array[wp.int32],
    out_twins: wp.array[wp.int32],
    out_nonmanifold: wp.array[wp.int32],
) -> None:
    # One thread per position in the hash-sorted halfedge list; only the first position of each run
    # of equal keys acts, so each undirected edge is resolved exactly once. Run length 1 is a
    # boundary edge (twin stays -1), 2 an interior edge, 3+ a non-manifold edge.
    i = int(wp.tid())
    key = sorted_keys[i]
    if i > 0 and sorted_keys[i - 1] == key:
        return
    n = int(sorted_keys.shape[0])
    if i + 1 >= n or sorted_keys[i + 1] != key:
        return
    if i + 2 < n and sorted_keys[i + 2] == key:
        wp.atomic_add(out_nonmanifold, 0, 1)
        return
    h0 = order[i]
    h1 = order[i + 1]
    out_twins[h0] = h1
    out_twins[h1] = h0


@wp.kernel
def ring_start_halfedges(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    out_interior_start: wp.array[wp.int32],
    out_boundary_start: wp.array[wp.int32],
) -> None:
    # Per vertex the lowest-indexed outgoing halfedge, tracked twice: over all of them, and over the
    # twin-less (boundary) ones alone. A boundary vertex must start its CCW walk at its boundary
    # halfedge — the clockwise-most edge of its fan — or the walk covers only part of the fan.
    # ``atomic_min`` makes both picks deterministic regardless of thread order.
    h = int(wp.tid())
    origin = faces[h]
    wp.atomic_min(out_interior_start, origin, h)
    if twins[h] == wp.int32(-1):
        wp.atomic_min(out_boundary_start, origin, h)


@wp.func
def select_ring_start(boundary_start: wp.int32, interior_start: wp.int32) -> wp.int32:
    # Prefer the boundary halfedge; ``INT32_MAX`` means "no candidate of this kind".
    if boundary_start != INT32_MAX_CONSTANT:
        return boundary_start
    if interior_start != INT32_MAX_CONSTANT:
        return interior_start
    return wp.int32(-1)


@wp.func
def has_boundary_halfedge(boundary_start: wp.int32) -> wp.bool:
    # A vertex lies on a boundary exactly when one of its outgoing halfedges has no twin.
    return boundary_start != INT32_MAX_CONSTANT


@wp.kernel
def write_one_rings(
    starts: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    out_halfedges: wp.array[wp.int32],
    out_incomplete: wp.array[wp.int32],
) -> None:
    # Rotate counter-clockwise through the outgoing halfedges at one vertex: ``h -> twin(prev(h))``
    # steps from edge ``v->a`` to edge ``v->b`` within the CCW-oriented face ``(v, a, b)``. The walk
    # is bounded by the vertex's known ring size, so a pinched (vertex-non-manifold) vertex — where
    # the rotation closes early on one of its fans — is reported instead of silently truncated.
    v = int(wp.tid())
    start = starts[v]
    if start == wp.int32(-1):
        return
    begin = offsets[v]
    degree = offsets[v + 1] - begin
    count = wp.int32(0)
    h = start
    while h != wp.int32(-1) and count < degree:
        out_halfedges[begin + count] = h
        count += wp.int32(1)
        nxt = twins[halfedge_prev(h)]
        if nxt == wp.int32(-1) or nxt == start:
            break
        h = nxt
    if count != degree:
        wp.atomic_add(out_incomplete, 0, 1)
