"""
Kernels for the halfedge structure (``triwarp.halfedge``).

Every function here is index arithmetic under the ``h = 3 * f + k`` convention: halfedge ``h``
belongs to face ``h // 3``, occupies corner ``h % 3`` of it, and runs
``faces[h] -> faces[3 * f + (k + 1) % 3]``. There is no stored ``next`` or ``prev`` pointer and no
face table to consult, which is why these are ``@wp.func``s over an ``int32`` rather than lookups.

**The decomposition itself is deliberately not wrapped.** ``h // 3`` for the face appears at ~20
sites across the tree and stays spelled that way: it is shorter than a call, the convention is
stated here and restated at the sites that need it (``adjacency.py`` puts it in one clause -- *"the
face of edge ``e`` is just ``e // 3``"*), and a ``halfedge_face(h)`` helper would add a name without
adding information. What *is* worth reaching for is anything with a branch in it --
``halfedge_next`` and ``halfedge_prev`` below -- because those get re-derived by hand instead:
``kernels/repair.py`` carried ``(forward // 3) * 3 + (forward + 1) % 3`` twice, in a file that
already imported from here.
"""

import warp as wp

from triwarp.constants import INT32_MAX_CONSTANT


@wp.func
def halfedge_next(h: wp.int32) -> wp.int32:
    # Next halfedge inside the same face, under the ``h = 3 * f + k`` convention of
    # ``triwarp.halfedge``: index arithmetic, no structure to look up.
    return h - h % 3 + (h + 1) % 3


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
    faces: wp.array[wp.int32],
    sorted_keys: wp.array[wp.uint64],
    order: wp.array[wp.int32],
    out_twins: wp.array[wp.int32],
    out_defect_counts: wp.array[wp.int32],
) -> None:
    # One thread per position in the hash-sorted halfedge list; only the first position of each run
    # of equal keys acts, so each undirected edge is resolved exactly once. Run length 1 is a
    # boundary edge (twin stays -1), 2 an interior edge, 3+ a non-manifold edge.
    #
    # ``out_defect_counts`` carries the two rejections in one buffer so the wrapper needs one
    # readback: slot 0 counts edge-non-manifold edges, slot 1 counts edges whose two halfedges run
    # the *same* way.
    #
    # **The direction test is what makes the twin contract true rather than merely plausible.** The
    # sort key is built from the *sorted* endpoint pair, so it identifies the undirected edge and
    # says nothing about which way either halfedge crosses it -- and on a mesh that is not
    # consistently wound, the two halfedges of an edge can both run ``a -> b``. Pairing those still
    # satisfies ``twins[twins[h]] == h``, so nothing downstream notices; what breaks is the
    # *orientation* half of the contract ``halfedge_twins`` documents, and with it the CCW rotation
    # ``h -> twins[prev(h)]`` that ``write_one_rings`` walks. On a closed, edge- and
    # vertex-manifold but *non-orientable* mesh that leaves ``vertex_one_rings`` succeeding while
    # returning ring entries whose halfedge does not originate at the owning vertex, and halfedges
    # appearing in two rings at once -- so ``halfedge_tangent_angles`` sums corner angles belonging
    # to other vertices and races two threads onto one slot, or the walk closes early and the
    # wrapper reports a "pinch point" on a mesh that has none.
    #
    # Two halfedges of one undirected edge run the same way exactly when they share an origin, so
    # the test is one gather and no geometry. Two further inputs it rejects, both correctly: an
    # exactly *duplicated* face, whose three edges each carry two halfedges pointing the same way
    # (a reversed duplicate is a consistently wound degenerate surface and is still accepted); and
    # a face with a repeated vertex, whose self-edge ``a -> a`` has no opposite direction to find.
    # The wrapper's message says "the same direction", which is true of all three.
    i = wp.int32(wp.tid())
    key = sorted_keys[i]
    if i > 0 and sorted_keys[i - 1] == key:
        return
    n = sorted_keys.shape[0]
    if i + 1 >= n or sorted_keys[i + 1] != key:
        return
    if i + 2 < n and sorted_keys[i + 2] == key:
        wp.atomic_add(out_defect_counts, 0, 1)
        return
    h0 = order[i]
    h1 = order[i + 1]
    if faces[h0] == faces[h1]:
        wp.atomic_add(out_defect_counts, 1, 1)
        return
    out_twins[h0] = h1
    out_twins[h1] = h0


@wp.kernel
def count_mispaired_twins(
    faces: wp.array[wp.int32], twins: wp.array[wp.int32], out_mispaired: wp.array[wp.int32]
) -> None:
    # One thread per halfedge, counting the entries of a *caller-supplied* twin table that are not
    # the opposite halfedge. ``halfedge_twins`` cannot produce one -- ``pair_sorted_halfedges``
    # makes the property hold by construction -- so this exists for the ``twins=`` keyword, which
    # three public modules accept and none of them can trust.
    #
    # The three tests together are the whole contract ``halfedge_twins`` documents, and none of
    # them implies the others: ``twins[twins[h]] == h`` alone permits a pair that spans two
    # different edges, and the endpoint test alone permits a third halfedge on the same edge
    # claiming one of them back. A boundary entry (``-1``) is legal and skipped.
    #
    # Written as separate ``if``s rather than one ``or`` chain because the range test has to run
    # *before* the gathers it guards, and kernel-scope ``or`` is not a short-circuit this code
    # should depend on: an out-of-range twin would otherwise index the face buffer, which on the
    # CPU device is a host-heap read rather than a fault.
    #
    # That range test is against ``twins.shape[0]`` and **not** ``faces.shape[0]``, which are not
    # the same number: the wrapper defines the halfedge count as ``faces.shape[0] // 3 * 3``, so a
    # ragged face buffer leaves up to two trailing entries that are not halfedges. A twin of
    # ``faces.shape[0] - 1`` there would pass a check against the face length and then send
    # ``halfedge_destination`` one past the end -- the very read this guard exists to stop.
    #
    # It is cheap, which is what the wrapper claims: a small fraction of the ``halfedge_twins``
    # rebuild it lets the caller skip, flat in the halfedge count because both are launch-bound.
    h = wp.int32(wp.tid())
    twin = twins[h]
    if twin < wp.int32(0):
        return
    if twin >= twins.shape[0]:
        wp.atomic_add(out_mispaired, 0, 1)
        return
    if twins[twin] != h:
        wp.atomic_add(out_mispaired, 0, 1)
        return
    if faces[twin] != halfedge_destination(faces, h):
        wp.atomic_add(out_mispaired, 0, 1)
        return
    if faces[h] != halfedge_destination(faces, twin):
        wp.atomic_add(out_mispaired, 0, 1)


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
    h = wp.int32(wp.tid())
    origin = faces[h]
    wp.atomic_min(out_interior_start, origin, h)
    if twins[h] == wp.int32(-1):
        wp.atomic_min(out_boundary_start, origin, h)


@wp.func
def ring_start_and_boundary(
    boundary_start: wp.int32, interior_start: wp.int32
) -> tuple[wp.int32, wp.bool]:
    # Both answers the ring walk needs about a vertex, from the one pair of candidates: where to
    # start, and whether the vertex is on a boundary. They are two reads of ``boundary_start``
    # and one comparison apart, so asking for them separately costs a second pass over the same
    # buffer to re-derive a test the first pass already made.
    #
    # Start: prefer the boundary halfedge; ``INT32_MAX`` means "no candidate of this kind".
    # Boundary: a vertex lies on one exactly when an outgoing halfedge has no twin.
    start = wp.int32(-1)
    if boundary_start != INT32_MAX_CONSTANT:
        start = boundary_start
    elif interior_start != INT32_MAX_CONSTANT:
        start = interior_start
    return start, boundary_start != INT32_MAX_CONSTANT


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
    v = wp.int32(wp.tid())
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
