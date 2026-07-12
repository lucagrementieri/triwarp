"""ECL-CC connected components (init + single-pass CAS hook + pointer-jumping flatten)."""

import warp as wp


@wp.func
def find_representative(parents: wp.array[wp.int32], v: wp.int32) -> wp.int32:
    parent = wp.int32(parents[v])
    if parent != v:
        child = v
        grandparent = wp.int32(parents[parent])
        while parent > grandparent:
            parents[child] = grandparent
            child = parent
            parent = grandparent
            grandparent = wp.int32(parents[parent])
    return parent


@wp.func
def ecl_hook_edge(parents: wp.array[wp.int32], rep_v: wp.int32, u: wp.int32) -> wp.int32:
    # ECL-CC hook: union the trees holding ``rep_v`` and ``u``, always pointing the larger root
    # at the smaller. The retry loop is unbounded but guaranteed to terminate: it continues from
    # the CAS return value (never re-running the find), and every failed CAS hands back a strictly
    # smaller root — parents are monotonically non-increasing — so max(rep_v, rep_u) strictly
    # decreases and is bounded below by 0. Returns the (possibly updated) representative of the
    # owning vertex so the caller can carry it across that vertex's remaining edges.
    rep_u = find_representative(parents, u)
    repeat = wp.bool(True)
    while repeat:
        repeat = wp.bool(False)
        if rep_v != rep_u:
            old = wp.int32(0)
            if rep_v < rep_u:
                old = wp.atomic_cas(parents, rep_u, rep_u, rep_v)
                if old != rep_u:
                    rep_u = old
                    repeat = wp.bool(True)
            else:
                old = wp.atomic_cas(parents, rep_v, rep_v, rep_u)
                if old != rep_v:
                    rep_v = old
                    repeat = wp.bool(True)
    return rep_v


@wp.kernel
def ecl_init_parent(
    offsets: wp.array[wp.int32], indices: wp.array[wp.int32], out_parents: wp.array[wp.int32]
) -> None:
    v = int(wp.tid())
    out_parents[v] = v
    start = offsets[v]
    end = offsets[v + 1]
    for j in range(start, end):
        u = wp.int32(indices[j])
        if u < v:
            out_parents[v] = u
            break


@wp.kernel
def ecl_hook(
    offsets: wp.array[wp.int32], indices: wp.array[wp.int32], parents: wp.array[wp.int32]
) -> None:
    # One thread per vertex; the larger-id endpoint owns each undirected edge, so every edge is
    # hooked exactly once. Fine for bounded-degree mesh graphs (a reversed star — hub id n-1 —
    # would serialize on the hub thread, but mesh degrees are ~6). ``rep_v`` is hoisted across
    # the row, ECL-CC's ``vstat`` carry.
    v = wp.int32(int(wp.tid()))
    start = offsets[v]
    end = offsets[v + 1]
    rep_v = find_representative(parents, v)
    for j in range(start, end):
        u = indices[j]
        if v > u:
            rep_v = ecl_hook_edge(parents, rep_v, u)


@wp.kernel
def ecl_flatten(parents: wp.array[wp.int32], out_labels: wp.array[wp.int32]) -> None:
    v = wp.int32(int(wp.tid()))
    out_labels[v] = find_representative(parents, v)
