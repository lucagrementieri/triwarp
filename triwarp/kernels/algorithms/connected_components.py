"""
ECL-CC connected components (init + single-pass CAS hook + pointer-jumping flatten).

Two variants over the same union-find core:

- the plain one (``ecl_init_parent`` / ``ecl_hook`` / ``ecl_flatten``) labels components from a CSR
  adjacency, one thread per node with the row's representative hoisted across it;
- the **parity** one (``ecl_init_parent_parity`` / ``ecl_hook_parity`` / ``ecl_flatten_parity``)
  carries a Z2 potential alongside the labelling, from a signed edge list, one thread per edge.
  See ``find_representative_parity`` for the packing that makes it work.
"""

import warp as wp


@wp.func
def find_representative(parents: wp.array[wp.int32], v: wp.int32) -> wp.int32:
    parent = parents[v]
    if parent != v:
        child = v
        grandparent = parents[parent]
        while parent > grandparent:
            parents[child] = grandparent
            child = parent
            parent = grandparent
            grandparent = parents[parent]
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
    v = wp.int32(wp.tid())
    out_parents[v] = v
    start = offsets[v]
    end = offsets[v + 1]
    for j in range(start, end):
        u = indices[j]
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
    v = wp.int32(wp.int32(wp.tid()))
    start = offsets[v]
    end = offsets[v + 1]
    rep_v = find_representative(parents, v)
    for j in range(start, end):
        u = indices[j]
        if v > u:
            rep_v = ecl_hook_edge(parents, rep_v, u)


@wp.kernel
def ecl_flatten(parents: wp.array[wp.int32], out_labels: wp.array[wp.int32]) -> None:
    v = wp.int32(wp.int32(wp.tid()))
    out_labels[v] = find_representative(parents, v)


@wp.func
def find_representative_parity(words: wp.array[wp.int32], v: wp.int32) -> tuple[wp.int32, wp.int32]:
    # Path-halving find that also accumulates a Z2 potential. ``words[x]`` packs both halves of the
    # union-find edge into ONE int32 — ``(parent << 1) | parity(x -> parent)`` — which is the whole
    # trick: a single 32-bit load or store moves parent and parity together, so the plain variant's
    # ``wp.atomic_cas`` hook still applies and its termination proof (parents monotonically
    # non-increasing; see ``ecl_hook_edge``) carries over unchanged, because the packed word is
    # monotone in the parent. Two separate arrays could not be CAS'd atomically.
    #
    # Returns ``(root, parity(v -> root))``. The compression store is the same benign race as in
    # ``find_representative``: ``grandparent`` stays an ancestor of ``child`` however the tree moves
    # under us, and on a consistent component the parity to a given ancestor is invariant, so the
    # written word is always valid — only possibly less compressed than it could be.
    word = words[v]
    parent = wp.int32(word >> 1)
    parity = wp.int32(word & 1)
    accumulated = wp.int32(0)
    child = v
    if parent != v:
        grandword = words[parent]
        grandparent = wp.int32(grandword >> 1)
        grandparity = wp.int32(grandword & 1)
        while parent > grandparent:
            words[child] = (grandparent << 1) | (parity ^ grandparity)
            accumulated = accumulated ^ parity
            child = parent
            parent = grandparent
            parity = grandparity
            grandword = words[parent]
            grandparent = wp.int32(grandword >> 1)
            grandparity = wp.int32(grandword & 1)
    return parent, accumulated ^ parity


@wp.func
def ecl_hook_edge_parity(
    words: wp.array[wp.int32], rep_v: wp.int32, parity_v: wp.int32, u: wp.int32, edge_sign: wp.int32
) -> tuple[wp.int32, wp.int32]:
    # Parity-carrying mirror of ``ecl_hook_edge``: union the trees holding ``rep_v`` and ``u``,
    # larger root at the smaller, with the offset that keeps the potential consistent. Writing
    # ``p(x)`` for the parity from ``x`` to its root, the union must satisfy
    # ``p(rep_u) ^ p(rep_v) == parity_v ^ edge_sign ^ parity_u``, which is ``offset`` below.
    # Termination is the plain variant's argument verbatim — every failed CAS hands back a strictly
    # smaller root, so ``max(root_v, root_u)`` strictly decreases and is bounded below by 0.
    #
    # A component with contradictory signs (a Mobius band, in the orientation application) simply
    # takes whichever branch reached its roots first: the potential does not exist, so no assignment
    # is correct and callers detect the contradiction afterwards by re-testing the edges.
    root_v = rep_v
    par_v = parity_v
    root_u = wp.int32(0)
    par_u = wp.int32(0)
    root_u, par_u = find_representative_parity(words, u)
    repeat = wp.bool(True)
    while repeat:
        repeat = wp.bool(False)
        if root_v != root_u:
            offset = par_v ^ edge_sign ^ par_u
            if root_v < root_u:
                expected = root_u << 1
                old = wp.atomic_cas(words, root_u, expected, (root_v << 1) | offset)
                if old != expected:
                    root_u = old >> 1
                    par_u = par_u ^ (old & 1)
                    repeat = wp.bool(True)
            else:
                expected = root_v << 1
                old = wp.atomic_cas(words, root_v, expected, (root_u << 1) | offset)
                if old != expected:
                    root_v = old >> 1
                    par_v = par_v ^ (old & 1)
                    repeat = wp.bool(True)
    return root_v, par_v


@wp.kernel
def ecl_init_parent_parity(out_words: wp.array[wp.int32]) -> None:
    # Every node its own root at parity 0. The plain ``ecl_init_parent`` also pre-hooks each node
    # to its smallest neighbour, which needs the edge's sign and so would need a signed CSR; the
    # hook kernel below is edge-parallel and does the same work in one pass anyway.
    v = wp.int32(wp.tid())
    out_words[v] = v << 1


@wp.kernel
def ecl_hook_parity(
    edges: wp.array2d[wp.int32], signs: wp.array[wp.int32], words: wp.array[wp.int32]
) -> None:
    # One thread per signed edge; each thread's retry loop only exits once its two endpoints share
    # a tree, so after this single launch the forest spans every edge — no host convergence loop.
    e = wp.int32(wp.tid())
    a = edges[e, 0]
    b = edges[e, 1]
    if a == b:
        return
    rep_a = wp.int32(0)
    par_a = wp.int32(0)
    rep_a, par_a = find_representative_parity(words, a)
    ecl_hook_edge_parity(words, rep_a, par_a, b, signs[e])


@wp.kernel
def ecl_flatten_parity(
    words: wp.array[wp.int32], out_labels: wp.array[wp.int32], out_parity: wp.array[wp.int32]
) -> None:
    v = wp.int32(wp.int32(wp.tid()))
    root = wp.int32(0)
    parity = wp.int32(0)
    root, parity = find_representative_parity(words, v)
    out_labels[v] = root
    out_parity[v] = parity
