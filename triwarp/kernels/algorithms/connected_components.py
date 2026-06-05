"""ECL-lite connected components (ECL-CC init + CAS hook + intermediate pointer jumping)."""

import warp as wp

# TODO: validate experimentally a good value for this and also for the max hook passes
ECL_HOOK_MAX_RETRY = wp.constant(wp.int32(32))


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
def ecl_hook_edge(
    parents: wp.array[wp.int32],
    u: wp.int32,
    v: wp.int32,
    out_changed: wp.array[wp.int32],
    out_incomplete: wp.array[wp.int32],
) -> None:
    parent_u = find_representative(parents, u)
    for _ in range(ECL_HOOK_MAX_RETRY):
        retry = False
        parent_v = find_representative(parents, v)
        if parent_v != parent_u:
            wp.atomic_add(out_changed, 0, wp.int32(1))
            old_parent = wp.int32(0)
            if parent_v < parent_u:
                old_parent = wp.atomic_cas(parents, parent_u, parent_u, parent_v)
                if old_parent != parent_u:
                    parent_u = old_parent
                    retry = True
            else:
                old_parent = wp.atomic_cas(parents, parent_v, parent_v, parent_u)
                if old_parent != parent_v:
                    parent_v = old_parent
                    retry = True
        if not retry:
            return
    wp.atomic_add(out_incomplete, 0, wp.int32(1))


@wp.kernel
def ecl_init_parent(
    offsets: wp.array[wp.int32],
    indices: wp.array[wp.int32],
    parents: wp.array[wp.int32],
) -> None:
    v = int(wp.tid())
    parents[v] = v
    start = offsets[v]
    end = offsets[v + 1]
    for j in range(start, end):
        u = wp.int32(indices[j])
        if u < v:
            parents[v] = u
            break


@wp.kernel
def ecl_hook(
    offsets: wp.array[wp.int32],
    indices: wp.array[wp.int32],
    parents: wp.array[wp.int32],
    out_changed: wp.array[wp.int32],
    out_incomplete: wp.array[wp.int32],
) -> None:
    v = wp.int32(int(wp.tid()))
    start = offsets[v]
    end = offsets[v + 1]
    for j in range(start, end):
        u = indices[j]
        if v > u:
            ecl_hook_edge(parents, u, v, out_changed, out_incomplete)


@wp.kernel
def ecl_finalize_and_verify(
    offsets: wp.array[wp.int32],
    indices: wp.array[wp.int32],
    parents: wp.array[wp.int32],
    out_labels: wp.array[wp.int32],
    out_violations: wp.array[wp.int32],
) -> None:
    v = wp.int32(int(wp.tid()))
    label_v = find_representative(parents, v)
    out_labels[v] = label_v
    start = offsets[v]
    end = offsets[v + 1]
    for j in range(start, end):
        u = indices[j]
        if v > u:
            u_rep = find_representative(parents, u)
            if u_rep != label_v:
                wp.atomic_add(out_violations, 0, wp.int32(1))
