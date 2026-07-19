import warp as wp


@wp.kernel
def dilate_vertex_mask(
    unique_edges: wp.array2d[wp.int32], in_mask: wp.array[wp.bool], out_mask: wp.array[wp.bool]
) -> None:
    # One edge dilation round: a vertex joins the mask if either endpoint of an incident edge is
    # already in it. ``out_mask`` must be pre-seeded with ``in_mask`` (this only adds neighbors).
    i = int(wp.tid())
    a = unique_edges[i, 0]
    b = unique_edges[i, 1]
    if in_mask[a]:
        out_mask[b] = wp.bool(True)
    if in_mask[b]:
        out_mask[a] = wp.bool(True)


@wp.kernel
def edge_region_counts(
    inverse: wp.array[wp.int32],
    face_mask: wp.array[wp.bool],
    out_count: wp.array[wp.int32],
    out_region_count: wp.array[wp.int32],
) -> None:
    # Per unique edge: total incident-face count and how many of those faces are in the region.
    i = int(wp.tid())
    e = inverse[i]
    wp.atomic_add(out_count, e, wp.int32(1))
    if face_mask[i // 3]:
        wp.atomic_add(out_region_count, e, wp.int32(1))


@wp.func
def region_boundary_flag(count: wp.int32, region_count: wp.int32) -> wp.bool:
    # Interior region-boundary edge: exactly two incident faces, exactly one in the region.
    return count == 2 and region_count == 1


@wp.func
def keep_selected(selected: wp.bool, keep_flag: wp.int32) -> wp.bool:
    # Stay selected only if the vertex was selected and its component is not fully selected.
    return selected and keep_flag != 0


@wp.kernel
def keep_component_scatter(
    mask: wp.array[wp.bool], labels: wp.array[wp.int32], out_keep: wp.array[wp.int32]
) -> None:
    # A component is "kept" (not fully selected) if it has at least one unselected vertex.
    v = int(wp.tid())
    if not mask[v]:
        out_keep[labels[v]] = wp.int32(1)
