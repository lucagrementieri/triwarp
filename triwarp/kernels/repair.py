import warp as wp

from triwarp.kernels.array import update_argmin_pair
from triwarp.kernels.triangles import triangle_cross


@wp.func
def cyclic_match(
    a0: wp.int32, a1: wp.int32, a2: wp.int32, b0: wp.int32, b1: wp.int32, b2: wp.int32
) -> wp.bool:
    """Whether ``(a0, a1, a2)`` is a cyclic rotation of ``(b0, b1, b2)`` (same orientation)."""
    return (
        (a0 == b0 and a1 == b1 and a2 == b2)
        or (a0 == b1 and a1 == b2 and a2 == b0)
        or (a0 == b2 and a1 == b0 and a2 == b1)
    )


@wp.kernel
def scatter_duplicate_face_stats(
    faces: wp.array[wp.int32],
    unique_faces: wp.array[wp.int32],
    inverse: wp.array[wp.int32],
    out_member_count: wp.array[wp.int32],
    out_signed_count: wp.array[wp.int32],
    out_first_member: wp.array[wp.int32],
    out_first_positive: wp.array[wp.int32],
    out_first_negative: wp.array[wp.int32],
) -> None:
    # Per input face: orientation sign against its group representative (the first-occurring
    # face, gathered via ``inverse``) scattered into per-group stats. ``first_*`` slots hold the
    # smallest face index of each sign class (seeded with the n_faces sentinel).
    f = int(wp.tid())
    ui = inverse[f]
    consistent = cyclic_match(
        faces[f * 3],
        faces[f * 3 + 1],
        faces[f * 3 + 2],
        unique_faces[ui * 3],
        unique_faces[ui * 3 + 1],
        unique_faces[ui * 3 + 2],
    )
    wp.atomic_add(out_member_count, ui, wp.int32(1))
    wp.atomic_min(out_first_member, ui, wp.int32(f))
    if consistent:
        wp.atomic_add(out_signed_count, ui, wp.int32(1))
        wp.atomic_min(out_first_positive, ui, wp.int32(f))
    else:
        wp.atomic_add(out_signed_count, ui, wp.int32(-1))
        wp.atomic_min(out_first_negative, ui, wp.int32(f))


@wp.kernel
def resolve_duplicate_groups(
    member_count: wp.array[wp.int32],
    signed_count: wp.array[wp.int32],
    first_member: wp.array[wp.int32],
    first_positive: wp.array[wp.int32],
    first_negative: wp.array[wp.int32],
    out_keep: wp.array[wp.int32],
    out_error_group: wp.array[wp.int32],
) -> None:
    # Keep-decision per duplicate group (igl::resolve_duplicated_faces): singletons stay; a net
    # +1/-1 orientation keeps the first member of the majority sign; a cancelling group drops;
    # anything else is non-orientable (the smallest offending group index is reported).
    ui = int(wp.tid())
    count = signed_count[ui]
    if member_count[ui] == 1:
        out_keep[ui] = first_member[ui]
    elif count == 1:
        out_keep[ui] = first_positive[ui]
    elif count == -1:
        out_keep[ui] = first_negative[ui]
    else:
        out_keep[ui] = wp.int32(-1)
        if count != 0:
            wp.atomic_min(out_error_group, 0, wp.int32(ui))


@wp.kernel(enable_backward=False)
def small_triangle_collapse_edges(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    min_dbl_area: wp.float32,
    out_pairs: wp.array2d[wp.int32],
    out_flag: wp.array[wp.int32],
) -> None:
    """Flag faces with double-area below ``min_dbl_area`` and emit their shortest edge (libigl)."""
    f = int(wp.tid())
    face = faces[f * 3 : (f + 1) * 3]
    i0 = face[0]
    i1 = face[1]
    i2 = face[2]
    dbl_area = wp.length(triangle_cross(vertices, face))
    if dbl_area < min_dbl_area:
        v0 = vertices[i0]
        v1 = vertices[i1]
        v2 = vertices[i2]
        # Shortest of the three edges (0,1), (1,2), (2,0); collapse its endpoints together.
        best = wp.length_sq(v1 - v0)
        a = i0
        b = i1
        update_argmin_pair(best, a, b, wp.length_sq(v2 - v1), i1, i2)
        update_argmin_pair(best, a, b, wp.length_sq(v0 - v2), i2, i0)
        out_pairs[f, 0] = a
        out_pairs[f, 1] = b
        out_flag[f] = wp.int32(1)
    else:
        out_pairs[f, 0] = i0
        out_pairs[f, 1] = i0
        out_flag[f] = wp.int32(0)


@wp.kernel
def flip_faces_masked(
    faces: wp.array[wp.int32], flip: wp.array[wp.int32], out_faces: wp.array[wp.int32]
) -> None:
    """Copy ``faces`` to ``out_faces``, reversing winding (swap corners 1,2) where ``flip > 0``."""
    f = int(wp.tid())
    base = f * wp.int32(3)
    i0 = faces[base]
    i1 = faces[base + wp.int32(1)]
    i2 = faces[base + wp.int32(2)]
    out_faces[base] = i0
    if flip[f] > wp.int32(0):
        out_faces[base + wp.int32(1)] = i2
        out_faces[base + wp.int32(2)] = i1
    else:
        out_faces[base + wp.int32(1)] = i1
        out_faces[base + wp.int32(2)] = i2


@wp.func
def negative_volume_flag(volume: wp.float32) -> wp.int32:
    """Flag a face for flipping when its component's signed volume is negative (inward)."""
    return wp.where(volume < wp.float32(0.0), wp.int32(1), wp.int32(0))
