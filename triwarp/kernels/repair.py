import warp as wp


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


@wp.kernel
def accumulate_component_volume(
    labels: wp.array[wp.int32], volumes: wp.array[wp.float32], out_accum: wp.array[wp.float32]
) -> None:
    """Atomically add each face's signed volume into its connected component's accumulator."""
    f = int(wp.tid())
    wp.atomic_add(out_accum, labels[f], volumes[f])


@wp.kernel
def mark_negative_component(
    labels: wp.array[wp.int32], component_volume: wp.array[wp.float32], out_flip: wp.array[wp.int32]
) -> None:
    """Flag a face for flipping when its component's signed volume is negative (inward)."""
    f = int(wp.tid())
    if component_volume[labels[f]] < 0.0:
        out_flip[f] = wp.int32(1)
    else:
        out_flip[f] = wp.int32(0)
