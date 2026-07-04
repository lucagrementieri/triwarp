import warp as wp

from triwarp.kernels.triangles import triangle_cross


@wp.kernel
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
        length_12 = wp.length_sq(v2 - v1)
        if length_12 < best:
            best = length_12
            a = i1
            b = i2
        length_20 = wp.length_sq(v0 - v2)
        if length_20 < best:
            a = i2
            b = i0
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
