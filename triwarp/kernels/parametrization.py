import warp as wp


@wp.kernel
def flipped_faces_mask(
    vertices: wp.array[wp.vec2],
    faces: wp.array[wp.int32],
    out_mask: wp.array[wp.bool],
) -> None:
    fi = int(wp.tid())
    v0 = vertices[faces[fi * 3]]
    e0 = vertices[faces[fi * 3 + 1]] - v0
    e1 = vertices[faces[fi * 3 + 2]] - v0
    # 2D signed area * 2 == det of libigl's homogeneous 3x3 matrix
    signed_area2 = e0[0] * e1[1] - e0[1] * e1[0]
    out_mask[fi] = signed_area2 < 0.0
