import warp as wp


@wp.kernel
def average_onto_faces(
    faces: wp.array[wp.int32],
    vertex_values: wp.array[wp.float32],
    out_face_values: wp.array[wp.float32],
) -> None:
    f = int(wp.tid())
    v0 = faces[f * 3 + 0]
    v1 = faces[f * 3 + 1]
    v2 = faces[f * 3 + 2]
    out_face_values[f] = (vertex_values[v0] + vertex_values[v1] + vertex_values[v2]) / wp.float32(
        3.0
    )


