import warp as wp

from triwarp.kernels.triangles import face_vertices


@wp.kernel
def average_onto_faces(
    faces: wp.array[wp.int32],
    vertex_values: wp.array[wp.float32],
    out_face_values: wp.array[wp.float32],
) -> None:
    f = int(wp.tid())
    x0, x1, x2 = face_vertices(vertex_values, faces, wp.int32(f))
    out_face_values[f] = (x0 + x1 + x2) / wp.float32(3.0)
