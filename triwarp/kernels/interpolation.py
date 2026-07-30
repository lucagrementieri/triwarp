from typing import Any

import warp as wp

from triwarp.kernels.triangles import face_vertices, point_barycentric_cramer


@wp.kernel
def average_onto_faces(
    faces: wp.array[wp.int32],
    vertex_values: wp.array[wp.float32],
    out_face_values: wp.array[wp.float32],
) -> None:
    f = int(wp.tid())
    x0, x1, x2 = face_vertices(vertex_values, faces, wp.int32(f))
    out_face_values[f] = (x0 + x1 + x2) / wp.float32(3.0)


@wp.kernel
def transfer_onto_vertices(
    source_vertices: wp.array[wp.vec3],
    source_faces: wp.array[wp.int32],
    source_values: wp.array[Any],
    closest: wp.array[wp.vec3],
    face_id: wp.array[wp.int32],
    out_values: wp.array[Any],
    out_distance: wp.array[wp.float32],
) -> None:
    # Barycentric resample of a source per-vertex field at each target vertex's closest point.
    #
    # ``face_id < 0`` is the ``max_dist`` miss case. The value slot keeps whatever the wrapper
    # pre-filled (so a caller who narrowed the search still gets a deterministic buffer), and the
    # distance is promoted to ``inf``: ``wp.mesh_query_point_no_sign`` reports ``max_dist`` there,
    # which is indistinguishable from a genuine hit at exactly that range.
    i = int(wp.tid())
    f = face_id[i]
    if f < 0:
        out_distance[i] = wp.float32(wp.INF)
        return
    v0, v1, v2 = face_vertices(source_vertices, source_faces, f)
    bary = point_barycentric_cramer(v0, v1, v2, closest[i])
    a0, a1, a2 = face_vertices(source_values, source_faces, f)
    out_values[i] = a0 * bary[0] + a1 * bary[1] + a2 * bary[2]
