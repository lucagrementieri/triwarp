import warp as wp

from triwarp.constants import TOLERANCE_MERGE_CONSTANT


@wp.kernel
def face_adjacency_projections(
    vertices: wp.array[wp.vec3],
    face_normals: wp.array[wp.vec3],
    face_adjacency: wp.array2d[wp.int32],
    face_adjacency_edges: wp.array2d[wp.int32],
    face_adjacency_unshared: wp.array2d[wp.int32],
    out_projections: wp.array[wp.float32],
) -> None:
    tid = int(wp.tid())
    normal = face_normals[face_adjacency[tid, 0]]
    origin = vertices[face_adjacency_edges[tid, 0]]
    vid_other = face_adjacency_unshared[tid, 1]
    vector_other = vertices[vid_other] - origin
    out_projections[tid] = wp.dot(vector_other, normal)


@wp.kernel
def face_adjacency_convex(
    projections: wp.array[wp.float32],
    out_convex: wp.array[wp.bool],
) -> None:
    tid = int(wp.tid())
    out_convex[tid] = projections[tid] < TOLERANCE_MERGE_CONSTANT
