import warp as wp
from triwarp.kernels import triangles as kernel_triangles
from typing import Literal


def face_areas(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> wp.array[wp.float32]:
    f = faces.shape[0] // 3
    out_area = wp.empty(f, dtype=wp.float32, device=vertices.device)
    wp.launch(kernel_triangles.face_areas, dim=f, inputs=[vertices, faces, out_area], device=vertices.device)
    return out_area


def face_normals(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> wp.array[wp.vec3]:
    f = faces.shape[0] // 3
    out_normal = wp.empty(f, dtype=wp.vec3, device=vertices.device)
    wp.launch(kernel_triangles.face_normals, dim=f, inputs=[vertices, faces, out_normal], device=vertices.device)
    return out_normal


def angles(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> wp.array[wp.vec3]:
    f = faces.shape[0] // 3
    out_angle = wp.empty(f, dtype=wp.vec3, device=vertices.device)
    wp.launch(kernel_triangles.angles, dim=f, inputs=[vertices, faces, out_angle], device=vertices.device)
    return out_angle


def nondegenerate(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> wp.array[wp.bool]:
    f = faces.shape[0] // 3
    out_nondegenerate = wp.empty(f, dtype=wp.bool, device=vertices.device)
    wp.launch(
        kernel_triangles.nondegenerate, dim=f, inputs=[vertices, faces, out_nondegenerate], device=vertices.device
    )
    return out_nondegenerate


def barycentric_to_points(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], barycentric: wp.array[wp.vec3]
) -> wp.array[wp.vec3]:
    f = faces.shape[0] // 3
    out_points = wp.empty(f, dtype=wp.vec3, device=vertices.device)
    wp.launch(
        kernel_triangles.barycentric_to_points,
        dim=f,
        inputs=[vertices, faces, barycentric, out_points],
        device=vertices.device,
    )
    return out_points


def points_to_barycentric(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    points: wp.array[wp.vec3],
    method: Literal["cramer", "cross"] = "cramer",
) -> wp.array[wp.vec3]:
    f = faces.shape[0] // 3
    out_barycentric = wp.empty(f, dtype=wp.vec3, device=vertices.device)
    kernel = (
        kernel_triangles.points_to_barycentric_cramer
        if method == "cramer"
        else kernel_triangles.points_to_barycentric_cross
    )
    wp.launch(kernel, dim=f, inputs=[vertices, faces, points, out_barycentric], device=vertices.device)
    return out_barycentric
