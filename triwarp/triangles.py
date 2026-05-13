import warp as wp
from triwarp.kernels import triangles as kernel_triangles
from typing import Literal


def face_normals_and_areas(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> tuple[wp.array[wp.vec3], wp.array[wp.float32]]:
    f = faces.shape[0] // 3
    out_normal = wp.empty(f, dtype=wp.vec3, device=vertices.device)
    out_area = wp.empty(f, dtype=wp.float32, device=vertices.device)
    wp.launch(
        kernel_triangles.face_normals_and_areas,
        dim=f,
        inputs=[vertices, faces, out_normal, out_area],
        device=vertices.device,
    )
    return out_normal, out_area


def face_angles(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> wp.array[wp.vec3]:
    f = faces.shape[0] // 3
    out_angle = wp.empty(f, dtype=wp.vec3, device=vertices.device)
    wp.launch(
        kernel_triangles.angles,
        dim=f,
        inputs=[vertices, faces, out_angle],
        device=vertices.device,
    )
    return out_angle


def centroid(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> wp.vec3:
    f = faces.shape[0] // 3
    if f == 0:
        return wp.vec3(float("nan"), float("nan"), float("nan"))
    device = vertices.device
    out_centroid = wp.zeros(3, dtype=wp.float32, device=device)
    out_total_area = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(
        kernel_triangles.centroid,
        dim=f,
        inputs=[vertices, faces, out_centroid, out_total_area],
        device=device,
    )
    centroid = out_centroid.numpy()
    total_area = out_total_area.numpy().item()
    return wp.vec3(float(centroid[0] / total_area), float(centroid[1] / total_area), float(centroid[2] / total_area))


def nondegenerate(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> wp.array[wp.bool]:
    f = faces.shape[0] // 3
    out_nondegenerate = wp.empty(f, dtype=wp.bool, device=vertices.device)
    wp.launch(
        kernel_triangles.nondegenerate,
        dim=f,
        inputs=[vertices, faces, out_nondegenerate],
        device=vertices.device,
    )
    return out_nondegenerate


def barycentric_to_points(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    barycentric: wp.array[wp.vec3],
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
    wp.launch(
        kernel,
        dim=f,
        inputs=[vertices, faces, points, out_barycentric],
        device=vertices.device,
    )
    return out_barycentric


def closest_point(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], points: wp.array[wp.vec3]
) -> wp.array[wp.vec3]:
    f = faces.shape[0] // 3
    out_closest = wp.empty(f, dtype=wp.vec3, device=vertices.device)
    wp.launch(
        kernel_triangles.closest_point,
        dim=f,
        inputs=[vertices, faces, points, out_closest],
        device=vertices.device,
    )
    return out_closest
