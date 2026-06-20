import warp as wp

from triwarp.kernels import points as kernel_points


def point_plane_distance(
    points: wp.array[wp.vec3], plane_normal: wp.vec3, plane_origin: wp.vec3 = None
) -> wp.array[wp.float32]:
    if plane_origin is None:
        plane_origin = wp.vec3(0.0, 0.0, 0.0)
    n = int(points.shape[0])
    out_distances = wp.empty(n, dtype=wp.float32, device=points.device)
    wp.launch(
        kernel_points.point_plane_distance,
        dim=n,
        inputs=[points, plane_normal, plane_origin, out_distances],
        device=points.device,
    )
    return out_distances
