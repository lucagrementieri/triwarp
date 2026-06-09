"""Ray and point-in-mesh queries on ``wp.Mesh``."""

from __future__ import annotations

import warp as wp

import triwarp as tw
from triwarp.kernels import ray as kernel_ray
from triwarp.points import aabb_bounds


def _default_max_t(mesh: wp.Mesh, ray_origins: wp.array[wp.vec3]) -> float:
    mesh_min, mesh_max = aabb_bounds(mesh.points)
    ray_min, ray_max = aabb_bounds(ray_origins)
    combined_min = wp.vec3(min(mesh_min[0], ray_min[0]), min(mesh_min[1], ray_min[1]), min(mesh_min[2], ray_min[2]))
    combined_max = wp.vec3(max(mesh_max[0], ray_max[0]), max(mesh_max[1], ray_max[1]), max(mesh_max[2], ray_max[2]))
    return float(wp.length(combined_max - combined_min))


def _validate_ray_inputs(mesh: wp.Mesh, ray_origins: wp.array[wp.vec3], ray_directions: wp.array[wp.vec3]) -> None:
    if ray_origins.shape != ray_directions.shape:
        raise ValueError("Ray origin and direction don't match!")
    if mesh.device != ray_origins.device or mesh.device != ray_directions.device:
        devices = f"{mesh.device}, {ray_origins.device}, {ray_directions.device}"
        raise ValueError(f"mesh, ray_origins, and ray_directions must live on the same device, got {devices}")


def intersects_location(
    mesh: wp.Mesh, ray_origins: wp.array[wp.vec3], ray_directions: wp.array[wp.vec3], *, max_t: float | None = None
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]:
    """Return world-space locations where rays hit the mesh surface (first hit per ray).

    Uses ``wp.mesh_query_ray`` on the mesh BVH. Ray directions are unitized before
    querying. Returns only rays that hit within ``max_t`` as sparse ``(m,)`` arrays.
    Equivalent to compressing the dense output of :func:`intersects_first`.

    Parameters
    ----------
    mesh
        Triangle mesh with a built BVH (``wp.Mesh``).
    ray_origins
        ``(n,)`` ray origin positions as ``wp.vec3``.
    ray_directions
        ``(n,)`` ray direction vectors as ``wp.vec3`` (need not be unit length).
    max_t
        Optional maximum parametric distance along each normalized ray. When
        ``None``, derived from the combined mesh-and-origin AABB diagonal.

    Returns
    -------
    locations
        ``(m,)`` intersection points.
    index_ray
        ``(m,)`` index of the ray that produced each hit.
    index_tri
        ``(m,)`` face indices for each hit.
    """
    n = ray_origins.shape[0]
    device = ray_origins.device
    if n == 0:
        empty_int = wp.empty(0, dtype=wp.int32, device=device)
        return wp.empty(0, dtype=wp.vec3, device=device), empty_int, empty_int

    _validate_ray_inputs(mesh, ray_origins, ray_directions)
    if max_t is None:
        max_t = _default_max_t(mesh, ray_origins)

    faces_dense = wp.empty(n, dtype=wp.int32, device=device)
    locations_dense = wp.empty(n, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_ray.intersects_first_detail,
        dim=n,
        inputs=[mesh.id, ray_origins, ray_directions, wp.float32(max_t), faces_dense, locations_dense],
        device=device,
    )

    hit_mask = wp.empty(n, dtype=wp.bool, device=device)
    wp.launch(kernel_ray.face_hit_mask, dim=n, inputs=[faces_dense, hit_mask], device=device)
    index_ray = tw.array.flatnonzero(hit_mask)
    k = int(index_ray.shape[0])
    index_tri = wp.empty(k, dtype=wp.int32, device=device)
    wp.copy(index_tri, faces_dense[index_ray])
    locations = wp.empty(k, dtype=wp.vec3, device=device)
    wp.copy(locations, locations_dense[index_ray])
    return locations, index_ray, index_tri


def intersects_first(
    mesh: wp.Mesh, ray_origins: wp.array[wp.vec3], ray_directions: wp.array[wp.vec3], *, max_t: float | None = None
) -> wp.array[wp.int32]:
    """Find the index of the first triangle each ray hits.

    Uses ``wp.mesh_query_ray`` on the mesh BVH. Ray directions are unitized before
    querying. The search distance along each ray defaults to the diagonal of the
    axis-aligned bounding box enclosing mesh vertices and ray origins.

    Parameters
    ----------
    mesh
        Triangle mesh with a built BVH (``wp.Mesh``).
    ray_origins
        ``(n,)`` ray origin positions as ``wp.vec3``.
    ray_directions
        ``(n,)`` ray direction vectors as ``wp.vec3`` (need not be unit length).
    max_t
        Optional maximum parametric distance along each normalized ray. When
        ``None``, derived from the combined mesh-and-origin AABB diagonal.

    Returns
    -------
    wp.array[wp.int32]
        ``(n,)`` face indices; ``-1`` when a ray misses within ``max_t``.
    """
    n = ray_origins.shape[0]
    if n == 0:
        return wp.empty(0, dtype=wp.int32, device=ray_origins.device)
    _validate_ray_inputs(mesh, ray_origins, ray_directions)
    if max_t is None:
        max_t = _default_max_t(mesh, ray_origins)

    out_triangle_index = wp.empty(n, dtype=wp.int32, device=ray_origins.device)
    wp.launch(
        kernel_ray.intersects_first,
        dim=n,
        inputs=[mesh.id, ray_origins, ray_directions, wp.float32(max_t), out_triangle_index],
        device=ray_origins.device,
    )
    return out_triangle_index


def intersects_any(
    mesh: wp.Mesh, ray_origins: wp.array[wp.vec3], ray_directions: wp.array[wp.vec3], *, max_t: float | None = None
) -> wp.array[wp.bool]:
    """Check whether each ray hits the mesh surface.

    Uses ``wp.mesh_query_ray_anyhit`` on the mesh BVH. Ray directions are unitized
    before querying. The search distance along each ray defaults to the diagonal of
    the axis-aligned bounding box enclosing mesh vertices and ray origins.

    Parameters
    ----------
    mesh
        Triangle mesh with a built BVH (``wp.Mesh``).
    ray_origins
        ``(n,)`` ray origin positions as ``wp.vec3``.
    ray_directions
        ``(n,)`` ray direction vectors as ``wp.vec3`` (need not be unit length).
    max_t
        Optional maximum parametric distance along each normalized ray. When
        ``None``, derived from the combined mesh-and-origin AABB diagonal.

    Returns
    -------
    wp.array[wp.bool]
        ``(n,)`` hit flags; ``True`` when a ray hits within ``max_t``.
    """
    n = ray_origins.shape[0]
    if n == 0:
        return wp.empty(0, dtype=wp.bool, device=ray_origins.device)
    _validate_ray_inputs(mesh, ray_origins, ray_directions)
    if max_t is None:
        max_t = _default_max_t(mesh, ray_origins)

    out_hit = wp.empty(n, dtype=wp.bool, device=ray_origins.device)
    wp.launch(
        kernel_ray.intersects_any,
        dim=n,
        inputs=[mesh.id, ray_origins, ray_directions, wp.float32(max_t), out_hit],
        device=ray_origins.device,
    )
    return out_hit


def contains_points(
    mesh: wp.Mesh, points: wp.array[wp.vec3], *, n_sample: int = 5, perturbation_scale: float = 0.1
) -> wp.array[wp.bool]:
    """Test whether query points lie inside a closed mesh (ray parity sign).

    Uses ``wp.mesh_query_point_sign_parity`` on the mesh BVH. Points outside the
    mesh axis-aligned bounding box are rejected without a ray test. The closest-point
    search radius is the mesh AABB diagonal from :func:`triwarp.points.aabb_bounds`.
    Behavior for points on the surface is undefined, matching
    :func:`trimesh.ray.ray_util.contains_points`.

    Parameters
    ----------
    mesh
        Triangle mesh with a built BVH (``wp.Mesh``).
    points
        ``(n,)`` query positions as ``wp.vec3``.
    n_sample
        Number of perturbed rays for parity voting (higher is more robust).
    perturbation_scale
        Uniform perturbation scale applied to the base ray direction.

    Returns
    -------
    wp.array[wp.bool]
        ``(n,)`` flags; ``True`` when the point is classified as inside the mesh.
    """
    n = points.shape[0]
    if n == 0:
        return wp.empty(0, dtype=wp.bool, device=points.device)
    if mesh.device != points.device:
        raise ValueError(f"mesh and points must live on the same device, got {mesh.device} vs {points.device}")

    mesh_min, mesh_max = aabb_bounds(mesh.points)
    max_dist = wp.length(mesh_max - mesh_min)
    out_contains = wp.empty(n, dtype=wp.bool, device=points.device)
    wp.launch(
        kernel_ray.contains_points_sign_parity,
        dim=n,
        inputs=[
            mesh.id,
            points,
            wp.float32(max_dist),
            wp.int32(n_sample),
            wp.float32(perturbation_scale),
            mesh_min,
            mesh_max,
            out_contains,
        ],
        device=points.device,
    )
    return out_contains
