"""Ray and point-in-mesh queries on ``wp.Mesh``."""

from __future__ import annotations

import warp as wp

import triwarp as tw
from triwarp.bounds import aabb_bounds
from triwarp.constants import TOLERANCE_PLANAR
from triwarp.kernels import proximity as kernel_proximity
from triwarp.kernels import ray as kernel_ray
from triwarp.proximity import _default_mesh_query_max_dist as default_mesh_query_max_dist


def _validate_ray_inputs(
    mesh: wp.Mesh, ray_origins: wp.array[wp.vec3], ray_directions: wp.array[wp.vec3]
) -> None:
    if ray_origins.shape != ray_directions.shape:
        raise ValueError("Ray origin and direction don't match!")


def intersects_location(
    mesh: wp.Mesh,
    ray_origins: wp.array[wp.vec3],
    ray_directions: wp.array[wp.vec3],
    *,
    max_t: float | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Return world-space locations where rays hit the mesh surface (first hit per ray).

    Uses ``wp.mesh_query_ray`` on the mesh BVH. Ray directions are unitized before
    querying. Returns only rays that hit within ``max_t`` as sparse ``(m,)`` arrays.
    Equivalent to compressing the dense output of
    [`intersects_first`][triwarp.ray.intersects_first].

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
        max_t = default_mesh_query_max_dist(mesh.points, ray_origins)

    index_ray = wp.empty(n, dtype=wp.int32, device=device)
    index_tri = wp.empty(n, dtype=wp.int32, device=device)
    locations = wp.empty(n, dtype=wp.vec3, device=device)
    counter = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        kernel_ray.first_hit_append,
        dim=n,
        inputs=[
            wp.uint64(mesh.id),
            ray_origins,
            ray_directions,
            wp.float32(max_t),
            index_ray,
            index_tri,
            locations,
            counter,
        ],
        device=device,
    )
    _, (locations, index_ray, index_tri) = tw.array.trim_to_count(
        counter, locations, index_ray, index_tri
    )
    return locations, index_ray, index_tri


def intersects_first(
    mesh: wp.Mesh,
    ray_origins: wp.array[wp.vec3],
    ray_directions: wp.array[wp.vec3],
    *,
    max_t: float | None = None,
) -> wp.array[wp.int32]:
    """
    Find the index of the first triangle each ray hits.

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
        max_t = default_mesh_query_max_dist(mesh.points, ray_origins)

    out_triangle_index = wp.empty(n, dtype=wp.int32, device=ray_origins.device)
    locations_scratch = wp.empty(n, dtype=wp.vec3, device=ray_origins.device)
    wp.map(
        kernel_ray.first_hit,
        wp.uint64(mesh.id),
        ray_origins,
        ray_directions,
        wp.float32(max_t),
        out=[out_triangle_index, locations_scratch],
    )
    return out_triangle_index


def intersects_any(
    mesh: wp.Mesh,
    ray_origins: wp.array[wp.vec3],
    ray_directions: wp.array[wp.vec3],
    *,
    max_t: float | None = None,
) -> wp.array[wp.bool]:
    """
    Check whether each ray hits the mesh surface.

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
        max_t = default_mesh_query_max_dist(mesh.points, ray_origins)

    out_hit = wp.empty(n, dtype=wp.bool, device=ray_origins.device)
    wp.map(
        kernel_ray.any_hit,
        wp.uint64(mesh.id),
        ray_origins,
        ray_directions,
        wp.float32(max_t),
        out=out_hit,
    )
    return out_hit


def longest_ray(
    mesh: wp.Mesh,
    ray_origins: wp.array[wp.vec3],
    ray_directions: wp.array[wp.vec3],
    *,
    max_t: float | None = None,
    planar_tol: float = TOLERANCE_PLANAR,
) -> wp.array[wp.float32]:
    """
    Find the length of the longest unobstructed ray segment along each direction.

    Uses iterative ``wp.mesh_query_ray`` on the mesh BVH. Ray directions are
    unitized before querying. For each ray, returns the distance to the first mesh
    intersection strictly beyond ``planar_tol`` (to ignore degenerate on-surface
    hits), or ``inf`` when no such intersection exists within ``max_t``.

    Equivalent to [`trimesh.proximity.longest_ray`][].

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
    planar_tol
        Ignore intersections closer than this distance from the ray origin.

    Returns
    -------
    wp.array[wp.float32]
        ``(n,)`` unobstructed ray lengths; ``inf`` when a ray misses within ``max_t``.
    """
    n = ray_origins.shape[0]
    if n == 0:
        return wp.empty(0, dtype=wp.float32, device=ray_origins.device)
    _validate_ray_inputs(mesh, ray_origins, ray_directions)
    if max_t is None:
        max_t = default_mesh_query_max_dist(mesh.points, ray_origins)

    out_distances = wp.empty(n, dtype=wp.float32, device=ray_origins.device)
    wp.map(
        kernel_ray.longest_ray_distance,
        wp.uint64(mesh.id),
        ray_origins,
        ray_directions,
        wp.float32(max_t),
        wp.float32(planar_tol),
        out=out_distances,
    )
    return out_distances


def contains_points(
    mesh: wp.Mesh, points: wp.array[wp.vec3], *, n_sample: int = 5, perturbation_scale: float = 0.1
) -> wp.array[wp.bool]:
    """
    Test whether query points lie inside a closed mesh (ray parity sign).

    Uses ``wp.mesh_query_point_sign_parity`` on the mesh BVH, sharing the same
    parity path as
    [`signed_distance_on_mesh`][triwarp.proximity.signed_distance_on_mesh]. Points
    outside the mesh axis-aligned bounding box are rejected without a parity test.

    Triwarp mesh queries use **Warp's SDF sign convention** (outside positive,
    inside negative). A point is classified as inside when its signed distance
    would be negative; behavior on the on-surface tolerance band is undefined.
    Boolean results still match [`trimesh.Trimesh.contains`][].

    !!! note "Non-watertight meshes"

        Ray parity has no principled answer on an open or holed surface. For those, use the
        generalized winding-number sign instead:

        ```python
        inside = tw.proximity.signed_distance_on_mesh(v, f, pts, sign_mode="winding") < 0.0
        ```

        That mode is deliberately **not** offered here. It needs a ``wp.Mesh`` built with
        ``support_winding_number=True``, and Warp neither records that flag on the mesh object nor
        errors when it is missing — it silently falls back to ray parity. Since this function takes
        a caller-supplied ``wp.Mesh``, triwarp cannot verify the flag, so the option would be able
        to quietly return the parity answer.
        [`signed_distance_on_mesh`][triwarp.proximity.signed_distance_on_mesh] builds its own mesh
        and therefore can guarantee it.

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

    See Also
    --------
    [`signed_distance_on_mesh`][triwarp.proximity.signed_distance_on_mesh]
    """
    n = points.shape[0]
    if n == 0:
        return wp.empty(0, dtype=wp.bool, device=points.device)
    mesh_min, mesh_max = aabb_bounds(mesh.points)
    max_dist = default_mesh_query_max_dist(mesh.points)
    out_contains = wp.empty(n, dtype=wp.bool, device=points.device)
    wp.launch(
        kernel_proximity.contains_points_sign_parity,
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
