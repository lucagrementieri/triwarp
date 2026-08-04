"""Per-triangle geometry queries (Warp), mirroring `trimesh.triangles`."""

from typing import Literal

import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.constants import ITEMS_PER_SLICE
from triwarp.kernels import triangles as kernel_triangles


def face_normals_and_areas(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> tuple[wp.array[wp.vec3], wp.array[wp.float32]]:
    """
    Compute the unit normal and area of each triangle.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.

    Returns
    -------
    normals : wp.array[wp.vec3]
        Length-``n_faces`` unit face normals on ``vertices.device``. Zero when a face is
        degenerate.
    areas : wp.array[wp.float32]
        Length-``n_faces`` triangle areas on ``vertices.device``.

    See Also
    --------
    [`trimesh.Trimesh.face_normals`][]
    [`trimesh.Trimesh.area_faces`][]
    """
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


def face_angles(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> twt.Array2dFloat32:
    """
    Interior angle at each triangle vertex, in radians.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.

    Returns
    -------
    twt.Array2dFloat32
        Shape ``(n_faces, 3)`` interior angles on ``vertices.device``, aligned with the
        corners ``(i0, i1, i2)`` of each face.

    See Also
    --------
    [`trimesh.triangles.angles`][]
    """
    f = faces.shape[0] // 3
    out_angle = twt.empty_float32_2d((f, 3), device=vertices.device)
    wp.launch(
        kernel_triangles.angles, dim=f, inputs=[vertices, faces, out_angle], device=vertices.device
    )
    return twt.as_array2d_float32(out_angle)


_QUALITY_METRICS: dict[str, wp.int32] = {
    "aspect_ratio": kernel_triangles.QUALITY_ASPECT_RATIO,
    "radius_ratio": kernel_triangles.QUALITY_RADIUS_RATIO,
    "area_max_side": kernel_triangles.QUALITY_AREA_MAX_SIDE,
    "mean_ratio": kernel_triangles.QUALITY_MEAN_RATIO,
    "area": kernel_triangles.QUALITY_AREA,
}

FaceQualityMetric = Literal["aspect_ratio", "radius_ratio", "area_max_side", "mean_ratio", "area"]
"""Shape-quality measure selected by [`face_quality`][triwarp.triangles.face_quality]."""


def face_quality(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    metric: FaceQualityMetric = "aspect_ratio",
) -> wp.array[wp.float32]:
    """
    Per-face shape-quality measure.

    This is the quantity the library already uses internally to decide whether a triangle is worth
    keeping — [`isotropic_remesh`][triwarp.remesh.isotropic_remesh] and
    [`flip_to_delaunay`][triwarp.remesh.flip_to_delaunay] gate on ``aspect_ratio``, and the
    hole-filling cost functions weight candidate triangles by it — exposed so a caller can inspect
    or threshold it directly.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    metric
        Which measure to compute. All but ``"area"`` are invariant to a uniform scaling of the
        mesh, and all but ``"aspect_ratio"`` are *larger is better*:

        - ``"aspect_ratio"`` (default) — circumradius over twice the inradius. ``1`` for an
          equilateral triangle and unbounded above, so **smaller is better**; a degenerate
          triangle reads ``+inf``. This is MeshLib's ``triangleAspectRatio`` and the measure the
          remeshing gates use.
        - ``"radius_ratio"`` — inradius over circumradius, rescaled so an equilateral triangle
          reads ``1``; ``0`` when degenerate. MeshLab's ``inradius/circumradius``.
        - ``"area_max_side"`` — twice the area over the longest side squared, ``sqrt(3)/2`` at
          best. MeshLab's ``area/max side`` (scale-invariant despite the name).
        - ``"mean_ratio"`` — ``4 sqrt(3) A / (a^2 + b^2 + c^2)``, ``1`` at best. MeshLab's
          ``Mean ratio``.
        - ``"area"`` — the plain triangle area, for parity with MeshLab's ``Area``; identical to
          the second return of
          [`face_normals_and_areas`][triwarp.triangles.face_normals_and_areas].

    Returns
    -------
    wp.array[wp.float32]
        Length-``n_faces`` quality values on ``vertices.device``.

    Raises
    ------
    ValueError
        If ``metric`` is not one of the listed names.

    See Also
    --------
    [`face_normals_and_areas`][triwarp.triangles.face_normals_and_areas]
    [`nondegenerate`][triwarp.triangles.nondegenerate]
    [`triwarp.remesh.isotropic_remesh`][triwarp.remesh.isotropic_remesh]
    """
    if metric not in _QUALITY_METRICS:
        raise ValueError(f"unknown metric {metric!r}, expected one of {sorted(_QUALITY_METRICS)}")
    f = faces.shape[0] // 3
    out_quality = wp.empty(f, dtype=wp.float32, device=vertices.device)
    wp.launch(
        kernel_triangles.face_quality,
        dim=f,
        inputs=[vertices, faces, _QUALITY_METRICS[metric], out_quality],
        device=vertices.device,
    )
    return out_quality


def centroid(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> wp.vec3:
    """
    Area-weighted centroid of the mesh surface.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.

    Returns
    -------
    wp.vec3
        Area-weighted mean of per-face centroids. All-``NaN`` when ``faces`` is empty.

    See Also
    --------
    [`trimesh.Trimesh.centroid`][]
    """
    f = faces.shape[0] // 3
    if f == 0:
        return wp.vec3(float("nan"), float("nan"), float("nan"))
    device = vertices.device
    out_centroid = wp.zeros(1, dtype=wp.vec3, device=device)
    out_total_area = wp.zeros(1, dtype=wp.float32, device=device)
    n_slices = max(1, (f + ITEMS_PER_SLICE - 1) // ITEMS_PER_SLICE)
    wp.launch(
        kernel_triangles.centroid,
        dim=n_slices,
        inputs=[vertices, faces, wp.int32(f), wp.int32(n_slices), out_centroid, out_total_area],
        device=device,
    )
    # Two unavoidable readbacks: the return type is a host-side wp.vec3, so the sums have to
    # cross to the host to be divided.
    centroid = out_centroid.numpy()[0]
    total_area = float(out_total_area.numpy()[0])
    return wp.vec3(
        float(centroid[0]) / total_area,
        float(centroid[1]) / total_area,
        float(centroid[2]) / total_area,
    )


def volume(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> float:
    """
    Signed volume enclosed by the mesh.

    Sum of per-face signed tetrahedron volumes measured from the origin
    (``dot(v0, cross(v1, v2)) / 6``); for a closed, consistently wound surface this is
    independent of the reference point, and its sign follows the orientation of the face
    normals (positive for outward-facing normals). For an open or inconsistently wound mesh
    the result is not a meaningful volume.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.

    Returns
    -------
    float
        Signed volume in ``float32``. ``0.0`` for an empty mesh.

    See Also
    --------
    [`centroid`][triwarp.triangles.centroid]
    [`is_volume`][triwarp.validation.is_volume]
    [`trimesh.Trimesh.volume`][]
    """
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return 0.0
    device = vertices.device
    volumes = wp.empty(n_faces, dtype=wp.float32, device=device)
    wp.launch(
        kernel_triangles.signed_tet_volumes,
        dim=n_faces,
        inputs=[vertices, faces, wp.vec3(0.0, 0.0, 0.0), volumes],
        device=device,
    )
    return tw.reduce.sum(volumes)


def nondegenerate(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> wp.array[wp.bool]:
    """
    Flag triangles with non-zero area.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.

    Returns
    -------
    wp.array[wp.bool]
        Length-``n_faces`` mask on ``vertices.device``; ``True`` where the triangle area is
        non-zero.

    See Also
    --------
    [`trimesh.triangles.nondegenerate`][]
    """
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
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], barycentric: wp.array[wp.vec3]
) -> wp.array[wp.vec3]:
    """
    Convert barycentric coordinates to Cartesian points, one triangle per row.

    Operates on the "triangle soup" convention shared with `trimesh.triangles`: row ``i``
    of ``barycentric`` is evaluated against triangle ``i`` of ``faces`` (not every triangle
    against every coordinate).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    barycentric
        Length-``n_faces`` barycentric coordinates ``(u, v, w)`` as ``wp.vec3``, aligned
        one-to-one with the triangles in ``faces``.

    Returns
    -------
    wp.array[wp.vec3]
        Length-``n_faces`` Cartesian points on ``vertices.device``.

    See Also
    --------
    [`trimesh.triangles.barycentric_to_points`][]
    """
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
    """
    Convert Cartesian points to barycentric coordinates, one triangle per row.

    Operates on the "triangle soup" convention shared with `trimesh.triangles`: row ``i`` of
    ``points`` is projected onto triangle ``i`` of ``faces`` (not every triangle against every
    point).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    points
        Length-``n_faces`` query positions as ``wp.vec3``, aligned one-to-one with the
        triangles in ``faces``.
    method
        ``"cramer"`` solves the 2x2 linear system via Cramer's rule; ``"cross"`` uses the
        cross-product-ratio formulation. Both should agree up to floating-point error.

    Returns
    -------
    wp.array[wp.vec3]
        Length-``n_faces`` barycentric coordinates ``(u, v, w)`` on ``vertices.device``.

    See Also
    --------
    [`trimesh.triangles.points_to_barycentric`][]
    """
    f = faces.shape[0] // 3
    out_barycentric = wp.empty(f, dtype=wp.vec3, device=vertices.device)
    kernel = (
        kernel_triangles.points_to_barycentric_cramer
        if method == "cramer"
        else kernel_triangles.points_to_barycentric_cross
    )
    wp.launch(
        kernel, dim=f, inputs=[vertices, faces, points, out_barycentric], device=vertices.device
    )
    return out_barycentric


def closest_point(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], points: wp.array[wp.vec3]
) -> wp.array[wp.vec3]:
    """
    Closest point on each triangle to a query point, one triangle per row.

    Operates on the "triangle soup" convention shared with `trimesh.triangles`: row ``i`` of
    ``points`` is projected onto triangle ``i`` of ``faces`` (not every triangle against every
    point). For nearest-triangle queries over an entire mesh, see
    [`closest_point_on_mesh`][triwarp.proximity.closest_point_on_mesh] instead.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    points
        Length-``n_faces`` query positions as ``wp.vec3``, aligned one-to-one with the
        triangles in ``faces``.

    Returns
    -------
    wp.array[wp.vec3]
        Length-``n_faces`` closest points on ``vertices.device``.

    See Also
    --------
    [`trimesh.triangles.closest_point`][]
    """
    f = faces.shape[0] // 3
    out_closest = wp.empty(f, dtype=wp.vec3, device=vertices.device)
    wp.launch(
        kernel_triangles.closest_point,
        dim=f,
        inputs=[vertices, faces, points, out_closest],
        device=vertices.device,
    )
    return out_closest
