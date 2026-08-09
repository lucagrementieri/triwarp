"""
Quantities carried by one triangle at a time: its normal, area, angles, shape and barycentre.

[`face_normals_and_areas`][triwarp.triangles.face_normals_and_areas],
[`face_angles`][triwarp.triangles.face_angles],
[`face_centroids`][triwarp.triangles.face_centroids] and
[`face_quality`][triwarp.triangles.face_quality] are the per-face fields the rest of the package
reduces over; [`nondegenerate`][triwarp.triangles.nondegenerate] flags the faces those quantities
are meaningless on. [`barycentric_to_points`][triwarp.triangles.barycentric_to_points],
[`points_to_barycentric`][triwarp.triangles.points_to_barycentric] and
[`closest_point`][triwarp.triangles.closest_point] work one query point against one triangle, row
by row, without a BVH -- for a query against the whole surface see
[`triwarp.proximity`][triwarp.proximity].
"""

from typing import Literal

import warp as wp

import triwarp.typing as twt
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
    out_angle = twt.empty_2d((f, 3), wp.float32, device=vertices.device)
    wp.launch(
        kernel_triangles.angles, dim=f, inputs=[vertices, faces, out_angle], device=vertices.device
    )
    return twt.as_array2d(out_angle, wp.float32)


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


def face_centroids(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> wp.array[wp.vec3]:
    """
    Barycentre of every face: the mean of its three corners.

    Not to be confused with [`surface_centroid`][triwarp.totals.surface_centroid], which is the
    *mesh's* single area-weighted centre. This is one point per triangle and no weighting is
    involved.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.

    Returns
    -------
    wp.array[wp.vec3]
        ``(n_faces,)`` face barycentres on ``vertices.device``.

    See Also
    --------
    [`centroid`][triwarp.totals.surface_centroid]
    [`moments`][triwarp.totals.moments]
    ``igl.barycenter``
    """
    n_faces = int(faces.shape[0]) // 3
    out_centroids = wp.empty(n_faces, dtype=wp.vec3, device=vertices.device)
    if n_faces == 0:
        return out_centroids
    wp.launch(
        kernel_triangles.face_centroids,
        dim=n_faces,
        inputs=[vertices, faces, out_centroids],
        device=vertices.device,
    )
    return out_centroids


def face_signed_volumes(
    vertices: wp.array[wp.vec3] | wp.array[wp.vec3d],
    faces: wp.array[wp.int32],
    apex: wp.vec3 | wp.vec3d | None = None,
) -> wp.array[wp.float32] | wp.array[wp.float64]:
    """
    Signed volume of the tetrahedron ``(apex, v0, v1, v2)`` for each face.

    ``dot(v0 - apex, cross(v1 - apex, v2 - apex)) / 6``, so the sign follows the face's winding:
    positive where the face turns its front to the apex. Summed over a closed, consistently wound
    surface this is the enclosed volume ([`volume`][triwarp.totals.volume]) and is independent of
    ``apex``; per face it is not, which is why the argument exists —
    [`sample_volume`][triwarp.sample.sample_volume] fans from the surface centroid so that every
    tetrahedron is positive on a star-shaped mesh, and uses the array as a cumulative distribution.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` positions, ``wp.vec3`` or ``wp.vec3d``. The output dtype follows: pass
        ``vec3d`` where the sum's low digits matter, as
        [`filter_laplacian`][triwarp.smoothing.filter_laplacian]'s volume constraint does.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    apex
        Common apex of every tetrahedron, in ``vertices``' dtype. ``None`` (the default) means the
        world origin, which is what every whole-mesh volume wants.

    Returns
    -------
    wp.array[wp.float32] | wp.array[wp.float64]
        ``(n_faces,)`` signed volumes on ``vertices.device``, in ``vertices``' scalar type. Empty
        when ``faces`` is empty.

    See Also
    --------
    [`volume`][triwarp.totals.volume]
        The sum of these over the whole mesh.
    [`is_volume`][triwarp.validation.is_volume]
        Whether that sum is a meaningful volume at all.
    [`face_centroids`][triwarp.triangles.face_centroids]
    """
    n_faces = int(faces.shape[0]) // 3
    device = vertices.device
    scalar = wp.float64 if vertices.dtype is wp.vec3d else wp.float32
    volumes = wp.empty(n_faces, dtype=scalar, device=device)
    if n_faces == 0:
        return volumes
    if apex is None:
        apex = (
            wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
            if vertices.dtype is wp.vec3d
            else wp.vec3(0.0, 0.0, 0.0)
        )
    wp.launch(
        kernel_triangles.face_signed_volumes,
        dim=n_faces,
        inputs=[vertices, faces, apex, volumes],
        device=device,
    )
    return volumes


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
