"""Per-triangle geometry queries (Warp), mirroring `trimesh.triangles`."""

from typing import Literal

import warp as wp

import triwarp.typing as twt
from triwarp.constants import TILE_1D
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
    out_centroid = wp.zeros(3, dtype=wp.float32, device=device)
    out_total_area = wp.zeros(1, dtype=wp.float32, device=device)
    n_tiles = (f + TILE_1D - 1) // TILE_1D
    wp.launch_tiled(
        kernel_triangles.centroid,
        dim=[n_tiles],
        inputs=[vertices, faces, wp.int32(f), out_centroid, out_total_area],
        block_dim=TILE_1D,
        device=device,
    )
    centroid = out_centroid.numpy()
    total_area = float(out_total_area.numpy()[0])
    return wp.vec3(
        float(centroid[0]) / total_area,
        float(centroid[1]) / total_area,
        float(centroid[2]) / total_area,
    )


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
