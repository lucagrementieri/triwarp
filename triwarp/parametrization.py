from __future__ import annotations

import warp as wp

import triwarp as tw
from triwarp.kernels import parametrization as kernel_parametrization


def flipped_faces_mask(
    vertices: wp.array[wp.vec2], faces: wp.array[wp.int32]
) -> wp.array[wp.bool]:
    """
    Per-face flag: whether a triangle is inverted (negative 2D signed area) in the parametrization.

    For each triangle the 2D signed area of its three UV vertices is computed; a face is flagged
    ``True`` when that area is strictly negative, i.e. the triangle has folded over (flipped
    orientation) in the 2D domain. Mirrors libigl's ``flipped_triangles`` per-triangle test
    (determinant of the homogeneous ``3 x 3`` vertex matrix ``< 0``). Degenerate (zero-area)
    triangles are **not** flagged, matching the strict ``< 0`` comparison.
    [`flipped_faces`][triwarp.parametrization.flipped_faces] is the index form of this mask.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` 2D vertex positions (the parametrization / UV coordinates) as ``wp.vec2``.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.

    Returns
    -------
    wp.array[wp.bool]
        Length ``n_faces`` on ``vertices.device``. Empty for an empty mesh.

    See Also
    --------
    [`flipped_faces`][triwarp.parametrization.flipped_faces]

    Notes
    -----
    Equivalent to the per-triangle predicate behind libigl ``flipped_triangles``: the 2D cross
    product ``(v1 - v0) x (v2 - v0)`` equals ``det([[x0, x1, x2], [y0, y1, y2], [1, 1, 1]])``, so a
    ``True`` entry corresponds exactly to a triangle libigl would list as flipped.
    """
    device = vertices.device
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return wp.empty(0, dtype=wp.bool, device=device)

    out_mask = wp.empty(n_faces, dtype=wp.bool, device=device)
    wp.launch(
        kernel_parametrization.flipped_faces_mask,
        dim=n_faces,
        inputs=[vertices, faces, out_mask],
        device=device,
    )
    return out_mask


def flipped_faces(
    vertices: wp.array[wp.vec2], faces: wp.array[wp.int32]
) -> wp.array[wp.int32]:
    """
    Return the indices of triangles inverted (negative 2D signed area) in the parametrization.

    Convenience wrapper returning ``flatnonzero`` of
    [`flipped_faces_mask`][triwarp.parametrization.flipped_faces_mask]: the indices into ``faces``
    of triangles whose 2D signed area is strictly negative (folded over in the UV domain). Matches
    libigl's ``flipped_triangles``, which returns the same list of flipped-triangle indices.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` 2D vertex positions (the parametrization / UV coordinates) as ``wp.vec2``.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.

    Returns
    -------
    wp.array[wp.int32]
        Ascending face indices of the flipped triangles on ``vertices.device``. Empty when no
        triangle is flipped.

    See Also
    --------
    [`flipped_faces_mask`][triwarp.parametrization.flipped_faces_mask]
    [`flatnonzero`][triwarp.array.flatnonzero]
    """
    return tw.array.flatnonzero(flipped_faces_mask(vertices, faces))
