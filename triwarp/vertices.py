import warp as wp

from triwarp.kernels import array as kernel_array


def mean_vertex_normals(
    n_vertices: int, faces: wp.array[wp.int32], face_normals: wp.array[wp.vec3]
) -> wp.array[wp.vec3]:
    """
    Vertex normals as the (unnormalized) sum of incident face normals, then unit-length.

    For each vertex, face normals sharing that corner are accumulated in ``float32`` on
    ``faces.device``, cast to ``wp.vec3``, and L2-normalized. Vertices not referenced by
    any face remain zero.

    Parameters
    ----------
    n_vertices
        Number of vertices indexed by ``faces`` (output length).
    faces
        Triangle indices as ``wp.int32``; interpreted as ``(f, 3)`` via ``reshape((-1, 3))``
        (row-major ``[i0, i1, i2, …]`` flat layout is fine).
    face_normals
        One unit (or unnormalized) normal per triangle, length ``f`` as ``wp.vec3``, aligned
        with the rows of ``faces``.

    Returns
    -------
    wp.array[wp.vec3]
        Length-``n_vertices`` device array of unit normals where the accumulated vector was
        non-zero; otherwise the corresponding entry is zero.
    """
    normals = wp.zeros((n_vertices, 3), dtype=wp.float32, device=faces.device)
    faces2d = faces.reshape((-1, 3))
    wp.launch(kernel_array.scatter_sum_vec, dim=face_normals.shape[0], inputs=[face_normals, faces2d, normals])
    vec_normals = wp.empty(n_vertices, dtype=wp.vec3, device=faces.device)
    wp.utils.array_cast(normals, vec_normals)
    wp.launch(kernel_array.normalize, dim=n_vertices, inputs=[vec_normals])
    return vec_normals


# TODO: check management of degenerate faces
def weighted_vertex_normals(
    n_vertices: int, faces: wp.array[wp.int32], face_normals: wp.array[wp.vec3], face_angles: wp.array2d[wp.float32]
) -> wp.array[wp.vec3]:
    """
    Angle-weighted vertex normals (Thuerrner & Wuethrich, 1998).

    Each face contributes its normal scaled by the interior angle at the corner vertex;
    contributions are summed per vertex in ``float32`` on ``faces.device``, cast to
    ``wp.vec3``, and L2-normalized. This matches the “polygonal facets” recipe in
    *Computing Vertex Normals from Polygonal Facets*, Journal of Graphics Tools 3:1,
    43-46 (1998).

    Parameters
    ----------
    n_vertices
        Number of vertices indexed by ``faces`` (output length).
    faces
        Triangle indices as ``wp.int32``; interpreted as ``(f, 3)`` via ``reshape((-1, 3))``.
    face_normals
        One normal per triangle, length ``f`` as ``wp.vec3``, aligned with ``faces``.
    face_angles
        Interior angles at the three corners of each triangle, shape ``(f, 3)`` as
        ``wp.array2d[wp.float32]`` with rows matching ``faces`` / ``face_normals``.

    Returns
    -------
    wp.array[wp.vec3]
        Length-``n_vertices`` device array of unit normals where the accumulated vector was
        non-zero; otherwise the corresponding entry is zero.
    """
    normals = wp.zeros((n_vertices, 3), dtype=wp.float32, device=faces.device)
    faces2d = faces.reshape((-1, 3))
    wp.launch(
        kernel_array.scatter_weighted_sum_vec,
        dim=face_normals.shape[0],
        inputs=[face_normals, faces2d, face_angles, normals],
    )
    vec_normals = wp.empty(n_vertices, dtype=wp.vec3, device=faces.device)
    wp.utils.array_cast(normals, vec_normals)
    wp.launch(kernel_array.normalize, dim=n_vertices, inputs=[vec_normals])
    return vec_normals
