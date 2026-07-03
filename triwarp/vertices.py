from __future__ import annotations

import numpy as np
import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.kernels import array as kernel_array
from triwarp.kernels import vertices as kernel_vertices


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
    wp.launch(
        kernel_array.scatter_sum_vec,
        dim=face_normals.shape[0],
        inputs=[face_normals, faces2d, normals],
    )
    vec_normals = wp.empty(n_vertices, dtype=wp.vec3, device=faces.device)
    wp.utils.array_cast(normals, vec_normals)
    wp.launch(kernel_array.normalize, dim=n_vertices, inputs=[vec_normals])
    return vec_normals


def weighted_vertex_normals(
    n_vertices: int,
    faces: wp.array[wp.int32],
    face_normals: wp.array[wp.vec3],
    face_weights: twt.Array2dFloat32,
) -> wp.array[wp.vec3]:
    """
    Vertex normals from a weighted sum of incident face normals, then unit-length.

    Each face contributes its normal scaled by the per-corner weight in ``face_weights``;
    contributions are summed per vertex in ``float32`` on ``faces.device``, cast to
    ``wp.vec3``, and L2-normalized.

    Parameters
    ----------
    n_vertices
        Number of vertices indexed by ``faces`` (output length).
    faces
        Triangle indices as ``wp.int32``; interpreted as ``(f, 3)`` via ``reshape((-1, 3))``.
    face_normals
        One normal per triangle, length ``f`` as ``wp.vec3``, aligned with ``faces``.
    face_weights
        Per-corner weights, shape ``(f, 3)`` as ``twt.Array2dFloat32`` with rows matching
        ``faces`` / ``face_normals``.

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
        inputs=[face_normals, faces2d, face_weights, normals],
    )
    vec_normals = wp.empty(n_vertices, dtype=wp.vec3, device=faces.device)
    wp.utils.array_cast(normals, vec_normals)
    wp.launch(kernel_array.normalize, dim=n_vertices, inputs=[vec_normals])
    return vec_normals


def area_weighted_vertex_normals(
    n_vertices: int,
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_normals: wp.array[wp.vec3] | None = None,
    face_areas: wp.array[wp.float32] | None = None,
) -> wp.array[wp.vec3]:
    """
    Area-weighted vertex normals (``igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA``).

    Each face contributes its normal scaled by the triangle area at all three corners;
    contributions are summed per vertex and L2-normalized. This matches libigl's default
    ``per_vertex_normals`` weighting (up to the constant ``2`` factor from ``doublearea``,
    which cancels during normalization).

    Parameters
    ----------
    n_vertices
        Number of vertices indexed by ``faces`` (output length).
    vertices
        Mesh vertex positions as ``wp.vec3``, length ``n_vertices``.
    faces
        Triangle indices as ``wp.int32``; interpreted as ``(f, 3)`` via ``reshape((-1, 3))``.
    face_normals
        One normal per triangle, length ``f`` as ``wp.vec3``. When ``None``, computed from
        ``vertices`` and ``faces``.
    face_areas
        Scalar area per triangle, length ``f`` as ``wp.float32``. When ``None``, computed
        from ``vertices`` and ``faces``.

    Returns
    -------
    wp.array[wp.vec3]
        Length-``n_vertices`` device array of unit normals where the accumulated vector was
        non-zero; otherwise the corresponding entry is zero.
    """
    if face_normals is None or face_areas is None:
        computed_normals, computed_areas = tw.triangles.face_normals_and_areas(vertices, faces)
        if face_normals is None:
            face_normals = computed_normals
        if face_areas is None:
            face_areas = computed_areas
    n_faces = int(face_areas.shape[0])
    areas_np = np.ascontiguousarray(face_areas.numpy(), dtype=np.float32).reshape(-1, 1)
    face_weights_np = np.broadcast_to(areas_np, (n_faces, 3))
    face_weights = wp.array(face_weights_np, dtype=wp.float32, device=faces.device)
    return weighted_vertex_normals(n_vertices, faces, face_normals, face_weights)


def angle_weighted_vertex_normals(
    n_vertices: int,
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_normals: wp.array[wp.vec3] | None = None,
    face_angles: twt.Array2dFloat32 | None = None,
) -> wp.array[wp.vec3]:
    """
    Angle-weighted vertex normals (Thuerrner & Wuethrich, 1998).

    Each face contributes its normal scaled by the interior angle at the corner vertex;
    contributions are summed per vertex and L2-normalized. This matches the "polygonal
    facets" recipe in *Computing Vertex Normals from Polygonal Facets*, Journal of
    Graphics Tools 3:1, 43-46 (1998), and ``igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE``.

    Parameters
    ----------
    n_vertices
        Number of vertices indexed by ``faces`` (output length).
    vertices
        Mesh vertex positions as ``wp.vec3``, length ``n_vertices``.
    faces
        Triangle indices as ``wp.int32``; interpreted as ``(f, 3)`` via ``reshape((-1, 3))``.
    face_normals
        One normal per triangle, length ``f`` as ``wp.vec3``. When ``None``, computed from
        ``vertices`` and ``faces``.
    face_angles
        Interior angles at the three corners of each triangle, shape ``(f, 3)`` as
        ``twt.Array2dFloat32``. When ``None``, computed from ``vertices`` and ``faces``.

    Returns
    -------
    wp.array[wp.vec3]
        Length-``n_vertices`` device array of unit normals where the accumulated vector was
        non-zero; otherwise the corresponding entry is zero.
    """
    if face_normals is None:
        face_normals, _ = tw.triangles.face_normals_and_areas(vertices, faces)
    if face_angles is None:
        face_angles = tw.triangles.face_angles(vertices, faces)
    return weighted_vertex_normals(n_vertices, faces, face_normals, face_angles)


def sine_and_edge_length_weighted_vertex_normals(
    n_vertices: int,
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_normals: wp.array[wp.vec3] | None = None,
) -> wp.array[wp.vec3]:
    """
    Nelson Max MWSELR vertex normals (sine and edge-length reciprocal weighting).

    At each corner, the incident face contributes ``sin(angle) / (||E_i|| ||E_{i+1}||) * N_i``,
    which simplifies to ``(e1 x e2) / (||e1||^2 * ||e2||^2)`` for the two outgoing edges at that
    vertex. Contributions are summed per vertex and L2-normalized. This matches the
    Mean Weighted by Sine and Edge Length Reciprocals recipe optimized by Max for smooth
    surface reconstruction (Jin et al., 2005).

    Parameters
    ----------
    n_vertices
        Number of vertices indexed by ``faces`` (output length).
    vertices
        Mesh vertex positions as ``wp.vec3``, length ``n_vertices``.
    faces
        Triangle indices as ``wp.int32``; interpreted as ``(f, 3)`` via ``reshape((-1, 3))``.
    face_normals
        One normal per triangle, length ``f`` as ``wp.vec3``, aligned with ``faces``. When
        ``None``, the unnormalized triangle cross product is used instead.

    Returns
    -------
    wp.array[wp.vec3]
        Length-``n_vertices`` device array of unit normals where the accumulated vector was
        non-zero; otherwise the corresponding entry is zero.
    """
    n_faces = faces.shape[0] // 3
    if n_faces == 0:
        return wp.zeros(n_vertices, dtype=wp.vec3, device=vertices.device)
    unit_face_normals = face_normals is not None
    if face_normals is None:
        face_normals = wp.empty(n_faces, dtype=wp.vec3, device=vertices.device)
        wp.launch(
            kernel_vertices.face_crosses,
            dim=n_faces,
            inputs=[vertices, faces, face_normals],
            device=vertices.device,
        )
    face_weights = twt.empty_float32_2d((n_faces, 3), device=vertices.device)
    wp.launch(
        kernel_vertices.max_vertex_normal_weights,
        dim=n_faces,
        inputs=[vertices, faces, unit_face_normals, face_weights],
        device=vertices.device,
    )
    return weighted_vertex_normals(n_vertices, faces, face_normals, face_weights)


def vertex_defects(
    n_vertices: int, faces: wp.array[wp.int32], face_angles: twt.Array2dFloat32
) -> wp.array[wp.float32]:
    """
    Discrete angle defect per vertex: ``2π`` minus the sum of incident corner angles.

    For each vertex, interior angles from every triangle corner that references that vertex
    are accumulated in ``float32`` on ``faces.device``, then subtracted from a full turn.
    This is the standard piecewise-linear angle defect (related to discrete Gaussian
    curvature via the Gauss—Bonnet viewpoint on triangle meshes).

    Parameters
    ----------
    n_vertices
        Number of vertices indexed by ``faces`` (output length).
    faces
        Triangle indices as ``wp.int32``; interpreted as ``(f, 3)`` via ``reshape((-1, 3))``
        (row-major flat layout is fine).
    face_angles
        Interior angles at the three corners of each triangle, shape ``(f, 3)`` as
        ``twt.Array2dFloat32``, with rows aligned with ``faces``.

    Returns
    -------
    wp.array[wp.float32]
        Length-``n_vertices`` device array ``2π - Σ angles`` at each vertex. Vertices not
        referenced by any face have defect ``2π`` (empty angle sum).
    """
    angle_sum = wp.zeros(n_vertices, dtype=wp.float32, device=faces.device)
    faces2d = faces.reshape((-1, 3))
    wp.launch(
        kernel_array.scatter_sum_scalar,
        dim=face_angles.shape[0],
        inputs=[face_angles, faces2d, angle_sum],
    )
    defect = (2 * wp.pi) - angle_sum
    return defect
