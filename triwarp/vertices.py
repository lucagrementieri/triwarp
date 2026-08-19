"""
Per-vertex quantities: normals under five averaging rules, and the angle defect.

A vertex has no normal of its own -- only the faces around it do -- so a vertex normal is always a
weighted average of incident face normals, and the weight is a modelling choice rather than a
detail. The five here differ only in that weight: unweighted
([`mean_vertex_normals`][triwarp.vertices.mean_vertex_normals]), caller-supplied
([`weighted_vertex_normals`][triwarp.vertices.weighted_vertex_normals]), by face area
([`area_weighted_vertex_normals`][triwarp.vertices.area_weighted_vertex_normals]), by incident angle
([`angle_weighted_vertex_normals`][triwarp.vertices.angle_weighted_vertex_normals], the one that is
invariant to how a neighbour is triangulated), and by sine times edge length
([`sine_and_edge_length_weighted_vertex_normals`][triwarp.vertices.sine_and_edge_length_weighted_vertex_normals]).

[`vertex_defects`][triwarp.vertices.vertex_defects] is the odd one out and not a normal at all:
``2 * pi`` minus the incident angle sum, which is the intrinsic curvature concentrated at that
vertex and the quantity
[`discrete_gaussian_curvature`][triwarp.curvature.discrete_gaussian_curvature] reports.
"""

from __future__ import annotations

import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.kernels import predicates as kernel_predicates
from triwarp.kernels import scatter as kernel_scatter
from triwarp.kernels import vertices as kernel_vertices


def n_vertices(indices: twt.IntArray) -> int:
    """
    Vertex count inferred from an index buffer as ``max(indices) + 1``.

    Follows libigl's ``F.maxCoeff() + 1`` convention: the vertex count is one past the largest
    referenced index. Accepts any ``wp.int32`` index buffer — a length-``3 * n_faces`` flat
    triangle buffer, an ``(n, 2)`` edge array, etc. — reading its maximum on the host.

    Parameters
    ----------
    indices
        A ``wp.int32`` index buffer of any shape (e.g. a flat ``faces`` array or a ``(n, 2)``
        edge array).

    Returns
    -------
    int
        ``max(indices) + 1``, or ``0`` when ``indices`` is empty.
    """
    if int(indices.size) == 0:
        return 0
    # Device-side tiled max: only the 4-byte result crosses to the host, not the whole buffer.
    return int(tw.reduce.max(indices)) + 1


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
    return _accumulate_and_normalize(
        n_vertices, faces, kernel_scatter.scatter_sum_vec, face_normals
    )


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
    return _accumulate_and_normalize(
        n_vertices, faces, kernel_scatter.scatter_weighted_sum_vec, face_normals, face_weights
    )


def _accumulate_and_normalize(
    n_vertices: int,
    faces: wp.array[wp.int32],
    scatter_kernel: wp.Kernel,
    values: wp.array[wp.vec3],
    *extra: wp.array,
) -> wp.array[wp.vec3]:
    """
    Scatter per-face vectors onto their corners and unit-normalize the sums.

    The shared body of the module's two primitives: the scatter kernel decides whether each face
    contributes its vector once or scaled by a per-corner weight, and the other three public
    functions reach this through one of them. Accumulation is ``float32`` in an ``(n_vertices, 3)``
    buffer, which is what the scatter kernels write; the cast to ``wp.vec3`` is a reinterpretation
    of the same bytes.

    Parameters
    ----------
    n_vertices
        Output length.
    faces
        Flat ``wp.int32`` triangle index buffer; reshaped to ``(f, 3)`` for the scatter.
    scatter_kernel
        ``kernels.scatter`` kernel taking ``(values, faces2d, *extra, out_sums)`` -- the face table
        is the *second* argument in that family, not the last input.
    values
        ``(f,)`` per-face vectors to accumulate.
    extra
        Any further per-face arrays the kernel takes between the face table and the output, such as
        ``scatter_weighted_sum_vec``'s per-corner weights.

    Returns
    -------
    wp.array[wp.vec3]
        Length-``n_vertices`` unit normals; zero where the accumulated vector was zero.
    """
    device = faces.device
    sums = wp.zeros((n_vertices, 3), dtype=wp.float32, device=device)
    wp.launch(
        scatter_kernel,
        dim=int(values.shape[0]),
        inputs=[values, faces.reshape((-1, 3)), *extra, sums],
        device=device,
    )
    vec_normals = wp.empty(n_vertices, dtype=wp.vec3, device=device)
    wp.utils.array_cast(sums, vec_normals)
    wp.map(wp.normalize, vec_normals, out=vec_normals)
    return vec_normals


def area_weighted_vertex_normals(
    n_vertices: int,
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_normals: wp.array[wp.vec3] | None = None,
    face_areas: wp.array[wp.float32] | None = None,
) -> wp.array[wp.vec3]:
    """
    Area-weighted vertex normals.

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
    scaled_normals = wp.empty(n_faces, dtype=wp.vec3, device=faces.device)
    wp.map(wp.mul, face_normals, face_areas, out=scaled_normals)
    return mean_vertex_normals(n_vertices, faces, scaled_normals)


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
    face_weights = twt.empty_2d((n_faces, 3), wp.float32, device=vertices.device)
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

    See Also
    --------
    [`discrete_gaussian_curvature`][triwarp.curvature.discrete_gaussian_curvature]
        The same quantity at a *scale*: the Cohen-Steiner/Morvan ball measure sums these defects
        over a ball of given radius, where this is the pointwise value at one vertex. The ball
        measure is what converges under refinement; the pointwise defect does not.
    [`face_angles`][triwarp.triangles.face_angles]
        The angles this sums.
    """
    angle_sum = wp.zeros(n_vertices, dtype=wp.float32, device=faces.device)
    faces2d = faces.reshape((-1, 3))
    wp.launch(
        kernel_scatter.scatter_sum_scalar,
        dim=int(face_angles.shape[0]),
        inputs=[face_angles, faces2d, angle_sum],
        device=faces.device,
    )
    # In place over the accumulator: ``angle_sum`` is scratch, and the operator spelling
    # ``TWO_PI - angle_sum`` would run the same wp.map into a second allocation.
    wp.map(kernel_predicates.angle_defect, angle_sum, out=angle_sum)
    return angle_sum
