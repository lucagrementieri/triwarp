import warp as wp

from triwarp.kernels import array as kernel_array
from triwarp.kernels import curvature as kernel_curvature
import triwarp.typing as twt
import triwarp as tw
from triwarp.vertices import vertex_defects


def discrete_gaussian_curvature(
    points: wp.array[wp.vec3],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_angles: twt.Array2dFloat32,
    radius: float,
) -> wp.array[wp.float32]:
    """
    Return the discrete gaussian curvature measure of a sphere
    centered at a point as detailed in 'Restricted Delaunay
    triangulations and normal cycle'- Cohen-Steiner and Morvan.

    This is the sum of the vertex defects at all vertices
    within the radius for each point.

    Parameters
    ----------
    points : (n, 3) float
      Points in space
    radius : float ,
      The sphere radius, which can be zero if vertices
      passed are points.

    Returns
    --------
    gaussian_curvature:  (n,) float
      Discrete gaussian curvature measure.
    """
    nearest_indices, _, nearest_offsets = tw.points.query_hashgrid_ball_with_offsets(vertices, points, radius)
    defects = vertex_defects(vertices.shape[0], faces, face_angles)
    gauss_curvature = wp.zeros(points.shape[0], dtype=wp.float32, device=points.device)
    wp.launch(
        kernel_array.scatter_offset_sum,
        dim=nearest_indices.shape[0],
        inputs=[defects, nearest_indices, nearest_offsets, gauss_curvature],
    )
    return gauss_curvature


def discrete_mean_curvature(
    points: wp.array[wp.vec3],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    radius: float,
    *,
    face_adjacency: twt.Array2dInt32 | None = None,
    face_adjacency_edges: twt.Array2dInt32 | None = None,
) -> wp.array[wp.float32]:
    """
    Return the discrete mean curvature measure of a sphere
    centered at a point as detailed in 'Restricted Delaunay
    triangulations and normal cycle'- Cohen-Steiner and Morvan.

    This is the sum of the angle at all edges contained in the
    sphere for each point.

    Parameters
    ----------
    points
        ``(n,)`` query positions in space as ``wp.vec3``.
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        Length-``3 * n_faces`` flat triangle index buffer.
    radius
        Sphere radius which should typically be greater than zero.
    face_adjacency
        Optional ``(m, 2)`` face index pairs from
        :func:`triwarp.graph.face_adjacency`. When ``None``, adjacency and
        shared edges are computed from ``faces``.
    face_adjacency_edges
        Optional ``(m, 2)`` sorted shared vertex pairs. Must be supplied
        together with ``face_adjacency`` or omitted with it.

    Returns
    -------
    wp.array[wp.float32]
        Length ``n`` discrete mean curvature measure on ``points.device``.

    See Also
    --------
    :func:`discrete_gaussian_curvature`
    :func:`trimesh.curvature.discrete_mean_curvature_measure`
    """
    device = points.device
    n_points = int(points.shape[0])
    if n_points == 0:
        return wp.empty(0, dtype=wp.float32, device=device)

    if vertices.device != device or faces.device != device:
        raise ValueError(
            f"points, vertices, and faces must live on the same device, got {device}, {vertices.device}, {faces.device}"
        )

    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return wp.zeros(n_points, dtype=wp.float32, device=device)

    if (face_adjacency is None) != (face_adjacency_edges is None):
        raise ValueError("face_adjacency and face_adjacency_edges must both be provided or both omitted")
    if face_adjacency is None:
        face_adjacency, face_adjacency_edges = tw.graph.face_adjacency(faces, return_edges=True)
    assert face_adjacency is not None and face_adjacency_edges is not None

    m = int(face_adjacency.shape[0])
    if m == 0:
        return wp.zeros(n_points, dtype=wp.float32, device=device)

    angles = tw.graph.face_adjacency_angles(vertices, faces, face_adjacency=face_adjacency)
    convex = tw.convex.face_adjacency_convex(
        vertices, faces, face_adjacency=face_adjacency, face_adjacency_edges=face_adjacency_edges
    )

    edge_lower = wp.empty(m, dtype=wp.vec3, device=device)
    edge_upper = wp.empty(m, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_curvature.edge_aabb_from_endpoints,
        dim=m,
        inputs=[vertices, face_adjacency_edges, edge_lower, edge_upper],
        device=device,
    )

    bvh = tw.points.bvh_from_bounds(edge_lower, edge_upper)
    candidate_edges, offsets = tw.points.query_bvh_aabb_with_offsets(bvh, points, radius)

    mean_curvature = wp.zeros(n_points, dtype=wp.float32, device=device)
    n_candidates = int(candidate_edges.shape[0])
    if n_candidates > 0:
        wp.launch(
            kernel_curvature.accumulate_mean_curvature,
            dim=n_candidates,
            inputs=[
                points,
                vertices,
                face_adjacency_edges,
                angles,
                convex,
                candidate_edges,
                offsets,
                wp.float32(radius),
                mean_curvature,
            ],
            device=device,
        )

    return mean_curvature
