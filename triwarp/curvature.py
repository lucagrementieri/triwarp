import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.kernels import array as kernel_array
from triwarp.kernels import curvature as kernel_curvature
from triwarp.vertices import vertex_defects


def discrete_gaussian_curvature(
    points: wp.array[wp.vec3],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_angles: twt.Array2dFloat32,
    radius: float,
) -> wp.array[wp.float32]:
    """
    Return the discrete Gaussian curvature measure of a sphere centered at each query point.

    As detailed in Cohen-Steiner and Morvan, "Restricted Delaunay triangulations and
    normal cycle". This is the sum of vertex defects at all vertices within the radius
    for each point.

    Parameters
    ----------
    points
        ``(n,)`` query positions in space as ``wp.vec3``.
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        Length-``3 * n_faces`` flat triangle index buffer.
    face_angles
        ``(n_faces, 3)`` interior angles per face (from
        :func:`triwarp.triangles.face_angles`).
    radius
        Sphere radius; may be zero when ``vertices`` are the query points.

    Returns
    -------
    wp.array[wp.float32]
        Length ``n`` discrete Gaussian curvature measure on ``points.device``.
    """
    nearest_indices, _, nearest_offsets = tw.proximity.query_hashgrid_ball_with_offsets(
        vertices, points, radius
    )
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
    Return the discrete mean curvature measure of a sphere centered at each query point.

    As detailed in Cohen-Steiner and Morvan, "Restricted Delaunay triangulations and
    normal cycle". This is the sum of edge angles contained in the sphere for each point.

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

    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return wp.zeros(n_points, dtype=wp.float32, device=device)

    if (face_adjacency is None) != (face_adjacency_edges is None):
        raise ValueError(
            "face_adjacency and face_adjacency_edges must both be provided or both omitted"
        )
    if face_adjacency is None:
        face_adjacency, face_adjacency_edges = tw.graph.face_adjacency(faces, return_edges=True)
    assert face_adjacency is not None
    assert face_adjacency_edges is not None

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

    bvh = tw.proximity.bvh_from_bounds(edge_lower, edge_upper)
    candidate_edges, offsets = tw.proximity.query_bvh_aabb_with_offsets(bvh, points, radius)

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
