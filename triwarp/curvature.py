import warp as wp

from triwarp.kernels import array as kernel_array
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
    nearest_indices, _, nearest_offsets = tw.points.query_ball_with_offsets(vertices, points, radius)
    defects = vertex_defects(vertices.shape[0], faces, face_angles)
    gauss_curvature = wp.zeros(points.shape[0], dtype=wp.float32, device=points.device)
    wp.launch(
        kernel_array.scatter_offset_sum,
        dim=nearest_indices.shape[0],
        inputs=[defects, nearest_indices, nearest_offsets, gauss_curvature],
    )
    return gauss_curvature
