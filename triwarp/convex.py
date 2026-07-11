from __future__ import annotations

import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.constants import TILE_1D, TOLERANCE_MERGE_CONSTANT
from triwarp.kernels import array as kernel_array
from triwarp.kernels import convex as kernel_convex


def face_adjacency_projections(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_adjacency: twt.Array2dInt32 | None = None,
    face_adjacency_edges: twt.Array2dInt32 | None = None,
    face_adjacency_unshared: twt.Array2dInt32 | None = None,
    face_normals: wp.array[wp.vec3] | None = None,
) -> wp.array[wp.float32]:
    """
    Project each adjacent face pair's non-shared vertex onto the first face plane.

    For each row of ``face_adjacency``, the dot product is taken between the
    normal of face ``face_adjacency[k, 0]`` and the vector from one endpoint of
    the shared edge to the unshared vertex on ``face_adjacency[k, 1]``.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        Length-``3 * n_faces`` flat triangle index buffer (same layout as
        [`face_adjacency`][triwarp.graph.face_adjacency]).
    face_adjacency
        Optional ``(m, 2)`` face index pairs from
        [`face_adjacency`][triwarp.graph.face_adjacency]. When ``None``, adjacency and
        shared edges are computed from ``faces``.
    face_adjacency_edges
        Optional ``(m, 2)`` sorted shared vertex pairs (as from
        [`face_adjacency`][triwarp.graph.face_adjacency] with ``return_edges=True``).
        Must be supplied together with ``face_adjacency`` or omitted with it.
    face_adjacency_unshared
        Optional ``(m, 2)`` unshared vertex indices per face pair from
        [`face_adjacency_unshared`][triwarp.graph.face_adjacency_unshared]. When ``None``,
        computed from ``faces`` and the adjacency data.
    face_normals
        Optional length-``n_faces`` unit face normals. When ``None``, normals
        are computed from ``vertices`` and ``faces`` via
        [`face_normals_and_areas`][triwarp.triangles.face_normals_and_areas].

    Returns
    -------
    wp.array[wp.float32]
        Length ``m`` projections on ``faces.device``, one per ``face_adjacency``
        row. Empty when there are no faces or no adjacency pairs.

    Raises
    ------
    ValueError
        If ``vertices`` and ``faces`` live on different devices, or if only one
        of ``face_adjacency`` and ``face_adjacency_edges`` is provided.

    See Also
    --------
    [`face_adjacency_convex`][triwarp.convex.face_adjacency_convex]
    [`trimesh.Trimesh.face_adjacency_projections`][]
    """
    device = faces.device
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return wp.empty(0, dtype=wp.float32, device=device)

    if (face_adjacency is None) != (face_adjacency_edges is None):
        raise ValueError(
            "face_adjacency and face_adjacency_edges must both be provided or both omitted"
        )
    if face_adjacency is None:
        face_adjacency, face_adjacency_edges = tw.graph.face_adjacency(faces, return_edges=True)
    assert face_adjacency is not None
    assert face_adjacency_edges is not None

    if face_adjacency_unshared is None:
        face_adjacency_unshared = tw.graph.face_adjacency_unshared(
            faces, face_adjacency=face_adjacency, face_adjacency_edges=face_adjacency_edges
        )
    if face_normals is None:
        face_normals, _ = tw.triangles.face_normals_and_areas(vertices, faces)

    m = int(face_adjacency.shape[0])
    if m == 0:
        return wp.empty(0, dtype=wp.float32, device=device)

    out_projections = wp.empty(m, dtype=wp.float32, device=device)
    wp.launch(
        kernel_convex.face_adjacency_projections,
        dim=m,
        inputs=[
            vertices,
            face_normals,
            face_adjacency,
            face_adjacency_edges,
            face_adjacency_unshared,
            out_projections,
        ],
        device=device,
    )
    return out_projections


def face_adjacency_convex(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    face_adjacency: twt.Array2dInt32 | None = None,
    face_adjacency_edges: twt.Array2dInt32 | None = None,
    face_adjacency_unshared: twt.Array2dInt32 | None = None,
    face_normals: wp.array[wp.vec3] | None = None,
) -> wp.array[wp.bool]:
    """
    Return face pairs that are adjacent and locally convex.

    A pair is locally convex when the unshared vertex of the second face,
    projected onto the plane of the first face, has a projection less than
    [`TOLERANCE_MERGE`][triwarp.constants.TOLERANCE_MERGE].

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        Length-``3 * n_faces`` flat triangle index buffer (same layout as
        [`face_adjacency`][triwarp.graph.face_adjacency]).
    face_adjacency
        Optional ``(m, 2)`` face index pairs from
        [`face_adjacency`][triwarp.graph.face_adjacency]. When ``None``, adjacency and
        shared edges are computed from ``faces``.
    face_adjacency_edges
        Optional ``(m, 2)`` sorted shared vertex pairs. Must be supplied
        together with ``face_adjacency`` or omitted with it.
    face_adjacency_unshared
        Optional ``(m, 2)`` unshared vertex indices per face pair.
    face_normals
        Optional length-``n_faces`` unit face normals.

    Returns
    -------
    wp.array[wp.bool]
        Length ``m`` boolean mask on ``faces.device``, one per
        ``face_adjacency`` row. Empty when there are no faces or no adjacency
        pairs.

    See Also
    --------
    [`face_adjacency_projections`][triwarp.convex.face_adjacency_projections]
    [`trimesh.Trimesh.face_adjacency_convex`][]
    """
    device = faces.device
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return wp.empty(0, dtype=wp.bool, device=device)

    if (face_adjacency is None) != (face_adjacency_edges is None):
        raise ValueError(
            "face_adjacency and face_adjacency_edges must both be provided or both omitted"
        )
    if face_adjacency is None:
        face_adjacency, face_adjacency_edges = tw.graph.face_adjacency(faces, return_edges=True)
    assert face_adjacency is not None

    m = int(face_adjacency.shape[0])
    if m == 0:
        return wp.empty(0, dtype=wp.bool, device=device)

    projections = face_adjacency_projections(
        vertices,
        faces,
        face_adjacency=face_adjacency,
        face_adjacency_edges=face_adjacency_edges,
        face_adjacency_unshared=face_adjacency_unshared,
        face_normals=face_normals,
    )
    out_convex = wp.empty(m, dtype=wp.bool, device=device)
    wp.map(kernel_array.less, projections, TOLERANCE_MERGE_CONSTANT, out=out_convex)
    return out_convex


def fast_convex_set_mask(
    points: wp.array[wp.vec3], n_directions: int = 128, tolerance: float = 1e-6
) -> wp.array[wp.bool]:
    """
    Approximate the convex-hull vertices of a point cloud as a boolean mask.

    For any direction ``n``, the point maximizing ``⟨n, p⟩`` is a vertex of the
    convex hull, and the point minimizing it is the hull vertex farthest along
    ``-n``. A single dot-product sweep over the points therefore yields the two
    hull vertices supporting ``+n`` and ``-n``. Directions are drawn on the
    positive-``z`` hemisphere with the deterministic Fibonacci spiral
    ([`sample_fibonacci_hemisphere`][triwarp.sample.sample_fibonacci_hemisphere]);
    because the hemisphere and its reflection tile the full sphere, taking both
    the max and min per direction covers all ``2 * n_directions`` antipodal
    orientations at half the dot-product cost of sampling the full sphere.

    Marking every support extremum yields an approximation of the hull-vertex set
    with no false positives: every marked point is a true hull vertex, and coverage
    of the full vertex set improves as ``n_directions`` grows. The extrema are
    computed with a tiled block reduction (``wp.tile`` /
    [`TILE_1D`][triwarp.constants.TILE_1D]-wide ``wp.tile_max`` and ``wp.tile_min``),
    then a second pass marks the maximizers and minimizers.

    Parameters
    ----------
    points
        ``(n_points,)`` point positions on the target device.
    n_directions
        Number of Fibonacci hemisphere directions. Each covers two antipodal
        orientations, so the effective coverage is ``2 * n_directions``. Larger
        values recover more of the hull vertices.
    tolerance
        Absolute slack on the support test; a point is marked when its dot product
        with a direction is within ``tolerance`` of that direction's maximum or
        minimum. This captures coplanar ties and floating-point jitter. For
        widely-scaled data, scale this with the coordinate magnitude.

    Returns
    -------
    wp.array[wp.bool]
        Length-``n_points`` mask on ``points.device``; ``True`` for points selected
        as approximate hull vertices. Empty when there are no points.

    See Also
    --------
    [`fast_convex_set`][triwarp.convex.fast_convex_set]
    [`sample_fibonacci_hemisphere`][triwarp.sample.sample_fibonacci_hemisphere]
    [`flatnonzero`][triwarp.array.flatnonzero]
    [`scipy.spatial.ConvexHull`][]
    """
    device = points.device
    n_points = int(points.shape[0])
    if n_points == 0:
        return wp.empty(0, dtype=wp.bool, device=device)

    n_dir = int(n_directions)
    directions = tw.sample.sample_fibonacci_hemisphere(n_dir, device=device)

    best_max = wp.full(n_dir, value=-float("inf"), dtype=wp.float32, device=device)
    best_min = wp.full(n_dir, value=float("inf"), dtype=wp.float32, device=device)
    n_point_tiles = (n_points + TILE_1D - 1) // TILE_1D
    wp.launch_tiled(
        kernel_convex.hull_support_extremes,
        dim=[n_dir, n_point_tiles],
        inputs=[points, directions, n_points, best_max, best_min],
        block_dim=TILE_1D,
        device=device,
    )

    out_mask = wp.zeros(n_points, dtype=wp.bool, device=device)
    wp.launch_tiled(
        kernel_convex.mark_hull_support,
        dim=[n_dir, n_point_tiles],
        inputs=[points, directions, best_max, best_min, wp.float32(tolerance), n_points, out_mask],
        block_dim=TILE_1D,
        device=device,
    )
    return out_mask


def fast_convex_set(
    points: wp.array[wp.vec3], n_directions: int = 128, tolerance: float = 1e-6
) -> wp.array[wp.vec3]:
    """
    Approximate the convex-hull vertices of a point cloud as a point subset.

    Convenience wrapper around
    [`fast_convex_set_mask`][triwarp.convex.fast_convex_set_mask] that returns the
    selected points directly, gathered from ``points`` in ascending index order.

    Parameters
    ----------
    points
        ``(n_points,)`` point positions on the target device.
    n_directions
        Number of Fibonacci hemisphere directions (see
        [`fast_convex_set_mask`][triwarp.convex.fast_convex_set_mask]).
    tolerance
        Absolute slack on the support test (see
        [`fast_convex_set_mask`][triwarp.convex.fast_convex_set_mask]).

    Returns
    -------
    wp.array[wp.vec3]
        The subset of ``points`` selected as approximate hull vertices, on
        ``points.device``. Empty when there are no points.

    See Also
    --------
    [`fast_convex_set_mask`][triwarp.convex.fast_convex_set_mask]
    [`gather`][triwarp.array.gather]
    [`scipy.spatial.ConvexHull`][]
    """
    mask = fast_convex_set_mask(points, n_directions=n_directions, tolerance=tolerance)
    indices = tw.array.flatnonzero(mask)
    return tw.array.gather(points, indices)
