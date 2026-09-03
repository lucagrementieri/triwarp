"""
Discrete curvature: principal directions and magnitudes, Gaussian and mean.

Three quantities computed three different ways, because on a mesh "curvature" is not one thing:

- [`principal_curvature`][triwarp.curvature.principal_curvature] fits a quadric to a geodesic-ball
  neighbourhood of each vertex and reads the shape operator off it, giving both principal
  directions and magnitudes. This is the extrinsic, neighbourhood-scale answer, and the only one
  with a ``radius`` to tune (``igl::principal_curvature``).
- [`discrete_gaussian_curvature`][triwarp.curvature.discrete_gaussian_curvature] is the angle
  defect: purely *intrinsic*, exact rather than fitted, and computable from the angles alone.
- [`discrete_mean_curvature`][triwarp.curvature.discrete_mean_curvature] sums dihedral angle times
  edge length over the one ring, which is the integrated mean curvature rather than a pointwise one.

The neighbourhood the first one fits over is a *geodesic* ball
([`geodesic_ball`][triwarp.neighbors.geodesic_ball]), not a Euclidean one: a spatial query would
pull in vertices across a fold of the surface and corrupt the fit.
"""

import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.kernels import curvature as kernel_curvature
from triwarp.kernels import edges as kernel_edges
from triwarp.kernels import scatter as kernel_scatter
from triwarp.vertices import vertex_defects
from triwarp.vertices import vertex_normals as _vertex_normals


def principal_curvature(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    radius: int = 5,
    *,
    frame_independent: bool = True,
    face_normals: wp.array[wp.vec3] | None = None,
    face_areas: wp.array[wp.float32] | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.vec3], wp.array[wp.float32], wp.array[wp.float32]]:
    """
    Principal curvature directions and magnitudes per vertex via quadric fitting.

    For each vertex a quadric surface is fitted to a sphere-neighborhood of vertices in the
    local tangent frame. The principal curvatures and directions are extracted from the
    eigendecomposition of the resulting shape operator (Weingarten map), built over a
    sphere-search neighborhood of radius ``radius * avg_edge_length``.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions as ``wp.vec3``.
    faces
        Length-``3 * n_faces`` flat triangle index buffer as ``wp.int32``.
    radius
        Neighborhood size multiplier applied to the average edge length. Larger values
        collect more neighbors and produce smoother curvature estimates.
    frame_independent
        When ``True`` (default), the principal curvatures are the eigenvalues of the true
        (textbook) Weingarten map ``II*v = lam*I*v``; these are surface invariants and do not
        depend on the chosen tangent frame. When ``False``, the symmetrized shape operator of
        ``igl::principal_curvature`` is reproduced verbatim (frame-dependent), matching
        ``igl.principal_curvature`` exactly. The two agree closely on well-sampled smooth
        surfaces; they differ only in the off-diagonal of the shape operator.
    face_normals
        Optional length-``n_faces`` unit face normals and matching areas from
        [`face_normals_and_areas`][triwarp.triangles.face_normals_and_areas]; recomputed together
        when either is ``None``. [`Trimesh.face_normals`][triwarp.mesh.Trimesh.face_normals] and
        [`Trimesh.face_areas`][triwarp.mesh.Trimesh.face_areas] cache the pair.
    face_areas
        See ``face_normals``.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.vec3], wp.array[wp.float32], wp.array[wp.float32]]
        ``(PD1, PD2, PV1, PV2)`` where ``PV1 >= PV2`` at every vertex. Vertices for which
        the quadric fit failed (fewer than 6 neighbors or degenerate system) have zero
        directions and zero curvature values.

    Notes
    -----
    A failed fit is signalled by that all-zero output and nothing else -- there is no separate
    validity mask. The two states it conflates are a failed fit and a genuinely flat vertex, which
    is why the test is worth stating: ``PD1`` is zero only where the fit did not produce a frame.
    """
    device = vertices.device
    n_vertices = int(vertices.shape[0])

    # Compute vertex normals via face normals
    if face_normals is None or face_areas is None:
        face_normals, face_areas = tw.triangles.face_normals_and_areas(vertices, faces)
    vertex_normals = _vertex_normals(
        vertices, faces, face_normals=face_normals, face_weights=face_areas
    )

    # ``mean_edge_length`` is the per-face average, matching libigl's
    # ``CurvatureCalculator::getAverageEdge`` -- the one ``igl::principal_curvature`` uses to set
    # ``scaledRadius``. Not ``mean_unique_edge_length``: switching to it takes the mean deviation
    # from ``igl.principal_curvature`` on an open half-torus from 0.0056 to 0.0117 and drops the
    # within-5% fraction from 99.3% to 98.5%.
    avg_edge = tw.edges.mean_edge_length(vertices, faces)
    scaled_radius = float(radius) * avg_edge

    # Collect vertex neighborhoods as geodesic balls (libigl getSphere) on device. A Euclidean ball
    # would pull in vertices across surface folds and corrupt the quadric fit; see
    # tw.neighbors.geodesic_ball.
    neighbor_indices, offsets, reference_neighbors = tw.neighbors.geodesic_ball(
        vertices, faces, scaled_radius
    )

    # Fit quadric and extract principal curvature per vertex
    pd1 = wp.zeros(n_vertices, dtype=wp.vec3, device=device)
    pd2 = wp.zeros(n_vertices, dtype=wp.vec3, device=device)
    pv1 = wp.zeros(n_vertices, dtype=wp.float32, device=device)
    pv2 = wp.zeros(n_vertices, dtype=wp.float32, device=device)

    wp.launch(
        kernel_curvature.fit_principal_curvature,
        dim=n_vertices,
        inputs=[
            vertices,
            vertex_normals,
            neighbor_indices,
            offsets,
            reference_neighbors,
            frame_independent,
            pd1,
            pd2,
            pv1,
            pv2,
        ],
        device=device,
    )

    return pd1, pd2, pv1, pv2


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
        [`face_angles`][triwarp.triangles.face_angles]).
    radius
        Sphere radius; may be zero when ``vertices`` are the query points.

    Returns
    -------
    wp.array[wp.float32]
        Length ``n`` discrete Gaussian curvature measure on ``points.device``.

    See Also
    --------
    [`vertex_defects`][triwarp.vertices.vertex_defects]
        The pointwise angle defect this integrates. Same quantity, no scale: at ``radius = 0`` with
        the vertices as query points the two agree, and it is the ball measure rather than the
        pointwise defect that converges under refinement.
    [`discrete_mean_curvature`][triwarp.curvature.discrete_mean_curvature]
        The mean-curvature measure over the same ball.
    """
    nearest_indices, _, nearest_offsets = tw.neighbors.query_ball_with_offsets(
        vertices, points, radius
    )
    defects = vertex_defects(vertices.shape[0], faces, face_angles)
    gauss_curvature = wp.zeros(int(points.shape[0]), dtype=wp.float32, device=points.device)
    wp.launch(
        kernel_scatter.SCATTER_OFFSET_SUM[defects.dtype],
        dim=int(nearest_indices.shape[0]),
        inputs=[defects, nearest_indices, nearest_offsets, gauss_curvature],
        device=points.device,
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
        [`face_adjacency`][triwarp.adjacency.face_adjacency]. When ``None``, adjacency and
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
    [`discrete_gaussian_curvature`][triwarp.curvature.discrete_gaussian_curvature]
    [`trimesh.curvature.discrete_mean_curvature_measure`][]
    """
    device = points.device
    n_points = int(points.shape[0])
    if n_points == 0:
        return wp.empty(0, dtype=wp.float32, device=device)

    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return wp.zeros(n_points, dtype=wp.float32, device=device)

    face_adjacency, face_adjacency_edges = tw.adjacency.resolve_face_adjacency(
        faces, face_adjacency, face_adjacency_edges, n_vertices=int(vertices.shape[0])
    )

    m = int(face_adjacency.shape[0])
    if m == 0:
        return wp.zeros(n_points, dtype=wp.float32, device=device)

    angles = tw.adjacency.face_adjacency_angles(vertices, faces, face_adjacency=face_adjacency)
    convex = tw.adjacency.face_adjacency_convex(
        vertices, faces, face_adjacency=face_adjacency, face_adjacency_edges=face_adjacency_edges
    )

    edge_lower = wp.empty(m, dtype=wp.vec3, device=device)
    edge_upper = wp.empty(m, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_edges.edge_aabb_bounds,
        dim=m,
        inputs=[vertices, face_adjacency_edges, edge_lower, edge_upper],
        device=device,
    )

    bvh = tw.neighbors.bvh_from_bounds(edge_lower, edge_upper)
    # The broad phase is the **cube** ``[q +- radius]`` while the narrow phase below keeps only the
    # edge length inside the *ball* of that radius, so every candidate the cube admits and the ball
    # rejects is a wasted slot in the flat buffer and a wasted narrow-phase test. Measured rather
    # than inferred from the 6/pi volume ratio, which overstates it: on ``icosphere(4)`` /
    # ``icosphere(6)`` at radius scales 0.5 / 1.0 / 2.0 of the mean edge, **70.7-76.6 %** of
    # candidates meet the ball, i.e. ~**29 %** waste and not the 48 % a volume argument predicts --
    # the BVH holds edge *AABBs*, so a box near the cube's corner usually still overlaps the ball.
    #
    # ``wp.bvh_query_sphere`` (Warp 1.17) is the tighter broad phase and would remove that 29 %,
    # which shows up mostly on the radius axis where the cost grows (1.795 -> 3.927 ms on
    # ``icosphere(4)``, 1.977 -> 4.308 on ``icosphere(6)``). Not built: it needs a ball variant of
    # ``query_bvh_aabb_with_offsets``' count and emit kernels, and that function's *name* says aabb,
    # so a shape selector renames a public entry point and drags section 14's five move artifacts
    # with it -- against ~1 ms of the loss table. Sized here so the next pass starts from the
    # number rather than the volume ratio.
    candidate_edges, offsets = tw.neighbors.query_bvh_aabb_with_offsets(bvh, points, radius)

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
