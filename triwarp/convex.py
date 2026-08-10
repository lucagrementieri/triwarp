"""
Convexity of a mesh's face adjacency, and approximate convex-hull vertices of a point cloud.

A pair of edge-adjacent faces is *convex* when each face's third vertex lies on the inner side of
the other's plane. [`face_adjacency_convex`][triwarp.convex.face_adjacency_convex] answers that per
adjacency row, and
[`face_adjacency_projections`][triwarp.convex.face_adjacency_projections] returns the signed
distances it thresholds, for callers that want the margin rather than the verdict.

The remaining entry points work on a *point cloud* rather than a mesh, and answer a different
question: which points are convex-hull vertices. Both are exact-hull-free and run in a fixed number
of parallel launches, and both are one-sided -- but in *opposite* directions, which is what the
names say:

- [`convex_subset_mask`][triwarp.convex.convex_subset_mask] (and its point-returning form
  [`convex_subset`][triwarp.convex.convex_subset]) accumulates *positive* certificates: a direction
  a point is extremal along proves it is on the hull. Cheap and precise, but a hull vertex with no
  sampled certificate is dropped, so the result can be **smaller** than the hull-vertex set.
- [`convex_superset_mask`][triwarp.convex.convex_superset_mask] accumulates *negative* certificates:
  a tetrahedron of hull points strictly containing a point proves that point is interior. It keeps
  everything it cannot rule out, so the result is **never smaller** than the hull-vertex set -- a
  conservative filter, suitable as a prefilter before an exact hull.

Neither computes hull connectivity; for that use an exact hull library
([`scipy.spatial.ConvexHull`][] or the qhull-backed mesh packages).
"""

from __future__ import annotations

import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp._device import slice_count
from triwarp.constants import TOLERANCE_MERGE_CONSTANT
from triwarp.kernels import array as kernel_array
from triwarp.kernels import convex as kernel_convex

# Relative slack, against the support extent along a direction, for recognizing a point as that
# direction's support point. It cannot be zero: the reducing kernel and the marking kernel compute
# the same dot product in different code, so FMA contraction can leave them a ULP apart and an
# exact-equality test would match nothing and fall back to an arbitrary shell vertex. It must also
# stay small -- a wide slack lets the lowest-index *near*-support point win, pulling shell vertices
# inward and costing selectivity. This is a float32 epsilon question, not a tuning knob, so it is
# not exposed.
SUPPORT_TIE_SLACK = wp.constant(wp.float32(1e-6))

# Minimum normalized determinant (against the edge-length product) for a shell tetrahedron to be
# used. Rejection is free -- neighbouring, well-shaped tetrahedra cover the same region -- while a
# sliver's face normals are ill-conditioned cross products of nearly parallel edges, and that error
# is the one that can cost the superset guarantee. Chosen well above ``TOLERANCE_PLANAR`` for that
# reason, and paired with ``convex_superset_mask``'s ``margin`` default: measured, dropping this to
# 1e-9 makes every margin in the useful range unsafe.
TETRAHEDRON_FLATNESS = wp.constant(wp.float32(1e-3))


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
        [`face_adjacency`][triwarp.adjacency.face_adjacency]).
    face_adjacency
        Optional ``(m, 2)`` face index pairs from
        [`face_adjacency`][triwarp.adjacency.face_adjacency]. When ``None``, adjacency and
        shared edges are computed from ``faces``.
    face_adjacency_edges
        Optional ``(m, 2)`` sorted shared vertex pairs (as from
        [`face_adjacency`][triwarp.adjacency.face_adjacency] with ``return_edges=True``).
        Must be supplied together with ``face_adjacency`` or omitted with it.
    face_adjacency_unshared
        Optional ``(m, 2)`` unshared vertex indices per face pair from
        [`face_adjacency_unshared`][triwarp.adjacency.face_adjacency_unshared]. When ``None``,
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
        If only one of ``face_adjacency`` and ``face_adjacency_edges`` is provided.

    See Also
    --------
    [`face_adjacency_convex`][triwarp.convex.face_adjacency_convex]
    [`trimesh.Trimesh.face_adjacency_projections`][]
    """
    device = faces.device
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return wp.empty(0, dtype=wp.float32, device=device)

    face_adjacency, face_adjacency_edges = tw.adjacency.resolved_face_adjacency(
        faces, face_adjacency, face_adjacency_edges, n_vertices=int(vertices.shape[0])
    )

    if face_adjacency_unshared is None:
        face_adjacency_unshared = tw.adjacency.face_adjacency_unshared(
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
        [`face_adjacency`][triwarp.adjacency.face_adjacency]).
    face_adjacency
        Optional ``(m, 2)`` face index pairs from
        [`face_adjacency`][triwarp.adjacency.face_adjacency]. When ``None``, adjacency and
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

    face_adjacency, face_adjacency_edges = tw.adjacency.resolved_face_adjacency(
        faces, face_adjacency, face_adjacency_edges, n_vertices=int(vertices.shape[0])
    )

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


def convex_subset_mask(
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

    The extrema are computed by one thread per ``(direction, point slice)``, each reducing a
    strided slice of [`ITEMS_PER_SLICE_CUDA`][triwarp.constants.ITEMS_PER_SLICE_CUDA] points
    (``ITEMS_PER_SLICE_CPU`` on the CPU device) and committing one atomic, then a second pass marks
    the maximizers and minimizers. At most roughly ``2 * n_directions`` points (plus ties) can be
    marked.

    !!! warning "The result is an inner approximation, not a superset"

        Marking a point requires a *certificate* -- a sampled direction it is extremal
        along -- so a hull vertex with no such direction is **dropped**. The mask is
        therefore a subset of the hull boundary, never a conservative superset of the
        hull vertices; see Notes for the exact guarantee and how to raise recall. When
        losing a hull vertex is unacceptable, use
        [`convex_superset_mask`][triwarp.convex.convex_superset_mask], which errs the
        other way by construction.

    Parameters
    ----------
    points
        ``(n_points,)`` point positions on the target device.
    n_directions
        Number of Fibonacci hemisphere directions. Each covers two antipodal
        orientations, so the effective coverage is ``2 * n_directions``. Larger
        values recover more of the hull vertices.
    tolerance
        Relative slack on the support test; a point is marked when its dot product
        with a direction is within ``tolerance * (max - min)`` of that direction's
        maximum or minimum, where ``max - min`` is the cloud's support extent along
        the direction. Scaling the slack with the extent captures coplanar ties and
        float32 roundoff at any coordinate scale.

    Returns
    -------
    wp.array[wp.bool]
        Length-``n_points`` mask on ``points.device``; ``True`` for points selected
        as approximate hull vertices. Empty when there are no points.

    Notes
    -----
    **What is guaranteed.** Every marked point lies on the convex-hull *boundary*, to
    within the ``tolerance`` slack. That is the invariant that holds for every input.
    The stronger statement -- every marked point is a hull *vertex* -- holds only when
    no support direction ties, which is the generic case for a cloud in general
    position but fails on structured data: on a 5x5 grid over each face of a cube
    (98 points, 8 hull vertices) the mask returns 14 points at any ``n_directions``,
    the 8 corners plus 6 face-edge midpoints that tie with a corner along a face
    normal. Raising ``tolerance`` widens that effect deliberately.

    **What is not guaranteed.** A hull vertex is recovered only when a sampled
    direction falls inside its *normal cone*, so recall degrades with the flatness of
    the hull around a vertex, not with the point count. On 500 standard-normal points
    (31 hull vertices) the recovered fraction measures 0.61 at ``n_directions=32``,
    0.81 at 128, 0.87 at 256, 0.94 at 512 and 1.00 at 16384; the four vertices missed
    at 256 have normal cones spanning 3e-5 to 2e-3 of the sphere, so 512 antipodal
    orientations are expected to hit them 0.02 to 1.1 times. Increasing
    ``n_directions`` is the only knob that raises recall -- ``tolerance`` trades
    precision for it and is not a substitute. When *all* hull vertices are required,
    use an exact hull ([`scipy.spatial.ConvexHull`][] and the qhull-backed mesh
    libraries): no setting of these parameters makes this function conservative.

    See Also
    --------
    [`convex_subset`][triwarp.convex.convex_subset]
    [`convex_superset_mask`][triwarp.convex.convex_superset_mask]
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
    best_max, best_min = _support_extremes(points, directions)

    out_mask = wp.zeros(n_points, dtype=wp.bool, device=device)
    wp.launch(
        kernel_convex.mark_hull_support,
        dim=(n_dir, n_points),
        inputs=[points, directions, best_max, best_min, wp.float32(tolerance), out_mask],
        device=device,
    )
    return out_mask


def convex_subset(
    points: wp.array[wp.vec3], n_directions: int = 128, tolerance: float = 1e-6
) -> wp.array[wp.vec3]:
    """
    Approximate the convex-hull vertices of a point cloud as a point subset.

    Convenience wrapper around
    [`convex_subset_mask`][triwarp.convex.convex_subset_mask] that returns the
    selected points directly, gathered from ``points`` in ascending index order. The
    accuracy is the mask's: the returned subset lies on the hull boundary but can omit
    hull vertices, and is not a conservative superset of them.

    Parameters
    ----------
    points
        ``(n_points,)`` point positions on the target device.
    n_directions
        Number of Fibonacci hemisphere directions (see
        [`convex_subset_mask`][triwarp.convex.convex_subset_mask]).
    tolerance
        Relative slack on the support test (see
        [`convex_subset_mask`][triwarp.convex.convex_subset_mask]).

    Returns
    -------
    wp.array[wp.vec3]
        The subset of ``points`` selected as approximate hull vertices, on
        ``points.device``. Empty when there are no points.

    See Also
    --------
    [`convex_subset_mask`][triwarp.convex.convex_subset_mask]
    [`gather`][triwarp.array.gather]
    [`scipy.spatial.ConvexHull`][]
    """
    mask = convex_subset_mask(points, n_directions=n_directions, tolerance=tolerance)
    indices = tw.array.flatnonzero(mask)
    return tw.array.gather(points, indices)


def convex_superset_mask(
    points: wp.array[wp.vec3], subdivisions: int = 2, margin: float = 1e-5
) -> wp.array[wp.bool]:
    """
    Conservatively discard interior points, keeping a superset of the convex-hull vertices.

    Unlike [`convex_subset_mask`][triwarp.convex.convex_subset_mask], which keeps only points it can
    *prove* are on the hull, this keeps every point it cannot prove is *interior*. The certificate
    is a tetrahedron: the support point of the cloud along each direction of an
    [`icosphere`][triwarp.creation.icosphere] is a hull vertex, so for every triangle ``(a, b, c)``
    of the icosphere the tetrahedron ``(centroid, s_a, s_b, s_c)`` has all four corners in the hull
    and therefore lies inside it. Any point strictly inside such a tetrahedron is strictly inside
    the hull and cannot be a hull vertex; every other point is kept.

    Because a hull vertex lies on the hull *boundary*, it is in no tetrahedron's strict interior, so
    **no hull vertex is ever discarded** -- for any input, any ``subdivisions``, and any ``margin``.
    The parameters trade only how many interior points survive. That makes this a prefilter: run it,
    then hand the survivors to an exact hull, which then does its superlinear work on a small
    fraction of the cloud.

    Cost is one support sweep (``n_points`` times the ``10 * 4 ** subdivisions + 2`` icosphere
    directions) plus one pass testing each point against the ``20 * 4 ** subdivisions``
    tetrahedra, whose precomputed face planes every thread reads in lockstep.

    Parameters
    ----------
    points
        ``(n_points,)`` point positions on the target device.
    subdivisions
        Icosphere refinement level for the direction set, which also fixes the tetrahedron count.
        Higher values wrap the hull more tightly and so discard more interior points, at a
        proportionally higher cost; the guarantee is unaffected.
    margin
        Distance, as a fraction of the shell radius, by which a point must clear all four faces of a
        tetrahedron to count as strictly inside it. Raising it keeps *more* points, so it can only
        weaken the filter, never the guarantee. The default is set by float32 arithmetic rather than
        by taste: a point lying *exactly* on a face must not be read as inside. See Notes.

    Returns
    -------
    wp.array[wp.bool]
        Length-``n_points`` mask on ``points.device``; ``True`` for points that may be hull
        vertices, which includes every actual hull vertex. Empty when there are no points.

    Notes
    -----
    Measured selectivity on 200k points (``subdivisions=3``), against the exact hull-vertex count
    from [`scipy.spatial.ConvexHull`][]: standard normal, 88 hull vertices, 206 kept (0.10% of the
    cloud); uniform in a ball, 2060 hull vertices, 3411 kept (1.7%); uniform in a cube, 223 hull
    vertices, 1834 kept (0.9%). Flat-faced clouds are the weak case -- a triangulated inner shell
    cannot hug a plane, so the survivors form a thin slab under each face -- and near-spherical
    clouds are the strong one.

    The guarantee is exact in real arithmetic; in float32 it rests on ``margin`` covering the error
    in the plane evaluation, so the default was measured rather than picked. Over 192 cases (six
    distributions x eight seeds x ``subdivisions`` 0-3, 4k points, checked against
    [`scipy.spatial.ConvexHull`][]), ``margin=1e-8`` discards a true hull vertex in 119 of them and
    ``1e-7`` in 2 -- always a point lying *essentially exactly* on a tetrahedron face, where the
    computed distance straddles zero -- while everything from ``1e-6`` up is clean. The ``1e-5``
    default therefore sits 100x above the largest margin observed to fail, and costs 0.8 percentage
    points of selectivity against the unsafe floor. ``TETRAHEDRON_FLATNESS`` is the other half of
    the same protection: it discards sliver tetrahedra whose face normals are too ill-conditioned to
    trust, and without it no margin in this range is safe.

    Measuring the margin as a *distance* is what makes that trade cheap. The obvious alternative --
    barycentric coordinates from an inverse of the tetrahedron's edge matrix, one matrix-vector
    product instead of four plane evaluations -- needs a barycentric slack of ``1e-3`` for the same
    safety, because these tetrahedra run from the centroid out to the shell and a fixed barycentric
    slack cuts a thick layer off the base while cutting nothing off the sides. Measured on 200k
    standard-normal points, that formulation keeps 4035 points where this one keeps 206.

    Degenerate input is handled by the same conservative logic rather than by a special case. A
    coplanar or collinear cloud makes every tetrahedron flat; flat tetrahedra are rejected as
    ill-conditioned, so nothing is certified interior and every point is kept, which is a valid
    (if useless) superset. Fewer than four points returns an all-``True`` mask directly.

    See Also
    --------
    [`convex_subset_mask`][triwarp.convex.convex_subset_mask]
    [`icosphere`][triwarp.creation.icosphere]
    [`flatnonzero`][triwarp.array.flatnonzero]
    [`scipy.spatial.ConvexHull`][]
    """
    device = points.device
    n_points = int(points.shape[0])
    if n_points == 0:
        return wp.empty(0, dtype=wp.bool, device=device)
    if n_points < 4:
        # No tetrahedron exists, so nothing can be certified interior.
        return wp.full(n_points, value=True, dtype=wp.bool, device=device)

    directions, shell_faces = tw.creation.icosphere(subdivisions=int(subdivisions), device=device)
    n_dir = int(directions.shape[0])
    n_tetra = int(shell_faces.shape[0]) // 3
    best_max, best_min = _support_extremes(points, directions)

    # Seeded with the last index rather than a sentinel: a direction that somehow marks nothing
    # then yields a real point, which keeps the gather in range and the tetrahedra valid.
    support = wp.full(n_dir, value=n_points - 1, dtype=wp.int32, device=device)
    wp.launch(
        kernel_convex.support_indices,
        dim=(n_dir, n_points),
        inputs=[points, directions, best_max, best_min, SUPPORT_TIE_SLACK, support],
        device=device,
    )

    shell_vertices = tw.array.gather(points, support)
    centroid = wp.empty(1, dtype=wp.vec3, device=device)
    radius = wp.empty(1, dtype=wp.float32, device=device)
    wp.launch(
        kernel_convex.shell_bounds, dim=1, inputs=[shell_vertices, centroid, radius], device=device
    )

    planes = wp.zeros((n_tetra, 4), dtype=wp.vec4, device=device)
    valid = wp.empty(n_tetra, dtype=wp.bool, device=device)
    wp.launch(
        kernel_convex.tetrahedron_planes,
        dim=n_tetra,
        inputs=[shell_vertices, shell_faces, centroid, TETRAHEDRON_FLATNESS, planes, valid],
        device=device,
    )

    out_mask = wp.empty(n_points, dtype=wp.bool, device=device)
    wp.launch(
        kernel_convex.mark_hull_superset,
        dim=n_points,
        inputs=[points, planes, valid, radius, wp.float32(margin), out_mask],
        device=device,
    )
    return out_mask


def _support_extremes(
    points: wp.array[wp.vec3], directions: wp.array[wp.vec3]
) -> tuple[wp.array[wp.float32], wp.array[wp.float32]]:
    """
    Per-direction maximum and minimum of the support function over ``points``.

    Each thread reduces a strided slice of the cloud, so the launch is sized by
    [`items_per_slice`][triwarp._device.items_per_slice] points per thread rather than by the point
    count -- enough parallelism to fill the device while keeping the number of atomics into the
    ``n_directions`` accumulator slots low. This is the only reduction of this shape that still runs
    the strided form on CUDA, which is why the slice length is chosen per device.
    """
    device = points.device
    n_points = int(points.shape[0])
    n_dir = int(directions.shape[0])
    n_slices = slice_count(n_points, device)

    best_max = wp.full(n_dir, value=-float("inf"), dtype=wp.float32, device=device)
    best_min = wp.full(n_dir, value=float("inf"), dtype=wp.float32, device=device)
    wp.launch(
        kernel_convex.hull_support_extremes,
        dim=(n_dir, n_slices),
        inputs=[points, directions, n_slices, best_max, best_min],
        device=device,
    )
    return best_max, best_min
