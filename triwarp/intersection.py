"""
Cross-sections and level sets: where a mesh meets a plane, another mesh, or an isovalue.

A plane section and an isocontour are the same operation seen twice: intersecting a mesh with a
plane is exactly extracting the zero level set of that plane's signed distance. So
[`mesh_with_plane`][triwarp.intersection.mesh_with_plane] and
[`marching_triangles`][triwarp.intersection.marching_triangles] sit side by side here — the first
takes the field implicitly as a plane, the second takes any per-vertex scalar field — and both emit
the [`triwarp.polyline`][triwarp.polyline] convention: one array of points per curve, closed curves
not repeating their first point, and a parallel list of closed flags.

[`mesh_with_mesh`][triwarp.intersection.mesh_with_mesh] is the genuine intersection in the set
sense, and [`slice_mesh_with_plane`][triwarp.intersection.slice_mesh_with_plane] keeps the cut
geometry rather than the curve.

Every entry point here that takes a plane takes it as ``(plane_normal, plane_origin)``, in that
order, and so does every plane argument elsewhere in the package. Both are ``wp.vec3``, so a
transposed call type-checks and sections the wrong plane -- pass them by keyword where the call site
is not obvious. [`points.fit_plane`][triwarp.points.fit_plane] returns the pair in this order, so
its result splats straight into any of them.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp._device import require_nonempty_mesh
from triwarp.constants import TOLERANCE_MERGE
from triwarp.kernels import intersection as kernel_intersections
from triwarp.kernels import triangles as kernel_triangles


def segments_with_plane(
    start_points: wp.array[wp.vec3],
    end_points: wp.array[wp.vec3],
    plane_normal: wp.vec3,
    plane_origin: wp.vec3,
    *,
    line_segments: bool = True,
) -> tuple[wp.array[wp.vec3], wp.array[wp.bool]]:
    """
    Calculate plane-line intersections for batched segment endpoints.

    Each row pair ``(start_points[i], end_points[i])`` defines one line to test.
    Matches [`trimesh.intersections.plane_lines`][] with Trimesh's ``(2, n, 3)``
    layout expressed as two length-``n`` ``wp.vec3`` arrays.

    Parameters
    ----------
    plane_normal
        Plane normal vector.
    plane_origin
        Point on the plane.
    start_points
        ``(n,)`` first endpoint of each segment.
    end_points
        ``(n,)`` second endpoint of each segment.
    line_segments
        When ``True``, only mark intersections valid if endpoints lie on
        different sides of the plane.

    Returns
    -------
    intersections
        ``(n,)`` intersection points (undefined where ``valid`` is ``False``).
    valid
        ``(n,)`` mask indicating a valid intersection per segment.

    Raises
    ------
    ValueError
        If ``start_points`` and ``end_points`` do not have the same shape.
    """
    if start_points.shape != end_points.shape:
        raise ValueError("start_points and end_points must have the same shape")
    n = int(start_points.shape[0])
    device = start_points.device
    intersections = wp.empty(n, dtype=wp.vec3, device=device)
    valid = wp.empty(n, dtype=wp.bool, device=device)
    if n == 0:
        return intersections, valid

    wp.map(
        kernel_intersections.plane_with_line,
        plane_normal,
        plane_origin,
        start_points,
        end_points,
        wp.bool(line_segments),
        out=[intersections, valid],
    )
    return intersections, valid


def mesh_with_plane(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    plane_normal: wp.vec3,
    plane_origin: wp.vec3,
    *,
    return_faces: bool = False,
) -> wp.array[wp.vec3] | tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Intersect a mesh with a plane, returning line segments on the plane.

    Matches [`trimesh.intersections.mesh_plane`][] for indexed triangle meshes.
    To section a face subset, extract a submesh first (e.g.
    [`submesh_from_face_indices`][triwarp.selection.submesh_from_face_indices]).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` flat triangle index buffer.
    plane_normal
        Normal vector of the plane.
    plane_origin
        Point on the plane.
    return_faces
        If ``True``, also return the source face index for each segment.

    Returns
    -------
    lines
        ``(m, 2)`` ``wp.vec3`` array of segment endpoints (logical shape ``(m, 2, 3)``).
    face_index
        Returned only when ``return_faces=True``; ``(m,)`` ``wp.int32`` source
        face indices into the mesh.
    """
    device = vertices.device
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        empty_segments = wp.empty((0, 2), dtype=wp.vec3, device=device)
        if return_faces:
            return empty_segments, wp.empty(0, dtype=wp.int32, device=device)
        return empty_segments

    vertex_dots = _plane_dots(vertices, plane_normal, plane_origin)

    valid = wp.empty(n_faces, dtype=wp.bool, device=device)
    segments = wp.empty((n_faces, 2), dtype=wp.vec3, device=device)
    wp.launch(
        kernel_intersections.mesh_with_plane_segments,
        dim=n_faces,
        inputs=[vertices, faces, vertex_dots, plane_normal, plane_origin, valid, segments],
        device=device,
    )

    hit_faces = tw.array.flatnonzero(valid)
    n_hit = int(hit_faces.shape[0])
    if n_hit == 0:
        empty_segments = wp.empty((0, 2), dtype=wp.vec3, device=device)
        if return_faces:
            return empty_segments, wp.empty(0, dtype=wp.int32, device=device)
        return empty_segments

    lines = tw.array.gather(segments, hit_faces)
    if not return_faces:
        return lines
    # ``hit_faces`` is already a fresh dense buffer out of ``flatnonzero``; nothing else holds it,
    # so it is returned directly rather than copied.
    return lines, hit_faces


def marching_triangles(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    values: wp.array[wp.float32] | wp.array[wp.float64],
    isovalue: float = 0.0,
    n_vertices: int | None = None,
) -> tuple[list[wp.array[wp.vec3]], list[bool]]:
    """
    Extract the ``values == isovalue`` level set as a list of polylines.

    Each returned curve is a connected component of the level set, oriented so that the region where
    ``values > isovalue`` lies to its left (with the vertex normals as up). Curves close up unless
    they run into a mesh boundary, so an open curve begins and ends on a boundary edge.

    A value equal to the isovalue counts as positive, which keeps every cut face at exactly one
    segment; a contour running exactly through a vertex therefore yields zero-length segments rather
    than an ambiguous junction.

    Crossings are matched by [`edges_unique_inverse`][triwarp.edges.edges_unique_inverse], then
    linked on the host: the segment list is compacted on device first, so the readback is one
    ``int32`` pair per segment, and the linking itself is a vectorized pointer-doubling ranking over
    the compacted arrays with no per-segment iteration (the same successor-graph shape
    [`boundary_loops`][triwarp.boundary.boundary_loops] solves on device).

    !!! note "The returned arrays are views"
        Every curve slices one packed buffer, so holding a single curve keeps them all alive and
        writing into one writes into the shared allocation. ``wp.clone`` a curve for an independent
        buffer.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    values
        ``(n_vertices,)`` scalar field, ``wp.float32`` or ``wp.float64``. A ``float64`` field (what
        [`heat_geodesic`][triwarp.heat.distance.heat_geodesic] returns) is interpolated in
        ``float64``.
    isovalue
        Level to extract.
    n_vertices
        Total vertex count, forwarded to
        [`edges_unique_inverse`][triwarp.edges.edges_unique_inverse] as the hash base. When
        ``None`` it is inferred there with a host readback.

    Returns
    -------
    curves : list[wp.array[wp.vec3]]
        One array of points per level-set component, in order along the curve, on
        ``vertices.device``. Closed curves do not repeat their first point.
    closed : list[bool]
        Whether each curve is a closed loop.

    Raises
    ------
    ValueError
        If two segments start on the same mesh edge, which means the faces are not consistently
        oriented (the level set cannot then be linked into oriented curves). Repair the winding with
        [`make_winding_consistent`][triwarp.repair.make_winding_consistent] first.

    See Also
    --------
    [`clip_mesh_with_field`][triwarp.intersection.clip_mesh_with_field]
    [`mesh_with_plane`][triwarp.intersection.mesh_with_plane]
    [`heat_geodesic`][triwarp.heat.distance.heat_geodesic]
    [`polyline_length`][triwarp.polyline.polyline_length]
    ``potpourri3d.MarchingTrianglesSolver``
    """
    device = vertices.device
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return [], []

    # Subtract the isovalue once so the kernel only has to test signs; ``wp.map`` keeps the
    # element-wise work out of the kernel and preserves the field's dtype.
    shifted = wp.empty(int(values.shape[0]), dtype=values.dtype, device=device)
    wp.map(wp.sub, values, values.dtype(isovalue), out=shifted)

    edge_ids = tw.edges.edges_unique_inverse(faces, n_vertices=n_vertices)
    valid = wp.empty(n_faces, dtype=wp.bool, device=device)
    segments = wp.empty((n_faces, 2), dtype=wp.vec3, device=device)
    segment_edges = twt.empty_2d((n_faces, 2), wp.int32, device=device)
    wp.launch(
        kernel_intersections.marching_triangles_segments,
        dim=n_faces,
        inputs=[vertices, faces, shifted, edge_ids, valid, segments, segment_edges],
        device=device,
    )

    cut_faces = tw.array.flatnonzero(valid)
    n_segments = int(cut_faces.shape[0])
    if n_segments == 0:
        return [], []

    hit_segments = tw.array.gather(segments, cut_faces)
    hit_edges = twt.as_array2d(tw.array.gather(segment_edges, cut_faces), wp.int32)

    slots_np, starts_np, closed = _link_segments(hit_edges.numpy())

    # One gather assembles every curve: the slots index the flattened endpoint buffer, so the
    # packed result can be sliced per curve without a launch each.
    endpoints = hit_segments.reshape((2 * n_segments,))
    slots = wp.array(slots_np, dtype=wp.int32, device=device)
    packed = tw.array.gather(endpoints, slots)
    offsets = wp.array(starts_np, dtype=wp.int32, device=device)
    return tw.array.split(packed, offsets), closed


def _link_segments(segment_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[bool]]:
    """
    Chain oriented segments into curves, returning endpoint slots, curve starts and closed flags.

    ``segment_edges[i]`` holds the unique-edge ids the two endpoints of segment ``i`` lie on. Since
    the segments are consistently oriented, an interior crossing edge appears once as some segment's
    outgoing endpoint and once as another's incoming endpoint, so the successor relation is a
    permutation on all but the boundary-terminated chains -- the same successor-graph structure
    [`boundary_loops`][triwarp.boundary.boundary_loops] walks.

    Slots index the flattened endpoint buffer: endpoint ``e`` of segment ``i`` is ``2 * i + e``.
    Curves come out in the same order the serial walk produced: the open ones first, each from its
    unique predecessor-free segment, then the closed ones, each entered at its lowest-indexed
    segment so the result does not depend on face order.

    **Nothing here iterates per segment.** The successor comes from a lookup table rather than a
    sort -- an endpoint's unique-edge id indexes "which segment starts here", three ``O(n)`` passes
    against an ``argsort``, measured 0.205 ms against 2.382 at 26 246 segments -- and the ordering
    is two passes of Wyllie pointer doubling, ``O(n log L)`` in NumPy rather than one Python
    iteration per segment. Measured on level sets of an ``icosphere(6)``, against the ``while``
    loop this replaced: **1.56x at 742 segments, 4.13x at 8 280, 5.04x at 26 246 and 5.12x at
    47 898**, with bit-identical slots, starts and flags at every size and on 18 mixed open/closed
    level sets of ``hemisphere`` and ``half_torus``. The Python walk was 84-88 % of
    ``marching_triangles``' whole cost on the many-curve fields, flat at 0.3 us per segment across
    a 61x range of them; end to end the call measures **1.94x at wave40** and unchanged (1.06x) on
    the single-contour fields, where the linking was never the cost.
    """
    n = int(segment_edges.shape[0])
    start_edge = np.ascontiguousarray(segment_edges[:, 0])
    end_edge = np.ascontiguousarray(segment_edges[:, 1])
    index = np.arange(n, dtype=np.int64)

    # ``owner[e]`` is the segment whose *outgoing* endpoint lies on edge ``e``, so the successor of
    # segment ``i`` is whoever owns ``i``'s incoming edge. A second segment claiming an edge
    # overwrites the first, and the loser then fails to find itself -- which is exactly the
    # inconsistent-winding case, detected without a duplicate scan of its own.
    owner = np.full(int(max(start_edge.max(), end_edge.max())) + 1, -1, dtype=np.int64)
    owner[start_edge] = index
    if not np.array_equal(owner[start_edge], index):
        raise ValueError(
            "marching_triangles cannot link the level set: two segments start on the same edge, "
            "which means the faces are not consistently oriented."
        )
    successor = owner[end_edge]
    rounds = max(1, math.ceil(math.log2(max(n, 2))))

    # Pass 1: pointer doubling with a fixed point at every open curve's last segment, carrying the
    # smallest index seen along the way. A segment on an open curve lands on that curve's end; one
    # on a closed curve never does, and its window wraps, so its minimum becomes the whole loop's.
    ahead = np.where(successor >= 0, successor, index)
    lowest = np.minimum(index, ahead)
    for _ in range(rounds):
        lowest = np.minimum(lowest, lowest[ahead])
        ahead = ahead[ahead]
    is_closed = successor[ahead] >= 0

    # Cut every loop at its lowest-indexed segment, which turns it into a chain headed there. After
    # this every curve is a chain, so one ranking pass covers both kinds.
    entry = is_closed & (lowest == index)
    cut = successor.copy()
    cut[(successor >= 0) & entry[np.maximum(successor, 0)]] = -1

    has_predecessor = np.zeros(n, dtype=bool)
    has_predecessor[cut[cut >= 0]] = True
    is_head = ~has_predecessor

    # Pass 2: the same doubling on the cut graph gives each segment its hop count to its curve's
    # last segment. It stops as soon as every pointer has reached one, which is ``log2`` of the
    # *longest curve* rather than of the segment count -- 6 rounds against 15 on a 26k-segment
    # level set whose curves are 34 segments at their longest.
    tail = np.where(cut >= 0, cut, index)
    steps = (cut >= 0).astype(np.int64)
    for _ in range(rounds):
        if not (cut[tail] >= 0).any():
            break
        steps = steps + steps[tail]
        tail = tail[tail]

    # A curve is named by its head, reached from any of its segments through its shared last one.
    head_by_tail = np.empty(n, dtype=np.int64)
    head_by_tail[tail[is_head]] = index[is_head]
    head_of = head_by_tail[tail]
    sizes = np.bincount(head_of, minlength=n)
    position = sizes[head_of] - 1 - steps

    heads = np.flatnonzero(is_head)
    head_closed = is_closed[heads]
    curve_heads = np.concatenate([heads[~head_closed], heads[head_closed]])
    closed = is_closed[curve_heads]
    curve_of_head = np.empty(n, dtype=np.int64)
    curve_of_head[curve_heads] = np.arange(curve_heads.shape[0])
    curve = curve_of_head[head_of]

    # An open curve carries one extra slot: its last segment contributes both endpoints.
    slot_counts = sizes[curve_heads] + ~closed
    starts = np.concatenate([[0], np.cumsum(slot_counts)[:-1]]).astype(np.int32)
    slots = np.empty(int(slot_counts.sum()), dtype=np.int32)
    slots[starts[curve] + position] = 2 * index
    last = successor < 0
    slots[starts[curve[last]] + position[last] + 1] = 2 * index[last] + 1
    return slots, starts, closed.tolist()


def mesh_with_mesh(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    *,
    max_triangle_collisions: int = 16,
) -> wp.array[wp.vec3]:
    """
    Intersect two meshes, returning line segments along the intersection curve(s).

    Broad phase builds a ``wp.Mesh`` over the mesh with fewer faces and queries
    triangle AABBs via ``wp.mesh_query_aabb``; each triangle of the other mesh
    supplies the query box. Narrow phase runs Moller's interval test and clips the intersection
    line to both triangles. Coplanar overlapping faces produce no segments.

    Parameters
    ----------
    vertices_a, faces_a
        First indexed triangle mesh.
    vertices_b, faces_b
        Second indexed triangle mesh.
    max_triangle_collisions
        Maximum broad-phase candidate pairs recorded per query triangle.

    Returns
    -------
    lines
        ``(m, 2)`` ``wp.vec3`` segment endpoints (logical shape ``(m, 2, 3)``).

    Raises
    ------
    ValueError
        If ``max_triangle_collisions`` is less than 1.
    """
    device = vertices_a.device
    crossing = _colliding_face_pairs(
        vertices_a, faces_a, vertices_b, faces_b, max_triangle_collisions, "mesh_with_mesh"
    )
    if crossing is None:
        return wp.empty((0, 2), dtype=wp.vec3, device=device)
    hit_pairs, query_vertices, query_faces, target_vertices, target_faces, _swapped = crossing
    n_hit = int(hit_pairs.shape[0])

    segments = wp.empty((n_hit, 2), dtype=wp.vec3, device=device)
    wp.launch(
        kernel_intersections.triangle_pair_segments,
        dim=n_hit,
        inputs=[query_vertices, query_faces, target_vertices, target_faces, hit_pairs, segments],
        device=device,
    )

    seg_valid = wp.empty(n_hit, dtype=wp.bool, device=device)
    wp.launch(
        kernel_intersections.segment_nondegenerate,
        dim=n_hit,
        inputs=[segments, seg_valid],
        device=device,
    )

    keep = tw.array.flatnonzero(seg_valid)
    n_keep = int(keep.shape[0])
    if n_keep == 0:
        return wp.empty((0, 2), dtype=wp.vec3, device=device)

    return tw.array.gather(segments, keep)


def _colliding_face_pairs(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    max_triangle_collisions: int,
    caller: str,
) -> (
    tuple[
        twt.Array2dInt32,
        wp.array[wp.vec3],
        wp.array[wp.int32],
        wp.array[wp.vec3],
        wp.array[wp.int32],
        bool,
    ]
    | None
):
    """
    Broad phase plus narrow phase for two meshes: the crossing face pairs, or ``None`` if none.

    Shared by [`mesh_with_mesh`][triwarp.intersection.mesh_with_mesh], which goes on to compute a
    segment per pair, and [`mesh_collision_pairs`][triwarp.intersection.mesh_collision_pairs], which
    returns the pairs themselves. It exists because the two would otherwise carry the same fifty
    lines twice, and because those lines contain the one thing a caller must not get wrong: the pair
    columns are ``(query, target)``, and which input is the query depends on the **face counts**.

    Parameters
    ----------
    vertices_a, faces_a, vertices_b, faces_b
        The two meshes.
    max_triangle_collisions
        Broad-phase candidate cap per query triangle.
    caller
        Name to report in the empty-mesh guard's message.

    Returns
    -------
    tuple | None
        ``(pairs, query_vertices, query_faces, target_vertices, target_faces, swapped)``, where
        ``pairs`` is ``(n_hit, 2)`` in ``(query, target)`` order and ``swapped`` says whether the
        query is mesh **b** -- i.e. whether the columns are the caller's ``(a, b)`` order reversed.
        ``None`` when either mesh is empty or nothing crosses.

    Raises
    ------
    ValueError
        If ``max_triangle_collisions`` is less than 1.
    """
    device = vertices_a.device
    n_faces_a = int(faces_a.shape[0]) // 3
    n_faces_b = int(faces_b.shape[0]) // 3
    if n_faces_a == 0 or n_faces_b == 0:
        return None
    if max_triangle_collisions < 1:
        raise ValueError("max_triangle_collisions must be >= 1")

    # The smaller mesh supplies the BVH, so the larger one's faces are the queries.
    swapped = n_faces_a <= n_faces_b
    if swapped:
        target_vertices, target_faces = vertices_a, faces_a
        query_vertices, query_faces = vertices_b, faces_b
    else:
        target_vertices, target_faces = vertices_b, faces_b
        query_vertices, query_faces = vertices_a, faces_a

    n_query = int(query_faces.shape[0]) // 3
    require_nonempty_mesh(target_faces, caller)
    target_mesh = wp.Mesh(points=target_vertices, indices=target_faces)

    query_lower = wp.empty(n_query, dtype=wp.vec3, device=device)
    query_upper = wp.empty(n_query, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_triangles.face_aabb_bounds,
        dim=n_query,
        inputs=[query_vertices, query_faces, query_lower, query_upper],
        device=device,
    )

    target_indices, offsets, hit_counts = tw.proximity.query_mesh_aabb_with_offsets(
        target_mesh, query_lower, query_upper, max_hits=max_triangle_collisions
    )
    n_pairs = int(target_indices.shape[0])
    if n_pairs == 0:
        return None

    pairs = twt.empty_2d((n_pairs, 2), wp.int32, device=device)
    wp.launch(
        kernel_intersections.expand_query_target_pairs,
        dim=n_query,
        inputs=[offsets, hit_counts, target_indices, pairs],
        device=device,
    )

    valid = wp.empty(n_pairs, dtype=wp.bool, device=device)
    wp.launch(
        kernel_intersections.filter_intersecting_pairs,
        dim=n_pairs,
        inputs=[query_vertices, query_faces, target_vertices, target_faces, pairs, valid],
        device=device,
    )

    hit_pair_indices = tw.array.flatnonzero(valid)
    if int(hit_pair_indices.shape[0]) == 0:
        return None
    hit_pairs = twt.as_array2d(tw.array.gather(pairs, hit_pair_indices), wp.int32)
    return hit_pairs, query_vertices, query_faces, target_vertices, target_faces, swapped


def mesh_collision_pairs(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    *,
    max_triangle_collisions: int = 16,
) -> twt.Array2dInt32:
    """
    Which faces of two meshes cross each other, as index pairs.

    The **collision** question, as against
    [`mesh_with_mesh`][triwarp.intersection.mesh_with_mesh]'s intersection *curve*: this answers
    "do these two touch, and where" without computing the geometry of the contact, which is what a
    contact resolver, a fit check or an assembly validator wants.
    [`triwarp.validation.face_self_intersecting_mask`][triwarp.validation.face_self_intersecting_mask]
    is the same question asked of one mesh against itself.

    Parameters
    ----------
    vertices_a, faces_a
        First mesh: ``(n_vertices_a,)`` positions and a length-``3 * n_faces_a`` index buffer.
    vertices_b, faces_b
        Second mesh, in the same form.
    max_triangle_collisions
        Broad-phase candidate cap per query triangle. A pair beyond the cap is **dropped**, so raise
        it on meshes whose triangles pile into overlapping boxes; the answer is a subset, never a
        superset.

    Returns
    -------
    twt.Array2dInt32
        ``(n_pairs, 2)`` face-index pairs, column 0 into ``faces_a`` and column 1 into ``faces_b``.
        Empty ``(0, 2)`` when the meshes do not cross. Pairs that merely touch or are coplanar are
        **not** collisions -- the same convention
        [`triwarp.validation.is_self_intersecting`][triwarp.validation.is_self_intersecting] uses.

    Raises
    ------
    ValueError
        If ``max_triangle_collisions`` is less than 1.

    Examples
    --------
    ```python
    pairs = tw.intersection.mesh_collision_pairs(v, f, v, f)
    ```

    See Also
    --------
    [`collision_masks`][triwarp.intersection.collision_masks]
        The same answer as one boolean mask per mesh, which is the form a repair pass wants.
    [`mesh_with_mesh`][triwarp.intersection.mesh_with_mesh]
        The intersection curve, when the contact geometry is wanted and not just its existence.
    [`triwarp.proximity.closest_point_on_mesh`][triwarp.proximity.closest_point_on_mesh]
        What to reach for when the meshes do *not* touch and the clearance is the question.
    """
    device = vertices_a.device
    crossing = _colliding_face_pairs(
        vertices_a, faces_a, vertices_b, faces_b, max_triangle_collisions, "mesh_collision_pairs"
    )
    if crossing is None:
        return twt.empty_2d((0, 2), wp.int32, device=device)
    hit_pairs, _qv, _qf, _tv, _tf, swapped = crossing
    if not swapped:
        return hit_pairs

    ordered = twt.empty_2d((int(hit_pairs.shape[0]), 2), wp.int32, device=device)
    wp.launch(
        kernel_intersections.swap_pair_columns,
        dim=int(hit_pairs.shape[0]),
        inputs=[hit_pairs, ordered],
        device=device,
    )
    return ordered


def collision_masks(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    *,
    max_triangle_collisions: int = 16,
) -> tuple[wp.array[wp.bool], wp.array[wp.bool]]:
    """
    Which faces of each mesh are involved in a collision with the other, as one mask per mesh.

    [`mesh_collision_pairs`][triwarp.intersection.mesh_collision_pairs] reduced to the two questions
    a repair or a selection actually asks -- *which of my faces are in trouble* -- and the form a
    per-mesh collision bitset takes. It exists as its own entry point because
    deriving it from the pairs means scattering a **column** of a rank-2 array, and a column is a
    strided view that Warp's Python-scope gather silently misreads (CLAUDE.md section 4).

    Parameters
    ----------
    vertices_a, faces_a, vertices_b, faces_b
        The two meshes, as in [`mesh_collision_pairs`][triwarp.intersection.mesh_collision_pairs].
    max_triangle_collisions
        Broad-phase candidate cap per query triangle.

    Returns
    -------
    tuple[wp.array[wp.bool], wp.array[wp.bool]]
        Length-``n_faces_a`` and length-``n_faces_b`` masks on ``vertices_a.device``, ``True`` for a
        face that crosses some face of the other mesh.

    Raises
    ------
    ValueError
        If ``max_triangle_collisions`` is less than 1.

    See Also
    --------
    [`mesh_collision_pairs`][triwarp.intersection.mesh_collision_pairs]
        The pair list this reduces, when *which* faces meet matters.
    """
    device = vertices_a.device
    n_faces_a = int(faces_a.shape[0]) // 3
    n_faces_b = int(faces_b.shape[0]) // 3
    mask_a = wp.zeros(n_faces_a, dtype=wp.bool, device=device)
    mask_b = wp.zeros(n_faces_b, dtype=wp.bool, device=device)

    pairs = mesh_collision_pairs(
        vertices_a, faces_a, vertices_b, faces_b, max_triangle_collisions=max_triangle_collisions
    )
    n_pairs = int(pairs.shape[0])
    if n_pairs > 0:
        wp.launch(
            kernel_intersections.mark_pair_masks,
            dim=n_pairs,
            inputs=[pairs, mask_a, mask_b],
            device=device,
        )
    return mask_a, mask_b


def slice_mesh_with_plane(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    plane_normal: wp.vec3,
    plane_origin: wp.vec3,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Slice a mesh with a plane, returning the portion on the positive normal side.

    Matches [`trimesh.intersections.slice_faces_plane`][] for indexed triangle meshes
    (without UV handling). To slice a face subset, extract a submesh first (e.g.
    [`submesh_from_face_indices`][triwarp.selection.submesh_from_face_indices]).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` flat triangle index buffer.
    plane_normal
        Normal vector of the plane.
    plane_origin
        Point on the plane.

    Returns
    -------
    new_vertices
        Vertices of the sliced mesh.
    new_faces
        Length-``3 * m`` flat triangle index buffer for the sliced mesh.

    Notes
    -----
    The plane's signed distance is one per-vertex scalar field, so this is
    [`clip_mesh_with_field`][triwarp.intersection.clip_mesh_with_field] over
    ``dot(v - plane_origin, plane_normal)`` — the only thing the plane adds is the tie-break for a
    face lying *in* the plane, whose side is decided from its own normal rather than from its
    vertices.

    Every face falls into exactly one of three kept classes — wholly inside, cut into a quad, cut
    into a triangle — and the three are compacted by a *single* scan over one blocked flag buffer,
    so the call makes **one** host readback (the three class counts, 12 bytes) rather than one per
    class. Those counts then size the output buffers for their final use, and the two cut kernels
    write their triangles and their intersection points straight into them, so nothing is
    concatenated afterwards.

    Measured back to back on an RTX 5090 (`benchmarks/test_intersection.py`'s
    ``slice_mesh_with_plane`` group, medians from ``--benchmark-json``): **2.40x** on ``bunny``
    (1.94 -> 0.81 ms), 2.17x on ``happy_buddha``, 1.90x on ``dragon`` (1.75 -> 0.92) and 1.15x on
    ``lucy``. It is *flat* on ``bunny_decimated`` (1.20 -> 1.23 ms), the smallest mesh, because
    three readbacks or one, that row is at the ~340 µs wrapper floor — which is also why the cost
    here barely tracks the face count: the plane still meets only ``O(sqrt(n_faces))`` triangles.

    See Also
    --------
    [`clip_mesh_with_field`][triwarp.intersection.clip_mesh_with_field]
    [`mesh_with_plane`][triwarp.intersection.mesh_with_plane]
    [`trimesh.intersections.slice_faces_plane`][]
    """
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    if n_vertices == 0:
        return vertices, faces
    if int(faces.shape[0]) == 0:
        return wp.empty(0, dtype=wp.vec3, device=device), wp.empty(0, dtype=wp.int32, device=device)

    return _clip_with_vertex_field(
        vertices,
        faces,
        _plane_dots(vertices, plane_normal, plane_origin),
        plane_normal=plane_normal,
    )


def split_mesh_with_plane(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    plane_normal: wp.vec3,
    plane_origin: wp.vec3,
    *,
    tolerance: float = TOLERANCE_MERGE,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.bool]]:
    """
    Insert a plane's cross-section into the mesh as real edges, keeping both sides.

    The third thing a plane can do to a mesh, next to
    [`mesh_with_plane`][triwarp.intersection.mesh_with_plane], which returns the section curve and
    leaves the mesh alone, and
    [`slice_mesh_with_plane`][triwarp.intersection.slice_mesh_with_plane], which keeps one side and
    discards the other: this keeps **everything** and refines the triangles the plane crosses so the
    section becomes a set of mesh edges, then labels every output face by the side it fell on. It is
    what to reach for to select a region by a plane and then remesh, decimate or smooth only that
    region, since the complementary side is still there to be welded back to.

    The result is **crack-free**: the crossing point of an edge is computed once per *edge*, so the
    two triangles sharing it reference one vertex index and a watertight input stays watertight.
    That is the difference from
    [`slice_mesh_with_plane`][triwarp.intersection.slice_mesh_with_plane], whose per-face cut leaves
    two coincident copies along the section.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` flat triangle index buffer.
    plane_normal
        Normal vector of the plane. Need not be unit length, but ``tolerance`` is a band on
        ``dot(v - plane_origin, plane_normal)``, so a non-unit normal scales it.
    plane_origin
        Point on the plane.
    tolerance
        Half-width of the band around the plane within which a vertex counts as lying *on* it. Such
        a vertex is used as the crossing itself rather than having a near-duplicate inserted beside
        it, which is what keeps a plane through an existing vertex from producing slivers. Defaults
        to [`TOLERANCE_MERGE`][triwarp.constants.TOLERANCE_MERGE]; scale it with the model when the
        coordinates are large.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        Original vertices followed by one crossing point per crossed edge.
    new_faces : wp.array[wp.int32]
        Length-``3 * m`` flat triangle index buffer for the refined mesh, both sides included.
    above : wp.array[wp.bool]
        ``(m,)`` per-face mask: ``True`` where the face is on the ``plane_normal`` side. This is the
        side [`slice_mesh_with_plane`][triwarp.intersection.slice_mesh_with_plane] returns,
        including its tie-break for a face lying *in* the plane (kept when its own normal opposes
        ``plane_normal``), so
        ``submesh_from_face_mask(*split_mesh_with_plane(...))`` and ``slice_mesh_with_plane`` agree.

    Notes
    -----
    A plane crosses at most **two** of a triangle's three edges — two of three vertices always share
    a side, so their edge is not crossed — which means every crossed face is refined by the 1-split
    or 2-split case of [`subdivide_to_size`][triwarp.remesh.subdivide_to_size]'s crack-free
    templates, reused unchanged. Those templates are indexed by *which* corners carry a new vertex
    and never assume the new vertex is a midpoint, so the plane crossing drops straight in. The
    2-split quad is cut along its shorter diagonal, as there; both of its diagonals lie wholly on
    one side of the plane, so the choice cannot make a face straddle.

    Faces are **not** reordered: an output face's position is the compaction order of the split
    templates, not grouped by side. Pass ``above`` to
    [`submesh_from_face_mask`][triwarp.selection.submesh_from_face_mask] (or its negation) to
    materialize either half.

    Equivalent to VTK's ``vtkClipPolyData`` with ``GenerateClippedOutput``, which pyvista exposes as
    ``PolyData.clip(..., return_clipped=True)``.

    Examples
    --------
    Split at ``z = 0`` and confirm both halves are non-empty and together account for every face:

    ```python
    split_v, split_f, above = tw.intersection.split_mesh_with_plane(
        v, f, wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 0.0, 0.0)
    )
    n_above = int(tw.array.flatnonzero(above).shape[0])
    print(0 < n_above < int(above.shape[0]))
    ```

    See Also
    --------
    [`slice_mesh_with_plane`][triwarp.intersection.slice_mesh_with_plane]
    [`mesh_with_plane`][triwarp.intersection.mesh_with_plane]
    [`triwarp.remesh.split_edges`][triwarp.remesh.split_edges]
    [`triwarp.remesh.subdivide_to_size`][triwarp.remesh.subdivide_to_size]
    [`triwarp.selection.submesh_from_face_mask`][triwarp.selection.submesh_from_face_mask]
    """
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    n_faces = int(faces.shape[0]) // 3
    if n_vertices == 0 or n_faces == 0:
        return wp.clone(vertices), wp.clone(faces), wp.empty(n_faces, dtype=wp.bool, device=device)

    vertex_dots = _plane_dots(vertices, plane_normal, plane_origin)

    unique_edges, inverse = tw.edges.edges_unique(faces, n_vertices=n_vertices)
    n_edges = int(unique_edges.shape[0])
    crossed = wp.empty(n_edges, dtype=wp.bool, device=device)
    wp.launch(
        kernel_intersections.plane_crossed_edge_mask,
        dim=n_edges,
        inputs=[unique_edges, vertex_dots, wp.float32(tolerance), crossed],
        device=device,
    )
    # ``split_edges`` scans the mask itself; this scan exists because the crossing points must be
    # written at the slots that scan assigns, which is the ordering it documents for them.
    offsets, n_crossed = tw.array.counts_to_offsets(tw.array.astype(crossed, wp.int32))

    crossing_points = wp.empty(n_crossed, dtype=wp.vec3, device=device)
    if n_crossed > 0:
        wp.launch(
            kernel_intersections.plane_edge_crossing_points,
            dim=n_edges,
            inputs=[vertices, unique_edges, crossed, offsets, vertex_dots, crossing_points],
            device=device,
        )
    new_vertices, new_faces = tw.remesh.split_edges(
        vertices, faces, crossed, crossing_points, unique_edges=unique_edges, inverse=inverse
    )

    n_out = int(new_faces.shape[0]) // 3
    above = wp.empty(n_out, dtype=wp.bool, device=device)
    wp.launch(
        kernel_intersections.label_faces_by_plane_side,
        dim=n_out,
        inputs=[
            new_vertices,
            new_faces,
            # The appended crossing points sit on the plane by construction, so their dots are zero
            # to rounding; recomputing over the grown buffer is one map and avoids tracking which
            # tail entries to zero.
            _plane_dots(new_vertices, plane_normal, plane_origin),
            plane_normal,
            wp.float32(tolerance),
            above,
        ],
        device=device,
    )
    return new_vertices, new_faces, above


def clip_mesh_with_field(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    values: wp.array[wp.float32] | wp.array[wp.float64],
    isovalue: float = 0.0,
    *,
    cap: bool = False,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Keep the ``values >= isovalue`` region of the mesh, cutting faces the level set crosses.

    The region counterpart of
    [`marching_triangles`][triwarp.intersection.marching_triangles], which returns the level set
    itself: this returns the surface on one side of it, with every crossed triangle re-triangulated
    against the crossing points. The caller supplies the field, so one implementation covers
    clipping by a plane, by another surface's signed distance
    ([`signed_distance_on_mesh`][triwarp.proximity.signed_distance_on_mesh]), by a box, by geodesic
    distance ([`heat_geodesic`][triwarp.heat.distance.heat_geodesic]) or by any per-vertex quantity
    the caller can threshold.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` flat triangle index buffer.
    values
        ``(n_vertices,)`` scalar field, ``wp.float32`` or ``wp.float64``. A ``float64`` field is
        shifted in ``float64`` and then compared in ``float32``, which is the vertex buffer's
        precision.
    isovalue
        Level to clip at. A vertex whose value equals it counts as kept, matching
        [`marching_triangles`][triwarp.intersection.marching_triangles].
    cap
        When ``True``, weld the section rim and seal **every** boundary loop of the result with
        [`fill_min_weight`][triwarp.holes.fill_min_weight] — so a closed input gives a closed
        output, and the returned vertices are the welded ones. On an input that already had a
        boundary, that boundary is sealed too; clip first and cap yourself with
        [`fill_loops_min_weight`][triwarp.holes.fill_loops_min_weight] if only the section should
        close.

    Returns
    -------
    new_vertices
        Vertices of the clipped region, with the crossing points appended.
    new_faces
        Length-``3 * m`` flat triangle index buffer for the clipped region.

    Notes
    -----
    A face whose three values all equal ``isovalue`` is dropped: it lies *in* the level set, so
    neither side claims it and the field gives no tie-break.
    [`slice_mesh_with_plane`][triwarp.intersection.slice_mesh_with_plane] is the one caller that has
    one, and keeps such a face when its normal opposes the plane's.

    The uncapped result is **cracked along the section**, like
    [`trimesh.intersections.slice_faces_plane`][]'s: each cut face writes its own copy of the
    crossing points, so a rim edge carries two coincident vertices. Those copies are bitwise equal
    by construction, which is what makes ``cap=True``'s weld exact rather than a tolerance choice.

    Measured against ``clip_closed_surface`` on ``icosphere(3)`` at ``z = 0.1``: same 762 faces and
    the same volume to seven digits (1.7651057), from a min-weight cap against VTK's own
    triangulation of the section.

    Equivalent to VTK's ``clip_scalar`` (uncapped) and ``clip_closed_surface`` (capped, over a
    plane's signed distance), which pyvista exposes on ``PolyData``; VTK's default keeps
    ``scalar >= value`` as this does.

    Examples
    --------
    Keep the geodesic disk of radius 1 around vertex 0, sealed into a solid. The field is the
    ``float64`` one [`heat_geodesic`][triwarp.heat.distance.heat_geodesic] returns, and the region
    wanted is the *near* side, so the field enters negated:

    ```python
    source = wp.array([0], dtype=wp.int32, device=v.device)
    distance = tw.heat.distance.heat_geodesic(v, f, source)
    negated = wp.empty_like(distance)
    wp.map(wp.neg, distance, out=negated)
    disk_v, disk_f = tw.intersection.clip_mesh_with_field(v, f, negated, -1.0, cap=True)
    print(tw.validation.is_edge_manifold(disk_f, allow_boundary_edges=False))
    ```

    See Also
    --------
    [`marching_triangles`][triwarp.intersection.marching_triangles]
    [`slice_mesh_with_plane`][triwarp.intersection.slice_mesh_with_plane]
    [`fill_min_weight`][triwarp.holes.fill_min_weight]
    """
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    if n_vertices == 0:
        return vertices, faces
    if int(faces.shape[0]) == 0:
        return wp.empty(0, dtype=wp.vec3, device=device), wp.empty(0, dtype=wp.int32, device=device)

    new_vertices, new_faces = _clip_with_vertex_field(
        vertices, faces, _shifted_field(values, isovalue)
    )
    if cap:
        # The cut writes its crossing points per face, so a rim edge shared by two cut faces arrives
        # as two coincident vertices and the section is a set of loose edges rather than a loop.
        # They are *bitwise* equal by construction (the crossing is evaluated from the lower-
        # numbered endpoint on both sides), so ``epsilon=0`` collapses exactly those and the filler
        # then sees a real boundary loop.
        new_vertices, _, _, new_faces = tw.repair.remove_duplicated_vertices(
            new_vertices, new_faces
        )
        new_faces = tw.holes.fill_min_weight(new_vertices, new_faces)
    return new_vertices, new_faces


def split_faces_along_field(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    values: wp.array[wp.float32] | wp.array[wp.float64],
    isovalue: float = 0.0,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.bool]]:
    """
    Cut every face the level set crosses, keeping **both** sides as one connected mesh.

    The both-sides form of [`clip_mesh_with_field`][triwarp.intersection.clip_mesh_with_field]: that
    one keeps the ``values >= isovalue`` region and discards the rest, this one keeps everything and
    makes the level set a set of real mesh edges. Nothing moves and nothing is welded -- every input
    vertex survives at its own position, and the crossing points are appended and **shared** by the
    faces on both sides, so the result is as watertight as the input was.

    That sharing is the whole point. It is what makes the returned mask a genuine partition of the
    surface along the curve rather than two overlapping selections, so
    [`submesh_from_face_mask`][triwarp.selection.submesh_from_face_mask] on the mask and on its
    negation gives two pieces that meet exactly along the level set.

    A face with one corner exactly on the level set splits into **two** triangles, not three: the
    cut runs from that corner to the single crossing on the opposite edge. Emitting three would put
    a zero-area sliver in the output.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` flat triangle index buffer.
    values
        ``(n_vertices,)`` scalar field, ``wp.float32`` or ``wp.float64``. A ``float64`` field is
        shifted in ``float64`` and then compared in ``float32``, which is the vertex buffer's
        precision -- the same rule as
        [`marching_triangles`][triwarp.intersection.marching_triangles].
    isovalue
        Level to cut at. A vertex whose value equals it counts as positive, so a face lying wholly
        in the level set is kept whole on the positive side rather than cut along itself.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        The input vertices, unchanged and in order, with the crossing points appended.
    new_faces : wp.array[wp.int32]
        Length-``3 * m`` flat triangle index buffer. Uncut faces keep their winding and their
        vertex indices; cut faces are replaced by the two or three triangles they split into.
    positive : wp.array[wp.bool]
        Length-``m`` mask, ``True`` for the faces on the ``values >= isovalue`` side.

    See Also
    --------
    [`clip_mesh_with_field`][triwarp.intersection.clip_mesh_with_field]
        Keeps one side and discards the other, which is cheaper when that is all you need.
    [`marching_triangles`][triwarp.intersection.marching_triangles]
        Returns the level set itself, as polylines, instead of cutting along it.
    [`split_mesh_with_plane`][triwarp.intersection.split_mesh_with_plane]
        The plane special case of the clip.
    [`faces_left_of_contour`][triwarp.selection.faces_left_of_contour]
        The mask this returns for free, for a contour that was *given* rather than just created.
    """
    device = vertices.device
    if int(faces.shape[0]) == 0 or int(vertices.shape[0]) == 0:
        return (
            wp.clone(vertices),
            wp.empty(0, dtype=wp.int32, device=device),
            wp.empty(0, dtype=wp.bool, device=device),
        )
    return _split_with_vertex_field(vertices, faces, _shifted_field(values, isovalue))


def _shifted_field(
    values: wp.array[wp.float32] | wp.array[wp.float64], isovalue: float
) -> wp.array[wp.float32]:
    """
    Re-zero the field at ``isovalue``, in the vertex buffer's precision.

    Shifted in the field's own dtype (``wp.map`` preserves it, as in ``marching_triangles``), then
    narrowed once: every classifier and crossing kernel downstream works in ``float32``, which is
    what the vertex buffer carries, so narrowing earlier would lose the subtraction's precision and
    narrowing later would mean doing it per face.
    """
    device = values.device
    shifted = wp.empty(int(values.shape[0]), dtype=values.dtype, device=device)
    wp.map(wp.sub, values, values.dtype(isovalue), out=shifted)
    if values.dtype is wp.float32:
        return shifted
    narrowed = wp.empty(int(values.shape[0]), dtype=wp.float32, device=device)
    wp.utils.array_cast(shifted, narrowed)
    return narrowed


def _plane_dots(
    vertices: wp.array[wp.vec3], plane_normal: wp.vec3, plane_origin: wp.vec3
) -> wp.array[wp.float32]:
    """Signed plane distance of every vertex, the field all three plane entry points classify on."""
    dots = wp.empty(int(vertices.shape[0]), dtype=wp.float32, device=vertices.device)
    wp.map(kernel_intersections.point_plane_dot, vertices, plane_normal, plane_origin, out=dots)
    return dots


def _split_with_vertex_field(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], vertex_dots: wp.array[wp.float32]
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.bool]]:
    """Cut along ``vertex_dots == 0``, keeping both sides; the engine behind the public split."""
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    n_faces = int(faces.shape[0]) // 3

    face_classes = wp.empty(n_faces, dtype=wp.int32, device=device)
    face_signs = twt.empty_2d((n_faces, 3), wp.int32, device=device)
    wp.launch(
        kernel_intersections.classify_faces_for_split,
        dim=n_faces,
        inputs=[faces, vertex_dots, face_classes, face_signs],
        device=device,
    )
    blocks, counts = _slice_class_partition(
        face_classes, n_faces, kernel_intersections.SPLIT_CLASSES
    )
    positive_idx, negative_idx, edges_idx, corner_idx = blocks
    n_positive, n_negative, n_edges, n_corner = counts
    if n_edges + n_corner == 0:
        # Nothing crossed, so the mesh is untouched and only the side labels are new.
        positive = wp.empty(n_faces, dtype=wp.bool, device=device)
        wp.map(kernel_intersections.is_positive_split_class, face_classes, out=positive)
        return wp.clone(vertices), wp.clone(faces), positive

    # One new vertex per crossed *edge*: both faces sharing it address the same index, which is
    # what makes the cut watertight rather than a seam of coincident pairs. Sign agreement with the
    # classifier is a correctness requirement, so the mask is built at ``TOLERANCE_MERGE``, the
    # dead zone ``tolerance_sign`` uses.
    unique_edges, halfedge_edges = tw.edges.edges_unique(faces)
    crossed = wp.empty(int(unique_edges.shape[0]), dtype=wp.bool, device=device)
    wp.launch(
        kernel_intersections.plane_crossed_edge_mask,
        dim=int(unique_edges.shape[0]),
        inputs=[unique_edges, vertex_dots, TOLERANCE_MERGE, crossed],
        device=device,
    )
    edge_vertex_rank, n_new = tw.array.mask_to_index_map(crossed)
    crossed_edge_indices = tw.array.flatnonzero(crossed)

    n_uncut = n_positive + n_negative
    n_emitted = n_uncut + 3 * n_edges + 2 * n_corner
    all_vertices = wp.empty(n_vertices + n_new, dtype=wp.vec3, device=device)
    wp.copy(all_vertices[:n_vertices], vertices)
    wp.launch(
        kernel_intersections.level_set_edge_vertices,
        dim=n_new,
        inputs=[
            vertices,
            unique_edges,
            crossed_edge_indices,
            vertex_dots,
            all_vertices[n_vertices:],
        ],
        device=device,
    )

    all_faces = wp.empty(3 * n_emitted, dtype=wp.int32, device=device)
    positive = wp.empty(n_emitted, dtype=wp.bool, device=device)
    face_rows = all_faces.reshape((n_emitted, 3))
    for offset, count, index_block, side in (
        (0, n_positive, positive_idx, True),
        (n_positive, n_negative, negative_idx, False),
    ):
        if count == 0:
            continue
        # Not ``tw.array.gather``: the destination is a slice of a larger buffer, and gather
        # allocates its own.
        wp.copy(face_rows[offset : offset + count], faces.reshape((-1, 3))[index_block])
        positive[offset : offset + count].fill_(side)

    face_base = n_uncut
    for count, index_block, emit, rows_per_cut in (
        (n_edges, edges_idx, kernel_intersections.emit_split_cut_edges, 3),
        (n_corner, corner_idx, kernel_intersections.emit_split_cut_corner, 2),
    ):
        if count == 0:
            continue
        emitted = rows_per_cut * count
        wp.launch(
            emit,
            dim=count,
            inputs=[
                faces,
                index_block,
                face_signs,
                halfedge_edges,
                edge_vertex_rank,
                wp.int32(n_vertices),
                face_rows[face_base : face_base + emitted],
                positive[face_base : face_base + emitted],
            ],
            device=device,
        )
        face_base += emitted

    return all_vertices, all_faces, positive


def _clip_with_vertex_field(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    vertex_dots: wp.array[wp.float32],
    plane_normal: wp.vec3 | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Keep the ``vertex_dots >= 0`` region, the engine behind both public clippers.

    ``plane_normal`` is the plane clip's tie-break for a face lying in the level set, and its
    presence is the *only* difference between the two: a general field has no orientation to
    compare, so such a face is left in the ``ON_PLANE`` class, which no kept block claims.
    """
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    n_faces = int(faces.shape[0]) // 3

    face_classes = wp.empty(n_faces, dtype=wp.int32, device=device)
    face_signs = twt.empty_2d((n_faces, 3), wp.int32, device=device)
    wp.launch(
        kernel_intersections.classify_faces_for_slice,
        dim=n_faces,
        inputs=[faces, vertex_dots, face_classes, face_signs],
        device=device,
    )
    if plane_normal is not None:
        wp.launch(
            kernel_intersections.resolve_on_plane_faces,
            dim=n_faces,
            inputs=[vertices, faces, plane_normal, face_classes],
            device=device,
        )
    (inside_idx, quad_idx, tri_idx), (n_in, n_quad, n_tri) = _slice_class_partition(
        face_classes, n_faces, kernel_intersections.SLICE_CLASSES
    )
    n_in = int(inside_idx.shape[0])
    n_quad = int(quad_idx.shape[0])
    n_tri = int(tri_idx.shape[0])

    if n_quad + n_tri == 0:
        if n_in == 0:
            return wp.empty(0, dtype=wp.vec3, device=device), wp.empty(
                0, dtype=wp.int32, device=device
            )
        return tw.selection.submesh_from_face_indices(
            vertices, faces, inside_idx, unique_indices=True
        )

    # Both cuts contribute two intersection points and keep the original vertices addressable, so
    # the un-compacted output is exactly this long -- allocated once, written in place.
    all_vertices = wp.empty(n_vertices + 2 * (n_quad + n_tri), dtype=wp.vec3, device=device)
    wp.copy(all_vertices[:n_vertices], vertices)
    all_faces = wp.empty(3 * (n_in + 2 * n_quad + n_tri), dtype=wp.int32, device=device)
    if n_in > 0:
        # Not ``tw.array.gather``: the destination is a *slice* of a larger buffer, and ``gather``
        # allocates its own.
        wp.copy(all_faces[: 3 * n_in].reshape((n_in, 3)), faces.reshape((-1, 3))[inside_idx])

    vertex_base, face_base = n_vertices, 3 * n_in
    for face_indices, n_cut, emit, faces_per_cut in (
        (quad_idx, n_quad, kernel_intersections.emit_quad_cut, 2),
        (tri_idx, n_tri, kernel_intersections.emit_tri_cut, 1),
    ):
        if n_cut == 0:
            continue
        edge_points = wp.empty((n_cut, 3), dtype=wp.vec3, device=device)
        wp.launch(
            kernel_intersections.edge_level_crossings,
            dim=n_cut,
            inputs=[vertices, faces, face_indices, vertex_dots, edge_points],
            device=device,
        )
        n_emitted = faces_per_cut * n_cut
        wp.launch(
            emit,
            dim=n_cut,
            inputs=[
                faces,
                face_indices,
                face_signs,
                edge_points,
                wp.int32(vertex_base),
                all_vertices[vertex_base : vertex_base + 2 * n_cut],
                all_faces[face_base : face_base + 3 * n_emitted].reshape((n_emitted, 3)),
            ],
            device=device,
        )
        vertex_base += 2 * n_cut
        face_base += 3 * n_emitted

    new_vertices, new_faces, _ = tw.repair.remove_unreferenced_vertices(all_vertices, all_faces)
    return new_vertices, new_faces


def _slice_class_partition(
    face_classes: wp.array[wp.int32], n_faces: int, n_classes: int
) -> tuple[list[wp.array[wp.int32]], list[int]]:
    """
    Split the classified faces into the inside, cut-quad and cut-tri index arrays.

    The three classes are disjoint, so their selection flags live as three blocks of one buffer and
    one inclusive scan compacts all of them: block ``b``'s indices land contiguously, and the
    running totals at the block ends give the three counts in a single 12-byte readback. Each
    returned array is a contiguous view of one buffer, ascending in face index like the
    [`flatnonzero`][triwarp.array.flatnonzero] calls it replaces.
    """
    device = face_classes.device
    blocked = n_classes * n_faces
    flags = wp.empty(blocked, dtype=wp.int32, device=device)
    wp.launch(
        kernel_intersections.slice_class_flags,
        dim=n_faces,
        inputs=[face_classes, wp.int32(n_faces), wp.int32(n_classes), flags],
        device=device,
    )
    inclusive = wp.empty(blocked, dtype=wp.int32, device=device)
    wp.utils.array_scan(flags, out_array=inclusive, inclusive=True)
    counts = wp.empty(n_classes, dtype=wp.int32, device=device)
    wp.launch(
        kernel_intersections.slice_class_counts,
        dim=1,
        inputs=[inclusive, wp.int32(n_faces), wp.int32(n_classes), counts],
        device=device,
    )
    # The one host synchronization in the slice: these counts size every buffer downstream.
    class_counts = [int(count) for count in counts.numpy()]

    indices = wp.empty(sum(class_counts), dtype=wp.int32, device=device)
    wp.launch(
        kernel_intersections.scatter_slice_class,
        dim=blocked,
        inputs=[flags, inclusive, wp.int32(n_faces), indices],
        device=device,
    )

    def block(start: int, count: int) -> wp.array[wp.int32]:
        # Warp rejects a zero-length slice outright, so an empty class gets its own empty buffer.
        if count == 0:
            return wp.empty(0, dtype=wp.int32, device=device)
        return indices[start : start + count]

    starts = [sum(class_counts[:block_index]) for block_index in range(n_classes)]
    return [block(start, count) for start, count in zip(starts, class_counts, strict=True)], (
        class_counts
    )
