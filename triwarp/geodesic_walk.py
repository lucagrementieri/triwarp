"""
Straightest geodesics: walk a fixed distance across a mesh in a fixed direction.

A *straightest* geodesic is what you get by walking forward and, at every edge, unfolding the two
incident triangles into a common plane and continuing in a straight line. It is the surface analogue
of "go that way for this far", which makes it the tool for exponential maps, streamline tracing
and
extending a direction field along a surface — and unlike a shortest path it is fixed by an initial
condition rather than by two endpoints.

Both entry points are batched over many rays: one thread walks one ray, and the traced polylines
come back packed into one buffer with CSR offsets, the same shape
[`boundary_loops_batched`][triwarp.boundary.boundary_loops_batched] uses.

[`trace_from_vertex`][triwarp.geodesic_walk.trace_from_vertex] is the **exponential map** of the
surface, and its inverse is [`log_map`][triwarp.heat.vector.log_map]: one takes a tangent direction
and a distance to a point, the other takes a point back to the direction and distance that reach it.
The two live apart because they are different methods -- this one unfolds triangles combinatorially,
that one solves a vector-heat system -- and geodesic *distance* by the heat method is a third,
[`heat_geodesic`][triwarp.heat.distance.heat_geodesic].

!!! note "Cone points"
    A path that runs exactly into a vertex has no unique straightest continuation — the angle around
    a vertex is not ``2 * pi``, so "straight through" is ambiguous. Such a crossing is resolved by
    unfolding across the edge the walk reached the vertex along: the correct limit for a path
    passing arbitrarily close to the vertex, but not the split-the-angle convention. Paths through
    high-curvature vertices therefore drift from geometry-central's by about the angle defect.
"""

from __future__ import annotations

import warp as wp

import triwarp as tw
from triwarp.halfedge import halfedge_twins, vertex_one_rings
from triwarp.kernels import geodesic_walk as kernel_geodesic_walk

_DEFAULT_MAX_STEPS = 4096


def trace_from_vertex(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    start_vertices: wp.array[wp.int32],
    directions: wp.array[wp.vec3],
    twins: wp.array[wp.int32] | None = None,
    rings: tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.bool]] | None = None,
    frames: tuple[wp.array[wp.vec3], wp.array[wp.vec3], wp.array[wp.vec3]] | None = None,
    max_steps: int = _DEFAULT_MAX_STEPS,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Trace a straightest geodesic from each of a batch of vertices.

    Each ray starts at ``vertices[start_vertices[r]]`` and walks in ``directions[r]``, for an arc
    length equal to that direction's length *after projection into the vertex's tangent plane* — a
    unit direction traces at most a unit distance, and a direction along the normal traces nothing.
    That is ``potpourri3d.GeodesicTracer``'s convention too.

    Which incident face the ray starts in is decided in the vertex's *flattened* tangent space
    ([`halfedge_tangent_angles`][triwarp.tangent_space.halfedge_tangent_angles]): rescaling the
    incident corner angles to a full turn makes the fan a disk, so every tangent direction lands in
    exactly one wedge, including directions a naive per-face projection would place outside all of
    them. At a boundary vertex the fan spans only half a disk, and a direction outside it — pointing
    off the surface — traces nothing.

    A walk stops early when it reaches the mesh boundary or exceeds ``max_steps`` edge crossings.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    start_vertices
        ``(n_rays,)`` ``wp.int32`` start vertex per ray.
    directions
        ``(n_rays,)`` initial directions; the length of each sets how far its ray is traced.
    twins
        Optional precomputed [`halfedge_twins`][triwarp.halfedge.halfedge_twins].
    rings
        Optional precomputed [`vertex_one_rings`][triwarp.halfedge.vertex_one_rings], used to find
        each ray's starting face.
    frames
        Optional precomputed [`vertex_tangent_frames`][triwarp.tangent_space.vertex_tangent_frames].
        They define each vertex's tangent plane, which sets both the trace length and the starting
        wedge.
    max_steps
        Maximum edge crossings per ray.

    Returns
    -------
    points : wp.array[wp.vec3]
        All traced points, packed ray after ray, on ``vertices.device``.
    offsets : wp.array[wp.int32]
        Length ``n_rays + 1``; ray ``r`` owns ``points[offsets[r] : offsets[r + 1]]``, beginning at
        its start vertex. A ray always contributes at least one point.

    See Also
    --------
    [`trace_from_face`][triwarp.geodesic_walk.trace_from_face]
    [`trace_polylines`][triwarp.geodesic_walk.trace_polylines]
    [`heat_geodesic`][triwarp.heat.distance.heat_geodesic]
    """
    device = vertices.device
    n_rays = int(start_vertices.shape[0])
    n_vertices = int(vertices.shape[0])
    if n_rays == 0 or int(faces.shape[0]) == 0:
        return wp.empty(0, dtype=wp.vec3, device=device), wp.zeros(
            max(n_rays + 1, 1), dtype=wp.int32, device=device
        )

    if twins is None:
        twins = halfedge_twins(faces, n_vertices=n_vertices)
    if rings is None:
        rings = vertex_one_rings(faces, twins=twins, n_vertices=n_vertices)
    ring_offsets, ring_halfedges, is_boundary = rings
    if frames is None:
        frames = tw.tangent_space.vertex_tangent_frames(vertices, faces, rings=rings)
    basis_x, basis_y, normals = frames

    inputs = [
        vertices,
        faces,
        twins,
        tw.triangles.face_angles(vertices, faces),
        ring_offsets,
        ring_halfedges,
        is_boundary,
        basis_x,
        basis_y,
        normals,
        start_vertices,
        directions,
        wp.int32(max_steps),
        wp.float32(_length_epsilon(vertices, faces)),
    ]
    return _trace(kernel_geodesic_walk.trace_from_vertices, inputs, n_rays, device)


def trace_from_face(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    start_faces: wp.array[wp.int32],
    start_barycentric: wp.array[wp.vec3],
    directions: wp.array[wp.vec3],
    twins: wp.array[wp.int32] | None = None,
    max_steps: int = _DEFAULT_MAX_STEPS,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Trace a straightest geodesic from each of a batch of barycentric points inside faces.

    As [`trace_from_vertex`][triwarp.geodesic_walk.trace_from_vertex], but each ray
    starts at an interior point of a known face, so no wedge search is needed. This is the form for
    streamlines of a face-based vector field.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    start_faces
        ``(n_rays,)`` ``wp.int32`` starting face per ray.
    start_barycentric
        ``(n_rays,)`` barycentric coordinates of each start point within its face.
    directions
        ``(n_rays,)`` initial directions; the length of each sets how far its ray is traced.
    twins
        Optional precomputed [`halfedge_twins`][triwarp.halfedge.halfedge_twins].
    max_steps
        Maximum edge crossings per ray.

    Returns
    -------
    points : wp.array[wp.vec3]
        All traced points, packed ray after ray, on ``vertices.device``.
    offsets : wp.array[wp.int32]
        Length ``n_rays + 1`` CSR bounds into ``points``.

    See Also
    --------
    [`trace_from_vertex`][triwarp.geodesic_walk.trace_from_vertex]
    [`barycentric_to_points`][triwarp.triangles.barycentric_to_points]
    """
    device = vertices.device
    n_rays = int(start_faces.shape[0])
    if n_rays == 0 or int(faces.shape[0]) == 0:
        return wp.empty(0, dtype=wp.vec3, device=device), wp.zeros(
            max(n_rays + 1, 1), dtype=wp.int32, device=device
        )

    if twins is None:
        twins = halfedge_twins(faces, n_vertices=int(vertices.shape[0]))

    inputs = [
        vertices,
        faces,
        twins,
        start_faces,
        start_barycentric,
        directions,
        wp.int32(max_steps),
        wp.float32(_length_epsilon(vertices, faces)),
    ]
    return _trace(kernel_geodesic_walk.trace_from_faces, inputs, n_rays, device)


def trace_polylines(
    points: wp.array[wp.vec3], offsets: wp.array[wp.int32], *, copy: bool = False
) -> list[wp.array[wp.vec3]]:
    """
    Slice a packed trace result into one polyline per ray.

    !!! note "The returned arrays are views"
        Each polyline slices the packed buffer, so holding one keeps them all alive and writing into
        one writes into the shared allocation. Pass ``copy=True`` for independent buffers.

    Parameters
    ----------
    points
        Packed traced points from
        [`trace_from_vertex`][triwarp.geodesic_walk.trace_from_vertex] or
        [`trace_from_face`][triwarp.geodesic_walk.trace_from_face].
    offsets
        The matching length-``n_rays + 1`` CSR bounds.
    copy
        Return independent buffers instead of views.

    Returns
    -------
    list[wp.array[wp.vec3]]
        One open polyline per ray, in ray order.

    See Also
    --------
    [`polyline_length`][triwarp.polyline.polyline_length]
    [`boundary_loops`][triwarp.boundary.boundary_loops]
    """
    # ``offsets`` is the total-terminated n + 1 form, and ``split`` wants the length-n one, whose
    # last segment already runs to the end of ``points``. The guard is required rather than
    # defensive: a no-ray trace returns a length-1 offsets array, and Warp rejects the resulting
    # zero-length slice outright ("Invalid indexing in slice: 0:0:1").
    if int(offsets.shape[0]) <= 1:
        return []
    return tw.array.split(points, offsets[:-1], copy=copy)


def _length_epsilon(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> float:
    """
    Smallest crossing distance the walk will accept, as a fraction of the mean edge length.

    The walk needs *some* scale-aware floor: a ray starting at a vertex stands on two of its face's
    edges, and one starting on an edge stands on that edge, so a crossing at distance ~0 has to be
    rejected or the walk rotates on the spot. Tying it to the mean edge length keeps the behaviour
    invariant under a global rescale of the mesh.
    """
    return 1e-6 * tw.edges.mean_edge_length(vertices, faces)


def _trace(
    kernel: wp.Kernel, inputs: list, n_rays: int, device: wp.DeviceLike
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Run a tracing kernel twice: once to count each ray's points, once to write them.

    The walk is cheap arithmetic and its length is not known in advance, so counting first and
    sizing the output exactly beats reserving ``max_steps`` points per ray, which at the default cap
    would be tens of megabytes for a few thousand rays.
    """
    counts = wp.empty(n_rays, dtype=wp.int32, device=device)
    no_offsets = wp.empty(0, dtype=wp.int32, device=device)
    no_points = wp.empty(0, dtype=wp.vec3, device=device)
    wp.launch(kernel, dim=n_rays, inputs=[*inputs, no_offsets, counts, no_points], device=device)

    # Host readback: only the device knows the walk's total length, and it sizes the point buffer.
    offsets, total = tw.array.counts_to_offsets(counts, include_total=True)
    points = wp.empty(total, dtype=wp.vec3, device=device)
    wp.launch(kernel, dim=n_rays, inputs=[*inputs, offsets, counts, points], device=device)
    return points, offsets
