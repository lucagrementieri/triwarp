"""
Queries that ask where a point stands relative to a triangle mesh.

Three answers, in increasing order of what they need from the mesh:
[`closest_point_on_mesh`][triwarp.proximity.closest_point_on_mesh] and
[`normals_at_closest_faces`][triwarp.proximity.normals_at_closest_faces] need only a surface;
[`signed_distance_on_mesh`][triwarp.proximity.signed_distance_on_mesh] needs a consistent winding
to give the distance a sign; and [`winding_number`][triwarp.proximity.winding_number] needs neither
watertightness nor manifoldness, which is why it is the robust inside test on damaged input.
[`containing_faces_2d`][triwarp.proximity.containing_faces_2d] is the planar case -- point location
in a 2D triangulation -- and
[`query_mesh_aabb_with_offsets`][triwarp.proximity.query_mesh_aabb_with_offsets] is
the low-level box query the others are built over.

Everything here takes raw ``(vertices, faces)`` buffers. The queries phrased the other way round --
*how far away* the surface is rather than *where* it is, all taking a prebuilt ``wp.Mesh`` and a set
of points to measure at -- live in [`triwarp.visibility`][triwarp.visibility]: ambient occlusion and
obscurance outward, shape diameter and thickness inward, and the maximal tangent sphere in every
direction at once.

Everything signed here, plus [`contains_points`][triwarp.ray.contains_points], follows Warp's SDF
sign convention: outside positive, inside negative. Trimesh's ``signed_distance`` uses the opposite
sign.

Point-set acceleration structures (``wp.Bvh`` / ``wp.HashGrid``) and raw neighbor queries live in
[`triwarp.neighbors`][triwarp.neighbors]; axis-aligned bounding boxes in
[`triwarp.bounds`][triwarp.bounds].
"""

from __future__ import annotations

import math
from typing import Literal

import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp._device import require_nonempty_mesh
from triwarp.kernels import proximity as kernel_proximity
from triwarp.kernels import triangles as kernel_triangles
from triwarp.triangles import face_normals_and_areas

# Elements per thread for the two *per-query* lane-free reductions here (solid-angle sum, packed
# support argmax). Unlike a global reduction, these have one accumulator per query, so the query
# dimension already supplies the parallelism and a short slice only multiplies the atomic traffic:
# measured on 20k faces x 5000 queries, 128 runs 0.49 ms against 3.7 ms at 8 and 13.8 ms at 4. The
# optimum is flat over 128-256 and degrades again past 512, so this is not a sensitive knob.
ITEMS_PER_QUERY_SLICE = 128

# First search radius for [`closest_point_on_edges`][triwarp.proximity.closest_point_on_edges], as a
# fraction of the ``max_dist`` its deepening loop is capped at. The loop doubles from here and jumps
# straight to the certified radius as soon as it holds any candidate, so this only decides how many
# empty scans a query far from every edge pays; too *large* a start is the expensive mistake, since
# the first scan then enumerates the whole edge set.
_EDGE_INITIAL_RADIUS_SCALE = 0.01

# Ray-origin offset *below* the surface along the inward normal, as a fraction of the query AABB
# diagonal: without it the cone's own starting triangle is the nearest hit for every ray.
_SDF_SURFACE_OFFSET = 1e-4

# [`containing_faces_2d`][triwarp.proximity.containing_faces_2d]'s candidate search radius, as a
# fraction of the triangulation's bounding-box diagonal, and the barycentric slack that then decides
# containment. The radius only has to exceed the float32 rounding of a closest-point query on a flat
# mesh (measured at ~1e-5 of the diagonal), and being generous costs only BVH descent on queries
# that land outside; the sign test classifies, so the two are not a precision trade-off.
_CONTAINMENT_SEARCH_SCALE = 1e-3
_CONTAINMENT_BARYCENTRIC_EPS = wp.float32(1e-6)


def closest_point_on_mesh(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    points: wp.array[wp.vec3],
    *,
    max_dist: float | None = None,
    mesh: wp.Mesh | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.float32], wp.array[wp.int32]]:
    """
    For each query point, find the closest point on any triangle of the mesh.

    Uses ``wp.mesh_query_point_no_sign`` via ``wp.Mesh``. Distances are unsigned
    Euclidean lengths in ``float32``.

    Parameters
    ----------
    vertices
        ``(n,)`` mesh vertex positions as ``wp.vec3``.
    faces
        ``(f * 3,)`` flat triangle index array as ``wp.int32``.
        The internally built ``wp.Mesh`` aliases these buffers rather than copying them;
        do not mutate them for the duration of the call.
    points
        ``(m,)`` query positions in space as ``wp.vec3``.
    max_dist
        Maximum search radius per query. Faces farther than this are ignored.
        When ``None``, derived from the axis-aligned box enclosing mesh
        vertices and query points.
    mesh
        A ``wp.Mesh`` already built over ``vertices`` and ``faces``, to spare the clone and BVH
        build this otherwise pays on every call. Purely an optimization: the answer is identical
        either way, and it is not checked against ``vertices`` / ``faces`` -- passing a mesh over
        *different* geometry silently answers for that geometry, since only ``mesh`` is queried.
        Measured saving on an RTX 5090: a flat **0.15-0.27 ms** (the clone plus the build), so 32%
        of a single-query call and 1.7% of a 100 000-query one on 82k faces.
        [`Trimesh.warp_mesh`][triwarp.mesh.Trimesh.warp_mesh] is a cached property and is what to
        pass.

    Returns
    -------
    closest
        ``(m, 3)`` closest point on the mesh surface for each query.
    distance
        ``(m,)`` unsigned distance from each query to its closest surface point.
    triangle_id
        ``(m,)`` index of the triangle containing each closest point, or ``-1``
        when no face lies within ``max_dist``.
    """
    device = vertices.device
    m = int(points.shape[0])
    n_faces = int(faces.shape[0]) // 3
    if m == 0:
        return (
            wp.empty(0, dtype=wp.vec3, device=device),
            wp.empty(0, dtype=wp.float32, device=device),
            wp.empty(0, dtype=wp.int32, device=device),
        )
    if n_faces == 0:
        nan = float("nan")
        out_closest = wp.full(m, wp.vec3(nan, nan, nan), dtype=wp.vec3, device=device)
        out_distance = wp.full(m, float("inf"), dtype=wp.float32, device=device)
        out_face = wp.full(m, -1, dtype=wp.int32, device=device)
        return out_closest, out_distance, out_face

    if mesh is None:
        require_nonempty_mesh(faces, "closest_point_on_mesh")
        # The mesh aliases the caller's buffers and is discarded here, so it needs no copy.
        mesh = wp.Mesh(points=vertices, indices=faces)
    if max_dist is None:
        max_dist = tw.bounds.enclosing_diagonal(mesh.points, points)

    out_closest = wp.empty(m, dtype=wp.vec3, device=device)
    out_distance = wp.empty(m, dtype=wp.float32, device=device)
    out_face = wp.empty(m, dtype=wp.int32, device=device)
    wp.launch(
        kernel_proximity.closest_point_on_mesh,
        dim=m,
        inputs=[mesh.id, points, wp.float32(max_dist), out_closest, out_distance, out_face],
        device=device,
    )
    return out_closest, out_distance, out_face


def closest_point_on_edges(
    vertices: wp.array[wp.vec3],
    edges: twt.Array2dInt32,
    queries: wp.array[wp.vec3],
    *,
    max_dist: float | None = None,
    bvh: wp.Bvh | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.float32], wp.array[wp.int32]]:
    """
    For each query point, find the closest point on any edge of an edge set.

    The **wireframe** counterpart of
    [`closest_point_on_mesh`][triwarp.proximity.closest_point_on_mesh]: same return triple, same
    ``max_dist`` semantics, but the geometry is a set of segments rather than a surface. That is the
    query a crease set, a seam, a boundary rim or a feature curve wants -- all four are already
    produced as an edge array by [`triwarp.seams`][triwarp.seams],
    [`triwarp.boundary`][triwarp.boundary] and [`triwarp.edges`][triwarp.edges], and none of them
    could be measured against before.

    Parameters
    ----------
    vertices
        ``(n,)`` positions the edges index, as ``wp.vec3``.
    edges
        ``(n_edges, 2)`` ``wp.int32`` vertex-index pairs. Order within a pair is irrelevant, and
        edges may share vertices or repeat.
    queries
        ``(m,)`` query positions in space as ``wp.vec3``.
    max_dist
        Maximum search distance per query; an edge farther than this is ignored and the query
        reports a miss. When ``None``, derived from the box enclosing ``vertices`` and ``queries``,
        which no real query can exceed.
    bvh
        A ``wp.Bvh`` already built over this edge set's per-edge boxes, to spare the bounds pass and
        the build. Purely an optimization -- and, as with
        [`closest_point_on_mesh`][triwarp.proximity.closest_point_on_mesh]'s ``mesh``, it is not
        checked against ``vertices`` / ``edges``: a BVH over a *different* edge set silently answers
        for the boxes it holds while the distances are computed from these vertices.

    Returns
    -------
    closest
        ``(m,)`` closest point on the edge set for each query, as ``wp.vec3``.
    distance
        ``(m,)`` unsigned distance from each query to that point.
    edge_id
        ``(m,)`` row of ``edges`` the closest point lies on, or ``-1`` when no edge lies within
        ``max_dist``.

    Raises
    ------
    ValueError
        If ``edges`` is not a rank-2 ``wp.int32`` array with two columns.

    Notes
    -----
    A miss reports the query point itself and ``max_dist``, alongside the ``-1`` id -- the same
    convention [`closest_point_on_mesh`][triwarp.proximity.closest_point_on_mesh] uses, so the two
    are interchangeable in a caller that only reads the id.

    The traversal is iterative deepening over a per-edge-box BVH and is **exact**, not a broad-phase
    approximation: a scan of the cube of half-extent ``r`` about a query enumerates every edge whose
    closest point lies within ``r``, so a best distance under ``r`` certifies the answer. The
    tempting shortcut -- one degenerate ``(a, b, b)`` triangle per edge, queried with
    ``wp.mesh_query_point_no_sign`` -- does **not** work: Warp's mesh BVH rejects a zero-area
    triangle, measured as 64 misses out of 64 queries on both devices (Warp 1.16).

    See Also
    --------
    [`closest_point_on_mesh`][triwarp.proximity.closest_point_on_mesh]
        The surface form. On a closed mesh its answer is never farther than this one's.
    [`triwarp.polyline.polyline_point_distance`][triwarp.polyline.polyline_point_distance]
        The same computation for an *ordered* chain, where the segments are consecutive vertices and
        no index structure is built.
    """
    device = vertices.device
    twt.ensure_ndim(edges, 2, dtype=wp.int32)
    if int(edges.shape[1]) != 2:
        raise ValueError("edges must have two columns")
    m = int(queries.shape[0])
    n_edges = int(edges.shape[0])

    if m == 0:
        return (
            wp.empty(0, dtype=wp.vec3, device=device),
            wp.empty(0, dtype=wp.float32, device=device),
            wp.empty(0, dtype=wp.int32, device=device),
        )
    if n_edges == 0:
        return (
            wp.clone(queries),
            wp.full(m, float("inf"), dtype=wp.float32, device=device),
            wp.full(m, -1, dtype=wp.int32, device=device),
        )

    if bvh is None:
        lower = wp.empty(n_edges, dtype=wp.vec3, device=device)
        upper = wp.empty(n_edges, dtype=wp.vec3, device=device)
        wp.launch(
            kernel_proximity.edge_bounds,
            dim=n_edges,
            inputs=[vertices, edges, lower, upper],
            device=device,
        )
        bvh = tw.neighbors.bvh_from_bounds(lower, upper)
    if max_dist is None:
        max_dist = tw.bounds.enclosing_diagonal(vertices, queries)
    # The scene box bounds each query's *complete* search radius, so a query outside the geometry
    # still terminates exactly rather than growing to ``max_dist``.
    min_bound, max_bound = tw.bounds.aabb(vertices)

    out_closest = wp.empty(m, dtype=wp.vec3, device=device)
    out_distance = wp.empty(m, dtype=wp.float32, device=device)
    out_edge = wp.empty(m, dtype=wp.int32, device=device)
    wp.launch(
        kernel_proximity.closest_point_on_edges,
        dim=m,
        inputs=[
            vertices,
            edges,
            queries,
            bvh.id,
            wp.float32(max_dist),
            wp.float32(_EDGE_INITIAL_RADIUS_SCALE * max_dist),
            min_bound,
            max_bound,
            out_closest,
            out_distance,
            out_edge,
        ],
        device=device,
    )
    return out_closest, out_distance, out_edge


def mesh_to_mesh_distance(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    *,
    upper_bound: float | None = None,
) -> tuple[float, int, int]:
    """
    Smallest distance between two triangle meshes, with the pair of faces that achieves it.

    The clearance between two parts, and zero when they touch or overlap. Unlike a vertex-to-mesh
    query this is the true minimum over the *surfaces*: two boxes edge to edge realise their
    clearance between edge interiors, and every vertex of each is further from the other than
    that.

    Two phases. An upper bound comes first -- the smallest distance from a vertex of ``A`` to ``B``,
    which is a real distance between the surfaces and therefore an upper bound on their minimum.
    Then every face of ``A`` queries a BVH over ``B``'s faces with its own bounding box grown by
    that bound, and each candidate pair gets the exact triangle-triangle distance. The bound is what
    makes the broad phase sound rather than heuristic: the true minimum is at most the bound, so the
    pair achieving it has boxes within that distance and cannot be culled.

    Parameters
    ----------
    vertices_a, faces_a
        First mesh: ``(n_vertices,)`` positions and a length-``3 * n_faces`` index buffer.
    vertices_b, faces_b
        Second mesh, same layout.
    upper_bound
        A distance known to be at least the answer, which prunes the broad phase. Supply one when
        you have it -- from a previous frame, or from a bounding-volume gap -- and the vertex query
        that would otherwise derive it is skipped. **Too small a bound gives a wrong answer**, not a
        slow one: it culls the pair that would have won. ``None`` derives a sound bound.

    Returns
    -------
    distance : float
        The minimum distance. ``0.0`` exactly when some pair of faces crosses.
    face_a : int
        The face of the first mesh achieving it, or ``-1`` if either mesh is empty.
    face_b : int
        The face of the second mesh achieving it, or ``-1``.

    Raises
    ------
    ValueError
        If ``upper_bound`` is negative.

    !!! note "Witness faces are ambiguous under ties"
        Two parallel plates have a continuum of closest pairs and any of them is a correct answer;
        the tie-break here is the lowest ``face_a``, then whichever ``face_b`` that face's own scan
        reached first. Compare *distances* against another implementation, and faces only where the
        configuration is generic.

    See Also
    --------
    [`closest_point_on_mesh`][triwarp.proximity.closest_point_on_mesh]
        Point-to-mesh, which is the query this derives its bound from.
    [`mesh_with_mesh`][triwarp.intersection.mesh_with_mesh]
        The zero-distance case in detail: every intersecting pair and the segments they cross on.
    [`face_self_intersecting_mask`][triwarp.validation.face_self_intersecting_mask]
        The one-mesh analogue of that.
    """
    if upper_bound is not None and upper_bound < 0.0:
        raise ValueError(f"upper_bound must be non-negative, got {upper_bound}")
    device = faces_a.device
    n_faces_a = int(faces_a.shape[0]) // 3
    n_faces_b = int(faces_b.shape[0]) // 3
    if n_faces_a == 0 or n_faces_b == 0:
        return math.inf, -1, -1

    if upper_bound is None:
        # A vertex-to-surface distance is a distance between the surfaces, so its minimum bounds the
        # answer from above. One readback, and it is what lets the broad phase cull at all.
        _points, distances, _faces = closest_point_on_mesh(vertices_b, faces_b, vertices_a)
        upper_bound = float(tw.reduce.min(distances))

    lower = wp.empty(n_faces_b, dtype=wp.vec3, device=device)
    upper = wp.empty(n_faces_b, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_triangles.face_aabb_bounds,
        dim=n_faces_b,
        inputs=[vertices_b, faces_b, lower, upper],
        device=device,
    )
    bvh = tw.neighbors.bvh_from_bounds(lower, upper)
    distance_sq = wp.empty(n_faces_a, dtype=wp.float32, device=device)
    witness = wp.empty(n_faces_a, dtype=wp.int32, device=device)
    wp.launch(
        kernel_proximity.face_to_mesh_distance,
        dim=n_faces_a,
        inputs=[
            vertices_a,
            faces_a,
            vertices_b,
            faces_b,
            lower,
            upper,
            bvh.id,
            wp.float32(upper_bound),
            # Seeded at infinity, **not** at ``upper_bound ** 2``: when the bound *is* the answer
            # -- two spheres whose closest points are vertices -- seeding it there prunes the very
            # pair that achieves it and the result comes back ``inf``.
            wp.full(1, math.inf, dtype=wp.float32, device=device),
            distance_sq,
            witness,
        ],
        device=device,
    )
    keys = wp.empty(n_faces_a, dtype=wp.int64, device=device)
    wp.launch(
        kernel_proximity.face_distance_keys,
        dim=n_faces_a,
        inputs=[distance_sq, keys],
        device=device,
    )
    # The key's low 32 bits are the winning face, so one reduction and one 8-byte read give both the
    # distance and the argmin -- no second pass over the candidates.
    best_face_a = int(tw.reduce.min(keys)) & 0xFFFFFFFF
    return (
        math.sqrt(float(distance_sq.numpy()[best_face_a])),
        best_face_a,
        int(witness.numpy()[best_face_a]),
    )


def normals_at_closest_faces(
    mesh: wp.Mesh, points: wp.array[wp.vec3], *, max_dist: float | None = None
) -> wp.array[wp.vec3]:
    """
    Return unit face normals at the closest mesh triangle for each query point.

    For each position in ``points``, runs an unsigned closest-point query on
    ``mesh`` and returns the normal of the hit triangle.

    Parameters
    ----------
    mesh
        Warp mesh (BVH built by caller).
    points
        ``(m,)`` query positions as ``wp.vec3``.
    max_dist
        Maximum search radius per query. When ``None``, derived from the
        axis-aligned box enclosing mesh vertices and query points.

    Returns
    -------
    wp.array[wp.vec3]
        ``(m,)`` face normals at the closest triangle for each query. A query with no face
        within ``max_dist`` reports the first face's normal; use
        [`closest_point_on_mesh`][triwarp.proximity.closest_point_on_mesh] directly, whose
        ``triangle_id`` is ``-1`` there, when a miss has to be detected.

    See Also
    --------
    [`closest_point_on_mesh`][triwarp.proximity.closest_point_on_mesh]
    """
    device = points.device
    m = int(points.shape[0])
    if m == 0:
        return wp.empty(0, dtype=wp.vec3, device=device)

    _closest, _distance, out_face = closest_point_on_mesh(
        mesh.points, mesh.indices, points, max_dist=max_dist, mesh=mesh
    )
    # A miss leaves ``-1``, which would gather out of bounds; clamping to face 0 costs one map
    # over ``m`` and keeps the read in range.
    hit_face = wp.empty(m, dtype=wp.int32, device=device)
    wp.map(wp.max, out_face, wp.int32(0), out=hit_face)
    all_face_normals, _ = face_normals_and_areas(mesh.points, mesh.indices)
    return tw.array.gather(all_face_normals, hit_face)


def signed_distance_on_mesh(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    points: wp.array[wp.vec3],
    *,
    max_dist: float | None = None,
    sign_mode: Literal["parity", "winding"] = "parity",
    n_sample: int = 5,
    perturbation_scale: float = 0.1,
    accuracy: float = 2.0,
    winding_threshold: float = 0.5,
    mesh: wp.Mesh | None = None,
) -> wp.array[wp.float32]:
    """
    Signed distance from each query point to a triangle mesh (Warp SDF convention).

    Distances follow Warp's signed-distance field convention:

    * Points **outside** the mesh have **positive** distance.
    * Points **inside** have **negative** distance.
    * Points within [`TOLERANCE_MERGE`][triwarp.constants.TOLERANCE_MERGE] of the surface
      return positive unsigned distance.

    Trimesh ``signed_distance`` uses the opposite sign; negate its output to compare.
    See also [`contains_points`][triwarp.ray.contains_points] (inside iff signed distance is
    negative, except on the on-surface tolerance band).

    The **unsigned** distance is identical in both ``sign_mode`` values — only the sign differs.

    !!! note "Choosing a `sign_mode`"

        ``"parity"`` (default) uses ``wp.mesh_query_point_sign_parity``: it casts ``n_sample``
        perturbed rays and votes on the crossing parity. Exact on a watertight mesh, cheap, but
        it has no principled answer on an open or holed surface — a ray that escapes through a
        hole flips the verdict.

        ``"winding"`` uses ``wp.mesh_query_point_sign_winding_number``, which evaluates the
        *generalized winding number* on the mesh BVH (a Barnes-Hut style traversal governed by
        ``accuracy``) and compares it against ``winding_threshold``. This is the
        Jacobson et al. robust inside/outside criterion and it degrades gracefully on
        non-watertight input, which is why it is the mode to reach for on raw scan data.

        Measured on this repo's fixtures: the two modes agree on watertight meshes
        (icosahedron, ``cave_cube``), but on a sphere with a patch of faces removed ``"winding"``
        reproduces the exact generalized winding number's sign on 100% of query points while
        ``"parity"`` manages 93.2%. The costs are a 1.2-1.5x slower query
        (``benchmarks/test_proximity.py``) and a substantially larger ``wp.Mesh``:
        ``support_winding_number=True`` stores a solid-angle expansion per BVH node, measured at
        roughly 3x the mesh's device memory (+235 MB on dragon's 871k faces).

        ``"winding"`` is still much cheaper than thresholding
        [`winding_number`][triwarp.proximity.winding_number] yourself, because that sums the exact
        solid angle over *every* face for *every* query: at 10k queries the same sign decision costs
        8.1 ms this way versus 167 ms exactly on dragon (871k faces), and the gap widens with the
        face count. Reach for [`winding_number`][triwarp.proximity.winding_number] only when you
        need the winding *value* — Warp exposes no builtin for the approximated value, only its
        sign.

    Parameters
    ----------
    vertices
        ``(n,)`` mesh vertex positions as ``wp.vec3``.
    faces
        ``(f * 3,)`` flat triangle index array as ``wp.int32``.
        The internally built ``wp.Mesh`` aliases these buffers rather than copying them;
        do not mutate them for the duration of the call.
    points
        ``(m,)`` query positions in space as ``wp.vec3``.
    max_dist
        Maximum search radius per query. When ``None``, derived from the
        axis-aligned box enclosing mesh vertices and query points.
    sign_mode
        ``"parity"`` (default) for ray-parity sign, ``"winding"`` for the generalized
        winding-number sign. See the note above.
    n_sample
        Perturbed rays for parity voting (off-triangle sign branch). ``"parity"`` only.
    perturbation_scale
        Uniform perturbation scale for parity rays. ``"parity"`` only.
    accuracy
        Barnes-Hut accuracy for the winding-number traversal: a node is expanded when the query
        point is within ``accuracy`` times the node's radius, so larger values are more accurate
        and slower. ``"winding"`` only; Warp's default is ``2.0``.
    winding_threshold
        Winding number above which a point counts as inside. ``"winding"`` only; ``0.5`` is the
        standard choice for a once-wound closed surface.
    mesh
        A ``wp.Mesh`` already built over ``vertices`` and ``faces``, to spare the clone and BVH
        build. **Only valid with ``sign_mode="parity"``**: the winding mode needs a mesh built with
        ``support_winding_number=True``, and ``wp.Mesh`` exposes no way to read that flag back, so a
        supplied mesh cannot be checked and is refused rather than silently degraded to parity.
        See [`closest_point_on_mesh`][triwarp.proximity.closest_point_on_mesh] for the measured
        saving.

    Returns
    -------
    wp.array[wp.float32]
        ``(m,)`` signed distances in ``float32``.

    Raises
    ------
    ValueError
        If ``sign_mode`` is not ``"parity"`` or ``"winding"``, or if ``mesh`` is supplied together
        with ``sign_mode="winding"``.

    See Also
    --------
    [`winding_number`][triwarp.proximity.winding_number]
    [`contains_points`][triwarp.ray.contains_points]
    """
    if sign_mode not in ("parity", "winding"):
        raise ValueError(f"sign_mode must be 'parity' or 'winding', got {sign_mode!r}")

    device = vertices.device
    m = int(points.shape[0])
    n_faces = int(faces.shape[0]) // 3
    if m == 0:
        return wp.empty(0, dtype=wp.float32, device=device)
    if n_faces == 0:
        return wp.full(m, float("inf"), dtype=wp.float32, device=device)

    if mesh is not None and sign_mode == "winding":
        # wp.Mesh exposes no way to read back support_winding_number, so a supplied mesh cannot be
        # checked for the per-node solid-angle expansion the winding builtin needs -- and without it
        # the builtin silently degrades to ray parity. Refusing is the only safe answer; the parity
        # mode has no such requirement and accepts any mesh.
        raise ValueError(
            "sign_mode='winding' cannot use a supplied mesh: it needs "
            "wp.Mesh(support_winding_number=True), which cannot be verified after construction. "
            "Omit mesh=, or use sign_mode='parity'."
        )
    if mesh is None:
        require_nonempty_mesh(faces, "signed_distance_on_mesh")
        # The winding-number builtin silently degrades to ray parity unless the mesh carries the
        # per-node solid-angle expansion, so the flag is bound to sign_mode here rather than
        # exposed.
        # The mesh aliases the caller's buffers and is discarded here, so it needs no copy.
        mesh = wp.Mesh(
            points=vertices, indices=faces, support_winding_number=sign_mode == "winding"
        )
    if max_dist is None:
        max_dist = tw.bounds.enclosing_diagonal(mesh.points, points)
    out_distance = wp.empty(m, dtype=wp.float32, device=device)
    if sign_mode == "winding":
        wp.launch(
            kernel_proximity.signed_distance_on_mesh_winding,
            dim=m,
            inputs=[
                mesh.id,
                points,
                wp.float32(max_dist),
                wp.float32(accuracy),
                wp.float32(winding_threshold),
                out_distance,
            ],
            device=device,
        )
        return out_distance
    wp.launch(
        kernel_proximity.signed_distance_on_mesh,
        dim=m,
        inputs=[
            mesh.id,
            points,
            wp.float32(max_dist),
            wp.int32(n_sample),
            wp.float32(perturbation_scale),
            out_distance,
        ],
        device=device,
    )
    return out_distance


def signed_distance_grid(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    voxel_size: float | None = None,
    *,
    bounds: tuple[wp.vec3, wp.vec3] | None = None,
    pad: int = 2,
    sign_mode: Literal["parity", "winding"] = "parity",
    mesh: wp.Mesh | None = None,
) -> tuple[twt.Array3dFloat32, tuple[wp.vec3, wp.vec3]]:
    """
    Sample the signed distance to a mesh on a regular lattice, as a field and the box it spans.

    The bridge from a surface to a **level set**, and the missing half of the implicit round trip:
    [`triwarp.voxels.to_field`][triwarp.voxels.to_field] already gives
    [`triwarp.levelset.marching_cubes`][triwarp.levelset.marching_cubes] an
    *occupancy* lattice, but occupancy is ``0`` or ``1`` and thresholding it at anything other than
    ``0.5`` does not move the surface anywhere. A distance field does, which is what makes
    ``marching_cubes(*signed_distance_grid(...), iso=d)`` an offset surface at distance ``d`` --
    see [`triwarp.levelset.offset_mesh`][triwarp.levelset.offset_mesh], the named entry point for
    it.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    voxel_size
        Lattice spacing, isotropic. ``None`` takes
        [`triwarp.voxels.resolve_voxel_grid`][triwarp.voxels.resolve_voxel_grid]'s default of 1 % of
        the bounding-box diagonal, which is this package's one definition of an unspecified grid.
    bounds
        ``(lower, upper)`` box to sample, **before** padding. ``None`` uses the mesh's own
        axis-aligned box, which is what an offset wants -- an inward offset needs no more, and an
        outward one needs ``pad`` to cover it.
    pad
        Cells of margin added on every side, so the lattice extends ``pad * voxel_size`` beyond the
        box. Two is enough for the surface itself to be enclosed; an **outward offset of ``d``
        needs ``pad >= d / voxel_size + 1``** or its level set is clipped by the lattice boundary.
    sign_mode
        How the sign is decided, forwarded to
        [`signed_distance_on_mesh`][triwarp.proximity.signed_distance_on_mesh]: ``"parity"`` counts
        ray crossings, ``"winding"`` sums solid angles and is the one that survives a mesh with open
        rims.
    mesh
        A ``wp.Mesh`` already built over ``vertices`` and ``faces``, to spare the build. Forwarded
        as-is, so ``signed_distance_on_mesh``'s rule applies unchanged: it is usable with
        ``sign_mode="parity"`` only, since the winding sign needs a mesh built with
        ``support_winding_number=True`` and that cannot be verified after construction.

    Returns
    -------
    field : twt.Array3dFloat32
        ``(nx, ny, nz)`` signed distances, negative inside. Exactly the first argument
        [`triwarp.levelset.marching_cubes`][triwarp.levelset.marching_cubes] takes.
    bounds : tuple[wp.vec3, wp.vec3]
        The ``(lower, upper)`` corners the lattice actually spans, padded and snapped so the spacing
        is exactly ``voxel_size`` on every axis. Pass it straight through as that function's
        ``bounds``.

    Raises
    ------
    ValueError
        If ``voxel_size`` is not positive, ``pad`` is negative, or ``faces`` is empty.

    Examples
    --------
    ```python
    field, box = tw.proximity.signed_distance_grid(v, f, voxel_size=0.05, pad=4)
    shell_v, shell_f = tw.levelset.marching_cubes(field, 0.1, bounds=box)
    ```

    Notes
    -----
    **The whole lattice is sampled, so size it deliberately**: the field costs
    ``4 * nx * ny * nz`` bytes and one closest-point query per sample, which is 16.7 M queries and
    67 MB at ``256 ** 3``. There is no narrow band, and that is deliberate rather than missing --
    ``signed_distance_on_mesh`` reports ``+max_dist`` for a query that finds no face within the
    limit, so a banded field would carry a **positive** value deep inside the solid and silently
    invert the level set. Reduce the resolution instead.

    The lattice is a *corner* lattice: ``field[0, 0, 0]`` sits exactly on the returned ``lower``.
    That is [`triwarp.voxels.grid_points`][triwarp.voxels.grid_points]'s convention and
    ``marching_cubes``'s, and it is **not** the voxel-centre convention the rest of
    [`triwarp.voxels`][triwarp.voxels] uses.

    See Also
    --------
    [`signed_distance_on_mesh`][triwarp.proximity.signed_distance_on_mesh]
        The per-query form this samples, and where the sign conventions are documented.
    [`triwarp.levelset.offset_mesh`][triwarp.levelset.offset_mesh]
        What to call instead when the answer wanted is the offset surface rather than the field.
    [`triwarp.voxels.to_field`][triwarp.voxels.to_field]
        The occupancy lattice, when a binary inside test is all that is needed.
    """
    if pad < 0:
        raise ValueError("pad must be non-negative")
    if int(faces.shape[0]) == 0:
        raise ValueError("signed_distance_grid needs at least one face")
    device = vertices.device
    spacing, _origin = tw.voxels.resolve_voxel_grid(
        vertices, voxel_size, None, caller="signed_distance_grid"
    )

    lower, upper = bounds if bounds is not None else tw.bounds.aabb(vertices)
    margin = float(pad) * spacing
    lower = wp.vec3(lower[0] - margin, lower[1] - margin, lower[2] - margin)
    # One sample per spacing, and at least the two marching cubes needs to have a cell at all.
    shape = tuple(
        max(2, math.floor((float(upper[axis]) + margin - float(lower[axis])) / spacing) + 1)
        for axis in range(3)
    )
    snapped_upper = wp.vec3(
        *(float(lower[axis]) + (shape[axis] - 1) * spacing for axis in range(3))
    )

    samples = tw.voxels.grid_points(shape, bounds=(lower, snapped_upper), device=device)
    distances = signed_distance_on_mesh(vertices, faces, samples, sign_mode=sign_mode, mesh=mesh)
    return twt.as_array3d(distances.reshape(shape), wp.float32), (lower, snapped_upper)


def winding_number(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    points: wp.array[wp.vec3],
    *,
    tiled: bool = True,
) -> wp.array[wp.float32]:
    """
    Generalized winding number at each query point.

    Sums the signed solid angle subtended by each oriented triangle. For a
    closed, consistently oriented watertight mesh, interior points have
    winding number near ``1`` and exterior points near ``0``.

    Parameters
    ----------
    vertices
        ``(n,)`` mesh vertex positions as ``wp.vec3``.
    faces
        ``(f * 3,)`` flat triangle index array as ``wp.int32``.
    points
        ``(m,)`` query positions in space as ``wp.vec3``.
    tiled
        When ``True`` (default), sum solid angles with the face list partitioned across
        threads: one thread per ``(query, face slice)`` walks a strided slice of
        [`ITEMS_PER_QUERY_SLICE`][triwarp.proximity.ITEMS_PER_QUERY_SLICE] faces and accumulates
        one ``wp.atomic_add`` per slice, so the summation order is nondeterministic and the result
        can differ in the last float32 digits between runs. When ``False``, each query thread
        loops over all faces serially — orders of magnitude slower on large meshes,
        but the fixed left-to-right summation makes it the exact-sum reference.

    Returns
    -------
    wp.array[wp.float32]
        ``(m,)`` winding numbers in ``float32``.
    """
    device = points.device
    n_queries = int(points.shape[0])
    n_faces = int(faces.shape[0]) // 3
    if n_queries == 0:
        return wp.empty(0, dtype=wp.float32, device=device)
    if n_faces == 0:
        return wp.zeros(n_queries, dtype=wp.float32, device=device)

    out_winding = (
        wp.zeros(n_queries, dtype=wp.float32, device=device)
        if tiled
        else wp.empty(n_queries, dtype=wp.float32, device=device)
    )
    if tiled:
        n_face_slices = max(1, (n_faces + ITEMS_PER_QUERY_SLICE - 1) // ITEMS_PER_QUERY_SLICE)
        wp.launch(
            kernel_proximity.winding_number_tiled,
            dim=(n_queries, n_face_slices),
            inputs=[
                vertices,
                faces,
                wp.int32(n_faces),
                wp.int32(n_face_slices),
                points,
                out_winding,
            ],
            device=device,
        )
    else:
        wp.launch(
            kernel_proximity.winding_number,
            dim=n_queries,
            inputs=[vertices, faces, wp.int32(n_faces), points, out_winding],
            device=device,
        )
    return out_winding


def query_mesh_aabb_with_offsets(
    mesh: wp.Mesh,
    query_lower: wp.array[wp.vec3],
    query_upper: wp.array[wp.vec3],
    *,
    max_hits: int = 16,
) -> tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Low-level mesh AABB query with per-query axis-aligned bounds.

    For each query primitive ``k``, tests intersection of ``[query_lower[k],
    query_upper[k]]`` against every triangle in ``mesh`` via ``wp.mesh_query_aabb``.
    At most ``max_hits`` candidate face indices are recorded per query.

    Requires the default Warp mesh BVH backend; ``bvh_constructor="cubql"`` meshes
    do not support AABB queries.

    Parameters
    ----------
    mesh
        Target ``warp.Mesh`` built with the default BVH backend.
    query_lower
        Length-``m`` lower corners of the query boxes, on the target device.
    query_upper
        Length-``m`` upper corners of the query boxes.
    max_hits
        Maximum candidate faces recorded per query. Hits past this cap are dropped, so the
        result is a bounded sample rather than the full candidate set when a query straddles
        more than ``max_hits`` triangles.

    Returns
    -------
    candidate_indices_flat, offsets, hit_counts
        ``offsets`` is the exclusive prefix sum of per-query hit counts.
        Query ``k`` owns ``candidate_indices_flat[offsets[k] : offsets[k] + hit_counts[k]]``.
        All three are empty when ``m == 0``.

    Raises
    ------
    ValueError
        If ``query_lower`` and ``query_upper`` have different lengths, or ``max_hits < 1``.
    """
    device = query_lower.device
    m = int(query_lower.shape[0])
    if int(query_upper.shape[0]) != m:
        raise ValueError("query_lower and query_upper must have the same length")
    if max_hits < 1:
        raise ValueError("max_hits must be >= 1")

    if m == 0:
        return (
            wp.empty(0, dtype=wp.int32, device=device),
            wp.empty(0, dtype=wp.int32, device=device),
            wp.empty(0, dtype=wp.int32, device=device),
        )

    hit_counts = wp.empty(m, dtype=wp.int32, device=device)
    wp.launch(
        kernel_proximity.query_mesh_aabb_count,
        dim=m,
        inputs=[query_lower, query_upper, mesh.id, wp.int32(max_hits), hit_counts],
        device=device,
    )

    # One scan pass yields both the row starts and their total; an all-zero ``hit_counts`` scans to
    # all-zero offsets, which is exactly what the empty case wants to return.
    offsets, total_hits = tw.array.counts_to_offsets(hit_counts)
    if total_hits == 0:
        return wp.empty(0, dtype=wp.int32, device=device), offsets, hit_counts

    candidate_indices_flat = wp.empty(total_hits, dtype=wp.int32, device=device)
    wp.launch(
        kernel_proximity.query_mesh_aabb_neighbors,
        dim=m,
        inputs=[
            query_lower,
            query_upper,
            mesh.id,
            wp.int32(max_hits),
            offsets,
            candidate_indices_flat,
        ],
        device=device,
    )

    return candidate_indices_flat, offsets, hit_counts


def containing_faces_2d(
    vertices: wp.array[wp.vec2], faces: wp.array[wp.int32], points: wp.array[wp.vec2]
) -> wp.array[wp.int32]:
    """
    For each 2D query point, the triangle of a planar triangulation that contains it.

    Point location in the plane -- the primitive an inverse UV lookup needs, and the counterpart of
    [`triwarp.texture`][]'s forward direction: ``remap_attribute_from_uv`` *samples an image* at a
    UV coordinate, where this answers which triangle of a UV atlas a coordinate falls in, and so
    which surface point it corresponds to. Pair it with
    [`points_to_barycentric`][triwarp.triangles.points_to_barycentric] on the returned face to
    finish the inverse map.

    The triangulation must not overlap itself -- a UV atlas, a Delaunay triangulation, or the
    ``xy`` projection of a height field. Where triangles do overlap, each query gets one of the
    containing faces and which one is unspecified.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` planar vertex positions as ``wp.vec2``.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    points
        ``(m,)`` planar query positions as ``wp.vec2``.

    Returns
    -------
    wp.array[wp.int32]
        Length ``m`` on ``vertices.device``: the containing triangle's index, or ``-1`` where the
        query lies outside the triangulation. A query exactly on a shared edge is inside *both* its
        triangles and which one is returned is not specified.

    Notes
    -----
    Two stages: a closest-point query against the triangulation lifted to the ``z = 0`` plane picks
    a candidate face -- sufficient because a point inside any triangle is at distance zero from it,
    so the *closest* triangle contains it whenever one does -- and a barycentric sign test on the
    query's own coordinates decides. That reuses the BVH ``wp.Mesh`` already builds, at the cost of
    one lifted ``wp.vec3`` copy of ``vertices``.

    The second stage is not redundant. Deciding on the query radius alone misclassifies ~0.2% of
    random queries on a 3 979-triangle Delaunay mesh, because an in-plane point's closest-point
    distance is not exactly zero in ``float32``: measured against
    ``scipy.spatial.Delaunay.find_simplex``, a radius of 1e-7 / 1e-6 / 1e-5 of the bounding diagonal
    misses 73 / 28 / 6 interior points, while 1e-5 / 1e-4 / 1e-3 falsely accepts 0 / 14 / 59
    exterior ones -- no radius separates them. The barycentric test is ~1000x sharper, so the radius
    is only a search bound and there is no tolerance to tune.

    Building that BVH is per call, so a caller locating several point sets in one triangulation pays
    for it each time -- there is no prebuilt-index entry point, which is also why the benchmark's
    scipy row (``scipy.spatial.Delaunay.find_simplex`` on a triangulation built outside the timed
    region) is the *unfavourable* comparison for triwarp rather than the flattering one.

    See Also
    --------
    [`closest_point_on_mesh`][triwarp.proximity.closest_point_on_mesh]
    [`points_to_barycentric`][triwarp.triangles.points_to_barycentric]
    [`remap_attribute_from_uv`][triwarp.texture.remap_attribute_from_uv]
    """
    device = vertices.device
    m = int(points.shape[0])
    n_faces = int(faces.shape[0]) // 3
    if m == 0:
        return wp.empty(0, dtype=wp.int32, device=device)
    if n_faces == 0:
        return wp.full(m, -1, dtype=wp.int32, device=device)

    lifted = wp.empty(int(vertices.shape[0]), dtype=wp.vec3, device=device)
    wp.map(kernel_proximity.lift_vec2, vertices, out=lifted)
    # One readback, the same one `closest_point_on_mesh` pays and for the same reason: the search
    # radius has to be in the triangulation's own units and nothing else knows its scale.
    search_radius = _CONTAINMENT_SEARCH_SCALE * tw.bounds.enclosing_diagonal(lifted)

    require_nonempty_mesh(faces, "containing_faces_2d")
    # Both buffers are local to this call (``lifted`` is built above), so no copy is needed.
    mesh = wp.Mesh(points=lifted, indices=faces)
    out_face = wp.empty(m, dtype=wp.int32, device=device)
    wp.launch(
        kernel_proximity.face_containing_point_2d,
        dim=m,
        inputs=[
            mesh.id,
            vertices,
            faces,
            points,
            wp.float32(search_radius),
            _CONTAINMENT_BARYCENTRIC_EPS,
            out_face,
        ],
        device=device,
    )
    return out_face
