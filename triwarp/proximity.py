"""
Queries against a ``wp.Mesh`` surface: closest point, signed distance, winding number, thickness.

[`thickness`][triwarp.proximity.thickness] and
[`shape_diameter`][triwarp.proximity.shape_diameter] both measure how thick the volume is under a
surface point, at two levels of robustness: the first fires one inward ray (or fits one tangent
sphere), the second fires a whole cone and takes an outlier-trimmed mean. For the *outward*
counterpart -- how open a point is rather than how thick -- see
[`triwarp.shading`][triwarp.shading].

Mesh point queries ([`closest_point_on_mesh`][triwarp.proximity.closest_point_on_mesh],
[`signed_distance_on_mesh`][triwarp.proximity.signed_distance_on_mesh])
and [`contains_points`][triwarp.ray.contains_points] follow Warp's SDF sign convention: outside
positive, inside negative. Trimesh ``signed_distance`` uses the opposite sign.

Point-set acceleration structures (``wp.Bvh`` / ``wp.HashGrid``) and raw neighbor queries live in
[`triwarp.neighbors`][triwarp.neighbors]; axis-aligned bounding boxes in
[`triwarp.bounds`][triwarp.bounds].
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp._device import require_nonempty_mesh
from triwarp.kernels import proximity as kernel_proximity
from triwarp.kernels import shading as kernel_shading
from triwarp.triangles import face_normals_and_areas

# Elements per thread for the two *per-query* lane-free reductions here (solid-angle sum, packed
# support argmax). Unlike a global reduction, these have one accumulator per query, so the query
# dimension already supplies the parallelism and a short slice only multiplies the atomic traffic:
# measured on 20k faces x 5000 queries, 128 runs 0.49 ms against 3.7 ms at 8 and 13.8 ms at 4. The
# optimum is flat over 128-256 and degrades again past 512, so this is not a sensitive knob.
ITEMS_PER_QUERY_SLICE = 128

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
    points
        ``(m,)`` query positions in space as ``wp.vec3``.
    max_dist
        Maximum search radius per query. Faces farther than this are ignored.
        When ``None``, derived from the axis-aligned box enclosing mesh
        vertices and query points.

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
        nan_closest_np = np.full((m, 3), np.nan, dtype=np.float32)
        out_closest = wp.array(nan_closest_np, dtype=wp.vec3, device=device)
        out_distance = wp.full(m, float("inf"), dtype=wp.float32, device=device)
        out_face = wp.full(m, -1, dtype=wp.int32, device=device)
        return out_closest, out_distance, out_face

    require_nonempty_mesh(faces, "closest_point_on_mesh")
    mesh = wp.Mesh(points=wp.clone(vertices), indices=wp.clone(faces))
    if max_dist is None:
        max_dist = _default_mesh_query_max_dist(mesh.points, points)

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


def normals_at_closest_faces(
    mesh: wp.Mesh, points: wp.array[wp.vec3], *, max_dist: float | None = None
) -> wp.array[wp.vec3]:
    """
    Return unit face normals at the closest mesh triangle for each query point.

    For each position in ``points``, runs an unsigned closest-point query on
    ``mesh`` and returns the normal of the hit triangle. When no face lies
    within ``max_dist``, the corresponding output is undefined (same as the
    underlying ``wp.mesh_query_point_no_sign`` miss case).

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
        ``(m,)`` face normals at the closest triangle for each query.
    """
    device = points.device
    m = int(points.shape[0])
    if m == 0:
        return wp.empty(0, dtype=wp.vec3, device=device)

    if max_dist is None:
        max_dist = _default_mesh_query_max_dist(mesh.points, points)

    out_closest = wp.empty(m, dtype=wp.vec3, device=device)
    out_dist = wp.empty(m, dtype=wp.float32, device=device)
    out_face = wp.empty(m, dtype=wp.int32, device=device)
    wp.launch(
        kernel_proximity.closest_point_on_mesh,
        dim=m,
        inputs=[mesh.id, points, wp.float32(max_dist), out_closest, out_dist, out_face],
        device=device,
    )
    all_face_normals, _ = face_normals_and_areas(mesh.points, mesh.indices)
    normals = wp.empty(m, dtype=wp.vec3, device=device)
    wp.copy(normals, all_face_normals[out_face])
    return normals


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

    Returns
    -------
    wp.array[wp.float32]
        ``(m,)`` signed distances in ``float32``.

    Raises
    ------
    ValueError
        If ``sign_mode`` is not ``"parity"`` or ``"winding"``.

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

    require_nonempty_mesh(faces, "signed_distance_on_mesh")
    # The winding-number builtin silently degrades to ray parity unless the mesh carries the
    # per-node solid-angle expansion, so the flag is bound to sign_mode here rather than exposed.
    mesh = wp.Mesh(
        points=wp.clone(vertices),
        indices=wp.clone(faces),
        support_winding_number=sign_mode == "winding",
    )
    if max_dist is None:
        max_dist = _default_mesh_query_max_dist(mesh.points, points)
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


def winding_number(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    query_points: wp.array[wp.vec3],
    *,
    tiled: bool = True,
) -> wp.array[wp.float32]:
    """
    Generalized winding number at each query point (``igl::winding_number``).

    Sums the signed solid angle subtended by each oriented triangle. For a
    closed, consistently oriented watertight mesh, interior points have
    winding number near ``1`` and exterior points near ``0``.

    Parameters
    ----------
    vertices
        ``(n,)`` mesh vertex positions as ``wp.vec3``.
    faces
        ``(f * 3,)`` flat triangle index array as ``wp.int32``.
    query_points
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
    device = query_points.device
    n_queries = int(query_points.shape[0])
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
                query_points,
                out_winding,
            ],
            device=device,
        )
    else:
        wp.launch(
            kernel_proximity.winding_number,
            dim=n_queries,
            inputs=[vertices, faces, wp.int32(n_faces), query_points, out_winding],
            device=device,
        )
    return out_winding


def max_tangent_sphere(
    mesh: wp.Mesh,
    points: wp.array[wp.vec3],
    *,
    inwards: bool = True,
    normals: wp.array[wp.vec3] | None = None,
    threshold: float = 1e-6,
    max_iter: int = 100,
) -> tuple[wp.array[wp.vec3], wp.array[wp.float32]]:
    """
    Find the center and radius of the sphere tangent to the mesh at each point.

    Implements the shrinking-sphere algorithm (Inui et al. 2016): iteratively
    finds the largest sphere tangent to the mesh at ``points`` with no
    non-tangential intersections.

    Parameters
    ----------
    mesh
        Warp mesh (BVH built by caller).
    points
        ``(m,)`` surface points as ``wp.vec3``.
    inwards
        If ``True``, sphere grows inward (into the mesh interior). If ``False``,
        grows outward.
    normals
        ``(m,)`` unit surface normals at ``points``. If ``None``, computed from
        the closest triangle.
    threshold
        Convergence threshold as a fraction of the scene diagonal.
    max_iter
        Maximum number of shrink iterations.

    Returns
    -------
    centers
        ``(m,)`` sphere center positions as ``wp.vec3``.
    radii
        ``(m,)`` sphere radii as ``float32``. ``inf`` when the sphere is
        unbounded.
    """
    device = points.device
    m = int(points.shape[0])
    if m == 0:
        return (
            wp.empty(0, dtype=wp.vec3, device=device),
            wp.empty(0, dtype=wp.float32, device=device),
        )

    if normals is None:
        normals = normals_at_closest_faces(mesh, points)

    ray_dirs: wp.array[wp.vec3] = -normals if inwards else normals

    max_t = _default_mesh_query_max_dist(mesh.points, points)
    distances = tw.ray.longest_ray(mesh, points, ray_dirs, max_t=max_t)

    n_verts = int(mesh.points.shape[0])
    radii = wp.empty(m, dtype=wp.float32, device=device)
    not_converged = wp.empty(m, dtype=wp.bool, device=device)
    needs_support = wp.empty(m, dtype=wp.bool, device=device)
    wp.map(
        kernel_proximity.init_sphere_radii_finite,
        distances,
        out=[radii, not_converged, needs_support],
    )
    # Escaped rays (typically exterior/reach queries) need the support point of the vertex
    # cloud in the ray direction. Compact them first — interior queries usually leave the
    # subset empty — then run one grid-stride packed-argmax pass over the vertices for just
    # that subset instead of a serial all-vertices loop per query thread.
    support_indices = tw.array.flatnonzero(needs_support)
    k = int(support_indices.shape[0])
    if k > 0:
        n_vert_slices = max(1, (n_verts + ITEMS_PER_QUERY_SLICE - 1) // ITEMS_PER_QUERY_SLICE)
        packed_support = wp.zeros(k, dtype=wp.uint64, device=device)
        wp.launch(
            kernel_proximity.support_argmax_tiled,
            dim=(k, n_vert_slices),
            inputs=[
                mesh.points,
                wp.int32(n_verts),
                wp.int32(n_vert_slices),
                ray_dirs,
                support_indices,
                packed_support,
            ],
            device=device,
        )
        wp.launch(
            kernel_proximity.init_sphere_radii_support,
            dim=k,
            inputs=[
                mesh.points,
                points,
                ray_dirs,
                support_indices,
                packed_support,
                radii,
                not_converged,
            ],
            device=device,
        )

    centers = wp.empty(m, dtype=wp.vec3, device=device)
    wp.map(kernel_proximity.sphere_center, points, ray_dirs, radii, out=centers)

    mesh_min, mesh_max = tw.bounds.aabb_bounds(mesh.points)
    D = float(wp.length(mesh_max - mesh_min))  # noqa: N806
    convergence_threshold = wp.float32(threshold * D)

    # All per-iteration buffers are preallocated once and ping-ponged (the step kernel writes
    # every lane, passing converged state through). The convergence count is checked every
    # iteration on purpose: an extra iteration runs a full BVH closest-point pass, far more
    # expensive than the 8-byte readback the check costs.
    n_pts_wp = wp.empty(m, dtype=wp.vec3, device=device)
    n_dists_wp = wp.empty(m, dtype=wp.float32, device=device)
    n_face_wp = wp.empty(m, dtype=wp.int32, device=device)
    new_radii = wp.empty(m, dtype=wp.float32, device=device)
    new_centers = wp.empty(m, dtype=wp.vec3, device=device)
    new_nc = wp.empty(m, dtype=wp.bool, device=device)

    for _ in range(max_iter):
        if tw.reduce.sum(not_converged) == 0:
            break

        wp.launch(
            kernel_proximity.closest_point_on_mesh,
            dim=m,
            inputs=[mesh.id, centers, wp.float32(max_t), n_pts_wp, n_dists_wp, n_face_wp],
            device=device,
        )
        wp.launch(
            kernel_proximity.step_sphere_shrink,
            dim=m,
            inputs=[
                points,
                ray_dirs,
                n_pts_wp,
                n_dists_wp,
                centers,
                radii,
                convergence_threshold,
                not_converged,
                new_radii,
                new_centers,
                new_nc,
            ],
            device=device,
        )
        radii, new_radii = new_radii, radii
        centers, new_centers = new_centers, centers
        not_converged, new_nc = new_nc, not_converged

    return centers, radii


def query_mesh_aabb_bounds_with_offsets(
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
        kernel_proximity.query_mesh_aabb_bounds_count,
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
        kernel_proximity.query_mesh_aabb_bounds_neighbors,
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


def thickness(
    mesh: wp.Mesh,
    points: wp.array[wp.vec3],
    *,
    exterior: bool = False,
    normals: wp.array[wp.vec3] | None = None,
    method: Literal["max_sphere", "ray"] = "max_sphere",
) -> wp.array[wp.float32]:
    """
    Thickness of the mesh at each point.

    Parameters
    ----------
    mesh
        Warp mesh (BVH built by caller).
    points
        ``(m,)`` surface points as ``wp.vec3``.
    exterior
        If ``True``, compute exterior thickness (reach). If ``False``, interior.
    normals
        ``(m,)`` unit surface normals. If ``None``, computed automatically.
    method
        ``"max_sphere"`` (default) or ``"ray"``.

    Returns
    -------
    wp.array[wp.float32]
        ``(m,)`` thickness values. ``inf`` for unbounded.
    """
    if method == "max_sphere":
        _centers, radii = max_tangent_sphere(mesh, points, inwards=not exterior, normals=normals)
        return radii * wp.float32(2.0)

    elif method == "ray":
        if normals is None:
            normals = normals_at_closest_faces(mesh, points)

        ray_dirs = normals if exterior else -normals
        max_t = _default_mesh_query_max_dist(mesh.points, points)
        return tw.ray.longest_ray(mesh, points, ray_dirs, max_t=max_t)

    else:
        raise ValueError('Invalid method, use "max_sphere" or "ray"')


def shape_diameter(
    mesh: wp.Mesh,
    points: wp.array[wp.vec3],
    *,
    normals: wp.array[wp.vec3] | None = None,
    n_rays: int = 64,
    cone_angle: float = math.pi / 3.0,
    trim: float = 1.0,
    max_t: float | None = None,
) -> wp.array[wp.float32]:
    """
    Shape diameter function: local thickness of the volume, from an inward cone of rays.

    Shapira et al.'s SDF and MeshLab's ``compute_scalar_by_shape_diameter_function_per_vertex``. A
    cone of ``n_rays`` rays is fired *into* the volume about ``-normal``, each is traced to the far
    side, and the result is the cosine-weighted mean of those distances **after discarding the
    outliers** — the rays that escaped through a nearby opening or crossed the entire model, which
    would otherwise dominate the average near any concavity.

    This is the many-ray generalization of
    [`thickness(method="ray")`][triwarp.proximity.thickness]: at ``n_rays=1`` with a vanishing
    ``cone_angle`` the bundle collapses to the inward normal and the two agree to float32. The extra
    rays are what make it stable — a single ray through a thin sliver of geometry reads a thickness
    the neighbourhood does not have — and the reason it is the quantity used for skeleton extraction
    and part segmentation rather than the one-ray version.

    Parameters
    ----------
    mesh
        Triangle mesh with a built BVH (``wp.Mesh``). Should be closed: on an open surface the rays
        that find nothing on the far side are simply absent from the mean.
    points
        ``(m,)`` surface positions to measure at, normally the mesh's own vertices.
    normals
        ``(m,)`` **outward** unit normals; the cone opens along ``-normals``. When ``None`` they are
        taken from the closest face of ``mesh``. Pass
        [`area_weighted_vertex_normals`][triwarp.vertices.area_weighted_vertex_normals] for a smooth
        field over a mesh's own vertices.
    n_rays
        Rays per point. MeshLab's default is ``64``, which is this one. Note that the single ray of
        ``n_rays=1`` is the *centroid* of the cone's Fibonacci lattice rather than its axis, so it
        only coincides with the inward normal as ``cone_angle`` goes to zero.
    cone_angle
        Half-angle of the cone in **radians**, measured from the inward normal. The default
        ``pi / 3`` (60 degrees) is Shapira's 120-degree cone. Must be in ``(0, pi / 2]`` — beyond
        that the cone reaches around to the outside of the surface and the distances stop meaning
        thickness.
    trim
        Keep only rays whose distance is within ``trim`` standard deviations of the mean before
        averaging. ``1.0`` (the default) is Shapira's rule; a large value keeps everything and turns
        this into a plain weighted mean. Must be non-negative.
    max_t
        Maximum ray length. When ``None``, the diagonal of the AABB enclosing the mesh and the query
        points.

    Returns
    -------
    wp.array[wp.float32]
        ``(m,)`` diameters in the mesh's own length units on ``points.device``. ``inf`` at a point
        where no ray in the cone found the far side at all.

    Raises
    ------
    ValueError
        If ``n_rays < 1``, ``cone_angle`` is outside ``(0, pi / 2]``, ``trim < 0``, or ``normals``
        has a different length from ``points``.

    See Also
    --------
    [`thickness`][triwarp.proximity.thickness]
    [`max_tangent_sphere`][triwarp.proximity.max_tangent_sphere]
    [`triwarp.shading.ambient_occlusion`][triwarp.shading.ambient_occlusion]
    [`triwarp.sample.sample_fibonacci_cone`][triwarp.sample.sample_fibonacci_cone]

    Notes
    -----
    MeshLab's ``cone_amplitude`` parameter is a **no-op** in the 2025.07 build — its output is
    byte-identical at ``90`` and ``120`` degrees — and its trimming rule is not the one documented
    in the paper, so its values differ from these by a roughly constant factor on a given mesh.
    Compare against it by rank rather than by value; the exactly-checkable statements are the
    reduction to [`thickness`][triwarp.proximity.thickness] at ``n_rays=1`` and the analytic ``2 R``
    on a sphere.
    """
    if n_rays < 1:
        raise ValueError(f"shape_diameter requires n_rays >= 1, got {n_rays}")
    if not 0.0 < cone_angle <= math.pi / 2.0:
        raise ValueError(f"cone_angle must be in (0, pi / 2] radians, got {cone_angle}")
    if trim < 0.0:
        raise ValueError(f"trim must be non-negative, got {trim}")

    device = points.device
    m = int(points.shape[0])
    if m == 0:
        return wp.empty(0, dtype=wp.float32, device=device)

    if normals is None:
        normals = normals_at_closest_faces(mesh, points)
    elif int(normals.shape[0]) != m:
        raise ValueError(
            f"normals must have one entry per point, got {normals.shape[0]} for {m} points"
        )

    diagonal = _default_mesh_query_max_dist(mesh.points, points)
    directions = tw.sample.sample_fibonacci_cone(n_rays, cone_angle, device=device)
    # Distances are kept so the trimming pass can revisit them against a mean the first pass had not
    # finished computing; re-tracing instead would double the only expensive part of the kernel.
    scratch = twt.empty_float32_2d((m, n_rays), device=device)
    out_diameter = wp.empty(m, dtype=wp.float32, device=device)
    wp.launch(
        kernel_shading.shape_diameter,
        dim=m,
        inputs=[
            mesh.id,
            points,
            normals,
            directions,
            wp.float32(max_t if max_t is not None else diagonal),
            wp.float32(_SDF_SURFACE_OFFSET * max(diagonal, 1e-12)),
            wp.float32(trim),
            scratch,
            out_diameter,
        ],
        device=device,
    )
    return out_diameter


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
    search_radius = _CONTAINMENT_SEARCH_SCALE * _default_mesh_query_max_dist(lifted)

    require_nonempty_mesh(faces, "containing_faces_2d")
    mesh = wp.Mesh(points=lifted, indices=wp.clone(faces))
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


def _default_mesh_query_max_dist(
    mesh_points: wp.array[wp.vec3], query_points: wp.array[wp.vec3] | None = None
) -> float:
    """
    Diagonal of the AABB enclosing ``mesh_points`` and ``query_points``.

    When ``query_points`` is ``None`` (or empty), returns the diagonal of the
    axis-aligned bounding box of ``mesh_points`` alone.
    """
    mesh_min, mesh_max = tw.bounds.aabb_bounds(mesh_points)
    if query_points is None or int(query_points.shape[0]) == 0:
        return tw.bounds.aabb_diagonal(mesh_min, mesh_max)
    query_min, query_max = tw.bounds.aabb_bounds(query_points)
    combined_min, combined_max = tw.bounds.aabb_union(mesh_min, mesh_max, query_min, query_max)
    return tw.bounds.aabb_diagonal(combined_min, combined_max)
