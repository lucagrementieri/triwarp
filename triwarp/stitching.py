"""
Basic boundary hole filling (Warp).

Every boundary of a triangle mesh is an ordered vertex loop
([`boundary_loops`][triwarp.boundary.boundary_loops], the analog of
``trimesh.repair.fill_holes``'s ``nx.cycle_basis`` and ``igl::boundary_loop_all``). Each loop
of ``B`` vertices is sealed with a purely topological triangulation — **no smoothing or
refinement**:

- [`fill_holes_fan`][triwarp.stitching.fill_holes_fan] fans ``B - 2`` triangles from the loop's
  first vertex, reusing only existing vertices (``trimesh.repair.fill_holes(use_fan=True)``).
- [`fill_holes_cone`][triwarp.stitching.fill_holes_cone] inserts one centroid vertex per hole
  and cones ``B`` triangles onto it (``igl::topological_hole_fill``,
  ``trimesh.repair.stitch(insert_vertices=True)``).

Because ``boundary_loops`` orders each loop following the face-winding direction of the
boundary half-edges, a fill triangle sharing a rim edge is emitted **reversed** on that edge so
its winding is consistent with the adjacent original face. This assumes the input mesh is
consistently wound; run [`make_winding_consistent`][triwarp.repair.make_winding_consistent]
first on meshes with mixed winding.

The complementary operation joins **two** open meshes across one boundary loop each:
[`stitch`][triwarp.stitching.stitch] (and the lower-level
[`triangulate_boundaries`][triwarp.stitching.triangulate_boundaries]) zippers the two rims with a
band of bridge triangles, producing a single watertight seam. No smoothing or refinement is
applied.
"""

from __future__ import annotations

import numpy as np
import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.kernels import stitching as kernel_stitching


def _hole_loops(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    preserve_largest_hole: bool = False,
) -> tuple[wp.array[wp.int32], wp.array[wp.int32], int, int] | None:
    """
    Pack fillable boundary loops (>= 3 vertices) for on-device triangulation.

    Returns ``(flat_loops, loop_starts, n_loops, total)`` where ``flat_loops`` concatenates the
    loop vertex indices, ``loop_starts`` is each loop's start offset in ``flat_loops`` (the
    exclusive scan of the loop sizes, on device), ``n_loops`` is the loop count, and ``total`` is
    ``flat_loops.shape[0]``. Returns ``None`` when there is no fillable boundary loop. Both counts
    are read from array shapes (host metadata), so no device buffer is copied to the host.

    When ``preserve_largest_hole`` is ``True`` the single largest loop — the one with the greatest
    perimeter arc length ([`closed_polyline_length`][triwarp.polyline.closed_polyline_length]; the
    first one on a tie) — is excluded from the returned pack, leaving it open. This is the standard
    cut for disk-topology repair and UV parametrization, where exactly one boundary must survive.
    """
    loops = [loop for loop in tw.boundary.boundary_loops(vertices, faces) if int(loop.shape[0]) >= 3]
    if preserve_largest_hole and len(loops) > 0:
        perimeters = [
            tw.polyline.closed_polyline_length(tw.array.gather(vertices, loop)) for loop in loops
        ]
        largest = max(range(len(loops)), key=lambda i: perimeters[i])
        del loops[largest]
    if len(loops) == 0:
        return None
    flat_loops, loop_starts = tw.array.pack_1d_arrays(loops)
    return flat_loops, loop_starts, len(loops), int(flat_loops.shape[0])


def fill_holes_fan(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    preserve_largest_hole: bool = False,
) -> wp.array[wp.int32]:
    """
    Fill every boundary hole with a triangle fan from each loop's first vertex.

    A boundary loop of ``B`` vertices is sealed with ``B - 2`` triangles all sharing the loop's
    first vertex, reusing only existing vertices (``trimesh.repair.fill_holes(use_fan=True)``).
    The fill is topological: a fan is only geometrically ideal for convex holes. The vertex
    buffer is unchanged.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    preserve_largest_hole
        When ``True``, leave the single largest boundary loop (most vertices) open and fill only
        the rest. This turns a mesh with a known disk-like topology into a single-boundary disk,
        which is the input a robust UV parametrization expects.

    Returns
    -------
    wp.array[wp.int32]
        Flat face buffer of the original faces followed by the new fill triangles, on
        ``faces.device``. A watertight or empty mesh — or, with ``preserve_largest_hole``, a mesh
        whose only hole is the largest — is returned unchanged (a copy).

    See Also
    --------
    [`fill_holes_cone`][triwarp.stitching.fill_holes_cone]
    [`boundary_loops`][triwarp.boundary.boundary_loops]
    [`make_winding_consistent`][triwarp.repair.make_winding_consistent]

    Notes
    -----
    Fill winding is consistent with the adjacent faces only when the input mesh is consistently
    wound (see [`make_winding_consistent`][triwarp.repair.make_winding_consistent]).
    """
    device = faces.device
    n_faces = int(faces.shape[0]) // 3
    packed = _hole_loops(vertices, faces, preserve_largest_hole) if n_faces > 0 else None
    if packed is None:
        return wp.clone(faces)

    flat_loops, loop_starts, n_loops, total = packed
    n_tri = total - 2 * n_loops
    fill_faces = wp.empty(3 * n_tri, dtype=wp.int32, device=device)
    wp.launch(
        kernel_stitching.fan_faces,
        dim=n_loops,
        inputs=[flat_loops, loop_starts, wp.int32(total), wp.int32(n_loops), fill_faces],
        device=device,
    )
    return tw.array.concatenate([faces, fill_faces])


def fill_holes_cone(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    preserve_largest_hole: bool = False,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Fill every boundary hole by coning it onto a new centroid vertex.

    A boundary loop of ``B`` vertices is sealed with ``B`` triangles fanning from one new vertex
    placed at the loop's centroid (``igl::topological_hole_fill``,
    ``trimesh.repair.stitch(insert_vertices=True)``). Unlike
    [`fill_holes_fan`][triwarp.stitching.fill_holes_fan] this appends one vertex per hole, so the
    vertex buffer grows.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    preserve_largest_hole
        When ``True``, leave the single largest boundary loop (most vertices) open and fill only
        the rest. This turns a mesh with a known disk-like topology into a single-boundary disk,
        which is the input a robust UV parametrization expects.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        Original vertices followed by one centroid per filled hole, on ``vertices.device``.
    new_faces : wp.array[wp.int32]
        Original faces followed by the new cone triangles, on ``faces.device``. A watertight or
        empty mesh — or, with ``preserve_largest_hole``, a mesh whose only hole is the largest —
        is returned unchanged (copies).

    See Also
    --------
    [`fill_holes_fan`][triwarp.stitching.fill_holes_fan]
    [`boundary_loops`][triwarp.boundary.boundary_loops]
    [`make_winding_consistent`][triwarp.repair.make_winding_consistent]

    Notes
    -----
    Fill winding is consistent with the adjacent faces only when the input mesh is consistently
    wound (see [`make_winding_consistent`][triwarp.repair.make_winding_consistent]).
    """
    device = faces.device
    n_faces = int(faces.shape[0]) // 3
    packed = _hole_loops(vertices, faces, preserve_largest_hole) if n_faces > 0 else None
    if packed is None:
        return wp.clone(vertices), wp.clone(faces)

    flat_loops, loop_starts, n_loops, total = packed
    n_vertices = int(vertices.shape[0])

    centroids = wp.empty(n_loops, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_stitching.loop_centroids,
        dim=n_loops,
        inputs=[vertices, flat_loops, loop_starts, wp.int32(total), wp.int32(n_loops), centroids],
        device=device,
    )

    fill_faces = wp.empty(3 * total, dtype=wp.int32, device=device)
    wp.launch(
        kernel_stitching.cone_faces,
        dim=n_loops,
        inputs=[
            flat_loops,
            loop_starts,
            wp.int32(total),
            wp.int32(n_loops),
            wp.int32(n_vertices),
            fill_faces,
        ],
        device=device,
    )
    return (
        tw.array.concatenate([vertices, centroids]),
        tw.array.concatenate([faces, fill_faces]),
    )


def _longest_increasing_subsequence(numbers: np.ndarray) -> np.ndarray:
    """
    Longest strictly increasing subsequence of ``numbers`` (patience-sorting, O(N log N)).

    Repeated values must be pre-perturbed to distinct values (see
    [`_non_increasing_indices`][triwarp.stitching._non_increasing_indices]); the algorithm does
    not handle ties. Mirrors promesh's private helper of the same name.
    """
    p = np.zeros_like(numbers, dtype=np.int64)
    m = -np.ones(numbers.size + 1, dtype=np.int64)
    size = 0
    for i, x in enumerate(numbers):
        subseq_size = int(np.searchsorted(numbers[m[1 : size + 1]], x))
        p[i] = m[subseq_size]
        m[subseq_size + 1] = i
        size = max(subseq_size + 1, size)
    subseq = np.empty(size, numbers.dtype)
    k = m[size]
    for i in range(size - 1, -1, -1):
        subseq[i] = numbers[k]
        k = p[k]
    return subseq


def _non_increasing_indices(numbers: np.ndarray) -> np.ndarray:
    """
    Return the entry indices **not** in the longest non-decreasing subsequence of ``numbers``.

    Repeated integers are perturbed by adding ``linspace(0, 1, count, endpoint=False)`` within
    each equal-value group, so a run of equal values is treated as (weakly) increasing and kept.
    Mirrors promesh's private helper (with a NumPy grouping in place of ``trimesh.grouping.group``
    to avoid a runtime ``trimesh`` dependency).
    """
    different = numbers.astype(np.float64)
    order = np.argsort(numbers, kind="stable")
    sorted_values = numbers[order]
    cut = np.flatnonzero(np.diff(sorted_values)) + 1
    for group in np.split(order, cut):
        different[group] += np.linspace(0.0, 1.0, num=len(group), endpoint=False)
    subsequence = _longest_increasing_subsequence(different)
    absence_mask = np.isin(different, subsequence, assume_unique=True, invert=True)
    return np.flatnonzero(absence_mask)


def triangulate_boundaries(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    loop_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
    loop_b: wp.array[wp.int32],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Join two meshes by triangulating the band between one boundary loop on each.

    The two rims are zippered into a watertight seam by a greedy, order-preserving correspondence
    (the Warp port of promesh's ``triangulate_boundaries``): every A-edge is matched to the B
    vertex minimizing the added-triangle perimeter, the association is rotated so its shortest
    pair comes first, and a longest-increasing-subsequence correction forces the matching to be
    monotone (non-self-intersecting). **No smoothing or refinement** is applied — the seam reuses
    only the two loops' existing vertices, adding ``len(loop_a) + len(loop_b)`` bridge triangles.

    The larger loop is treated as A (the meshes are swapped internally when
    ``len(loop_a) < len(loop_b)``), so the output is invariant to argument order. Both loops must
    wind following the face orientation, as returned by
    [`boundary_loops`][triwarp.boundary.boundary_loops]; the two boundaries are assumed to wind in
    opposite directions (as two open meshes facing each other do), so loop A is reversed to align
    them.

    Parameters
    ----------
    vertices_a, vertices_b
        ``(n_vertices,)`` vertex positions of each mesh.
    faces_a, faces_b
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffers of each mesh.
    loop_a, loop_b
        Ordered vertex-index loops (``>= 3`` vertices each) around the boundary to join on each
        mesh, indexing ``vertices_a`` / ``vertices_b`` respectively.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        Concatenation of ``vertices_a`` then ``vertices_b`` (larger loop first), on
        ``faces_a.device``.
    new_faces : wp.array[wp.int32]
        Original faces (B reindexed by ``len(vertices_a)``) followed by the bridge triangles, on
        ``faces_a.device``.

    Raises
    ------
    ValueError
        If either loop has fewer than 3 vertices.

    See Also
    --------
    [`stitch`][triwarp.stitching.stitch]
    [`boundary_loops`][triwarp.boundary.boundary_loops]
    [`concatenate`][triwarp.graph.concatenate]

    Notes
    -----
    The correspondence and its monotonicity correction are inherently sequential and run on the
    host over the length-``len(loop_a)`` association array; the O(N·M) perimeter matrix, the
    reductions, and the triangle emission run in Warp kernels, and the perimeter matrix itself is
    never copied off the device.
    """
    device = faces_a.device
    n = int(loop_a.shape[0])
    m = int(loop_b.shape[0])
    if n < m:
        vertices_a, vertices_b = vertices_b, vertices_a
        faces_a, faces_b = faces_b, faces_a
        loop_a, loop_b = loop_b, loop_a
        n, m = m, n
    if m < 3:
        raise ValueError(f"each boundary loop must have at least 3 vertices, got {n} and {m}")
    n_vertices_a = int(vertices_a.shape[0])

    # Reverse loop A so both rims wind the same way, then take rim positions.
    flipped_loop_a = wp.empty(n, dtype=wp.int32, device=device)
    wp.launch(
        kernel_stitching.flip_loop,
        dim=n,
        inputs=[loop_a, wp.int32(n), flipped_loop_a],
        device=device,
    )
    a_pos = tw.array.gather(vertices_a, flipped_loop_a)
    b_pos = tw.array.gather(vertices_b, loop_b)

    # perimeters[i, j] = |a_i - b_j| + |a_{i+1} - b_j| for A-edge i and B-vertex j.
    perimeters = twt.empty_float32_2d((n, m), device=device)
    wp.launch(
        kernel_stitching.boundary_perimeters,
        dim=(n, m),
        inputs=[a_pos, b_pos, wp.int32(n), perimeters],
        device=device,
    )

    col_min = wp.empty(n, dtype=wp.int32, device=device)
    val_min = wp.empty(n, dtype=wp.float32, device=device)
    wp.launch(
        kernel_stitching.row_argmin,
        dim=n,
        inputs=[perimeters, wp.int32(m), col_min, val_min],
        device=device,
    )

    shift = wp.empty(2, dtype=wp.int32, device=device)
    wp.launch(
        kernel_stitching.global_argmin,
        dim=1,
        inputs=[col_min, val_min, wp.int32(n), shift],
        device=device,
    )
    shift_np = shift.numpy()
    shift_a = int(shift_np[0])
    shift_b = int(shift_np[1])

    edge_dev = wp.empty(n, dtype=wp.int32, device=device)
    wp.launch(
        kernel_stitching.rolled_edge_map,
        dim=n,
        inputs=[col_min, wp.int32(shift_a), wp.int32(shift_b), wp.int32(n), wp.int32(m), edge_dev],
        device=device,
    )
    edge = edge_dev.numpy()

    row_roll = shift_a
    col_roll = shift_b

    # Wraparound-group fix: if the lowest-index group is split across the ends of ``edge``, roll
    # A so the group is contiguous (promesh's correction before the monotonicity pass).
    if edge[-1] == edge[0]:
        trailing = int(np.argmin(np.flip(edge) == edge[0]))
        if trailing > 0:
            edge = np.roll(edge, trailing)
            row_roll = (shift_a - trailing) % n

    # Monotonicity correction: re-pick the B vertex of every edge outside the longest
    # non-decreasing subsequence within the bracket of its stable neighbours, so the matching is
    # order-preserving. The sentinel value ``m`` closes the last bracket.
    edge_ext = np.append(edge, np.int32(m))
    out_edge = wp.array(edge_ext.astype(np.int32), dtype=wp.int32, device=device)
    if not np.all(np.diff(edge) >= 0):
        unsorted_indices = _non_increasing_indices(edge_ext)
        stable_indices = np.delete(np.arange(edge_ext.size), unsorted_indices)
        next_indices = stable_indices[np.searchsorted(stable_indices, unsorted_indices)]
        wp.launch(
            kernel_stitching.resolve_corrections,
            dim=1,
            inputs=[
                perimeters,
                wp.array(unsorted_indices.astype(np.int32), dtype=wp.int32, device=device),
                wp.array(next_indices.astype(np.int32), dtype=wp.int32, device=device),
                wp.int32(int(unsorted_indices.size)),
                wp.int32(row_roll),
                wp.int32(col_roll),
                wp.int32(n),
                wp.int32(m),
                out_edge,
            ],
            device=device,
        )

    # Rolled loops referencing the concatenated vertex buffer (A first, then B offset).
    roll_a = wp.empty(n, dtype=wp.int32, device=device)
    wp.launch(
        kernel_stitching.rolled_loop_a,
        dim=n,
        inputs=[flipped_loop_a, wp.int32(row_roll), wp.int32(n), roll_a],
        device=device,
    )
    roll_b = wp.empty(m, dtype=wp.int32, device=device)
    wp.launch(
        kernel_stitching.rolled_loop_b,
        dim=m,
        inputs=[loop_b, wp.int32(col_roll), wp.int32(n_vertices_a), wp.int32(m), roll_b],
        device=device,
    )

    bridge_a = wp.empty(3 * n, dtype=wp.int32, device=device)
    wp.launch(
        kernel_stitching.bridge_a_faces,
        dim=n,
        inputs=[roll_a, roll_b, out_edge, wp.int32(n), bridge_a],
        device=device,
    )
    bridge_b = wp.empty(3 * m, dtype=wp.int32, device=device)
    wp.launch(
        kernel_stitching.bridge_b_faces,
        dim=m,
        inputs=[roll_a, roll_b, out_edge, wp.int32(n), wp.int32(m), bridge_b],
        device=device,
    )

    combined_vertices, combined_faces = tw.graph.concatenate(
        [(vertices_a, faces_a), (vertices_b, faces_b)]
    )
    return combined_vertices, tw.array.concatenate([combined_faces, bridge_a, bridge_b])


def stitch(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array[wp.int32],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array[wp.int32],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Stitch two single-boundary open meshes into one watertight mesh.

    Extracts the single boundary loop of each mesh
    ([`boundary_loops`][triwarp.boundary.boundary_loops]) and joins them with
    [`triangulate_boundaries`][triwarp.stitching.triangulate_boundaries]. Each mesh must have
    exactly one boundary loop (the promesh assumption); use ``triangulate_boundaries`` directly to
    join specific loops of multi-boundary meshes. **No smoothing** is applied.

    Parameters
    ----------
    vertices_a, vertices_b
        ``(n_vertices,)`` vertex positions of each mesh.
    faces_a, faces_b
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffers of each mesh.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        Concatenated vertices (larger-boundary mesh first), on ``faces_a.device``.
    new_faces : wp.array[wp.int32]
        Concatenated, reindexed faces followed by the bridge triangles, on ``faces_a.device``.

    Raises
    ------
    ValueError
        If either mesh does not have exactly one boundary loop of at least 3 vertices.

    See Also
    --------
    [`triangulate_boundaries`][triwarp.stitching.triangulate_boundaries]
    [`fill_holes_fan`][triwarp.stitching.fill_holes_fan]
    [`boundary_loops`][triwarp.boundary.boundary_loops]
    """
    loops_a = [
        loop for loop in tw.boundary.boundary_loops(vertices_a, faces_a) if int(loop.shape[0]) >= 3
    ]
    loops_b = [
        loop for loop in tw.boundary.boundary_loops(vertices_b, faces_b) if int(loop.shape[0]) >= 3
    ]
    if len(loops_a) != 1 or len(loops_b) != 1:
        raise ValueError(
            "stitch requires each mesh to have exactly one boundary loop (>= 3 vertices); "
            f"got {len(loops_a)} and {len(loops_b)}"
        )
    return triangulate_boundaries(vertices_a, faces_a, loops_a[0], vertices_b, faces_b, loops_b[0])
