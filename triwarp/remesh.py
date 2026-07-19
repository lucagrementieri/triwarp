"""Mesh subdivision on NVIDIA Warp."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal, overload

import numpy as np
import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.kernels import array as kernel_array
from triwarp.kernels import remesh as kernel_remesh

_INT32_MAX = 2147483647

# Callback that launches a predicate kernel filling ``out_flip``/``out_quad`` for one flip
# iteration. Supplied by each consumer of ``_flip_interior_edges`` (3D Delone / 2D incircle).
_LaunchCandidates = Callable[
    [
        twt.Array2dInt32,  # adjacency (m, 2)
        twt.Array2dInt32,  # adjacency_edges (m, 2)
        twt.Array2dInt32,  # unshared (m, 2)
        "wp.array[wp.uint64]",  # sorted edge keys
        int,  # number of keys
        "wp.uint64",  # key base (n_vertices)
        "wp.array[wp.bool]",  # out_flip (m,)
        twt.Array2dInt32,  # out_quad (m, 4)
    ],
    None,
]


def _flip_interior_edges(
    faces: wp.array[wp.int32], n_vertices: int, launch_candidates: _LaunchCandidates, max_iter: int
) -> int:
    """
    Repeatedly flip an independent set of interior edges until none is a candidate.

    ``faces`` is mutated in place. Each iteration rebuilds face adjacency, lets
    ``launch_candidates`` mark flippable edges (predicate-specific), then commits a
    conflict-free subset (no two committed flips touch a shared face or create the same new
    edge). Returns the total number of flips performed. The winding rewrite matches MeshLib
    ``flipEdge`` and ``igl::flip_edge``.
    """
    device = faces.device
    n_faces = int(faces.shape[0]) // 3
    total = 0
    for _ in range(max_iter):
        edges_sorted = tw.edges.faces_to_edges(faces, sorted=True)
        adjacency, adjacency_edges = tw.graph.face_adjacency(faces, edges_sorted, return_edges=True)
        m = int(adjacency.shape[0])
        if m == 0:
            break
        unshared = tw.graph.face_adjacency_unshared(faces, adjacency, adjacency_edges)

        # Sorted table of existing undirected-edge keys, for the "flip would duplicate an edge"
        # guard. Keys match kernels.unique.pack_indices (min + max * n_vertices).
        n_rows = int(edges_sorted.shape[0])
        keys = tw.unique.hash_indices_rows(edges_sorted, max_index=n_vertices)
        keys_buffer = wp.empty(2 * n_rows, dtype=wp.uint64, device=device)
        wp.copy(keys_buffer, keys, count=n_rows)
        vals_buffer = tw.array.init_sort_pair_indices(n_rows, -1, device)
        wp.utils.radix_sort_pairs(keys_buffer, vals_buffer, count=n_rows)
        sorted_keys = keys_buffer[:n_rows]

        out_flip = wp.zeros(m, dtype=wp.bool, device=device)
        out_quad = twt.empty_int32_2d((m, 4), device=device)
        launch_candidates(
            adjacency,
            adjacency_edges,
            unshared,
            sorted_keys,
            n_rows,
            wp.uint64(n_vertices),
            out_flip,
            out_quad,
        )

        # Independent-set selection: a flip commits only if it wins both incident faces and the
        # hashed slot of its new edge (prevents two disjoint flips creating the same edge).
        table = 1
        while table < 4 * m + 1:
            table <<= 1
        face_claim = wp.full(n_faces, _INT32_MAX, dtype=wp.int32, device=device)
        edge_claim = wp.full(table, _INT32_MAX, dtype=wp.int32, device=device)
        wp.launch(
            kernel_remesh.claim_flips,
            dim=m,
            inputs=[
                out_flip,
                out_quad,
                adjacency,
                wp.int32(table - 1),
                wp.uint64(n_vertices),
                face_claim,
                edge_claim,
            ],
            device=device,
        )
        count = wp.zeros(1, dtype=wp.int32, device=device)
        wp.launch(
            kernel_remesh.commit_flips,
            dim=m,
            inputs=[
                out_flip,
                out_quad,
                adjacency,
                face_claim,
                edge_claim,
                wp.int32(table - 1),
                wp.uint64(n_vertices),
                faces,
                count,
            ],
            device=device,
        )
        n = int(count.numpy()[0])
        total += n
        if n == 0:
            break
    return total


def flip_to_delaunay(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    region: wp.array[wp.bool] | None = None,
    max_angle_change: float | None = None,
    max_deviation: float | None = None,
    critical_aspect_ratio: float = 1000.0,
    max_iter: int = 100,
) -> wp.array[wp.int32]:
    """
    Improve triangle quality by flipping interior edges toward the Delaunay criterion.

    Ports MeshLib ``makeDeloneEdgeFlips``: for every interior edge whose two incident faces are
    both in ``region``, the shared diagonal is flipped when doing so satisfies the local Delone
    (empty-circumcircle) test — subject to an optional dihedral-angle-change gate and a
    surface-deviation gate, so the flips never distort the surface. Rim edges (with a face
    outside the region, or on the mesh boundary) are never flipped. Vertices, face count and
    region membership are unchanged; only the triangulation of the region is rewritten.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    region
        Optional length-``n_faces`` ``wp.bool`` mask; only edges interior to the ``True`` faces
        are flippable. ``None`` treats the whole mesh as flippable.
    max_angle_change
        Maximum dihedral-angle change (radians) a flip may introduce
        (``maxAngleChangeAfterFlip``). ``None`` disables the gate.
    max_deviation
        Maximum surface deviation a flip may introduce (``maxDeviationAfterFlip``). ``None``
        disables the gate.
    critical_aspect_ratio
        Triangle aspect ratio above which the dihedral-angle gate is lifted (so degenerate
        triangles can still be repaired), matching ``criticalAspectRatioFlip``.
    max_iter
        Maximum number of parallel flip passes.

    Returns
    -------
    wp.array[wp.int32]
        Flat face buffer with the region re-triangulated, on ``faces.device`` (a copy; the
        input is not modified).

    See Also
    --------
    [`subdivide_region_to_size`][triwarp.remesh.subdivide_region_to_size]
    [`delaunay_triangulation`][triwarp.reconstruction.delaunay_triangulation]
    [`face_adjacency`][triwarp.graph.face_adjacency]
    """
    device = faces.device
    n_faces = int(faces.shape[0]) // 3
    out_faces = wp.clone(faces)
    if n_faces == 0:
        return out_faces
    if region is not None and int(region.shape[0]) != n_faces:
        raise ValueError(f"region must have length n_faces={n_faces}, got {int(region.shape[0])}")

    n_vertices = tw.vertices.n_vertices(faces)
    region_flags = wp.empty(n_faces, dtype=wp.int32, device=device)
    if region is None:
        region_flags.fill_(1)
    else:
        wp.utils.array_cast(region, region_flags)

    mac = wp.float32(max_angle_change if max_angle_change is not None else float(2.0 * math.pi))
    mdsq = wp.float32(max_deviation * max_deviation if max_deviation is not None else 3.0e38)
    car = wp.float32(critical_aspect_ratio)

    def launch(
        adjacency, adjacency_edges, unshared, sorted_keys, n_keys, key_base, out_flip, out_quad
    ):
        wp.launch(
            kernel_remesh.delone_flip_candidates,
            dim=int(adjacency.shape[0]),
            inputs=[
                vertices,
                out_faces,
                adjacency,
                adjacency_edges,
                unshared,
                region_flags,
                sorted_keys,
                wp.int32(n_keys),
                key_base,
                mac,
                mdsq,
                car,
                out_flip,
                out_quad,
            ],
            device=device,
        )

    _flip_interior_edges(out_faces, n_vertices, launch, max_iter)
    return out_faces


def subdivide(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Subdivide a mesh by splitting every face into four triangles.

    Each triangle is split by placing a new vertex at the midpoint of each
    edge. The four child triangles share these midpoints and preserve the
    original winding order, matching [`trimesh.remesh.subdivide`][] exactly.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        Length-``3 * n_faces`` flat triangle index buffer.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(new_vertices, new_faces)`` on ``vertices.device``.

    See Also
    --------
    [`trimesh.remesh.subdivide`][]
    """
    device = vertices.device
    n_vertices = int(vertices.shape[0])

    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return vertices, faces

    unique_edges, inverse = tw.edges.edges_unique(faces, n_vertices=n_vertices)
    n_unique = int(unique_edges.shape[0])

    # Compute midpoint vertex for each unique edge
    out_midpoints = wp.empty(n_unique, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_remesh.compute_midpoints,
        dim=n_unique,
        inputs=[vertices, unique_edges, out_midpoints],
        device=device,
    )

    # Build (n_faces, 3) array of midpoint vertex indices
    mid_idx = twt.empty_int32_2d((n_faces, 3), device=device)
    wp.launch(
        kernel_remesh.build_mid_idx,
        dim=n_faces,
        inputs=[inverse, wp.int32(n_vertices), mid_idx],
        device=device,
    )

    # Emit 4 new triangles per face, shape (n_faces*12,)
    out_new_faces = wp.empty(n_faces * 12, dtype=wp.int32, device=device)
    wp.launch(
        kernel_remesh.subdivide_faces,
        dim=n_faces,
        inputs=[faces, mid_idx, out_new_faces],
        device=device,
    )

    new_vertices, _ = tw.array.pack_1d_arrays([vertices, out_midpoints])
    return new_vertices, out_new_faces


@overload
def subdivide_to_size(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    max_edge: float,
    max_iter: int = 10,
    return_index: Literal[False] = False,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]: ...
@overload
def subdivide_to_size(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    max_edge: float,
    max_iter: int = 10,
    *,
    return_index: Literal[True],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]: ...
def subdivide_to_size(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    max_edge: float,
    max_iter: int = 10,
    return_index: bool = False,
) -> (
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
    | tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]
):
    """
    Subdivide a mesh until every edge is at most ``max_edge`` long.

    Every edge longer than ``max_edge`` is bisected at a single shared midpoint,
    so the two faces on either side stay in sync and a watertight input stays
    watertight — no T-junctions (cracks) are introduced. Faces already small
    enough are left untouched. Each pass splits every over-long edge once and
    re-triangulates the incident faces with per-face templates (1, 2, or 3 split
    edges; the 2-split quad is cut along its shorter diagonal), iterating until
    no edge exceeds the threshold, matching
    [`trimesh.remesh.subdivide_to_size`][].

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        Length-``3 * n_faces`` flat triangle index buffer.
    max_edge
        Maximum length of any edge in the result.
    max_iter
        Maximum number of subdivision passes. A ``ValueError`` is raised if the
        mesh still has an over-long edge after this many passes.
    return_index
        If ``True``, also return the source face index of each output face.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        Refined vertex positions on ``vertices.device`` (original vertices first,
        then the inserted edge midpoints).
    new_faces : wp.array[wp.int32]
        Flat buffer of the refined faces.
    index : wp.array[wp.int32]
        Only returned when ``return_index`` is ``True``: length ``n_out_faces``,
        the index of the original face each output face was refined from.

    Raises
    ------
    ValueError
        If any edge is still longer than ``max_edge`` after ``max_iter`` passes.

    See Also
    --------
    [`subdivide`][triwarp.remesh.subdivide]
    [`trimesh.remesh.subdivide_to_size`][]
    """
    device = vertices.device
    max_edge_f = wp.float32(max_edge)

    current_vertices = vertices
    current_faces = faces
    n_faces = int(faces.shape[0]) // 3
    index = tw.array.init_range(n_faces, device)

    if n_faces == 0:
        if return_index:
            return current_vertices, current_faces, index
        return current_vertices, current_faces

    for i in range(max_iter + 1):
        n_faces = int(current_faces.shape[0]) // 3
        n_vertices = int(current_vertices.shape[0])

        unique_edges, inverse = tw.edges.edges_unique(current_faces, n_vertices=n_vertices)
        m = int(unique_edges.shape[0])
        lengths = tw.edges.edges_unique_length(
            current_vertices, current_faces, unique_edges=unique_edges
        )

        # Flag the edges that are longer than the target length.
        long_mask = wp.empty(m, dtype=wp.bool, device=device)
        wp.map(kernel_array.greater, lengths, max_edge_f, out=long_mask)

        # Exclusive scan of the flags gives each long edge its new-vertex slot;
        # the inclusive total is the number of midpoints to add this pass.
        flags = wp.empty(m, dtype=wp.int32, device=device)
        wp.utils.array_cast(long_mask, flags)
        offsets = wp.empty(m, dtype=wp.int32, device=device)
        inclusive = wp.empty(m, dtype=wp.int32, device=device)
        wp.utils.array_scan(flags, out_array=offsets, inclusive=False)
        wp.utils.array_scan(flags, out_array=inclusive, inclusive=True)
        n_long = int(inclusive.numpy()[-1])

        # Every edge is short enough: we are done.
        if n_long == 0:
            break
        # Ran out of passes with over-long edges still present.
        if i >= max_iter:
            raise ValueError("max_iter exceeded!")

        midpoint_idx = wp.empty(m, dtype=wp.int32, device=device)
        wp.launch(
            kernel_remesh.build_midpoint_index,
            dim=m,
            inputs=[long_mask, offsets, wp.int32(n_vertices), midpoint_idx],
            device=device,
        )

        new_mid = wp.empty(n_long, dtype=wp.vec3, device=device)
        wp.launch(
            kernel_remesh.fill_edge_midpoints,
            dim=m,
            inputs=[current_vertices, unique_edges, long_mask, offsets, new_mid],
            device=device,
        )
        # Append midpoints so the new indices resolve during face emission.
        current_vertices, _ = tw.array.pack_1d_arrays([current_vertices, new_mid])

        face_mid = twt.empty_int32_2d((n_faces, 3), device=device)
        wp.launch(
            kernel_remesh.build_face_mid,
            dim=n_faces,
            inputs=[inverse, midpoint_idx, face_mid],
            device=device,
        )

        # Emit up to four triangles per face into fixed slots, then compact.
        out_faces = twt.empty_int32_2d((n_faces * 4, 3), device=device)
        out_valid = wp.empty(n_faces * 4, dtype=wp.bool, device=device)
        out_slot_index = wp.empty(n_faces * 4, dtype=wp.int32, device=device)
        wp.launch(
            kernel_remesh.emit_size_faces,
            dim=n_faces,
            inputs=[
                current_faces,
                face_mid,
                current_vertices,
                index,
                out_faces,
                out_valid,
                out_slot_index,
            ],
            device=device,
        )

        kept = tw.array.flatnonzero(out_valid)
        current_faces = tw.array.gather(out_faces, kept).reshape(-1)
        index = tw.array.gather(out_slot_index, kept)

    if return_index:
        return current_vertices, current_faces, index
    return current_vertices, current_faces


def _flip_region_faces(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    region_flags: wp.array[wp.int32],
    max_angle_change: float | None,
    max_deviation: float | None,
    max_iter: int,
) -> int:
    """Run the parallel Delone flip pass over the region, mutating ``faces`` in place."""
    device = faces.device
    n_vertices = int(vertices.shape[0])
    mac = wp.float32(max_angle_change if max_angle_change is not None else float(2.0 * math.pi))
    mdsq = wp.float32(max_deviation * max_deviation if max_deviation is not None else 3.0e38)
    car = wp.float32(1000.0)

    def launch(
        adjacency, adjacency_edges, unshared, sorted_keys, n_keys, key_base, out_flip, out_quad
    ):
        wp.launch(
            kernel_remesh.delone_flip_candidates,
            dim=int(adjacency.shape[0]),
            inputs=[
                vertices,
                faces,
                adjacency,
                adjacency_edges,
                unshared,
                region_flags,
                sorted_keys,
                wp.int32(n_keys),
                key_base,
                mac,
                mdsq,
                car,
                out_flip,
                out_quad,
            ],
            device=device,
        )

    return _flip_interior_edges(faces, n_vertices, launch, max_iter)


def subdivide_region_to_size(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    region: wp.array[wp.bool],
    max_edge: float,
    max_iter: int = 10,
    max_splits: int | None = None,
    delaunay: bool = True,
    max_angle_change: float | None = math.pi / 6.0,
    max_deviation: float | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.bool]]:
    """
    Subdivide only a face region until its edges are at most ``max_edge`` long.

    Ports MeshLib ``subdivideMesh`` restricted to a face region (as used by ``fillHoleNicely``'s
    ``subdivideFillingNicely``): every edge with at least one incident region face and length
    greater than ``max_edge`` is bisected, the incident faces are re-triangulated crack-free
    (the [`subdivide_to_size`][triwarp.remesh.subdivide_to_size] 1/2/3-split templates, so faces
    outside the region that touch a split edge stay watertight), and — unless disabled — a
    parallel Delaunay edge-flip pass ([`flip_to_delaunay`][triwarp.remesh.flip_to_delaunay])
    improves the region triangulation after each pass. New vertices are appended after the
    originals, so the caller derives the new-vertex set as the index range
    ``[len(vertices), len(new_vertices))``.

    Unlike MeshLib's sequential longest-edge-first priority queue, splitting is done in parallel
    passes; ``max_splits`` is honoured as a soft budget by keeping only the longest eligible
    edges of the pass that would exceed it.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    region
        Length-``n_faces`` ``wp.bool`` mask; only edges touching a ``True`` face are refined.
    max_edge
        Target maximum edge length inside the region.
    max_iter
        Maximum number of subdivision passes.
    max_splits
        Optional soft cap on the total number of edge splits (``maxEdgeSplits``). ``None`` keeps
        splitting until convergence and raises if ``max_iter`` is exhausted first.
    delaunay
        When ``True`` (default), interleave and finish with the Delaunay flip pass.
    max_angle_change
        Dihedral-angle-change gate (radians) for the flip pass
        (``maxAngleChangeAfterFlip``; default 30°). ``None`` disables the gate.
    max_deviation
        Surface-deviation gate for the flip pass. ``None`` disables it.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        Original vertices followed by the inserted midpoints, on ``vertices.device``.
    new_faces : wp.array[wp.int32]
        Flat buffer of the refined faces.
    new_region : wp.array[wp.bool]
        Length ``n_out_faces`` region mask; child faces inherit their parent's membership.

    Raises
    ------
    ValueError
        If ``region`` length does not match the face count, or if over-long region edges remain
        after ``max_iter`` passes and ``max_splits`` is ``None``.

    See Also
    --------
    [`subdivide_to_size`][triwarp.remesh.subdivide_to_size]
    [`flip_to_delaunay`][triwarp.remesh.flip_to_delaunay]
    [`fill_holes_nicely`][triwarp.stitching.fill_holes_nicely]
    """
    device = vertices.device
    n_faces = int(faces.shape[0]) // 3
    if int(region.shape[0]) != n_faces:
        raise ValueError(f"region must have length n_faces={n_faces}, got {int(region.shape[0])}")
    if n_faces == 0:
        return vertices, faces, region

    max_edge_f = wp.float32(max_edge)
    current_vertices = vertices
    current_faces = faces
    region_flags = wp.empty(n_faces, dtype=wp.int32, device=device)
    wp.utils.array_cast(region, region_flags)
    splits_done = 0

    for i in range(max_iter + 1):
        n_faces = int(current_faces.shape[0]) // 3
        n_vertices = int(current_vertices.shape[0])

        unique_edges, inverse = tw.edges.edges_unique(current_faces, n_vertices=n_vertices)
        m = int(unique_edges.shape[0])
        lengths = tw.edges.edges_unique_length(
            current_vertices, current_faces, unique_edges=unique_edges
        )

        edge_in_region = wp.zeros(m, dtype=wp.bool, device=device)
        wp.launch(
            kernel_remesh.mark_region_edges,
            dim=3 * n_faces,
            inputs=[region_flags, inverse, edge_in_region],
            device=device,
        )
        long_mask = wp.empty(m, dtype=wp.bool, device=device)
        wp.map(kernel_remesh.long_region_edge, lengths, max_edge_f, edge_in_region, out=long_mask)

        flags = wp.empty(m, dtype=wp.int32, device=device)
        wp.utils.array_cast(long_mask, flags)
        inclusive = wp.empty(m, dtype=wp.int32, device=device)
        wp.utils.array_scan(flags, out_array=inclusive, inclusive=True)
        n_long = int(inclusive.numpy()[-1]) if m > 0 else 0

        if n_long == 0:
            break
        if i >= max_iter:
            if max_splits is None:
                raise ValueError("max_iter exceeded!")
            break

        if max_splits is not None:
            remaining = max_splits - splits_done
            if remaining <= 0:
                break
            if n_long > remaining:
                long_mask = _keep_longest_edges(long_mask, lengths, remaining, m, device)
                wp.utils.array_cast(long_mask, flags)
                wp.utils.array_scan(flags, out_array=inclusive, inclusive=True)
                n_long = int(inclusive.numpy()[-1])

        offsets = wp.empty(m, dtype=wp.int32, device=device)
        wp.utils.array_scan(flags, out_array=offsets, inclusive=False)

        midpoint_idx = wp.empty(m, dtype=wp.int32, device=device)
        wp.launch(
            kernel_remesh.build_midpoint_index,
            dim=m,
            inputs=[long_mask, offsets, wp.int32(n_vertices), midpoint_idx],
            device=device,
        )
        new_mid = wp.empty(n_long, dtype=wp.vec3, device=device)
        wp.launch(
            kernel_remesh.fill_edge_midpoints,
            dim=m,
            inputs=[current_vertices, unique_edges, long_mask, offsets, new_mid],
            device=device,
        )
        current_vertices, _ = tw.array.pack_1d_arrays([current_vertices, new_mid])

        face_mid = twt.empty_int32_2d((n_faces, 3), device=device)
        wp.launch(
            kernel_remesh.build_face_mid,
            dim=n_faces,
            inputs=[inverse, midpoint_idx, face_mid],
            device=device,
        )
        out_faces = twt.empty_int32_2d((n_faces * 4, 3), device=device)
        out_valid = wp.empty(n_faces * 4, dtype=wp.bool, device=device)
        out_slot_index = wp.empty(n_faces * 4, dtype=wp.int32, device=device)
        wp.launch(
            kernel_remesh.emit_size_faces,
            dim=n_faces,
            inputs=[
                current_faces,
                face_mid,
                current_vertices,
                region_flags,
                out_faces,
                out_valid,
                out_slot_index,
            ],
            device=device,
        )
        kept = tw.array.flatnonzero(out_valid)
        current_faces = tw.array.gather(out_faces, kept).reshape(-1)
        region_flags = tw.array.gather(out_slot_index, kept)
        splits_done += n_long

        if delaunay:
            _flip_region_faces(
                current_vertices, current_faces, region_flags, max_angle_change, max_deviation, 8
            )

    if delaunay:
        _flip_region_faces(
            current_vertices, current_faces, region_flags, max_angle_change, max_deviation, 50
        )

    new_region = wp.empty(int(current_faces.shape[0]) // 3, dtype=wp.bool, device=device)
    wp.utils.array_cast(region_flags, new_region)
    return current_vertices, current_faces, new_region


def _keep_longest_edges(
    long_mask: wp.array[wp.bool],
    lengths: wp.array[wp.float32],
    remaining: int,
    m: int,
    device: wp.DeviceLike,
) -> wp.array[wp.bool]:
    """Keep only the ``remaining`` longest edges currently flagged in ``long_mask`` (host-side)."""
    mask_np = long_mask.numpy()
    lengths_np = lengths.numpy()
    eligible = np.flatnonzero(mask_np)
    keep = eligible[np.argsort(-lengths_np[eligible])[:remaining]]
    truncated = np.zeros(m, dtype=bool)
    truncated[keep] = True
    return wp.array(truncated, dtype=wp.bool, device=device)
