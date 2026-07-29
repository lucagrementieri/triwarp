"""Mesh subdivision on NVIDIA Warp."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal, overload

import numpy as np
import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp._device import require_nonempty_mesh
from triwarp.constants import INT32_MAX, TOLERANCE_MOLLIFY
from triwarp.kernels import array as kernel_array
from triwarp.kernels import edges as kernel_edges
from triwarp.kernels import remesh as kernel_remesh
from triwarp.laplacian import mollify_intrinsic

# Callback that launches a predicate kernel filling ``out_flip``/``out_quad`` for one flip
# iteration. Supplied by each consumer of ``_flip_interior_edges`` (3D Delone / 2D incircle).
_LaunchCandidates = Callable[
    [
        twt.Array2dInt32,  # adjacency (m, 2)
        twt.Array2dInt32,  # adjacency_edges (m, 2)
        twt.Array2dInt32,  # unshared (m, 2)
        "wp.array[wp.uint64]",  # sorted edge keys
        "wp.uint64",  # key base (n_vertices)
        "wp.array[wp.bool]",  # out_flip (m,)
        twt.Array2dInt32,  # out_quad (m, 4)
    ],
    None,
]


def isotropic_remesh(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    target_length: float | None = None,
    iterations: int = 10,
    adaptive: bool = False,
    feature_angle: float = 30.0,
    check_surface_distance: bool = True,
    max_surface_distance: float | None = None,
    split: bool = True,
    collapse: bool = True,
    swap: bool = True,
    smooth: bool = True,
    reproject: bool = True,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Isotropic explicit remeshing (Botsch-Kobbelt split / collapse / flip / smooth / reproject).

    GPU port of the classic incremental isotropic remesher (PyMeshLab's
    ``meshing_isotropic_explicit_remeshing``, vcglib ``IsotropicRemeshing``): each iteration drives
    all edge lengths toward ``target_length`` by (1) **splitting** every edge longer than
    ``4/3 * target_length`` at its midpoint (crack-free, reusing
    [`subdivide_to_size`][triwarp.remesh.subdivide_to_size]); (2) **collapsing** every edge shorter
    than ``4/5 * target_length`` (a parallel primitive with full 1-ring locking and a manifold
    link-condition guard); (3) **flipping** interior edges toward the ideal vertex valence (6
    interior, 4 boundary); (4) **tangentially smoothing** free vertices (area-equalizing Laplacian
    projected onto the tangent plane); and (5) **reprojecting** free vertices back onto the original
    surface. Feature and boundary structure is preserved: each vertex is classified FREE / CREASE
    (on a boundary loop or a dihedral crease sharper than ``feature_angle``) / CORNER (feature
    junction or endpoint), corners are frozen, crease vertices only move along their feature, and
    feature edges are never flipped.

    Inputs are cloned and never mutated.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions on the target device.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    target_length
        Desired uniform edge length. Defaults to ``1 %`` of the bounding-box diagonal.
    iterations
        Number of full remeshing passes.
    adaptive
        Curvature-adaptive target length. Not implemented in this version (raises
        ``NotImplementedError`` when ``True``).
    feature_angle
        Dihedral angle in **degrees** above which an interior edge is treated as a sharp feature
        (protected from flipping and collapsing across).
    check_surface_distance
        Reserved for an explicit surface-deviation rejection gate; in this version surface fidelity
        is maintained by the ``reproject`` step (see Notes).
    max_surface_distance
        Companion tolerance for ``check_surface_distance`` (defaults to ``1 %`` of the bounding-box
        diagonal). Reserved (see Notes).
    split, collapse, swap, smooth, reproject
        Enable/disable each stage of the per-iteration pipeline.

    Returns
    -------
    vertices : wp.array[wp.vec3]
        Remeshed vertex positions on ``vertices.device``.
    faces : wp.array[wp.int32]
        Flat ``3 * n_faces`` triangle index buffer.

    Raises
    ------
    ValueError
        If ``target_length`` is non-positive.
    NotImplementedError
        If ``adaptive`` is ``True``.

    See Also
    --------
    [`subdivide_to_size`][triwarp.remesh.subdivide_to_size]
    [`flip_to_delaunay`][triwarp.remesh.flip_to_delaunay]

    Notes
    -----
    Hysteresis (split above ``4/3 t``, collapse below ``4/5 t``) keeps split and collapse from
    fighting. The collapse primitive guarantees manifoldness through the link condition but has no
    normal-flip guard in this version, relying on the reprojection step to keep free vertices on the
    original surface; ``check_surface_distance`` / ``max_surface_distance`` are accepted for API
    parity and reserved for a future explicit rejection gate. Adaptive (curvature-driven) sizing is
    a documented follow-up.
    """
    if adaptive:
        raise NotImplementedError(
            "adaptive isotropic remeshing is not implemented yet; call with adaptive=False."
        )
    n_faces = int(faces.shape[0]) // 3
    current_vertices = wp.clone(vertices)
    current_faces = wp.clone(faces)
    if n_faces == 0 or iterations <= 0:
        return current_vertices, current_faces

    lo, hi = tw.bounds.aabb_bounds(vertices)
    diag = float(wp.length(hi - lo))
    target = target_length if target_length is not None else 0.01 * diag
    if target <= 0.0:
        raise ValueError(f"isotropic_remesh requires target_length > 0, got {target}.")
    low = wp.float32(4.0 / 5.0 * target)
    high = wp.float32(4.0 / 3.0 * target)
    feature = wp.float32(math.radians(feature_angle))
    _ = max_surface_distance if max_surface_distance is not None else 0.01 * diag

    # Original surface, built once, for reprojecting free vertices (never a 0-triangle mesh).
    original_mesh = None
    if reproject:
        require_nonempty_mesh(faces, "isotropic_remesh")
        original_mesh = wp.Mesh(points=wp.clone(vertices), indices=wp.clone(faces))

    for _ in range(iterations):
        if split:
            current_vertices, current_faces = subdivide_to_size(
                current_vertices, current_faces, float(high), max_iter=20
            )
        if collapse:
            current_vertices, current_faces = _collapse_pass(
                current_vertices, current_faces, low, high, feature
            )
        if int(current_faces.shape[0]) == 0:
            break
        if swap:
            _valence_flip_pass(current_vertices, current_faces, feature)
        if smooth or reproject:
            codes, _boundary = _classify(current_vertices, current_faces, feature)
            if smooth:
                current_vertices = _smooth_pass(current_vertices, current_faces, codes)
            if reproject and original_mesh is not None:
                current_vertices = _reproject_pass(
                    current_vertices, codes, original_mesh, max(diag, 1.0)
                )

    return current_vertices, current_faces


def _classify(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    feature: wp.float32,
    edges_sorted: twt.Array2dInt32 | None = None,
) -> tuple[wp.array[wp.int32], wp.array[wp.bool]]:
    """
    Per-vertex FREE / CREASE / CORNER codes plus a boundary-vertex mask.

    A boundary edge or an interior edge sharper than ``feature`` counts as one incident feature
    edge; a vertex with zero is FREE, exactly two is CREASE (a smooth feature/boundary line), and
    anything else (a feature endpoint or a junction) is a frozen CORNER.

    ``edges_sorted`` may be passed when the caller already built it for the same ``faces``: this is
    called up to seven times per ``isotropic_remesh`` iteration, and rebuilding the sorted edge rows
    each time was a measurable share of the total.
    """
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    n_faces = int(faces.shape[0]) // 3
    codes = wp.zeros(n_vertices, dtype=wp.int32, device=device)
    boundary_vertex = wp.zeros(n_vertices, dtype=wp.bool, device=device)
    if n_faces == 0:
        return codes, boundary_vertex

    feature_count = wp.zeros(n_vertices, dtype=wp.int32, device=device)
    boundary_count = wp.zeros(n_vertices, dtype=wp.int32, device=device)

    if edges_sorted is None:
        edges_sorted = tw.edges.faces_to_edges(faces, sorted=True)
    boundary = tw.boundary.boundary_edges(vertices, faces, edges_sorted=edges_sorted)
    if int(boundary.shape[0]) > 0:
        wp.launch(
            kernel_remesh.scatter_edge_endpoint_counts,
            dim=int(boundary.shape[0]),
            inputs=[boundary, feature_count],
            device=device,
        )
        wp.launch(
            kernel_remesh.scatter_edge_endpoint_counts,
            dim=int(boundary.shape[0]),
            inputs=[boundary, boundary_count],
            device=device,
        )

    adjacency, adjacency_edges = tw.adjacency.face_adjacency(
        faces, edges_sorted, return_edges=True, n_vertices=n_vertices
    )
    if int(adjacency.shape[0]) > 0:
        angles = tw.adjacency.face_adjacency_angles(vertices, faces, face_adjacency=adjacency)
        wp.launch(
            kernel_remesh.scatter_feature_endpoint_counts,
            dim=int(adjacency.shape[0]),
            inputs=[adjacency_edges, angles, feature, feature_count],
            device=device,
        )

    wp.map(kernel_remesh.finalize_vertex_codes, feature_count, out=codes)
    wp.map(kernel_array.greater, boundary_count, wp.int32(0), out=boundary_vertex)
    return codes, boundary_vertex


def _collapse_pass(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    low: wp.float32,
    high: wp.float32,
    feature: wp.float32,
    max_passes: int = 5,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """Collapse short edges in parallel with 1-ring locking; returns compacted (vertices, faces)."""
    device = vertices.device
    for _ in range(max_passes):
        n_vertices = int(vertices.shape[0])
        n_faces = int(faces.shape[0]) // 3
        if n_faces == 0:
            break

        # One sorted edge-row build per pass, shared by the unique-edge pass and ``_classify``.
        edges_sorted = tw.edges.faces_to_edges(faces, sorted=True)
        unique_edges, inverse = tw.edges.edges_unique(
            faces, edges_sorted=edges_sorted, n_vertices=n_vertices
        )
        m = int(unique_edges.shape[0])
        if m == 0:
            break
        lengths = tw.edges.edges_unique_length(vertices, faces, unique_edges=unique_edges)

        edge_face_count = wp.zeros(m, dtype=wp.int32, device=device)
        wp.launch(
            kernel_edges.count_edge_faces,
            dim=int(inverse.shape[0]),
            inputs=[inverse, edge_face_count],
            device=device,
        )
        codes, _boundary = _classify(vertices, faces, feature, edges_sorted=edges_sorted)
        csr = tw.graph.edges_to_csr(n_vertices, unique_edges)

        survivor = wp.full(m, -1, dtype=wp.int32, device=device)
        removed = wp.empty(m, dtype=wp.int32, device=device)
        target_pos = wp.empty(m, dtype=wp.vec3, device=device)
        wp.launch(
            kernel_remesh.collapse_candidates,
            dim=m,
            inputs=[
                unique_edges,
                lengths,
                vertices,
                codes,
                edge_face_count,
                csr.offsets,
                csr.columns,
                low,
                high,
                survivor,
                removed,
                target_pos,
            ],
            device=device,
        )

        claim = wp.full(n_vertices, INT32_MAX, dtype=wp.int32, device=device)
        wp.launch(
            kernel_remesh.claim_collapses,
            dim=m,
            inputs=[survivor, removed, csr.offsets, csr.columns, claim],
            device=device,
        )
        remap = tw.array.init_range(n_vertices, device)
        positions = wp.clone(vertices)
        count = wp.zeros(1, dtype=wp.int32, device=device)
        wp.launch(
            kernel_remesh.commit_collapses,
            dim=m,
            inputs=[
                survivor,
                removed,
                target_pos,
                csr.offsets,
                csr.columns,
                claim,
                remap,
                positions,
                count,
            ],
            device=device,
        )
        if int(count.numpy()[0]) == 0:
            break

        remapped = tw.array.gather(remap, faces)
        valid = wp.empty(n_faces, dtype=wp.bool, device=device)
        wp.launch(
            kernel_remesh.mark_distinct_faces, dim=n_faces, inputs=[remapped, valid], device=device
        )
        kept = tw.array.flatnonzero(valid)
        faces = tw.array.gather(remapped.reshape((n_faces, 3)), kept).reshape(-1)
        vertices, faces, _ = tw.repair.remove_unreferenced_vertices(positions, faces)

    return vertices, faces


def _valence_flip_pass(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], feature: wp.float32, max_iter: int = 10
) -> None:
    """Flip interior edges toward ideal valence (6 interior, 4 boundary); mutates ``faces``."""
    device = faces.device
    n_vertices = int(vertices.shape[0])
    _codes, boundary_vertex = _classify(vertices, faces, feature)

    def launch(adjacency, adjacency_edges, unshared, sorted_keys, key_base, out_flip, out_quad):
        # Valence is recomputed from the (in-place mutated) faces each pass to avoid staleness.
        valence = wp.zeros(n_vertices, dtype=wp.int32, device=device)
        unique_edges, _ = tw.edges.edges_unique(faces, n_vertices=n_vertices)
        wp.launch(
            kernel_remesh.accumulate_vertex_valence,
            dim=int(unique_edges.shape[0]),
            inputs=[unique_edges, valence],
            device=device,
        )
        wp.launch(
            kernel_remesh.valence_flip_candidates,
            dim=int(adjacency.shape[0]),
            inputs=[
                vertices,
                faces,
                adjacency,
                adjacency_edges,
                unshared,
                sorted_keys,
                key_base,
                valence,
                boundary_vertex,
                feature,
                out_flip,
                out_quad,
            ],
            device=device,
        )

    _flip_interior_edges(faces, n_vertices, launch, max_iter)


def _smooth_pass(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], codes: wp.array[wp.int32]
) -> wp.array[wp.vec3]:
    """
    One tangential Laplacian smoothing step over free vertices.

    This is the *unweighted* one-ring centroid, which is a documented gap rather than a choice: the
    area-equalizing form Botsch-Kobbelt specify (and which ``isotropic_remesh``'s Notes describe) is
    what actually removes anisotropy, and this one cannot -- on a regular graded grid every vertex
    already sits at the plain average of its neighbours, so the smoother is at a fixed point. See
    ``accumulate_one_ring`` for the measurement and the blocker.
    """
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    normals = tw.vertices.area_weighted_vertex_normals(n_vertices, vertices, faces)
    unique_edges, _ = tw.edges.edges_unique(faces, n_vertices=n_vertices)

    ring_sum = wp.zeros(n_vertices, dtype=wp.vec3, device=device)
    degree = wp.zeros(n_vertices, dtype=wp.int32, device=device)
    wp.launch(
        kernel_remesh.accumulate_one_ring,
        dim=int(unique_edges.shape[0]),
        inputs=[unique_edges, vertices, ring_sum, degree],
        device=device,
    )
    out_positions = wp.empty(n_vertices, dtype=wp.vec3, device=device)
    wp.map(
        kernel_remesh.tangential_smooth_step,
        vertices,
        codes,
        normals,
        ring_sum,
        degree,
        wp.float32(1.0),
        out=out_positions,
    )
    return out_positions


def _reproject_pass(
    vertices: wp.array[wp.vec3], codes: wp.array[wp.int32], original_mesh: wp.Mesh, max_dist: float
) -> wp.array[wp.vec3]:
    """Snap free vertices onto the closest point of the original surface."""
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    out_positions = wp.empty(n_vertices, dtype=wp.vec3, device=device)
    # ``wp.uint64(...)`` is required: a bare ``wp.Mesh.id`` is a Python int and ``wp.map`` would
    # infer ``int32`` for it (see tests/test_map_uniform_probe.py).
    wp.map(
        kernel_remesh.reproject_vertices,
        vertices,
        codes,
        wp.uint64(original_mesh.id),
        wp.float32(max_dist),
        out=out_positions,
    )
    return out_positions


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
        adjacency, adjacency_edges = tw.adjacency.face_adjacency(
            faces, edges_sorted, return_edges=True, n_vertices=n_vertices
        )
        m = int(adjacency.shape[0])
        if m == 0:
            break
        unshared = tw.adjacency.face_adjacency_unshared(faces, adjacency, adjacency_edges)

        # Sorted table of existing undirected-edge keys, for the "flip would duplicate an edge"
        # guard. Keys match kernels.grouping.pack_indices (min + max * n_vertices).
        keys = tw.grouping.hash_indices_rows(edges_sorted, max_index=n_vertices, validate=False)
        sorted_keys, _order = tw.array.sort_pairs(keys)

        out_flip = wp.zeros(m, dtype=wp.bool, device=device)
        out_quad = twt.empty_int32_2d((m, 4), device=device)
        launch_candidates(
            adjacency,
            adjacency_edges,
            unshared,
            sorted_keys,
            wp.uint64(n_vertices),
            out_flip,
            out_quad,
        )

        # Independent-set selection: a flip commits only if it wins both incident faces and the
        # hashed slot of its new edge (prevents two disjoint flips creating the same edge).
        table = 1
        while table < 4 * m + 1:
            table <<= 1
        face_claim = wp.full(n_faces, INT32_MAX, dtype=wp.int32, device=device)
        edge_claim = wp.full(table, INT32_MAX, dtype=wp.int32, device=device)
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
    [`face_adjacency`][triwarp.adjacency.face_adjacency]
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

    def launch(adjacency, adjacency_edges, unshared, sorted_keys, key_base, out_flip, out_quad):
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


def intrinsic_delaunay(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    epsilon: float = TOLERANCE_MOLLIFY,
    max_iter: int = 100,
) -> tuple[wp.array[wp.int32], twt.Array2dFloat32, int]:
    """
    Retriangulate to the intrinsic Delaunay triangulation, without moving a vertex.

    Flips edges whose two opposite angles sum past ``pi`` — exactly the edges whose cotangent weight
    is negative — until none is left. The flip is *intrinsic*: the new edge is not a straight line
    in space but the geodesic across the two triangles, and its length comes from unfolding them
    into a plane and measuring the other diagonal. The surface, its vertices and its metric are all
    untouched; only which pairs of vertices count as connected changes, so every operator built from
    the result is a better-behaved operator for the *same* geometry.

    Its practical effect is that the cotangent weights all become non-negative, which is what a
    Laplacian needs to satisfy a maximum principle: no spurious extrema, no negative diffusion, far
    better conditioned solves on a badly-shaped mesh.

    Flips run in parallel rounds, each committing a conflict-free independent set (the same engine
    behind [`flip_to_delaunay`][triwarp.remesh.flip_to_delaunay]) with the edge-length table carried
    alongside the connectivity.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer. Not modified.
    epsilon
        Mollification margin applied before flipping, relative to the mean edge length: a degenerate
        triangle has no well-defined angles to test.
    max_iter
        Cap on the number of parallel flip rounds.

    Returns
    -------
    intrinsic_faces : wp.array[wp.int32]
        Length-``3 * n_faces`` connectivity of the intrinsic triangulation, over the same vertices.
    edge_lengths : twt.Array2dFloat32
        ``(n_faces, 3)`` intrinsic edge lengths for those faces, column ``e`` opposite corner ``e``.
    n_flips : int
        How many edges were flipped. Zero means the input was already intrinsically Delaunay.

    Notes
    -----
    A flip that would duplicate an existing edge is skipped rather than allowed to create a
    multi-edge, so a few non-Delaunay edges can survive on coarse meshes — geometry-central's
    signpost machinery represents those, this does not. The count is small in practice: on the
    fixtures used in the tests the result matches ``igl.intrinsic_delaunay_cotmatrix`` exactly.

    See Also
    --------
    [`robust_laplacian`][triwarp.laplacian.robust_laplacian]
    [`mollify_intrinsic`][triwarp.laplacian.mollify_intrinsic]
    [`flip_to_delaunay`][triwarp.remesh.flip_to_delaunay]
    """
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    n_faces = int(faces.shape[0]) // 3
    lengths, _ = mollify_intrinsic(vertices, faces, epsilon=epsilon)
    intrinsic_faces = wp.clone(faces)
    if n_faces == 0:
        return intrinsic_faces, lengths, 0

    total = 0
    for _ in range(max_iter):
        edges_sorted = tw.edges.faces_to_edges(intrinsic_faces, sorted=True)
        adjacency, adjacency_edges = tw.adjacency.face_adjacency(
            intrinsic_faces, edges_sorted, return_edges=True, n_vertices=n_vertices
        )
        n_interior = int(adjacency.shape[0])
        if n_interior == 0:
            break
        unshared = tw.adjacency.face_adjacency_unshared(intrinsic_faces, adjacency, adjacency_edges)
        keys = tw.grouping.hash_indices_rows(edges_sorted, max_index=n_vertices, validate=False)
        sorted_keys, _order = tw.array.sort_pairs(keys)

        flip = wp.zeros(n_interior, dtype=wp.bool, device=device)
        quad = twt.empty_int32_2d((n_interior, 4), device=device)
        new_length = wp.empty(n_interior, dtype=wp.float32, device=device)
        wp.launch(
            kernel_remesh.intrinsic_delaunay_candidates,
            dim=n_interior,
            inputs=[
                intrinsic_faces,
                lengths,
                adjacency,
                adjacency_edges,
                unshared,
                sorted_keys,
                wp.uint64(n_vertices),
                flip,
                quad,
                new_length,
            ],
            device=device,
        )

        # Independent set: a flip commits only if it wins both incident faces and its new edge's
        # hashed slot, so no two committed flips share a face or invent the same edge.
        table = 1
        while table < 4 * n_interior + 1:
            table <<= 1
        face_claim = wp.full(n_faces, INT32_MAX, dtype=wp.int32, device=device)
        edge_claim = wp.full(table, INT32_MAX, dtype=wp.int32, device=device)
        wp.launch(
            kernel_remesh.claim_flips,
            dim=n_interior,
            inputs=[
                flip,
                quad,
                adjacency,
                wp.int32(table - 1),
                wp.uint64(n_vertices),
                face_claim,
                edge_claim,
            ],
            device=device,
        )
        # Lengths first: this pass needs the *old* connectivity to know which corner holds which
        # vertex, and ``commit_flips`` is about to overwrite it.
        wp.launch(
            kernel_remesh.update_flipped_lengths,
            dim=n_interior,
            inputs=[
                intrinsic_faces,
                flip,
                quad,
                adjacency,
                new_length,
                face_claim,
                edge_claim,
                wp.int32(table - 1),
                wp.uint64(n_vertices),
                lengths,
            ],
            device=device,
        )
        count = wp.zeros(1, dtype=wp.int32, device=device)
        wp.launch(
            kernel_remesh.commit_flips,
            dim=n_interior,
            inputs=[
                flip,
                quad,
                adjacency,
                face_claim,
                edge_claim,
                wp.int32(table - 1),
                wp.uint64(n_vertices),
                intrinsic_faces,
                count,
            ],
            device=device,
        )
        committed = int(count.numpy()[0])
        total += committed
        if committed == 0:
            break
    return intrinsic_faces, lengths, total


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

    # Build (n_faces, 3) array of midpoint vertex indices: unique-edge index + vertex offset.
    mid_idx_flat = wp.empty(3 * n_faces, dtype=wp.int32, device=device)
    wp.map(wp.add, inverse, wp.int32(n_vertices), out=mid_idx_flat)
    mid_idx = mid_idx_flat.reshape((n_faces, 3))

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
        offsets, n_long = tw.array.counts_to_offsets(flags)

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

        face_mid = tw.array.gather(midpoint_idx, inverse).reshape((n_faces, 3))

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

    def launch(adjacency, adjacency_edges, unshared, sorted_keys, key_base, out_flip, out_quad):
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
    [`fill_holes_nicely`][triwarp.hole_filling.fill_holes_nicely]
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
        offsets, n_long = tw.array.counts_to_offsets(flags)

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
                offsets, n_long = tw.array.counts_to_offsets(flags)

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

        face_mid = tw.array.gather(midpoint_idx, inverse).reshape((n_faces, 3))
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
