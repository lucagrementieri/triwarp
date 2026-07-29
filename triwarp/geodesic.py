"""Geodesic distance and geodesic-ball neighborhoods on triangle meshes."""

from __future__ import annotations

import warnings

import warp as wp
import warp.optim.linear as wpl
import warp.sparse as wps

import triwarp as tw
from triwarp.edges import mean_edge_length
from triwarp.intrinsic import mollify_intrinsic
from triwarp.kernels import geodesic as kernel_geodesic
from triwarp.kernels.algorithms import bfs as kernel_bfs
from triwarp.laplacian import (
    cotmatrix,
    cotmatrix_entries,
    cotmatrix_entries_intrinsic,
    mass_matrix_entries,
)
from triwarp.triangles import face_normals_and_areas

_CG_TOLERANCE = 1e-8


HeatOperators = tuple[
    wps.BsrMatrix[wp.float64],
    wps.BsrMatrix[wp.float64],
    wp.array[wp.float32],
    wp.array[wp.vec3],
    wp.array[wp.float32],
]
"""What [`heat_operators`][triwarp.geodesic.heat_operators] returns for the heat method's solves."""


def heat_operators(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    t: float | None = None,
    *,
    use_robust: bool = False,
) -> HeatOperators:
    """
    Assemble everything [`heat_geodesic`][triwarp.geodesic.heat_geodesic] needs before its solves.

    Every quantity here depends on the mesh alone, not on the source set, so a caller computing
    distance from many different sources on one mesh can build these once and pass them back through
    ``heat_geodesic(..., operators=...)``. That is the split
    ``potpourri3d.MeshHeatMethodDistanceSolver`` and ``igl::heat_geodesics`` expose as a stateful
    solver object; here it stays a plain tuple of buffers.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    t
        Diffusion time. When ``None``, defaults to the squared mean edge length (the
        ``igl::heat_geodesics`` default).
    use_robust
        Build the Laplacian from *mollified* edge lengths
        ([`mollify_intrinsic`][triwarp.intrinsic.mollify_intrinsic]) instead of straight from vertex
        positions. Costs one extra pass and two host readbacks, and is what lets the method run on a
        mesh with degenerate triangles at all. It leaves a clean mesh's operator unchanged.

        This is mollification **only**, not the intrinsic Delaunay retriangulation that
        [`robust_laplacian`][triwarp.intrinsic.robust_laplacian] also does by default (and that
        ``potpourri3d``'s identically-named flag includes). The reason is structural rather than a
        shortcut: flipping changes which faces exist, and the gradient and divergence stages below
        integrate over faces. Swapping in an operator built on a different triangulation while those
        stages still use the original one is not a cheap approximation, it is inconsistent — so a
        fully intrinsic heat method needs intrinsic *mass*, *gradient* and *divergence* as well. Use
        [`robust_laplacian`][triwarp.intrinsic.robust_laplacian] directly where only the operator
        matters (smoothing, parametrization, spectral work).

    Returns
    -------
    heat_system : warp.sparse.BsrMatrix
        ``M - t * L`` in ``float64``, the heat-diffusion system.
    laplacian : warp.sparse.BsrMatrix
        The ``float64`` cotangent stiffness matrix ``L`` (igl sign convention, so ``-L`` is positive
        semi-definite), reused for the Poisson stage.
    cot_entries : wp.array[wp.float32]
        Per-face half-cotangent weights, reused by the divergence.
    face_normals : wp.array[wp.vec3]
        One unit normal per face.
    face_areas : wp.array[wp.float32]
        One area per face.

    See Also
    --------
    [`heat_geodesic`][triwarp.geodesic.heat_geodesic]
    [`cotmatrix`][triwarp.laplacian.cotmatrix]
    [`mass_matrix_entries`][triwarp.laplacian.mass_matrix_entries]
    """
    if t is None:
        h = mean_edge_length(vertices, faces)
        t = h * h

    # Per-face half-cotangent weights (float32, O(1) and safe) reused for both the Laplacian and
    # the divergence. The cotangent stiffness follows the igl convention (negative diagonal, so
    # ``-L`` is positive semi-definite) but is assembled here in float64.
    if use_robust:
        # Mollified lengths: one global constant added to every edge so no triangle is degenerate.
        # The gradient and divergence stages below still use the extrinsic positions, so this makes
        # the *solves* robust rather than turning the whole method intrinsic.
        lengths, _ = mollify_intrinsic(vertices, faces)
        cot_entries = cotmatrix_entries_intrinsic(lengths)
    else:
        cot_entries = cotmatrix_entries(vertices, faces)
    # ``cotmatrix`` casts the shared float32 half-cotangent weights to float64 and assembles the
    # operator natively in a single build (see issue_report.md).
    laplacian = cotmatrix(vertices, faces, cot_entries=cot_entries, dtype=wp.float64)

    # Face normals / areas (float32) for the gradient; the lumped mass is built natively in float64
    # by ``mass_matrix_entries``.
    normals, areas = face_normals_and_areas(vertices, faces)
    mass = mass_matrix_entries(vertices, faces, dtype=wp.float64)

    # Heat system (M - t L). ``bsr_axpy`` overwrites the mass matrix in place (no longer needed).
    mass_diag = wps.bsr_diag(diag=mass)
    heat_system = wps.bsr_axpy(x=laplacian, y=mass_diag, alpha=-float(t), beta=1.0)
    return heat_system, laplacian, cot_entries, normals, areas


def heat_geodesic(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    sources: wp.array[wp.int32],
    t: float | None = None,
    operators: HeatOperators | None = None,
    *,
    use_robust: bool = False,
) -> wp.array[wp.float64]:
    """
    Approximate geodesic distance to the nearest source vertex (Crane et al. heat method).

    Diffuses heat from the source vertices for a short time ``t``, normalizes the resulting
    gradient into a unit vector field pointing away from the sources, and integrates it back into a
    distance field by solving a Poisson problem. Both solves are sparse, symmetric positive
    (semi-)definite systems handled on-device by conjugate gradient. The result is an
    *approximation* of the true geodesic distance (typically a few percent error), matching
    ``igl::heat_geodesics``.

    The computation runs in ``float64``: the diffused heat decays exponentially away from the
    source and would underflow ``float32``, collapsing the far field.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    sources
        ``(n_sources,)`` ``wp.int32`` source vertex indices. The returned distance is measured to
        the nearest source and is zero at the source set.
    t
        Diffusion time. When ``None``, defaults to the squared mean edge length (the
        ``igl::heat_geodesics`` default), which balances accuracy and smoothing. Ignored when
        ``operators`` is given, which already fixes it.
    operators
        Optional precomputed [`heat_operators`][triwarp.geodesic.heat_operators] for this mesh. They
        depend on the mesh only, so passing them back skips the assembly on every solve after the
        first — worth it when computing distance from many different source sets.
    use_robust
        Forwarded to [`heat_operators`][triwarp.geodesic.heat_operators]: build the Laplacian from
        mollified edge lengths, which is what makes the solves survive degenerate triangles. Ignored
        when ``operators`` is supplied. ``potpourri3d.MeshHeatMethodDistanceSolver`` has the same
        flag and defaults it to ``True``; this defaults to ``False`` so the plain call stays exactly
        ``igl::heat_geodesics``.

    Returns
    -------
    wp.array[wp.float64]
        ``(n_vertices,)`` geodesic distance field on ``vertices.device``.

    Raises
    ------
    NotImplementedError
        If a non-trivial solve is required on the CPU device. The two conjugate-gradient solves use
        ``warp.optim.linear.cg``, which returns NaN on the CPU device in Warp 1.14-1.15; a CUDA
        device is required. (Empty meshes or empty source sets return a zero field without solving
        and are allowed on any device.)

    See Also
    --------
    [`heat_operators`][triwarp.geodesic.heat_operators]
    [`cotmatrix`][triwarp.laplacian.cotmatrix]
    [`mean_edge_length`][triwarp.edges.mean_edge_length]
    [`marching_triangles`][triwarp.contour.marching_triangles]
    """
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    n_faces = int(faces.shape[0]) // 3

    if n_vertices == 0 or n_faces == 0 or int(sources.shape[0]) == 0:
        return wp.zeros(n_vertices, dtype=wp.float64, device=device)

    # The two linear solves rely on ``warp.optim.linear.cg``, which returns NaN on the CPU device
    # in Warp 1.14-1.15 (even for a trivial well-conditioned system). Require a CUDA device.
    if wp.get_device(device).is_cpu:
        raise NotImplementedError(
            "heat_geodesic requires a CUDA device: warp.optim.linear.cg produces NaN on the CPU "
            "device in Warp 1.14-1.15."
        )

    if operators is None:
        operators = heat_operators(vertices, faces, t, use_robust=use_robust)
    heat_system, laplacian, cot_entries, normals, areas = operators

    # Heat solve: (M - t L) u = u0, with u0 the source indicator.
    u0 = wp.zeros(n_vertices, dtype=wp.float64, device=device)
    wp.launch(
        kernel_geodesic.seed_source_indicator,
        dim=int(sources.shape[0]),
        inputs=[sources, u0],
        device=device,
    )

    heat = wp.zeros(n_vertices, dtype=wp.float64, device=device)
    wpl.cg(
        heat_system,
        u0,
        heat,
        tol=_CG_TOLERANCE,
        maxiter=10 * n_vertices,
        M=wpl.preconditioner(heat_system, "diag"),
    )

    # Unit vector field X = -grad(u)/|grad(u)|.
    field = wp.empty(n_faces, dtype=wp.vec3d, device=device)
    wp.launch(
        kernel_geodesic.face_gradient_normalized,
        dim=n_faces,
        inputs=[vertices, faces, normals, areas, heat, field],
        device=device,
    )

    # Integrated divergence b = div(X), then Poisson solve L phi = b, i.e. (-L) phi = -b with the
    # positive semi-definite operator.
    divergence = wp.zeros(n_vertices, dtype=wp.float64, device=device)
    wp.launch(
        kernel_geodesic.integrated_divergence,
        dim=n_faces,
        inputs=[vertices, faces, cot_entries, field, divergence],
        device=device,
    )
    poisson_system = wps.bsr_axpy(x=laplacian, alpha=-1.0)
    # Flip sign so the Poisson right-hand side matches the positive semi-definite operator ``-L``.
    neg_divergence = wp.empty(n_vertices, dtype=wp.float64, device=device)
    wp.map(wp.neg, divergence, out=neg_divergence)

    phi = wp.zeros(n_vertices, dtype=wp.float64, device=device)
    wpl.cg(
        poisson_system,
        neg_divergence,
        phi,
        tol=_CG_TOLERANCE,
        maxiter=10 * n_vertices,
        M=wpl.preconditioner(poisson_system, "diag"),
    )

    # Shift so the distance field is zero at the (nearest) source. For a correctly signed field
    # the global minimum sits at the source set, so subtracting it yields a nonnegative field.
    offset = float(phi.numpy().min())
    wp.map(wp.sub, phi, wp.float64(offset), out=phi)
    return phi


def geodesic_ball(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], radius: float, min_count: int = 6
) -> tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Per-vertex geodesic-ball neighborhoods matching ``igl::principal_curvature``'s ``getSphere``.

    For each vertex this is a breadth-first traversal of the mesh edge graph, enqueueing a neighbor
    only when it lies within Euclidean ``radius`` of the center — i.e. the connected component of
    the center within the radius ball. This is a *geodesic* ball rather than a pure Euclidean one,
    so it excludes vertices that are spatially close but lie across a fold of the surface (e.g. the
    opposite wall of a torus tube), which a Euclidean hash-grid query would wrongly include and
    which corrupts the quadric fit. When fewer than ``min_count`` vertices are reachable, the
    nearest out-of-ball vertices are appended (libigl's ``extra_candidates`` path).

    Also returns the per-vertex reference neighbor used to build the tangent frame: the
    lowest-indexed edge neighbor, matching libigl's ``adjacency_list[i][0]``. libigl's symmetrized
    shape operator is frame-dependent, so reproducing its principal values (
    [`principal_curvature`][triwarp.curvature.principal_curvature] with ``frame_independent=False``)
    requires this exact frame; the default frame-independent computation does not depend on it.
    Isolated vertices reference themselves.

    The traversal runs entirely on device. Vertex adjacency is built as a CSR graph via
    [`edges_unique`][triwarp.edges.edges_unique] +
    [`edges_to_csr`][triwarp.graph.edges_to_csr], then a single-pass BFS collects each ball into
    its per-source queue row (the queue prefix *is* the result) and a scan + gather compacts the
    rows into the CSR neighbor buffer. Each source uses fixed-capacity scratch of 512 neighbors;
    if a vertex collects more than that the surplus is dropped and a warning is emitted.

    !!! note

        Distances and tie-breaking are computed in ``float32`` (set-equivalent to the libigl
        reference; borderline ties between equidistant neighbors may resolve differently but leave
        the order-independent quadric fit unchanged).

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions as ``wp.vec3``.
    faces
        Length-``3 * n_faces`` flat triangle index buffer as ``wp.int32``.
    radius
        Geodesic-ball radius in world units.
    min_count
        Minimum neighbors per vertex; the nearest out-of-ball vertices backfill any shortfall.

    Returns
    -------
    tuple[wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32]]
        ``(neighbor_indices, offsets, reference_neighbors)``. ``offsets`` is the
        length-``n_vertices`` exclusive prefix sum of per-vertex neighbor counts (CSR starts);
        vertex ``i`` owns ``neighbor_indices[offsets[i] : offsets[i + 1]]`` with ``offsets[n]``
        implied as the total. ``reference_neighbors`` has length ``n_vertices``.
    """
    device = vertices.device
    n = int(vertices.shape[0])
    if n == 0:
        empty = wp.empty(0, dtype=wp.int32, device=device)
        return (
            empty,
            wp.empty(0, dtype=wp.int32, device=device),
            wp.empty(0, dtype=wp.int32, device=device),
        )

    unique_edges, _ = tw.edges.edges_unique(faces, n_vertices=n)
    adjacency = tw.graph.edges_to_csr(n, unique_edges)
    adj_offsets = adjacency.offsets
    adj_columns = adjacency.columns

    reference_neighbors = wp.empty(n, dtype=wp.int32, device=device)
    wp.launch(
        kernel_geodesic.geodesic_ball_reference_neighbors,
        dim=n,
        inputs=[adj_offsets, adj_columns, reference_neighbors],
        device=device,
    )

    # Per-source scratch lives in shared global-memory pools sized for one chunk of sources
    # (queue rows, an open-addressing visited row pre-filled with -1 per launch, and a small
    # nearest-fallback pool) instead of ~8 KB of per-thread local arrays.
    chunk = min(n, 1 << 15)
    queue_pool = wp.empty(
        (chunk, kernel_bfs._PER_SOURCE_MAX_NEIGHBORS), dtype=wp.int32, device=device
    )
    visited_pool = wp.empty(
        (chunk, kernel_bfs._VISITED_HASH_CAPACITY), dtype=wp.int32, device=device
    )
    ext_dist_pool = wp.empty((chunk, kernel_bfs._EXTRAS_CAPACITY), dtype=wp.float32, device=device)
    ext_idx_pool = wp.empty((chunk, kernel_bfs._EXTRAS_CAPACITY), dtype=wp.int32, device=device)

    overflow = wp.zeros(1, dtype=wp.int32, device=device)
    counts = wp.empty(n, dtype=wp.int32, device=device)
    local_offsets = wp.empty(chunk, dtype=wp.int32, device=device)
    chunk_total_buf = wp.empty(1, dtype=wp.int32, device=device)
    chunk_flats: list[wp.array[wp.int32]] = []
    for start in range(0, n, chunk):
        m = min(chunk, n - start)
        visited_pool.fill_(-1)
        wp.launch(
            kernel_geodesic.query_geodesic_ball_collect,
            dim=m,
            inputs=[
                vertices,
                adj_offsets,
                adj_columns,
                wp.float32(radius),
                wp.int32(min_count),
                wp.int32(start),
                queue_pool,
                visited_pool,
                ext_dist_pool,
                ext_idx_pool,
                counts,
                overflow,
            ],
            device=device,
        )
        # Gather this chunk's queue rows before the next chunk reuses the pools: chunk-local
        # exclusive scan of counts, one 4-byte readback for the chunk total, then a coalesced
        # 2D copy into the chunk's flat buffer.
        wp.utils.array_scan(counts[start : start + m], out_array=local_offsets[:m], inclusive=False)
        wp.map(
            wp.add, local_offsets[m - 1 : m], counts[start + m - 1 : start + m], out=chunk_total_buf
        )
        chunk_total = int(chunk_total_buf.numpy()[0])
        flat_chunk = wp.empty(chunk_total, dtype=wp.int32, device=device)
        if chunk_total > 0:
            wp.launch(
                kernel_geodesic.gather_queue_rows,
                dim=(m, kernel_bfs._PER_SOURCE_MAX_NEIGHBORS),
                inputs=[queue_pool, counts, local_offsets, wp.int32(start), flat_chunk],
                device=device,
            )
        chunk_flats.append(flat_chunk)

    n_overflow = int(overflow.numpy()[0])
    if n_overflow > 0:
        warnings.warn(
            f"geodesic_ball: {n_overflow} neighborhood capacity breaches "
            f"(fixed cap 512); surplus neighbors dropped.",
            stacklevel=2,
        )

    offsets = wp.empty(n, dtype=wp.int32, device=device)
    wp.utils.array_scan(counts, out_array=offsets, inclusive=False)

    if len(chunk_flats) == 1:
        # Single chunk (n <= chunk): the chunk buffer already is the global CSR neighbor buffer.
        return chunk_flats[0], offsets, reference_neighbors

    # Chunk order equals ascending source order, so concatenation lines up with the global scan.
    total = sum(int(flat_chunk.shape[0]) for flat_chunk in chunk_flats)
    neighbor_indices = wp.empty(total, dtype=wp.int32, device=device)
    position = 0
    for flat_chunk in chunk_flats:
        length = int(flat_chunk.shape[0])
        if length > 0:
            wp.copy(neighbor_indices[position : position + length], flat_chunk)
            position += length
    return neighbor_indices, offsets, reference_neighbors
