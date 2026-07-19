"""
Kernels for the heat-method geodesic distance solver (Crane et al. 2013).

Everything runs in ``float64``: far from the source the diffused heat decays exponentially and
would underflow ``float32``, destroying the gradient direction and collapsing the far field. The
per-face half-cotangent weights are reused from :mod:`triwarp.laplacian` (they are ``O(1)`` and
numerically safe in ``float32``); only the assembled operators, the diffused field, and the two
linear solves need double precision.
"""

import warp as wp

from triwarp.kernels.algorithms import bfs as kernel_bfs
from triwarp.kernels.array import to_vec3d
from triwarp.kernels.triangles import face_vertices_vec3d


@wp.kernel
def seed_source_indicator(sources: wp.array[wp.int32], out_u0: wp.array[wp.float64]) -> None:
    # Set the initial heat to 1 at each source vertex (out_u0 pre-zeroed by the caller).
    t = int(wp.tid())
    out_u0[sources[t]] = wp.float64(1.0)


@wp.kernel
def face_gradient_normalized(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    normals: wp.array[wp.vec3],
    areas: wp.array[wp.float32],
    u: wp.array[wp.float64],
    out_x: wp.array[wp.vec3d],
) -> None:
    # Per-face gradient of the scalar field u, then X = -grad(u)/|grad(u)| (unit, points away
    # from the source).  grad(u) = 1/(2A) * sum_i u_i (n x e_i^opp), e_i^opp the CCW edge opposite
    # vertex i. Geometry is read in float32 (input precision) and promoted; u is float64.
    f = int(wp.tid())
    i0 = faces[f * 3 + 0]
    i1 = faces[f * 3 + 1]
    i2 = faces[f * 3 + 2]
    v0, v1, v2 = face_vertices_vec3d(vertices, faces, wp.int32(f))
    n = to_vec3d(normals[f])
    area = wp.float64(areas[f])

    grad = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    if area > wp.float64(0.0):
        e0 = v2 - v1  # opposite vertex i0
        e1 = v0 - v2  # opposite vertex i1
        e2 = v1 - v0  # opposite vertex i2
        grad = (u[i0] * wp.cross(n, e0) + u[i1] * wp.cross(n, e1) + u[i2] * wp.cross(n, e2)) / (
            wp.float64(2.0) * area
        )

    length = wp.length(grad)
    unit = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    if length > wp.float64(0.0):
        unit = -grad / length
    out_x[f] = unit


@wp.kernel
def integrated_divergence(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    cot_entries: wp.array2d[wp.float32],
    field: wp.array[wp.vec3d],
    out_div: wp.array[wp.float64],
) -> None:
    # Cotangent integrated divergence of the per-face vector field, accumulated per vertex.
    # cot_entries[f, k] = 1/2 cot(angle at corner k); each vertex gets contributions from the two
    # edges of the triangle incident to it, weighted by the cotangent opposite those edges.
    f = int(wp.tid())
    i0 = faces[f * 3 + 0]
    i1 = faces[f * 3 + 1]
    i2 = faces[f * 3 + 2]
    v0, v1, v2 = face_vertices_vec3d(vertices, faces, wp.int32(f))
    x = field[f]
    c0 = wp.float64(cot_entries[f, 0])
    c1 = wp.float64(cot_entries[f, 1])
    c2 = wp.float64(cot_entries[f, 2])

    d0 = c2 * wp.dot(v1 - v0, x) + c1 * wp.dot(v2 - v0, x)
    d1 = c0 * wp.dot(v2 - v1, x) + c2 * wp.dot(v0 - v1, x)
    d2 = c1 * wp.dot(v0 - v2, x) + c0 * wp.dot(v1 - v2, x)

    wp.atomic_add(out_div, i0, d0)
    wp.atomic_add(out_div, i1, d1)
    wp.atomic_add(out_div, i2, d2)


@wp.kernel
def geodesic_ball_reference_neighbors(
    adj_offsets: wp.array[wp.int32],
    adj_columns: wp.array[wp.int32],
    out_reference: wp.array[wp.int32],
) -> None:
    """Lowest-indexed edge neighbor per vertex (libigl ``adjacency_list[i][0]``); self if alone."""
    i = int(wp.tid())
    start = int(adj_offsets[i])
    end = int(adj_offsets[i + 1])
    if start == end:
        out_reference[i] = i
        return
    minimum = adj_columns[start]
    for k in range(start + 1, end):
        if adj_columns[k] < minimum:
            minimum = adj_columns[k]
    out_reference[i] = minimum


@wp.kernel
def query_geodesic_ball_collect(
    vertices: wp.array[wp.vec3],
    adj_offsets: wp.array[wp.int32],
    adj_columns: wp.array[wp.int32],
    radius: wp.float32,
    min_count: wp.int32,
    chunk_start: wp.int32,
    queue_pool: wp.array2d[wp.int32],
    visited_pool: wp.array2d[wp.int32],
    ext_dist_pool: wp.array2d[wp.float32],
    ext_idx_pool: wp.array2d[wp.int32],
    out_counts: wp.array[wp.int32],
    out_overflow: wp.array[wp.int32],
) -> None:
    # Scratch lives in wrapper-allocated global-memory pools (one row per thread of the current
    # chunk) instead of ~8 KB of per-thread local arrays; the wrapper pre-fills the visited pool
    # with -1 before each launch. Single pass: after this kernel the thread's queue row holds
    # the collected set (``queue_pool[t][:out_counts[chunk_start + t]]``) ready to gather.
    t = int(wp.tid())
    i = int(chunk_start) + t
    out_counts[i] = kernel_bfs.per_source_bfs_collect(
        wp.int32(i),
        vertices,
        adj_offsets,
        adj_columns,
        radius,
        min_count,
        queue_pool[t],
        visited_pool[t],
        ext_dist_pool[t],
        ext_idx_pool[t],
        out_overflow,
    )


@wp.kernel
def gather_queue_rows(
    queue_pool: wp.array2d[wp.int32],
    counts: wp.array[wp.int32],
    local_offsets: wp.array[wp.int32],
    chunk_start: wp.int32,
    out_flat: wp.array[wp.int32],
) -> None:
    # Compact the chunk's queue rows into its flat CSR buffer. Adjacent j threads read one
    # queue row and write one out_flat segment contiguously (coalesced on both sides).
    # ``counts`` is the global per-source array (indexed at chunk_start + t); ``local_offsets``
    # is the chunk-local exclusive scan of this chunk's counts.
    t, j = wp.tid()
    if j >= counts[int(chunk_start) + t]:
        return
    out_flat[local_offsets[t] + j] = queue_pool[t, j]
