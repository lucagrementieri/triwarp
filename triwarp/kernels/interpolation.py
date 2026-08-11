from typing import Any

import warp as wp

from triwarp.kernels.triangles import face_vertices, point_barycentric_cramer


@wp.kernel
def average_onto_faces(
    faces: wp.array[wp.int32],
    vertex_values: wp.array[wp.float32],
    out_face_values: wp.array[wp.float32],
) -> None:
    f = int(wp.tid())
    x0, x1, x2 = face_vertices(vertex_values, faces, wp.int32(f))
    out_face_values[f] = (x0 + x1 + x2) / wp.float32(3.0)


@wp.kernel
def transfer_onto_vertices(
    source_vertices: wp.array[wp.vec3],
    source_faces: wp.array[wp.int32],
    source_values: wp.array[Any],
    closest: wp.array[wp.vec3],
    face_id: wp.array[wp.int32],
    out_values: wp.array[Any],
    out_distance: wp.array[wp.float32],
) -> None:
    # Barycentric resample of a source per-vertex field at each target vertex's closest point.
    #
    # ``face_id < 0`` is the ``max_dist`` miss case. The value slot keeps whatever the wrapper
    # pre-filled (so a caller who narrowed the search still gets a deterministic buffer), and the
    # distance is promoted to ``inf``: ``wp.mesh_query_point_no_sign`` reports ``max_dist`` there,
    # which is indistinguishable from a genuine hit at exactly that range.
    i = int(wp.tid())
    f = face_id[i]
    if f < 0:
        out_distance[i] = wp.float32(wp.INF)
        return
    v0, v1, v2 = face_vertices(source_vertices, source_faces, f)
    bary = point_barycentric_cramer(v0, v1, v2, closest[i])
    a0, a1, a2 = face_vertices(source_values, source_faces, f)
    out_values[i] = a0 * bary[0] + a1 * bary[1] + a2 * bary[2]


@wp.func
def gaussian_kernel_weight(distance: wp.float32, inverse_scale: wp.float32) -> wp.float32:
    # ``exp(-(sharpness * d / radius) ** 2)``, VTK's ``vtkGaussianKernel``, with the caller having
    # folded ``sharpness / radius`` into one reciprocal length. An unused neighbour slot arrives at
    # distance ``inf`` and weighs exactly 0, which is what lets the padded k-nearest rows and the
    # ragged ball rows share this kernel.
    scaled = distance * inverse_scale
    return wp.exp(-scaled * scaled)


@wp.kernel
def interpolate_from_points(
    source_values: wp.array[Any],
    neighbor_indices: wp.array[wp.int32],
    neighbor_distances: wp.array[wp.float32],
    offsets: wp.array[wp.int32],
    inverse_scale: wp.float32,
    out_values: wp.array[Any],
) -> None:
    # Gaussian-weighted mean of one query's neighbours, over the CSR the neighbour queries return.
    # A query with no neighbours, or whose every weight underflowed, is left at the null value the
    # wrapper pre-filled -- so the miss case needs no per-dtype null argument here.
    q = int(wp.tid())
    start = offsets[q]
    stop = offsets[q + 1]
    if start >= stop:
        return

    # A zero of the field's own dtype, which a generic kernel cannot spell any other way.
    accumulated = source_values[0] * wp.float32(0.0)
    total = float(0.0)  # noqa: UP018 — float() declares a mutable Warp dynamic variable
    coincident = int(-1)  # noqa: UP018, RUF046 — int() declares a mutable Warp dynamic variable
    for slot in range(start, stop):
        index = neighbor_indices[slot]
        if index < 0:
            continue
        distance = neighbor_distances[slot]
        if distance <= wp.float32(0.0):
            coincident = index
        weight = gaussian_kernel_weight(distance, inverse_scale)
        accumulated = accumulated + source_values[index] * weight
        total += weight

    # VTK returns the source's own value where a query sits on a data point, rather than blending
    # its neighbours in; the interpolant is then exact at the data.
    if coincident >= 0:
        out_values[q] = source_values[coincident]
    elif total > wp.float32(0.0):
        out_values[q] = accumulated / total


# Concrete overloads, registered at import -- rationale in ``triwarp/kernels/reduce.py``, rule in
# CLAUDE.md section 4. Measured at 2 overloads across **3** module loads.
#
# The dtype set is the one ``transfer_onto_vertices``'s own docstring promises -- "any Warp dtype
# closed under scaling and addition works: ``wp.float32`` for a scalar, ``wp.vec3`` for a normal or
# a colour" -- so registering exactly those two keeps the code and the documentation agreeing.
# ``interpolate_from_points`` documents the same pair.
def _register_overloads() -> None:
    """Instantiate every concrete overload of this module's generic kernels."""
    for dtype in (wp.float32, wp.vec3):
        wp.overload(
            transfer_onto_vertices,
            [
                wp.array[wp.vec3],
                wp.array[wp.int32],
                wp.array[dtype],
                wp.array[wp.vec3],
                wp.array[wp.int32],
                wp.array[dtype],
                wp.array[wp.float32],
            ],
        )
        wp.overload(
            interpolate_from_points,
            [
                wp.array[dtype],
                wp.array[wp.int32],
                wp.array[wp.float32],
                wp.array[wp.int32],
                wp.float32,
                wp.array[dtype],
            ],
        )


_register_overloads()
