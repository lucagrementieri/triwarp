from typing import Any

import warp as wp

from triwarp.kernels.array import trilinear_cell, trilinear_weight
from triwarp.kernels.triangles import face_vertices, point_barycentric_cramer


@wp.kernel
def average_onto_faces(
    faces: wp.array[wp.int32],
    vertex_values: wp.array[wp.float32],
    out_face_values: wp.array[wp.float32],
) -> None:
    f = wp.int32(wp.tid())
    x0, x1, x2 = face_vertices(vertex_values, faces, f)
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
    i = wp.int32(wp.tid())
    f = face_id[i]
    if f < 0:
        out_distance[i] = wp.float32(wp.INF)
        return
    v0, v1, v2 = face_vertices(source_vertices, source_faces, f)
    bary = point_barycentric_cramer(v0, v1, v2, closest[i])
    a0, a1, a2 = face_vertices(source_values, source_faces, f)
    out_values[i] = a0 * bary[0] + a1 * bary[1] + a2 * bary[2]


@wp.kernel
def apply_transfer_operator(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    weights: wp.array[wp.float32],
    source_values: wp.array[Any],
    out_values: wp.array[Any],
) -> None:
    # One CSR row per output element: the field's value there is the weighted sum of the source
    # values the row references. Written rather than delegated to ``warp.sparse.bsr_mv``, which
    # requires the vector's dtype to match the matrix's 1x1 ``float32`` block, so it can move a
    # scalar field and nothing else -- where an attribute is as often a ``wp.vec3`` colour or a
    # ``wp.vec2`` UV, and those are exactly what a topology edit drops today.
    #
    # The accumulator is seeded as ``source_values[0] - source_values[0]``, which is how a generic
    # ``Any`` kernel spells "the additive identity of this dtype": there is no ``dtype(0)``
    # constructor to reach for, and ``x * wp.float32(0.0)`` would not parse for a dtype whose scalar
    # is not float32 (Warp requires both operands' scalars to match). It also gives an empty row --
    # an output element no source reaches -- the right answer rather than uninitialized memory.
    #
    # The weights being ``float32`` is what bounds the dtype set: the product below needs the
    # field's own scalar to be float32 too, so ``wp.float32`` / ``wp.vec2`` / ``wp.vec3`` work and a
    # ``float64`` field does not. That is not a gap -- these operators are assembled from float32
    # vertex data, so a float64 field carried through one would advertise precision the weights do
    # not have.
    row = wp.int32(wp.tid())
    total = source_values[0] - source_values[0]
    for k in range(offsets[row], offsets[row + 1]):
        total = total + source_values[columns[k]] * weights[k]
    out_values[row] = total


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
    q = wp.int32(wp.tid())
    start = offsets[q]
    stop = offsets[q + 1]
    if start >= stop:
        return

    # A zero of the field's own dtype, which a generic kernel cannot spell any other way.
    accumulated = source_values[0] * wp.float32(0.0)
    total = wp.float32(0.0)
    coincident = wp.int32(-1)
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
@wp.kernel(enable_backward=False)
def sample_grid_trilinear(
    field: wp.array3d[Any],
    lower: wp.vec3,
    inverse_spacing: wp.vec3,
    points: wp.array[wp.vec3],
    out_values: wp.array[Any],
) -> None:
    # Read a dense lattice at one arbitrary position, as the trilinearly weighted sum of the eight
    # corners around it -- the transpose of ``kernels/scatter.splat_grid_trilinear``. The wrapper's
    # ``Notes`` records which round trips through the pair are exact; averaging makes the two
    # adjoint rather than inverse.
    #
    # Summed rather than nested as ``wp.lerp``. ``kernels/reconstruction.poisson_sample_grid`` is
    # the lerp spelling over a flat ``res**3`` float32 buffer; the two are the same function and
    # differ only in float32 summation order, and they are deliberately kept apart because that
    # one's arithmetic is what the Poisson iso-value was measured against. All three share
    # ``trilinear_cell``.
    #
    # A position outside the lattice reads the nearest cell's stencil rather than a null value,
    # because ``trilinear_cell`` clamps: the field is extended by its boundary cells, which is what
    # a signed-distance or density lattice wants and is the convention ``poisson_sample_grid``
    # already had.
    s = wp.int32(wp.tid())
    shape = wp.vec3i(field.shape[0], field.shape[1], field.shape[2])
    base, fractions = trilinear_cell(wp.cw_mul(points[s] - lower, inverse_spacing), shape)
    accumulator = out_values.dtype(0.0)
    for offset_x in range(2):
        for offset_y in range(2):
            for offset_z in range(2):
                accumulator += (
                    trilinear_weight(fractions, offset_x, offset_y, offset_z)
                    * field[base[0] + offset_x, base[1] + offset_y, base[2] + offset_z]
                )
    out_values[s] = accumulator


# CLAUDE.md section 4. Measured at 2 overloads across **3** module loads.
#
# The dtype set is the one ``transfer_onto_vertices``'s own docstring promises -- "any Warp dtype
# closed under scaling and addition works: ``wp.float32`` for a scalar, ``wp.vec3`` for a normal or
# a colour" -- so registering exactly those two keeps the code and the documentation agreeing.
# ``interpolate_from_points`` documents the same pair. ``apply_transfer_operator`` adds ``wp.vec2``,
# since it is the one that carries a *stored attribute* through a topology edit and a UV pair is one
# of the things such an attribute is. It cannot add ``wp.float64``: its weights are float32.
def _register_overloads() -> None:
    """Instantiate every concrete overload of this module's generic kernels."""
    for dtype in (wp.float32, wp.vec2, wp.vec3):
        wp.overload(
            apply_transfer_operator,
            [
                wp.array[wp.int32],
                wp.array[wp.int32],
                wp.array[wp.float32],
                wp.array[dtype],
                wp.array[dtype],
            ],
        )
    for dtype in (wp.float32, wp.vec3):
        wp.overload(
            sample_grid_trilinear,
            [wp.array3d[dtype], wp.vec3, wp.vec3, wp.array[wp.vec3], wp.array[dtype]],
        )
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
