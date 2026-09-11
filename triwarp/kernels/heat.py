"""
Kernels for the three heat-diffusion solvers of [`triwarp.heat`][triwarp.heat].

Everything runs in ``float64``: far from the source the diffused field decays exponentially and
would underflow ``float32``, destroying the gradient direction and collapsing the far field. The
per-face half-cotangent weights are reused from ``triwarp.laplacian`` (they are ``O(1)`` and
numerically safe in ``float32``); only the assembled operators, the diffused field and the linear
solves need double precision.

Three sections, in the order their wrappers appear: the scalar heat method's diffusion and gradient
normalization, the signed method's curve seeding and level-set constraints, and the vector method's
transport, extension and log map. They share `face_unit_gradient` and the ``float64`` convention,
which is why they are one module rather than three.
"""

import warp as wp

from triwarp.constants import TOLERANCE_ZERO_CONSTANT
from triwarp.kernels.linalg import free_row
from triwarp.kernels.predicates import normalize_or_zero, unit_tangent
from triwarp.kernels.scatter import add_corner_triple
from triwarp.kernels.triangles import corner_triple, face_unit_gradient, face_vertices_vec3d

# --------------------------------------------------------------------------------------
# Scalar heat method: geodesic distance (Crane et al. 2013)
# --------------------------------------------------------------------------------------


@wp.kernel
def seed_source_indicator(sources: wp.array[wp.int32], out_u0: wp.array[wp.float64]) -> None:
    # Set the initial heat to 1 at each source vertex (out_u0 pre-zeroed by the caller).
    t = wp.int32(wp.tid())
    out_u0[sources[t]] = wp.float64(1.0)


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
    f = wp.int32(wp.tid())
    i0, i1, i2 = corner_triple(faces, f)
    v0, v1, v2 = face_vertices_vec3d(vertices, faces, f)
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


# --------------------------------------------------------------------------------------
# Signed heat method: signed distance to oriented curves (Feng & Crane 2024)
# --------------------------------------------------------------------------------------


@wp.kernel
def splat_curve_normals(
    vertices: wp.array[wp.vec3],
    segments: wp.array2d[wp.int32],
    normals: wp.array[wp.vec3],
    basis_x: wp.array[wp.vec3],
    basis_y: wp.array[wp.vec3],
    out_field: wp.array[wp.vec2d],
) -> None:
    # The signed heat method's source term: each curve segment contributes its own *normal* -- the
    # tangent direction perpendicular to it -- to the two vertices it connects, weighted by half the
    # segment's length. Diffusing normals rather than an indicator is what makes the result signed:
    # the field arrives at a point already knowing which side of the curve it is on.
    s = wp.int32(wp.tid())
    a = segments[s, 0]
    b = segments[s, 1]
    edge = vertices[b] - vertices[a]
    length = wp.length(edge)
    if length <= TOLERANCE_ZERO_CONSTANT:
        return
    direction = edge / length
    weight = wp.float64(0.5 * length)

    for k in range(2):
        v = a
        if k == 1:
            v = b
        normal = normals[v]
        # The segment direction as this vertex sees it, then rotated a quarter turn in the tangent
        # plane. ``cross(normal, direction)`` is the left normal, which is the orientation
        # geometry-central signs with: the region a counter-clockwise curve encloses comes out
        # positive.
        tangential, tangential_length = unit_tangent(direction, normal, TOLERANCE_ZERO_CONSTANT)
        if tangential_length <= TOLERANCE_ZERO_CONSTANT:
            continue
        curve_normal = wp.cross(normal, tangential)
        wp.atomic_add(
            out_field,
            v,
            weight
            * wp.vec2d(
                wp.float64(wp.dot(curve_normal, basis_x[v])),
                wp.float64(wp.dot(curve_normal, basis_y[v])),
            ),
        )


@wp.kernel
def vertex_field_to_face_field(
    faces: wp.array[wp.int32],
    normals: wp.array[wp.vec3],
    field: wp.array[wp.vec2d],
    basis_x: wp.array[wp.vec3],
    basis_y: wp.array[wp.vec3],
    out_face_field: wp.array[wp.vec3d],
) -> None:
    # Average the three corners' tangent vectors into one per-face vector, in world space, so the
    # existing cotangent divergence can integrate it. Each corner's 2D components mean nothing
    # outside its own frame, so they have to be expanded to 3D *before* averaging.
    #
    # ``field`` must already be normalized per vertex -- the caller floors the diffused field
    # against its own maximum and passes the unit directions. That precondition is what makes the
    # *absolute* ``TOLERANCE_ZERO_CONSTANT`` below correct here, where everywhere else in this
    # module a diffused field is compared against a relative floor (see ``scale_to_magnitude``):
    # the sum of three unit vectors carries no coordinate scale, so the test only asks whether the
    # three corners cancelled. Hand it a raw diffused field and it zeroes whole faces on any mesh
    # away from unit scale.
    f = wp.int32(wp.tid())
    normal = normals[f]
    total = wp.vec3(0.0, 0.0, 0.0)
    for k in range(3):
        v = faces[f * 3 + k]
        value = field[v]
        total += wp.float32(value[0]) * basis_x[v] + wp.float32(value[1]) * basis_y[v]
    tangential, _length = unit_tangent(total, normal, TOLERANCE_ZERO_CONSTANT)
    out_face_field[f] = wp.vec3d(
        wp.float64(tangential[0]), wp.float64(tangential[1]), wp.float64(tangential[2])
    )


@wp.kernel
def scatter_free_rhs(
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    values: wp.array[wp.float64],
    out_rhs: wp.array2d[wp.float64],
) -> None:
    # Compact a full-length right-hand side down to the unpinned degrees of freedom, in the layout
    # ``linalg.solve_spd_columns`` expects (one row per right-hand side).
    #
    # **Not factored with ``gather_free_solution`` below or with
    # ``smoothing.scatter_free_scalar``, deliberately, and the near-duplicate scan's 0.917 on the
    # last pair is a false positive worth knowing about.** After ``linalg.free_row`` -- which is
    # already the shared guard, and is what makes the statement-run scan's biggest group (nine
    # sites) an extraction rather than a duplicate -- each of the three is *one assignment*, and
    # the three assignments are three different operations:
    #
    #   * this one **compacts** (full-length -> reduced), into a rank-2 destination's row 0;
    #   * ``gather_free_solution`` **expands** (reduced -> full-length) and writes an explicit
    #     zero at every pinned entry, which is why it tests ``fixed_mask`` directly instead of
    #     calling ``free_row`` at all;
    #   * ``smoothing.scatter_free_scalar`` expands from a rank-1 source and **leaves** the pinned
    #     entries alone, because there they hold boundary values the caller set.
    #
    # So the pair the scan matched at 0.917 runs in *opposite directions*, and what is left to
    # share after the guard is the destination's rank and the pinned-entry policy -- which is the
    # whole of what distinguishes them. This is the same verdict, for the same reason, as
    # ``holes.fill_dp_span``'s written decline about the prologue it shares with its tiled sibling:
    # a helper here would cost more at the three call sites than it removes. Recorded because a
    # text-keyed scan cannot see a direction and will match this pair again.
    i = wp.int32(wp.tid())
    ri = free_row(fixed_mask, free_map, i)
    if ri < 0:
        return
    out_rhs[0, ri] = values[i]


@wp.kernel
def gather_free_solution(
    fixed_mask: wp.array[wp.bool],
    free_map: wp.array[wp.int32],
    solution: wp.array2d[wp.float64],
    out_field: wp.array[wp.float64],
) -> None:
    # Expand the reduced solution back over every vertex; the pinned ones keep the value they were
    # pinned to, which for a zero level set is zero.
    i = wp.int32(wp.tid())
    if fixed_mask[i]:
        out_field[i] = wp.float64(0.0)
        return
    out_field[i] = solution[0, free_map[i]]


# --------------------------------------------------------------------------------------
# Vector heat method: transport, scalar extension and the log map (Sharp et al. 2019)
# --------------------------------------------------------------------------------------


@wp.func
def block_mass(mass: wp.float64) -> wp.mat22d:
    # The scalar lumped mass, as one 2x2 block per vertex: the vector problem carries two unknowns
    # per vertex and the same area weight applies to both.
    return wp.mat22d(mass, wp.float64(0.0), wp.float64(0.0), mass)


@wp.kernel
def seed_source_scalars(
    sources: wp.array[wp.int32],
    values: wp.array[wp.float64],
    out_indicator: wp.array[wp.float64],
    out_weighted: wp.array[wp.float64],
) -> None:
    # Scalar extension needs two right-hand sides: where the sources are, and what they carry.
    s = wp.int32(wp.tid())
    v = sources[s]
    wp.atomic_add(out_indicator, v, wp.float64(1.0))
    wp.atomic_add(out_weighted, v, values[s])


@wp.func
def divide_positive(
    numerator: wp.float64, denominator: wp.float64, floor: wp.float64
) -> wp.float64:
    # Away from every source the indicator decays to ~0; guard the ratio rather than emit inf.
    # ``floor`` is a fraction of the indicator field's own maximum, never an absolute value -- see
    # ``scale_to_magnitude`` below for why an absolute one is a silent wrong answer on a large mesh.
    #
    # Not folded into ``array.divide_if_positive`` despite the shared shape: that one's threshold is
    # a fixed zero and its fallback is the unchanged numerator, where this one's threshold is the
    # caller's own floor and its fallback is zero -- two differences, and a four-argument
    # ``divide_or(numerator, denominator, floor, fallback)`` covering both would be a mode argument
    # with one caller per mode (§14).
    if denominator <= floor:
        return wp.float64(0.0)
    return numerator / denominator


@wp.func
def scale_to_magnitude(direction: wp.vec2d, magnitude: wp.float64, floor: wp.float64) -> wp.vec2d:
    # The vector heat method splits a transported vector into a direction (from the vector
    # diffusion) and a magnitude (from a scalar extension): short-time vector diffusion smears
    # magnitudes but preserves directions well.
    #
    # ``floor`` is a fraction of the direction field's own maximum, never an absolute length. The
    # diffused field carries the mesh's scale as ~1/scale^2 -- measured max |direction| of 6.24e-01
    # at unit scale against 6.24e-13 at 1e6 -- so an absolute cutoff turns into "return zero
    # everywhere" on a large mesh: 91 of 162 vertices came back at zero magnitude on an
    # ``icosphere(2)`` scaled by 1e5. Relative, the same field is identical to every digit printed
    # at both scales.
    #
    # The floor separates a vanished direction from a represented one. It does *not* separate signal
    # from round-off and must not be raised in an attempt to: the smallest genuinely diffused value
    # on ``half_torus`` is 8.0e-10 of the maximum, *below* the 8.7e-09 of round-off left at a point
    # where the transported copies cancel exactly. The two populations overlap, so no magnitude cut
    # tells them apart -- which is why ``transport_tangent_vectors`` *reports* resolution as a
    # second mask (one ``array.greater`` at a higher floor) instead of acting on it here.
    #
    # That round-off is the float32 *transport angles*, not the solve and not the frames (which the
    # connection Laplacian never reads). Three measurements: it does not move when the
    # conjugate-gradient tolerance is tightened from 1e-8 to 1e-14; injecting angle noise moves it
    # linearly, extrapolating back to ~2e-07 rad of effective error, which is float32 epsilon on an
    # O(1) angle; and redoing the ring accumulation in float64 while still storing float32 leaves it
    # at 9.9e-09, so it is the angles' storage precision rather than the accumulation order.
    return magnitude * normalize_or_zero(direction, floor)


@wp.func
def tangent_to_world(tangent: wp.vec2, basis_x: wp.vec3, basis_y: wp.vec3) -> wp.vec3:
    return tangent[0] * basis_x + tangent[1] * basis_y


@wp.kernel
def scatter_face_field_to_vertices(
    faces: wp.array[wp.int32],
    face_areas: wp.array[wp.float32],
    field: wp.array[wp.vec3d],
    out_vertex_field: wp.array[wp.vec3],
) -> None:
    # Area-weighted average of a per-face vector field onto vertices, as a plain scatter-add: the
    # weights are the same for all three corners so no normalization is needed before projecting.
    f = wp.int32(wp.tid())
    area = face_areas[f]
    value = wp.vec3(
        wp.float32(field[f][0]) * area,
        wp.float32(field[f][1]) * area,
        wp.float32(field[f][2]) * area,
    )
    add_corner_triple(out_vertex_field, faces, f, value, value, value)


@wp.kernel
def face_unit_gradients(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    normals: wp.array[wp.vec3],
    areas: wp.array[wp.float32],
    values: wp.array[wp.float64],
    sign: wp.float64,
    out_gradient: wp.array[wp.vec3d],
) -> None:
    # The field's unit gradient direction, times an explicit sign. For a distance field this is the
    # radial direction: `sign=1` is the direction *away* from the source (the log map's angle is
    # measured against it), `sign=-1` is `X = -grad(u)/|grad(u)|` (the direction the heat solve's
    # Poisson stage integrates back into a distance).
    #
    # One kernel serving both callers, not two: they used to be `face_gradient_normalized` (the
    # `sign=-1` heat-solve shim) and `face_gradient_unit` (this one, `sign=1`), identical apart from
    # the sign, and both are launched from `triwarp/heat.py`. Deliberately still not merged with
    # `triangles.face_gradients` behind a `normalize` flag, though: the arithmetic is already
    # shared -- this is a one-line launch shim over a `triangles` @wp.func, and
    # `face_unit_gradient` is `normalize(face_gradient(...))` -- so a flag would save one shim while
    # putting a mode argument on `laplacian.face_gradients`' path that only this module's callers
    # would ever set (§14, speculative generality: one caller per mode). The two also return
    # different quantities: a gradient carries the field's rate of change, this carries only a
    # direction.
    f = wp.int32(wp.tid())
    out_gradient[f] = sign * face_unit_gradient(vertices, faces, normals, areas, values, f)


@wp.func
def world_to_tangent_unit(
    value: wp.vec3, basis_x: wp.vec3, basis_y: wp.vec3, tolerance: wp.float32
) -> wp.vec2:
    # Express a 3D vertex field in each vertex's tangent basis, normalized. Only the direction
    # survives, which is all the log map's angle needs.
    #
    # ``tolerance`` must be relative to ``value``'s own field maximum, not a fixed constant:
    # ``value`` is an area-weighted *sum* of unit vectors (see ``scatter_face_field_to_vertices``),
    # so its magnitude carries the mesh's coordinate scale squared and a fixed floor collapses the
    # whole log map to angle zero on any mesh not near unit scale (confirmed: every one of 162
    # vertices on a unit icosphere scaled by 1e-7).
    tangent = wp.vec2(wp.dot(value, basis_x), wp.dot(value, basis_y))
    return normalize_or_zero(tangent, tolerance)


@wp.kernel
def log_map_from_angles(
    radial: wp.array[wp.vec2],
    transported: wp.array[wp.vec2],
    distance: wp.array[wp.float64],
    reference_tolerance: wp.float32,
    out_log: wp.array[wp.vec2],
) -> None:
    # Polar coordinates of each vertex as seen from the source.
    #
    # ``transported`` is the source's reference direction parallel-transported to this vertex, and
    # ``radial`` points away from the source here. The angle between them is preserved by transport
    # along the connecting geodesic, so it *is* the angle at which that geodesic leaves the
    # source -- which with the distance gives the vertex's position in the source's tangent plane.
    v = wp.int32(wp.tid())
    reference = transported[v]
    outward = radial[v]
    r = wp.float32(distance[v])
    # ``reference`` is the raw (unnormalized) diffused field -- the same quantity
    # ``transport_tangent_vectors`` calls ``direction`` and floors relative to its own maximum, so
    # this comparison must be too, for the identical reason ``world_to_tangent_unit`` above needs
    # one. ``outward`` is different: ``world_to_tangent_unit`` already normalized it to unit length
    # or exactly zero, so comparing it to the fixed ``TOLERANCE_ZERO_CONSTANT`` here is only asking
    # "was it zeroed", not re-testing a raw physical magnitude.
    if wp.length(reference) <= reference_tolerance or wp.length(outward) <= TOLERANCE_ZERO_CONSTANT:
        # On the cut locus the transported directions arriving from either side cancel and there is
        # no angle to report -- the log map genuinely has none there. Keep the radius and use angle
        # zero, so the magnitude still means what it should.
        out_log[v] = wp.vec2(r, 0.0)
        return
    angle = wp.atan2(
        reference[0] * outward[1] - reference[1] * outward[0],
        reference[0] * outward[0] + reference[1] * outward[1],
    )
    out_log[v] = wp.vec2(r * wp.cos(angle), r * wp.sin(angle))
