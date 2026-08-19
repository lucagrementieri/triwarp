"""
Precision-generic geometric predicates and the small triangle quantities they share.

These are the standard triangle quantities -- double area, circumcircle and minimum-enclosing-
circle diameters, aspect ratio, dihedral angle -- and they were duplicated three times before this
module existed: a ``float32`` set in ``kernels/reconstruction.py``, a byte-equivalent ``float64``
set in ``kernels/remesh.py``, and a third partial copy in ``kernels/holes.py``. Each
`@wp.func` here is generic over the scalar type, so one definition instantiates at whatever
precision the calling kernel uses.

The three later arrivals are the same defect one level out: ``triangle_aabb`` came from
``kernels/intersection.py`` and ``triangle_double_area`` / ``circumcircle_diameter`` from
``kernels/holes.py``, where a general triangle quantity had ended up inside a module that owns an
*algorithm* and other modules were importing the algorithm to reach the geometry.

Degenerate inputs return [`float_inf`][triwarp.kernels.predicates.float_inf] — an actual infinity,
so callers detect the case with ``wp.isinf`` rather than by comparing against a magic large value.

!!! note
    Vector parameters are annotated ``Any`` rather than a concrete ``wp.vec3`` / ``wp.vec3d``:
    Warp has no generic vector annotation, and ``Any`` lets the kernel's own vector type flow
    through. Scalars use ``wp.Float``. Literals inside these functions must be built with
    ``type(x)(...)`` so they instantiate at the caller's precision.
"""

import math
from typing import Any

import warp as wp

from triwarp.kernels.array import cross2

# Full turn in ``float64``; ``type(x)(TWO_PI_F64)`` narrows it to the caller's precision, and at
# ``float32`` that is bit-identical to ``2 * wp.PI`` (verified on both devices, Warp 1.16).
TWO_PI_F64 = wp.constant(wp.float64(2.0 * math.pi))


@wp.func
def float_inf(sample: wp.Float) -> wp.Float:
    # Positive infinity at ``sample``'s precision: the degenerate-case sentinel for every predicate
    # below. Callers test it with ``wp.isinf``.
    return type(sample)(wp.INF)


@wp.func
def orient2d(a: Any, b: Any, c: Any):
    # Twice the signed area of triangle ABC in 2D: > 0 when C is left of the directed line A->B.
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


@wp.func
def triangle_normal(a: Any, b: Any, c: Any):
    # Unit normal of triangle ABC, or the zero vector when the triangle is degenerate. Warp's
    # ``kEps`` is 0, so ``normalize`` already returns the zero vector for a zero-length input.
    return wp.normalize(wp.cross(b - a, c - a))


@wp.func
def triangle_double_area(a: Any, b: Any, c: Any) -> wp.Float:
    # Twice the area of triangle ABC: the norm of the edge cross product.
    # Kept undivided because most callers either compare it against zero or fold the half into a
    # constant of their own.
    return wp.length(wp.cross(b - a, c - a))


@wp.func
def triangle_aabb(a: Any, b: Any, c: Any):
    # Lower and upper corners of triangle ABC's axis-aligned bounding box. ``wp.min`` / ``wp.max``
    # on vectors are element-wise.
    return wp.min(a, wp.min(b, c)), wp.max(a, wp.max(b, c))


@wp.func
def vector_angle(a: Any, b: Any) -> wp.Float:
    # Unsigned angle in [0, pi] between two vectors, as ``atan2(|a x b|, a . b)`` rather than
    # ``acos(a . b)``.
    #
    # The atan2 form is the accurate one and the reason this is the single spelling in the tree.
    # ``acos`` has an infinite derivative at +-1, so for nearly-parallel vectors -- the *common*
    # case here: two coplanar faces across an edge, a straight run of a polyline, two near-collinear
    # endpoint normals -- it amplifies the round-off already in the dot product, while the cross
    # product carries the small angle directly. Measured at a true separation of 1e-7 rad in
    # float64: this form returns 1.0e-07, ``acos`` returns 9.996e-08, four digits already gone.
    # Against a float64 reference on float32 face normals, worst error over all adjacent face pairs:
    # 1.3e-06 -> 7.6e-08 on an icosphere(3), and 3.5e-04 -> 4.7e-08 on a 64-section cylinder, whose
    # cap fans are coplanar. That 3.5e-04 is *past* the 1e-5 the parity tests compare at, so the old
    # spelling was one fixture away from failing rather than merely less tidy.
    # It is also scale-free, so callers may pass unnormalized vectors (``tris_angle_profit`` passes
    # raw cross products) and a zero-length input gives ``atan2(0, 0) == 0`` rather than the
    # spurious pi/2 that ``acos`` of a zeroed ``normalize`` returns.
    return wp.atan2(wp.length(wp.cross(a, b)), wp.dot(a, b))


@wp.func
def dihedral_angle(left_normal: Any, right_normal: Any, edge_vector: Any) -> wp.Float:
    # Signed angle between the two face normals about the shared edge.
    edge_direction = wp.normalize(edge_vector)
    sine = wp.dot(edge_direction, wp.cross(left_normal, right_normal))
    cosine = wp.dot(left_normal, right_normal)
    return wp.atan2(sine, cosine)


@wp.func
def circumcircle_diameter_sq(a: Any, b: Any, c: Any) -> wp.Float:
    # Squared diameter of triangle ABC's circumcircle.
    # A zero-length side collapses to the opposite side; zero area means no circumcircle at all.
    ab = wp.length_sq(b - a)
    ca = wp.length_sq(a - c)
    bc = wp.length_sq(c - b)
    zero = type(ab)(0.0)
    if ab <= zero:
        return ca
    if ca <= zero:
        return bc
    if bc <= zero:
        return ab
    f = wp.length_sq(wp.cross(b - a, c - a))
    if f <= zero:
        return float_inf(ab)
    return ab * ca * bc / f


@wp.func
def circumcircle_diameter(a: Any, b: Any, c: Any) -> wp.Float:
    # Diameter (not squared) of triangle ABC's circumcircle; +inf when degenerate, which the
    # square root preserves.
    return wp.sqrt(circumcircle_diameter_sq(a, b, c))


@wp.func
def delone_metrics(a: Any, b: Any, c: Any, d: Any):
    # For quadrangle ABCD, the pair of Delone metrics compared when deciding whether to flip the
    # diagonal AC to BD: each is the larger circumcircle of the two triangles that diagonal makes.
    # Either may be infinite (a degenerate triangle), so callers test with ``wp.isinf`` rather than
    # subtracting blindly.
    metric_ac = wp.max(circumcircle_diameter_sq(a, c, d), circumcircle_diameter_sq(c, a, b))
    metric_bd = wp.max(circumcircle_diameter_sq(b, d, a), circumcircle_diameter_sq(d, b, c))
    return metric_ac, metric_bd


@wp.func
def mincircle_diameter_sq(a: Any, b: Any, c: Any) -> wp.Float:
    # Squared diameter of the smallest circle enclosing triangle ABC: for an obtuse triangle that
    # is the circle on the longest side, otherwise it is the circumcircle.
    ab = wp.length_sq(b - a)
    ca = wp.length_sq(a - c)
    bc = wp.length_sq(c - b)
    if ca >= bc + ab:
        return ca
    if bc >= ab + ca:
        return bc
    if ab >= ca + bc:
        return ab
    f = wp.length_sq(wp.cross(b - a, c - a))
    if f <= type(ab)(0.0):
        return float_inf(ab)
    return ab * ca * bc / f


@wp.func
def triangle_aspect_ratio(a: Any, b: Any, c: Any) -> wp.Float:
    # Circum-radius over twice the in-radius. Grows without bound for slivers, so a degenerate
    # triangle returns +inf.
    bc = wp.length(c - b)
    ca = wp.length(a - c)
    ab = wp.length(b - a)
    half_perimeter = (bc + ca + ab) / type(bc)(2.0)
    denominator = (
        type(bc)(8.0) * (half_perimeter - bc) * (half_perimeter - ca) * (half_perimeter - ab)
    )
    if denominator <= type(bc)(0.0):
        return float_inf(bc)
    return bc * ca * ab / denominator


@wp.func
def unfold_on_plane(b: Any, c: Any, d: Any, to_left: wp.bool):
    # Place ``c`` in the plane relative to the already-placed 2D point ``d``, preserving the angle
    # and length of the 3D pair (b, c).
    dot_bc = wp.dot(b, c)
    cross_bc = wp.length(wp.cross(b, c))
    dd = wp.length_sq(d)
    zero = type(dd)(0.0)
    if dd <= zero:
        return d * zero
    # Initialised before the branch: a variable assigned only inside an ``if`` is uninitialised in
    # Warp when the branch is not taken.
    orthogonal = type(d)(-d[1], d[0])
    if not to_left:
        orthogonal = type(d)(d[1], -d[0])
    return (dot_bc * d + cross_bc * orthogonal) / dd


@wp.func
def line_isect(b: Any, c: Any, d: Any) -> wp.Float:
    # Parameter along segment 0->B where it meets line C-D.
    c1 = cross2(d, c)
    c2 = cross2(c - b, d - b)
    zero = type(c1)(0.0)
    if c1 == zero and c2 == zero:
        bb = wp.length_sq(b)
        if bb == zero:
            return zero
        return (wp.dot(c, b) + wp.dot(d, b)) / (type(c1)(2.0) * bb)
    cc = c1 + c2
    if cc == zero:
        return zero
    return c1 / cc


@wp.func
def is_unfold_quadrangle_convex(a: Any, b: Any, c: Any, d: Any) -> wp.bool:
    # Unfold triangles ABC and ACD into a common plane; the quadrangle is convex exactly when the
    # shortest B->D path crosses diagonal AC strictly between A and C. ``wp.vector`` seeds B's 2D
    # image at the caller's precision, which is how this stays generic despite mapping
    # vec3 -> vec2 / vec3d -> vec2d.
    unfold_b = wp.vector(wp.length(b - a), type(a[0])(0.0))
    unfold_c = unfold_on_plane(b - a, c - a, unfold_b, wp.bool(True))
    unfold_d = unfold_on_plane(c - a, d - a, unfold_c, wp.bool(True))
    x = line_isect(unfold_c, unfold_b, unfold_d)
    # Clamping ``x`` to [0, 1] first is a common formulation and cannot change this strict test,
    # so it is dropped.
    return x > type(x)(0.0) and x < type(x)(1.0)


@wp.func
def project_out_normal(vector: Any, normal: Any):
    # Component of ``vector`` in the plane orthogonal to unit ``normal``.
    return vector - wp.dot(vector, normal) * normal


@wp.func
def unit_tangent(vector: Any, normal: Any, tolerance: Any):
    # Project ``vector`` into the plane orthogonal to unit ``normal`` and normalize it, returning
    # the projection's length alongside.
    #
    # The length is returned rather than folded into a fallback because every caller wants a
    # different one when it is degenerate: keep a precomputed perpendicular, return the unprojected
    # input, skip the contribution entirely, or report failure. Callers compare the length against
    # their own tolerance and branch; when it is above tolerance the first result is the unit
    # tangent, and when it is not the first result is the *unnormalized* projection.
    tangential = project_out_normal(vector, normal)
    length = wp.length(tangential)
    if length > tolerance:
        return tangential / length, length
    return tangential, length


@wp.func
def angle_defect(angle_sum: wp.Float) -> wp.Float:
    """Angle defect at a vertex: a full turn minus the incident corner angles."""
    return type(angle_sum)(TWO_PI_F64) - angle_sum


@wp.func
def barycentric_2d(q0: wp.vec2, q1: wp.vec2, q2: wp.vec2, p: wp.vec2) -> wp.vec3:
    """
    Barycentric coordinates ``(b0, b1, b2)`` of ``p`` in the 2D triangle ``(q0, q1, q2)``.

    Returns a vector with a negative component for degenerate triangles, so a caller testing
    ``min(b) >= -eps`` for containment rejects them without a separate area check.
    """
    v0 = q1 - q0
    v1 = q2 - q0
    v2 = p - q0
    d00 = wp.length_sq(v0)
    d01 = wp.dot(v0, v1)
    d11 = wp.length_sq(v1)
    d20 = wp.dot(v2, v0)
    d21 = wp.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if wp.abs(denom) < wp.float32(1e-20):
        return wp.vec3(-1.0, -1.0, -1.0)
    inverse_denominator = 1.0 / denom
    b1 = (d11 * d20 - d01 * d21) * inverse_denominator
    b2 = (d00 * d21 - d01 * d20) * inverse_denominator
    b0 = 1.0 - b1 - b2
    return wp.vec3(b0, b1, b2)
