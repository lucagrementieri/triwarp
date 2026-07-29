"""
Precision-generic geometric predicates (ports of MeshLib ``MRTriMath.h`` / ``MRReducePath``).

These were duplicated three times before this module existed: a ``float32`` set in
``kernels/reconstruction.py``, a byte-equivalent ``float64`` set in ``kernels/remesh.py``, and a
third partial copy in ``kernels/hole_filling.py``. Each `@wp.func` here is generic over the scalar
type, so one definition instantiates at whatever precision the calling kernel uses.

Degenerate inputs return [`float_inf`][triwarp.kernels.predicates.float_inf] — an actual infinity,
so callers detect the case with ``wp.isinf`` rather than by comparing against a magic large value.

!!! note
    Vector parameters are annotated ``Any`` rather than a concrete ``wp.vec3`` / ``wp.vec3d``:
    Warp has no generic vector annotation, and ``Any`` lets the kernel's own vector type flow
    through. Scalars use ``wp.Float``. Literals inside these functions must be built with
    ``type(x)(...)`` so they instantiate at the caller's precision.
"""

from typing import Any

import warp as wp

from triwarp.kernels.array import cross2


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
def dihedral_angle(left_normal: Any, right_normal: Any, edge_vector: Any) -> wp.Float:
    # Signed angle between the two face normals about the shared edge (MeshLib ``dihedralAngle``).
    edge_direction = wp.normalize(edge_vector)
    sine = wp.dot(edge_direction, wp.cross(left_normal, right_normal))
    cosine = wp.dot(left_normal, right_normal)
    return wp.atan2(sine, cosine)


@wp.func
def circumcircle_diameter_sq(a: Any, b: Any, c: Any) -> wp.Float:
    # MRTriMath.h ``circumcircleDiameterSq``: squared diameter of triangle ABC's circumcircle.
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
def mincircle_diameter_sq(a: Any, b: Any, c: Any) -> wp.Float:
    # MRTriMath.h ``minCircleDiameterSq``: for an obtuse triangle the smallest enclosing circle is
    # the one on the longest side, otherwise it is the circumcircle.
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
    # MRTriMath.h ``triangleAspectRatio``: circum-radius over twice the in-radius. Grows without
    # bound for slivers, so a degenerate triangle returns +inf.
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
    # MRReducePath.cpp ``unfoldOnPlane``: place ``c`` in the plane relative to the already-placed
    # 2D point ``d``, preserving the angle and length of the 3D pair (b, c).
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
    # MRReducePath.cpp ``lineIsect``: parameter along segment 0->B where it meets line C-D.
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
    # MRReducePath ``isUnfoldQuadrangleConvex``: unfold triangles ABC and ACD into a common plane;
    # the quadrangle is convex exactly when the shortest B->D path crosses diagonal AC strictly
    # between A and C. ``wp.vector`` seeds B's 2D image at the caller's precision, which is how this
    # stays generic despite mapping vec3 -> vec2 / vec3d -> vec2d.
    unfold_b = wp.vector(wp.length(b - a), type(a[0])(0.0))
    unfold_c = unfold_on_plane(b - a, c - a, unfold_b, wp.bool(True))
    unfold_d = unfold_on_plane(c - a, d - a, unfold_c, wp.bool(True))
    x = line_isect(unfold_c, unfold_b, unfold_d)
    # MeshLib clamps ``x`` to [0, 1] first; that cannot change this strict test, so it is dropped.
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
