"""
Kernels for point-cloud surface reconstruction.

The heart is
[`build_local_triangulations`][triwarp.kernels.reconstruction.build_local_triangulations], a
per-point fan triangulation: each point orders its neighbours by angle in the tangent plane and
then improves the fan toward a Delaunay-like one. The geometry helpers below are the standard
triangle predicates, shared with ``kernels/predicates.py``.
"""

import warp as wp

from triwarp.constants import FLOAT32_INF_CONSTANT, PI, TWO_PI
from triwarp.kernels.array import (
    lattice_position,
    ravel_index,
    trilinear_cell,
    trilinear_corner,
    trilinear_weight,
    update_argmax,
)
from triwarp.kernels.predicates import (
    delone_metrics,
    is_unfold_quadrangle_convex,
    orient2d,
    project_out_normal,
    triangle_aspect_ratio,
    vector_angle,
)

# Compile-time upper bound on the per-point fan size (neighbours kept for one center).
# Per-thread scratch arrays are sized to this; the runtime ``max_neighbours`` must not exceed it.
MAX_NEIGHBOURS = 64

# Faces with aspect ratio above this are treated as degenerate and removed first.
CRITICAL_ASPECT_RATIO = wp.constant(wp.float32(1e3))
# Neighbour filter: drop a neighbour whose oriented normal opposes the center normal.
NORMAL_FILTER_DOT = wp.constant(wp.float32(-0.3))


# --------------------------------------------------------------------------------------
# Lexicographic incremental triangulation (the Delaunay seed)
# --------------------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def lexicographic_triangulation(
    points: wp.array[wp.vec2d],
    order: wp.array[wp.int32],
    max_faces: wp.int32,
    boundary: wp.array[wp.int32],
    boundary_next: wp.array[wp.int32],
    orientations: wp.array[wp.float64],
    out_faces: wp.array[wp.int32],
    out_counts: wp.array[wp.int32],
) -> None:
    # One thread, deliberately: this is a hull sweep whose every step depends on the previous
    # boundary, so there is nothing to parallelise. It is launched on the **CPU** device for the
    # same reason -- measured on 20 000 points, the identical sweep costs 352 ms as the Python loop
    # this replaces, 93 ms in a single CUDA thread, and 1.40 ms here. A single GPU thread is the
    # wrong tool and the numbers say so by a factor of 66.
    #
    # ``out_counts`` carries [faces written, faces wanted]. They differ only when a degenerate input
    # (duplicate or collinear points) drives the visible arc past the 2n bound a real triangulation
    # obeys; the wrapper raises on that rather than letting the writes wrap.
    if wp.tid() != 0:
        return
    n = order.shape[0]
    zero = wp.float64(0.0)
    n_faces = wp.int32(0)
    n_boundary = wp.int32(0)

    for i in range(2, n):
        ci = order[i]
        curr = points[ci]

        if n_boundary == 0:
            # Every point so far is collinear; the first off-line point fans the whole prefix.
            side = orient2d(points[order[0]], points[order[1]], curr)
            if side != zero:
                for j in range(i - 1):
                    if n_faces < max_faces:
                        first = order[j]
                        second = order[j + 1]
                        if side < zero:
                            first, second = second, first
                        out_faces[3 * n_faces + 0] = first
                        out_faces[3 * n_faces + 1] = second
                        out_faces[3 * n_faces + 2] = ci
                    n_faces += 1
                for j in range(i + 1):
                    # The prefix in lex order, plus ``curr``; reversed when ``curr`` is right of it,
                    # so the boundary comes out counter-clockwise either way.
                    if side > zero:
                        boundary[j] = order[j]
                    else:
                        boundary[j] = order[i - j]
                n_boundary = i + 1
            continue

        nb = n_boundary
        for j in range(nb):
            following = j + 1
            if following == nb:
                following = 0
            orientations[j] = orient2d(points[boundary[j]], points[boundary[following]], curr)

        # Every edge ``curr`` can see becomes a triangle, wound so the new face agrees with the
        # boundary's orientation.
        for j in range(nb):
            if orientations[j] < zero:
                following = j + 1
                if following == nb:
                    following = 0
                if n_faces < max_faces:
                    out_faces[3 * n_faces + 0] = boundary[following]
                    out_faces[3 * n_faces + 1] = boundary[j]
                    out_faces[3 * n_faces + 2] = ci
                n_faces += 1

        # The visible edges form one contiguous arc: ``left`` starts it, ``right`` ends it (the
        # first kept vertex).
        left = wp.int32(-1)
        right = wp.int32(-1)
        for j in range(nb):
            previous = j - 1
            if previous < 0:
                previous = nb - 1
            if orientations[j] >= zero and orientations[previous] < zero:
                right = j
            elif orientations[j] < zero and orientations[previous] >= zero:
                left = j
        # No visible edge at all means a degenerate insertion (a duplicate point, since a lex sweep
        # always sees the hull from the new rightmost point otherwise). Both indices stay -1, which
        # in the Python original wrapped to the last boundary entry and broke the walk immediately;
        # mapping them to ``nb - 1`` reproduces that exactly rather than reading out of bounds. That
        # collapse also fires -- and drops every other boundary vertex from the hull, unreferenced,
        # rather than only the duplicate -- when ``curr`` is exactly collinear with the whole
        # current boundary rather than a duplicate of one point (all ``orientations`` land on
        # exactly zero). Searched for a genuinely-lex-ordered general-position input that reaches
        # this branch with ``nb >= 2`` and confirmed a real hand-traced case exists in principle at
        # ``nb == 2``: two boundary points and a later point exactly collinear with both. But a
        # randomized search over 220 000+ properly lex-sorted point sets (3-7 points, general
        # position) never reduced
        # ``n_boundary`` to 2 at all, degenerate or not -- reaching it seems to itself require an
        # earlier degenerate (already-collinear) step, which is the same input class
        # ``test_delaunay_collinear`` already accepts producing zero faces for. So the dropped
        # vertex ends up in the same "unreferenced by any face" outcome that class already produces
        # deliberately, rather than corrupting anything downstream (the rest of the sweep proceeds
        # correctly against whatever boundary this collapse leaves, degenerate or not). Left as is;
        # revisit only with a concrete reachable repro, not a hand-traced hypothetical one.
        if right < 0:
            right = nb - 1
        if left < 0:
            left = nb - 1

        # Keep the non-visible arc right..left going forward, then insert ``curr`` after it.
        n_kept = wp.int32(0)
        k = right
        for _ in range(nb):
            boundary_next[n_kept] = boundary[k]
            n_kept += 1
            if k == left:
                break
            k += 1
            if k == nb:
                k = 0
        boundary_next[n_kept] = ci
        n_kept += 1
        for j in range(n_kept):
            boundary[j] = boundary_next[j]
        n_boundary = n_kept

    out_counts[0] = wp.min(n_faces, max_faces)
    out_counts[1] = n_faces


# --------------------------------------------------------------------------------------
# Geometry primitives
# --------------------------------------------------------------------------------------


@wp.func
def delone_flip_profit_sq(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3) -> wp.float32:
    # Empty-circumcircle (Delone) profit of flipping diagonal AC to BD.
    metric_ac, metric_bd = delone_metrics(a, b, c, d)
    return metric_ac - metric_bd


@wp.func
def tris_angle_profit(
    a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3, crit_ang: wp.float32
) -> wp.float32:
    # Dihedral angle across edge AC minus the critical angle.
    ac = c - a
    ab = b - a
    ad = d - a
    dir_abc = wp.cross(ab, ac)
    dir_acd = wp.cross(ac, ad)
    return vector_angle(dir_abc, dir_acd) - crit_ang


@wp.func
def cycle_step(nbr: wp.array[wp.int32], m: wp.int32, i: wp.int32, step: wp.int32) -> wp.int32:
    # Walk the ring of ``m`` fan slots from ``i`` in direction ``step`` until a live neighbour
    # (``nbr[j] >= 0``) is found; return ``i`` if the whole ring is dead. One function with an
    # explicit direction rather than a ``+1`` and a ``-1`` copy that differed only in their wrap
    # guard -- the fan optimizer removes slots as it goes, so both directions have to skip holes
    # and the skipping is the whole body.
    #
    # This is called from `edge_removal_weight` inside `build_local_triangulations`'s O(m) x O(m)
    # greedy removal loop, so an adversarial removal order (long runs of dead slots) makes one call
    # O(m) and the whole loop O(m^3) against the O(m^2) shape it otherwise implies -- an
    # incrementally-maintained doubly-linked `prev`/`next` pair would cap it at O(1)/O(m^2).
    # Measured instead of assumed
    # (`benchmarks/test_reconstruction.py::test_triangulate_point_cloud`'s own cloud,
    # `num_neighbours` swept 8/16/32/64 at fixed point count, RTX 5090, device time only): the
    # kernel's own cost scales 3.5-4.05x per doubling of `m`, matching the O(m^2) shape the loop
    # nominally has, not the O(m^3) worst case -- so on this ordinary point cloud, the walk is not
    # hitting long dead-slot runs and the doubly-linked rewrite would buy little. `m` is capped at
    # `MAX_NEIGHBOURS` regardless, bounding the absolute worst case. Declined; re-measure before
    # reopening this if a future caller's removal pattern looks different (e.g. a very anisotropic
    # or highly clustered cloud).
    j = i
    for _ in range(m):
        j = j + step
        if j >= m:
            j = 0
        elif j < 0:
            j = m - 1
        if nbr[j] >= 0:
            return j
    return i


@wp.func
def cycle_next(nbr: wp.array[wp.int32], m: wp.int32, i: wp.int32) -> wp.int32:
    return cycle_step(nbr, m, i, wp.int32(1))


@wp.func
def cycle_prev(nbr: wp.array[wp.int32], m: wp.int32, i: wp.int32) -> wp.int32:
    return cycle_step(nbr, m, i, wp.int32(-1))


# --------------------------------------------------------------------------------------
# Fan edge-removal weight: the priority with which a fan edge is dropped
# --------------------------------------------------------------------------------------
@wp.func
def edge_removal_weight(
    center: wp.int32,
    i: wp.int32,
    m: wp.int32,
    border: wp.int32,
    crit_angle: wp.float32,
    normalizer_sq: wp.float32,
    nbr: wp.array[wp.int32],
    ang: wp.array[wp.float32],
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
) -> wp.vec2:
    # Returns (weight, stable) with stable encoded as result[1] > 0.5.
    stable = wp.vec2(0.0, 1.0)
    prev = cycle_prev(nbr, m, i)
    nxt = cycle_next(nbr, m, i)

    a = points[center]
    n_center = normals[center]

    # --- boundary-edge handling ---
    if border >= 0 and (nbr[i] == border or nbr[prev] == border):
        next_el = nbr[prev] == border
        if next_el:
            prev_ind = i
            next_ind = nxt
            other_id = nxt
        else:
            prev_ind = prev
            next_ind = i
            other_id = prev
        length_sq = wp.length_sq(a - points[nbr[i]])
        other_length_sq = wp.length_sq(a - points[nbr[other_id]])
        if length_sq < other_length_sq:
            return stable
        bb = points[nbr[prev_ind]]
        cc = points[nbr[next_ind]]
        if triangle_aspect_ratio(a, bb, cc) <= CRITICAL_ASPECT_RATIO:
            return stable
        return wp.vec2(FLOAT32_INF_CONSTANT, 0.0)

    # --- interior-edge handling ---
    dif_angle = ang[nxt] - ang[prev]
    if dif_angle < 0.0:
        dif_angle += TWO_PI
    if dif_angle > PI:
        return stable  # removing this edge would leave an angle > pi

    av = center
    bv = nbr[nxt]
    cv = nbr[i]
    dv = nbr[prev]
    b = points[bv]
    c = points[cv]
    d = points[dv]
    ab = b - a
    ac = c - a
    ad = d - a

    ac_length_sq = wp.length_sq(ac)
    if (
        ac_length_sq > wp.length_sq(ab) and triangle_aspect_ratio(a, b, c) > CRITICAL_ASPECT_RATIO
    ) or (
        ac_length_sq > wp.length_sq(ad) and triangle_aspect_ratio(a, c, d) > CRITICAL_ASPECT_RATIO
    ):
        # degenerate triangle, longest edge -> remove as fast as possible
        return wp.vec2(FLOAT32_INF_CONSTANT, 0.0)

    # flip possibility: trusted normals allow a flip across a fold, else require convexity
    flip = False
    if wp.dot(n_center, normals[cv]) < 0.0:
        flip = True
    else:
        flip = is_unfold_quadrangle_convex(a, b, c, d)
    if not flip:
        return stable

    delone_prof = delone_flip_profit_sq(a, b, c, d)
    if delone_prof == 0.0 and wp.min(av, cv) > wp.min(bv, dv):
        delone_prof = -1.0
    angle_prof = tris_angle_profit(a, b, c, d, crit_angle)
    if delone_prof < 0.0 and angle_prof <= 0.0:
        return stable

    weight = 0.0
    if delone_prof > 0.0:
        weight += delone_prof / normalizer_sq
    if angle_prof > 0.0:
        weight += angle_prof

    # ``ac_length_sq`` is already the squared norm of ``ac`` (computed above for the aspect-ratio
    # guard), so its square root is ``norm_val`` -- no need for a second length computation.
    norm_val = wp.sqrt(ac_length_sq)
    if norm_val == 0.0:
        return wp.vec2(FLOAT32_INF_CONSTANT, 0.0)
    plane_dist = wp.abs(wp.dot(n_center, ac))
    weight += plane_dist / norm_val

    # trusted-normal agreement bonuses
    c_norm = normals[cv]
    weight += 5.0 * (1.0 - wp.dot(n_center, c_norm))
    tri_norm = wp.normalize(wp.cross(ab, ac) + wp.cross(ac, ad))
    tri_norm_weight = wp.dot(tri_norm, c_norm)
    if tri_norm_weight < 0.0:
        return wp.vec2(FLOAT32_INF_CONSTANT, 0.0)
    weight += 5.0 * (1.0 - tri_norm_weight)

    return wp.vec2(weight, 0.0)


# --------------------------------------------------------------------------------------
# Local fan triangulation: a Boissonnat-style fan about each point, greedily optimized
# --------------------------------------------------------------------------------------
@wp.kernel(enable_backward=False)
def build_local_triangulations(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    neighbor_idx: wp.array2d[wp.int32],
    neighbor_dist: wp.array2d[wp.float32],
    radius: wp.float32,
    crit_angle: wp.float32,
    boundary_angle: wp.float32,
    out_tris: wp.array3d[wp.int32],
    out_valid: wp.array2d[wp.bool],
) -> None:
    v = wp.int32(wp.tid())
    k = neighbor_idx.shape[1]
    cap = out_tris.shape[1]

    a = points[v]
    n_center = normals[v]

    # Both rows stay ``wp.zeros`` rather than becoming ``wp.types.vector(length=MAX_NEIGHBOURS)``
    # register rows: every access below is a *runtime* index (the gather cursor, the selection
    # sort's ``mn``, the fan scan's ``i``), which spills a vector to local memory anyway, and
    # ``edge_removal_weight`` / ``cycle_prev`` / ``cycle_next`` take them as ``wp.array``, so the
    # vector form would additionally need ``wp.ref`` variants of all three. Measured on the clean
    # case of the same class — ``algorithms/ball_pivoting.seed_triangles``, one 64-wide row with no
    # helper passing — the register row is 0.994x (min) / 1.000x (median) end to end, i.e. no gain
    # to trade that complexity for.
    #
    # And ``wp.fixedarray`` is not a third option: its own docstring says it is "only used during
    # codegen, and for type hints" -- it *is* the codegen type of a kernel-scope ``wp.zeros``, not a
    # separate storage class. So the choice here is registers or the stack, and both are measured.
    nbr = wp.zeros(shape=MAX_NEIGHBOURS, dtype=wp.int32)
    ang = wp.zeros(shape=MAX_NEIGHBOURS, dtype=wp.float32)

    # --- gather + filter neighbours ---
    # A constructor call -- wp.int32(...) / wp.float32(...) -- declares a mutable Warp dynamic
    # variable; a bare literal is a compile-time constant that freezes the enclosing loop
    # (out_valid stays all-False -> no faces). The constructor is what matters, not which spelling
    # of it: the older int()/float() forms are the same builtins under a different name.
    m = wp.int32(0)
    for i in range(k):
        if m >= MAX_NEIGHBOURS:
            # Silent truncation, not a check: this file cannot itself raise (CLAUDE.md section 1.4)
            # if the caller's ``max_neighbours`` violated the module-level invariant. The wrapper's
            # ``max_neighbours > MAX_NEIGHBOURS`` guard is what makes this branch unreachable at
            # the public entry point; a direct launch of this kernel bypasses it.
            break
        nb = neighbor_idx[v, i]
        if nb < 0 or nb == v:
            continue
        if radius > 0.0 and neighbor_dist[v, i] > radius:
            continue
        pn = points[nb]
        if pn[0] == a[0] and pn[1] == a[1] and pn[2] == a[2]:
            continue  # coincident with center
        if wp.dot(n_center, normals[nb]) < NORMAL_FILTER_DOT:
            continue
        nbr[m] = nb
        m += 1

    if m < 2:
        return

    # --- tangent-plane basis (project neighbours onto plane through center) ---
    base = wp.vec3(0.0, 0.0, 0.0)
    normalizer_sq = wp.float32(0.0)
    for i in range(m):
        d = points[nbr[i]] - a
        pv = project_out_normal(d, n_center)
        if normalizer_sq <= 0.0:
            base = pv
            normalizer_sq = wp.length_sq(pv)
    if normalizer_sq <= 0.0:
        normalizer_sq = 1.0
    base = wp.normalize(base)  # zero-length base normalizes to the zero vector (Warp kEps == 0)

    # --- polar angle of each neighbour around the center in the tangent plane ---
    for i in range(m):
        d = points[nbr[i]] - a
        pv = project_out_normal(d, n_center)
        if wp.length_sq(pv) > 0.0:
            vec = wp.normalize(pv)
        else:
            vec = base
        cp = wp.cross(vec, base)
        # wp.sign is -1 below zero and +1 otherwise, matching the guard this replaces.
        s = wp.sign(wp.dot(cp, n_center))
        ang[i] = wp.atan2(s * wp.length(cp), wp.dot(vec, base))

    # --- sort neighbours by angle (selection sort; m <= MAX_NEIGHBOURS) ---
    for i in range(m):
        mn = i
        for j in range(i + 1, m):
            if ang[j] < ang[mn]:
                mn = j
        if mn != i:
            ta = ang[i]
            ang[i] = ang[mn]
            ang[mn] = ta
            tn = nbr[i]
            nbr[i] = nbr[mn]
            nbr[mn] = tn

    # --- boundary detection: angular gaps wider than boundary_angle ---
    border = wp.int32(-1)
    gap_count = wp.int32(0)
    for i in range(m):
        if i + 1 < m:
            diff = ang[i + 1] - ang[i]
        else:
            diff = ang[0] + TWO_PI - ang[i]
        if diff > boundary_angle:
            gap_count += 1
            if border < 0:
                border = nbr[i]
    if gap_count > 1:
        # Two or more disjoint gaps: this point sits between separate regions of the cloud (a thin
        # bridge/peninsula, or a point equidistant from two disjoint holes). `border` can only ever
        # mark one gap, so the others would read as ordinary interior edges below, and a spurious
        # triangle could bridge the two disconnected regions across an unregistered gap. Rather than
        # guess which one to keep, this point's fan contributes no triangle this round -- a real fix
        # needs the fan to carry more than one boundary marker (or split into multiple arcs), which
        # is a redesign of the boundary representation this function does not attempt. Leaving a
        # point untriangulated is an accepted outcome elsewhere in the reconstruction pipeline (a
        # confirmed triangle needs two of its three vertices' fans to agree, so no single point's
        # fan is required to succeed), and `crit_hole_length` fills the small holes this leaves.
        return

    # --- greedy fan optimisation (linear-scan replacement of the priority queue) ---
    current = m  # inherits m's dynamic-variable type (m is already mutable)
    for _step in range(m):
        best_w = wp.float32(-FLOAT32_INF_CONSTANT)  # the constructor keeps this mutable
        best_pos = wp.int32(-1)
        for i in range(m):
            if nbr[i] < 0:
                continue
            res = edge_removal_weight(
                v, i, m, border, crit_angle, normalizer_sq, nbr, ang, points, normals
            )
            if res[1] > 0.5:
                continue  # stable, cannot remove
            update_argmax(best_w, best_pos, res[0], i)
        if best_pos < 0:
            break
        old = nbr[best_pos]
        prv = cycle_prev(nbr, m, best_pos)
        nbr[best_pos] = -1
        current -= 1
        if old == border:
            border = nbr[prv]
        if current < 2:
            for i in range(m):
                nbr[i] = -1
            break

    # --- emit fan triangles between consecutive surviving neighbours ---
    slot = wp.int32(0)
    for i in range(m):
        if nbr[i] < 0:
            continue
        if border >= 0 and nbr[i] == border:
            continue  # boundary gap: no triangle here
        nxt = cycle_next(nbr, m, i)
        if nxt == i:
            continue
        if border < 0 and nbr[i] > nbr[nxt] and cycle_prev(nbr, m, i) == nxt:
            # No boundary gap and exactly two live neighbours remain: the ring is a 2-cycle, so
            # ``prev(i) == next(i)`` and this wedge and (nxt, i)'s wedge describe the identical
            # triangle (there is no third live neighbour to make them distinct). Emit it once, from
            # the lower-indexed neighbour, rather than writing it twice.
            continue
        bidx = nbr[i]
        cidx = nbr[nxt]
        pb = points[bidx]
        pc = points[cidx]
        # orient so the face normal agrees with the (trusted) center normal
        if wp.dot(wp.cross(pb - a, pc - a), n_center) < 0.0:
            bidx, cidx = cidx, bidx
        if slot < cap:
            out_tris[v, slot, 0] = v
            out_tris[v, slot, 1] = bidx
            out_tris[v, slot, 2] = cidx
            out_valid[v, slot] = True
            slot += 1


# ======================================================================================
# Screened-Poisson dense-grid kernels (Kazhdan PoissonRecon-style, index-space h = 1)
#
# A node-centered dense grid of ``res**3`` scalar nodes is stored flat (row-major
# ``(i * res + j) * res + k``). The pipeline: (1) trilinear-splat the oriented normals into
# a vector field ``V`` with a density weight ``W``; (2) normalize ``V /= max(W, eps)``;
# (3) build the RHS ``b = -div V`` by central differences; (4) solve
# ``(L_N + point_weight * W) x = b`` matrix-free (7-point homogeneous-Neumann Laplacian plus a
# lumped screening diagonal); (5) sample ``x`` at the input points for the iso-value. The screening
# diagonal is strictly positive at occupied nodes, so the operator is SPD (no constant null space).
# Everything is index-space (grid spacing 1); the world scale is reapplied by marching cubes.
# ======================================================================================
POISSON_WEIGHT_EPS = wp.constant(wp.float32(1e-8))


@wp.func
def poisson_grid_index(i: wp.int32, j: wp.int32, k: wp.int32, res: wp.int32) -> wp.int32:
    # Row-major flat index into a res**3 node grid (cube: ny = nz = res).
    return ravel_index(i, j, k, res, res)


@wp.func
def poisson_sample_grid(
    field: wp.array[wp.float32], res: wp.int32, gx: wp.float32, gy: wp.float32, gz: wp.float32
) -> wp.float32:
    # Trilinear interpolation of ``field`` at grid coordinate (gx, gy, gz) in [0, res - 1].
    #
    # The cell decomposition is shared with ``splat_normals`` below and with
    # ``kernels/interpolation.sample_grid_trilinear`` / ``kernels/scatter.splat_grid_trilinear``.
    # What is *not* shared is the evaluation: this reads a **flat** ``res**3`` float32 buffer and
    # nests ``wp.lerp``, where the public pair reads a rank-3 array of any dtype and sums the eight
    # corner weights. The two are the same function of the same inputs and differ only in float32
    # summation order, so they are deliberately not merged: this one's arithmetic is what the
    # Poisson iso-value was measured against, and the storage differs anyway.
    base, next_corner, fractions = trilinear_cell(wp.vec3(gx, gy, gz), wp.vec3i(res, res, res))
    i0, j0, k0 = base[0], base[1], base[2]
    # The far corner comes from ``trilinear_cell`` rather than being written ``i0 + 1`` here, so a
    # ``res == 1`` grid cannot index past this flat ``res**3`` buffer. Identical to ``i0 + 1`` at
    # every ``res >= 2``, which is every resolution the Poisson solver builds.
    i1, j1, k1 = next_corner[0], next_corner[1], next_corner[2]
    fx, fy, fz = fractions[0], fractions[1], fractions[2]
    c000 = field[poisson_grid_index(i0, j0, k0, res)]
    c100 = field[poisson_grid_index(i1, j0, k0, res)]
    c010 = field[poisson_grid_index(i0, j1, k0, res)]
    c110 = field[poisson_grid_index(i1, j1, k0, res)]
    c001 = field[poisson_grid_index(i0, j0, k1, res)]
    c101 = field[poisson_grid_index(i1, j0, k1, res)]
    c011 = field[poisson_grid_index(i0, j1, k1, res)]
    c111 = field[poisson_grid_index(i1, j1, k1, res)]
    c00 = wp.lerp(c000, c100, fx)
    c10 = wp.lerp(c010, c110, fx)
    c01 = wp.lerp(c001, c101, fx)
    c11 = wp.lerp(c011, c111, fx)
    c0 = wp.lerp(c00, c10, fy)
    c1 = wp.lerp(c01, c11, fy)
    return wp.lerp(c0, c1, fz)


@wp.kernel(enable_backward=False)
def splat_normals(
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    cube_lower: wp.vec3,
    inv_cell: wp.float32,
    res: wp.int32,
    confidence: wp.int32,
    out_vx: wp.array[wp.float32],
    out_vy: wp.array[wp.float32],
    out_vz: wp.array[wp.float32],
    out_w: wp.array[wp.float32],
) -> None:
    s = wp.int32(wp.tid())
    n = normals[s]
    length = wp.length(n)
    # Confidence weighting scales the splat by the normal magnitude; otherwise unit weight.
    weight = wp.float32(1.0)
    if confidence != 0:
        weight = length
    # Unit direction, reusing ``length`` instead of ``wp.normalize``'s own redundant sqrt; a zero
    # vector's length is already zero, so it is left unchanged -- the same zero-vector result
    # ``wp.normalize`` gives that input (magnitude is carried separately by ``weight``).
    if length > 0.0:
        n = n / length

    g = (points[s] - cube_lower) * inv_cell
    base, next_corner, fractions = trilinear_cell(g, wp.vec3i(res, res, res))

    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                w = trilinear_weight(fractions, di, dj, dk) * weight
                corner = trilinear_corner(base, next_corner, wp.vec3i(di, dj, dk))
                idx = poisson_grid_index(corner[0], corner[1], corner[2], res)
                wp.atomic_add(out_vx, idx, w * n[0])
                wp.atomic_add(out_vy, idx, w * n[1])
                wp.atomic_add(out_vz, idx, w * n[2])
                wp.atomic_add(out_w, idx, w)


@wp.func
def normalized_field_component(
    field: wp.array[wp.float32], weights: wp.array[wp.float32], idx: wp.int32
) -> wp.float32:
    # One splatted vector-field component divided by its own density weight -- the per-element
    # operation `normalize_vector_field` used to do as a separate pass. Kept as a `@wp.func` (not a
    # standalone kernel) so `poisson_level_setup` can read a normalized value at each of its six
    # neighbour offsets directly, rather than reading back a value a prior launch wrote: see the
    # measurement in ``poisson_level_setup`` below.
    inv = 1.0 / wp.max(weights[idx], POISSON_WEIGHT_EPS)
    return field[idx] * inv


@wp.func
def poisson_neighbor_degree(i: wp.int32, j: wp.int32, k: wp.int32, res: wp.int32) -> wp.float32:
    # Number of (i, j, k)'s up-to-6 axis-aligned grid neighbours that lie inside [0, res) -- the
    # diagonal degree of the homogeneous-Neumann 7-point Poisson stencil. Shared so
    # ``screened_laplacian_matvec``'s operator and ``poisson_level_setup``'s Jacobi
    # preconditioner can never disagree about which grid nodes are boundary nodes.
    deg = wp.float32(0.0)
    if i + 1 < res:
        deg += 1.0
    if i - 1 >= 0:
        deg += 1.0
    if j + 1 < res:
        deg += 1.0
    if j - 1 >= 0:
        deg += 1.0
    if k + 1 < res:
        deg += 1.0
    if k - 1 >= 0:
        deg += 1.0
    return deg


@wp.kernel(enable_backward=False)
def screened_laplacian_matvec(
    x: wp.array[wp.float32],
    y: wp.array[wp.float32],
    weights: wp.array[wp.float32],
    screen: wp.float32,
    alpha: wp.float32,
    beta: wp.float32,
    res: wp.int32,
    out_z: wp.array[wp.float32],
) -> None:
    # z = alpha * (A @ x) + beta * y, with A = L_N + screen * diag(W).
    i, j, k = wp.tid()
    idx = poisson_grid_index(i, j, k, res)
    xc = x[idx]
    deg = poisson_neighbor_degree(i, j, k, res)
    # Same boundary test as ``poisson_neighbor_degree``, expanded to also gather the live
    # neighbours' values -- the six bounds checks are free next to the six global loads they guard.
    acc = wp.float32(0.0)
    if i + 1 < res:
        acc += x[poisson_grid_index(i + 1, j, k, res)]
    if i - 1 >= 0:
        acc += x[poisson_grid_index(i - 1, j, k, res)]
    if j + 1 < res:
        acc += x[poisson_grid_index(i, j + 1, k, res)]
    if j - 1 >= 0:
        acc += x[poisson_grid_index(i, j - 1, k, res)]
    if k + 1 < res:
        acc += x[poisson_grid_index(i, j, k + 1, res)]
    if k - 1 >= 0:
        acc += x[poisson_grid_index(i, j, k - 1, res)]
    ax = (deg * xc - acc) + screen * weights[idx] * xc
    out_z[idx] = alpha * ax + beta * y[idx]


@wp.kernel(enable_backward=False)
def poisson_level_setup(
    vx: wp.array[wp.float32],
    vy: wp.array[wp.float32],
    vz: wp.array[wp.float32],
    weights: wp.array[wp.float32],
    screen: wp.float32,
    res: wp.int32,
    out_b: wp.array[wp.float32],
    out_inv_diag: wp.array[wp.float32],
) -> None:
    # The two per-node passes a solve level needs, in one launch: the right-hand side and the
    # Jacobi preconditioner's inverse diagonal. They are independent -- neither reads the other --
    # and both already read ``weights`` at this node, so fusing removes a launch over the whole
    # ``res**3`` grid and shares the node index. A level is solved once per octree depth, so this
    # launch pair ran several times per call.
    i, j, k = wp.tid()
    # Central differences in index space; one-sided at the grid boundary (denominator 1 there).
    # That denominator is 0, not 1, at res == 1 (every axis's clamp collapses to ip == im), but
    # reconstruction.screened_poisson validates 3 <= full_depth <= depth <= 10, so the smallest
    # reachable grid is res = 2**3 + 1 = 9 and no caller in the tree reaches res == 1. Not guarded
    # here per CLAUDE.md section 4.2 ("no speculative generality") -- add a guard only if a future
    # caller can legitimately reach res == 1.
    ip = wp.min(i + 1, res - 1)
    im = wp.max(i - 1, 0)
    jp = wp.min(j + 1, res - 1)
    jm = wp.max(j - 1, 0)
    kp = wp.min(k + 1, res - 1)
    km = wp.max(k - 1, 0)
    # Normalized in place here rather than by a separate `normalize_vector_field` pass over the
    # whole grid: that pass measured 5.93% of screened_poisson's device time at the default depth=8
    # (RTX 5090), almost all of it the extra launch plus a full res**3-element write-back of
    # vx/vy/vz that this function immediately reads back at six neighbour offsets anyway. Fusing
    # removes one launch and that write-back, at the cost of reading `weights` at the same six
    # offsets instead of once per node -- a real trade given six values are already read here per
    # component. `normalized_field_component` reproduces the deleted pass's exact
    # `field[idx] * (1.0 / max(weight, eps))` operation (a multiply by the reciprocal, not a divide)
    # so the result is bit-identical to computing it as a separate pass first.
    dx = (
        normalized_field_component(vx, weights, poisson_grid_index(ip, j, k, res))
        - normalized_field_component(vx, weights, poisson_grid_index(im, j, k, res))
    ) / wp.float32(ip - im)
    dy = (
        normalized_field_component(vy, weights, poisson_grid_index(i, jp, k, res))
        - normalized_field_component(vy, weights, poisson_grid_index(i, jm, k, res))
    ) / wp.float32(jp - jm)
    dz = (
        normalized_field_component(vz, weights, poisson_grid_index(i, j, kp, res))
        - normalized_field_component(vz, weights, poisson_grid_index(i, j, km, res))
    ) / wp.float32(kp - km)
    centre = poisson_grid_index(i, j, k, res)
    out_b[centre] = -(dx + dy + dz)

    deg = poisson_neighbor_degree(i, j, k, res)
    d = deg + screen * weights[centre]
    if d <= 0.0:
        d = 1.0
    out_inv_diag[centre] = 1.0 / d


@wp.func
def diagonal_precond_axpby(
    x: wp.float32, y: wp.float32, inv_diag: wp.float32, alpha: wp.float32, beta: wp.float32
) -> wp.float32:
    # One element of ``z = alpha * M^-1 x + beta * y`` for the Jacobi preconditioner
    # ``M = diag(d)``. Mapped (not a kernel) so the wrapper can hoist the launch out of the
    # CG iteration; see ``reconstruction._diagonal_operator``.
    return alpha * inv_diag * x + beta * y


@wp.kernel(enable_backward=False)
def prolong_grid(
    coarse: wp.array[wp.float32], res_c: wp.int32, res_f: wp.int32, out_fine: wp.array[wp.float32]
) -> None:
    # Trilinear factor-2 prolongation: res_f - 1 == 2 * (res_c - 1), so fine node i sits at
    # coarse coordinate i / 2.
    i, j, k = wp.tid()
    val = poisson_sample_grid(
        coarse, res_c, wp.float32(i) * 0.5, wp.float32(j) * 0.5, wp.float32(k) * 0.5
    )
    out_fine[poisson_grid_index(i, j, k, res_f)] = val


@wp.kernel(enable_backward=False)
def sample_field_trilinear(
    field: wp.array[wp.float32],
    res: wp.int32,
    cube_lower: wp.vec3,
    inv_cell: wp.float32,
    points: wp.array[wp.vec3],
    out_values: wp.array[wp.float32],
) -> None:
    s = wp.int32(wp.tid())
    g = (points[s] - cube_lower) * inv_cell
    out_values[s] = poisson_sample_grid(field, res, g[0], g[1], g[2])


@wp.kernel
def lattice_points(
    resolution: wp.vec3i, origin: wp.vec3, spacing: wp.vec3, out_points: wp.array[wp.vec3]
) -> None:
    # World positions of a dense ``res_x * res_y * res_z`` node lattice, in the row-major order
    # ``wp.MarchingCubes`` expects of a ``(nx, ny, nz)`` field: ``x`` is the slowest axis. The
    # position itself is ``array.lattice_position``, shared with ``voxels.lattice_points``, which
    # writes the same quantity into a ``wp.array3d`` instead of flattening.
    i, j, k = wp.tid()
    index = (i * resolution[1] + j) * resolution[2] + k
    out_points[index] = lattice_position(origin, spacing, i, j, k)
