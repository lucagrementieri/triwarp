import warp as wp

from triwarp.constants import TILE_1D
from triwarp.kernels.predicates import triangle_double_area
from triwarp.kernels.triangles import face_vertices, face_vertices_vec3d


@wp.kernel
def centroid_tiled(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    n_faces: wp.int32,
    out_weighted_centroid: wp.array[wp.float32],
    out_total_area: wp.array[wp.float32],
) -> None:
    # CUDA path, launched via wp.launch_tiled with block_dim=TILE_1D: each lane computes one face's
    # area-weighted centroid contribution, the block reduces each component cooperatively with
    # wp.tile_sum, and lane 0 commits four atomics per block (out-of-range lanes contribute zero).
    # CPU MUST NOT use this: the lanes partition the **outer** work (the face list), so the stride
    # is the constant `TILE_1D` and not `wp.block_dim()` -- and `wp.launch_tiled` runs one lane per
    # block on the CPU device through Warp 1.17, where that lane would see one face per tile. This
    # is the constant-stride case of the rule in `.claude/CLAUDE.md` section 2.2, and it is why this
    # reduction keeps a device pair where `kernels/visibility.py::obscurance` needs only one
    # kernel. See `centroid_sliced` and `_device.prefers_tiled_reduction`.
    #
    # **Rewriting it into the lane-strided `ITEMS_PER_BLOCK_1D` fold was measured and declined**,
    # and the number is worth keeping: the same rewrite is worth 2.8-10.7x on the `registration`
    # and `points` accumulators (`.claude/CLAUDE.md` section 13.2). Built as
    # `tile_chunk(n_faces, chunk, ITEMS_PER_BLOCK_1D)` plus a `wp.block_dim()` stride -- which would
    # also make it portable and retire `centroid_sliced` -- it measures **0.98-1.01x** at 1 280 /
    # 20 480 / 81 920 / 327 680 faces, the areas agreeing to 1.5e-07: flat everywhere. The fold pays
    # on *atomic contention*, and contention here is `blocks x slots`: 5 120 blocks x 4 slots is
    # ~2e4 atomics at 327k faces, where the accumulators that won were at ~1e5 (3 125 blocks x 25 or
    # 43 slots at 200k-1M points). Below that the launch floor hides it. With no CUDA win to pay for
    # it the portability is not free either -- `blocks_1d(n)` would give the CPU path `n / 1024`
    # single-lane blocks against `slice_count`'s `n / 32` threads -- so the device pair stays.
    #
    # The area comes from ``triangle_double_area`` over the corners already in registers, not from
    # ``face_normals_and_area``: that helper re-enters ``face_vertices`` through ``triangle_cross``,
    # so the pair loaded all three corners twice, and it also normalizes a face normal this kernel
    # then binds to ``_`` and discards. The expression is the same one either way
    # (``0.5 * |cross(v1 - v0, v2 - v0)|``), so the areas are bit-identical.
    i, t = wp.tid()
    f = i * TILE_1D + t
    contrib = wp.vec3(0.0, 0.0, 0.0)
    area = wp.float32(0.0)
    if f < n_faces:
        v0, v1, v2 = face_vertices(vertices, faces, f)
        area = 0.5 * triangle_double_area(v0, v1, v2)
        contrib = (v0 + v1 + v2) * (area / 3.0)
    sum_x = wp.tile_sum(wp.tile(contrib[0]))
    sum_y = wp.tile_sum(wp.tile(contrib[1]))
    sum_z = wp.tile_sum(wp.tile(contrib[2]))
    area_sum = wp.tile_sum(wp.tile(area))
    if t == 0:
        wp.tile_atomic_add(out_weighted_centroid, sum_x, (0,))
        wp.tile_atomic_add(out_weighted_centroid, sum_y, (1,))
        wp.tile_atomic_add(out_weighted_centroid, sum_z, (2,))
        wp.tile_atomic_add(out_total_area, area_sum, (0,))


@wp.kernel
def centroid_sliced(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    n_faces: wp.int32,
    n_slices: wp.int32,
    out_weighted_centroid: wp.array[wp.float32],
    out_total_area: wp.array[wp.float32],
) -> None:
    # Portable path: one thread per face slice walks a strided slice, accumulates locally and
    # commits four atomics. Lane-free, so it is correct on the CPU device where `centroid_tiled`
    # is not; it gives up the block shuffle-reduce and measures 1.67x slower on CUDA at 327k faces
    # (16.0 -> 26.7 us), which is why both exist.
    #
    # The bug this sibling exists to prevent, since the sentence naming it was lost in an edit and
    # only its second half survived: running `centroid_tiled` on the CPU device silently averaged
    # **every TILE_1D-th face** rather than every face, because `wp.launch_tiled` gives that device
    # one lane per block, so `t` is always 0 and `f = i * TILE_1D + t` skips the rest of each tile.
    # It was invisible on a symmetric mesh, whose every-Nth-face centroid is still the true
    # centroid.
    #
    # Same corner-loading note as `centroid_tiled` above: the area is taken from the corners this
    # loop already holds.
    j = wp.int32(wp.tid())
    total = wp.vec3(0.0, 0.0, 0.0)
    area_total = wp.float32(0.0)
    for f in range(j, n_faces, n_slices):
        v0, v1, v2 = face_vertices(vertices, faces, f)
        area = 0.5 * triangle_double_area(v0, v1, v2)
        total = total + (v0 + v1 + v2) * (area / 3.0)
        area_total = area_total + area
    wp.atomic_add(out_weighted_centroid, 0, total[0])
    wp.atomic_add(out_weighted_centroid, 1, total[1])
    wp.atomic_add(out_weighted_centroid, 2, total[2])
    wp.atomic_add(out_total_area, 0, area_total)


@wp.kernel
def moment_integrands(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    out_volume: wp.array[wp.float64],
    out_first: wp.array[wp.vec3d],
    out_squares: wp.array[wp.vec3d],
    out_products: wp.array[wp.vec3d],
) -> None:
    # Per-face contribution to the mass integrals of the solid bounded by the mesh, taking each
    # face with the origin as a tetrahedron. Everything accumulates in float64: the second moments
    # scale as length^5, so a float32 sum over a large mesh loses the answer's low digits well
    # before the reduction finishes.
    #
    # For the tetrahedron (0, a, b, c) with det = dot(a, cross(b, c)):
    #   ∫dV      = det / 6
    #   ∫x dV    = det * (a.x + b.x + c.x) / 24
    #   ∫x^2 dV  = det * (a.x^2 + b.x^2 + c.x^2 + a.x b.x + a.x c.x + b.x c.x) / 60
    #   ∫xy dV   = det * (2(a.x a.y + b.x b.y + c.x c.y)
    #                     + a.x b.y + b.x a.y + a.x c.y + c.x a.y + b.x c.y + c.x b.y) / 120
    f = wp.int32(wp.tid())
    a, b, c = face_vertices_vec3d(vertices, faces, f)
    det = wp.dot(a, wp.cross(b, c))

    out_volume[f] = det / wp.float64(6.0)
    out_first[f] = det * (a + b + c) / wp.float64(24.0)
    out_squares[f] = (
        det
        * wp.vec3d(
            a[0] * a[0] + b[0] * b[0] + c[0] * c[0] + a[0] * b[0] + a[0] * c[0] + b[0] * c[0],
            a[1] * a[1] + b[1] * b[1] + c[1] * c[1] + a[1] * b[1] + a[1] * c[1] + b[1] * c[1],
            a[2] * a[2] + b[2] * b[2] + c[2] * c[2] + a[2] * b[2] + a[2] * c[2] + b[2] * c[2],
        )
        / wp.float64(60.0)
    )
    # (xy, xz, yz), in the same order the wrapper reads them back.
    out_products[f] = (
        det
        * wp.vec3d(
            wp.float64(2.0) * (a[0] * a[1] + b[0] * b[1] + c[0] * c[1])
            + a[0] * b[1]
            + b[0] * a[1]
            + a[0] * c[1]
            + c[0] * a[1]
            + b[0] * c[1]
            + c[0] * b[1],
            wp.float64(2.0) * (a[0] * a[2] + b[0] * b[2] + c[0] * c[2])
            + a[0] * b[2]
            + b[0] * a[2]
            + a[0] * c[2]
            + c[0] * a[2]
            + b[0] * c[2]
            + c[0] * b[2],
            wp.float64(2.0) * (a[1] * a[2] + b[1] * b[2] + c[1] * c[2])
            + a[1] * b[2]
            + b[1] * a[2]
            + a[1] * c[2]
            + c[1] * a[2]
            + b[1] * c[2]
            + c[1] * b[2],
        )
        / wp.float64(120.0)
    )
