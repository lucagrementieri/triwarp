"""Shared numeric tolerances (import from here to avoid circular imports)."""

import warp as wp

TOLERANCE_MERGE = 1e-8
TOLERANCE_PLANAR = 1e-5
TOLERANCE_ZERO = 1e-12

# Intrinsic mollification margin, as a fraction of the longest edge: the triangle inequality is
# satisfied with this much slack rather than exactly, so a barely-valid triangle still yields a
# finite cotangent weight. Shared by `laplacian.mollify_intrinsic` and `remesh.intrinsic_delaunay`.
TOLERANCE_MOLLIFY = 1e-5

TOLERANCE_MERGE_CONSTANT = wp.constant(wp.float32(TOLERANCE_MERGE))
TOLERANCE_PLANAR_CONSTANT = wp.constant(wp.float32(TOLERANCE_PLANAR))
TOLERANCE_ZERO_CONSTANT = wp.constant(wp.float32(TOLERANCE_ZERO))

# Largest representable values, usable as "sorts last" sentinels inside kernels (computing them
# here in Python scope avoids the literals Warp cannot evaluate at kernel scope).
INT32_MAX = 2**31 - 1
INT32_MAX_CONSTANT = wp.constant(wp.int32(INT32_MAX))
INT64_MAX = 2**63 - 1
INT64_MAX_CONSTANT = wp.constant(wp.int64(INT64_MAX))
FLOAT32_INF_CONSTANT = wp.constant(wp.INF)

PI = wp.constant(wp.PI)
TWO_PI = wp.constant(2 * wp.PI)

TILE_1D = 64
TILE_2D = 8

# Elements reduced per thread by the *lane-free* reductions -- those that cannot use ``wp.tile``
# because they must also be correct on the CPU device, where ``wp.launch_tiled`` runs exactly one
# lane per block (Warp 1.15) and any lane-parallel body silently reduces a single element per tile.
# Each thread walks a strided slice of its input and commits one atomic, so this trades launch width
# against atomic traffic.
#
# 32 is for reductions landing in ONE or a few accumulators, where the slice count is the only
# source of parallelism. Measured over 1.2k-1M elements (area-weighted centroid, chamfer loss, hull
# support sweep), the optimum walks from 8 to 64 across that range and 32 is within ~7% of it
# everywhere, while 128 costs up to 2.4x on the small end. A per-query reduction wants a much
# longer slice -- the query dimension already fills the device -- and sets its own value locally.
ITEMS_PER_SLICE = 32
