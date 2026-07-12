"""Shared numeric tolerances (import from here to avoid circular imports)."""

import warp as wp

TOLERANCE_MERGE = 1e-8
TOLERANCE_PLANAR = 1e-5
TOLERANCE_ZERO = 1e-12

TOLERANCE_MERGE_CONSTANT = wp.constant(wp.float32(TOLERANCE_MERGE))
TOLERANCE_PLANAR_CONSTANT = wp.constant(wp.float32(TOLERANCE_PLANAR))
TOLERANCE_ZERO_CONSTANT = wp.constant(wp.float32(TOLERANCE_ZERO))

# Largest representable values, usable as "sorts last" sentinels inside kernels (computing them
# here in Python scope avoids the literals Warp cannot evaluate at kernel scope).
INT32_MAX_CONSTANT = wp.constant(wp.int32(2**31 - 1))
INT64_MAX = 2**63 - 1
INT64_MAX_CONSTANT = wp.constant(wp.int64(INT64_MAX))
FLOAT32_INF_CONSTANT = wp.constant(wp.INF)

PI = wp.constant(wp.PI)
TWO_PI = wp.constant(2 * wp.PI)

TILE_1D = 64
TILE_2D = 8
