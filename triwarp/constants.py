"""Shared numeric tolerances (import from here to avoid circular imports)."""

import warp as wp

TOLERANCE_MERGE = 1e-8
TOLERANCE_ZERO = 1e-12

TOLERANCE_MERGE_CONSTANT = wp.constant(wp.float32(TOLERANCE_MERGE))
TOLERANCE_ZERO_CONSTANT = wp.constant(wp.float32(TOLERANCE_ZERO))

TILE_1D = 64
TILE_2D = 8
