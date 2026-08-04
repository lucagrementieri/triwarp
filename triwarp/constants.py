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

# Elements reduced per thread by the *lane-free* reductions -- those whose body must stay correct on
# the CPU device, where ``wp.launch_tiled`` runs exactly one lane per block (Warp 1.15) and any
# lane-parallel body silently reduces a single element per tile. Each thread walks a strided slice
# of its input and commits one atomic, so this trades launch width against atomic traffic.
#
# The optimum splits by device, so there are two values and
# [`items_per_slice`][triwarp._device.items_per_slice] picks between them; do not read either
# directly. Swept over 8-256 on a 5k and a 200k point cloud (hull support extremes, the only CUDA
# consumer left after the tiled reductions were restored) plus the CPU-only centroid and chamfer
# reductions:
#
# - CUDA wants long slices: at 200k points 32 costs 1.54x of the 256 optimum, while 128 is within 1%
#   of it and within 4% at 5k points, where the whole sweep is flat.
# - CPU wants short ones: 128 costs 1.43x of the 32 optimum on the 5k cloud, and the CPU sweep is
#   otherwise flat (32 within 1.11x of best everywhere measured).
#
# A *per-query* reduction wants a much longer slice on both devices -- the query dimension already
# fills the device -- and sets its own value locally (see ``proximity.ITEMS_PER_QUERY_SLICE``).
ITEMS_PER_SLICE_CUDA = 128
ITEMS_PER_SLICE_CPU = 32
