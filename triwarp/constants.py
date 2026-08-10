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

# Tiles folded per block by the *global* (``axis=None``) 1-D reductions in ``kernels/reduce.py``.
# One block per tile means one ``atomic_add`` per 64 elements, and at scale that single accumulator
# address is the bottleneck rather than bandwidth: 219k blocks contending on one slot cost 309 us to
# sum 14M float32 (56 MB), against a ~31 us bandwidth floor on this card. Folding several tiles into
# a register first divides the atomic traffic by this factor.
#
# Swept 1/4/16/64/256 at 36k, 438k, 1.09M and 14M elements, interleaved under one clock state:
#
# - 16 is 1.28x / 1.50x / 1.90x / **4.89x** over the one-tile form and never loses;
# - 4 is better below ~1M (1.58x / 1.70x) but only 3.33x at 14M;
# - 64 matches 16 at 14M and loses 1.4x below it; 256 loses everywhere (the tail block does too much
#   serial work while the rest of the device idles).
#
# 16 wins where the difference is visible: below ~1M the whole reduction sits under the ~82 us of
# host-side launch + readback that ends any scalar-returning call, so the 5 us that 4 would save
# there is unobservable, while the 30 us it gives up at 14M is not.
#
# The CPU device pays for it, and the ratio is recorded here rather than left to be rediscovered:
# ``wp.launch_tiled`` runs one lane per block there (through Warp 1.16), so folding 16 tiles means
# 16x fewer blocks and correspondingly less parallelism -- measured **1.28x slower at 36k, 1.07x at
# 438k, 1.02x at 14M**.
# Accepted on the CUDA number per CLAUDE.md section 13: the loss is bounded, shrinks with size, and
# is at its worst exactly where the host floor already hides it.
TILES_PER_BLOCK_1D = 16

# Elements reduced per thread by the *lane-free* reductions -- those whose body must stay correct on
# the CPU device, where ``wp.launch_tiled`` runs exactly one lane per block (through Warp 1.16) and
# any lane-parallel body silently reduces a single element per tile. Each thread walks a strided
# slice of its input and commits one atomic, so this trades launch width against atomic traffic.
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
