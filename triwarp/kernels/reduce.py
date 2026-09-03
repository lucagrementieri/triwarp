"""Global reductions: tiled ``wp.tile_load`` reductions generated from a shared template."""

import warp as wp

# Tile-reduction builtins are resolved only inside kernel source text and are not
# exposed as Python-scope attributes (``wp.tile_max`` etc. raise ``AttributeError``),
# unlike the atomic and scalar builtins (``wp.atomic_max``, ``wp.max`` …). To let the
# kernel factories below *capture* them, pull the concrete ``Function`` objects out of
# Warp's builtin registry. A captured builtin is emitted inline at codegen (same as a
# literal ``wp.tile_sum`` call), so it templates on the tile dtype correctly — wrapping
# it in a ``@wp.func`` instead would erase the dtype and produce ambiguous C++ overloads.
from warp._src.context import builtin_functions as _warp_builtins

from triwarp.constants import TILE_1D, TILE_2D, TILES_PER_BLOCK_1D

_tile_min = _warp_builtins["tile_min"]
_tile_max = _warp_builtins["tile_max"]
_tile_sum = _warp_builtins["tile_sum"]


@wp.func
def tile_chunk(n: wp.int32, chunk: wp.int32, width: wp.int32) -> tuple[wp.int32, wp.int32]:
    # The ``(offset, remaining)`` of one chunk of a 1-D array: where chunk ``chunk`` of ``width``
    # elements starts, and how many elements are left from there. ``remaining <= 0`` means the
    # chunk is past the end and the block has nothing to do; a positive ``remaining`` below
    # ``width`` is the ragged last chunk, which the caller clamps with ``wp.min(remaining, width)``.
    #
    # Named because the *contract* is what goes wrong here, not the arithmetic: ``width`` is
    # ``TILE_1D`` for a one-tile-per-block kernel and ``TILES_PER_BLOCK_1D * TILE_1D`` for the
    # folding ones in this module, and a launch that assumes the wrong one silently folds every
    # element 16 times -- see ``blocks_1d``, which is the same hazard from the launch side.
    offset = chunk * width
    return offset, n - offset


# Elements one block of a 1-D global reduction owns. It is a ``wp.constant`` because both halves
# of the contract read it and they must read the *same* number: [`blocks_1d`] derives the launch
# width from it at Python scope, and a kernel-scope ``tile_chunk(n, chunk, ITEMS_PER_BLOCK_1D)``
# derives each block's slice from it. A launch and a kernel that disagree here do not fail -- they
# fold some elements twice and drop others, which for a sum is a wrong answer and for an extremum
# looks right. See [`blocks_1d`] for the same hazard stated from the launch side.
ITEMS_PER_BLOCK_1D = wp.constant(TILE_1D * TILES_PER_BLOCK_1D)


def blocks_1d(n: int) -> int:
    """
    Launch width for the 1-D global reduction kernels in this module.

    Every ``*1d_tiled`` kernel here folds ``TILES_PER_BLOCK_1D`` tiles per block, so its grid is
    ``n / (TILE_1D * TILES_PER_BLOCK_1D)`` and **not** ``n / TILE_1D``. Call this rather than
    open-coding the division: passing the tile count would give each block the same 16 tiles the
    next 15 blocks also claim, folding every element 16 times over -- and, the fold being
    idempotent for the extrema, that would leave ``min`` / ``max`` / ``any`` / ``all`` looking
    correct while ``sum`` silently returned 16x its answer.

    It is also the launch width for the *lane-strided* reductions outside this module --
    ``registration.transform_and_accumulate_cost`` / ``accumulate_procrustes_moments`` /
    ``accumulate_point_to_plane``, ``points.centered_covariance``,
    ``polyline.accumulate_newell_normal`` / ``accumulate_turning_angle`` /
    ``accumulate_loop_frame`` -- which own the same
    [`ITEMS_PER_BLOCK_1D`][triwarp.kernels.reduce.ITEMS_PER_BLOCK_1D] chunk per block but partition
    it across lanes with ``wp.block_dim()`` rather than loading tiles from it.

    Parameters
    ----------
    n
        Number of elements in the 1-D array being reduced.

    Returns
    -------
    int
        Block count to pass as ``dim`` to ``wp.launch_tiled`` with ``block_dim=TILE_1D``.
    """
    items = ITEMS_PER_BLOCK_1D
    return (n + items - 1) // items


# ---------------------------------------------------------------------------
# Kernel factories.
#
# Each builds one ``@wp.kernel`` capturing a ``(tile_reduce, atomic, scalar)``
# triple. The full-tile branch loads a tile and reduces it with ``tile_reduce``;
# the partial-tile remainder folds elements with ``scalar``; the result is
# committed with ``atomic``. A unique ``name`` per instantiation is REQUIRED:
# Warp keys generated kernels by name, so reusing one name would alias distinct
# reductions to the same compiled kernel.
# ---------------------------------------------------------------------------


def _reduce_1d_tiled(tile_reduce, atomic, scalar, name, dtype=wp.Scalar):
    """
    axis=None on a 1-D array: ``TILES_PER_BLOCK_1D`` tiles per block, one atomic per block.

    The accumulator is seeded from the block's *first* chunk rather than from an identity, which
    keeps the kernel generic over ``wp.Scalar`` without the wrapper having to pass a per-dtype
    identity value in. Every subsequent chunk folds in with ``scalar``.

    ``dtype`` widens the template past ``wp.Scalar`` — ``wp.vec3`` is the one in use, for the vec3
    sum, whose ``wp.add`` fold and ``wp.atomic_add`` commit work component-wise. It is a *codegen*
    parameter, so each instantiation is a concrete kernel: annotating ``wp.array[Any]`` and letting
    one kernel serve both dtypes also works, but measured ~18 us per launch of host-side overload
    resolution (0.59x at 10 000 elements, 0.94x at 10M), so the generic form is declined here.
    """

    def _k(values: wp.array[wp.Scalar], out: wp.array[wp.Scalar]) -> None:
        i, t = wp.tid()
        n = values.shape[0]
        base, remaining = tile_chunk(n, i, TILES_PER_BLOCK_1D * TILE_1D)
        if remaining <= 0:
            return

        # First chunk seeds the accumulator (both branches assign it -- see CLAUDE.md section 5 on
        # Warp's conditional scoping).
        if remaining >= TILE_1D:
            tile = wp.tile_load(values, shape=TILE_1D, offset=base, storage="register")
            result = tile_reduce(tile)[0]
        else:
            result = values[base]
            for k in range(1, remaining):
                result = scalar(result, values[base + k])

        for s in range(1, TILES_PER_BLOCK_1D):
            offset = base + s * TILE_1D
            rest = n - offset
            if rest >= TILE_1D:
                chunk = wp.tile_load(values, shape=TILE_1D, offset=offset, storage="register")
                result = scalar(result, tile_reduce(chunk)[0])
            elif rest > 0:
                for k in range(rest):
                    result = scalar(result, values[offset + k])

        if t == 0:
            atomic(out, 0, result)

    _k.__name__ = name
    _k.__qualname__ = name
    _k.__annotations__["values"] = wp.array[dtype]
    _k.__annotations__["out"] = wp.array[dtype]
    return wp.kernel(_k)


def _weighted_sum_1d_tiled(name, dtype):
    """
    Weighted 1-D sum ``sum_k values[k] * weights[k]``, in ``_reduce_1d_tiled``'s block shape.

    Its own factory rather than an instantiation of the one above because the second array makes
    the tile path a *pair* of loads and an element-wise product, which the single-array template
    cannot express; everything else -- the ``TILES_PER_BLOCK_1D`` fold, the seed-from-first-chunk
    accumulator, the one atomic per block -- is the same skeleton and the same launch contract
    ([`blocks_1d`][triwarp.kernels.reduce.blocks_1d], never the tile count).

    ``dtype`` is the *value* dtype (``wp.float32`` or ``wp.vec3``); weights are always
    ``wp.float32``, and a ``wp.vec3`` tile times a ``wp.float32`` tile scales component-wise. The
    accumulator is seeded from the first weighted element for the same reason as above: neither a
    ``wp.float32(0.0)`` nor a ``wp.vec3(0.0)`` literal spells both instantiations.
    """

    def _k(
        values: wp.array[wp.Scalar], weights: wp.array[wp.float32], out_sum: wp.array[wp.Scalar]
    ) -> None:
        i, t = wp.tid()
        n = values.shape[0]
        base, remaining = tile_chunk(n, i, TILES_PER_BLOCK_1D * TILE_1D)
        if remaining <= 0:
            return

        if remaining >= TILE_1D:
            v_tile = wp.tile_load(values, shape=TILE_1D, offset=base, storage="register")
            w_tile = wp.tile_load(weights, shape=TILE_1D, offset=base, storage="register")
            result = wp.tile_sum(v_tile * w_tile)[0]
        else:
            result = values[base] * weights[base]
            for k in range(1, remaining):
                result = result + values[base + k] * weights[base + k]

        for s in range(1, TILES_PER_BLOCK_1D):
            offset = base + s * TILE_1D
            rest = n - offset
            if rest >= TILE_1D:
                v_chunk = wp.tile_load(values, shape=TILE_1D, offset=offset, storage="register")
                w_chunk = wp.tile_load(weights, shape=TILE_1D, offset=offset, storage="register")
                result = result + wp.tile_sum(v_chunk * w_chunk)[0]
            elif rest > 0:
                for k in range(rest):
                    result = result + values[offset + k] * weights[offset + k]

        if t == 0:
            wp.atomic_add(out_sum, 0, result)

    _k.__name__ = name
    _k.__qualname__ = name
    _k.__annotations__["values"] = wp.array[dtype]
    _k.__annotations__["out_sum"] = wp.array[dtype]
    return wp.kernel(_k)


def _reduce_2d_tiled(tile_reduce, atomic, scalar, name):
    """axis=None on a 2-D array: one 2-D tile per block, atomically fold into slot 0."""

    def _k(values: wp.array2d[wp.Scalar], out: wp.array[wp.Scalar]) -> None:
        i, j, t = wp.tid()
        n_rows = values.shape[0]
        n_cols = values.shape[1]
        row_offset = i * TILE_2D
        col_offset = j * TILE_2D
        if row_offset >= n_rows or col_offset >= n_cols:
            return

        remaining_rows = n_rows - row_offset
        remaining_cols = n_cols - col_offset
        tile_rows = TILE_2D if remaining_rows >= TILE_2D else remaining_rows
        tile_cols = TILE_2D if remaining_cols >= TILE_2D else remaining_cols
        if remaining_rows >= TILE_2D and remaining_cols >= TILE_2D:
            tile = wp.tile_load(
                values,
                shape=(TILE_2D, TILE_2D),
                offset=(row_offset, col_offset),
                storage="register",
            )
            result = tile_reduce(tile)[0]
        else:
            result = values[row_offset, col_offset]
            for r in range(tile_rows):
                for c in range(tile_cols):
                    if r == 0 and c == 0:
                        continue
                    result = scalar(result, values[row_offset + r, col_offset + c])

        if t == 0:
            atomic(out, 0, result)

    _k.__name__ = name
    _k.__qualname__ = name
    return wp.kernel(_k)


# Element access for a per-axis reduction, as two ``@wp.func``s a factory *captures* rather than as
# a branch every factory repeats: ``i`` is always the output slot and ``k`` always the position
# along the reduced extent, so each loop below is written once and reads the same for both axes.
# Two things still name the axis directly -- which ``shape`` the extent comes from, and the tile's
# shape -- and both are ``wp.static(rows)``, so nothing is decided at runtime.
#
# The axis is a **codegen** parameter, never a kernel argument: Warp indexes ``array2d`` row-major,
# so the two axes have genuinely different access patterns and a runtime ``axis`` selector would
# compile one of them into a strided walk regardless. Each instantiation is therefore exactly the
# specialised kernel it was hand-written as -- one source of truth, two compiled kernels. Verified
# as such: against the four hand-written rows/cols kernels this replaced, output arrays are equal
# element-for-element and the timings are flat (0.99-1.01x on an RTX 5090 over a (20k, 512) table
# and a (4M, 3) one, both axes, tiled and serial).


@wp.func
def _element_along_row(values: wp.array2d[wp.Scalar], i: wp.int32, k: wp.int32):
    # axis=1: slot ``i`` is a row, ``k`` walks its columns.
    return values[i, k]


@wp.func
def _element_along_col(values: wp.array2d[wp.Scalar], i: wp.int32, k: wp.int32):
    # axis=0: slot ``i`` is a column, ``k`` walks its rows.
    return values[k, i]


def _reduce_2d_axis_tiled(tile_reduce, atomic, scalar, name, rows):
    """
    axis=1 (``rows=True``) or axis=0: one output slot per grid index, tiled along its extent.

    Each block folds one ``TILE_1D`` tile of slot ``i``'s extent and commits it with ``atomic``, so
    the wrapper must pre-fill the output with the reduction's identity.
    """
    element = _element_along_row if rows else _element_along_col

    def _k(values: wp.array2d[wp.Scalar], out: wp.array[wp.Scalar]) -> None:
        i, j, t = wp.tid()
        if wp.static(rows):
            extent = values.shape[1]
        else:
            extent = values.shape[0]
        offset = j * TILE_1D
        if offset >= extent:
            return
        remaining = extent - offset
        if remaining >= TILE_1D:
            # A tile is a shape, so this cannot go through ``element``: the reduced extent has to
            # be the tile's long side for the load to be contiguous.
            if wp.static(rows):
                tile = wp.tile_load(
                    values, shape=(1, TILE_1D), offset=(i, offset), storage="register"
                )
            else:
                tile = wp.tile_load(
                    values, shape=(TILE_1D, 1), offset=(offset, i), storage="register"
                )
            result = tile_reduce(tile)[0]
        else:
            result = element(values, i, offset)
            for k in range(1, remaining):
                result = scalar(result, element(values, i, offset + k))
        if t == 0:
            atomic(out, i, result)

    _k.__name__ = name
    _k.__qualname__ = name
    return wp.kernel(_k)


# When the reduced extent is narrower than ``TILE_1D`` the tiled kernels above never take their
# ``tile_load`` branch — all ``TILE_1D`` lanes of every block redundantly run the serial remainder
# loop, a ``TILE_1D``-fold read amplification (measured 6.6 ms -> 0.13 ms for ``max(axis=1)`` on a
# ``(14M, 3)`` table, 49x). These serial variants launch one plain thread per *output* element and
# write directly: no tiles, no atomics, and no init fill needed on the output buffer. The wrapper
# picks them whenever ``reduced extent < TILE_1D``; past that the tiled kernels stay (a tall
# ``(n, 3)`` table reduced along axis=0 has only 3 outputs — 3 serial threads would be 89x slower).


def _reduce_2d_axis_serial(scalar, name, rows):
    """
    axis=1 (``rows=True``) or axis=0 with a reduced extent under ``TILE_1D``: one thread per output.

    Writes ``out`` directly, so unlike the tiled form it needs no identity pre-fill. The
    ``rows=False`` instantiation walks a column, whose consecutive threads read consecutive
    addresses -- coalesced, where the ``rows=True`` one is strided by the row length.
    """
    element = _element_along_row if rows else _element_along_col

    def _k(values: wp.array2d[wp.Scalar], out: wp.array[wp.Scalar]) -> None:
        i = wp.tid()
        if wp.static(rows):
            extent = values.shape[1]
        else:
            extent = values.shape[0]
        result = element(values, i, 0)
        for k in range(1, extent):
            result = scalar(result, element(values, i, k))
        out[i] = result

    _k.__name__ = name
    _k.__qualname__ = name
    return wp.kernel(_k)


# ---------------------------------------------------------------------------
# Scalar reductions (min / max / sum) over ``wp.Scalar`` arrays.
# ---------------------------------------------------------------------------

min1d_tiled = _reduce_1d_tiled(_tile_min, wp.atomic_min, wp.min, "min1d_tiled")
min2d_tiled = _reduce_2d_tiled(_tile_min, wp.atomic_min, wp.min, "min2d_tiled")
min_2d_rows_tiled = _reduce_2d_axis_tiled(
    _tile_min, wp.atomic_min, wp.min, "min_2d_rows_tiled", rows=True
)
min_2d_cols_tiled = _reduce_2d_axis_tiled(
    _tile_min, wp.atomic_min, wp.min, "min_2d_cols_tiled", rows=False
)
min_2d_rows_serial = _reduce_2d_axis_serial(wp.min, "min_2d_rows_serial", rows=True)
min_2d_cols_serial = _reduce_2d_axis_serial(wp.min, "min_2d_cols_serial", rows=False)

max1d_tiled = _reduce_1d_tiled(_tile_max, wp.atomic_max, wp.max, "max1d_tiled")
max2d_tiled = _reduce_2d_tiled(_tile_max, wp.atomic_max, wp.max, "max2d_tiled")
max_2d_rows_tiled = _reduce_2d_axis_tiled(
    _tile_max, wp.atomic_max, wp.max, "max_2d_rows_tiled", rows=True
)
max_2d_cols_tiled = _reduce_2d_axis_tiled(
    _tile_max, wp.atomic_max, wp.max, "max_2d_cols_tiled", rows=False
)
max_2d_rows_serial = _reduce_2d_axis_serial(wp.max, "max_2d_rows_serial", rows=True)
max_2d_cols_serial = _reduce_2d_axis_serial(wp.max, "max_2d_cols_serial", rows=False)

sum1d_tiled = _reduce_1d_tiled(_tile_sum, wp.atomic_add, wp.add, "sum1d_tiled")
sum2d_tiled = _reduce_2d_tiled(_tile_sum, wp.atomic_add, wp.add, "sum2d_tiled")
sum_2d_rows_tiled = _reduce_2d_axis_tiled(
    _tile_sum, wp.atomic_add, wp.add, "sum_2d_rows_tiled", rows=True
)
sum_2d_cols_tiled = _reduce_2d_axis_tiled(
    _tile_sum, wp.atomic_add, wp.add, "sum_2d_cols_tiled", rows=False
)
sum_2d_rows_serial = _reduce_2d_axis_serial(wp.add, "sum_2d_rows_serial", rows=True)
sum_2d_cols_serial = _reduce_2d_axis_serial(wp.add, "sum_2d_cols_serial", rows=False)

# ---------------------------------------------------------------------------
# Boolean reductions over int32 0/1 masks. ``any`` == OR == max; ``all`` == AND
# == min (a 0/1 mask minimises to 1 iff every element is 1). The 2-D global case
# is handled by the wrapper flattening the mask and reusing the 1-D kernel, so no
# ``*2d_tiled`` bool kernel is needed.
# ---------------------------------------------------------------------------

any_1d_tiled = _reduce_1d_tiled(_tile_max, wp.atomic_max, wp.max, "any_1d_tiled")
any_2d_rows_tiled = _reduce_2d_axis_tiled(
    _tile_max, wp.atomic_max, wp.max, "any_2d_rows_tiled", rows=True
)
any_2d_cols_tiled = _reduce_2d_axis_tiled(
    _tile_max, wp.atomic_max, wp.max, "any_2d_cols_tiled", rows=False
)
any_2d_rows_serial = _reduce_2d_axis_serial(wp.max, "any_2d_rows_serial", rows=True)
any_2d_cols_serial = _reduce_2d_axis_serial(wp.max, "any_2d_cols_serial", rows=False)

all_1d_tiled = _reduce_1d_tiled(_tile_min, wp.atomic_min, wp.min, "all_1d_tiled")
all_2d_rows_tiled = _reduce_2d_axis_tiled(
    _tile_min, wp.atomic_min, wp.min, "all_2d_rows_tiled", rows=True
)
all_2d_cols_tiled = _reduce_2d_axis_tiled(
    _tile_min, wp.atomic_min, wp.min, "all_2d_cols_tiled", rows=False
)
all_2d_rows_serial = _reduce_2d_axis_serial(wp.min, "all_2d_rows_serial", rows=True)
all_2d_cols_serial = _reduce_2d_axis_serial(wp.min, "all_2d_cols_serial", rows=False)


# ---------------------------------------------------------------------------
# Min/max together. The dual tile-reduction, dual atomic and two output slots do
# not fit the single-primitive factory template, so these get their own — the
# axis-parameterized pair below — and the two axis=None forms stay hand-written,
# there being one instantiation of each to generate.
# ---------------------------------------------------------------------------


@wp.kernel
def minmax1d_tiled(values: wp.array[wp.Scalar], out_minmax: wp.array[wp.Scalar]) -> None:
    # Same TILES_PER_BLOCK_1D fold as the factory kernels above, with two accumulators seeded from
    # the block's first chunk; two atomics per block instead of two per tile.
    i, t = wp.tid()
    n = values.shape[0]
    base, remaining = tile_chunk(n, i, TILES_PER_BLOCK_1D * TILE_1D)
    if remaining <= 0:
        return

    if remaining >= TILE_1D:
        tile = wp.tile_load(values, shape=TILE_1D, offset=base, storage="register")
        tile_min = wp.tile_min(tile)[0]
        tile_max = wp.tile_max(tile)[0]
    else:
        tile_min = values[base]
        tile_max = values[base]
        for k in range(1, remaining):
            v = values[base + k]
            tile_min = wp.min(tile_min, v)
            tile_max = wp.max(tile_max, v)

    for s in range(1, TILES_PER_BLOCK_1D):
        offset = base + s * TILE_1D
        rest = n - offset
        if rest >= TILE_1D:
            chunk = wp.tile_load(values, shape=TILE_1D, offset=offset, storage="register")
            tile_min = wp.min(tile_min, wp.tile_min(chunk)[0])
            tile_max = wp.max(tile_max, wp.tile_max(chunk)[0])
        elif rest > 0:
            for k in range(rest):
                v = values[offset + k]
                tile_min = wp.min(tile_min, v)
                tile_max = wp.max(tile_max, v)

    if t == 0:
        wp.atomic_min(out_minmax, 0, tile_min)
        wp.atomic_max(out_minmax, 1, tile_max)


@wp.kernel
def minmax2d_tiled(values: wp.array2d[wp.Scalar], out_minmax: wp.array[wp.Scalar]) -> None:
    i, j, t = wp.tid()
    n_rows = values.shape[0]
    n_cols = values.shape[1]
    row_offset = i * TILE_2D
    col_offset = j * TILE_2D
    if row_offset >= n_rows or col_offset >= n_cols:
        return

    remaining_rows = n_rows - row_offset
    remaining_cols = n_cols - col_offset
    tile_rows = wp.where(remaining_rows >= TILE_2D, TILE_2D, remaining_rows)
    tile_cols = wp.where(remaining_cols >= TILE_2D, TILE_2D, remaining_cols)
    if remaining_rows >= TILE_2D and remaining_cols >= TILE_2D:
        tile = wp.tile_load(
            values, shape=(TILE_2D, TILE_2D), offset=(row_offset, col_offset), storage="register"
        )
        tile_min = wp.tile_min(tile)[0]
        tile_max = wp.tile_max(tile)[0]
    else:
        tile_min = values[row_offset, col_offset]
        tile_max = values[row_offset, col_offset]
        for r in range(tile_rows):
            for c in range(tile_cols):
                if r == 0 and c == 0:
                    continue
                v = values[row_offset + r, col_offset + c]
                tile_min = wp.min(tile_min, v)
                tile_max = wp.max(tile_max, v)

    if t == 0:
        wp.atomic_min(out_minmax, 0, tile_min)
        wp.atomic_max(out_minmax, 1, tile_max)


def _minmax_2d_axis_tiled(name, rows):
    """axis=1 (``rows=True``) or axis=0: ``_reduce_2d_axis_tiled`` with both extrema at once."""
    element = _element_along_row if rows else _element_along_col

    def _k(
        values: wp.array2d[wp.Scalar], out_min: wp.array[wp.Scalar], out_max: wp.array[wp.Scalar]
    ) -> None:
        i, j, t = wp.tid()
        if wp.static(rows):
            extent = values.shape[1]
        else:
            extent = values.shape[0]
        offset = j * TILE_1D
        if offset >= extent:
            return
        remaining = extent - offset
        if remaining >= TILE_1D:
            if wp.static(rows):
                tile = wp.tile_load(
                    values, shape=(1, TILE_1D), offset=(i, offset), storage="register"
                )
            else:
                tile = wp.tile_load(
                    values, shape=(TILE_1D, 1), offset=(offset, i), storage="register"
                )
            tile_min = wp.tile_min(tile)[0]
            tile_max = wp.tile_max(tile)[0]
        else:
            tile_min = element(values, i, offset)
            tile_max = tile_min
            for k in range(1, remaining):
                v = element(values, i, offset + k)
                tile_min = wp.min(tile_min, v)
                tile_max = wp.max(tile_max, v)
        if t == 0:
            wp.atomic_min(out_min, i, tile_min)
            wp.atomic_max(out_max, i, tile_max)

    _k.__name__ = name
    _k.__qualname__ = name
    return wp.kernel(_k)


def _minmax_2d_axis_serial(name, rows):
    """
    axis=1 (``rows=True``) or axis=0 with a reduced extent under ``TILE_1D``: one thread per output.

    The tiled form above loses ``TILE_1D``-fold there for the reason given on
    [`_reduce_2d_axis_serial`][triwarp.kernels.reduce._reduce_2d_axis_serial].
    """
    element = _element_along_row if rows else _element_along_col

    def _k(
        values: wp.array2d[wp.Scalar], out_min: wp.array[wp.Scalar], out_max: wp.array[wp.Scalar]
    ) -> None:
        i = wp.tid()
        if wp.static(rows):
            extent = values.shape[1]
        else:
            extent = values.shape[0]
        slot_min = element(values, i, 0)
        slot_max = slot_min
        for k in range(1, extent):
            v = element(values, i, k)
            slot_min = wp.min(slot_min, v)
            slot_max = wp.max(slot_max, v)
        out_min[i] = slot_min
        out_max[i] = slot_max

    _k.__name__ = name
    _k.__qualname__ = name
    return wp.kernel(_k)


minmax_2d_rows_tiled = _minmax_2d_axis_tiled("minmax_2d_rows_tiled", rows=True)
minmax_2d_cols_tiled = _minmax_2d_axis_tiled("minmax_2d_cols_tiled", rows=False)
minmax_2d_rows_serial = _minmax_2d_axis_serial("minmax_2d_rows_serial", rows=True)
minmax_2d_cols_serial = _minmax_2d_axis_serial("minmax_2d_cols_serial", rows=False)


# ---------------------------------------------------------------------------
# Sums whose dtype the ``wp.Scalar`` template above does not admit: ``wp.vec3``
# values, and the weighted forms whose second array the single-array template
# cannot express. All three come from the same two factories as the scalar
# reductions, so the file has one way of generating a 1-D reduction.
#
# Folding ``TILES_PER_BLOCK_1D`` tiles per block is what these three gained by
# moving onto the template. Measured against one tile per block, on an RTX 5090:
# weighted ``float32`` 1.25x at 1M elements and 3.34x at 10M, weighted ``vec3``
# 1.95x and 4.14x, the unweighted ``vec3`` sum 2.08x and 5.43x. Below ~200k the
# fold has too few blocks to fill *that* device and costs ~10 us (0.66-0.81x at
# 10k) -- the trade the scalar reductions have made unconditionally since they
# were written. On CPU there is no such crossover: one lane per block means the
# fold is strictly fewer blocks for the same work, and it wins 1.27-1.55x
# (weighted ``float32``), 1.67-2.06x (weighted ``vec3``) and 1.50-1.96x (the
# ``vec3`` sum) at every size from 10k to 10M.
# ---------------------------------------------------------------------------

sum_vec3_1d_tiled = _reduce_1d_tiled(
    _tile_sum, wp.atomic_add, wp.add, "sum_vec3_1d_tiled", dtype=wp.vec3
)

weighted_sum1d_tiled = _weighted_sum_1d_tiled("weighted_sum1d_tiled", wp.float32)
weighted_sum_vec3_1d_tiled = _weighted_sum_1d_tiled("weighted_sum_vec3_1d_tiled", wp.vec3)


@wp.func
def outer_sum_chunk(
    points: wp.array[wp.vec3],
    center: wp.vec3,
    offset: wp.int32,
    remaining: wp.int32,
    lane: wp.int32,
    stride: wp.int32,
) -> wp.mat33:
    # ``_chunk``, not ``_tile``: this walks its share of the block's chunk with a plain loop and
    # uses no tile primitive, so it returns one lane's *partial* matrix and the caller is
    # responsible for reducing across lanes. The name says so rather than promising a cooperative
    # reduction that is not here.
    #
    # ``lane`` / ``stride`` are passed rather than read from ``wp.block_dim()`` because this is a
    # ``@wp.func``: the caller is the kernel that knows its own launch shape.
    count = wp.min(remaining, ITEMS_PER_BLOCK_1D)
    # M = sum_k outer(x_k, x_k) where x_k = points[k] - center
    m = wp.mat33(0.0)
    for k in range(lane, count, stride):
        x = points[offset + k] - center
        m += wp.outer(x, x)
    return m


@wp.kernel
def minmax_vec3_chunked(points: wp.array[wp.vec3], out_corners: wp.array[wp.float32]) -> None:
    # Component-wise min and max of a ``wp.vec3`` array, in one launch into one buffer.
    #
    # ``out_corners`` is ``[min_x, min_y, min_z, -max_x, -max_y, -max_z]``, *negating* the upper
    # half so a single ``wp.full(6, inf)`` initializes both and every update is an
    # ``atomic_min``. The alternative — separate min and max buffers — needs two allocations, two
    # fills and two readbacks, and at this size the reduction is entirely host-latency-bound.
    #
    # One thread per ``TILE_1D`` points, so the atomics see a few hundred contenders per address
    # rather than one per point.
    offset, remaining = tile_chunk(points.shape[0], wp.int32(wp.tid()), TILE_1D)
    if remaining <= 0:
        return
    count = wp.min(remaining, TILE_1D)

    lower = points[offset]
    upper = points[offset]
    for k in range(1, count):
        p = points[offset + k]
        lower = wp.min(lower, p)  # wp.min / wp.max on a vector are component-wise
        upper = wp.max(upper, p)

    for c in range(3):
        wp.atomic_min(out_corners, c, lower[c])
        wp.atomic_min(out_corners, 3 + c, -upper[c])


# ---------------------------------------------------------------------------
# Concrete overloads, registered at import.
#
# Every kernel above annotated ``wp.Scalar`` is *generic*, and Warp instantiates
# an overload lazily -- on the first launch at each new dtype. A module's hash
# covers the set of instantiated overloads, so that first launch changes the hash
# and recompiles **every** kernel in this file. Measured on an RTX 5090, Warp
# 1.16: 14.2 s per fork, 10.4 s of it nvcc, and reducing ``int32`` axis=0, then
# ``int32`` axis=1, then ``float32`` axis=0 paid it three times over -- 40+
# kernels rebuilt to gain one. Worse, the chain is *order-dependent*: a caller
# reaching the dtypes in a different order walks links that were never compiled,
# so the cost came back on every change of test selection (``tests/test_reduce.py``
# alone: 1 561 s on a fresh selection against 1.30 s repeating it).
#
# Registering every (kernel, dtype) pair the wrapper's dispatch can reach gives
# the module one hash for its whole lifetime: one compile ever, then one cached
# load per process. Registration is not compilation -- ``wp.overload`` builds the
# overload's ``Adjoint`` and nothing else -- so this costs milliseconds of import
# and nothing at all on a process that never reduces anything. Measured over the
# whole suite: 66 distinct ``(hash, block_dim)`` loads of this module before,
# **3** after (the two block_dim variants plus CPU's), and ``pytest`` end to end
# 1 033 s -> 190 s with ``tests/test_reduce.py`` itself 535 s -> 2.0 s.
#
# The trade is honest about one thing: the *single* compile is now bigger, since
# the module holds ~110 concrete kernels rather than the ~40 a lazy fork built.
# It measured 80 s of nvcc per block_dim variant, paid once and then cached
# (a warm cached load is 59-75 ms). That is the cost of editing this file, not of
# using it -- and against 66 forks of 14.2 s it is not close. If it ever does
# become the bottleneck, the escape is to give each generated kernel its own
# module with ``@wp.kernel(module="unique")``, the way ``warp.sparse`` does, so
# a rebuild touches one kernel instead of all of them.
#
# Two rules for keeping it that way:
#
# - **A new generic kernel in this file must be added to a group below, and a new
#   dtype to the right tuple.** ``test_generic_kernels_register_their_overloads``
#   catches the first; nothing catches the second, because a missing dtype does not
#   fail, it just re-forks the chain on its first launch. The symptom is a test or
#   a script that suddenly takes tens of seconds -- read it as a rebuild and come
#   back here (CLAUDE.md section 13).
# - **The dtype set is the one ``triwarp.reduce`` dispatches over**, not every
#   dtype ``wp.Scalar`` admits (CLAUDE.md section 14, no speculative generality):
#   an unused overload is compile time paid on every rebuild. The boolean
#   reductions are ``wp.int32`` only because ``_reduce_bool`` converts the mask
#   before launching, and ``wp.bool`` is not a ``wp.Scalar`` in any case.
#
# ``block_dim`` forks the hash independently of the dtypes and is deliberately
# left forked. It is not the same pathology: the values in use are fixed by *this
# package's* launch code -- ``TILE_1D`` for the ``wp.launch_tiled`` reductions and
# Warp's 256 default for the plain ones, plus 1 on CPU, where Warp pins it -- so
# they are a bounded set of two or three variants, not a chain whose length grows
# with what a caller happens to reduce first. Collapsing them by passing
# ``block_dim=TILE_1D`` at the plain-launch sites was measured and declined: it
# costs 0.60x on ``max(axis=1)`` over a ``(4M, 3)`` table and 0.67x on
# ``max(axis=0)`` over ``(3, 4M)``, buying only ``minmax_vec3_chunked`` at 100k
# (1.63x, 14 us) and nothing at all by 14M (0.97x) -- an RTX 5090, min of 20
# interleaved reps.
# ---------------------------------------------------------------------------

# A **global** reduction takes whatever scalar dtype the caller's buffer carries, and the package
# itself hands it two families: geometry and solver values (``wp.float32``, ``wp.float64`` in the
# heat and smoothing solvers) and *keys* --
# [`isin`][triwarp.array.isin] and [`hash_indices_rows`][triwarp.grouping.hash_indices_rows] bound
# an index range with ``reduce.minmax`` over the caller's key dtype, whose public surface is
# ``wp.int32`` / ``wp.int64`` / ``wp.uint32`` / ``wp.uint64`` (``isin`` widens anything narrower
# with ``sortable_dtype`` before reducing, so sub-32-bit dtypes never reach a kernel here).
_GLOBAL_DTYPES = (wp.int32, wp.int64, wp.uint32, wp.uint64, wp.float32, wp.float64)

# A **per-axis** reduction is only ever reached with an index or a geometry dtype: nothing in the
# package reduces a table of 64-bit keys along an axis, and the six-dtype cross product over these
# 24 kernels would be 72 more kernels to compile on every rebuild for no call site (CLAUDE.md
# section 14). A caller who does reduce a ``wp.uint64`` table along an axis pays one fork, once.
_AXIS_DTYPES = (wp.int32, wp.float32, wp.float64)

# Grouped by signature shape, which is what ``wp.overload`` matches on; the operator each kernel
# folds with does not enter into it.
_GLOBAL_1D = (min1d_tiled, max1d_tiled, sum1d_tiled, minmax1d_tiled)
_GLOBAL_2D = (min2d_tiled, max2d_tiled, sum2d_tiled, minmax2d_tiled)
_AXIS_SINGLE_OUT = (
    min_2d_rows_tiled,
    min_2d_cols_tiled,
    min_2d_rows_serial,
    min_2d_cols_serial,
    max_2d_rows_tiled,
    max_2d_cols_tiled,
    max_2d_rows_serial,
    max_2d_cols_serial,
    sum_2d_rows_tiled,
    sum_2d_cols_tiled,
    sum_2d_rows_serial,
    sum_2d_cols_serial,
)
_AXIS_DUAL_OUT = (
    minmax_2d_rows_tiled,
    minmax_2d_cols_tiled,
    minmax_2d_rows_serial,
    minmax_2d_cols_serial,
)
# The boolean reductions reach these kernels only through ``_reduce_bool``, which casts the mask to
# a 0/1 ``wp.int32`` first, so one dtype covers them -- and ``wp.bool`` is not a ``wp.Scalar``
# anyway.
_MASK_1D = (any_1d_tiled, all_1d_tiled)
_MASK_2D = (
    any_2d_rows_tiled,
    any_2d_cols_tiled,
    any_2d_rows_serial,
    any_2d_cols_serial,
    all_2d_rows_tiled,
    all_2d_cols_tiled,
    all_2d_rows_serial,
    all_2d_cols_serial,
)


def _register_overloads() -> None:
    """Instantiate every concrete overload of this module's generic kernels."""
    for dtype in _GLOBAL_DTYPES:
        for kernel in _GLOBAL_1D:
            wp.overload(kernel, [wp.array[dtype], wp.array[dtype]])
        for kernel in _GLOBAL_2D:
            wp.overload(kernel, [wp.array2d[dtype], wp.array[dtype]])
    for dtype in _AXIS_DTYPES:
        for kernel in _AXIS_SINGLE_OUT:
            wp.overload(kernel, [wp.array2d[dtype], wp.array[dtype]])
        for kernel in _AXIS_DUAL_OUT:
            wp.overload(kernel, [wp.array2d[dtype], wp.array[dtype], wp.array[dtype]])
    for kernel in _MASK_1D:
        wp.overload(kernel, [wp.array[wp.int32], wp.array[wp.int32]])
    for kernel in _MASK_2D:
        wp.overload(kernel, [wp.array2d[wp.int32], wp.array[wp.int32]])


_register_overloads()
