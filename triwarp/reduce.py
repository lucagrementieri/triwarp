"""Global reductions on Warp arrays."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal, NamedTuple, cast, overload

import warp as wp

import triwarp.typing as twt
from triwarp._device import read_scalar
from triwarp.array import _sorted_copy, astype
from triwarp.constants import TILE_1D, TILE_2D
from triwarp.kernels import reduce as kernel_reduce


@overload
def min(array: twt.Array1dInt32 | twt.Array2dInt32, *, axis: None = ...) -> int: ...
@overload
def min(array: twt.Array1dFloat | twt.Array2dFloat, *, axis: None = ...) -> float: ...
@overload
def min(array: twt.Array2dScalar, *, axis: Literal[0, 1]) -> twt.Array1dScalar: ...
def min(
    array: twt.ScalarArray, *, axis: Literal[0, 1] | None = None
) -> float | int | twt.Array1dScalar:
    """
    Minimum of ``array``.

    With ``axis=None`` (default), reduces every element to one Python scalar using
    tiled kernels (``wp.tile_load`` + ``wp.tile_min`` + ``wp.atomic_min``).

    With ``axis=0`` or ``axis=1`` on a rank-2 input, reduces along that axis to a
    1D ``wp.array`` of the same dtype — one plain thread per output element when the
    reduced extent is narrower than a tile, a tiled block reduction per output otherwise.

    Parameters
    ----------
    array
        Rank-1 ``(n,)`` or rank-2 ``(n, m)`` scalar Warp array. Must be non-empty.
    axis
        ``None`` for a global scalar result. ``0`` or ``1`` for a per-axis 1D
        result (rank-2 input only).

    Returns
    -------
    float | int | wp.array
        Global scalar when ``axis=None``; 1D array of length ``n`` (``axis=1``) or
        ``m`` (``axis=0``) otherwise.

    Raises
    ------
    ValueError
        If ``array`` is empty, its rank is not 1 or 2, or ``axis`` is not
        ``None`` for a rank-1 input.
    """
    return cast(float | int | twt.Array1dScalar, _reduce_scalar(array, axis, _SCALAR_REDUCE["min"]))


@overload
def max(array: twt.Array1dInt32 | twt.Array2dInt32, *, axis: None = ...) -> int: ...
@overload
def max(array: twt.Array1dFloat | twt.Array2dFloat, *, axis: None = ...) -> float: ...
@overload
def max(array: twt.Array2dScalar, *, axis: Literal[0, 1]) -> twt.Array1dScalar: ...
def max(
    array: twt.ScalarArray, *, axis: Literal[0, 1] | None = None
) -> float | int | twt.Array1dScalar:
    """
    Maximum of ``array``.

    With ``axis=None`` (default), reduces every element to one Python scalar using
    tiled kernels (``wp.tile_load`` + ``wp.tile_max`` + ``wp.atomic_max``).

    With ``axis=0`` or ``axis=1`` on a rank-2 input, reduces along that axis to a
    1D ``wp.array`` of the same dtype — one plain thread per output element when the
    reduced extent is narrower than a tile, a tiled block reduction per output otherwise.

    Parameters
    ----------
    array
        Rank-1 ``(n,)`` or rank-2 ``(n, m)`` scalar Warp array. Must be non-empty.
    axis
        ``None`` for a global scalar result. ``0`` or ``1`` for a per-axis 1D
        result (rank-2 input only).

    Returns
    -------
    float | int | wp.array
        Global scalar when ``axis=None``; 1D array of length ``n`` (``axis=1``) or
        ``m`` (``axis=0``) otherwise.

    Raises
    ------
    ValueError
        If ``array`` is empty, its rank is not 1 or 2, or ``axis`` is not
        ``None`` for a rank-1 input.
    """
    return cast(float | int | twt.Array1dScalar, _reduce_scalar(array, axis, _SCALAR_REDUCE["max"]))


@overload
def minmax(array: twt.Array1dInt32 | twt.Array2dInt32, *, axis: None = ...) -> tuple[int, int]: ...
@overload
def minmax(
    array: twt.Array1dFloat | twt.Array2dFloat, *, axis: None = ...
) -> tuple[float, float]: ...
@overload
def minmax(array: wp.array[wp.vec3], *, axis: None = ...) -> tuple[wp.vec3, wp.vec3]: ...
@overload
def minmax(
    array: twt.Array2dScalar, *, axis: Literal[0, 1]
) -> tuple[twt.Array1dScalar, twt.Array1dScalar]: ...
def minmax(
    array: twt.ScalarArray | wp.array[wp.vec3], *, axis: Literal[0, 1] | None = None
) -> (
    tuple[float, float]
    | tuple[int, int]
    | tuple[wp.vec3, wp.vec3]
    | tuple[twt.Array1dScalar, twt.Array1dScalar]
):
    """
    Minimum and maximum of ``array``.

    With ``axis=None`` (default), reduces every element to two Python scalars using
    tiled kernels (``wp.tile_load`` + ``wp.tile_min``/``tile_max`` +
    ``wp.atomic_min``/``atomic_max``). Either direction alone would be one launch, one
    buffer and one readback too, so asking for both costs nothing extra.

    A rank-1 ``wp.vec3`` array reduces component-wise to a ``(wp.vec3, wp.vec3)`` corner
    pair — one chunked kernel into a single six-slot buffer (the upper corner negated so
    one ``inf`` fill seeds both ends) and one readback, which is what keeps
    [`aabb`][triwarp.bounds.aabb] host-latency-bound and nothing more.

    With ``axis=0`` or ``axis=1`` on a rank-2 input, reduces along that axis to a
    pair of 1D ``wp.array`` buffers (min, max) of the same dtype.

    Parameters
    ----------
    array
        Rank-1 ``(n,)`` or rank-2 ``(n, m)`` scalar Warp array, or a rank-1 ``wp.vec3``
        array. Must be non-empty.
    axis
        ``None`` for a global scalar result. ``0`` or ``1`` for per-axis 1D
        results (rank-2 scalar input only).

    Returns
    -------
    tuple[float, float] | tuple[int, int] | tuple[wp.vec3, wp.vec3] | tuple[wp.array, wp.array]
        ``(min, max)`` as Python scalars — or ``wp.vec3`` corners for a ``wp.vec3`` input —
        when ``axis=None``; pair of 1D arrays of length ``n`` (``axis=1``) or ``m``
        (``axis=0``) otherwise.

    Raises
    ------
    ValueError
        If ``array`` is empty, its rank is not 1 or 2, ``axis`` is not ``None`` for a
        rank-1 or ``wp.vec3`` input.

    See Also
    --------
    [`aabb`][triwarp.bounds.aabb]
        The mesh-facing spelling of the ``wp.vec3`` reduction, with the empty-input
        ``(+inf, -inf)`` convention instead of a raise.
    """
    if array.dtype == wp.vec3:
        if axis is not None:
            raise ValueError("minmax requires axis=None for a wp.vec3 array.")
        return _launch_global_vec3_minmax(cast("wp.array[wp.vec3]", array))
    return cast(
        tuple[float, float] | tuple[int, int] | tuple[twt.Array1dScalar, twt.Array1dScalar],
        _reduce_scalar(cast(twt.ScalarArray, array), axis, _SCALAR_REDUCE["minmax"]),
    )


def any(array: wp.array[wp.bool], *, axis: Literal[0, 1] | None = None) -> wp.array[wp.bool] | bool:
    """
    Reduce a boolean array with logical OR.

    For rank-1 input, or rank-2 input with ``axis=None``, reduce all elements
    to one Python ``bool``. For rank-2 input with an explicit axis, reduce along
    that axis to a 1D ``wp.bool`` array.

    Parameters
    ----------
    array
        Rank-1 ``(n,)`` or rank-2 ``(n, m)`` ``wp.bool`` array. Must be non-empty.
    axis
        ``0``, ``1``, or ``None``. For rank-2 input, ``None`` reduces all elements
        to a single scalar. Ignored for rank-1 input.

    Returns
    -------
    bool | wp.array[wp.bool]
        Python ``bool`` for rank-1 input or rank-2 with ``axis=None``.
        Length ``array.shape[1]`` when ``axis=0``, else ``array.shape[0]``,
        for rank-2 input with an explicit axis.

    Raises
    ------
    ValueError
        If ``array`` is empty, its rank is not 1 or 2, or ``axis`` is not
        ``0``, ``1``, or ``None`` for rank-2 input.
    """
    return _reduce_bool(array, axis, _BOOL_REDUCE["any"])


@overload
def sum(array: wp.array[wp.vec3], *, axis: None = ...) -> wp.vec3: ...
@overload
def sum(array: twt.Array1dInt32 | twt.Array2dInt32, *, axis: None = ...) -> int: ...
@overload
def sum(array: twt.Array1dFloat | twt.Array2dFloat, *, axis: None = ...) -> float: ...
@overload
def sum(array: twt.Array2dScalar, *, axis: Literal[0, 1]) -> twt.Array1dScalar: ...
@overload
def sum(array: wp.array[wp.bool], *, axis: None = ...) -> int: ...
@overload
def sum(array: wp.array[wp.bool], *, axis: Literal[0, 1]) -> twt.Array1dInt32: ...
def sum(
    array: twt.ScalarArray | wp.array[wp.bool] | wp.array[wp.vec3],
    *,
    axis: Literal[0, 1] | None = None,
) -> float | int | twt.Array1dScalar | twt.Array1dInt32 | wp.vec3:
    """
    Sum of ``array``.

    With ``axis=None`` (default), reduces every element to one Python scalar using
    tiled kernels (``wp.tile_load`` + ``wp.tile_sum`` + ``wp.atomic_add``).

    With ``axis=0`` or ``axis=1`` on a rank-2 input, reduces along that axis to a
    1D ``wp.array`` of the same dtype — one plain thread per output element when the
    reduced extent is narrower than a tile, a tiled block reduction per output otherwise.

    For ``wp.bool`` input, counts ``True`` values and returns ``int`` (global) or
    ``wp.int32`` (per-axis).

    Parameters
    ----------
    array
        Rank-1 ``(n,)`` or rank-2 ``(n, m)`` scalar or ``wp.bool`` Warp array.
        Must be non-empty.
    axis
        ``None`` for a global scalar result. ``0`` or ``1`` for a per-axis 1D
        result (rank-2 input only).

    Returns
    -------
    float | int | wp.array
        Global scalar when ``axis=None``; 1D array of length ``n`` (``axis=1``) or
        ``m`` (``axis=0``) otherwise.

    Raises
    ------
    ValueError
        If ``array`` is empty, its rank is not 1 or 2, or ``axis`` is not
        ``None`` for a rank-1 input.
    """
    if array.dtype == wp.vec3:
        if axis is not None:
            raise ValueError("sum over a vec3 array supports only axis=None.")
        if array.ndim != 1:
            raise ValueError("sum over a vec3 array requires a 1D array.")
        n = int(array.shape[0])
        if n == 0:
            raise ValueError("sum requires a non-empty array.")
        out_vec = wp.zeros(1, dtype=wp.vec3, device=array.device)
        wp.launch_tiled(
            kernel_reduce.sum_vec3_1d_tiled,
            dim=[kernel_reduce.blocks_1d(n)],
            inputs=[array, out_vec],
            block_dim=TILE_1D,
            device=array.device,
        )
        return out_vec.list()[0]
    if array.dtype == wp.bool:
        mask = cast(wp.array[wp.bool], array)
        mask_i32 = astype(mask, wp.int32)
        if mask.ndim == 2 and axis is None:
            mask_i32 = mask_i32.flatten()
        result = _reduce_scalar(cast(twt.ScalarArray, mask_i32), axis, _SCALAR_REDUCE["sum"])
        if axis is None:
            return int(cast(int, result))
        return cast(twt.Array1dInt32, result)
    return cast(
        float | int | twt.Array1dScalar,
        _reduce_scalar(cast(twt.ScalarArray, array), axis, _SCALAR_REDUCE["sum"]),
    )


@overload
def mean(array: wp.array[wp.vec3], *, axis: None = ...) -> wp.vec3: ...
@overload
def mean(array: twt.ScalarArray | wp.array[wp.bool], *, axis: None = ...) -> float: ...
@overload
def mean(
    array: twt.Array2dScalar | wp.array[wp.bool], *, axis: Literal[0, 1]
) -> twt.Array1dFloat32: ...
def mean(
    array: twt.ScalarArray | wp.array[wp.bool] | wp.array[wp.vec3],
    *,
    axis: Literal[0, 1] | None = None,
) -> float | twt.Array1dFloat32 | wp.vec3:
    """
    Arithmetic mean of ``array``.

    Delegates the reduction to the tiled [`sum`][triwarp.reduce.sum], then divides by the
    element count. With ``axis=None`` (default), returns one Python ``float``. With
    ``axis=0`` or ``axis=1`` on a rank-2 input, returns a 1D ``wp.float32`` array.

    The result is always floating point regardless of input dtype. For ``wp.bool``
    input, the mean is the fraction of ``True`` values. For a 1D ``wp.vec3`` input
    the mean is the component-wise average ``wp.vec3`` (``axis=None`` only).

    Parameters
    ----------
    array
        Rank-1 ``(n,)`` or rank-2 ``(n, m)`` scalar or ``wp.bool`` Warp array, or
        a rank-1 ``(n,)`` ``wp.vec3`` array. Must be non-empty.
    axis
        ``None`` for a global scalar result. ``0`` or ``1`` for a per-axis 1D
        result (rank-2 input only).

    Returns
    -------
    float | wp.vec3 | wp.array
        Global ``float`` (scalar/``wp.bool`` input) or ``wp.vec3`` (``wp.vec3``
        input) when ``axis=None``; 1D ``wp.float32`` array of length ``n``
        (``axis=1``) or ``m`` (``axis=0``) otherwise.

    Raises
    ------
    ValueError
        If ``array`` is empty, its rank is not 1 or 2, or ``axis`` is not
        ``None`` for a rank-1 (or ``wp.vec3``) input.
    """
    if array.dtype == wp.vec3:
        return cast(wp.vec3, sum(array, axis=axis)) / float(int(array.size))
    total = sum(array, axis=axis)
    if axis is None:
        return float(total) / float(int(array.size))
    sums = cast(twt.Array1dScalar, total)
    out = astype(sums, wp.float32)
    wp.map(wp.div, out, wp.float32(array.shape[axis]), out=out)
    return cast(twt.Array1dFloat32, out)


@overload
def weighted_sum(values: twt.Array1dFloat32, weights: twt.Array1dFloat32) -> float: ...
@overload
def weighted_sum(values: wp.array[wp.vec3], weights: twt.Array1dFloat32) -> wp.vec3: ...
def weighted_sum(
    values: twt.Array1dFloat32 | wp.array[wp.vec3], weights: twt.Array1dFloat32
) -> float | wp.vec3:
    """
    Weighted sum ``sum_i weights[i] * values[i]``.

    Parameters
    ----------
    values
        Rank-1 ``(n,)`` ``wp.float32`` or ``wp.vec3`` array. Must be non-empty.
    weights
        Rank-1 ``(n,)`` ``wp.float32`` array of the same length as ``values``.

    Returns
    -------
    float | wp.vec3
        Scalar weighted sum on the host (``float`` for scalar ``values``,
        ``wp.vec3`` for ``wp.vec3`` ``values``).

    Raises
    ------
    ValueError
        If either array is empty or their lengths differ.
    """
    n_values = int(values.shape[0])
    n_weights = int(weights.shape[0])
    if n_values == 0 or n_weights == 0:
        raise ValueError("weighted_sum requires non-empty arrays.")
    if n_values != n_weights:
        raise ValueError("weighted_sum requires values and weights of equal length.")

    n_blocks = kernel_reduce.blocks_1d(n_values)
    if values.dtype == wp.vec3:
        out_vec = wp.zeros(1, dtype=wp.vec3, device=values.device)
        wp.launch_tiled(
            kernel_reduce.weighted_sum_vec3_1d_tiled,
            dim=[n_blocks],
            inputs=[values, weights, out_vec],
            block_dim=TILE_1D,
            device=values.device,
        )
        return out_vec.list()[0]

    out = wp.zeros(1, dtype=wp.float32, device=values.device)
    wp.launch_tiled(
        kernel_reduce.weighted_sum1d_tiled,
        dim=[n_blocks],
        inputs=[values, weights, out],
        block_dim=TILE_1D,
        device=values.device,
    )
    return float(out.numpy().item())


def all(array: wp.array[wp.bool], *, axis: Literal[0, 1] | None = None) -> wp.array[wp.bool] | bool:
    """
    Reduce a boolean array with logical AND.

    For rank-1 input, or rank-2 input with ``axis=None``, reduce all elements
    to one Python ``bool``. For rank-2 input with an explicit axis, reduce along
    that axis to a 1D ``wp.bool`` array.

    Parameters
    ----------
    array
        Rank-1 ``(n,)`` or rank-2 ``(n, m)`` ``wp.bool`` array. Must be non-empty.
    axis
        ``0``, ``1``, or ``None``. For rank-2 input, ``None`` reduces all elements
        to a single scalar. Ignored for rank-1 input.

    Returns
    -------
    bool | wp.array[wp.bool]
        Python ``bool`` for rank-1 input or rank-2 with ``axis=None``.
        Length ``array.shape[1]`` when ``axis=0``, else ``array.shape[0]``,
        for rank-2 input with an explicit axis.

    Raises
    ------
    ValueError
        If ``array`` is empty, its rank is not 1 or 2, or ``axis`` is not
        ``0``, ``1``, or ``None`` for rank-2 input.
    """
    return _reduce_bool(array, axis, _BOOL_REDUCE["all"])


def median(array: twt.Array1dScalar) -> float:
    """
    Median of a 1D scalar array (``numpy.median``).

    Sorts a copy of the values on-device with ``warp.utils.radix_sort_pairs`` and reads the
    middle element (odd length) or averages the two middle elements (even length). The result
    is always a Python ``float``, regardless of input dtype.

    Parameters
    ----------
    array
        Rank-1 ``(n,)`` Warp array of any 32- or 64-bit scalar dtype accepted by
        ``warp.utils.radix_sort_pairs`` (``wp.int32``, ``wp.uint32``, ``wp.float32``,
        ``wp.int64``, ``wp.uint64``, ``wp.float64``). Must be non-empty.

    Returns
    -------
    float
        The median value on the host.

    Raises
    ------
    ValueError
        If ``array`` is empty or not rank-1.
    """
    if array.ndim != 1:
        raise ValueError("median requires a 1D array.")
    n = int(array.shape[0])
    if n == 0:
        raise ValueError("median requires a non-empty array.")

    sorted_values = cast(twt.Array1dScalar, _sorted_copy(cast(twt.Array1dScalar, array)))
    if n % 2 == 1:
        return float(read_scalar(sorted_values, n // 2))
    middle = sorted_values[n // 2 - 1 : n // 2 + 1].numpy()
    return (float(middle[0]) + float(middle[1])) / 2.0


class _ScalarReduceSpec(NamedTuple):
    name: str
    axis_rows_tiled: Callable[..., None]
    axis_cols_tiled: Callable[..., None]
    axis_rows_serial: Callable[..., None]
    axis_cols_serial: Callable[..., None]
    tiled_1d: Callable[..., None]
    tiled_2d: Callable[..., None]
    init_global: Callable[[type], int | float]
    global_output_slots: int
    dual_axis: bool


class _BoolReduceSpec(NamedTuple):
    name: str
    axis_rows_tiled: Callable[..., None]
    axis_cols_tiled: Callable[..., None]
    axis_rows_serial: Callable[..., None]
    axis_cols_serial: Callable[..., None]
    tiled_1d: Callable[..., None]
    init_global: int


_SCALAR_REDUCE: dict[str, _ScalarReduceSpec] = {
    "min": _ScalarReduceSpec(
        name="min",
        axis_rows_tiled=kernel_reduce.min_2d_rows_tiled,
        axis_cols_tiled=kernel_reduce.min_2d_cols_tiled,
        axis_rows_serial=kernel_reduce.min_2d_rows_serial,
        axis_cols_serial=kernel_reduce.min_2d_cols_serial,
        tiled_1d=kernel_reduce.min1d_tiled,
        tiled_2d=kernel_reduce.min2d_tiled,
        init_global=twt.dtype_max,
        global_output_slots=1,
        dual_axis=False,
    ),
    "max": _ScalarReduceSpec(
        name="max",
        axis_rows_tiled=kernel_reduce.max_2d_rows_tiled,
        axis_cols_tiled=kernel_reduce.max_2d_cols_tiled,
        axis_rows_serial=kernel_reduce.max_2d_rows_serial,
        axis_cols_serial=kernel_reduce.max_2d_cols_serial,
        tiled_1d=kernel_reduce.max1d_tiled,
        tiled_2d=kernel_reduce.max2d_tiled,
        init_global=twt.dtype_min,
        global_output_slots=1,
        dual_axis=False,
    ),
    "minmax": _ScalarReduceSpec(
        name="minmax",
        axis_rows_tiled=kernel_reduce.minmax_2d_rows_tiled,
        axis_cols_tiled=kernel_reduce.minmax_2d_cols_tiled,
        axis_rows_serial=kernel_reduce.minmax_2d_rows_serial,
        axis_cols_serial=kernel_reduce.minmax_2d_cols_serial,
        tiled_1d=kernel_reduce.minmax1d_tiled,
        tiled_2d=kernel_reduce.minmax2d_tiled,
        init_global=twt.dtype_max,
        global_output_slots=2,
        dual_axis=True,
    ),
    "sum": _ScalarReduceSpec(
        name="sum",
        axis_rows_tiled=kernel_reduce.sum_2d_rows_tiled,
        axis_cols_tiled=kernel_reduce.sum_2d_cols_tiled,
        axis_rows_serial=kernel_reduce.sum_2d_rows_serial,
        axis_cols_serial=kernel_reduce.sum_2d_cols_serial,
        tiled_1d=kernel_reduce.sum1d_tiled,
        tiled_2d=kernel_reduce.sum2d_tiled,
        init_global=twt.dtype_zero,
        global_output_slots=1,
        dual_axis=False,
    ),
}

_BOOL_REDUCE: dict[str, _BoolReduceSpec] = {
    "any": _BoolReduceSpec(
        name="any",
        axis_rows_tiled=kernel_reduce.any_2d_rows_tiled,
        axis_cols_tiled=kernel_reduce.any_2d_cols_tiled,
        axis_rows_serial=kernel_reduce.any_2d_rows_serial,
        axis_cols_serial=kernel_reduce.any_2d_cols_serial,
        tiled_1d=kernel_reduce.any_1d_tiled,
        init_global=0,
    ),
    "all": _BoolReduceSpec(
        name="all",
        axis_rows_tiled=kernel_reduce.all_2d_rows_tiled,
        axis_cols_tiled=kernel_reduce.all_2d_cols_tiled,
        axis_rows_serial=kernel_reduce.all_2d_rows_serial,
        axis_cols_serial=kernel_reduce.all_2d_cols_serial,
        tiled_1d=kernel_reduce.all_1d_tiled,
        init_global=1,
    ),
}


# --- private helpers ---------------------------------------------------------------------
#
# The launch/dispatch layer every public reduction above shares, kept together because that is what
# makes the six of them one mechanism rather than six. ``_launch_global_vec3_minmax`` is the only
# member with a single caller (``minmax``) and it stays here for that reason: it is the ``wp.vec3``
# sibling of ``_launch_global_scalar_tiled`` below, and moving it 20 functions up would put half of
# one dispatch table in the middle of the public surface.


def _launch_global_vec3_minmax(array: wp.array[wp.vec3]) -> tuple[wp.vec3, wp.vec3]:
    """Component-wise corner pair of a ``wp.vec3`` array: one launch, one buffer, one readback."""
    n = int(array.shape[0])
    if n == 0:
        raise ValueError("minmax requires a non-empty array.")
    if int(array.ndim) != 1:
        raise ValueError("minmax requires a rank-1 wp.vec3 array.")
    corners = wp.full(6, math.inf, dtype=wp.float32, device=array.device)
    wp.launch(
        kernel_reduce.minmax_vec3_chunked,
        dim=(n + TILE_1D - 1) // TILE_1D,
        inputs=[array, corners],
        device=array.device,
    )
    corners_np = corners.numpy()
    # Slots 3..5 hold the *negated* upper corner; see the kernel.
    return wp.vec3(*corners_np[:3].tolist()), wp.vec3(*(-corners_np[3:]).tolist())


def _reduce_scalar(
    array: twt.ScalarArray, axis: Literal[0, 1] | None, spec: _ScalarReduceSpec
) -> (
    float
    | int
    | twt.Array1dScalar
    | tuple[twt.Array1dScalar, twt.Array1dScalar]
    | tuple[float, float]
    | tuple[int, int]
):
    _validate_scalar_array(array, spec, axis)

    if array.ndim == 2 and axis is not None:
        return _launch_axis_scalar(array, axis, spec)

    return _launch_global_scalar_tiled(array, spec)


def _launch_global_scalar_tiled(
    array: twt.ScalarArray, spec: _ScalarReduceSpec
) -> float | int | tuple[float, float] | tuple[int, int]:
    if spec.global_output_slots == 2:
        out = wp.array(
            [twt.dtype_max(array.dtype), twt.dtype_min(array.dtype)],
            dtype=array.dtype,
            device=array.device,
        )
    else:
        out = wp.full(1, spec.init_global(array.dtype), dtype=array.dtype, device=array.device)

    flat = _flattened_for_global(array)
    if flat is not None:
        wp.launch_tiled(
            spec.tiled_1d,
            dim=[kernel_reduce.blocks_1d(int(flat.shape[0]))],
            inputs=[flat, out],
            block_dim=TILE_1D,
            device=array.device,
        )
    else:
        n, m = array.shape
        n_tiles = (n + TILE_2D - 1) // TILE_2D
        m_tiles = (m + TILE_2D - 1) // TILE_2D
        wp.launch_tiled(
            spec.tiled_2d,
            dim=[n_tiles, m_tiles],
            inputs=[array, out],
            block_dim=TILE_2D * TILE_2D,
            device=array.device,
        )

    out_np = out.numpy()
    if spec.global_output_slots == 2:
        return cast(tuple[int, int] | tuple[float, float], (out_np[0].item(), out_np[1].item()))
    return cast(float | int, out_np.item())


def _flattened_for_global(array: twt.ScalarArray) -> twt.Array1dScalar | None:
    """
    Return a 1-D view when the 1-D kernel is the better way to reduce ``array``, else ``None``.

    Rank-1 input is already the 1-D case. Rank-2 input is worth flattening when its trailing extent
    is under ``TILE_2D``, because the rank-2 kernel tiles ``TILE_2D``-squares and a table that
    narrow clips every tile — so its ``wp.tile_load`` branch never runs and all ``TILE_2D**2`` lanes
    walk the same short block, the rank-2 form of the amplification ``_launch_axis_scalar``
    documents. Flattening costs nothing on a contiguous buffer and hands the work to the 1-D kernel,
    which also carries the ``TILES_PER_BLOCK_1D`` fold. Measured on ``minmax``: **5.95x** on a
    ``(14M, 3)`` table and **6.18x** on ``(8M, 2)``, against 1.02x on ``(40k, 128)`` — where the
    tile branch does fire, which is why the width test is there and not just a contiguity test.

    The cost is at the small end: ``(36k, 3)`` measures **0.79x**, because 16 tiles per block leaves
    too few blocks to fill the device. That is 13 µs of kernel time under the ~82 µs host floor
    every scalar-returning reduction already pays, so it is invisible end to end.

    ``wp.array.flatten()`` raises on a non-contiguous array rather than copying, so a strided view
    keeps the rank-2 kernel — correctness first, and such a view is not the common case.

    Parameters
    ----------
    array
        Rank-1 or rank-2 scalar array being reduced with ``axis=None``.

    Returns
    -------
    wp.array | None
        A 1-D view to reduce with the 1-D kernel, or ``None`` to use the rank-2 kernel.
    """
    if array.ndim == 1:
        return cast(twt.Array1dScalar, array)
    if int(array.shape[1]) < TILE_2D and array.is_contiguous:
        return cast(twt.Array1dScalar, array.flatten())
    return None


def _launch_axis_scalar(
    array: twt.Array2dScalar, axis: Literal[0, 1], spec: _ScalarReduceSpec
) -> twt.Array1dScalar | tuple[twt.Array1dScalar, twt.Array1dScalar]:
    n_rows, n_cols = int(array.shape[0]), int(array.shape[1])
    n_out, reduced = (n_rows, n_cols) if axis == 1 else (n_cols, n_rows)

    if reduced < TILE_1D:
        # The tiled kernels never take their tile_load branch below TILE_1D: every lane of every
        # block redundantly runs the serial remainder, a TILE_1D-fold read amplification (measured
        # 49x on a (14M, 3) axis=1 max). One plain thread per output, direct write, no init fill.
        serial = spec.axis_rows_serial if axis == 1 else spec.axis_cols_serial
        if spec.dual_axis:
            out_min = wp.empty(n_out, dtype=array.dtype, device=array.device)
            out_max = wp.empty(n_out, dtype=array.dtype, device=array.device)
            wp.launch(serial, dim=n_out, inputs=[array, out_min, out_max], device=array.device)
            return cast(twt.Array1dScalar, out_min), cast(twt.Array1dScalar, out_max)
        out = wp.empty(n_out, dtype=array.dtype, device=array.device)
        wp.launch(serial, dim=n_out, inputs=[array, out], device=array.device)
        return cast(twt.Array1dScalar, out)

    tiled = spec.axis_rows_tiled if axis == 1 else spec.axis_cols_tiled
    n_tiles = (reduced + TILE_1D - 1) // TILE_1D
    if spec.dual_axis:
        out_min = wp.full(n_out, twt.dtype_max(array.dtype), dtype=array.dtype, device=array.device)
        out_max = wp.full(n_out, twt.dtype_min(array.dtype), dtype=array.dtype, device=array.device)
        wp.launch_tiled(
            tiled,
            dim=[n_out, n_tiles],
            inputs=[array, out_min, out_max],
            block_dim=TILE_1D,
            device=array.device,
        )
        return cast(twt.Array1dScalar, out_min), cast(twt.Array1dScalar, out_max)

    out = wp.full(n_out, spec.init_global(array.dtype), dtype=array.dtype, device=array.device)
    wp.launch_tiled(
        tiled, dim=[n_out, n_tiles], inputs=[array, out], block_dim=TILE_1D, device=array.device
    )
    return cast(twt.Array1dScalar, out)


def _validate_scalar_array(
    array: twt.ScalarArray, spec: _ScalarReduceSpec, axis: Literal[0, 1] | None
) -> None:
    if int(array.size) == 0:
        raise ValueError(f"{spec.name} requires a non-empty array.")
    if array.ndim == 1 and axis is not None:
        raise ValueError(f"{spec.name} requires axis=None for a 1D array.")
    if array.ndim not in (1, 2):
        raise ValueError(f"{spec.name} requires a 1D or 2D array.")


def _reduce_bool(
    array: wp.array[wp.bool], axis: Literal[0, 1] | None, spec: _BoolReduceSpec
) -> wp.array[wp.bool] | bool:
    if int(array.size) == 0:
        raise ValueError(f"{spec.name} requires a non-empty array.")

    if array.ndim == 1:
        mask_i32 = astype(array, wp.int32)
        return _launch_global_bool_tiled(mask_i32, spec)

    if array.ndim == 2:
        mask_i32 = astype(array, wp.int32)
        n_rows, n_cols = int(array.shape[0]), int(array.shape[1])
        if axis is None:
            return _launch_global_bool_tiled(mask_i32.flatten(), spec)
        n_out, reduced = (n_rows, n_cols) if axis == 1 else (n_cols, n_rows)
        if reduced < TILE_1D:
            # Same rule as _launch_axis_scalar: below TILE_1D the tiled form only amplifies reads.
            serial = spec.axis_rows_serial if axis == 1 else spec.axis_cols_serial
            out_i32 = wp.empty(n_out, dtype=wp.int32, device=array.device)
            wp.launch(serial, dim=n_out, inputs=[mask_i32, out_i32], device=array.device)
        else:
            tiled = spec.axis_rows_tiled if axis == 1 else spec.axis_cols_tiled
            n_tiles = (reduced + TILE_1D - 1) // TILE_1D
            out_i32 = wp.full(n_out, spec.init_global, dtype=wp.int32, device=array.device)
            wp.launch_tiled(
                tiled,
                dim=[n_out, n_tiles],
                inputs=[mask_i32, out_i32],
                block_dim=TILE_1D,
                device=array.device,
            )
        out = astype(out_i32, wp.bool)
        return out

    raise ValueError(f"{spec.name} requires a 1D or 2D array.")


def _launch_global_bool_tiled(mask_i32: wp.array[wp.int32], spec: _BoolReduceSpec) -> bool:
    out = wp.full(1, spec.init_global, dtype=wp.int32, device=mask_i32.device)
    n = int(mask_i32.shape[0])
    wp.launch_tiled(
        spec.tiled_1d,
        dim=[kernel_reduce.blocks_1d(n)],
        inputs=[mask_i32, out],
        block_dim=TILE_1D,
        device=mask_i32.device,
    )
    return bool(out.numpy().item() != 0)
