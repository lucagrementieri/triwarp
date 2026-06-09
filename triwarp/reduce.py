"""Global reductions on Warp arrays."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, NamedTuple, cast, overload

import warp as wp

from triwarp.constants import TILE_1D, TILE_2D
from triwarp.kernels import reduce as kernel_reduce
import triwarp.typing as twt


@overload
def min(array: twt.Array1dInt32 | twt.Array2dInt32, *, axis: None = ...) -> int: ...
@overload
def min(array: twt.Array1dFloat32 | twt.Array2dFloat32, *, axis: None = ...) -> float: ...
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
    1D ``wp.array`` of the same dtype using one thread per output element.

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
def max(array: twt.Array1dFloat32 | twt.Array2dFloat32, *, axis: None = ...) -> float: ...
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
    1D ``wp.array`` of the same dtype using one thread per output element.

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
    array: twt.Array1dFloat32 | twt.Array2dFloat32, *, axis: None = ...
) -> tuple[float, float]: ...
@overload
def minmax(
    array: twt.Array2dScalar, *, axis: Literal[0, 1]
) -> tuple[twt.Array1dScalar, twt.Array1dScalar]: ...
def minmax(
    array: twt.ScalarArray, *, axis: Literal[0, 1] | None = None
) -> tuple[float, float] | tuple[int, int] | tuple[twt.Array1dScalar, twt.Array1dScalar]:
    """
    Minimum and maximum of ``array``.

    With ``axis=None`` (default), reduces every element to two Python scalars using
    tiled kernels (``wp.tile_load`` + ``wp.tile_min``/``tile_max`` + ``wp.atomic_min``/``atomic_max``).

    With ``axis=0`` or ``axis=1`` on a rank-2 input, reduces along that axis to a
    pair of 1D ``wp.array`` buffers (min, max) of the same dtype.

    Parameters
    ----------
    array
        Rank-1 ``(n,)`` or rank-2 ``(n, m)`` scalar Warp array. Must be non-empty.
    axis
        ``None`` for a global scalar result. ``0`` or ``1`` for per-axis 1D
        results (rank-2 input only).

    Returns
    -------
    tuple[float, float] | tuple[int, int] | tuple[wp.array, wp.array]
        ``(min, max)`` as Python scalars when ``axis=None``; pair of 1D arrays
        of length ``n`` (``axis=1``) or ``m`` (``axis=0``) otherwise.

    Raises
    ------
    ValueError
        If ``array`` is empty, its rank is not 1 or 2, or ``axis`` is not
        ``None`` for a rank-1 input.
    """
    return cast(
        tuple[float, float] | tuple[int, int] | tuple[twt.Array1dScalar, twt.Array1dScalar],
        _reduce_scalar(array, axis, _SCALAR_REDUCE["minmax"]),
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


@overload
def max_for_dtype(dtype: type[wp.Int]) -> int: ...
@overload
def max_for_dtype(dtype: type[wp.Float]) -> float: ...
def max_for_dtype(dtype: type[wp.Scalar]) -> int | float:
    if not wp.types.type_is_int(dtype):
        return float("inf")
    bits = wp.types.type_size_in_bytes(dtype) * 8
    if not dtype.__name__.lower().startswith("u"):
        bits -= 1
    return (1 << bits) - 1


@overload
def min_for_dtype(dtype: type[wp.Int]) -> int: ...
@overload
def min_for_dtype(dtype: type[wp.Float]) -> float: ...
def min_for_dtype(dtype: type[wp.Scalar]) -> int | float:
    if not wp.types.type_is_int(dtype):
        return float("-inf")
    bits = wp.types.type_size_in_bytes(dtype) * 8
    if dtype.__name__.lower().startswith("u"):
        return 0
    return -(1 << (bits - 1))


class _ScalarReduceSpec(NamedTuple):
    name: str
    axis_rows: Callable[..., None]
    axis_cols: Callable[..., None]
    tiled_1d: Callable[..., None]
    tiled_2d: Callable[..., None]
    init_global: Callable[[type], int | float]
    global_output_slots: int
    dual_axis: bool


class _BoolReduceSpec(NamedTuple):
    name: str
    axis_rows: Callable[..., None]
    axis_cols: Callable[..., None]
    tiled_1d: Callable[..., None]
    init_global: int


_SCALAR_REDUCE: dict[str, _ScalarReduceSpec] = {
    "min": _ScalarReduceSpec(
        name="min",
        axis_rows=kernel_reduce.min_2d_rows,
        axis_cols=kernel_reduce.min_2d_cols,
        tiled_1d=kernel_reduce.min1d_tiled,
        tiled_2d=kernel_reduce.min2d_tiled,
        init_global=max_for_dtype,
        global_output_slots=1,
        dual_axis=False,
    ),
    "max": _ScalarReduceSpec(
        name="max",
        axis_rows=kernel_reduce.max_2d_rows,
        axis_cols=kernel_reduce.max_2d_cols,
        tiled_1d=kernel_reduce.max1d_tiled,
        tiled_2d=kernel_reduce.max2d_tiled,
        init_global=min_for_dtype,
        global_output_slots=1,
        dual_axis=False,
    ),
    "minmax": _ScalarReduceSpec(
        name="minmax",
        axis_rows=kernel_reduce.minmax_2d_rows,
        axis_cols=kernel_reduce.minmax_2d_cols,
        tiled_1d=kernel_reduce.minmax1d_tiled,
        tiled_2d=kernel_reduce.minmax2d_tiled,
        init_global=max_for_dtype,
        global_output_slots=2,
        dual_axis=True,
    ),
}

_BOOL_REDUCE: dict[str, _BoolReduceSpec] = {
    "any": _BoolReduceSpec(
        name="any",
        axis_rows=kernel_reduce.any_2d_rows,
        axis_cols=kernel_reduce.any_2d_cols,
        tiled_1d=kernel_reduce.any_1d_tiled,
        init_global=0,
    ),
    "all": _BoolReduceSpec(
        name="all",
        axis_rows=kernel_reduce.all_2d_rows,
        axis_cols=kernel_reduce.all_2d_cols,
        tiled_1d=kernel_reduce.all_1d_tiled,
        init_global=1,
    ),
}


def _validate_scalar_array(
    array: twt.ScalarArray, spec: _ScalarReduceSpec, axis: Literal[0, 1] | None
) -> None:
    if int(array.size) == 0:
        raise ValueError(f"{spec.name} requires a non-empty array.")
    if array.ndim == 1 and axis is not None:
        raise ValueError(f"{spec.name} requires axis=None for a 1D array.")
    if array.ndim not in (1, 2):
        raise ValueError(f"{spec.name} requires a 1D or 2D array.")


def _launch_axis_scalar(
    array: twt.Array2dScalar, axis: Literal[0, 1], spec: _ScalarReduceSpec
) -> twt.Array1dScalar | tuple[twt.Array1dScalar, twt.Array1dScalar]:
    n_rows, n_cols = int(array.shape[0]), int(array.shape[1])
    if spec.dual_axis:
        if axis == 1:
            out_min = wp.empty(n_rows, dtype=array.dtype, device=array.device)
            out_max = wp.empty(n_rows, dtype=array.dtype, device=array.device)
            wp.launch(
                spec.axis_rows, dim=n_rows, inputs=[array, out_min, out_max], device=array.device
            )
        else:
            out_min = wp.empty(n_cols, dtype=array.dtype, device=array.device)
            out_max = wp.empty(n_cols, dtype=array.dtype, device=array.device)
            wp.launch(
                spec.axis_cols, dim=n_cols, inputs=[array, out_min, out_max], device=array.device
            )
        return cast(twt.Array1dScalar, out_min), cast(twt.Array1dScalar, out_max)

    if axis == 1:
        out = wp.empty(n_rows, dtype=array.dtype, device=array.device)
        wp.launch(spec.axis_rows, dim=n_rows, inputs=[array, out], device=array.device)
    else:
        out = wp.empty(n_cols, dtype=array.dtype, device=array.device)
        wp.launch(spec.axis_cols, dim=n_cols, inputs=[array, out], device=array.device)
    return cast(twt.Array1dScalar, out)


def _launch_global_scalar_tiled(
    array: twt.ScalarArray, spec: _ScalarReduceSpec
) -> float | int | tuple[float, float] | tuple[int, int]:
    if spec.global_output_slots == 2:
        out = wp.array(
            [max_for_dtype(array.dtype), min_for_dtype(array.dtype)],
            dtype=array.dtype,
            device=array.device,
        )
    else:
        out = wp.full(1, spec.init_global(array.dtype), dtype=array.dtype, device=array.device)

    if array.ndim == 1:
        n = int(array.shape[0])
        n_tiles = (n + TILE_1D - 1) // TILE_1D
        wp.launch_tiled(
            spec.tiled_1d,
            dim=[n_tiles],
            inputs=[array, out],
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


def _bool_mask_as_int32(mask: wp.array[wp.bool]) -> wp.array[wp.int32]:
    out = wp.empty(mask.shape, dtype=wp.int32, device=mask.device)
    wp.utils.array_cast(mask.flatten(), out.flatten())
    return out


def _launch_global_bool_tiled(mask_i32: wp.array[wp.int32], spec: _BoolReduceSpec) -> bool:
    out = wp.full(1, spec.init_global, dtype=wp.int32, device=mask_i32.device)
    n = int(mask_i32.shape[0])
    n_tiles = (n + TILE_1D - 1) // TILE_1D
    wp.launch_tiled(
        spec.tiled_1d,
        dim=[n_tiles],
        inputs=[mask_i32, out],
        block_dim=TILE_1D,
        device=mask_i32.device,
    )
    return bool(out.numpy().item() != 0)


def _reduce_bool(
    array: wp.array[wp.bool], axis: Literal[0, 1] | None, spec: _BoolReduceSpec
) -> wp.array[wp.bool] | bool:
    if int(array.size) == 0:
        raise ValueError(f"{spec.name} requires a non-empty array.")

    if array.ndim == 1:
        mask_i32 = _bool_mask_as_int32(array)
        return _launch_global_bool_tiled(mask_i32, spec)

    if array.ndim == 2:
        mask_i32 = _bool_mask_as_int32(array)
        n_rows, n_cols = int(array.shape[0]), int(array.shape[1])
        if axis is None:
            return _launch_global_bool_tiled(mask_i32.flatten(), spec)
        if axis == 1:
            out = wp.empty(n_rows, dtype=wp.bool, device=array.device)
            wp.launch(spec.axis_rows, dim=n_rows, inputs=[mask_i32, out], device=array.device)
        else:
            out = wp.empty(n_cols, dtype=wp.bool, device=array.device)
            wp.launch(spec.axis_cols, dim=n_cols, inputs=[mask_i32, out], device=array.device)
        return out

    raise ValueError(f"{spec.name} requires a 1D or 2D array.")
