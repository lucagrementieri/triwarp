"""
Warp array type aliases and runtime rank checks for Python wrappers.

Use these aliases in ``triwarp`` Python APIs instead of ``wp.array2d[dtype]``, which
static checkers treat as Warp annotation objects (no ``.shape`` / indexing).

Kernels should continue to use ``wp.array2d[dtype]`` in ``@wp.kernel`` signatures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias, TypeVar, cast

import warp as wp

T = TypeVar("T")

if TYPE_CHECKING:
    Array1dInt32: TypeAlias = wp.array[wp.int32, Literal[1]]
    Array1dFloat32: TypeAlias = wp.array[wp.float32, Literal[1]]
    Array2dInt32: TypeAlias = wp.array[wp.int32, Literal[2]]
    Array2dFloat32: TypeAlias = wp.array[wp.float32, Literal[2]]

    # wp.Int / wp.Float / wp.Scalar are TypeVars; subscripting wp.array[...] with them
    # yields a generic alias that requires type arguments under basedpyright.
    Array1dInt: TypeAlias = Array1dInt32
    Array2dInt: TypeAlias = Array2dInt32
    Array1dFloat: TypeAlias = Array1dFloat32
    Array2dFloat: TypeAlias = Array2dFloat32
    Array1dScalar: TypeAlias = Array1dInt32 | Array1dFloat32
    Array2dScalar: TypeAlias = Array2dInt32 | Array2dFloat32

    IntArray: TypeAlias = Array1dInt | Array2dInt
    FloatArray: TypeAlias = Array1dFloat | Array2dFloat
    ScalarArray: TypeAlias = Array1dScalar | Array2dScalar
else:
    Array1dInt32 = wp.array
    Array1dFloat32 = wp.array
    Array2dInt32 = wp.array
    Array2dFloat32 = wp.array
    Array1dInt = wp.array
    Array2dInt = wp.array
    Array1dFloat = wp.array
    Array2dFloat = wp.array
    Array1dScalar = wp.array
    Array2dScalar = wp.array
    IntArray = wp.array
    FloatArray = wp.array
    ScalarArray = wp.array

__all__ = [
    "Array1dFloat",
    "Array1dFloat32",
    "Array1dInt",
    "Array1dInt32",
    "Array1dScalar",
    "Array2dFloat",
    "Array2dFloat32",
    "Array2dInt",
    "Array2dInt32",
    "Array2dScalar",
    "FloatArray",
    "IntArray",
    "ScalarArray",
    "as_array2d_float32",
    "as_array2d_int32",
    "empty_float32_2d",
    "empty_int32_2d",
    "ensure_ndim",
]


def ensure_ndim(arr: wp.array[T], ndim: int, *, dtype: type | None = None) -> wp.array[T]:
    """Validate rank (and optionally dtype) of a Warp array."""
    if int(arr.ndim) != ndim:
        raise TypeError(f"expected {ndim}D array, got ndim={arr.ndim}")
    if dtype is not None and arr.dtype != dtype:
        raise TypeError(f"expected dtype {dtype}, got {arr.dtype}")
    return arr


def as_array2d_int32(arr: wp.array[T]) -> Array2dInt32:
    """
    Validate and narrow a Warp array to [`Array2dInt32`][triwarp.typing.Array2dInt32].

    Parameters
    ----------
    arr
        Warp array expected to be rank-2 ``int32``.

    Returns
    -------
    Array2dInt32
        ``arr`` unchanged, narrowed to the checked alias.

    Raises
    ------
    TypeError
        If ``arr`` is not rank-2 ``int32``.
    """
    ensure_ndim(arr, 2, dtype=wp.int32)
    return cast(Array2dInt32, arr)


def as_array2d_float32(arr: wp.array[T]) -> Array2dFloat32:
    """
    Validate and narrow a Warp array to [`Array2dFloat32`][triwarp.typing.Array2dFloat32].

    Parameters
    ----------
    arr
        Warp array expected to be rank-2 ``float32``.

    Returns
    -------
    Array2dFloat32
        ``arr`` unchanged, narrowed to the checked alias.

    Raises
    ------
    TypeError
        If ``arr`` is not rank-2 ``float32``.
    """
    ensure_ndim(arr, 2, dtype=wp.float32)
    return cast(Array2dFloat32, arr)


def _shape_2d(shape: tuple[int, int] | list[int]) -> tuple[int, int]:
    dims = tuple(int(x) for x in shape)
    if len(dims) != 2:
        raise ValueError(f"2D shape must have length 2, got {shape!r}")
    return (dims[0], dims[1])


def empty_int32_2d(
    shape: tuple[int, int] | list[int], *, device: wp.DeviceLike = None
) -> Array2dInt32:
    """
    Allocate an uninitialized rank-2 ``int32`` Warp array.

    Parameters
    ----------
    shape
        ``(rows, cols)`` shape of the allocated array.
    device
        Target Warp device.

    Returns
    -------
    Array2dInt32
        Uninitialized ``(rows, cols)`` ``int32`` array on ``device``.
    """
    return cast(Array2dInt32, wp.empty(_shape_2d(shape), dtype=wp.int32, device=device))


def empty_float32_2d(
    shape: tuple[int, int] | list[int], *, device: wp.DeviceLike = None
) -> Array2dFloat32:
    """
    Allocate an uninitialized rank-2 ``float32`` Warp array.

    Parameters
    ----------
    shape
        ``(rows, cols)`` shape of the allocated array.
    device
        Target Warp device.

    Returns
    -------
    Array2dFloat32
        Uninitialized ``(rows, cols)`` ``float32`` array on ``device``.
    """
    return cast(Array2dFloat32, wp.empty(_shape_2d(shape), dtype=wp.float32, device=device))
