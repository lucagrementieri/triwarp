"""Unique values for Warp arrays (1D scalars and 2D rows)."""

from __future__ import annotations

import math
from typing import Literal, TypeVar, overload

import warp as wp

import triwarp as tw
import triwarp.typing as twt
from triwarp.array import gather, init_range
from triwarp.kernels import array as kernel_array
from triwarp.kernels import edges as kernel_edges
from triwarp.kernels import unique as kernel_unique

Scalar = TypeVar("Scalar", bound=wp.Scalar)


@overload
def unique_1d(
    data: wp.array[Scalar],
    *,
    return_inverse: Literal[False] = False,
    return_counts: Literal[False] = False,
) -> wp.array[Scalar]: ...
@overload
def unique_1d(
    data: wp.array[Scalar], *, return_inverse: Literal[True], return_counts: Literal[False] = False
) -> tuple[wp.array[Scalar], wp.array[wp.int32]]: ...
@overload
def unique_1d(
    data: wp.array[Scalar], *, return_inverse: Literal[False] = False, return_counts: Literal[True]
) -> tuple[wp.array[Scalar], wp.array[wp.int32]]: ...
@overload
def unique_1d(
    data: wp.array[Scalar], *, return_inverse: Literal[True], return_counts: Literal[True]
) -> tuple[wp.array[Scalar], wp.array[wp.int32], wp.array[wp.int32]]: ...
def unique_1d(
    data: wp.array[Scalar], *, return_inverse: bool = False, return_counts: bool = False
) -> (
    wp.array[Scalar]
    | tuple[wp.array[Scalar], wp.array[wp.int32]]
    | tuple[wp.array[Scalar], wp.array[wp.int32], wp.array[wp.int32]]
):
    """
    Find sorted unique elements of a 1D Warp array (``numpy.unique`` subset).

    Uses an open-addressing hash table: O(n) inserts plus an O(n_unique log n_unique) sort of only
    the unique values.

    ``return_inverse`` maps each input position to the index of its value in the sorted unique
    output.
    ``return_index`` is not supported.

    Parameters
    ----------
    data
        Rank-1 ``wp.array`` with any scalar ``dtype``.
    return_inverse
        If ``True``, include the inverse mapping in the return tuple.
    return_counts
        If ``True``, include per-unique occurrence counts in the return tuple.

    Returns
    -------
    unique : wp.array
        Sorted unique values (same ``dtype`` and ``device`` as ``data``).
    inverse : wp.array[wp.int32], optional
        Present if ``return_inverse=True``. ``inverse[i]`` indexes ``unique`` such
        that conceptually ``unique[inverse[i]] == data[i]`` (float ``NaN`` matches
        NumPy: all ``NaN`` values share one slot).
    counts : wp.array[wp.int32], optional
        Present if ``return_counts=True``. Number of occurrences per ``unique`` row.

    Raises
    ------
    ValueError
        If ``data`` is not rank-1 or length is ``>= 2**31``.
    """
    if int(data.ndim) != 1:
        raise ValueError(f"unique_1d expects a rank-1 array, got ndim={data.ndim}")

    device = data.device
    n = int(data.shape[0])

    if n == 0:
        empty_unique = wp.empty(0, dtype=data.dtype, device=device)
        empty_i32 = wp.empty(0, dtype=wp.int32, device=device)
        return _pack_unique_result(
            empty_unique,
            inverse=empty_i32 if return_inverse else None,
            counts=empty_i32 if return_counts else None,
        )
    if n > (1 << 30):
        raise ValueError(f"unique_1d requires length <= 2**30, got length {n}")

    log2_capacity = max(3, math.ceil(math.log2(n) + 1))
    mask = wp.int32((1 << log2_capacity) - 1)

    data_int = reinterpret_cast_to_int(data, n)
    return _unique_hash(data_int, data.dtype, n, mask, return_inverse, return_counts)


def _unique_hash(
    data_int: wp.array[wp.int32] | wp.array[wp.int64],
    original_dtype: type[Scalar],
    n: int,
    mask: wp.int32,
    return_inverse: bool,
    return_counts: bool,
) -> (
    wp.array[Scalar]
    | tuple[wp.array[Scalar], wp.array[wp.int32]]
    | tuple[wp.array[Scalar], wp.array[wp.int32], wp.array[wp.int32]]
):
    cap = int(mask) + 1
    key_dtype = data_int.dtype
    device = data_int.device

    # Phase 1: parallel insert into open-addressing hash table (slot_key 0 = empty).
    slot_key = wp.zeros(cap, dtype=key_dtype, device=device)
    slot_counts = wp.zeros(cap, dtype=wp.int32, device=device)
    wp.launch(
        kernel_unique.hash_insert,
        dim=n,
        inputs=[data_int, slot_key, slot_counts, mask],
        device=device,
    )

    # Phase 2: mark occupied slots, prefix-scan to get compact positions.
    occ_mask = wp.zeros(cap, dtype=wp.int32, device=device)
    wp.launch(kernel_unique.mark_occupied, dim=cap, inputs=[slot_key, occ_mask], device=device)
    scan_pos = wp.empty(cap, dtype=wp.int32, device=device)
    wp.utils.array_scan(occ_mask, scan_pos, inclusive=True)
    wp.launch(kernel_array.sub, dim=cap, inputs=[scan_pos, wp.int32(1)], device=device)
    n_unique = int(tw.reduce.max(scan_pos)) + 1

    # Phase 3: compact unique keys and their occurrence counts.
    keys_compact = wp.empty(n_unique, dtype=key_dtype, device=device)
    cnts_compact = wp.empty(n_unique, dtype=wp.int32, device=device)
    wp.launch(
        kernel_unique.compact_from_table,
        dim=cap,
        inputs=[slot_key, slot_counts, occ_mask, scan_pos, keys_compact, cnts_compact],
        device=device,
    )

    # Phase 4: sort only the n_unique keys (typically n_unique << n).
    keys_buf = wp.empty(2 * n_unique, dtype=key_dtype, device=device)
    wp.copy(keys_buf, keys_compact, count=n_unique)
    perm_buf = init_range(2 * n_unique, device)
    wp.utils.radix_sort_pairs(keys_buf, perm_buf, count=n_unique)

    unique_values = reinterpret_cast_from_int(keys_buf, original_dtype, count=n_unique)

    unique_counts = None
    if return_counts:
        sort_perm = wp.empty(n_unique, dtype=wp.int32, device=device)
        wp.copy(sort_perm, perm_buf, count=n_unique)
        unique_counts = wp.empty(n_unique, dtype=wp.int32, device=device)
        wp.copy(unique_counts, cnts_compact[sort_perm])

    unique_inverse = None
    if return_inverse:
        sorted_dense = wp.empty(n_unique, dtype=key_dtype, device=device)
        wp.copy(sorted_dense, keys_buf, count=n_unique)
        unique_inverse = wp.empty(n, dtype=wp.int32, device=device)
        wp.launch(
            kernel_array.map_sorted_inverse,
            dim=n,
            inputs=[data_int, sorted_dense, unique_inverse],
            device=device,
        )

    return _pack_unique_result(unique_values, inverse=unique_inverse, counts=unique_counts)


def _pack_unique_result(
    unique: wp.array[Scalar],
    *,
    inverse: wp.array[wp.int32] | None = None,
    counts: wp.array[wp.int32] | None = None,
) -> (
    wp.array[Scalar]
    | tuple[wp.array[Scalar], wp.array[wp.int32]]
    | tuple[wp.array[Scalar], wp.array[wp.int32], wp.array[wp.int32]]
):
    if inverse is not None and counts is not None:
        return unique, inverse, counts
    if inverse is not None:
        return unique, inverse
    if counts is not None:
        return unique, counts
    return unique


@overload
def unique_faces(
    faces: wp.array[wp.int32], *, return_inverse: Literal[False] = False
) -> wp.array[wp.int32]: ...
@overload
def unique_faces(
    faces: wp.array[wp.int32], *, return_inverse: Literal[True]
) -> tuple[wp.array[wp.int32], wp.array[wp.int32]]: ...
def unique_faces(
    faces: wp.array[wp.int32], *, return_inverse: bool = False
) -> wp.array[wp.int32] | tuple[wp.array[wp.int32], wp.array[wp.int32]]:
    """
    Find unique triangular faces up to vertex permutation (orientation-agnostic).

    Two faces are equal when they share the same three vertices regardless of order. Each
    face's indices are sorted before deduplication; the representative returned for a class is
    the first occurring face, with its original vertex order preserved.

    Parameters
    ----------
    faces
        Flat ``wp.int32`` triangle index buffer of length ``3 * n_faces`` (common 1D format).
    return_inverse
        If ``True``, also return the inverse mapping from each input face to its slot in the
        unique output.

    Returns
    -------
    unique_faces : wp.array[wp.int32]
        Flat buffer of the unique faces (length ``3 * n_unique``).
    inverse : wp.array[wp.int32], optional
        Present if ``return_inverse=True``. Length ``n_faces``; ``inverse[i]`` is the unique
        slot of input face ``i``.
    """
    device = faces.device
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        empty_faces = wp.empty(0, dtype=wp.int32, device=device)
        if return_inverse:
            return empty_faces, wp.empty(0, dtype=wp.int32, device=device)
        return empty_faces

    faces2d = faces.reshape((-1, 3))
    sorted_faces = twt.empty_int32_2d((n_faces, 3), device=device)
    wp.launch(
        kernel_unique.sort_face_indices, dim=n_faces, inputs=[faces2d, sorted_faces], device=device
    )
    _, inverse = unique_rows(sorted_faces, return_inverse=True)
    n_unique = int(tw.reduce.max(inverse)) + 1
    first = wp.full(n_unique, wp.int32(n_faces), dtype=wp.int32, device=device)
    wp.launch(
        kernel_edges.scatter_first_occurrence, dim=n_faces, inputs=[inverse, first], device=device
    )
    unique_faces_out = gather(faces2d, first).reshape((-1,))
    if return_inverse:
        return unique_faces_out, inverse
    return unique_faces_out


def reinterpret_cast_to_int(
    data: wp.array[wp.Scalar], count: int | None = None
) -> wp.array[wp.int32] | wp.array[wp.int64]:
    n_bits = wp.types.type_size_in_bytes(data.dtype) * 8
    n = data.shape[0]
    count = count or n
    copy_count = min(n, count)

    if n_bits > 32:
        reinterpreted = wp.empty(count, dtype=wp.int64, device=data.device)
        wp.copy(reinterpreted, data, count=copy_count)
        return reinterpreted

    reinterpreted = wp.empty(count, dtype=wp.int32, device=data.device)
    if wp.types.type_is_float(data.dtype):
        src = data
        if n_bits < 32:
            src = wp.empty(copy_count, dtype=wp.float32, device=data.device)
            wp.utils.array_cast(data, src, count=copy_count)
        wp.copy(reinterpreted, src, count=copy_count)
    else:
        wp.utils.array_cast(data, reinterpreted, count=copy_count)
    return reinterpreted


def reinterpret_cast_from_int(
    data: wp.array[wp.int32] | wp.array[wp.int64], dtype: type[Scalar], count: int | None = None
) -> wp.array[Scalar]:
    n_bits = wp.types.type_size_in_bytes(data.dtype) * 8
    n_target_bits = wp.types.type_size_in_bytes(dtype) * 8
    n = data.shape[0]
    count = count or n
    copy_count = min(n, count)

    if n_bits == n_target_bits:
        reinterpreted = wp.empty(count, dtype=dtype, device=data.device)
        wp.copy(reinterpreted, data, count=copy_count)
        return reinterpreted

    if wp.types.type_is_float(dtype) and n_bits > n_target_bits:
        wide_dtype = getattr(wp, f"float{n_bits}")
        wide = wp.empty(count, dtype=wide_dtype, device=data.device)
        wp.copy(wide, data, count=copy_count)
        reinterpreted_casted = wp.empty(count, dtype=dtype, device=data.device)
        wp.utils.array_cast(wide, reinterpreted_casted, count=copy_count)
    else:
        reinterpreted_casted = wp.empty(count, dtype=dtype, device=data.device)
        wp.utils.array_cast(data, reinterpreted_casted, count=copy_count)
    return reinterpreted_casted


def hash_vector_rows(data: wp.array[wp.vec3], epsilon: float = 0.0) -> wp.array[wp.uint64]:
    """
    Pack each ``wp.vec3`` row into a single ``uint64`` key.

    With ``epsilon == 0.0``, each coordinate is interpreted as ``float32`` bits,
    right-shifted by 11 to drop low mantissa bits, then concatenated into one key
    per row for bucketing or near-duplicate grouping. With ``epsilon > 0.0``, each
    coordinate is instead rounded to the nearest multiple of ``epsilon`` and the
    resulting integer row is packed via [`hash_indices_rows`][triwarp.unique.hash_indices_rows],
    giving exact (non-bucketed) equality up to the tolerance.

    Parameters
    ----------
    data
        ``(n,)`` device array of ``wp.vec3`` values.
    epsilon
        Uniqueness tolerance. ``0`` uses the bit-truncation hash; positive values
        round coordinates to ``round(v / epsilon)`` before packing.

    Returns
    -------
    wp.array[wp.uint64]
        Length-``n`` array on ``data.device`` with one packed key per row.

    See Also
    --------
    [`hash_indices_rows`][triwarp.unique.hash_indices_rows]
    [`hash_rows`][triwarp.unique.hash_rows]
    """
    if data.dtype != wp.vec3:
        raise ValueError(f"data must be a wp.array[wp.vec3], got wp.array[{data.dtype}]")
    if epsilon > 0.0:
        rounded = twt.empty_int32_2d((data.shape[0], 3), device=data.device)
        wp.launch(
            kernel_unique.round_vec3_scaled,
            dim=data.shape[0],
            inputs=[data, wp.float32(1.0 / epsilon), rounded],
            device=data.device,
        )
        return hash_indices_rows(rounded)
    hashes = wp.empty(data.shape[0], dtype=wp.uint64, device=data.device)
    wp.launch(kernel_unique.pack_vec3, dim=data.shape[0], inputs=[data, hashes], device=data.device)
    return hashes


def hash_indices_rows(data: twt.Array2dInt32, max_index: int | None = None) -> wp.array[wp.uint64]:
    """
    Pack each row of non-negative ``int32`` values into a single ``uint64`` key.

    Each row is treated as digits in base ``max_index`` (or ``max(data)+1`` when
    ``max_index`` is omitted) and combined into one integer key per row. A global
    min/max over ``data`` (via ``triwarp.reduce.minmax``) validates the range and
    supplies the radix when it is inferred.

    Parameters
    ----------
    data
        ``(n, w)`` device array of ``int32``; every entry must be non-negative. If
        ``max_index`` is given, every entry must satisfy ``entry < max_index``.
    max_index
        Optional exclusive upper bound on entries and radix for packing; must be
        positive when provided. If ``None``, set to ``max(data) + 1`` after validation.

    Returns
    -------
    wp.array[wp.uint64]
        Length-``n`` array on ``data.device`` with one packed key per row.

    See Also
    --------
    [`hash_vector_rows`][triwarp.unique.hash_vector_rows]
    [`hash_rows`][triwarp.unique.hash_rows]
    """
    twt.ensure_ndim(data, 2, dtype=wp.int32)
    if max_index is not None and max_index <= 0:
        raise ValueError(f"max_index must be positive, got {max_index}")
    min_data, max_data = tw.reduce.minmax(data)
    if min_data < 0:
        raise ValueError(f"data must be non-negative, got a minimum of {min_data}")
    if max_index is not None and max_data >= max_index:
        raise ValueError(
            f"data must be less than max_index {max_index}, got a maximum of {max_data}"
        )
    if max_index is None:
        max_index = max_data + 1
    hashes = wp.empty(data.shape[0], dtype=wp.uint64, device=data.device)
    wp.launch(
        kernel_unique.pack_indices,
        dim=data.shape[0],
        inputs=[data, wp.uint64(max_index), hashes],
        device=data.device,
    )
    return hashes


@overload
def hash_rows(data: wp.array[wp.vec3]) -> wp.array[wp.uint64]: ...
@overload
def hash_rows(data: twt.Array2dInt32) -> wp.array[wp.uint64]: ...
@overload
def hash_rows(data: twt.Array2dFloat32) -> wp.array[wp.uint64]: ...
def hash_rows(
    data: wp.array[wp.vec3] | twt.Array2dInt32 | twt.Array2dFloat32,
) -> wp.array[wp.uint64]:
    """
    Pack each row of a 2D or ``wp.vec3`` array into a single ``uint64`` key.

    Dispatches to [`hash_vector_rows`][triwarp.unique.hash_vector_rows] for ``wp.vec3``
    (including ``float32`` arrays with width 3) or
    [`hash_indices_rows`][triwarp.unique.hash_indices_rows] for ``int32`` rows.

    Parameters
    ----------
    data
        ``(n, w)`` ``int32`` or ``float32`` array, or length-``n`` ``wp.vec3`` array.

    Returns
    -------
    wp.array[wp.uint64]
        Length-``n`` array on ``data.device`` with one packed key per row.

    Raises
    ------
    ValueError
        If ``data`` has an unsupported dtype or shape.
    """
    if data.dtype == wp.vec3:
        return hash_vector_rows(data)
    twt.ensure_ndim(data, 2)
    if data.dtype == wp.int32:
        return hash_indices_rows(data)
    if data.dtype == wp.float32:
        n = int(data.shape[0])
        if int(data.shape[1]) != 3:
            raise ValueError("float32 hash_rows currently requires width 3")
        vec = wp.empty(n, dtype=wp.vec3, device=data.device)
        wp.utils.array_cast(data, vec)
        return hash_vector_rows(vec)
    raise ValueError(f"hash_rows unsupported dtype {data.dtype}")


@overload
def unique_rows(
    data: wp.array[wp.vec3],
    *,
    return_inverse: Literal[False] = False,
    return_counts: Literal[False] = False,
) -> wp.array[wp.vec3]: ...
@overload
def unique_rows(
    data: wp.array[wp.vec3], *, return_inverse: Literal[True], return_counts: Literal[False] = False
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]: ...
@overload
def unique_rows(
    data: wp.array[wp.vec3], *, return_inverse: Literal[False] = False, return_counts: Literal[True]
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]: ...
@overload
def unique_rows(
    data: wp.array[wp.vec3], *, return_inverse: Literal[True], return_counts: Literal[True]
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]: ...
@overload
def unique_rows(
    data: twt.Array2dInt32,
    *,
    return_inverse: Literal[False] = False,
    return_counts: Literal[False] = False,
) -> twt.Array2dInt32: ...
@overload
def unique_rows(
    data: twt.Array2dInt32, *, return_inverse: Literal[True], return_counts: Literal[False] = False
) -> tuple[twt.Array2dInt32, wp.array[wp.int32]]: ...
@overload
def unique_rows(
    data: twt.Array2dInt32, *, return_inverse: Literal[False] = False, return_counts: Literal[True]
) -> tuple[twt.Array2dInt32, wp.array[wp.int32]]: ...
@overload
def unique_rows(
    data: twt.Array2dInt32, *, return_inverse: Literal[True], return_counts: Literal[True]
) -> tuple[twt.Array2dInt32, wp.array[wp.int32], wp.array[wp.int32]]: ...
@overload
def unique_rows(
    data: twt.Array2dFloat32,
    *,
    return_inverse: Literal[False] = False,
    return_counts: Literal[False] = False,
) -> twt.Array2dFloat32: ...
@overload
def unique_rows(
    data: twt.Array2dFloat32,
    *,
    return_inverse: Literal[True],
    return_counts: Literal[False] = False,
) -> tuple[twt.Array2dFloat32, wp.array[wp.int32]]: ...
@overload
def unique_rows(
    data: twt.Array2dFloat32,
    *,
    return_inverse: Literal[False] = False,
    return_counts: Literal[True],
) -> tuple[twt.Array2dFloat32, wp.array[wp.int32]]: ...
@overload
def unique_rows(
    data: twt.Array2dFloat32, *, return_inverse: Literal[True], return_counts: Literal[True]
) -> tuple[twt.Array2dFloat32, wp.array[wp.int32], wp.array[wp.int32]]: ...
def unique_rows(
    data: wp.array, *, return_inverse: bool = False, return_counts: bool = False
) -> (
    wp.array
    | tuple[wp.array, wp.array[wp.int32]]
    | tuple[wp.array, wp.array[wp.int32], wp.array[wp.int32]]
):
    """
    Find unique rows of a 2D Warp array (``numpy.unique`` along axis 0).

    Each row is hashed with [`hash_rows`][triwarp.unique.hash_rows], then deduplicated via the
    same open-addressing hash table as [`unique_1d`][triwarp.unique.unique_1d]. Unique rows are
    returned in
    sorted hash-key order; the representative row for each key is the first input
    row in that equivalence class.

    ``return_inverse`` maps each input row index to its slot in the sorted unique
    output. ``return_index`` is not supported.

    Parameters
    ----------
    data
        ``(n, w)`` ``int32`` or ``float32`` array, or length-``n`` ``wp.vec3`` array.
    return_inverse
        If ``True``, include the inverse mapping in the return tuple.
    return_counts
        If ``True``, include per-unique occurrence counts in the return tuple.

    Returns
    -------
    unique
        ``(n_unique, w)`` or length-``n_unique`` ``wp.vec3`` array on ``data.device``.
    inverse : wp.array[wp.int32], optional
        Present if ``return_inverse=True``. ``inverse[i]`` indexes ``unique``.
    counts : wp.array[wp.int32], optional
        Present if ``return_counts=True``. Occurrences per ``unique`` row.

    Raises
    ------
    ValueError
        If ``data`` is not rank-2 ``int32``/``float32`` or rank-1 ``wp.vec3``.
    """
    device = data.device
    n = int(data.shape[0])
    is_vec3 = data.dtype == wp.vec3

    if n == 0:
        if is_vec3:
            empty_unique = wp.empty(0, dtype=wp.vec3, device=device)
        elif data.dtype == wp.int32:
            empty_unique = twt.empty_int32_2d((0, int(data.shape[1])), device=device)
        else:
            empty_unique = twt.empty_float32_2d((0, int(data.shape[1])), device=device)
        empty_i32 = wp.empty(0, dtype=wp.int32, device=device)
        return _pack_unique_rows_result(
            empty_unique,
            inverse=empty_i32 if return_inverse else None,
            counts=empty_i32 if return_counts else None,
        )

    row_keys = hash_rows(data)
    keys_result = unique_1d(row_keys, return_inverse=True, return_counts=return_counts)
    if return_counts:
        _, inverse, counts = keys_result
    else:
        _, inverse = keys_result
        counts = None

    n_unique = int(tw.reduce.max(inverse)) + 1
    first_idx = wp.full(n_unique, wp.int32(n), dtype=wp.int32, device=device)
    wp.launch(
        kernel_edges.scatter_first_occurrence, dim=n, inputs=[inverse, first_idx], device=device
    )

    if is_vec3:
        unique_rows_out = gather(data, first_idx)
    else:
        twt.ensure_ndim(data, 2)
        n_cols = int(data.shape[1])
        if data.dtype == wp.int32:
            unique_rows_out = twt.empty_int32_2d((n_unique, n_cols), device=device)
        else:
            unique_rows_out = twt.empty_float32_2d((n_unique, n_cols), device=device)
        wp.launch(
            kernel_array.gather_rows,
            dim=n_unique,
            inputs=[data, first_idx, unique_rows_out],
            device=device,
        )

    if not return_inverse:
        inverse = None
    return _pack_unique_rows_result(unique_rows_out, inverse=inverse, counts=counts)


def _pack_unique_rows_result(
    unique: wp.array,
    *,
    inverse: wp.array[wp.int32] | None = None,
    counts: wp.array[wp.int32] | None = None,
) -> (
    wp.array
    | tuple[wp.array, wp.array[wp.int32]]
    | tuple[wp.array, wp.array[wp.int32], wp.array[wp.int32]]
):
    if inverse is not None and counts is not None:
        return unique, inverse, counts
    if inverse is not None:
        return unique, inverse
    if counts is not None:
        return unique, counts
    return unique
