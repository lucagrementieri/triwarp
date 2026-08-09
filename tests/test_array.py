"""Regression tests for ``triwarp.geometry`` against ``trimesh.geometry`` (CPU reference)."""

import numpy as np
import numpy.typing as npt
import pytest
import scipy.sparse
import trimesh as tm
import warp as wp

import triwarp as tw


def test_init_range(device: str) -> None:
    n = 8
    out_wp = tw.array.init_range(n, device)
    assert np.array_equal(out_wp.numpy(), np.arange(n, dtype=np.int32))


def test_init_range_zero(device: str) -> None:
    out_wp = tw.array.init_range(0, device)
    assert out_wp.shape == (0,)


def test_init_range_step(device: str) -> None:
    count, step = 6, 3
    out_wp = tw.array.init_range_step(count, step, device)
    assert np.array_equal(out_wp.numpy(), np.arange(0, count * step, step, dtype=np.int32))


def test_init_range_step_zero_count(device: str) -> None:
    out_wp = tw.array.init_range_step(0, 5, device)
    assert out_wp.shape == (0,)


def test_init_sort_pair_indices(device: str) -> None:
    n, fill = 5, -1
    out_wp = tw.array.init_sort_pair_indices(n, fill, device)
    expected_np = np.array([0, 1, 2, 3, 4, -1, -1, -1, -1, -1], dtype=np.int32)
    assert np.array_equal(out_wp.numpy(), expected_np)


def test_init_sort_pair_indices_zero(device: str) -> None:
    out_wp = tw.array.init_sort_pair_indices(0, -1, device)
    assert out_wp.shape == (0,)


def test_init_repeat_index(device: str) -> None:
    count, repeats = 9, 3
    out_wp = tw.array.init_repeat_index(count, repeats, device)
    expected_np = np.repeat(np.arange(count // repeats, dtype=np.int32), repeats)
    assert np.array_equal(out_wp.numpy(), expected_np)


def test_init_repeat_index_zero_count(device: str) -> None:
    out_wp = tw.array.init_repeat_index(0, 4, device)
    assert out_wp.shape == (0,)


def test_concatenate(device: str) -> None:
    parts = [
        wp.array([0, 3], dtype=wp.int32, device=device),
        wp.array([], dtype=wp.int32, device=device),
        wp.array([7, 12], dtype=wp.int32, device=device),
    ]
    out_wp = tw.array.concatenate(parts)
    assert np.array_equal(out_wp.numpy(), np.array([0, 3, 7, 12], dtype=np.int32))


def test_concatenate_single_returns_input(device: str) -> None:
    arr_wp = wp.array([1, 2], dtype=wp.int32, device=device)
    out_wp = tw.array.concatenate([arr_wp])
    assert out_wp is arr_wp


def test_concatenate_empty_segments(device: str) -> None:
    out_wp = tw.array.concatenate([wp.empty(0, dtype=wp.int32, device=device)])
    assert out_wp.shape == (0,)


def test_split_roundtrips_pack_1d_arrays(device: str) -> None:
    """``split(*pack_1d_arrays(arrays))`` recovers every input segment (the inverse pair)."""
    rng = np.random.default_rng(3)
    parts_np = [rng.integers(0, 100, size=size).astype(np.int32) for size in (4, 1, 0, 7)]
    parts_wp = [wp.array(part, dtype=wp.int32, device=device) for part in parts_np]

    flat_wp, offsets_wp = tw.array.pack_1d_arrays(parts_wp)
    segments_wp = tw.array.split(flat_wp, offsets_wp)

    assert len(segments_wp) == len(parts_np)
    for segment_wp, part_np in zip(segments_wp, parts_np, strict=True):
        assert np.array_equal(segment_wp.numpy(), part_np)


def test_split_views_share_storage_and_copies_do_not(device: str) -> None:
    """Default segments alias the packed buffer; ``copy=True`` detaches them."""
    flat_wp = wp.array(np.arange(6, dtype=np.int32), dtype=wp.int32, device=device)
    offsets_wp = wp.array(np.array([0, 2], dtype=np.int32), dtype=wp.int32, device=device)

    views = tw.array.split(flat_wp, offsets_wp)
    copies = tw.array.split(flat_wp, offsets_wp, copy=True)
    flat_wp.fill_(9)

    assert np.array_equal(views[0].numpy(), np.array([9, 9], dtype=np.int32))
    assert np.array_equal(copies[0].numpy(), np.array([0, 1], dtype=np.int32))
    assert np.array_equal(copies[1].numpy(), np.array([2, 3, 4, 5], dtype=np.int32))


def test_split_rejects_bad_offsets(device: str) -> None:
    flat_wp = wp.array(np.arange(4, dtype=np.int32), dtype=wp.int32, device=device)
    for bad in ([1, 2], [0, 3, 2], [0, 5]):
        offsets_wp = wp.array(np.array(bad, dtype=np.int32), dtype=wp.int32, device=device)
        with pytest.raises(ValueError, match="offsets must start at 0"):
            tw.array.split(flat_wp, offsets_wp)


def test_split_empty_offsets(device: str) -> None:
    flat_wp = wp.array(np.arange(3, dtype=np.int32), dtype=wp.int32, device=device)
    assert tw.array.split(flat_wp, wp.empty(0, dtype=wp.int32, device=device)) == []


@pytest.mark.parametrize("copy", [False, True])
def test_split_trailing_empty_segment(device: str, copy: bool) -> None:
    """A segment that is empty *at the end* of the buffer splits (Warp rejects ``arr[n:n]``)."""
    flat_wp = wp.array(np.arange(4, dtype=np.int32), dtype=wp.int32, device=device)
    offsets_wp = wp.array(np.array([0, 4, 4], dtype=np.int32), dtype=wp.int32, device=device)

    segments_wp = tw.array.split(flat_wp, offsets_wp, copy=copy)

    assert [int(segment.shape[0]) for segment in segments_wp] == [4, 0, 0]
    assert np.array_equal(segments_wp[0].numpy(), np.arange(4, dtype=np.int32))


@pytest.mark.parametrize(
    ("dtype_wp", "dtype_np"),
    [(wp.float32, np.float32), (wp.float64, np.float64), (wp.vec3, np.float32)],
    ids=["float32", "float64", "vec3"],
)
def test_allclose_matches_numpy(device: str, dtype_wp: type, dtype_np: type) -> None:
    """The tolerances must instantiate at the input's precision, not always at ``float32``."""
    rng = np.random.default_rng(95)
    shape = (6, 3) if dtype_wp is wp.vec3 else (18,)
    a_np = rng.standard_normal(shape).astype(dtype_np)
    a_wp = wp.array(a_np, dtype=dtype_wp, device=device)
    close_wp = wp.array(a_np + dtype_np(1e-9), dtype=dtype_wp, device=device)
    far_wp = wp.array(a_np + dtype_np(1e-2), dtype=dtype_wp, device=device)
    assert tw.array.allclose(a_wp, close_wp) == bool(np.allclose(a_np, a_np + dtype_np(1e-9)))
    assert tw.array.allclose(a_wp, far_wp) == bool(np.allclose(a_np, a_np + dtype_np(1e-2)))


def test_allclose_empty_is_true(device: str) -> None:
    empty_wp = wp.empty(0, dtype=wp.vec3, device=device)
    assert tw.array.allclose(empty_wp, empty_wp) is True


def test_allclose_rejects_mismatched_dtypes(device: str) -> None:
    a_wp = wp.zeros(3, dtype=wp.float32, device=device)
    b_wp = wp.zeros(3, dtype=wp.float64, device=device)
    with pytest.raises(ValueError, match="matching dtypes"):
        tw.array.allclose(a_wp, b_wp)


def test_sort_and_argsort_distinct_keys(device: str) -> None:
    """With distinct keys the permutation is unique, so it must equal ``numpy.argsort``."""
    rng = np.random.default_rng(31)
    keys_np = rng.permutation(64).astype(np.int32)
    keys_wp = wp.array(keys_np, dtype=wp.int32, device=device)
    sorted_keys_wp, order_wp = tw.array.sort_and_argsort(keys_wp)
    assert np.array_equal(sorted_keys_wp.numpy(), np.sort(keys_np))
    assert np.array_equal(order_wp.numpy(), np.argsort(keys_np).astype(np.int32))


def test_sort_and_argsort_duplicate_keys(device: str) -> None:
    """
    With ties the radix sort is not documented as stable, so only the permutation property holds.

    ``sorted_keys[i] == keys[order[i]]`` and ``order`` is a permutation of ``0..n-1`` -- that is
    everything the contract promises, and it is what distinguishes a correct sort from one that
    dropped or duplicated an entry.
    """
    rng = np.random.default_rng(32)
    keys_np = rng.integers(0, 8, size=64).astype(np.int32)
    keys_wp = wp.array(keys_np, dtype=wp.int32, device=device)
    sorted_keys_wp, order_wp = tw.array.sort_and_argsort(keys_wp)
    assert np.array_equal(sorted_keys_wp.numpy(), np.sort(keys_np))
    assert np.array_equal(np.sort(order_wp.numpy()), np.arange(64, dtype=np.int32))
    assert np.array_equal(keys_np[order_wp.numpy()], sorted_keys_wp.numpy())


def test_sort_and_argsort_empty(device: str) -> None:
    keys_wp = wp.empty(0, dtype=wp.int32, device=device)
    sorted_keys_wp, order_wp = tw.array.sort_and_argsort(keys_wp)
    assert sorted_keys_wp.shape == (0,)
    assert order_wp.shape == (0,)


def test_pack_1d_wp_arrays(device: str):
    parts = [
        wp.array([1, 2, 3], dtype=wp.int32, device=device),
        wp.array([], dtype=wp.int32, device=device),
        wp.array([4], dtype=wp.int32, device=device),
    ]
    flat, offsets = tw.array.pack_1d_arrays(parts)
    assert flat.device == device
    assert offsets.device == device
    assert np.array_equal(flat.numpy(), np.array([1, 2, 3, 4], dtype=np.int32))
    assert np.array_equal(offsets.numpy(), np.array([0, 3, 3], dtype=np.int32))


def test_pack_1d_wp_arrays_vec3(device: str):
    parts = [
        wp.array([wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0)], dtype=wp.vec3, device=device),
        wp.array([wp.vec3(2.0, 2.0, 2.0)], dtype=wp.vec3, device=device),
    ]
    flat, offsets = tw.array.pack_1d_arrays(parts)
    exp = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 2.0, 2.0]], dtype=np.float32)
    got = flat.numpy().reshape(-1, 3)
    assert np.allclose(got, exp, rtol=1e-5, atol=1e-5)
    assert np.array_equal(offsets.numpy(), np.array([0, 2], dtype=np.int32))


def test_pack_1d_wp_arrays_dtype_mismatch(device: str):
    parts = [
        wp.array([1], dtype=wp.int32, device=device),
        wp.array([2.0], dtype=wp.float32, device=device),
    ]
    with pytest.raises(ValueError, match="same dtype"):
        tw.array.pack_1d_arrays(parts)


def test_sort_rows(device: str):
    rng = np.random.default_rng(42)
    data = rng.random(size=(32, 4), dtype=np.float32)
    sorted_data_np = np.sort(data, axis=1)

    data_wp = wp.array(data, dtype=wp.float32, device=device)
    tw.array.sort_rows(data_wp)
    assert np.array_equal(data_wp.numpy(), sorted_data_np)


@pytest.mark.parametrize("data", [None, np.arange(1, 13, dtype=np.int32)])
def test_index_sparse(data: npt.NDArray[np.int32] | None, device: str):
    n_rows = 4
    indices = np.array([[0, 1, 2], [0, 3, 1], [1, 2, 3], [0, 2, 3]])

    result_np = tm.geometry.index_sparse(n_rows, indices, data).tocsr()

    indices_wp = wp.array(indices, dtype=wp.int32, device=device)
    data_wp = wp.array(data, dtype=wp.int32, device=device) if data is not None else None

    result_wp = tw.array.index_sparse(n_rows, indices_wp, data_wp)
    assert result_wp.values.dtype == (data_wp.dtype if data_wp is not None else wp.float32)
    assert np.array_equal(result_wp.offsets.numpy(), result_np.indptr)
    assert np.array_equal(result_wp.values.numpy(), result_np.data)


def test_index_sparse_repeated_indices(device: str):
    n_rows = 4
    indices = np.array([[0, 1, 0], [3, 3, 1], [1, 2, 3]])
    data = np.ones(9, dtype=np.int32)
    result_np = tm.geometry.index_sparse(n_rows, indices, data).tocsr()

    indices_wp = wp.array(indices, dtype=wp.int32, device=device)
    data_wp = wp.array(data, dtype=wp.int32, device=device)

    result_wp = tw.array.index_sparse(n_rows, indices_wp, data_wp, dtype=wp.float64)
    assert result_wp.values.dtype == wp.float64
    result_csr = scipy.sparse.csr_matrix(
        (result_wp.values.numpy(), result_wp.columns.numpy(), result_wp.offsets.numpy()),
        shape=result_wp.shape,
    )
    assert np.array_equal(result_csr.todense(), result_np.todense())


def test_isin_1d(device: str) -> None:
    rng = np.random.default_rng(42)
    elements_np = rng.integers(0, 20, size=50, dtype=np.int32)
    test_np = rng.choice(20, size=8, replace=False).astype(np.int32)

    elements_wp = wp.array(elements_np, dtype=wp.int32, device=device)
    test_wp = wp.array(test_np, dtype=wp.int32, device=device)
    mask_wp = tw.array.isin(elements_wp, test_wp)
    mask_np = np.isin(elements_np, test_np)
    assert np.array_equal(mask_wp.numpy(), mask_np)


def test_isin_2d(device: str) -> None:
    rng = np.random.default_rng(7)
    elements_np = rng.integers(0, 15, size=(12, 3), dtype=np.int32)
    test_np = rng.choice(15, size=5, replace=False).astype(np.int32)

    elements_wp = wp.array(elements_np, dtype=wp.int32, device=device)
    test_wp = wp.array(test_np, dtype=wp.int32, device=device)
    mask_wp = tw.array.isin(elements_wp, test_wp)
    mask_np = np.isin(elements_np, test_np)
    assert np.array_equal(mask_wp.numpy(), mask_np)


def test_isin_empty_test(device: str) -> None:
    elements_np = np.array([0, 1, 2, 3], dtype=np.int32)
    elements_wp = wp.array(elements_np, dtype=wp.int32, device=device)
    test_wp = wp.empty(0, dtype=wp.int32, device=device)
    mask_wp = tw.array.isin(elements_wp, test_wp)
    mask_ref_np = np.zeros_like(elements_np, dtype=bool)
    assert np.array_equal(mask_wp.numpy(), mask_ref_np)


@pytest.mark.parametrize("shape", [(4, 3, 2), (2, 2, 3, 5)], ids=["rank3", "rank4"])
def test_isin_higher_rank(device: str, shape: tuple[int, ...]) -> None:
    """Membership is a per-element predicate, so any rank round-trips through flatten/reshape."""
    rng = np.random.default_rng(13)
    elements_np = rng.integers(0, 15, size=shape, dtype=np.int32)
    test_np = rng.choice(15, size=5, replace=False).astype(np.int32)

    elements_wp = wp.array(elements_np, dtype=wp.int32, device=device)
    test_wp = wp.array(test_np, dtype=wp.int32, device=device)
    mask_wp = tw.array.isin(elements_wp, test_wp)
    assert mask_wp.shape == shape
    assert np.array_equal(mask_wp.numpy(), np.isin(elements_np, test_np))


def test_isin_sparse_large_indices(device: str) -> None:
    """Forces sort + binary-search path (max index >> len(test_elements))."""
    elements_np = np.array([1, 1_000_000, 2, 999_999, 3], dtype=np.int32)
    test_np = np.array([1, 2, 3], dtype=np.int32)
    elements_wp = wp.array(elements_np, dtype=wp.int32, device=device)
    test_wp = wp.array(test_np, dtype=wp.int32, device=device)
    mask_wp = tw.array.isin(elements_wp, test_wp)
    mask_np = np.isin(elements_np, test_np)
    assert np.array_equal(mask_wp.numpy(), mask_np)


def test_isin_negative_values(device: str) -> None:
    """
    Regression: the membership table is anchored at the global minimum, not at zero.

    Anchored at zero this returned a *silent wrong answer* -- ``mark_membership_mask`` drops a
    negative test value as out of range, so ``-5`` read back as absent while ``3`` was found.
    """
    elements_np = np.array([-5, 3, -1, 4, -5], dtype=np.int32)
    test_np = np.array([3, -5], dtype=np.int32)
    elements_wp = wp.array(elements_np, dtype=wp.int32, device=device)
    test_wp = wp.array(test_np, dtype=wp.int32, device=device)
    assert np.array_equal(
        tw.array.isin(elements_wp, test_wp).numpy(), np.isin(elements_np, test_np)
    )


_ISIN_DTYPES = [
    (wp.int8, np.int8),
    (wp.uint8, np.uint8),
    (wp.int16, np.int16),
    (wp.uint16, np.uint16),
    (wp.int32, np.int32),
    (wp.uint32, np.uint32),
    (wp.int64, np.int64),
    (wp.uint64, np.uint64),
]


@pytest.mark.parametrize(
    ("dtype_wp", "dtype_np"), _ISIN_DTYPES, ids=lambda d: getattr(d, "__name__", "")
)
def test_isin_integer_dtypes(device: str, dtype_wp: type, dtype_np: type) -> None:
    """Every Warp integer dtype, on the membership-table branch. Sub-32-bit ones are widened."""
    info = np.iinfo(dtype_np)
    low, high = max(int(info.min), -60), min(int(info.max), 60)
    rng = np.random.default_rng(19)
    elements_np = rng.integers(low, high + 1, size=97).astype(dtype_np)
    test_np = rng.integers(low, high + 1, size=11).astype(dtype_np)

    elements_wp = wp.array(elements_np, dtype=dtype_wp, device=device)
    test_wp = wp.array(test_np, dtype=dtype_wp, device=device)
    assert np.array_equal(
        tw.array.isin(elements_wp, test_wp).numpy(), np.isin(elements_np, test_np)
    )


@pytest.mark.parametrize(
    ("dtype_wp", "dtype_np"), _ISIN_DTYPES, ids=lambda d: getattr(d, "__name__", "")
)
def test_isin_integer_dtypes_sparse(device: str, dtype_wp: type, dtype_np: type) -> None:
    """The same dtypes on the sort + binary-search branch, where the span dwarfs the test set."""
    span = min(int(np.iinfo(dtype_np).max), 2**40)
    test_np = np.array([0, span // 3, span // 2, span], dtype=dtype_np)
    elements_np = np.array([0, 1, span // 2, span], dtype=dtype_np)

    elements_wp = wp.array(elements_np, dtype=dtype_wp, device=device)
    test_wp = wp.array(test_np, dtype=dtype_wp, device=device)
    assert np.array_equal(
        tw.array.isin(elements_wp, test_wp).numpy(), np.isin(elements_np, test_np)
    )


def test_isin_rejects_mismatched_and_non_integer_dtypes(device: str) -> None:
    """Both arrays must share one integer dtype: the kernels are instantiated per dtype."""
    elements_wp = wp.array(np.arange(4, dtype=np.int64), dtype=wp.int64, device=device)
    with pytest.raises(TypeError, match="one dtype"):
        tw.array.isin(elements_wp, wp.array(np.array([1], np.int32), dtype=wp.int32, device=device))
    floats_wp = wp.array(np.zeros(3, np.float32), dtype=wp.float32, device=device)
    with pytest.raises(TypeError, match="integer dtype"):
        tw.array.isin(floats_wp, floats_wp)


def test_flatnonzero(device: str) -> None:
    rng = np.random.default_rng(11)
    mask_np = rng.choice([False, True], size=64, replace=True)
    mask_wp = wp.array(mask_np, dtype=wp.bool, device=device)
    indices_wp = tw.array.flatnonzero(mask_wp)
    indices_ref_np = np.flatnonzero(mask_np).astype(np.int32)
    assert np.array_equal(indices_wp.numpy(), indices_ref_np)


def test_flatnonzero_empty(device: str) -> None:
    mask_wp = wp.array(np.zeros(8, dtype=bool), dtype=wp.bool, device=device)
    indices_wp = tw.array.flatnonzero(mask_wp)
    assert indices_wp.shape == (0,)


@pytest.mark.parametrize(
    ("dtype_wp", "dtype_np"),
    [(wp.int32, np.int32), (wp.int8, np.int8), (wp.float32, np.float32)],
    ids=["int32", "int8", "float32"],
)
def test_flatnonzero_nonboolean(device: str, dtype_wp: type, dtype_np: type) -> None:
    """
    Any non-zero value selects its index, matching ``numpy.flatnonzero``.

    Values above one and negative values both count -- the flag pass must map to 0/1 rather than
    cast, or the prefix sum would total the values instead of counting them.
    """
    values_np = np.array([0, 3, 0, -2, 1, 0, 7, -1], dtype=dtype_np)
    values_wp = wp.array(values_np, dtype=dtype_wp, device=device)
    indices_wp = tw.array.flatnonzero(values_wp)
    assert np.array_equal(indices_wp.numpy(), np.flatnonzero(values_np).astype(np.int32))


def test_flatnonzero_rejects_rank2(device: str) -> None:
    values_wp = wp.array(np.zeros((3, 4), dtype=np.int32), dtype=wp.int32, device=device)
    with pytest.raises(ValueError, match="1D"):
        tw.array.flatnonzero(values_wp)


@pytest.mark.parametrize("n", [16, 257], ids=["small", "large"])
def test_flatnonzero_indices_to_mask_round_trip(device: str, n: int) -> None:
    """
    The two are inverses: each recovers the other's input.

    ``flatnonzero`` returns ascending unique indices, so the mask direction round-trips exactly
    while the index direction round-trips up to ``numpy.unique`` (duplicate indices mark the same
    slot once).
    """
    rng = np.random.default_rng(23)
    mask_np = rng.random(n) < 0.3
    mask_wp = wp.array(mask_np, dtype=wp.bool, device=device)
    assert np.array_equal(
        tw.array.indices_to_mask(tw.array.flatnonzero(mask_wp), n).numpy(), mask_np
    )

    indices_np = rng.choice(n, size=n // 4, replace=True).astype(np.int32)
    indices_wp = wp.array(indices_np, dtype=wp.int32, device=device)
    assert np.array_equal(
        tw.array.flatnonzero(tw.array.indices_to_mask(indices_wp, n)).numpy(), np.unique(indices_np)
    )


@pytest.mark.parametrize("include_total", [False, True], ids=["plain", "total"])
def test_counts_to_offsets(device: str, include_total: bool) -> None:
    """
    Exclusive prefix sum, in both offset conventions.

    The total-terminated form is the length-``n + 1`` CSR array ``segmented_sort_pairs`` wants; it
    the same buffer, so the two must agree on their common prefix and the total.
    """
    rng = np.random.default_rng(41)
    counts_np = rng.integers(0, 7, size=32).astype(np.int32)
    counts_wp = wp.array(counts_np, dtype=wp.int32, device=device)
    offsets_wp, total = tw.array.counts_to_offsets(counts_wp, include_total=include_total)

    exclusive_np = np.concatenate([[0], np.cumsum(counts_np)]).astype(np.int32)
    assert total == int(counts_np.sum())
    assert np.array_equal(offsets_wp.numpy(), exclusive_np if include_total else exclusive_np[:-1])


@pytest.mark.parametrize("include_total", [False, True], ids=["plain", "total"])
def test_counts_to_offsets_empty(device: str, include_total: bool) -> None:
    counts_wp = wp.empty(0, dtype=wp.int32, device=device)
    offsets_wp, total = tw.array.counts_to_offsets(counts_wp, include_total=include_total)
    assert total == 0
    assert np.array_equal(offsets_wp.numpy(), np.zeros(1 if include_total else 0, dtype=np.int32))


def test_remap_indices_passes_negative_sentinels_through(device: str) -> None:
    """
    ``-1`` entries survive the remap unchanged; everything else reads the table.

    This is the behavior that separates ``remap_indices`` from a plain ``gather`` — a gather
    would read out of bounds on the sentinel — and what ``repair``'s sentinel-preserving face
    remaps rely on.
    """
    indices_wp = wp.array(
        np.array([2, -1, 0, 1, -1], dtype=np.int32), dtype=wp.int32, device=device
    )
    remap_wp = wp.array(np.array([10, 11, 12], dtype=np.int32), dtype=wp.int32, device=device)
    remapped_wp = tw.array.remap_indices(indices_wp, remap_wp)
    assert np.array_equal(remapped_wp.numpy(), np.array([12, -1, 10, 11, -1], dtype=np.int32))


def test_remap_indices_empty(device: str) -> None:
    indices_wp = wp.empty(0, dtype=wp.int32, device=device)
    remap_wp = wp.array(np.array([0, 1], dtype=np.int32), dtype=wp.int32, device=device)
    assert tw.array.remap_indices(indices_wp, remap_wp).shape == (0,)


def test_indices_to_mask_empty(device: str) -> None:
    indices_wp = wp.empty(0, dtype=wp.int32, device=device)
    mask_wp = tw.array.indices_to_mask(indices_wp, 5)
    assert np.array_equal(mask_wp.numpy(), np.zeros(5, dtype=bool))


def test_gather_1d(device: str) -> None:
    rng = np.random.default_rng(5)
    values_np = rng.integers(0, 1000, size=32, dtype=np.int32)
    indices_np = rng.integers(0, 32, size=10, dtype=np.int32)

    values_wp = wp.array(values_np, dtype=wp.int32, device=device)
    indices_wp = wp.array(indices_np, dtype=wp.int32, device=device)
    gathered_wp = tw.array.gather(values_wp, indices_wp)

    assert np.array_equal(gathered_wp.numpy(), values_np[indices_np])


def test_gather_2d_rows(device: str) -> None:
    rng = np.random.default_rng(6)
    rows_np = rng.integers(0, 1000, size=(20, 2), dtype=np.int32)
    indices_np = rng.integers(0, 20, size=7, dtype=np.int32)

    rows_wp = wp.array(rows_np, dtype=wp.int32, device=device)
    indices_wp = wp.array(indices_np, dtype=wp.int32, device=device)
    gathered_wp = tw.array.gather(rows_wp, indices_wp)

    assert gathered_wp.shape == (7, 2)
    assert np.array_equal(gathered_wp.numpy(), rows_np[indices_np])


def test_gather_vec3(device: str) -> None:
    rng = np.random.default_rng(7)
    points_np = rng.standard_normal((16, 3)).astype(np.float32)
    indices_np = rng.integers(0, 16, size=5, dtype=np.int32)

    points_wp = wp.array(points_np, dtype=wp.vec3, device=device)
    indices_wp = wp.array(indices_np, dtype=wp.int32, device=device)
    gathered_wp = tw.array.gather(points_wp, indices_wp)

    assert np.array_equal(gathered_wp.numpy(), points_np[indices_np])


def test_gather_empty_indices(device: str) -> None:
    rows_wp = wp.array(np.zeros((4, 2), dtype=np.int32), dtype=wp.int32, device=device)
    indices_wp = wp.empty(0, dtype=wp.int32, device=device)
    gathered_wp = tw.array.gather(rows_wp, indices_wp)
    assert gathered_wp.shape == (0, 2)


bitcast_test_data = (
    wp.array([-128, -127, -1, 0, 1, 127], dtype=wp.int8),
    wp.array([255, 0, 254, 1], dtype=wp.uint8),
    wp.array([-32768, -32767, 10, 0, -1, 32767, 32765], dtype=wp.int16),
    wp.array([65535, 32, 65534, 0, 1], dtype=wp.uint16),
    wp.array([-2147483648, -2147483647, 10, 0, -1, 2147483647, 2147483645], dtype=wp.int32),
    wp.array([4294967295, 32, 4294967294, 0, 1], dtype=wp.uint32),
    wp.array(
        [
            -9223372036854775808,
            -9223372036854775807,
            10,
            0,
            -1,
            9223372036854775806,
            9223372036854775807,
        ],
        dtype=wp.int64,
    ),
    wp.array([18446744073709551615, 65535, 18446744073709551614, 131070, 1], dtype=wp.uint64),
    wp.array(
        np.asarray(
            [
                0x0000,  # +0
                0x8000,  # -0
                0x7F80,  # +inf
                0xFF80,  # -inf
                0x7FFF,  # quiet NaN (preserved through float32 widen/narrow on CUDA)
                0x7F7F,  # largest finite
                0xFF7F,  # smallest (most negative) finite
                0x0080,  # smallest subnormal
                0x8100,  # negative subnormal
            ],
            dtype=np.uint16,
        ),
        dtype=wp.bfloat16,
    ),
    wp.array(
        [
            np.finfo(np.float16).min,
            np.finfo(np.float16).max,
            -np.finfo(np.float16).max,
            np.finfo(np.float16).smallest_subnormal,
            -np.finfo(np.float16).smallest_subnormal,
            0.0,
            -0.0,
            np.inf,
            -np.inf,
            np.nan,
        ],
        dtype=wp.float16,
    ),
    wp.array(
        [
            np.finfo(np.float32).min,
            np.finfo(np.float32).max,
            -np.finfo(np.float32).max,
            np.finfo(np.float32).smallest_subnormal,
            -np.finfo(np.float32).smallest_subnormal,
            0.0,
            -0.0,
            np.inf,
            -np.inf,
            np.nan,
        ],
        dtype=wp.float32,
    ),
    wp.array(
        [
            np.finfo(np.float64).min,
            np.finfo(np.float64).max,
            -np.finfo(np.float64).max,
            np.finfo(np.float64).smallest_subnormal,
            -np.finfo(np.float64).smallest_subnormal,
            0.0,
            -0.0,
            np.inf,
            -np.inf,
            np.nan,
        ],
        dtype=wp.float64,
    ),
)


@pytest.mark.parametrize(
    "data", bitcast_test_data, ids=[a.dtype.__name__ for a in bitcast_test_data]
)
def test_bitcast_int_reciprocity(device: str, data: wp.array[wp.Scalar]):
    as_int = tw.array.bitcast_to_int(data.to(device))
    recovered = tw.array.bitcast_from_int(as_int, data.dtype)
    assert np.array_equal(data.numpy(), recovered.numpy(), equal_nan=True)
