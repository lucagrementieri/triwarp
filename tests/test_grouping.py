from __future__ import annotations

import numpy as np
import pytest
import trimesh as tm
import warp as wp

import triwarp as tw
import triwarp.typing as twt

group_test_data = (
    (wp.array([1, 3, 2, 3, 4, 4, 7, 5, -1, 5, 5], dtype=wp.int32), 2, wp.array([[1, 3], [4, 5]])),
    (
        wp.array([0, 1, 2, 1, 5, 6, 1, 0, 0, 0, 6, 4, 6], dtype=wp.uint64),
        3,
        wp.array([[1, 3, 6], [5, 10, 12]]),
    ),
    (
        wp.array([-1, 3, 2, -3, 4, 2, -1, 2, 2, 2], dtype=wp.int64),
        4,
        wp.empty((0, 4), dtype=wp.int32),
    ),
    # High-bit uint64 keys sort natively as unsigned (after low keys) in Warp 1.15.
    (wp.array([2**63 + 5, 1, 2**63 + 5, 1], dtype=wp.uint64), 2, wp.array([[1, 3], [0, 2]])),
)


@pytest.mark.parametrize(("values", "length", "expected"), group_test_data)
def test_group(
    device: str, values: wp.array[wp.Int], length: int, expected: twt.Array2dInt32
) -> None:
    values_wp = wp.array(values, dtype=values.dtype, device=device)
    groups_wp = tw.grouping.group(values_wp, length)
    assert np.array_equal(groups_wp.numpy(), expected.numpy())


def test_group_int_rows(device: str) -> None:
    data_np = np.array([[1, 2], [3, 4], [1, 2], [2, 1], [3, 4], [0, 1], [3, 4]], dtype=np.int32)
    length = 2
    groups_np = np.sort(tm.grouping.group_rows(data_np, require_count=length), axis=1)

    data_wp = wp.array(data_np, dtype=wp.int32, device=device)
    groups_wp = tw.grouping.group_int_rows(data_wp, length)
    assert np.array_equal(np.sort(groups_wp.numpy(), axis=1), groups_np)
