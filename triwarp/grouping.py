import warp as wp

from triwarp.kernels import grouping as kernel_grouping
import triwarp as tw


def hash_vector_rows(data: wp.array[wp.vec3]) -> wp.array[wp.uint64]:
    if data.dtype != wp.vec3:
        raise ValueError(f"data must be a wp.array[wp.vec3], got wp.array[{data.dtype}]")
    hashes = wp.empty(data.shape[0], dtype=wp.uint64, device=data.device)
    wp.launch(kernel_grouping.pack_vec3, dim=data.shape[0], inputs=[data, hashes], device=data.device)
    return hashes


def hash_indices_rows(data: wp.array2d[wp.int32], max_index: int | None = None) -> wp.array[wp.uint64]:
    if data.dtype != wp.int32:
        raise ValueError(f"data must be a wp.array2d[wp.int32], got wp.array2d[{data.dtype}]")
    if max_index is not None and max_index <= 0:
        raise ValueError(f"max_index must be positive, got {max_index}")
    min_data, max_data = tw.reduce.minmax(data)
    if min_data < 0:
        raise ValueError(f"data must be non-negative, got a minimum of {min_data}")
    if max_index is not None and max_data >= max_index:
        raise ValueError(f"data must be less than max_index {max_index}, got a maximum of {max_data}")
    if max_index is None:
        max_index = max_data + 1
    hashes = wp.empty(data.shape[0], dtype=wp.uint64, device=data.device)
    wp.launch(
        kernel_grouping.pack_indices, dim=data.shape[0], inputs=[data, wp.uint64(max_index), hashes], device=data.device
    )
    return hashes
