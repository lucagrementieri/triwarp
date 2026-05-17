import warp as wp

from triwarp.kernels import grouping as kernel_grouping


def hash_vector_rows(data: wp.array[wp.vec3]) -> wp.array[wp.uint64]:
    if data.dtype != wp.vec3:
        raise ValueError("data must be a wp.array[wp.vec3]")
    out = wp.empty(data.shape[0], dtype=wp.uint64, device=data.device)
    wp.launch(kernel_grouping.pack_vec3, dim=data.shape[0], inputs=[data, out], device=data.device)
    return out
