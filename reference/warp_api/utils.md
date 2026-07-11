# warp.utils API

Source: https://nvidia.github.io/warp/stable/api_reference/warp_utils.html (Warp 1.15.0)
> Regenerate after a Warp upgrade — see `reference/warp_api/REGENERATE.md`.

Python-scope utilities. Called as `wp.utils.<name>(...)`.

## Array Operations
- `array_cast(src, dst)` — cast elements from one array to another with a different dtype (see CLAUDE.md §4 for Python-scope dtype conversion).
- `array_inner(a, b)` — inner product of two arrays.
- `array_scan(in, out, inclusive=True)` — scan (prefix sum); since 1.15 supports 64-bit scalar and vector types (still no `wp.bool` — cast to int32 first).
- `array_sum(...)` — sum of array elements.

## Sorting
- `radix_sort_pairs(keys, values, count)` — sort key-value pairs; keys may be `int32`, `uint32`, `float32`, `int64`, `uint64`, or `float64` (unsigned + float64 since 1.15), values any 4- or 8-byte type; both buffers need `2 * count` capacity.
- `segmented_sort_pairs(keys, values, count, segment_start_indices)` — sort key-value pairs within segments; keys still `int32`/`float32` only, values `int32` (not extended in 1.15).
- `runlength_encode(...)` — run-length encode an array.

## Graph Coloring
- `GraphColoringAlgorithm` — graph coloring algorithm selection enum.
- `graph_coloring_assign(...)` — assign colors so no two adjacent nodes share a color.
- `graph_coloring_balance(...)` — balance the sizes of color groups.
- `graph_coloring_get_groups(...)` — convert node colors into per-color groups.

## Allocators
- `AllocatorRmm` — routes Warp device memory through RAPIDS Memory Manager (RMM).

## Misc
- `create_warp_function(...)` — create a Warp function from a Python function.
