# warp.utils API

Source: https://nvidia.github.io/warp/stable/api_reference/warp_utils.html (Warp 1.14.0)
> Regenerate after a Warp upgrade — see `reference/warp_api/REGENERATE.md`.

Python-scope utilities. Called as `wp.utils.<name>(...)`.

## Array Operations
- `array_cast(src, dst)` — cast elements from one array to another with a different dtype (see CLAUDE.md §4 for Python-scope dtype conversion).
- `array_inner(a, b)` — inner product of two arrays.
- `array_scan(...)` — scan (prefix sum) over an array.
- `array_sum(...)` — sum of array elements.

## Sorting
- `radix_sort_pairs(...)` — sort key-value pairs by radix sort.
- `segmented_sort_pairs(...)` — sort key-value pairs within segments.
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
