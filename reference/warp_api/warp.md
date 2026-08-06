# warp module API (Python scope)

Source: https://nvidia.github.io/warp/stable/api_reference/warp.html (Warp 1.16.0)
> Regenerate after a Warp upgrade — see `reference/warp_api/REGENERATE.md`.

Called as `wp.<name>(...)` at Python scope.

## Type Annotations
- `DeviceLike` — union type for device representation.
- `Float` / `Int` / `Scalar` — type variables for float/int/scalar types.
- `ref` — pass-by-reference parameter annotation.

## Data Types — Scalars
- `bool`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`.
- `float16`, `float32`, `float64`, `bfloat16`.
- `handle` — type for object handles (Mesh, Volume, BVH).

## Data Types — Vectors
- `vec2`/`vec3`/`vec4` — aliases for the float32 variants.
- Typed suffixes: `b` (int8), `s` (int16), `i` (int32), `l` (int64), `ub` (uint8), `us` (uint16), `ui` (uint32), `ul` (uint64), `h` (float16), `f` (float32), `d` (float64). E.g. `vec3d`, `vec4i`.

## Data Types — Matrices
- `mat22`/`mat33`/`mat44` — aliases for float32 variants.
- Suffixes `h`/`f`/`d`: e.g. `mat33d`, `mat44h`.
- `matrix_from_cols()` / `matrix_from_rows()` — construct matrix from column/row vectors.

## Data Types — Quaternions / Transforms / Spatial
- `quat` (=`quatf`), `quatd`, `quath`; `quat_between_vectors()` — quaternion rotating a to b.
- `transform` (=`transformf`), `transformd`, `transformh`; `transform_expand()` — expand 7-elem sequence.
- `spatial_vector`/`spatial_matrix` (=f variants) plus `d`/`h` suffixes.

## Arrays
- `array()` — multi-dimensional array of same-type values.
- `array1d/2d/3d/4d()` — create or annotate N-dimensional array.
- `fixedarray()` — stack-allocated fixed-size array.
- `tile()` / `tile_stack()` — Warp tile / tile-stack object.
- `clone()` — clone array, copying source memory.
- `copy()` — copy contents from source to destination.
- `empty()` / `empty_like()` — uninitialized array.
- `full()` / `full_like()` — array initialized to a given value.
- `ones()` / `ones_like()` — one-initialized array.
- `zeros()` / `zeros_like()` — zero-initialized array.
- `from_ptr()` — array from raw pointer (deprecated).

## Indexed Arrays
- `indexedarray()` — indexed access to a subset of a source array.
- `indexedarray1d/2d/3d/4d()` — create or annotate N-dimensional indexed array.

## Spatial Acceleration
- `Bvh()` / `BvhQuery()` / `BvhQueryTiled()` — bounding volume hierarchy + query state.
- `HashGrid()` / `HashGridQuery()` — hash-based spatial grid for neighbor queries.
  - `build(points, radius, groups=None)` — (1.16) an optional `int32` `groups` array assigns each
    point a group; `wp.hash_grid_query(..., group)` then restricts traversal to that group, so one
    grid can serve many independent point sets.
  - `reserve(num_points, with_groups=False)` — (1.16) pass `with_groups=True` to record grouped
    rebuilds inside a CUDA graph without a warm-up build.
- `Mesh()` — triangle mesh for collision / ray casting.
- `MeshQueryAABB()` / `MeshQueryAABBTiled()` / `MeshQueryPoint()` / `MeshQueryRay()` — mesh query state/outputs.
- `Volume()` — sparse volumetric (NanoVDB) data structure.
  - `allocate_by_tiles(tile_points, voxel_size=None, ..., rebuildable=False, max_tiles=None, max_lower_nodes=None, max_upper_nodes=None, status=None, point_mask=None)`
  - `allocate_by_voxels(voxel_points, voxel_size=None, ..., rebuildable=False, max_active_voxels=None, max_leaf_nodes=None, max_lower_nodes=None, max_upper_nodes=None, status=None, point_mask=None)`
  - `rebuild(points, status=None, point_mask=None)` — (1.16) refresh a **rebuildable** volume's
    topology in place at fixed capacity, with no new allocation. Works on CPU and is CUDA
    graph-capturable when memory-pool allocation is enabled; `warp.fem.Nanogrid` topologies built on
    a rebuildable volume refresh in place too.
  - `is_rebuildable` / `get_rebuild_info()` / `get_active_stats()` — (1.16) capacity introspection.

## Textures
- `Texture()` / `Texture1D/2D/3D()` — texture classes for hardware-accelerated sampling.
- `GLTextureResource()` — register/use an OpenGL texture.
- `TextureAddressMode()` / `TextureFilterMode()` / `TextureResourceFlags()` — texture config enums.

## Runtime
- `init()` — initialize the Warp runtime.
- `clear_kernel_cache()` / `clear_lto_cache()` — clear cache directories.
- `is_cpu_available()` / `is_cuda_available()` / `is_cubql_available()` — backend availability.
- `print_diagnostics()` — print build/runtime snapshot.

## Kernel Programming
- `kernel()` — decorator registering a Warp kernel.
- `func()` — decorator defining a Warp function callable from kernels.
- `func_grad()` / `func_replay()` / `func_native()` — register custom gradient/replay/native snippet.
- `grad()` — callable computing a function gradient.
- `address_of()` — return the address of an addressable expression.
- `map()` — map function over array elements.
- `overload()` — overload a generic kernel with argument types.
- `static()` — evaluate static expression and inline its result.
- `struct()` — decorator defining a Warp struct.
- `WarpCodegen*Error()` — codegen error classes (Attribute/Index/Key/Type/Value/general).

## Kernel Execution
- `launch()` — launch a kernel on a device.
- `launch_tiled()` — launch grid with trailing dim = block size.
- `Launch()` — launch data object for quick replay.
- `Kernel()` / `Function()` / `Module()` — kernel/function/module objects.
- `get_suggested_block_size()` — suggested CUDA block size for occupancy.
- `synchronize()` — synchronize CPU with outstanding CUDA work.

## Automatic Differentiation
- `Tape()` — record kernel launches for autodiff.

## Device Management
- `Device()` — device for allocation/launching.
- `ScopedDevice()` — context manager to temporarily change default device.
- `get_device()` / `set_device()` / `get_devices()` — device get/set/list.
- `get_preferred_device()` — preferred device (cuda:0 or cpu).
- `is_device_available()` — check device availability.
- `can_access()` — check device access to a resource.
- `get_cuda_device()` / `get_cuda_devices()` / `get_cuda_device_count()` — CUDA device queries.
- `get_cuda_driver_version()` / `get_cuda_toolkit_version()` / `get_cuda_supported_archs()` — CUDA versions/archs.
- `map_cuda_device()` / `unmap_cuda_device()` — assign/remove a CUDA device alias.
- `synchronize_device()` — synchronize CPU with device work.

## Module Management
- `get_module()` / `get_module_options()` / `set_module_options()` — module access/options.
- `load_module()` / `force_load()` — compile/load user kernels.
- `compile_aot_module()` / `load_aot_module()` — ahead-of-time compile/load.

## CUDA Streams / Events
- `Stream()` / `ScopedStream()` — CUDA stream wrapper + context manager. Properties include
  `is_complete`, `is_capturing`, `priority`, and `is_blocking` (1.16 — whether the stream blocks
  against the legacy default stream; relevant for streams borrowed from PyTorch).
- `get_stream()` / `set_stream()` — current stream get/set.
- `synchronize_stream()` / `wait_stream()` — stream synchronization.
- `Event()` — recordable CUDA event.
- `record_event()` / `wait_event()` / `synchronize_event()` — event ops.
- `get_event_elapsed_time()` — elapsed time between two events.

## CUDA Memory Management
- `Allocator()` / `ScopedAllocator()` — allocator protocol + context manager.
- `CudaManagedAllocator()` — CUDA managed-memory allocator.
- `MemoryKind()` — memory kind backing an array.
- `get_device_allocator()` / `set_device_allocator()` / `set_cuda_allocator()` — allocator get/set.
- `ScopedMempool()` / `ScopedMempoolAccess()` / `ScopedPeerAccess()` — mempool/peer context managers.
- `is_mempool_supported/enabled()` / `set_mempool_enabled()` — mempool support/toggle.
- `is_mempool_access_supported/enabled()` / `set_mempool_access_enabled()` — mempool access.
- `get_mempool_release_threshold()` / `set_mempool_release_threshold()` — release threshold.
- `get_mempool_used_mem_current()` / `get_mempool_used_mem_high()` — mempool usage / high-water mark.
- `get_cuda_max_cluster_dim()` — maximum thread-block cluster dimension.
- `is_peer_access_supported/enabled()` / `set_peer_access_enabled()` — peer device access.

## Graph Management
- `Graph()` — handle to a captured graph.
- `ScopedCapture()` — context manager for graph capture.
- `capture_begin()` / `capture_end()` / `capture_launch()` — capture + launch.
- `capture_if()` / `capture_while()` — dynamic branch / loop nodes. Since 1.16 an array whose last
  reference is dropped while a body graph is being recorded is no longer freed prematurely (was a
  use-after-free in 1.15).
- **Capturable built-in ops (1.16, experimental).** `wp.utils.array_sum()` / `wp.utils.array_inner()`
  capture and replay on both CPU and CUDA. `wp.HashGrid.build()`, `wp.Bvh.refit()` and
  `wp.Bvh.rebuild()` additionally capture on **CPU**, but replay-only — those captures cannot be
  serialized with `capture_save()`.
- `capture_save()` / `capture_load()` — serialize/load graph (.wrp).
- `capture_debug_dot_print()` — export graph to DOT.
- `CaptureMode()` — stream capture mode enum.
- `is_conditional_graph_supported()` — conditional graph node support.

## IPC
- `from_ipc_handle()` / `event_from_ipc_handle()` — array/event from IPC handle.

## Profiling
- `ScopedTimer()` / `TimingResult()` — timing context manager + result.
- `timing_begin()` / `timing_end()` / `timing_print()` — detailed activity timing.
- `TIMING_ALL/GRAPH/KERNEL/KERNEL_BUILTIN/MEMCPY/MEMSET()` — timing flags.
- `ScopedMemoryTracker()` / `print_memory_report()` — memory tracking.
- `cuda_profiler_start()` / `cuda_profiler_stop()` / `ScopedCudaProfiler(device=None)` — (1.16)
  bracket an external profiler's (Nsight, `ncu`) capture range from Python, equivalent to
  `cuProfilerStart`/`cuProfilerStop`. Acts on the given device's current CUDA context. Use it to
  exclude warm-up and kernel compilation from a profile.

## Logging
- `get_logger()` / `set_logger()` / `Logger()` — logger access/protocol.
- `ScopedLogLevel()` / `ScopedLogger()` — context managers.
- `LOG_DEBUG/INFO/WARNING/ERROR()` — log levels.

## Interop
- NumPy: `from_numpy()`, `dtype_from_numpy()`, `dtype_to_numpy()`.
- DLPack: `from_dlpack()`, `to_dlpack()`.
- PyTorch: `from_torch()`, `to_torch()`, `device_from/to_torch()`, `dtype_from/to_torch()`, `stream_from/to_torch()`.
- JAX: `from_jax()`, `to_jax()`, `jax_callable()`, `jax_kernel()`, `device_from/to_jax()`, `dtype_from/to_jax()`, `clear_jax_callable_graph_cache()`, `JaxCallableGraphMode()`, `JaxModulePreloadMode()`.
- Paddle: `from_paddle()`, `to_paddle()`, `device_from/to_paddle()`, `dtype_from/to_paddle()`, `stream_from_paddle()`.
- Omniverse Fabric: `fabricarray()`, `indexedfabricarray()`, `fabricarrayarray()`, `indexedfabricarrayarray()`.

## Constants
- `constant()` — declare a compile-time constant for kernels.
- Math constants (upper and lower case): `PI`, `TAU`, `E`, `PHI`, `HALF_PI`, `INF`, `NAN`, `LN2`, `LN10`, `LOG2E`, `LOG10E`.

## Configuration Modes
- `DeterministicMode()` — deterministic atomic-operation mode enum.

## Miscellaneous
- `MarchingCubes()` — reusable marching-cubes extraction context.
- `RegisteredGLBuffer()` — register a GL buffer with CUDA.
