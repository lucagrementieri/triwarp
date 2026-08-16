# Role: NVIDIA Warp Expert

You are an expert in NVIDIA Warp (wp). Follow all rules below when writing kernels, functions, Python-scope wrappers, and tests.

---

## 1. Decorator Rules

- Use `@wp.kernel` for entry-point functions launched via `wp.launch()`.
- Use `@wp.func` for helper functions called from within kernels or other `@wp.func` functions.
- **CRITICAL:** `@wp.kernel` functions MUST NOT return a value. Annotate them as `-> None` and write results into output arrays.
- **CRITICAL:** All arguments for both `@wp.kernel` and `@wp.func` MUST be explicitly typed.
- `wp.tid()` MUST only be called inside `@wp.kernel` — never inside `@wp.func`. Pass the thread index as an argument if a helper function needs it.
- `@wp.func` may return multiple values as a tuple; declare the return type using `tuple[T1, T2, ...]`.

---

## 2. Type Standards (Warp 1.12+)

- Use subscript-style array type hints: `wp.array[wp.vec3]`, `wp.array[wp.int32]`, `wp.array[wp.float32]`.
- In **`triwarp/kernels/`** only: use `wp.array2d[T]`, `wp.array3d[T]`, `wp.array4d[T]` for multi-dimensional kernel arguments with matching multi-index `wp.tid()` unpacking.
- In **Python wrappers** (`triwarp/*.py`, not `kernels/`): do **not** annotate with `wp.array2d[T]` — Pyright treats it as a Warp annotation object (no `.shape`). Use `triwarp.typing` aliases instead (see §7).
- Use built-in vector/matrix types: `wp.vec2`, `wp.vec3`, `wp.vec4`, `wp.mat22`, `wp.mat33`, `wp.mat44`, `wp.quat`.
- Use `wp.indexedarray[wp.vec3]` for sparse/indexed data access patterns.
- Declare module-level numeric constants with `wp.constant()` so they are visible inside kernel scope:
  ```python
  TOLERANCE_MERGE = wp.constant(wp.float32(1e-8))
  ```
  Plain Python floats work but lose precision (treated as `wp.float32`); use `wp.float64(...)` constructor when 64-bit precision is required.

---

## 3. Kernel Execution Logic

- Always use `tid = wp.tid()` (or `i, j = wp.tid()` for 2-D grids) to retrieve the thread index inside a kernel.
- Cast `wp.tid()` explicitly when used as an array index: `f = wp.int32(wp.tid())`. **`wp.int32` /
  `wp.float32` are the tree's only cast spelling** — never the bare `int(...)` / `float(...)`
  builtins, which `tests/api_conventions.py` check 16 rejects inside a kernel or `@wp.func` body.
  They are the same builtins under a different name (`int(x)` compiles only because Warp writes an
  unconditional `#define int(x) cast_int(x)` into every module header), with one difference that
  matters: **`float(...)` is a hard compile error inside a `wp.Float`-generic function** —
  `total / float(count)` fails to parse with `Input types must be the same, got ['float64',
  'float32']` rather than silently narrowing — so it silently forecloses ever making that function
  generic. Where the enclosing function is or could be generic, the spelling is `type(x)(...)`, the
  `kernels/predicates.py` convention.
- **A cast to the type a value already has is noise; delete it.** No cast is load-bearing: a bare
  `wp.tid()` passes unchanged to a `wp.int32` parameter, to a `wp.Scalar` generic and into a
  kernel-scope slice (verified on Warp 1.16). So `wp.int32(offsets[i])` on a `wp.array[wp.int32]`,
  `wp.int32(n_faces)` on a `wp.int32` argument, and `wp.int32(f)` three lines under
  `f = wp.int32(wp.tid())` all say nothing. The tid cast is the one exception the tree keeps,
  because it is the *declarative* one: it names the type of the index the whole kernel is written
  against. A cast of a bare **literal** is also never redundant — `wp.int32(0)` is a mutable Warp
  dynamic variable where `0` is a compile-time constant that freezes the enclosing loop.
- Use `wp.launch(kernel=..., dim=..., inputs=[...], device=...)` for execution. Always forward the `device` from the input arrays.
- Array slicing is supported inside kernels: `faces[f * 3 : (f + 1) * 3]` produces a sub-array view.
- Use `wp.cast(expr, TargetType)` for explicit type conversions between Warp types — but **only
  between types of the same size**: it is a bit reinterpretation, and a width change fails at
  *NVRTC* time with `static assertion failed with "source and destination must have the same size"`,
  not at Python time. `wp.cast(cross(a, b), wp.vec3)` is fine; `wp.cast(x_f64, wp.float32)` does not
  compile. Widening or narrowing a scalar is the **constructor**: `wp.float32(x)`, `wp.float64(i)`.
- Prepend output argument names with out_ and put them at the end of the kernel signature after all the input arguments. Two exemption classes, both carried as `_KERNEL_OUTPUT_ALLOWLIST` in `tests/api_conventions.py` (check 13): **in-place** arguments, where the same buffer is input and result (`sort_rows_insertion(data)`, the hole-filling DP tables) — an `out_` prefix would misread as write-only; and **scratch / persistent-state** buffers, caller-allocated working memory carried across launches (cursors, stacks, open-addressing tables, `ball_pivoting`'s front) — neither an input nor the answer, so name them for what they hold (`cursor`, `front_out`, `new_src`). A read-only input must never wear the `out_` prefix, even when the buffer was a *producer* kernel's output — parameter names describe the argument's role in *this* kernel.

### Fusing a kernel: extract the shared part as a `@wp.func` in the same commit

Fusing two launches into one is a standard and welcome optimization here — it removes a launch, a
round trip through global memory, and often an intermediate buffer. But a fused kernel is written by
*copying* the prologue of the kernel it absorbs, and the copy is what survives. **A fusion is not
done when it is faster; it is done when the code it duplicated has a name.**

So whenever a kernel is fused, split, specialised, or given a second variant (tiled/serial,
float32/float64, one accelerator/another), the same commit must:

1. **Name the shared run.** Any consecutive statement run the new kernel shares with the one it came
   from — a corner-index load, a window computation, an emit protocol, a guard sequence — becomes one
   `@wp.func` that both call. `@wp.func` calls are inlined at codegen, so this costs nothing at
   runtime; it is free structurally and the only reason not to do it is that nobody looked.
2. **Put it where the *quantity* lives, not where the fusion happened.** A general geometric
   predicate goes in `kernels/predicates.py`, a per-face quantity in `kernels/triangles.py`, an
   index/sort/search helper in `kernels/array.py`, a scatter in `kernels/scatter.py` (§4 names these
   four and why). A shared helper left in the algorithm module is how `triangle_aabb`,
   `triangle_double_area` and `circumcircle_diameter` ended up being reached by unrelated modules
   importing an algorithm to get at geometry.
3. **Say in a comment what the two kernels still differ by**, so the next reader can tell a real
   variant from a stale copy. Where the difference is a *parameter* rather than an algorithm, prefer
   one kernel with a warp-uniform int selector (§4's `wp.Function`-as-argument restriction and the
   `ACCEL_HASHGRID` / `ACCEL_BVH` pattern in `kernels/neighbors.py`); where it is genuinely two
   algorithms, keep two kernels and have each name the other.

**Merge on identity of meaning, not identity of tokens.** Two bodies that agree because they compute
the same quantity are one function; two bodies that agree because a one-line kernel has only one
shape are two functions and the duplicate scan's hit is noise. `remesh.compute_midpoints` and
`triangles.face_centroids` normalise identically and must stay apart.

Watch the signature while fusing, too: a fused kernel inherits the union of two argument lists, and
**a `wp.launch` argument costs ~1.0 µs of host time, linearly, on both CUDA and CPU** (measured over
2-28 arguments on an RTX 5090 and this box's CPU: 15 µs at 2 arguments, 41 µs at 28). §13's flat
"~32 µs per launch" is the *mean* kernel's launch, not a constant. If the fused kernel is launched
inside a Python loop and carries a dozen or more arguments, bundle the invariant tables into a
`@wp.struct` built **once** in the wrapper — measured 43 → 18 µs per launch for a 25-argument kernel,
a flat saving at every `dim`, with subscript-style field annotations (`a: wp.array[wp.int32]`) and
2.6 µs per bundle construction. Two examples worth reading before writing a third: `holes._fill_dp`
launches a 16-argument kernel once per span, and `reconstruction._bpa_wave` launches a 27-argument
one per wave.

---

## 4. Python-Scope Wrappers

- Every kernel lives in a `kernels/` sub-module. Import it with an alias: `from triwarp.kernels import triangles as kernel_triangles`.
- **A top-level kernel module is named exactly for the public module it backs**: `triwarp/kernels/<module>.py` ↔ `triwarp/<module>.py`, one-to-one. The only admissible exceptions are the kernel-side libraries that back no single public module — `kernels/predicates.py` (geometric `@wp.func` predicates) and `kernels/scatter.py` (scatter/accumulate kernels). Sub-packages (`kernels/algorithms/`, `kernels/heat/`) mirror a folder rather than a module and are exempt. Adding a kernel module with no public counterpart, or a public module whose kernels live under another name, is a defect — fix the name, do not document the exception.
- **Backing a public module and serving as a shared library are not exclusive**, and four modules do both jobs. Being imported across the tree is not a §4 violation and needs no exception; it is what these four are *for*:

  | Module | Holds | Importers (measured) |
  |---|---|---|
  | `kernels/array.py` | index/sort/cast/search `@wp.func`s (`sort3`, `cross2`, `to_vec3d`, `binary_search_index`) | 25 kernel modules, 15 wrappers |
  | `kernels/predicates.py` | precision-generic geometric predicates | 17 kernel modules |
  | `kernels/triangles.py` | per-face corner/quality/gradient `@wp.func`s | 15 kernel modules, 2 wrappers |
  | `kernels/scatter.py` | scatter/accumulate kernels | 11 wrappers |

  What *is* a defect is placement: a general geometric predicate living in a module that owns an **algorithm**, so that unrelated modules import the algorithm to reach the geometry. `triangle_aabb` sat in `kernels/intersection.py` and `triangle_double_area` / `circumcircle_diameter` in `kernels/holes.py` for exactly that reason; all three are now in `predicates.py`, generic over the scalar type. When a helper is reached from a second module, ask which of the four it belongs in before adding the import.
- Python-scope wrapper functions accept `wp.array[T]` for 1D buffers; use `twt.Array2dInt32`, `twt.Array2dFloat32`, etc. for rank-2 results (see §7).
- For **2D** outputs, allocate with `twt.empty_int32_2d((rows, cols), device=...)` or `twt.empty_float32_2d(...)` instead of bare `wp.empty((rows, cols), ...)`.
- For **1D** outputs, keep `wp.empty(n, dtype=..., device=input.device)` when all elements will be written by the kernel (avoid unnecessary zero-initialization).
- Return rank-2 buffers with `return twt.as_array2d_int32(arr)` (or `as_array2d_float32`) so callers get a checked, correctly typed value.
- Always forward `device=vertices.device` (or the relevant input's device) to `wp.launch` and allocation helpers.
- Derive the face count as `f = faces.shape[0] // 3` from the flat face index array.

### Python-scope gather indexing (prefer over trivial gather kernels)

At **Python scope**, Warp supports **gather** with integer indexing: `view = src[indices]` yields a `wp.indexedarray`. Materialize a dense `wp.array` with `wp.copy(dst, view)` when callers need `.reshape()` or a guaranteed `wp.array` return type (see `triwarp/selection.py` face gather and `triwarp/array.py` `isin`).

- **1D gather:** `vertices[indices]`, `lookup[elements]`, etc.
- **2D index arrays:** Warp requires **1D** index arrays for `[]` gather — flatten first (`elements.flatten()`), gather, `wp.copy`, then `.reshape(original_shape)`.
- **⚠️ The index array must be CONTIGUOUS.** Warp 1.15 reads the index buffer as if contiguous and
  **silently ignores a view's stride** — no exception is raised. `payload[edges[:, 0]]` returns the
  flattened buffer's leading entries (`[0, 10, 1, 11, …]`), not column 0. A contiguous *prefix*
  slice (`arr[:n]`) is safe; a column (`arr[:, k]`) or step slice (`arr[::2]`) is not — `wp.copy`
  it into a dense buffer first. This is why `kernels/edges.py:edge_lengths` stays a kernel rather
  than becoming `wp.map(seg_len, verts[edges[:, 0]], verts[edges[:, 1]])`. When converting a gather,
  verify **values**, not just that it runs and is faster: the corrupt version reads a contiguous
  prefix and is measurably *faster* than the correct one.
- **Indexed assignment** (`arr[indices] = value`) is **not** supported on `wp.array` at Python scope — keep a small kernel for scatter / mask marking (e.g. `mark_membership_mask` in `triwarp/kernels/array.py`).

Do **not** add custom per-element gather kernels when `[]` plus `wp.copy` suffices. Probe tests live in `tests/test_*_indexing_probe.py`.

### Dtype conversion at Python scope (`wp.utils.array_cast`)

`wp.cast(expr, TargetType)` is for **kernel / `@wp.func` scope** only — there is no `wp.cast` on whole arrays at Python scope.

For element-wise dtype conversion of `wp.array` buffers at Python scope, allocate the destination and call **`wp.utils.array_cast(src, dst)`** (same device, matching shape). Example: `wp.bool` → `wp.int32` `0`/`1` flags for `wp.utils.array_scan` in `flatnonzero` — do **not** add a `bool_to_int32` gather-style kernel.

Inside kernels, keep using `wp.cast(expr, TargetType)` for same-width scalar and vector
conversions, and the constructor (`wp.float32(x)`) where the width changes — see §3.

### `BsrMatrix.nnz` is a stale capacity; `nnz_sync()` is the entry count

**Never size a buffer, slice, or launch dim off `matrix.nnz`.** After `bsr_from_triplets` the `nnz`
field still holds the *triplet count it was handed*, duplicates included — so for any
duplicate-emitting build it is an upper bound, measured at 8400 against a true 4516 on the synthetic
Laplacian of `downloads/issue_report.md` and **3.4x** (15 360 against 4 482) on
`laplacian.cotmatrix`, which emits 12 triplets per face. Use **`matrix.nnz_sync()`** (one host
readback, §13's ~0.1 ms) or read `offsets[nrow]`, which `energies.k_harmonic` already does.

**And `nnz` is a *cache*, not a fixed field: `nnz_sync()` repairs it in place.** Measured on 1.16.0 —
`int(m.nnz)` reads 15 360, then `m.nnz_sync()` returns 4 482, and `int(m.nnz)` now reads 4 482 too;
nothing else syncs it (`bsr_mv`, `values.numpy()`, `offsets.numpy()` all leave it stale). So whether
a `.nnz` read is correct depends on whether unrelated earlier code happened to sync that matrix,
which makes the bug order-dependent and is a live trap **for the test as much as the code**: a guard
that measures the capacity and then hands the *same* matrix to the function under test has already
repaired it, and passes against the broken implementation. Build two operators — one to measure, one
to hand over (see `test_filter_laplacian_implicit_duplicate_built_operator`).

The failure is silent and it is not a Warp bug. Sizing a `triplet_buffers` allocation by `nnz` leaves
the tail `[nnz_sync(), nnz)` unwritten, and since those buffers are `wp.empty` (§4, deliberately) the
gap reaches the next `bsr_from_triplets` as **uninitialized triplets**. `bsr_from_triplets` drops an
out-of-range row/column index silently — verified for both `999999` and `-7`, no exception and no
CUDA fault — so most garbage vanishes and the answer looks right; the entries whose garbage index
happens to land in `[0, nrow)` accumulate a garbage value into a **real** entry. Measured on
`_build_implicit_system` with a `cotmatrix` operator and plausible indices left in the memory pool:
`‖values‖ = 1.1e13` against the correct `84.3`, plus one spurious entry. This is what the
long-standing "`bsr_mm` is nondeterministic on CUDA" claim in this package really was, in
`downloads/issue_report.md` and in six code comments — **`bsr_mm` is sound**; do not reintroduce that
explanation. A rebuild sliced to `nnz_sync()` is safe.

Two corollaries. A matrix built by *duplicate-free* triplets (`laplacian.laplacian`,
`smoothing._edge_weight_matrix`) has `nnz == nnz_sync()`, which is why the default paths never
showed this — so a probe on the default operator proves nothing, and the check belongs on a
`cotmatrix`-shaped input. And where a triplet writer legitimately leaves slots unwritten (a
conditional emit, as in `dirichlet_system_triplets` / `laplacian_ls_triplets`), `wp.zeros` rather
than `triplet_buffers` is correct and deliberate: a `(0, 0, 0.0)` triplet is a harmless structural
zero.

### NumPy at Python scope is sanctioned; leaking it through the API is not

`warp-lang` carries an unconditional `Requires-Dist: numpy` and `import warp` loads it eagerly, and
`wp.array(list, dtype=...)` itself ends in `np.asarray` inside
`warp._src.types.array._init_from_data`. So NumPy is present wherever triwarp runs, it is a declared
core dependency in `pyproject.toml`, and deleting `import numpy as np` from a wrapper shrinks
nothing — it only moves the same NumPy call into Warp, more slowly. **Do not open a "remove NumPy"
pass**; 13 modules import it and that is correct. Host-side metadata math (offset scans, launch
dims, per-loop sizes, small candidate tables) and host-*sequential* algorithms (patience sorting in
`combine`, DP traceback in `holes`, `lexsort` Delaunay in `reconstruction`, `argsort` +
`searchsorted` chain linking in `intersection`, most of the procedural mesh templates in `creation`)
stay in NumPy: they are not device work, and porting them buys Python loops.

**The exception is a template that is a closed-form parallel *map* whose output scales with a
resolution parameter** — no sequential dependence between elements, so a kernel buys no Python loop
and the host build is pure assembly plus an upload. `creation.icosphere` and `creation.grid` are
both in that class and both are kernels: `grid` was 78 % NumPy prologue at `count=512` and came out
**33x faster and bit-identical** (6.77 → 0.205 ms; 112x at `(1024, 1024)`), because a `float64`
kernel followed by the same `float32` store rounds the same way the host build did. Decide by
whether the elements depend on each other, not by which module the function lives in.

Three things are still defects:

- **A public signature or return that names `np.ndarray`**, which forces the dependency on the
  *caller*. Return `wp.mat33d` / `wp.vec3` (`totals.moments` returns the inertia tensor as
  `wp.mat33d` — `wp.mat33` would discard the `float64` digits the integrals exist to keep); annotate
  inputs `Sequence[Sequence[float]]` when the body is a duck-typed `np.asanyarray`, which is
  *widening*, since the old annotation was narrower than the implementation. The one sanctioned
  exception is `triwarp/io.py`, where meshio hands back `np.ndarray` unconditionally and NumPy-in is
  `mesh_from_numpy`'s entire purpose.
- **NumPy standing in for a Warp Python-scope equivalent that exists.** `wp.full`,
  `arr[k:].fill_()`, `wp.array([wp.mat44(...)])`, `wp.determinant`, `wp.inverse`, `wp.transpose` and
  `wp.svd3` all work at Python scope (verified on 1.16) and need no host buffer; `arr.list()[0]`
  gives a row-indexable `wp.mat44` from a `wp.array[wp.mat44]`. `math.pi` / `float("nan")` /
  `float("inf")` beat `np.pi` / `np.nan` / `np.inf`. Two traps: **`wp.svd3` is not a substitute for
  `np.linalg.svd` of a non-square matrix** — `creation._align_vectors` takes the SVD of a `(3, 1)`
  for basis completion and its free rotation about the axis is a *gauge* the trimesh comparison
  pins element-wise; and **Warp raises on a zero-length slice** (`RuntimeError: Invalid indexing in
  slice: 20:20:1`), so a trailing-mask `fill_` needs an `if stop > start` guard where the NumPy
  version silently no-opped.
- **NumPy reducing a full `.numpy()` readback** — `.min()`, `.max()`, `.any()`, `.sum(axis=0)` — is a
  §13 defect wearing NumPy's clothes: the whole array crossed the bus to produce one scalar. Use
  `triwarp.reduce` (or `wp.utils.array_sum`, which reduces a `wp.vec3d` array componentwise and so
  needs no kernel of its own), and check whether a kernel for it already exists before writing one —
  `holes._mean_rim_edge_length` was reading back the entire vertex buffer while `_loop_perimeters`,
  three hundred lines up in its own file, already computed the answer on the device. **Decide these
  on the CUDA measurement and accept the CPU regression** (§13), but keep the host path where the
  buffer never scales with the mesh, as `graph.bfs_multi_source`'s `k`-element source check does.

### Elementwise ops at Python scope (`wp.map`, Warp 1.15+ — prefer over trivial map kernels)

Do **not** write a `@wp.kernel` whose body is only `out[i] = f(in[i], ...)`. Keep the op as a
named `@wp.func` in the `kernels/` module (or use a builtin like `wp.neg`, `wp.add`, `wp.div`,
`wp.normalize`) and call **`wp.map(op, *inputs, out=...)`** from the wrapper. The generated
kernel is cached (in-memory per process + Warp's on-disk kernel cache) and its GPU time is
identical to a hand-written kernel; cached calls cost ~11 µs extra host-side Python.

- **Named `@wp.func` only, never lambdas**: the map cache is keyed by the *unqualified*
  function name plus input dtypes — two different ops with the same name would collide, and
  lambdas re-derive the function each call.
- **Always pass `out=`** so allocation stays with `wp.empty` / `twt.empty_*` in the wrapper.
  In-place is `out=<an input>`; multi-output funcs (`tuple[...]` return) take `out=[a, b]`.
- Scalars mix freely with arrays (`wp.map(is_long_edge, lengths, max_edge_f, out=mask)`);
  device is inferred from the array inputs.
- **Slice views** work as inputs and outputs: adjacent-element ops map over shifted views
  (`wp.map(segment_length, polyline[:-1], polyline[1:], out=lengths)`), and offset writes map
  into `out=dst[o : o + n]`. CSR row degrees: pass `offsets[:-1]` and `offsets[1:]`.
- **Python-scope gather composes**: `wp.map(pred, table[indices], out=mask)` maps over the
  `wp.indexedarray` view (see `repair.make_volume`, which maps a sign predicate over a
  per-component volume table gathered by face label).
- Inside **per-iteration wrapper loops**, hoist the kernel once with
  `wp.map(..., return_kernel=True)` and `wp.launch(kernel, dim, inputs=[...], outputs=[...])`
  in the loop (see `triwarp/smoothing.py`) — this removes the per-call Python overhead.
- Still a real kernel: ops needing the thread index as *data* (`init_range`,
  `seed_orientation`), whole arrays as uniform arguments (binary-search tables), scatters,
  and multi-element/row-indexed outputs.

### Function-valued parameters and kernel factories (Warp 1.15+)

- A `@wp.func` may take `fn: wp.Function` parameters; the target is bound at **compile time**
  per call site (user `@wp.func`s and simple builtins like `wp.min` are valid targets; tile
  intrinsics, variadic and LTO builtins are not). `wp.launch` can NOT pass a `wp.Function` as
  a kernel argument — for runtime selection, pass an **int/enum kernel argument** and branch
  over `wp.Function` targets inside a dispatch `@wp.func` (warp-uniform branch, one compiled
  module; see `registration.robust_weight` for the pattern).
- Builtins with no Python-scope handle (e.g. tile intrinsics) can still parameterize kernel
  factories by pulling the concrete `Function` object from
  `warp._src.context.builtin_functions["tile_max"]` and closure-capturing it — captured
  builtins emit inline at codegen and template on the tile dtype (see
  `triwarp/kernels/reduce.py`). Give each factory instantiation a unique kernel `name`.

### A generic kernel registers its overloads at import (`wp.overload`)

**A `@wp.kernel` generic over a dtype (`wp.Scalar`, `wp.Float`, `wp.Int`, `Any`) must have its
concrete overloads registered at module import**, in a `_register_overloads()` called at the bottom
of the file. Warp instantiates a generic kernel's overload *lazily*, on the first launch at each new
dtype, and a module's hash covers the set of **instantiated** overloads — so that first launch
changes the hash and recompiles **every kernel in the module**. Nothing fails; it only costs, which
is why nothing but a check catches it.

Measured on an RTX 5090, Warp 1.16, over one full-suite run before the registrations existed:
`kernels.reduce` rebuilt across **66** distinct `(hash, block_dim)` module loads at ~14 s each,
`laplacian` 16, `array` 13, `scatter` 11, `grouping`/`energies` 9 — 206 loads across
`triwarp.kernels.*`, against 87 after, where 2 per module is the irreducible CPU/CUDA floor.
`tests/test_reduce.py` alone went from 535 s to 1.4 s and the whole suite from 1 033 s to 29 s.
Independent corroboration: 464 of the 1 269 directories in the Warp kernel cache were dead `reduce`
hash links.

- **The chain is order-dependent, which is what makes it a developer-loop tax rather than a
  one-time cost.** A caller reaching the dtypes in a different order walks links that were never
  compiled, so changing which tests you select re-pays it from scratch.
- **Register what the wrapper's dispatch can reach, not every dtype the template admits** (§14, no
  speculative generality): an unused overload is compile time paid on every rebuild. Derive the set
  from the wrapper — a public `dtype=` keyword documented "float32 or float64", the key dtypes
  `sortable_dtype` maps onto, a docstring naming its own admissible dtypes — and say so in a
  comment. Where two generic arguments are independent (`laplacian.cotmatrix_triplets`' entry
  precision and matrix precision), it is a genuine cross product, not a diagonal.
- **Registration is not compilation.** `wp.overload` builds the overload's `Adjoint` and nothing
  else, so this costs milliseconds of import and nothing on a process that never launches the
  kernel. Do **not** `wp.load_module` / `wp.force_load` at import — that *would* compile eagerly.
- **`block_dim` forks the hash independently of dtype and is normally left alone.** Its values are
  fixed by the package's own launch code (a bounded two or three: a tiled launch's `block_dim`,
  Warp's 256 default, and 1 on CPU), so it is not a chain that grows with call order. Collapsing
  them is a perf change and needs §13's measurement — it was measured for `reduce` and declined.
- `tests/test_api_conventions.py::test_generic_kernels_register_their_overloads` fails when a
  generic kernel has **no** overload registered. Nothing checks that a registered dtype *set* is
  complete, and nothing cheaply can — proving it means launching the whole dispatch, and the cost
  of getting it wrong is a rebuild rather than a wrong answer. **A missing dtype is diagnosed from
  the clock, not from a failing assert**; see §13.

### In-place `@wp.func` parameters (`wp.ref[T]`, Warp 1.15+)

`@wp.func` helpers may declare `wp.ref[T]` parameters to mutate caller-owned storage (locals,
array elements, struct fields) — use for multi-value updates like argmin/minmax/swap helpers
(see `update_argmin` in `kernels/array.py`).
**Constraint:** any kernel calling a `wp.ref` helper must be decorated
`@wp.kernel(enable_backward=False)` — the per-kernel flag specifically; a module-level
`wp.set_module_options({"enable_backward": False})` is NOT consulted at kernel-parse time in
Warp 1.15 and the module still fails to compile. Never use `wp.ref` in
`triwarp/kernels/distance.py` — the chamfer kernels are differentiated via `wp.Tape`.

---

## 5. Kernel-Scope Restrictions

The following Python features are **not supported** inside `@wp.kernel` and `@wp.func`:

- Lambda functions, list comprehensions, sets, dicts, `list.append()`, `eval()`, recursion, exceptions.
- Python tuples for initialization — use explicit typed constructors: `wp.vec3(1.0, 2.0, 3.0)`, not `(1.0, 2.0, 3.0)`.
- For small fixed-size collections use vector types (`wp.vec3`, etc.); for larger ones use `wp.zeros(shape=N, dtype=T)` (stack-allocated inside a kernel).
- The `%` operator follows C++11 semantics (sign of result = sign of dividend), not Python semantics.
- **`//` truncates toward zero like `/`, not toward −infinity like CPython's `//`, and on integers the two operators are the same operation.** Measured on Warp 1.16: `[-8, -7, -1, 0, 1, 7, 8] ÷ 3` gives `[-2, -2, 0, 0, 0, 2, 2]` for both spellings, where CPython's `//` gives `[-3, -3, -1, 0, 0, 2, 2]`. This is consistent with the `%` rule above (`-8 % 3 == -2`, and `-2 * 3 + (-2) == -8`). **Spell integer division `//`** — `/` on two `int32`s reads as real division and only truncates because the operands happen to be integers, so a reader has to recover the types before knowing what the line does. Every dividend in this package is a non-negative index, where the two conventions coincide; the hazard is *porting* a line with a negative dividend between host Python and kernel scope, which changes its answer silently.
- `wp.asin()` / `wp.acos()` auto-clamp inputs to [-1, 1]; explicit `wp.clamp` before these calls is redundant but harmless.
- Variable scope inside conditional blocks may differ from CPython: variables defined only inside an `if` branch are accessible afterward in Warp, but are uninitialized if the branch was not taken — always initialize variables before branching.

---

## 6. Testing Against Trimesh

All new geometry functions MUST have regression tests that compare against the `trimesh` CPU reference implementation (`trimesh.triangles`).

### Conventions

- Test file: `tests/test_<module>.py`; import pattern:
  ```python
  import trimesh.<module> as tm
  import triwarp.<module> as tw
  ```
- Use the `device` fixture from `tests/conftest.py` (runs on `cuda:0` if available, else `cpu`). Every test function must accept `device` as a parameter.
- Generate reproducible random data with `np.random.default_rng(seed)` (use a fixed integer seed per test).
- Convert triangle-soup arrays `(n, 3, 3)` to an indexed mesh before passing to Warp:
  ```python
  def _triangle_soup_to_vertices_faces_wp(tri_np, device):
      n = tri_np.shape[0]
      vertices = tri_np.reshape(-1, 3).astype(np.float64)
      faces = np.arange(n * 3, dtype=np.int32)
      v_wp = wp.array(np.ascontiguousarray(vertices), dtype=wp.vec3, device=device)
      f_wp = wp.array(faces, dtype=wp.int32, device=device)
      return v_wp, f_wp
  ```
- Call `.numpy()` on Warp output arrays before passing to NumPy comparison functions. Use it inline, and not defining a new variable.
- Use `np.allclose(got, exp, rtol=1e-5, atol=1e-5)` for floating-point results.
- Use `np.array_equal(got, exp)` for boolean or integer results.
- Name variables with a suffix for the library: `_np` for NumPy/SciPy, `_tm` for Trimesh, `_wp` for Warp. Avoid `got` / `exp` but use instead clear names.
- When passing a NumPy 1D vector to a `wp.vec3` scalar argument at Python scope, use `wp.vec3(*array_np.tolist())` — not `wp.vec3(*map(float, np.asanyarray(...).reshape(3)))`.

### Fallback references: libigl, potpourri3d and pymeshlab

When `trimesh` has no equivalent function, use the `igl` Python package (bindings for the C++
reference mirrored under `reference/libigl/`) as the CPU reference instead — import as
`import igl`, name reference variables with an `_igl` suffix. Pass triwarp's flat face buffer
as `mesh_tm.faces` (`(n_faces, 3)` int array) to the igl function. Otherwise follow the same
comparison conventions (`np.array_equal`/`np.allclose`, inline `.numpy()`).

igl is the reference whose input convention matches triwarp's most closely — `float64` `(n, 3)`
vertices and `int64` `(n_faces, 3)` faces, which is exactly what `mesh_tm.vertices` / `mesh_tm.faces`
already are — and every bound function is *pure* (arrays in, arrays out), so there is no in-place
mutation to defend against. The exceptions are the stateful solver objects (`HeatGeodesicsData`,
`ARAPData`, `min_quad_with_fixed_data`, `AABB`), which cache a factorization and must therefore be
constructed **inside** a timed callable. Five hazards, all measured:

- **An out-of-range face index is a SIGSEGV, not an exception.** `igl.cotmatrix(V, F)` with one entry
  of `F` set to `len(V) + 500` kills the interpreter with exit code 139 and no traceback — igl
  bounds-checks nothing. Never hand it a reduced `V` with the original `F`. The same class of crash
  hits `igl.principal_curvature` on a non-manifold vertex, and `igl.heat_geodesics_precompute` /
  `igl.harmonic` / `igl.lscm` refuse (raise) rather than crash on meshes they cannot factor.
- **Three bound functions are memory-unsafe on ordinary input, so a "works" probe is not enough** —
  check *values*, and prefer a fixture class where the function is known safe. `igl.loop` aborts with
  `free(): invalid pointer` on a five-vertex mesh with three faces on one edge and SIGSEGVs (139) on
  `bunny_decimated`, whose 87 duplicated faces leave it non-edge-manifold; on `bunny` it silently
  returns 1 113 `NaN` rows, one per unreferenced vertex, because it indexes `igl::adjacency_list`
  (sized `F.max() + 1`) up to `n_verts`. And **`igl.in_element` is unusable outright**: on a
  two-triangle square it never reports element 0 for any query inside it, the same query returns a
  face in a 3-query batch and `-1` in a 7-query batch, and a 200-point Delaunay input aborts with
  `malloc(): invalid size`. Use `scipy.spatial.Delaunay.find_simplex` as the point-location oracle.
  And **`igl.upsample` corrupts the process heap on the scan meshes**, which matters more than the
  other two because the SIGSEGV lands *later*, in unrelated code, and `--benchmark-json` is written
  at session end — so it silently destroyed every row of `benchmarks/test_remesh.py` for two
  measurement rounds. Measured one selection per process: **2 of 8** runs crash with the igl rows
  alone, **8 of 8** with other libraries co-resident, and per mesh **7 of 8** on `bunny_decimated`,
  **7 of 8** on `bunny`, **0 of 8** on `dragon`. Two readings to *not* take from that: it is not an
  interaction with triwarp (an earlier round concluded it was, from a single nondeterministic
  failure), and it is not a size limit — the smallest mesh fails most and the largest never does.
  Compacting the unreferenced vertices away makes it worse, not better. It is safe on `icosahedron`
  (1 200 calls, six processes, clean), so it stays a tested reference there and is *not* a
  benchmarked one.
- **F-only functions size their output by `F.max() + 1`, not by `len(V)`.** `igl.adjacency_matrix`,
  `igl.vertex_components` and `igl.is_vertex_manifold` return `F.max() + 1` rows where
  `igl.cotmatrix` and `igl.gaussian_curvature` return `len(V)`. So on a mesh with unreferenced
  vertices the two families disagree with each other and only the `(V, F)` family matches triwarp;
  a comparison against the F-only family is class B with the transform named ("pad igl's answer to
  `n_vertices`"). Also note `igl.connected_components(igl.adjacency_matrix(F))` counts every isolated
  vertex as its own component.
- **Several call signatures are not what the docs suggest, and two fail silently.**
  `igl.exact_geodesic(V, F, vs, vt)` returns an *empty array* rather than raising, because `VS/FS/VT/FT`
  all default to `array([])` and a 4-argument call binds `vt` to `FS` — pass all six, with the face
  arrays explicitly `np.array([], dtype=np.int64)`. `igl.knn(P, V, k, *igl.octree(V)[:4])` takes seven
  positional arguments. `igl.in_element` needs a live `igl.AABB`. `igl.crouzeix_raviart_*` need
  `(V, F, E, EMAP)` from `igl.unique_edge_map(F)`. `igl.average_onto_vertices`'s `S` is a per-face
  *scalar*; `igl.cut_mesh`'s `C` is a per-corner **bool** mask, not an edge list.
- **`collapse_small_triangles` and `resolve_duplicated_faces` are not bound**, despite the C++ headers
  existing and `triwarp.repair` carrying functions named after them (`AttributeError`). Generally: the
  C++ surface is ~493 headers and only 150 functions are bound, so confirm a name exists in the wheel
  before planning a comparison around it. A hand port of the C++ into a test file is a legitimate
  *test* oracle (see `tests/test_distance.py`, `tests/test_polyline.py`, `tests/test_seams.py`) but
  never a benchmark row.

**Licensing:** libigl's core is MPL2, but everything under `reference/libigl/include/igl/copyleft/`
is **GPL** — the CGAL boolean suite, `progressive_hulls`, `quadprog`, tetgen and
`copyleft/marching_cubes`. No triwarp code may be derived from that subtree; read the MPL2 top-level
`marching_cubes.h` if a reference is needed, never the `copyleft/` one.

For the heat-method family, tangent spaces and isocontours — where neither trimesh nor igl has an
equivalent — use `potpourri3d` (pybind11 over geometry-central, mirrored under
`reference/potpourri3d/`): `import potpourri3d as pp3d`, reference variables suffixed `_pp`. It takes
`float64` `(n, 3)` vertices and `(n_faces, 3)` `int32` faces. Four things to know before writing the
comparison:

- **Construct solvers with the non-default flags** `use_robust=False` /
  `use_intrinsic_delaunay=False` so both sides discretize the same triangulation; the defaults
  mollify and flip to an intrinsic Delaunay triangulation first.
- **Tangent-space quantities are gauge-dependent.** Frames agree only up to a rotation about the
  normal, and transport angles only through gauge-invariant combinations (the holonomy around a
  face). Never compare 2D tangent components or single connection phases element-wise.
- **Barycentric output must be decoded.** `marching_triangles` returns `(element_index, coords)`
  pairs in geometry-central's *own* element numbering — decode edges through `pp3d.edges(V, F)`,
  dispatching on `len(coords)` (0 = vertex, 1 = edge, 2 = face). Its closed curves repeat their first
  point; open ones do not.
- **It rejects some inputs outright**, with a `RuntimeError` rather than a wrong answer:
  `MeshVectorHeatSolver` / `GeodesicTracer` / fast marching need a manifold mesh, and `pp3d.edges`
  needs every vertex referenced by a face. Pick fixtures accordingly instead of catching the error.

A zero cotangent weight (an edge whose two opposite angles are both right angles, i.e. every quad
grid split by a diagonal — `cave_cube`, `half_torus`) erases that edge's phase from
`get_connection_laplacian()`, so it cannot serve as an oracle there at all; see
`tests/test_tangent.py`.

**pymeshlab** (pybind11 over MeshLab / VCGlib, mirrored under `reference/PyMeshLab/`) is the
broadest reference of the three — 281 filters — and a hard test dependency like `igl`, so
`import pymeshlab as ml` plainly, never through `pytest.importorskip`. Reference variables take a
**`_pml`** suffix. Build the MeshSet with `tests.conversions.trimesh_to_pymeshlab(mesh_tm)` (or
`warp_to_pymeshlab(vertices_wp, faces_wp)` for a triwarp output) rather than hand-rolling
`ml.MeshSet()`. Use it where it is a *better* oracle than the incumbent, not everywhere — trimesh /
igl / potpourri3d stay the reference where they already are one. Six hazards, all found by probing:

- **Almost every filter mutates `current_mesh()` in place.** `apply_coord_*` moves vertices,
  `meshing_*` rewrites the topology, `compute_*_per_vertex` writes an attribute, `generate_*` pushes
  a *new* mesh onto the set. So one MeshSet serves one filter call; build a fresh one per comparison.
  `compute_curvature_principal_directions_per_vertex` and
  `meshing_decimation_quadric_edge_collapse` additionally default to `autoclean=True` and will
  delete unreferenced vertices under you.
- **`get_*` filters return a dict; `compute_*` / `meshing_*` / `apply_*` return `None`** (or a small
  dict of statistics) and the answer must be read back off `current_mesh()` — `vertex_matrix`,
  `face_matrix`, `vertex_normal_matrix`, `vertex_scalar_array`, `face_scalar_array`,
  `vertex_selection_array`, `face_selection_array`,
  `vertex_curvature_principal_dir{1,2}_matrix`, `edge_matrix`. Selections come back as bool arrays
  and scalars as float arrays, so `np.array_equal` / `np.allclose` apply directly.
- **Length parameters take a wrapper type.** `ml.PercentageValue(1)` is 1% of the bbox diagonal;
  `ml.PureValue(x)` is an absolute length (this version has no `AbsoluteValue` — that name is gone).
  Pass `PureValue` fed from the same number triwarp gets so both sides see the identical parameter.
- **It rejects some inputs outright.** `compute_texcoord_parametrization_harmonic` and
  `..._least_squares_conformal_maps` raise `PyMeshLabException: Failed to apply filter` on a
  **closed** mesh — they need a boundary, so use `hemisphere` / `half_torus`, not `icosahedron`.
  `face_face_adjacency_matrix()` raises `MissingComponentException` unless the FF component was
  requested (`update_topology()` alone does not enable it). And `generate_boolean_*` takes
  `first_mesh` / `second_mesh`, not the `first` / `second` older docs show.
- **Some parameters are silent no-ops, and one runs backwards.**
  `compute_scalar_by_shape_diameter_function_per_vertex`'s `cone_amplitude` produces byte-identical
  output at 90 and 120 degrees; `apply_normal_smoothing_per_face` and
  `apply_scalar_smoothing_per_vertex` expose no parameters at all (one pass, take it or leave it); and
  `generate_resampled_uniform_mesh`'s `offset` as a `PercentageValue` runs from *full erosion* at 0%
  to full dilation at 100%, so its own 50% default is the **zero** offset — pass `PureValue(0.0)` when
  you mean zero. Probe a parameter before building an axis on it.
- **Two filters differ from triwarp's port by definition, not tolerance**, and the tests say so
  rather than papering over it: `apply_scalar_smoothing_per_vertex` averages a *boundary* vertex over
  its two boundary neighbours alone (so its oracle runs on closed fixtures), and
  `apply_coord_two_steps_smoothing` at its own defaults moves a noisy cube *further* from clean than
  the noise was, because its fitting step rounds corners in.

**open3d** (pybind11 over Open3D, mirrored under `reference/Open3D/`) is a hard test dependency
like `pymeshlab` and `igl` — import it plainly as `import open3d as o3d` (the alias is pinned in
ruff's import conventions), never through `pytest.importorskip`; reference variables take an
**`_o3d`** suffix. Build meshes with `tests.conversions.trimesh_to_open3d` (reusable across calls,
unlike a MeshSet), clouds with `points_to_open3d`, and tensor-API meshes with
`trimesh_to_open3d_t` — never chain off an unbound `from_legacy(...)` (see the freed-memory hazard
below). The installed wheel is a CUDA build whose legacy `open3d.geometry` / `open3d.pipelines`
APIs are CPU-only; only `open3d.t` has GPU kernels. Seven hazards, all measured:

- **Legacy `remove_*` / `orient_*` / `filter_*` methods mutate in place** (build inside the timed
  callable, the `test_repair.py` rule); the pure `compute_*` / `get_*` / `is_*` calls recompute
  unconditionally and can share one mesh. The trap in the second family: **`get_volume` validates
  before it integrates**, and the validation is the full brute-force `IsWatertight` composition —
  13.8 s on a watertight 82k-face sphere whose integral is microseconds — and it *raises* on
  non-watertight input. Never benchmark it as "volume".
- **k-NN distances come back squared** from both `KDTreeFlann` and `o3d.core.nns` — take the square
  root before `allclose`. Use `o3d.core.nns.NearestNeighborSearch` for anything batched (indices
  match `scipy.spatial.KDTree` byte-for-byte on a tie-free cloud); the legacy tree's only query is
  a per-point Python loop, 6x slower at 20k queries. `KDTreeFlann`'s radius search is **exclusive
  at exactly `r`** where triwarp's ball queries are inclusive — random clouds never tie, so only a
  constructed fixture can expose it.
- **`is_vertex_manifold` tests connectivity, not a fan**: three faces sharing one edge pass it
  (their faces are edge-connected) and fail triwarp's and igl's fan definition. The answers agree
  exactly on edge-manifold input — restrict the comparison to that class and pin the divergence.
  `is_edge_manifold` shares triwarp's `allow_boundary_edges` switch with identical semantics.
- **Smoothing filters re-derive inverse-distance weights from current positions every pass**
  (`filter_smooth_laplacian`, `filter_smooth_taubin`), so they match triwarp's fixed assembled
  operator at one iteration (6.6e-08) and diverge over ten; Taubin's `number_of_iterations` counts
  lambda-mu *pairs* like MeshLab's. `filter_sharpen` adds `strength * (deg(v) * v - Σ neighbours)`
  — the *unnormalized* residual, so its displacement is triwarp's times the vertex degree (measured
  ratio = degree to 7 digits) and no parameter mapping fixes an irregular mesh. All three are D2
  exemptions with the numbers in the `noparity` reasons.
- **Platonic solids come in rotated frames and odd scales**: the octahedron matches triwarp's
  vertex table exactly, but the tetrahedron is rotated (nearest-vertex distance 0.92 after
  scaling) and the icosahedron is the raw `(0, ±1, ±φ)` table at circumradius 1.902 — compare
  rigid-motion invariants after scaling to unit circumradius, never positions. There is no
  `create_dodecahedron`.
- **`RaycastingScene.compute_signed_distance` shares triwarp's convention exactly** (negative
  inside, parity-ray sign; 1.8e-7 agreement) — no negation, unlike trimesh. But
  `compute_closest_points` diverges ~2e-4 at equidistant-face ties, so compare *distances*, not
  the returned points.
- **`get_oriented_bounding_box` is PCA of the hull and minimizes nothing** (12.9% above triwarp's
  volume on a tilted half_torus); the comparable entry point is
  `get_minimal_oriented_bounding_box`, the hull-face search of trimesh's family.

**pyvista** (VTK 9.6 through its own Python layer, mirrored under `reference/pyvista`) is the
**VTK** reference and a hard test dependency like the three above — `import pyvista as pv`, never
`pytest.importorskip`; reference variables take a **`_pv`** suffix. Build meshes with
`tests.conversions.trimesh_to_pyvista` and clouds with `points_to_pyvista`; one `PolyData` serves
many comparisons because pyvista caches nothing. `pyvista.core` needs no renderer and every
comparison in the suite runs headless. Two things about the mirror: it is byte-identical to the
installed wheel for `core/utilities/parametric_objects.py`, so those algorithms may be read from the
submodule; but the parametric surfaces' **domains and periodicity are not in pyvista at all** — they
live in VTK's `vtkParametric*` constructors as `JoinU` / `JoinV` / `TwistU` / `TwistV` and were read
off the objects at runtime. And a `uv run` whose working directory is inside `reference/pyvista`
resolves *that* project and builds a second virtualenv (it downloads its own VTK); always invoke
probes from the repo root. Licence: pyvista is MIT and VTK is BSD-3, so there is no `copyleft/`
subtree to avoid as there is in `reference/libigl`. Ten hazards, all measured:

- **Float64 in, sometimes float32 out.** Point storage is exact (round-trip error `0.0` on
  `[1/3, π, e]`) and `regular_faces` is a real `(n, 3)` `int64` array — that is why pyvista, not
  vedo, is the VTK oracle. But `compute_normals`' `Normals`, `ray_trace`'s hit points,
  `fit_plane_to_points(return_meta=True)`'s centre and normal and `texture_map_to_*`'s coordinates
  come back **float32**, while `multi_ray_trace`, `principal_axes`, `curvature` and
  `compute_implicit_distance` are float64. Check the dtype per row. Where a difference of large
  numbers is taken (the angle defect), use `atol` and know the residual is **triwarp's** float32
  vertex buffer: the same comparison measures 5.10e-05 against pyvista's float64 and 5.4e-05
  against vedo's float32.
- **Nothing is cached, and one filter of twelve mutates.** Repeat calls recompute (`cell_quality`
  10.1 / 7.2 ms, `decimate` 210 / 201 ms), so a shared `PolyData` is right — unlike a
  `pymeshlab.MeshSet`. The exception measured: **`edge_mask` writes `point_ind` into its input.**
  Every filter with an `inplace` switch defaults to `False`; never pass `True`.
- **An extraction renumbers its points.** `extract_feature_edges` returns a new `PolyData` carrying
  only the points its lines touch, in its own order — on a unit box it keeps all 8 and still orders
  them differently, so a comparison that skips the remap fails on a mesh whose *counts* match and
  reads as a real disagreement. Go through `tests.conversions.pyvista_edges_to_indices`.
  `extract_cells` / `extract_points` / `threshold` / `clip_box` return an `UnstructuredGrid` (index
  maps in `vtkOriginalPointIds` / `vtkOriginalCellIds`); `split_bodies` / `bounding_box` /
  `oriented_bounding_box` return a `MultiBlock` unless passed `as_composite=False`; and
  `remove_points` / `collision` / `ray_trace` / `contour_banded` return **tuples**.
- **`cell_quality` is 12 usable measures of 28 on triangles, with two naming inversions and one
  constant.** pyvista's `radius_ratio` is triwarp's `aspect_ratio`; triwarp's `radius_ratio` is its
  reciprocal; `shape` is triwarp's `mean_ratio` and `aspect_frobenius` is one over it; `condition`
  duplicates `aspect_frobenius` exactly; `min_angle` / `max_angle` are in **degrees**;
  `distortion` is a constant **1.0** on ordinary input, so a threshold test on it passes on
  anything; and the 16 measures that do not apply come back as the `-1.0` null value rather than
  raising. `tests/test_triangles.py` carries the decoding.
- **`is_manifold` is `n_open_edges == 0`, and `n_open_edges` counts boundary *plus* non-manifold
  edges.** So `is_manifold` maps to `is_edge_manifold(allow_boundary_edges=False)` and the *count*
  does not map to `boundary_edges`: on three faces sharing one edge it reads **7** where triwarp
  counts 6 boundary edges. Likewise **`DataSet.center` is the bounding-box centre**, not a
  centroid, and `bounding_sphere` returns `(radius, center)` — radius first — and is a genuine
  near-minimal sphere (`vtkCell::ComputeBoundingSphere`, square-rooted), not the sphere about the
  AABB centre; the two coincide on any centrally symmetric mesh, which is how the wrong reading
  survived three fixtures, and differ by 11.8% on a hemisphere.
- **`curvature('maximum')` / `('minimum')` are algebra, not estimation** — measured *exactly*
  `H ± √(H² − K)` from VTK's own Gauss and mean curvature, max abs difference **0.0** in float64,
  and complex on the 300 of 642 icosphere vertices where `H² < K`. They are not an independent
  principal-curvature implementation. `curvature('gaussian')`, by contrast, is exactly the angle
  defect over the barycentric lumped area and is a genuine oracle.
- **Surface operators return surface quantities.** `compute_derivative`'s gradient is the
  *tangential* one — mean `2/3 · e_x` for `f = x` on the unit sphere, which triwarp's
  `face_gradients` reproduces to 1.8e-07 after `average_onto_vertices` — and it stays on the
  **points** for a point-data field even with `preference='cell'`. Correct for a 2-manifold; it
  looks like a bug if read as 3-D.
- **Both smoothers are different algorithms, not different tunings.** `smooth` moves each vertex
  along its incident edge directions under VTK's own convergence test, so it diverges with the
  iteration count against a fixed assembled operator (0.103 / 0.363 / 0.581 at 1 / 5 / 10 at
  `relaxation_factor=1.0`); `smooth_taubin` is the windowed-sinc filter, parameterized by a
  `pass_band` it maps to its own weights, and it warns *"An optimal offset for the smoothing filter
  could not be found"* on ordinary input (0.835 at one iteration, then 6.5e-03 / 2.6e-02). Its
  iteration count is in lambda-mu **pairs**, like MeshLab's. Both are D2 exemptions.
- **Several answers are empty, constant or unchanged rather than wrong, so assert non-vacuity
  first.** `clip_surface(pv.Sphere(radius=0.6))` returns **0 cells** on `icosphere(3)`;
  `extract_values(0.010, scalars='area')` returns 0 cells (it matches values *exactly* — a range is
  `ranges=`); `edge_mask(30)` is all-`False` on a smooth sphere (use a box);
  `integrate_data` of a symmetric point field reads −1.2e-15; `validate_mesh().coincident_points`
  is **empty** on ten exactly duplicated vertices (`clean` is the dedup oracle, and its
  `zero_size` — not `degenerate_faces` — is where a zero-area triangle lands);
  `lines_from_points` gives one two-point line cell per segment rather than one polyline, so
  `compute_arc_length` restarts every segment and `decimate_polyline` is a no-op at every
  reduction; `tube` / `ribbon` emit triangle **strips** (`n_faces == 0` — `.triangulate()` first);
  and `extrude(capping=True)` leaves 16 open edges.
- **Where the reference put the answer, and four deprecated names.** `align(return_matrix=True)`
  returns `(aligned_mesh, 4×4 matrix)` and *does* move the points (unlike MeshLab, which writes a
  layer transform); `geodesic` puts the ordered path in `vtkOriginalPointIds` and its Euclidean
  length equals `geodesic_distance` to 1e-8; `sample` marks misses with `vtkValidPointMask` *and* a
  `vtkGhostType` array; `voxelize_binary_mask` writes a **point** array named `mask` on a
  cell-centred grid, so it is *solid* and its set is contained in triwarp's `mode="solid"` answer
  rather than equal to it. Deprecated in 0.48.4: module-level `pv.voxelize` / `pv.voxelize_volume`
  (a hard `DeprecationError`), `select_enclosed_points` (→ `select_interior_points`, and the array
  name is now lowercase `selected_points`), `extract_geometry` (→ `extract_surface(algorithm=None)`)
  and `n_faces_strict` (→ `n_faces`).

Two more, for the parametric surfaces specifically: **they arrive open and `clean` defaults
differently per surface** — `surface_from_para(clean=False)` is the underlying default and at that
setting every one of the 21 has 156 or 236 boundary edges (a raw `ParametricMobius()` is a disk),
while pyvista overrides it to `clean=True` on 9 of them, so `klein` arrives welded and `mobius` does
not. **Always pass `clean=True` explicitly.** And **`klein` is not a Klein bottle** as VTK
parameterizes it: it welds to two boundary loops and reads *orientable*, so only `figure8_klein` is
the closed non-orientable χ = 0 surface. A comparison reaching for `klein` expecting
non-orientability is testing nothing.

### Mesh fixtures (prefer over inline construction)

Reuse shared mesh fixtures from `tests/conftest.py` instead of building meshes in each test. Fixtures return `(mesh_tm: tm.Trimesh, mesh_wp: wp.Mesh)` via `tests.conversions.trimesh_to_warp`.

| Fixture | Use when |
|---------|----------|
| `icosahedron` | Default watertight solid; inside/outside, surface sampling, sign tests |
| `cave_cube` | Hollow / non-convex shell (boolean difference) |
| `hemisphere`, `half_torus` | Curved or open surfaces |
| `boy_surface` | Closed, watertight and **non-orientable**, χ = 1 — the `False` branch of `is_orientable` / `face_orientation_bits`, and `make_winding_consistent`'s impossible one |
| `mobius` | Non-orientable *with* a boundary: the same three predicates, one loop of 78 edges, χ = 0 |
| `bohemian_dome` | Closed genus 1 that self-intersects — `homology_generators` / `tree_cotree` at genus 1, and `is_self_intersecting` on a *closed* input |

The last three are built by `creation.parametric_surface` rather than by trimesh, and they are the
only inputs in the suite that are non-orientable or that have an odd Euler characteristic. A boolean
predicate asserted only on the first three fixtures is testing one branch; that is what these close.
`creation.parametric_surface` builds thirteen more surfaces that are not fixtures yet — reach for
one (and add the fixture) rather than hand-rolling a degenerate mesh, and see its docstring for
which class each is.

- **Do not** call `tm.creation.box()` or hand-roll `wp.Mesh(...)` in tests unless the case requires a bespoke degenerate mesh (e.g. empty faces, unreferenced vertices).
- When a simple cube would suffice, prefer **`icosahedron`** or **`cave_cube`** for richer geometry.
- Parametrize over multiple fixtures with `request.getfixturevalue(mesh_name)` when coverage should span mesh types (see `test_signed_distance_on_mesh_random`).
- Use `mesh_wp.device` (not the `device` fixture) for query-point allocation when a mesh fixture is already in scope.
- Edge-case tests (empty points/faces, single-triangle pathology) may still use minimal inline buffers.

### The parity gate: a benchmarked reference must be a tested reference

`benchmarks/` asserts only shapes and finiteness, so on its own it cannot tell whether two timed
implementations compute the same thing. `tests/test_parity.py` closes that loop and **fails the
default `pytest` run** when a benchmarked `(group, library)` pair is neither tested nor exempted.
Two markers join the suites; the benchmark's `benchmark(group=...)` name is the key, which makes
group names a cross-suite API — renaming one breaks every `parity` marker that cites it.

```python
# tests/test_edges.py -- "this test proves triwarp agrees with trimesh for that group"
@pytest.mark.parity("faces_to_edges", "trimesh")

# benchmarks/test_curvature.py -- "timed, but the results are not comparable"
@pytest.mark.noparity("pymeshlab", oracle="trimesh", reason="MeshLab computes the Meyer/Desbrun "
                      "pointwise 1-ring operator, not the Cohen-Steiner/Morvan ball measure, so "
                      "its absolute value is not comparable; measured 0.982 correlation with a 7% "
                      "offset. trimesh is the oracle for this group.")
```

Both are stackable and take string **literals** only — a computed argument is invisible to a static
scan, so the scanner rejects it. Run `uv run python -m tests.parity` for the full matrix.

Classify every comparison, and say which class it is in the docstring:

- **A** direct `np.allclose` / `np.array_equal`. The default.
- **B** equal after a *named* transform, still at `1e-5`: a dict index, a unit fix
  (`igl.doublearea / 2`), a reduction (igl's per-vertex mask → triwarp's bool), a projection
  (`igl.boundary_loop` is the longest loop), `lexsort` for unordered rows, a sign or gauge fix.
  **Most apparent non-equivalence lands here.** The benchmark docstrings' "does strictly less",
  "upper bound" and "output shape differs" caveats are about *cost*, not about the value.
- **C** a derived scalar, set distance or statistic, because no correspondence exists. Must name the
  bug class it excludes, and record a mutation probe *and its margin* in the docstring — threshold
  ≥ 3x from the measured agreement. `fraction_within` bounds must be shown to fail under shuffling
  one side, or they are testing marginal distributions rather than the correspondence.
- **D** exemption. Only for: not an independent implementation (`oracle=` required); a different
  algorithm with a measured disagreement; a parameter the reference lacks; an answer not observable
  in isolation; stochastic with no invariant; or input classes where triwarp is undefined.
  **Not** admissible: "awkward", "the tolerance would be loose", or any class-B situation.

Never a parity assert: shape-only or `isfinite`-only (that is the *benchmark's* assert, and this
gate exists to stop it migrating inward); triwarp compared with itself; a threshold a constant
output would pass. A boolean assert must be parametrized over inputs producing both answers.

**Check the comparison is not vacuous on its fixture**, which the gate cannot do for you. A test
comparing two *empty* answers passes, reads as coverage, and tests nothing — measured: `test_ears`
compared `igl.ears` against `boundary.ears` on `hemisphere` and `half_torus`, where neither library
finds a single ear, so the assert was `[] == []` and the loop body checking the corner convention
never ran. Making it non-vacuous immediately surfaced a real disagreement (the two number the ear's
local edge differently, `triwarp_opp == (igl_opp + 1) % 3`). So: assert the reference produced a
non-empty answer, or assert its expected count, before comparing to it — and treat "this function is
already the oracle in tests/" as no evidence at all that the comparison is live.

Reuse `tests/comparisons.py` (`lexsort_rows`, `assert_unordered_rows_equal`, `canonical_labels`,
`same_partition`, `canonical_winding`, `assert_same_up_to_sign`, `assert_cyclic_permutation_equal`,
`fraction_within`, `symmetric_chamfer`, `hausdorff_two_sided`) and `tests/conversions.py`
(`trimesh_to_open3d`, `points_to_open3d`, `open3d_to_trimesh`, `trimesh_to_open3d_t`,
`trimesh_to_pymeshlab`, `warp_to_pymeshlab`, `points_to_pymeshlab`, `trimesh_to_pyvista`,
`points_to_pyvista`, `pyvista_edges_to_indices`, `warp_to_trimesh`, `faces_igl`) rather than
re-rolling either. `canonical_labels` is the label-packing transform every component comparison
needs — triwarp names a component after a representative element, igl and scipy number `0..k-1` in
their own traversal orders and VTK's `RegionId` numbers them in a third, so only the *partition* is
shared. `open3d` and `pyvista` are hard test dependencies like `pymeshlab` and `igl` — import them
plainly as `import open3d as o3d` / `import pyvista as pv`, never through `pytest.importorskip`; see
their hazard blocks above.

Two measured gotchas worth not rediscovering: MeshLab's `face_normal_matrix()` after
`compute_normal_per_face()` is the **unnormalised** cross product (magnitude exactly `2 * area`), so
it checks normals *and* areas; and `symmetric_chamfer` has a sampling noise floor — a mesh against
itself scores ~0.028 on `icosphere(3)`, so a threshold must clear that, not sit under it.

### Where the reference put the answer

A reference that looks like it *disagrees* is more often one whose result was read from the wrong
place. Every one of these was measured, and each first presented as a total failure:

- **MeshLab writes a layer transform, not vertices.** `compute_matrix_by_icp_between_meshes` leaves
  `vertex_matrix()` byte-identical to the input; the answer is `transform_matrix()` /
  `transformed_vertex_matrix()`. Read the wrong one and a converged ICP looks like a no-op.
- **Open3D's tensor meshes must be held in a name.** `o3d.t.geometry.TriangleMesh.from_legacy(x)
  .fill_holes()` lets the temporary be collected and the result reads freed memory — garbage floats
  (2052.1, 4.4e-41) rather than an exception. Bind the intermediate.
- **Open3D's `fill_holes` winds its cap against the rest of the mesh**, so a raw signed volume of its
  output is meaningless (−1.06 where the truth is 2.02). `trimesh.repair.fix_winding` first.
- **A reference's zero is not always "off".** `generate_surface_reconstruction_ball_pivoting`
  reconstructs **nothing** at `clustering=0` (0 faces against 1 277 at its 20% default) and returns
  *faster* for it; `meshing_close_holes` closes nothing at its `maxholesize=30` default. Assert the
  reference produced output before comparing to it.
- **MeshLab has two uniform coordinate umbrellas, and neither is documented.** Recovered by solving
  least-squares for the per-vertex stencil over 12 random position sets on one connectivity (residual
  2e-16): `apply_coord_laplacian_smoothing` and `apply_coord_unsharp_mask` weight each neighbour by
  its shared-face count and include the vertex itself once (`1/(2d+1)` self, `2/(2d+1)` per
  neighbour on a closed mesh), while `apply_coord_taubin_smoothing` uses the plain 1-ring mean. The
  difference is 8% of the displacement — far too large to read as a tolerance. That technique is the
  general one: **one pass of a linear filter is a linear map, so its stencil is solvable.**
- **Pass counts are conventions too.** MeshLab's Taubin `stepsmoothnum` counts lambda-mu *pairs*
  where triwarp and trimesh do one half-step per `iterations`, so the mapping is `2 *
  stepsmoothnum`; and `get_scalar_statistics_per_vertex`'s `"med"` is the sorted element at index
  `n // 2 - 1`, one *below* the middle, for both parities.

### `lexsort` is unusable on float coordinates with ties

`tests.comparisons.lexsort_rows` sorts exactly, so two sides that tie in `float32` but differ in the
16th digit in `float64` order those rows differently and the compare fails by the full coordinate
range (measured 1.59 on a subdivided icosahedron, 0.83 on a star ring). Both are false negatives.
For **positions**, match with a `cKDTree` nearest-neighbour query plus a bijection check, or use
[`hausdorff_two_sided`]; keep `lexsort_rows` for integer index rows, where it is exact.

---

## 7. Python wrapper typing (`triwarp.typing`)

Import once per Python wrapper module:

```python
import triwarp.typing as twt
```

Do **not** re-export typing symbols from `triwarp/__init__.py`; import `twt` where needed.

### Why not `wp.array2d` in wrappers?

At runtime every buffer is `warp.array`. `wp.array2d[dtype]` in Python signatures is a static annotation helper; type checkers do not treat it like a real array (missing `.shape`, bad assignability from `wp.empty`). `isinstance(x, wp.array2d)` is always `False`.

Use **`wp.array[dtype, Literal[ndim]]`** via the aliases in `triwarp/typing.py`:

| Alias | Meaning |
|-------|---------|
| `twt.Array2dInt32` | `(rows, cols)` `int32` |
| `twt.Array2dFloat32` | `(rows, cols)` `float32` |
| `twt.Array1dInt32` | 1D `int32` |
| `twt.IntArray`, `twt.FloatArray`, `twt.ScalarArray` | 1D or 2D unions (e.g. `reduce.py`) |

Kernels in `triwarp/kernels/` keep `wp.array2d[dtype]` unchanged.

### Empty 2D allocation

```python
edges_wp = twt.empty_int32_2d((n_faces * 3, 2), device=faces_wp.device)
angles_wp = twt.empty_float32_2d((f, 3), device=vertices_wp.device)
```

Empty mesh / no adjacency early return:

```python
if n_faces == 0:
    return twt.empty_int32_2d((0, 2), device=faces_wp.device)
```

### Returns and parameters

```python
def faces_to_edges(faces_wp: wp.array[wp.int32], sorted: bool = False) -> twt.Array2dInt32:
    out_wp = twt.empty_int32_2d((n_faces * 3, 2), device=faces_wp.device)
    wp.launch(kernel_graph.faces_to_edges, dim=n_faces, inputs=[faces_wp, out_wp], device=faces_wp.device)
    return twt.as_array2d_int32(out_wp)
```

Optional 2D arguments: `edges_sorted_wp: twt.Array2dInt32 | None = None`.

### Runtime checks (not `isinstance`)

- `twt.ensure_ndim(arr_wp, 2, dtype=wp.int32)` — validate rank and dtype on inputs.
- `twt.as_array2d_int32(arr_wp)` / `twt.as_array2d_float32(arr_wp)` — check then narrow the return type for Pyright.

Do not use `isinstance(..., wp.array2d)`; use the helpers above.

### Tests

Tests may use `import triwarp.typing as twt` for annotations (e.g. `expected: twt.Array2dInt32`). Compare via `.numpy()` and `np.array_equal(got, exp)` as in `tests/test_graph.py`.

---

## 8. Device Checks

**Do not check that input arrays share the same device.** Manual `if arr.device != device: raise ValueError(...)` guards are redundant — the harness rejects the mismatch for you (below), and §14 forbids a docstring documenting a `ValueError` for arrays "on different devices". Omit them entirely.

**But do not believe that `wp.launch` raises on a device mismatch — it has not since Warp 1.14.** That release removed the unconditional same-device check (`NVIDIA/warp` GH-1461) so that hardware-coherent launches would be legal, and the default `wp.config.launch_array_access_mode` is `RELAXED`, which passes the pointers straight through and validates nothing. On this box the consequences are asymmetric and both are silent:

- **CPU arrays, CUDA launch** (a launch that forgot `device=`, resolving to `cuda:0`): the GPU reads the host arrays over HMM (`is_cpu_memory_access_from_gpu_supported` is `True` here) and computes the **right answer** — then the launch is *asynchronous*, so when those host arrays are freed while the kernel is still running the heap is corrupted and the process aborts in `malloc` much later. Measured on a 97-line repro: 20/20 aborts when the arrays are freed without a sync, **0/20** when nothing is freed (`os._exit`), and **0/20** when `wp.synchronize()` precedes the free. The same pattern on `cuda:0` arrays is safe because CUDA frees are stream-ordered; host frees carry no ordering, and nothing at the call site distinguishes the two.
- **CUDA arrays, CPU launch**: immediate `SIGSEGV`, no Python exception (GH-1693).

Four consequences for triwarp:

- **`tests/conftest.py` sets `LaunchArrayAccessMode.STRICT`**, the only mode that rejects a *genuine* cross-device argument; `CHECKED` validates addressability, which HMM genuinely provides, so it permits the launch and still corrupts (measured 2/30). The full suite passes under `STRICT` (2 113 tests), so no triwarp launch is intentionally cross-device — keep it that way.
- **`STRICT` alone would not have caught the bug that motivated it, which is why check 15 exists.** It only fires when an argument is *not* on the launch device, so on a CUDA run — this box, and CI — an omitted `device=` resolves to `cuda:0`, which *is* the arrays' device, and nothing is rejected. The corruption then waits for a CPU run. `test_launches_name_their_device` scans every launch site statically and is the half that sees it; the two guards cover different halves and neither replaces the other.
- **§3's "always forward the `device`" is a memory-safety rule, not a tidiness one.** All 476 `wp.launch` / `wp.launch_tiled` calls in `triwarp/` name a device; a new one that does not is the defect above.
- **`.numpy()` is not a sync on a CPU array.** On a CUDA array it synchronizes; on a host array it is a zero-copy view, so "I read the result and it was correct" proves nothing about whether the kernel finished.

---

## 9. Warp API Reference

Authoritative Warp function lists are mirrored locally under `reference/warp_api/`:

| File | Scope |
|------|-------|
| `reference/warp_api/builtins.md` | Built-in functions usable inside `@wp.kernel` / `@wp.func` (`wp.<name>`) |
| `reference/warp_api/warp.md` | `warp` module API at Python scope (`wp.<name>`) |
| `reference/warp_api/sparse.md` | `warp.sparse` BSR/CSR matrix API |
| `reference/warp_api/utils.md` | `warp.utils` Python-scope utilities |
| `reference/warp_api/fem_linalg.md` | `warp.fem.linalg` linear-algebra utilities |

BEFORE using an unfamiliar Warp builtin, sparse, or utils function, `grep` these files to confirm the exact name, signature, and scope rather than guessing. Each file stamps the Warp version it was transcribed from, and its source URL, at the top — fetch the URL for full argument details or examples when the one-line description is insufficient. Do not restate that version here; run `uv run reference/warp_api/warp_version.py` to compare every stamp against the installed `warp-lang` and see which files a Warp upgrade has left stale. `reference/warp_api/REGENERATE.md` records how to re-extract them.

---

## 10. Documentation (MkDocs + mkdocstrings)

Docs are built with **MkDocs Material** + **mkdocstrings** (`python` handler, `docstring_style: numpy`), configured in `mkdocs.yml`. `docs/gen_ref_pages.py` auto-generates one API reference page per public module under `triwarp/` on every build, including subpackages such as `triwarp/heat/` (named by dotted path, e.g. `heat.distance`; `triwarp/kernels/` is excluded) — a new module needs **no manual nav entry**, it appears automatically. Preview locally with `DISABLE_MKDOCS_2_WARNING=true uv run mkdocs serve`; validate with `DISABLE_MKDOCS_2_WARNING=true uv run mkdocs build --strict` (fails the build on any broken cross-reference or unresolved external inventory). The `DISABLE_MKDOCS_2_WARNING` prefix silences a promotional banner injected by the `properdocs` transitive dependency of `mkdocs-gen-files`/`mkdocs-literate-nav`/`mkdocs-section-index`.

**The one-line summary says what the function returns, never which C++ call it wraps.** mkdocstrings renders that first line as the function's entry in its module's API index, so a reference library's name there turns the index into a table of bindings — `laplacian.cotmatrix` read `"Cotangent stiffness matrix / discrete Laplacian (``igl::cotmatrix``)"` where it should read `"Cotangent stiffness matrix of the mesh: the discrete Laplace-Beltrami operator."` This is §14's naming rule one level out. Attribution is *wanted* and stays — 32 of the 49 wrapper modules mention a reference library somewhere in their prose, and that is right — but one line down, in `Notes` or `See Also`. Enforced by `tests/test_api_conventions.py`, whose allowlist is `mesh.py`'s "mirrors `trimesh.Trimesh`" alone (the property names are chosen to match, which a reader has to be told at the top). The same test forbids a module summary ending in `(Warp)` or `on NVIDIA Warp`: the whole package is Warp.

Docstrings stay **NumPy-style** (`Parameters`/`Returns`/`Raises`/`See Also`), but cross-references use **mkdocs-autorefs** link syntax, not Sphinx roles — Sphinx interpreted-text roles (`:func:`, `:attr:`, `:meth:`, `:class:`, `:data:`, `:mod:`) have no Markdown equivalent and render as literal, broken text (e.g. `:func:`x``) under MkDocs.

### Cross-reference rules

| Target | Syntax | Example |
|---|---|---|
| Internal (`triwarp.*`) | `` [`short_name`][fully.qualified.path] `` | `` [`face_adjacency`][triwarp.graph.face_adjacency] `` |
| External **with** a configured inventory (`trimesh`, `numpy`, `scipy`, stdlib) | `` [`fully.qualified.name`][] `` (empty brackets — resolved via the `inventories:` list in `mkdocs.yml`) | `` [`trimesh.grouping.group_rows`][] `` |
| External **without** an inventory (e.g. `warp`, `igl` — no public Sphinx inventory exists) | Plain double-backtick code span, no link | ``` ``warp.sparse.BsrMatrix`` ``` |
| Shapes, literals, C++ names (`igl::foo`), file paths | Leave as plain single/double-backtick code — already renders correctly, no role needed | ``` ``(n_vertices,)`` ``` |

- Always resolve internal refs to the **fully-qualified path** as the link target, even when referencing a function in the same module or using a bare/unqualified name (numpydoc's old auto-linking of bare `See Also` names does not carry over to mkdocstrings — every `See Also` entry needs the explicit `[`name`][full.path]` form).
- Before adding a new external inventory to `mkdocs.yml`, verify it actually serves an `objects.inv` (`curl -I <url>/objects.inv`) — do not assume one exists just because the project has Sphinx docs.
- RST admonitions (`.. note::`, etc.) don't exist in Markdown — use MkDocs Material's `!!! note` admonition syntax instead (requires the `admonition` / `pymdownx.details` extensions, already enabled in `mkdocs.yml`).
- Module-level constants/type aliases without their own docstring are only linkable because `show_if_no_docstring: true` is set in `mkdocs.yml`; don't remove that option without re-checking `triwarp/constants.py` and `triwarp/typing.py` cross-refs still resolve.
- Every public function should have a docstring — undocumented public functions still get a page entry (via `show_if_no_docstring`) but render with an empty description, which looks broken in the generated site.
- After editing docstrings, sanity-check with `grep -rnE ':(func|attr|meth|class|data|mod):\`' triwarp/` — it should return nothing. (`kernels/` counts too: nothing there renders, but a reader meets the broken role text all the same.)

---

## 11. Function Ordering Within a Module

`mkdocstrings` is configured with `members_order: source` (see `mkdocs.yml`), so **source order is the rendered docs order** — placement in the file is part of the public API's discoverability, not cosmetic.

### Which module a function belongs in

- **One operation family, one module.** Functions with the same shape of signature computing the
  same *kind* of answer belong together, and a family split across two modules is a defect however
  reasonable each half looked when it landed. The five per-point surface descriptors
  (`ambient_occlusion`, `volumetric_obscurance`, `shape_diameter`, `thickness`,
  `max_tangent_sphere`) sat in two modules while sharing one kernel module, and
  `benchmarks/test_proximity.py` had already voted by holding all five rows — **the suite disagreeing
  with the split is the signal to look for**, since coverage follows the family and not the file.
- **Place a function by what it computes and what machinery it shares, not by where the reference
  library keeps it.** Several modules mirror a trimesh module, and mirrored membership is a weak
  reason: `triangles.volume` / `moments` / `centroid` sat in `triangles.py` because
  `trimesh.triangles.mass_properties` exists, and they are whole-mesh reductions in a per-triangle
  module. Names keep their trimesh / igl spelling where that is the field's vocabulary — this rule
  is about *placement* only.
- **The machinery half of that test outranks the subject half, and it is measurable.** A function
  that is a *component* of another module's solver stays with it, however well it reads elsewhere:
  `heat.distance.heat_geodesic` computes geodesic distance, but `heat/vector.py` calls it,
  `VectorHeatOperators` embeds its operator tuple, and `log_map`'s radius is asserted to be its
  answer — moving it would invert the dependency. Check with an import/call scan before moving
  anything, not by reading.

### Python wrapper modules (`triwarp/*.py`)

- File layout: module docstring → imports → module constants / type aliases → functions.
- **Group public functions thematically, then order the groups by importance and expected frequency of use.** The primary entry points a user reaches for first go at the top of the file; niche or low-level variants go last. Within a group, put closely related functions consecutively (e.g. `expand_vertex_mask` / `shrink_vertex_mask` in `selection.py`, placed after the more frequently used `submesh_from_*` family), and put the simple form of an operation before its advanced variants (`query_bvh_ball` before `query_bvh_ball_with_offsets` / `query_bvh_ball_count`).
- **Stepdown rule for private helpers:** a private function called by exactly one public function goes **immediately after** that function. A private helper shared by several functions in the same file goes after its **last** caller. Cross-cutting utilities (validation guards, tiny converters used throughout the file) go in a trailing "private helpers" section at the bottom. A private helper must never appear above its first caller — a reader should never need to jump backward to a definition they haven't been introduced to yet.

### Kernel modules (`triwarp/kernels/*.py`)

The stepdown rule **inverts** for kernel helpers: Warp resolves `@wp.func` references at kernel-decoration time, so a `@wp.func` **must textually precede** every kernel (or other `@wp.func`) that calls it — Python's usual "helper after its caller" convention is not legal here. Order kernels to mirror their wrapper module's public-function order, and place each kernel's `@wp.func` helpers immediately **before** that kernel; a `@wp.func` shared by several kernels in the file goes before its **first** user.

### Tests (`tests/test_<module>.py`)

Mirror the corresponding wrapper module's public-function order, so a reader scanning tests top-to-bottom sees the same story as the API page.

---

## 12. Tooling & Local Validation

Two tools are configured in `pyproject.toml` and are the authority for style and typing — do not hand-roll equivalents or add competing tools. Run them (and the tests) after any change to `triwarp/` and before considering work done.

### Ruff (`[tool.ruff]`) — lint + format

Ruff is the **only** linter and formatter (no black/isort/flake8). Config lives in `[tool.ruff]`; do not override it inline.

```bash
uv run ruff format triwarp tests     # format (100-col, skip-magic-trailing-comma)
uv run ruff check --fix triwarp tests # lint + autofix
```

- Enabled rule families and per-file ignores are in `[tool.ruff.lint]` — respect them; do not add blanket `# noqa` to silence a rule the config selects. If a rule is genuinely wrong for a construct, prefer a scoped `per-file-ignores` entry over inline suppression.
- Import conventions are enforced: `import numpy as np`, `import trimesh as tm`, `import warp as wp` (`[tool.ruff.lint.flake8-import-conventions.aliases]`).
- Docstrings are enforced (`D`), so every public function needs one (see §10).
- `reference/` is excluded from both lint and type-checking — never edit vendored code to satisfy a tool.

### basedpyright (`[tool.basedpyright]`) — type checking

basedpyright (a Pyright superset) is the configured type checker — it is what the IDE runs, so match its verdict. Config lives in `[tool.basedpyright]`.

```bash
uv run basedpyright               # type-check triwarp/ (uses pyproject config)
```

- **`triwarp/kernels/` is excluded.** The Warp kernel DSL (`wp.array[wp.vec3]` subscripting, `wp.tid()`, typed vector constructors, kernel-scope slicing) is not modeled by any stubs and is inherently un-typecheckable. Do not try to make kernels type-clean or add `# pyright: ignore` there.
- **Warp-stub type-flow rules are disabled** (`reportArgumentType`, `reportCallIssue`, `reportReturnType`, `reportAttributeAccessIssue`, `reportIndexIssue`, `reportOperatorIssue`, `reportGeneralTypeIssues`, etc.). Warp's Python-scope stubs are weak (`wp.empty` typed as returning `array[float]`; the `twt.Array2d*` aliases not assignable to `array[Unknown, int]`; `BsrMatrix.offsets/.columns/.values` absent from the stub; `wp.max`/`wp.mean` overloads unmatched), so these rules fire almost entirely on false positives. **Consequence:** basedpyright will *not* catch genuine argument/return/index type errors in wrappers — rely on the §6 Trimesh regression tests for correctness, not the type checker.
- **`reportPossiblyUnboundVariable` is kept as an error** — it catches the §5 gotcha (a variable assigned only inside an `if` branch, read as uninitialized when the branch isn't taken). When it fires on a *correlated* condition (e.g. two separate `if is_mesh:` blocks), fix it the way `triwarp/registration.py` does: initialize the variable to `None` before the branch and `assert x is not None` at the use site — do **not** suppress it.
- The gate is expected to stay at **0 errors**. If basedpyright reports an error, it is almost always either a real possibly-unbound bug or a missing dependency — resolve it, do not add suppressions or widen the disabled-rule list without cause.

### Full environment for validation

basedpyright and the test suite need the optional/test dependencies (e.g. `meshio`, `trimesh`) resolvable, so sync all groups first:

```bash
uv sync --all-groups
```

Running basedpyright in a dev-only env yields spurious `reportMissingImports` on `meshio` and other test-group packages.

---

## 13. Performance Work: Measure Before You Change

- **Something suddenly slow is a Warp rebuild until proven otherwise — check that first.** Before
  profiling anything, before believing a kernel got slower, before deleting a test for being slow:
  a single-digit-second operation that now takes tens of seconds, or a test file whose cost appears
  and disappears as you change *which* tests you select, is almost always a module recompile from
  an unregistered generic-kernel overload (§4). Measured: `tests/test_reduce.py` took **1 561 s** on
  a fresh selection and **1.30 s** repeating the identical one — same tests, same asserts, same
  machine, 1 200x apart — and the whole suite ran 1 033 s where the tests themselves account for
  ~29 s. Two confirmations, both cheap:
  ```bash
  # 1. The compile is single-threaded nvcc, so the GPU is idle while the clock runs.
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader   # 0 % during the stall
  # 2. Warp says so outright.
  uv run python -c "import warp as wp; wp.config.verbose = True; ..." 2>&1 \
      | grep -E "Module hash changed, recompiling|took .* ms  \(compiled\)"
  ```
  Any `Module hash changed, recompiling: <module>` line for a `triwarp.kernels.*` module is the
  defect: add the dtype to that module's `_register_overloads`. A *second* line for the same module
  in one run means the chain is still forking. This is the first thing to inspect because the
  alternative reading — "this test is inherently slow, cap its input or delete it" — removes
  coverage to work around a fixable compile, which is exactly the trade that hid the problem for as
  long as it lasted.
- **A benchmark lands before the optimization does.** Never restructure code for speed without a
  `benchmarks/test_<module>.py` group timing the *current* implementation first. A belief about
  where the cost sits ("two Python loops", "too many derived launches") is a hypothesis until that
  group exists and prints a number.
- **Attribute a change only with a back-to-back A/B in one session.** Saved baselines drift ±10%
  (±30% under 100 µs) between sessions, so a comparison against a stored number proves nothing —
  run old and new in the same session, in the same process order.
- **Verify values, not just timing.** This is the §4 gather warning generalised: the *wrong*
  implementation is frequently the faster one, because it reads less. Every perf change must keep
  its parity / regression test green, which means a function about to be optimized needs one first.
- **CUDA is the target; a CPU regression is an acceptable price for a GPU win.** triwarp exists to
  run geometry on the GPU, so when the two devices disagree, **decide on the CUDA number.** Do not
  reject a device-side change because Warp's CPU backend is slower at it — Warp's CPU reductions run
  ~1 lane per block while NumPy's are vectorized C, so a host readback plus NumPy wins on CPU almost
  every time and would veto nearly every reduction if it were allowed a vote. Still *measure* both
  (the CPU path must stay correct, and the ratio belongs in the comment), and still decline a change
  that wins nowhere. Worked example — the seven candidate sites in the "NumPy reducing a full
  readback" sweep: on CPU, four lose 4-25x and only `moments` / `_mean_rim_edge_length` win outright,
  so a CPU-decided sweep converts two. On CUDA **all seven** win once the array is big enough, with
  crossovers spread over 100k-1M elements — so the CUDA-decided sweep converts seven and leaves five
  fewer defects. The case to imitate when declining is the *eighth* site,
  `graph.bfs_multi_source`: it keeps its host range check because `sources` is `k` seeds and never
  grows with the mesh, so there is no GPU gain to be had at any size — **"no gain on CUDA" is the
  reason to decline, not "slower on CPU."**
- **A device reduction costs ~0.1–0.3 ms flat on CUDA; a readback costs bytes.** The crossover is
  wherever the copy exceeds ~0.15 ms, which measured out at ~200k `int32`, ~200k `float32`,
  ~1M `bool` and ~16k `vec3d` elements. Below it a reduction launch is pure overhead; above it the
  readback grows without bound (`.numpy().sum(axis=0)` over `(n_faces,)` `vec3d` moves 72 B/face —
  33.6 ms on `dragon`, against 0.19 ms for four `wp.utils.array_sum` calls). So size the array
  before reaching for either, and prefer the reduction on any buffer that scales with the mesh.
- **Interleave A and B in one loop, and read the `min`.** Timing all of A then all of B lets GPU
  clock state decide the winner: the first pass of this same sweep produced non-monotonic ratios
  (8.4x, 0.03x, 4.35x for one site across three sizes) and a recurring ~2.27 ms artifact, which
  reversed into a clean monotonic trend once the variants were interleaved under one clock state
  with the GPU pre-warmed. Report the `min` alongside the median: a one-off Warp kernel compile or a
  scheduler hiccup inflates a median but cannot deflate a minimum.
- **Cost model for wrappers.** Host-side Python is a real cost: ~11 µs per cached `wp.map` call,
  ~32 µs of launch marshalling per `wp.launch` (measured in `combine.concatenate`), ~0.1 ms per
  host readback, against 0.9–2.4 ms for a full extra device pass on a mid-size mesh. So: collapse
  per-item launches into one launch over a packed buffer, and single-pass the host-side metadata
  loops — but do not chase Python microseconds in a wrapper whose cost is really per-segment
  `wp.copy` launches. Measure which regime you are in before optimizing for either.
- **That ~32 µs is the *mean* kernel's launch, not a constant: a launch costs ~1.0 µs per
  argument.** Measured over kernels differing only in argument count, small `dim` so marshalling
  dominates, interleaved, 400 reps — 15 µs at 2 arguments, 22 at 8, 31 at 16, 41 at 28, linear, and
  **identical on CUDA and CPU**, so this one is not a device judgement call. The package's mean
  kernel takes 5.1 arguments and only 18 of 440 take 12 or more, so this matters in exactly one
  place: a wide kernel launched inside a Python loop. Bundle its invariant arrays into a
  `@wp.struct` built **once** in the wrapper (construction is ~2.6 µs; rebuilding per launch gives
  the saving back), which collapses them to a single argument — measured 1.17–1.93x on
  `holes._fill_dp` and 1.34x on `reconstruction._bpa_wave`'s launch path, or ~1.08 µs per dropped
  argument. Struct fields take the §2 subscript annotation (`a: wp.array[wp.int32]`). Do **not**
  open a tree-wide bundling pass: bundling a 5-argument kernel trades ~3 µs for a struct the reader
  has to open, and §14's "no speculative generality" applies to argument bundles too.
- **Budget the host–device syncs.** Every `.numpy()` / `int(<device value>)` readback in a wrapper
  carries a comment naming why it is unavoidable. When the caller can supply the bound the readback
  infers, expose it as a keyword (`face_adjacency(n_vertices=...)`,
  `hash_indices_rows(validate=False)`) **and pass it from every in-repo caller that knows it** — an
  escape hatch nothing uses is not an optimization. Note that trading one readback for an extra
  device pass is usually a *loss*; see the cost model above.
- **Size buffers for their final use at allocation time.** Do not allocate-then-grow at Python
  scope. When a consumer needs an `n + 1` sentinel-terminated form, the *producer* allocates
  `n + 1` and hands back a view (`counts_to_offsets`); a helper whose only job is to patch up
  another function's output convention is a smell to be fixed at the producer.

---

## 14. Evolving the Public API

**`tests/test_api_conventions.py` is the mechanical half of this section**, and it fails the default
`pytest` run. Fifteen checks. Eight scan the public surface of `triwarp/` (excluding `kernels/`): a
summary line naming a reference library (§10); a `*_mask` producer that does not return
`wp.array[wp.bool]`; a module summary advertising Warp; a module without a `tests/` **and** a
`benchmarks/` file named for it; a private name reached across a module boundary; one public name
exported by two modules; a top-level `kernels/<name>.py` without its `triwarp/<name>.py` or the
reverse (§4); and a private helper defined above its first caller (§11). The ninth scans `kernels/`
**as well**: a comment or docstring blaming a Warp version older than the installed `warp-lang`.
Two enforce an earlier section's convention on kernel code: §3's `out_` prefix and end-of-signature
position for a written argument (its two exemption classes carried as `_KERNEL_OUTPUT_ALLOWLIST`),
and §2's subscript-style array annotation — the latter scans the whole package, because only in an
*annotation* position is `wp.array(dtype=T)` the stale spelling rather than a legal allocation.
The last four are newer and each exists because the same defect was found twice:

- **A `wp.launch` / `wp.launch_tiled` with no `device=`.** Check 15, and it is a memory-safety guard
  rather than a style one — see §8 for the measured failure. It is also the *load-bearing* half of
  that guard: `tests/conftest.py`'s `STRICT` mode only fires when an argument is genuinely off the
  launch device, so on a **CUDA** run the omitted argument resolves to the arrays' own device and
  nothing is rejected, which is precisely how the original defect passed CI and hurt only CPU users.
  Only the scan sees it there.

- **An allocation with no `device=`.** `wp.zeros` / `empty` / `ones` / `full` / `array` at Python
  scope land on Warp's *current* device, and the suite cannot see the difference because a test
  runs with its arrays' device already current — `array.index_sparse`'s `wp.ones` was wrong for
  as long as it existed and every test passed. Scans all of `triwarp/` including `_*.py` modules,
  since a misplaced buffer is not a question about the API's shape.
- **A public function that raises with no `Raises` block.** Only a *direct* `raise` in the
  function's own body counts; the 42 functions that delegate validation to a shared guard and
  document its `Raises` are correct and are not scanned.
- **A fenced ```python docstring example that does not run.** The one static check that is not
  static: `tests/api_conventions.py` extracts the blocks and `tests/test_api_conventions.py`
  `exec`s them against a mesh fixture, because both defects it was written for were *runtime*
  ones (`wp.array` compared with a float, a NumPy bool array handed to `flatnonzero`) and
  `ast.parse` sees nothing wrong with either. Blocks holding a bare `...` are deliberate outlines
  and skip. A new example that needs a name the fixture does not bind fails with `NameError` —
  extend `example_namespace`, do not weaken the test.

Each check carries a written allowlist — read the reason before adding an entry, and prefer fixing
the code. It does not replace review: it cannot tell whether a *new* name is a good one, only that
it does not break a convention the package already holds to.

**A Warp version claim is spelled `Warp 1.16`, with the word immediately before the number.** The
staleness check reads that anchored form and nothing else, because this package writes measured
ratios in the same shape (`within 1.25x of best`, `1.06 ms`, `1.13x on CUDA`) and a bare `1.N` token
matched 30 of those against 3 real claims. So write "still present in Warp 1.16.0", never "still
present in 1.16.0" with the word three lines up — the looser form is invisible to the check and will
survive the next upgrade unexamined, which is exactly the failure the check exists to catch.

- **Name a function after what it returns, in NumPy vocabulary — never after the Warp call it
  wraps.** `sort_pairs` named `warp.utils.radix_sort_pairs`'s key/value mechanism rather than its
  result (a sort *and* an argsort), which is why it became `sort_and_argsort`.
- **Docstring, signature, and body must agree.** Three specific gates:
  - A documented `Raises` must be reachable. In particular §8 forbids device-mismatch checks, so
    **no docstring may document a `ValueError` for arrays "on different devices"** — Warp raises
    that itself, and the wrapper never does.
  - A documented validation must actually be performed, or the claim goes.
  - Annotations must cover every rank and dtype the docstring claims and the body supports (a
    docstring promising rank-2 support needs an annotation that admits rank 2).
- **No public signature or return type may name `np.ndarray`**, outside `triwarp/io.py` where
  meshio makes it unavoidable — a caller should not need NumPy to *consume* a triwarp answer. See
  §4 for what to use instead, and for why internal host-side NumPy is fine.
- **A guard must encode a real limitation.** When the implementation is naturally rank- or
  dtype-agnostic — a flatten/reshape, a generic `@wp.func` — drop the `ensure_ndim` cap and widen
  the annotation instead of validating a restriction that is not there.
- **Prefer dtype-generic `@wp.func`s** (`wp.Float` / `wp.Scalar` for scalars, `Any` for vectors —
  the `kernels/predicates.py` convention) over hardcoded `float32` / `vec3` variants, *as long as
  the dispatch stays readable*. Where Warp cannot express the generic — there is no `wp.any` /
  `wp.all` over vector components and no generic vector annotation — keep named per-type funcs
  behind a small dtype-keyed dispatch rather than contorting the kernel.
- **No speculative generality.** Add an axis, parameter, or mode only when an in-repo call site
  needs it. The absence of a caller is a reason not to build it, not a gap to fill.
- **No near-duplicate wrappers.** Two public functions that are the same algorithm with different
  returns share one private helper (`concatenate` / `pack_1d_arrays` behind `_pack_segments`).
- **Inverse and dual pairs cross-reference each other and have a round-trip test** — e.g.
  `flatnonzero` / `indices_to_mask`. Bidirectional `See Also` is required for inverse pairs and for
  simple/advanced variants of one operation; it is *not* required for hub→spoke references (most
  of the ~250 one-way links in the package are correct — `cotmatrix` should not list every
  consumer).
- **Coverage is per module.** Every public `triwarp/<module>.py` gets both `tests/test_<module>.py`
  and `benchmarks/test_<module>.py`, and a function's tests live in the file mirroring *its* module
  (§11), not in a neighbour's.
- **Moving or renaming a public function moves everything derived from it — in the same commit.**
  A move is not done when the wrapper compiles; it is done when nothing still points at the old
  home. Five artifacts, every time:
  1. **Its kernels, if they are exclusively its.** A kernel referenced by only the moved function
     moves to the destination's `kernels/` module; a kernel shared with a function that stays put
     does **not** move, and the new kernel module imports it (kernel-to-kernel imports are normal —
     `kernels/predicates.py` has 17 importers). Decide by measuring, not by reading: an AST scan of
     which wrappers reference each `kernel_<mod>.<name>` is the authority, because a kernel that
     *looks* single-purpose is often reached from a private helper in a third module.
  2. **Its tests**, into `tests/test_<destination>.py`, keeping the §11 source order.
  3. **Its benchmark rows**, into `benchmarks/test_<destination>.py`.
  4. **Its `benchmark(group=...)` name**, when the group is named after the function or its old
     module. Renaming a group is allowed and is sometimes required to keep the suites consistent —
     but the group name is the parity key, so **every `parity` / `noparity` marker citing it must be
     updated in the same commit**, and `uv run python -m tests.parity` must show the same pair count
     before and after (the *names* change, the matrix shape does not).
  5. **Its docs entry** in `docs/gen_ref_pages.py` `SECTIONS`, plus every
     `[`name`][triwarp.old.path]` cross-reference — `mkdocs build --strict` is what finds the ones
     you missed.

---

## 15. Running Long Commands: Never Poll With `until`

The benchmark suite and the measurement probes §13 asks for routinely run for minutes. **Do not
write a wait loop around them.** A backgrounded command re-invokes you when it exits, reporting its
exit code and its output file path — verified in-session: a 25-second background command returned
`completed (exit code 0)` on its own with no polling. Launch it, do something else, and read the
output file when the notification arrives.

- **`until ! pgrep -f <name>; do sleep 5; done` never terminates.** `pgrep -f` matches the *full
  command line of every process*, including the polling shell itself, whose command line contains
  `<name>`. The condition is therefore permanently true. Measured cost of not knowing this: **seven
  such loops in one session, each spinning for 3-4 hours** until killed by hand, every one of them
  redundant because the command it was watching had already sent its completion notification and its
  output file was already on disk.
- **A watcher is never the record.** The benchmark's or probe's own stdout file is. If you find
  yourself launching a second command to learn whether the first finished, delete it.
- **When a poll genuinely is unavoidable** — external state the harness cannot see, such as a CI run
  — make the pattern unable to self-match (`pgrep -f '[p]robe_p4'`) or test a sentinel file the
  process writes on exit, and give the loop a **bounded** iteration count so a logic error costs
  seconds instead of hours.
- **Do not chain short sleeps** to approximate a long wait. Pass a longer `timeout` to the command,
  or background it and let the notification arrive.

A foreground command that outruns its timeout is *also* moved to the background and notified the same
way, so exceeding a timeout is not a reason to start polling either — the result is still coming.
