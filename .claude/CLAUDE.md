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
- **A constructor of the type a value already has is the same noise, and the cast scan does not see
  it.** `wp.vec3(*(vertices[face[1]] - vertices[face[0]]))` splats a `wp.vec3` and rebuilds it; the
  difference of two `wp.vec3`s is already a `wp.vec3`. Check 16 classifies *casts*, so a
  `wp.vecN(*(...))` / `wp.matNM(*(...))` splat-and-reconstruct survives it — three such sites
  outlived the pass that deleted 184 redundant casts, in `triangles.triangle_edges`. When deleting
  one class of no-op conversion, grep the constructor spelling too, and read the operand's type
  rather than trusting the scan's silence.
- Use `wp.launch(kernel=..., dim=..., inputs=[...], device=...)` for execution. Always forward the `device` from the input arrays.
- Array slicing is supported inside kernels: `faces[f * 3 : (f + 1) * 3]` produces a sub-array view.
- Use `wp.cast(expr, TargetType)` for explicit type conversions between Warp types — but **only
  between types of the same size**: it is a bit reinterpretation, and a width change fails at
  *NVRTC* time with `static assertion failed with "source and destination must have the same size"`,
  not at Python time. `wp.cast(cross(a, b), wp.vec3)` is fine; `wp.cast(x_f64, wp.float32)` does not
  compile. Widening or narrowing a scalar is the **constructor**: `wp.float32(x)`, `wp.float64(i)`.
- Prepend output argument names with out_ and put them at the end of the kernel signature after all the input arguments. Two exemption classes, both carried as `_KERNEL_OUTPUT_ALLOWLIST` in `tests/api_conventions.py` (check 13): **in-place** arguments, where the same buffer is input and result (`sort_rows_insertion(data)`, the hole-filling DP tables) — an `out_` prefix would misread as write-only; and **scratch / persistent-state** buffers, caller-allocated working memory carried across launches (cursors, stacks, open-addressing tables, `ball_pivoting`'s front) — neither an input nor the answer, so name them for what they hold (`cursor`, `front_out`, `new_src`). A read-only input must never wear the `out_` prefix, even when the buffer was a *producer* kernel's output — parameter names describe the argument's role in *this* kernel.

### A lane-parallel kernel strides by `wp.block_dim()`, or it stays lane-free

A kernel whose lanes cooperate — a `wp.tile_sum` / `tile_min` / `tile_max` over one value per lane,
a `wp.tile_bvh_query_aabb` walk — is correct on **both** devices exactly when its lanes partition a
sequence **the block already owns**, with the stride taken from `wp.block_dim()`. Never from a
kernel argument, never from a module constant.

The reason is that `wp.launch_tiled` runs exactly **one lane per block on the CPU device** through
Warp 1.16 whatever `block_dim=` is passed, and `wp.block_dim()` reads `1` there — so that single
lane walks the whole sequence and the one-element tile it reduces holds the right answer. **A
one-element tile is not the bug.** Measured on one 1 000-element sum whose stride came from an
`n_slices` argument instead:

| device | `n_slices` | `block_dim` | result | expected |
|---|---|---|---|---|
| cpu | 64 | 64 or 256 | **16.0** | 1000.0 (short by exactly the stride — one lane walked 1/64) |
| cuda:0 | 64 | 64 | 1000.0 | 1000.0 |
| cuda:0 | 64 | 256 | **3616.0** | 1000.0 (lanes 64-255 re-walk what 0-63 counted) |

So the arg-strided form is wrong on **both** devices, and its correctness on CUDA silently depends
on a *wrapper* passing `n_slices == block_dim`, which no signature expresses. Do not read a CPU
failure of a tiled kernel as "tiles do not work on the CPU backend" — that reading cost this package
four kernels that were converted *away* from tile reductions and four comments stating the
prohibition in general terms, all of which the block-per-item kernels then contradicted correctly.

Where the lanes would partition the **outer** work the grid is over — a whole-array reduction with
no per-item dimension, `measures.centroid_tiled`, `metrics.chamfer_*_tiled` — there is no
block-owned sequence and no `wp.block_dim()` to take. Those kernels must either keep a *constant*
stride and be launched on CUDA only, with a lane-free `_sliced` sibling for the CPU (the
`_device.prefers_tiled_reduction` pair), or stay lane-free on both. Both are correct; a single
kernel is not available.

Read `kernels/visibility.py::obscurance` (one block per point, lanes over its ray bundle, measured
3.2-11.8x against one thread per point) for the first form and `kernels/measures.py::centroid_tiled`
for the second.

**Eligibility is an occupancy question, not a shape question, and getting that backwards costs 2-8x.**
Having an outer per-item dimension is necessary and not sufficient: the block-per-item form pays
exactly when that outer dimension *alone* would starve the device. `obscurance` qualified because it
launched `dim = n_points` — 8 171 threads, under 3 % of an RTX 5090 — with no second dimension at
all. A kernel that already carries a **slice dimension** does not qualify, because that dimension is
what fills the device and collapsing it into `block_dim` lanes throws the occupancy away. Measured on
`points.hull_support_extremes`, converting it both ways:

| points | grid today | block-per-direction |
|---|---|---|
| 5 000 | 13 x 40 threads | **2.3x faster** (the grid was starved) |
| 200 000 | 13 x 1 563 threads | **0.12-0.60x — a 2-8x loss** (13 blocks on 170 SMs) |

`proximity.winding_number_tiled` shows the same trend from the other end: 2.16x at 1 280 faces and
4 096 queries, 1.19x at 5 120 faces, **1.01x** at 65 536 queries — a gain that shrinks to nothing as
the grid fills, which section 13 calls a decline rather than a small win. So four kernels in this tree
keep the arg-strided form deliberately, and each says so with its number.

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

**A comment that names the other copy is the finding, not the fix.** *"Feature handling mirrors
`collapse_candidates`"*, *"the `dart_priorities` argument"*, *"every candidate kernel opens with the
same two lines"* — each of those was written by an author who had already seen the duplication and
answered it with prose. A cross-reference is a claim that **only a shared `@wp.func` can keep true**:
the two bodies are equivalent on the day it is written and nothing stops the next edit to one of them
from diverging silently. So when writing "same as X" / "mirrors X" / "as in X" about *code* rather
than about a reason, extract instead; and when reading one, treat it as an unfactored duplicate that
has been located for you. **Re-count the run from the code, not from the comment** — the "same two
lines" above was seven statements, and the prose had been factored where the code had not.

**A clean duplicate scan is not a clean file.** Every scan this package has used keys on α-renamed
statement *text*, so two bodies expressing the same decision through different control flow — a
`reject` flag against an early `return`, a position against a boolean — score as unrelated. The
best find of the fourth `kernels/` pass was exactly that shape and came from reading a closed item,
not from a scan: `remesh.collapse_candidates` and `quadric_collapse_candidates` hold one 4-way
feature-collapse rule twice, one through a flag and one through returns. **A duplicated *decision
rule* is a live correctness hazard where a duplicated arithmetic run is only noise**, so it
outranks the longer runs a scan does find: extract it as a `@wp.func` returning the classification
(a sentinel for "reject", the `_resolve_flip_quad_guarded` convention) and let each caller map the
free branch onto its own answer.

**Factor the family, not the pair — a helper pinned to one rank or one precision breeds the copies
it was meant to prevent.** Two measured instances: `corner_triple` landed for flat 3-stride buffers
and reached 16 callers while its rank-2 sibling (`arr[row, 0..2]`) stayed nameless at eight sites;
and `laplacian.squared_edge_lengths`, hardcoded `wp.vec3` / `float32`, could not be reached by the
three `float64` sites in `energies.py`, which open-coded it instead. So when extracting, write the
generic form §14 asks for (`wp.Float` / `wp.Scalar` / `Any`, the `kernels/predicates.py` convention)
and check for the *other* rank and the *other* precision before declaring the run named. This is
also the recurring half of rule 2 above: a general per-triangle quantity left in an algorithm module
is now a **three-time** defect — `triangle_aabb` in `intersection.py`, `triangle_double_area` and
`circumcircle_diameter` in `holes.py`, `squared_edge_lengths` in `laplacian.py` — and the tell is a
kernel module importing an *algorithm* to reach *geometry*.

**One predicate, one spelling per module — this is a correctness rule, not a style one.**
`wp.length(d) < r` and `wp.length_sq(d) < r * r` are **not the same predicate in float32**: measured,
10 rows of 200k disagree at the boundary. So a module that tests the same rule both ways can accept a
candidate in one kernel and reject it in another — `ball_pivoting` tested one clustering rule as
`wp.length(...) < min_cluster` in `seed_triangles` and as `wp.length_sq(...) < min_cluster_sq` forty
lines later. Pick one spelling per predicate and say which; expect the speed to be flat (`neighbors`
measured 0.997-1.003x, and a hash-grid cell probe is worth ~600 point tests) and keep the number
either way.

**A green suite does not prove a `@wp.func` extraction was behaviour-neutral.** `@wp.func` calls
inline at codegen, so an extraction that reorders an expression's evaluation changes `float32`
results without moving any comparison asserted at `1e-5`. The evidence is **reading the diff**, not
the suite. Two consequences for the gate: prefer to check the *decision* arrays a kernel writes over
the positions it produces where positions drift on their own (`isotropic_remesh` moves ~3e-06 run to
run from atomic ordering, so a byte comparison fails on an unchanged build), and where a reference's
answer is a combinatorial object — a triangulation, a face buffer — gate on that rather than on a
tolerance.

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
- For **2D** outputs, allocate with `twt.empty_2d((rows, cols), wp.int32, device=...)` — the dtype is an argument, not part of the name — instead of bare `wp.empty((rows, cols), ...)`. `twt.empty_3d` is the rank-3 counterpart.
- For **1D** outputs, keep `wp.empty(n, dtype=..., device=input.device)` when all elements will be written by the kernel (avoid unnecessary zero-initialization).
- Return rank-2 buffers with `return twt.as_array2d(arr, wp.int32)` so callers get a checked, correctly typed value. The `dtype` argument selects the overload that narrows to `twt.Array2dInt32` / `Array2dFloat32` / `Array2dFloat64`.
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

**And this is now a pattern rather than one API's quirk: `wp.Volume.get_voxel_count()` is a
capacity too.** It reports the grid's allocated voxel count, not its active one, so `triwarp.voxels`
goes through `Volume.get_active_stats().voxel_count` and says so at both sites. Same shape as
`nnz` — a field that reads like the answer, is an upper bound, and fails silently when it sizes a
buffer. When a Warp object offers a count, check whether it is the count or the capacity before
sizing anything with it.

**And this is now a pattern rather than one API's quirk: a Warp object's `*_count` / `.nnz` field is
a *capacity* until proven otherwise.** The second instance is `wp.volume_voxel_count`, which returns
the grid's allocated capacity and not its active voxel count — `triwarp/voxels.py` rejects it in
source at two sites and goes through `Volume.get_active_stats` instead. So before sizing a buffer,
a slice or a launch `dim` off any such field, probe it against a construction whose true count you
know (a duplicate-emitting triplet build, a sparse voxel set) and record the number where the
rejection lives. The failure mode is the same both times: the wrong reading is an *upper* bound, so
nothing raises and the tail is garbage.

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
  *caller*. Return `wp.mat33d` / `wp.vec3` (`measures.moments` returns the inertia tensor as
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
`triwarp/kernels/metrics.py` — the chamfer kernels are differentiated via `wp.Tape`.

---

## 5. Kernel-Scope Restrictions

The following Python features are **not supported** inside `@wp.kernel` and `@wp.func`:

- Lambda functions, list comprehensions, sets, dicts, `list.append()`, `eval()`, recursion, exceptions.
- Python tuples for initialization — use explicit typed constructors: `wp.vec3(1.0, 2.0, 3.0)`, not `(1.0, 2.0, 3.0)`.
- For small fixed-size collections use vector types (`wp.vec3`, etc.); for larger ones use `wp.zeros(shape=N, dtype=T)` (stack-allocated inside a kernel).
- The `%` operator follows C++11 semantics (sign of result = sign of dividend), not Python semantics.
- **`//` truncates toward zero like `/`, not toward −infinity like CPython's `//`, and on integers the two operators are the same operation.** Measured on Warp 1.16: `[-8, -7, -1, 0, 1, 7, 8] ÷ 3` gives `[-2, -2, 0, 0, 0, 2, 2]` for both spellings, where CPython's `//` gives `[-3, -3, -1, 0, 0, 2, 2]`. This is consistent with the `%` rule above (`-8 % 3 == -2`, and `-2 * 3 + (-2) == -8`). **Spell integer division `//`** — `/` on two `int32`s reads as real division and only truncates because the operands happen to be integers, so a reader has to recover the types before knowing what the line does. Every dividend in this package is a non-negative index, where the two conventions coincide; the hazard is *porting* a line with a negative dividend between host Python and kernel scope, which changes its answer silently.
- **And spell the remainder `%`, not `i - (i // stride) * stride`.** The long form is the same
  operation — every dividend here is a non-negative index — but it reads as though it is *avoiding*
  `%` for a reason a reader then goes looking for. `conjugate_gradient` wrote it long-hand where the
  sibling `multigrid` three files over wrote `t % stride`; there is no such reason.
- `wp.asin()` / `wp.acos()` auto-clamp inputs to [-1, 1]; explicit `wp.clamp` before these calls is redundant but harmless.
- Variable scope inside conditional blocks may differ from CPython: variables defined only inside an `if` branch are accessible afterward in Warp, but are uninitialized if the branch was not taken — always initialize variables before branching.
- **A tuple cannot be subscripted by a runtime index; a vector can.** The `tuple[T, T, T]` a
  `@wp.func` returns unpacks into names and nothing more, so `corner[k]` for a loop variable `k`
  does not compile — which is why a handful of kernels reach for the flat slice
  `faces[f * 3 : (f + 1) * 3]` that §3's index-form rule otherwise retired. The spelling that keeps
  both is `wp.vec3i(*corner_triple(faces, f))`, which accepts a runtime `[]` (verified on Warp
  1.16) and takes no sub-array view. So the rule is **not** "no slices" but *no slice where an index
  form exists* — `holes.directed_edge_opposites` reintroduced one and it was the legitimate case.

---

## 6. Testing Against Trimesh

All new geometry functions MUST have regression tests that compare against the `trimesh` CPU reference implementation (`trimesh.triangles`).

### Conventions

- Test file: `tests/test_<module>.py`; import pattern:
  ```python
  import trimesh.<module> as tm
  import triwarp.<module> as tw
  ```
- Use the `device` fixture from `tests/conftest.py`. Every test function must accept `device` as a
  parameter. **It is parametrized, and `--device={auto,cpu,cuda,both}` selects for *this
  process*** — `auto` picks one device (cuda if available), exactly like `benchmarks/conftest.py`,
  and every test id carries a `[cpu]` / `[cuda0]` suffix. `both` means "every device this process
  can see, skip nothing".
- **Both-device coverage is a two-process job: `uv run python -m tests.devices`.** It runs a CUDA
  pass, then a CPU pass with **`CUDA_VISIBLE_DEVICES=""`**, and that variable is the whole point:
  **Warp's CPU work is ~36x slower once CUDA has been initialised in the process.** Measured on one
  `heat_signed_distance` call, same mesh, same code, only the variable differing — 50.57 s against
  **1.40 s** — and unchanged by `launch_array_access_mode` (`RELAXED` 50.34 s, `CHECKED` 49.77 s),
  so it is CUDA *presence* and not §8's launch guard, which stays `STRICT` for free. Same tests at
  the pytest level: `tests/test_heat_signed.py` is 166.77 s with CUDA visible and **8.53 s**
  without. Whole suite: in-process `--device=both` measured **717 s** against ~37.6 s + ~155 s as
  two passes. So never reach for `--device=both` on a GPU box to get CPU coverage — use the runner.
- **Run the runner before calling a change done**, not just the default `pytest`. Both-device
  coverage is what caught the `warp.fem` ambient-device leak in
  `reconstruction._screened_poisson_adaptive` — broken for CPU input on any box with a GPU, and
  invisible to a CUDA-only run (the devices matched) *and* to a `CUDA_VISIBLE_DEVICES=""` run
  (`warp.fem` then defaults to CPU too). That defect class needs CUDA present *and* the arrays on
  the host, which is a configuration neither single-device run reaches. `wp.ScopedDevice(device)` is
  the fix when a dependency picks the device for us.
- **A test that costs more than ~15 s on CPU wears `@pytest.mark.slow_cpu(<measured seconds>)`**,
  which skips it when it would run on `cpu` unless `--device=both`. Four `screened_poisson` tests
  carry it, and they were 277.6 s of a 432.6 s CPU-only run — one ~90 s depth-6 solve each, under a
  second on CUDA. In a CUDA-hidden process `--device=both` therefore means "all of CPU, including
  these", which is how the runner's `--slow-cpu` asks for a full CPU pass. Use the marker only where
  the *device* is the cost and the claim is device-independent, and put the measured number in it so
  the next reader can re-derive the cut; a test slow on both devices belongs on a smaller input
  instead.
- Generate reproducible random data with `np.random.default_rng(seed)` (use a fixed integer seed per test).
- **Upload a NumPy mesh with `conversions.numpy_to_warp(vertices_np, faces_np, device)`**, never a
  local helper. This section used to *print the body* of one, and the result was six private copies
  across six modules at 54 call sites, differing only in where the `float32` cast sat — printing an
  implementation is an invitation to paste it. Its `wp.vec2` sibling is `numpy_to_warp_uv`, for the
  parametrization tests' 2-D vertex buffers, and the inverse is `warp_to_trimesh`. Triangle-soup
  arrays `(n, 3, 3)` become an indexed mesh first:
  ```python
  vertices_wp, faces_wp = numpy_to_warp(
      tri_np.reshape(-1, 3), np.arange(tri_np.shape[0] * 3, dtype=np.int32), device
  )
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
  *test* oracle (see `tests/test_metrics.py`, `tests/test_polyline.py`, `tests/test_seams.py`) but
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
`import pymeshlab as ml` plainly, never through `pytest.importorskip`. **That rule covers every
package in `[dependency-groups] test`, not only the six reference libraries** — `libigl`, `shapely`
and `moderngl` are in the same list and were guarded by eight `importorskip` calls, which is a
latent hole rather than a safety net: if the import breaks the tests *vanish* instead of failing, and
one of those eight sat under a `parity` marker that `tests/test_parity.py` would have kept passing
because the marker is static and the skip is not. Where a *runtime* precondition genuinely cannot be
declared as a dependency — an EGL driver for `moderngl`'s OpenGL reference — keep a skip, but make it
name the driver and put it in the fixture that needs the context, never at module scope over the
import. Reference variables take a
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
APIs are CPU-only; only `open3d.t` has GPU kernels. Nine hazards, all measured:

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
- **`remove_radius_outlier` is nondeterministic**, so it cannot be a class-A oracle. It shares one
  `KDTreeFlann` across an `#pragma omp parallel for` whose radius search is not thread-safe under
  that sharing: measured three distinct keep sets (43 / 44 / 45 points) over eight repetitions of one
  500-point cloud, differing by one or two points each time. Its published rule —
  `count > nb_points`, self counted — is sound, and evaluating it through the *same* tree one query
  at a time reproduces `points.radius_outlier_mask` exactly on that cloud. So the comparison goes
  through `search_radius_vector_3d` in a loop and the filter keeps only the benchmark row. No other
  `remove_*` method shares the defect (`remove_statistical_outlier` reduces per point independently
  and is stable), which is why this one had to be found rather than assumed.
- **A down-sampler's output order is its own, not its algorithm's.** Every legacy selection routes
  through `SelectByIndex`, which walks a *mask* over the input and therefore emits the survivors in
  ascending index order — so `farthest_point_down_sample`'s greedy sequence is destroyed on the way
  out and only the selected *set* can be compared (measured equal at counts 4 / 32 / 64). Where the
  order is the claim, transcribe the C++ loop into the test: `FarthestPointDownSample` takes its
  arg-max with a strict `>`, so the lowest index wins a tie. Two small conventions from the same
  family: `num_samples=0` returns an empty cloud rather than raising, and
  `compute_nearest_neighbor_distance` reports **`0.0`** for a cloud of fewer than two points where
  the honest answer is `inf`.

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
subtree to avoid as there is in `reference/libigl`. Fifteen hazards, all measured:

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
- **`multi_ray_trace` is trimesh + embree, not VTK.** It imports `trimesh`, checks
  `trimesh.ray.has_embree` and calls `tmesh.ray.intersects_location` — measured identical to
  trimesh's own call (first-hit faces 1.0000 over 2 000 rays, 7.3 against 8.9 ms), so a `pyvista`
  row for the `intersects_*` groups would be a **trimesh row under another name**, the
  one-VTK-two-wrappers double count in a new pair. VTK's own `ray_trace` *is* independent (face
  agreement 1.0000, hit point 2.80e-07 against triwarp) but takes one ray per call at
  **398.6 µs/ray** — 89x embree — so it is a test oracle and never a benchmark row.
- **`validate_mesh()`'s cell fields are per *cell*, not per mesh.** `intersecting_faces` is "two
  faces of a **3D cell**", so on a triangle mesh it is identically empty: 0 on two interpenetrating
  icospheres where `face_self_intersecting_mask` flags **92**, and 0 on `bohemian_dome` where it
  flags 205. `inverted_faces` likewise reads 0 on a mesh with ten reversed faces. The degeneracy
  field that does fire is **`zero_size`** (`degenerate_faces` stays empty even for a repeated vertex
  id), and `clean()` **keeps** those faces — 82 of 82 cells at the default, at `tolerance=0.0` and at
  `absolute=False` — so there is a detector here and no filter. A degeneracy comparison also needs a
  *scale-aware* input: a float64-exactly-collinear face (area 1.25e-17) survives triwarp's float32
  altitude test on both devices, so pyvista flags two where `remove_degenerate_faces` drops one.
  Relatedly, `collision` is a **two-mesh** filter and cannot see a self-intersection either: it
  reports 2 600 hits for a 320-cell mesh against its own copy (56 for two genuinely offset spheres).
- **`compute_implicit_distance` needs polygons.** On a line-set `PolyData` VTK logs
  *"No polygons to evaluate function!"* once per query and returns a field **3.35** off the truth
  rather than raising. Polyline distance goes through `find_closest_cell` on a **single-cell**
  polyline instead (2.49e-07 against `polyline.distance_to_polyline`) — and that single cell is the
  whole trick: `pv.lines_from_points` makes one cell per segment, which is what makes
  `compute_arc_length` read 0.0638 for a polyline of length 12.7049. Its locator collapses on a long
  cell, though: 24.8 ms at 4 096 queries against 268 segments, **104 s** at 65 536 queries against
  65 536 segments.
- **`find_containing_cell` is the point-location oracle that works**, batched, `-1` outside, and
  measured **1.0000** against `scipy.spatial.Delaunay.find_simplex` on 10 000 queries with the batch
  and the per-point loop byte-identical. Worth stating because the libigl block above records
  `igl.in_element` as unusable for the identical question, so generalizing from it skips a good
  reference. `find_closest_cell` is likewise the most accurate closest-point reference registered
  (4.4e-16 against `igl.point_mesh_squared_distance` on distance *and* point) — but its **cell id is
  not comparable**: it disagrees with igl on 28 % of exterior queries, every one a point lying on a
  shared edge to ~1e-16, because for a query far outside a convex mesh the nearest point is a vertex
  (the vertex normal fans exhaust 4π). Compare the distance, and the point at Warp's own 2.2e-04
  `mesh_query_point_no_sign` floor.
- **Two filters answer a *different* question than their name suggests.** `sample()` interpolates
  only where the query lands **inside** a source cell — 476 of 2 562 target points valid against
  `interpolation.transfer_onto_vertices`, agreeing to 7.20e-08 on those — and
  `snap_to_closest_point=True` snaps to the nearest source **vertex**, not the nearest point on the
  surface, which makes it *worse* (0.256). And `delaunay_2d(edge_source=loop)` does not clip to the
  loop: on a 40-point star it returns 63 cells covering area **4.465** against the polygon's 3.264.
  The polygon-fill oracle is `triangulate_contours`, which adds zero Steiner points and matches
  `polyline.triangulate_polyline`'s `n - 2` count and area to nine digits.

Two more, for the parametric surfaces specifically: **they arrive open and `clean` defaults
differently per surface** — `surface_from_para(clean=False)` is the underlying default and at that
setting every one of the 21 has 156 or 236 boundary edges (a raw `ParametricMobius()` is a disk),
while pyvista overrides it to `clean=True` on 9 of them, so `klein` arrives welded and `mobius` does
not. **Always pass `clean=True` explicitly.** And **`klein` is not a Klein bottle** as VTK
parameterizes it: it welds to two boundary loops and reads *orientable*, so only `figure8_klein` is
the closed non-orientable χ = 0 surface. A comparison reaching for `klein` expecting
non-orientability is testing nothing.

**meshlib** (pybind11 over MeshLib's C++ core, mirrored under `reference/MeshLib`) is a hard test
dependency like the four above — `from meshlib import mrmeshpy as mm` / `mrmeshnumpy as mn`, both
aliases pinned in ruff's import conventions, never through `pytest.importorskip`; reference
variables take an **`_ml`** suffix. Build meshes with `tests.conversions.trimesh_to_meshlib` /
`numpy_to_meshlib` / `warp_to_meshlib`, clouds with `points_to_meshlib`, and read a result back with
`meshlib_to_trimesh` — never hand-roll `mn.meshFromFacesVerts`, whose argument order is the trap
below. It earns its seat on three counts the other five cannot cover: it is the **only
multi-threaded** CPU reference (143 OS threads measured live during `findSelfCollidingTrianglesBS` +
`computePerVertNormals`), so a `triwarp-cuda` vs `meshlib` ratio is a fair fight where the others
are not — and the other edge of that knife is that a `triwarp-cpu` row loses to it on any parallel
op regardless of algorithm, which is §13's "decide on the CUDA number" again. It is the only
reference that binds a real minimum-weight Liepa/Klincsek `fillHole` with a 12-metric family and a
two-loop `stitchHoles`, where trimesh fans and pymeshlab ear-clips. And it is the only oracle for
`repair.collapse_small_triangles`, which libigl does not bind despite the C++ header existing.
`meshlib.mrcudapy` exists and is deliberately **not** used: the plain `mrmeshpy` free functions are
the reference, and a CUDA module would make the row incomparable with the other five. Nine hazards,
all measured:

- **Almost every free function mutates its `Mesh` in place and returns something else.** `relax`,
  `fillHole`, `fillHoles`, `decimateMesh`, `remesh`, `subdivideMesh`, `fixMeshDegeneracies`,
  `filterCreaseEdges`, `denoiseNormals`, `smoothRegionBoundary`, `expand` and `shrink` return a
  status `bool`, a count, an `EdgeId` or a `FaceBitSet` of *new* faces — never the mesh. So **one
  `mm.Mesh` serves one mutating call**: build a fresh one per comparison and per benchmark round,
  the `new_meshset_pml` rule rather than the `mesh_o3d` one (`BenchCase.new_mesh_ml()` is a method,
  not a cached property, for exactly this). The exceptions are worth knowing per function rather
  than assumed: `marchingCubes` reads its `SimpleVolume` and returns a fresh `Mesh`, verified by two
  calls on one volume returning the identical face count with `dims` intact, so *its* input is
  cacheable.
- **`meshFromFacesVerts` takes faces *first*, and it sizes the vertex buffer by `F.max() + 1`.** The
  argument order is the reverse of every other converter in `tests/conversions.py`, and a swapped
  call raises nothing — the two arrays differ in shape only when the counts differ. That is why the
  converters exist and why they all take vertices first. The sizing is igl's F-only hazard in a new
  place and it behaves *differently* by position: on `icosphere(2)` a **trailing** unreferenced
  vertex is dropped outright (163 in → `points.size()` 162, `getNumpyVerts` 162 rows) while an
  **interior** one is kept in the buffer and excluded from `numValidVerts` (163 in → 163 back,
  `numValidVerts` 162, value verbatim). So on `bunny`, whose 1 113 unreferenced vertices are
  interior, the buffers line up and the *validity* mask does not; where the spares are trailing, the
  indices shift. Never hand MeshLib a compacted `V` with the original `F`, and never assume
  `getNumpyVerts(...).shape[0] == len(V)`.
- **`pack()` is mandatory before reading topology back, and skipping it is silent.** Measured after
  `decimateMesh(maxDeletedFaces=200)` on a 320-face mesh: `numValidFaces` is 120,
  `topology.faceSize()` is 320, and **`getNumpyFaces` returns 319 rows** — `last_valid_face_id + 1`
  — of which **199 are `[0, 0, 0]`**, degenerate triangles on vertex 0. `getNumpyVerts` still
  returns all 162 positions while `numValidVerts` is 62. No exception, no warning; after `pack()`
  the same reads give 120 and 62. `meshlib_to_trimesh` packs by default for this reason.
- **`np.asarray` on a scalar container silently returns a 0-d `object` array.** `VertScalars`,
  `FaceScalars` and `UndirectedEdgeScalars` are not accepted by `mn.toNumpyArray` (which binds only
  `VertCoords` / `FaceNormals` / `std_vector_Vector3_float` and raises a clear `TypeError`
  otherwise — that part is safe), and `np.asarray(vert_scalars)` produces `dtype=object, shape=()`
  rather than raising, so the failure surfaces several lines later in whatever NumPy call comes
  next. Go through `conversions.meshlib_scalars_to_numpy` — and for a container of **ids** rather
  than numbers (`Buffer_VertId` from `findNClosestPointsPerPoint`, `VertMap` from
  `findSmallestCloseVertices`) through `conversions.meshlib_indices_to_numpy`, which exists because
  the scalar reader *raises* on them: a `VertId` implements `__index__` but not `__float__`, so
  `np.fromiter(..., np.float64)` fails with `float() argument must be a string or a real number`.
- **A returned bitset is only as long as its highest set bit, and which functions do that is not
  guessable.** `mn.getNumpyBitSet` reads the bitset at *its own* length: one MeshLib sized against
  the mesh comes back domain-sized (`getBoundaryVerts` gives 162 entries on a 162-vertex mesh with
  one bit set), but one built by insertion does not — `findSelfCollidingTrianglesBS` returns **608**
  entries on a 640-face pair of overlapping spheres (last colliding face 607) and an **empty** array
  on a clean mesh. Neither raises, and both break an `np.array_equal` against a triwarp mask by
  *shape* rather than by value, which reads as a converter bug. Go through
  `conversions.meshlib_bitset_to_numpy(bitset, size)`, which states the domain and pads.
- **Every bitset converts in bulk, in both directions — a per-bit Python loop is never the answer.**
  A `TypedBitSet` (`FaceBitSet`, `VertBitSet`, `VoxelBitSet`) derives from `MR::BitSet`, and
  `mn.getNumpyBitSet` is declared over the base, so pybind11 upcasts any of them into the readback
  above. The *load* direction is `BitSet.fromBlocks`, which takes the packed `uint64` blocks — so
  `np.packbits(flags, bitorder="little")` fills a whole set in one call, and
  `conversions.numpy_to_meshlib_bitset` is that (wrap the result in the typed set:
  `mm.VoxelBitSet(bitset)`). Measured against the per-cell `set()` loop it replaced: **258x** at
  110 592 voxels (0.33 ms against 84.9 ms), and the readback **1 199x** (0.14 ms against 168 ms).
  Three details: `bitorder="little"` is not NumPy's default and is not optional; `fromBlocks`
  rejects a NumPy `uint64` array with `TypeError` (its argument is `std_vector_unsigned_long`, so
  pass `.tolist()`) and rounds the size up to whole 64-bit blocks, so `resize` back to the domain;
  and the element order is the container's, which for a `VoxelBitSet` addressed by a
  `VolumeIndexer` is `x` fastest, i.e. `dense.ravel(order="F")`. Iteration also works —
  `list(bitset)` yields one `VoxelId` per *set* bit, despite `MR_BIND_IGNORE_PY` on the C++
  `begin`/`end` — but it is a convenience, not the fast path. **A "MeshLib binds no converter for
  this" claim is a reason to probe the base class, not to write the loop**: one such claim in
  `tests/test_voxels.py` cost a benchmark row that then read as unbenchmarkable.
- **`(*args, **kwargs)` in a signature is an overload set, and `inspect` / `help()` cannot see it.**
  This wheel's pybind11 docstrings are stripped: `inspect.signature` gives `(*args, **kwargs)` and
  `__doc__` carries no overload lines. **Read the real signatures by calling the function with one
  junk argument and reading the `TypeError`**, which pybind11 renders as a numbered list of every
  overload — measured 4 for `expand`, 3 for `relax` and `getAllComponents`, 2 for `shrink` and
  `stitchHoles`.
- **Two overloads of one name can have opposite output conventions, and the wrong one binds
  silently.** `expand(topology, region: FaceBitSet, hops)` returns `None` and **mutates `region`**
  (measured: 1 face → 12), while `expand(topology, f: FaceId, hops)` **returns** a new `FaceBitSet`
  (also 12); same for `shrink`. `stitchHoles(mesh, a, b, params)` takes two named hole edges and
  `stitchHoles(mesh, params)` finds them itself — argument *count* is the only tell. `relax`'s first
  overload takes a `PointCloud` and the second a `Mesh`, with different params types
  (`PointCloudRelaxParams` vs `MeshRelaxParams`). One `getAllComponents` form returns a
  `(components, count)` **tuple** rather than the vector. Resolve the overload explicitly and assert
  the result's type or count before comparing, so a future rebinding cannot quietly pick the other.
- **`findOutliers`' default mask segfaults on a cloud with no normals.**
  `FindOutliersParams.mask` defaults to `OutlierTypeMask.All`, which includes `AwayNormal`, and that
  criterion dereferences the cloud's normals — measured as a `SIGSEGV` with no exception on a
  415-point cloud built by `pointCloudFromPoints`. The other three modes (`SmallComponents`,
  `WeaklyConnected`, `FarSurface`) run fine without normals. Set the mode explicitly, or supply
  normals; and note the same rule as the other normal-consuming functions listed under
  `points_to_meshlib`.
- **A projector stores a raw pointer to the mesh or cloud it was given, so a temporary segfaults.**
  `PointsToMeshProjector.updateMeshData(build_a_mesh())` and
  `PointsProjector.setPointCloud(build_a_cloud())` both return normally and then read freed memory
  in `findProjections` — measured as a hard `SIGSEGV` (no exception, no traceback) at 10 000 queries
  against `bunny_decimated` and at 40 queries against a 300-point cloud. Bind the mesh or cloud to a
  name that outlives every query, which is Open3D's `from_legacy` hazard in a second library.
  **`mm.MeshPart` does *not* share the rule — it keeps a real Python reference, and
  `mm.MeshPart(trimesh_to_meshlib(mesh_tm))` over a temporary is safe.** Measured:
  `sys.getrefcount(mesh_ml)` goes 2 → 3 across the constructor and `part.mesh is mesh_ml`, where the
  two setters above leave it at 2. That is the whole test, and it is the one to run on the next
  binding of this shape rather than reasoning from a sibling: **probe the refcount, do not infer the
  lifetime.** (An earlier version of this block said `MeshPart` had the same rule and could not even
  hold the reference in an attribute; the attribute part is true — the pybind11 object has no
  `__dict__` — but it is irrelevant, because nothing needs to hold it. The wrong half made 28 correct
  call sites across 13 test files read as latent use-after-frees.)
  `findProjections`'s `upDistLimitSq` is a second crash of the same shape: pass MeshLib's own
  `FLT_MAX`, since `math.inf` segfaults rather than raising.
- **The AABB tree is lazily built and cached on the `Mesh`, and the ratio depends on whether the
  query is the process's first.** On `icosphere(4)`, first-vs-second `findProjection` measures
  **68x** on the process's first mesh (2.03 ms → 0.030 ms) and settles at **17-20x** on later fresh
  meshes (0.28-0.30 ms → 0.014-0.017 ms), the difference being the thread pool spinning up inside
  the first build. So every query row and every timed callable must state whether the build is
  inside it — recommended: build outside and pre-warm with one throwaway query, so the row times the
  *query*, matching what triwarp's `wp.Mesh`-in-hand rows already do with their BVH. This is also a
  *correctness* trap next to the mutation hazard: a mutating call invalidates the tree, so a test
  that queries, mutates and queries again is not measuring what it looks like. **A `PointCloud`
  caches its point tree the same way**, which silently changes what a *self*-query row measures:
  `findNClosestPointsPerPoint` on a 160 000-point cloud runs 10.2 ms with the tree rebuilt and
  3.8 ms reusing it, against triwarp and open3d, both of which build their structure inside every
  call. `cloud.invalidateCaches()` in the benchmark's `setup` is the lever — dropping and rebuilding
  the *cloud* instead adds the point allocation and with it an 11-54 ms spread.
- **Per-vertex free functions are per-*vertex*, and `mrmeshnumpy` has the batched form.**
  `discreteGaussianCurvature(topology, points, v)` and `sumAngles(...)` take one `VertId` per call; a
  Python loop over 642 vertices measures **2.79 ms** against **0.046 ms** for
  `mn.getNumpyGaussianCurvature(mesh)` — **49-67x**, bit-identical results, and the gap grows with
  the mesh. Prefer `mn.getNumpyGaussianCurvature` / `getNumpyMeanCurvature` / `getNumpyCurvature`,
  and never put a per-vertex Python loop in a benchmark row: it would time the loop, not MeshLib.
- **`getNumpyVerts` is float64 but the storage is float32.** Round-trip error on `icosphere(2)` is
  **2.58e-08**, so a MeshLib comparison bottoms out around 1e-7 for the same reason a triwarp one
  does. `computePerFaceNormals` is **normalized** (measured |n| = 1.0 ± 1e-7) — note the contrast
  with pymeshlab's `face_normal_matrix()`, which is the unnormalised cross product at magnitude
  `2 * area`.
- **Four parameter conventions that read as a disagreement, and one function whose name lies.**
  `sampleHalfSphere()` is **not** a half sphere: measured, its 145 directions span `z` from -1 to +1
  and only 72 have `z > 0`, so feeding it to `computeSkyViewFactor` as a sky dome halves the answer
  (0.52 where the open sky reads 0.98). `InSphereSearchSettings.maxRadius` defaults to **1**
  whatever the mesh's scale, silently capping every thickness on anything larger — pass half the
  smallest bounding-box side, the article's own recommendation. `makeUVSphere`'s
  `verticalResolution` counts interior latitude **rings**, not profile points, so it pairs with
  `creation.uv_sphere(count=(v + 2, h // 2))` — and at *that* mapping the two are the same mesh
  vertex for vertex (bijection at 4.7e-07), where the same nominal resolution differs by 74 % in the
  vertex count. And `leftCotan(e)` is the **plain** cotangent keyed by the directed edge whose left
  face owns it, against `laplacian.cotmatrix_entries`' *half* cotangent keyed by `(face, corner)`;
  `cotan(ue)` is the two summed, i.e. the assembled off-diagonal rather than the table. Two more
  found the same way: `MarchingCubesParams.origin` addresses the voxel **centre**, so a lattice
  whose sample `[0, 0, 0]` sits at `lower` is marched with `origin = lower - voxel / 2` and the
  un-shifted call is a rigid half-diagonal off (0.0369 against 1.2e-07, measured); and
  `findNClosestPointsPerPoint` returns a **heap, not a sorted list** — the ids are exactly scipy's
  `k` nearest (set equality 1.0000 at `k=3`) but only 90.8 % of rows are in distance order and the
  *nearest* is the **last** entry, which at `numNei=2` holds in 100 % of rows only because a
  two-element heap is ordered by construction. Ask for `numNei=1` when one neighbour is the
  question.
- **`computeRayThicknessAtVertices` takes the direction from the *pseudonormal*.** So it pairs with
  `visibility.thickness(method="ray", normals=angle_weighted_vertex_normals(...))` to **5.96e-07**
  and with the area-weighted normals to **0.031** — five orders worse, and the kind of gap that
  reads as an algorithm bug. Same family as the two normal pairings below. Both thickness functions
  and `computeInSphereThicknessAtVertices` also take **no query set** (they answer at every vertex),
  which is what keeps them out of a benchmark group whose input is a subsample.

Two of its pairings are worth knowing before writing a comparison, because neither is guessable from
the names: `computePerVertNormals` matches `vertices.area_weighted_vertex_normals` to **1.19e-07**
while `computePerVertPseudoNormals` matches `angle_weighted_vertex_normals` to **1.19e-07**, and
each sits **6.8e-03** from the other's partner — so MeshLib pins a weighting convention no other
reference distinguishes. And `mn.getNumpyGaussianCurvature` is the pointwise **angle defect**, which
pairs with `vertices.vertex_defects` (9.5e-07 abs / 5.2e-05 rel on `icosphere(3)`) and **not** with
`curvature.discrete_gaussian_curvature`, the Cohen-Steiner/Morvan *ball* measure — that pairing
lands in the same D2 exemption pymeshlab already holds on that group.

**Licensing: MeshLib is the one reference here that is not open source.** The wheel and
`reference/MeshLib` are under AMV Consulting's *"NON-COMMERCIAL & education"* agreement — a
terminable, non-transferable licence for "non-commercial, evaluation or educational purposes", with
a separate commercial licence required otherwise, and an explicit bar on modifying or transferring
the Software. That is a stronger constraint than libigl's `copyleft/` subtree or pymeshlab's GPL,
because it restricts *use* rather than distribution, and triwarp itself ships `MIT OR Apache-2.0`.
Two rules follow, and `triwarp/` currently satisfies both — **keep it that way**, because both are
one careless docstring away from being false again.

**Nothing under `triwarp/` may name MeshLib at all.** Not the library, not a function
(`fillHole`, `positionVertsSmoothly`, `triangleAspectRatio`), not a source file (`MRMeshDelone.cpp`,
`MRTriMath.h`, `MRMeshMetrics.cpp`). 89 such references were removed in one pass across 19 files;
they had accumulated as ordinary attribution and collectively read as a claim that a package shipped
under `MIT OR Apache-2.0` is derived from a proprietary one. Describe what the code **computes**, or
name the algorithm in the literature's vocabulary — "the Liepa/Klincsek interval DP", "the Delone
empty-circumcircle test", "circum-radius over twice the in-radius" — which is what a reader needed
anyway and what §10 asks for independently. Read `reference/MeshLib` to understand an operation's
*interface* and its parameters; never to port its body.

**Keep it a test/benchmark dependency.** It belongs in `tests/` and `benchmarks/`, where naming it
is correct and required — a comparison has to say what it compares against. Nothing in the shipped
package imports it or mentions it; `grep -rni 'meshlib\|MRTriMath\|MRReducePath\|MRMesh' triwarp/`
must stay empty.

One more, for the fixtures rather than the API: **`trimesh.slice_plane`'s output is a poor MeshLib
input.** A hemisphere built that way from `icosphere(2)` reports **17** `findHoleRepresentiveEdges`
where the surface has one rim; a properly built open mesh reports the correct count. Use the
`tests/conftest.py` fixtures (`hemisphere`, `half_torus`), and **assert the hole count** before
comparing a per-hole answer.

**pymeshfix** (nanobind over Marco Attene's MeshFix / the TMesh kernel, mirrored under
`reference/pymeshfix`) is a hard test dependency like the five above — `import pymeshfix` plainly and
reach the low-level class as `from pymeshfix import _meshfix`, never through
`pytest.importorskip`; reference variables take a **`_pmf`** suffix. Build every `PyTMesh` through
`tests.conversions.numpy_to_pymeshfix` / `trimesh_to_pymeshfix` / `warp_to_pymeshfix` and read one
back with `pymeshfix_to_numpy`, `pymeshfix_intersecting_faces` or `pymeshfix_face_remap` — the
converters exist because the raw calls are unsafe in three separate ways listed below. It is the
**narrowest deep** reference here: `dir(PyTMesh)` is 19 members, of which **nine are algorithms**
(`fill_small_boundaries`, `select_intersecting_triangles`, `strong_degeneracy_removal`,
`strong_intersection_removal`, `clean`, `remove_smallest_components`, `join_closest_components`,
`fix_connectivity`, plus the module-level `clean_from_arrays`) against MeshLib's 246 — and all nine
are *repair*, which is where triwarp has 20 public functions and, before this, only trimesh's
`repair` module and MeshLib as oracles. It performs one operation no other reference here does end to
end: arrays of a broken digitised surface in, a single watertight solid out. Do **not** plan a
comparison for anything else: there is no curvature, geodesic, parametrization, registration,
reconstruction, decimation, remeshing, point-cloud, boolean, proximity or signed-distance entry
point, and the C++ names for several of those (`cutAndStitch`, `iterativeEdgeSwaps`,
`loopSubdivision`, `isInnerPoint`, `openToDisk`, `marchIntersections.cpp`) are in the headers and
**unbound** — the libigl lesson, one notch worse, since here only nine algorithms of a ~120-method
class are reachable. Thirteen hazards, all measured:

- **`load_array` is not a load; it is already a repair, and it renumbers.** It runs the kernel's
  connectivity fix and Euler update before returning. Measured on `icosphere(1)` (42 v / 80 f): a
  trailing *or* interior unreferenced vertex is dropped (43 → 42, surviving positions and their
  relative order intact); an exactly duplicated face is **kept** and the non-manifold edges it
  creates are cut instead (80 → **81 f**, 42 → **45 v**), while a *reversed* duplicate is refused
  and the vertices are still cut; two coincident *referenced* vertices are **not** merged; one
  backwards face is rewound and *every* face backwards is left alone (volume −3.6587 in and out),
  because consistent is not outward. On the scan meshes `bunny_decimated` loads as
  **8 372 v / 16 220 f** from 8 171 / 16 301 and `bunny` as **34 834 v / 69 451 f** (its 1 113
  unreferenced vertices); `dragon` is unchanged. On a non-orientable closed surface it cuts the
  orientation-reversing seam — `boy` 1 483 → **1 559 v** at an unchanged 2 964 faces — leaving two
  coincident sheets where the surface had one, which is why `select_intersecting_triangles` reads
  436 there against triwarp's 177 and the number is not a disagreement. So **every comparison must
  be index-free** (positions, canonically sorted rows, sets, counts); where a face index is
  unavoidable, go through `pymeshfix_face_remap`, which checks the load first and refuses rather
  than guessing. The flip side is that `load_array` is itself an oracle — for
  `remove_unreferenced_vertices`, for `make_winding_consistent`, and partly for
  `split_non_manifold_vertices`.
- **One `PyTMesh` serves one load and one mutating call.** A second `load_array` raises
  `RuntimeError: Cannot load arrays after arrays have already been loaded`, and every algorithm
  mutates in place and returns a status, a count or an array — never the mesh. So build a fresh
  object per comparison and per benchmark round: the `new_mesh_ml` shape, and here there is no
  cached alternative at all.
- **`select_intersecting_triangles` returns a mostly-uninitialised array.** It allocates `(n, 3)`
  `int32` and writes the `n` face indices into the **flat** prefix, leaving `2n` entries of heap
  garbage. Measured on two `icosphere(2)`s translated 1.2 apart: shape `(72, 3)`, a flat prefix of
  72 ascending indices all below 640, and `arr.max()` reading **30 751** — a value that varies
  between processes. `out.ravel()[: out.shape[0]]` is the only defined read; go through
  `pymeshfix_intersecting_faces`. The prefix *is* deterministic (identical across repeat calls and
  across a fresh object), so a naive `np.array_equal(out1, out2)` reports nondeterminism that is not
  there.
- **`tris_per_cell` and `justproper` are no-ops on ordinary input.** Measured on that same pair,
  `tris_per_cell` ∈ {10, 50, 200} crossed with `justproper` ∈ {False, True} all return **72**.
  `tris_per_cell` tunes the broad phase and should not change the answer; `justproper` should, and
  does not on any fixture probed. Pass both explicitly at those values so a wheel that starts
  honouring either one fails a test rather than drifting, and do **not** build a triwarp flag around
  `justproper` until a fixture is found where its two settings differ.
- **`nbe` is inclusive, both its docstrings say otherwise, and pymeshlab's counterpart is
  exclusive.** `fill_small_boundaries(nbe, …)` fills loops of **at most** `nbe` boundary edges where
  the C++ comment and the Python docstring both say "less than". Measured on a 24-edge rim: `nbe`
  23 → **0** patched, 24 → **1**, 25 → 1; `nbe = 0` means all. This is also the one place two
  references disagree about the *same* parameter — on a 16-edge rim pymeshfix fills at `nbe = 16`
  and pymeshlab only at `maxholesize = 17` — so `holes.fill_small(max_edges=...)` follows pymeshfix
  (the precedence rule below) and a pymeshlab comparison passes `max_edges + 1` as its named
  class-B transform.
- **The "MeshFix could not fix everything" line on stderr is printed when it *succeeded*.** The
  wrapper does `if (result) cerr << …` where `result` is *true only if the mesh was completely
  cleaned*. Measured: `clean_from_arrays` printed it for both `bunny_decimated` and `bunny` and both
  outputs are watertight with χ = 2, while `clean()` on a clean `icosphere(3)` returns **True**. The
  message is inverted, `set_quiet` does not suppress it, and **nothing about it may be used as a
  signal** — read the boolean, or read the mesh. Every builder sets `set_quiet(True)` so a benchmark
  round does not print it once per repetition.
- **`remove_smallest_components` ranks by face count, not area or diameter, and returns the number
  removed.** Measured on three disjoint spheres — 80 f, 320 f, and 80 f at radius 10, much the
  largest by area and diameter — it removed **2** and kept the **320-face** one (extent
  `[2, 2, 2]`). It always reduces to exactly one component. That is the rule
  `repair.remove_small_components(keep_largest=True)` defaults to.
- **The output face buffer is a reordering, even when nothing was repaired.** `icosphere(2)` round
  trips with **byte-identical float64 vertices** (max abs difference `0.0`) and an identical
  triangle *set* under `np.sort(rows, axis=1)` plus a lexsort, but the rows come back in a different
  order and each starts at a different corner. Never compare face buffers positionally.
- **`n_boundaries` is a property in 0.18.1, and `boundaries()` raises.** The older Cython wheel
  exposed `boundaries()`; the nanobind one keeps the name bound only to raise `"boundaries() is
  deprecated. Use n_boundaries instead."`, and `n_points` / `n_faces` became properties in the same
  change. Code written against an example older than 0.17 fails with
  `TypeError: 'int' object is not callable`.
- **`strong_degeneracy_removal` measures degeneracy in `double`, so it is *stricter* than triwarp's
  `float32` test rather than merely different.** Measured on a flat 12-column strip: exactly
  collinear vertices are removed by both (24 v / 22 f → 0 / 0), and the same strip offset by `1e-9`
  is removed by triwarp and **kept unchanged** by pymeshfix, because `1e-9` is not zero in `double`.
  Where the degeneracy is exact the two agree completely -- a sphere with zero-area faces appended
  comes back at 162 v / 320 f, watertight, χ = 2, volume 4.0470 from both -- so compare on an
  *exactly* degenerate fixture and pin the near-degenerate class as the divergence.
- **`strong_intersection_removal` is a different algorithm from
  `repair.fix_self_intersections(method="local")`, not a different tuning**, and no transform
  rescues the pair: on the 16x16 self-intersecting torus triwarp cuts and refills each sheet and
  ends with **two** closed components (χ = 4, volume −10.42, 528 v) where pymeshfix removes far more
  and ends with **one** (χ = 2, volume −6.53, 80 v), two-sided surface distance 0.50; on
  `bohemian_dome` the same shapes at 2.23. All they share is the post-condition, so that pair is
  neither benchmarked nor a parity claim. The comparable level is the whole pipeline:
  `repair.make_solid` against `clean_from_arrays` agrees to **3.11e-08** on interpenetrating shells
  and returns **8 188 v / 16 372 f from both sides** on `bunny_decimated`.
- **Reproducing `clean_from_arrays` needs the loader's repair as an explicit first stage.** It is
  invisible in the C++ pipeline because `load_array` does it, and its absence is invisible in the
  output too until you check the right predicate: without `remove_unreferenced_vertices` +
  `make_winding_consistent` + `split_non_manifold_vertices` first, `bunny_decimated` comes back with
  χ = 2 and one component and is **not watertight**, because nothing downstream looks at edge
  manifoldness. Two more orderings that were measured rather than reasoned: the component filter has
  to run *inside* the intersection loop as well as before it (cutting a band out can disconnect the
  surface -- the torus above goes from one component to two), and **nothing geometric may run after
  the final fill** (filling a 3-vertex rim makes one sliver, a degeneracy pass deletes it and
  reopens the rim, and the two trade the same 122 faces for ever: χ = 2 before, χ = −56 after, and
  stable there).
- **`trimesh.slice_plane`'s output is a poor input**, the same hazard the MeshLib block records and
  worse here: a hemisphere sliced from `icosphere(2)` without `merge_vertices()` loads as
  121 → **137 v** and reports **17** boundary loops where the surface has one; after
  `merge_vertices()` it loads unchanged at 97 v and reports **1**. Use the `tests/conftest.py`
  fixtures, and **assert `n_boundaries`** before comparing a per-hole answer.

**It is single-threaded**, which makes its ratios easy to read: measured on `bunny`,
`select_intersecting_triangles` runs 435 ms of wall clock against 435 ms of `process_time`, a ratio
of **1.00**. So it belongs with trimesh / igl / pymeshlab / pyvista rather than with `meshlib`.

**Benchmark rule: on most rows the load *is* the row.** Because a `PyTMesh` takes exactly one load,
the build has to sit inside the timed callable (`BenchCase.new_tmesh_pmf()`), so every pymeshfix row
prices the load — measured 67.9 ms on `bunny_decimated` and 439.6 ms on `bunny`, against 64.2 /
435.3 ms for `select_intersecting_triangles`, 5.8 / 51.4 ms for `fill_small_boundaries` and
6.7 / 60.5 ms for `remove_smallest_components`. **Create a `pymeshfix` row only where the operation
is at least ~30 % of the round, and state the measured share in the group docstring.** By that rule
the intersection family (49–50 %) and `clean_from_arrays` (68–73 %) are timed, and the hole-fill
(8–10 %) and component-removal (9–12 %) comparisons carry
`pytest.mark.parity(<group>, "pymeshfix", benchmarked=False, reason=…)` with the ratio in the reason.
Cap it at `bunny`; `dragon` is seconds of load plus seconds of query per round.

**Licensing: pymeshfix is GPL-3.0**, and the TMesh headers under `reference/pymeshfix/src/` carry
Attene's dual licence — GPLv3 *or* a commercial agreement with IMATI-GE/CNR. triwarp ships
`MIT OR Apache-2.0`. This is a **different** constraint from MeshLib's and conflating the two
over- or under-restricts:

| | MeshLib | pymeshfix |
|---|---|---|
| May `triwarp/` name it? | **No** | **Yes**, in `Notes` / `See Also` — pymeshlab is GPL and is named throughout |
| May `triwarp/` be derived from its source? | No | **No** |
| May `tests/` and `benchmarks/` import it? | Yes | Yes |
| Why | proprietary, restricts *use* | copyleft, restricts *distribution of derivatives* |

So: read `reference/pymeshfix/src/` for the interface and the parameters, never for the body; cite
the **paper** rather than the file when porting — Attene, *"A lightweight approach to repairing
digitized polygon meshes"* (The Visual Computer 26, 2010) for the repair pipeline and the
component-joining rule, Liepa, *"Filling holes in meshes"* (SGP 2003) §3 for the density refinement,
Barequet & Sharir (1995) for the loop pairing — and keep
`grep -rnE 'MeshFix|Basic_TMesh|TMesh|_meshfix' triwarp/` empty. Prose mentions of `pymeshfix`
itself are allowed there; C++ symbols are not.

**And the precedence rule, which is the part a future author most needs and would never guess:**

> **Where pymeshfix and MeshLib both answer a question and their answers differ, triwarp's default
> is pymeshfix's answer and MeshLib's is reachable by a flag** — not the reverse. Where only MeshLib
> answers it, nothing changes.

The reason is not preference. A function whose *behaviour* was pinned against MeshLib alone is a
function whose specification lives in a proprietary binary nobody may read; pymeshfix's source is
mirrored and readable by anyone. The one place this currently bites is `holes.fill_small`, which
takes a **perimeter** threshold because MeshLib's `fillHoles` does, where pymeshfix and pymeshlab
both take a **boundary-edge count** — two of three references cannot express the incumbent
signature.

**pytorch3d** (a torch C++/CUDA extension over PyTorch, mirrored under `reference/pytorch3d`) is a
hard test dependency like the five above — `import pytorch3d.ops as p3d_ops` / `.loss as p3d_loss`
/ `.structures as p3d_structures`, all three aliases pinned in ruff's import conventions, never
through `pytest.importorskip`; reference variables take a **`_p3d`** suffix. Build meshes with
`tests.conversions.trimesh_to_pytorch3d` / `numpy_to_pytorch3d` / `warp_to_pytorch3d`, clouds with
`points_to_pytorch3d` (the `loss` container form) or `points_to_torch` (the bare batched tensor the
`ops` entry points take), and read a result back with `pytorch3d_to_numpy`.

It is the **first and only reference here with CUDA kernels of its own**, so it takes *two*
`LIBRARIES` rows (`pytorch3d-cpu` / `pytorch3d-cuda`) and a `triwarp-cuda` vs `pytorch3d-cuda`
ratio is the one GPU-against-GPU comparison in the suite. It is also the first reference triwarp
**already cited in its own prose without testing against**: `metrics.py`, `registration.py` and
`kernels/metrics.py` named it nine times, so two public functions had their specification pinned to
a library nothing in the suite ran. Those nine sentences now carry the measured number instead.
Fifteen hazards, all measured:

- **Everything is batched, with a leading minibatch axis, and the wrap fails silently.**
  `knn_points(p, q)` handed a bare `(P, 3)` reads it as `(N=P, P1=3, D)` and compares three points
  at full speed — no exception, and a *faster* wrong answer. So a single cloud goes in as `x[None]`
  and its answer comes out as `result[0]`, which is what `points_to_torch` is for, and every
  comparison asserts the reference's output **shape** before its values.
- **Every neighbour and Chamfer distance it returns is squared.** `knn_points().dists`,
  `ball_query().dists`, `loss.chamfer_distance` and both `point_mesh_*` scalars. Take the square
  root before comparing (measured **0.0** afterwards for `knn_points` on the host, 1.19e-07 on
  CUDA where its own kernel is a different reduction order).
- **`torch.cuda.is_available()` is the wrong probe for the CUDA extension.** A wheel whose arch
  list stops short of the device returns `True` and then fails every kernel with *"no kernel image
  is available"*; a build that compiled the CPU extension only — which is what `setup.py` selects
  whenever it finds no `CUDA_HOME`, the normal state of a CI runner — raises `RuntimeError: Not
  compiled with GPU support.` on a CUDA tensor. `benchmarks/conftest.py` gates its `-cuda` row on
  launching a two-point `knn_points`, which is the only probe that distinguishes them.
- **`Meshes` and `Pointclouds` are immutable *caching* containers** — the opposite end of the scale
  from a `ml.MeshSet`. Every `ops.*` / `loss.*` entry point is pure, so **one object serves many
  comparisons** and there is no freshness rule; but `verts_normals_packed`, `edges_packed`,
  `faces_packed_to_edges_packed`, `laplacian_packed` and `faces_areas_packed` are memoized on first
  request, so a *benchmark* row naming one of them has to build the container **inside** the timed
  callable or it reports nothing. And the answer lives on the `*_packed()` accessors, never in the
  constructor arguments: `SubdivideMeshes` and `ops.cubify` both return a `Meshes` whose inputs the
  caller never saw.
- **It does not cast for you, and one entry point casts anyway.** A float64 `Meshes` keeps float64
  through `verts_packed()`, so an unconverted reference compares a float64 answer against triwarp's
  float32 one and reads as triwarp being wrong by ~1e-7 — hence float32 in the converters. The
  exception is `ops.mesh_face_areas_normals`, whose C++ kernel returns **float32** whatever it was
  handed. Faces are stored as int64 regardless (an int32 tensor is silently widened).
- **`corresponding_points_alignment` and `iterative_closest_point` are row-vector.** They solve
  `s·X·R + T = Y`, so `R` is the transpose of `registration.procrustes`' linear block divided by the
  scale (measured 2.54e-07 / 2.62e-07); `T` needs no transform. Compare the **converged transform
  and the rmse**, never the iteration count — `relative_rmse_thr` is its own stopping rule.
- **`ops.cot_laplacian` returns two conventions in one call and neither is guessable.** Its
  off-diagonal is **twice** triwarp's half-cotangent table and its diagonal is identically **0.0**
  (not merely small) where `laplacian.cotmatrix` assembles the row sum; and its second return is
  `1 / inv_areas == 3 * M_ii`, the *reciprocal* of three times the barycentric lumped mass. Both
  cancel out of `mesh_laplacian_smoothing`'s ratios, which is why `energies.laplacian_smoothing_loss`
  can assemble from triwarp's own `cotmatrix`. `ops.laplacian` likewise writes **-1** on the diagonal
  where `laplacian.laplacian(equal_weight=True)` writes 0, and `ops.norm_laplacian` **is**
  `laplacian(equal_weight=False)` before its row normalization — same formula, same literal
  `eps = 1e-12`, measured 1.49e-08 after dividing by the row sum.
- **`ops.marching_cubes` does not exist**: it is not re-exported from `pytorch3d.ops`, only from
  `pytorch3d.ops.marching_cubes`, so `p3d_ops.marching_cubes` is an `AttributeError`. With
  `return_local_coords=False` it emits lattice indices, which is `levelset.marching_cubes`' own
  default and needs no convention fix at all — the one reference of five that does not.
- **`ops.cubify`'s three `align` modes are one uniform scale and translation apart.** Measured on a
  6³ occupancy sphere, `"topleft"` / `"corner"` / `"center"` return the **identical** face buffer
  and vertex count with bounding boxes `[-0.6, 1.0]`, `[-0.667, 0.667]` and `[-0.8, 0.8]` — all
  three reachable through `voxels.from_cells(cells, voxel_size, origin)`, so an `align=` keyword
  would be a second spelling of one that exists. It also **compacts**, which is what exposed
  `to_boxes(cull_internal=True)` returning the whole corner lattice.
- **`packed_to_padded` / `padded_to_packed` bounds-check nothing and corrupt the heap.** A
  mismatched `max_size` or `total_size` does not raise — it writes out of bounds and the process
  dies in `malloc`/`free` much later, in unrelated code. Their `first_idxs` are **starting indices,
  not counts**, which makes them `array.pack_1d_arrays`' `offsets` unchanged (measured `[0, 3, 8]`
  from both sides for lengths `(3, 5, 2)`); read the sizes off the buffers rather than passing
  literals.
- **`sample_points_from_meshes` caps the face count at 2²⁴.** It draws the face index with
  `torch.multinomial`, whose category limit is 16 777 216, so `lucy`'s 28 055 742 faces raise
  `RuntimeError: number of categories cannot exceed 2^24` rather than sampling.
  `happy_buddha`'s 1 087 716 are fine.
- **`add_points_features_to_volume_densities_features` is `[-1, 1]` local space, `[z, y, x]`
  storage, and `rescale_features=True`.** Its volume is `(minibatch, channels, D, H, W)` and a
  point's `(x, y, z)` indexes `(W, H, D)`, so its lattice is the **transpose** of
  `voxels.splat_onto_grid`'s; `align_corners=True` is triwarp's `bounds`; and the default
  `rescale_features` divides by `density.clamp(min_weight)`, which is what makes both sides an
  *average* rather than an accumulation. With those three lined up the two agree at exactly **0.0**
  on the host and 4.8e-07 on CUDA (atomic order).
- **`mesh_normal_consistency` counts pairs, not adjacencies.** It enumerates every pair of faces
  sharing an edge through its own `_C.mesh_normal_consistency_find_verts`, so an edge with `k`
  incident faces contributes `C(k, 2)` terms; `adjacency.face_adjacency` keeps only edges with
  *exactly* two faces and reports none at all there. The two agree exactly on edge-manifold input
  (measured 0.0155947 over 480 pairs) and read **0.777 against 0.0** on three faces sharing one
  edge. Restrict the comparison and pin the divergence.
- **`ops.taubin_smoothing` rebuilds its operator every half-pass**, from the current positions, and
  row-normalizes it — where every triwarp smoother assembles once. One fixed operator sits
  4.1e-03 / 6.5e-03 / 9.9e-03 from it at 1 / 3 / 10 of its iterations, the size of the displacement
  itself; `smoothing.filter_taubin(recompute=True)` closes that to 2.4e-07 at **~23x** the cost.
  Its `num_iter` counts lambda-mu **pairs**, like MeshLab's and open3d's.
- **`ico_sphere` is *not* the rotated-frame hazard open3d's Platonic solids are.** It starts from
  the identical `(±0.5257, ±0.8507, 0)` table `creation.icosphere` uses and subdivides the same way,
  so the positions correspond one-to-one and the 5.8e-05 residual is pytorch3d's table being
  *written* to four decimal places. Match by nearest vertex with a bijection check at 1e-4, not by
  index. `utils.torus`, by contrast, takes the **minor** radius first and builds its vertex table in
  a Python double loop, so it needs a real parameter mapping and its benchmark column is a
  per-vertex Python floor.

**Its CPU rows are Θ(N²) on anything with a neighbour query**, because `_C` carries no spatial
structure on *either* device — no tree, no grid, just the pairwise loop. Measured on this box:
`knn_points` 299.8 / 1 189.5 / 4 562.8 ms and `chamfer_distance` 557.8 / 2 319.1 / 9 077.1 ms at
10 k / 20 k / 40 k self-queries, i.e. 3.84-4.16x per doubling against the 4x a quadratic predicts,
which extrapolates to ~3.5 s per `knn` round on `bunny` and **~9 minutes per round** on `dragon`.
So a `pytorch3d-cpu` neighbour or chamfer row is capped at a feature mesh; the `-cuda` rows run the
scan meshes. Its `-cpu` row is nonetheless a *threaded* one (`torch.get_num_threads()` is 24 here),
so it belongs with `meshlib` rather than with trimesh / igl / pyvista / pymeshfix.

**And the GPU ratio is a crossover, not a bar — which is the most useful thing this reference
measures.** Brute force with perfect coalescing *beats* a BVH descent while the whole problem still
fits the device's bandwidth: measured 2.31 ms against triwarp's 3.29 at 20 000 points
(`knn_points`, **0.70x** — a loss for triwarp) and 74.65 against 0.87 at 200 000 (**85x**). So the
neighbour and chamfer groups need the point count as an **axis**; a one-size row reports whichever
side of the crossover it landed on. Half of that swing is *triwarp's*, and a row must not be read as
a statement about brute force: pytorch3d is a clean quadratic over the sweep (0.63, 2.26, 5.59,
20.18, 73.82 ms at 5 k / 20 k / 50 k / 100 k / 200 k) while `query_nearest` is **non-monotonic**
(0.82, 3.25, 7.38, **0.46**, 0.75 ms) — a 16x drop between 50 k and 100 k, identical to three digits
between its `bvh` and `hashgrid` backends, so the cost is in a stage the two share. That is the
search-radius heuristic (memory `hashgrid-nearest-radius-is-cubic`) and it is an open finding.

**Every CUDA row must synchronize torch's stream.** `wp.synchronize_device` synchronizes *Warp's*
and says nothing about torch's, so a `pytorch3d-cuda` row synchronized the Warp way times the launch
and not the kernel — the same class of silent error as section 8's launch-device hazard.
`BenchCase.run` branches on `kind == "pytorch3d"` for exactly this.

**Licensing: pytorch3d is BSD-3 (Meta) and torch is BSD-3**, so there is no `copyleft/` subtree to
avoid as in `reference/libigl`, no *use* restriction as with MeshLib, and no copyleft as with
pymeshlab or pymeshfix. `triwarp/` may name it — it already does, nine times — and may be derived
from it with attribution. It stays a test and benchmark dependency all the same: nothing in the
shipped package imports torch, and it must not start, since `torch` is a 2.5 GB install for a
library whose whole premise is Warp.

### Mesh fixtures (prefer over inline construction)

Reuse shared mesh fixtures from `tests/conftest.py` instead of building meshes in each test. Fixtures return `(mesh_tm: tm.Trimesh, mesh_wp: wp.Mesh)` via `tests.conversions.trimesh_to_warp`.

| Fixture | Use when |
|---------|----------|
| `icosahedron` | Default watertight solid, 12 vertices; inside/outside, surface sampling, sign tests |
| `icosphere`, `icosphere_coarse` | Closed and *curved* — `subdivisions=3` (642 vertices) and `2` (162). Reach for these wherever `icosahedron` is too coarse, rather than calling `tm.creation.icosphere` |
| `unit_box` | Sharp features: 12 creases at exactly 90° with 6 flat face diagonals, untranslated so a coordinate sign picks out one face. Crease / seam / dihedral-angle tests — `icosahedron` has no right angles and `cave_cube` is non-convex |
| `cave_cube` | Hollow / non-convex shell (boolean difference) |
| `hemisphere`, `half_torus` | Curved or open surfaces |
| `boy_surface` | Closed, watertight and **non-orientable**, χ = 1 — the `False` branch of `is_orientable` / `face_orientation_bits`, and `make_winding_consistent`'s impossible one |
| `mobius` | Non-orientable *with* a boundary: the same three predicates, one loop of 78 edges, χ = 0 |
| `bohemian_dome` | Closed genus 1 that self-intersects — `homology_generators` / `tree_cotree` at genus 1, and `is_self_intersecting` on a *closed* input |

The last three are built by `creation.parametric_surface` rather than by trimesh, and they are the
only inputs in the suite that are non-orientable or that have an odd Euler characteristic. A boolean
predicate asserted only on the orientable fixtures above is testing one branch; that is what these
close. `creation.parametric_surface` builds thirteen more surfaces that are not fixtures yet — reach
for one (and add the fixture) rather than hand-rolling a degenerate mesh, and see its docstring for
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

**Where a reference library computes the same quantity, one test must compare the two outputs.**
That is the obligation the classes below describe, and it is not discharged by an invariant: a
function can be watertight, symmetric, idempotent and manifold while computing the wrong answer.
So if trimesh / igl / potpourri3d / pymeshlab / open3d / pyvista / meshlib / pymeshfix / pytorch3d
has the quantity — check, do not
assume: §6's hazard blocks list what each one actually binds, and several names that *look* present
are not — there is a class A/B/C comparison against it, and `tests/test_parity.py` enforces that for
every *benchmarked* pair. Below that bar, `pytest.mark.parity` is the marker that records it.

**Invariant checks are welcome and belong in the same test.** Watertightness, an involution, a
counting identity, a round trip, a conservation law — these catch failure modes no reference
comparison sees (a reference agreeing with you on a shared bug, or a quantity the reference computes
in a different gauge). Assert them *alongside* the output comparison rather than in a test of their
own, so one test carries the whole claim about the function and the parity marker sits on the test
that does the comparing. Split them out only when the invariant needs an input the comparison cannot
use — a fixture the reference rejects, or a degenerate case it crashes on.

**Where no reference computes the quantity, an invariant-only test is the honest answer**, and its
docstring says so in those words: *"Not a library comparison: <why none exists>"*, followed by what
the invariant excludes. That is a fifth label beside A–D, not a class-D exemption — D is for a
*benchmarked* pair whose results are genuinely incomparable and needs a `noparity` entry; this is
for a quantity with no counterpart to benchmark. `halfedge_twins` (no reference has a halfedge
structure), `homology.tree_cotree` (nothing computes a basis) and `geodesic_walk`'s arc-length checks
are the shape of it.

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

Write the label as **`Class A`** (capital, the word before the letter). A lowercase `class b` reads
the same to a human and is invisible to a grep for the convention — 21 of them had accumulated
before anyone looked, then 9 more after that count was taken to 0, which is why
`test_api_conventions.py` now gates it rather than trusting the convention.

**Four phrases carry a label, not two**, and a scan or a review that knows only the first two
misreads 14 correct tests as unlabelled. Measured over the 523 tests whose `assert` reads a
reference-suffixed variable: `Class [ABCD]` (464), `Not a library comparison` (48), `Triwarp against
triwarp` (9), `Not a parity assert` (5). The last two are the triwarp-against-triwarp family below,
and they are labels in good standing — do not reword them to fit a narrower grep.

Never a parity assert: shape-only or `isfinite`-only (that is the *benchmark's* assert, and this
gate exists to stop it migrating inward); triwarp compared with itself; a threshold a constant
output would pass. A boolean assert must be parametrized over inputs producing both answers.

**Triwarp-against-triwarp is not a parity assert but is still a legitimate test**, for one job:
pinning two entry points to each other where only one has an oracle — a mask form against an index
form, a precomputed path against the deriving one, a CPU run against a CUDA one. Say which of the
two carries the oracle, so the pair does not read as a comparison against a reference.

**Check the comparison is not vacuous on its fixture**, which the gate cannot do for you. A test
comparing two *empty* answers passes, reads as coverage, and tests nothing — measured: `test_ears`
compared `igl.ears` against `boundary.ears` on `hemisphere` and `half_torus`, where neither library
finds a single ear, so the assert was `[] == []` and the loop body checking the corner convention
never ran. Making it non-vacuous immediately surfaced a real disagreement (the two number the ear's
local edge differently, `triwarp_opp == (igl_opp + 1) % 3`). So: assert the reference produced a
non-empty answer, or assert its expected count, before comparing to it — and treat "this function is
already the oracle in tests/" as no evidence at all that the comparison is live.

**And a constant answer is as vacuous as an empty one — the sweep must look for both.** Two measured
cases, each of which had a docstring *asserting the non-vacuity that was absent*, and in both the
sentence is what stopped anyone re-checking. `test_connected_component_labels_random` said "200
random edges over 64 nodes gives several components rather than one" and produced **1** component
holding all 64 nodes, so the class-B label-packing transform it exists for was the identity (its
`_matches_igl` sibling said the same false sentence and was single-component on *both* of its
shapes — one because 200 edges is past the giant-component threshold, the other because its
`n_edges == node_count - 1` branch builds a path). `test_discrete_mean_curvature` said "over every
vertex" and ran on a *regular* icosahedron, where trimesh's answer is **one value at spread 0.0**, so
a permuted result, an off-by-one in the gather and a query/vertex index swap all pass. The rule that
follows: **a claim about the input's shape is a claim an assert can carry cheaply, so make it an
assert and not a sentence** — `assert np.unique(labels_np).shape[0] > 1`,
`assert np.ptp(reference) > 1e-3`.

Reuse `tests/comparisons.py` (`lexsort_rows`, `assert_unordered_rows_equal`, `undirected_edges`,
`edge_multiplicity`, `euler_characteristic`, `open_edge_count`, `canonical_labels`,
`same_partition`, `canonical_winding`, `assert_same_up_to_sign`, `assert_cyclic_permutation_equal`,
`assert_same_loop_set`, `trimesh_outline_loops`, `fraction_within`, `symmetric_chamfer`,
`chamfer_two_sided`, `symmetric_surface_distance`, `hausdorff_two_sided`,
`hausdorff_surface_two_sided`) and `tests/conversions.py` (`numpy_to_warp`, `numpy_to_warp_uv`,
`points_to_warp`, `points_to_warp_uv`, `trimesh_to_warp`, `warp_to_trimesh`, `trimesh_to_open3d`, `points_to_open3d`, `open3d_to_trimesh`,
`trimesh_to_open3d_t`, `trimesh_to_pymeshlab`, `warp_to_pymeshlab`, `points_to_pymeshlab`,
`trimesh_to_pyvista`, `points_to_pyvista`, `pyvista_edges_to_indices`, `numpy_to_meshlib`,
`trimesh_to_meshlib`, `warp_to_meshlib`, `points_to_meshlib`, `meshlib_to_trimesh`,
`numpy_to_meshlib_bitset`, `meshlib_scalars_to_numpy`, `meshlib_bitset_to_numpy`,
`numpy_to_pymeshfix`, `trimesh_to_pymeshfix`, `warp_to_pymeshfix`, `pymeshfix_to_numpy`,
`pymeshfix_intersecting_faces`, `pymeshfix_face_remap`, `points_to_torch`, `numpy_to_pytorch3d`,
`trimesh_to_pytorch3d`, `warp_to_pytorch3d`, `points_to_pytorch3d`, `pytorch3d_to_numpy`,
`faces_igl`, `mesh_igl`)
rather than re-rolling either. **Check both modules before writing a private helper in a test
file** — every one of the six consolidated in 2026-08 was written by someone who did not, and
`undirected_edges` alone had been spelled three different ways across six files.

**`points_to_warp` and `warp_to_trimesh` are the two most-reached-for, and both were re-rolled for a
long time before they existed or were adopted.** The bare-cloud upload — query points, normals, ray
origins and directions, a sampled surface — had been written out **403** times in four equivalent
spellings across 35 files *plus* six one-line private copies carrying 181 more calls, because every
reference library had a `points_to_*` and Warp did not; it is now one helper at 532 call sites. The
readback direction is the mirror image and the asymmetry is worth knowing about yourself: a test
author reaches for the shared helper when *building* the reference and writes the readback by hand,
every time, so `warp_to_trimesh` sat at 4 mentions in 2 files while 38 sites inlined
`tm.Trimesh(x.numpy(), f.numpy().reshape(-1, 3), process=False)`.

**And the fixture *sets* are shared too**: `CLOSED_MESHES`, `OPEN_MESHES` and
`MESHES = CLOSED_MESHES + OPEN_MESHES` live in `tests/conftest.py`. "The four that span closed/open
and convex/non-convex" is a decision about coverage, and it had been restated in ten files under
three names, which meant a fixture added to the set reached exactly one of them. Import them; keep a
local list only where it is genuinely a different set, and say in a comment why (`test_adjacency.py`
drops `cave_cube` because its coplanar box faces make every adjacency angle 0 or pi/2).

`canonical_labels` is the label-packing transform every component comparison
needs — triwarp names a component after a representative element, igl and scipy number `0..k-1` in
their own traversal orders and VTK's `RegionId` numbers them in a third, so only the *partition* is
shared. `open3d`, `pyvista` and `pytorch3d` are hard test dependencies like `pymeshlab` and
`igl` — import them plainly as `import open3d as o3d` / `import pyvista as pv` /
`import pytorch3d.ops as p3d_ops`, never through `pytest.importorskip`; see their hazard blocks
above.

**Two of the class-C helpers take different inputs and the mesh one raises on point arrays.**
`symmetric_chamfer(mesh_a, mesh_b)` takes two *meshes* and samples them itself, where
`chamfer_two_sided(points_a, points_b)` takes two clouds already drawn — which is what a comparison
between two *samplers* needs. And prefer the mean form over `hausdorff_two_sided` where the claim is
distributional: measured on two independent 1 000-point samplings of `icosphere(2)`, the mean
statistic separates the same mesh from one scaled by 1.15 by **6.8x** (0.00768 against 0.05227)
where the worst-case Hausdorff separates them by **1.4** (0.162 against 0.228), because one stray
sample in a tail dominates a maximum.

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
edges_wp = twt.empty_2d((n_faces * 3, 2), wp.int32, device=faces_wp.device)
angles_wp = twt.empty_2d((f, 3), wp.float32, device=vertices_wp.device)
```

Empty mesh / no adjacency early return:

```python
if n_faces == 0:
    return twt.empty_2d((0, 2), wp.int32, device=faces_wp.device)
```

### Returns and parameters

```python
def faces_to_edges(faces_wp: wp.array[wp.int32], sorted: bool = False) -> twt.Array2dInt32:
    out_wp = twt.empty_2d((n_faces * 3, 2), wp.int32, device=faces_wp.device)
    wp.launch(kernel_graph.faces_to_edges, dim=n_faces, inputs=[faces_wp, out_wp], device=faces_wp.device)
    return twt.as_array2d(out_wp, wp.int32)
```

Optional 2D arguments: `edges_sorted_wp: twt.Array2dInt32 | None = None`.

### Runtime checks (not `isinstance`)

- `twt.ensure_ndim(arr_wp, 2, dtype=wp.int32)` — validate rank and dtype on inputs.
- `twt.as_array2d(arr_wp, wp.int32)` — check then narrow the return type for Pyright. Overloaded on the `dtype` argument for `wp.int32` / `wp.float32` / `wp.float64`; `twt.as_array3d` covers `wp.float32` / `wp.bool` at rank 3.

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

**A name in `dir(wp)` that is missing from those mirrors is usually hidden on purpose, not missed by
the transcription.** `dir(wp)` exposes ~510 names against ~137 the package uses, and browsing the
remainder for adoption candidates is how the `wp.dense_chol` / `dense_subs` / `dense_solve` family
gets proposed as a replacement for a hand-written 6×6 Cholesky. Introspect before planning around one:

```python
from warp._src.context import builtin_functions
f = builtin_functions["dense_chol"]      # a Function, the same handle §4's kernel factories capture
print(f.hidden, f.doc, f.input_types)    # True  'WIP'  {n: int32, A: array(ndim=1, float32), ...}
```

`hidden: True` / `doc: "WIP"` is the answer, and the mirrors' silence was the same answer read one
step earlier.

**Then check the *quantity*, not the name — adopting a matching builtin is sometimes a regression.**
Four measured rejections worth not re-deriving, each of which reads as a match by name:
`wp.sample_unit_hemisphere_surface` would replace `visibility`'s low-discrepancy Fibonacci lattice
(whose `local[2] == dot(direction, normal)` identity the kernel depends on) with a Monte-Carlo
estimate at the same ray count — variance where there was none, and every occlusion parity test would
need a tolerance instead of an equality; `wp.norm_huber` is the Huber *norm* where
`registration.robust_weight` needs the IRLS *weight* `ρ'(r)/r`; `wp.volume_voxel_count` is a capacity
(§4); and the `dense_*` family takes `wp.array[float32]` where the caller holds a
`wp.spatial_matrix` in registers, which memory record `warp-per-thread-row-storage` measured as a 2x
loss. Also check the signature's *storage class* and its precision: a builtin that only speaks
`float32` cannot serve the `float64` half of a dispatch, so half the hand-written code stays either
way. And where a builtin *does* fit, the argument is often single-source-of-truth rather than speed —
`wp.volume_index_to_world` measured perf-neutral (1.08x at 200k voxels, 1.005x at 2M, both
launch-dominated) against a hand-rolled half-voxel transform that agrees with it to 3.58e-07; adopt
it for the convention, and if only half the sites can convert (the voxelizers run *before* a volume
exists), **name the split in the module docstring** rather than leaving two silent conventions in one
file.

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
  kernel takes 4.9 arguments and only 15 of 483 take 12 or more, so this matters in exactly one
  place: a wide kernel launched inside a Python loop. Bundle its invariant arrays into a
  `@wp.struct` built **once** in the wrapper (construction is ~2.6 µs; rebuilding per launch gives
  the saving back), which collapses them to a single argument — measured 1.17–1.93x on
  `holes._fill_dp`, **1.87–1.90x** on `holes._stitch_halves`' anti-diagonal loop (200 to 1 024
  launches, flat in the rim size, gated on an identical traceback) and 1.34x on
  `reconstruction._bpa_wave`'s launch path. ~1.0 µs per dropped argument is a **floor**, not a
  fit: the stitch bundle returned ~1.8 µs each, its arguments being six `wp.array` handles rather
  than scalars. Struct fields take the §2 subscript annotation (`a: wp.array[wp.int32]`). Do
  **not** open a tree-wide bundling pass: bundling a 5-argument kernel trades ~3 µs for a struct
  the reader has to open, and §14's "no speculative generality" applies to argument bundles too.
- **Width alone does not qualify a kernel for a bundle — the launch count around it does.** The
  three bundled kernels each launch hundreds of times with nothing else in the loop.
  `remesh.objective_flip_candidates` is the same 14 arguments and is declined, because a flip
  *round* rebuilds the whole face adjacency around its one launch: measured on a noisy
  icosphere(4), `flip_to_delaunay` converges in **2** rounds at 514 µs each, so the nine
  bundleable arguments bound the saving at 18 µs, **1.75 %** of the call. Before proposing a
  bundle, count the launches per call and divide.
- **Graph capture and argument bundling address different loops, and neither substitutes for the
  other: capture pays on a launch sequence that repeats identically, a bundle on one that runs
  once.** Recording a graph costs at least what issuing the launches costs, because capture
  intercepts each one. Measured in one session, 14-argument kernel at `dim=64` so host cost
  dominates, 30 reps, median/min at 400 launches: loose arguments 11.3/9.8 ms; the `@wp.struct`
  bundle 6.0/4.6 ms (**1.88x**); capture-and-replay-**once** 13.5/10.4 ms (**0.84x — a loss**);
  replay of an already-recorded sequence 2.6/0.5 ms (**4.29x**). That last row is where capture's
  reputation comes from and it is unreachable without a *repeated* sequence. Two rules follow.
  **Do not bundle a kernel whose launch is already captured** — a replayed launch costs ~1.17 µs
  whatever its argument list (memory: `warp-cg-iteration-launch-floor`), so `remesh._issue_pass`'s
  14-argument kernel and the multigrid V-cycle inside `wp.capture_while` gain nothing. And **do not
  reach for capture on a once-through Python loop** — `holes._fill_dp`'s span loop and the stitch
  DP's diagonal loop are each recorded and replayed exactly once, which is the 0.84x row. The
  unexplored lever is the reverse: a *repeated* wrapper loop issuing an identical sequence that is
  not yet captured is worth 4-45x, and `wp.capture_if` (a device-side conditional, unused here) is
  the primitive for a stage that currently spends a host readback deciding.
- **A decline is a result — write it at the site, with the number, and resolve every site the
  finding named.** The exemplary case is `kernels/neighbors.py:122-132`, where the `wp.length_sq`
  swap was measured, declined, and the reasoning (including that the two spellings are not the same
  predicate) written into the source; the counter-case is the same finding's other two sites in
  `ball_pivoting`, which were neither converted nor annotated and were therefore re-derived a pass
  later. A finding that names five sites is closed when all five are converted **or** annotated —
  a partially applied one reads as an open question and costs the next pass the whole re-derivation.
- **Before proposing an optimization, read what *calls* the thing — the decline may already be
  written there.** A scan reads bodies; a measured decision is prose, and it lives at the call site
  rather than in the function. The fifth kernel pass proposed replacing `holes._closest_loop_pair`'s
  three launches and `(n_a, n_b)` matrix with a one-kernel atomic reduction, having read the three
  kernels and the private wrapper — and its caller's preamble comment, **three lines above the
  call**, already read *"Measured, and deliberately left alone: this preamble … is 5.0 % of the call
  at a 100-vertex rim on CUDA and falls to 0.7 % at 1 000 and 0.4 % at 4 000."* Grep the caller for a
  number first. Where the decline is genuinely absent, measure it and write it at *both* ends: the
  sibling item that pass did measure (folding `holes.global_argmin` into `row_argmin`, **1.51 % /
  0.58 % / 0.41 %** of `stitch_loops` at rims of 100 / 1 000 / 4 000, launch-dominated so the
  single-lane walk is not the cost) is now recorded on the kernel *and* at the wrapper.
- **A share that falls as the input grows is a decline, not a small win.** Both of the above shrink
  with rim size, which means the saving is largest exactly where the call is already cheap — the
  same shape as §13's `objective_flip_candidates` bundle (1.75 % of a two-round pass). Report the
  share at more than one size before deciding; a single operating point cannot show the trend.
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
`pytest` run. Nineteen checks. Eight scan the public surface of `triwarp/` (excluding `kernels/`): a
summary line naming a reference library (§10); a `*_mask` producer that does not return
`wp.array[wp.bool]`; a module summary advertising Warp; a module without a `tests/` **and** a
`benchmarks/` file named for it; a private name reached across a module boundary; one public name
exported by two modules; a top-level `kernels/<name>.py` without its `triwarp/<name>.py` or the
reverse (§4); and a private helper defined above its first caller (§11). The ninth scans `kernels/`
**as well**: a comment or docstring blaming a Warp version older than the installed `warp-lang`.
Five enforce an earlier section's convention on kernel code: §3's `out_` prefix and
end-of-signature position for a written argument (its two exemption classes carried as
`_KERNEL_OUTPUT_ALLOWLIST`); §2's subscript-style array annotation — this one scans the whole
package, because only in an *annotation* position is `wp.array(dtype=T)` the stale spelling rather
than a legal allocation; §3's cast spelling, **check 16**, no bare `int(...)` / `float(...)`; §5's
integer division, **check 17**, no `/` between two operands that are integers *by declaration*; and
§2's type standard, **check 18**, no bare `bool` / `int` / `float` annotation in a `@wp.kernel` /
`@wp.func` signature. Those last three are one family, and the family is the point: all three
spellings are *legal* and generate identical code, so the defect is invisible to the compiler and to
the suite, and nothing but a scan holds the line. Check 18 reads kernel-scope signatures **only** —
a kernel *factory* is ordinary Python and its `row_size: int` / `name: str` parameters are correct,
which is why `str` is not in its table. The last six are newer and each exists because the same
defect was found twice:

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
- **An integer division spelled `/` inside a kernel.** Check 17, and it is legibility rather than
  correctness — the two spellings are the *same* operation on integers in Warp. The third pass
  converted eight sites and wrote the rule into §5; `algorithms/multigrid.py`, written afterwards,
  reintroduced two, and the scan built for the check turned up four more in
  `algorithms/blue_noise.py` that a textual pass had missed. The scan types an operand only **by
  declaration** — an annotated `wp.int*` / `wp.uint*` / `wp.Int` parameter, an element of an array
  whose annotated dtype is one of those, a module-level integer `wp.constant`, an integer literal,
  a `.shape[...]`, `wp.tid()`, an integer constructor, or an integer-preserving expression over
  those — because a scan that misfires on float division gets switched off by the first person it
  annoys. `test_integer_division_scan_ignores_float_operands` pins the negative cases.
- **A fenced ```python docstring example that does not run.** The one static check that is not
  static: `tests/api_conventions.py` extracts the blocks and `tests/test_api_conventions.py`
  `exec`s them against a mesh fixture, because both defects it was written for were *runtime*
  ones (`wp.array` compared with a float, a NumPy bool array handed to `flatnonzero`) and
  `ast.parse` sees nothing wrong with either. Blocks holding a bare `...` are deliberate outlines
  and skip. A new example that needs a name the fixture does not bind fails with `NameError` —
  extend `example_namespace`, do not weaken the test.
- **A test comparing against a reference library with no class label.** Check 19, the second one
  that reads `tests/` rather than `triwarp/`, and it exists because the convention has decayed
  **twice**: the lowercase `class b` spelling went 21 → 0 → 9, invisible to §6's own prescribed grep
  because a human reads `class B` and `Class B` the same. Four decisions keep it from misfiring, each
  of which cost a wrong count while it was being built. It keys on `ast.Assert`, not on the function
  body — a fixture unpack `mesh_tm, mesh_wp = icosphere` names a `_tm` variable in every mesh test,
  and keying on the body takes it from 0 hits to 120. It accepts **all four** label phrases (§6), not
  the two headline ones, or it would fail 14 correct tests and the author's fix would be to reword
  good docstrings. It leaves `_np` out of its suffix list, measured at 290 false positives. And it
  checks only that a label is *present*, never that it is the right one: a `_tm` name in an assert is
  not proof of an oracle — `test_split_single_component` compares `split`'s output against the
  *input* mesh's vertices, which is a round trip. The rarer defect it also closes is a comparison
  with **no docstring at all**, which ruff cannot see because `D103` is in the ignore list.

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
- **A packed buffer and its offsets are returned, and accepted, values first.** `(flat, offsets)`,
  never `(offsets, flat)`; where a third array rides along it is a *per-item* one and goes last
  (`(ring_halfedges, offsets, is_boundary)`). The convention is stated in
  `array.pack_1d_arrays`' docstring, the primitive the rest are built on. Both halves are usually
  `wp.array[wp.int32]`, so **a transposed unpack type-checks, runs, and indexes garbage** — there
  is nothing but the convention to lean on, which is why it is written down here. Measured before
  it was made unanimous: 13 of 15 public returns and **4 of 4** argument lists were already values
  first, `igl.vertex_triangle_adjacency` orders it the same way, and the two exceptions
  (`adjacency.vertex_face_adjacency`, `halfedge.vertex_one_rings`) were swapped to match. The one
  place a caller *constructs* such a pair by hand is a precomputed keyword
  (`descend_field(vertex_faces=…)`); transposing it there segfaulted the CPU backend several
  launches after the call, not at it, so that composition carries a test of its own.
- **A mask is named `<element>_<property>_mask`, element first.** Element-first sorts and
  completes: type `tw.validation.face_` and every per-face predicate appears, which is exactly what
  the property-first spellings did not do — `bad_face_mask` and `flipped_faces_mask` were the two
  nobody could find. The 2026-08 pass converted five (`repair.bad_face_mask` →
  `validation.face_defective_mask`, `parametrization.flipped_faces_mask` / `flipped_face_indices` →
  `face_flipped_*`, `points.finite_point_mask` / `duplicate_point_mask` → `point_finite_mask` /
  `point_duplicate_mask`) and deliberately left five alone, because **a convention followed
  everywhere regardless of fit is not worth having**: `radius_outlier_mask` /
  `statistical_outlier_mask` (the property *is* the name, and `point_radius_outlier_mask` is worse),
  `half_space_mask` (the element is implicit and the geometry is the point), `fillable_loop_mask`
  (already element-first — "loop" is the element), `uv_seam_vertex_mask` (element-final, and the
  qualifier is a namespace) and `convex_subset_mask` / `convex_superset_mask` (the module docstring
  turns on the subset/superset opposition and the names carry it). Check 2 already scans these
  names for their dtype, so it is where the spelling rule belongs too — but note nothing enforces
  the *order*, which is a review question.
- **Two public names that differ by one character are a defect even when both are correct.**
  `boundary.boundary_loop` and `boundary.boundary_loops` meant "the longest one" and "all of them";
  the singular is now `longest_boundary_loop`, which is what its own docstring already said. Look
  for this whenever a plural is added next to an existing singular.
- **A tuning choice is a keyword, not a name — and if the kernel already branches on it, the
  Python layer is the only place it doubled.** `neighbors` exposed each ball and nearest query
  twice, once per accelerator, for eight names covering four operations: identical positional
  arguments, identical returns, differing only in the name of the optional prebuilt structure.
  `kernels/neighbors.py` had *already* unified them behind `ACCEL_HASHGRID` / `ACCEL_BVH` selectors
  in one warp-uniform kernel, the benchmark groups already treated the backend as an axis, and
  `query_bvh_nearest`'s docstring carried a note telling the caller which of the two names to type
  — a naming scheme that needs such a note is doing the caller's dispatch for them. They are now
  `query_ball` / `query_ball_count` / `query_ball_with_offsets` / `query_nearest` with
  `backend="hashgrid" | "bvh"` and an `accelerator=` that infers it. Three consequences worth
  keeping:
    - **A default that must be distinguishable from "not passed" is spelled `None`.** `backend`
      defaults to `None`, documented as "`hashgrid` when no `accelerator` is given", so that
      handing over a `wp.Bvh` and nothing else does not read as contradicting a default the caller
      never wrote. Only an *explicit* mismatch raises.
    - **Keep the discriminator in the benchmark group name, not in the function name.** The two
      backends are genuinely different cost rows for one function, so the groups are
      `query_ball_bvh` / `query_ball_hashgrid` and `query_nearest_{bvh,hashgrid}_k{1,7,64}`. The
      group name is the parity key, so all 19 markers moved in the same commit and the matrix
      stayed at its count.
    - **A merge like this needs a triwarp-against-triwarp test that the two paths agree**, or the
      shared group name is an unchecked claim. `test_the_two_backends_agree` is that test; it is
      not a parity assert, and says so.
  What stays split: `query_bvh_aabb_with_offsets` and `query_bvh_box` are genuinely BVH-only — a
  hash grid has no box query — so naming the structure there is informative rather than redundant.
  And where the *pairings are different algorithms over different inputs* rather than one algorithm
  with a tuning knob, the name should keep carrying the type: `metrics.chamfer_*` / `hausdorff_*`
  were considered for the same treatment and declined.
- **When a comment and the body disagree, decide which one is load-bearing before "fixing" it — the
  usual answer is the comment.** `tangent_space.any_perpendicular`'s comment claims it crosses with
  *"whichever coordinate axis the normal is least aligned with"* while the body compares only
  `|n[0]|` against `|n[1]|` and never returns z. The body is **correct for its purpose** (it needs
  any axis not parallel to the normal, and x or y always qualifies), and "correcting" it into a
  three-way argmin would move the tangent frame at every z-dominant normal, under `visibility`'s
  ray bundles and `tangent_space`'s frames. Fix the sentence, leave the branch, and say in the
  commit which of the two you changed and why.
- **A new Warp construct can silently switch off a static check that predates it.** Memory record
  `wp-struct-hides-writes-from-check-13`: moving buffers into a `@wp.struct` removed them from check
  13's view entirely, because the check resolved store targets to a bare `Name`. So after moving
  writes behind a struct field, a `@wp.func` return, or any new spelling, **confirm the checks that
  used to see those lines still see them** — a green suite after a refactor is equally consistent
  with "still covered" and "no longer looked at". Extending the check is the fix; dropping the
  construct is not.
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
- **The axis is not always the dtype: `Any` is generic over the *rank* and the *dimension* too, and
  a helper pinned to either breeds copies exactly as a precision-pinned one does.** Two measured
  instances, both merged in the fifth kernel pass. A 5×5 and a 6×6 Householder normal-equation solve
  lived in `curvature` and `smoothing`, and what pinned them was one line — a `for k in range(N)`
  singularity test, because **a matrix has no readable `.shape` in kernel scope**
  (`r.shape[0]` is a `WarpCodegenAttributeError` at parse time on Warp 1.16). The rank-free spelling
  is a reduction over the diagonal: `wp.min(wp.abs(wp.get_diag(r))) < tol` asks the identical
  question with no loop, and `linalg.solve_normal_equations` now serves both. And a Cramer's-rule
  barycentric solve lived once per *dimension* (`wp.vec2`, `wp.vec3`) although `wp.length_sq` and
  `wp.dot` say nothing about the ambient dimension — one `Any` body serves both. **When two bodies
  differ only in a size, look for the one statement that names it and ask whether Warp has a
  reduction for it.**
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
- **Renaming a *keyword argument* has its own artifact list, and a call-site scan sees none of it.**
  Three sites survived a paren-aware rewrite of all 33 `neighbors` query calls in the 2026-08 pass,
  each invisible for a different reason, and each was caught by a *runtime* check rather than a
  static one:
    1. **A `TypedDict` field feeding a `**splat`.** `registration._TargetIndex` declared `bvh:
       wp.Bvh` and the call site read `query_nearest(..., **target_index)` — the keyword's name is
       nowhere near the call. Eight tests failed with `TypeError: unexpected keyword argument
       'bvh'`. Grep the *old keyword name on its own*, not just at call sites.
    2. **A rename that ran after the function rename.** A bulk pass had already turned
       `query_hashgrid_ball_count` into `query_ball_count`, so the later keyword pass no longer
       matched it and left `grid=` behind. Sweep for "merged name still carrying the old keyword"
       as a separate, final pass — ordering two mechanical passes wrong silently skips their
       intersection.
    3. **A fenced ```python docstring example.** Check 14 is what found it, which is the second
       time that check has paid for itself; `ast.parse` and every grep were clean.
  The general rule: after any mechanical rename, re-derive the *residual* set from the new names
  rather than trusting that the pass which produced them was complete.

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
