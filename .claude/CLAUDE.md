# Role: NVIDIA Warp Expert

You are an expert in NVIDIA Warp (`wp`) working on **triwarp**, a GPU geometry-processing library.
Follow every rule below when writing kernels, `@wp.func` helpers, Python-scope wrappers, tests and
benchmarks.

**This file has two parts, and they answer different questions.**

- **Part I — Rules (§1-§11).** How code in this repository must be written. Obey these.
- **Part II — Measured facts (§12-§16).** What has already been measured on this hardware and this
  Warp version: platform bugs, a cost model, kernel-shape verdicts, measurement traps, and the
  status of specific triwarp functions. Read the relevant section **before** proposing an
  optimization or diagnosing a slowdown — most plausible ideas in this repository have already been
  built and priced, and several of them lost.

**Part II is the knowledge base, and it is the only one.** There is no separate memory store: a new
measured finding — a ratio, a refuted idea, a platform quirk, a reference library's trap — is written
into the section of Part II that owns it, next to the numbers it revises or contradicts, and a rule
that follows from it goes into Part I with a cross-reference. Do not start a parallel notes file; a
finding recorded anywhere else is a finding the next reader will re-derive.

Two conventions that hold throughout:

- **A measured decline is a result.** When a change is measured and rejected, the number is written
  at the site *and* recorded here. Do not re-derive it; do not re-propose it without new evidence.
- **A Warp version claim is spelled `Warp 1.17`** — the word immediately before the number. The
  staleness check (§4.5, check 9) reads only that anchored form, because this package writes
  measured ratios in the same shape (`1.06 ms`, `within 1.25x`) and a bare `1.N` token matched 30 of
  those against 3 real claims. Installed today: **warp-lang 1.17.0** on an **RTX 5090**.

## Contents

**Part I — Rules**

1. [Kernel syntax and semantics](#1-kernel-syntax-and-semantics) — decorators, types, casts,
   kernel-scope restrictions, arithmetic spellings, mutable accumulators
2. [Kernel architecture and launches](#2-kernel-architecture-and-launches) — launch discipline,
   `wp.block_dim()`, block-per-item, fusing, `wp.overload`, `wp.ref`, factories, `@wp.struct`,
   per-thread storage
3. [Python-scope wrappers](#3-python-scope-wrappers) — layout, `triwarp.typing`, allocation, gather,
   `wp.map`, dtype conversion, `BsrMatrix`, the NumPy policy, device rules, readbacks
4. [Evolving the public API](#4-evolving-the-public-api) — naming, signatures, docstring agreement,
   moving/renaming, the 23 mechanical checks
5. [Function ordering within a module](#5-function-ordering-within-a-module)
6. [Documentation](#6-documentation-mkdocs--mkdocstrings)
7. [Testing](#7-testing) — conventions, devices, fixtures, the parity gate, shared helpers, the nine
   reference libraries
8. [Tooling and local validation](#8-tooling-and-local-validation)
9. [Performance work: measure before you change](#9-performance-work-measure-before-you-change)
10. [Warp API reference mirrors](#10-warp-api-reference-mirrors)
11. [Running long commands](#11-running-long-commands-never-poll-with-until)

**Part II — Measured facts**

12. [The Warp platform](#12-the-warp-platform-bugs-quirks-and-version-status) — memory safety, CPU
    tiles, `wp.ref`, numerics, casts, compilation, `warp.sparse`, builtin verdicts, `wp.Volume`,
    upgrade discipline
13. [The cost model (RTX 5090)](#13-the-cost-model-rtx-5090) — host per-call, device/memory, tuning
    constants
14. [Kernel-shape verdicts](#14-kernel-shape-verdicts) — what wins, what is refuted
15. [Benchmark and measurement traps](#15-benchmark-and-measurement-traps)
16. [triwarp component status](#16-triwarp-component-status) — the launch-resolution pass,
    shipped results, open defects, refuted plans

---
---

# PART I — RULES

---

## 1. Kernel syntax and semantics

### 1.1 Decorators

- `@wp.kernel` for entry points launched via `wp.launch()`; `@wp.func` for helpers called from
  kernels or other `@wp.func`s.
- **`@wp.kernel` functions MUST NOT return a value.** Annotate `-> None` and write into output
  arrays.
- **Every argument of both `@wp.kernel` and `@wp.func` MUST be explicitly typed.**
- `wp.tid()` may be called **only inside `@wp.kernel`**, never inside `@wp.func` — pass the thread
  index as an argument.
- `@wp.func` may return several values as a tuple; declare `tuple[T1, T2, ...]`. **The generic
  forms parse on Warp 1.17** — probed directly: `tuple[Any, Any, Any]`, `tuple[wp.Float,
  wp.Float]` and the mixed `tuple[Any, wp.Float]` all compile and run on an `Any`-generic
  helper, so a dtype- or rank-generic helper is no reason to leave the annotation off.
- A `@wp.func` may read `values.shape[1]` and may carry **default argument values**
  (`def f(a: wp.int32, b: wp.int32 = 3)`). `wp.launch` also accepts **`None` for an array argument**
  (a null descriptor) — legal as long as no thread indexes it.

### 1.2 Type standards

- Use subscript-style array annotations: `wp.array[wp.vec3]`, `wp.array[wp.int32]`,
  `wp.array[wp.float32]`. Check 14 scans the **whole package** for this, because only in an
  *annotation* position is `wp.array(dtype=T)` the stale spelling rather than a legal allocation.
- In **`triwarp/kernels/` only**: `wp.array2d[T]`, `wp.array3d[T]`, `wp.array4d[T]` for
  multi-dimensional kernel arguments, with matching multi-index `wp.tid()` unpacking. In Python
  wrappers use the `triwarp.typing` aliases instead (§3.2).
- Built-in vector/matrix types: `wp.vec2/3/4`, `wp.mat22/33/44`, `wp.quat`. `wp.indexedarray[wp.vec3]` (and its
  siblings) for sparse/indexed access patterns.
- **No bare `bool` / `int` / `float` annotation** in a `@wp.kernel` / `@wp.func` signature
  (check 18). This reads kernel-scope signatures only — a kernel **factory** is ordinary Python and
  its `row_size: int` / `name: str` parameters are correct, which is why `str` is not in the check's
  table.
- Module-level numeric constants go through `wp.constant()` so they are visible in kernel scope:
  ```python
  TOLERANCE_MERGE = wp.constant(wp.float32(1e-8))
  ```
  A plain Python float works but is treated as `wp.float32`; use the `wp.float64(...)` constructor
  where 64-bit precision is required.
- **Prefer dtype-generic `@wp.func`s** — `wp.Float` / `wp.Scalar` for scalars, `Any` for vectors,
  the `kernels/predicates.py` convention — as long as the dispatch stays readable. Coverage and its
  limits are measured in §12.4.
- **The generic axis is not always the dtype: `Any` is generic over the *rank* and the *dimension*
  too.** A helper pinned to either breeds copies exactly as a precision-pinned one does. Two merged
  instances: a 5x5 and a 6x6 Householder normal-equation solve differed only in a
  `for k in range(N)` singularity test — because **a matrix has no readable `.shape` in kernel
  scope** (`r.shape[0]` is a `WarpCodegenAttributeError` at parse time on Warp 1.16) — and the
  rank-free spelling is a reduction over the diagonal,
  `wp.min(wp.abs(wp.get_diag(r))) < tol`, now `linalg.solve_normal_equations`; and a Cramer's-rule
  barycentric solve lived once per *dimension* (`wp.vec2`, `wp.vec3`) although `wp.length_sq` and
  `wp.dot` say nothing about the ambient dimension. **When two bodies differ only in a size, look
  for the one statement that names it and ask whether Warp has a reduction for it.**

### 1.3 Casts and conversions

- Cast `wp.tid()` explicitly when used as an array index: `f = wp.int32(wp.tid())`. **`wp.int32` /
  `wp.float32` are the tree's only cast spelling** — never the bare `int(...)` / `float(...)`
  builtins (check 16 rejects them inside a kernel or `@wp.func` body). They are the same builtins
  under a different name (`int(x)` compiles only because Warp writes an unconditional
  `#define int(x) cast_int(x)` into every module header), with one difference that matters:
  **`float(...)` is a hard compile error inside a `wp.Float`-generic function** — `total /
  float(count)` fails to parse with `Input types must be the same, got ['float64', 'float32']`
  rather than silently narrowing. Where the enclosing function is or could be generic, the spelling
  is `type(x)(...)` (the `kernels/predicates.py` convention). Measured detail in §12.5.
- **A cast to the type a value already has is noise; delete it.** No cast is load-bearing: a bare
  `wp.tid()` passes unchanged to a `wp.int32` parameter, to a `wp.Scalar` generic and into a
  kernel-scope slice (verified on Warp 1.16). So `wp.int32(offsets[i])` on a `wp.array[wp.int32]`,
  `wp.int32(n_faces)` on a `wp.int32` argument, and `wp.int32(f)` three lines under
  `f = wp.int32(wp.tid())` all say nothing.
- **The tid cast is the one exception the tree keeps**, because it is the *declarative* one: it
  names the type of the index the whole kernel is written against. **Check 22** enforces it — no
  bare single-index `wp.tid()`. It reads single-`Name` assignment targets only: a multi-index
  `i, j = wp.tid()` cannot carry a cast, and a scan keying on the *call* instead would report 61
  correct sites as defects and be switched off by the first person it annoyed
  (`test_bare_tid_scan_ignores_multi_index_unpacks` pins that).
- A cast of a bare **literal** is never redundant — `wp.int32(0)` is a mutable Warp dynamic variable
  where `0` is a compile-time constant that freezes the enclosing loop (§1.6).
- **A constructor of the type a value already has is the same noise, and the cast scan does not see
  it.** `wp.vec3(*(vertices[face[1]] - vertices[face[0]]))` splats a `wp.vec3` and rebuilds it; the
  difference of two `wp.vec3`s is already a `wp.vec3`. Check 16 classifies *casts*, so a
  `wp.vecN(*(...))` / `wp.matNM(*(...))` splat-and-reconstruct survives it — three such sites
  outlived the pass that deleted 184 redundant casts, in `triangles.triangle_edges` (since fixed —
  that helper now returns the differences directly, and a re-scan of `kernels/` finds no surviving
  `wp.vecN(*(...))` / `wp.matNM(*(...))` splat; the nine remaining hits are Python-scope
  constructions from host sequences, which are not this defect). When deleting
  one class of no-op conversion, grep the constructor spelling too, and read the operand's type
  rather than trusting the scan's silence.
- `wp.cast(expr, TargetType)` is an explicit conversion **only between types of the same size** — it
  is a bit reinterpretation, and a width change fails at *NVRTC* time with
  `static assertion failed with "source and destination must have the same size"`, not at Python
  time. `wp.cast(cross(a, b), wp.vec3)` is fine; `wp.cast(x_f64, wp.float32)` does not compile.
  Widening or narrowing a scalar is the **constructor**: `wp.float32(x)`, `wp.float64(i)`.
  Whole-array conversion at Python scope is a different API — §3.6.

### 1.4 Kernel-scope restrictions

Not supported inside `@wp.kernel` / `@wp.func`:

- Lambdas, list comprehensions, sets, dicts, `list.append()`, `eval()`, recursion, exceptions.
- Python tuples for initialization — use typed constructors: `wp.vec3(1.0, 2.0, 3.0)`.
- For small fixed-size collections use vector types; for larger ones `wp.zeros(shape=N, dtype=T)`
  (stack-allocated) — but see §2.9, which measures that as a **2x loss** against
  `wp.types.vector(length=N)`.
- **A tuple cannot be subscripted by a runtime index; a vector can.** The `tuple[T, T, T]` a
  `@wp.func` returns unpacks into names and nothing more, so `corner[k]` for a loop variable `k`
  does not compile — which is why a handful of kernels reach for the flat slice
  `faces[f * 3 : (f + 1) * 3]`. The spelling that keeps both is `wp.vec3i(*corner_triple(faces, f))`,
  which accepts a runtime `[]` (verified on Warp 1.16) and takes no sub-array view. The rule is
  **not** "no slices" but *no slice where an index form exists* — `holes.directed_edge_opposites`
  reintroduced one and it was the legitimate case.
- **Variable scope inside conditionals differs from CPython**: a variable defined only inside an
  `if` branch is accessible afterwards in Warp but uninitialized if the branch was not taken —
  always initialize before branching.
- `wp.asin()` / `wp.acos()` auto-clamp to [-1, 1]; an explicit `wp.clamp` before them is redundant
  but harmless.

### 1.5 Arithmetic and conditional spellings

- **`%` follows C++11 semantics** (sign of result = sign of dividend), not Python's.
- **`//` truncates toward zero like `/`, and on integers the two operators are the same
  operation.** Measured on Warp 1.16: `[-8, -7, -1, 0, 1, 7, 8] ÷ 3` gives `[-2, -2, 0, 0, 0, 2, 2]`
  for both spellings, where CPython's `//` gives `[-3, -3, -1, 0, 0, 2, 2]`. Consistent with the `%`
  rule (`-8 % 3 == -2`, and `-2 * 3 + (-2) == -8`).
- **Spell integer division `//`** (check 17). `/` on two `int32`s reads as real division and only
  truncates because the operands happen to be integers, so a reader has to recover the types before
  knowing what the line does. Every dividend in this package is a non-negative index, where the two
  conventions coincide; the hazard is *porting* a line with a negative dividend between host Python
  and kernel scope, which changes its answer silently. The scan types an operand only **by
  declaration** — an annotated `wp.int*` / `wp.uint*` / `wp.Int` parameter, an element of an array
  whose annotated dtype is one of those, a module-level integer `wp.constant`, an integer literal, a
  `.shape[...]`, `wp.tid()`, an integer constructor, or an integer-preserving expression over those
  — because a scan that misfires on float division gets switched off by the first person it annoys.
- **Spell the remainder `%`, not `i - (i // stride) * stride`.** The long form is the same operation
  but reads as though it is *avoiding* `%` for a reason a reader then goes looking for.
  `conjugate_gradient` wrote it long-hand where the sibling `multigrid` three files over wrote
  `t % stride`; there is no such reason.
- **A conditional value is `wp.where(cond, a, b)`, not a Python ternary** (check 20). 33+ sites in
  16 kernel modules. A ternary compiles to the same code — Warp lowers `ast.IfExp` exactly as it
  lowers `wp.where` — so like checks 16, 17 and 18 this is legibility, and nothing but a scan holds
  the line. Note `wp.where` eagerly evaluates both arms where a ternary short-circuits; every
  kernel-scope ternary found to date has both arms already evaluated (an argsort output, an index),
  so a future site where they are not needs a decision rather than a mechanical conversion.

Checks 16, 17, 18, 20 and 22 are **one family**, and the family is the point: every one of those
spellings is *legal* and generates identical code, so the defect is invisible to the compiler and to
the suite, and nothing but a scan holds the line.

### 1.6 Mutable loop accumulators

Inside `@wp.kernel` / `@wp.func`, a variable initialized to a **bare numeric literal**
(`count = 0.0`, `n = 0`) is a **compile-time constant** and cannot be mutated inside a dynamic
`for` / `while` loop. Warp raises `WarpCodegenError: "Error mutating a constant count inside a
dynamic loop…"` — or, in some module contexts, **silently keeps the initial value**, so `mean /
count` divides by zero and produces `inf` with no error.

Declare mutable accumulators with the typed constructor: `count = wp.float32(0.0)`,
`n = wp.int32(0)`. (The `float(0.0)` / `int(0)` spellings work identically — §12.5 — but §1.3's cast
rule means the tree writes `wp.float32` / `wp.int32`.)

**Never let a ruff auto-fix strip the constructor off a kernel-scope accumulator** — it introduces a
real bug. Where the legacy `int(0)` spelling survives, **two rules fire, not one**: `float(0.0)`
trips only `UP018`, but `int(0)` also trips `RUF046` and `# noqa: UP018` alone does not suppress it
— `ruff check --fix` silently rewrites it to `0` and the kernel then fails to compile on the next
launch.

---

## 2. Kernel architecture and launches

### 2.1 Launch discipline

- `wp.launch(kernel=..., dim=..., inputs=[...], device=...)`. **Always forward the `device` from the
  input arrays** — this is a memory-safety rule, not tidiness (§3.9). Check 15 scans every launch
  site; all 476 launches in `triwarp/` name a device.
- Array slicing is supported inside kernels: `faces[f * 3 : (f + 1) * 3]` produces a sub-array view
  (but see §1.4 — prefer an index form where one exists).
- **Prefix output argument names with `out_` and put them at the end of the signature**, after all
  inputs (check 13). Two exemption classes, both carried as `_KERNEL_OUTPUT_ALLOWLIST`:
  **in-place** arguments, where the same buffer is input and result
  (`sort_rows_insertion(data)`, the hole-filling DP tables) — an `out_` prefix would misread as
  write-only; and **scratch / persistent-state** buffers, caller-allocated working memory carried
  across launches (cursors, stacks, open-addressing tables, `ball_pivoting`'s front) — neither an
  input nor the answer, so name them for what they hold (`cursor`, `front_out`, `new_src`). A
  read-only input must never wear the `out_` prefix, even when the buffer was a *producer* kernel's
  output — parameter names describe the argument's role in *this* kernel.
- Derive the face count as `f = faces.shape[0] // 3` from the flat face index array.

### 2.2 A lane-parallel kernel strides by `wp.block_dim()`, or it stays lane-free

A kernel whose lanes cooperate — a `wp.tile_sum` / `tile_min` / `tile_max` over one value per lane,
a `wp.tile_bvh_query_aabb` walk — is correct on **both** devices exactly when its lanes partition a
sequence **the block already owns**, with the stride taken from `wp.block_dim()`. Never from a
kernel argument, never from a module constant.

`wp.launch_tiled` runs exactly **one lane per block on the CPU device** through Warp 1.17 whatever
`block_dim=` is passed, and `wp.block_dim()` reads `1` there — so that single lane walks the whole
sequence and the one-element tile it reduces holds the right answer. **A one-element tile is not the
bug.** Measured on one 1 000-element sum whose stride came from an `n_slices` argument instead:

| device | `n_slices` | `block_dim` | result | expected |
|---|---|---|---|---|
| cpu | 64 | 64 or 256 | **16.0** | 1000.0 (short by exactly the stride — one lane walked 1/64) |
| cuda:0 | 64 | 64 | 1000.0 | 1000.0 |
| cuda:0 | 64 | 256 | **3616.0** | 1000.0 (lanes 64-255 re-walk what 0-63 counted) |

So the arg-strided form is wrong on **both** devices, and its correctness on CUDA silently depends
on a *wrapper* passing `n_slices == block_dim`, which no signature expresses.

**Do not read a CPU failure of a tiled kernel as "tiles do not work on the CPU backend."** That
reading cost this package four kernels converted *away* from tile reductions and four comments
stating the prohibition in general terms, all of which the block-per-item kernels then contradicted
correctly. Full platform detail and the probe that distinguishes the two cases: §12.2.

Where the lanes would partition the **outer** work the grid is over — a whole-array reduction with
no per-item dimension, `measures.centroid_tiled`, `metrics.chamfer_*_tiled` — there is no
block-owned sequence and no `wp.block_dim()` to take. Those kernels must either keep a *constant*
stride and be launched on CUDA only, with a lane-free `_sliced` sibling for the CPU (the
`_device.prefers_tiled_reduction` pair), or stay lane-free on both. Both are correct; a single
kernel is not available. **Rule of thumb: a strided loop is one token from portable; a block-index
partition is a rewrite.**

Read `kernels/visibility.py::obscurance` (one block per point, lanes over its ray bundle, measured
3.2-11.8x against one thread per point) for the first form and `kernels/measures.py::centroid_tiled`
for the second.

**A lane-strided reduction's tuning variable is the fold width, not the lane redundancy.** Launch it
`dim=kernel_reduce.blocks_1d(n)` so each block owns `ITEMS_PER_BLOCK_1D` elements, never
`dim=n / TILE_1D`: what reaches a constant-slot accumulator is *one atomic per block*, so the block
count is the contended quantity and the fold is what pays for the `wp.tile_sum` calls. Two kernels
that already had one atomic per block, and looked finished for that reason, measured **4.1-10.7x**
from the fold alone — while removing their 64-fold redundant lane arithmetic, the thing that *looks*
wasteful, was worth 1.02-1.13x. Numbers, the wide-accumulator caveat and the commit spelling
(`wp.atomic_add` reads trailing indices as array dimensions, so assemble the summed vector or matrix
first): §13.2.

### 2.3 Block-per-item is an occupancy trade, not a shape choice

**Eligibility is an occupancy question, and getting it backwards costs 2-8x.** Having an outer
per-item dimension is necessary and not sufficient: the block-per-item form pays exactly when that
outer dimension *alone* would starve the device. `obscurance` qualified because it launched
`dim = n_points` — 8 171 threads, under 3 % of an RTX 5090 — with no second dimension at all. A
kernel that already carries a **slice dimension** does not qualify, because that dimension is what
fills the device and collapsing it into `block_dim` lanes throws the occupancy away.

Measured on `points.hull_support_extremes`, converted both ways:

| points | grid today | block-per-direction |
|---|---|---|
| 5 000 | 13 x 40 threads | **2.3x faster** (the grid was starved) |
| 200 000 | 13 x 1 563 threads | **0.12-0.60x — a 2-8x loss** (13 blocks on 170 SMs) |

`proximity.winding_number_tiled` shows the same trend from the other end: 2.16x at 1 280 faces and
4 096 queries, 1.19x at 5 120 faces, **1.01x** at 65 536 queries — a gain that shrinks to nothing as
the grid fills, which §9 calls a decline rather than a small win.

**Before proposing this rewrite, compute the launch's *current* thread count**, and measure at the
large end first — the small end shows a flattering 2x. Four kernels in this tree keep the
arg-strided form deliberately and each says so with its number. The full win/loss table is §14.1.

### 2.4 Fusing a kernel: extract the shared part as a `@wp.func` in the same commit

Fusing two launches into one is a standard and welcome optimization here — it removes a launch, a
round trip through global memory, and often an intermediate buffer. But a fused kernel is written by
*copying* the prologue of the kernel it absorbs, and the copy is what survives. **A fusion is not
done when it is faster; it is done when the code it duplicated has a name.**

Whenever a kernel is fused, split, specialised, or given a second variant (tiled/serial,
float32/float64, one accelerator/another), the same commit must:

1. **Name the shared run.** Any consecutive statement run the new kernel shares with the one it came
   from — a corner-index load, a window computation, an emit protocol, a guard sequence — becomes
   one `@wp.func` that both call. `@wp.func` calls are inlined at codegen, so this costs nothing at
   runtime; it is free structurally and the only reason not to do it is that nobody looked.
2. **Put it where the *quantity* lives, not where the fusion happened.** A general geometric
   predicate goes in `kernels/predicates.py`, a per-face quantity in `kernels/triangles.py`, an
   index/sort/search helper in `kernels/array.py`, a scatter in `kernels/scatter.py` (§3.1 names
   these four and why). A shared helper left in the algorithm module is how `triangle_aabb`,
   `triangle_double_area` and `circumcircle_diameter` ended up being reached by unrelated modules
   importing an algorithm to get at geometry.
3. **Say in a comment what the two kernels still differ by**, so the next reader can tell a real
   variant from a stale copy. Where the difference is a *parameter* rather than an algorithm, prefer
   one kernel with a warp-uniform int selector (§2.7's `wp.Function`-as-argument restriction and the
   `ACCEL_HASHGRID` / `ACCEL_BVH` pattern in `kernels/neighbors.py`); where it is genuinely two
   algorithms, keep two kernels and have each name the other.

**Merge on identity of meaning, not identity of tokens.** Two bodies that agree because they compute
the same quantity are one function; two bodies that agree because a one-line kernel has only one
shape are two functions and the duplicate scan's hit is noise. `remesh.compute_midpoints` and
`triangles.face_centroids` normalise identically and must stay apart.

**A comment that names the other copy is the finding, not the fix.** *"Feature handling mirrors
`collapse_candidates`"*, *"as in X"* — each of those was written by an author who had already seen
the duplication and answered it with prose. A cross-reference is a claim that **only a shared
`@wp.func` can keep true**. So when writing "same as X" / "mirrors X" about *code* rather than about
a reason, extract instead; and when reading one, treat it as an unfactored duplicate that has been
located for you. **Re-count the run from the code, not from the comment** — one "same two lines" was
seven statements, and the prose had been factored where the code had not.

**A clean duplicate scan is not a clean file.** Every scan this package has used keys on α-renamed
statement *text*, so two bodies expressing the same decision through different control flow — a
`reject` flag against an early `return`, a position against a boolean — score as unrelated. The best
find of the fourth `kernels/` pass was exactly that shape and came from reading a closed item:
`remesh.collapse_candidates` and `quadric_collapse_candidates` hold one 4-way feature-collapse rule
twice, one through a flag and one through returns. **A duplicated *decision rule* is a live
correctness hazard where a duplicated arithmetic run is only noise**, so it outranks the longer runs
a scan does find: extract it as a `@wp.func` returning the classification (a sentinel for "reject",
the `_resolve_flip_quad_guarded` convention) and let each caller map the free branch onto its own
answer.

**Factor the family, not the pair — a helper pinned to one rank or one precision breeds the copies
it was meant to prevent.** Two measured instances: `corner_triple` landed for flat 3-stride buffers
and reached 16 callers while its rank-2 sibling (`arr[row, 0..2]`) stayed nameless at eight sites;
and `laplacian.squared_edge_lengths`, hardcoded `wp.vec3` / `float32`, could not be reached by the
three `float64` sites in `energies.py`, which open-coded it instead. When extracting, write the
generic form §1.2 asks for and check for the *other* rank and the *other* precision before declaring
the run named. A general per-triangle quantity left in an algorithm module is now a **three-time**
defect, and the tell is a kernel module importing an *algorithm* to reach *geometry*.

**One predicate, one spelling per module — this is a correctness rule, not a style one.**
`wp.length(d) < r` and `wp.length_sq(d) < r * r` are **not the same predicate in float32**:
measured, 10 rows of 200k disagree at the boundary. So a module that tests the same rule both ways
can accept a candidate in one kernel and reject it in another — `ball_pivoting` tested one
clustering rule as `wp.length(...) < min_cluster` in `seed_triangles` and as
`wp.length_sq(...) < min_cluster_sq` forty lines later. Pick one spelling per predicate and say
which; expect the speed to be flat (`neighbors` measured 0.997-1.003x — §13.2) and keep the number
either way.

**A green suite does not prove a `@wp.func` extraction was behaviour-neutral.** `@wp.func` calls
inline at codegen, so an extraction that reorders an expression's evaluation changes `float32`
results without moving any comparison asserted at `1e-5`. The evidence is **reading the diff**, not
the suite. Two consequences for the gate: prefer to check the *decision* arrays a kernel writes over
the positions it produces where positions drift on their own (§16.4), and where a reference's answer
is a combinatorial object — a triangulation, a face buffer — gate on that rather than on a
tolerance.

Watch the signature while fusing: a fused kernel inherits the union of two argument lists, and a
launch argument costs ~1.0 µs of host time (§2.8, §13.1).

### 2.5 A generic kernel registers its overloads at import (`wp.overload`)

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

- **The chain is order-dependent**, which makes it a developer-loop tax rather than a one-time cost:
  a caller reaching the dtypes in a different order walks links that were never compiled, so
  changing which tests you select re-pays it from scratch.
- **Register what the wrapper's dispatch can reach, not every dtype the template admits** (§4.2, no
  speculative generality): an unused overload is compile time paid on every rebuild. Derive the set
  from the wrapper — a public `dtype=` keyword documented "float32 or float64", the key dtypes
  `sortable_dtype` maps onto, a docstring naming its own admissible dtypes — and say so in a
  comment. Where two generic arguments are independent (`laplacian.cotmatrix_triplets`' entry
  precision and matrix precision), it is a genuine cross product, not a diagonal.
- **A diagonal registration where the wrapper admits a cross product is the expensive half of this
  rule, and it now has a clock reading.** `energies.crouzeix_raviart_cotmatrix_triplets` was
  registered `for dtype in _MATRIX_DTYPES` with one loop variable serving both its `cot_entries`
  and its value precision, while the public
  `crouzeix_raviart_cotmatrix(cot_entries: twt.Array2dFloat, dtype: type)` takes the two
  *independently* and the kernel body casts one to the other. Measured on an RTX 5090, Warp 1.17,
  with the diagonal path warmed first: the first `float64`-entries / `float32`-matrix call logged
  `Module hash changed, recompiling: triwarp.kernels.energies` and took **80.3 s**, returning the
  right answer — §2.5's failure mode with a number on it. The nested loop takes the same call to
  **1.47 ms**. So when a wrapper exposes two dtype knobs, check whether the registration crosses
  them; `laplacian.py`'s sibling did and says so in a comment, which is the model.
- **Registration is not compilation.** `wp.overload` builds the overload's `Adjoint` and nothing
  else, so this costs milliseconds of import and nothing on a process that never launches the
  kernel. Do **not** `wp.load_module` / `wp.force_load` at import — that *would* compile eagerly.
- **`block_dim` forks the hash independently of dtype and is normally left alone.** Its values are
  fixed by the package's own launch code (a bounded two or three: a tiled launch's `block_dim`,
  Warp's 256 default, and 1 on CPU), so it is not a chain that grows with call order. Collapsing
  them is a perf change and needs §9's measurement — it was measured for `reduce` and declined.
- `tests/test_api_conventions.py::test_generic_kernels_register_their_overloads` fails when a
  generic kernel has **no** overload registered. Nothing checks that a registered dtype *set* is
  complete, and nothing cheaply can. **A missing dtype is diagnosed from the clock, not from a
  failing assert** — see §15.1.
- **`wp.map` has the same chain**, keyed by the unqualified op name and forked per *signature*, and
  it needs the same treatment — §3.5 (check 23) and §12.6.
- **And the registration pays twice: keep the `wp.Kernel` `wp.overload` returns, and launch through
  it.** Registering fixes the *rebuild*; it does nothing about the ~12 µs of `infer_argument_types`
  every generic launch runs before it can even look the overload up (§13.1, measured 2.17x on one
  launch and 1.09-1.58x end to end across 20 wrappers). So a module's registration builds a
  dtype-keyed [`OverloadTable`][triwarp.kernels.array.OverloadTable] and the wrapper writes
  `kernel_laplacian.COTMATRIX_TRIPLETS[cot_entries.dtype, dtype]` rather than naming the generic
  kernel. Nothing extra is compiled — the overloads already existed — and the kernel source stays
  generic, so §1.2 is untouched. **All 42 generic launch sites in the tree are converted**; a new
  generic kernel adds a table rather than a bare `wp.overload` call. A dtype the table lacks raises
  a `KeyError` naming the kernel, which is this section's failure mode made visible instead of
  costing 80 s of silent rebuild. `kernels/reduce.py` goes one step further and has **no** generic
  kernels at all: its factories always could bake the dtype in, and its own `_reduce_1d_tiled`
  docstring had said so since it was written — the 34 that stayed at the `wp.Scalar` default were
  simply never revisited. They are now `KernelTable`s of concrete instantiations, compiling exactly
  the 106 kernels the registration used to create.

### 2.6 In-place `@wp.func` parameters (`wp.ref[T]`)

`@wp.func` helpers may declare `wp.ref[T]` parameters to mutate caller-owned storage (locals, array
elements, struct fields) — use for multi-value updates like argmin/minmax/swap helpers (see
`update_argmin` in `kernels/array.py`).

- **Constraint:** any kernel calling a `wp.ref` helper must be decorated
  `@wp.kernel(enable_backward=False)` — the per-kernel flag specifically; a module-level
  `wp.set_module_options({"enable_backward": False})` is NOT consulted at kernel-parse time and the
  module still fails to compile. This is inference-neutral (no forward-runtime effect).
- **Never use `wp.ref` in `triwarp/kernels/metrics.py`** — the chamfer kernels are differentiated
  via `wp.Tape`.
- **`wp.ref[T]` requires a concrete `T`** — generics do not instantiate inside it. Details and the
  knock-on consequences: §12.3.

### 2.7 Function-valued parameters and kernel factories

- A `@wp.func` may take `fn: wp.Function` parameters; the target is bound at **compile time** per
  call site (user `@wp.func`s and simple builtins like `wp.min` are valid targets; tile intrinsics,
  variadic and LTO builtins are not).
- **`wp.launch` can NOT pass a `wp.Function` as a kernel argument.** For runtime selection, pass an
  **int/enum kernel argument** and branch over `wp.Function` targets inside a dispatch `@wp.func`
  (warp-uniform branch, one compiled module; see `registration.robust_weight`).
- Builtins with no Python-scope handle (e.g. tile intrinsics) can still parameterize kernel
  factories by pulling the concrete `Function` object from
  `warp._src.context.builtin_functions["tile_max"]` and closure-capturing it — captured builtins
  emit inline at codegen and template on the tile dtype (see `triwarp/kernels/reduce.py`). Give each
  factory instantiation a unique kernel `name`.
- **Name it with `wp.kernel(f, name=...)`, not by mutating `__name__` / `__qualname__`.** Warp 1.17
  added a `name=` parameter to the decorator (NVIDIA/warp#1561) that sets both the registration key
  and the base of the generated native entry point; it must be a valid C++ identifier. That is one
  line instead of two assignments the reader has to recognise as a naming convention, and it is the
  documented route to ahead-of-time compilation. `kernels/reduce.py` uses it at all nine factories.
- **A factory is also how you avoid a generic kernel's dispatch cost.** A generic kernel pays
  ~12 µs of host-side overload resolution on *every* launch — the same for `Any`, `wp.Float` and
  `wp.Scalar`, and more when several parameters are generic (§13.1). A factory that writes
  `__annotations__` after defining the body produces concrete kernels from one source:
  ```python
  def _factory(name, dtype):
      def _k(values: wp.array[wp.Scalar], out: wp.array[wp.Scalar]) -> None: ...
      _k.__annotations__["values"] = wp.array[dtype]
      _k.__annotations__["out"] = wp.array[dtype]
      return wp.kernel(_k, name=name)
  ```
  **A factory whose `dtype` parameter has a generic default is a factory nobody specialised**, which
  is exactly how `kernels/reduce.py` ended up with 34 generic kernels behind a template that could
  always have baked the dtype in. If a factory takes a dtype, give it no default.
- **Where the kernel body should stay generic, the table form in §2.5 gets the same launch cost for
  no restructuring at all** — `wp.overload` already returns the concrete kernel. Reach for a factory
  when the *body* needs specialising (an axis, a storage class, a captured builtin) and for the
  table when only the dtype does.
- **`if wp.static(flag):` prunes the untaken branch even when the two branches bind locals of
  *different types*** (probed: `wp.types.vector(length=K)()` vs `wp.zeros(shape=K)`, and
  `values[i, k]` vs `values[k, i]`). That is what lets one body generate axis- or
  storage-specialised kernels with no runtime branch.

### 2.8 `@wp.struct` argument bundles

**A `wp.launch` argument costs ~1.0 µs of host time, linearly, on both CUDA and CPU** (§13.1). §9's
flat "~32 µs per launch" is the *mean* kernel's launch (mean argument count 5.1), not a constant.

If a kernel is launched inside a Python loop and carries a dozen or more arguments, bundle the
invariant tables into a `@wp.struct` built **once** in the wrapper — measured 43 → 18 µs per launch
for a 25-argument kernel, a flat saving at every `dim`, with subscript-style field annotations
(`a: wp.array[wp.int32]`) and 2.6 µs per bundle construction. Rebuilding per launch gives the win
back.

- **Width alone does not qualify a kernel — the launch count around it does.** Count the launches
  per call and divide. `remesh.objective_flip_candidates` is the same 14 arguments and is declined:
  a flip *round* rebuilds the whole face adjacency around its one launch, `flip_to_delaunay`
  converges in **2** rounds at 514 µs each, so nine bundleable arguments bound the saving at 18 µs —
  **1.75 %** of the call.
- **Do not bundle a kernel whose launch is already captured** — a replayed launch costs ~1.17 µs
  whatever its argument list (§13.1), so `remesh._issue_pass` and the multigrid V-cycle inside
  `wp.capture_while` gain nothing.
- Do **not** open a tree-wide bundling pass: only 18 of 440 kernels take ≥12 arguments, and §4.2's
  "no speculative generality" applies to argument bundles too.
- **A new Warp construct can switch off a static check written before it.** Moving buffers into a
  `@wp.struct` removed them from check 13's view entirely, because the check resolved store targets
  to a bare `ast.Name` — see §4.5.

Two examples worth reading before writing a third: `holes._fill_dp` launches a 16-argument kernel
once per span, and `reconstruction._bpa_wave` launches a 27-argument one per wave. Measured gains
and the graph-capture comparison: §13.1, §14.3.

### 2.9 Per-thread row storage

**`wp.zeros(shape=K, dtype=T)` inside a kernel is not per-thread fast storage.** Warp wraps the
stack buffer in an `array_t`, so every access goes through a pointer and nvcc never promotes it to
registers — measured a **2x loss** against a global-memory row (9.87 ms against 4.59 at k=32).

Use **`wp.types.vector(length=K, dtype=T)`** (the `vec5d` pattern in `kernels/curvature.py`):
genuinely register-resident, 1.09x at k=1, 1.5x at k=7, 1.9x at k=16, 2.9x at k=30, **7.8x at
k=64**, collapsing to 1.66x at 96 where it spills (the row costs `2 * K` registers).

**Keeping it in registers constrains the code shape:**

- A vector must never be passed to a `@wp.func` (that takes its address and spills it) — the insert
  has to be inline, duplicated per call site.
- **Any runtime index spills the whole vector**: read the k-th element with an unrolled
  `for slot in range(K): if slot == k - 1:` compare, not `row[k - 1]`. `K` is closure-captured in a
  kernel factory so `range(K)` unrolls with literal indices; one kernel per bucket, unique `name`
  each (`KNN_ROW_BUCKETS = (1, 4, 8, 16, 32, 64)`).
- **A register insertion sort needs an explicit `placed` flag.** Carrying the displaced element down
  with `if carry < row[slot]` breaks the chain on a run of **equal** distances already in the row,
  dropping a neighbour — one differing row in 20 000 at k=32 with the distance array still
  byte-identical, i.e. invisible to a distance comparison.

Cost of admission: `kernels/neighbors.py`'s cold-cache compile went 3.9 s → ~12 s for 12 generated
kernels. Related: §14.4 (tile solves are a different trade), §13.1.

---

## 3. Python-scope wrappers

### 3.1 Module layout and imports

- Every kernel lives in a `kernels/` sub-module, imported with an alias:
  `from triwarp.kernels import triangles as kernel_triangles`.
- **A top-level kernel module is named exactly for the public module it backs**:
  `triwarp/kernels/<module>.py` ↔ `triwarp/<module>.py`, one-to-one (check 7). The only admissible
  exceptions are the kernel-side libraries that back no single public module —
  `kernels/predicates.py` (geometric `@wp.func` predicates) and `kernels/scatter.py` (scatter /
  accumulate kernels). Sub-packages (`kernels/algorithms/`) mirror a folder rather than a module and
  are exempt. Adding a kernel module with no public counterpart, or a public module whose kernels
  live under another name, is a defect — fix the name, do not document the exception.
- **Backing a public module and serving as a shared library are not exclusive**, and four modules do
  both jobs. Being imported across the tree is not a violation and needs no exception; it is what
  these four are *for*:

  | Module | Holds | Importers (measured) |
  |---|---|---|
  | `kernels/array.py` | index/sort/cast/search `@wp.func`s (`sort3`, `cross2`, `to_vec3d`, `binary_search_index`) | 25 kernel modules, 15 wrappers |
  | `kernels/predicates.py` | precision-generic geometric predicates | 17 kernel modules |
  | `kernels/triangles.py` | per-face corner/quality/gradient `@wp.func`s | 15 kernel modules, 2 wrappers |
  | `kernels/scatter.py` | scatter/accumulate kernels | 11 wrappers |

  What *is* a defect is placement: a general geometric predicate living in a module that owns an
  **algorithm**, so that unrelated modules import the algorithm to reach the geometry.
  `triangle_aabb` sat in `kernels/intersection.py` and `triangle_double_area` /
  `circumcircle_diameter` in `kernels/holes.py` for exactly that reason; all three are now in
  `predicates.py`, generic over the scalar type. When a helper is reached from a second module, ask
  which of the four it belongs in before adding the import.
- **`triwarp/__init__.py` is lazy (PEP 562 `__getattr__`) and must stay that way.** `@wp.kernel`
  builds an `Adjoint` at *import* time, so an eager `__init__` cost 0.60 s and made
  `import triwarp as tw` a whole-package pull; see §16.2 for the measurement and the subprocess test
  that guards it.
- **Three searches in `kernels/array.py` are not interchangeable**, and picking wrong is silent:
  `binary_search_index` is `searchsorted(side="right")` (returns `slot + 1` on an exact hit),
  `binary_search_index_left` is `side="left"` (the exact slot for a present key — use this plus
  `index < n and values[index] == v` for a lookup), and `binary_search_sorted_contains` is
  membership only. Detail and the bug it caused: §16.5.

### 3.2 Typing (`triwarp.typing`)

Import once per Python wrapper module:

```python
import triwarp.typing as twt
```

Do **not** re-export typing symbols from `triwarp/__init__.py`; import `twt` where needed.

**Why not `wp.array2d` in wrappers?** At runtime every buffer is `warp.array`. `wp.array2d[dtype]`
in a Python signature is a static annotation helper; type checkers do not treat it like a real array
(missing `.shape`, bad assignability from `wp.empty`), and `isinstance(x, wp.array2d)` is always
`False`.

Use **`wp.array[dtype, Literal[ndim]]`** via the aliases:

| Alias | Meaning |
|-------|---------|
| `twt.Array2dInt32` | `(rows, cols)` `int32` |
| `twt.Array2dFloat32`, `twt.Array2dFloat64` | `(rows, cols)` `float32` / `float64` |
| `twt.Array1dInt32` | 1D `int32` |
| `twt.IntArray`, `twt.FloatArray`, `twt.ScalarArray` | 1D or 2D unions (e.g. `reduce.py`) |

Kernels in `triwarp/kernels/` keep `wp.array2d[dtype]` unchanged. Optional 2D arguments:
`edges_sorted_wp: twt.Array2dInt32 | None = None`.

**Runtime checks, not `isinstance`:**

- `twt.ensure_ndim(arr_wp, 2, dtype=wp.int32)` — validate rank and dtype on inputs.
- `twt.as_array2d(arr_wp, wp.int32)` — check then narrow the return type for Pyright. Overloaded on
  the `dtype` argument for `wp.int32` / `wp.float32` / `wp.float64`; `twt.as_array3d` covers
  `wp.float32` / `wp.bool` at rank 3.

### 3.3 Allocation and returns

- Python-scope wrappers accept `wp.array[T]` for 1D buffers; use `twt.Array2dInt32`,
  `twt.Array2dFloat32`, etc. for rank-2 results.
- For **2D** outputs allocate with `twt.empty_2d((rows, cols), wp.int32, device=...)` — the dtype is
  an argument, not part of the name. `twt.empty_3d` is the rank-3 counterpart.
- For **1D** outputs keep `wp.empty(n, dtype=..., device=input.device)` when all elements will be
  written by the kernel (avoid unnecessary zero-initialization).
- Return rank-2 buffers as `return twt.as_array2d(arr, wp.int32)` so callers get a checked,
  correctly typed value.
- **Always pass `device=` to every allocation** (`wp.zeros` / `empty` / `ones` / `full` / `array`) —
  check 10, and it scans all of `triwarp/` including `_*.py`. Without it the buffer lands on Warp's
  *current* device and the suite cannot see the difference, because a test runs with its arrays'
  device already current: `array.index_sparse`'s `wp.ones` was wrong for as long as it existed and
  every test passed.
- Empty mesh / no adjacency early return:
  ```python
  if n_faces == 0:
      return twt.empty_2d((0, 2), wp.int32, device=faces_wp.device)
  ```
- **Size buffers for their final use at allocation time.** Do not allocate-then-grow at Python
  scope. When a consumer needs an `n + 1` sentinel-terminated form, the *producer* allocates `n + 1`
  and hands back a view (`counts_to_offsets`); a helper whose only job is to patch up another
  function's output convention is a smell to be fixed at the producer.
- **Warp raises on a zero-length slice** (`RuntimeError: Invalid indexing in slice: 20:20:1`), so a
  trailing-mask `fill_` needs an `if stop > start` guard where a NumPy version silently no-opped.
- **A buffer whose initial value matters is *allocated holding it* — never `wp.empty` followed by
  `fill_` / `zero_`.** One statement instead of two, and no window in which the buffer holds
  garbage. `wp.zeros`, `wp.full(n, value, dtype=..., device=...)`, and at rank 2
  `twt.as_array2d(wp.full((rows, cols), value, ...), dtype)` — there is deliberately no
  `twt.full_2d`, because that spelling already appears at the one site that needs it and §4.2
  forbids the speculative helper. **The reason is legibility, not speed**: measured on an RTX 5090,
  Warp 1.17, 200 calls between two syncs, min of 7 — `wp.empty` plus `fill_` against `wp.full` is
  **0.96-1.00x** at 1 024 / 200 000 / 4 000 000 elements, rank 1 and rank 2 alike, so the two-call
  form is if anything a few tenths of a microsecond *cheaper* and the choice is entirely about how
  the code reads.
    - `wp.empty` stays correct — and is the rule — where **every** element is written before it is
      read (the bullet above). What this rule forbids is allocating uninitialized and then
      initializing.
    - **Where the branches initialize differently, allocate inside each branch.**
      `energies.laplacian_smoothness` held two `wp.empty` buffers above a three-way `method`
      branch that filled them, launched into them, or half-zeroed them; each branch now allocates
      what it will hold. A shared allocation above a branch that initializes reads as one thing
      and is as many things as there are branches.
    - **A *partial* write into a buffer another writer already filled is not this pattern.** Ten
      `arr[a:b].fill_(...)` sites carry a running state counter, a padded triplet index, or a
      mask whose head a kernel wrote; those stay. The scan that finds the real thing keys on a
      whole-buffer `fill_` / `zero_` on a name assigned from `wp.empty` / `twt.empty_*` within a
      few lines — measured 8 sites, all converted — and a second pass keying on a *slice* target
      found one more (`boundary._loop_owner_labels`' terminator), so run both.
    - There is **no** Python-scope scalar write to pair with `_device.read_scalar`, and asking for
      one is usually the wrong question: `arr[k] = v` raises `TypeError: 'array' object does not
      support item assignment` on both devices (Warp 1.17) and `arr[k : k + 1].fill_(v)` is
      already the primitive such a helper would wrap. The site that prompted the question wanted
      the *allocation* to carry the value instead, after which the write disappears.

### 3.4 Python-scope gather indexing (prefer over trivial gather kernels)

At **Python scope**, Warp supports **gather** with integer indexing: `view = src[indices]` yields a
`wp.indexedarray`. Materialize a dense `wp.array` with `wp.copy(dst, view)` when callers need
`.reshape()` or a guaranteed `wp.array` return type (see `triwarp/selection.py`'s face gather and
`triwarp/array.py`'s `isin`).

- **1D gather:** `vertices[indices]`, `lookup[elements]`.
- **2D index arrays:** Warp requires **1D** index arrays — flatten first (`elements.flatten()`),
  gather, `wp.copy`, then `.reshape(original_shape)`. `.flatten()` on rank-3/rank-4 arrays returns a
  **contiguous** rank-1 view and `reshape` round-trips fine, which is why `array.isin` accepts any
  rank instead of capping at 2. But `wp.array.flatten()` **raises** `RuntimeError` on a
  non-contiguous view rather than copying.
- **⚠️ The index array must be CONTIGUOUS.** Warp reads the index buffer as if contiguous and
  **silently ignores a view's stride** — no exception. `payload[edges[:, 0]]` returns the flattened
  buffer's leading entries (`[0, 10, 1, 11, …]`), not column 0. A contiguous *prefix* slice
  (`arr[:n]`) is safe; a column (`arr[:, k]`) or step slice (`arr[::2]`) is not — `wp.copy` it into
  a dense buffer first. Measurements: §12.1. This is why `kernels/edges.py:edge_lengths` stays a
  kernel. **When converting a gather, verify values, not just that it runs and is faster**: the
  corrupt version reads a contiguous prefix and is measurably *faster* than the correct one.
- **Indexed assignment** (`arr[indices] = value`) is **not** supported on `wp.array` at Python scope
  — keep a small kernel for scatter / mask marking (e.g. `mark_membership_mask` in
  `triwarp/kernels/array.py`).

Do **not** add custom per-element gather kernels when `[]` plus `wp.copy` suffices. Probe tests live
in `tests/test_*_indexing_probe.py` — `test_array_indexing_probe.py` pins the stride hazard
itself (still live on Warp 1.17), written as *"the gather returns the flat prefix"* rather than
*"not the column"* so that a Warp release honouring the stride fails it and this rule is revisited
rather than silently kept.

### 3.5 Elementwise ops at Python scope (`wp.map`)

Do **not** write a `@wp.kernel` whose body is only `out[i] = f(in[i], ...)`. Keep the op as a named
`@wp.func` in the `kernels/` module (or use a builtin like `wp.neg`, `wp.add`, `wp.div`,
`wp.normalize`) and call **`wp.map(op, *inputs, out=...)`** from the wrapper. The generated kernel is
cached (in-memory per process + Warp's on-disk cache) and its GPU time is identical to a
hand-written kernel; cached calls cost ~11 µs extra host-side Python.

- **Named `@wp.func` only, never lambdas**: the map cache is keyed by the *unqualified* function name
  plus input dtypes — two different ops with the same name would collide, and lambdas re-derive the
  function each call.
- **Always pass `out=`** so allocation stays with `wp.empty` / `twt.empty_*` in the wrapper. In-place
  is `out=<an input>`; multi-output funcs (`tuple[...]` return) take `out=[a, b]`.
- Scalars mix freely with arrays (`wp.map(is_long_edge, lengths, max_edge_f, out=mask)`); device is
  inferred from the array inputs. **But `wp.map` infers a bare Python `int` as `wp.int32`, not as the
  mapped func's declared dtype** — unlike `wp.launch`. A `bvh.id` handed to a `wp.uint64` parameter
  fails at codegen with `TypeError: Function <fn> does not support the provided argument types
  int32, ...`. Cast explicitly: `wp.map(fn, wp.uint64(bvh.id), ...)`.
- **Slice views** work as inputs and outputs: adjacent-element ops map over shifted views
  (`wp.map(segment_length, polyline[:-1], polyline[1:], out=lengths)`), and offset writes map into
  `out=dst[o : o + n]`. CSR row degrees: pass `offsets[:-1]` and `offsets[1:]`.
- **Python-scope gather composes**: `wp.map(pred, table[indices], out=mask)` maps over the
  `wp.indexedarray` view (see `repair.make_volume`).
- Inside **per-iteration wrapper loops**, hoist the kernel once with `wp.map(..., return_kernel=True)`
  and `wp.launch(kernel, dim, inputs=[...], outputs=[...])` in the loop (see `triwarp/smoothing.py`)
  — this removes the per-call Python overhead, and the overhead has a number: **a cached `wp.map`
  call costs 23.8-26.6 µs against 13.4-14.3 for the launch it wraps, 1.78-1.86x, ~11 µs** (Warp
  1.17, RTX 5090, 100 calls between two syncs, flat from 1 024 to 200 000 elements). It is the same
  host-side resolution a generic kernel pays (§13.1), so the two conversions look alike and are
  priced alike. `smoothing.filter_normals` ran two maps over 20 passes for ~0.44 ms of pure host
  time and hoisting them is most of its measured 1.48x. **Hoist where the loop body is cheap; a map
  inside a solve loop is a fraction of a percent** — `linalg._multigrid_hierarchy`, the ARAP and CG
  component loops and the remesh pass loops are all left alone deliberately.
- **An op reached at more than one call signature must have those signatures declared at import**
  (check 23) — the same rule and the same reason as `wp.overload` in §2.5. `wp.map` names its
  generated module `map_<unqualified op name>`, each distinct signature forks that module's hash, and
  a module's hash covers the kernels instantiated in it — so an op reached at three signatures builds
  its module three times, each build containing every kernel accumulated so far. Measured on
  Warp 1.17: `wp.mul` at float32 → vec3 → float64 in a fresh cache compiles 245 + 43 + 43 ms where
  the final hash alone is one 253 ms build, and over the tree's eight longest chains, cold cache
  1 365-1 485 → 647-665 ms (**2.1x**) and warm cache 114.9 → 97.1 ms (1.18x).
  **Declaration is not compilation** — `return_kernel=True` on a zero-length host array costs
  ~0.5 ms and reaches the final module directly. The tables live in `_declare_map_kernels()` at the
  bottom of the module that owns the op (`declare_map_signatures` in `kernels/array.py` carries them
  and the reasoning); the Warp *builtins* live in `kernels/array.py` because one generated module is
  shared across several wrappers and every declaration for it has to run before the first launch from
  any of them.
- **The fork axis is not only the dtype, which is the part that is not guessable.**
  `warp._src.utils.map` keys on `(is_array, type(input).__name__, dtype, ndim, broadcast_mask)` per
  input, where `broadcast_mask` is `tuple(d == 1 for d in shape)` — so a **length-1** array forks a
  module (12 ops fork on that axis *alone*, and it is the normal path for every
  reduction-into-a-scalar wrapper), as does an **`indexedarray`** from a Python-scope gather, as does
  the rank. Derive the table by instrumenting `wp.map` over a suite run and recording Warp's own key;
  check 23 fails when a module needs a table and has none, and the completeness gate is the load
  census (182 distinct `map_*` loads over 143 `(module, device, block_dim)` pairs before, **143 over
  143** after — 143 is the floor). Zero-length **CPU** arrays are enough to declare a CUDA overload:
  the module is keyed by dtype, not device.
- Never declare map kernels in `triwarp/__init__.py` (§3.1), and never via `wp.load_module` /
  `wp.force_load`, which compile eagerly.
- Still a real kernel: ops needing the thread index as *data* (`init_range`, `seed_orientation`),
  whole arrays as uniform arguments (binary-search tables), scatters, and multi-element/row-indexed
  outputs.

### 3.6 Dtype conversion at Python scope

`wp.cast(expr, TargetType)` is for **kernel / `@wp.func` scope** only — there is no `wp.cast` on
whole arrays at Python scope.

For element-wise dtype conversion of `wp.array` buffers at Python scope, allocate the destination and
call **`wp.utils.array_cast(src, dst)`** (same device, matching shape). Example: `wp.bool` →
`wp.int32` `0`/`1` flags for `wp.utils.array_scan` in `flatnonzero` — do **not** add a
`bool_to_int32` gather-style kernel.

### 3.7 Sparse: `BsrMatrix.nnz` is a stale capacity

**Never size a buffer, slice, or launch dim off `matrix.nnz`.** After `bsr_from_triplets` the `nnz`
field still holds the *triplet count it was handed*, duplicates included — so for any
duplicate-emitting build it is an upper bound, measured at 8400 against a true 4516 on a synthetic
Laplacian and **3.4x** (15 360 against 4 482) on `laplacian.cotmatrix`, which emits 12 triplets per
face. Use **`matrix.nnz_sync()`** (one host readback, ~0.1 ms) or read `offsets[nrow]`, which
`energies.k_harmonic` already does.

**And `nnz` is a *cache*, not a fixed field: `nnz_sync()` repairs it in place.** Measured —
`int(m.nnz)` reads 15 360, then `m.nnz_sync()` returns 4 482, and `int(m.nnz)` now reads 4 482 too;
nothing else syncs it (`bsr_mv`, `values.numpy()`, `offsets.numpy()` all leave it stale). So whether
a `.nnz` read is correct depends on whether unrelated earlier code happened to sync that matrix,
which makes the bug order-dependent and is a live trap **for the test as much as the code**: a guard
that measures the capacity and then hands the *same* matrix to the function under test has already
repaired it, and passes against the broken implementation. Build two operators — one to measure, one
to hand over (see `test_filter_laplacian_implicit_duplicate_built_operator`).

The failure is silent and it is not a Warp bug. Sizing a `triplet_buffers` allocation by `nnz` leaves
the tail `[nnz_sync(), nnz)` unwritten, and since those buffers are `wp.empty` (§3.3, deliberately)
the gap reaches the next `bsr_from_triplets` as **uninitialized triplets**. `bsr_from_triplets` drops
an out-of-range row/column index silently — verified for both `999999` and `-7`, no exception and no
CUDA fault — so most garbage vanishes and the answer looks right; the entries whose garbage index
happens to land in `[0, nrow)` accumulate a garbage value into a **real** entry. Measured on
`_build_implicit_system` with a `cotmatrix` operator and plausible indices left in the memory pool:
`‖values‖ = 1.1e13` against the correct `84.3`. **This is what the long-standing "`bsr_mm` is
nondeterministic on CUDA" claim really was** — `bsr_mm` is sound; do not reintroduce that
explanation.

Two corollaries. A matrix built by *duplicate-free* triplets (`laplacian.laplacian`,
`smoothing._edge_weight_matrix`) has `nnz == nnz_sync()`, which is why the default paths never showed
this — so a probe on the default operator proves nothing, and the check belongs on a
`cotmatrix`-shaped input. And where a triplet writer legitimately leaves slots unwritten (a
conditional emit, as in `dirichlet_system_triplets` / `laplacian_ls_triplets`), `wp.zeros` rather
than `triplet_buffers` is correct **for correctness** — but it is a **cost** defect: see §12.7, where
unwritten `(0, 0, 0.0)` triplets measured **31.8x**.

**This is a pattern rather than one API's quirk: a Warp object's `*_count` / `.nnz` field is a
*capacity* until proven otherwise.** `wp.Volume.get_voxel_count()` and `wp.volume_voxel_count` report
the grid's allocated capacity, not its active voxel count, so `triwarp.voxels` goes through
`Volume.get_active_stats().voxel_count` and says so at both sites. Before sizing a buffer, a slice or
a launch `dim` off any such field, probe it against a construction whose true count you know
(a duplicate-emitting triplet build, a sparse voxel set) and record the number where the rejection
lives. The failure mode is the same both times: the wrong reading is an *upper* bound, so nothing
raises and the tail is garbage.

More `warp.sparse` behaviour — `bsr_mm`'s structural superset, `bsr_compress`'s in-place return,
operands read from the stale `nnz` field — is in §12.7.

### 3.8 NumPy at Python scope is sanctioned; leaking it through the API is not

`warp-lang` carries an unconditional `Requires-Dist: numpy` and `import warp` loads it eagerly, and
`wp.array(list, dtype=...)` itself ends in `np.asarray`. So NumPy is present wherever triwarp runs,
it is a declared core dependency, and deleting `import numpy as np` from a wrapper shrinks nothing —
it only moves the same NumPy call into Warp, more slowly. **Do not open a "remove NumPy" pass**; 13
modules import it and that is correct.

Host-side metadata math (offset scans, launch dims, per-loop sizes, small candidate tables) and
host-*sequential* algorithms (patience sorting in `combine`, DP traceback in `holes`, `lexsort`
Delaunay in `reconstruction`, `argsort` + `searchsorted` chain linking in `intersection`, most of the
procedural mesh templates in `creation`) stay in NumPy: they are not device work, and porting them
buys Python loops.

**The exception is a template that is a closed-form parallel *map* whose output scales with a
resolution parameter** — no sequential dependence between elements, so a kernel buys no Python loop
and the host build is pure assembly plus an upload. `creation.icosphere` and `creation.grid` are both
in that class and both are kernels: `grid` was 78 % NumPy prologue at `count=512` and came out **33x
faster and bit-identical** (6.77 → 0.205 ms; 112x at `(1024, 1024)`), because a `float64` kernel
followed by the same `float32` store rounds the same way the host build did. Decide by whether the
elements depend on each other, not by which module the function lives in.

Three things are still defects:

- **A public signature or return that names `np.ndarray`**, which forces the dependency on the
  *caller*. Return `wp.mat33d` / `wp.vec3` (`measures.moments` returns the inertia tensor as
  `wp.mat33d` — `wp.mat33` would discard the `float64` digits the integrals exist to keep); annotate
  inputs `Sequence[Sequence[float]]` when the body is a duck-typed `np.asanyarray`, which is
  *widening*. The one sanctioned exception is `triwarp/io.py`, where meshio hands back `np.ndarray`
  unconditionally and NumPy-in is `mesh_from_numpy`'s entire purpose.
- **NumPy standing in for a Warp Python-scope equivalent that exists.** `wp.full`, `arr[k:].fill_()`,
  `wp.array([wp.mat44(...)])`, `wp.determinant`, `wp.inverse`, `wp.transpose` and `wp.svd3` all work
  at Python scope (verified on 1.16) and need no host buffer; `arr.list()[0]` gives a row-indexable
  `wp.mat44` from a `wp.array[wp.mat44]`. `math.pi` / `float("nan")` / `float("inf")` beat `np.pi` /
  `np.nan` / `np.inf`. One trap: **`wp.svd3` is not a substitute for `np.linalg.svd` of a non-square
  matrix** — `creation._align_vectors` takes the SVD of a `(3, 1)` for basis completion and its free
  rotation about the axis is a *gauge* the trimesh comparison pins element-wise.
- **NumPy reducing a full `.numpy()` readback** — `.min()`, `.max()`, `.any()`, `.sum(axis=0)` — is a
  §9 defect wearing NumPy's clothes: the whole array crossed the bus to produce one scalar. Use
  `triwarp.reduce` (or `wp.utils.array_sum`, which reduces a `wp.vec3d` array componentwise and so
  needs no kernel of its own), and check whether a kernel for it already exists before writing one —
  `holes._mean_rim_edge_length` was reading back the entire vertex buffer while `_loop_perimeters`,
  three hundred lines up in its own file, already computed the answer on the device. **Decide these
  on the CUDA measurement and accept the CPU regression** (§9), but keep the host path where the
  buffer never scales with the mesh, as `graph.bfs_multi_source`'s `k`-element source check does. The
  seven-site A/B and its four rejections are §14.5.

### 3.9 Device checks and the launch-device memory-safety rule

**Do not check that input arrays share the same device.** Manual
`if arr.device != device: raise ValueError(...)` guards are redundant, and §4.3 forbids a docstring
documenting a `ValueError` for arrays "on different devices". Omit them entirely.

**But do not believe that `wp.launch` raises on a device mismatch — it has not since Warp 1.14.**
That release removed the unconditional same-device check (`NVIDIA/warp` GH-1461) so that
hardware-coherent launches would be legal, and the default `wp.config.launch_array_access_mode` is
`RELAXED`, which passes the pointers straight through and validates nothing. On this box the
consequences are asymmetric and both are silent:

- **CPU arrays, CUDA launch** (a launch that forgot `device=`, resolving to `cuda:0`): the GPU reads
  the host arrays over HMM (`is_cpu_memory_access_from_gpu_supported` is `True` here) and computes
  the **right answer** — then the launch is *asynchronous*, so when those host arrays are freed while
  the kernel is still running the heap is corrupted and the process aborts in `malloc` much later.
- **CUDA arrays, CPU launch**: immediate `SIGSEGV`, no Python exception (GH-1693).

Four consequences:

- **`tests/conftest.py` sets `LaunchArrayAccessMode.STRICT`**, the only mode that rejects a *genuine*
  cross-device argument; `CHECKED` validates addressability, which HMM genuinely provides, so it
  permits the launch and still corrupts. The full suite passes under `STRICT` — keep it that way.
- **`STRICT` alone would not have caught the bug that motivated it, which is why check 15 exists.**
  It only fires when an argument is *not* on the launch device, so on a CUDA run — this box, and CI —
  an omitted `device=` resolves to `cuda:0`, which *is* the arrays' device, and nothing is rejected.
  The corruption then waits for a CPU run. `test_launches_name_their_device` scans every launch site
  statically and is the half that sees it; the two guards cover different halves.
- **§2.1's "always forward the `device`" is a memory-safety rule.**
- **`.numpy()` is not a sync on a CPU array.** On a CUDA array it synchronizes; on a host array it is
  a zero-copy view, so "I read the result and it was correct" proves nothing about whether the kernel
  finished.

Full root-cause history, the free-while-in-flight measurements and the two bisection techniques that
found them: §12.1.

### 3.10 Readbacks

- **Budget the host-device syncs.** Every `.numpy()` / `int(<device value>)` readback in a wrapper
  carries a comment naming why it is unavoidable. When the caller can supply the bound the readback
  infers, expose it as a keyword (`face_adjacency(n_vertices=...)`,
  `hash_indices_rows(validate=False)`) **and pass it from every in-repo caller that knows it** — an
  escape hatch nothing uses is not an optimization.
- **A readback costs ~0.1 ms; an extra device pass costs 0.9-2.4 ms.** Trading one readback for an
  extra pass is usually a *loss* (§13.1, §14.6).
- **Use `triwarp._device.read_scalar(arr, index=-1)` for a tail read**, not a hand-rolled spelling —
  the fast path is device-split and a pinned scratch is a **race** (§12.1). It takes any index and
  any dtype, so `arr[k : k + 1].numpy()[0]` and `arr.numpy()[k]` are both it, spelled slower
  (27.8 against 15.7 µs on CUDA); the sites converted in this pass were `creation.sweep_polygon`'s
  two path endpoints, `bounds`' two winning frames and `reconstruction`'s two face counters.

---

## 4. Evolving the public API

### 4.1 Naming

- **Name a function after what it returns, in NumPy vocabulary — never after the Warp call it
  wraps.** `sort_pairs` named `warp.utils.radix_sort_pairs`'s key/value mechanism rather than its
  result (a sort *and* an argsort), which is why it became `sort_and_argsort`.
- **A mask is named `<element>_<property>_mask`, element first.** Element-first sorts and completes:
  type `tw.validation.face_` and every per-face predicate appears, which is exactly what the
  property-first spellings did not do — `bad_face_mask` and `flipped_faces_mask` were the two nobody
  could find. Five names were converted and five deliberately left alone, because **a convention
  followed everywhere regardless of fit is not worth having**: `radius_outlier_mask` /
  `statistical_outlier_mask` (the property *is* the name), `half_space_mask` (the element is implicit
  and the geometry is the point), `fillable_loop_mask` (already element-first), `uv_seam_vertex_mask`
  (element-final, and the qualifier is a namespace) and `convex_subset_mask` / `convex_superset_mask`
  (the module docstring turns on the subset/superset opposition). Check 2 scans these names for their
  dtype; nothing enforces the *order*, which is a review question.
- **Two public names that differ by one character are a defect even when both are correct.**
  `boundary.boundary_loop` and `boundary.boundary_loops` meant "the longest one" and "all of them";
  the singular is now `longest_boundary_loop`. Look for this whenever a plural is added next to an
  existing singular.
- **A wrapper whose whole device side is one kernel nobody else launches shares that kernel's
  name.** `array.sort_pair_indices` launched `init_sort_pair_indices` and `array.arange` launched
  `init_range`; the `init_` prefix says "this kernel initializes a buffer", which is true of every
  fill kernel in the tree and therefore says nothing, so all it did was stop `grep sort_pair_indices`
  from finding both halves at once. Both are renamed, and the affix classes to reject are
  `init_` / `do_` / `compute_` / `run_` / `make_` / `kernel_` / `_impl` / `_inner`.
  **Do not apply this literally — it is a rule about *filler*, and a literal scan is 129 rows of
  which one was the defect.** Measured: an AST scan over `triwarp/` pairing each wrapper with the
  kernels it references, restricted to wrappers referencing exactly one kernel that no other
  wrapper references, flags 129 sites; narrowing to "the two names differ only by a filler affix or
  a plural" leaves **6**, and every one of those six is *informative*:
    - a **plural** because the kernel is batched over what the wrapper answers for one thing
      (`geodesic_walk.trace_from_face` → `trace_from_faces`);
    - **`_pass` / `_step`** because the kernel is one iteration of a loop the wrapper runs
      (`graph.shortest_path_envelope` → `shortest_path_envelope_pass`, `smoothing.relax_approx` →
      `relax_approx_step`);
    - **`finalize_`** because the kernel is the second half of a two-stage reduction whose first
      half is another wrapper (`points.fit_line` / `fit_plane` / `principal_axes`).

  Two further exemptions the 129 make obvious. A kernel in a **shared kernel library** (§3.1's four)
  keeps that library's vocabulary — `vertices.vertex_defects` launches `scatter.scatter_sum_scalar`
  and must, because the name has to read correctly for the other ten importers. And a kernel that
  computes one *ingredient* of the wrapper's answer keeps the ingredient's name, the rest of the
  wrapper being host-side index arithmetic: `remesh.subdivide` → `compute_midpoints` is right about
  the midpoints and would be wrong called `subdivide`. So the question to ask is not "do the names
  match" but **"does the kernel's name carry information the wrapper's name does not"** — and if it
  does not, the kernel takes the wrapper's name. Re-run the scan
  (`plans/`-local, ~60 lines of `ast`) rather than re-deriving the 129.
- **A tuning choice is a keyword, not a name — and if the kernel already branches on it, the Python
  layer is the only place it doubled.** `neighbors` exposed each ball and nearest query twice, once
  per accelerator, for eight names covering four operations: identical arguments, identical returns,
  differing only in the name of the optional prebuilt structure — while `kernels/neighbors.py` had
  *already* unified them behind `ACCEL_HASHGRID` / `ACCEL_BVH` selectors. They are now `query_ball` /
  `query_ball_count` / `query_ball_with_offsets` / `query_nearest` with
  `backend="hashgrid" | "bvh"` and an `accelerator=` that infers it. Three consequences:
    - **A default that must be distinguishable from "not passed" is spelled `None`.** `backend`
      defaults to `None`, documented as "`hashgrid` when no `accelerator` is given", so that handing
      over a `wp.Bvh` and nothing else does not read as contradicting a default the caller never
      wrote. Only an *explicit* mismatch raises.
    - **Keep the discriminator in the benchmark group name, not the function name** —
      `query_ball_bvh` / `query_ball_hashgrid`, `query_nearest_{bvh,hashgrid}_k{1,7,64}`. The group
      name is the parity key, so all 19 markers moved in the same commit.
    - **A merge like this needs a triwarp-against-triwarp test that the two paths agree**, or the
      shared group name is an unchecked claim (`test_the_two_backends_agree`).

  What stays split: `query_bvh_ball` and `query_bvh_box` are genuinely BVH-only — a hash
  grid has no box query. And where the *pairings are different algorithms over different inputs*
  rather than one algorithm with a tuning knob, the name keeps carrying the type: `metrics.chamfer_*`
  / `hausdorff_*` were considered for the same treatment and declined.
- **One operation family, one module.** Functions with the same shape of signature computing the same
  *kind* of answer belong together, and a family split across two modules is a defect however
  reasonable each half looked when it landed. The five per-point surface descriptors
  (`ambient_occlusion`, `volumetric_obscurance`, `shape_diameter`, `thickness`,
  `max_tangent_sphere`) sat in two modules while sharing one kernel module, and
  `benchmarks/test_proximity.py` had already voted by holding all five rows — **the suite disagreeing
  with the split is the signal to look for.**
- **Place a function by what it computes and what machinery it shares, not by where the reference
  library keeps it.** `triangles.volume` / `moments` / `centroid` sat in `triangles.py` because
  `trimesh.triangles.mass_properties` exists, and they are whole-mesh reductions in a per-triangle
  module. Names keep their trimesh / igl spelling where that is the field's vocabulary — this rule is
  about *placement* only. **The machinery half outranks the subject half, and it is measurable**: a
  function that is a *component* of another module's solver stays with it. Check with an import/call
  scan, not by reading.

### 4.2 Signatures and returns

- **A packed buffer and its offsets are returned, and accepted, values first.** `(values, offsets)`,
  never `(offsets, values)`; where a third array rides along it is a *per-item* one and goes last
  (`(ring_halfedges, offsets, is_boundary)`). The convention is stated in `array.pack_1d_arrays`'
  docstring. Both halves are usually `wp.array[wp.int32]`, so **a transposed unpack type-checks,
  runs, and indexes garbage** — there is nothing but the convention to lean on. Measured before it
  was made unanimous: 13 of 15 public returns and **4 of 4** argument lists were already values
  first, and `igl.vertex_triangle_adjacency` orders it the same way. The one place a caller
  *constructs* such a pair by hand is a precomputed keyword (`descend_field(vertex_faces=…)`);
  transposing it there **segfaulted the CPU backend several launches after the call, not at it**, so
  that composition carries a test of its own. Do not debug such a crash as a Warp problem — check the
  pair order first.
- **No public signature or return type may name `np.ndarray`**, outside `triwarp/io.py` (§3.8).
- **A guard must encode a real limitation.** When the implementation is naturally rank- or
  dtype-agnostic — a flatten/reshape, a generic `@wp.func` — drop the `ensure_ndim` cap and widen the
  annotation instead of validating a restriction that is not there.
- **When a function mirrors a NumPy one, mirror its *positional* signature too, and make `device`
  keyword-only.** `array.arange(n, device)` / `arange_step(count, step, device)` were two functions
  covering the one NumPy call whose whole convention is its positional arity —
  `arange(stop)` / `arange(start, stop)` / `arange(start, stop, step)` — so a caller who knew
  `numpy.arange` had to learn a second spelling and a reader could not tell `arange_step(6, 3)`
  from `arange(6, 3)`. They are now one `arange(start, stop=None, step=1, dtype=wp.int32, *,
  device)`, which is `numpy.arange`'s signature with `device` promoted from optional to required
  (nothing here allocates onto Warp's ambient device, §3.3). Two things that fall out:
    - **A required keyword-only `device` breaks every positional call site, and the residual set
      must be re-derived from the *new* name** (§4.4). A `tw.array.arange(` grep found 15 and
      missed 4 more reached through `from triwarp.array import arange`; those failed at runtime,
      not at lint or type-check time.
    - **Keep the specialised kernel for the common case.** One `start + i * step` kernel would be
      the tidy answer and costs two more marshalled arguments (~2 µs of a ~26 µs 200k-element
      call, §13.1) on the only path any in-repo caller takes, so `arange` dispatches: the
      zero-argument `arange` kernel when `start == 0 and step == 1`, `arange_affine` otherwise.
- **No speculative generality.** Add an axis, parameter, or mode only when an in-repo call site needs
  it. The absence of a caller is a reason not to build it, not a gap to fill.
- **No near-duplicate wrappers.** Two public functions that are the same algorithm with different
  returns share one private helper (`concatenate` / `pack_1d_arrays` behind `_pack_segments`).
- **Inverse and dual pairs cross-reference each other and have a round-trip test** — e.g.
  `flatnonzero` / `indices_to_mask`. Bidirectional `See Also` is required for inverse pairs and for
  simple/advanced variants of one operation; it is *not* required for hub→spoke references (most of
  the ~250 one-way links in the package are correct — `cotmatrix` should not list every consumer).
- **Coverage is per module.** Every public `triwarp/<module>.py` gets both `tests/test_<module>.py`
  and `benchmarks/test_<module>.py` (check 4), and a function's tests live in the file mirroring
  *its* module (§5), not in a neighbour's.

### 4.3 Docstring, signature and body must agree

- A documented `Raises` must be reachable, and any function with a direct `raise` in its own body
  needs a `Raises` block (check 11; the 42 functions that delegate validation to a shared guard and
  document its `Raises` are correct and are not scanned). In particular §3.9 forbids device-mismatch
  checks, so **no docstring may document a `ValueError` for arrays "on different devices"**.
- A documented validation must actually be performed, or the claim goes.
- Annotations must cover every rank and dtype the docstring claims and the body supports.
- **When a comment and the body disagree, decide which one is load-bearing before "fixing" it — the
  usual answer is the comment.** `tangent_space.any_perpendicular`'s comment claims it crosses with
  *"whichever coordinate axis the normal is least aligned with"* while the body compares only
  `|n[0]|` against `|n[1]|`. The body is **correct for its purpose** (it needs any axis not parallel
  to the normal, and x or y always qualifies), and "correcting" it into a three-way argmin would move
  the tangent frame at every z-dominant normal. Fix the sentence, leave the branch, and say in the
  commit which of the two you changed and why.

### 4.4 Moving or renaming

**Moving or renaming a public function moves everything derived from it — in the same commit.** A
move is not done when the wrapper compiles; it is done when nothing still points at the old home.
Five artifacts, every time:

1. **Its kernels, if they are exclusively its.** A kernel referenced by only the moved function moves
   to the destination's `kernels/` module; a kernel shared with a function that stays put does
   **not** move, and the new kernel module imports it (kernel-to-kernel imports are normal —
   `kernels/predicates.py` has 17 importers). Decide by measuring, not by reading: an AST scan of
   which wrappers reference each `kernel_<mod>.<name>` is the authority, because a kernel that
   *looks* single-purpose is often reached from a private helper in a third module.
2. **Its tests**, into `tests/test_<destination>.py`, keeping the §5 source order.
3. **Its benchmark rows**, into `benchmarks/test_<destination>.py`.
4. **Its `benchmark(group=...)` name**, when the group is named after the function or its old module.
   Renaming a group is allowed and sometimes required — but the group name is the parity key, so
   **every `parity` / `noparity` marker citing it must be updated in the same commit**, and
   `uv run python -m tests.parity` must show the same pair count before and after.
5. **Its docs entry** in `docs/gen_ref_pages.py` `SECTIONS`, plus every `[`name`][triwarp.old.path]`
   cross-reference — `mkdocs build --strict` is what finds the ones you missed.

**Renaming a *keyword argument* has its own artifact list, and a call-site scan sees none of it.**
Three sites survived a paren-aware rewrite of all 33 `neighbors` query calls, each invisible for a
different reason, and each caught by a *runtime* check rather than a static one:

1. **A `TypedDict` field feeding a `**splat`.** `registration._TargetIndex` declared `bvh: wp.Bvh`
   and the call site read `query_nearest(..., **target_index)` — the keyword's name is nowhere near
   the call. Eight tests failed with `TypeError: unexpected keyword argument 'bvh'`. Grep the *old
   keyword name on its own*, not just at call sites.
2. **A rename that ran after the function rename.** A bulk pass had already turned
   `query_hashgrid_ball_count` into `query_ball_count`, so the later keyword pass no longer matched
   it and left `grid=` behind. Sweep for "merged name still carrying the old keyword" as a separate,
   final pass — ordering two mechanical passes wrong silently skips their intersection.
3. **A fenced ```python docstring example.** Check 14's runtime `exec` is what found it; `ast.parse`
   and every grep were clean.

The general rule: after any mechanical rename, **re-derive the *residual* set from the new names**
rather than trusting that the pass which produced them was complete. And the full suite catches what
a targeted per-file run does not — a rename is not done until the whole suite has run.

### 4.5 The mechanical gate: `tests/api_conventions.py`

**Twenty-three checks**, and they fail the default `pytest` run.

- **Eight scan the public surface of `triwarp/` (excluding `kernels/`)**: a summary line naming a
  reference library (1); a `*_mask` producer that does not return `wp.array[wp.bool]` (2); a module
  summary advertising Warp (3); a module without a `tests/` **and** a `benchmarks/` file named for it
  (4); a private name reached across a module boundary (5); one public name exported by two modules
  (6); a top-level `kernels/<name>.py` without its `triwarp/<name>.py` or the reverse (7); a private
  helper defined above its first caller (8).
- **Check 9 scans `kernels/`, `tests/` and `benchmarks/` as well**: a comment or docstring blaming a
  Warp version older than the installed `warp-lang`. It reads **only** the anchored spelling
  `Warp 1.17`. `_WARP_VERSION_ALLOWLIST` holds deliberate history and is where the next upgrade's
  re-verification notes go. **A stale `pytest.skip` is worse than a stale comment** — the comment
  misinforms, the skip deletes a branch, and on a CUDA box the deleted branch is the one nobody runs;
  16 skips for one long-fixed `cg` bug survived because check 9 originally scanned `triwarp/` only.
- **Check 5 has a `_`-prefixed-module escape, and reaching for it is usually the wrong of two
  answers.** The check reads only public wrapper modules (`scan_package` skips `kernels/` and any
  path component starting with `_`) and only `_`-prefixed *names*, so a shared wrapper-side helper
  with a plain name inside `triwarp/_thing.py` is invisible to it — which is what `_device.py` is.
  But a new `_*.py` holding one helper is a module created to dodge a check, and the check's premise
  is sound: a shared *operation* wants a home, not a hiding place. Worked through on
  `adjacency.resolve_face_adjacency`, which review wanted off the public surface while
  `curvature` and `validation` both called it. Moving it to `triwarp/_adjacency.py` passed every
  gate and was still reverted, because splitting it showed which half was actually shared: the
  **rule** (both tables or neither) is a contract three modules must enforce identically, and is
  now the public `adjacency.require_paired_adjacency`; the **derivation** is one
  `face_adjacency(return_edges=True)` call that each caller writes inline with its own
  `n_vertices`. So when a private cross-module helper has to stop being public, first ask whether
  it is really one operation — a validator plus a one-line default is two, and only one of them
  needs to be reachable. Two knock-ons either way: every `[`x`][triwarp.mod.x]` cross-reference to
  a name that moves into a `_*.py` breaks `mkdocs build --strict` (a private module generates no
  page), and a *newly* public validator needs its own `Raises` block (check 11) and a test that
  covers the accepting cases as well as the raise.
- **Check 10**: an allocation with no `device=` (§3.3). **Check 11**: a public function that raises
  with no `Raises` block (§4.3).
- **Check 12**: a fenced ```python docstring example that does not run. The one static check that is
  not static — `tests/api_conventions.py` extracts the blocks and `tests/test_api_conventions.py`
  `exec`s them against a mesh fixture, because both defects it was written for were *runtime* ones (a
  `wp.array` compared with a float, a NumPy bool array handed to `flatnonzero`) and `ast.parse` sees
  nothing wrong with either. Blocks holding a bare `...` are deliberate outlines and skip. A new
  example needing a name the fixture does not bind fails with `NameError` — extend
  `example_namespace`, do not weaken the test.
- **Checks 13-18, 20, 22 enforce kernel conventions**: `out_` prefix and position (13, §2.1);
  subscript-style array annotations, whole package (14, §1.2); a launch with no `device=` (15, §3.9);
  cast spelling (16, §1.3); integer division (17, §1.5); bare `bool`/`int`/`float` annotations (18,
  §1.2); kernel-scope ternary (20, §1.5); bare single-index `wp.tid()` (22, §1.3 — landed at 45 sites
  against 422, clustered per file rather than scattered).
- **Check 19** reads `tests/`: a test comparing against a reference library with no class label
  (§7.4). It exists because the convention decayed **twice** — the lowercase `class b` spelling went
  21 → 0 → 9, invisible to the prescribed grep because a human reads `class B` and `Class B` the
  same. Four decisions keep it from misfiring: it keys on `ast.Assert`, not the function body (a
  fixture unpack `mesh_tm, mesh_wp = icosphere` names a `_tm` variable in every mesh test, and keying
  on the body takes it from 0 hits to 120); it accepts **all four** label phrases (§7.4), or it would
  fail 14 correct tests; it leaves `_np` out of its suffix list (290 false positives); and it checks
  only that a label is *present*, never that it is the right one. The rarer defect it also closes is a
  comparison with **no docstring at all**, which ruff cannot see because `D103` is ignored.
- **Check 21**: a MeshLib name anywhere under `triwarp/` — a *licensing* guard (§7.6).
- **Check 23**: a kernel module whose `@wp.func` is `wp.map`'d from several sites with no
  declaration table (§3.5). Like its `wp.overload` sibling it asserts a table *exists* and never that
  it is complete; the completeness gate is the load census, which is a clock measurement (§15.1).

Each check carries a written allowlist — read the reason before adding an entry, and prefer fixing
the code. **The gate does not replace review**: it cannot tell whether a *new* name is a good one,
only that it does not break a convention the package already holds to.

**Two rules about the gate itself:**

- **A new Warp construct can silently switch off a static check that predates it.** Bundling buffers
  into a `@wp.struct` removed them from check 13's view entirely, because the check resolved store
  targets to a bare `ast.Name` — `edges.count[slot] = 1` and `wp.atomic_add(edges.count, slot, 1)`
  both stopped matching, and nothing went red. Fixed by walking `ast.Attribute` as well as
  `ast.Subscript`. **After introducing a construct the package has not used before** (the first
  `@wp.struct`, the first `wp.ref`, the first tile intrinsic), grep `tests/api_conventions.py` for
  checks that pattern-match on AST node types and ask whether the new construct is invisible to them.
  A green suite after a refactor is equally consistent with "still covered" and "no longer looked at".
- **Keep writing checks with a staleness half.** The tell that caught the struct case was check 13
  complaining that two allowlist entries no longer matched anything.

---

## 5. Function ordering within a module

`mkdocstrings` is configured with `members_order: source`, so **source order is the rendered docs
order** — placement in the file is part of the public API's discoverability, not cosmetic.

### Python wrapper modules (`triwarp/*.py`)

- File layout: module docstring → imports → module constants / type aliases → functions.
- **Group public functions thematically, then order the groups by importance and expected frequency
  of use.** Primary entry points at the top; niche or low-level variants last. Within a group put
  closely related functions consecutively (`expand_vertex_mask` / `shrink_vertex_mask`), and the
  simple form before its advanced variants (`query_ball` before `query_ball_with_offsets` /
  `query_ball_count`).
- **Stepdown rule for private helpers** (check 8): a private function called by exactly one public
  function goes **immediately after** that function. A private helper shared by several functions in
  the same file goes after its **last** caller. Cross-cutting utilities go in a trailing "private
  helpers" section at the bottom. A private helper must never appear above its first caller — a
  reader should never need to jump backward to a definition they haven't been introduced to yet.
  Two things the draining sweep learned: reordering has to consider *every* private helper in a
  module, not only the flagged ones (moving a flagged helper below a callee it uses strands that
  callee); and a name can appear several times at module scope, since `@overload` stubs precede their
  implementation, so verifying a "pure move" by comparing functions keyed on name silently collapses
  those duplicates — compare the multiset.

### Kernel modules (`triwarp/kernels/*.py`)

The stepdown rule **inverts**: Warp resolves `@wp.func` references at kernel-decoration time, so a
`@wp.func` **must textually precede** every kernel (or other `@wp.func`) that calls it. Order kernels
to mirror their wrapper module's public-function order, and place each kernel's `@wp.func` helpers
immediately **before** that kernel; a `@wp.func` shared by several kernels goes before its **first**
user.

### Tests (`tests/test_<module>.py`)

Mirror the corresponding wrapper module's public-function order, so a reader scanning tests
top-to-bottom sees the same story as the API page.

---

## 6. Documentation (MkDocs + mkdocstrings)

Docs are built with **MkDocs Material** + **mkdocstrings** (`python` handler,
`docstring_style: numpy`), configured in `mkdocs.yml`. `docs/gen_ref_pages.py` auto-generates one API
reference page per public module under `triwarp/` on every build (`triwarp/kernels/` is excluded) — a
new module needs **no manual nav entry**. Preview with `DISABLE_MKDOCS_2_WARNING=true uv run mkdocs
serve`; validate with `DISABLE_MKDOCS_2_WARNING=true uv run mkdocs build --strict` (fails on any
broken cross-reference or unresolved external inventory). The `DISABLE_MKDOCS_2_WARNING` prefix
silences a promotional banner injected by the `properdocs` transitive dependency.

**The one-line summary says what the function returns, never which C++ call it wraps.** mkdocstrings
renders that first line as the function's entry in its module's API index, so a reference library's
name there turns the index into a table of bindings — `laplacian.cotmatrix` read *"Cotangent
stiffness matrix / discrete Laplacian (``igl::cotmatrix``)"* where it should read *"Cotangent
stiffness matrix of the mesh: the discrete Laplace-Beltrami operator."* Attribution is *wanted* and
stays — 32 of the 49 wrapper modules mention a reference library somewhere in their prose — but one
line down, in `Notes` or `See Also`. Enforced by check 1, whose allowlist is `mesh.py`'s "mirrors
`trimesh.Trimesh`" alone. Check 3 forbids a module summary ending in `(Warp)` or `on NVIDIA Warp`:
the whole package is Warp.

**No measured timing belongs in a public function's docstring.** Not a millisecond figure, not a
speedup ratio, not a launch or byte count — those are facts about *this* box, *this* Warp version
and *this* mesh, and mkdocstrings publishes them as though they were part of the contract, where a
reader on other hardware reads a number that is simply false for them. The docstring keeps the
*claim* the number supports, in terms a caller can act on: "roughly doubles the call", "a host
readback serialises the device pipeline", "the fixed per-segment cost dominates at these widths".

**Moving the number, not deleting it.** §9 requires a measured result to live at its site, so the
figure goes into a `#` comment in the same function's body — or into the private helper that
actually pays it, which is usually better, because two public forms sharing one cost then carry the
sentence once. Numbers stay welcome in private helpers' docstrings, in `kernels/` (nothing there
renders), in `benchmarks/` group docstrings and in Part II here.

The scan is an `ast` walk over `triwarp/`'s public functions matching
`\d[\d.,]*\s*(ms|us|µs|ns|GB|MB|kB)\b` or `\b\d+(\.\d+)?x\b` against each docstring — a plain
grep for `ms` is unusable, and the same regex over *prose* words (`measured`, `faster`,
`benchmark`) returns 35 kB of legitimate behavioural text, so key on the *quantity*.
**Measured 2026-09-03: 50 public functions across 30 modules, 174 lines.** The three modules
reviewed that day (`adjacency`, `array`, `boundary`) are clean; **45 functions across 24 modules are
not yet converted** — `linalg` 5, `proximity` 4, then `voxels` / `selection` / `remesh` / `holes` 3
each — and the list regenerates from the scan rather than being maintained here.

Docstrings stay **NumPy-style** (`Parameters`/`Returns`/`Raises`/`See Also`), but cross-references use
**mkdocs-autorefs** link syntax, not Sphinx roles — Sphinx interpreted-text roles (`:func:`, `:attr:`,
`:meth:`, `:class:`, `:data:`, `:mod:`) have no Markdown equivalent and render as literal, broken text
under MkDocs.

| Target | Syntax | Example |
|---|---|---|
| Internal (`triwarp.*`) | `` [`short_name`][fully.qualified.path] `` | `` [`face_adjacency`][triwarp.graph.face_adjacency] `` |
| External **with** a configured inventory (`trimesh`, `numpy`, `scipy`, stdlib) | `` [`fully.qualified.name`][] `` (empty brackets) | `` [`trimesh.grouping.group_rows`][] `` |
| External **without** an inventory (`warp`, `igl`) | plain double-backtick code span, no link | ``` ``warp.sparse.BsrMatrix`` ``` |
| Shapes, literals, C++ names, file paths | plain code span | ``` ``(n_vertices,)`` ``` |

- Always resolve internal refs to the **fully-qualified path**, even for a function in the same module
  — numpydoc's old auto-linking of bare `See Also` names does not carry over.
- Before adding a new external inventory to `mkdocs.yml`, verify it serves an `objects.inv`
  (`curl -I <url>/objects.inv`).
- RST admonitions (`.. note::`) don't exist in Markdown — use MkDocs Material's `!!! note`.
- Module-level constants / type aliases without their own docstring are only linkable because
  `show_if_no_docstring: true` is set; don't remove that option without re-checking
  `triwarp/constants.py` and `triwarp/typing.py` cross-refs.
- Every public function should have a docstring — an undocumented one still gets a page entry and
  renders with an empty description, which looks broken.
- After editing docstrings, sanity-check with `grep -rnE ':(func|attr|meth|class|data|mod):\`'
  triwarp/` — it should return nothing (`kernels/` counts too: nothing there renders, but a reader
  meets the broken role text all the same).

---

## 7. Testing

Every new geometry function MUST have regression tests comparing against a CPU reference
implementation. `trimesh` is the default; §7.6 lists the eight others and what each one actually
binds.

### 7.1 Conventions

- Test file `tests/test_<module>.py`; import pattern:
  ```python
  import trimesh.<module> as tm
  import triwarp.<module> as tw
  ```
- Use the `device` fixture from `tests/conftest.py`. **Every test function must accept `device`.**
- Generate reproducible random data with `np.random.default_rng(seed)` (a fixed integer seed per
  test).
- **Upload a NumPy mesh with `conversions.numpy_to_warp(vertices_np, faces_np, device)`**, never a
  local helper. This section used to *print the body* of one, and the result was six private copies
  across six modules at 54 call sites, differing only in where the `float32` cast sat — printing an
  implementation is an invitation to paste it. Its `wp.vec2` sibling is `numpy_to_warp_uv`; the
  inverse is `warp_to_trimesh`. Triangle-soup arrays `(n, 3, 3)` become an indexed mesh first:
  ```python
  vertices_wp, faces_wp = numpy_to_warp(
      tri_np.reshape(-1, 3), np.arange(tri_np.shape[0] * 3, dtype=np.int32), device
  )
  ```
- Call `.numpy()` on Warp outputs **inline**, before passing to NumPy comparison functions; do not
  bind a new variable for it.
- `np.allclose(got, exp, rtol=1e-5, atol=1e-5)` for floats; `np.array_equal` for booleans/integers.
- Name variables with a library suffix — `_np` (NumPy/SciPy), `_tm` (trimesh), `_wp` (Warp), `_igl`,
  `_pp` (potpourri3d), `_pml` (pymeshlab), `_o3d` (open3d), `_pv` (pyvista), `_ml` (meshlib),
  `_pmf` (pymeshfix), `_p3d` (pytorch3d). Avoid `got` / `exp`; use clear names.
- Passing a NumPy 1D vector to a `wp.vec3` scalar argument at Python scope:
  `wp.vec3(*array_np.tolist())` — not `wp.vec3(*map(float, np.asanyarray(...).reshape(3)))`.

### 7.2 Devices, and the two-process runner

The `device` fixture **is parametrized, and `--device={auto,cpu,cuda,both}` selects for *this
process*** — `auto` picks one device (cuda if available), exactly like `benchmarks/conftest.py`, and
every test id carries a `[cpu]` / `[cuda0]` suffix. `both` means "every device this process can see,
skip nothing".

**Both-device coverage is a two-process job: `uv run python -m tests.devices`.** It runs a CUDA pass,
then a CPU pass with **`CUDA_VISIBLE_DEVICES=""`**, and that variable is the whole point: **Warp's
CPU work is ~36x slower once CUDA has been initialised in the process** (§12.1). Measured at the
pytest level: `tests/test_heat_signed.py` is 166.77 s with CUDA visible and **8.53 s** without. Whole
suite: in-process `--device=both` measured **717 s** against ~37.6 s + ~155 s as two passes. **Never
reach for `--device=both` on a GPU box to get CPU coverage** — use the runner.

**Run the runner before calling a change done**, not just the default `pytest`. Both-device coverage
is what caught the `warp.fem` ambient-device leak in `reconstruction._screened_poisson_adaptive` —
broken for CPU input on any box with a GPU, and invisible to a CUDA-only run (the devices matched)
*and* to a `CUDA_VISIBLE_DEVICES=""` run (`warp.fem` then defaults to CPU too). That defect class
needs CUDA present *and* the arrays on the host, which is a configuration neither single-device run
reaches. `wp.ScopedDevice(device)` is the fix when a dependency picks the device for us.

**A test that costs more than ~15 s on CPU wears `@pytest.mark.slow_cpu(<measured seconds>)`**, which
skips it when it would run on `cpu` unless `--device=both`. Four `screened_poisson` tests carry it,
and they were 277.6 s of a 432.6 s CPU-only run — one ~90 s depth-6 solve each, under a second on
CUDA (§16.3). In a CUDA-hidden process `--device=both` therefore means "all of CPU, including these",
which is how the runner's `--slow-cpu` asks for a full CPU pass. Use the marker only where the
*device* is the cost and the claim is device-independent, and put the measured number in it; a test
slow on both devices belongs on a smaller input instead.

**When a kernel is launched with `launch_tiled`, pin `"cpu"` explicitly in a parametrize** rather
than trusting the `device` fixture — it returns `cuda:0` whenever CUDA is available, so the CPU path
of every tiled kernel is otherwise unexercised (§12.2 records two defects that hid there).

**When two devices differ but neither is wrong, compare both to a common oracle rather than to each
other.** `heat_signed_distance` differs cross-device by 3.4e-03, but against `heat_geodesic` from the
same loop (which agrees across devices to 1.1e-08) both sit at max error 2.768921e-02 — the
cross-device gap is 5x *smaller* than either device's own discretization error, so no guard is
warranted.

### 7.3 Mesh fixtures (prefer over inline construction)

Reuse shared fixtures from `tests/conftest.py` instead of building meshes in each test. Fixtures
return `(mesh_tm: tm.Trimesh, mesh_wp: wp.Mesh)` via `tests.conversions.trimesh_to_warp`.

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

The last three are built by `creation.parametric_surface` and are the only inputs in the suite that
are non-orientable or that have an odd Euler characteristic. A boolean predicate asserted only on the
orientable fixtures is testing one branch; that is what these close. `creation.parametric_surface`
builds thirteen more surfaces that are not fixtures yet — reach for one (and add the fixture) rather
than hand-rolling a degenerate mesh.

- **Do not** call `tm.creation.box()` or hand-roll `wp.Mesh(...)` in tests unless the case requires a
  bespoke degenerate mesh (empty faces, unreferenced vertices). **Never construct a `wp.Mesh` with
  zero triangles on a CUDA device** — it silently corrupts the allocator (§12.1).
- When a simple cube would suffice, prefer **`icosahedron`** or **`cave_cube`** for richer geometry.
- Parametrize over multiple fixtures with `request.getfixturevalue(mesh_name)` when coverage should
  span mesh types.
- Use `mesh_wp.device` (not the `device` fixture) for query-point allocation when a mesh fixture is
  already in scope.
- Edge-case tests (empty points/faces, single-triangle pathology) may still use minimal inline
  buffers — a **single-triangle** mesh is the safe way to reach an "empty mesh" guard on CUDA.
- **The fixture *sets* are shared too**: `CLOSED_MESHES`, `OPEN_MESHES` and
  `MESHES = CLOSED_MESHES + OPEN_MESHES` live in `tests/conftest.py`. "The four that span closed/open
  and convex/non-convex" is a decision about coverage, and it had been restated in ten files under
  three names, which meant a fixture added to the set reached exactly one of them. Import them; keep
  a local list only where it is genuinely a different set, and say in a comment why
  (`test_adjacency.py` drops `cave_cube` because its coplanar box faces make every adjacency angle 0
  or pi/2).
- **`trimesh.slice_plane`'s output is a poor input to several references.** A hemisphere built that
  way from `icosphere(2)` reports **17** boundary loops in MeshLib and loads 121 → **137 v** in
  pymeshfix where the surface has one rim; after `merge_vertices()` it loads unchanged at 97 v and
  reports **1**. Use the `tests/conftest.py` fixtures (`hemisphere` calls `merge_vertices()` for
  exactly this reason) and **assert the hole count** before comparing a per-hole answer. The same
  duplication makes `boundary_loops`' documented "last write wins" device-dependent, which once read
  as total harmonic/tutte/arap disagreement.

### 7.4 The parity gate: a benchmarked reference must be a tested reference

`benchmarks/` asserts only shapes and finiteness, so on its own it cannot tell whether two timed
implementations compute the same thing. `tests/test_parity.py` closes that loop and **fails the
default `pytest` run** when a benchmarked `(group, library)` pair is neither tested nor exempted. Two
markers join the suites; the benchmark's `benchmark(group=...)` name is the key, which makes group
names a cross-suite API — renaming one breaks every `parity` marker that cites it.

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

**Where a reference library computes the same quantity, one test must compare the two outputs.** That
is the obligation, and it is not discharged by an invariant: a function can be watertight, symmetric,
idempotent and manifold while computing the wrong answer. So if any of the nine references has the
quantity — **check, do not assume**: §7.6's hazard blocks list what each one actually binds, and
several names that *look* present are not — there is a class A/B/C comparison against it.

**Invariant checks are welcome and belong in the same test.** Watertightness, an involution, a
counting identity, a round trip, a conservation law — these catch failure modes no reference
comparison sees. Assert them *alongside* the output comparison rather than in a test of their own, so
one test carries the whole claim and the parity marker sits on the test that does the comparing.
Split them out only when the invariant needs an input the comparison cannot use.

**Where no reference computes the quantity, an invariant-only test is the honest answer**, and its
docstring says so in those words: *"Not a library comparison: <why none exists>"*, followed by what
the invariant excludes. That is a fifth label beside A-D, not a class-D exemption — D is for a
*benchmarked* pair whose results are genuinely incomparable and needs a `noparity` entry; this is for
a quantity with no counterpart to benchmark. `halfedge_twins` (no reference has a halfedge
structure), `homology.tree_cotree` (nothing computes a basis) and `geodesic_walk`'s arc-length checks
are the shape of it.

**Classify every comparison, and say which class it is in the docstring:**

- **A** — direct `np.allclose` / `np.array_equal`. The default.
- **B** — equal after a *named* transform, still at `1e-5`: a dict index, a unit fix
  (`igl.doublearea / 2`), a reduction (igl's per-vertex mask → triwarp's bool), a projection
  (`igl.boundary_loop` is the longest loop), `lexsort` for unordered rows, a sign or gauge fix.
  **Most apparent non-equivalence lands here.** The benchmark docstrings' "does strictly less",
  "upper bound" and "output shape differs" caveats are about *cost*, not about the value.
- **C** — a derived scalar, set distance or statistic, because no correspondence exists. Must name
  the bug class it excludes, and record a mutation probe *and its margin* in the docstring —
  threshold ≥ 3x from the measured agreement. `fraction_within` bounds must be shown to fail under
  shuffling one side, or they are testing marginal distributions rather than the correspondence.
- **D** — exemption. Only for: not an independent implementation (`oracle=` required); a different
  algorithm with a measured disagreement; a parameter the reference lacks; an answer not observable
  in isolation; stochastic with no invariant; or input classes where triwarp is undefined. **Not**
  admissible: "awkward", "the tolerance would be loose", or any class-B situation.

**Write the label as `Class A`** (capital, the word before the letter). A lowercase `class b` reads
the same to a human and is invisible to a grep — 21 had accumulated before anyone looked, then 9 more
after that count was taken to 0, which is why check 19 gates it.

**Four phrases carry a label, not two**, and a scan or review that knows only the first two misreads
14 correct tests as unlabelled. Measured over the 523 tests whose `assert` reads a
reference-suffixed variable: `Class [ABCD]` (464), `Not a library comparison` (48), `Triwarp against
triwarp` (9), `Not a parity assert` (5). The last two are labels in good standing — do not reword
them to fit a narrower grep.

**Never a parity assert:** shape-only or `isfinite`-only (that is the *benchmark's* assert, and this
gate exists to stop it migrating inward); triwarp compared with itself; a threshold a constant output
would pass. A boolean assert must be parametrized over inputs producing both answers.

**Triwarp-against-triwarp is not a parity assert but is still a legitimate test**, for one job:
pinning two entry points to each other where only one has an oracle — a mask form against an index
form, a precomputed path against the deriving one, a CPU run against a CUDA one. Say which of the two
carries the oracle.

**Check the comparison is not vacuous on its fixture**, which the gate cannot do for you.

- **An empty answer.** `test_ears` compared `igl.ears` against `boundary.ears` on `hemisphere` and
  `half_torus`, where neither library finds a single ear, so the assert was `[] == []` and the loop
  body checking the corner convention never ran. Making it non-vacuous immediately surfaced a real
  disagreement (`triwarp_opp == (igl_opp + 1) % 3`).
- **A constant answer is as vacuous as an empty one**, and in both measured cases the docstring
  *asserted the non-vacuity that was absent* — and the sentence is what stopped anyone re-checking.
  `test_connected_component_labels_random` said "200 random edges over 64 nodes gives several
  components rather than one" and produced **1** component holding all 64 nodes, so the class-B
  label-packing transform it exists for was the identity. `test_discrete_mean_curvature` said "over
  every vertex" and ran on a *regular* icosahedron, where trimesh's answer is **one value at spread
  0.0**, so a permuted result, an off-by-one in the gather and a query/vertex index swap all pass.

  **A claim about the input's shape is a claim an assert can carry cheaply, so make it an assert and
  not a sentence** — `assert np.unique(labels_np).shape[0] > 1`, `assert np.ptp(reference) > 1e-3`.
- **Assert the reference produced a non-empty answer, or its expected count, before comparing to
  it** — and treat "this function is already the oracle in tests/" as no evidence that the comparison
  is live.
- **A docstring that says the vacuous branch is "covered separately" is a claim to check, not a
  reason to stop.** `test_face_nondegenerate_mask` compared against `trimesh.triangles.nondegenerate`
  on a fixture where both sides are all-`True`, and said so — *"which is why the degenerate branch is
  covered separately by the zero-area tests in this file"*. **Those tests did not exist**; the
  nearest one exercised a different function, and every call in the suite that feeds the mask a
  degenerate face lives in `tests/test_repair.py`. So the guard §7.7 records as the regression
  detector for `screened_poisson(point_weight=0.0)`'s zero-area triangles had its own reference
  comparison untested on the condition it guards. It is now parametrized over a clean and a
  degenerate arm with `assert np.count_nonzero(~nondegenerate_tm) == (2 if with_degenerate else 0)`
  carrying the claim, and the mutation probe confirms it is the *trimesh comparison* that fails when
  the mask is forced all-`True`. **Grep for the tests a docstring names before believing it.**

**Proving a guard test "bites" means identifying WHICH assertion fails under the mutation, not that
the test fails.** A multi-assert test can fail for a reason unrelated to the defect it was written
for: one k-NN tie-break test "verified the gate" by deleting a carry flag and seeing 10 of 12 cases
fail — but only the *index* assert failed, distances stayed bit-identical, and the docstring already
declared the identity of a tied neighbour unspecified. So the test pinned a convention the public API
disclaims and cost a second kernel path per bucket. Corollaries: if a test compares triwarp against
itself, ask what an external oracle would say instead; and **before building a test around a plan's
failure-mode claim, reproduce the claim** — a plan's "measured" claims are hypotheses about what was
probed.

**A Class C threshold with a large headroom is a threshold that does not bite, and the probe is
what shows it.** §7.4 asks for a margin of at least 3x *between the threshold and the measured
agreement*, which is a floor against flakiness; it says nothing about the ceiling, and four of the
tests probed in one pass sat at 9.5-38x. Two were retightened on the probe's own numbers —
`offset_mesh` against both MeshLab and pymeshlab from `0.5 * _VOXEL` to `0.25` (14x and 9.5x → 7.0x
and 4.8x, which is what makes a **10 %** offset error fail where only a 25 % one did before), and
`resample_uniform` from `2.0 * voxel_size` to `0.5` (38x → 9.6x, separating a **5 %** scale error
where 10 % used to pass). The probe to run is the one that re-runs the *reference* on a deliberately
wrong input, not one that perturbs the triwarp side: it measures what the threshold can actually
distinguish. Two results that went the other way and are recorded as such: `heat_geodesic`'s 5 % bar
against the exact great-circle field cannot be tightened, because a third of it is genuine
discretization error in both methods; and the two `heat_signed_distance` correlations pair with an
error bound that has only **1.19x** headroom on `hemisphere`, so that one is at its floor already.

**A rank correlation is scale-invariant and an error bound is not, so a Class C test carrying both
is carrying two different guards — say which catches what.** Measured on
`heat_signed_distance`: shuffling one side fails the error bound (0.478 against a 0.150 bar) but
leaves the correlation at 0.73 on a 12-vertex fixture; negating one side fails the correlation and
leaves the error bound's magnitude untouched; scaling one side by 1.5 fails the error bound and
leaves the correlation *exactly* unchanged. Neither statistic alone excludes the bug class.

**Run the mutation probe on a test whose whole point is a boundary predicate.** A branch-agreement
test for `greedy_downsample_mask` did not bite: mutating the search's `>=` to `>` left all three
spacings (uniform / clustered / duplicate points) passing, because none put two points *exactly* one
step apart. "Random inputs plus a ties case" does not reach an exact tie — a fourth spacing of 0.25
segments with step 1.0, both exact in float32, does.

### 7.5 Shared helpers — check both modules before writing a private one

Reuse `tests/comparisons.py` (`lexsort_rows`, `assert_unordered_rows_equal`, `undirected_edges`,
`edge_multiplicity`, `euler_characteristic`, `open_edge_count`, `canonical_labels`, `same_partition`,
`canonical_winding`, `assert_same_up_to_sign`, `assert_cyclic_permutation_equal`,
`assert_same_loop_set`, `trimesh_outline_loops`, `fraction_within`, `symmetric_chamfer`,
`chamfer_two_sided`, `symmetric_surface_distance`, `hausdorff_two_sided`,
`hausdorff_surface_two_sided`) and `tests/conversions.py` (`numpy_to_warp`, `numpy_to_warp_uv`,
`points_to_warp`, `points_to_warp_uv`, `trimesh_to_warp`, `warp_to_trimesh`, `trimesh_to_open3d`,
`points_to_open3d`, `open3d_to_trimesh`, `trimesh_to_open3d_t`, `trimesh_to_pymeshlab`,
`warp_to_pymeshlab`, `points_to_pymeshlab`, `trimesh_to_pyvista`, `points_to_pyvista`,
`pyvista_edges_to_indices`, `numpy_to_meshlib`, `trimesh_to_meshlib`, `warp_to_meshlib`,
`points_to_meshlib`, `meshlib_to_trimesh`, `numpy_to_meshlib_bitset`, `meshlib_scalars_to_numpy`,
`meshlib_indices_to_numpy`, `meshlib_bitset_to_numpy`, `numpy_to_pymeshfix`, `trimesh_to_pymeshfix`,
`warp_to_pymeshfix`, `pymeshfix_to_numpy`, `pymeshfix_intersecting_faces`, `pymeshfix_face_remap`,
`points_to_torch`, `numpy_to_pytorch3d`, `trimesh_to_pytorch3d`, `warp_to_pytorch3d`,
`points_to_pytorch3d`, `pytorch3d_to_numpy`, `faces_igl`, `mesh_igl`) rather than re-rolling either.
Every one of the six helpers consolidated in 2026-08 was written by someone who did not check, and
`undirected_edges` alone had been spelled three different ways across six files.

**`points_to_warp` and `warp_to_trimesh` are the two most-reached-for, and both were re-rolled for a
long time.** The bare-cloud upload had been written out **403** times in four equivalent spellings
across 35 files *plus* six one-line private copies carrying 181 more calls, because every reference
library had a `points_to_*` and Warp did not; it is now one helper at 532 call sites. The readback
direction is the mirror image and the asymmetry is worth knowing about yourself: **a test author
reaches for the shared helper when *building* the reference and writes the readback by hand, every
time** — `warp_to_trimesh` sat at 4 mentions in 2 files while 38 sites inlined
`tm.Trimesh(x.numpy(), f.numpy().reshape(-1, 3), process=False)`.

`canonical_labels` is the label-packing transform every component comparison needs — triwarp names a
component after a representative element, igl and scipy number `0..k-1` in their own traversal orders
and VTK's `RegionId` numbers them in a third, so only the *partition* is shared.

**Two of the class-C helpers take different inputs and the mesh one raises on point arrays.**
`symmetric_chamfer(mesh_a, mesh_b)` takes two *meshes* and samples them itself, where
`chamfer_two_sided(points_a, points_b)` takes two clouds already drawn — which is what a comparison
between two *samplers* needs. **Prefer the mean form over `hausdorff_two_sided` where the claim is
distributional**: measured on two independent 1 000-point samplings of `icosphere(2)`, the mean
statistic separates the same mesh from one scaled by 1.15 by **6.8x** (0.00768 against 0.05227) where
the worst-case Hausdorff separates them by **1.4** (0.162 against 0.228), because one stray sample in
a tail dominates a maximum. `symmetric_chamfer` also has a sampling noise floor — a mesh against
itself scores ~0.028 on `icosphere(3)`, so a threshold must clear that, not sit under it.

**`lexsort` is unusable on float coordinates with ties.** `lexsort_rows` sorts exactly, so two sides
that tie in `float32` but differ in the 16th digit in `float64` order those rows differently and the
compare fails by the full coordinate range (measured 1.59 on a subdivided icosphere, 0.83 on a star
ring). Both are false negatives. For **positions**, match with a `cKDTree` nearest-neighbour query
plus a bijection check, or use `hausdorff_two_sided`; keep `lexsort_rows` for integer index rows,
where it is exact.

### 7.6 The nine reference libraries

**All nine are hard test dependencies. Import them plainly — never through `pytest.importorskip`.**
That rule covers every package in `[dependency-groups] test`, not only the reference libraries:
`libigl`, `shapely` and `moderngl` were guarded by eight `importorskip` calls, which is a latent hole
rather than a safety net — if the import breaks the tests *vanish* instead of failing, and one of
those eight sat under a `parity` marker that `tests/test_parity.py` would have kept passing because
the marker is static and the skip is not. Where a *runtime* precondition genuinely cannot be declared
as a dependency (an EGL driver for `moderngl`'s OpenGL reference), keep a skip, but make it name the
driver and put it in the fixture that needs the context, never at module scope over the import.

Aliases are pinned in ruff's import conventions: `import trimesh as tm`, `import igl`,
`import potpourri3d as pp3d`, `import pymeshlab as ml`, `import open3d as o3d`, `import pyvista as
pv`, `from meshlib import mrmeshpy as mm` / `mrmeshnumpy as mn`, `import pymeshfix` /
`from pymeshfix import _meshfix`, `import pytorch3d.ops as p3d_ops` / `.loss` / `.structures`.

**Threading, because it decides how a ratio reads:** single-threaded — trimesh, igl, pyvista,
pymeshfix (measured 435 ms wall against 435 ms `process_time`, ratio 1.00). Multi-threaded —
`meshlib` (143 OS threads measured live) and `pytorch3d-cpu` (24 torch threads). GPU —
`pytorch3d-cuda` only. A `triwarp-cpu` row loses to a threaded reference on any parallel op
regardless of algorithm, which is §9's "decide on the CUDA number" again.

#### libigl (`igl`)

The reference whose input convention matches triwarp's most closely — `float64` `(n, 3)` vertices and
`int64` `(n_faces, 3)` faces, which is exactly what `mesh_tm.vertices` / `mesh_tm.faces` already are
— and every bound function is *pure*. The exceptions are the stateful solver objects
(`HeatGeodesicsData`, `ARAPData`, `min_quad_with_fixed_data`, `AABB`), which cache a factorization
and must be constructed **inside** a timed callable. Five hazards, all measured:

- **An out-of-range face index is a SIGSEGV, not an exception.** `igl.cotmatrix(V, F)` with one entry
  of `F` set to `len(V) + 500` kills the interpreter with exit code 139 and no traceback — igl
  bounds-checks nothing. Never hand it a reduced `V` with the original `F`. The same class of crash
  hits `igl.principal_curvature` on a non-manifold vertex, and `igl.heat_geodesics_precompute` /
  `igl.harmonic` / `igl.lscm` refuse (raise) rather than crash on meshes they cannot factor.
- **Three bound functions are memory-unsafe on ordinary input, so a "works" probe is not enough** —
  check *values*, and prefer a fixture class where the function is known safe. `igl.loop` aborts with
  `free(): invalid pointer` on a five-vertex mesh with three faces on one edge and SIGSEGVs on
  `bunny_decimated`; on `bunny` it silently returns 1 113 `NaN` rows, one per unreferenced vertex,
  because it indexes `igl::adjacency_list` (sized `F.max() + 1`) up to `n_verts`. **`igl.in_element`
  is unusable outright**: on a two-triangle square it never reports element 0 for any query inside
  it, the same query returns a face in a 3-query batch and `-1` in a 7-query batch, and a 200-point
  Delaunay input aborts with `malloc(): invalid size` — use `scipy.spatial.Delaunay.find_simplex`, or
  pyvista's `find_containing_cell`. And **`igl.upsample` corrupts the process heap on the scan
  meshes**, which matters more than the other two because the SIGSEGV lands *later*, in unrelated
  code, and `--benchmark-json` is written at session end — so it silently destroyed every row of
  `benchmarks/test_remesh.py` for two measurement rounds. Measured one selection per process: **2 of
  8** runs crash with the igl rows alone, **8 of 8** with other libraries co-resident, and per mesh
  **7 of 8** on `bunny_decimated`, **7 of 8** on `bunny`, **0 of 8** on `dragon`. Two readings to
  *not* take: it is not an interaction with triwarp (an earlier round concluded it was, from a single
  nondeterministic failure), and it is not a size limit — the smallest mesh fails most and the
  largest never does. Compacting the unreferenced vertices away makes it worse. It is safe on
  `icosahedron` (1 200 calls, six processes, clean), so it stays a tested reference there and is
  *not* a benchmarked one.
- **F-only functions size their output by `F.max() + 1`, not by `len(V)`.**
  `igl.adjacency_matrix`, `igl.vertex_components` and `igl.is_vertex_manifold` return `F.max() + 1`
  rows where `igl.cotmatrix` and `igl.gaussian_curvature` return `len(V)`. So on a mesh with
  unreferenced vertices the two families disagree with each other and only the `(V, F)` family
  matches triwarp; a comparison against the F-only family is class B with the transform named. Also
  `igl.connected_components(igl.adjacency_matrix(F))` counts every isolated vertex as its own
  component.
- **Several call signatures are not what the docs suggest, and two fail silently.**
  `igl.exact_geodesic(V, F, vs, vt)` returns an *empty array* rather than raising, because
  `VS/FS/VT/FT` all default to `array([])` and a 4-argument call binds `vt` to `FS` — pass all six,
  with the face arrays explicitly `np.array([], dtype=np.int64)`.
  `igl.knn(P, V, k, *igl.octree(V)[:4])` takes seven positional arguments. `igl.in_element` needs a
  live `igl.AABB`. `igl.crouzeix_raviart_*` need `(V, F, E, EMAP)` from `igl.unique_edge_map(F)`.
  `igl.average_onto_vertices`'s `S` is a per-face *scalar*; `igl.cut_mesh`'s `C` is a per-corner
  **bool** mask, not an edge list.
- **`collapse_small_triangles` and `resolve_duplicated_faces` are not bound**, despite the C++
  headers existing and `triwarp.repair` carrying functions named after them (`AttributeError`).
  Generally: the C++ surface is ~493 headers and only 150 functions are bound, so confirm a name
  exists in the wheel before planning a comparison around it. A hand port of the C++ into a test file
  is a legitimate *test* oracle (see `tests/test_metrics.py`, `test_polyline.py`, `test_seams.py`)
  but never a benchmark row.

**Licensing:** libigl's core is MPL2, but everything under `reference/libigl/include/igl/copyleft/`
is **GPL** — the CGAL boolean suite, `progressive_hulls`, `quadprog`, tetgen and
`copyleft/marching_cubes`. No triwarp code may be derived from that subtree; read the MPL2 top-level
`marching_cubes.h` if a reference is needed, never the `copyleft/` one.

#### potpourri3d (`pp3d`)

For the heat-method family, tangent spaces and isocontours — where neither trimesh nor igl has an
equivalent. pybind11 over geometry-central; takes `float64` `(n, 3)` vertices and `(n_faces, 3)`
`int32` faces.

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
- A zero cotangent weight (an edge whose two opposite angles are both right angles, i.e. every quad
  grid split by a diagonal — `cave_cube`, `half_torus`) erases that edge's phase from
  `get_connection_laplacian()`, so it cannot serve as an oracle there at all; see
  `tests/test_tangent.py`.

#### pymeshlab (`ml`)

The broadest reference — 281 filters. Build the MeshSet with
`tests.conversions.trimesh_to_pymeshlab(mesh_tm)` (or `warp_to_pymeshlab`) rather than hand-rolling
`ml.MeshSet()`. Use it where it is a *better* oracle than the incumbent, not everywhere. Every one of
these traps **fails green** — a row that reports the build, the `k=1` solve, or a zero-hole no-op
looks like a plausible measurement:

- **Almost every filter mutates `current_mesh()` in place.** `apply_coord_*` moves vertices,
  `meshing_*` rewrites the topology, `compute_*_per_vertex` writes an attribute, `generate_*` pushes
  a *new* mesh onto the set. So one MeshSet serves one filter call; build a fresh one per comparison.
  `compute_curvature_principal_directions_per_vertex` and
  `meshing_decimation_quadric_edge_collapse` additionally default to `autoclean=True` and will delete
  unreferenced vertices under you. Build cost ~0.47 µs/vertex (17 ms on `bunny`), so the MeshSet goes
  inside the timed callable — with one measured exception, the selection filters, whose cost is
  independent of how much is selected (0.86-1.21 ms across 0.9 % → 86 % of 81 920 faces).
- **`get_*` filters return a dict; `compute_*` / `meshing_*` / `apply_*` return `None`** (or a small
  dict of statistics) and the answer must be read off `current_mesh()` — `vertex_matrix`,
  `face_matrix`, `vertex_normal_matrix`, `vertex_scalar_array`, `face_scalar_array`,
  `vertex_selection_array`, `face_selection_array`, `vertex_curvature_principal_dir{1,2}_matrix`,
  `edge_matrix`. Selections come back as bool arrays and scalars as float arrays.
- **Length parameters take a wrapper type.** `ml.PercentageValue(1)` is 1 % of the bbox diagonal;
  `ml.PureValue(x)` is an absolute length (this version has no `AbsoluteValue`). Feed `PureValue`
  from the same number triwarp gets.
- **`harm_function` is a no-op.** `compute_texcoord_parametrization_harmonic` returns bit-identical
  texture coordinates at `harm_function=1`, `2`, `3` (max deviation exactly 0.0) and identical
  timings, where libigl's own `k=2` costs 4.3x its `k=1`. Never map triwarp's `k` onto it.
- **Four defaults silently measure nothing.** `meshing_close_holes(maxholesize=30)` closes *zero*
  512-edge rims; `get_hausdorff_distance(samplenum=8)` samples 8 points of the whole cloud;
  `generate_sampling_poisson_disk(radius=0%)` autoguesses instead of using yours; and
  `generate_surface_reconstruction_ball_pivoting(clustering=0)` reconstructs **nothing** (0 faces
  against 1 277 at its 20 % default) and returns *faster* for it. Always assert on the returned dict
  and assert the reference produced output before comparing to it.
- **`get_hausdorff_distance` has a *second* silent default: `maxdist` returns `inf`.** It defaults to
  a percentage of the bbox diagonal, and a pair separated further comes back with `min` / `max` /
  `mean` / `RMS` of `inf` — no exception, no warning. Two unit spheres at a 2.0 gap read `inf` at
  every `samplenum` until `maxdist=ml.PureValue(1e6)` is passed, then read 2.00000000. Its `min` is a
  sound *upper bound* on the surface-to-surface minimum and is tight whenever a sample lands on the
  witness — always, for a convex pair, because the support point of a **polytope** is a vertex
  (identical at 1 000 and 100 000 samples). It is a registered oracle for `mesh_to_mesh_distance` on
  that basis.
- **Some parameters are silent no-ops, and one runs backwards.**
  `compute_scalar_by_shape_diameter_function_per_vertex`'s `cone_amplitude` produces byte-identical
  output at 90 and 120 degrees; `apply_normal_smoothing_per_face` and
  `apply_scalar_smoothing_per_vertex` expose no parameters at all; and
  `generate_resampled_uniform_mesh`'s `offset` as a `PercentageValue` runs from *full erosion* at 0 %
  to full dilation at 100 %, so its own 50 % default is the **zero** offset — pass `PureValue(0.0)`
  when you mean zero. **Probe a parameter before building an axis on it.**
- **Non-manifold input raises, it does not degrade.** `meshing_surface_subdivision_midpoint` and
  `generate_polyline_from_planar_section` both fail on *every* scan mesh. Same boundary as libigl /
  potpourri3d. Other preconditions that raise:
  `compute_texcoord_parametrization_{harmonic,lscm}` need a *boundary* (a closed mesh fails — use
  `hemisphere` / `half_torus`); `compute_matrix_by_fitting_to_plane` needs `set_selection_all()`
  first; `compute_matrix_by_icp_between_meshes` needs **both** layers to carry faces;
  `face_face_adjacency_matrix()` raises `MissingComponentException` unless the FF component was
  requested (`update_topology()` alone does not enable it). `generate_boolean_*` takes `first_mesh` /
  `second_mesh`, not `first` / `second`.
- **Selection morphology is face-based.** `apply_selection_dilatation` / `..._erosion` dilate the
  *face* set; a vertex selection handed to them is simply cleared. From a vertex seed the bridge is
  `compute_selection_transfer_vertex_to_face(inclusive=False)` — `inclusive=True` (the default)
  selects only faces whose *every* vertex is selected. Dilate then maps exactly onto
  `expand_vertex_mask`; **erode does not** map onto `shrink_vertex_mask`.
- **`generate_surface_reconstruction_vcg` returns 0 faces** on an oriented cloud at every voxel size
  probed. Rejected as a reference.
- **Two filters differ from triwarp's port by definition, not tolerance**, and the tests say so:
  `apply_scalar_smoothing_per_vertex` averages a *boundary* vertex over its two boundary neighbours
  alone (so its oracle runs on closed fixtures), and `apply_coord_two_steps_smoothing` at its own
  defaults moves a noisy cube *further* from clean than the noise was, because its fitting step
  rounds corners in.
- `MeshSet(verbose=False)` is the default but several filters print to stdout regardless (ICP, the
  point-cloud normal estimator, the VCG reconstructor); pytest's fd capture absorbs it.
- **`face_normal_matrix()` after `compute_normal_per_face()` is the *unnormalised* cross product**
  (magnitude exactly `2 * area`), so it checks normals *and* areas.
- **Pass counts are conventions**: MeshLab's Taubin `stepsmoothnum` counts lambda-mu *pairs* where
  triwarp and trimesh do one half-step per `iterations`, so the mapping is `2 * stepsmoothnum`; and
  `get_scalar_statistics_per_vertex`'s `"med"` is the sorted element at index `n // 2 - 1`, one
  *below* the middle, for both parities.
- **MeshLab has two uniform coordinate umbrellas, and neither is documented.** Recovered by solving
  least-squares for the per-vertex stencil over 12 random position sets on one connectivity (residual
  2e-16): `apply_coord_laplacian_smoothing` and `apply_coord_unsharp_mask` weight each neighbour by
  its shared-face count and include the vertex itself once (`1/(2d+1)` self, `2/(2d+1)` per neighbour
  on a closed mesh), while `apply_coord_taubin_smoothing` uses the plain 1-ring mean. The difference
  is 8 % of the displacement — far too large to read as a tolerance. **That technique is the general
  one: one pass of a linear filter is a linear map, so its stencil is solvable.**
- **MeshLab writes a layer transform, not vertices.** `compute_matrix_by_icp_between_meshes` leaves
  `vertex_matrix()` byte-identical to the input; the answer is `transform_matrix()` /
  `transformed_vertex_matrix()`. Read the wrong one and a converged ICP looks like a no-op.

**Licensing:** pymeshlab is GPL. `triwarp/` may name it in prose (it does, throughout); no triwarp
code may be derived from its source.

#### open3d (`o3d`)

Build meshes with `tests.conversions.trimesh_to_open3d` (reusable across calls, unlike a MeshSet),
clouds with `points_to_open3d`, and tensor-API meshes with `trimesh_to_open3d_t` — never chain off an
unbound `from_legacy(...)`. The installed wheel is a CUDA build whose legacy `open3d.geometry` /
`open3d.pipelines` APIs are CPU-only; only `open3d.t` has GPU kernels. Nine hazards, all measured:

- **Legacy `remove_*` / `orient_*` / `filter_*` methods mutate in place** (build inside the timed
  callable); the pure `compute_*` / `get_*` / `is_*` calls recompute unconditionally and can share
  one mesh. The trap in the second family: **`get_volume` validates before it integrates**, and the
  validation is the full brute-force `IsWatertight` composition — 13.8 s on a watertight 82k-face
  sphere whose integral is microseconds — and it *raises* on non-watertight input. Never benchmark it
  as "volume".
- **Open3D's tensor meshes must be held in a name.**
  `o3d.t.geometry.TriangleMesh.from_legacy(x).fill_holes()` lets the temporary be collected and the
  result reads freed memory — garbage floats (2052.1, 4.4e-41) rather than an exception.
- **`fill_holes` winds its cap against the rest of the mesh**, so a raw signed volume of its output is
  meaningless (−1.06 where the truth is 2.02). `trimesh.repair.fix_winding` first.
- **k-NN distances come back squared** from both `KDTreeFlann` and `o3d.core.nns`. Use
  `o3d.core.nns.NearestNeighborSearch` for anything batched (indices match `scipy.spatial.KDTree`
  byte-for-byte on a tie-free cloud); the legacy tree's only query is a per-point Python loop, 6x
  slower at 20k queries. `KDTreeFlann`'s radius search is **exclusive at exactly `r`** where
  triwarp's ball queries are inclusive — random clouds never tie, so only a constructed fixture can
  expose it.
- **`is_vertex_manifold` tests connectivity, not a fan**: three faces sharing one edge pass it and
  fail triwarp's and igl's fan definition. The answers agree exactly on edge-manifold input —
  restrict the comparison to that class and pin the divergence. `is_edge_manifold` shares triwarp's
  `allow_boundary_edges` switch with identical semantics.
- **Smoothing filters re-derive inverse-distance weights from current positions every pass**
  (`filter_smooth_laplacian`, `filter_smooth_taubin`), so they match triwarp's fixed assembled
  operator at one iteration (6.6e-08) and diverge over ten; Taubin's `number_of_iterations` counts
  lambda-mu *pairs*. `filter_sharpen` adds `strength * (deg(v) * v - Σ neighbours)` — the
  *unnormalized* residual, so its displacement is triwarp's times the vertex degree (measured ratio =
  degree to 7 digits) and no parameter mapping fixes an irregular mesh. All three are D2 exemptions.
- **Platonic solids come in rotated frames and odd scales**: the octahedron matches triwarp's vertex
  table exactly, but the tetrahedron is rotated (nearest-vertex distance 0.92 after scaling) and the
  icosahedron is the raw `(0, ±1, ±φ)` table at circumradius 1.902 — compare rigid-motion invariants
  after scaling to unit circumradius, never positions. There is no `create_dodecahedron`.
- **`RaycastingScene.compute_signed_distance` shares triwarp's convention exactly** (negative inside,
  parity-ray sign; 1.8e-7 agreement) — no negation, unlike trimesh. But `compute_closest_points`
  diverges ~2e-4 at equidistant-face ties, so compare *distances*, not the returned points.
- **`get_oriented_bounding_box` is PCA of the hull and minimizes nothing** (12.9 % above triwarp's
  volume on a tilted half_torus); the comparable entry point is `get_minimal_oriented_bounding_box`.
- **`remove_radius_outlier` is nondeterministic**, so it cannot be a class-A oracle. It shares one
  `KDTreeFlann` across an `#pragma omp parallel for` whose radius search is not thread-safe under
  that sharing: measured three distinct keep sets (43 / 44 / 45 points) over eight repetitions of one
  500-point cloud. Its published rule (`count > nb_points`, self counted) is sound, and evaluating it
  through the *same* tree one query at a time reproduces `points.radius_outlier_mask` exactly. So the
  comparison goes through `search_radius_vector_3d` in a loop and the filter keeps only the benchmark
  row. No other `remove_*` method shares the defect, which is why this one had to be found rather
  than assumed.
- **A down-sampler's output order is its own, not its algorithm's.** Every legacy selection routes
  through `SelectByIndex`, which walks a *mask* over the input and emits survivors in ascending index
  order — so `farthest_point_down_sample`'s greedy sequence is destroyed on the way out and only the
  selected *set* can be compared (equal at counts 4 / 32 / 64). Where the order is the claim,
  transcribe the C++ loop into the test: `FarthestPointDownSample` takes its arg-max with a strict
  `>`, so the lowest index wins a tie. Also `num_samples=0` returns an empty cloud rather than
  raising, and `compute_nearest_neighbor_distance` reports **`0.0`** for a cloud of fewer than two
  points where the honest answer is `inf`.
- Benchmark gotchas: `remove_duplicated_triangles` mutates in place, returns `self` and is
  idempotent; `compute_vertex_normals` overwrites but never caches (a shared mesh is fine);
  `select_by_index` takes **vertex** not triangle indices; `Vector3dVector` rejects non-writeable
  arrays, so use `np.array(...)` not `np.ascontiguousarray(...)` on a trimesh `TrackedArray`;
  `fill_holes` is tensor-API only.

#### pyvista (`pv`)

The **VTK** reference (VTK 9.6 through pyvista's own layer). Build meshes with
`tests.conversions.trimesh_to_pyvista` and clouds with `points_to_pyvista`; one `PolyData` serves many
comparisons because pyvista caches nothing. `pyvista.core` needs no renderer and every comparison runs
headless. A `uv run` whose working directory is inside `reference/pyvista` resolves *that* project and
builds a second virtualenv — always invoke probes from the repo root. Licence: pyvista is MIT and VTK
is BSD-3, so there is no `copyleft/` subtree to avoid. Fifteen hazards, all measured:

- **Float64 in, sometimes float32 out.** Point storage is exact (round-trip error `0.0` on
  `[1/3, π, e]`) and `regular_faces` is a real `(n, 3)` `int64` array — that is why pyvista, not
  vedo, is the VTK oracle. But `compute_normals`' `Normals`, `ray_trace`'s hit points,
  `fit_plane_to_points(return_meta=True)`'s centre and normal and `texture_map_to_*`'s coordinates
  come back **float32**, while `multi_ray_trace`, `principal_axes`, `curvature` and
  `compute_implicit_distance` are float64. Check the dtype per row. Where a difference of large
  numbers is taken (the angle defect), use `atol` and know the residual is **triwarp's** float32
  vertex buffer: 5.10e-05 against pyvista's float64 and 5.4e-05 against vedo's float32.
- **Nothing is cached, and one filter of twelve mutates.** Repeat calls recompute (`cell_quality`
  10.1 / 7.2 ms, `decimate` 210 / 201 ms), so a shared `PolyData` is right. The exception:
  **`edge_mask` writes `point_ind` into its input.** Every filter with an `inplace` switch defaults
  to `False`; never pass `True`.
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
  duplicates `aspect_frobenius` exactly; `min_angle` / `max_angle` are in **degrees**; `distortion`
  is a constant **1.0** on ordinary input, so a threshold test on it passes on anything; and the 16
  measures that do not apply come back as the `-1.0` null value rather than raising.
  `tests/test_triangles.py` carries the decoding.
- **`is_manifold` is `n_open_edges == 0`, and `n_open_edges` counts boundary *plus* non-manifold
  edges.** So `is_manifold` maps to `is_edge_manifold(allow_boundary_edges=False)` and the *count*
  does not map to `boundary_edges`: on three faces sharing one edge it reads **7** where triwarp
  counts 6 boundary edges. Likewise **`DataSet.center` is the bounding-box centre**, not a centroid,
  and `bounding_sphere` returns `(radius, center)` — radius first — and is a genuine near-minimal
  sphere, not the sphere about the AABB centre; the two coincide on any centrally symmetric mesh
  (which is how the wrong reading survived three fixtures) and differ by 11.8 % on a hemisphere.
- **`curvature('maximum')` / `('minimum')` are algebra, not estimation** — measured *exactly*
  `H ± √(H² − K)` from VTK's own Gauss and mean curvature, max abs difference **0.0** in float64, and
  complex on the 300 of 642 icosphere vertices where `H² < K`. They are not an independent
  principal-curvature implementation. `curvature('gaussian')` is exactly the angle defect over the
  barycentric lumped area and is a genuine oracle.
- **Surface operators return surface quantities.** `compute_derivative`'s gradient is the
  *tangential* one — mean `2/3 · e_x` for `f = x` on the unit sphere, which triwarp's
  `face_gradients` reproduces to 1.8e-07 after `average_onto_vertices` — and it stays on the
  **points** for a point-data field even with `preference='cell'`.
- **Both smoothers are different algorithms, not different tunings.** `smooth` moves each vertex
  along its incident edge directions under VTK's own convergence test, so it diverges with the
  iteration count against a fixed assembled operator (0.103 / 0.363 / 0.581 at 1 / 5 / 10 at
  `relaxation_factor=1.0`); `smooth_taubin` is the windowed-sinc filter and warns *"An optimal offset
  for the smoothing filter could not be found"* on ordinary input (0.835 at one iteration, then
  6.5e-03 / 2.6e-02). Its iteration count is in lambda-mu **pairs**. Both are D2 exemptions.
- **Several answers are empty, constant or unchanged rather than wrong, so assert non-vacuity
  first.** `clip_surface(pv.Sphere(radius=0.6))` returns **0 cells** on `icosphere(3)`;
  `extract_values(0.010, scalars='area')` returns 0 cells (it matches values *exactly* — a range is
  `ranges=`); `edge_mask(30)` is all-`False` on a smooth sphere (use a box); `integrate_data` of a
  symmetric point field reads −1.2e-15; `validate_mesh().coincident_points` is **empty** on ten
  exactly duplicated vertices (`clean` is the dedup oracle, and its `zero_size` — not
  `degenerate_faces` — is where a zero-area triangle lands); `lines_from_points` gives one two-point
  line cell per segment rather than one polyline, so `compute_arc_length` restarts every segment and
  `decimate_polyline` is a no-op at every reduction; `tube` / `ribbon` emit triangle **strips**
  (`n_faces == 0` — `.triangulate()` first); and `extrude(capping=True)` leaves 16 open edges.
- **Where the reference put the answer, and four deprecated names.** `align(return_matrix=True)`
  returns `(aligned_mesh, 4×4 matrix)` and *does* move the points (unlike MeshLab); `geodesic` puts
  the ordered path in `vtkOriginalPointIds` and its Euclidean length equals `geodesic_distance` to
  1e-8; `sample` marks misses with `vtkValidPointMask` *and* a `vtkGhostType` array;
  `voxelize_binary_mask` writes a **point** array named `mask` on a cell-centred grid, so it is
  *solid* and its set is contained in triwarp's `mode="solid"` answer rather than equal to it.
  Deprecated in 0.48.4: module-level `pv.voxelize` / `pv.voxelize_volume` (a hard
  `DeprecationError`), `select_enclosed_points` (→ `select_interior_points`, array name now lowercase
  `selected_points`), `extract_geometry` (→ `extract_surface(algorithm=None)`) and `n_faces_strict`
  (→ `n_faces`).
- **`multi_ray_trace` is trimesh + embree, not VTK.** It imports `trimesh`, checks
  `trimesh.ray.has_embree` and calls `tmesh.ray.intersects_location` — measured identical to
  trimesh's own call (first-hit faces 1.0000 over 2 000 rays, 7.3 against 8.9 ms), so a `pyvista` row
  for the `intersects_*` groups would be a **trimesh row under another name**. VTK's own `ray_trace`
  *is* independent (face agreement 1.0000, hit point 2.80e-07) but takes one ray per call at
  **398.6 µs/ray** — 89x embree — so it is a test oracle and never a benchmark row.
- **`validate_mesh()`'s cell fields are per *cell*, not per mesh.** `intersecting_faces` is "two
  faces of a **3D cell**", so on a triangle mesh it is identically empty: 0 on two interpenetrating
  icospheres where `face_self_intersecting_mask` flags **92**, and 0 on `bohemian_dome` where it
  flags 205. `inverted_faces` likewise reads 0 on a mesh with ten reversed faces. The degeneracy
  field that does fire is **`zero_size`**, and `clean()` **keeps** those faces — 82 of 82 cells at
  the default, at `tolerance=0.0` and at `absolute=False` — so there is a detector here and no
  filter. A degeneracy comparison also needs a *scale-aware* input: a float64-exactly-collinear face
  (area 1.25e-17) survives triwarp's float32 altitude test on both devices, so pyvista flags two
  where `remove_degenerate_faces` drops one (§12.4). Relatedly, `collision` is a **two-mesh** filter
  and cannot see a self-intersection either: it reports 2 600 hits for a 320-cell mesh against its
  own copy.
- **`compute_implicit_distance` needs polygons.** On a line-set `PolyData` VTK logs *"No polygons to
  evaluate function!"* once per query and returns a field **3.35** off the truth rather than raising.
  Polyline distance goes through `find_closest_cell` on a **single-cell** polyline instead (2.49e-07
  against `polyline.distance_to_polyline`) — and that single cell is the whole trick:
  `pv.lines_from_points` makes one cell per segment, which is what makes `compute_arc_length` read
  0.0638 for a polyline of length 12.7049. Its locator collapses on a long cell: 24.8 ms at 4 096
  queries against 268 segments, **104 s** at 65 536 queries against 65 536 segments.
- **`find_containing_cell` is the point-location oracle that works**, batched, `-1` outside, and
  measured **1.0000** against `scipy.spatial.Delaunay.find_simplex` on 10 000 queries with the batch
  and the per-point loop byte-identical. Worth stating because igl records `in_element` as unusable
  for the identical question, so generalizing from it skips a good reference. `find_closest_cell` is
  likewise the most accurate closest-point reference registered (4.4e-16 against
  `igl.point_mesh_squared_distance` on distance *and* point) — but its **cell id is not comparable**:
  it disagrees with igl on 28 % of exterior queries, every one a point lying on a shared edge to
  ~1e-16. Compare the distance, at Warp's own 2.2e-04 `mesh_query_point_no_sign` floor (§12.4).
- **Two filters answer a *different* question than their name suggests.** `sample()` interpolates
  only where the query lands **inside** a source cell — 476 of 2 562 target points valid against
  `interpolation.transfer_onto_vertices`, agreeing to 7.20e-08 on those — and
  `snap_to_closest_point=True` snaps to the nearest source **vertex**, not the nearest point on the
  surface, which makes it *worse* (0.256). And `delaunay_2d(edge_source=loop)` does not clip to the
  loop: on a 40-point star it returns 63 cells covering area **4.465** against the polygon's 3.264.
  The polygon-fill oracle is `triangulate_contours`, which adds zero Steiner points and matches
  `polyline.triangulate_polyline`'s `n - 2` count and area to nine digits.
- **Parametric surfaces arrive open and `clean` defaults differently per surface.**
  `surface_from_para(clean=False)` is the underlying default and at that setting every one of the 21
  has 156 or 236 boundary edges (a raw `ParametricMobius()` is a disk), while pyvista overrides it to
  `clean=True` on 9 of them, so `klein` arrives welded and `mobius` does not. **Always pass
  `clean=True` explicitly.** And **`klein` is not a Klein bottle** as VTK parameterizes it: it welds
  to two boundary loops and reads *orientable*, so only `figure8_klein` is the closed non-orientable
  χ = 0 surface. The mirror is byte-identical to the installed wheel for
  `core/utilities/parametric_objects.py`, so those algorithms may be read from the submodule; but the
  surfaces' **domains and periodicity are not in pyvista at all** — they live in VTK's
  `vtkParametric*` constructors as `JoinU` / `JoinV` / `TwistU` / `TwistV`.

#### meshlib (`mm` / `mn`)

Build meshes with `tests.conversions.trimesh_to_meshlib` / `numpy_to_meshlib` / `warp_to_meshlib`,
clouds with `points_to_meshlib`, and read a result back with `meshlib_to_trimesh` — never hand-roll
`mn.meshFromFacesVerts`, whose argument order is the trap below. It earns its seat on three counts
the other eight cannot cover: it is the **only multi-threaded** CPU reference, so a `triwarp-cuda` vs
`meshlib` ratio is a fair fight; it is the only reference that binds a real minimum-weight
Liepa/Klincsek `fillHole` with a 12-metric family and a two-loop `stitchHoles`; and it is the only
oracle for `repair.collapse_small_triangles`. `meshlib.mrcudapy` exists and is deliberately **not**
used: the plain `mrmeshpy` free functions are the reference, and a CUDA module would make the row
incomparable with the others. Fourteen hazards, all measured:

- **Almost every free function mutates its `Mesh` in place and returns something else.** `relax`,
  `fillHole`, `fillHoles`, `decimateMesh`, `remesh`, `subdivideMesh`, `fixMeshDegeneracies`,
  `filterCreaseEdges`, `denoiseNormals`, `smoothRegionBoundary`, `expand` and `shrink` return a
  status `bool`, a count, an `EdgeId` or a `FaceBitSet` of *new* faces — never the mesh. So **one
  `mm.Mesh` serves one mutating call**: build a fresh one per comparison and per benchmark round
  (`BenchCase.new_mesh_ml()` is a method, not a cached property, for exactly this). The exceptions
  are worth knowing per function rather than assumed: `marchingCubes` reads its `SimpleVolume` and
  returns a fresh `Mesh`, so *its* input is cacheable.
- **`meshFromFacesVerts` takes faces *first*, and it sizes the vertex buffer by `F.max() + 1`.** The
  argument order is the reverse of every other converter in `tests/conversions.py`, and a swapped
  call raises nothing. The sizing behaves *differently* by position: on `icosphere(2)` a **trailing**
  unreferenced vertex is dropped outright (163 in → `points.size()` 162) while an **interior** one is
  kept in the buffer and excluded from `numValidVerts` (163 in → 163 back, `numValidVerts` 162). So
  on `bunny`, whose 1 113 unreferenced vertices are interior, the buffers line up and the *validity*
  mask does not; where the spares are trailing, the indices shift. Never hand MeshLib a compacted `V`
  with the original `F`, and never assume `getNumpyVerts(...).shape[0] == len(V)`.
- **`pack()` is mandatory before reading topology back, and skipping it is silent.** Measured after
  `decimateMesh(maxDeletedFaces=200)` on a 320-face mesh: `numValidFaces` is 120,
  `topology.faceSize()` is 320, and **`getNumpyFaces` returns 319 rows** — `last_valid_face_id + 1` —
  of which **199 are `[0, 0, 0]`**. No exception, no warning; after `pack()` the same reads give 120
  and 62. `meshlib_to_trimesh` packs by default.
- **`np.asarray` on a scalar container silently returns a 0-d `object` array.** `VertScalars`,
  `FaceScalars` and `UndirectedEdgeScalars` are not accepted by `mn.toNumpyArray` (which binds only
  `VertCoords` / `FaceNormals` / `std_vector_Vector3_float` and raises a clear `TypeError` otherwise
  — that part is safe), and `np.asarray(vert_scalars)` produces `dtype=object, shape=()` rather than
  raising, so the failure surfaces several lines later. Go through
  `conversions.meshlib_scalars_to_numpy` — and for a container of **ids** rather than numbers through
  `conversions.meshlib_indices_to_numpy`, which exists because the scalar reader *raises* on them: a
  `VertId` implements `__index__` but not `__float__`.
- **A returned bitset is only as long as its highest set bit, and which functions do that is not
  guessable.** `mn.getNumpyBitSet` reads the bitset at *its own* length: one MeshLib sized against
  the mesh comes back domain-sized (`getBoundaryVerts` gives 162 entries on a 162-vertex mesh with
  one bit set), but one built by insertion does not — `findSelfCollidingTrianglesBS` returns **608**
  entries on a 640-face pair of overlapping spheres and an **empty** array on a clean mesh. Neither
  raises, and both break an `np.array_equal` by *shape* rather than by value, which reads as a
  converter bug. Go through `conversions.meshlib_bitset_to_numpy(bitset, size)`.
- **Every bitset converts in bulk, in both directions — a per-bit Python loop is never the answer.**
  A `TypedBitSet` derives from `MR::BitSet` and `mn.getNumpyBitSet` is declared over the base, so
  pybind11 upcasts any of them. The *load* direction is `BitSet.fromBlocks`, which takes packed
  `uint64` blocks — so `np.packbits(flags, bitorder="little")` fills a whole set in one call
  (`conversions.numpy_to_meshlib_bitset`; wrap the result in the typed set). Measured against the
  per-cell `set()` loop it replaced: **258x** at 110 592 voxels (0.33 ms against 84.9 ms), and the
  readback **1 199x** (0.14 ms against 168 ms). Three details: `bitorder="little"` is not NumPy's
  default and is not optional; `fromBlocks` rejects a NumPy `uint64` array with `TypeError` (pass
  `.tolist()`) and rounds the size up to whole 64-bit blocks, so `resize` back to the domain; and the
  element order is the container's, which for a `VoxelBitSet` addressed by a `VolumeIndexer` is `x`
  fastest, i.e. `dense.ravel(order="F")`. **A "MeshLib binds no converter for this" claim is a reason
  to probe the base class, not to write the loop.**
- **`(*args, **kwargs)` in a signature is an overload set, and `inspect` / `help()` cannot see it.**
  This wheel's pybind11 docstrings are stripped. **Read the real signatures by calling the function
  with one junk argument and reading the `TypeError`**, which pybind11 renders as a numbered list of
  every overload — measured 4 for `expand`, 3 for `relax` and `getAllComponents`, 2 for `shrink` and
  `stitchHoles`.
- **Two overloads of one name can have opposite output conventions, and the wrong one binds
  silently.** `expand(topology, region: FaceBitSet, hops)` returns `None` and **mutates `region`**
  (1 face → 12), while `expand(topology, f: FaceId, hops)` **returns** a new `FaceBitSet`; same for
  `shrink`. `stitchHoles(mesh, a, b, params)` takes two named hole edges and `stitchHoles(mesh,
  params)` finds them itself — argument *count* is the only tell. `relax`'s first overload takes a
  `PointCloud` and the second a `Mesh`, with different params types. One `getAllComponents` form
  returns a `(components, count)` **tuple**. Resolve the overload explicitly and assert the result's
  type or count before comparing.
- **`findOutliers`' default mask segfaults on a cloud with no normals.** `FindOutliersParams.mask`
  defaults to `OutlierTypeMask.All`, which includes `AwayNormal`, and that criterion dereferences the
  cloud's normals — measured as a `SIGSEGV` with no exception on a 415-point cloud. The other three
  modes run fine without normals.
- **A projector stores a raw pointer to the mesh or cloud it was given, so a temporary segfaults.**
  `PointsToMeshProjector.updateMeshData(build_a_mesh())` and
  `PointsProjector.setPointCloud(build_a_cloud())` both return normally and then read freed memory in
  `findProjections`. Bind the mesh or cloud to a name that outlives every query — Open3D's
  `from_legacy` hazard in a second library. **`mm.MeshPart` does *not* share the rule — it keeps a
  real Python reference**, so `mm.MeshPart(trimesh_to_meshlib(mesh_tm))` over a temporary is safe:
  `sys.getrefcount(mesh_ml)` goes 2 → 3 across the constructor and `part.mesh is mesh_ml`, where the
  two setters above leave it at 2. That is the whole test, and it is the one to run on the next
  binding of this shape: **probe the refcount, do not infer the lifetime.**
  `findProjections`'s `upDistLimitSq` is a second crash of the same shape: pass MeshLib's own
  `FLT_MAX`, since `math.inf` segfaults rather than raising.
- **The AABB tree is lazily built and cached on the `Mesh`, and the ratio depends on whether the
  query is the process's first.** On `icosphere(4)`, first-vs-second `findProjection` measures
  **68x** on the process's first mesh (2.03 ms → 0.030 ms) and settles at **17-20x** on later fresh
  meshes, the difference being the thread pool spinning up inside the first build. Every query row
  and every timed callable must state whether the build is inside it — recommended: build outside and
  pre-warm with one throwaway query. This is also a *correctness* trap next to the mutation hazard: a
  mutating call invalidates the tree. **A `PointCloud` caches its point tree the same way**:
  `findNClosestPointsPerPoint` on a 160 000-point cloud runs 10.2 ms with the tree rebuilt and 3.8 ms
  reusing it, against triwarp and open3d, both of which build inside every call.
  `cloud.invalidateCaches()` in the benchmark's `setup` is the lever.
- **Per-vertex free functions are per-*vertex*, and `mrmeshnumpy` has the batched form.**
  `discreteGaussianCurvature(topology, points, v)` and `sumAngles(...)` take one `VertId` per call; a
  Python loop over 642 vertices measures **2.79 ms** against **0.046 ms** for
  `mn.getNumpyGaussianCurvature(mesh)` — **49-67x**, bit-identical, and the gap grows with the mesh.
  Never put a per-vertex Python loop in a benchmark row.
- **`getNumpyVerts` is float64 but the storage is float32.** Round-trip error on `icosphere(2)` is
  **2.58e-08**, so a MeshLib comparison bottoms out around 1e-7. `computePerFaceNormals` is
  **normalized** (|n| = 1.0 ± 1e-7) — note the contrast with pymeshlab's unnormalised
  `face_normal_matrix()`.
- **Four parameter conventions that read as a disagreement, and one function whose name lies.**
  `sampleHalfSphere()` is **not** a half sphere: its 145 directions span `z` from -1 to +1 and only
  72 have `z > 0`, so feeding it to `computeSkyViewFactor` as a sky dome halves the answer (0.52
  where the open sky reads 0.98). `InSphereSearchSettings.maxRadius` defaults to **1** whatever the
  mesh's scale, silently capping every thickness on anything larger — pass half the smallest
  bounding-box side. `makeUVSphere`'s `verticalResolution` counts interior latitude **rings**, not
  profile points, so it pairs with `creation.uv_sphere(count=(v + 2, h // 2))` — and at *that*
  mapping the two are the same mesh vertex for vertex (bijection at 4.7e-07), where the same nominal
  resolution differs by 74 % in the vertex count. `leftCotan(e)` is the **plain** cotangent keyed by
  the directed edge whose left face owns it, against `laplacian.cotmatrix_entries`' *half* cotangent
  keyed by `(face, corner)`; `cotan(ue)` is the two summed. Two more found the same way:
  `MarchingCubesParams.origin` addresses the voxel **centre**, so a lattice whose sample `[0, 0, 0]`
  sits at `lower` is marched with `origin = lower - voxel / 2` and the un-shifted call is a rigid
  half-diagonal off (0.0369 against 1.2e-07); and **`findNClosestPointsPerPoint` returns a heap, not
  a sorted list** — the ids are exactly scipy's `k` nearest (set equality 1.0000 at `k=3`) but only
  90.8 % of rows are in distance order and the *nearest* is the **last** entry, which at `numNei=2`
  holds in 100 % of rows only because a two-element heap is ordered by construction. Ask for
  `numNei=1` when one neighbour is the question.
- **`computeRayThicknessAtVertices` takes the direction from the *pseudonormal*.** So it pairs with
  `visibility.thickness(method="ray", normals=angle_weighted_vertex_normals(...))` to **5.96e-07**
  and with the area-weighted normals to **0.031** — five orders worse, the kind of gap that reads as
  an algorithm bug. Both thickness functions and `computeInSphereThicknessAtVertices` also take **no
  query set**, which keeps them out of a benchmark group whose input is a subsample.
- **Its detectors are reliable oracles; several of its mutators are not.** `mm.eliminateTunnels`
  leaves the mesh **byte-identical** on every configuration probed — a 2 048-face torus and a genus-2
  union, at `maxTunnelLength` 4.0 and 1e9, `maxIters` 1 / 2 / 5 / 100, all three `TunnelLoopType`
  values, `buildCoLoops = False`, and through the `FillHoleNicelySettings` overload — while on that
  same torus `detectTunnelFaces` returns **128 faces** and `detectBasisTunnels` the 2 correct loops.
  And `inflate(mesh, verts, InflateSettings)` takes the **unselected** vertices as its Dirichlet
  condition, so selecting every vertex leaves the system with no anchor and collapses `icosphere(3)`
  to the origin (max radius 0.0000) at every pressure probed; given a *region* it solves a different
  problem (volume drops 4.153 → 2.843 before pressure raises it, because the implicit Laplacian
  flattens the cap onto its pinned rim). Meanwhile `findSpikeVertices` and `findInnerVertsOfDegree`
  are exact Class A oracles. **Before building a comparison on a MeshLib mutator, run it once and
  assert it changed the mesh**; where it did not, fall back to the invariant that carries the claim
  (χ arithmetic for tunnels, volume monotonicity and normal alignment for inflation) and say in the
  test docstring what was probed.
- **`mm.localFixSelfIntersections` is the third inert mutator, and it cost four rounds of reading
  the suite's largest loss backwards.** On the `fix_self_intersections` benchmark's own fixture —
  `sphere_med` concatenated with a copy of itself offset by 0.35 of the diagonal — it returns its
  input: 81 924 v / 163 840 f in and out, `np.array_equal` on **both** buffers, with all **1 176**
  colliding triangles still colliding. Inert at every configuration probed: both
  `SelfIntersections.Settings.Method` values (`Relax`, `CutAndFill`), `relaxIterations` 0 / 5 / 20 /
  100, `maxExpand` 1 / 3 / 10, `subdivideEdgeLen` at 1.0 / 0.5 / 0.1 of the mean edge,
  `touchIsIntersection=False` and `mimicPatch=True`.

  **It needs a single-component input, and that is the part no signature says.** On the 512-face
  `torus_self_intersecting` it does mutate — 512 → 2 512 faces — and *still* does not clear: 64 →
  **128** colliding by its own detector. The benchmark's fixture is two welded copies of one sphere,
  so two components, and it declines outright. Its sibling `mm.fixSelfIntersections` (the voxel
  path) has no such limit and genuinely repairs the same input, 163 840 → 82 052 faces with 0
  intersecting, which is why only the `local` cell is skipped.

  Two general lessons, both of which the rule above already states and neither of which was applied
  here. The benchmark's callable returned `numValidFaces()` and asserted `> 0` — **a no-op passes
  that**, and it read as a 4.3-5.0x triwarp loss (127.8 ms, the largest single row in the suite)
  against triwarp's real 1 176 → 126 repair. And the earlier "both libraries reduce, and neither
  clears" measurement had counted only triwarp's side with a detector; **apply the same detector to
  both outputs**, which is what `tests/test_repair.py` already does correctly by using MeshLib as a
  *detector* rather than as a fixer.

  **The fix was the fixture, not a skip — which is the better move whenever a reference declines an
  input rather than being wrong about it.** Skipping the cell would have dropped a comparison;
  giving the group a single-component input restored one. `benchmarks/meshes.py` now registers a
  self-intersecting **torus** (`tangle` axis, `tangle_torus_small` / `tangle_torus`, tube wider
  than hole), where all four cells do real work and the row is like-for-like for the first time:
  triwarp's local path goes from **3.14x behind** at 8 192 faces to **1.03x** at 163 840, and the
  voxel path wins 4.8-5.8x throughout. So the suite's largest single loss was neither a real loss
  nor an unmeasurable one — it was the wrong input.

  **And its behaviour is fixture-dependent in both directions, so neither reading generalizes.** On
  the trimesh-built torus the local fixer *clears* the self-intersections; on the MeshLib-built
  16x16 torus `tests/test_repair.py` uses it makes them **worse** (64 intersecting faces in, 128
  out, subdividing 512 faces into 2 512 — that figure had been recorded as "281" and is corrected
  at the site); on two welded spheres it declines outright. Probe the mutator on the *specific*
  input a row or a test will use.

**Two pairings worth knowing before writing a comparison, neither guessable from the names:**
`computePerVertNormals` matches `vertices.area_weighted_vertex_normals` to **1.19e-07** while
`computePerVertPseudoNormals` matches `angle_weighted_vertex_normals` to **1.19e-07**, and each sits
**6.8e-03** from the other's partner — so MeshLib pins a weighting convention no other reference
distinguishes. And `mn.getNumpyGaussianCurvature` is the pointwise **angle defect**, which pairs with
`vertices.vertex_defects` (9.5e-07 abs / 5.2e-05 rel on `icosphere(3)`) and **not** with
`curvature.discrete_gaussian_curvature`, the Cohen-Steiner/Morvan *ball* measure.

**Licensing: MeshLib is the one reference here that is not open source.** The wheel and
`reference/MeshLib` are under AMV Consulting's *"NON-COMMERCIAL & education"* agreement — a
terminable, non-transferable licence, with a separate commercial licence required otherwise, and an
explicit bar on modifying or transferring the Software. That is a stronger constraint than libigl's
`copyleft/` subtree or pymeshlab's GPL, because it restricts *use* rather than distribution, and
triwarp ships `MIT OR Apache-2.0`.

**Nothing under `triwarp/` may name MeshLib at all.** Not the library, not a function (`fillHole`,
`positionVertsSmoothly`, `triangleAspectRatio`), not a source file (`MRMeshDelone.cpp`,
`MRTriMath.h`). 89 such references were removed in one pass across 19 files; they had accumulated as
ordinary attribution and collectively read as a claim that a package shipped under
`MIT OR Apache-2.0` is derived from a proprietary one. Describe what the code **computes**, or name
the algorithm in the literature's vocabulary — "the Liepa/Klincsek interval DP", "the Delone
empty-circumcircle test", "circum-radius over twice the in-radius" — which is what a reader needed
anyway. Read `reference/MeshLib` to understand an operation's *interface* and its parameters; never
to port its body. **Keep it a test/benchmark dependency**, where naming it is correct and required.
**Check 21 enforces this** rather than trusting a remembered grep — the rule has failed twice, once
at 89 sites and once at five, two of which said "port of" in the imperative. Its pattern is wider
than four symbols, because those matched **none** of the three file-name comments the eighth kernels
pass found: it keys on `meshlib` / `mrmeshpy` / `mrmeshnumpy`, on an `MR<CamelCase>` prefix
generically, and on the C++ identifiers that carry no `MR` at all (`FanOptimizer`,
`buildLocalTriangulation`, `positionVertsSmoothly`, `calcQueueElement_`, `updateBorderQueueElement_`).
Add to that pattern when a new symbol is found; do not narrow it.

#### pymeshfix (`_pmf`)

nanobind over Marco Attene's MeshFix / the TMesh kernel. Build every `PyTMesh` through
`tests.conversions.numpy_to_pymeshfix` / `trimesh_to_pymeshfix` / `warp_to_pymeshfix` and read one
back with `pymeshfix_to_numpy`, `pymeshfix_intersecting_faces` or `pymeshfix_face_remap` — the
converters exist because the raw calls are unsafe in three separate ways.

It is the **narrowest deep** reference: `dir(PyTMesh)` is 19 members, of which **nine are
algorithms** (`fill_small_boundaries`, `select_intersecting_triangles`, `strong_degeneracy_removal`,
`strong_intersection_removal`, `clean`, `remove_smallest_components`, `join_closest_components`,
`fix_connectivity`, plus module-level `clean_from_arrays`) against MeshLib's 246 — and all nine are
*repair*, which is where triwarp has 20 public functions. It performs one operation no other
reference does end to end: arrays of a broken digitised surface in, a single watertight solid out.
Do **not** plan a comparison for anything else: there is no curvature, geodesic, parametrization,
registration, reconstruction, decimation, remeshing, point-cloud, boolean, proximity or
signed-distance entry point, and the C++ names for several (`cutAndStitch`, `iterativeEdgeSwaps`,
`loopSubdivision`, `isInnerPoint`, `openToDisk`, `marchIntersections.cpp`) are in the headers and
**unbound**. Thirteen hazards, all measured:

- **`load_array` is not a load; it is already a repair, and it renumbers.** It runs the kernel's
  connectivity fix and Euler update before returning. On `icosphere(1)` (42 v / 80 f): a trailing
  *or* interior unreferenced vertex is dropped; an exactly duplicated face is **kept** and the
  non-manifold edges it creates are cut instead (80 → **81 f**, 42 → **45 v**), while a *reversed*
  duplicate is refused and the vertices are still cut; two coincident *referenced* vertices are
  **not** merged; one backwards face is rewound and *every* face backwards is left alone (volume
  −3.6587 in and out), because consistent is not outward. On the scan meshes `bunny_decimated` loads
  as **8 372 v / 16 220 f** from 8 171 / 16 301 and `bunny` as **34 834 v / 69 451 f**; `dragon` is
  unchanged. On a non-orientable closed surface it cuts the orientation-reversing seam — `boy`
  1 483 → **1 559 v** at an unchanged 2 964 faces — leaving two coincident sheets where the surface
  had one, which is why `select_intersecting_triangles` reads 436 there against triwarp's 177 and the
  number is not a disagreement. So **every comparison must be index-free** (positions, canonically
  sorted rows, sets, counts); where a face index is unavoidable, go through `pymeshfix_face_remap`,
  which checks the load first and refuses rather than guessing. The flip side is that `load_array` is
  itself an oracle — for `remove_unreferenced_vertices`, for `make_winding_consistent`, and partly
  for `split_non_manifold_vertices`.
- **One `PyTMesh` serves one load and one mutating call.** A second `load_array` raises
  `RuntimeError`, and every algorithm mutates in place and returns a status, a count or an array.
- **`select_intersecting_triangles` returns a mostly-uninitialised array.** It allocates `(n, 3)`
  `int32` and writes the `n` face indices into the **flat** prefix, leaving `2n` entries of heap
  garbage. On two `icosphere(2)`s translated 1.2 apart: shape `(72, 3)`, a flat prefix of 72
  ascending indices all below 640, and `arr.max()` reading **30 751** — a value that varies between
  processes. `out.ravel()[: out.shape[0]]` is the only defined read. The prefix *is* deterministic,
  so a naive `np.array_equal(out1, out2)` reports nondeterminism that is not there.
- **`tris_per_cell` and `justproper` are no-ops on ordinary input.** `tris_per_cell` ∈ {10, 50, 200}
  crossed with `justproper` ∈ {False, True} all return **72**. Pass both explicitly at those values
  so a wheel that starts honouring either one fails a test rather than drifting, and do **not** build
  a triwarp flag around `justproper`.
- **`nbe` is inclusive, both its docstrings say otherwise, and pymeshlab's counterpart is
  exclusive.** `fill_small_boundaries(nbe, …)` fills loops of **at most** `nbe` boundary edges where
  the C++ comment and the Python docstring both say "less than". On a 24-edge rim: `nbe` 23 → **0**
  patched, 24 → **1**, 25 → 1; `nbe = 0` means all. This is the one place two references disagree
  about the *same* parameter — on a 16-edge rim pymeshfix fills at `nbe = 16` and pymeshlab only at
  `maxholesize = 17` — so `holes.fill_small(max_edges=...)` follows pymeshfix (the precedence rule
  below) and a pymeshlab comparison passes `max_edges + 1` as its named class-B transform.
- **The "MeshFix could not fix everything" line on stderr is printed when it *succeeded*.** The
  wrapper does `if (result) cerr << …` where `result` is *true only if the mesh was completely
  cleaned*. `clean_from_arrays` printed it for both `bunny_decimated` and `bunny` and both outputs
  are watertight with χ = 2, while `clean()` on a clean `icosphere(3)` returns **True**. The message
  is inverted, `set_quiet` does not suppress it, and **nothing about it may be used as a signal** —
  read the boolean, or read the mesh.
- **`remove_smallest_components` ranks by face count, not area or diameter, and returns the number
  removed.** On three disjoint spheres — 80 f, 320 f, and 80 f at radius 10, much the largest by area
  and diameter — it removed **2** and kept the **320-face** one. It always reduces to exactly one
  component; that is the rule `repair.remove_small_components(keep_largest=True)` defaults to.
- **The output face buffer is a reordering, even when nothing was repaired.** `icosphere(2)` round
  trips with **byte-identical float64 vertices** and an identical triangle *set* under
  `np.sort(rows, axis=1)` plus a lexsort, but the rows come back in a different order and each starts
  at a different corner. Never compare face buffers positionally.
- **`n_boundaries` is a property in 0.18.1, and `boundaries()` raises.** The older Cython wheel
  exposed `boundaries()`; the nanobind one keeps the name bound only to raise, and `n_points` /
  `n_faces` became properties in the same change. Code written against an example older than 0.17
  fails with `TypeError: 'int' object is not callable`.
- **`strong_degeneracy_removal` measures degeneracy in `double`, so it is *stricter* than triwarp's
  `float32` test rather than merely different.** On a flat 12-column strip: exactly collinear
  vertices are removed by both (24 v / 22 f → 0 / 0), and the same strip offset by `1e-9` is removed
  by triwarp and **kept unchanged** by pymeshfix. Where the degeneracy is exact the two agree
  completely — a sphere with zero-area faces appended comes back at 162 v / 320 f, watertight, χ = 2,
  volume 4.0470 from both — so compare on an *exactly* degenerate fixture and pin the near-degenerate
  class as the divergence.
- **`strong_intersection_removal` is a different algorithm from
  `repair.fix_self_intersections(method="local")`, not a different tuning**, and no transform rescues
  the pair: on the 16x16 self-intersecting torus triwarp cuts and refills each sheet and ends with
  **two** closed components (χ = 4, volume −10.42, 528 v) where pymeshfix removes far more and ends
  with **one** (χ = 2, volume −6.53, 80 v), two-sided surface distance 0.50; on `bohemian_dome` 2.23.
  All they share is the post-condition, so that pair is neither benchmarked nor a parity claim. The
  comparable level is the whole pipeline: `repair.make_solid` against `clean_from_arrays` agrees to
  **3.11e-08** and returns **8 188 v / 16 372 f from both sides** on `bunny_decimated`.
- **Reproducing `clean_from_arrays` needs the loader's repair as an explicit first stage.** It is
  invisible in the C++ pipeline because `load_array` does it, and its absence is invisible in the
  output too until you check the right predicate: without `remove_unreferenced_vertices` +
  `make_winding_consistent` + `split_non_manifold_vertices` first, `bunny_decimated` comes back with
  χ = 2 and one component and is **not watertight**, because nothing downstream looks at edge
  manifoldness. Two more orderings measured rather than reasoned: the component filter has to run
  *inside* the intersection loop as well as before it (cutting a band out can disconnect the
  surface), and **nothing geometric may run after the final fill** (filling a 3-vertex rim makes one
  sliver, a degeneracy pass deletes it and reopens the rim, and the two trade the same 122 faces for
  ever: χ = 2 before, χ = −56 after, and stable there).
- **`trimesh.slice_plane`'s output is a poor input** — see §7.3.

**Benchmark rule: on most rows the load *is* the row.** Because a `PyTMesh` takes exactly one load,
the build has to sit inside the timed callable (`BenchCase.new_tmesh_pmf()`), so every pymeshfix row
prices the load — 67.9 ms on `bunny_decimated` and 439.6 ms on `bunny`, against 64.2 / 435.3 ms for
`select_intersecting_triangles`, 5.8 / 51.4 ms for `fill_small_boundaries` and 6.7 / 60.5 ms for
`remove_smallest_components`. **Create a `pymeshfix` row only where the operation is at least ~30 %
of the round, and state the measured share in the group docstring.** By that rule the intersection
family (49-50 %) and `clean_from_arrays` (68-73 %) are timed, and the hole-fill (8-10 %) and
component-removal (9-12 %) comparisons carry
`pytest.mark.parity(<group>, "pymeshfix", benchmarked=False, reason=…)` with the ratio in the reason.
Cap it at `bunny`.

**Licensing: pymeshfix is GPL-3.0**, and the TMesh headers under `reference/pymeshfix/src/` carry
Attene's dual licence — GPLv3 *or* a commercial agreement with IMATI-GE/CNR. This is a **different**
constraint from MeshLib's, and conflating the two over- or under-restricts:

| | MeshLib | pymeshfix |
|---|---|---|
| May `triwarp/` name it? | **No** | **Yes**, in `Notes` / `See Also` — pymeshlab is GPL and is named throughout |
| May `triwarp/` be derived from its source? | No | **No** |
| May `tests/` and `benchmarks/` import it? | Yes | Yes |
| Why | proprietary, restricts *use* | copyleft, restricts *distribution of derivatives* |

Read `reference/pymeshfix/src/` for the interface and the parameters, never for the body; cite the
**paper** rather than the file when porting — Attene, *"A lightweight approach to repairing digitized
polygon meshes"* (The Visual Computer 26, 2010) for the repair pipeline and the component-joining
rule, Liepa, *"Filling holes in meshes"* (SGP 2003) §3 for the density refinement, Barequet & Sharir
(1995) for the loop pairing — and keep `grep -rnE 'MeshFix|Basic_TMesh|TMesh|_meshfix' triwarp/`
empty. Prose mentions of `pymeshfix` itself are allowed there; C++ symbols are not.

**And the precedence rule, which is the part a future author most needs and would never guess:**

> **Where pymeshfix and MeshLib both answer a question and their answers differ, triwarp's default is
> pymeshfix's answer and MeshLib's is reachable by a flag** — not the reverse. Where only MeshLib
> answers it, nothing changes.

The reason is not preference. A function whose *behaviour* was pinned against MeshLib alone is a
function whose specification lives in a proprietary binary nobody may read; pymeshfix's source is
mirrored and readable by anyone. The one place this currently bites is `holes.fill_small`, which
takes a **perimeter** threshold because MeshLib's `fillHoles` does, where pymeshfix and pymeshlab
both take a **boundary-edge count** — two of three references cannot express the incumbent signature.

#### pytorch3d (`p3d_ops` / `p3d_loss` / `p3d_structures`)

Build meshes with `tests.conversions.trimesh_to_pytorch3d` / `numpy_to_pytorch3d` /
`warp_to_pytorch3d`, clouds with `points_to_pytorch3d` (the `loss` container form) or `points_to_torch`
(the bare batched tensor the `ops` entry points take), and read back with `pytorch3d_to_numpy`.

It is the **first and only reference with CUDA kernels of its own**, so it takes *two* `LIBRARIES`
rows (`pytorch3d-cpu` / `pytorch3d-cuda`) and a `triwarp-cuda` vs `pytorch3d-cuda` ratio is the one
GPU-against-GPU comparison in the suite. It is also the first reference triwarp **already cited in
its own prose without testing against**: `metrics.py`, `registration.py` and `kernels/metrics.py`
named it nine times, so two public functions had their specification pinned to a library nothing in
the suite ran. Fifteen hazards, all measured:

- **Everything is batched, with a leading minibatch axis, and the wrap fails silently.**
  `knn_points(p, q)` handed a bare `(P, 3)` reads it as `(N=P, P1=3, D)` and compares three points at
  full speed — no exception, and a *faster* wrong answer. A single cloud goes in as `x[None]` and its
  answer comes out as `result[0]`, which is what `points_to_torch` is for, and every comparison
  asserts the reference's output **shape** before its values.
- **Every neighbour and Chamfer distance it returns is squared.** `knn_points().dists`,
  `ball_query().dists`, `loss.chamfer_distance` and both `point_mesh_*` scalars. Take the square root
  before comparing (measured **0.0** afterwards for `knn_points` on the host, 1.19e-07 on CUDA where
  its own kernel is a different reduction order).
- **`torch.cuda.is_available()` is the wrong probe for the CUDA extension.** A wheel whose arch list
  stops short of the device returns `True` and then fails every kernel with *"no kernel image is
  available"*; a build that compiled the CPU extension only — which is what `setup.py` selects
  whenever it finds no `CUDA_HOME`, the normal state of a CI runner — raises `RuntimeError: Not
  compiled with GPU support.` on a CUDA tensor. `benchmarks/conftest.py` gates its `-cuda` row on
  launching a two-point `knn_points`, which is the only probe that distinguishes them.
- **`Meshes` and `Pointclouds` are immutable *caching* containers** — the opposite end of the scale
  from a `ml.MeshSet`. Every `ops.*` / `loss.*` entry point is pure, so **one object serves many
  comparisons**; but `verts_normals_packed`, `edges_packed`, `faces_packed_to_edges_packed`,
  `laplacian_packed` and `faces_areas_packed` are memoized on first request, so a *benchmark* row
  naming one has to build the container **inside** the timed callable or it reports nothing. And the
  answer lives on the `*_packed()` accessors, never in the constructor arguments.
- **It does not cast for you, and one entry point casts anyway.** A float64 `Meshes` keeps float64
  through `verts_packed()`, so an unconverted reference compares a float64 answer against triwarp's
  float32 one and reads as triwarp being wrong by ~1e-7 — hence float32 in the converters. The
  exception is `ops.mesh_face_areas_normals`, whose C++ kernel returns **float32** whatever it was
  handed. Faces are stored as int64 regardless.
- **`corresponding_points_alignment` and `iterative_closest_point` are row-vector.** They solve
  `s·X·R + T = Y`, so `R` is the transpose of `registration.procrustes`' linear block divided by the
  scale (2.54e-07 / 2.62e-07); `T` needs no transform. Compare the **converged transform and the
  rmse**, never the iteration count.
- **`ops.cot_laplacian` returns two conventions in one call and neither is guessable.** Its
  off-diagonal is **twice** triwarp's half-cotangent table and its diagonal is identically **0.0**
  (not merely small) where `laplacian.cotmatrix` assembles the row sum; and its second return is
  `1 / inv_areas == 3 * M_ii`, the *reciprocal* of three times the barycentric lumped mass. Both
  cancel out of `mesh_laplacian_smoothing`'s ratios. `ops.laplacian` writes **-1** on the diagonal
  where `laplacian.laplacian(equal_weight=True)` writes 0, and `ops.norm_laplacian` **is**
  `laplacian(equal_weight=False)` before its row normalization — same formula, same literal
  `eps = 1e-12`, measured 1.49e-08 after dividing by the row sum. **It also never coalesces** — see
  §16.6 for why the `cotmatrix` ratio is a scope mismatch and the rewrite it invited was declined.
- **`ops.marching_cubes` does not exist**: it is not re-exported from `pytorch3d.ops`, only from
  `pytorch3d.ops.marching_cubes`. With `return_local_coords=False` it emits lattice indices, which is
  `levelset.marching_cubes`' own default and needs no convention fix at all — the one reference of
  five that does not.
- **`ops.cubify`'s three `align` modes are one uniform scale and translation apart.** On a 6³
  occupancy sphere, `"topleft"` / `"corner"` / `"center"` return the **identical** face buffer and
  vertex count with bounding boxes `[-0.6, 1.0]`, `[-0.667, 0.667]` and `[-0.8, 0.8]` — all three
  reachable through `voxels.from_cells(cells, voxel_size, origin)`, so an `align=` keyword would be a
  second spelling of one that exists. It also **compacts**, which is what exposed
  `to_boxes(cull_internal=True)` returning the whole corner lattice.
- **`packed_to_padded` / `padded_to_packed` bounds-check nothing and corrupt the heap.** A mismatched
  `max_size` or `total_size` does not raise — it writes out of bounds and the process dies in
  `malloc`/`free` much later. Their `first_idxs` are **starting indices, not counts**, which makes
  them `array.pack_1d_arrays`' `offsets` unchanged (measured `[0, 3, 8]` from both sides for lengths
  `(3, 5, 2)`); read the sizes off the buffers rather than passing literals.
- **`sample_points_from_meshes` caps the face count at 2²⁴.** It draws the face index with
  `torch.multinomial`, whose category limit is 16 777 216, so `lucy`'s 28 055 742 faces raise
  `RuntimeError` rather than sampling. `happy_buddha`'s 1 087 716 are fine.
- **`add_points_features_to_volume_densities_features` is `[-1, 1]` local space, `[z, y, x]` storage,
  and `rescale_features=True`.** Its volume is `(minibatch, channels, D, H, W)` and a point's
  `(x, y, z)` indexes `(W, H, D)`, so its lattice is the **transpose** of `voxels.splat_onto_grid`'s;
  `align_corners=True` is triwarp's `bounds`; and the default `rescale_features` divides by
  `density.clamp(min_weight)`, which makes both sides an *average* rather than an accumulation. With
  those three lined up the two agree at exactly **0.0** on the host and 4.8e-07 on CUDA.
- **`mesh_normal_consistency` counts pairs, not adjacencies.** It enumerates every pair of faces
  sharing an edge, so an edge with `k` incident faces contributes `C(k, 2)` terms;
  `adjacency.face_adjacency` keeps only edges with *exactly* two faces. The two agree exactly on
  edge-manifold input (0.0155947 over 480 pairs) and read **0.777 against 0.0** on three faces
  sharing one edge.
- **`ops.taubin_smoothing` rebuilds its operator every half-pass**, from the current positions, and
  row-normalizes it — where every triwarp smoother assembles once. One fixed operator sits
  4.1e-03 / 6.5e-03 / 9.9e-03 from it at 1 / 3 / 10 of its iterations, the size of the displacement
  itself; `smoothing.filter_taubin(recompute=True)` closes that to 2.4e-07 at **~23x** the cost. Its
  `num_iter` counts lambda-mu **pairs**.
- **`ico_sphere` is *not* the rotated-frame hazard open3d's Platonic solids are.** It starts from the
  identical `(±0.5257, ±0.8507, 0)` table `creation.icosphere` uses and subdivides the same way, so
  the positions correspond one-to-one and the 5.8e-05 residual is pytorch3d's table being *written*
  to four decimal places. Match by nearest vertex with a bijection check at 1e-4, not by index.
  `utils.torus`, by contrast, takes the **minor** radius first and builds its vertex table in a
  Python double loop, so it needs a real parameter mapping and its benchmark column is a per-vertex
  Python floor.

**Its CPU rows are Θ(N²) on anything with a neighbour query**, because `_C` carries no spatial
structure on *either* device. Measured: `knn_points` 299.8 / 1 189.5 / 4 562.8 ms and
`chamfer_distance` 557.8 / 2 319.1 / 9 077.1 ms at 10 k / 20 k / 40 k self-queries — 3.84-4.16x per
doubling, extrapolating to ~3.5 s per `knn` round on `bunny` and **~9 minutes per round** on
`dragon`. So a `pytorch3d-cpu` neighbour or chamfer row is capped at a feature mesh; the `-cuda` rows
run the scan meshes.

**And the GPU ratio is a crossover, not a bar — the most useful thing this reference measures.**
Brute force with perfect coalescing *beats* a BVH descent while the whole problem still fits the
device's bandwidth: 2.31 ms against triwarp's 3.29 at 20 000 points (`knn_points`, **0.70x — a loss
for triwarp**) and 74.65 against 0.87 at 200 000 (**85x**). So the neighbour and chamfer groups need
the point count as an **axis**; a one-size row reports whichever side of the crossover it landed on.

**Every CUDA row must synchronize torch's stream.** `wp.synchronize_device` synchronizes *Warp's* and
says nothing about torch's, so a `pytorch3d-cuda` row synchronized the Warp way times the launch and
not the kernel. `BenchCase.run` branches on `kind == "pytorch3d"` for exactly this. **And
`torch.cuda.empty_cache()` belongs in that teardown** — torch never frees device memory, and 16
triwarp rows once failed to allocate 65 368 bytes on a 32 GB card after an uncapped pytorch3d row
(§15.4).

**Licensing: pytorch3d is BSD-3 (Meta) and torch is BSD-3**, so there is no subtree to avoid, no use
restriction and no copyleft. `triwarp/` may name it and may be derived from it with attribution. It
stays a test and benchmark dependency all the same: nothing in the shipped package imports torch, and
it must not start, since `torch` is a 2.5 GB install for a library whose whole premise is Warp.

### 7.7 Where the reference put the answer

A reference that looks like it *disagrees* is more often one whose result was read from the wrong
place, or which was handed something other than what you thought. Two general rules, each the
generalization of several measured cases above:

- **Ask what a new reference was handed, and whether it finished the job, before reading a ratio.**
  Two instances in one round: `chamfer_backward` (the *inputs* differed) and `cot_laplacian` (the
  *output* is a different object — an uncoalesced COO tensor with no diagonal, §16.6).
- **A reference's zero is not always "off"**, and a reference's warning is usually about *our* input.
  A `RuntimeWarning: invalid value encountered in divide` from `trimesh/triangles.py:659` was
  reported as "reference-library numerics on a stochastic reconstruction" and was **wrong**: that
  line divides by `d1 - d3`, which expands to the squared length of edge AB, so it is a zero-length
  edge and nothing else — `screened_poisson(point_weight=0.0)` was returning 27-64 zero-area
  triangles, every run. Two things made the wrong reading comfortable: the stack frame was entirely
  inside the reference (nothing in the traceback named triwarp), and the warning was *intermittent*
  (3, then 1, then 0 across three suite runs), which reads like reference flakiness — but it was our
  output that varied, and **the degenerate faces were present every single run**; only whether a
  query point's nearest triangle happened to be one of them varied. **Read the reference's source at
  the warned line, solve for what input makes it fire, then assert that property on the buffer we
  passed** — that check is a better detector than the warning, because it fires on every run, and it
  is the right regression guard (`face_nondegenerate_mask`, not `-W error::RuntimeWarning`).

---

## 8. Tooling and local validation

Two tools are configured in `pyproject.toml` and are the authority for style and typing — do not
hand-roll equivalents or add competing tools. Run them, and the tests, after any change to `triwarp/`
and before considering work done.

### Ruff — lint + format

Ruff is the **only** linter and formatter (no black/isort/flake8). Config lives in `[tool.ruff]`; do
not override it inline.

```bash
uv run ruff format triwarp tests      # format (100-col, skip-magic-trailing-comma)
uv run ruff check --fix triwarp tests # lint + autofix
```

- Enabled rule families and per-file ignores are in `[tool.ruff.lint]` — respect them; do not add
  blanket `# noqa` to silence a rule the config selects. If a rule is genuinely wrong for a
  construct, prefer a scoped `per-file-ignores` entry over inline suppression. The one standing
  exception is §1.6's kernel accumulator, which needs `# noqa: UP018, RUF046`.
- Import conventions are enforced (`[tool.ruff.lint.flake8-import-conventions.aliases]`) — see §7.6
  for the reference-library aliases.
- Docstrings are enforced (`D`), so every public function needs one (§6).
- `reference/` is excluded from both lint and type-checking — never edit vendored code to satisfy a
  tool.
- **Ruff's `F811` is a dead-code check, not a collision check.** After merging modules (or their test
  / benchmark files), two same-named definitions collide silently and only one is linted: a duplicate
  `def` trips `F811` **only if the first binding is unused**; a duplicate `def` whose first binding
  *is* used before the redefinition is **invisible** and still changes behaviour, since module-level
  `def`s all execute at import so the *last* definition wins for every call site including those
  written above it (two `_sources_wp` helpers in `benchmarks/`, one cached and one not, silently
  dropped the cache); and duplicate module-level **constants** are never linted at all. **After any
  module merge, run an AST scan for repeated top-level `FunctionDef` / `ClassDef` / `Assign` names,
  and confirm the test count is unchanged across the merge** — that is the check that catches a
  silently deleted test.

### basedpyright — type checking

basedpyright (a Pyright superset) is the configured type checker — it is what the IDE runs, so match
its verdict. Config lives in `[tool.basedpyright]`.

```bash
uv run basedpyright
```

- **`triwarp/kernels/` is excluded.** The Warp kernel DSL is not modeled by any stubs and is
  inherently un-typecheckable. Do not try to make kernels type-clean or add `# pyright: ignore` there.
- **Warp-stub type-flow rules are disabled** (`reportArgumentType`, `reportCallIssue`,
  `reportReturnType`, `reportAttributeAccessIssue`, `reportIndexIssue`, `reportOperatorIssue`,
  `reportGeneralTypeIssues`). Warp's Python-scope stubs are weak (`wp.empty` typed as returning
  `array[float]`; the `twt.Array2d*` aliases not assignable to `array[Unknown, int]`;
  `BsrMatrix.offsets/.columns/.values` absent from the stub), so these rules fire almost entirely on
  false positives. **Consequence: basedpyright will *not* catch genuine argument/return/index type
  errors in wrappers** — rely on the §7 regression tests for correctness, not the type checker. It
  also cannot see a cross-module private call that goes through an attribute path
  (`tw.holes._mean_rim_edge_length`).
- **`reportPossiblyUnboundVariable` is kept as an error** — it catches §1.4's conditional-scope
  gotcha. When it fires on a *correlated* condition (two separate `if is_mesh:` blocks), fix it the
  way `triwarp/registration.py` does: initialize to `None` before the branch and
  `assert x is not None` at the use site. Do **not** suppress it.
- The gate is expected to stay at **0 errors**. An error is almost always either a real
  possibly-unbound bug or a missing dependency — resolve it, do not widen the disabled-rule list.

### Full environment

Test/reference dependencies live under `[dependency-groups] test`, **not** `[project] dependencies`
(triwarp itself only needs `warp-lang`). There is no `[tool.uv] default-groups`, so a bare `uv sync`
does **not** install them and will uninstall trimesh/pytest:

```bash
uv sync --all-groups
```

`uv add <pkg>` puts a package in main `dependencies` by default — use `uv add --group test <pkg>`.
Prerelease pins need `--prerelease allow`; trimesh is pinned `>=5.0.0rc1` because the crack-free
`trimesh.remesh.subdivide_to_size` (the reference for `triwarp.remesh.subdivide_to_size`) only exists
in the 5.0.0rc prereleases — stable 4.x is the older T-junction "soup" variant.

Running basedpyright in a dev-only env yields spurious `reportMissingImports` on `meshio` and other
test-group packages.

**`plans/` is gitignored** — plan documents are local working notes and never appear in a commit.

---

## 9. Performance work: measure before you change

The methodology is here; the *numbers* are Part II (§13 cost model, §14 kernel-shape verdicts,
§15 measurement traps, §16 component status).

- **Something suddenly slow is a Warp rebuild until proven otherwise — check that first.** Before
  profiling anything, before believing a kernel got slower, before deleting a test for being slow.
  See §15.1 for the two cheap confirmations and the measurements.
- **A benchmark lands before the optimization does.** Never restructure code for speed without a
  `benchmarks/test_<module>.py` group timing the *current* implementation first. A belief about where
  the cost sits ("two Python loops", "too many derived launches") is a hypothesis until that group
  exists and prints a number.
- **Attribute a cost with *one measurement of the thing itself*.** A projection built by subtracting
  two measurements of *different* things is a hypothesis, and in this repo it has been optimistic by
  **3-10x every time** (§15.2).
- **Attribute at the *benchmark's own* operating point.** A kernel's share of the cost can reverse
  the sign of an optimization between two points on the same axis — grep the benchmark for how it
  derives its parameter and profile at *that* value (§15.3).
- **Read device time before diagnosing anything.** Six of triwarp's ten biggest losses are 92-99 %
  host-side launch/allocation cost, not slow kernels (§16.1).
- **But first ask whether the function graph-captures, because `wp.timing_begin` cannot see a
  replayed kernel and will report a device-bound function as ~100 % host** (§15.10). That covers
  triwarp's own five capture sites *and* every CG solve, since `warp.optim.linear` captures by
  default. Three functions were attributed backwards this way. Disable the capture, measure there,
  and carry the device total back.
- **Before proposing an optimization, read what *calls* the thing — the decline may already be
  written there.** A scan reads bodies; a measured decision is prose, and it lives at the call site,
  in a constant's comment, or in the benchmark's docstring rather than in the function. All three
  items worked in one plan round were refuted by text already in the file the item proposed changing
  (§15.5). Grep the caller, the constant and the benchmark docstring for a number first.
- **Attribute a change only with a back-to-back A/B in one session.** Saved baselines drift ±10 %
  (±30 % under 100 µs) between sessions, so a comparison against a stored number proves nothing.
  Run old and new in the same session, in the same process order — and **use a detached git worktree,
  never `git stash`** (§15.6).
- **Interleave A and B in one loop, and read the `min`.** Timing all of A then all of B lets GPU
  clock state decide the winner. Report the `min` alongside the median: a one-off Warp kernel compile
  or a scheduler hiccup inflates a median but cannot deflate a minimum (§15.7).
- **Verify values, not just timing.** This is §3.4's gather warning generalised: the *wrong*
  implementation is frequently the faster one, because it reads less. Every perf change must keep its
  parity / regression test green, which means a function about to be optimized needs one first.
- **CUDA is the target; a CPU regression is an acceptable price for a GPU win.** triwarp exists to
  run geometry on the GPU, so when the two devices disagree, **decide on the CUDA number.** Do not
  reject a device-side change because Warp's CPU backend is slower at it — Warp's CPU reductions run
  ~1 lane per block while NumPy's are vectorized C, so a host readback plus NumPy wins on CPU almost
  every time and would veto nearly every reduction if it were allowed a vote. Still *measure* both
  (the CPU path must stay correct, and the ratio belongs in the comment), and still decline a change
  that wins nowhere. Worked example: §14.5. **"No gain on CUDA" is the reason to decline, not "slower
  on CPU."** The exception where the *device* becomes part of the branch condition is an algorithm
  that does asymptotically more work to expose parallelism — see §14.7.
- **A share that falls as the input grows is a decline, not a small win.** The saving is largest
  exactly where the call is already cheap. Report the share at more than one size before deciding; a
  single operating point cannot show the trend.
- **A decline is a result — write it at the site, with the number, and resolve every site the finding
  named.** The exemplary case is `kernels/neighbors.py`, where the `wp.length_sq` swap was measured,
  declined, and the reasoning written into the source; the counter-case is the same finding's other
  two sites in `ball_pivoting`, which were neither converted nor annotated and were re-derived a pass
  later. A finding that names five sites is closed when all five are converted **or** annotated.
- **A written decline can expire because a *neighbour* got faster.** `greedy_downsample_mask`'s own
  comment declined a rewrite and said *"revisit if a benchmark ever puts it on top"*; a
  level-synchronous RDP then took `polyline_simplify` from 84 ms to 0.68 ms and **inverted the
  module's ordering**. Re-read the declines in a module after any big win in it.
- **A decline written in one place is not a decline applied everywhere, and the tree's own source is
  the census you should not trust.** `kernels/reduce.py::_reduce_1d_tiled` has documented "the
  generic form is declined here, measured ~18 us per launch" since it was written — and 34 of that
  module's kernels took the generic default anyway, because the factory's `dtype` parameter had one
  (§16.0). **When a finding is a property Warp computes rather than one the source spells, take the
  census from the runtime**: an AST scan of annotations found 39 of the 71 generic kernels, and
  `wp.get_module(name).kernels` found all 71.
- **A probe that instruments the thing it measures must carry a third, do-nothing arm.** A
  monkeypatched `wp.launch` reported two wrappers *regressing* under a change that was in fact worth
  1.15-1.28x on them: the patch added its own Python frame to every launch in one arm only, and
  silently failed to specialise non-array generic parameters. Adding a "same wrapper, no
  substitution" control separated the two, and a detached-worktree A/B (§15.6) settled it. **Instrument
  both arms identically, or measure without instrumenting.**
- **Re-probe a tuning constant after a Warp upgrade.** The hole-DP block-size reading *"measured flat
  between 32 and 128"* was taken on Warp 1.16 and does not hold on 1.17.
- **Sweep both devices before writing a single tuning number**, and if the two optima differ by more
  than a tolerance, split the constant rather than averaging (§13.3).
- **Count the launches a numerical-method change adds per iteration before proposing it.** A cycle
  here is launch-bound, not flop-bound; any smoother that buys iterations with launches loses
  (§14.8).

---

## 10. Warp API reference mirrors

Authoritative Warp function lists are mirrored locally under `reference/warp_api/`:

| File | Scope |
|------|-------|
| `reference/warp_api/builtins.md` | Built-ins usable inside `@wp.kernel` / `@wp.func` (`wp.<name>`) |
| `reference/warp_api/warp.md` | `warp` module API at Python scope |
| `reference/warp_api/sparse.md` | `warp.sparse` BSR/CSR matrix API |
| `reference/warp_api/utils.md` | `warp.utils` Python-scope utilities |
| `reference/warp_api/fem_linalg.md` | `warp.fem.linalg` linear-algebra utilities |

BEFORE using an unfamiliar Warp builtin, sparse, or utils function, `grep` these files to confirm the
exact name, signature, and scope rather than guessing. Each file stamps the Warp version it was
transcribed from, and its source URL, at the top — fetch the URL for full argument details when the
one-line description is insufficient. Run `uv run reference/warp_api/warp_version.py` to compare every
stamp against the installed `warp-lang` and see which files an upgrade has left stale.
`reference/warp_api/REGENERATE.md` records how to re-extract them.

**A name in `dir(wp)` that is missing from those mirrors is usually hidden on purpose, not missed by
the transcription.** `dir(wp)` exposes ~510 names against ~137 the package uses, and browsing the
remainder for adoption candidates is how the `wp.dense_chol` / `dense_subs` / `dense_solve` family
gets proposed as a replacement for a hand-written 6×6 Cholesky. Introspect before planning around one:

```python
from warp._src.context import builtin_functions
f = builtin_functions["dense_chol"]      # a Function, the same handle §2.7's factories capture
print(f.hidden, f.doc, f.input_types)    # True  'WIP'  {n: int32, A: array(ndim=1, float32), ...}
```

`hidden: True` / `doc: "WIP"` is the answer, and the mirrors' silence was the same answer read one
step earlier.

**Then check the *quantity*, not the name — adopting a matching builtin is sometimes a regression.**
Also check the signature's *storage class* and its precision: a builtin that only speaks `float32`
cannot serve the `float64` half of a dispatch, so half the hand-written code stays either way. And
where a builtin *does* fit, the argument is often single-source-of-truth rather than speed —
`wp.volume_index_to_world` measured perf-neutral (1.08x at 200k voxels, 1.005x at 2M, both
launch-dominated) against a hand-rolled half-voxel transform that agrees with it to 3.58e-07; adopt
it for the convention, and if only half the sites can convert, **name the split in the module
docstring** rather than leaving two silent conventions in one file. The measured adoptions and
rejections are §12.8.

**When a plan proposes swapping in a Warp geometry builtin, sweep the *scale* axis and the degenerate
cases, not just random inputs at unit scale** — unit-scale random pairs showed perfect parity and
would have hidden every defect §12.8 records.

---

## 11. Running long commands: never poll with `until`

The benchmark suite and the measurement probes §9 asks for routinely run for minutes. **Do not write
a wait loop around them.** A backgrounded command re-invokes you when it exits, reporting its exit
code and its output file path. Launch it, do something else, and read the output file when the
notification arrives.

- **`until ! pgrep -f <name>; do sleep 5; done` never terminates.** `pgrep -f` matches the *full
  command line of every process*, including the polling shell itself, whose command line contains
  `<name>`. The condition is therefore permanently true. Measured cost of not knowing this: **seven
  such loops in one session, each spinning for 3-4 hours** until killed by hand, every one redundant
  because the command it was watching had already sent its completion notification.
- **A watcher is never the record.** The benchmark's or probe's own stdout file is. If you find
  yourself launching a second command to learn whether the first finished, delete it.
- **When a poll genuinely is unavoidable** — external state the harness cannot see — make the pattern
  unable to self-match (`pgrep -f '[p]robe_p4'`) or test a sentinel file the process writes on exit,
  and give the loop a **bounded** iteration count. **The bracket trick is not sufficient on its
  own**: this harness runs a command as `zsh -c '… && eval '\''<your command>'\''…'`, so the pattern
  string appears *literally* in the invoking shell's command line and `pgrep -f` matches that shell
  even though it does not match the pattern. Measured: `pgrep -af '[p]ytest'` returned exactly one
  row, its own wrapper. **Prefer the sentinel file, and read a `pgrep` hit's command line before
  believing the process it names is real.**
- **Do not chain short sleeps** to approximate a long wait. Pass a longer `timeout`, or background it.

A foreground command that outruns its timeout is *also* moved to the background and notified the same
way, so exceeding a timeout is not a reason to start polling either.

**Do not run a timing probe while `pytest` or `mkdocs` is running** — the same reconstruction read
527-752 ms under contention and 295-400 ms on a quiet box.

---
---

# PART II — MEASURED FACTS

Everything below was measured on this box (**RTX 5090**, 170 SMs) against the Warp version stamped at
each item. Read the relevant section before proposing an optimization, diagnosing a slowdown, or
re-deriving a number.

---

## 12. The Warp platform: bugs, quirks and version status

### 12.1 Memory safety — every failure here is silent

- **A `wp.launch` with `device=` omitted is memory corruption, not a style nit.** It resolves to the
  *default* device — `cuda:0` whenever CUDA is present — while the array arguments sit on the CPU. It
  raises nothing, loads the module on `cuda:0`, and returns **numerically correct results** while
  corrupting the host heap. Isolated with `device=` as the only variable in a standalone script (no
  triwarp): **9/20 aborts omitted, 0/20 explicit, 0/20 with `CUDA_VISIBLE_DEVICES=""`**; a further
  reduction reached 29/30 on 1.16.0 and 20/20 on 1.15.0.

  **That cross-device launch is a supported configuration, not a validation gap.** On an HMM box
  (`is_cpu_memory_access_from_gpu_supported == True`) the GPU may legitimately dereference `malloc`'d
  host memory, so `RELAXED` and `CHECKED` both permit it by design; `CHECKED` validates
  *addressability* only. What the launch does **not** acquire is ordering or lifetime: it is
  asynchronous, so the host arrays are freed at scope exit while the kernel is still reading them.
  Three variants settle it — launch then `os._exit` with nothing freed and nothing read: **0/20**;
  launch then an explicit `del` with no sync: **20/20**; the same `del` after `wp.synchronize()`:
  **0/20**. A `wp.synchronize()` after the launch is therefore the *fix*, not a workaround.
  **The asymmetry, measured not read:** the identical free-while-in-flight pattern is **0/20** with
  the arrays on `cuda:0` and **20/20** (`free(): invalid pointer`) with them on the CPU, because a
  CUDA array's storage is freed with the stream-ordered `cudaFreeAsync` whereas
  `CpuDefaultAllocator.deallocate` is an immediate `free()` ordered against nothing. Nothing at the
  call site says which. `pinned=True` was immune only because `wp_free_pinned` calls the
  synchronizing `cudaFreeHost`. The reverse mismatch (CUDA arrays, `device="cpu"`) segfaults
  immediately (GH-1693). Filed as GH-1766.
- **An out-of-bounds kernel write on the CPU device *is* glibc heap corruption**, because a CPU Warp
  array is host heap. `graph.successor_cycles` deferred its range check to a downstream call while
  `scatter_successor` ran first: `tests/test_graph.py` on CPU was 6/6 aborts before the check moved
  and 8/8 clean after. Guarded by a test that counts `wp.launch` calls — the old test asserted only
  the `ValueError`, which was raised either way, so it passed on the broken ordering.
  `connected_component_parity_from_edges` still range-checks nothing, which is the same shape without
  the guard.
- **Neither of the two was "the Warp CPU backend corrupts the heap"** — the framing that survived
  three reports and months of subprocess isolation. The separate-process discipline is not needed for
  either cause any more.
- **Two bisection techniques, which are the transferable part.** `wp.config.mode = "debug"` compiles
  kernel-side bounds checks; a violation trips `Assertion failed: 'i >= -arr.shape[0] && i <
  arr.shape[0]'` at `warp/native/array.h`, and it took the OOB scatter from "unexplained" to a named
  line in one run. **But a clean debug run is evidence about timing, not about correctness**: it
  reported zero assertions against the old pipeline, where enough work followed the launch that the
  kernel finished before teardown, and **10/10** against a reduced repro whose arrays are freed
  immediately. The launch-device bug fell to a **component-swap bisection**: run the standalone
  (clean) pipeline and swap in one library component at a time — `laplacian`, `reduce.max` and the
  volume reduction were 0/20, the normals path 7/20, then `face_normals_and_areas` 0/20 and
  `mean_vertex_normals` 9/20, and diffing that one function against its inline twin left exactly one
  difference. **Swap components, not lines.**
- **Two dead ends, so they are not re-run:** the resident-JIT-module-count hypothesis is refuted
  (0/16/48 synthetic CPU kernel modules: 0/20 each), and a probe against *current* triwarp says
  "fixed" on both 1.15.0 and 1.16.0 — it cannot distinguish a Warp fix from our trigger moving, so
  any re-probe must pin the library revision and vary only the Warp version. That false negative put
  a wrong FIXED row in the workaround table.
- **A `wp.Mesh` with zero triangles silently corrupts CUDA allocator state.** The constructor
  "succeeds" (`mesh.id` valid, `wp.synchronize()` clean) but the *next* unrelated CUDA allocation
  anywhere later in the process fails with `out of memory` despite 30+ GiB free and cascades into
  `CUDA error 700`. Point count doesn't matter — only `indices.shape[0] == 0`. Safe on `cpu`; safe
  with at least one triangle. Still broken on 1.16 (10/10 subprocesses died on the first 4 MiB alloc
  after). Never construct one on CUDA, **including in tests** — use a single-triangle mesh to reach
  an `n_faces < 2` guard. `triwarp.mesh.Trimesh.warp_mesh` builds a `wp.Mesh` unconditionally and
  would hit this for a zero-face mesh — a known latent issue, not fixed.
- **Python-scope gather silently ignores a non-contiguous index view's stride.** Measured with
  `payload[i] == i` and `edges = [[0,10],[1,11],[2,12],[3,13],[4,14]]`: `edges[:, 0]` itself is
  correct (`[0 1 2 3 4]`, `strides=(8,)`, `is_contiguous=False`) but `payload[edges[:, 0]]` returns
  `[0. 10. 1. 11. 2.]` — the flattened buffer's first five entries — and `payload[edges[:, 1]]`
  returns the same offset by one. `wp.copy(dense, edges[:, 0])` first, then `payload[dense]`, is
  correct. `wp.map` over such a view inherits the corruption **and is faster** (15.7 µs against
  26.7 µs on 900k edges), because it reads a contiguous prefix — so a benchmark alone reads as a win.
  Audited 2026-07-26: every existing Python-scope gather in `triwarp/` passes a full array or a
  contiguous prefix. Rule: §3.4.
- **`wp.copy` into a *pinned* host buffer is a genuine async memcpy with no event, so reading it
  races.** `wp_memcpy_d2h` is a bare `cudaMemcpyAsync` on the current stream, and CUDA only blocks the
  host when the destination is **pageable**. Measured behind a 140 ms kernel: a pinned scratch read
  the *previous* value **20 of 20** times; an unpinned one 0 of 20; the package's own slice spelling
  0 of 20. A plan had prescribed pinned at 11.4 µs against 28.9 — that 11.4 µs was the cost of a racy
  read. The corrected table (20 calls between two syncs, min of 15) is **device-split**, which is why
  the helper branches rather than picking a portable spelling:

  | spelling | CUDA | CPU |
  |---|---|---|
  | `int(arr[n - 1 :].numpy()[0])` — the old idiom | 27.8 µs | 6.6 µs |
  | `int(one_element_device_array.numpy()[0])` | 22.6 | — |
  | **`wp.copy(unpinned_scratch, arr[k : k + 1])` + `.numpy()[0]`** | **15.7** | 9.5 |
  | `wp.copy(pinned, ...)` + `.numpy()[0]` | 12.1 **(WRONG)** | — |
  | pinned + `wp.synchronize_device` | 14.7 | — |
  | **`arr.numpy()[k]`** (host `.numpy()` is a zero-copy whole-buffer view) | (copies all) | **1.7** |

  Pinning buys nothing even done correctly (14.7 against 15.7). `triwarp._device.read_scalar` is
  this, with a per-dtype scratch cached at module level; measured `array.flatnonzero(200k)`
  114.1 → 96.2 µs on CUDA, 1.01x on CPU. **Not reentrant** — the scratch is shared.

  **And the shared scratch has a second, silent hazard the scalar dtypes hide: for a vector or
  matrix dtype `.numpy()[0]` is a *view*, so two sequential reads both alias the one cached row and
  the first takes the second's value.** Found by converting `creation.sweep_polygon`'s two endpoint
  readbacks, which then decided every open path was closed; `read_scalar` now copies before
  returning (and on the host branch the view is onto the caller's own buffer, where a caller writing
  through it would corrupt the array). Pinned by
  `test_read_scalar_returns_a_detached_row_for_a_vector_dtype`, whose mutation probe confirms it is
  the *vector* assert that fails without the copy. **The general shape: a helper that caches one
  buffer per dtype is safe for as long as everything it returns is a scalar, and the day someone
  passes a `wp.vec3` it is wrong without an error.**
- **Warp's CPU work runs ~36x slower in a process where CUDA has been initialised.** Measured on one
  `heat_signed_distance` call, same mesh, same code, only `CUDA_VISIBLE_DEVICES` differing: `STRICT`
  50.57 s vs **1.40 s**, `RELAXED` 50.34 vs 1.40, `CHECKED` 49.77. **It is CUDA *presence*, not the
  launch-access guard** — all three modes measure the same, so `tests/conftest.py` keeps `STRICT` for
  free. **It is not GPU contention either**: re-probed with the GPU at 0 %, 49.57 / 49.52 s visible
  against 1.42 / 1.34 s hidden — 35x, unchanged. `CUDA_VISIBLE_DEVICES` must be set before the
  process starts, so this can only ever be a second process, never a fixture or context manager
  (§7.2).

### 12.2 `wp.launch_tiled` runs one lane per block on the CPU

Still true on **Warp 1.17**. `wp.launch_tiled(kernel, dim=[...], block_dim=64)` executes **one thread
per block** on the CPU backend — `wp.tid()`'s lane index is always 0. Measured with a counting
kernel: `dim=[4]`, `block_dim=64` runs 256 threads on `cuda:0` and **4** on `cpu`, max lane index 63
vs 0.

**The obvious probe says "fixed", and that is the trap.** `wp.tile_load` reads its whole tile out of
an array, is lane-independent, and was **never** affected — `wp.tile_sum(wp.tile_load(...))` totals
512.0 on *both* devices over 8 blocks of 64 ones. Only `wp.tile(x)`, built from *per-lane* values,
collapses: it totals **8.0** on CPU against 512.0 on CUDA. **Probe the lane-constructed tile**, or
you will conclude it is fixed and delete a correctness branch.

Two silent consequences, no exception either way: a `wp.tile_max(wp.tile(scalar))` block reduction
returns that block leader's contribution (a max over 256 shuffled floats returned 185, the max of
every 64th element); and even a kernel with **no** tile intrinsics breaks if it relies on lanes
covering the tile (`idx = tile_i * TILE_1D + t` touches only every 64th element).

Two defects this hid, both because `tests/conftest.py`'s `device` fixture returns `cuda:0` whenever
CUDA is available: `convex_subset_mask` was silently unsound on CPU (an under-estimated support max
widens the marking test, so it marked interior points), and `holes.fill_dp_span_tiled` minimized over
every 32nd apex and returned an equal-count, plausible, **wrong** triangulation — 42 of 44 triangles
differed from the serial engine.

**The fix is one token: stride by `wp.block_dim()`.** It reads the launch's `block_dim` on CUDA and
**1** on CPU, costs nothing, and makes the single CPU lane cover every element, after which the
`wp.tile_min(wp.tile(...))` reductions degenerate to one-element tiles that correctly return that
lane's own answer. Byte-identical on both devices and measured neutral on CUDA (400-vertex loop,
three alternating pairs: min 10.07/10.25/10.19 ms against 9.98/10.23/9.77). **It does not generalize
to *partitioned* kernels** — `measures.centroid_tiled` computes `f = i * TILE_1D + t` at
`dim = n_blocks`, so one lane per block drops 63 of every 64 faces and no stride change reaches it;
that would need the kernel to loop over its block's range, which is why the `centroid_sliced` /
`centroid_tiled` pair and `_device.prefers_tiled_reduction` stay.

**The *statement* of the rule mattered more than the rule.** Four kernels carried comments
generalizing this to "a `wp.tile_*` reduction is wrong on the CPU backend", which forbids the correct
form and blocked two measured wins. The one-element CPU tile is harmless; what breaks is a partition
stride that is not `wp.block_dim()` — and that form is wrong on **CUDA too** (§2.2's table). A tile
reduction also works fine inside a `@wp.func` (probed generic over `wp.Float`, both devices,
`block_dim` 1/32/64/256), which is what made `array.tile_argmin` possible.

**Do not file this upstream and do not re-probe it blind.** `wp.launch` documents `block_dim` as
"always 1 for cpu devices", `launch_tiled` forces it before appending the trailing lane dimension,
and Warp's tiles guide states the consequence. Open issues: **NVIDIA/warp#1480** (*CPU/GPU parity for
all tile code*, which names `wp.tile(lane_value)` followed by reductions as an affected pattern) and
**NVIDIA/warp#1638** (*Add efficient CPU block execution with fibers*). The branch becomes removable
only when CPU blocks run more than one logical thread; both numbers are cited in the
`_device.prefers_tiled_reduction` docstring.

**Warp exposes no grid-wide barrier** (no cooperative groups, no `__threadfence`), and a spin-wait
emulation over `atomic_*` needs all blocks co-resident and a fence Warp cannot spell. That is the
ceiling on every level-synchronous rewrite (§14.9).

**Warp's hash grid has no per-cell entry point.** `wp.HashGrid` exposes only `hash_grid_query` /
`hash_grid_query_next` (a sequential per-thread iterator) and `hash_grid_point_id`, so there is **no
way to hand lane `t` cells `t, t+32, …`** — the cell *walk* cannot be split across lanes, only the
per-candidate arithmetic can. Measured: the walk is **70-73 %** of the per-candidate cost, flat from
64 to 8 171 queries, so a cooperative search that keeps `wp.HashGrid` is capped at **1.37x**. **Never
propose a warp-per-edge search that keeps the hash grid.** Warp *does* ship a block-cooperative BVH
walk (`tile_bvh_query_aabb` / `tile_query_valid` / `tile_bvh_query_next`) — §14.2 — and it has a
silent correctness bug, next.

**`wp.tile_bvh_query_aabb` returns out-of-range primitive indices once a traversal round overruns
its result buffer, and every caller must bound-check what it hands back.** Read from
`warp/native/tile_bvh.h` (1.17): a round appends each hit with an **unconditional**
`atomicAdd(&query.result_counter_shared_mem[0], 1)` and guards only the *write* against
`result_buffer_capacity = WP_TILE_BLOCK_DIM * 5` — **160** at the `block_dim=32` both triwarp
consumers use. The consumer then reads
`result_buffer_shared_mem[counter - block_size + lane_id]`, so as soon as the counter has run past
the capacity that index is past the written region and a lane is handed **uninitialised shared
memory as a primitive index**. The stack has the same shape at
`stack_capacity = 64 * BVH_QUERY_STACK_SIZE`, dropping children instead. Nothing raises, and the
`>= 0` test every documented example uses does not catch it — a garbage word is positive about half
the time and arbitrarily large.

Confirmed rather than read: `proximity.mesh_to_mesh_distance` on `lucy` against a translated copy,
7 007 straggler faces reaching `face_to_mesh_distance_tiled`, dereferences one at
**12.26 GB past the nearest allocation** — `compute-sanitizer --tool memcheck` naming that kernel
and that load, 7 errors, and **0 errors** with a `candidate < n_target_faces` test added (the run
genuinely instrumented: 278 ms against 30 ms uninstrumented, §12.10's rule). It is deterministic at
that face count and **not** allocator-sensitive — 6/6 with the CUDA mempool on and 6/6 with it off —
which is what distinguishes it from §16.3's `ball_pivoting` fault.

Three consequences:

- **Bound-check the index at every `tile_bvh_query_next` site**, `candidate >= 0 and candidate < n`.
  Both of triwarp's do. It is output-neutral by construction — an out-of-range index is never a
  primitive of the BVH — so it can only reject what was already garbage.
- **The guard stops the corruption and cannot restore the dropped primitives.** An overrun round
  silently loses hits, so a guarded walk may return an incomplete candidate set; that half is
  upstream's. In practice the two triwarp callers survive it because each has a *second*, sound
  bound on the answer — the global running minimum, and the pivot's own acceptance test.
- **The obvious attribution was tested and is wrong.** §16.3's long-open intermittent
  `ball_pivoting` `CUDA error 700` is *not* this: its own repro faults **6 of 6 with the guard in
  place**, on the same cloud. So this rules the garbage-index path out there rather than closing it.

**A query box grown by a distance bound is what reaches the overrun**, which is why this had gone
years unseen: the round count scales with how many primitives one box meets, so a big mesh plus a
generous box is the trigger and a tight box on a small mesh never gets near 160.

**Warp exposes no node-by-node BVH traversal** either (`bvh_query_aabb` / `bvh_query_ray` /
`bvh_query_sphere` / `bvh_get_group_root` only), so a BVH-pair wavefront means writing our own
hierarchy — price it as that, not as a rewrite of the query.

### 12.3 `wp.ref[T]` requires concrete types

Verified by compile probes; still broken on 1.16.

- **Generic type-vars do not instantiate inside `wp.ref[...]`** — `wp.ref[wp.Scalar]`,
  `wp.ref[wp.Int]` and `wp.ref[typing.Any]` all fail resolution *even for float32* ("Couldn't find
  function overload"). `wp.Any` does not exist. Concrete `wp.ref[wp.float32]` / `wp.ref[wp.int32]`
  work.
- **No `@wp.func` name-overloading**: a second `def foo` just shadows at Python scope, and
  `wp.overload()` is **kernels-only**. So one helper name cannot serve both float32 and float64.
- **Tuple-assignment swap** (`a, b = b, a`) works only for **simple local variables** (like `sort3`),
  NOT for array/matrix/vector elements — `arr[i], arr[j] = ...` raises *"Multiple return functions
  can only assign to simple variables"*.

Consequence: the shared argmin/argmax/swap helpers in `kernels/array.py` are concrete `float32`
(`update_argmin`, `update_argmax`, `update_argmax_lowest_index`, `update_argmax_vec3`,
`update_argmin_pair`). Float64 sites and int32 array-element swaps keep hand-written loops.

### 12.4 Numerics and precision

- **`wp.Float` covers `float16` / `float32` / `float64` from one generic definition** — that is what
  let `array.allclose` drop its hardcoded `float32` path. **`wp.Scalar` does NOT instantiate for
  `wp.bool`** (`TypeError: Function <name> does not support the provided argument types
  warp._src.types.bool`; it does work for `int8` / `int32` / `float32`), so a "works on masks and on
  numbers" wrapper needs a dtype branch, not one generic func.
- **Scalar arguments must be constructed at the input's precision**: `wp.map(f, a_float64,
  wp.float32(tol))` will not resolve the generic — use `a.dtype(tol)`. Literals *inside* a generic
  `@wp.func` need `type(x)(...)`.
- **There is no `wp.any` / `wp.all` over vector components** and no generic vector annotation beyond
  `Any`, so per-component predicates stay as named per-type funcs behind a dtype-keyed dispatch
  (`is_close_vec3`).
- **FMA fusion makes a degenerate triangle's area ~1e-8 on CUDA and exactly 0 on CPU.**
  `triangle_cross` computes `wp.cross(v1-v0, v2-v0)`; for a face with a repeated vertex the two edges
  are numerically identical, so the cross is exactly `0` on CPU but ~`1e-8` on CUDA, because Warp's
  `fuse_fp` is ON by default and fuses `a*b - c*d` into `fma(a,b,-(c*d))`. That flips a zero-area
  face to "nondegenerate" against the absolute `TOLERANCE_MERGE = 1e-8` altitude test at unit scale.
  **Decision: leave the imprecision — do NOT disable `fuse_fp`.** Disabling it would kill FMA fusion
  across the whole triangles kernel module (closest_point, barycentric, normals — proximity/sampling
  hot paths), and because `@wp.func`s compile under each *calling* kernel's module options it would
  have to be disabled in every module using `triangle_cross`. **So write degeneracy tests with
  scale-aware inputs** — scale vertices to ~1e-2 so genuine altitudes stay far above `1e-8` while the
  FMA residual stays far below it — and never assume CPU/GPU bit-agreement on zero-area faces.
- **`wp.mesh_query_point_no_sign` + `wp.mesh_eval_position` is not an exact closest-point query.**
  Measured on a remeshed `icosphere(3)` against `trimesh.proximity.closest_point` on the *same*
  float32 positions: worst-case disagreement **2.1e-05** absolute, mean 1.6e-08, always Warp
  reporting the *smaller* distance. It is a **fixed point** — clamping a vertex to distance `d` from
  Warp's reported closest point and re-querying returns the same answer (re-clamp moves ≤1.5e-07), so
  iterating does not converge. Consequence for anything building a distance *bound* on it
  (`isotropic_remesh`'s `max_deviation`, `_reproject_pass`): the bound is exact against Warp's own
  query (≤1.4e-07) but only approximate against an independent float64 one — 1.00x / 1.01x / **1.37x**
  the bound at bounds of 1.07e-03 / 3.21e-04 / 1.07e-04. **How to test such a bound:** assert
  exactness against `proximity.closest_point_on_mesh` (the same query, so the implementation's real
  contract), and assert the *improvement ratio* against the independent oracle rather than the bound
  itself. Asserting `trimesh_distance <= bound` fails at tight bounds and the failure is not a bug in
  the code under test.
- **`wp.length(d) < r` and `wp.length_sq(d) < r*r` are not the same predicate in float32** — 10 rows
  of 200k disagree at the boundary, the same 10 on CPU and CUDA. Rule: §2.4.
- **`NaN` breaks a binary search, and which way it breaks depends on the search's convention.**
  Every comparison against `NaN` is false, so a midpoint landing on one is steered by a predicate
  that never fires. `searchsorted(side="right")`'s `values[mid] > value` then advances `left`, i.e.
  *into* Warp's float radix sort's trailing `NaN` block, and the search returns the array length for
  every finite value; `side="left"`'s `wp.lower_bound` is steered away from it and is correct for
  every finite value while landing `NaN` itself at slot 0. Neither raises. Where a `NaN` can only
  belong to the last slot — which is exactly a sorted-unique table — the fix is the left search plus
  `if index >= n or values[index] != value: index = n - 1`, since `NaN` compares unequal to
  everything including itself. §16.5 has the case this cost.
- **float32 storage sets a hard noise floor that no solver tolerance reaches.**
  `tangent_space.halfedge_transport_angles` (and `halfedge_tangent_angles`, and `triangles.face_angles`
  under them) are float32, and that storage precision — not the CG tolerance, not the frames — sets
  the noise floor of everything built on `laplacian.connection_laplacian`: **8.7e-09 of the field
  maximum** on `cave_cube`. Two traps: **`vertex_tangent_frames` is not the lever**
  (`connection_laplacian` never reads it — its phases come from the transport angles, and the frames
  only name the basis the answer is *reported* in, so a "make the frames float64" plan spends memory
  and changes nothing); and **the noise floor overlaps real signal**, so no magnitude threshold
  separates them (`half_torus` resolves genuine directions down to 8.0e-10 of its maximum, *below*
  the 8.7e-09 of round-off). Report it with a mask instead — which is why
  `heat.transport_tangent_vectors` returns `(transported, resolved)`.

  **The three-probe sequence that settles "is this solver noise or input precision?", in order of
  cost:** (1) sweep the solver tolerance — flat ⇒ not convergence (1e-8 → 1e-14 left it at 8.7e-09 →
  1.0e-08); (2) inject noise into the suspected input and check linearity, extrapolating back to the
  unperturbed point — landing on ~1e-7 relative names float32; (3) split storage from arithmetic —
  redo in float64 but round the result back to the shipped dtype; unchanged ⇒ it is the *storage*,
  and an internal-only fix cannot work.
- **Diffused heat fields scale as 1/scale² with the mesh coordinates**, so any absolute tolerance
  applied to one is a silent wrong answer on a rescaled mesh (43 of 97 vertices zeroed at scale 1e5
  before the fix).

### 12.5 Kernel-scope cast semantics

All measured on Warp 1.16 (probe scripts written to a file — Warp refuses `exec()`-defined kernels).

- **`int(x)` and `wp.int32(x)` are the same operation.** Generated C++ is identical apart from the
  call name (`wp::int32(var_0)` versus `wp::int(var_0)`), and the latter only compiles because Warp
  writes an **unconditional** `#define int(x) cast_int(x)` / `#define float(x) cast_float(x)` into
  every module header (`warp._src.codegen.cpu_module_header`) — `wp::int(x)` is not valid C++. There is no cost to either spelling. The one
  recorded difference is in Warp's own type system: `int`'s `value_type` is Python `int` and
  `float`'s is Python `float`, where `int32`/`float32` carry the Warp types.
- **That difference bites exactly once, and it is fatal.** `total / float(count)` inside a
  `wp.Float`-generic `@wp.func` does **not** silently narrow — it fails to parse: `Input types must
  be the same, got ['float64', 'float32']`. So every `float()` in a kernel is a hard commitment that
  its enclosing function will never be made generic.
- **No cast on a thread index is load-bearing.** A bare `wp.tid()` passes unchanged to a `wp.int32`
  `@wp.func` parameter, to a `wp.Scalar`-generic parameter, and into a kernel-scope slice. All five
  spellings compile and agree. As of 2026-08-16 triwarp carried 329 `int(wp.tid())`, 98 redundant
  `int(...)`, 85 redundant `wp.int32()`/`wp.float32()`, and **46 sites in 41 kernels that do both**.
- **`//` is not CPython's `//`** — §1.5 has the table.

Rules: §1.3, §1.5, §1.6.

### 12.6 Compilation, module hashing and import cost

- **A generic kernel's lazy overload instantiation rebuilds its whole module** — §2.5 carries the
  measurement (206 → 87 module loads, suite 1 033 s → 29 s).
- **`wp.map` forks its generated module per call *signature*, on axes wider than the dtype** — §3.5
  carries the measurement (182 → 143 loads; cold cache 2.1x).
- **`import triwarp` cost 0.603 s because `@wp.kernel` builds an `Adjoint` at import time** — 553
  decorated kernels plus ~110 generated by `kernels/reduce.py`'s factories. `-X importtime` self
  times: `kernels.reduce` 85.6 ms, `kernels.algorithms.ball_pivoting` 73.3, `kernels.intersection`
  68.7, `kernels.remesh` 34.2, `kernels.creation` 32.9. **Importing one submodule did not avoid it** —
  Python imports a parent package before its child, so `import triwarp.edges` ran
  `triwarp/__init__.py` and measured the same 0.586 s. Fixed with a PEP 562 module `__getattr__`:
  `import triwarp` 0.603 → **0.001 s**; `import triwarp.edges` 0.586 s / 56 kernel modules → 0.050 s
  / 5; `import triwarp.mesh` 0.589 / 56 → 0.022 / 0. Details in §16.2.
- **A deferral one module makes can be silently cancelled by an unrelated module's top-level
  import.** `triwarp/reconstruction.py` defers `import warp.fem` on purpose and its comment measured
  the cost — but two *kernel* modules were loading the whole package eagerly for two `@wp.func`s, so
  the deferral saved nothing. **Check `sys.modules`, do not trust the comment.** Isolated:
  `import warp.fem.linalg` **0.24-0.29 s** against `import warp._src.fem.linalg`
  **0.008-0.010 s**, because the public module is a re-export shim that runs `warp/fem/__init__.py`
  and pulls in `adaptivity`, `dirichlet`, `domain`, `field.*`, `geometry.*`. End to end, interleaved
  A/B with a `git checkout` between reps: `import triwarp` **1.49 → 1.18 s** median. So the tree uses
  **`warp._src.fem.linalg`**, with a comment naming the version probed —
  `kernels/reduce.py`'s `warp._src.context` import is the same bet and the stated precedent. The risk
  is that an upgrade moves `warp._src`; both sites fail loudly at import. (No fem *codegen* is
  triggered — these inline — but the inlined QR raised `kernels.curvature`'s cold-cache build from
  1.4 s to 5.7 s, and runtime was neutral.)
- **Only host readbacks block a CUDA graph capture.** `array_scan`, `bsr_from_triplets`,
  `radix_sort_pairs` and a nested `wp.capture_while` all capture and replay fine on Warp 1.16.
  `edges_unique` / `flatnonzero` / `remove_unreferenced_vertices` each read back the count that sizes
  their own output — that raises `CUDA error 906` inside a capture — and each was replaced by the
  inclusive scan it was reading, with the count left in a device `state` array. **`warp.Graph`
  retains modules but not arrays**: keep a reference list for anything allocated inside a capture
  (probing could not reproduce corruption after dropping it, so this is defensive, not proven).
- **`block_dim=32` takes the `warp_count == 1` fast path** in `tile_reduce_impl` (a ballot plus a warp
  shuffle, no cross-warp shared round trip) — that is the whole 126-vs-369 ns gap in §13.2.
- **"Clear the kernel cache before the first run" is a no-op across an upgrade**: Warp namespaces the
  cache by version (`~/.cache/warp/1.17.0`), so a new version is cold by construction. Also
  `wp.config.kernel_cache_dir` reads `None` until init — the real path only appears in the init banner.
- **`wp.config.verbose = True` is deprecated in Warp 1.17** — it prints a deprecation notice to
  stderr, which is noise in exactly the output you are grepping. Use
  `wp.config.log_level = wp.LOG_DEBUG`; the log lines themselves are unchanged.
- **A tile `shape=` must be a plain integer, so `wp.constant(wp.int32(n))` cannot serve as one.**
  `wp.constant(256)` works as a `wp.tile_load` / `wp.tile_zeros` `shape=`; the typed spelling fails
  at *parse* time with an `AttributeError` naming the kernel. That matters because §1.5's check 17
  types an operand by declaration and only recognises an integer `wp.constant` in the typed form, so
  a constant that is also a tile shape can never be made visible to it — `algorithms/bfs.py`'s
  `BFS_SCAN_BLOCK` converted (it is only an offset and an index) and `conjugate_gradient.py`'s
  `CG_TILE` cannot, and says so at the site. A typed constant is also not usable in *host* arithmetic
  (`(n + c - 1) // c` raises `unsupported operand type(s) for //`), so the wrapper reads `int(...)`.

### 12.7 `warp.sparse`

The `nnz`-is-a-capacity rule and its consequences are §3.7. Three further behaviours, all silent:

- **`bsr_mm` returns a structural *superset* of the product.** On `A @ P0` where the true pattern
  (counted independently in NumPy) is **5 578** entries, `bsr_mm` gives **9 590**, the extra 4 029
  exactly zero. Values are correct to 1.1e-16. `topology="compact"` does not change it, and
  `"masked"` / `"padded"` raise without a `z`. scipy's product gives the true pattern. **The zeros
  are not cosmetic once the result is multiplied again**: an explicit zero at `(i, c)` makes column
  `c` "see" row `i`, so a Galerkin product `PᵀAP` inherits every aggregate reachable from it —
  **191 181 entries for a 587-row coarse operator against a true 4 084**, a 47x pattern blowup out of
  a 1.7x one, operator complexity 2.19 instead of 1.03. **The extra entries are interspersed in
  column order, not trailing capacity** (over the 639 rows carrying a zero, **0** have their zeros
  only at the row's end; columns stay strictly increasing; the padding is a variable per-row gap fill,
  mean 1.96, max 22) — so this is **not** NVIDIA/warp#1769, whose title says the opposite. **Still
  present on Warp 1.17**: a `PtAP` measures 5 458 entries against scipy's true 2 488 (2.19x), so the
  prune is part of the algorithm.
- **`bsr_compress`'s illegal-memory-access is FIXED on Warp 1.17.** On 1.16, compressing a `bsr_mm`
  result made the *next* `bsr_mm` die with `CUDA error 700` in `wp_free_device_async` — reached from
  a product's result and from the **default `inplace=False`** rather than the `inplace=True` the
  issue title names. On 1.17 it completes, so `linalg._multigrid_prune` is now **one call** to
  `wps.bsr_compress(matrix, prune_numerical_zeros=True)` instead of a CSR-to-triplets rebuild, and it
  is **~2.9x faster** on the step (0.175 ms median / 0.166 min against 0.513 / 0.462, `float64`, a
  4 000-row `PtAP`). **New trap found doing it: `bsr_compress(m)` at the documented default
  `inplace=False` returns `m` itself, pruned in place.** Measured: `result is m`,
  `result.values.ptr == m.values.ptr`, and `m.nnz_sync()` goes 5 458 → 2 488 across the call; with
  nothing to prune it also returns `src` rather than a copy. Safe at both `_multigrid_prune` call
  sites only because each passes a freshly built temporary. **Copy first if you still need the
  unpruned matrix.**
- **`bsr_set_transpose`, `bsr_mm` and `bsr_axpy` all read the `nnz` *field*, never `nnz_sync()`**, so
  a matrix whose count is stale carries garbage into whatever consumes it. §3.7 is the same rule for
  buffer sizing; this is it for *operands*.
- **`bsr_mv` takes one vector**, so a cycle over three right-hand sides pays three launches per
  mat-vec: replacing it with one hand-written batched CSR kernel took a V-cycle from **34 launches
  and 189 µs to 15 and 85**.
- **Unwritten `(0, 0, 0.0)` triplets in a `wp.zeros` buffer all accumulate on entry `(0, 0)`**, and
  `bsr_from_triplets`' duplicate accumulation costs **O(duplicates on the hottest address)**, not
  O(triplets). Measured on `smoothing.laplacian_ls_triplets` / `bunny` (244 523 triplets, only 9 377
  rows written, so ~185 000 collide), interleaved, min of 6: zero-padded **14.345 ms** →
  `rows.fill_(n_rows)` (out of range, silently dropped) **0.451 ms** — **31.79x**; `bunny_decimated`
  3.697 → 0.406 ms (9.10x); `nnz` and `sum(abs(values))` identical, so the fix is output-identical.
  §3.7 is right about *correctness* (a structural zero is harmless) and wrong about *cost*.
  **Before launching a conditional triplet writer, `rows.fill_(n_rows)`; price any conditional emit
  by its collision count, not its buffer size.**
- **`warp.optim.linear`'s `TiledDot` has three paths and `batch_offsets` picks the bad one.** It takes
  a **direct batched** kernel whenever `batch_offsets` is set and `batch_count > 1`, launched
  `dim=(columns, batch_count, tile_size)` — one block per (column, subproblem), every lane reducing
  `n / tile_size` entries serially, so the dot is `O(n)`:

  | n | batched (`batch_offsets`, 2 columns) | tiled tree (unbatched) |
  |---|---|---|
  | 4 356 | 4.55 µs | 5.09 µs |
  | 17 161 | 9.51 µs | 5.13 µs |
  | 40 962 | 18.66 µs | 5.16 µs |
  | 163 842 | **66.15 µs** | 6.06 µs |

  Two dots run per CG iteration; on `harmonic[saddle] k=2` that is **19 µs of a 41 µs iteration**,
  where the sparse matvec is only 13-27 %. `use_bounded_tree` — the escape — requires
  `batch_count == 1`, so a batched solve can never reach it. `linalg.replicated_operator` attaches
  `batch_offsets` for per-subproblem convergence, which is why `linalg._BatchedCg` exists: same
  iteration, same worst-column stopping rule, a real two-stage per-column tree — **1.10-1.50x end to
  end**, iteration counts unchanged. Two near-identical ideas that are losses: separate unbatched
  `cg` calls per column do reach the tree reduction and measure **0.60-0.81x** (a second copy of
  every other kernel in the iteration); and dropping `batch_offsets` is not a tuning change —
  `alpha` and `beta` become global rather than per column, which is CG on the block system.

### 12.8 Warp builtins: adoption verdicts

**Adopted:**

- **`wp.mesh_query_point_sign_winding_number`** → `proximity.signed_distance_on_mesh(...,
  sign_mode="winding")`. Agrees with the exact generalized winding number's sign on **100 %** of query
  points across icosahedron / cave_cube / hemisphere / half_torus / holed-sphere, where ray parity
  manages only 93.2 % on the holed sphere. Costs 1.2-1.5x the parity query and ~3x mesh device memory
  (+235 MB on dragon's 871k faces). Default stays `"parity"`. ⚠️ Two traps:
  `support_winding_number=True` is required or the builtin **silently returns the ray parity answer**
  (no error), and `wp.Mesh` does not retain the flag, so it cannot be checked from Python — that is
  why `ray.contains_points`, which takes a caller-supplied `wp.Mesh`, deliberately does *not* get the
  option while `signed_distance_on_mesh` (which builds its own) can guarantee it. Warp exposes only
  the thresholded *sign*; the value-returning `solid_angle_iterative` is not a registered builtin, so
  `proximity.winding_number` (the igl-matching value) still needs a custom LBVH.
- **`wp.bvh_query_sphere`** (Warp 1.17) in `kernels/neighbors.py` (ball count/collect, all three k-NN
  enumerations, weighted nearest) and `kernels/proximity.py::closest_point_on_edges`. It prunes on an
  **exact sphere-AABB squared-distance test**, so on a BVH of degenerate point bounds that test *is*
  the point-in-ball test and the narrow phase disappears. **It is bit-exactly `wp.length_sq(d) <=
  r*r`, not `wp.length(d) <= r`** (over 100 000 queries against 1M points: 0 rows differ from the
  squared spelling, 1 row from the sqrt spelling, identically on both devices), so adopting it *is*
  adopting the squared predicate and the hash-grid branch had to switch too. Wins: ball count 1.18x /
  1.63x / 2.69x at r = 0.01 / 0.02 / 0.05 (200k pts, 20k queries) and 1.42 / 1.96 / 2.61x at 1M /
  100k; register-row k-NN 1.14x (k=1), 1.21x (k=7), 1.58x (k=16), 1.28x (k=30), 0.94x (k=64) at 200k
  and 1.73 / 1.80 / 1.41 / **0.90** / 1.61x at 1M; 1.14-1.38x on CPU. The k-NN dip at one bucket is
  **not** the deepening sequence (identical, because the certificate compares the k-th *distance*),
  not the Euclidean `complete_radius`, and not a spill (`local_memory_size` 0 at every bucket, and
  *fewer* registers at five of six) — what is left is per-node arithmetic. **The counter-example:**
  the same conversion in `ball_pivoting.ball_is_empty` was byte-identical and **reverted as a 1.75x
  loss** (16.20/15.74/15.54 ms on the hash grid against 27.61/27.99/28.12), because a *small-radius,
  well-centred* query is the hash grid's best case — the grid was built with cell width `radius`, so
  a probe reaches 27 cells by address arithmetic where the BVH pays a ~11-level root descent per
  call, millions of times. **Convert a ball query when the enumeration radius is large relative to
  the structure and a BVH already exists; do not convert a probe whose radius equals the hash-grid
  cell width.** And do not read §14.2's tiled 2.4-8.9x as transferring — there is no
  `tile_bvh_query_sphere` in 1.17.
- **`wp.bvh_query_sphere` again, as a broad phase over *bounds* — and the reason to prefer it is
  the traversal, not the candidates it does not return.** `neighbors.query_bvh_ball`
  is the ball sibling of `query_bvh_box`, and adopting it in
  `curvature.discrete_mean_curvature` measured **2.40-4.26x on the whole public call** (harness
  medians, pymeshlab's own column reproducing within 4.4 % as the control): `sphere_small` 4.109 →
  1.553 ms, `sphere_med` 4.592 → 1.760 and 11.791 → 3.042, `sphere_large` 9.458 → 3.949 and
  26.665 → 8.531 at radius scales 2.0 / 4.0. The `sphere_small` cell flips from a 0.94x **loss** to
  pymeshlab into a 2.37x win.

  **The platform fact underneath it is worth more than the one adoption: on Warp 1.17
  `wp.bvh_query_aabb`'s traversal costs 6.5-15x `wp.bvh_query_sphere`'s per candidate returned, on
  the identical BVH.** Both are exact — checked against brute-force oracles, the cube query
  returning 228 396 candidates against the cube oracle's 228 396 and the ball query 169 164 against
  the ball oracle's 169 164 — and the ball returns only 26-30 % fewer, so the candidate trim
  explains almost none of the gap. **The control that isolates it is the *inscribed* cube**
  (half extent `r / sqrt(3)`, strictly contained in the ball, so strictly fewer candidates): it
  returns 91 164 candidates and still costs **6.5x** the ball query's time. Count-pass only,
  `leaf_size=1`, `icosphere(4)`: cube 1.468 ms / inscribed cube 0.723 / ball **0.111**. `root=-1`
  versus the default root makes no difference. So **wherever a caller's predicate is a ball, the
  cube broad phase is the wrong query even before its extra candidates are counted** — and a
  decline sized on the candidate ratio alone (this one was, at "~29 % waste, ~1 ms of the loss
  table") is sized on the smaller half. Not transferable to `ball_pivoting`'s pivot search: that
  walk is `wp.tile_bvh_query_aabb` and 1.17 still ships no `tile_bvh_query_sphere`, so converting
  it would trade §14.2's measured 4.5-4.9x tiled win for this one — **unmeasured, and the one open
  lead this finding creates.**
- **`wp.mesh_get_bvh`** (Warp 1.17) — `proximity.mesh_to_mesh_distance` now builds **one** structure
  over mesh B instead of two; §16.6.
- **`wp.volume_index_to_world`** — perf-neutral (1.08x at 200k voxels, 1.005x at 2M, both
  launch-dominated) against a hand-rolled half-voxel transform that agrees to 3.58e-07. Adopted for
  the *convention*, not the speed.

**Rejected on measured evidence — do not re-propose without new data:**

- **`wp.intersect_tri_tri` cannot replace `intersection.triangles_intersect_sat`.** Möller's
  `NoDivTriTriIsect` carries an **absolute** `EPSILON = 1e-6` applied to *unnormalized* plane
  distances, so its verdict moves with mesh scale while triwarp's SAT is scale-invariant (pure sign
  comparisons). Over 200k random pairs, disagreement was 0.00 % at scale 1 and 100, but **44.6 % at
  scale ≤ 1e-3** (every pair snaps to "coplanar") and 10-16 % at scale ≥ 1e3. There *is* a float64
  overload and it fixes the large-scale end completely but **not** the small-scale end — the epsilon
  is absolute in either precision. Also 36 % disagreement on coplanar pairs and 26 % on degenerate
  ones, precision-independent. `mesh_with_mesh` is public API on arbitrary meshes, so a small object
  measured in metres would silently return wrong results. *(Noted while measuring: triwarp's SAT
  reports **every** coplanar pair as intersecting — its `vec3_equal(normal, other_normal)` guard only
  catches bit-equal normals, and the 13 cross-product axes degenerate to zero vectors whose intervals
  trivially overlap. The wrapper's "coplanar faces produce no segments" contract still holds, but by
  accident, via NaN segments failing the length filter. Pre-existing; not fixed.)*
- **`wp.closest_point_edge_edge` cannot replace `remesh._segments_dist_sq_d`.** The native C++ path
  is the same Ericson algorithm and *does* clamp correctly (the un-clamped version in
  `_src/builtins.py` is a Python reference, not what codegen uses). Rejected because: `max_deviation`
  defaults to `None`, so the float64 branch never executes on the default path — zero upside; it is
  float32-only, measured ~20x worse relative error on near-parallel segments (2.0e-3 vs 1.1e-4),
  exactly the near-degenerate quad geometry the float64 port exists for; and it returns distance, not
  squared distance, adding a `sqrt` to the flip inner loop.
- **`wp.sample_unit_hemisphere_surface`** would replace `visibility`'s low-discrepancy Fibonacci
  lattice (whose `local[2] == dot(direction, normal)` identity the kernel depends on) with a
  Monte-Carlo estimate at the same ray count — variance where there was none, and every occlusion
  parity test would need a tolerance instead of an equality.
- **`wp.norm_huber`** is the Huber *norm* where `registration.robust_weight` needs the IRLS *weight*
  `ρ'(r)/r`.
- **`wp.tile_arange`** cannot express `array.arange` and loses where it can. Its bounds are read
  at *codegen* — `tile_arange_value_func` computes the tile length from them — so a runtime
  `block * TILE` start is a `TypeError` at parse (`unsupported operand type(s) for -: 'Var' and
  'Var'`), and the only expressible form is a constant tile shifted by a
  `tile_map(wp.add, ..., tile_broadcast(tile(start)))`. Measured that way against the plain
  `out[i] = i` kernel, values identical, `block_dim=256`, min of 9 interleaved reps of 100
  launches: **0.85x at 1 024, 0.91x at 200 192, 0.998x at 13 999 872**. A range fill is a pure
  streaming store with no reuse for a tile to exploit, and §13.1 already prices the whole
  200k-element call as launch- and allocation-bound.
- **`wp.volume_voxel_count`** is a capacity (§3.7).
- **The `dense_chol` / `dense_subs` / `dense_solve` family** is `hidden: True` / `doc: "WIP"`, and it
  takes `wp.array[float32]` where the caller holds a `wp.spatial_matrix` in registers — a 2x loss
  (§2.9).

### 12.9 `wp.Volume` as a voxel-set container

Measured first on 1.15 then re-measured on 1.16; `triwarp/voxels.py` ships all of this and its module
docstring carries the table.

**Two 1.15 facts that 1.16 falsified — do not carry them forward:** `allocate_by_voxels` works on
**CPU** on 1.16 (`rebuildable=True` included), so a volume-backed module does not have to be
CUDA-only, and `fem.Nanogrid` also works on CPU with identical counts, with `get_voxels()` row order
**byte-identical across CPU and CUDA**; and an empty point set now **raises**
`RuntimeError("Failed to create volume")` on both devices instead of aborting the process. Still
guard it, but it is catchable.

- **`Volume.allocate_by_voxels(world_points, voxel_size, translation)` deduplicates** and beats our
  dedup: 10⁶ points → 794 875 voxels (identical counts) in **0.64 ms vs `unique_rows` 1.51 ms on
  CUDA (2.35x)** and **138 ms vs 316 ms on CPU (2.28x)**. `get_voxels()` reads the `(n,3)` int32 cells
  back in 0.03 ms.
- **`volume_lookup_index(grid, i, j, k) == k`-th row of `get_voxels()`**, and `-1` when absent — so
  the grid and the cell array share one canonical numbering and a per-voxel payload is just a
  `wp.array(n_voxels)`. No side table, no hash map, O(1) membership.
- **NanoVDB centres voxels on integers**, so to make its index equal an Open3D/trimesh cell index
  (`floor((p-origin)/s)`, corners on the grid) pass `translation = origin + 0.5*voxel_size`. Without
  the shift every cell comes back **+1 on every axis** — silent, not an error.
- **`get_voxels()` order is leaf-major** (8³ leaves in lexicographic leaf order, then lexicographic
  `(x,y,z)` within a leaf) — deterministic and reproducible, but *not* globally lexicographic, so
  lexsort before comparing to a reference and never assume it matches a C-order reshape.
- **`voxel_points` may be integer.** A contiguous `(n,3)` int32 (or `vec3i`) array is read as
  *index-space* cells, so a cell array goes straight in with no float round trip.
- **`point_mask` (int32, one per point, 0 = ignore) filters during the build**, so a compaction pass
  is never needed — and a **fully-masked one-point build is the only way to make a legal EMPTY
  volume** (`active=0`, `get_voxels()` returns 0 rows) where a zero-length input raises.
- **Rebuildable volumes: traps resolved, and the payoff is not there.** `rebuild()` into a
  pre-reserved topology is only **5-7 % faster** than a fresh `allocate_by_voxels` (0.344 vs 0.369 ms
  at 179 k voxels; 0.836 vs 0.885 at 262 k) — the cost is inserting points and building leaves, not
  the allocation — so it does not pay for `dilate`-style loops. Two traps: **`get_voxel_count()`
  returns the reserved CAPACITY** and `get_voxels()` is padded with `[0,0,0]` (read
  `get_active_stats().voxel_count`, §3.7); and the four `max_*` capacities **cascade**
  (`max_leaf_nodes` defaults to `max_active_voxels`, `max_lower_nodes` to `max_leaf_nodes`, …), so
  supplying only `max_active_voxels` at 800 k voxels reserves 800 k *upper* nodes, **runs out of
  device memory**, reports "Failed to create volume", and leaves the CUDA context throwing
  illegal-memory-access on everything after. Pass all four, each ≥ the exact build's `ActiveStats`
  field — derivable a priori as leaves ≤ ⌈E/8⌉³, lowers ≤ ⌈E/128⌉³, uppers ≤ ⌈E/4096⌉³ for cell
  extent E. `status` must be `uint32`. (A plan's "silently empty topology at ×1.25" was that OOM
  cascade, not a semantic quirk.)
- **`warp.fem.Nanogrid(volume)` derives topology triwarp would otherwise hand-write** — verified
  exact on a solid 2×2×2 voxel block (8 cells / 36 sides / 24 boundary sides / 27 vertices), 1.23 ms
  to construct at 179 k voxels: `.vertex_grid` is the deduplicated corner lattice;
  `boundary_side_index()` + `side_position` + `side_normal` enumerate the outward faces;
  `side_inner_cell_index` over boundary sides is the 6-connected surface-voxel set; and
  `PicQuadrature(Cells(Nanogrid(v)), positions)` exposes `cell_particle_offsets` /
  `cell_particle_indices`, atomic-free per-voxel segment offsets. Cost of admission: import
  `warp.fem` **inside the function**, never at module scope (§12.6).

### 12.10 Upgrade discipline and the workaround table

Status of every version-stamped workaround, last full re-probe against 1.16 with 1.17 deltas noted:

| Workaround | Status |
|---|---|
| `warp.optim.linear.cg` returns NaN on CPU | **FIXED in 1.16** — cpu and cuda:0 identical to 1.7e-10 on a 64x64 SPD system. `_device.require_cuda` and all seventeen call sites deleted |
| CPU backend corrupts the process heap | **FIXED / never was Warp** — see §12.1; the repro that aborted ~60 % on 1.15 ran clean 12/12 |
| `bsr_mm` nondeterministic on a chained triple product | **REFUTED — never a Warp bug** (§3.7) |
| `bsr_compress` illegal memory access (#1769) | **FIXED in 1.17** (§12.7) |
| `bsr_mm` structural superset | **still present on 1.17**, and it is not #1769 (§12.7) |
| `wp.launch_tiled` one lane per block on CPU | **still broken on 1.17** (§12.2) |
| empty `wp.Mesh` corrupts the CUDA allocator | **still broken** — 10/10 subprocesses died on the next 4 MiB alloc (§12.1) |
| `wp.ref[wp.Scalar]` generics | **still broken** — `WarpCodegenError` at kernel parse (§12.3) |
| radix key dtypes | unchanged — int32/int64/uint32/uint64/float32/float64 only, 4- or 8-byte values. `segmented_sort_pairs` was NOT extended (int32/float32 keys only) |
| no sparse triangular solve | unchanged |
| generic `Any`-typed `@wp.func` wrappers around tile intrinsics | still fail (NVRTC "more than one instance of overloaded function"); the working route is the builtin-capture factory of §2.7 |
| `@wp.kernel(grid_stride=False)` | benchmarked as noise (±5 %, sign flips between runs) — not adopted |

**Three things that make an upgrade's verification honest, all of which default to a *false pass*:**

- **`compute-sanitizer` is at `/usr/local/cuda-12.8/bin/compute-sanitizer`, not on `PATH`**, and a
  minor version behind the toolkit Warp reports — it still works. Run through `uv` it must instrument
  two process hops (`uv` → `python`), so `--target-processes all` is mandatory. **A run that
  instrumented *nothing* also prints `ERROR SUMMARY: 0 errors`, so the proof is the slowdown**:
  measured 1.71 s uninstrumented vs 20.01 s under memcheck (**11.7x**). Always time the same suite
  both ways.
- **Both `wp.capture_while` sites sit behind `wp.is_conditional_graph_supported()`**
  (`triwarp/graph.py`, `triwarp/polyline.py`), so on a box where that returns `False` the whole
  CUDA-graph path is skipped and the tests pass green having tested the fallback. It is `True` here;
  confirm with a pytest plugin that monkeypatches `wp.capture_while` / `wp.capture_launch` and counts
  calls — measured **34 each, paired**.
- **Re-running our own repro script validates the repro, not the upstream bug.** That is exactly how
  the `bsr_mm` misattribution survived both the 1.15 and 1.16 re-probes. **When a workaround's
  justification is an upstream bug nobody else has confirmed, suspect the repro.**

**Gates for an upgrade**, all four: the full suite (1953 passed / 1 skipped at the 1.16 bump),
`basedpyright` 0 errors, `mkdocs build --strict` clean, `tests.parity` with an unchanged pair count.
The `pyproject.toml` specifier stays `warp-lang>=1.15` where no newer-only API is used — the pin
lives in `uv.lock`.

**Also re-probe any tuning constant** (§9), and note mkdocstrings renders triwarp's **source-level**
annotations, so Warp's 1.16 change to `repr()` of array annotations never reached the docs.

**Two probe-fixture traps, both of which faked a "DIFFERS" for a whole family:** drawing random input
**inside** a per-device callable hands the two devices different problems (hoist every `rng` draw to
host scope); and `slice_plane` without `merge_vertices()` leaves duplicated boundary vertices, so
harmonic / tutte / arap were being pinned to different boundaries and read as total disagreement
(§7.3).

---

## 13. The cost model (RTX 5090)

### 13.1 Host-side, per call

**Measure `n` calls between two syncs and divide.** A per-call cost taken with a sync *inside* the
loop is up to **14x wrong**, because Warp leaves `cudaMemPoolAttrReleaseThreshold` at 0, so every
sync drains the pool and the next allocation is cold: `wp.empty(36, int32)` reads **114.6 µs** with a
sync per call and **4.5 µs** with 200 calls between syncs — and it is flat in the size (64, 65 536
and 1 048 576 elements all 4.4-4.5 µs). *(That 14x reading led to reporting the mempool release
threshold as a hidden wrapper floor; measured on real wrappers, raising it to 8 GB is worth
**1.00-1.02x**.)*

| primitive (correct regime) | cost |
|---|---|
| `wp.launch` | **9.7 µs**, independent of `dim` |
| `wp.empty` | 4.5 µs, flat in size |
| `wp.zeros` | 7.6 µs |
| `wp.full` | 8.6 µs |
| `arr.fill_` | 3.1 µs |
| `wp.copy` | 5.2 µs |
| `wp.clone` | 10.8 µs |
| `wp.array(numpy)` | 16.1 µs |
| a slice view | 2.8 µs |
| `array.flatnonzero(200k)` | 115 µs |
| a cached `wp.map` call (Python overhead above the kernel) | ~11 µs |
| a host readback | ~0.1 ms |
| a **replayed** kernel in a captured chain | **1.17 µs** at n=17 689, 1.57 µs at n=163 842, exactly linear from 1 to 12 kernels |

A host-cost model of `allocations × per-call cost + kernels × 9.7 µs` accounts for **84-122 %** of a
wrapper's measured host time, verified on seven wrappers spanning 0.3-2.9 ms.

**Launch marshalling is ~1.0 µs per argument, linear, identical on both devices** (400 reps, small
`dim`, min alongside median):

| Args | CUDA median / min | CPU median / min |
|---:|---|---|
| 2 | 16.6 / 14.8 µs | 12.7 / 11.2 µs |
| 8 | 22.5 / 20.8 | 19.9 / 17.5 |
| 16 | 30.6 / 27.2 | 29.2 / 27.2 |
| 24 | 39.0 / 36.3 | 39.2 / 34.5 |
| 28 | 43.5 / 41.1 | 41.0 / 37.2 |

So the often-quoted "~32 µs per `wp.launch`" is the *mean* kernel's launch (triwarp's mean argument
count is 5.1), not a constant. A `@wp.struct` bundle collapses it: a 25-argument kernel against the
identical kernel taking one bundle measures **43.1 → 18.2 µs median (0.40x)** at `n = 1024`, 42.9 →
18.1 at `n = 100 000`, 222 → 195 at `n = 4M` — a flat ~25 µs saving at every size, which is what a
host-side cost should look like. Building the bundle costs **2.6 µs**. Rule and eligibility: §2.8.
Only 18 of 440 triwarp kernels take ≥12 arguments.

**A generic kernel costs a further ~12 µs of host-side overload resolution on every launch, which
is roughly *double* the host cost of a concrete one — and `wp.Float` / `wp.Scalar` cost exactly what
`Any` costs.** That last clause is the correction: this entry read "a `wp.array[Any]` kernel" for
several rounds, and the annotation is not the axis. `wp.launch` runs `infer_argument_types` over the
**whole argument list** and only then looks the overload up, so the cost scales with how many
parameters are generic, not with which spelling names them. Re-measured on Warp 1.17 / RTX 5090, 100
launches between two synchronization points, min of 25 interleaved reps:

| kernel | per launch |
|---|---|
| concrete, hand-written | 12.1-12.3 µs |
| concrete, factory-generated (§2.7) | 12.0-12.2 — identical to hand-written |
| one generic array parameter, `wp.Float` | 23.4-24.2 (**+11.3-12.3**) |
| the same body annotated `Any` | 24.2-24.5 — **the same** |
| three generic parameters (`triangles.face_signed_volumes`: `wp.array[Any]`, `Any`, `wp.array[wp.Float]`) | 26.6 against 12.2 concrete — **2.17x, 14.3 µs** |

Flat in `dim` (1 024 to 4M), elevated in both `min` and median, so it is per-launch and not a
first-call effect.

**The fix is one line per module and compiles nothing new: `wp.overload()` *returns* the concrete
`wp.Kernel`, and `_register_overloads()` was already calling it and discarding the result.** Keeping
it in a dtype-keyed table (`kernels/array.py::OverloadTable`, whose base `KernelTable` also serves
`kernels/reduce.py`'s factory instantiations) lets the wrapper hand `wp.launch` the resolved kernel.
The kernel source stays dtype-generic, so §1.2's preference is untouched; what changes is only which
object reaches `wp.launch`. Landed across all 42 generic launch sites in the tree — measured
end to end against a detached baseline worktree, interleaved processes, min of 3 rounds of 12 reps:

Ratios are the range over **two** such sessions, because §15.7's ±10 % drift is the same size as
several of these:

| call | before → after (µs) | ratio |
|---|---|---|
| `smoothing.filter_normals` (20 passes) | 2 030 → 1 373 | **1.46-1.48x** (also §3.5's map hoist) |
| `voxels.sample_grid_trilinear` (50k queries) | 60.4 → 38.6 | 1.35-1.56x |
| `voxels.splat_onto_grid` (200k) | 124.5 → 82.5 | 1.34-1.51x |
| `measures.volume` (81 920 faces) | 147.1 → 98.7 | 1.29-1.49x |
| `reduce.max(axis=1)` on `(60k, 3)` | 39.8 → 25.2 | 1.35-1.58x |
| `vertices.vertex_defects` | 92.5 → 62.4 | 1.33-1.48x |
| `reduce.min(axis=0)` | 50.5 → 34.9 | 1.28-1.45x |
| `array.arange(200k)` | 33.8 → 25.6 | 1.30-1.32x |
| `grouping.unique_1d(200k, inverse)` | 317.0 → 240.9 | 1.21-1.32x |
| `array.isin` / `grouping.unique_rows` / `edges.edges_unique` | 364 → 300, 641 → 526, 736 → 629 | 1.14-1.25x |
| `laplacian.cotmatrix` / `laplacian` / `mass_matrix` | 410 → 355, 381 → 339, 238 → 207 | 1.11-1.17x |
| `reduce.sum` / `minmax` / `any` (200k) | 83.9 → 69.8, 93.6 → 84.0, 122 → 109 | 1.08-1.22x |

19 of 20 probed calls improved and none regressed (`bounds.aabb` is flat at 0.99-1.00x — its kernel
was already concrete, which is the control this table needed). **A missing dtype now raises rather
than silently rebuilding the module**, which is the §2.5 failure mode turned from a clock reading
into an error naming the kernel.

**A cached `wp.map` call carries the same kind of overhead: 23.8-26.6 µs against 13.4-14.3 for the
launch it wraps — 1.78-1.86x, ~11 µs.** That is what §3.5's `return_kernel=True` hoist removes, and
it prices the hoist for any *loop*: `smoothing.filter_normals` ran two maps over 20 passes and was
~0.44 ms of pure host time. The 199 `wp.map` call sites reached once per wrapper call each pay it
too; converting those would fight §3.5's readability rule and has **not** been measured end to end.

**Per-segment packing costs are host constants and flat in the data** — the same rows measure 1.46 ms
at 0.07 MB total and 2.42 ms at 268 MB, a 4 000x range:

| operation | per segment |
|---|---|
| `wp.copy` (`pack_1d_arrays`, `concatenate`) | **6.02 µs** |
| `wp.clone` (`split(copy=True)`) | **15.06 µs** — a ~10 µs allocation plus a ~6 µs copy |
| a `wp.array` slice view (`split(copy=False)`) | **3.63 µs** |
| one whole-buffer `wp.copy`, 48 903 to 2 614 242 elements | **0.010-0.015 ms**, flat |

So 99 % of a 256-segment pack is per-call overhead; the validation/size loop is 2.6 % and the
`wp.array(offsets_list)` transfer is 0.027 ms flat — do **not** "optimize the Python side". **The
NumPy crossover is a segment SIZE, ~98 kB (~24 576 int32), and it does not move with the segment
count** (1.01x at 64 segments, 0.93x at 256): every ratio in these benchmark groups is
`98 kB / segment size`, which predicts 1.50x at 65.5 kB (measured 1.50x) and 2.40x at `dragon`'s
`n_segments=256` (measured 2.47x). Rows invert on their own axis — `dragon` at `n_segments=4` is
2.6 MB a segment and wins 0.05-0.19x. **The lever is reducing the *number* of segments at the call
site**, never the Python around them. Probed and rejected: `wp.array(list_of_arrays, ...)` (arrays of
arrays cannot be built this way, and a kernel cannot dereference a raw pointer); the slice-view
spelling `wp.copy(out[o:o+n], a)` (**2x worse** than `dest_offset=`); dropping `src_offset=` / `count=`
or passing an explicit `stream=` (neutral to 1.7x worse); the private `core.wp_memcpy_d2d` loop (2.4x
faster, so most of `wp.copy` is Python-side validation — but it is private API and skips exactly that
validation). Two further declines are written at their sites in `triwarp/array.py`: building `split`'s
views with a raw `wp.array(ptr=...)` (1.57-1.69x, but it re-implements `wp.array.__getitem__`'s
20-attribute contract and already drops the `grad` view), and giving `split(copy=True)` one shared
output buffer (3.93-4.02x, but it drops half of what `copy=True` promises).

**A device reduction costs ~0.10-0.32 ms flat on CUDA regardless of `n`** (launch + 4-byte read),
while a readback scales with bytes copied. The crossover is wherever the copy exceeds ~0.15 ms, which
measured out at ~200k `int32`, ~200k `float32`, ~1M `bool` and ~16k `vec3d` elements. Below it a
reduction launch is pure overhead; above it the readback grows without bound
(`.numpy().sum(axis=0)` over `(n_faces,)` `vec3d` moves 72 B/face — **33.6 ms on `dragon`**, against
0.19 ms for four `wp.utils.array_sum` calls).

**Under ~100k elements both reduction forms sit at the ~18 µs launch floor**, so small inputs show no
difference at all.

### 13.2 Device-side and memory access

| primitive, in-kernel, amortized over 100k iterations | cost |
|---|---|
| dependent L2 load (160 KB / 640 KB working set) | **118 / 147 ns** |
| dependent load, 64 MB working set | 678 ns |
| 15 *independent* loads from one thread | **524 ns total** (~35 ns each) |
| `wp.tile_sum(wp.tile(x))[0]`, `block_dim` 32 / 64 / 128 | **126 / 325 / 369 ns** |
| `wp.tile_scan_exclusive(wp.tile(x))` + `untile`, any width to 32 | **353 ns** |

**These five numbers decide whether a cooperative (one-block, barrier-synchronized) rewrite of a
serial kernel can possibly win, before writing it.** A correct level-synchronous round needs ≥2
barriers plus a prefix scan ≈ **600 ns**; if the level's own work is under that, the rewrite loses.
That is exactly how the BFS drain went (§14.9).

Two facts about the primitives themselves:

- **`wp.tile_sum(wp.tile(x))[0]` is a genuine block-wide broadcast reduction** — correct on *every*
  lane, not just lane 0. `tile_reduce_impl` computes the sum on one thread, but
  `tile_register_t::extract` routes it through a `__shared__` scalar with `WP_TILE_SYNC()` either
  side. It is therefore also a usable **full block barrier**, and so is `tile_scan_exclusive`, which
  syncs before *and* after.
- **Tile ops are legal inside a dynamic `while` loop** on CUDA and behave correctly across lanes
  (probed directly). Keep the loop condition block-uniform — read it from a global that every lane
  reads after a barrier, or from registers every lane updates identically — since `__syncthreads` in
  divergent flow is UB.

**A single thread's throughput is ~35 ns per independent memory op**, so a serial pointer-chasing
kernel is usually **throughput** bound rather than latency bound; check that before trying to hide a
stall.

**A `wp.hash_grid_query` cell probe costs ~600 linear-scan point tests.** Each probe is a hash plus
two dependent, uncoalesced global loads; a linear scan over `points` has every thread in a warp
reading the *same* `points[j]`, so it streams out of L2 as a broadcast. Consequence: once the search
radius outgrows a couple of cell widths, an exact `O(n)` scan is genuinely cheaper than widening the
walk — widening `MAX_CELL_SPAN` from 4 to 8/16/32 made the far-query case *worse* (14 → 34 → 27 ms on
`bunny`). The break-even span scales as `n ** (1/3)` — one cell at 36k points, four at 438k — which is
why `_knn_widest_grid_radius` is `n`-aware rather than a fixed constant.

**`wp.launch_tiled` with every lane walking the whole chunk and lane 0 doing the atomics is FASTER
than one thread per chunk**, even though it looks 64x redundant: all lanes read the same
`a[offset + k]` at each step so the loads broadcast, where one-thread-per-chunk gives each lane its
own 64-element run and the reads stop coalescing. Measured ~10 % end to end on a 20k-point ICP
(`icp_mesh[bunny]` 4.46 → 4.90 ms when the loop was "de-duplicated", back to 4.00 ms when restored).
The redundant arithmetic is free — these reductions are memory-bound. See
`accumulate_procrustes_moments`.

**Dropping a `sqrt` from a ball query's narrow phase is flat** — replacing `wp.length(d) <= r` with
`wp.length_sq(d) <= r*r`, which discards the distance so the root looks free to drop, measures
**0.997-1.003x** over hash grid and BVH at 200k and 1M points and two radii. The root costs nothing
against the candidate walk's memory traffic. Declined on that count *and* because the two are not the
same predicate (§12.4).

**A single-slot atomic reduction serializes on one address, so its cost is linear in the launch.**
Converting `wp.atomic_add(acc, CONST_SLOT, x)` per thread to the tree's `tile_chunk` +
`chunk, lane = wp.tid()` + `wp.tile_sum(wp.tile(local))` + `if lane == 0` idiom, values agreeing to
≤1.6e-05 relative:

| kernel | 4-5k | 20k | 65k | 200-262k | 1M |
|---|---|---|---|---|---|
| `registration.accumulate_cost` | 1.23x | 2.23x | — | 11.90x | 32.06x |
| `polyline.accumulate_newell_normal` | 1.61x | — | 12.32x | 32.23x | — |
| `polyline.accumulate_turning_angle` | 1.19x | — | 5.79x | 16.55x | — |

The converted form needs **no** `prefers_tiled_reduction` branch, because its lanes partition a chunk
the block already owns with a `wp.block_dim()` stride (§2.2) — verified on CPU with
`CUDA_VISIBLE_DEVICES=""`: 3.5e-06 relative and 0.204 → 0.192 ms at n = 65 536. **Do not convert a
conditional atomic** (a compaction cursor, a change flag, a rare-event counter — 27 of the 31
constant-slot atomics in `kernels/`): contention there is proportional to hits, not to the launch.

**The same finding at *one atomic per block*, and it names the real variable — the block count, not
the redundant lanes.** Two `registration` kernels launched `dim=[n / TILE_1D]` with **every** lane
walking the block's whole 64-element chunk and lane 0 publishing, so they were already 64x better
than a per-thread atomic and looked done. They were not: 25 hot addresses
(`accumulate_procrustes_moments`) and 43 (`accumulate_point_to_plane`, a 6x6 normal matrix plus a
right-hand side plus a cost) still took one add per block, at `n / 64` blocks. Re-launching at
`blocks_1d(n)` — the same `ITEMS_PER_BLOCK_1D` fold the reduce module uses — cuts the block count
16x:

| kernel | 5k | 20k | 200k | 1M |
|---|---|---|---|---|
| `registration.accumulate_procrustes_moments` (25 slots) | 1.03x | 1.01x | **4.14x** | **10.67x** |
| `registration.accumulate_point_to_plane` (43 slots) | — | 1.38x | **4.66x** | **9.64x** |
| `points.centered_covariance` (9 slots, a `wp.mat33`) | 1.02x | 1.03x | **2.80x** | **9.82x** |

**The 64-fold redundant arithmetic was never the cost, and a plan that says it is should be
measured before it is believed.** The comment those kernels carried claimed the redundancy was free
because all lanes read the same `a[offset + k]` and the loads broadcast out of one cache line; that
claim is **correct**. Isolated by keeping the grid and giving each lane exactly one element — which
removes the redundancy and changes nothing else — the gain is **1.03x / 1.02x / 1.13x / 1.03x** at
5k / 20k / 200k / 1M. So the shape to look for is not "lanes doing discarded work" but "how many
blocks reach the accumulator", and the lever is the fold width.

**The shape is greppable, and `points.centered_covariance` was the third kernel found this way** —
its helper `reduce.outer_sum_chunk` *documented* the every-lane form in its own comment, which is
what made it findable. Look for `wp.launch_tiled` at a `dim` of `n / TILE_1D` (rather than
`kernel_reduce.blocks_1d(n)`) with a `lane == 0` / `t == 0` atomic commit. The remaining three sites
in the tree that match the `dim` half do **not** match the shape: `metrics.chamfer_*_tiled` and
`measures.centroid_tiled` compute `f = i * TILE_1D + t` and so partition the *outer* work at a
constant stride — the `_sliced` / `prefers_tiled_reduction` case, where changing the fold is a
rewrite of the partition and not a one-token change.

**That rewrite has now been built for `measures.centroid_tiled` and it is flat, which puts a
threshold on this whole family.** The lane-strided form (`tile_chunk(n_faces, chunk,
ITEMS_PER_BLOCK_1D)` plus a `wp.block_dim()` stride, which would additionally retire
`centroid_sliced` and one device branch) measures **0.98-1.01x** at 1 280 / 20 480 / 81 920 /
327 680 faces, areas agreeing to 1.5e-07. **The quantity the fold reduces is `blocks x slots`, and
the threshold is around 1e5, not 1e4**: centroid is 5 120 blocks x 4 slots ≈ 2e4 at 327k faces and
sits under the launch floor, where every kernel that won was at ~1e5 (3 125 blocks x 25 or 43 slots
at 200k-1M points, and `polyline.accumulate_loop_frame`'s three atomics over 262 144 elements). So
**compute `blocks x accumulator slots` before proposing this rewrite**; under ~1e5 it will be flat.
The chamfer pair is one slot and further under. With no CUDA win to pay for it the portability is
not free either — `blocks_1d(n)` gives the CPU path `n / 1024` single-lane blocks against
`slice_count`'s `n / 32` threads — so all three keep their `_sliced` siblings, and the numbers are
written at `kernels/measures.py::centroid_tiled` and `kernels/metrics.py`.

Two consequences for the next conversion of this shape. **The fold is what pays for the
`wp.tile_sum` calls, so a wide accumulator needs it more, not less**: 43 reductions per block is a
much larger fixed cost than 25, which is why the point-to-plane kernel trails at n = 20 000 (1.38x
against 4.14x) and catches up once there is enough per block to amortize them. And **a `wp.tile_sum`
is block-collective, so all 25 or 43 run outside the `if lane == 0` guard** and only the commit is
guarded — `wp.atomic_add` also reads trailing indices as *array dimensions*, not vector components,
so a per-component commit is impossible: assemble the summed `wp.vec3` / `wp.mat33` /
`wp.spatial_matrix` from the tile sums and atomic-add the whole object. As everywhere else in this
family the tree is the *more* accurate arm (relevant here because `out_jtj` is then factorized):
against float64 at 200k / 1M the normal matrix sits at 3.71e-07 / 1.08e-06 against the serialized
form's 1.14e-06 / 1.51e-06, and the whole Procrustes accumulator at 2.35e-07 / 4.42e-07 against
7.30e-07 / 1.21e-06. CPU is flat (0.98-1.02x) and needs no branch, for §2.2's reason.

**Flattening a reduction to its launch floor can make a *neighbouring* fusion worth doing, and the
decline that says otherwise expires silently.** `registration._procrustes_into` ran
`apply_transform_mat44` at `dim=n` immediately above its cost reduction, and fusing the two was
declined on a measurement that the pair cost 0.054 / 0.282 ms at n = 20 000 / 200 000 of which the
reduction was **70 % / 94 %** — so a launch was 6 % of the pair at worst. Once the reduction became
flat at ~0.0165 ms the pair was two launch *floors* and removing one is 40 % of it: **1.61x / 1.72x /
1.58x** at n = 20 000 / 200 000 / 1 000 000, with `acc[ACC_COST]` and every element of
`out_transformed` **bit-identical** (max |delta| exactly 0.0), because the fusion only removes a
round trip of the transformed points through global memory. **After landing a large win on a kernel,
re-read the declines about the launches either side of it** — this is §9's "a written decline can
expire because a neighbour got faster", and the neighbour here was the same function.

End to end the reduction wins are much smaller than the kernel wins, which is the honest number to
quote: `procrustes` is **1.11x / 1.51x** at n = 35 947 / 437 645, but `icp` and `icp_point_to_plane`
are only **1.03-1.08x**, because an ICP iteration is dominated by the correspondence search and by
the per-iteration readback §16.1 already attributes (24 % of a 10-iteration run at 2 562 points).

**A single-address `float64` `wp.atomic_add` serializes the launch entirely** — 102 µs/iteration
against Warp's 22.6. A global dot needs a two-stage reduction.

**Padding must be a hole, not a value.** Pointing padded edge rows at a dummy vertex made
`bsr_from_triplets` accumulate ~96 000 triplets into one entry and its atomic serialized: **4.25 ms
of a 4.82 ms pass**; sending them out of range (silently dropped) → 0.596 ms. Same trap as §12.7's
zero-padded triplets from the other direction, and as the one-atomic-per-tile defect below. **Look
for it whenever padding has a *value* rather than being a hole.**

**Two tiling antipatterns, both measured in `kernels/reduce.py`:**

1. **A `wp.tile_load` kernel below one tile is pure loss (49x).** The axis kernels branch
   `if remaining >= TILE_1D: tile_load(...) else: <serial loop>`. When the *reduced extent* is under
   `TILE_1D` the tile branch never runs, and because `launch_tiled` gives every block 64 lanes, all
   64 redundantly walk the same short row — a 64-fold read amplification. On a `(14M, 3)` table
   `max(axis=1)` went 6.64 ms → **0.13 ms** with one plain thread per output row and a direct write.
   **But the opposite direction must stay tiled**: the same table on `axis=0` has only 3 outputs, so
   3 serial threads are **89x slower** than the tiled form. The dispatch key is the reduced extent,
   not the axis. **And it recurs one rank up**: the rank-2 `axis=None` kernels tile `TILE_2D`-squares,
   which an `(n, 3)` vertex table or `(m, 2)` edge table clips the same way — fixed by flattening to
   the 1-D kernel when the trailing extent is `< TILE_2D` **and** the array is contiguous (5.95x on
   `(14M, 3)`, 6.18x on `(8M, 2)`, 1.02x on `(40k, 128)` where the tile branch does fire, 0.79x at
   `(36k, 3)` where there are too few blocks). Gate on the trailing extent, not on contiguity alone;
   the contiguity guard is *required* because `flatten()` raises on a non-contiguous view.
2. **One atomic per tile does not scale (4.9x).** The global 1-D reductions issued one `atomic_add`
   per 64-element block, so 14M elements put 219k blocks on a single accumulator address — 309 µs
   against a ~31 µs bandwidth floor for 56 MB. Folding `TILES_PER_BLOCK_1D = 16` tiles into a register
   before the atomic gives 63 µs. Swept 1/4/16/64/256: 16 never loses, 4 is better below ~1M but only
   3.33x at 14M, 256 loses everywhere. Seed the accumulator from the block's *first* chunk rather than
   an identity — that keeps the kernel generic over `wp.Scalar` with no per-dtype identity argument.

   **The coupling hazard this creates:** changing how much work a kernel does per block silently
   breaks every *other* module that launches it and computes its own `dim`. `triwarp/smoothing.py`
   launched `kernel_reduce.sum1d_tiled` directly at two sites with a stale `n / TILE_1D`; against the
   folded kernel each block would re-fold the same 16 tiles the next 15 blocks also claim, returning
   **16x** the true sum — and invisibly, since the fold is idempotent for min/max/any/all so only
   `sum` is wrong. Fixed by exporting `kernels.reduce.blocks_1d(n)`. **`grep` for direct launches of
   a kernel before changing its per-block contract.** (CPU cost of the fold, for the record: 1.28x
   slower at 36k, 1.07x at 438k, 1.02x at 14M — accepted on the CUDA number.)

**Two more shape facts:** `wp.tile(vec3)` decomposes to a scalar tile, so tile-reduce a vec3 per
component or pack it; and `wp.array.view(wp.float32)` on a vec3 array gives a zero-copy `(n, 3)` view
for `reduce.minmax`.

### 13.3 Tuning constants are per-device

`ITEMS_PER_SLICE` (elements per thread in the lane-free strided-slice reductions) has a
device-dependent optimum, swept 8-256:

- **CUDA wants long slices.** On `convex_subset_mask` at 200k points, 32 costs **1.54x** of the 256
  optimum; 128 is within 1 % of best there and within 4 % at 5k points.
- **CPU wants short ones.** 128 costs **1.43x** of the 32 optimum on a 5k cloud; the CPU sweep is
  otherwise flat (32 within 1.11x everywhere).

So it became `ITEMS_PER_SLICE_CUDA = 128` / `ITEMS_PER_SLICE_CPU = 32` behind
`_device.items_per_slice(device)`. **Slice length is not one number even within a device**:
`ITEMS_PER_SLICE = 32` suits reductions into one or a few accumulators (centroid, chamfer loss, hull
support); a *per-query* reduction wants ~128 (`proximity.ITEMS_PER_QUERY_SLICE`) because the query
dimension already fills the device and a short slice only multiplies atomics — measured 0.49 ms at
128 against 13.8 ms at 4 on 20k faces × 5k queries.

**When adding a device-dispatched path, re-ask which device still reaches each branch, and re-sweep
the constants that branch reads** — the old measurement may no longer describe any live call site.
`ITEMS_PER_SLICE` had been tuned against `triangles.centroid` and the chamfer losses, then a dual
CUDA/CPU path was added so those two stopped reaching the sliced form on CUDA at all, and the
constant's only remaining CUDA consumer became the convex-hull support sweep, which nothing in the
original sweep had weighted for.

**Sweep the values you did not try the first time, or the re-probe inherits the original's blind
spot.** §9's "re-probe after a Warp upgrade" was run on the three constants stamped Warp 1.16, and
the two outcomes are opposite. `polyline._DOWNSAMPLE_DOUBLING_FROM = 8192` **reproduces exactly** on
1.17 (0.19 / 0.40 / 0.65 / 1.14 / 2.24 / 7.58x at 528 → 65 536 against the recorded 0.19 / 0.37 /
0.62 / 1.16 / 1.92 / 6.72x, masks byte-identical, CPU still losing at every size) — no change.
`kernels/points.FARTHEST_BLOCK_*` did not: the original was a two-value sweep (256 and 1024) and
read as two brackets split at 4 096, but sweeping 128 / 256 / 512 / 1024 shows **three** — 256 below
~2 048, **512 from ~2 048 to ~5 120**, 1 024 from ~6 144 — so the old crossover handed a 4 096-point
cloud that wants 512 lanes to 1 024, a **1.29x loss at exactly the bracket point**. The brackets are
stable across `count` (256 and 1 024 pick the same width at every size) and the selected indices are
identical at every width, so it is a pure cost choice. **A constant whose sweep sampled two values
has not been shown to be a two-bracket problem.**

---

## 14. Kernel-shape verdicts

### 14.1 Block-per-item (one block per item, lanes stride the inner sequence)

**Shipped wins:**

| kernel | measured |
|---|---|
| `kernels/visibility.py::obscurance` — block per point, lanes over rays, `block_dim=64` | bunny_decimated 6.35 → 1.14 ms at 64 rays (5.6x) and 23.2 → **1.96** at 256 (**11.8x**, row flips against pymeshlab); bunny 7.75 → 2.43 (3.2x) and 28.8 → 6.47 (4.5x). Max abs diff **0.0** |
| `shape_diameter` (same thread-per-point shape, converted with it) | harness `shape_diameter[bunny_decimated,256]` 31.1 → **3.05 ms** |
| `points.farthest_point_sample` as **one persistent block** (`launch_tiled(dim=(1,))`, lanes stride the cloud, `wp.tile_max` over `pack_farthest_key` = the argmax *and* the barrier, identical tie-break) | sphere_small 2 562 @1024: 4.32 → **1.20 ms** (**3.61x**, open3d 2.68, row flips); @64: 0.41 → 0.12; sphere_med 40 962 @1024: 21.1 → **9.5** (2.2x); @64: 1.43 → 0.67. Indices identical |

`block_dim=64` is best for the visibility bundle (32 ≈ 64; 256 loses on bunny/64 rays);
`block_dim=1024` on large clouds and 256 on small for FPS. FPS wins where the hole DP lost because
its per-iteration work is `n` distances (tiny), so one SM is enough and the cost was 2 replayed
kernels (~2 µs) per sample against the block's ~1 µs round.

**Declined and annotated with their numbers** (all four already carry a *slice* dimension, so the
outer dimension is not what starves the device): `points.hull_support_extremes` (2.3x at 5 000 points,
**0.12-0.60x at 200 000**), `visibility.support_argmax_tiled`, `proximity.winding_number_tiled`
(2.16x → 1.19x → **1.01x** as the grid fills), `bounds.oriented_box_extents`. CPU flat (1.01-1.06x)
either way. `ITEMS_PER_SLICE_CUDA` keeps all four consumers. Criterion and tables: §2.3.

### 14.2 Cooperative BVH walks

**`tile_bvh_query_aabb` beats the hash grid in the narrow-query regime and regresses above the
crossover.** Measured on one cloud/radius/query set, all walkers agreeing exactly:

| walker | ball of `r` | ball of `2r` |
|---|---|---|
| `wp.HashGrid` serial (incumbent) | 70-83 µs | 359-394 µs |
| `wp.Bvh` **serial** | 132-148 µs (**worse**) | 317-399 µs |
| `wp.Bvh` **tiled** | **29.5-67.6 µs** | **40-99 µs** |
| hashed cell grid, cooperative | 8.6-12.2 µs | 18.3-28.1 µs |

The serial BVH being *worse* than the hash grid is the tell: **the win is the tiled traversal, not the
structure.** BVH build 0.166 ms, cheaper than the cell grid's 0.28. But it **spends 32 lanes on a
query the device could already saturate**: against the hash grid it is 2.4x / 9.0x (small / large
ball) at 64 concurrent queries, 2.2x / 8.2x at 1 024, then **0.74x / 2.1x** at 8 171 and **0.12x /
0.36x** at 65 536. Crossover ~4-8 k queries for a ball of `r`, ~16-65 k for `2r`.

**Swept across triwarp: exactly one of three hash-grid usages can take it, and it shipped.**
`neighbors.query_hashgrid_*` runs at 20 000 queries and every in-repo caller passes a whole cloud —
past the crossover. `poisson_fem.refinement_oracle` is a `fem.ImplicitField` func, so `warp.fem` owns
the launch shape and there is no block to cooperate over. Only `ball_pivoting` qualified, its front
being 312-424 median: its pivot search moved to one warp per front edge walking the BVH cooperatively
— **116.7 → 25.9 ms** and **203.4 → 41.6 ms** (4.5x / 4.9x), taking the group from 3.78x behind
pymeshlab to 1.34x ahead and 1.54x behind to 3.65x ahead. **Do not re-run this sweep.** The empty-ball
test stayed a per-lane serial *hash-grid* query — at that point each lane tests a different ball, so
there is nothing to cooperate on and it wins by running 32-way concurrently.

**The cell-list numbers, kept because they bound what a bespoke index could be worth:** a hand-rolled
cell list (cell index per point → `counts_to_offsets` → `sort_and_argsort`) is **1.3-1.7x faster even
single-threaded** — Warp's iterator pays for generality this workload does not use — and with lanes
splitting the *cells* it is 11.3-13.5x (dense) / **6.8-9.7x** (hashed, `O(n)`, shippable) on a ball of
`r`, and 22.0-29.7x / **13.6-21.2x** on a ball of `2r`, at a build cost of 0.28 ms. A *dense* grid
cannot ship (for a surface cloud the cell count grows as `n^1.5`, ~3e9 cells on `lucy`), so store the
packed cell key per entry and compare it exactly. **Only build one if a measurement shows the query
is *still* the bottleneck** — its extra 2.2-5.5x is probably unspendable against a kernel's own
throughput ceiling.

**A thread-per-query BVH launch is usually load-imbalanced, not under-pruned.** On `bunny` against a
translated copy, 69 451 query faces / 2.01 M candidate tests: **0.16 %** of candidates survive the box
prune (the leaf test is not the cost), **98.2 %** of faces return **no candidate at all** (the BVH
culls them at the root, free), and **0.5 %** of faces carry half the traversal — the busiest walks
**3 428** candidates alone. Fix: a capped thread pass that appends stragglers to a work list, then
**one warp per straggler** via `tile_bvh_query_aabb` — **3.1-10.2x** end to end, distances
bit-identical, cap 64, only 202-1 194 faces overflow. Two levers measured and declined: `block_dim`
(256, the default, wins at every value from 32) and tightening the query margin (**the vertex bound
equals the answer to all 16 digits** on every benchmarked row, so the query only ever *confirms* what
the bound found). **Before optimizing any thread-per-query BVH kernel, histogram the per-thread
candidate count** — and note the imbalance does not necessarily persist at scale (§16.6).

### 14.3 CUDA graph capture

**Capture and argument bundling address different loops, and neither substitutes for the other:
capture pays on a launch sequence that repeats identically, a bundle on one that runs once.**
Recording a graph costs at least what issuing the launches costs, because capture intercepts each
one. Measured in one session, 14-argument kernel at `dim=64` so host cost dominates, 30 reps,
median/min at 400 launches:

| arm | median / min | vs loose |
|---|---|---|
| loose arguments | 11.3 / 9.8 ms | — |
| `@wp.struct` bundle | 6.0 / 4.6 ms | **1.88x** |
| capture-and-replay-**once** | 13.5 / 10.4 ms | **0.84x — a loss** |
| replay of an already-recorded sequence | 2.6 / 0.5 ms | **4.29x** |

That last row is where capture's reputation comes from and it is unreachable without a *repeated*
sequence. **Do not reach for capture on a once-through Python loop** — `holes._fill_dp`'s span loop
and the stitch DP's diagonal loop are each recorded and replayed exactly once, which is the 0.84x
row. The same 0.84x shows up independently in the packing family, where an earlier record claimed a
crossover at ~1024 segments that **does not reproduce**: re-measured at 300 000 elements and again at
2 614 242, record-and-replay-once is **0.62x / 0.86x / 0.85x / 0.84x** at 4 / 256 / 1 024 / 4 096,
flattening near 0.85x rather than crossing 1. Replaying an *already recorded* graph there is 7.7-8.5x
— but a pack's segment pointers change every call, so the recording is never reused.

**And a captured function cannot be attributed with `wp.timing_begin`** — it reports zero kernels
for the replay, so the very functions capture helps most read as ~100 % host (§15.10).

**Where capture pays, it pays large.** `quadric_decimate`'s whole pass is now one captured graph
replayed at a fixed width: **115 → 37 ms** on `saddle@0.1` (2.6-4.5x across fixtures), now faster
than pyvista, igl, open3d and pymeshlab on every benchmarked cell. What made it legal is that the
pass is **flat in the mesh size** — 3.13 ms per pass at 32 258 faces against 2.82 ms at 3 227, and the
*device* half flatter still (0.805 against 0.697) — so running every pass at the pass-0 width costs
**1.01x**. A captured wrapper chain also pays for its Python once (2.69x issued vs replayed), so the
wrapper chains stayed and the capture removed their cost. See §12.6 for what blocks a capture.

**`wp.capture_while` is slower than a batched host loop** where the per-iteration conditional-graph
overhead exceeds the sync it removes: 1.61 vs 1.36 ms/wave on `bunny_decimated` in ball pivoting
(shipped: `_BPA_WAVES_PER_BATCH = 8` with one readback per batch). Nesting one inside a capture is
fine.

**The unexplored lever is the reverse:** a *repeated* wrapper loop issuing an identical sequence that
is not yet captured is worth 4-45x, and `wp.capture_if` (a device-side conditional, unused here) is
the primitive for a stage that currently spends a host readback deciding.

### 14.4 Tile solves: the crossover is K ≥ 16-32

Batched K×K SPD solves, one thread per system (Cholesky in registers on a `wp.matrix` type) vs one
block per system (`wp.tile_cholesky` + `wp.tile_cholesky_solve`, best of `block_dim` ∈ {32,64,128}),
values cross-checked. Above-floor medians (harness floor = 13.9 µs for an empty `dim=1` launch +
`synchronize` — subtract it or small-K rows read as false parity):

| K | N | per-thread | tiled | |
|---|---|---|---|---|
| 6 | 65536 | 8.5 | 38.5 | tiles **4.5x slower** |
| 8 | 65536 | 17.3 | 53.6 | tiles 3.1x slower |
| 16 | 1024 | 27.7 | 7.2 | tiles 3.9x faster |
| 16 | 65536 | 54.1 | 100.6 | tiles 1.9x slower |
| 32 | 1024 | 585.6 | 15.6 | tiles 38x faster |
| 64 | 1024 | 3989 | 85 | tiles 47x faster |

Tiles win only when **occupancy-starved**: few systems, large matrix. **triwarp's only dense solves
are K=6 (`registration.solve_spd6`, N=1) and K=5 (the curvature quadric, N=n_vertices), both on the
wrong side**, so the recurring idea "rewrite the small dense solves as tiles for a CUDA-only fast
path" is refuted — do not re-propose it. It is *not* blocked by the CPU lane constraint; there was no
opportunity to lose. **Only reach for tile solves at K ≥ 16 with low N.**

### 14.5 NumPy readback vs device reduction: 3 of 7 converted

Interleaved back-to-back A/B (min-of-30, GPU clocks warmed) on bunny_decimated / bunny / dragon, **on
both `cuda:0` and `cpu`**. A = readback + numpy, B = device reduction; ratios > 1 mean B wins. **The
CPU axis is what rejected 4 of 7.**

**CONVERTED (win on both devices):**

- `holes._mean_rim_edge_length` → reuse the existing `_loop_perimeters` kernel: 71/253/1941x CUDA,
  39/111x CPU. **But end-to-end `fill_smooth` is only 1.13-1.30x** — the DP dominates, so quote the
  end-to-end number. **The axis is the LOOP COUNT, not the vertex buffer**: on CPU the old form ran
  0.09 ms flat from 8k to 438k vertices, so `vertices.numpy()` on CPU is a *view* and the "whole
  vertex buffer copy" never existed — what cost 427 ms was iterating 8 978 loops in Python with one
  `.numpy()` each. At `combine.stitch_smooth`'s **two** rims the conversion is a measured LOSS
  (0.77x CUDA, 0.38x CPU); accepted, because it is 0.15 ms against a 100+ ms refine-and-smooth.
- `measures.moments`' three `.numpy().sum(axis=0)` → `wp.utils.array_sum`, which **reduces `wp.vec3d`
  componentwise with no kernel needed**: 4.8/16.9/174x CUDA, 3.9/4.7x CPU; end-to-end 8.5x (bunny) /
  42x (dragon), which **turned the benchmark's documented igl loss into a win** (443 µs vs 806).
- `reconstruction._poisson_iso_value` **weighted branch only**: tie at 8k, 2.6x at 36k, 50x at 1M
  CUDA / 3.5x CPU, with the `lengths` alloc counted inside.

**REJECTED — do not re-propose; the current numpy code is faster:**

- `graph` edge-range validation (`.numpy().min()/.max()` → `tw.reduce.minmax`): 14.7x win on CUDA at
  dragon but a **20x LOSS on CPU** (0.059 → 1.275 ms on bunny). Worst offender.
- `smoothing`'s `bool(mask.numpy().any())` → `tw.reduce.any`: **A wins everywhere** — 0.26-0.77x
  CUDA, 0.04-0.05x CPU. A bool array is 1 byte/element; the copy never clears the launch cost.
- `sample_volume`'s `.sum()` + `.min()` (+ the re-upload, credited to B): A wins on CPU 4x and on
  CUDA below ~500k faces. Only 4.3x at dragon.
- `_poisson_iso_value` **uniform branch**: A wins 5x on CPU, and up to ~440k on CUDA.

**Method notes that cost a wasted first run:** sequential (non-interleaved) A-then-B timing produced
non-monotonic garbage with a recurring ~2.27 ms artifact; and pre-allocating B's scratch outside the
timed callable flatters it — count the alloc. **And `grep` the whole package for a private helper's
callers, not just its own module**: `combine.stitch_smooth` reaches `tw.holes._mean_rim_edge_length`
across the module boundary, so a grep scoped to `holes.py` found one caller where there were two, and
the signature change failed `test_combine.py` in the full suite only. basedpyright cannot see it
either (§8).

### 14.6 Readbacks inside device loops

A per-pass `count.numpy()[0]` readback costs **~0.1 ms**, while one extra loop pass costs
**0.9-2.4 ms** (`flip_to_delaunay`) — so any scheme that trades redundant passes for fewer syncs
loses. `_flip_interior_edges` checking every 2 passes instead of every pass is **+5 % to +30 %** with
bit-identical output; `cg(check_every=25/50)` instead of `10` is **+0 % to +6 %** because the
overshoot adds real CG iterations.

`cg(check_every=0)` is the exception on paper — Warp's `_run_capturable_loop` uses `wp.capture_while`
with an on-device condition kernel, so it converges device-side with zero host readbacks, measured
0.93-1.00x — **but it is not a lever in the harness**: an isolated probe read 2.1-2.3x on
`transport_tangent_vectors` while the harness reads 1.05-1.23x on transport, ~1.1x on
`heat_geodesic[sphere_small]`, and **0.94x** on the `saddle_graded` conditioning rows (§15.8). Do not
re-try it. It also changes `cg`'s return from scalars to device arrays and depends on
conditional-graph support.

**Batching convergence checks loses whenever an extra iteration runs a real kernel** (a BVH query in
`max_tangent_sphere`, a whole-graph ECL hook): keep per-iteration 4-8-byte readbacks there and fuse
multiple flags into one buffer instead.

**Before removing a host sync from a loop, price one loop iteration first.** Only remove the sync if
the host could actually run ahead; if the next iteration depends on this one's result, it cannot. A
readback's *self-time* in a profile is the queue depth in front of it, not its own cost, so profiles
make syncs look far more expensive than removing them turns out to be.

### 14.7 Per-device algorithm choice (rare, and the criterion is asymptotic work)

Two functions in the tree branch on the *device*, not just on a constant, and both do so for the same
reason: the parallel form does **asymptotically more work** to expose parallelism, so on a backend
that runs a launch grid as one serial loop there is no GPU win being paid for.

- **`_device.prefers_tiled_reduction`** — a *correctness* branch (§12.2), not a performance one.
- **`polyline_downsample`'s pointer-doubled greedy walk** (`_DOWNSAMPLE_DOUBLING_FROM = 8192`, CUDA
  only). The kept set is the orbit of point 0 under "the next point at least `step` further along",
  so build that step function for every point at once (`greedy_successors`, a lower bound) and
  pointer-double it: `ceil(log2(n+1))` rounds of `succ ← succ∘succ` plus an in-place reached-set
  spread, `2*rounds + 1` launches.

  | n | serial | doubling | |
  |---|---|---|---|
  | 528 | 0.066 ms | 0.354 ms | 0.19x |
  | 4 096 | 0.269 | 0.434 | 0.62x |
  | 8 192 | 0.499 | 0.430 | **1.16x** |
  | 65 536 | 3.735 | 0.556 | **6.72x** |

  Crossover **8 192**, not the 4 096 first guessed. Harness: `rim_long` 3.93 → **0.874 ms**. On CPU
  the doubling loses at *every* size — **30x at 65 536** (0.266 ms serial against 8.105 ms) — because
  it does `n log n` work where the serial form does `n`. (Warp's CPU walk is also *faster than
  CUDA's*, 0.266 vs 3.735 ms — no launch to issue, cache-friendly stride.) **Exactness is the claim**:
  the successor search evaluates the same float32 `cumulative[j] - cumulative[i] >= step` the walk
  does, so the masks agree bit for bit — `cum[mid] - step` would *not* be the same predicate. A
  large-`n` comparison against a NumPy oracle is unavailable (float64 sequential `cumsum` vs Warp's
  float32 tree scan differ ~2e-4 over 8 192 segments, the same order as the gaps between decisions),
  so exactness is pinned triwarp-against-triwarp and the large-`n` test is invariants only.

### 14.8 Solvers: the cycle is launch-bound

**Any smoother that buys iterations with launches loses on this machine.** A Chebyshev multigrid
smoother was built, measured and reverted: interleaved against damped Jacobi in one process over five
systems, **Jacobi wins four of five**, the one loss is 1.03x, the worst Chebyshev cell is a **2.07x
regression**, and no single `(degree, interval)` is best everywhere (`rho/5` is a cliff of up to 17x,
so the interval has no safe default). A Chebyshev step costs three launches where a Jacobi sweep
costs two — a 1.5x launch increase for a ~1.03x iteration decrease. The same reasoning predicts the
sweep-count table being flat: iterations fall 1.4-1.7x from 1 to 4 sweeps while the clock is flat to
rising.

**That 1.5x is the smoother's own arithmetic and it overstates the cycle's by ~4x — the verdict
survives on the other ground, not this one.** Counted at HEAD: one V-cycle apply on
`smooth_region[bunny]` (n = 11 426, 4 levels) issues **34 launches**, of which the smoother is
**9 (26.5 %)** and `csr_matvec` is 18 — so swapping every Jacobi sweep for a Chebyshev step raises
the *cycle's* launch count by **1.132x**, not 1.5x, against a ~1.03x iteration decrease. Still a
loss, and only by ~10 %, so a Chebyshev variant reaching the 1.15x iteration reduction some
intervals gave would flip it on launches alone. **What still refutes it is interval robustness**:
no single `(degree, interval)` is best everywhere and `rho/5` is a cliff of up to 17x, so there is
no safe default to ship — that is the reason to keep, and "the cycle is launch-bound" is the reason
to stop quoting.

Two traps from that work: **a small synthetic system lied** (on a 576-unknown grid Laplacian,
Chebyshev degree 2 over `[rho/3, rho]` gave **12 iterations against Jacobi's 87** — a 7x that
vanished entirely on the real meshes); and **`sweeps` is a default argument of
`_MultigridCycle.__init__`**, frozen at import, so patching the module constant does nothing and the
first probe returned an identical iteration count at every value while looking like a working
measurement.

**A direct GPU factorization (cuDSS 0.8 via nvmath-python + CuPy) wins only where the multigrid gate
already fires.** Measured on the *actual* `(Q_uu, rhs)` systems captured from
`linalg.solve_spd_columns`:

| call | warp CG | cuDSS | reference |
|---|---|---|---|
| `smooth_region[bunny_decimated]` (n=2043, dom 3.0) | 32.2 | **22.2** (1.45x) | meshlib 16.4 |
| `smooth_region[bunny]` (n=8987, dom 2.5) | 72.0 | **44.8** (1.61x) | meshlib 55.6 → flips to a win |
| `harmonic[saddle_small] k=2` (dom 2.6) | 36.7 | **26.3** (1.40x) | igl 27.3 → flips |
| `harmonic[saddle] k=2` | 68.0 | 60.4 (1.13x) | igl 145.7 |
| `harmonic[hemisphere] k=2` (dom 1.94) | 58.6 | 85.5 (**0.68x**) | |
| `harmonic[*] k=1` (dom ~1.0) | 7-11.5 | 22-74 (**0.12-0.31x**) | |

**The cost is the symbolic plan, not the numeric work**: plan 30-160 ms vs factor 0.8-4.4 ms vs solve
0.3-0.6 ms — flat in conditioning, so it wins exactly on the systems `CG_MULTIGRID_DOMINANCE` already
routes to multigrid and loses everywhere CG converges quickly. **Do not add cuDSS as a one-shot
backend**; the real lever, unmeasured, is **plan reuse** (same pattern, `Qg.data` updated in place,
`factorize()+solve()`: **0.95-4.2 ms**) for ARAP-shaped loops. Traps: AMD reordering is 1.7-2x faster
than DEFAULT/NESTED_DISSECTION without the MT layer; `nsa.direct_solver` one-shot re-creates the
handle (30-90 ms extra); `reset_operands(a=new)` drops the plan; `libcudss.so` is not on the loader
path; and systems with **empty rows** (unreferenced free vertices — 7 on `bunny_decimated`, 297 on
`bunny`) are singular for a direct solver where CG leaves them at the initial guess. Footprint:
`libcudss.so` 143 MB + CuPy, CUDA-only.

**Plan reuse re-checked, and there is still nothing to spend it on.** `parametrization.arap` is the
ARAP-shaped loop the lead names — one operator build, then `max_iterations` reuses of
`spd_column_solver` — and it **beats igl in all six benchmarked cells**: 1.76x and 2.81x on
`saddle_small`, 4.20x and 6.08x on `saddle`, 13.97x and 17.25x on `hemisphere`, at 10 and 3
iterations. So no loss row justifies a 143 MB dependency, and neither `nvmath` nor `cupy` is
installed to measure one with. One caveat if this is ever opened: **a host launch count cannot bound
the solve's share here** — an ARAP outer iteration issues only ~12 host launches because
`spd_column_solver`'s CG is graph-captured, which is §15.10 exactly, so the bound has to be taken
with the capture disabled.

**Every direct-factorization reference is flat across the conditioning axis and triwarp's CG is
not**, which says where to look: a conditioning regression in triwarp is an **iteration-count**
problem, not an assembly or operator problem. Across `saddle` vs `saddle_graded` (identical
connectivity, worst aspect ratio 1.6 → 4 719): `min_quad_with_fixed` pymeshlab **39.0 → 39.6 ms**
against triwarp 33.7 → 83.5 (2.5x); `heat_geodesic_conditioning` pymeshlab **71.4 → 71.1**,
potpourri3d 58.4 → 60.0, triwarp 22.4 → 68.7 (3.1x). Three independent implementations agreeing on
flatness is the strongest version of this statement the suite has. `isotropic_remesh` **inverts** it
— pymeshlab 339 → 1 116 ms (3.3x) while triwarp is flat at 123 ms — because triwarp runs a fixed
`iterations` × five launches whatever the input looks like, so **its flatness is a fixed work budget,
not insensitivity** (§16.4).

### 14.9 Refuted, with the code written — do not re-propose

- **A single-block cooperative BFS drain for `graph.bfs`.** Written, verified **byte-identical** in
  `order` / `parents` / `distances` at block widths 4, 8, 16 and 32, and it runs **39.9-44.1 ms
  against the serial engine's 19.7**. The serial drain is memory-op *throughput* bound on one thread
  (481 ns/node against 524 ns for 15 independent loads), not latency-chain bound — so software
  pipelining measured 1.05x and a register-vector batch of the `dist` loads was a loss. And a
  ribbon's frontier is ~2 nodes, so the level's work is under the ~600 ns of barriers a correct round
  needs (§13.2); 20 480 levels puts a ~12 ms floor on the synchronization alone. `graph.bfs` on
  `ribbon_long` stays at 23.3 ms against scipy's 0.74 ms and that is its ceiling on this stack.

  **And `ribbon_long` does not pay per-level dispatch at all, which is the part every write-up of
  this row has had backwards.** `graph.bfs` has *two* engines and hands over to the serial drain as
  soon as the frontier is narrow and no longer growing — its own docstring says so — and a path
  graph trips that immediately. Measured at HEAD: `ribbon_long` runs **20 481 levels** and issues
  **5** host launches, and `wp.timing_begin` attributes **22.755 ms of the 23.5 ms call to
  `resume_bfs_kernel` alone** — the single-thread serial walk, 555 ns a node, matching the 481-544
  ns/node this section already measured for it. So *fusing the four-kernel level body would do
  nothing here*: the level body never runs. The refutation above is the right one; the reason
  usually given for it is not.

  The *separate* `sphere_med` gap is where the level body does run — **193 levels at ~20 µs each**,
  4.03 ms against scipy's 2.24 (1.79x, not the 2.2x once recorded) — and four replayed kernels are
  ~4.7 µs of that 20 µs, so a perfect fusion of all four into one caps at ~1.3x on a 1.79x gap.
  Worth knowing before opening it: the level is mostly device work, as §16.9 says.
- **A persistent one-block-per-loop tiled kernel for the Liepa hole-fill DP.** Built,
  byte-identical, and it loses **0.89x / 0.12x / 0.03x** at rims of 128 / 512 / 2048. The DP is
  `B³/6` apex evaluations (22 M at B = 512) and a block is one SM of ~170; the shipped engine pays
  ~25 µs per span launch — the launch floor, GPU mostly idle — and is *still* 8x faster, because
  510 launches × 25 µs = 12.9 ms against 22 M evaluations at ~5 ns each on one SM. Barriers are not
  the cost (≈1 µs × 3 × 510 ≈ 1.5 ms); the SM count is. (`block_dim=1024` fails to launch, CUDA 701.)
  Same conclusion as the BFS drain from the opposite direction: there the level had too little work
  for a block, here far too much. The only design that beats both is whole-GPU work between cheap
  level barriers, i.e. a grid-wide barrier — which Warp does not expose (§12.2).
- **Tile solves at triwarp's K** — §14.4.
- **A Chebyshev smoother** — §14.8.
- **Voxel aggregation for the multigrid hierarchy** (via `voxels.cell_indices` + `unique_rows`):
  operator complexity **4.2-56** smoothed, while the affordable unsmoothed variant nets only 1.0-1.6x
  and does not break the h-dependence (47 → 85 iterations for a 4x increase in n, the same rate as
  Jacobi).
- **Four blue-noise micro-optimizations of the old Bridson propose kernel**, superseded by an
  algorithm change (§16.7) but each still a valid null: a lazy affine `a*k+b` shell permutation
  (−44 %, because the affine family is tiny and structured so "first valid cell" is a *biased*
  selection against the shuffle's uniform one → more rounds); a keyed Feistel permutation which fixes
  that bias (round count held at 97 vs 99) and is still **−2x** on propose (so the per-thread 728-int
  array was *not* the occupancy bottleneck); removing `prune_spawn_neighborhoods` (output-preserving
  to drop, and propose **exploded 172 → 2529 ms, −15x** — the prune costs 70 ms and saves ~2360 ms of
  `far_enough` work, so it is load-bearing); and dropping `far_enough` from `find_far_candidate`
  (only 172 → 148 ms for the loss of the min-distance safety backstop). A fifth was a wash: making
  the eager Fisher-Yates over up to 728 shell cells lazy, whose result was almost always consumed
  only at its first element.

---

## 15. Benchmark and measurement traps

### 15.1 A sudden slowdown is a Warp rebuild until proven otherwise

Before profiling anything, before believing a kernel got slower, before deleting a test for being
slow: a single-digit-second operation that now takes tens of seconds, or a test file whose cost
appears and disappears as you change *which* tests you select, is almost always a module recompile
from an unregistered generic-kernel overload (§2.5) or an undeclared `wp.map` signature (§3.5).
Measured: `tests/test_reduce.py` took **1 561 s** on a fresh selection and **1.30 s** repeating the
identical one — same tests, same asserts, same machine, **1 200x apart**.

Two cheap confirmations:

```bash
# 1. The compile is single-threaded nvcc, so the GPU is idle while the clock runs.
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader   # 0 % during the stall
# 2. Warp says so outright.
uv run python -c "import warp as wp; wp.config.log_level = wp.LOG_DEBUG; ..." 2>&1 \
    | grep -E "Module hash changed, recompiling|took .* ms  \(compiled\)"
```

Any `Module hash changed, recompiling: <module>` line for a `triwarp.kernels.*` or `map_*` module is
the defect. A *second* line for the same module in one run means the chain is still forking. **This is
the first thing to inspect because the alternative reading — "this test is inherently slow, cap its
input or delete it" — removes coverage to work around a fixable compile**, which is exactly the trade
that hid the problem for as long as it lasted. Check 9 cannot see this, because CLAUDE.md is not one
of its scan roots.

**The second candidate is host-side per-element Python, and it is a distinct class.** It is not a
rebuild (the GPU is idle for both, so that tell does not separate them) and not launch overhead
(those are milliseconds, not minutes). `benchmarks/test_holes.py::test_bridge_edges` ran **>20
minutes without finishing** while the GPU sat at 0 %; the first hypothesis was a reference library's
`relaxApprox`, which timed at **0.02 s on bunny**. The real cause was `holes.bridge_edges`'
`validate=True` path building two Python `set` comprehensions over
`tw.edges.faces_to_edges(faces).numpy()` — one interpreter iteration per mesh edge. Correct, and
invisible everywhere except `lucy` (28 M faces, 84 M edge rows). Replaced with a device scan plus a
four-flag readback: **0.035 s on lucy**. **When a benchmark group stalls, grep the triwarp function
it times for a Python `for` / `set` / comprehension over anything derived from `faces` or `vertices`
— a `.numpy()` feeding a comprehension is the signature** — and confirm by timing the triwarp call
alone on the largest registry mesh. The fix is a device scan returning a handful of flags, **not a
smaller benchmark mesh**: capping the mesh hides the defect, which is what the largest row exists to
prevent.

### 15.2 Attribute against one number

A projection built by subtracting two measurements of *different* things is a hypothesis, and in this
repo it has been **optimistic by 3-10x every time**. Measured across five plan items: the one
projection that held — and beat its target — came from a single attributed number
(`_bsr_accumulate_triplet_values` is 20.7 ms in one launch, 91 % of device time →
`min_quad_with_fixed` gained 2.4-9.8x). The three that came in far under all subtracted unlike
quantities:

- "the min-area fallback pass costs 33 ms" was `plane_normalized` minus `min_area` as *primary*
  metrics — two different per-span kernels. Real value of skipping the pass: **2 %**.
- "the amortized heat solve is 96 % host, so cache the operators for 5x" read a warm, near-converged
  solve; the real solves run 30/90/330 CG iterations and the host cost is *per iteration*. Only
  0.61 ms of 6.29 was hoistable → **4.5-8 %**.
- "hoisting the decimation loop's allocations removes ~1 000 memsets" counted memsets that
  *initialize kernel inputs* and therefore cannot be removed by moving the allocation. All twelve
  allocations are **3.6 %** of the call.

**Price the candidate directly** — time the exact call in isolation, or count its launches and
multiply. Two specific traps: a *warm* repeat of a stage measures a different regime than the cold
one inside the real call (a CG solve especially), and `wp.empty` vs `wp.zeros` differ by ~4 µs of
memset against ~8.6 µs of allocator, so "remove the memset" and "remove the allocation" are different
claims.

**And price it at more than one size, because the *sign of the trend* is the decision.** Two items
worked in one pass had nearly the same share at the small end and opposite verdicts, which only the
second point revealed. `mesh_to_mesh_distance`'s two tail readbacks are 3.3 % of the call on `bunny`
and **12.5 % on `lucy`** — a share that *grows*, so the fix is worth more the more it matters, and
it landed. `hash_indices_rows`' validation readback is 14.9 % of `edges_unique` on
`bunny_decimated` and **4.8 % on `dragon`** — a share that falls, which §9 calls a decline, and it
was declined. A single operating point would have read the two as the same item.

### 15.3 Attribute at the benchmarked operating point

**A profiled share is a share at one point on the parameter axis, and it can reverse an
optimization's sign.** On `sample.sample_surface_blue_noise`, attribution at
`radius = sqrt(area / n_faces)` (~46 000 samples) found `dart_cover_neighbors` at 52-66 % of device
time. Inverting it (one thread per *accepted* point instead of per *alive* point — 128-255x fewer
threads, set-identical by definition) measured:

- dense, ~46 k samples: cover 116.1 → 39.5 ms, whole call 185.5 → 108.3 — **1.71x win**
- **benchmark radius, ~2 k samples: cover 6.05 → 7.85 ms, whole call 16.2 → 18.4 — 0.88x LOSS**

At the benchmarked radius the cells are larger and only ~50-280 points are accepted per round, so the
inverted kernel runs 50-280 busy threads on a 170-SM device — a bandwidth-bound kernel traded for a
latency-bound one. Reverted. **Grep the benchmark for how it derives its parameter and profile at
*that* value.** §9's interleave rule does not catch this — both arms were timed correctly, at the
wrong radius.

**The same rule applies to the *fixture*, not just the parameter.** `mesh_to_mesh_distance`'s V7a
item was written up as a decline from a first pass that measured against a copy translated 0.6x the
extent on *all three* axes — heavily interpenetrating, where the whole call is 5.6 / 37.0 / **8 754**
ms instead of 2.1 / 3.1 / 5.4 and the removed build reads **3.8 % falling to 0.7 %**. The benchmark
translates along **x only** by 1.2-2.0x the x-extent, i.e. disjoint side by side, where the same build
is 9.2 % / 7.5 / 18.1 and the change is a win. **Read the fixture, not the call.**

### 15.4 Benchmark-harness hazards

- **`benchmarks/test_meshes.py` is a real gate and the default `pytest` run does not collect it.**
  It self-checks the registry — recorded vertex/face counts, and a `_TOPOLOGY` table of
  `(bodies, watertight, loops, peak valence)` every feature mesh must match. Registering a mesh
  without adding its `_TOPOLOGY` row fails there with a bare `KeyError`, and **the full suite,
  `basedpyright`, `mkdocs --strict` and `tests.parity` all stay green while it does** — measured,
  after adding the `tangle` axis. So **after touching `benchmarks/meshes.py`, run
  `pytest benchmarks/test_meshes.py`** (9 s) as a fifth gate. The upside of the table is that it
  makes a new mesh state its claim: `tangle_torus` is `(1, True, 0, 6)`, which is what says out
  loud that its self-intersection is *geometric* and not topological — one closed watertight
  component whose embedding crosses itself — and therefore what distinguishes it from `tangle_2`'s
  two bodies.

- **`--benchmark-json` is written at session end, so one pathological row costs the whole module.**
  Round 9 lost **three modules of 48** to a newly added reference, all three the harness rather than
  triwarp, all three sharing one false premise stated in a docstring (*"Cap the `pytorch3d-cpu` row
  alone: the CUDA one is a different cost curve entirely"*). The three failures:
  `test_chamfer_points_to_mesh`'s pytorch3d branch `return`s *above* `skip_larger_than`, so the cap
  governed everyone except the branch that needed it and `lucy`'s 28 M faces held the GPU for **87
  minutes**; `test_icp_point_cloud`'s cap is after the branch and still inert, because
  `skip_larger_than` opens `if bench_case.mesh_name not in MESHES_BY_NAME: return` and `MESH_ORDER`
  holds only the five scan meshes — so `sphere_large`'s 163 842 points ran a brute-force quadratic ICP
  for **926 s**; and `test_laplacian` has no cap at all, so after `lucy`'s 14 M vertices went through
  torch, **Warp could not allocate 65 368 bytes on a 32 GB card and 16 `triwarp-cuda` rows failed** —
  which silently *understates* the loss table, because the missing rows are triwarp's.
  **Put `skip_larger_than` first in every group, before any library branch; give it a vertex-count
  comparison for meshes outside `MESHES_BY_NAME`; call `torch.cuda.empty_cache()` in the pytorch3d
  teardown outside the timed region; and put a bounded per-module wall-clock cap in the runner** (15
  min is generous — the longest healthy module is `test_visibility` at 535 s).
- **Never benchmark Poisson surface reconstruction on `dragon`-sized (>100k point) clouds.**
  `test_screened_poisson[dragon-open3d-*]` ran for **93 minutes without completing one row** (GPU
  idle, 42 CPU cores saturated) and had to be killed, blocking the remaining 18 modules. open3d's
  `create_from_point_cloud_poisson` is ~7.5 s/call at depth 9 on `bunny`'s 35 947 points; `dragon`
  has 435 545. Ten rounds of that is hours, and it measures open3d rather than triwarp. **When a
  benchmark module's wall clock looks wrong, get the per-library split before trimming anything
  triwarp does.**
- **A `benchmarks/` number is 1.2-2.5x an isolated probe number for the identical call**, because the
  harness syncs *inside* the timed callable and runs each round against a cold pool while a hand probe
  warms up and syncs *around* the timed region. Measured at one commit, same machine, same session:
  `ball_pivoting[bunny_decimated / bunny]` 45.5 / 64.0 ms isolated against 67.8 / 87.7 harness;
  `delaunay_triangulation[2000 / 20000]` 8.65 / 22.16 against 23.58 / 46.70. **Comparing across the
  two nearly produced a false regression report** — a round recorded `delaunay`'s post-fix numbers
  from an isolated A/B (7.16 / 18.70 ms), the next round read the harness's, and it looked like a
  2.5x regression. **Compare probe to probe or harness to harness, never across, and label which kind
  a quoted number is.**
- **A median at `rounds=3` can be unrepresentative of its own samples.** `marching_triangles[sphere_large]`
  was one round's *only* regression above 1.31x — a reported 3.25x (3.003 → 9.770 ms) that a whole
  bisect was planned against. It read `min=3.092 median=9.770 max=9.777 stddev=3.857 rounds=3`: the
  floor never moved, and two of three samples carried ~6.7 ms of one-off cost (the shape of a Warp
  module load). pytest-benchmark drops `rounds` to 3 for the slowest groups, and at n=3 the median
  *is* the middle sample. **Run `aggregate.py --suspect` and read it before the loss table** — it
  flags every cell whose median exceeds its own minimum by >1.5x. Over 1 437 triwarp cells exactly
  **two** exceed it, so this is cheap insurance rather than a common problem. It flags *reference*
  cells too, and those tilt the other way — an inflated reference median flatters triwarp
  (`bvh_from_points[bunny-igl-4]` is **7.72x**, min 28.704 ms against median 221.460, and four
  `query_nearest_*[bunny-igl]` rows are 2.2-3.1x), **so a *win* against one of those needs the
  reference's own min checked.**
  **Confirmed a third time on a fresh round**, which is what closing this kind of item looks like:
  the same cell re-measured `median=2.915 min=2.854` — a ratio of **1.02** — so the floor never
  moved and the one-off is gone. Two things worth carrying away rather than re-deriving. A cell
  once flagged suspect should be **re-read on the next round before anything is built against it**,
  because a fresh median is the cheapest possible disconfirmation. And the *real* shape of that row
  was hidden behind the artifact: `marching_triangles` genuinely loses to meshlib by **7.7x at
  `sphere_large` and 17x at `sphere_med`** — but on 2.9 ms against 0.38, a **2.5 ms** absolute gap,
  which is why it never appeared in a loss table ranked by milliseconds and is not a target.

### 15.5 A plan item may be refuted by its own target

In one plan round, **all three items worked were refuted, and in every case the refutation was
already written in the file the item proposed changing**: a size threshold whose counterexample was
named in the constant's own comment; a scope mismatch the benchmark's docstring anticipated verbatim
(*"must not be discounted as 'triwarp does strictly more work' … that discount is available on the
torus fixture, it is not available here"* — the plan had taken a number measured on a `tests/`
fixture and applied it to a different benchmark input); and one where the plan correctly caught that
a threshold was tuned on the solve alone, then re-diagnosed it from the **setup** alone and reached
the opposite wrong answer.

**A plan is written by reading loss tables and diffs; a measured decline is prose and lives next to
the code, so a table-driven pass systematically cannot see it.** Before measuring a plan item, grep
the target function, its constant's comment *and its benchmark docstring* for a number.

**And when a "nothing separates these" note exists, check which *kind* of quantity it ruled out
before accepting it.** Three attempts at a multigrid predictor (size, Jacobi iteration count,
probe-rate extrapolation) all interleaved the classes and the constant's comment concluded nothing
separates them — but every one had tried a property of the *solve*. The property of the *operator*
was never tried, and it works (§16.8).

### 15.6 A/B without `git stash`

**Do not `git stash` to measure current code against pre-change code.** The working tree is shared: a
second stash cycle swept up in-flight edits to two files that arrived from outside the session while
it was running. They popped back intact, but the files were reverted on disk for ~2 minutes, and a
save in that window would have conflicted.

Use a detached worktree, which never touches the working tree:

```bash
git worktree add -q --detach $SCRATCH/baseline HEAD
```

**The catch that makes it non-obvious:** `uv sync` installs triwarp through a *MetaPathFinder*
(`site-packages/__editable___triwarp_0_1_0_finder.py`), which outranks both `sys.path` and
`PYTHONPATH` — so `PYTHONPATH=$SCRATCH/baseline python probe.py` still imports the working tree and
silently measures the wrong thing. Drop the finder first:

```python
sys.meta_path = [f for f in sys.meta_path if "editable" not in getattr(f, "__module__", "")]
sys.path.insert(0, BASELINE)
```

**A `sed`-based in-place sweep is the same hazard as `git stash`, and it is worse because it looks
harmless.** Sweeping a tuning constant by editing `triwarp/*.py` between runs mutates the tree
*another agent may be running `pytest` against* — measured this session: three other `claude`
processes and two foreign `pytest` runs were live, so the sweep was simultaneously reading their
GPU load and writing their source. **A constant baked into a kernel cannot be swept in one
process** (Warp fixes it at codegen), which is exactly the case that tempts the in-place edit; put
the value in a detached worktree instead and drive it with the main venv's interpreter.

**And check whether the box is yours before taking any clock reading at all.** `nvidia-smi`'s
utilization and `ps -eo pcpu,comm | grep pytest` cost nothing and are the difference between a
measurement and a fiction: a theta sweep interleaved at pass granularity read a **4x swing at
*fixed* theta** (`multigrid_preconditioner[saddle]` 16.8 ms in pass 1, 71.7 in pass 2) while two
foreign `pytest` processes held the GPU at 100 %, against round 10's 20.4 ms for the same cell.
Two rules follow. **Interleave A and B inside one process where the change permits it** — pass-level
interleaving is too coarse when the noise timescale is seconds. And **when the box is not quiet,
measure the quantity that is not a clock**: iteration counts, launch counts, level counts,
candidate counts, `nnz`, and whether a reference mutated its input are all deterministic, and every
one of this session's five re-probes of a declined item was settled by one of them.

**Verify with `print(triwarp.__file__)` before trusting a single number.** Do **not** `uv run` from
inside the worktree — it resolves that copy as its own project and builds a second virtualenv (the
same trap `reference/pyvista` has); use `.venv/bin/python` directly.

### 15.7 Timing hygiene

- **Interleave A and B in one loop and report the `min` alongside the median.** Timing all of A then
  all of B lets GPU clock state decide the winner: one sweep produced non-monotonic ratios (8.4x,
  0.03x, 4.35x for one site across three sizes) and a recurring ~2.27 ms artifact, which reversed
  into a clean monotonic trend once interleaved under one clock state with the GPU pre-warmed.
- **Saved baselines drift ±10 % (±30 % under 100 µs) between sessions.** Flagged deltas of +9 % to
  +30 % appeared on benchmarks whose code had not changed at all; re-running both arms back-to-back
  showed the true delta was ±2 %, with the sign flipping between reps.
- **A reference library's own column is the control that licenses a cross-session comparison.**
  `benchmarks/test_homology.py` carried medians of 51.8 / 50.4 / 65.1 ms plus two findings built on
  them and the conclusion that being 6.9-10.9x behind meshlib was the module's largest gap.
  Re-measured: **5.96 / 5.76 / 12.07 ms** — both findings false at the new numbers, and triwarp
  1.24x *ahead* at genus 0. What licensed the conclusion is that **meshlib's column was unchanged
  within noise** (7.38 / 5.52 / 5.99 against the recorded 7.6 / 5.5 / 6.0). A reference nothing in
  the repo touched is a control: if its number reproduces, the session is comparable and the whole
  delta belongs to triwarp. **Re-run the *whole group* including its reference rows**, and a table
  that stale is worth rewriting rather than annotating, since its *conclusions* mislead more than its
  numbers.
- **Do not `wp.synchronize_device` around each launch** when timing a microsecond kernel — that
  measures sync latency. It read 2 265 µs at 256 proposals and 22.7 µs at 4 096 for the *same*
  kernel, which is impossible and should have been the tell. Batch K launches, sync once, divide.
- **The device/wall timer's per-launch synchronization inflates *wall* badly** — it turned
  `min_quad_with_fixed[saddle_graded]` from 89 ms into 1 377 ms. Take launch counts and device totals
  from `wp.timing_begin(cuda_filter=wp.TIMING_KERNEL | wp.TIMING_MEMSET, synchronize=True)`, and wall
  from a separate un-instrumented loop.

### 15.8 Probe-process contamination

**A loop-over-configurations probe script is not the same experiment as the code under test.**
Measured while checking whether the adaptive screened-Poisson backend worked on CPU: **in one
process**, looping over 3 cloud sizes × 3 depths, the same `(2562, depth=6)` config gave `0 faces` on
one run and `ValueError: data must be non-negative, got a minimum of -1` on the next, while
neighbouring configs succeeded — reading exactly like a real nondeterministic backend bug. **One
config per process, five processes: `31 832 faces` every time**, identical to CUDA. The actual pytest
suite: all five adaptive tests pass on CPU. So the "defect" was the probe.

**Before attributing a failure seen in a multi-config probe to the product, re-run the single failing
configuration in a fresh process**, and prefer running the real tests over re-deriving them in a
script — `for i in 1..5; do uv run python probe_one.py; done`, not a Python `for` loop inside one
interpreter. **Corollary for the reverse direction:** a probe that *passes* in one process says
nothing about a suite that runs hundreds of allocations before it. And holding several 28 M-primitive
structures at once in one process raised `CUDA error 700` in `wp_memcpy_d2h` — the probe's own
footprint, not a reproduced defect.

### 15.9 Decide on the harness number, not the isolated one

`check_every=0` for the heat-family CG read 2.1-2.3x in an isolated probe and 1.05-1.23x in the
harness, with **0.94x** on the `saddle_graded` conditioning rows (§14.6). Same shape as §15.4's
1.2-2.5x factor: **decide on the harness number.**

### 15.10 `wp.timing_begin` is blind to graph-replayed kernels, so a captured function reads as ~100 % host

**This invalidates the device/wall split for every function that graph-captures**, and it fails in
the most expensive direction: a device-bound function reads as host-bound, which points the next
optimization at launch elimination when the kernels are the cost. Measured directly — 20 launches
of one kernel, timed both ways in one process:

| | `timing_begin` reports |
|---|---|
| 20 loose `wp.launch` calls | 20 kernels, 0.113 ms |
| the identical 20 inside a `wp.ScopedCapture`, replayed | **0 kernels, 0.000 ms** |

**The reach is much wider than triwarp's own five capture sites** (`graph`, `polyline`, `remesh`,
`linalg`, `reconstruction`), because **`warp.optim.linear`'s solvers capture their iteration by
default** (`use_cuda_graph=True`). So every triwarp CG solve — heat, parametrization, smoothing,
`min_quad_with_fixed`, `solve_spd*` — has its dominant kernels hidden from this measurement. Three
corrections measured on Warp 1.17 / RTX 5090:

| call | reads as | actually |
|---|---|---|
| `graph.bfs[handles_64]` | 97 % host (0.29 ms, 62 kernels) | **~100 % device** — 3.73 ms over **572** kernels against a 3.53 ms captured wall |
| `heat.heat_geodesic[sphere_med]` | 97 % host (0.48 ms, 85 kernels) | **87 % device** — 13.03 ms over **2 949** kernels, 362 CG iterations |
| `heat.transport_tangent_vectors[sphere_med]` | 93 % host (1.03 ms, 168 kernels) | **71 % device** — 4.17 ms over 864 kernels |

**How to measure it correctly.** Re-run with capture disabled and read `timing_begin` there; the
kernel *set* is unchanged, so the device total transfers back to the captured run (only the wall
does not — the uncaptured wall is 3.7-9.5x higher, which is what the capture is worth).
`warp.optim.linear` takes `use_cuda_graph=False`; triwarp's own `wp.capture_while` sites fall back
to direct execution when `wp.is_conditional_graph_supported` returns `False`, so monkeypatching that
is the lever. **Both arms must be instrumented identically** (§9) — the uncaptured arm is the
measurement, not the baseline.

**The tell that this was wrong all along was already in the tree.** `heat.heat_operators`' docstring
had concluded *"the call is iteration-bound, and the Poisson half is the expensive one … the cost is
device-side rather than host"* — derived from a tolerance sweep (12.65 / 10.21 / 5.16 / 2.44 ms at
`tol` = 1e-10 / 1e-6 / 1e-3 / 1e-1) rather than from `timing_begin`. When a device/wall split
contradicts a tolerance or input-size sweep on the same function, **the sweep is right**: it cannot
be fooled by where the kernels were issued from.

**Two second-order traps in the same family**, both of which make host time look larger than it is:
a `.numpy()` readback's wall time is the queue depth in front of it (§14.6), and once ~1 000
launches are pending the driver's launch queue fills and `wp.launch` itself blocks — measured 41 µs
per `wp.launch` inside `transport_tangent_vectors`' CG against the 12 µs floor of §13.1, which is a
device-bound signature and not a marshalling cost.

---

## 16. triwarp component status

Shipped results, open defects and refuted plans, by area. **Check here before opening work on any of
these.**

### 16.0 The launch-resolution pass, and what it says about where to look next

The single largest cross-cutting win the tree has taken: **every generic-kernel launch site (42) and
`kernels/reduce.py`'s 34 generic kernels now go through a dtype-keyed table of concrete kernels**,
removing ~12 µs of `infer_argument_types` from each launch. Measured against a detached baseline
worktree, 19 of 20 probed wrappers improved and none regressed, from 1.09x (`laplacian.mass_matrix`)
to 1.58x (`reduce.max(axis=1)`); the table and the mechanism are §13.1, the rule is §2.5, the
factory guidance §2.7. Full suite unchanged at 3 121 passed, and the suite's own wall clock is flat
(63.9 s against 60.5 s, inside session noise) once the one-time cold compile of the renamed kernels
is paid — nothing extra is compiled, since a registered overload *is* a compiled kernel.

**Three method points from that pass, because they generalise past this one finding:**

- **An AST scan of kernel annotations found 39 of the 71 generic kernels; a runtime census found all
  71.** The 34 it missed are in `kernels/reduce.py`, where a *factory* leaves the dtype generic by
  default — nothing in the source text says `Any`. Take the census from Warp, not from the parser:
  `[k for k, v in wp.get_module(name).kernels.items() if v.is_generic]` after importing every
  `triwarp.kernels.*`. The same reasoning applies to any property Warp computes rather than reads.
- **The module that had already written the finding down was the one still paying it.**
  `_reduce_1d_tiled`'s docstring has said "the generic form is declined here, measured ~18 us per
  launch" since it was written — and 34 of its instantiations took the generic default anyway,
  because the `dtype` parameter had one. §9's "read what calls the thing, the decline may already be
  written there" has a converse: **a written decline is not a landed one; grep for the sites that
  should have obeyed it.**
- **The probe that measures a fix must not also change the thing it measures.** A first pass
  monkeypatched `wp.launch` to substitute concrete kernels and read *losses* on two wrappers; both
  were artifacts — the patch left non-array generic parameters generic, and it added its own Python
  wrapper to every launch in the arm. Adding a third "control" arm with the wrapper but no
  substitution, and then re-measuring against a real baseline worktree, turned both losses into
  1.28x and 1.15x wins.

### 16.1 Where triwarp's benchmark losses actually are

**Six of the ten biggest losses are 92-99 % host-side launch/allocation cost, not slow kernels.**
Device time from `wp.timing_begin(cuda_filter=wp.TIMING_KERNEL | wp.TIMING_MEMSET, synchronize=True)`,
wall from the benchmark suite.

!!! warning "Every reading in this section is only valid for a function that does **not** graph-capture"
    `wp.timing_begin` reports **zero** kernels for graph-replayed work (§15.10), and
    `warp.optim.linear` captures its solver iteration by default — so this method reads any CG-backed
    or `wp.capture_while`-driven function as ~100 % host whatever it really is. The rows below were
    taken before their functions were captured, or on functions that never were; three that this
    method got backwards are corrected in §15.10. **Check for capture before trusting a host share,
    and re-derive one that was taken before a capture landed.**

| group / case | wall | device | host share |
|---|---|---|---|
| `remesh.quadric_decimate` [saddle_graded, 0.1] | 354 ms | 28.5 ms | **92 %** |
| `heat_geodesic` amortized [sphere_small] | 6.57 ms | 0.27 ms | **96 %** |
| `intersection.slice_mesh_with_plane` [bunny_decimated] | 4.90 ms | 0.08 ms | **95 %** |
| `creation.icosphere` [subdiv 3] | 2.92 ms | 0.21 ms | **94 %** |
| `creation.triangulate_polygon` [ring 64] | 2.62 ms | 0.09 ms | **95 %** |
| `creation.cone` [32 sections] | 0.47 ms | 0.01 ms (2 launches) | **99 %** |

The two that *are* device-bound are so for the same reason — a grid too narrow to fill the machine,
not an inefficient kernel: the hole-fill DP is 153 of 155 ms across 1 020 launches with a grid of at
most 1 024 threads, and `graph.bfs[ribbon_long]` is 22.28 of 22.89 ms in **one** kernel launch on
**one thread** (544 ns/node over 40 962 nodes).

A later attribution sweep across every module with a mid-band loss found that everything in the
0.3-20 ms band except `ambient_occlusion` (0 % host), `lscm` (10 %), `bfs[sphere_med]` (11 %) and
`query_nearest_bvh_k1` (11 %) is **66-97 % host** at 19-350 kernels — launch floors (`split` 97 %,
`robust_laplacian` 88 % at 168k kernels, `delete_region_keep_boundary` 87 %/98k, `cluster_decimate`
85 %/84k, `remove_degenerate_faces` 88 %/18k). **Tiling cannot touch those; only launch *elimination*
can.**

**My own first hypotheses attributed all of these to kernel or algorithm cost, and were wrong** — I
expected the solver losses to be CG iteration count; the CG is 1.65 ms of device time across 176
launches. **Reading the benchmark table alone misdiagnoses six of ten. Get the device/wall split
before optimizing anything.**

One single-kernel outlier, unrelated to launch count: `linalg.assemble_interior_system` spends
**20.4 ms in one `_bsr_accumulate_triplet_values`** (91 % of device time) because it routes an
already-row-sorted, already-deduplicated CSR through `bsr_from_triplets` — 418 176 triplets in,
418 176 nnz out. That is 61 % of `min_quad_with_fixed[saddle]` and is paid by `harmonic`, `lscm` and
every constrained solve.

**Also host-bound, from a different angle:** ICP's per-iteration cost over 10 iterations on 2 562
points (1 419 µs) is 343 µs (24 %) of `cost_acc.numpy()` readback and 139 µs of `dim=1` solve launch,
against only 60 µs (4.2 %) of actual Cholesky.

### 16.2 `import triwarp`

**0.603 s → 0.001 s** via a PEP 562 module `__getattr__` in `triwarp/__init__.py`, resolving each
submodule (and `Trimesh`, a class, as a special case) on first access and caching it into the module
namespace so later `tw.laplacian` is a plain global lookup. Cause and per-module timings: §12.6.

**Four tests pin it, and the load-bearing one runs `import triwarp` in a subprocess** and asserts
zero kernel modules are pulled in — in-process the answer is always "all of them", because by the
time a test runs the session imported what it needed. **Do not "simplify" it to an in-process check.**
Two things the laziness deliberately does not change: `_register_overloads()` still runs before the
first launch through its module (a wrapper cannot be reached without importing it), and nothing calls
`wp.load_module` / `wp.force_load` at import.

**This was not a Warp lever.** Warp 1.17's compilation/startup guide prompted the look, but every
genuine Warp-side item there was worth far less — `wp.load_module(max_workers=)` is warmup-only,
`module="unique"` was already argued and priced in `kernels/reduce.py`, and the `wp.map`
cross-process cache fix buys triwarp nothing because named `@wp.func`s already cached.

### 16.3 `reconstruction`

- **`screened_poisson` is correct on the CPU device but not fast.** Per call on a 642-point sphere
  cloud, `full_depth=4`:

  | depth | CUDA | CPU | faces | mean radial error |
  |---|---|---|---|---|
  | 4 | 0.37 s | 1.19 s | 2 144 | 0.00911 |
  | 5 | 0.01 s | 11.91 s | 7 976 | 0.00626 |
  | 6 | 0.01 s | 99.43 s | 32 552 | 0.00445 |
  | 7 | 0.03 s | (~13 min) | 136 664 | 0.01723 |

  The `dense` solve is over the `2^depth` cubed node grid **whatever the cloud size**, so each level
  is roughly 8x. The `adaptive` (warp.fem) backend is far cheaper on CPU — 12 s at depth 6 on a
  2 562-point cloud — and gives the same faces as CUDA. `tests/test_reconstruction.py::_poisson_depth`
  returns **5 on CPU and 6 on CUDA**, which took the CPU suite from 23 min to **13:09**. Three tests
  keep a depth-6 literal because their claims stop holding at 5, each for a measured reason:
  `test_poisson_sphere_watertight_manifold` (at depth 5 the surface is edge-manifold with zero
  boundary edges but **self-intersecting**, on *both* devices), and the two class-C metric
  comparisons whose margin falls from 3.4x / 3.7x to **2.64x / 2.44x**, under §7.4's 3x floor (the
  mutation probe still fires — it is the margin that fails, not the sensitivity).
  **Most of the residue is not triwarp**: pymeshlab's own depth-5 reconstruction on this cloud is
  **203.5 s**, against 98.3 s for triwarp's depth-6 solve beside it.
  **And the error is not monotone in depth** — 0.00911 / 0.00626 / 0.00445 / 0.01723 at depths 4-7 on
  a fixed cloud, because past some depth the octree resolves sampling noise rather than the surface.
  A test asserting "finer depth reduces error" is asserting something false in general; the one that
  did was removed for that reason.
- **`screened_poisson(point_weight=0.0)` returns 27-64 zero-area triangles, every run.** It was the
  only one of the four reconstruction entry points with no degenerate-face tail (the other three run
  `_clean_reconstruction`). Found through a reference's warning — §7.7.
- **`ball_pivoting`'s persistent-front rewrite also fixed the documented v1 watertightness
  limitation.** `bunny` 1070 → 214 ms. The old design rebuilt the front from the whole triangle soup
  every wave and produced 3.08 faces per referenced vertex with 23.6 % boundary edges; the persistent
  front gives **1.97 faces/vertex and 1.7 % boundary**, and a subdivided icosphere now reconstructs
  to exactly `2v - 4` faces with **zero** boundary edges. So the "overlapping sheets" artefact was a
  consequence of the per-wave rebuild, not inherent to a wave-parallel front. **Do not reinstate that
  caveat.** Measured design decisions, all counter to the obvious guess: `wp.capture_while` is
  *slower* than a batched host loop here (§14.3); **hash-grid cell width should be the *ball* radius,
  not the `2 * radius` pivot neighbourhood** (12 % end to end — the inner `ball_is_empty` is the most
  frequent query and a 2x cell made it enumerate ~8x the points; finer than `radius` is much worse);
  and compacting the front to only live edges was a measured **10 % loss** in the pre-rewrite design
  (fewer threads = less parallelism) and is worth it now only because the compaction is a free side
  effect of the pivot pass. What remains (94 % of kernel time) is `pivot_front_edges` and it is
  **occupancy**-bound. The tiled-BVH pivot search then took it 4.5-4.9x further (§14.2).
- **`ball_pivoting`'s wave loop is reproducible; `repair.make_winding_consistent` is not.** BPA *was*
  run-to-run nondeterministic (44 179 / 44 243 / 44 208 faces over three runs of one build), and the
  cause was one line: `propose_triangle` hands out a proposal slot with `wp.atomic_add` and
  `claim_triangle_vertices` used that slot as the `wp.atomic_min` priority, so which triangle won a
  contested vertex was GPU scheduling. **It compounded rather than permuted** — a losing proposal's
  front edge is retried a wave later against mutated state. Note **BPA has no float atomics at all**,
  so this was never FP non-associativity; it was integer arrival order, which is exactly fixable. The
  fix is `proposal_key(a, b)`, a `uint64` bit-pack of the proposal's *source edge*, unique within a
  wave (`wp.atomic_min` works on `uint64` and `int64` on both devices). Two other order leaks closed
  in the same change: the triangle budget now declines a *whole* wave, and both consumers clamp
  `CNT_PROPOSAL`, which overshoots its buffer on overflow (a latent OOB read).

  **But a globally fixed key starves the front**, and that cost 1.66-1.77x: the same proposals win the
  same contested vertices every wave, so each wave commits a smaller independent set —
  `bunny_decimated` 24.83 ms / 80 waves → 43.88 ms / **208 waves**, `bunny` 37.09 / 152 → 61.71 / 304,
  face counts unchanged. Arrival order had been a tie-break re-drawn each wave. **Fixed** by
  `wave_salt`, Knuth's hash of the wave counter xored into both halves of the pack (masked to 31 bits
  so it cannot set a point index's sign bit): recovered **1.70x / 1.35x**, waves 208 → 96 and
  304 → 176, still reproducible call to call. **When making an order-dependent algorithm
  deterministic, check whether the replacement order is fixed across *rounds*.**

  **The fixture lesson, which cost the most time: no icosphere can test this.** icosphere(3), (4), (5)
  and a jittered icosphere are all reproducible *without* the fix — a uniformly sampled closed sphere
  reconstructs to its exact Euler triangulation, so wave order has nothing to decide, and the first
  regression test would have passed against the broken code. Use
  `tm.creation.torus(1.0, 0.35, 64, 32)` (2048 points, irregular spacing): pre-fix 2 980 / 3 036 /
  3 042 / 3 089 faces over four runs, post-fix a constant 3 269.

  **Still open:** `repair.make_winding_consistent` seeds each connected component from an arbitrary
  face and that choice varies run to run, so `ball_pivoting`'s public return agrees as an *unoriented*
  triangle set but not in per-component winding. Bisected stage by stage —
  `resolve_duplicated_faces`, `remove_degenerate_faces`, `remove_non_manifold_faces` and
  `holes.fill_small` are all deterministic. This is why sorting the face buffer would not buy
  bit-identity.
- **OPEN: `ball_pivoting` intermittently dies with CUDA error 700** on ~50-70 % of runs that
  reconstruct three *different* clouds in one process, surfacing at a `.numpy()` readback.
  **Pre-existing** (baseline faults 3 of 5, HEAD 4 of 6, verified in a detached worktree with
  `triwarp.__file__` printed). Each cloud alone is fine. **Not a lifetime bug** — holding every
  input/output/intermediate alive makes it *worse* (6 of 6). **Invisible to both sanitizers**:
  `compute-sanitizer --tool memcheck` reports zero invalid accesses and passes, and Warp
  `mode="debug"` bounds checking also passes — do not conclude "clean" from either, they perturb the
  timing/layout the fault depends on. Not queued-wave runaway. **The unguarded-scatter suspect was
  investigated and CLEARED** (all five now route through a guarded `push_front_edge` with a
  `CNT_OVERFLOW` counter; instrumented on the faulting sequence, overflow reads **0** and the front
  peaks at **0.1 %** of capacity). **What it does depend on is the allocator:** default (CUDA mempool
  on) faults 4 of 6; `wp.set_mempool_enabled(dev, False)` faults **0 of 6** — so the fault needs
  **async allocation overlapping in-flight work**, an ordering bug between a free / pool-block reuse
  and a kernel still reading it, not a static OOB. **Workaround:**
  `wp.set_mempool_enabled(wp.get_device("cuda:0"), False)` around the reconstruction. **Where to look
  next:** what `_BpaState.grow()` / `compact()` / `_adopt_next_front()` free or rebind while waves are
  still queued, and whether the `wp.Bvh` or `wp.HashGrid` outlives the kernels holding its `id` — a
  `wp.uint64` id is not a reference Python can see. Treat every `ball_pivoting` benchmark and test
  result as provisional until closed.
- **Slab-chunked marching cubes is not viable and was abandoned.** `wp.MarchingCubes` is crack-free
  only *within* one grid: its per-cell face triangulation is not consistent across independent
  invocations, so welding independent z-slabs leaves ~2·(seam-verts) non-manifold edges spread over
  the whole surface (χ = 1187 at depth 8, growing with slab count) even though each slab is
  individually manifold and seam vertices weld perfectly. A 1-ULP x drift was confirmed and fixed
  with integer index-space MC bounds plus a uniform index→world transform (seam verts then
  bitwise-identical), but the face-triangulation inconsistency is the real blocker and needs a
  face-consistent MC reimplementation. Since the solve is spacing-capped anyway, the payoff is
  marginal; the capped full-array extraction was kept.
- **`warp.fem` gotchas from the adaptive backend**, each of which produced a plausible wrong answer:
  an `ImplicitField` func must have **no return annotation** (it builds an arg-struct from
  `argspec.annotations`, which includes `'return'`); `allocate_by_voxels` is voxel-**centered**, so
  the domain is `[-cv/2, extent-cv/2]` and `translation=(0.5*coarse_voxel,)*3` is mandatory or the
  outer extraction-lattice shell lands outside the domain, `fem.lookup` returns NULL and a spurious
  surface component appears at the cube corner; solve in **index space** (finest voxel = 1) so the FEM
  screening-vs-stiffness balance matches the dense index-space calibration (world-space stiffness
  ~O(h) is negligible against screening → near-singular); and **a point-source weak form RINGS when
  cells ≪ sample spacing** (χ = 413 at depth 6 on a sparse cloud), fixed by capping
  `grid_depth = floor(log2(2*cube/spacing))` clamped to `[full_depth, depth]` — `round` was too
  aggressive.

### 16.4 `remesh`

- **OPEN — `isotropic_remesh`'s `_smooth_pass` is uniform where its own Notes promise the
  Botsch-Kobbelt area-equalizing relaxation.** Area-weighting each neighbour by its barycentric area
  takes aspect p99 on a graded patch from **352 to 20** and improves the icospheres too (min angle
  45 → 54°) — but it makes `is_watertight` fail on `cave_cube` via a **self-intersection at every
  step size down to lam=0.1**, so it is not an overshoot and damping does not fix it; clamping the
  step to 0.5x the shortest incident edge does not fix it either *and* throttles the benefit away
  (asp99 back to 374). **Not shipped.** It needs a real fold guard: propose positions, reject any
  vertex whose move would invert an incident face normal. The area-equalizing solve itself now exists
  as `smoothing.equalize_triangle_areas` (a per-vertex 3x3 float64 solve carrying a `no_shrinkage`
  tangent-plane constraint the barycentric weighting did not) — that is the obvious thing to try
  against the `cave_cube` self-intersection, but **it has not been tried, and the fold guard is still
  unwritten.**
- **FIXED — `valence_flip_candidates` had no shape guard.** It flipped on valence alone with only a
  convexity test, and convexity makes a flip legal but bounds nothing about the shape produced: on a
  graded mesh it turned slivers into worse slivers and in float32 hit exactly-zero area — **2 738 of
  84 406 faces**, worst aspect ratio 5.7e6. Now also rejects a flip creating a degenerate triangle or
  worsening the pair's aspect ratio (its Delaunay sibling already had deviation and aspect gates).
  Result: 0 degenerate faces everywhere, and `saddle` reached parity with MeshLab (aspect p99 1.58 vs
  1.57).
- **Why a graded regular grid is the pathological input**, and why `tests/test_remesh.py` could not
  see either bug: it is valence-perfect (100 % of interior vertices have valence exactly 6), already
  Delaunay, *and* a fixed point of the unweighted Laplacian — so **three of the five stages are blind
  to its anisotropy by construction** and only split/collapse act, reaching a dynamic equilibrium at
  ~41 % short edges of which 94.8 % *are* collapsible (so the link-condition and anti-oscillation
  guards are not the cause, and raising the collapse pass cap 12x barely helps). The suite had only
  run the remesher on clean **closed icospheres**.
- **Never quote the `isotropic_remesh` speedup over MeshLab without the quality caveat** — triwarp is
  flat at 123 ms against pymeshlab's 339 → 1 116 because it spends a fixed budget, not because it does
  the same job (§14.8).
- **CLOSED — `remesh.claim_collapses` locked by the raw edge index.** Fixed in `3428899` by
  merging both collapse paths onto one `claim_collapse_key` kernel, so the isotropic path now hashes
  its lock key like the quadric one; `kernels/remesh.py` reads `scramble_index(k)` at all four
  sites. The measurement is kept because it prices the *shape* of the defect, which recurs wherever
  a parallel independent set locks on a spatially monotone index. The **quadric** path always hashed
  (`scramble_index`, whose comment records that a raw index gives 1 winner out of 51 546 candidates
  because `edges_unique` is spatially monotone); the **isotropic** path (`claim_collapses` →
  `commit_collapses`, backing `isotropic_remesh`) used `key = k` — the thing that comment warns
  about:

  | mesh | candidates | winners raw | winners hashed |
  |---|---|---|---|
  | `saddle` | 40 934 | **1** | 608 |
  | `saddle_graded` | 36 234 | **1** | 491 |
  | `icosphere(5)` | 30 720 | 162 | 328 |

  End to end through `_collapse_pass` at `max_passes=5`, kernels monkeypatched so nothing else
  differs: `saddle` 17 689 → **17 684** verts (raw) against **14 928** (hashed), at **10.6 vs
  10.7 ms** — the cost is flat because a pass is dominated by rebuilding the edge incidence, so the
  hashed key buys ~500x the work for free. **Every existing test passed against the broken version —
  nothing asserted a collapse *count***, which is the lasting lesson: an independent-set kernel needs
  a test on how many winners a round produces, not only on the validity of the ones it commits.
- **`isotropic_remesh` is not byte-gateable.** Its face buffer is stable but vertex positions differ
  by ~2.7e-06 run to run (icosphere(2), 3 iterations, CUDA), because `accumulate_one_ring` and the
  area-weighted normals accumulate with atomics in nondeterministic order; on some fixtures (a
  48-section cylinder) the drift tips a split/collapse decision and even the **face count** changes.
  A byte gate that includes it reports false failures that read as a real regression — this cost a
  round of debugging where 5 of 44 cases "failed" for this reason alone. **To gate a change that
  `isotropic_remesh` merely *contains*, gate the contained stage directly** —
  `remesh._valence_flip_pass(vertices, faces_clone, feature)` mutates its face buffer in place and is
  exactly reproducible. Same rule for any other stage: find the deterministic sub-step, do not loosen
  the comparison.
- **`quadric_decimate` is one captured graph** — §14.3. The round cap is not the lever
  (`_QUADRIC_ROUNDS` swept 4-64 changes the pass count by ≤1).
- **Edge-length equilibrium is ~1.0t only when the target is a "nice" ratio of the input edge**
  (midpoint-split quantization); coarser-than-input targets plateau at ~0.73t.

### 16.5 `array`, `graph`, `polyline`

- **`kernels/array.binary_search_index` is `searchsorted(side="right")`** and returns `slot + 1` on an
  exact hit. Picking the wrong one of the three searches is silent and systematic, not a crash: in
  `selection.faces_left_of_contour` the payload was an argsort of halfedge keys, so `slot + 1`
  returned **a valid index of the wrong halfedge** — every seed landed on a neighbouring face and the
  flood fill returned the whole mesh on every input. The existing consumer `map_sorted_inverse`
  compensates with an explicit `- 1`, which is the tell that the "right" convention is the odd one
  here. **When a binary search feeds an argsort payload, verify one lookup by hand against NumPy** —
  the wrong answer is plausible, not garbage. And note a too-permissive *orientation* bug in the same
  function produced the identical symptom, so the second bug was only findable after the first was
  fixed: **when one fix does not change a symptom, suspect two causes rather than concluding the fix
  was wrong.**
- **FIXED, and the same convention was the cause: `side="right"` is poisoned by `NaN`, so
  `unique_1d(return_inverse=True)` was silently wrong on any float array containing one.** Warp's
  float radix sort puts `NaN` last and every comparison against it is false, so a binary search
  whose midpoint lands in that tail is steered by a predicate that never fires — and `side="right"`
  is steered *into* the tail, reporting the **last** slot for every finite value. Measured on
  Warp 1.17: `[3.5, 1.25, 3.5, nan, 1.25]` returned `[2, 0, 2, 2, 0]` against numpy's
  `[1, 0, 1, 2, 0]`, while `unique_1d`'s own docstring documents numpy's `NaN` semantics for exactly
  this path. `map_sorted_inverse` now searches `side="left"` (identical for an integer key, since
  every key it looks up is present) and falls back to the last slot when the found element compares
  unequal — which is the `NaN` branch, and costs one comparison. **How it was found is the
  transferable part: it fell out of *registering the float overloads*** (§2.5), because writing the
  registration meant asking what dtypes the wrapper's dispatch reaches, and then launching one. A
  dtype nothing had ever knowingly run is a dtype nothing had ever checked.
- **`array.isin` spent 48-54 % of every call inferring the value span, and the guarantee that
  removes it is a *bound*, not `assume_unique`.** Stage-timed on an RTX 5090, Warp 1.17: the two
  `reduce.minmax` reductions and their two host readbacks are **169-181 µs of a 315-350 µs call**,
  flat from 60k to 1M elements and identical on the dense and sparse branches, while
  `_sorted_copy` is 30 %. `assume_unique` — either side — buys **nothing**, and the reason is worth
  keeping: `numpy.isin` gains from one only because `numpy.in1d`'s sort path calls `numpy.unique`
  on both arrays first, where neither triwarp strategy dedups anything (the table is an idempotent
  scatter, the binary search reads the keys as given). So `isin` takes `max_index`, spelled and
  documented like `grouping.hash_indices_rows`' — **2.74-2.87x** (285.6 → 99.4 µs at 60k over a 20k
  table, 285.4 → 100.8 at 200k, 289.3 → 105.5 at 1M), threaded from both in-repo callers
  (`selection.face_indices_from_vertex_indices(n_vertices=...)`, which `submesh_from_vertex_indices`
  and `repair` both supply) per §3.10. Two details that came out of it: fusing the element side's
  `wp.map(shifted_index)` plus Python-scope gather into one `isin_lookup_mask` launch is a further
  **1.22x** (348.5 → 285.6 µs) on the *inferred* path, and that kernel's range guard is what makes
  the bound safe rather than merely wrong — without it an over-tight `max_index` is an
  out-of-bounds gather, i.e. §12.1's host-heap corruption on the CPU device. The guard's test
  bites on exactly that assert (mutation-probed).
- **`_unique_hash`'s two whole-table bookkeeping passes were 40 % of `unique_1d`.** Phase-timed at
  313 µs for 100k int32: `mark_occupied` + `array_scan` + `wp.map(sub)` + readback **93.4 µs**,
  `arange` for the sort's permutation **30.4**, `radix_sort_pairs` **55.3**, `hash_insert` + its two
  `wp.zeros` **56.1**, compaction **55.3**, `bitcast_to_int` **17.9**, two `wp.copy` **25.4**. Three
  output-identical folds took it to **225.2 µs (1.39x)** and 12 → **7** launches: `hash_insert`
  stamps a zero-filled 0/1 `out_occupied` itself (one benign duplicate store per thread, deleting the
  whole-table `mark_occupied` pass); `compact_from_table` takes the inclusive scan's `-1` in its own
  indexing and writes the identity permutation (deleting both the `wp.map(wp.sub)` pass and the
  `arange` launch); and `unique_1d` shares the input buffer when it is already `int32`/`int64`. The
  table is 2.6x the input, so a whole-table pass costs more than a whole-input one. **What is left is
  the radix sort (55 µs) and the two `wp.zeros` over the table; do not expect another 1.3x.**
  `unique_rows` gets 1.15x only, because it feeds `uint64` hashes and cannot skip the bitcast.
  `group(hashed, 2)` measures 388.0 → 388.4 µs, unchanged, and is the **control**.
- **`flatnonzero` uses an INCLUSIVE scan + single tail read** — `scatter_index_where` expects the
  inclusive scan and writes at `inclusive[i]-1`.
- **`intersection._link_segments` is vectorized NumPy pointer doubling**, not a per-segment Python
  walk (which was 0.3 µs per segment, flat over a 61x range, and 84-88 % of the whole call on
  many-curve fields). Two passes of Wyllie pointer doubling — `successor_cycles`' structure (cut each
  cycle at its lowest-indexed node, then rank the chains) extended to cover open chains — plus a
  **lookup-table successor**: an endpoint's unique-edge id indexes "which segment starts here",
  replacing the `argsort` + `searchsorted` and detecting inconsistent winding without a duplicate
  scan. Link only: **1.56x / 4.13x / 5.04x / 5.12x** at 742 / 8 280 / 26 246 / 47 898 segments; end to
  end **1.94x at wave40**, 0.99-1.06x on single-contour rows, output bit-identical at all sizes and on
  18 mixed open/closed level sets. **The device port was not taken**: profiling the vectorized version
  showed the sort was 1.54 ms of 2.46 at 26 246 segments while the two doubling passes were 0.53 and
  0.39, so the lookup table (0.205 ms, no device code) captured most of it — what remains is 1.6 ms of
  a 6.8 ms call, i.e. **~1.4 ms of headroom for a device walk, not the ~7 the Python loop was worth**,
  and a device version would also have *regressed* the single-contour rows. Two details: pass 2
  self-terminates at `log2(longest curve)` (6 rounds against 15 at wave40) while **pass 1 cannot** (a
  cycle's window minimum can stall for a round and then drop, so an unchanged-round test is unsound);
  and fusing the two pass-1 tables into one `(n, 2)` int32 array is a **2.8x loss**, because NumPy
  casts int32 index arrays to `intp` anyway and the column writes are strided.
- **`polyline_downsample` is pointer-doubled above 8 192 points on CUDA only** — §14.7.

### 16.6 `proximity`, `metrics`, `neighbors`

- **`query_hashgrid_nearest`'s cost is *cubic* in how far `initial_radius` under-estimates the actual
  answer distance**, because `_knn_cell_size` is `max(initial_radius, extent / grid_bins)` — *not*
  `initial_radius / sqrt(3)` — so that one scalar sets the cell width *and* the search seed, and a
  scan at radius `r` walks `(2 ceil(r / cell) + 1) ** 3` cells. The default comes from
  `knn_initial_radius(points, k)`, which inverts the **target's** density — the wrong estimator
  whenever the two clouds are displaced. On `dragon` (437 645 points), k=1:

  | `initial_radius` | offset 0 | 0.01x diag | 0.05x diag |
  |---|---|---|---|
  | default | 0.81 ms | 71.3 ms | 259.8 ms |
  | the queries' own median answer distance | 0.81 ms | **2.5 ms** | **28 ms** |

  **It is the cell walk, not the `knn_linear_scan` fallback.** An instrumented kernel counting branch
  outcomes: at 0.01x displacement only **9 rows of 437 645** reach the fallback, at 2.09 grid scans
  per row, and the call still costs 71 ms. Do not re-diagnose this as the fallback. Three probe
  designs were built and all withdrawn (each *exact*, verified against `scipy.spatial.cKDTree` at
  k = 1, 4, 30 — the cost was the obstacle, never correctness): clamping the growth ladder to
  `widest` (a regression — rows pay the widest walk *and* the fallback); a capped strided-sample probe
  (0.3-0.6 ms of fixed launch+readback on *every* call and 2-12 ms on its coarse rungs); and the same
  probe with the target thinned for coarse rungs (fixes the rung cost but still loses at 0.5x
  displacement and on small clouds).
- **SHIPPED, and it is the direction that worked: seed the *backward* search from the *forward*
  half's own answer.** `metrics.py`'s two-direction helpers now derive the backward `query_nearest`'s
  `initial_radius` from the forward distances (`_backward_radius`, a `tw.reduce.max` plus one
  readback). Both directions share one distance scale, so the forward half already holds the
  estimate: no probe, no subsample. Min of 7 interleaved reps, values bit-identical:

  | symmetric call | 8 171 pts | 35 947 | 437 645 |
  |---|---|---|---|
  | `chamfer_points_to_points` | 1.22x | 1.31x | 1.50x |
  | `chamfer_points_to_mesh` | 1.27x | 1.45x | **3.22x** |

  In the harness: `chamfer_points_to_mesh[dragon]` 199.2 → 64.2 ms (now a 2.60x **win** over meshlib,
  was a 1.20x loss) and `chamfer_points_to_points[dragon symmetric]` 363.7 → 244.7.
  **Two things measurement refuted — do not re-propose either:** `backend="bvh"` at these call sites
  is a **LOSS at every size** (0.45x / 0.73x / 0.95x on the displaced pair), and the old "290 ms
  hashgrid against 146 ms bvh at 0.05x offset on dragon" **no longer reproduces under Warp 1.17**
  (185 vs 194 ms) — that record's 2x has expired.
- **CLOSED — REFUTED: seeding the *forward* pass from a query-prefix probe cannot be gated on size,
  and the "sign change across scale" reading of it was wrong.** The design (probe a prefix, seed
  `initial_radius` from its answers, gate on the query count the way
  `CG_MULTIGRID_SIZE_FLOOR` gates) was measured properly and does not survive. Two findings, in the
  order they killed it:
  - **Across size it looks exactly like a gateable cliff.** Sweeping `dragon`'s vertices at the
    benchmark's own 0.05x displacement, probe 1 024, min of 5, distances bit-identical in every
    cell: **0.63x** at 8 171, **0.60x** at 35 947, 0.61-0.62x through 70 000, then **1.22x** at
    80 000, 1.37x at 160 000, 1.66x at 300 000, **1.95x** at 437 645. The transition is a cliff, not
    a slope — the *unseeded* cost jumps 8.52 → 18.83 ms between 70 000 and 80 000 — and it is
    genuinely `n`-driven, because `_knn_widest_grid_radius` is itself `n`-aware. A floor at 100 000
    fits that sweep perfectly.
  - **And the fit is an artefact of one displacement.** Holding `n` at 437 645 — far above any floor
    — and varying only how far the clouds sit apart: two independent samplings of one surface
    **1.10x**, a 0.005x translation **0.67x**, the benchmark's 0.05x **1.95x**, a 0.5x translation
    **0.42x**. A size gate would therefore ship a 2.4x regression on a widely separated pair. This is
    §16.8's "the size branch did not transfer" a second time, and it is why the earlier record of
    this idea as "0.90x / 0.96x / 2.40x, needing a defensible gate" was the wrong axis: the sign
    changes with the **ratio of the answer distance to the point spacing**, not with the size, and it
    is not monotonic in that either.

  **What the sweep found instead is a much larger prize in a different place.** `initial_radius`
  sets the hash-grid *cell width*, and that — not the ladder start — is the whole effect: sharing one
  grid between probe and query, so only the ladder start improves, measures **0.98x** at `dragon`
  against the fresh-grid arm's 1.94x, while the grid build itself is **0.10 ms**. Sweeping the width
  directly as a multiple of the default (`dragon`, 437 645 points, ms):

  | displacement | 1x | 2x | 4x | **8x** | 16x | 32x | 64x | probe's own seed |
  |---|---|---|---|---|---|---|---|---|
  | 0.005x diag | **1.0** | 1.2 | 2.7 | 10.2 | 42.2 | 115.5 | 210.3 | 2x |
  | 0.05x diag | 160.6 | 188.8 | 107.2 | **24.1** | 40.4 | 106.9 | 216.9 | 20x |
  | 0.5x diag | 147.3 | 148.4 | 148.5 | 148.3 | 148.5 | 275.7 | 341.1 | 173x |

  So the shipping default is optimal at 0.005x; at 0.05x the optimum is **8x the default and 24.1 ms
  against the shipping 160.6 — 6.7x**, three times what the probe's 1.95x recovers, because the probe
  seeds at 20x and overshoots; and at 0.5x nothing under 32x moves the number at all. **The optimum
  cell width is neither the density nor the answer distance**, which is exactly why no statistic of a
  probe finds it — `max` / `mean` / `p50` / `p90` all measure 0.45-0.65x below the cliff and
  1.38-1.87x above it, i.e. the statistic is not the variable.

  **The open lead is therefore `neighbors._knn_cell_size`, not `metrics.py`**: a 6.7x sits between the
  default width and the best one on the benchmark's own input, and finding it needs a model of the
  walk's cost against the width rather than another estimate of the answer. Do not re-propose the
  forward probe.
- **`query_nearest`'s famous 16x non-monotonic drop is real, but it is `k >= 8` and hash-grid-only.**
  `benchmarks/README.md` carried for several rounds the strongest negative claim in the benchmark
  prose — 0.82, 3.25, 7.38, **0.46**, 0.75 ms at 5 k / 20 k / 50 k / 100 k / 200 k, "identical to
  three digits between the `bvh` and `hashgrid` backends, so the cost is in a stage the two share",
  and "triwarp is 10x off its own 100 000-point cost here". Re-measured **interleaved across size and
  backend in one pre-warmed process** (the original was a sequential sweep, the shape that
  manufactures this artifact): the measurement **reproduces to ~3 %** and both conclusions drawn from
  it were wrong.

  | n | hashgrid k=8 | bvh k=8 | hashgrid k=1 | bvh k=1 |
  |---|---|---|---|---|
  | 5 000 | 0.837 | 0.695 | 0.229 | 0.316 |
  | 20 000 | 3.262 | 0.774 | 0.236 | 0.355 |
  | 50 000 | 7.567 | 0.854 | 0.249 | 0.402 |
  | 100 000 | **0.472** | 1.368 | 0.269 | 0.516 |
  | 200 000 | 0.801 | 2.023 | 0.346 | 0.667 |

  **At `k = 1` there is no effect at all** — flatly monotonic, and per query the cost *falls*
  45.8 → 1.7 ns; the published figures were a `k = 8` sweep read onto the `*_k1` rows. And **it is
  hash-grid-specific, not shared**: the BVH is monotonic and beats the grid **1.2x / 4.2x / 8.9x** at
  5 k / 20 k / 50 k before losing 2.9x / 2.5x at 100 k / 200 k. Mechanism: a row whose true `k`-th
  distance runs past `_knn_widest_grid_radius(cell, n)` abandons the walk for an exact linear scan
  (50 000² tests in 7.567 ms is 3.3e11 tests/s — a scan, not a search), and the uniform-density
  estimate has no distribution behind it, so the *tail* of the true `k`-th distance trips the cutover
  and how much of the cloud is in that tail moves with `n`. **So `backend="bvh"` is the workaround at
  moderate `k` on a uniform cloud — the opposite of what the docstring used to imply.**
- **`k`, not `n`, is what is still slow in the k-NN path.** On `sphere_small` (2 562 points,
  self-query): k=1 0.353 ms, k=8 0.619, k=16 1.035, **k=30 2.481**, **k=64 6.220** — super-linear,
  because `knn_sorted_insert` keeps the candidate row in the *output arrays* (global memory) and does
  a binary search plus two shift-inserts there per accepted candidate, while `knn_reset_row` rewrites
  the whole row on every deepening attempt. That is what `points.statistical_outlier_mask` (k=30) and
  `outlier_probability` pay for. The register-row rewrite (§2.9) addresses exactly this.
  *(Historical: an older record of a ~21.6 ms `query_bvh_nearest` at k=1 and a ~10x ICP loss to open3d
  is superseded — k=1 measured 0.545 ms on bunny and 1.918 ms on dragon against scipy's 17.99 / 140.3,
  and the ICP rows show no loss.)*
- **SHIPPED: `mesh_to_mesh_distance` now reads the `wp.Mesh`'s own BVH** (`wp.mesh_get_bvh`, Warp
  1.17) instead of building a second `wp.Bvh` over per-face AABBs. Distance and both witness face
  indices **bit-identical in all six cells**: 1.07x / 1.00 (bunny_decimated near/far), 1.01 / 1.05
  (bunny), **1.18 / 1.23** (dragon) — it grows with size because the removed build was 9.2 % / 7.5 /
  18.1 of the call. The per-face AABBs stay (the kernel's box-gap prune reads them), and Warp's own
  leaf policy (vs `leaf_size=4`) did not cost the traversal. **The operating point decided this
  item's sign** — §15.3.
- **CLOSED: `mesh_to_mesh_distance`'s cost was its own upper bound, and the bound did not need
  every vertex.** The remainder §16.1 had as "the structure builds and the bound" is 93.1 % **the
  bound alone**, and it is one kernel: stage-attributed on `lucy` at the benchmark's own operating
  point (a disjoint copy at 1.2x the x-extent), warm, one call between two syncs — the vertex query
  is **702.78 ms of a 758.47 ms call**, against 31.58 ms (4.2 %) for the `wp.Mesh` build and
  **6.13 ms (0.8 %) for both traversal passes together**. Fourteen million closest-point queries
  were being paid to prune a walk worth under a percent of the call.

  The bound only seeds the broad phase's prune limit, so a **subsample** of A's vertices is exactly
  as sound and merely looser — and it barely loosens, because the nearest approach is not a rare
  event on a surface: 0.0464651 against an exact 0.0459145 from **2 048** of `dragon`'s 437 645
  vertices, 1.2 %. Shipped as `proximity._BOUND_SAMPLE_TARGET = 16_384` (a stride, so no RNG and no
  gather). A/B against a detached baseline worktree, interleaved, min of 5, at both benchmarked
  offsets:

  | mesh | near | far |
  |---|---|---|
  | `bunny` (35 947 v) | 0.99x | 0.98x |
  | `dragon` (437 645) | 1.22x | 1.12x |
  | `happy_buddha` (543 652) | 1.36x | 1.37x |
  | **`lucy` (14 027 872)** | **10.1x** (744.3 → 73.5 ms) | **9.06x** (651.9 → 72.0) |

  The distance is **bit-identical in all eight cells** and so is `face_a`; `face_b` differs in the
  two `happy_buddha` cells, which is the tie the function's own Notes already declare unspecified.
  Bit-identity is not luck — a looser limit prunes *less*, so the narrow phase sees a superset of
  the candidates it saw before and the minimum over a superset containing the argmin is the same
  float. The gain grows with the mesh because the removed work does and the rest does not, which is
  the opposite of §9's falling-share decline; the target is a *count* for that reason.

  **The sweep is flat and that is the useful part** — 1.14-1.24x on `dragon` and 1.27-1.45x on
  `happy_buddha` across targets from 2 048 to 65 536 — so 16 384 is a middle with margin rather
  than a tuned optimum, and it does not need re-probing after an upgrade. One earlier reading of
  0.88x at 1 024 points on `dragon` was an **unwarmed** arm and does not reproduce.

  **It also uncovered a platform defect that gates it**: widening the query box is what first
  reaches `wp.tile_bvh_query_aabb`'s result-buffer overrun (§12.2), which faults deterministically
  on `lucy[near]` — through the *public* `upper_bound=` parameter on the shipping code, before this
  change existed. The bound-check went in first, as its own commit.
- **OPEN: `mesh_to_mesh_distance[lucy]`'s 105x for 26x the faces is NOT the load imbalance.**
  Instrumented at 28 055 742 faces: pass 1 (thread per face, cap 64) is **14.5 ms**, **zero** faces
  overflow the cap, and pass 2 is 0.003 ms — against 789-853 ms for the whole call. So the traversal
  is ~2 % and the cost is in the structure builds and the bound; the imbalance that motivated the
  two-pass design (98.2 % of faces returning no candidate, §14.2) simply does not appear at that
  scale. By contrast `happy_buddha` has 14 overflowing faces and pass 2 is **81.8 %** of its two
  passes. **Partly closed, and the missing piece was host time.** The remainder above was attributed
  to "the structure builds and the bound" because a device profile is where it was looked for;
  **12.5 % of it was two `.numpy()` calls in the function's own tail**, each copying a whole
  per-face array to the host to index one element. Priced directly at the benchmark's own operating
  point, min of 15: `bunny` 0.106 → 0.048 ms (3.34 % → 1.52 % of a 3.18 ms call), `dragon`
  0.490 → 0.058 (8.88 % → 1.05 % of 5.51 ms), **`lucy` 102.6 → 0.12 ms (12.46 % → 0.01 % of
  823.9 ms)** with `_device.read_scalar`, which takes an arbitrary index. The share *grows* with the
  mesh, the inverse of §9's falling-share decline. Two lessons: read the *host* half before
  accepting a device attribution, and a `.numpy()[k]` on an array that scales with the mesh is the
  shape to grep for (`wp.mesh_get_bvh` had already taken the build out of this same function).
- **`cotmatrix`'s 3.7-5.1x loss to pytorch3d is a scope mismatch, and the rewrite it invited is
  DECLINED.** `p3d_ops.cot_laplacian` does **not assemble a sparse matrix** — it wraps `3F` entries
  as an uncoalesced `torch.sparse_coo_tensor` and adds its transpose, so duplicate `(i, j)` pairs are
  never summed and **no diagonal is ever written**, where `laplacian.cotmatrix` sorts, dedups and
  accumulates 12 triplets a face into CSR *with* its assembled row sum:

  | mesh | triwarp | pytorch3d | + `.coalesce()` |
  |---|---|---|---|
  | bunny_decimated | 0.526 ms | 0.514 (1.02x) | 0.589 — **triwarp wins 1.12x** |
  | dragon | 3.363 | 0.728 (**4.62x**) | 3.042 — **1.11x**, parity |

  Structure confirms it: on `dragon` the uncoalesced tensor holds 5 228 484 entries (`6F`), coalesced
  2 618 512, and triwarp's `nnz_sync()` is 3 056 157 — a difference of **437 645, exactly the vertex
  count**, i.e. the diagonal, whose coalesced absmax measures **0**. Assembling into an `edges_unique`
  pattern with a `binary_search_index` per entry would have been a medium-large build against a gap
  that does not exist.

  **Re-verified on Warp 1.17, including at `lucy` — the row whose 3.7x headline made this look like
  the biggest assembly gap in the suite.** `dragon` 2.431 / 0.631 / **3.025** ms and `happy_buddha`
  3.046 / 0.803 / **3.906** (triwarp / raw / coalesced), reproducing the recorded table to within
  7 %, so triwarp is **1.24-1.28x ahead** once pytorch3d does the same job. `lucy`'s coalesce cannot
  be re-run on a box with 12 GiB already committed elsewhere (it OOMs at 10.4 GiB peak), but its
  **raw arm is the control and reproduces exactly** — 28.247 ms against the harness's 28.320 and
  `nnz` 168 334 452 to the digit — so the recorded 128.0 ms coalesced figure stands and triwarp is
  **1.22x ahead at `lucy` too** (105.3 ms). The scope mismatch therefore holds at *every* size.
  - **And the bookkeeping fix this invites is a defect: `cotmatrix / pytorch3d` is already COVERED
    by a live class-B parity test.** Adding `noparity` would delete a real comparison from the
    matrix — `noparity` is for a benchmarked pair whose *results* are incomparable, and these
    agree once the named coalesce transform is applied. What is incomparable is the **timing**, and
    the benchmark docstring already carries the full table. Leave both alone.
- **`repair.fix_self_intersections(method="local")` was the suite's largest single loss for four
  rounds, and it was never a loss: the reference call is a no-op.** `mm.localFixSelfIntersections`
  returns its input byte-for-byte on this row's fixture, all 1 176 colliding faces intact, at every
  configuration probed — full detail and the single-component requirement behind it in §7.6. So the
  4.3-5.0x (127.8 ms) was triwarp's real repair, 1 176 → **126** intersecting at the default
  `max_iter=3`, timed against a call that returns its argument. **Do not re-derive this as a scope
  discount argument** — it was not that triwarp does more work, it is that the reference did none
  on that input.
  - **The group now runs on the `tangle` axis instead — a self-intersecting single-component torus
    at 8 192 and 163 840 faces — and the loss is gone rather than exempted.** All four cells do
    real work, both libraries carry §7.6's assert-it-mutated guard, and the harness reads: local
    46.2 against meshlib's 14.7 at 8 192 faces (**3.14x behind**) and 188.3 against 183.0 at
    163 840 (**1.03x**), with voxel winning 4.8-5.8x at both. The size axis is there because this
    is a crossover — a serial C++ fixer leads while the mesh is small — and one row would report
    whichever side of it the fixture landed on.
  - **Read that 1.03x with its quality caveat, which runs the other way**: at 163 840 faces triwarp
    leaves **20** intersecting of 884 where MeshLib reaches 0 (both reach 0 at 8 192). So the large
    cell is parity for a marginally less complete repair, and `max_iter` is what closes the
    residue.
  - **What the earlier reading got wrong is instructive: it measured only triwarp's side with a
    detector.** "1 176 in, 158-365 out, and meshlib also reduces without clearing" was recorded
    twice and refuted twice as a scope mismatch (round 7's T3, round 8's U3), each time arguing
    about *whether both reduce*. Applying triwarp's detector to **both** outputs settles it in one
    call. That is §7.7's rule — ask what the reference was handed and whether it finished the job —
    and the benchmark's own assert (`numValidFaces() > 0`) could not see it.
  - **REFUTED: the DP is not this row's cost, and every DP lever is capped at 4-10 %.** The
    attribution above was taken on the retired two-sphere fixture and its "the cost is three
    min-weight DP sweeps … and that sweep is launch-bound" does **not** transfer to the `tangle`
    axis this group now runs on. Re-measured stage by stage on the benchmark's own inputs
    (`_run_hole_dp`, `delete_region_keep_boundary`, `subdivide_region_to_size` and both smoothers
    wrapped, warm, one call between two syncs):

    | stage | `tangle_torus_small` (8 192 f, 47.1 ms) | `tangle_torus` (163 840 f, 209.5 ms) |
    |---|---|---|
    | `refill_region` | 43.56 ms (92.5 %) | 194.54 ms (92.9 %) |
    |  `smooth_region` (least squares) | 19.54 (41.5 %) | 79.90 (38.1 %) |
    |  `subdivide_region_to_size` | 10.80 (22.9 %) | 53.42 (25.5 %) |
    |  `smooth_region_fixed_rim` | 5.07 (10.8 %) | 15.91 (7.6 %) |
    |  **the min-weight DP sweep** | **1.88 (4.0 %)** | **21.27 (10.2 %)** |
    |  `delete_region_keep_boundary` | 3.79 (8.0 %) | 13.32 (6.4 %) |
    | `_dilate_face_mask` | 1.69 (3.6 %) | 6.86 (3.3 %) |
    | `face_self_intersecting_mask` | 1.59 (3.4 %) | 6.79 (3.2 %) |

    **The rims are short.** One pass opens 4 rims of 64 vertices on the small mesh, and 4-6 rims of
    at most 327 on the large one (top sizes `[321, 321, 203, 54, 52, 43]`, median 128, over 4 DP
    calls) — not the "5 rims, longest 642" the retired fixture produced. So a **bounded candidate
    search** (MeshLib's `getOptimalSteps` caps the apex scan at ~20 past
    `maxPolygonSubdivisions`, making the DP `O(n²·20)` rather than `O(n³)`) is worth nothing at
    `max_size` 64 and at most a fraction of 10.2 % at 327 — while costing `fill_min_weight`'s
    documented exactness. The **blocked interval DP** is bounded by the same 4-10 %: it removes
    launches from a sweep that is already a tenth of the call. Both are declined **for this row**;
    the blocked DP is **also declined for `fill_min_weight` / `fill_smooth`**, where the sweep *is*
    the call but there is no loss to close: against `meshlib`, the only reference running the same
    minimum-weight DP (the others do a topological fill and the group docstring says so), triwarp
    is **5.0x ahead** on `rim_short` (14.88 ms against 74.47), **2.1x** on `holes_many` (4.86
    against 10.17), and ahead on `stitch_min_weight` (23.21 against 27.74). The ~17 µs-per-span
    launch floor is real and is nobody's bottleneck. **Build the blocked DP when a row appears that
    it would flip.** The one hole-family row that does lose, `fill_smooth[rim_short-refined]` at
    74.69 against 41.59, is 13.85 ms of DP and ~61 ms of refinement and smoothing — the same split
    `fix_self_intersections` shows above, which is the second independent sighting of it.
  - **What is left is the smoother and the refiner, 64-72 % between them**, and that is where a
    lever for this row has to come from. `face_self_intersecting_mask` is 3.2-3.4 %, so the
    detector is still not the cost. **Four refuted levers, unchanged:** graph capture of the chain
    (a once-through loop records and replays once, 0.84x); the DP block knob (it ships, but is
    gated on a narrow grid); one persistent block per loop (built and reverted, 0.03-0.89x); and
    the scope-discount argument above, superseded rather than refuted. **A fifth, already
    recorded elsewhere:** §16.8 measured `refill_region`'s patch solves at dominance 2.69-4.02 and
    found all four *lose* under a forced multigrid hierarchy, because a 17-25 ms setup is most of
    a patch solve — so the 38-41 % `smooth_region` share is not reachable that way either.
  - The `voxel` sibling on the identical input is a **5.19x win** (17.3 ms against meshlib's 90.0,
    re-measured), and meshlib genuinely repairs there, so **the method choice — not the method's
    implementation — is still the available answer for a caller today.**
- **The hole DP is launch-bound.** `holes._run_hole_dp` runs one launch per triangulation span,
  `max(B) - 1` for the whole mesh. Replaying the identical launch at `dim=(1, 1)` isolates
  marshalling: the floor is a flat **16.4-18.6 µs per launch** and accounts for **37 % / 61 / 74 /
  52** of the sweep at one rim of B = 128 / 256 / 512 / 1024 (5.610 / 7.305 / 12.893 / 32.675 ms).
  Apex throughput over the same range is 0.06 / 0.38 / 1.73 / 5.48 G evaluations a second — three
  orders below the arithmetic — so **the device is idle waiting for launches**. The older claim that
  the sweep is "device-bound at a large rim" holds only past B ≈ 1 000, which no benchmarked rim
  reaches. **Shipped: `hole_dp_block(max_size, n_loops)`, 32 or 128 lanes** — worth 1.10-1.13x on
  `rim_short` (faces bit-identical) and up to 1.45x at one rim of 1024. **It is an occupancy gate and
  getting that backwards costs 12 %**: the grid is `(n_loops, max_size - span)`, so a wide block pays
  only when that grid alone would starve the device — a first version keyed on `max_size` alone was
  caught at **0.88x** on a *scattered* deleted region (99 rims, max 3025), and 32 → 128 measures 1.13x
  at 2 rims, 1.03x at 8, **0.93x at 32**, **0.89x at 128**. Within the winning corner the size is
  mesh-dependent (1.01x on a capped tube against 1.13x on `rim_short` at the same shape), so the table
  is a floor, not a formula. **Remaining lever, unbuilt:** a **blocked** interval DP — tile the
  `(i, j)` plane into `T × T` tiles so the tile-level DAG keeps its shape at `1/T` the span count,
  cutting ~511 launches to ~16 at `T = 32`.
- **`holes._closest_loop_pair`'s three launches are deliberately left alone** — *"this preamble … is
  5.0 % of the call at a 100-vertex rim on CUDA and falls to 0.7 % at 1 000 and 0.4 % at 4 000."*
  Likewise folding `holes.global_argmin` into `row_argmin` is **1.51 % / 0.58 % / 0.41 %** of
  `stitch_loops` at rims of 100 / 1 000 / 4 000, launch-dominated. Both are §9's "a share that falls
  as the input grows is a decline".

### 16.7 `sample`

- **`sample_surface_blue_noise` is randomized-priority parallel dart throwing, not Bridson.** Every
  pool point draws a priority from the seed; a point is accepted when no smaller-priority point still
  in play lies within `r`; everything within `r` of an acceptance is discarded; iterate. Measured:
  `bunny_decimated` **263 → 42.9 ms** at the 2k-sample radius (6.1x) and **542 → 54.2** at half of it
  (10.0x). The row inverts — triwarp now beats pymeshlab 1.7x and 5.4x where it lost 4.9x and 2.2x.
  Spacing is **exactly** `1.000 r` (was 0.99-tolerant) and the worst coverage gap is 1.08-1.10 r,
  tighter than MeshLab's 1.11-1.19 and Open3D's 1.20-1.22. **This supersedes the earlier "propose is
  inherent, already well-tuned" conclusion** — that was true of the *kernel* and false of the
  *algorithm*: the cost was 98 rounds × a 729-cell shell; the replacement is 27 cells × a handful of
  rounds. `_bridson_blue_noise` and its fourteen kernels are deleted. **Reach for randomized-priority
  selection whenever a GPU port needs a maximal-packing / MIS-shaped result** — the tie-break-free
  correctness argument (the later of any too-close pair was already discarded) is what makes it safe
  against a stochastic output, and it is the serial algorithm's own distribution.
- **SHIPPED: a per-cell summary prunes both dart kernels' 27-cell shell scan for 1.70-2.33x,
  byte-identical.** `dart_cover_neighbors` retires an alive point only if an **ACCEPTED** point is
  within `r`, so a `wp.bool` per cell ("anything accepted here this round") skips the cell outright;
  `dart_select_minima` rejects a point only if a smaller-priority not-COVERED point is within `r`, so
  a `wp.uint32` per cell holding the minimum live priority skips it. At the radii the benchmark scores,
  interleaved, min of 5: **cover prune alone 1.51x / 2.06x, select alone 1.06-1.08x, both 1.70x /
  1.72x / 2.33x**, with the `state` array **byte-identical in all four modes**. End to end on the four
  benchmarked configurations: 42.32 → 24.59, 48.74 → 25.11, 46.68 → 27.34 and 132.07 → 57.38 ms. It
  came out as **two** extra launches per round, not three: the covering summary is written by
  `dart_select_minima` at the point it accepts, and the two resets are `fill_` memsets (~3.1 µs)
  rather than kernels (~9.7). **This is not the refuted thread-mapping inversion (§15.3)** — it
  removes work whose result was already determined, so the set is identical by construction. The
  select summary is built over the *alive* list, which excludes ACCEPTED points; that is safe because
  a point accepted in an earlier round had its `r`-neighbourhood covered in the same round.
- **`blue_noise`'s cover pass is load-bearing for termination.** A 26-cell shell or a 0.9 r cover
  radius does not merely change the answer, it **stops the loop converging within 500 s**, because
  `dart_select_minima` will not accept a point while a smaller-priority *alive* point sits within r.
  Its byte gate therefore bites hard and is cheap to run.
- **The residual `blue_noise` gap to meshlib is 16-46 % per-round dispatch, not "most likely" all
  of it.** The standing reading — two shipped fixes in, MeshLib still 1.26-2.44x ahead at
  `bunny` / `bunny_decimated` — was that the remainder "most likely reflects fixed per-round GPU
  dispatch cost against a tight single-threaded C++ loop", which is a hypothesis rather than a
  number. Counted at HEAD, **5 launches per round** (the dart pair, the cell summary, the alive
  flags and the compaction):

  | cell | rounds | launches | triwarp | meshlib | dispatch floor @ ~12 µs | share of the gap |
  |---|---|---|---|---|---|---|
  | `bunny_decimated` r=1.0 | 36 | 196 | 26.216 ms | 11.787 | 2.35 ms | **16 %** |
  | `bunny` r=1.0 | 112 | 576 | 29.066 | 11.909 | 6.91 | **40 %** |
  | `bunny` r=0.5 | 91 | 471 | 59.858 | 47.448 | 5.65 | **46 %** |

  So dispatch is a real and substantial share and not the whole story; the majority is device work
  at the tighter radius. **The unexpected lever is the round count, which does not track the output
  size**: `bunny` at r=1.0 spends **112** rounds producing 46 427 samples where `bunny_decimated`
  at r=0.5 spends **36** producing 43 052 — 3x the rounds for the same answer. Bringing the first
  to the second's round count would take its dispatch floor from 6.91 to 2.2 ms, ~27 % of that
  cell's gap. Why one cloud needs 3x the rounds of another at a comparable sample count is
  unmeasured and is where this row's remaining headroom is; §9's rule applies — a benchmark for the
  round count lands before any change to it.
- **SHIPPED: `points.farthest_point_sample` as one persistent block** — §14.1.

### 16.8 `linalg`, `smoothing`, `laplacian`

- **SHIPPED: `linalg.multigrid_preconditioner`** (aggregation, smoothed prolongator, Galerkin
  hierarchy, batched V-cycle inside the captured CG loop). On `smooth_region`'s normal equations the
  *solve* is **2.46x on `bunny`**, 2.55x on CPU, at a 12.5x iteration reduction; the harness row went
  **225.7 → 159.7 ms**. The parallel MIS-2 aggregation is within **7-12 %** of pyamg's serial
  `standard_aggregation` — it was never the risk.
- **The setup is the blocker, and it is 3-4x what was budgeted.** 12-17 ms, near flat in the operator,
  because it is 8 sparse-op calls *per coarsening level* at Warp's fixed per-call cost (`bsr_mm`
  alone 0.82 ms × 6 on `bunny`'s two levels, the aggregation's launches 3.53, the prune 1.78,
  `bsr_transposed` 0.76). It was 18 ms until the power iteration stopped using `bsr_mv`: one fused
  `power_step` kernel plus a sign-vector start (whose norm is exactly `sqrt(n)`, so one host sync
  instead of two) took that stage **2.57 → 0.76 ms** and the whole build 18.2 → 15.4. **Every** losing
  case loses by exactly that setup: `smooth_region_fixed_rim` 0.48-0.53x, `harmonic k=1` 0.43-1.13x,
  and `fill_smooth` **0.21x** when `smooth_region` asked for the hierarchy on every hole patch. With
  a free setup all of them would win. **Cutting `bsr_mm`'s per-call cost is the lever.**
- **Two predictors were built and refuted, which is why the shipped policy is a cap.** *Size*: at
  ~2 000 free unknowns `bunny_decimated` takes 1 784 Jacobi iterations and wins 1.88x, while an
  `icosphere` at the same size takes 521 and loses 0.69x. *Extrapolating the probe's convergence
  rate*: at probe lengths 200 / 400 / 600 the estimated remaining count of the *losing* systems
  (2 993 / 4 307 / 6 481) interleaves with the winning ones' (2 410 / 4 336 / 5 795) at every length —
  what decides the ratio is the V-cycle's *own* iteration count, unknowable without building the
  hierarchy (a one-point estimate from iteration 0 is worse still: 558 for a true 6 341, because CG's
  early reduction is far faster than its tail). So `preconditioner="auto"` runs Jacobi for
  `CG_PROBE_ITERATIONS = 2000` and escalates only on non-convergence — 0.98-1.01x on everything a
  hierarchy would not have helped, 1.47x on `bunny`'s benchmarked region, 2.59x on three quarters of
  it, and it forgoes the upside just past the cap. **The first `"auto"` returned the capped probe's
  unconverged iterate — silently wrong by 1.3 absolute while looking 20x faster**; the escalating
  branch is now pinned by a test with the probe forced to one iteration. Two sweeps per half-cycle,
  not one (735 / 524 / 445 / 400 iterations at 1 / 2 / 3 / 4 sweeps, 104.0 / 90.8 / 90.6 / 93.1 ms).
- **What separates the winning systems is the *operator's* off-diagonal dominance, not the unknown
  count.** `solve_spd_columns(preconditioner="auto")` gates on
  `max_i sum_{j!=i} |A_ij| / A_ii` (`linalg._offdiagonal_dominance`, 0.12-0.13 ms flat) crossed with a
  size floor. Regular meshes read 1.90-2.04 dominance, irregular ones 2.22+; that gap is what
  separates them, and the size floor exists **for the hole patches, not for smallness** —
  `fill_smooth` / `refill_region`'s patch solves carry dominance 2.69-4.02 (a freshly triangulated
  patch has worse triangles than any scan) and all four *lose* under a forced hierarchy because a
  17-25 ms setup is most of a patch solve. Worth 2.02x / 1.80x on the two benchmarked `smooth_region`
  rows; the gate runs *before* the probe and can only remove one, so a declined system is bit-for-bit
  the old behaviour.
- **The gate's two branches generalize differently, and `lscm` is the row that proves it.**

  | operator | dominance | forced V-cycle |
  |---|---|---|
  | `smooth_region` umbrella normal equations | 2.03-3.41 | wins 1.15-3.62x |
  | `harmonic k=2` (squared Laplacian) | 1.94-2.66 | wins 1.42-2.92x |
  | `lscm` coupled u/v | 1.55-1.80 | **loses 0.58-0.67x** |
  | `harmonic k=1`, `tutte`, `min_quad_with_fixed` | 1.00-1.20 | 0.41-1.09x |

  The **dominance** branch transferred to a class it was never fitted on. The **size** branch
  (n ≥ 7000) did not: it fired on `lscm` and on a large well-conditioned Laplacian and cost
  0.41-0.67x on rows that were winning. It now needs `CG_MULTIGRID_SIZE_FLOOR = 1.9` — between
  `lscm`'s 1.798 and `smooth_region`'s 2.035, the thinnest margin of the four thresholds. Two traps:
  **`min_quad_with_fixed` is not a good `"auto"` caller** (0.90-1.09x over 8 cells) — only a *squared*
  operator is, which is why `harmonic` switches on `k` rather than on the gate alone; and
  `_offdiagonal_dominance` read a flat **0.000** on a negative-diagonal Laplacian until it took the
  magnitude on both sides, a number that looks well-conditioned and silently declines the gate.
  **Re-run the whole 30-system table before moving any threshold, and never route a new caller
  through `"auto"` without measuring it.** `harmonic k=2` is 2.25x on `saddle` and 0.23x on
  `saddle_graded` at identical connectivity, so a strength-of-connection threshold (`theta > 0`) is
  the open lead for anisotropic operators.
- **The gate's decline branch generalizes to a fifth operator class it was never fitted on: the heat
  Poisson system.** `heat.heat_operators`' docstring names *"a preconditioner stronger than Jacobi on
  a cotangent operator"* as the only lever for `heat_geodesic`, and §15.10's correction confirms that
  call is 87 % device and iteration-bound — so the hierarchy is the obvious thing to try. Measured on
  `-L` with a mean-zero right-hand side (a random one is out of range — the constants are in the
  nullspace — and runs CG to its 25 620-iteration cap, which is the docstring's own trap in a second
  guise), `tol=1e-8`:

  | mesh | n | dominance | Jacobi | multigrid | build | solve | with build |
  |---|---|---|---|---|---|---|---|
  | `sphere_small` | 2 562 | **1.00** | 100 it / 3.59 ms | 10 it / 3.78 | 13.06 ms | 0.95x | **0.21x** |
  | `sphere_med` | 40 962 | **1.00** | 380 it / 10.60 | 20 it / 6.44 | 25.10 | 1.65x | **0.34x** |
  | `saddle` | 17 689 | **1.20** | 710 it / 17.17 | 20 it / 6.12 | 19.68 | 2.80x | **0.67x** |

  Iterations fall **10-35x** and the solve wins up to 2.80x, and the setup loses all of it — the same
  sentence as every other losing case above. **The gate is right without being touched**: all three
  read dominance 1.00-1.20, far below `CG_MULTIGRID_SIZE_FLOOR = 1.9`, so `"auto"` declines them. It
  flips only for a caller that solves **four or more** times against one `heat_operators`; no in-repo
  caller does, so §4.2 says do not build the keyword. Re-open this if one appears.
- **OPEN: `robust_laplacian` keeps a -16 off-diagonal on Dini's surface although
  `intrinsic_delaunay` reports convergence.** On `parametric_surface("dini")` (40x40, 1600 vertices,
  edge lengths 3.3e-4 to 4.02 — a 12 000:1 ratio, face areas 5.4e-5 to 0.033), `robust_laplacian` with
  its default `use_intrinsic_delaunay=True` returns a matrix whose smallest off-diagonal is
  **-16.13**, contradicting its own docstring; the plain `cotmatrix` on the same mesh reads -40.5.
  `remesh.intrinsic_delaunay` reports **3792 flips and is stable at `max_iter` 100 / 400 / 2000**, so
  it believes it converged; both matrices are finite (mollification is working) and no face is
  zero-area. **Untriaged** — either the parallel flipper has a fixed point that is not intrinsically
  Delaunay on strongly graded input, or float32 intrinsic edge lengths break the flip predicate at
  this length ratio. It is the first graded open patch the suite can build, which is why nobody had
  seen it. **Do not add `dini` to
  `test_intrinsic_delaunay_removes_negative_cotangent_weights` until this is resolved — it fails.**
  Build the mesh with `tw.creation.parametric_surface("dini")` to reproduce (there is no fixture,
  deliberately).
- **`smoothing`'s two conditional-emit triplet writers must `rows.fill_(n_rows)` before launch** —
  §12.7, measured 31.8x and 9.1x. `holes.fill_smooth` reaches both through
  `smoothing.refine_and_smooth_region`, so one fix lands on three benchmark rows.

### 16.9 The 0.3-5.0 ms loss band, attributed

Attributed per row before touching any of it (97 rows, 158.1 ms by gap). Top groups:

| group | gap ms | rows | status |
|---|---|---|---|
| `split_array` | 15.06 | 6 | per-segment `wp.clone` floor — 256 × ~15 µs, the measured constant (§13.1) |
| `transport_tangent_vectors` | 8.62 | 2 | **attributed** — 71 % *device*, `warp.optim.linear`'s two `TiledDot` kernels are 42 % of the solve (§15.10) |
| `cluster_decimate` | 8.59 | 4 | flat over a 67x face range, wins 9.14x at `dragon`. Not a work item |
| `chamfer_points_to_points` | 7.71 | 5 | **closed** by the backward-radius seeding (§16.6) |
| `laplacian_inverse_distance` | 7.50 | 4 | **closed** — `edges=` made two cells wins |
| `remove_degree3_vertices` | 6.94 | 2 | nothing to hoist: each pass deletes faces, so the next halfedge structure is over a different mesh |
| `concatenate_arrays` | 6.70 | 5 | the same per-segment floor; its pytorch3d row prices the alternative |
| `mesh_to_mesh_distance` | 6.69 | 4 | **improved** 1.00-1.23x (§16.6) |
| `is_watertight` | 5.88 | 2 | documented scope mismatch (meshlib reads a cached `isClosed`) |
| `cotmatrix` | 5.87 | 2 | **closed** as a scope mismatch (§16.6) |
| `fillable_loop_mask` | 5.67 | 3 | `edges_unique` at 62-66 %; deleting the chord test outright still leaves ~0.4 ms |
| `homology_generators` | 4.94 | 1 | **attributed** — `graph.bfs` is 3.57 of `tree_cotree`'s 6.09 ms and is ~100 % device (§15.10) |
| `lscm` | 4.83 | 1 | the ill-conditioned-solve family |
| `pack_1d_arrays` | 4.02 | 3 | the per-segment floor |
| `marching_triangles` | 3.98 | 2 | **not real** — a median artifact at `rounds=3` (§15.4) |
| `vector_heat_scale` | 3.82 | 1 | **attributed** — the same transport, so the same CG (§15.10) |
| `delaunay_triangulation` | 3.73 | 1 | **attributed** — genuinely host (no capture), and it is the *documented* CPU seed: 5.2 ms in `invoke` plus 2.8 in `_lexicographic_triangulation`, 8.0 of 18.1 ms at n = 20 000 |

**The finding: the band is mostly closed or floor, and the six largest genuinely-open rows were
addressed by other items rather than by working the band.** That is the argument for attributing
first and expecting the row to belong to someone else's item. One caveat on reading any of it: the
band is computed from **medians** — run `aggregate.py --suspect` first (§15.4).

**The four rows that had no attribution now have one, and three of the four are *device*-bound —
the opposite of what this section's method reported**, because all three graph-capture and
`wp.timing_begin` cannot see that (§15.10). So the band's remaining 21.1 ms is not launch overhead
and launch elimination will not touch it:

- **`transport_tangent_vectors` / `vector_heat_scale`** are one call. 71 % device; of the solve's
  4.17 ms the two `TiledDot` kernels are 1.75 ms (42 %) at 186 calls each, the sparse mat-vec only
  0.31. Same shape as §12.7's batched-dot finding, on the *unbatched* path — `wpl.cg`, not
  `_BatchedCg`. A CG iteration here is ~36 µs of which two dots are ~10, and every kernel in it is
  latency-bound at ~5 µs over 40 962 rows.
- **`homology_generators`** = `tree_cotree` (6.09 ms) + ~5.0 ms of Python loop tracing. Of the
  `tree_cotree` half, **`graph.bfs` is 3.57 ms and is ~100 % device** — 4 kernels x 142 levels at
  26.1 µs a level (`bfs_scatter_claims` 8.6, `bfs_expand_claim` 7.6, `bfs_count_and_scan` 7.0,
  `bfs_scan_and_advance` 2.9, the last at `dim=1`). Capture is already worth **3.7-5.0x** there
  (3.53 ms against 13.3 uncaptured), and `wp.capture_while`'s own per-iteration overhead is only
  ~4-6 µs against a captured fixed chain's 1.0-2.1 (measured on a 142-iteration synthetic; batching
  K bodies per conditional check is worth at most 1.38x there, best at K = 4, and nothing here).
  **So the level loop is real work and the only lever is fewer levels or fewer kernels per level**,
  which is §14.9's standing conclusion. The primal tree cannot take the dual side's Boruvka
  treatment — `tree_cotree` reads the dual tree as a *set* but walks the primal one's rooted
  `parents`, and rooting a forest in parallel is a different problem; `homology.py` says so at the
  site. Second-largest piece after that is `_device.read_scalar`, 11 calls for 0.154 ms.
- **`delaunay_triangulation`** is the one that is genuinely host, and its host half is the
  *designed* one: the single-thread CPU seed kernel, 5.2 ms in `invoke` plus 2.8 in
  `_lexicographic_triangulation` of an 18.1 ms call at n = 20 000. Already recorded in the
  benchmark's own docstring as the fixed design (a CUDA thread is 66x worse at it).

Two related notes: `homology_generators[handles_64]` spends 4.43 of 10.9 ms in the Python
`_loop_through_tree` walks (128 generators, mean loop 110), which a NumPy binary-lifting LCA would
vectorize; and `combine.split`'s cost scales with **component count**, not mesh size — ~0.64 ms of
host work per returned submesh against 0.29 ms (open3d) and 0.23 ms (trimesh), so on a mesh with 94
scan floaters triwarp is 78 ms where it wins 34-120x on few-component meshes. The per-component
allocation/launch sequence needs batching; the labelling is fine.

### 16.10 Where triwarp wins big (for context when reading a loss)

`screened_poisson` 15-25x over open3d (134 ms vs 2.0-2.3 s), `combine.split` on few-component meshes
34-120x, `winding_number` 13-18x over igl, `procrustes` ~7x over trimesh (but only even with open3d —
it is launch-latency bound, not compute bound), `cluster_decimate` 9.14x at `dragon`,
`polyline_simplify`'s level-synchronous RDP 84 → 0.68 ms at `rim_long`. Small-input (~16k) fixed
overhead of device reductions and scan pipelines costs ~40-120 µs against the host path — accepted,
because large meshes win 10-260x (`n_vertices` on `lucy` 259x).

**A third independent implementation is what makes an outlier legible**; trimesh alone is slow enough
everywhere that a 3x triwarp regression still looks like a win against it. That is the argument for
carrying nine references.
