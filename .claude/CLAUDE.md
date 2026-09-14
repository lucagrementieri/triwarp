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
   moving/renaming, the 25 mechanical checks
5. [Function ordering within a module](#5-function-ordering-within-a-module)
6. [Documentation](#6-documentation-zensical--mkdocstrings)
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
14. [Kernel-shape verdicts](#14-kernel-shape-verdicts) — what wins, what is refuted, and when a
    producer-consumer fusion pays
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
- **`wp.constant()` is not what makes a module-level value visible in kernel scope, and wrapping
  with it is optional.** On Warp 1.17 it is `return x` after an `is_value(x)` check (§12.6) — any
  module-level global that evaluates to a scalar/vector/matrix is resolvable from kernel scope
  whether or not it passed through `wp.constant()`. What *is* load-bearing is the typed
  constructor:
  ```python
  TOLERANCE_MERGE = wp.float32(1e-8)
  ```
  A plain Python float works too but is treated as `wp.float32` in kernel-scope arithmetic, and
  mixing that default against a `float64` kernel variable is a parse error — use the
  `wp.float64(...)` constructor where 64-bit precision is required. `triwarp/constants.py` carries
  no `wp.constant()` calls for this reason; do not add one to a new constant on the theory that it
  is required for visibility.
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

**And price the fused region itself, never the call that contains it — §14.10 is the verdict
table.** Every such pair measured in this tree is faster fused (1.04-3.46x); what the iteration
count changes is only whether a benchmark can resolve the win. Attributing through the enclosing
call declined a real `heat` win on noise once already.

**The converse of rule 1: a helper with one caller is a name, not an abstraction — inline it, then
look again.** Extracting the shared run is right when the run is *shared*; when a fusion absorbs its
own producer, the "helper" it leaves behind has a single call site and its only effect is to hide
the redundancy between the two halves. Inlining three such kernels exposed a doubled vertex load, a
doubled grid index and a `dbl_area` computed twice by two spellings of the same expression —
together worth more than the launch the fusion removed (§14.10).

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
  a dense buffer first. **Both halves of that sentence are load-bearing, and the tree got the first
  one backwards once**: `remesh._keep_longest_edges` wrote `gather(eligible, wp.clone(order[:k]))`
  with a comment citing this rule, where `order[:k]` is a prefix and needs no clone at all —
  measured 47.7 against 31.9 µs for the gather (**1.50x**), byte-identical. When a clone-of-slice
  cites this rule, check which kind of slice it is; `adjacency.face_adjacency` and
  `reconstruction._seed_candidates` clone a `[:, 0]` column and are the case it exists for. Measurements: §12.1. This is why `kernels/edges.py:edge_lengths` stays a
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

**The census has been taken, and it is the reason not to re-open this.** An AST scan of
`triwarp/` (excluding `kernels/`, which has none) finds **488 host-side sites**: 329 NumPy calls,
67 `.numpy()` readbacks, 55 `math.*` and 32 `.tolist()` / `.item()`. Bucketed: 126 metadata and
marshalling, 81 fixed-size 3x3/4x4/scalar math, 67 readbacks, 55 scalar math, 36 host-sequential,
32 host converts, 15 module-scope constant tables evaluated once at import, and ~76 lattice and
template index arithmetic. **Only the last bucket contained anything convertible** (§16.4's
`parametric_surface`), and the rest are settled by one measurement rather than site by site:

| host op | µs | | device floor | µs |
|---|---|---|---|---|
| `np.eye(4)` | 0.76 | | `wp.empty(1)` | 6.1 |
| 4x4 matmul | 0.72 | | **one `wp.launch(dim=1)`** | **21.5** |
| 3x3 determinant | 1.65 | | **one host readback** | **21.2** |
| 3x3 SVD | 6.14 | | | |

A minimum device round trip is ~43 µs against 0.7-6 µs of host arithmetic: a **26-60x loss**, and
structural rather than an implementation detail. That disposes of 262 of the 488 sites (54 %) at a
stroke, and the module-scope tables cost nothing per call.

**The crossover, per operation class** (RTX 5090, Warp 1.17, result staying *on the device*; ratio
numpy/device, >1 = the device wins). Use it to price a conversion before writing one:

| operation | 1e3 | 1e4 | 1e5 | 1e6 | crossover |
|---|---|---|---|---|---|
| `cumsum` / scan | 0.38x | **2.6x** | 26x | 251x | ~5 k |
| `unique` | 0.18x | **2.8x** | 36x | 542x | ~5 k |
| gather | 0.08x | 0.67x | **6.3x** | 58x | ~30 k |
| sort | 0.03x | 0.27x | **3.0x** | 31x | ~50 k |
| sum -> scalar | 0.07x | 0.16x | 0.96x | **10.3x** | ~100 k |
| elementwise | 0.02x | 0.09x | 0.48x | **14.5x** | ~200 k |
| flatnonzero | 0.01x | 0.06x | 0.45x | **4.9x** | ~200 k |
| min/max -> scalars | 0.02x | 0.03x | 0.12x | **2.9x** | ~500 k |

And the *readback* question separately — mask already on the device, caller needs a host bool:
`mask.numpy().any()` against `reduce.any` measures 0.71x at 100 k, **1.00x at 500 k**, 3.46x at
1 M, 32x at 16 M, which reproduces §12.4's ~0.5 M crossover exactly. `np.any` short-circuits, so
its cost depends on the data and not only on `n` — an all-False mask is the worst case and the one
a convergence loop actually hits; timing a half-True mask makes numpy look 100x better than it is.

**Measured NumPy share of real public calls**, at 320 / 5 120 / 81 920 faces: nine of eleven probed
sit at **0.6-6 %** with the share flat or falling, which §9 calls a decline. The two above 10 % were
`creation.uv_sphere` / `capsule` (35-67 %, which is `_revolve_kept_template`, the filter that
licenses the closed-form fast path — 33-42 µs for profiles up to 64 points against a ~43 µs device
floor, so a wash below ~256 profile points and declined) and `bounds.oriented_bounding_box`
(13-18 %, where the NumPy was **not** the cost — §16.4).

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
- **A `.tolist()` immediately splatted into a Warp vector/matrix constructor is noise — delete it.**
  `wp.vec3(*x.tolist())`, `wp.mat33(*x.ravel().tolist())`, `wp.mat44(*x.flatten().tolist())`,
  `wp.mat33d(*x.ravel().tolist())` all construct identically from the raw NumPy array with the
  `.tolist()` dropped — verified on Warp 1.17 for `float32` and `float64`, for a contiguous row and
  for a non-contiguous (transposed / column / negated) view. `.tolist()` changes neither the
  dtype-narrowing a `wp.mat33`/`wp.vec3` float32 constructor does (a Python `float` and a
  `np.float64` round to the same bits) nor anything else observable; it only allocates a throwaway
  Python list the constructor immediately consumes and discards. Swept and removed across the
  tree — **19 sites in 5 files** (`bounds.py`, `transform.py`, `measures.py`, `reduce.py`,
  `creation.py`), two of which were the identical redundancy one level down —
  `math.dist(a.tolist(), b.tolist())`, where `math.dist` also takes a raw NumPy array directly.
  Two shapes are **not** this defect and must stay: a `.tolist()` whose result **is** the return
  value, satisfying a genuine `list[...]`-typed public signature rather than feeding a Warp
  constructor (`intersection._link_segments`'s `closed.tolist()` — the function's declared return
  is `list[bool]`, so this is what keeps a public return from naming `np.ndarray`, the very thing
  this section's first bullet forbids); and a `.tolist()` used to get plain Python scalars for
  non-Warp bookkeeping such as a dict key (`creation._icosphere_face_table`'s
  `for f, (a, b, c) in enumerate(faces_np.tolist())`). §7.1's test-writing convention
  (`wp.vec3(*array_np.tolist())`) is unaffected by this finding and stays the sanctioned spelling
  in `tests/` for consistency across the suite, where the redundant allocation is immaterial — do
  not carry it into `triwarp/` production code as though the `.tolist()` there were load-bearing.
- **`arr.numpy().tolist()` is `arr.list()`, but only for a rank-1 array — verified on Warp 1.17.**
  `wp.array.list()`'s scalar-dtype branch is literally `self.numpy().flatten().tolist()`, so for an
  already-1D array the two spellings return the identical Python list (checked byte-for-byte across
  `int32`, `uint64`, `bool` and `float32`) at identical cost — `.list()` calls `.numpy()` internally,
  so there is no speed win, only one fewer visible readback call. **It is not a substitute for a
  rank-2 (or higher) array: `.list()` unconditionally flattens**, where `.numpy().tolist()` preserves
  the row structure. `boundary_edges`'s `(n, 2)` output as `.list()` returns one flat `2n`-element
  list rather than `n` pairs, which silently breaks anything iterating rows
  (`for edge in edges.numpy().tolist()`) or indexing into it (`cells(grid).numpy().tolist().index([0,
  0, 0])`) — both patterns exist in the tree and were left alone for exactly this reason (see
  `tests/test_boundary.py`, `tests/test_seams.py`, `tests/test_voxels.py`,
  `tests/test_geodesic_walk.py::test_shorten_loop_*`). Restrict the swap to a genuinely `ndim == 1`
  buffer — an offsets/index/mask array, a `list[wp.array]` element from `boundary_loops`, or an
  already-indexed row of a 2D array — which is what `array.py`'s `split` already asserts before this
  exact pattern (`if int(offsets.ndim) != 1: raise ValueError(...)`), and is the cheap tell to check
  before converting a site.
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

**Every public function that accepts two or more device-bearing arguments calls
`_device.require_same_device(**named)` as its first statement**, passing every array, mesh, BVH,
hash grid, `wp.Volume` or list/tuple of those it received — including an `X | None = None`
precomputed-cache argument; the helper skips `None` silently, so nothing is filtered beforehand.
It raises `RuntimeError` (deliberately not `ValueError` — see the reasoning below) naming the two
mismatched arguments and their devices. Document it with a `Raises` entry the same way a direct
`raise` is documented (§4.3) — check 11 cannot see through the delegation, but a caller reading the
docstring should not have to know that.

This reverses what an earlier revision of this file said (*"do not check that input arrays share
the same device"*), and the reversal is deliberate, not a relaxation of the reasoning that produced
the original rule. That rule was correct for triwarp's own internal call sites: every kernel
factory forwards `device=` from an input array (§2.1), the test harness runs under
`LaunchArrayAccessMode.STRICT`, and an internal wrapper calling another triwarp wrapper already
passes arrays it just validated or produced itself, so a repeated check there really was redundant.
None of that holds for an external caller of the *public* API, who has no reason to know Warp has a
device model at all and who can trivially construct a mismatch by accident — one mesh loaded from
disk (landing on CPU) and one built with a GPU default. For that caller, the two ways a mismatch
actually fails (below) are not a `ValueError` away; they are silent memory corruption or a bare
segfault with no Python traceback. A cheap comparison of a handful of `.device` attributes, once,
at the public boundary, converts an undebuggable native failure into an ordinary Python exception,
for a cost (a dict of attribute reads) that is unmeasurable next to a single `wp.launch` (§13.1).
**The rule is still "do not scatter ad hoc checks through internal helpers"** — it is now "the
public boundary checks once, through one shared helper, and everything behind it stays trusting."

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

Five consequences:

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
- **`require_same_device` raises `RuntimeError`, matching the exception type PyTorch raises for the
  same class of mistake, deliberately with different, triwarp-specific wording** — PyTorch's own
  message names tensors and an internal kernel-dispatch frame (`"Expected all tensors to be on the
  same device, but found at least two devices, cuda:0 and cpu!"`); triwarp's names the caller's own
  keyword-argument names and both devices, and suggests the fix (`"'a' is on cpu while 'c' is on
  cuda:0. Move one onto the other's device ... before calling this function."`). `RuntimeError`
  rather than `ValueError` because this is not a bad *value* in the NumPy sense (the arrays are each
  perfectly valid on their own device) — it names the same failure class PyTorch, a library with the
  same multi-device model, already reserves `RuntimeError` for.

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
- **That is a *single*-value rule, and it inverts at two.** A readback's cost is almost all fixed,
  so **one `.numpy()` of a small buffer beats two `read_scalar` calls** — measured 20.4 µs for a
  2-element `.numpy()` against 29.7 for two tail reads. So a function returning several small
  device values should write them into *one* buffer and read it once, which is what
  `points.fit_plane` / `principal_axes`, `measures.moments` and `registration.icp_point_to_plane`'s
  scalar accumulator all do (1.27-3.00x, §16.4). The rule only reverses once the buffer is large
  enough for the copy to matter: `.numpy()` of 61 440 `int32` is 36.6 µs against `read_scalar`'s
  14.7, so a *tail* read of a mesh-sized array stays `read_scalar`.

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
- **A `Literal`-typed *menu* argument is validated at the public boundary and raises `ValueError`
  naming the argument, the offending value and the options. The annotation is a hint, not a
  guard.** basedpyright rejects an off-menu *literal*; nothing rejects the same value arriving
  through a variable, a config dict or a `**kwargs` splat, so a menu read as `A if x == 1 else B`
  silently answers a different question. This is not a new convention — it is the one the package
  already had, measured by calling every one of the **43** `Literal`-annotated parameters in
  `triwarp/*.py` with an off-menu value: **38 already raised** exactly that `ValueError`, and the
  three that did not are now converted (`texture.remap_attribute_from_uv(order=2)` sampled
  nearest-neighbour and returned a plausible image; `registration.icp_point_to_plane` and both
  `seams.uv_seam_*` indexed a mode dict and surfaced a bare `KeyError('bogus')`, naming neither
  the argument nor the alternatives). Four details worth having:
    - **Two `Literal` shapes are not menus and need no guard**: the `Literal[True]` /
      `Literal[False]` pairs on a `return_*` keyword, which are `@overload` stubs over a `bool`
      where every value is legal, and a rank literal (`wp.array[DType, Literal[3]]`).
    - **Where the options live in a table, derive the message from it** —
      `f"match must be one of {list(_UV_MATCH_MODES)}, got {match!r}"` — so it cannot drift from
      the table. `list(...)` for an *ordered* table (a dict or tuple, so the message reads in the
      docstring's order) and `sorted(...)` for a `frozenset`, which is what `visibility.thickness`
      does. Where the branch is an `if` / `elif` chain, spell the names out, as most of the 38 do.
    - **A delegating wrapper does not repeat the check**; it documents the `ValueError` and lets
      the function it delegates to raise (§4.3, the same rule as `require_same_device`). **20 of
      the 43 are that shape** — a private validator (`holes._check_refine`,
      `reduce._validate_scalar_array`, `voxels._check_iterations`) or another public function
      (`levelset.offset_mesh` → `signed_distance_grid` → `signed_distance_on_mesh`) — against 23
      that check in their own body. But the *test* then has to call through the wrapper: that is
      the only thing that shows the guard is still reached.
    - **No static check guards this, deliberately.** A permissive scan (does the body mention the
      parameter?) passes the `order == 1` defect that motivated the rule, and a strict one
      (does the body contain an `In` / `NotIn` test on it?) flags every delegating wrapper —
      **20 of 43**, nearly half the sites as allowlist, which §4.5 says is how a check gets
      switched off. The gate is instead a **probe**: call each menu entry point with an off-menu
      value and read the exception type. It is ~60 lines, it needs one valid call per entry
      point, and it is the only thing that distinguishes "raises" from "raises for the wrong
      reason".
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
  needs a `Raises` block (check 11; the functions that delegate validation to a shared guard and
  document its `Raises` are correct and are not scanned — the largest such family is every public
  function with two or more device-bearing arguments, which documents
  `_device.require_same_device`'s `RuntimeError` per §3.9).
- A documented validation must actually be performed, or the claim goes.
- Annotations must cover every rank and dtype the docstring claims and the body supports.
- **When a comment and the body disagree, decide which one is load-bearing before "fixing" it — the
  usual answer is the comment.** `tangent_space.any_perpendicular`'s comment claims it crosses with
  *"whichever coordinate axis the normal is least aligned with"* while the body compares only
  `|n[0]|` against `|n[1]|`. The body is **correct for its purpose** (it needs any axis not parallel
  to the normal, and x or y always qualifies), and "correcting" it into a three-way argmin would move
  the tangent frame at every z-dominant normal. Fix the sentence, leave the branch, and say in the
  commit which of the two you changed and why.
- **When a private helper's docstring states a rule the public surface must obey, the defect is in
  the paths that *bypass* the helper — and the `@overload` stubs and the `Returns` block are two
  further statements of that rule that no test reads.** `neighbors._shape_nearest` exists to
  *"collapse the `(m, k)` result to the rank the caller's `queries` / `k` imply"* and collapses to
  rank-1 at `k == 1`; `query_nearest`'s two degenerate early returns were taken **before** it and
  came back rank-2, so `indices[i]` was a scalar on an ordinary cloud and a length-1 row on an
  empty one. The `k: Literal[1]` overload and the `Returns` prose independently declared rank-2 as
  well — one intent, four statements, one of them right. This is §2.4's duplicated *decision rule*
  one level up, and the fix is the same shape: route the bypassing paths back through the helper
  (`_empty_nearest` now returns `_shape_nearest(...)`) rather than repeating the collapse at each
  site. **After changing a rank/shape/dtype rule, grep for the early returns and the `@overload`
  stubs, not just the main path** — basedpyright cannot see the mismatch, because a wrong overload
  return type is self-consistent.
- **A test that hardcodes one value of the parameter the contract turns on is testing the one case
  that cannot fail.** `test_query_nearest_empty` pinned `k = 2` — the only `k` where the collapse
  above is a no-op — so the whole suite was green against it. §7.4's vacuity rule usually reads as
  "check the fixture is not degenerate"; this is the same rule about a *parameter*, and the cheap
  version is to parametrize over the boundary value rather than a comfortable one.

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
   cross-reference — `zensical build --strict` is what finds the ones you missed.

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

**Twenty-five checks**, and they fail the default `pytest` run.

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
  a name that moves into a `_*.py` breaks `zensical build --strict` (a private module generates no
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
- **Check 21**: a MeshLib **or promesh** name anywhere under `triwarp/` — a *licensing* guard
  (§7.6). Two libraries, one scan, opposite reasons: MeshLib's licence is readable and
  restricts *use*; promesh's mirror carries no licence file at all, so a "port of" comment
  cites terms nobody here has checked.
- **Check 23**: a kernel module whose `@wp.func` is `wp.map`'d from several sites with no
  declaration table (§3.5). Like its `wp.overload` sibling it asserts a table *exists* and never that
  it is complete; the completeness gate is the load census, which is a clock measurement (§15.1).
- **Check 24**: a `.claude/CLAUDE.md` cross-reference naming a section that does not exist, **or a
  bare chapter number where that chapter is subdivided**. The second clause is the one that earns
  the check, and it is a *staleness* rule rather than a convention one: 111 references across 46
  files were citing Part I's old numbering, and every one of them still resolved. `section 4` stood
  simultaneously for §3.5 (`wp.map` targets), §2.5 (overload registration), §2.7 (`wp.launch`
  cannot pass a `wp.Function`) and §3.7 (`triplet_buffers`' uninitialized tail); `section 6` for
  four different subsections of §7. One number meaning several sections is exactly what resolution
  alone cannot see. Chapters 5, 6, 8, 9, 10 and 11 carry no `###` heading, so a bare number is
  their only citation and is accepted — the check reads that from the file's own headings rather
  than from a list, which is what keeps `_CLAUDE_CHAPTER_ALLOWLIST` empty (rejecting *every* bare
  chapter reports 68 sites of which 38 are correct citations — more than half the hits as
  allowlist, which is how a check gets switched off). It scans `triwarp/`, `tests/` and
  `benchmarks/`, matches `AGENTS.md` too (a symlink to this file, cited that way at three sites),
  and abstains when `.claude/CLAUDE.md` is absent. **Two things it cannot see**, both fixed by hand
  in the pass that added it: a reference naming the file in one sentence and the number in the
  next, and a bare `section N` belonging to a *paper* — `kernels/remesh.py` cites "Liepa 2003,
  section 3", and widening the pattern to catch the first misfires on the second.
- **Check 25**: a `!!!` admonition inside a numpydoc **item-list** section — `Parameters`,
  `Returns`, `Yields`, `Receives`, `Raises`, `Warns`, `Attributes` or `See Also` (§6). griffe reads
  each entry's first line as a *name*, so an `!!! note "..."` header between two `Raises` entries
  becomes an exception type: confirmed by loading such a function and getting **three** raises
  entries, the middle one carrying the literal string where an annotation belongs. The published
  page grows a row for a type that does not exist, the admonition's body becomes that row's
  description, and the warning never renders as a warning. Nothing else sees it — `zensical build
  --strict` stays clean, because every cross-reference in the swallowed text still resolves, and
  ruff's `D` rules do not model section contents. It had decayed to **eight** sites in five modules
  (`holes`, `proximity`, `sample`, `smoothing`, `voxels`), seven of them in a `Raises` block, and
  the clustering is the lesson: an author writes the caveat where the thought occurs, and the
  thought occurs while documenting what the function rejects. **It ships with no allowlist and no
  allowlist machinery**, because unlike check 24 there is no legitimate instance — every admonition
  has a correct home in `Notes` or in the leading description, with no loss of meaning and no
  reordering of anything a caller reads first.

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

## 6. Documentation (Zensical + mkdocstrings)

Docs are built with **Zensical** + **mkdocstrings** (`python` handler, `docstring_style: numpy`).
Zensical is the Material for MkDocs team's ground-up replacement for the MkDocs stack, adopted
because MkDocs has been unmaintained since 2024-08; it reads **`mkdocs.yml`** unchanged, so the
config filename, `theme: name: material`, the palette and `extra.css` are all as they were.

**Two commands, and the first is not optional:**

```bash
uv run python docs/gen_ref_pages.py     # ALWAYS first -- materializes docs/api/ + docs/SUMMARY.md
uv run zensical build --strict          # validate: exits 1 on a broken cross-reference
uv run zensical serve                   # preview locally
```

`docs/gen_ref_pages.py` generates one API reference page per public module under `triwarp/`
(`triwarp/kernels/` is excluded), so a new module needs **no manual nav entry** — but it is a
**standalone pre-build script, not a plugin hook**, because Zensical has no `gen-files` equivalent
(`zensical/zensical#51`, still open). Five measured facts about that runtime, each of which fails
quietly:

- **Zensical ignores an unsupported plugin entry silently — no warning, no error.** A build with
  `gen-files` still listed in `plugins:` exits 0 and publishes a site with **no API reference at
  all**. `--strict` is the only thing that reports it, as one unresolved cross-reference per API
  symbol (measured: 169). Never register `gen-files`; always run the script first.
- **`--strict` is the cross-reference backstop** §4.4's five-artifact discipline leans on, and it
  works: it exits 1 on a dangling `[`name`][triwarp.old.path]`. Verified that the four external
  inventories still resolve under it too.
- **`zensical serve` watches `docs/` but not `triwarp/*.py`.** A docstring edit does not trigger a
  rebuild, and forcing one by touching a `docs/` file does not help either — mkdocstrings has the
  module cached in-process. **Restart `serve` to see a docstring change.**
- **There is no `exclude_docs:` equivalent.** The `assets/benchmarks/*.md` sidecar tables are kept
  out of the search index by `search: exclude: true` front matter instead, emitted by
  `benchmarks/plot_comparison.py`. (`draft: true` was probed and is a no-op.) They are still
  *built*, as unlinked pages nothing references.
- **`literate-nav` and `section-index` are implemented natively**, so neither package is installed
  — measured, the built site is byte-identical without them — and their `plugins:` entries are read
  as configuration rather than as a request to load a plugin. `mkdocs-gen-files` *is* installed,
  for its `Nav` helper alone.

`DISABLE_MKDOCS_2_WARNING` is gone: it silenced a banner recommending this exact migration.

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

**And no development-history narrative belongs in `triwarp/*.py` either — in a docstring or in a
code comment.** A sentence that reads as a lab notebook rather than documentation ("measured X,
declined Y", "reverted", "one session", "round N", "probe"/"sweep" used as a methodology rather than
an algorithm term, a cross-reference to an internal doc section) describes how the code came to be,
not what it does or how to use it. The public wrapper layer — every module under `triwarp/`,
excluding `kernels/` — is documentation for a caller, not an engineering log; keep the behavioral or
correctness fact a piece of history was attached to (a convention, a sign rule, an aliasing warning,
what raises and when), and cut the narrative around it.

**Delete it, don't relocate it.** This reverses the file's own earlier guidance, which said to move
a pruned number into a `#` comment in the same function's body — that is no longer the rule for
`triwarp/*.py`. §9's "a measured decline is a result, written at the site" still holds, but the site
for that discipline is now `kernels/`, `benchmarks/`, and `tests/`, not the public wrapper: a
decline's number and reasoning belong in the private helper that actually pays it, or in Part II
here, not in a comment a caller has to scroll past. Numbers and decline-history stay welcome in
private helpers' docstrings, in `kernels/` (nothing there renders), in `benchmarks/` group
docstrings, and in Part II.

A full pass across all 52 public modules (2026-09-04) removed every instance the scan below found,
plus the matching code-comment narrative in the same files, and fixed the two stale allowlist
entries in `tests/api_conventions.py`'s check 9 that removing two "Warp 1.16" history notes left
dangling. This is now a zero-tolerance completeness rule rather than a tracked backlog: a fresh hit
in a new or edited public function is a regression to fix in the commit that introduced it, not an
item for a future pass.

The scan is an `ast` walk over `triwarp/`'s public functions matching
`\d[\d.,]*\s*(ms|us|µs|ns|GB|MB|kB)\b` or `\b\d+(\.\d+)?x\b` against each docstring — a plain
grep for `ms` is unusable, and the same regex over *prose* words (`measured`, `faster`,
`benchmark`) returns 35 kB of legitimate behavioural text, so key on the *quantity*. It has no
comment-scanning counterpart yet; a code-comment narrative slipping back in is caught by review,
not by this scan.

Docstrings stay **NumPy-style** (`Parameters`/`Returns`/`Raises`/`See Also`), but cross-references use
**mkdocs-autorefs** link syntax, not Sphinx roles — Sphinx interpreted-text roles (`:func:`, `:attr:`,
`:meth:`, `:class:`, `:data:`, `:mod:`) have no Markdown equivalent and render as literal, broken text
in Markdown.

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
- RST admonitions (`.. note::`) don't exist in Markdown — use the `!!! note` admonition syntax.
- **An admonition goes in free prose — the leading description, `Notes` or `Examples` — never
  inside `Parameters` / `Returns` / `Raises` / `See Also` or any other item-list section**
  (check 25, §4.5). griffe reads each entry's first line as a name, so an admonition header
  between two `Raises` entries renders as an *exception type* and its body as that type's
  description. `zensical build --strict` cannot see it, and it had reached eight sites.
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

There are **nine** libraries here and **ten** subsections: the last one, `promesh`, is not one of
the nine and is not a reference at all — it is a design mirror that nothing installs, and it is
filed here only because that is where a reader looks for it.

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

#### promesh — not a reference library, and not citable from `triwarp/`

`reference/promesh/` is a bare source drop, and it is the one mirror here whose terms **cannot be
read from this repo at all**: no `LICENSE`, no `COPYING`, no `pyproject.toml`. It is also not a
published package — nothing installs it, `import promesh` fails, and it therefore can never be a
tested or benchmarked reference, so none of §7.6's machinery applies to it. Treat it as a design
mirror only, on the same footing as reading MeshLib for an *interface*.

**Nothing under `triwarp/` may name it**, for a reason that is the mirror image of MeshLib's rather
than the same one: MeshLib's licence is readable and restricts *use*, where promesh's is simply
unknown — so a "port of" comment cites terms nobody here has checked. Five such references had
accumulated in `holes.py`, two claiming a port outright (*"the Warp port of promesh's
`triangulate_boundaries`"*, *"Mirrors promesh's private helper"*). All five were rewritten into the
algorithm's own vocabulary — the gap-bridging problem of **Barequet & Sharir (1995)**, solved by the
minimal-perimeter heuristic with a longest-increasing-subsequence monotonicity correction — which is
what a reader wanted in the first place. **Check 21 covers both libraries in one scan** (keying
additionally on `promesh` and `triangulate_boundaries`) because the fix is identical either way,
and its own test pins the *replacement* wording as a negative so a later pass cannot widen the
pattern back onto the literature's vocabulary.

One mitigating fact worth knowing before re-reading that tree: `reference/promesh/deformation.py`
attributes its own mollification helper to `kentechx/HoleFillingPy` (**MIT**), so parts of it are
themselves ports of permissively licensed code — which is why the corroborating citation in
`kernels/laplacian.triangle_inequality_slack` names that upstream and not promesh.

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

### Version control

**Commit directly on `main` unless the user asks for a branch.** This is a single-developer
repository: there is no review queue for a feature branch to sit in and no second working copy to
integrate with, so a branch-and-merge cycle buys nothing and costs a merge. The default overrides
any general "branch before committing" habit — branch only when the user names one, or when a change
is genuinely speculative and expected to be thrown away.

Two things this does **not** change. Committing is still an explicit request: finish the work, run
the gates (§8, §12.10), and commit when asked, not on your own initiative. And the measurement
discipline that depends on a clean tree is unaffected — an A/B against a prior revision uses a
**detached worktree**, never `git stash` and never a branch checkout in the live tree (§15.6).

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
- **A mechanism validated on one member of a fixture pair built to isolate a variable must be
  measured on the *other* member before it ships.** `saddle`/`saddle_graded`, `sphere`/`tangle`,
  closed/open, uniform/graded — the pair exists because that variable matters, and a benchmark
  suite that ships one is telling you which variable a new mechanism has to survive. This is §7.4's
  vacuity rule ("check the comparison is not vacuous on its fixture") applied to a *benchmark*
  rather than a test, and it has cost a measured 2.59x regression once: a block conjugate gradient
  was probed on two well-conditioned saddles, won on both, shipped, and was a **0.386x** loss on the
  graded saddle of identical connectivity — which is precisely where its own documented failure mode
  lives (§16.8). The check is cheap: it is one more row in the probe you are already running.
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

**Do not run a timing probe while `pytest` or `zensical` is running** — the same reconstruction read
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

- **A `wp.launch` with `device=` omitted corrupts the host heap, not a style nit.** It silently
  resolves to the default device (`cuda:0` whenever CUDA is present) while the array arguments sit
  on the CPU, and returns numerically correct results (HMM lets the GPU legitimately dereference
  host memory) while corrupting the heap, because the launch is asynchronous and the host arrays get
  freed while the kernel is still reading them. **A `wp.synchronize()` after the launch is the fix**
  — the asymmetry is that a CUDA array's storage frees with the stream-ordered `cudaFreeAsync` while
  a CPU array's frees immediately and unordered, so the CPU side is the one that actually corrupts.
  Always forward `device=` from the input arrays (§2.1) so this configuration is never reached.
- **An out-of-bounds kernel write on the CPU device *is* glibc heap corruption**, because a CPU Warp
  array is host heap — deferring a range check to a downstream call is enough to trigger it.
  Range-check at the write, not later; `connected_component_parity_from_edges` still doesn't.
- **General lesson: neither of the above is "the Warp CPU backend corrupts the heap"**, the framing
  that survived months of subprocess isolation before these were found — both were application bugs.
  Two bisection techniques generalize past this case: `wp.config.mode = "debug"` compiles kernel-side
  bounds checks, but **a clean debug run is evidence about timing, not correctness** (enough work
  after a bad launch can let the kernel finish before teardown even with the bug present, giving a
  false-clean debug run); and a **component-swap bisection** — run a clean pipeline and substitute
  one library component at a time — finds a launch-ordering bug that line-level bisection can't.
- **A `wp.Mesh` with zero triangles silently corrupts CUDA allocator state.** The constructor
  succeeds, but the *next* unrelated CUDA allocation anywhere later in the process fails with a
  spurious OOM and cascades into `CUDA error 700`. Only `indices.shape[0] == 0` matters, not point
  count. Safe on `cpu`; still broken on CUDA. Never construct one on CUDA, **including in tests** —
  use a single-triangle mesh to reach an `n_faces < 2` guard. `triwarp.mesh.Trimesh.warp_mesh` would
  hit this for a zero-face mesh — a known latent issue, not fixed.
- **Python-scope gather silently ignores a non-contiguous index view's stride.** `payload[edges[:,
  0]]` returns the flattened buffer's leading entries instead of column 0, no exception, and a
  `wp.map` over the same view inherits the corruption while reading *faster* — so a benchmark alone
  reads as a win. `wp.copy` the index into a dense buffer first. Rule: §3.4.
- **`wp.copy(dst, src, count=0)` copies the *whole* source, and `wp.utils.array_cast` inherits it** —
  Warp's own back-compat rule reads `count == 0` as "not passed" rather than "nothing". A `count`
  derived from data (a compacted length, a readback) must be checked for zero at the call site and
  the zero case answered without calling `wp.copy` at all; same shape as §3.3's zero-length-slice
  raise and §3.7's capacity-versus-count rule — a legitimate empty case Warp spells as a special
  value rather than a length.
- **`wp.copy` into a *pinned* host buffer is a genuine async memcpy with no event, so reading it
  right after races and returns the *previous* value** — CUDA only blocks the host on a *pageable*
  destination. `triwarp._device.read_scalar` is the safe, device-split spelling; pinning buys
  nothing once done correctly. **It caches one scratch buffer per dtype, which is only safe when
  everything returned is a scalar** — for a vector/matrix dtype `.numpy()[0]` is a *view* onto the
  shared scratch, so two sequential reads alias and the first takes the second's value; the fix is
  to copy before returning. General lesson: a per-dtype cached buffer degrades silently the day
  someone passes a non-scalar dtype through it.
- **Warp's CPU work runs ~36x slower in a process where CUDA has been initialised** — it is CUDA's
  mere *presence* in the process, not the launch-access guard or GPU contention, and
  `CUDA_VISIBLE_DEVICES` must be set before the process starts. This is why both-device coverage is
  a two-process job (§7.2).

### 12.2 `wp.launch_tiled` runs one lane per block on the CPU

Still true on **Warp 1.17**. `wp.launch_tiled(kernel, dim=[...], block_dim=64)` executes **one thread
per block** on the CPU backend — `wp.tid()`'s lane index is always 0.

**The obvious probe says "fixed", and that is the trap.** `wp.tile_load` reads its whole tile out of
an array, is lane-independent, and was never affected. Only `wp.tile(x)`, built from *per-lane*
values, collapses on CPU. **Always probe the lane-constructed tile**, or you will conclude it is
fixed and delete a correctness branch. Two silent consequences, no exception either way: a block
reduction over a lane-constructed tile returns only the block leader's contribution, and even a
kernel with no tile intrinsics breaks if it relies on lanes covering the tile
(`idx = tile_i * TILE_1D + t` touches only every 64th element on CPU). Because
`tests/conftest.py`'s `device` fixture returns `cuda:0` whenever CUDA is available, this class of
bug is invisible to the default test run and needs an explicit `"cpu"` parametrize to catch (§7.2).

**The fix is one token: stride by `wp.block_dim()`** — it reads the launch's `block_dim` on CUDA and
**1** on CPU, costs nothing on CUDA, and makes the single CPU lane cover every element, after which a
tile reduction degenerates to a one-element tile that correctly returns that lane's own answer.
**It does not generalize to *partitioned* kernels** — a kernel computing `f = i * TILE_1D + t` at
`dim = n_blocks` has one lane per block dropping most of the range, and no stride change reaches it;
that needs the kernel to loop over its block's range, which is why some kernels keep a lane-free
`_sliced` sibling behind `_device.prefers_tiled_reduction`.

**The correctness boundary is narrower than it looks: only a partition stride that isn't
`wp.block_dim()` is wrong, on *either* device (§2.2) — a one-element CPU tile is harmless, and a
tile reduction works fine inside a `@wp.func` too.** A blanket comment forbidding "any tile reduction
on CPU" is itself a defect: it is over-broad and blocks real wins.

Open upstream issues, cited in `_device.prefers_tiled_reduction`'s docstring: **NVIDIA/warp#1480**
(CPU/GPU tile parity) and **NVIDIA/warp#1638** (efficient CPU block execution). The branch above
becomes removable only when CPU blocks run more than one logical thread.

**Warp exposes no grid-wide barrier** (no cooperative groups, no `__threadfence`), and a spin-wait
emulation over `atomic_*` needs all blocks co-resident and a fence Warp cannot spell. That is the
ceiling on every level-synchronous rewrite (§14.9).

**Warp's hash grid has no per-cell entry point.** `wp.HashGrid` exposes only `hash_grid_query` /
`hash_grid_query_next` (a sequential per-thread iterator) and `hash_grid_point_id`, so the cell
*walk* cannot be split across lanes, only the per-candidate arithmetic can — the walk is 70-73% of
the per-candidate cost, so a cooperative search that keeps `wp.HashGrid` is capped at ~1.37x. **Never
propose a warp-per-edge search that keeps the hash grid.** Warp *does* ship a block-cooperative BVH
walk (`tile_bvh_query_aabb` / `tile_query_valid` / `tile_bvh_query_next`) — §14.2 — and it has a
silent correctness bug, next.

**`wp.tile_bvh_query_aabb` returns out-of-range primitive indices once a traversal round overruns
its result buffer, and every caller must bound-check what it hands back.** A round appends each hit
with an unconditional atomic counter increment but guards only the *write* against a fixed
`result_buffer_capacity` (`WP_TILE_BLOCK_DIM * 5`, i.e. 160 at `block_dim=32`) — once the counter
runs past capacity, a lane reads uninitialised shared memory as a primitive index. Nothing raises,
and the `>= 0` test every documented example uses does not catch it, since a garbage word is
positive about half the time. **A query box grown by a distance bound is what reaches the overrun**
— the round count scales with how many primitives one box meets, so a big mesh plus a generous
box is the trigger and a tight box on a small mesh never gets near it. Confirmed via
`compute-sanitizer --tool memcheck` on `proximity.mesh_to_mesh_distance`'s tiled straggler pass.

Two consequences:

- **Bound-check the index at every `tile_bvh_query_next` site**, `candidate >= 0 and candidate < n`.
  Both of triwarp's do. It is output-neutral by construction — an out-of-range index is never a
  primitive of the BVH — so it can only reject what was already garbage.
- **The guard stops the corruption and cannot restore the dropped primitives.** An overrun round
  silently loses hits, so a guarded walk may return an incomplete candidate set; that half is
  upstream's. In practice the two triwarp callers survive it because each has a *second*, sound
  bound on the answer — the global running minimum, and the pivot's own acceptance test.

**A query box grown by a distance bound is what reaches the overrun**, which is why this had gone
years unseen: the round count scales with how many primitives one box meets, so a big mesh plus a
generous box is the trigger and a tight box on a small mesh never gets near 160.

**Warp exposes no node-by-node BVH traversal** either (`bvh_query_aabb` / `bvh_query_ray` /
`bvh_query_sphere` / `bvh_get_group_root` only), so a BVH-pair wavefront means writing our own
hierarchy — price it as that, not as a rewrite of the query.

A `wp.capture_while` loop body that issues several launches does **not** replay as one unit on both
devices, so a loop whose *result buffer* depends on the whole body running cannot rely on it.

Measured on Warp 1.17 while removing a per-pass `wp.copy` from `graph.shortest_path_envelope`. The
copy exists because a Bellman-Ford relaxation reads every label and writes every label, so it needs
a second buffer and the result has to come back. Unrolling **two** passes into the body, each
writing into the other's buffer, removes the copy entirely and halves the conditional-graph
evaluations — worth **1.26x at 2 562 nodes and 1.45x at 40 962**. On a 2 562-node sphere at
`max_iterations` 1 / 3 / 7 it relaxed 16 / 51 / 181 nodes on CUDA against **6 / 31 / 141** on the
CPU device; even caps agreed exactly. The CPU path records through `ScopedCapture`'s APIC recording
(`wp.is_conditional_graph_supported()` is a *machine* query and returns `True` even for a CPU-device
array, so the capture branch is taken there too) and the body did not replay as two passes per
round.

Reverted. **The generalisable rule: a Python-level ping-pong cannot help a captured loop either** —
the body is recorded once and replayed, so rebinding the names only takes effect at record time.
Between those two, a double-buffered iteration inside `capture_while` keeps its copy.

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
  - **Do not answer that branch by widening the mask to `int32` first — write the concrete bool
    kernel.** `triwarp.reduce`'s `sum` / `any` / `all` opened with `astype(mask, wp.int32)`, which is
    an allocation of `4n` bytes, a launch, a full read of `n` and a full write of `4n`, after which
    the reduction read `4n` rather than `n`: **nine bytes of traffic per mask byte**, plus a launch
    and an allocation, to answer one boolean. A concrete bool-input family measured **1.60-1.94x** on
    the reduction itself across 8 k-164 k elements. There is no `wp.tile_load` of a `bool` array, so
    such a kernel cannot take the tile-load shape: its lanes walk the block's chunk striding by
    `wp.block_dim()` (§2.2, which is also what keeps it right on the CPU device) and the block folds
    the per-lane registers with a single `wp.tile` reduction — one tile reduction per block where the
    tile-load form runs `TILES_PER_BLOCK_1D` of them, and the strided reads still coalesce.
  - **And once it exists, widening a mask before a whole-array reduction is a measured loss, not
    merely redundant.** `int(reduce.sum(astype(mask, wp.int32)))` against `int(reduce.sum(mask))`
    measures **105.1 against 53.3 µs, 1.97x** — an allocation of `4n` bytes and an `array_cast`
    launch for a count the reduction already knows how to take. Two sites carried it
    (`smoothing.filter_spikes`, `remesh.subdivide_region_to_size`, the second inside its pass
    loop) and both are converted. The `counts_to_offsets(astype(mask, wp.int32))` sites are *not*
    the same defect — `wp.utils.array_scan` genuinely cannot read a `wp.bool` buffer — but they
    are no longer an `array_cast` either: `kernels/array.bool_flags` is a plain kernel writing the
    same 0/1 bytes at **11.0-11.3 µs against `array_cast`'s 20.3-21.0**, flat from 1 024 to
    1 000 000 elements, and `flatnonzero` / `mask_to_compact_ranks` / `remesh._compact` all use it.
  - **It moves a documented crossover, so re-check the declines that cite one.** The
    readback-versus-device-reduction crossover for a `wp.bool` mask went from ~1 M elements to
    **~0.5 M** (`reduce.any` against `mask.numpy().any()`: 1.54x slower at 163 842, 1.15x at
    350 000, level at 524 288, **1.35x faster at 1 M**, 2.97x at 4 M). Both sites in `repair.py` that
    carry a written decline of that shape still sit under it and keep their readback — with the new
    number written in place of the old one, which is the §9 discipline, not a conversion.
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
- **`wp.mesh_query_point_no_sign` + `wp.mesh_eval_position` is not an exact closest-point query** —
  it disagrees with an independent float64 oracle by up to ~2e-5 absolute, always reporting the
  *smaller* distance, and is a fixed point (re-querying from its own answer doesn't converge closer).
  A distance *bound* built on it is therefore exact only against Warp's own query
  (`proximity.closest_point_on_mesh`), not against an independent oracle. **How to test such a
  bound:** assert exactness against the same Warp query the bound was built from, and assert an
  *improvement ratio* against an independent oracle rather than the bound value itself — asserting
  `independent_distance <= bound` fails at tight bounds for a reason that isn't a bug.
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
  `tangent_space.halfedge_transport_angles` (and its dependents) are float32, and that storage
  precision — not the CG tolerance, not `vertex_tangent_frames` (which only names the reporting
  basis and isn't read by `connection_laplacian` at all) — sets the noise floor of everything built
  on `laplacian.connection_laplacian`. **The noise floor overlaps real signal**, so no magnitude
  threshold separates them; report it with a resolved/unresolved mask instead of a cutoff, which is
  why `heat.transport_tangent_vectors` returns `(transported, resolved)`.

  **General technique for "is this solver noise or input precision?", cheapest first:** (1) sweep the
  solver tolerance — flat means it isn't convergence; (2) inject noise into the suspected input and
  check linearity, extrapolating back to the unperturbed point; (3) split storage from arithmetic —
  redo in float64 but round the result back to the shipped dtype; unchanged means it's the *storage*,
  and an internal-only fix can't work.
- **Diffused heat fields scale as 1/scale² with the mesh coordinates**, so an absolute tolerance
  applied to one is a silent wrong answer on a rescaled mesh.

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
  `@wp.func` parameter, to a `wp.Scalar`-generic parameter, and into a kernel-scope slice; all
  spellings compile and agree.
- **`//` is not CPython's `//`** — §1.5 has the table.

Rules: §1.3, §1.5, §1.6.

### 12.6 Compilation, module hashing and import cost

- **A generic kernel's lazy overload instantiation rebuilds its whole module** — §2.5 carries the
  measurement (206 → 87 module loads, suite 1 033 s → 29 s).
- **`wp.map` forks its generated module per call *signature*, on axes wider than the dtype** — §3.5
  carries the measurement (182 → 143 loads; cold cache 2.1x).
- **`import triwarp` was expensive because `@wp.kernel` builds an `Adjoint` at import time for every
  decorated kernel, and importing one submodule doesn't avoid it** (Python imports the parent package
  first). Fixed with a PEP 562 module `__getattr__` that resolves each submodule lazily — details and
  the guarding test are in §16.2.
- **A deferral one module makes can be silently cancelled by an unrelated module's top-level
  import — check `sys.modules`, don't trust the comment.** `triwarp/reconstruction.py` defers
  `import warp.fem` on purpose, but two *kernel* modules were loading the whole package eagerly for
  two `@wp.func`s, so the deferral saved nothing. `warp.fem`'s public re-export shim is far more
  expensive to import than `warp._src.fem.linalg` directly (it pulls in `adaptivity`, `dirichlet`,
  `domain`, `field.*`, `geometry.*`), so the tree imports the private module instead, with a comment
  naming the Warp version this was probed against — the risk is an upgrade moving `warp._src`, which
  fails loudly at import rather than silently.
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
- **`wp.constant(x)` is `return x` after an `is_value(x)` check — on Warp 1.17 it is an identity
  function, not a declaration.** A bare module-level global with no `wp.constant()` and no typed
  constructor compiles and runs correctly from kernel scope on both devices, because Warp's codegen
  resolves *any* free variable that evaluates to a static value, wrapped or not — so
  `wp.constant()` adds no kernel-scope visibility, and `triwarp/constants.py` no longer calls it
  (§1.2). It also fixes no dtype. What it still legitimately does: raise `TypeError` immediately at
  the definition site if the value isn't scalar/vector/matrix-shaped, rather than failing later and
  more confusingly inside a kernel that references it.
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

- **`bsr_mm` returns a structural *superset* of the product**, with the extra entries exactly zero
  but interspersed in column order rather than trailing capacity (not NVIDIA/warp#1769, which is
  about trailing capacity — this is a different, still-present behavior on 1.17). **The zeros are
  not cosmetic once the result is multiplied again**: an explicit zero at `(i, c)` makes column `c`
  "see" row `i`, so a Galerkin product `PᵀAP` inherits every aggregate reachable from it — a measured
  47x pattern blowup on one coarse operator, operator complexity 2.19 instead of a true 1.03.
- **`bsr_compress`'s illegal-memory-access is FIXED on Warp 1.17** (was: compressing a `bsr_mm`
  result made the *next* `bsr_mm` die with `CUDA error 700`). `linalg._multigrid_prune` is now one
  call to `wps.bsr_compress(matrix, prune_numerical_zeros=True)`, ~2.9x faster than the CSR-to-triplets
  rebuild it replaced. **Trap found doing it: at the documented default `inplace=False`, it still
  returns the *same* matrix, pruned in place** — copy first if the unpruned matrix is still needed.
- **`bsr_set_transpose`, `bsr_mm` and `bsr_axpy` all read the `nnz` *field*, never `nnz_sync()`**, so
  a matrix whose count is stale carries garbage into whatever consumes it — the operand-side version
  of §3.7's buffer-sizing rule.
- **`bsr_mv` takes one vector**, so a cycle over several right-hand sides pays one launch per vector
  per mat-vec; a hand-written batched CSR kernel amortizes that across columns (shipped in the
  multigrid V-cycle).
- **Unwritten `(0, 0, 0.0)` triplets in a `wp.zeros` buffer all accumulate on entry `(0, 0)`, and
  `bsr_from_triplets`'s duplicate accumulation costs O(duplicates on the hottest address), not
  O(triplets).** A conditional triplet writer must point its unwritten slots *out of range*
  (`rows.fill_(n_rows)`, silently dropped) rather than leave them at a zeroed default — measured
  9-32x depending on how many triplets collide on `(0, 0)`. §3.7 is right about correctness (a
  structural zero is harmless) and wrong about cost; price a conditional emit by its collision
  count, not its buffer size.
- **`warp.optim.linear`'s `TiledDot` silently picks an O(n) per-block reduction whenever
  `batch_offsets` is set with `batch_count > 1`**, instead of its normal O(log n) tiled-tree
  reduction — and `use_bounded_tree`, the escape hatch, requires `batch_count == 1`, so a batched
  solve can never reach it. Two dots run per CG iteration, and at moderate `n` this can dominate the
  iteration cost. `linalg._BatchedCg` works around it with a real two-stage per-column tree
  (1.10-1.50x end to end, iteration counts unchanged); running one unbatched `cg` per column instead
  reaches the good reduction but duplicates every other kernel in the iteration and loses overall.
- **`warp.optim.linear.cg` silently resolves an omitted `atol` to `atol := tol`, turning a relative
  residual tolerance into an absolute floor of the same numeric value** — its convergence criterion
  is `max(atol, tol * ‖b‖)`. Once a right-hand side's norm falls below that floor, the solve returns
  **zero iterations** and the untouched initial guess as "converged" — a correct-*looking* silent
  failure, not a flagged one. **FIXED**: every triwarp call site now passes `atol=0.0` explicitly
  alongside `tol=` (`linalg.solve_spd`, `linalg.solve_spd_columns`'s single-column path, and both of
  `reconstruction`'s direct `wpl.cg` calls) — triwarp's own multi-column path (`_BatchedCg`)
  was never affected, since it computes its stopping threshold with an explicit
  `atol_sq = 0.0` already. This is what let a mesh-scale-dependent right-hand side (as in
  `heat.log_map` at extreme mesh scale) silently return zero instead of solving. Verified free at
  ordinary mesh scale: iteration counts and benchmark timings are unchanged before/after on normal
  inputs, confirming the fix only changes behavior once `‖b‖` is already pathologically small.

### 12.8 Warp builtins: adoption verdicts

**Adopted:**

- **`wp.mesh_query_point_sign_winding_number`** → `proximity.signed_distance_on_mesh(...,
  sign_mode="winding")`. Exact where ray parity misclassifies on meshes with holes, at a memory and
  runtime cost, so parity stays the default. Two traps: `support_winding_number=True` is required or
  the builtin **silently returns the ray parity answer**, and `wp.Mesh` doesn't retain the flag, so
  it can't be checked from Python — only a function that builds its own mesh can guarantee it.
  Warp exposes only the thresholded *sign*, not the value, so `proximity.winding_number` still needs
  a custom LBVH.
- **`wp.bvh_query_sphere`** (Warp 1.17), adopted in `kernels/neighbors.py` and
  `proximity.py::closest_point_on_edges` — a real win where the enumeration radius is large relative
  to an existing BVH. **It is bit-exactly `wp.length_sq(d) <= r*r`, not `wp.length(d) <= r`**, so
  adopting it means adopting the squared predicate everywhere, including the hash-grid branch.
  **Do not convert a probe whose radius equals the hash-grid cell width** — that's the grid's own
  best case, and the identical conversion in `ball_pivoting.ball_is_empty` was reverted as a real
  loss there (a small, well-centred query reaches the hash grid's target cell by address arithmetic,
  where a BVH always pays a root-to-leaf descent).
- **`wp.bvh_query_sphere` again, as a broad phase over *bounds*.** Adopting it in
  `curvature.discrete_mean_curvature` (in place of a cube `wp.bvh_query_aabb`) won 2-4x on the whole
  public call. **The platform fact underneath it: `wp.bvh_query_aabb`'s traversal costs several times
  `wp.bvh_query_sphere`'s per candidate returned on the same BVH, even when made to return the same
  candidate count** — so a cube broad phase is the wrong query whenever the caller's real predicate
  is a ball, independent of the extra candidates it returns. Not yet applied to `ball_pivoting`'s
  pivot search, which uses the tiled BVH walk and has no `tile_bvh_query_sphere` counterpart in 1.17
  — an open lead, unmeasured.
- **`wp.mesh_get_bvh`** (Warp 1.17) — `proximity.mesh_to_mesh_distance` now builds one structure over
  mesh B instead of two; §16.6.
- **`wp.volume_index_to_world`** — adopted for the convention, not the speed (perf-neutral against
  the hand-rolled transform it replaced).

**Rejected on measured evidence — do not re-propose without new data:**

- **`wp.intersect_tri_tri` cannot replace `intersection.triangles_intersect_sat`.** Its epsilon is
  absolute on unnormalized plane distances, so its verdict moves with mesh scale where triwarp's SAT
  is scale-invariant — it disagrees badly at both very small and very large scale, in either
  precision, and `mesh_with_mesh` is public API on arbitrary meshes.
- **`wp.closest_point_edge_edge` cannot replace `remesh._segments_dist_sq_d`.** Float32-only (the
  float64 branch of the caller never executes against it), and measurably worse relative error on
  near-parallel segments — exactly the near-degenerate case the existing float64 port exists for.
- **`wp.sample_unit_hemisphere_surface`** would replace `visibility`'s low-discrepancy Fibonacci
  lattice with a Monte-Carlo estimate at the same ray count — variance where there was none.
- **`wp.norm_huber`** is the Huber *norm* where `registration.robust_weight` needs the IRLS *weight*
  `ρ'(r)/r`.
- **`wp.tile_arange`** cannot express `array.arange` and loses where it can — its bounds are read at
  codegen, so a runtime start is a parse error, and a range fill has no reuse for a tile to exploit
  anyway (§13.1 already prices that call as launch- and allocation-bound).
- **`wp.volume_voxel_count`** is a capacity (§3.7).
- **The `dense_chol` / `dense_subs` / `dense_solve` family** is `hidden: True` / `doc: "WIP"`, and it
  takes `wp.array[float32]` where the caller holds a `wp.spatial_matrix` in registers — a 2x loss
  (§2.9).

### 12.9 `wp.Volume` as a voxel-set container

`triwarp/voxels.py` ships all of this and its module docstring carries the detail. On Warp 1.16+,
`allocate_by_voxels` and `fem.Nanogrid` both work on **CPU** with identical counts and byte-identical
`get_voxels()` row order across devices, so a volume-backed module doesn't have to be CUDA-only; an
empty point set raises `RuntimeError` rather than aborting the process, but still guard it.

- **`Volume.allocate_by_voxels(world_points, voxel_size, translation)` deduplicates**, and beats a
  hand-rolled `unique_rows` dedup by ~2.3x on both devices.
- **`volume_lookup_index(grid, i, j, k)` is the `k`-th row of `get_voxels()`**, and `-1` when absent
  — the grid and the cell array share one canonical numbering, so a per-voxel payload is just a
  `wp.array(n_voxels)`. No side table, no hash map, O(1) membership.
- **NanoVDB centres voxels on integers**, so to make its index equal a corner-aligned cell index
  (`floor((p-origin)/s)`), pass `translation = origin + 0.5*voxel_size` — without the shift every
  cell comes back +1 on every axis, silently.
- **`get_voxels()` order is leaf-major**, deterministic but *not* globally lexicographic — lexsort
  before comparing to a reference and never assume it matches a C-order reshape.
- **`voxel_points` may be integer** — a contiguous `(n,3)` int32 (or `vec3i`) array is read directly
  as index-space cells, no float round trip.
- **`point_mask` (int32, one per point, 0 = ignore) filters during the build**, so no compaction
  pass is ever needed, and a fully-masked one-point build is the only way to make a legal EMPTY
  volume (a zero-length input raises instead).
- **Rebuildable volumes: the payoff isn't there** — `rebuild()` into a pre-reserved topology is only
  marginally faster than a fresh `allocate_by_voxels`, so it doesn't pay for `dilate`-style loops.
  Two traps if used anyway: `get_voxel_count()` returns the reserved *capacity*, not the active count
  (read `get_active_stats().voxel_count`, §3.7); and the four `max_*` capacities **cascade** — passing
  only `max_active_voxels` under-reserves the other three, runs out of device memory, and leaves the
  CUDA context throwing illegal-memory-access on everything after. Pass all four explicitly.
- **`warp.fem.Nanogrid(volume)` derives topology triwarp would otherwise hand-write** — verified
  exact against a hand-built solid-block topology. `.vertex_grid` is the deduplicated corner lattice;
  `boundary_side_index()` + `side_position` + `side_normal` enumerate outward faces;
  `side_inner_cell_index` over boundary sides is the 6-connected surface-voxel set. Import `warp.fem`
  **inside the function**, never at module scope (§12.6).

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

- **`compute-sanitizer` is at `/usr/local/cuda-12.8/bin/compute-sanitizer`, not on `PATH`**, and must
  run with `--target-processes all` (through `uv` it hops two processes). **A run that instrumented
  *nothing* also prints `ERROR SUMMARY: 0 errors`, so the proof is the slowdown** (~10x) — always
  time the same suite both ways to confirm it actually instrumented something.
- **Both `wp.capture_while` sites sit behind `wp.is_conditional_graph_supported()`**, so on a box
  where that returns `False` the whole CUDA-graph path is skipped and the tests pass green having
  tested only the fallback. Confirm it's actually `True` here with a pytest plugin that
  monkeypatches `wp.capture_while` / `wp.capture_launch` and counts calls.
- **Re-running our own repro script validates the repro, not the upstream bug.** That is exactly how
  the `bsr_mm` misattribution survived two version re-probes. When a workaround's justification is an
  upstream bug nobody else has confirmed, suspect the repro.

**Gates for an upgrade**, all four: the full suite (1953 passed / 1 skipped at the 1.16 bump),
`basedpyright` 0 errors, `zensical build --strict` clean, `tests.parity` with an unchanged pair count.
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
loop is up to 14x wrong, because Warp leaves the CUDA mempool release threshold at 0, so every sync
drains the pool and the next allocation is cold. *(Raising the release threshold to 8 GB, measured
on real wrappers, is only worth 1.00-1.02x — not a lever.)*

| primitive (correct regime) | cost |
|---|---|
| `wp.launch` / `wp.launch_tiled` | **11.8 / 12.2 µs**, independent of `dim` |
| `wp.empty` | 6.1 µs, flat in size (`wp.empty(0)` 2.0) |
| `wp.zeros` | 9.3 µs |
| `wp.full` | 9.3 µs |
| `arr.fill_` / `arr.zero_` | 3.2 / 2.5 µs |
| `wp.copy` | 4.5 µs |
| `wp.clone` | 13.3 µs |
| `wp.array(numpy)` | 16.1 µs |
| a slice view | 3.0 µs |
| **`wp.utils.array_cast`** | **20.8-22.1 µs — 1.8x a plain launch** |
| `wp.utils.array_scan` | 7.1 µs |
| **`wp.utils.array_sum`** | **39.8 µs — 3.4x a plain launch** |
| `wp.utils.radix_sort_pairs` | 15.6 (int32) / 18.6 (int64) µs at `n = 1`; 64.3 / 86.1 at 61 440 |
| `array.flatnonzero(200k)` | 115 µs (97 after the `array_cast` removal below) |
| a cached `wp.map` call (Python overhead above the kernel) | ~11 µs |
| a host readback | ~0.1 ms *queued*, **14.3 µs isolated** — the 0.1 ms is the pipeline drain in front of it, not its own cost (§16.7), so price it by what is queued |
| `wp.synchronize_device` | 1.1 µs |
| a **replayed** kernel in a captured chain | **1.17 µs** at n=17 689, 1.57 µs at n=163 842, exactly linear from 1 to 12 kernels |

The launch and allocation rows were re-measured on Warp 1.17 (400 calls between two syncs, min of 9)
and each is 20-25 % above what this table carried from an earlier version; the cost model below is
still accurate because every term moved together. **The three rows in bold are the ones that change
decisions**, and all three were being reached for as though they were free: `wp.utils.array_sum` is
3.4x a launch, which is why four of them were 62 % of `measures.moments` (§16.4);
`wp.utils.array_cast` is 1.8x a launch for a copy a plain kernel does identically, which is why
`flatnonzero` stopped using it; and an *isolated* readback is 14.3 µs rather than 0.1 ms, so a
function that already synchronized is not paying 0.1 ms for its second read.

A host-cost model of `allocations × per-call cost + kernels × 9.7 µs` accounts for 84-122% of a
wrapper's measured host time, verified on seven wrappers spanning 0.3-2.9 ms.

**Launch marshalling is ~1.0 µs per argument, linear, identical on both devices:**

| Args | CUDA median / min | CPU median / min |
|---:|---|---|
| 2 | 16.6 / 14.8 µs | 12.7 / 11.2 µs |
| 8 | 22.5 / 20.8 | 19.9 / 17.5 |
| 16 | 30.6 / 27.2 | 29.2 / 27.2 |
| 24 | 39.0 / 36.3 | 39.2 / 34.5 |
| 28 | 43.5 / 41.1 | 41.0 / 37.2 |

So the often-quoted "~32 µs per `wp.launch`" is the *mean* kernel's launch (triwarp's mean argument
count is 5.1), not a constant. A `@wp.struct` bundle collapses it: a 25-argument kernel against the
identical kernel taking one bundle measures roughly 0.4x median at every size, a flat ~25 µs saving
regardless of `dim` — what a host-side cost should look like. Building the bundle costs ~2.6 µs.
Rule and eligibility: §2.8. Only 18 of 440 triwarp kernels take ≥12 arguments.

**A generic kernel costs a further ~12 µs of host-side overload resolution on every launch — roughly
double the host cost of a concrete one — and `wp.Float` / `wp.Scalar` cost exactly what `Any`
costs.** `wp.launch` runs `infer_argument_types` over the *whole* argument list before looking the
overload up, so the cost scales with how many parameters are generic, not with which spelling names
them; a kernel with three generic parameters pays roughly double what one with a single generic
parameter does. This is flat in `dim`, so it's a genuine per-launch cost, not a first-call effect.
**The fix is one line per module: `wp.overload()` already *returns* the concrete `wp.Kernel`, and
the registration code was calling it and discarding the result.** Keeping it in a dtype-keyed table
lets the wrapper hand `wp.launch` the resolved kernel directly, with the kernel source staying
dtype-generic (§1.2 untouched). Landed across all 42 generic launch sites in the tree; verified
end-to-end against a detached baseline worktree across ~20 representative wrappers, 19 of 20
improved (typically 1.1-1.6x) and none regressed — full mechanism and rule at §2.5 and §16.0. **A
missing dtype now raises rather than silently rebuilding the module**, turning the §2.5 failure mode
from a clock reading into an error naming the kernel.

**A cached `wp.map` call carries the same kind of overhead**, roughly 1.8x the launch it wraps
(~11 µs) — that's what §3.5's `return_kernel=True` hoist removes, and it prices the hoist for any
loop that calls `wp.map` repeatedly.

**Per-segment packing costs are host constants and flat in the data** — the same rows measure 1.46 ms
at 0.07 MB total and 2.42 ms at 268 MB, a 4 000x range:

| operation | per segment |
|---|---|
| `wp.copy` (`pack_1d_arrays`, `concatenate`) | **6.02 µs** |
| `wp.clone` (`split(copy=True)`) | **15.06 µs** — a ~10 µs allocation plus a ~6 µs copy |
| a `wp.array` slice view (`split(copy=False)`) | **3.63 µs** |
| one whole-buffer `wp.copy`, 48 903 to 2 614 242 elements | **0.010-0.015 ms**, flat |

So the vast majority of a many-segment pack is per-call overhead, not the data volume.

**The lever is a single segmented-copy launch, and the claim that there was none was wrong.**
`_pack_segments` carried the reasoning *"there is no segmented alternative — Warp has no
array-of-arrays and a kernel cannot dereference a raw pointer"*. Warp has an array-of-arrays: a
`@wp.struct` may carry a `wp.array` field, and a `wp.array` of that struct is a descriptor table a
kernel indexes as `segments[s].data[k]`. One launch over it replaces the whole loop:

| segments (total fixed at 2 614 242) | 2 | 8 | 32 | 64 | 256 | 1 024 | 4 096 |
|---|---|---|---|---|---|---|---|
| copy loop | 0.026 ms | 0.063 | 0.172 | 0.341 | 1.372 | 5.187 | 21.529 |
| one launch | 0.094 | 0.092 | 0.091 | 0.094 | 0.092 | 0.102 | 0.144 |
| ratio | 0.27x | 0.68x | **1.89x** | 3.64x | **14.91x** | 50.76x | 149.75x |

The launch form is flat in the segment count *and* in the total size (0.078-0.094 ms from 48 903 to
2 614 242 elements), so the whole choice is a threshold — `array.PACK_SEGMENTS_KERNEL_FROM = 32`.
End to end it is **5.1-5.7x** on `concatenate` / `pack_1d_arrays` at 256 segments; the gap to the
14.9x above is the Python-side per-segment validation loop and `require_same_device`, which is the
§3.9 contract rather than a defect.

Two details decide whether it pays, and the first is most of it:

- **Build the descriptor vectorized.** Constructing the 256 struct instances one at a time costs
  **0.619 ms** of a 0.714 ms per-object build — it would give the whole win back. One NumPy
  structured array through the struct's own `numpy_dtype()`, uploaded once, costs **0.081 ms**.
- **One kernel serves every dtype, not a table of them.** Declare the descriptor's array field
  `wp.array[wp.int32]` and point it at the segment's storage with a length in 4-byte *words*, and
  alias the destination the same way (`wp.array(ptr=..., dtype=wp.int32, shape=...)`, because
  `wp.array.view` refuses a dtype of a different size). It is then a byte copy that never names the
  caller's dtype — verified byte-identical for `int32`, `float32`, `float64`, `vec3` and `uint64`.
  A dtype whose itemsize is not a multiple of 4 keeps the loop.

**`split(copy=True)` is not reachable this way** and keeps its 15.06 µs per segment: its outputs are
`n_segments` *separate arrays*, so the allocations are the return value.

Several alternate spellings were probed and rejected as real losses (a raw-pointer `wp.array` view
that re-implements `wp.array.__getitem__`'s contract, a shared output buffer for `split(copy=True)`
that drops half of what the copy promises, dropping `src_offset=`/`count=`, and others) — all written
at their sites in `triwarp/array.py`. **The NumPy crossover is a segment SIZE (~98 kB), and it does
not move with the segment count** — a useful rule of thumb when deciding whether to route a packing
call through NumPy or Warp.

**The general lesson, and it is worth more than the row: a "Warp has no X" comment is a claim to
probe, not a fact to inherit.** This one had stood through several optimization rounds and was
quoted as settled. The probe that refuted it is fifteen lines (§10 says the same thing about
introspecting a builtin before planning around its absence).

**A device reduction costs ~0.10-0.32 ms flat on CUDA regardless of `n`** (launch + 4-byte read),
while a readback scales with bytes copied — the crossover is wherever the copy exceeds ~0.15 ms
(around 200k `int32`/`float32`, 1M `bool`, 16k `vec3d` elements). Below it a reduction launch is
pure overhead; above it the readback grows without bound.

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
barriers plus a prefix scan ≈ 600 ns; if the level's own work is under that, the rewrite loses.
That is exactly how the BFS drain went (§14.9).

Two facts about the primitives themselves:

- **`wp.tile_sum(wp.tile(x))[0]` is a genuine block-wide broadcast reduction** — correct on *every*
  lane, not just lane 0, and it syncs before and after, so it also doubles as a usable full block
  barrier (as does `tile_scan_exclusive`).
- **Tile ops are legal inside a dynamic `while` loop** on CUDA and behave correctly across lanes.
  Keep the loop condition block-uniform (read it from a global every lane reads after a barrier, or
  from registers every lane updates identically) — `__syncthreads` in divergent flow is UB.

**A single thread's throughput is ~35 ns per independent memory op**, so a serial pointer-chasing
kernel is usually throughput bound rather than latency bound; check that before trying to hide a
stall.

**A `wp.hash_grid_query` cell probe costs ~600 linear-scan point tests** (a hash plus two dependent,
uncoalesced global loads per probe, against a broadcast-out-of-L2 linear scan). Consequence: once the
search radius outgrows a couple of cell widths, an exact O(n) scan is genuinely cheaper than widening
the walk — the break-even span scales as `n^(1/3)`, which is why `_knn_widest_grid_radius` is
`n`-aware rather than a fixed constant.

**`wp.launch_tiled` with every lane walking the whole chunk and lane 0 doing the atomics is FASTER
than one thread per chunk**, even though it looks 64x redundant — all lanes read the same address at
each step so the loads broadcast, where one-thread-per-chunk gives each lane its own run and the
reads stop coalescing. The redundant arithmetic is free; these reductions are memory-bound. See
`accumulate_procrustes_moments`.

**Dropping a `sqrt` from a ball query's narrow phase is flat** — the root costs nothing against the
candidate walk's memory traffic. Declined on that count *and* because `wp.length` vs `wp.length_sq`
are not the same predicate in float32 (§12.4).

**SHIPPED — a single-slot atomic reduction serializes on one address, so its cost is linear in the
launch; converting it to a lane-partition + `wp.tile_sum` + single guarded commit is a large,
real win that grows with the launch size** (roughly 1x at a few thousand elements up to 10-30x at a
million). The converted form needs no CPU/CUDA branch, because its lanes partition a chunk the block
already owns with a `wp.block_dim()` stride (§2.2). **Do not convert a *conditional* atomic** (a
compaction cursor, a change flag, a rare-event counter): contention there is proportional to hits,
not to the launch, so there's nothing to win.

**The same finding recurs at *one atomic per block*, and it names the real variable — the block
count, not the redundant lane arithmetic.** Several kernels already had every lane walk a whole
chunk and only lane 0 publish, so they looked done (64x better than a naive per-thread atomic) — but
they still issued one atomic per *block*, and the block count itself was the remaining cost.
Re-launching at a finer block grain (the same fold width the reduce module uses) cuts the block
count substantially and is worth a further several-x at large sizes. **Isolating the 64-fold "wasted"
lane arithmetic alone (keeping the grid, giving each lane one element) recovers almost nothing** — so
the shape to look for is not "lanes doing discarded work" but "how many blocks reach the
accumulator", and the lever is the fold width, not the redundancy.

**The shape is greppable**: look for `wp.launch_tiled` at a `dim` sized to the *item* count (rather
than the reduce module's own block-count helper) with a `lane == 0` commit. Not every kernel matching
that `dim` shape matches the defect, though — a kernel that partitions the *outer* work at a constant
stride (rather than reducing into one shared accumulator per block) is a different, `_sliced`-paired
case, where changing the fold is a rewrite of the partition, not a one-token change. **That rewrite
was tried for one such case and came back flat, which sets a threshold for the whole family: the
quantity the fold reduces is `blocks x accumulator slots`, and it needs to be around 1e5 or more
before there's anything to win** — compute that product before proposing this class of rewrite.

Two implementation notes for the next conversion of this shape: a wide accumulator needs the fold
*more*, not less (more per-block reductions to amortize); and because `wp.tile_sum` is block
collective, all of them run *outside* the `lane == 0` guard and only the final commit is inside it —
assemble the whole vector/matrix from the tile sums first, since `wp.atomic_add` reads trailing
indices as array dimensions, not vector components, so a per-component commit isn't possible.

**Flattening a reduction to its launch floor can make a *neighbouring* fusion worth doing, and a
written decline can expire silently as a result.** A fusion between a reduction and an adjacent
kernel was declined when the reduction was most of the pair's cost — once the reduction above became
a flat launch floor, the same fusion became a large win (both launches now cost about the same, so
removing one removes roughly half the pair), output bit-identical. **After landing a large win on a
kernel, re-read the declines about the launches either side of it** — a written decline is not
permanent if the thing next to it changes (§9).

**A single-address `float64` `wp.atomic_add` serializes the launch entirely** — a global dot needs a
two-stage reduction, not a single accumulator.

**Padding must be a hole, not a value.** Pointing padded rows at a dummy valid index makes
`bsr_from_triplets`-style accumulation collide on one entry and serialize; sending them out of range
(silently dropped) is the fix. Same trap as §12.7's zero-padded triplets, from the other direction —
**look for it whenever padding has a *value* rather than being a hole.**

**Two tiling antipatterns, both measured in `kernels/reduce.py`:**

1. **A `wp.tile_load` kernel below one tile is a large loss** (measured ~49x on one case). When the
   *reduced extent* is under the tile width, `launch_tiled` still gives every block many lanes, and
   all of them redundantly walk the same short row — fixed by falling back to one plain thread per
   output row when the reduced extent is small. **But the opposite reduction axis must stay tiled**
   — the same fallback on the other axis is a large loss the other way, because there are too few
   outputs to keep the device busy serially. **The dispatch key is the reduced extent, not the
   axis**, and this recurs one rank up for 2-D tables with a short trailing dimension — fixed the
   same way, gated on the trailing extent (and contiguity, since flattening a non-contiguous view
   raises).
2. **One atomic per tile does not scale** (measured ~4.9x lost at large sizes). Global 1-D reductions
   issuing one atomic per 64-element block put far too many blocks on one accumulator address at
   scale; folding several tiles into a register before the atomic recovers most of it. Swept fold
   widths found a middle value that never loses across the whole size range tested.

3. **`tile_chunk` reports what is left to the end of the array, not the block's share of it, and
   clamping is the caller's job.** Its own docstring says so; the tile-load factories satisfy it
   *implicitly*, through a fixed `TILES_PER_BLOCK_1D` loop that cannot overrun. A kernel whose loop
   is bounded by `remaining` instead — any `for k in range(t, remaining, wp.block_dim())` — must
   write `remaining = wp.min(remaining, ITEMS_PER_BLOCK_1D)` or block 0 walks the whole array. It
   fails loudly for a sum (measured 9 774 961 against 99 737) and **silently for min/max/any/all**,
   where re-reading elements another block already read is idempotent. Look for it whenever a new
   reduction kernel does not use `wp.tile_load`.

   **The coupling hazard this creates: changing how much work a kernel does per block silently
   breaks any *other* module that launches it directly and computes its own `dim`.** A caller that
   launched the unfolded kernel at a stale `dim` against the folded version returned a wrong answer
   (each block re-folding tiles another block also claimed) — invisible for min/max/any/all, since
   the fold happens to be idempotent for those, and wrong only for `sum`. Fixed by exporting the
   block-count helper for callers to reuse. **`grep` for direct launches of a kernel before changing
   its per-block contract.**

**Two more shape facts:** `wp.tile(vec3)` decomposes to a scalar tile, so tile-reduce a vec3 per
component or pack it; and `wp.array.view(wp.float32)` on a vec3 array gives a zero-copy `(n, 3)` view
for `reduce.minmax`.

### 13.3 Tuning constants are per-device

`ITEMS_PER_SLICE` (elements per thread in the lane-free strided-slice reductions) has a
device-dependent optimum: **CUDA wants long slices, CPU wants short ones** — shipped as
`ITEMS_PER_SLICE_CUDA = 128` / `ITEMS_PER_SLICE_CPU = 32` behind `_device.items_per_slice(device)`.
**Slice length is not one number even within a device**: 32 suits reductions into one or a few
accumulators, while a *per-query* reduction wants ~128 (`proximity.ITEMS_PER_QUERY_SLICE`) because
the query dimension already fills the device and a short slice only multiplies atomics.

**When adding a device-dispatched path, re-ask which device still reaches each branch, and re-sweep
the constants that branch reads** — the old measurement may no longer describe any live call site.
`ITEMS_PER_SLICE` had been tuned against two call sites that, after a later change, stopped reaching
the sliced form on CUDA at all — leaving the constant tuned for a consumer nothing in the original
sweep had weighted for.

**Sweep the values you didn't try the first time when re-probing after an upgrade, or the re-probe
inherits the original's blind spot.** Two constants re-probed after a Warp upgrade gave opposite
outcomes: one reproduced exactly (no change needed); the other's original sweep had only tried two
candidate values and read as a clean two-bracket split, but a finer sweep revealed a *third* bracket
in between, which the coarse sweep's two points straddled — costing a real loss at exactly the size
the old crossover was tuned around. **A constant whose sweep sampled only two values has not been
shown to be a two-bracket problem.**
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
**0.12-0.60x at 200 000**), `visibility.support_argmax_sliced`, `proximity.winding_number_tiled`
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

- **A single-block cooperative BFS drain for `graph.bfs`.** Built, verified byte-identical, and it
  loses badly against the existing serial engine. The serial drain is memory-throughput bound on one
  thread, not latency bound, so neither software pipelining nor a register-vector batch of loads
  helped; and on a narrow, non-growing frontier (a long thin "ribbon" graph) the per-level work is
  smaller than the barrier cost a correct cooperative round needs, so thousands of levels put a
  multi-millisecond floor on synchronization alone before any real work happens. **General lesson:
  a per-level dispatch cost only matters if the level body actually runs** — `graph.bfs` already
  hands off to the cheap serial path once the frontier stops growing, so on a graph shaped like that,
  "fuse the level's kernels" is a no-op fix for a stage that never executes.
- **A persistent one-block-per-loop tiled kernel for the Liepa hole-fill DP.** Built, byte-identical,
  and it loses at every rim size tried, worse as the rim grows — the DP's total work outgrows what
  one SM can do serially long before the existing many-small-launches design's launch overhead
  becomes the bottleneck. **Same conclusion as the BFS drain, from the opposite direction: one
  design had too little work per level for a block, this one has far too much.** The only design
  that would beat both is whole-GPU work between cheap level barriers — a grid-wide barrier, which
  Warp does not expose (§12.2).
- **Tile solves at triwarp's own problem size** — §14.4.
- **A Chebyshev smoother** — §14.8.
- **Voxel aggregation for the multigrid hierarchy**: the geometrically-natural aggregation blows up
  operator complexity far more than the algebraic aggregation already shipped, and a cheaper
  unsmoothed variant doesn't fix the underlying convergence-rate dependence on mesh resolution either.
- **Several micro-optimizations of the old Bridson blue-noise propose kernel**, all superseded by an
  algorithm change (§16.7) but each a valid null result in its own right: a couple of cheaper random
  permutation schemes introduced enough bias or overhead to be a net loss; dropping either of the two
  pruning/safety passes in the kernel was a large loss, confirming both are load-bearing rather than
  incidental; and making an eager shuffle lazy was a wash, since its result was almost always consumed
  at only its first element anyway.
- **A `wp.Stream`-overlap rewrite for any "independent-but-sequential" pair, tree-wide.** A
  repository-wide sweep for public functions with two kernel-launching branches that have no data
  dependency between them found five real candidates — `metrics._distances_mesh_to_mesh`'s two
  `closest_point_on_mesh` calls, `reconstruction.ball_pivoting`'s hash-grid-then-BVH build pair,
  `smoothing.filter_implicit_fairing`'s cotangent-stiffness/mass-matrix pair (repeated every
  smoothing iteration), and a same-operator/different-right-hand-side CG solve pair
  (`heat.extend_scalar`'s two `_solve_scalar` calls). Each was built and measured with two
  `wp.Stream`s joined by `wait_stream`, against the sequential default-stream form it already has, on
  an RTX 5090 / Warp 1.17 across 642-163 842 vertices: **0.87-1.07x** — indistinguishable from noise,
  never a repeatable win. Two structural reasons, not a per-pair fluke: (1) every branch here is
  host-launch-overhead dominated (§13.1) — device time measured at 0.01-0.19 ms against a
  0.13-1.4 ms wall time — so there is at most a sliver of device time for a second stream to hide
  behind, and the `ScopedStream`/`wait_stream` bookkeeping costs about as much as that sliver; (2) a
  CG solve issues its convergence check as a host readback every `CG_CHECK_EVERY_FALLBACK` (10)
  iterations, so wrapping a whole solve in one stream context still runs every one of its readbacks
  to completion before the *next* Python call — the other solve — issues a single kernel: sequential
  Python calls cannot overlap through streams alone unless the two loops' iterations are interleaved
  at the call site, which is a rewrite of the iteration, not a stream annotation. **Do not build a
  `wp.Stream`-overlap convention on the strength of "these two calls have no data dependency" alone**
  — check the device/wall split first (§13.1, §16.1). For a same-operator, multiple-right-hand-side
  solve specifically, reach for the existing batched machinery
  (`linalg.solve_spd_columns`'s `_BatchedCg`, §16.8) instead of streams, since it merges the
  columns into one *launch* sequence rather than trying to run two independent ones concurrently.
  (Merging their Krylov *subspaces* as well is a different mechanism, was built, and was removed —
  §16.8.) `heat.extend_scalar` already goes through it; `heat.transport_tangent_vectors` reaches it
  through that call.

### 14.10 Producer-consumer fusion: always fuse; the iteration count only decides whether a benchmark can see it

**Two consecutive launches at the same `dim` where the second reads the first's output only at its
own thread index are fusible, and the fused kernel is faster — every pair measured in this tree, at
every size, 1.04-2.65x — because it removes a launch, an allocation and a full round trip of the
intermediate buffer through global memory.** What varies is not whether the fusion wins but whether
any *benchmark* can resolve the win, and that is set by the region's share of the call it sits in.

A tree-wide scan found **93** runs of consecutive same-`dim` launches across `triwarp/`, of which
**89** adjacent pairs passed the index-locality test. The scan is ~120 lines of `ast` (walk each
wrapper's launches in line order, map each launch's argument expressions onto its kernel's parameter
names, and ask whether every array the first kernel *writes* is read by the second only at a name
bound from `wp.tid()`); re-derive it rather than re-reading 93 call sites.

**Measure the region, not the call that contains it** — §15.2's rule, and the one this pass got
wrong first time round. Time the two launches against the one, with nothing else in the window; a
whole-call A/B cannot resolve a region worth 0.1 % of a CG-dominated solve, and its noise then reads
as a verdict. Both worked examples are below, and they landed on opposite sides of the noise floor
for exactly this reason.

**SHIPPED — the explicit smoothing filters**, where the pair is inside a pass loop so the saving is
multiplied by the iteration count and the benchmark sees it directly. Every filter alternated
`kernels/laplacian.apply_operator` (one row of the row-stochastic averaging operator into an
`(n_vertices,)` `wp.vec3d` buffer) with a per-index step kernel reading that buffer at its own
vertex. The row apply is now a `@wp.func` (`kernels/laplacian.operator_row`, and
`operator_row_scalar` for the float32 field — two functions, because the vec3d one promotes the
float32 weight to float64 and the scalar one does not), and each filter launches one kernel per
pass. Measured on an RTX 5090, Warp 1.17, 10 passes, operator precomputed, **output bit-identical on
all nine paths**:

| group | min-ratio |
|---|---|
| `filter_scalar_laplacian` | **1.23-2.08x** |
| `filter_taubin` | **1.17-2.04x** |
| `filter_humphrey` — four launches per pass to **two** | **1.14-1.79x** |
| `filter_laplacian_integration` (explicit) | **1.48-1.77x** |
| `filter_sharpen` | **1.16-1.53x** (delegates to `filter_laplacian`) |
| `filter_normals` | **1.36-1.40x** |
| `filter_mut_dif_laplacian` | 1.04-1.33x |
| `filter_two_step` / `filter_spikes` / `inflate` | 1.01-1.23x (delegates) |

Two details worth carrying forward. `filter_humphrey` applies the operator *twice* per pass and the
update needs both `L.v` and `L.b`, so only `L.b` disappears — **a fusion can remove one of two
intermediate buffers and still be worth 1.79x**. And `filter_normals`' fusable pair is not inside an
iteration at all: its pass is seed → crease-gated neighbour *scatter* → normalize, and the scatter
needs the whole seeded buffer, so the pair is the normalization with the **next** pass's seed, which
only becomes adjacent once the first seed is peeled off the front of the loop. **When the middle
stage of a three-stage pass blocks the obvious fusion, look across the loop boundary.**

**SHIPPED — `heat`, and this one was declined once on a bad measurement before being re-measured and
landed.** `face_unit_gradients` / `vertex_field_to_face_field` fed `integrated_divergence` /
`scatter_face_field_to_vertices` at `dim=n_faces`, index-locally, at three sites (`heat_geodesic`,
`heat_signed_distance`, `log_map`); all three are now one kernel, sharing
`kernels/heat.accumulate_face_divergence` and `face_field_from_vertex_field`. Priced directly, on
icospheres from 1 280 to 327 680 faces:

| fused pair | ratio across the size sweep |
|---|---|
| `unit_gradient_divergence` | **1.04-2.47x** |
| `vertex_field_divergence` | **1.38-2.50x** |
| `scatter_unit_gradient_to_vertices` | **1.49-2.65x** |

**Never slower, at any size.** The ratio falls toward the large end because the region becomes
bandwidth-bound and the saved launch is a smaller share of it — *not* because the fusion stops
paying.

**What the first, wrong reading looked like, because the shape recurs.** Attributing through the
whole call gave 1.021x at 10 242 vertices and 1.006x at 40 962 — read as "a share that falls as the
input grows", which §9 calls a decline — and a 25-row harness sweep gave 0.987-1.041x, whose sub-1.0
cells read as possible regressions. Both were artifacts: every caller is dominated by two CG solves,
so the fused region is **0.14 %** of `log_map[saddle]`, and 0.1 % of a 28 ms call is far under the
harness's ~1-3 % noise. Re-measured interleaved, the worst cell (`log_map[saddle]`, twice reported
below 1.0) is a **tie at 0.998-1.000** and the other (`heat_signed_distance[band]`, reported 0.996)
is **1.004**; the second harness sweep after the rewrite read 22 of 25 cells at or above 1.0, to
1.12x. **A cell whose region is a fraction of a percent of the call cannot report that region's
speed — do not let it vote.**

Two deterministic cross-checks that settled it far faster than any clock, and that §15.6 recommends
for exactly this: the **kernel count** is `base - 1` per fused site (so no solver iteration count
moved, which was the live worry, since the fusion shifts the answer by 1-3 ulps and a changed
iteration count would have been a real mechanism for a slowdown); and the **allocation count** is
strictly lower by one `(n_faces,)` `wp.vec3d` buffer per site. With the same kernels, the same
iterations and one fewer allocation, there is no mechanism by which the rest of the call can get
slower — which is what makes the sub-1.0 cells provably noise rather than arguably noise.

**SHIPPED — the straight-line one-shot pairs, eight of them, and the surprise is how large they
are.** These were first declined as "marginal"; priced directly (interleaved, min-of-mins over five
process pairs) every one is a substantial win, because a pair that runs once still replaces two
launches with one and usually shares loads as well:

| fused region | ratio (small / large input) |
|---|---|
| `creation` two `offset_cap_faces` launches → one 2-D launch | **3.46x / 3.36x** |
| `vertices` `face_crosses` + `max_vertex_normal_weights` | **1.94x / 1.96x** |
| `creation` `sweep_plane_normals` + `sweep_transforms` | **1.89x** |
| `holes` `loop_centroids` + `cone_faces` | **1.80x** |
| `holes` `project_loop_to_plane` + `bridge_loop_to_ring` | **1.58x** |
| `graph` `edges_to_csr`'s weighted structure + values | **1.61x / 1.57x** |
| `reconstruction` `negative_divergence` + `screened_inverse_diagonal` | **1.62x** |
| `energies` `hessian_corner_gradients` + `voronoi_mass` | **1.29x / 1.13x** |

**Do not read these off the harness — its noise floor swamps them.** The same benchmark run reported
`cone[512]` at **0.873x** on a code path the change never touched (`cone` goes through `revolve`),
so ±13 % is the resolution for these sub-millisecond groups. Price the region (§15.2).

**Two launches of the *same* kernel merge into one wider launch, and that is the biggest win in the
table.** `offset_cap_faces` ran twice per extruded or swept solid — near cap reversed at offset 0,
far cap at offset N — into adjacent blocks of one buffer. One `dim=(2, n_cap)` launch with the row
index selecting the offset and the winding is 3.4x. **Grep for a kernel launched twice in a row with
different scalar arguments; it is the cheapest fusion there is.**

**Still declined:** `energies.laplacian_smoothing_loss`' `cot_row_scales` (the two launches are in
*different branches*, not sequential). **And a pair inside a graph-captured loop really is not a
candidate** — a replayed launch is ~1.17 µs (§14.3), so `polyline.polyline_triangulate`'s
ear-clipping round saves nothing by losing one of its four.

### A fused kernel is not finished until the duplication *inside* it is gone

**Inline any `@wp.func` the fusion leaves with a single caller, then look for what the inlining
exposes.** A helper that is not reused is a name, not an abstraction; and while it stays a helper
the redundancy between the two halves is invisible. This pass inlined **15** — nine created by the
fusions and six pre-existing step rules (`laplacian_step`, `neighborhood_average`,
`humphrey_residual`, `humphrey_update`, `mut_dif_adil`, `scalar_laplacian_step`) that stopped being
`wp.map` targets when their kernels absorbed them. Four survived on real reuse: `operator_row` (5
call sites), `heat.accumulate_face_divergence` (2), `vertices.write_max_corner_weights` (2),
`graph.write_adjacency_pair` (2).

**What the inlining then exposed, in three of eight kernels — and it is worth more than the launch:**

- `energies.hessian_face_terms` loaded the face's three vertices **twice** and computed `dbl_area`
  **twice** by two different spellings. They are the same expression: `triangle_double_area(a, b, c)`
  *is* `wp.length(wp.cross(b - a, c - a))`, which the gradient half had already formed. Reusing it
  took the region from **1.15x to 1.29x** at 20 480 faces and from **1.00x — a tie — to 1.13x** at
  327 680. That fusion would have shipped as "no measurable gain".
- `reconstruction.poisson_level_setup` computed the centre node's `poisson_grid_index` twice.
- `holes.extend_rim_to_ring` loaded `loop_vertices[t]` twice.

**The fix is a local, not a new `@wp.func`** — the redundancy is *within* one kernel, so a helper
would only be a named way to recompute it. Reach for a shared helper when two *kernels* repeat a
run; an α-renamed statement-run scan over all nine touched modules found none from this work (its
only hits are the pre-existing `fill_dp_span` / `fill_dp_span_tiled` tiled/serial pair).

**One near-duplicate is deliberate, and removing it was built, measured and reverted.**
`smoothing.diffuse_scalar_pass` spells out the CSR row walk rather than calling `operator_row`,
because that one promotes the float32 weight to float64 for a `wp.vec3d` accumulator where a scalar
field is float32 end to end. Both ways of merging them were tried:

- **Keep float32 storage, accumulate in float64** — the cheap version, no extra bytes. **Does not
  compile**: `wp.float64(w) * f32_value` is a hard parse error, `Input types must be the same, got
  ['float64', 'float32']`. This is §12.4's "scalar arguments must be constructed at the input's
  precision", and it is what makes the two irreducible rather than merely inconvenient.
- **Carry the field in float64**, which does let one `Any`-generic `operator_row` serve both (seed
  the accumulator with the first term instead of a typed zero — there is no way to spell "zero of
  `Any`'s type" in kernel scope, and `0 + a == a` exactly, so it stays bit-identical). Built, and
  **0.86x on `lucy`** (14 M vertices, 5.15 → 5.96 ms) and **0.63x at 655 k**: the field is one of
  four streams the CSR walk reads, so doubling its width costs bandwidth no launch saving repays,
  before counting the two conversion passes a float64 iterate adds per call.

So the duplication stays and the numbers live at the site. **The general point: two kernels that
differ only in a dtype are not always mergeable, and when the merge costs memory traffic the right
answer is to duplicate the loop and write down why.** The `Any`-generic form was also reverted on
§4.2 — with the scalar path gone it had exactly one instantiation.

**Three traps in the scan itself, because a re-run will hit all three.** Index-locality from an AST
walk is *necessary and not sufficient*: it does not follow `@wp.func` calls, so a kernel reaching a
neighbour's entry through a helper reads as local (`graph.connected_component_labels`' `ecl_hook`,
`voxels.fill_cavities`' `flood_hook`); it does not see **intervening host work**, so a pair with a
`counts_to_offsets` scan between the two launches is reported adjacent and is not fusible
(`intersection.split_mesh_with_plane`, `voxels.to_boxes`); and a *claim/commit* independent-set pair
is never fusible however local it looks, because commit must see every claim
(`remesh._flip_interior_edges`, `intrinsic_delaunay`). Verify each candidate by reading both kernel
bodies and the wrapper lines between the launches.

**And check what the fusion killed.** Removing the last launcher of a kernel leaves dead code that
still costs import time (§12.6) — this pass stranded and deleted `kernels/laplacian.apply_operator`,
`kernels/smoothing.apply_operator_scalar`, `smoothing._apply_operator`, and all four of
`kernels/heat`'s unfused halves — and it stales the `wp.map` bookkeeping, since a fused step kernel
is one fewer `wp.map` site (six generated `map_*` modules stopped being built, and check 23's
allowlist comment for `smoothing` named an op that is no longer mapped at all). **Move the deleted
kernel's prose onto what survives**: four `kernels/heat` comments carried the `sign` convention, a
normalization precondition and a deliberate non-merge decision, all of which had to be relocated
rather than lost, and five references to the deleted names in `triwarp/heat.py` and
`tests/test_heat.py` had to be repointed.

---

## 15. Benchmark and measurement traps

### 15.1 A sudden slowdown is a Warp rebuild until proven otherwise

Before profiling anything, before believing a kernel got slower, before deleting a test for being
slow: a single-digit-second operation that now takes tens of seconds, or a test file whose cost
appears and disappears as you change *which* tests you select, is almost always a module recompile
from an unregistered generic-kernel overload (§2.5) or an undeclared `wp.map` signature (§3.5).
Measured up to 1 200x apart between a fresh test selection and an identical repeat.

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
that hid the problem for as long as it lasted.

**The second candidate is host-side per-element Python, and it is a distinct class** — not a rebuild
(the GPU is idle for both, so that tell does not separate them) and not launch overhead (those are
milliseconds, not minutes). One benchmark ran over 20 minutes without finishing at 0% GPU utilization
because a `validate=True` path built Python `set` comprehensions over a `.numpy()`-read array — one
interpreter iteration per mesh edge, invisible on ordinary meshes and catastrophic on the largest scan
mesh. Replaced with a device scan plus a small readback. **When a benchmark group stalls, grep the
triwarp function it times for a Python `for` / `set` / comprehension over anything derived from
`faces` or `vertices`** — a `.numpy()` feeding a comprehension is the signature — and confirm by
timing the triwarp call alone on the largest registry mesh. The fix is a device scan, **not a smaller
benchmark mesh**: capping the mesh hides the defect, which is what the largest row exists to prevent.

### 15.2 Attribute against one number

A projection built by subtracting two measurements of *different* things is a hypothesis, and in this
repo it has been optimistic by 3-10x every time. The one projection that held came from a single
directly-attributed number; three others that came in far under a rough subtraction-based estimate
each mixed up two different quantities — a fallback pass's cost estimated from two different
per-span metrics rather than the pass itself; an amortized-cost claim read from a warm, unrepresentative
solve rather than the real per-iteration cost; and a memset count that couldn't actually be removed
because it was initializing kernel inputs, not padding an allocation.

**Price the candidate directly** — time the exact call in isolation, or count its launches and
multiply. Two specific traps: a *warm* repeat of a stage measures a different regime than the cold
one inside the real call (a CG solve especially), and `wp.empty` vs `wp.zeros` differ in what they
actually remove, so "remove the memset" and "remove the allocation" are different claims.

**And price it at more than one size, because the *sign of the trend* is the decision.** Two items in
one pass had nearly the same share at the small end and opposite verdicts, which only a second size
revealed: one readback's share of its call *grows* with mesh size, so the fix is worth more the more
it matters and it landed; a different readback's share *falls* with mesh size, which §9 calls a
decline, and it was declined. A single operating point would have read the two as the same item.

**The rule fails in *both* directions, and the pessimistic direction is the one that quietly throws
away real wins.** The failure above is a subtraction that flatters a candidate; the mirror is
measuring a candidate *through* a call it barely occupies, where the enclosing noise becomes the
verdict. A `heat` kernel fusion was declined on exactly that: whole-call A/B gave 1.021x then
1.006x — read as §9's falling share — and a 25-row harness sweep gave 0.987-1.041x, whose sub-1.0
cells read as regressions. Priced directly, the fused region is **1.04-2.65x at every size and never
slower**; it is 0.14 % of the call that reported it worst, and 0.1 % of 28 ms is far under harness
noise. **Before reading a ratio near 1.0 as a verdict, compute the candidate's share of what you
measured** — under a few percent, the measurement cannot see it and the honest next step is to
isolate the region, not to declare a decline. §14.10 has the full case.

**A loop with a convergence break reports the break point, not the change.** An
`icp_point_to_plane` hoist read **1.86x** end to end and was worth **1.02-1.06x**: the two arms
stopped at different iterations, and nothing else about the 874 µs "saving" was real. Two directly
measured maps at ~10.5 µs each on a ~200 µs iteration was the whole of it. **Pin the iteration
count before timing an iterative solver** — `threshold=0.0` here — or the ratio is fiction. The
same call is also not bit-reproducible once converged (the point-to-plane normal equations
accumulate through `float32` atomics), so two runs of the identical build disagree in the last
digits; a value gate on it has to compare a *trajectory* and expect the plateau to wobble.

### 15.3 Attribute at the benchmarked operating point

**A profiled share is a share at one point on the parameter axis, and it can reverse an
optimization's sign.** Attributing a blue-noise sampler at its natural, dense-sampling parameter
found one kernel dominating device time; inverting it (running one thread per *accepted* point
instead of per *alive* point) was a large win there — but at the actual benchmarked, sparser radius
the same inversion was a real *loss*, because far fewer points get accepted per round and the
inverted kernel starves the device instead of using it. Reverted. **Grep the benchmark for how it
derives its parameter and profile at *that* value** — §9's interleave rule doesn't catch this, since
both arms were timed correctly, just at the wrong radius.

**The same rule applies to the *fixture*, not just the parameter.** A `mesh_to_mesh_distance` fix was
first evaluated against a heavily-interpenetrating pair of copies, where the cost it targeted was a
small and shrinking share of a call that was mostly dominated by something else entirely — reading as
a decline. The actual benchmark fixture places the copies disjoint side by side, where that same cost
is a much larger, real share and the fix is a genuine win. **Read the fixture, not just the call.**

### 15.4 Benchmark-harness hazards

- **`benchmarks/test_meshes.py` is a real gate and the default `pytest` run does not collect it.**
  It self-checks the registry against a topology table every feature mesh must match, and a mesh
  registered without its matching row fails there with a bare `KeyError` while the full suite,
  `basedpyright`, `zensical build --strict` and `tests.parity` all stay green. **After touching
  `benchmarks/meshes.py`, run `pytest benchmarks/test_meshes.py`** (a few seconds) as a fifth gate.
- **`--benchmark-json` is written at session end, so one pathological row costs the whole module.**
  A newly added reference library lost several modules this way — a size cap that governed every
  case except the one it existed for, a cap check that was unreachable dead code, and an uncapped
  row that starved a later triwarp row of device memory entirely. **General rule: put a
  `skip_larger_than`-style cap first in every group, before any library-specific branch; make sure
  it actually covers meshes outside the reference's own registry; free any foreign library's device
  memory in its teardown, outside the timed region; and put a bounded per-module wall-clock cap in
  the runner** so one bad row can't silently blank out the rest of a module's results.
- **Never benchmark Poisson surface reconstruction on very large point clouds with a reference
  library that doesn't scale to them** — one run went for over 90 minutes without completing a
  single row and had to be killed, blocking every module after it. **When a benchmark module's wall
  clock looks wrong, get the per-library split before trimming anything triwarp does** — the slow
  side is often the reference, not the code under test.
- **A `benchmarks/` harness number and an isolated hand-probe number for the identical call are not
  directly comparable** (the harness syncs and cold-pools differently around the timed region than a
  hand probe does) — **compare probe to probe or harness to harness, never across, and label which
  kind a quoted number is.** Comparing across the two once nearly produced a false regression report.
- **A median at a low sample count (`rounds=3`) can be unrepresentative of its own samples** — a
  one-off cost (the shape of a Warp module load) landing in two of three samples can inflate a
  median far above the floor while the floor itself never moved. **Run the aggregate script's
  "suspect" check and read it before trusting a loss table** — it flags any cell whose median
  exceeds its own minimum by a wide margin, on both triwarp's and a reference's own cells (an
  inflated *reference* median flatters a triwarp "win" the same way). A flagged cell should be
  re-measured on the next round before anything is built against it; once confirmed clean, the
  underlying real result can still be a genuine, if small in absolute terms, loss that a
  milliseconds-ranked loss table would otherwise never surface.

### 15.5 A plan item may be refuted by its own target

In one plan round, **all three items worked were refuted, and in every case the refutation was
already written in the file the item proposed changing** — a counterexample named in the constant's
own comment, a scope mismatch the benchmark's own docstring already anticipated, and a threshold
whose tuning basis the plan misdiagnosed by looking at the wrong stage.

**A plan is written by reading loss tables and diffs; a measured decline is prose and lives next to
the code, so a table-driven pass systematically cannot see it.** Before measuring a plan item, grep
the target function, its constant's comment *and its benchmark docstring* for a number.

**And when a "nothing separates these" note exists, check which *kind* of quantity it ruled out
before accepting it.** Several attempts at a predictor tried different properties of the *solve*
(size, iteration count, convergence-rate extrapolation) and all failed the same way; the property
that actually separates the cases was a property of the *operator*, never tried (§16.8).

**Two refutations from the host-cost sweep, both cheap to re-derive and both wrong-looking-right:**

- **Narrowing edge/row keys to 32 bits to halve the radix sort.** `radix_sort_pairs` really is
  1.3-4.1x cheaper for `int32` than `int64` in isolation (72.8 against 17.8 µs at n = 4 096;
  148.1 against 81.8 at 1 M). End to end `unique_1d(return_inverse=True)` on `uint32` keys measures
  **0.91-1.01x** against `uint64` across icospheres 2-6 — no better, sometimes worse — because
  `uint32` is outside `_unique_hash`'s native `(int32, int64)` set and picks up a `bitcast_to_int`
  copy, and because the call is host-bound anyway. The lever was never the key *width*; it was the
  reduction *around* the packing (§16.4).
- **Replacing `map_sorted_inverse`'s binary search with a hash-slot lookup.** Recording each
  element's table slot at insert time and resolving the inverse as `inv_perm[scan_pos[slot] - 1]`
  trades `log2(n_unique)` dependent probes for three. Stage-profiled, `map_sorted_inverse` is
  **13 µs of `unique_1d`'s 254.9** — under 5 % best case, against an extra `n`-sized `int32` buffer
  and a wider `hash_insert`. Not built. **Stage-profile before optimising a stage**; the obvious
  suspect here is 5 % and the radix sort's *host floor* is 19 µs of it.

### 15.6 A/B without `git stash`

**Do not `git stash` to measure current code against pre-change code.** The working tree is shared —
a stash cycle can sweep up in-flight edits arriving from outside the session while it runs, reverting
them on disk for a window in which a save would conflict.

Use a detached worktree, which never touches the working tree:

```bash
git worktree add -q --detach $SCRATCH/baseline HEAD
```

**The catch that makes it non-obvious:** `uv sync` installs triwarp through a *MetaPathFinder* that
outranks both `sys.path` and `PYTHONPATH`, so `PYTHONPATH=$SCRATCH/baseline python probe.py` still
silently imports the working tree. Drop the finder first:

```python
sys.meta_path = [f for f in sys.meta_path if "editable" not in getattr(f, "__module__", "")]
sys.path.insert(0, BASELINE)
```

**A `sed`-based in-place sweep of a tuning constant is the same hazard as `git stash`, and it's
worse because it looks harmless** — it mutates a tree another process may be running tests against.
A constant baked into a kernel can't be swept within one process anyway (Warp fixes it at codegen),
which is exactly the case that tempts the in-place edit; put the value in a detached worktree instead
and drive it with the main venv's interpreter.

**And check whether the box is yours before taking any clock reading at all.** `nvidia-smi`'s
utilization and a process listing cost nothing and are the difference between a measurement and a
fiction — a sweep once read a 4x swing at *fixed* parameters purely because foreign processes held
the GPU busy during part of it. Two rules follow. **Interleave A and B inside one process where the
change permits it** — coarser-grained interleaving is too slow to track noise on a second-scale
timescale. And **when the box is not quiet, measure a quantity that is not a clock**: iteration
counts, launch counts, level counts, candidate counts, `nnz`, and whether a reference mutated its
input are all deterministic, and can settle a question a noisy clock can't.

**But a failing `nvidia-smi` is not evidence that CUDA is unusable — probe the driver through
Warp.** NVML is a separate userspace library from the CUDA driver API Warp calls, so a host whose
`nvidia-smi` dies with `Failed to initialize NVML: Driver/library version mismatch` can still
report `"cuda:0" : "NVIDIA GeForce RTX 5090" (31 GiB, sm_120, mempool enabled)` from `wp.init()`
and launch kernels correctly — measured, with the full CUDA suite green in that state. Cost of not
knowing this: one review pass recorded itself as CPU-only and wrote off its own CUDA evidence. So
`nvidia-smi` is the *box-is-quiet* check above and nothing more; availability is
`wp.get_cuda_device_count()` plus one real launch.

**Verify with `print(triwarp.__file__)` before trusting a single number.** Do **not** `uv run` from
inside the worktree — it resolves that copy as its own project and builds a second virtualenv; use
`.venv/bin/python` directly.

### 15.7 Timing hygiene

- **Interleave A and B in one loop and report the `min` alongside the median.** Timing all of A then
  all of B lets GPU clock state decide the winner — one sweep produced non-monotonic ratios and a
  recurring artifact that reversed into a clean monotonic trend once interleaved under one clock
  state with the GPU pre-warmed.
- **Saved baselines drift ±10% (±30% under 100 µs) between sessions.** Flagged deltas in that range
  have appeared on benchmarks whose code hadn't changed at all; re-running both arms back-to-back
  showed the true delta was much smaller, with the sign flipping between reps.
- **A reference library's own column is the control that licenses a cross-session comparison.** A
  stale benchmark table once carried findings built on numbers that had drifted since — but because
  the *reference's* column was unchanged within noise across the two sessions, that reference served
  as a control proving the drift was real and belonged to triwarp's side. **Re-run the *whole group*
  including its reference rows**, and a table that stale is worth rewriting rather than annotating,
  since its *conclusions* mislead more than its numbers.
- **Do not `wp.synchronize_device` around each launch** when timing a microsecond kernel — that
  measures sync latency, not the kernel, and can read wildly different numbers for the identical
  kernel depending on launch count. Batch K launches, sync once, divide.
- **The device/wall timer's per-launch synchronization inflates *wall* badly** — one case was off
  by more than 10x. Take launch counts and device totals from
  `wp.timing_begin(cuda_filter=wp.TIMING_KERNEL | wp.TIMING_MEMSET, synchronize=True)`, and wall time
  from a separate un-instrumented loop.

### 15.8 Probe-process contamination

**A loop-over-configurations probe script is not the same experiment as the code under test.** One
process looping over several cloud sizes and depths produced what looked exactly like a
nondeterministic backend bug (the same configuration returning correct output on one run and an
error on the next, sandwiched between succeeding neighbors) — but running each configuration in its
own fresh process gave the correct output every time, identical to CUDA, matching what the real test
suite already showed. **The "defect" was the probe's own cross-configuration state leaking, not the
product.**

**Before attributing a failure seen in a multi-config probe to the product, re-run the single
failing configuration in a fresh process**, and prefer running the real tests over re-deriving them
in an ad hoc script. **Corollary for the reverse direction:** a probe that *passes* in one process
says nothing about a suite that runs hundreds of allocations before it — holding several very large
structures alive at once in one process has separately produced a CUDA allocator error that was the
probe's own footprint, not a reproduced defect.

### 15.9 Decide on the harness number, not the isolated one

`check_every=0` for the heat-family CG read 2.1-2.3x in an isolated probe and 1.05-1.23x in the
harness, with **0.94x** on the `saddle_graded` conditioning rows (§14.6). Same shape as §15.4's
1.2-2.5x factor: **decide on the harness number.**

### 15.10 `wp.timing_begin` is blind to graph-replayed kernels, so a captured function reads as ~100 % host

**This invalidates the device/wall split for every function that graph-captures**, and it fails in
the most expensive direction: a device-bound function reads as host-bound, which points the next
optimization at launch elimination when the kernels are the cost. Measured directly: 20 loose
`wp.launch` calls report 20 kernels to `timing_begin`; the identical 20 calls replayed from inside a
capture report **zero** kernels and zero time.

**The reach is much wider than triwarp's own handful of explicit capture sites**, because
`warp.optim.linear`'s solvers capture their iteration by default. So every triwarp CG solve — heat,
parametrization, smoothing, `min_quad_with_fixed`, `solve_spd*` — has its dominant kernels hidden
from this measurement, and at least three functions were attributed backwards this way (one read as
97% host that was actually ~100% device; two others read as 90-97% host that were actually
70-87% device).

**How to measure it correctly.** Re-run with capture disabled and read `timing_begin` there; the
kernel *set* is unchanged, so the device total transfers back to the captured run (only the wall
does not — the uncaptured wall is several times higher, which is what the capture is worth).
`warp.optim.linear` takes `use_cuda_graph=False`; triwarp's own `wp.capture_while` sites fall back
to direct execution when `wp.is_conditional_graph_supported` returns `False`, so monkeypatching that
is the lever. **Both arms must be instrumented identically** (§9) — the uncaptured arm is the
measurement, not the baseline.

**The tell that this was wrong all along was already in the tree**: a docstring had concluded the
call was device-bound from a solver-tolerance sweep, not from `timing_begin` — and that sweep was
right. **When a device/wall split contradicts a tolerance or input-size sweep on the same function,
trust the sweep**: it cannot be fooled by where the kernels were issued from.

**Two second-order traps in the same family, both of which make host time look larger than it is:**
a `.numpy()` readback's wall time is the queue depth in front of it (§14.6), not its own cost; and
once enough launches are pending, the driver's launch queue fills and `wp.launch` itself blocks —
a device-bound signature that can look like marshalling cost if read naively.

---

## 16. triwarp component status

Shipped results, open defects and refuted plans, by area. **Check here before opening work on any of
these.**

### 16.0 The launch-resolution pass, and what it says about where to look next

**Done: every generic-kernel launch site and `kernels/reduce.py`'s generic kernels now go through a
dtype-keyed table of concrete kernels**, removing ~12 µs of `infer_argument_types` from each launch —
the table and mechanism are §13.1, the rule is §2.5, the factory guidance §2.7.

**Three method points from that pass, because they generalise past this one finding:**

- **A static AST scan of kernel annotations undercounted the generic kernels; a runtime census
  found them all.** A *factory* can leave a `dtype` parameter generic by default with nothing in the
  source text saying so. Take the census from Warp itself
  (`[k for k, v in wp.get_module(name).kernels.items() if v.is_generic]`), not from the parser — the
  same reasoning applies to any property Warp computes rather than the source spells out.
- **A written decline is not a landed one.** A docstring had documented "the generic form is
  declined here" for years, yet a third of its own instantiations took the generic default anyway,
  because a factory parameter had one. §9's "read what calls the thing, the decline may already be
  written there" has a converse: grep for the sites that should have obeyed a written decline and
  didn't.
- **The probe that measures a fix must not also change the thing it measures.** A first pass
  monkeypatched `wp.launch` to substitute concrete kernels and read *losses* on two wrappers; both
  were artifacts of the patch itself (it left non-array generic parameters generic, and added its
  own Python wrapper to every launch in that arm). Adding a "same wrapper, no substitution" control
  arm turned both apparent losses into real wins.

### 16.1 Where triwarp's benchmark losses actually are

**The whole mid-level surface is host-bound, and the cheapest way to see it is the flatness
census.** 44 public functions timed at 320 faces and at 81 920 faces — a 256x range — on an
RTX 5090, Warp 1.17: **43 of 44 came out flat within 1.25x**, one (`bounds.oriented_bounding_box`)
at 1.28x, and **none above 3x**. `validation.is_watertight` is 1 870 → 2 070 µs, `vertex_one_rings`
490 → 512, `grouping.unique_rows` 454 → 451, `measures.moments` 250 → 256. Corroborated
independently by `wp.timing_begin` on `grouping.unique_1d` (no capture in that path, so §15.10 does
not apply): **wall 254.9 µs against 46.9 µs of device time — 82 % host**, over 10 device ops.

Two consequences, and they set the shape of every optimization in this package:

- **There is usually no kernel to make faster.** The currency is the *count* of Warp API calls, and
  §13.1's table is the price list. A `cProfile` of `tw.reduce.sum(bool)` measures 52.0 µs of which
  the raw `wp.zeros(1)` + `launch_tiled` + `read_scalar` sequence is **50.1 µs**: about 2 µs is
  triwarp's own Python. Micro-optimising wrapper code is not a lever; removing Warp calls is.
- **Flatness across the mesh size is the measurement to take first**, before any profiler. It costs
  two timings, it needs no instrumentation, and it cannot be fooled by graph capture the way a
  device/wall split can (§15.10).

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

A later attribution sweep across every module with a mid-band loss found that almost everything in
the 0.3-20 ms band is **66-97% host** at high launch counts (`split`, `robust_laplacian`,
`delete_region_keep_boundary`, `cluster_decimate`, `remove_degenerate_faces` all above 85% host) —
a small number of exceptions (`ambient_occlusion`, `lscm`, `bfs[sphere_med]`,
`query_nearest_bvh_k1`) are genuinely device-bound. **Tiling cannot touch a launch floor; only launch
*elimination* can, and reading the benchmark table alone (without the device/wall split) misdiagnoses
most losses in this band** — first hypotheses that assumed kernel/algorithm cost were wrong more
often than not.

One single-kernel outlier, unrelated to launch count: `linalg.assemble_interior_system` spends
**20.4 ms in one `_bsr_accumulate_triplet_values`** (91 % of device time) because it routes an
already-row-sorted, already-deduplicated CSR through `bsr_from_triplets` — 418 176 triplets in,
418 176 nnz out. That is 61 % of `min_quad_with_fixed[saddle]` and is paid by `harmonic`, `lscm` and
every constrained solve.

**Also host-bound, from a different angle:** ICP's per-iteration cost over 10 iterations on 2 562
points (1 419 µs) is 343 µs (24 %) of `cost_acc.numpy()` readback and 139 µs of `dim=1` solve launch,
against only 60 µs (4.2 %) of actual Cholesky. **Half of that readback is now gone and the 24 % no
longer holds** — `accumulate_point_to_plane` commits its cost and its weight sum into *one*
length-2 buffer, so the loop reads both in the sync it was already taking right after that launch
rather than taking a second one three launches downstream (which drained a pipeline the first had
already drained, so it was a full sync, not §16.7's cheap ~0.02 ms follow-on read). One fewer
`zero_()` per iteration rides along. Measured **1.065x / 1.052x** on `sphere_small` / `sphere_med`
at ten iterations — real, and much smaller than 24 %, because that attribution predates the tiled
rewrite of the accumulate itself.

### 16.2 `import triwarp`

Fixed via a PEP 562 module `__getattr__` in `triwarp/__init__.py`, resolving each submodule (and
`Trimesh`, a class, as a special case) on first access and caching it into the module namespace so a
later access is a plain global lookup. Cause: §12.6 — `@wp.kernel` builds an `Adjoint` at import
time for every decorated kernel, and importing a submodule doesn't avoid it because Python imports
the parent package first.

**The guarding test must run `import triwarp` in a subprocess and assert zero kernel modules are
pulled in** — in-process the answer is always "all of them", because by the time a test runs the
session already imported what it needed. Do not "simplify" it to an in-process check. Two things
the laziness deliberately does not change: overload registration still runs before the first launch
through its module, and nothing calls `wp.load_module` / `wp.force_load` at import.

### 16.3 `reconstruction`

- **`screened_poisson` is correct on the CPU device but not fast** — the `dense` solve is over a
  `2^depth`-cubed node grid whatever the cloud size, so each level costs ~8x. Test depth is
  therefore device-dependent: 5 on CPU, 6 on CUDA (the `adaptive`/`warp.fem` backend is far cheaper
  on CPU and gives the same faces, but isn't the default). **Error is not monotone in depth** — past
  some depth the octree resolves sampling noise rather than the surface — so a test must never
  assert "finer depth reduces error".
- **`screened_poisson(point_weight=0.0)` returns a handful of zero-area triangles, every run** — the
  only one of the four reconstruction entry points with no degenerate-face cleanup pass. Found
  through a reference library's warning about it; see §7.7 for the general technique.
- **`ball_pivoting`'s persistent-front rewrite fixed the documented v1 watertightness limitation**:
  the old design rebuilt the front from the whole triangle soup every wave, producing many more
  boundary edges than the persistent front does; a subdivided icosphere now reconstructs exactly
  with zero boundary edges. **The "overlapping sheets" artefact was a consequence of the per-wave
  rebuild, not inherent to a wave-parallel front** — do not reinstate that caveat. Several design
  choices here were counter to the obvious guess and shouldn't be re-tried: `wp.capture_while` is
  *slower* than a batched host loop for this loop shape (§14.3); the hash-grid cell width should be
  the *ball* radius, not the wider pivot neighbourhood (a wider cell makes the far more frequent
  empty-ball query enumerate many more points than it needs to); and compacting the front to only
  live edges was a loss in the pre-rewrite design (fewer threads, less parallelism) and only pays off
  now because it's a free side effect of the pivot pass.
- **`ball_pivoting`'s wave loop was run-to-run nondeterministic, fixed; `repair.make_winding_consistent`
  still isn't.** The cause was integer arrival order (not FP non-associativity — BPA has no float
  atomics): a proposal slot handed out by one atomic was reused as the *priority* for a second,
  unrelated atomic, so which triangle won a contested vertex depended on GPU scheduling and a losing
  proposal's edge got retried later against already-mutated state. Fixed with a proposal key derived
  from the proposal's source edge instead of its arrival slot. **A globally fixed key alone starves
  the front** (the same proposals win the same contested vertices every wave, so each wave commits a
  smaller independent set) — fixed by salting the key with a hash of the wave counter, restoring
  performance while staying reproducible. **General lesson: when making an order-dependent algorithm
  deterministic, check whether the replacement order is fixed across *rounds*, not just within one.**
  **Fixture lesson: no uniformly-sampled closed sphere can test this** — its exact Euler triangulation
  leaves wave order nothing to decide, so only an irregular-spacing fixture (e.g. a torus) exposes
  the nondeterminism at all. Still open: `repair.make_winding_consistent` seeds each connected
  component from an arbitrary face, so per-component winding still isn't reproducible even though the
  unoriented triangle set is.
- **CLOSED: `ball_pivoting`'s intermittent CUDA error 700 was a host-side buffer swap, not an
  allocator or lifetime bug.** `_bpa_run` swapped its front buffers after every *queued* wave, but a
  wave kernel silently no-ops once the device has already stopped partway through a batch — so an odd
  number of no-op swaps leaves the pair exchanged, and the next compaction pass reads whatever
  garbage a stale (sometimes never-written) buffer holds as edge-table indices. Fixed by tracking how
  many waves actually ran and adding one more swap when that count is odd. **General lesson: when the
  memory tools (compute-sanitizer, debug mode) come back clean on a memory-shaped symptom, instrument
  the control flow instead — they can perturb the very timing the defect's schedule depends on.**
  Several plausible-sounding prior readings ("it's the allocator", "it's a lifetime issue") were each
  true as *symptoms* of the swap desync, not independent causes, which is why fixing the swap closed
  all of them at once.
- **`ball_pivoting`'s default (`radius=0`) auto-radius is nondeterministic** (an order-dependent
  atomic float sum), which the docstring's reproducibility claim didn't originally cover. **Resolved
  as a documentation fix, deliberately, not a determinism fix**: the reproducibility claim is now
  scoped to an explicit `radius=`, since the auto-guess is a heuristic nothing downstream is
  calibrated against, and a caller who needs a fixed triangulation needs a fixed radius anyway. Do
  not "fix" this by making the underlying reduction deterministic without a caller that needs it
  (§4.2).
- **Slab-chunked marching cubes is not viable and was abandoned.** `wp.MarchingCubes` is crack-free
  only *within* one grid — its per-cell face triangulation isn't consistent across independent
  invocations, so welding independently-computed z-slabs leaves non-manifold seams even though each
  slab is individually manifold. Fixing it needs a face-consistent MC reimplementation; since the
  solve is spacing-capped anyway the payoff is marginal, so the capped full-array extraction was
  kept instead.
- **`warp.fem` gotchas from the adaptive backend, each of which produces a plausible wrong answer
  rather than an error:** an `ImplicitField` func must have no return annotation;
  `allocate_by_voxels` is voxel-*centered*, so the extraction lattice needs a half-voxel translation
  or it lands outside the domain and produces a spurious surface component; solve in *index space*
  so the screening-vs-stiffness balance matches the dense calibration; and a point-source weak form
  rings when cells are much smaller than the sample spacing, fixed by capping the grid depth rather
  than rounding it.

### 16.4 `remesh`, `repair`, `creation`, `bounds`

- **SHIPPED — `isotropic_remesh`'s smooth pass is the Botsch-Kobbelt area-equalizing relaxation
  its Notes promise, with the fold veto the collapse stage already runs.** Each one-ring neighbour
  is weighted by its own barycentric area (`laplacian.mass_matrix_entries`), and every proposed
  move is vetoed if it would invert an incident face. `tests/test_remesh.py` passes, and the call is
  **1.35x faster** end to end (`icosphere(5)`, 3 iterations: 22.1-23.1 → 16.3-16.8 ms, stable over
  alternating arms) despite the extra launches — the same "a rejected move removes downstream work"
  effect the collapse veto showed.
    - **The `cave_cube` blocker this item recorded no longer exists, and it was never the smoothing's
      fault.** Area-weighting with *no* veto now leaves `cave_cube` watertight and
      self-intersection-free at every configuration probed — 8 and 20 iterations, target 0.5x and
      0.25x the mean edge, with `reproject=False` as well — and the whole remesh suite passes
      without it. What had made it fold was the missing **collapse** fold veto, fixed since (next
      item); the two readings were one defect seen from two stages.
    - **The quality gain is real but far smaller than the stale "352 → 20" this item quoted**, which
      was measured against a baseline that predates that collapse veto. Against today's baseline,
      99th-percentile aspect ratio: `icosphere(3)` **1.308 → 1.173** and `hemisphere` **1.804 →
      1.496** (both clear wins), graded patch 4.211 → 3.826 at 3 iterations and 1.575 → 1.596 at 10
      (a wash), `unit_box` 1.414 → 1.706 and `cave_cube` 1.414 → 1.483 (small regressions, because
      an already-uniform structured grid is a fixed point of the *unweighted* smoother and the area
      weights perturb it). Net: better on the curved fixtures, slightly worse on the flat-faced
      boxes, and it is what the docstring says.
    - **The veto is kept although nothing in the suite needs it, because it is free and it guards a
      hazard the sibling stage guards for the same reason** (§2.4's one-rule-one-spelling): veto
      against no-veto measures 16.5-16.9 against 17.1-17.3 ms, two ranges that overlap. It fires on
      the graded patch and nowhere else.
    - **A tangential step to a convex combination of the one ring cannot fold a *convex* vertex
      link** — the target is inside the link polygon by construction. So every well-shaped fixture
      in the suite is unable to reach the veto, and a test built on one asserts nothing; the guard's
      own test needs a deliberately non-convex link (a deep notch of near neighbours opposite far,
      heavy ones), which is `test_smooth_pass_vetoes_a_move_that_would_fold_a_face`. The weighting
      itself is pinned by a hand-computable NumPy oracle in
      `test_smooth_pass_is_the_area_weighted_centroid`, since no reference exposes a single
      relaxation step.
    - Not needed after all: `smoothing.equalize_triangle_areas`, which this item named as the
      obvious next step.
    - **The parity test that covers this group could not see the change at all, which is the part
      worth carrying forward.** `test_remesh_edge_concentration` ran on `icosphere(3)` alone, where
      an already-uniform sampling makes the area weights equal the uniform ones — both arms measured
      *bit-identically* (CV 0.0419, 5 120 faces, 0.00287 from meshlib). A library comparison on a
      uniform fixture cannot see a stage whose whole job is anisotropy. It is now parametrized over
      the uniform sphere **and** a graded patch, and tightened on measured margins: mean-within-20 %
      → 5 %, in-band `[0.5, 1.6]x` at 80 % → `[0.7, 1.4]x` at 95 %, and the spread bound against
      meshlib from a single `1.5x` to a per-fixture 0.5x (icosphere, 3.0x margin) and 0.85x (graded,
      1.4x). Disabling one stage at a time now fails it for `split`, `collapse` **and `smooth`**,
      where before only `split` did.
    - **Two defects the recheck turned up, neither caused by this change.** The meshlib reference
      skipped `pack()` — §7.6's documented trap — and so read **12 800 face rows of which 7 696 were
      `[0, 0, 0]` padding** against 5 104 real ones, which inflates the reference spread and
      therefore loosens the bound *in triwarp's favour* (measured small here, CV 0.2525 → 0.2522,
      but the direction is the one nothing notices). And every measured number in that test's
      docstring and in the group's `noparity` reason was stale, the worst by 41x: the exemption
      claimed a graded-patch aspect p99 of **352 against 1.87** where it now reads **8.60** — 352 →
      9.13 from the collapse fold veto and 9.13 → 8.60 from this change.
    - **General lesson: a green parity test is evidence only if its fixture can express the
      difference.** §7.4's vacuity rule is usually read as "check the reference produced a non-empty
      answer"; this is the same rule about the *input*, and the cheap check is to run the new code
      and the old against the parity fixture and diff the statistic. Identical output means the
      test is not covering the change, however green it is.
- **FIXED — the isotropic *collapse* had no fold veto**, where its quadric-decimation sibling did.
  Both kernels deliberately share their core decision helpers precisely because a duplicated
  *decision rule* diverging is the hazard (§2.4); this divergence was exactly that gap, not a
  deliberate variant. Fixed by running the same veto in both. **The adjacency build it needs is
  free** — a rejected collapse removes downstream work that pays for the build, so the fix measured
  as a real quality improvement (aspect p99 roughly halved on a graded patch) at no net cost.
- **Why a graded regular grid was the pathological input the test suite never covered**: it's
  valence-perfect, already Delaunay, and a fixed point of the unweighted Laplacian, so three of the
  five remesh stages are blind to its anisotropy by construction — the test suite had only run the
  remesher on clean closed icospheres, which share the same blind spot.
- **CLOSED — `remesh.claim_collapses` locked its parallel independent set on the raw edge index**,
  which is spatially monotone, so nearly every round had exactly one winner among tens of thousands
  of candidates — silently correct-looking (every test passed, because nothing asserted a collapse
  *count*) but wildly under-parallel. Fixed by hashing the lock key, matching what the sibling
  quadric-decimation path already did. **General lesson: an independent-set kernel needs a test on
  how many winners a round produces, not only on the validity of the ones it commits** — validity
  alone can't see "one winner instead of hundreds".
- **CLOSED — the same hashed key had a second, opposite defect: it wasn't injective**, so two
  distinct candidates could collide onto the same key and both win, corrupting the mesh (overlapping
  committed 1-rings) rather than merely under-parallelizing. Fixed by packing the hash and the raw
  index into separate halves of a 64-bit key, so ties resolve deterministically instead of both
  committing. **General, reusable lesson: a min-key parallel independent set needs its key to be
  both spatially incoherent *and* injective — these are two separate properties, and a correctness
  guard bolted onto only one of several callers is the tell that the key itself, not the caller, is
  wrong.**
- **FIXED — the collapse anti-oscillation test only walked the *removed* endpoint's ring**, missing
  that a free (non-pinned) placement also moves the survivor, so every edge from the survivor's own
  neighbours to the new position went untested. Fixed by walking both rings when the placement
  moves the survivor. Cost: a few percent fewer collapses accepted at a tight target, nothing
  measurable at a loose one.
- **SHIPPED — `_valence_flip_pass` recomputed vertex valence from scratch every pass via a full
  `edges_unique` call, when the edge-topology rebuild it already runs earlier in the same pass had
  already sorted the identical keys.** Reading valence directly off that sorted key buffer instead
  (one small kernel in place of a whole extra sort-and-count chain) measured ~1.4-1.55x on the
  stage, flat across mesh size — the tell that what was removed was host-side launch chain, not
  device work (§16.1's "remesh is launch-bound" again). **Trap in the first draft: the natural
  "runs of exactly two" marker undercounts** — it flags manifold-interior edges only and would
  silently drop every boundary edge from the valence count; the marker that matches `edges_unique`'s
  row set is the any-length run start.
- **`isotropic_remesh` is not byte-gateable** — vertex positions drift by a small amount run to run
  because normal/ring accumulation uses nondeterministic-order atomics, and on some fixtures that
  drift tips a split/collapse decision and changes even the face count. **To gate a change that
  `isotropic_remesh` merely *contains*, gate the exactly-reproducible contained stage directly**
  (e.g. `remesh._valence_flip_pass`, which mutates its face buffer in place deterministically) rather
  than loosening the comparison on the whole function.
- **`quadric_decimate` is one captured graph** — §14.3.
- **Edge-length equilibrium is ~1.0x target only when the target is a "nice" ratio of the input
  edge** (midpoint-split quantization); coarser-than-input targets plateau lower.

- **SHIPPED — `select_independent_degree3` counts its own selection.** `remove_degree3_vertices`
  recovered the count with `reduce.sum(astype(selected, wp.int32))` — a whole-array cast, a
  reduction and a readback per pass — for a number the kernel that built the selection already knew,
  one *conditional* `wp.atomic_add` away (§13.2: contention scales with the rare hits, not with the
  launch). Its six per-pass allocations are hoisted to the pass-0 size and sliced, which is legal
  because the vertex count is invariant (compaction is deferred to the end) and the face count only
  shrinks. Measured **1.109x / 1.083x** on `sphere_med` / `sphere_large`.
  - **The kernel's other caller pays a launch argument for this and is measured neutral**
    (`flatten_degree3_vertices`, 0.991x / 1.004x) — correct per §2.4: one decision rule, one kernel.
  - **Hoisting scratch is not free when the loop usually runs once.** That caller's two ping-pong
    position buffers, allocated upfront, cost **0.948-0.968x**; allocated lazily they cost nothing.
    The common mesh has no two adjacent candidates and runs a single pass, so the second buffer was
    pure overhead. Hoist against the *expected* pass count, not the maximum.
- **SHIPPED — `intrinsic_delaunay`'s four per-iteration resets ride in the kernels above them.**
  `face_claim.fill_(INT32_MAX)`, `remap.fill_(INT32_MAX)`, `no_remap.zero_()` and `count.zero_()`
  were whole device passes over buffers that scale with the mesh, each sitting immediately next to a
  launch at exactly the right `dim`; they are now stores in `intrinsic_delaunay_candidates` and
  `claim_intrinsic_flips` (before its early return). Measured **1.029-1.038x** standalone.
  **The pattern to look for is specific**: a whole-buffer `fill_` / `zero_` whose length equals the
  `dim` of an *adjacent* launch that does not read it. `sample._dart_throw_blue_noise` carries a
  written decline of the opposite shape and is right — there the buffers are cell-sized while the
  launch is over a shrinking alive count, so there is no same-`dim` launch to fold into.

- **SHIPPED — `creation.parametric_surface`'s lattice runs on the device above 9 216 samples:
  61x at the benchmark's top resolution, and a *size dispatch* because the device loses below it.**
  A host-side NumPy audit priced every public call's NumPy share (0.6-6 % for nine of eleven probed,
  flat or falling — §9 declines) and this was the outlier: **29-42 % and not falling**, 10.5 ms of a
  24.8 ms call at 512². `_parametric_lattice` is `meshgrid` + ~10 `np.where` passes + `np.unique` +
  a face assembly over an `n_u * n_v` lattice — a closed-form parallel map with no sequential
  dependence, which is the exception §3.8 already names and which `grid` and `icosphere` were
  converted under.
  - **Measured against a detached baseline worktree** (§15.6), `boy` / `dini` / `klein` / `mobius`
    at the benchmark's own 40 / 160 / 640: **1.05-1.22x / 2.12-2.34x / 50-61x**. At 640 that is
    42.7 → 0.70 ms.
  - **The device path has a flat ~0.66 ms floor and is a 2.4x LOSS below it** — 0.42x at 32², 0.65x
    at 64², 1.01x at 96², 1.55x at 128². The floor is `unique_1d`'s readback plus `flatnonzero`'s
    plus eight launches; the host path is quadratic in the resolution. So the shipped form is
    `_PARAMETRIC_LATTICE_DEVICE_FROM = 9216` with **both** implementations kept, the same shape as
    §14.7's `_DOWNSAMPLE_DOUBLING_FROM`. **Every resolution the suite and the fixtures use is below
    the gate**, so the device path is unreachable in an ordinary test run — which is why
    `test_parametric_lattice_paths_agree` monkeypatches the threshold both ways and compares
    bit-for-bit, over six surfaces chosen to straddle the gluing rules.
  - **Gate: 83 device-path cases byte-identical to the baseline** (16 surfaces x 5 resolution pairs,
    plus `super_ellipsoid` / `super_toroid` / `random_hills`), and 115 host-path cases likewise.
  - **The one subtle part is the pole anchor, and it is statically derivable — but prove it, do not
    reason it.** The host form collapses a pole row to `(i_canonical[on_row][0],
    j_canonical[on_row][0])`, the first surviving sample in C order, which reads as data-dependent.
    A probe over all 16 specs x 5 resolution pairs shows it is always `(0, 0)` for a `j == 0` or
    `i == 0` pole, `(0, n_v - 1)` for `j == n_v - 1` and `(n_u - 1, 0)` for `i == n_u - 1`, so the
    kernel needs no anchor arguments at all. Two orderings in the same function are equally easy to
    get wrong and are called out at the kernel: the u-seam re-canonicalisation after a v-twist is
    **unmasked**, and all four pole masks are snapshots of the post-wrap state taken *before* any
    collapse.
  - `super_ellipsoid`, `super_toroid` and `random_hills` share the lattice and move with it.

- **SHIPPED — `bounds.oriented_bounding_box`'s refinement selects its chains on the device:
  1.14-1.51x.** The trust-region loop scored its candidates on the device and then read the whole
  extent table back **every round** to take a per-chain argmin in numpy and refresh an improved
  chain with a 36-byte `wp.copy`. Counted at the default `refine_iterations=8`: 18 launches, **38
  copies, 10 readbacks**, of which 4.5 copies and 1 readback per round were that host loop.
  `kernels/bounds.oriented_box_select_chains` does the argmin and the update in place, one thread
  per chain, and the refinement reads back once at the end.
  - Measured interleaved against a detached baseline, clouds of 2 k to 2 M points: **1.51 / 1.46 /
    1.32 / 1.14x** at the default, and 1.15-1.58x for the `surface_area` and `diagonal` objectives.
    The win falls as the cloud grows because the extents kernel comes to dominate, but the
    *absolute* saving is a flat ~0.55-0.72 ms at every size.
  - **`refine_iterations=0` is the control and reads 0.98-1.00x at every size** — that path was not
    touched, which is what makes the rest of the column trustworthy.
  - The objective moved into the kernel as a warp-uniform int selector (§2.7), so one kernel serves
    all three. `chains` and `chain_state` are in-place loop state and are allowlisted for check 13
    rather than given an `out_` prefix (§2.1).
  - **The NumPy in this function was never the cost, which is the general lesson.** It profiled at
    13-18 % — the highest in the tree after `parametric_surface` — but that NumPy is over
    `rotations`-sized candidate tables and is microseconds; the 35 `wp.copy` calls above it were
    0.52 ms. **Read what the host time actually is before attributing it to NumPy.**

- **SHIPPED — `cone` and `cylinder` are closed-form kernels, not `revolve` compositions: 2.43-2.60x
  at every section count.** Both built a three- or four-point NumPy profile, **uploaded** it, and
  handed it to `revolve` — which **read it straight back**, decided the layout on the host, uploaded
  four more tables (`column`, `offsets`, `on_axis`, `keep`) and ran two launches. The result was a
  flat host floor independent of the section count: `cone` sat 307x behind open3d at 32 sections and
  4.2x at 4 096, triwarp flat while the reference scaled. `kernels/creation`'s `cone_mesh` /
  `cylinder_mesh` write the whole solid from six scalars, one thread per wedge — no upload, no
  readback, one launch — taking 0.304 ms to **0.091** (3.36x, measured 100 calls between two syncs).
  - **The shared piece is a `@wp.func`, not a copy.** `revolution_point` carries one profile point
    to one slice and is called by both new kernels *and* by `revolve_vertices`, so the three cannot
    drift in the last bit (§2.4); the span is passed as the same `wp.float32(2 pi)` constant for the
    same reason. That is what makes the result byte-identical rather than merely close.
  - **Reuse the reference implementation's *filter*, do not re-derive it.** A first version emitted
    a fixed face layout and differed from `revolve` at `sections` 1 and 2, where the triangles
    touching the axis collapse — cone: 2 faces against 0 at one section, 4 against 2 at two. The
    wrappers now call `revolve`'s own `_revolve_kept_template` on their own tiny profile (host
    arithmetic on 3-4 points, no upload) and pass its verdict as two flags. That also inherits the
    *absolute* area tolerance, so a cone under ~3e-4 radius drops its caps in both engines.
    **The general move: when specialising a general routine, call its predicate rather than
    reimplementing it — the specialisation is the layout, not the rule.**
  - Gate: byte-identical vertices *and* faces across 69 cases — sections 1/2/3/4/5/7/8/32/33/512
    and the default, three radius-height pairs, plus the `transform` and `segment` paths.
  - **The idea generalizes, and it needs one kernel rather than one per shape.** `annulus`,
    `torus`, `uv_sphere` and `capsule` all revolve a profile whose layout is *regular* — every
    column a full ring except an optional on-axis point at one or both ends — so
    `kernels/creation.revolve_uniform` addresses the slots and the face blocks arithmetically from
    the profile plus five scalars: one launch, no readback, no layout tables. Measured (100 calls
    between two syncs, min of 7): **annulus 2.12x / 2.08x**, **torus 1.99x / 1.35x**, **uv_sphere
    1.93x / 1.35x**, **capsule 1.90x / 1.34x** at 32 / 512 sections, with cone and cylinder on the
    same path at **3.2-3.4x**. The four profile-driven shapes fall off at 512 because their
    *profile* has ~512 points, so the host profile build and the degenerate-area filter (both
    `O(P)`) become the floor; cone and cylinder hold 3.2x because their profiles are three and four
    points whatever the section count.
  - **The fast path is *checked against the general engine*, not asserted.** `_revolve_regular`
    asks `_revolve_kept_template` whether the surviving triangles are exactly the regular pattern
    and returns `None` otherwise — an interior on-axis column, a duplicated interior column, or
    geometry small enough for the absolute area tolerance to bite in the middle. **The fallback is
    live**, not defensive: 10 of 27 probed configurations take it. That is the shape to copy when
    adding a fast path to a general routine — let the general one adjudicate, and measure that both
    arms are reached.
  - Gate: byte-identical across **115 cases** (four shapes x eleven section counts x radii, plus
    minor-section sweeps and tiny-radius cases), plus
    `tests/test_creation.py::test_solids_of_revolution_agree_with_the_general_engine`, which
    monkeypatches the fast path off and compares bit-for-bit in one process over 12 configurations
    chosen to straddle the gate — with an `expect_faces` flag so a degenerate case cannot pass by
    both sides being empty.
- **SHIPPED — `bounds.enclosing_diagonal` reduces both clouds into one buffer: 1.66-1.69x.** It
  computed two boxes — two allocations, two launches, **two readbacks** — and unioned them on the
  host. `minmax_vec3_chunked` accumulates with `wp.atomic_min` into a buffer the *caller* seeds, so
  a second launch over a second cloud continues the same reduction: one buffer, one readback, and
  `aabb_union` leaves the path. The all-empty case still returns `inf` with no branch, because the
  seeded `inf` survives. `aabb` itself is untouched and is *at* the floor — one launch plus one
  readback, and it returns host `wp.vec3`s.
- **SHIPPED — the producer-then-reduce fusion, tree-wide scan and three conversions.** An `ast`
  scan of `triwarp/*.py` for "a `wp.map` / `wp.launch` writes a buffer, a `tw.reduce.*` within a few
  statements reduces it to a scalar" finds **33 sites**; each carries one allocation, one launch and
  ~11 µs of `wp.map` resolution more than it needs. Converted where the reduce is actually a share
  of the call: `polyline_length` **2.42-2.47x** (and **6.32-6.61x** closed), `polyline_centroid`
  **3.14-3.25x**, `polyline_radius` **1.44-1.46x** through the centroid it calls, `icp` with a
  `max_distance` **1.10x** (its guard 1.43x).
  - **Three mechanisms, and the second is the one that is easy to miss.** Fold the reduction into
    its producer (`polyline_weighted_midpoint_sums` emits all four numbers a weighted centroid needs
    in one buffer, replacing a two-output map, two reductions and **two** readbacks). Express a
    `closed` polyline by **wrapping the index** rather than by `polyline_close`'s whole-buffer copy
    — that is what takes the closed form from 1.5x to 6.3x, because once the reduction was fused the
    copy *was* the call. And where the readback must stay (a Python loop cannot branch on a device
    value), drop the reduction around it: `icp`'s kernel writes the weights the fit needs *and*
    their sum.
  - **The summation order changes, and the tree is the more accurate arm.** A lane-strided fold plus
    `wp.tile_sum` is not the sequential `float32` accumulation `reduce.sum` performs below `TILE_1D`
    elements. Against a `float64` oracle: relative error **3.18e-09 / 3.75e-08 / 5.01e-08 /
    2.19e-07** at n = 50 / 1 000 / 100 000 / 1 000 000, against a sequential `float32` sum's
    9.04e-08 / 3.68e-08 / **3.07e-06** / **3.62e-06**. A tree halves the depth over which rounding
    compounds, so "this changes the last bits" is not the same as "this is less accurate" — say
    which, with a number.
  - **It does cost a reference comparison, and that is the trade to state explicitly.**
    `tests/test_polyline.py::test_polyline_length_matches_meshlib` asserted bit-identical equality
    with `mm.calcLength`'s sequential sum and now asserts `rel=1e-6`. Its docstring had predicted
    the failure (*"a summation-order change would show here first"*) and it caught it on the first
    run. **A reduction's summation order is part of some functions' contract; check for an exact
    comparison before fusing one, and if the trade is taken, rewrite the test's reasoning rather
    than just loosening its number.**
  - **Declined, with reasons**: the `heat` family (five sites), `energies`, `validation`, `repair`,
    `voxels`, `reconstruction` and `sample` all spend 1-4 % of the call in the reduce — e.g.
    `energies.edge_length_loss` is 0.722 ms of which the pair is ~25 µs, the rest being
    `edges_unique_length`. `array.allclose` *is* the pattern end to end but is generic over three
    dtypes, so fusing costs a dtype table for a primitive with one in-repo caller. And moving
    `icp`'s guard after the Procrustes fit — to read the weight sum `ACC_W_SUM` already holds — is
    **not behaviour preserving**: the guard breaks *before* assigning the returned transform.
  - **A scan that keys on line proximity over-counts.** `polyline_radius` looked like three
    reductions over one array and is three *mutually exclusive branches*. Read the function before
    believing the hit.
- **SHIPPED — `edges_unique` is a host-bound substrate under 57 call sites in 18 modules, and it
  was paying for two host readbacks it did not need.** Stage-timed on an RTX 5090, Warp 1.17, 200
  calls between two syncs, min of 7: the whole call measured **0.471 / 0.504 / 0.500 ms** at
  320 / 5 120 / 81 920 faces — *flat across a 256x face range*, which by §16.1's own method makes it
  100 % host. The stages summed to 90 % of it: `unique_1d(return_inverse=True)` 0.26 (53 %),
  `hash_indices_rows(validate=True)` 0.107 (21 %, of which the range check alone is **0.084**),
  `faces_to_edges` 0.026, `first_occurrence_indices` 0.027, `array.gather` 0.031. Three changes,
  and the census to re-derive them is `plans/`-local (an AST scan plus a monkeypatched `wp.launch`
  census — §16.0's "take the census from the runtime").
  - **The deduplicated rows are already in the keys.** `grouping.hash_indices_rows` packs a sorted
    `(lo, hi)` row as `lo + hi * base`, and `array.unpack_edge_key` inverts it exactly — its own
    docstring had named that packing as the shared convention since it was written. So
    `first_occurrence_indices` plus `array.gather` became one `edges_from_keys` launch, dropping a
    scatter, a gather and the first-occurrence buffer. **Row order is unchanged and the result is
    bit-identical**: both forms index by the same unique id, and `edges_sorted` is min-first, so the
    unpacked `(key % base, key // base)` *is* the row the gather used to fetch.
  - **`validate` is now a keyword, default `True`, and 27 internal call sites pass `False`.** The
    range check is a `reduce.minmax` plus a host readback. Keeping it on at the public boundary and
    off inside the library is what `_device.require_valid_faces`' docstring already prescribed —
    it is "meant for the small number of public entry points that are the real trust boundary",
    with every downstream helper "expected to trust the connectivity it was handed the same way
    every other per-face kernel wrapper in this package already does". `edges_unique` was the one
    downstream helper paying for a check its own neighbours skip: `laplacian.laplacian_entries`,
    one of its callers, hands the same `faces` straight to kernels that index `vertices[faces[...]]`
    unchecked, so the raise was protecting nothing the rest of the call did not already assume.
    `validation.is_watertight` makes the point sharper still — it *already* ran one stage checked
    (`is_edge_manifold`) and the next unchecked (`face_adjacency`, whose `_edge_groups` passes
    `validate=n_vertices is None`), so the guard was not a contract, only an inconsistency. Measured **0.471 → 0.341 ms**, flat at
    every size (**1.38x**), and every one of those 27 sites gets it.
  - **`index_bound(X)` followed by a validating hash of `X` reduces the same array twice**, and only
    the negative half of the second check can ever fire — the bound was *derived* from that array.
    `array.index_bound` grew a `require_non_negative=` that takes both ends out of the one
    `reduce.minmax` (free, per `require_valid_faces`' own measurement), so the pairing is now one
    readback with the same guarantees. Applied at `edges_unique`, both `validation` edge-manifold
    entry points, `holes._EdgeTable` and `validation.edge_winding_consistent_mask`.
  - **`grouping.unique_1d` made two byte-identical copies of its sorted keys.** On the
    `sort_dtype == original_dtype` branch — which every `uint64` edge- or row-key call takes —
    `unique_values` and the inverse pass's `sorted_dense` are the same `keys_buf` prefix copied out
    twice. The second allocation and copy are gone.
  - **Deterministic counts, which is the part a noisy box cannot argue with** (§15.6):
    `edges_unique` 8 launches / 15 allocations → **7 / 14**, `edges_unique_length` 9 / 16 → 8 / 15,
    `is_watertight` 30 / 43 → 28 / 43, `vertex_manifold_mask` 17 / 24 → 16 / 24, `cotmatrix`
    3 / 12 → 3 / 11, and the readbacks those figures do *not* show are the larger half.
  - **End to end, against a detached baseline worktree** (§15.6; min-of-9 over 200 calls between
    two syncs, one arm per process on a quiet box, 320 / 5 120 / 81 920 faces):

    | call | ratio |
    |---|---|
    | `energies.edge_length_loss` | **1.49 / 1.46 / 1.44x** |
    | `validation.face_watertight_mask` | 1.19 / 1.20 / 1.14x |
    | `validation.edge_manifold_mask` | 1.19 / 1.18 / 1.13x |
    | `validation.is_edge_manifold` | 1.16 / 1.20 / 1.10x |
    | `edges.edges_unique` (**public default**, `validate=True`) | 1.16 / 1.11 / 1.12x |
    | `edges.edges_unique_length` | 1.16 / 1.14 / 1.12x |
    | `validation.vertex_manifold_mask` | 1.14 / 1.13 / 1.10x |
    | `validation.is_watertight` | 1.15 / 1.11 / 1.06x |
    | `edges.mean_unique_edge_length` | 1.11 / 1.11 / 1.09x |
    | `heat.heat_geodesic` | 1.03 / 1.03 / 1.02x |
    | `laplacian.cotmatrix`, `adjacency.face_adjacency` | 0.98-1.06x (controls — neither reaches `edges_unique`) |

    **Quote the two `edges_unique` numbers separately or the table reads wrong**: 1.09-1.16x is the
    *public* call, which still validates; the 1.38x above is the `validate=False` path all 27
    internal callers now take, and the A/B harness deliberately exercises the default so the
    published default is what gets reported. `edge_length_loss` is the largest because it was
    paying *both* readbacks — it supplied no `n_vertices`, so it inferred the bound and then had
    the packing re-check it.
  - **Five of ten `graph.connected_component_labels_from_edges` call sites still validated**, though
    that function's own warning names "an adjacency list from `face_adjacency`" as the case to skip
    it. All five now pass `validate=False` with the reason at the site. A grep for `validate=False`
    on one line **undercounts** — half the converted sites already wrapped across lines, which is
    how the census first read 1 of 10 rather than 5 of 10.

- **SHIPPED — a two-column index row needs no bound at all, so the reduction that infers one is
  removable outright.** `grouping.hash_indices_rows` packs a row as `sum(digit[i] * radix ** i)`,
  so *any* radix above every entry is injective — and **every `int32` reinterpreted as `uint32` is
  below `2 ** 32`**, which two columns always fit inside a `uint64`. The packing is also monotone
  lexicographic in `(row[1], row[0])` for every valid radix, so a wider one leaves
  `unique_1d`'s sorted row order untouched. `constants.INDEX_RADIX_PAIR` is that radix, and
  `validate=False` with no `max_index` now means "use it" for a row of at most two columns (a
  wider row still raises: its radix has to keep `radix ** w` inside a `uint64`).

  Measured at 61 440 rows: `hash_indices_rows` **112.3 → 24.1 µs (4.7x)**, with byte-identical
  unique rows in byte-identical order (30 720 of them). End to end on the sites that took it:
  `halfedge.halfedge_twins` **1.29x** (308.1 → 238.8 µs), `edges.edges_unique(validate=False)`
  **1.28x**, `boundary.ears` **1.26x**.

  **What it does *not* reach is the more useful half of the finding.** Most callers' reduction is
  not only the radix — it is also the negative-index guard, and `hash_indices_rows`' own docstring
  already said so at four sites. So `unique_rows`, `group_int_rows` and `edges_unique`'s *public*
  default keep theirs and are unchanged (1.00-1.02x); only a caller whose count reaches nothing but
  the radix converts. **Before assuming a bound is free to widen, ask what else it is: in
  `validation.is_vertex_manifold` it is the *length* of the per-vertex mask** (unreferenced
  vertices are deliberately `False`), so passing `vertices.shape[0]` there would change the answer
  on a mesh with trailing spares — that sub-item was planned, measured against the semantics and
  refuted.

- **SHIPPED — `measures.moments` reduces inside its integrand kernel: 3.00x.** It launched one
  kernel writing four `(n_faces,)` buffers (80 bytes a face, 6.5 MB at 81 920) and then four
  `wp.utils.array_sum` calls, each with its own host readback, to extract ten scalars. The four
  reductions were **160.6 µs of a 260.0 µs call**. `kernels.measures.moment_integrals` now folds
  them: lane-strided over its block's chunk, ten `wp.tile_sum` trees, ten atomics per block, one
  `(10,)` accumulator and one readback. Measured 269.6 → 89.9 µs at 20 480 faces, **2.6-3.0x from
  80 to 327 680 faces**, and **1.6-2.4x on the CPU device** as well. Accuracy improved: against a
  sorted `float64` serial sum the tree is **6.4e-16** relative at 81 920 faces.
    - **Its chunk width is a launch argument, not `ITEMS_PER_BLOCK_1D`, and that is the
      generalisable part.** The per-face integrand is ~80 `float64` flops — far heavier than the
      outer product in `points.centered_covariance` — so the family's single 1 024-element chunk
      starves the device: 434.3 µs at 1 310 720 faces against 296.1 for a 512-element chunk, while
      at 1 280 faces the ordering reverses (14.2 against 54.9). `measures._moment_chunk_faces`
      doubles the chunk from one tile until the grid is no wider than 1 280 blocks. **A runtime
      width measured identical to a `wp.constant` one**, so nothing is lost by choosing it on the
      host — which is worth knowing before baking any chunk constant into a kernel.
    - `kernels/reduce.py`'s note about this skeleton being hand-written at three sites is updated:
      it is five now, and the two added are the reason it is still not factored — the *bodies*
      differ in more than the component count.

- **SHIPPED — a function returning several small device values reads them back once.**
  `points.principal_axes` **1.48x** (252.9 → 170.5 µs) and `fit_plane` **1.27x**: three and two
  `dim=1` outputs, written by one kernel, were read with three and two separate `.list()` calls —
  95.6 µs against 22.7 for a single `.numpy()` of the packed buffer (§3.10).
  `points.statistical_outlier_mask` is the same finding one level up at **1.40x** (378.5 → 270.8):
  two of its three reductions ran over one pass's outputs, and the `wp.map` that built a mask
  purely to be counted went with them.

- **SHIPPED — `creation.sphere_cap` is a kernel: flat at ~52 µs against a quadratic host loop.**
  The last template still assembled on the host, and the one §3.8's exception covers — a closed-form
  parallel map with no sequential dependence. Measured 71 / 216 / 784 / 1 771 / 4 531 / 12 702 µs at
  `subdivisions` 0 / 3 / 5 / 6 / 7 / 8, against **51.7 / 49.9 / 51.8 / 51.9 / 51.4 / 55.4** —
  **1.4x to 229x**, and the device path wins at *every* size, so unlike `_parametric_lattice` it
  needs no size gate. Gate: **byte-identical vertices and faces across 128 cases** (both devices,
  subdivisions 0-7, four angles, two radii).
    - The one subtlety is the inverse: a thread recovers its ring from its vertex index by solving
      `3 r^2 - 3 r + 1 <= v`, and the ring *starts* are `1 + 3 r (r - 1)` **except ring 0**, which
      is the lone apex at slot 0 rather than that formula's 1. Getting that wrong wound every
      apex triangle around its neighbour instead — caught by the byte-identity gate and by nothing
      else, since the mesh stayed watertight, correctly wound and the right size.

### 16.5 `array`, `graph`, `polyline`, `intersection`

- **`kernels/array.binary_search_index` is `searchsorted(side="right")` and returns `slot + 1` on an
  exact hit.** Picking the wrong one of the three searches is silent and systematic, not a crash — a
  wrong-search bug on an argsort payload returned a *valid* index of the wrong element, so seeds
  landed on neighbouring elements and a flood fill returned far more than it should have, plausibly.
  **When a binary search feeds an argsort payload, verify one lookup by hand against NumPy.** And:
  **when a fix doesn't change a symptom, suspect two causes rather than concluding the fix was
  wrong** — a second, unrelated bug in the same function produced the identical symptom.
- **FIXED — `side="right"` is poisoned by `NaN`, so `unique_1d(return_inverse=True)` was silently
  wrong on any float array containing one.** Every comparison against `NaN` is false, and Warp's
  float radix sort puts `NaN` last, so `side="right"`'s search steers *into* that tail and reports
  the last slot for every finite value. Fixed by searching `side="left"` instead and falling back to
  the last slot only on the `NaN` branch. **Found by registering the float overloads** (§2.5) —
  writing that registration meant asking what dtypes the wrapper's dispatch actually reaches, and a
  dtype nothing had ever knowingly launched was a dtype nothing had ever checked.
- **SHIPPED — `array.isin` spent about half of every call inferring the value span; the guarantee
  that removes it is a caller-supplied bound (`max_index`), not `assume_unique`.**
  `assume_unique` buys nothing here (NumPy's version only wins because its sort path dedups both
  sides first; neither triwarp strategy dedups anything). Threading a `max_index` bound from callers
  that already know it is a straightforward ~2.7-2.9x win. **The bound must be range-guarded inside
  the kernel** — an over-tight bound is an out-of-bounds gather, i.e. §12.1's host-heap corruption on
  CPU, not merely a wrong answer.
- **SHIPPED — `_unique_hash` ran two whole-table bookkeeping passes that were ~40% of `unique_1d`**,
  removable because a later kernel already computes what they existed to recover (an occupancy flag,
  an identity permutation) as a side effect. ~1.4x on `unique_1d`; smaller on `unique_rows`, which
  still needs its own bitcast pass.
- **`flatnonzero` uses an inclusive scan + single tail read** — `scatter_index_where` expects the
  inclusive scan and writes at `inclusive[i]-1`.
- **SHIPPED — `intersection._link_segments` is vectorized NumPy pointer doubling, not a per-segment
  Python walk**, which had been 84-88% of the whole call on many-curve fields. Two passes of Wyllie
  pointer doubling (extended to cover open chains, not just cycles) plus a lookup-table successor
  replace an argsort+searchsorted chain; both bit-identical to the old output at every size tested.
  **The device port was deliberately not taken** — profiling the vectorized version showed most of
  the remaining cost was already in one non-portable sort, leaving too little headroom to justify a
  device rewrite, and a device version would have regressed the common single-contour case anyway.
- **`polyline_downsample` is pointer-doubled above 8 192 points on CUDA only** — §14.7.

- **SHIPPED — `marching_triangles` densifies the *level set's* edges, not the mesh's: 1.39-2.54x.**
  Stage-timed on an 81 920-face sphere, `edges.edges_unique_inverse` was **0.536 ms of 1.318
  (41 %)**, and the whole call was *flat* between `sphere_small` and `sphere_med` — a mesh-sized
  pass sitting under a level-set-sized answer. The unique edge *id* was only ever a matching key:
  the two faces sharing a crossing must agree on it, and nothing downstream needs it dense or
  small. **The sorted vertex pair is already such a key**, so the segment kernel packs it itself
  (`kernels/intersection.edge_key`) and the densification `_link_segments` genuinely needs runs
  there over the crossing endpoints alone — `2n` values instead of 122 880 mesh edges.

  | 327 680-face sphere | curve points | before | after | |
  |---|---|---|---|---|
  | plane | 1 534 | 1.784 ms | 0.702 | **2.54x** |
  | wave12 | 11 924 | 2.503 | 1.798 | **1.39x** |
  | wave40 | 39 772 | 5.058 | 5.071 | 1.00x |

  - **The trade has a crossover, and naming it is the transferable part.** The host `np.unique`
    grows with the *level set* where the device pass grew with the *mesh*, so a contour touching a
    large fraction of the edges gives the win back. It levels out rather than losing, so no gate is
    needed — but the same swap on a function whose answer is mesh-sized would be a loss.
  - **DECLINED — replacing the densification with a sorted join, and the *reason* is the reusable
    part.** `argsort` the start keys, `searchsorted` the end keys into them: no densification, no
    `owner` table. Interleaved in one process on the same captured input, both producing the
    identical `successor`, it is **1.14-1.91x slower** — 0.096 / 1.296 / 2.899 ms against
    0.084 / 0.730 / 1.514 at 1 534 / 11 924 / 39 772 segments.

    It loses because it needs **two sort-class passes where densifying needs one**, not because of
    the extra gathers (0.09 ms of 2.80). Phase-timed at 39 772 segments: `np.searchsorted(n into
    n)` is **1.83 ms** on its own, about what `np.unique` over `2n` costs (1.34), because a binary
    search is `n log n` *dependent, cache-missing* probes into a 318 kB array rather than a
    streaming sort — the same search into a cache-resident 1 000-entry array is **7.6x** faster at
    the identical query count. And `np.argsort` is ~5.5x `np.sort` (0.905 against 0.164 at `n`),
    since it permutes indices through indirect comparisons; the join needs it specifically to
    recover *which* segment won.

    **The general lesson: densifying is not overhead you pay to enable a lookup table — the dense
    labels make the join itself O(1) per element (0.021 ms), which buys more than the search they
    avoid.** `np.searchsorted` reads as "cheap, it is just a binary search" and at these sizes it
    is a sort-class cost.
  - Gate: byte-identical points, curve lengths and closed flags across 30 (mesh x field x isovalue)
    cases. `n_vertices` keeps a real meaning — it is the base the vertex pair packs against — so the
    public signature is unchanged.
  - **The general shape to look for**: a helper that densifies or indexes *the whole mesh* feeding
    a consumer whose answer is a small subset of it. The tell is a call whose cost is flat in the
    mesh size while the answer is not.

- **SHIPPED — `graph.successor_cycles` finds its node set with a mask, not `unique_1d`: 1.05x on
  it, and the mechanism is worth more than the row.** It took the distinct endpoints with
  `unique_1d(edges.flatten())`; a zeroed `node_count` mask, a `mark_membership_mask` scatter and
  `flatnonzero` return the *same sorted distinct values* for a fraction of the work, because the
  range check the function already ran guarantees every endpoint indexes the mask. Measured
  **122 against 176 µs, flat in `node_count` from 2 562 to 1 000 000**, byte-identical at every
  size. **`unique_1d` is a hash table, a compaction, a radix sort and two readbacks; where the
  values are known-bounded indices, a mask is the cheaper spelling of the same answer.**

- **`boundary.boundary_loops` is ~1.95 ms and flat, and this pass took only 1.05x of it.** Recorded
  because the attribution is the useful part: on a one-loop hemisphere it issues **36 launches, 48
  `wp.empty`, 11 `wp.zeros`, 16 `wp.copy` and 7 readbacks, identical at 656 / 10 304 / 41 088
  faces**. The largest single piece is `successor_cycles`' pointer-doubling loop — `ceil(log2 n)`
  launches of one kernel with swapped buffers, 14 of them here, ~257 µs — and **both mechanisms
  for removing it are already refuted**: a CUDA graph cannot be reused across calls because the
  round count varies, so it is §14.3's record-and-replay-once at 0.84x, and a single persistent
  block is the shape §14.9 refuted for `graph.bfs`. What did convert: the node set above, and
  `_needs_unoriented_boundary_walk`'s second launch, which scanned the whole vertex count to reduce
  a degree table to two bits and is now stamped by the value each `wp.atomic_add` returns (only a
  boundary vertex ever has a non-zero degree). Gate: **32 cases byte-identical on both devices**,
  including `mobius` — the fixture that actually takes the non-orientable branch the probe guards,
  without which the gate is vacuous (§7.4).

### 16.6 `proximity`, `metrics`, `neighbors`

- **`query_hashgrid_nearest`'s cost is *cubic* in how far `initial_radius` under-estimates the actual
  answer distance**, because that one scalar sets both the hash-grid cell width and the search seed
  — a scan at radius `r` walks `O((r/cell)^3)` cells. The default estimator inverts the *target*
  cloud's density, which is wrong whenever the two clouds are meaningfully displaced, and the cost
  is genuinely in the cell walk, not the linear-scan fallback (verified by instrumenting the branch
  directly). Several probe-based fixes were tried and withdrawn because the fix itself cost as much
  as the walk it was meant to shrink.
- **SHIPPED — seed the *backward* search from the *forward* half's own answer.** Both directions of
  a symmetric distance query share one distance scale, so the forward pass's own answer is already a
  good `initial_radius` estimate for the backward pass — no probe, no subsample needed. This is a
  real win (1.2-3.2x depending on size) on `chamfer_points_to_points` / `chamfer_points_to_mesh`.
  **`backend="bvh"` at these call sites is a loss at every size tested — do not re-propose it.**
- **REFUTED — seeding the *forward* pass from a query-prefix probe cannot be safely gated on size.**
  Sweeping only mesh size made the idea look like a clean gateable cliff, but holding size fixed and
  varying only how far the two clouds sit apart shows the *actual* variable is the ratio of answer
  distance to point spacing, not size — and it isn't even monotonic in that ratio. A gate built on
  the wrong variable would ship a real regression on a widely-separated pair that happened to still
  be "large" by the sweep's own size axis. **General lesson: a sweep across one axis (size) can look
  like it identifies the real threshold while actually just correlating with the true variable on
  that one benchmark's fixture — vary the fixture's own free parameters before trusting a gate.**
- **REFUTED — there is no cheap automatic cell-width trigger for `neighbors._knn_cell_size`, and
  both halves of that now have numbers rather than an argument.** The full table, the probe that
  works and the two reasons it cannot be made unconditional are written at the function itself; the
  short form:
    - **The missing term is the query displacement, and a cheap probe does recover it** — brute-force
      256 queries against a 1 024-point subsample of the cloud, subtract *that subsample's* own
      spacing (skip this and the estimate inherits the subsample's sparsity and over-widens 3-5x),
      add `knn_initial_radius`. Worth **2.0-11.4x** through the middle band (queries displaced
      0.02-0.10 of the diagonal), near-neutral where nothing was wrong, and it beats
      `backend="bvh"` there too.
    - **It is declined on the probe's cost, not its accuracy.** The probe is **0.27-0.29 ms** of
      slices, copies, a launch and a reduction, which on its own doubles the 0.285 ms on-surface
      call that is the benchmarked operating point; and at displacement 0.20 the wide cell is a
      1.6-2.2x loss, because abandoning the grid for the linear scan is by then the right
      algorithm. A gate for either would have to decide *without* the probe. **The bar is now a
      number: get the probe under ~0.05 ms and it ships.** Meanwhile the lever is the public
      `initial_radius=`, which `metrics.chamfer_points_to_points` already uses that way.
    - **The block-cooperative lead this item proposed is refuted twice over.** The grid walk cannot
      be lane-split at all (§12.2: `wp.HashGrid` has no per-cell entry point), and the linear-scan
      fallback, which can, is *not* load-imbalanced where it costs: a fallback census over
      displacements 0.00-0.20 gives 0 % of rows falling back at 0.00-0.01, then 22 / 69 / 84 / 92 %
      at 0.02 / 0.05 / 0.10 / 0.20 for k=1 (0 / 8 / 54 / 77 % at k=30), so the expensive regime is
      one where nearly every row takes the scan and the launch is uniformly expensive. The genuinely
      imbalanced band is the same middle band the radius probe already covers more cheaply. The
      deepening ladder is also never deep — **at most 2 attempts** at every displacement probed.
    - Method note worth keeping: `knn_sorted_insert` binary-searches its row's **whole length**, so
      a probe handing it a row wider than `k` gets silent garbage — a 64-wide scratch row at k=30
      reported "1 neighbour found, 100 % fallback" and read exactly like a triwarp defect. The row
      width *is* `k` in production. Confirm a census against an independent oracle (here a
      `cKDTree` k-th distance) before believing it.
- **A documented "16x non-monotonic drop" in `query_nearest` was real but mischaracterized** — it
  only appears at `k >= 8` and is hash-grid-specific (the BVH backend is monotonic across the same
  size sweep). Mechanism: once a row's true k-th distance exceeds the grid's widest search radius,
  that row falls back to an exact O(n) linear scan, and how much of the cloud falls in that tail
  scales with n — so `backend="bvh"` is the workaround at moderate k on a uniform cloud, the opposite
  of what the earlier reading implied.
- **`k`, not `n`, is what's still slow in the k-NN path** — insertion cost is super-linear in k
  because the candidate row lives in global memory and both the shift-insert and the reset touch it
  in full on every deepening attempt. This is what `points.statistical_outlier_mask` and
  `outlier_probability` (both k~30) pay for, and is exactly what the register-row rewrite (§2.9)
  targets.
- **SHIPPED — `mesh_to_mesh_distance` now reads the `wp.Mesh`'s own BVH** (`wp.mesh_get_bvh`) instead
  of building a second one over per-face AABBs — bit-identical output, and a real win that grows
  with mesh size since the removed build's share of the call grows with it.
- **CLOSED — `mesh_to_mesh_distance`'s cost was overwhelmingly its own distance *bound*, and the
  bound doesn't need every vertex of the query mesh.** Stage attribution on the largest scan mesh
  found the vertex query computing that bound was over 90% of the call, dwarfing the actual BVH
  traversal — millions of closest-point queries paying to prune a walk worth under 1% of the total.
  **The bound only seeds a prune limit, so a subsample is exactly as sound and barely loosens it** —
  the nearest approach between two surfaces is not a rare event, so a modest fixed-size stride
  sample recovers a bound within ~1% of the exact one. Shipped as a fixed sample-size stride (no RNG,
  no gather); the resulting distances are bit-identical to the unsampled version, because a looser
  bound only prunes *less*, so the narrow phase still sees the true argmin. The win grows with mesh
  size (flat to ~10x on the largest scan mesh) since what's removed scales with mesh size and the
  rest doesn't.

  **Widening the query box to build this fix worked as a fuzzer and found two real, older bugs
  downstream of the bound — general lesson: an optimization that loosens a bound stresses everything
  that bound feeds.** One was the `tile_bvh_query_aabb` result-buffer overrun (§12.2), which this
  fixture reaches deterministically. The other was CUDA-only: `mesh_to_mesh_distance` returned `inf`
  on inputs where every query face overflowed its candidate cap, because the straggler re-walk pass
  *overwrote* the first pass's already-correct partial answer with a worse one, hitting the same
  "the bound has become the answer" trap the function's `global_best_sq` seeding already guarded
  against one level up. Fixed by keeping the smaller of the two passes' answers rather than
  unconditionally overwriting. **The CPU device, which runs no capped second pass, was the oracle
  that caught this** — another instance of §7.2's two-device rule paying off. A serial uncapped
  straggler pass was also tried as a fix and reverted (a real loss) — §14.2's load imbalance is real
  and the block-cooperative walk earns its place.
- **The remaining cost on the largest scan mesh was mostly *host* time, not device time** — two
  `.numpy()` calls in the function's tail were each copying a whole per-face array to read one
  element, and that share *grows* with the mesh (the opposite of the usual falling-share decline).
  Fixed with `_device.read_scalar`. **General lesson: read the host half of a device profile before
  accepting a device-side attribution, and a `.numpy()[k]` on an array that scales with the mesh is
  the shape to grep for.**
- **DECLINED — `cotmatrix`'s apparent multi-x loss to pytorch3d is a scope mismatch, not a real gap,
  and the rewrite it invited is declined.** `p3d_ops.cot_laplacian` never assembles a real sparse
  matrix — it returns an *uncoalesced* COO tensor with duplicate entries unsummed and no diagonal
  ever written, where `laplacian.cotmatrix` does the full sort/dedup/accumulate into CSR with an
  assembled row sum. Once pytorch3d's own output is coalesced to do the equivalent job, triwarp is
  consistently *ahead*, at every size tested. Since the two already agree once that transform is
  named, this stays a live parity comparison rather than a `noparity` exemption — what's incomparable
  is the *timing*, not the result.
- **`repair.fix_self_intersections(method="local")` was read as the suite's largest single loss for
  several rounds, and it was never a loss — the reference call was a no-op on that fixture**,
  returning its input byte-for-byte while every collision remained. **General lesson: apply the
  same detector to *both* sides' output before comparing costs** — a reference that silently
  declines an input isn't a cheap fixer, and a benchmark assert of "produced some output" can't see
  the difference.
  - **The benchmark now runs on the `tangle` axis instead — a self-intersecting single-component
    torus at two sizes — and the loss is gone rather than exempted.** Both libraries carry a
    guard that asserts they actually mutated the input. There's a real crossover in triwarp's favor
    as the mesh shrinks (a serial C++ fixer wins on small meshes) and near-parity at the large size,
    with a quality caveat running the other way: at the large size triwarp leaves a small residue
    MeshLib clears completely, and `max_iter` is what closes it.
  - **REFUTED — the DP triangulation stage inside it is not this row's cost, and every DP lever is
    capped at a small fraction of the call**, both at the small and large benchmarked size, because
    the rims filled on this fixture are short. Neither a bounded-candidate DP nor a blocked interval
    DP has enough to remove — both are declined for this row, and the blocked DP is also declined
    for the dedicated hole-filling benchmarks, where triwarp already wins against the one reference
    running the same minimum-weight algorithm.
  - **What's left is the smoothing/refinement stage, which dominates the call, and it has no known
    lever yet.** Several plausible levers were tried and declined: graph-capturing the chain (a
    once-through loop, so capture-and-replay-once is a net loss — §14.3), a per-loop persistent
    block, and a forced multigrid hierarchy on its patch solves (declined because the hierarchy's
    setup cost dominates a single patch solve — §16.8).
  - **A separate voxel-based method on the identical input wins outright**, and meshlib genuinely
    repairs there too — the method choice, not this method's implementation, is the available answer
    for a caller today.
- **SHIPPED — the hole-filling DP is launch-bound, not device-bound, at every rim size the benchmark
  reaches**, because it issues one launch per triangulation span and the launch floor dominates
  until the rim gets far larger than any benchmarked case. A wider-block variant recovers some of
  that (roughly 1.1-1.45x depending on rim size), gated on grid shape rather than rim size alone
  since a wide block on a narrow, scattered grid is a loss. **Remaining lever, unbuilt:** a blocked
  interval DP that tiles the recurrence to cut the launch count by an order of magnitude — worth
  building only if a row appears where it would flip a result, since the launch floor here is
  currently nobody's bottleneck.
  - **Being launch-bound makes the launch's *arguments* a lever, and that is where the next 1.11x
    came from.** `dp` and `prev` were launch arguments on a sweep of `max_B - 2` launches — 510 on
    a 512-vertex rim — at ~1.0 µs each (§13.1), and their pointers are invariant across the sweep.
    They now ride in the `HoleFillTables` bundle that was already built once. Measured **1.110x** on
    `fill_min_weight[rim_short]` (10.738 → 9.678 ms), flat within 1 % on `holes_many` /
    `holes_dense`, whose rims are short enough that the sweep is ~30 launches rather than 510. The
    bundle's own docstring had drawn the line at "only `span` and the two in-place DP tables stay as
    arguments, because those are what a launch is actually about" — correct as legibility and worth
    milliseconds here.
  - **REFUTED — transposed mirrors of `dp` / `prev` so the apex loop's second read coalesces.**
    `apex_cost` reads children `(i, k)` and `(k, j)`, and the apex loop varies `k`, which sits in the
    column of one and the row of the other — so with the tiled kernel's lanes striding `k`, one read
    runs at stride `4 * b` (2 048 bytes on a 512-vertex rim, a transaction per lane). Mirroring the
    tables so both reads are stride 1 was built, verified byte-identical, and is **a loss**: 0.917x,
    and against the same struct-resident tables 1.095x where the unmirrored form gives 1.150x. Two
    reasons the arithmetic does not carry. The DP tables are a few megabytes and **L2-resident**, so
    "one transaction per lane" is an L2 hit, not a DRAM fetch — the 32x transaction argument prices
    bandwidth this kernel is not paying. And the DP is launch-bound, so the mirror's two extra
    per-interval stores and two extra launch arguments cost more than the coalescing saves. Do not
    re-propose without a rim whose tables exceed L2.
  - **REFUTED, by a free PTX diff rather than a measurement — collapsing
    `triangle_fill_metric`'s duplicated geometry.** Its default branch forms the same three edge
    differences twice, the same three dot products twice and the same cross product twice, ~38 of
    ~90 flops, in the innermost function of an `O(B^3)` loop — and the duplicates sit in *different*
    basic blocks (`circumcircle_diameter_sq` and `triangle_aspect_ratio` each carry early returns),
    so their removal needs partial-redundancy elimination rather than local CSE and there was no
    reason to assume it. **nvcc already does it.** Hand-fusing the branch and diffing
    `fill_dp_span`'s forward entry in the regenerated `*.sm120.ptx`: `sqrt.rn.f32` 56 → 56,
    `div.rn.f32` 57 → 57, `fma.rn.f32` 227 → 227, `mul.f32` 445 → 445, `add.f32` 47 → 47,
    `sub.f32` 335 → **332**. Three subtractions out of ~1 100 arithmetic instructions. Reverted —
    the fused form is twenty lines where the original is three named calls. **The general move is
    the cheap one: a PTX op-count diff costs no GPU time and no measurement, and it settled this
    before any clock was read.**
- **A couple of small preambles in the hole/loop-stitching path are deliberately left unoptimized** —
  their share of the call falls as the input grows, so they're declines rather than wins waiting to
  happen (§9's "a share that falls as the input grows is a decline").

### 16.7 `sample`

- **`sample_surface_blue_noise` is randomized-priority parallel dart throwing, not Bridson.** Every
  pool point draws a priority from the seed; a point is accepted when no smaller-priority point still
  in play lies within `r`; everything within `r` of an acceptance is discarded; iterate. This
  supersedes an earlier "propose is inherent, already well-tuned" conclusion that was true of the old
  *kernel* and false of the *algorithm* — the old design's cost was many rounds over a wide cell
  shell, the replacement is a narrow cell shell over a handful of rounds. Real win over the old
  design and over pymeshlab/Open3D on spacing/coverage quality too. **General lesson: reach for
  randomized-priority selection whenever a GPU port needs a maximal-packing / MIS-shaped result** —
  the tie-break-free correctness argument (the later of any too-close pair was already discarded) is
  what makes it safe against a stochastic output, and it is the serial algorithm's own distribution.
- **SHIPPED — a per-cell summary prunes both dart kernels' shell scan, real win, byte-identical
  output.** One kernel retires an alive point early if an accepted point already covers it; the
  other rejects a point early if a smaller-priority point already covers it — both via a small
  per-cell summary rather than a full shell re-scan. **This is not the refuted thread-mapping
  inversion (§15.3)** — it removes work whose result was already determined, so the output set is
  identical by construction, not merely faster.
- **The cover pass is load-bearing for termination, not just an optimization** — a wider shell or a
  looser cover radius doesn't merely change the answer, it can stop the loop from converging at all,
  because the accept step won't take a point while a smaller-priority alive point still covers it.
  Its byte-identity gate is therefore cheap insurance worth keeping.
- **CLOSED — the round count was benchmarked, as this item asked, and the premise was wrong: it is
  stable at 4-6 whatever the cloud.** Measured over five mesh shapes (icosphere at two resolutions,
  torus, cylinder, box), a 55x pool-size range (12 633 → 692 820) and a 55x output range (229 →
  12 684): **4-6 rounds in all fifteen cells**, growing logarithmically with the pool exactly as
  randomized-priority maximal-independent-set theory predicts, with pool/output holding at 54-57.
  The "one cloud takes several times as many rounds as another" reading does not reproduce.
    - **Nor is the remainder per-round dispatch.** From `wp.timing_begin` (nothing here
      graph-captures, so §15.10 does not apply): **48 % device** at the small end (1.56 of 3.23 ms,
      52 kernels) and **69 %** at the large (3.81 of 5.56 ms, 58 kernels). Of the device half,
      **87-92 % is two kernels** — `dart_select_minima` and `dart_cover_neighbors`, the shell scans
      themselves. Any further win is in those two; the loop around them is near its launch floor.
    - **SHIPPED off the back of it: one readback per round instead of two**, by scanning the
      survivor flags **inclusively** (the last entry is then the count outright) and having
      `dart_compact_alive` write at `positions[t] - 1` — the same shape `array.flatnonzero` already
      used. Measured **2-3.5 %**, every one of six cells in both reps, output unchanged.
    - **And it corrects the cost model: a *second consecutive* readback costs ~0.02 ms, not the
      ~0.1 ms §13.1 quotes.** The 0.1 ms figure is the pipeline drain, which the first read has
      already paid — which is why removing the second one was worth a few percent and not the ~15 %
      a naive two-times-0.1-ms estimate predicts. §15.2's "price the candidate directly" applies to
      readbacks too.
- **SHIPPED — the whole dart loop runs in cell-sorted index space, 1.69-2.35x and growing with the
  pool.** `sample._dart_throw_blue_noise` already radix-sorts the pool on its cell key to build the
  cell list, so `bucket` *is* the cell-order permutation — and the two kernels this section measures
  at 87-92 % of device time then threw that locality away, reading `j = bucket[k]` and scattering
  every payload read into the *unsorted* pool. Permuting `pool_points` / `priority` / `point_cell`
  through `bucket` once at setup makes a cell's members the contiguous run its offsets already
  name, so the loop variable **is** the point and all three reads are stride 1;
  `dart_cover_neighbors` stops taking `bucket` at all. Measured 1.685x / 2.009x / 2.353x at
  5 040 / 20 056 / ~80 000 accepted points, **byte-identical** points *and* face indices across six
  (mesh, seed) configurations.
  - **The tie-break is what makes it byte-identical, and getting it wrong would have been
    invisible.** The rule is "smaller priority, then smaller *pool* index", and renumbering the
    points changes what "pool index" means — so the comparison still reads `bucket[k] > bucket[i]`,
    on the priority-tie branch only, which keeps `bucket` out of the hot path. A sorted-space
    tie-break yields a different-but-valid packing that every count- and spacing-based test passes.
  - **The output order is restored, not abandoned.** `state` is sorted-indexed, so the survivor mask
    is permuted back through `scatter_index`'s inversion of `bucket` (one launch) before the
    compaction, and the returned points stay in pool order as they were.
- **SHIPPED — `points.farthest_point_sample` as one persistent block** — §14.1.

### 16.8 `linalg`, `smoothing`, `laplacian`

- **SHIPPED — `filter_laplacian`'s volume constraint no longer reads the volume back, 1.20-1.68x
  and growing with the iteration count.** `_apply_volume_constraint` ran
  `tw.measures.volume(...)` — a reduction that ends in a host readback — then formed
  `(vol_ini / vol_new) ** (1/3)` in Python and handed it to a `wp.map` as a scalar. That is **one
  full pipeline drain per smoothing pass** for a cube root of two numbers, inside the one loop
  whose pass count is the parameter callers turn up. `kernels/smoothing.rescale_to_volume` forms
  the ratio on the device from a `wp.utils.array_sum(..., out=<device array>)` and applies the same
  two skip conditions (zero current volume, non-positive ratio → scale 1, a no-op pass).
  - Measured against a detached baseline worktree (§15.6), min-of-7 over 30 calls between two
    syncs, on `icosphere(3)` / `icosphere(5)`:

    | iterations | 1 280 faces | 20 480 faces |
    |---|---|---|
    | 3 | 1.235x | 1.242x |
    | **10 (the default)** | **1.445x** | **1.445x** |
    | 30 | 1.659x | 1.617x |

  - **The tell was a readback census parametrized by the loop count, not a clock**: patching
    `wp.array.numpy` and calling at `iterations` 1 / 3 / 10 gave **6 / 8 / 15** readbacks, which is
    a fixed 5 plus one per pass. Now flat at 5. That census is deterministic, so it settles the
    question on a busy box where a timing would not (§15.6), and it is how to find the rest of this
    class: `plans/`-local, an `ast` walk for a readback — `.numpy()`, `read_scalar`, or a triwarp
    wrapper that returns a host scalar — lexically inside a `range`/`while` loop. It reports ~25
    live sites; most are genuine host branches (a convergence test, a compaction count) that cannot
    move to the device without changing when the loop stops, which is what makes this one unusual:
    nothing branched on the value, it was only arithmetic.
  - **The skip condition has to decline to write, not write an identity — and that distinction was
    a real regression, caught by an empty-face input and nothing else.** On a mesh with no faces
    the caller's `center` is the centre of mass of nothing, i.e. `NaN`, so "skip" spelled as a
    scale of `1.0` is `(p - NaN) * 1 + NaN` and propagates where the host version — which never
    reached the rescale at all in that case — left the buffer alone. The whole suite was green
    across it: every fixture has faces. `test_filter_laplacian_leaves_a_faceless_mesh_alone` pins
    it. **When moving a host-side `if` into a kernel, check what the *skipped* branch used to do
    with the arguments it never evaluated** — an identity is only an identity for finite operands.
    The sibling trap in the same edit: `wp.utils.array_sum` writes nothing for an empty input, so
    its `out=` accumulator must be `wp.zeros`, not `wp.empty` (§3.3).
  - **Not every per-iteration readback in that report is removable, and `registration.icp_*` is the
    worked counter-example.** Its `tw.reduce.any(valid)` early-exit genuinely needs the host to
    decide whether to break, and its second scalar read was already merged into one buffer in an
    earlier round (§16.1). Check whether the host *branches* on the value before proposing this.


- **SHIPPED — the CG iteration's per-launch floor, attacked three ways.** A `_BatchedCg` iteration
  was eight launches at a replayed ~1.17 µs each (§13.1) before any arithmetic, which is most of the
  cost on a *small* system — `heat_geodesic[sphere_small]` at 2 562 vertices is exactly that shape.
  - **`cg_advance_condition` folded into `cg_step_p`'s thread 0**, unconditionally: it reads `dots`
    and `atol_sq`, both written by the finalize *before* the launch, and writes `out_state`, which
    nothing in `cg_step_p` reads — so no barrier is needed and nothing races. Verified
    deterministically by counting `wp.launch` + `wp.launch_tiled` around one `_iteration`: **8 → 7**.
  - **The `r.r` / `r.z` partials folded into the x/r/z update** (`cg_step_x_r_z_dot`), which is
    §14.10's producer-consumer fusion applied to a reduction's *first* stage — legal precisely
    because a block-level partial needs only its own block's data. **7 → 6.** It reaches only the
    `preconditioner="diag"` branch; the multigrid branch's `z` comes from the cycle, not from the
    update, so there is nothing to fuse there. Its lanes stride by `wp.block_dim()` rather than
    taking one element each, which is what keeps the CPU device correct (§2.2).
  - **DECLINED — folding the `p.Ap` partials into `csr_matvec`.** That kernel is shared with the
    multigrid V-cycle at four sites that want no partials, and making it tiled would expose all of
    them to the ragged-tail hazard `cg_dot_partials` documents at length (a 129-entry tail cost
    11.03 µs against a 9-entry tail's 2.32, which alone turned a 1.5x win into a 0.82x loss at
    nearly the same size). Not worth one launch of six.
  - **`wp.capture_while` drives a *run* of iterations, not one** (`CG_ITERATIONS_PER_CHECK = 4`).
    The conditional-graph evaluation costs ~4-6 µs against a replayed launch's ~1.2, and the
    iterations that run past convergence are no-ops by construction — `cg_step_p` and
    `cg_step_x_r_z` pin a converged column's `beta` / `alpha` to exactly zero. Swept 1/2/4/8,
    reproduced twice: geomean **1.028x** at K=4, best cell `harmonic[saddle_graded]` **1.11x**,
    worst `harmonic[saddle_small]` 0.98x, and K=8 a real 0.90x loss on short solves. **It is not
    the 1.38x the same batching is worth on `graph.bfs`** (§16.9) — a CG iteration is six or seven
    launches where a BFS level is four, so the conditional evaluation is a smaller share of it.
    `maxiter` is clamped inside `cg_step_p` so the reported count stays exact despite the overshoot.
- **SHIPPED — every explicit smoothing filter applies the averaging operator in the thread that
  consumes its row, one launch per pass instead of two.** `filter_laplacian` (explicit),
  `filter_taubin`, `filter_neighborhood_average`, `filter_humphrey`, `filter_mut_dif_laplacian`,
  `filter_scalar_laplacian` and `filter_normals`; 1.05-1.96x, output bit-identical. The shared row
  apply is `kernels/laplacian.operator_row`. Full table, the `heat` sibling that was measured and
  and the three ways the candidate scan lies: **§14.10**, which also carries the `heat` sibling —
  landed after a first, wrong reading declined it on whole-call noise.

- **SHIPPED — `linalg.multigrid_preconditioner`** (aggregation, smoothed prolongator, Galerkin
  hierarchy, batched V-cycle inside the captured CG loop). A real win on ill-conditioned systems: a
  large iteration-count reduction translates into a solve-time win once the hierarchy is built. The
  parallel aggregation is close to a serial reference implementation's own quality — that was never
  the risk.
- **The hierarchy *setup* is the blocker, dominated by several sparse-op calls per coarsening level
  at Warp's fixed per-call cost, not by the aggregation algorithm itself.** Every losing case loses
  by exactly this setup cost; with a free setup all of them would win. **Cutting the per-call cost of
  the underlying sparse-matrix-multiply is the open lever.**
- **Two predictors for "will the hierarchy pay for itself" were built and refuted — neither operator
  size nor an early-iteration convergence-rate extrapolation separates winning systems from losing
  ones.** What separates them is the *operator's* off-diagonal dominance, not the unknown count or
  the convergence rate — so the shipped policy runs the cheap smoother for a bounded number of
  iterations and escalates only on non-convergence, gated by a measured off-diagonal-dominance
  threshold. **General lesson: a numerical-method decision like this belongs on a property of the
  operator, not a property of the problem size or an extrapolated iteration count** — both of the
  latter were tried first because they're cheaper to compute, and both were wrong.
- **The dominance gate has been checked against several operator classes it was never fitted on, and
  holds in most but not all of them** — it transfers cleanly to squared-Laplacian systems but a
  size-based companion threshold used alongside it does *not* transfer to a differently-conditioned
  system with a similar unknown count, and had to be retightened. **Never route a new caller through
  the `"auto"` gate without re-measuring on its own systems** — an operator with a favorable dominance
  reading on one mesh can read unfavorably on a differently-shaped mesh of the identical connectivity
  (measured on `harmonic k=2`, which wins big on a regular mesh and loses badly on a graded one at
  identical topology), so a strength-of-connection threshold for anisotropic operators is the open
  lead. Two heat-equation operator families (the Poisson system and the connection-Laplacian
  vector-heat system) were checked against the same gate and both correctly decline it today, since
  neither is currently solved repeatedly enough to amortize the hierarchy's setup cost — re-open only
  if a caller appears that solves the same system many times, and note the connection-Laplacian
  operator can't reach the gate at all yet because the whole hierarchy is scalar-CSR-only and would
  need a block-generalized version first.
- **CLOSED — `robust_laplacian`'s negative-weight residue on Dini's surface is entirely *boundary*
  edges, which are not Delaunay violations at all; the interior half this item called open is
  already fixed.** Re-censused on the current code: 4 783 flips, 4 375 edges, 266 of them
  multi-edges, and **zero** negative interior edges against **10** negative boundary ones
  (min -29.06). The interior half was closed by the multi-edge support `intrinsic_delaunay` now
  carries — a flip may create a second, geometrically distinct edge between two already-adjacent
  vertices, which intrinsically is a different geodesic and not a duplicate — so this item's "the
  lever for the interior half (a Delta-complex / signpost data structure) is unbuilt" is stale.
    - **The boundary residue is not a defect and no flip can address it**: a boundary edge has one
      opposite angle, so the two-angle Delaunay condition has nothing to compare it against, and an
      obtuse boundary corner is simply an obtuse boundary corner. The only remedy is inserting
      Steiner points, which `robust_laplacian`'s own contract forbids — its docstring promises the
      vertex set, and so the matrix's shape and meaning, are unchanged.
    - Both claims are already pinned by
      `tests/test_remesh.py::test_intrinsic_delaunay_resolves_interior_violations_via_multi_edges`,
      which asserts the interior minimum is non-negative *and* that the boundary residue survives,
      with a non-vacuity check that the fixture really reaches a multi-edge. The public docstring
      already says "Not *until*" and tells the caller to check the result. Nothing to implement.
    - **General lesson kept from the original triage** (it is still the useful part): quoting a
      `min()` over a set whose members have two different causes can size a proposed fix by 100x the
      wrong number — the residue's largest-magnitude member was the unfixable boundary case, not the
      interior one the proposed fix addressed.
- **SHIPPED — `filter_laplacian(implicit_time_integration=True)` and `filter_implicit_fairing` now
  solve their three position components as one batched multi-column solve instead of three separate
  single-column ones — real win (~2-2.8x), bit-identical output.** All three columns share one
  operator, so a batched solve advances them together in one Krylov iteration whose count is set by
  the worst column rather than the sum of three. **The tell that this was worth checking is textual:
  a `for` loop around a single-column solver in a module whose sibling functions all call the batched
  one.**
- **FIXED — two silent correctness defects in `filter_implicit_fairing`'s Dirichlet (pin-boundary)
  path, both found while doing the batching work above.** A helper returned "no constraint" for a
  mesh with no free interior vertex (a single triangle, a small hole patch) exactly the same way it
  did for "nothing pinned", so a caller asking to pin the boundary of such a mesh silently ran the
  unconstrained solve instead and moved every vertex it asked to fix. And the reduced solve seeded
  its iterate from the wrong array, teleporting any unreferenced vertex to the origin — the exact
  failure a sibling helper already existed to prevent on a different code path. **General lesson: a
  helper that returns the same sentinel for two logically different "nothing to do" cases is a
  defect waiting for the rarer case to be reached.**
- **`smoothing`'s two conditional-emit triplet writers must zero-pad their unwritten slots out of
  range, not to zero** — §12.7, a real cost fix (order of magnitude), reached by `holes.fill_smooth`
  through a shared helper so one fix lands on multiple benchmark rows.
- **REMOVED — block conjugate gradient over exactly two columns (`linalg._BlockCg2`), shipped and
  then retired.** Classical block CG (O'Leary 1980), sharing one Krylov subspace across both
  columns instead of solving them independently. It was gated to `n_columns == 2` under
  `preconditioner="diag"` — `harmonic`/`tutte` at `k=1`, `arap`'s global step, `heat.extend_scalar`
  — and it is gone. The dispatch, the 309-line class and its eight kernels are deleted;
  `_BatchedCg` handles every multi-column solve again. `linalg._cg_columns` carries the measurement
  at the site.
  - **The verdict, measured across every system in this package that reached the gate** (iteration
    counts are deterministic; wall is interleaved in one process, min of 15):

    | system | diag spread | block it | batched it | block/batched wall |
    |---|---|---|---|---|
    | `harmonic[saddle_small]` | 1.24 | 140 | 182 | 1.031x |
    | `harmonic[saddle]` | 1.24 | 275 | 355 | 1.032x |
    | `harmonic[hemisphere]` | 1.17 | 234 | 237 | **0.881x** |
    | `harmonic[saddle_graded]` | 2422 | **4774** | **2230** | **0.386x** |
    | `extend_scalar[sphere_small]` | 1.04 | 24 | 24 | 1.000x |
    | `extend_scalar[saddle]` | 4.10 | 31 | 31 | 1.000x |
    | `extend_scalar[saddle_graded]` | 4102 | 580 | 427 | 0.831x |

    Best case **+3 %**, worst case **−159 %**. Removing it measured **2.318x** end to end on
    `harmonic[saddle_graded]` (115.868 → 49.975 ms) and 1.075x on `harmonic[hemisphere]`, against a
    0.97x give-back on the two uniform saddles.
  - **It was never what made `extend_scalar` faster.** That function returns the *identical*
    iteration count under both solvers, so `12a5efe`'s batching of its two solves into one call
    carries its whole 1.79-1.95x — and removing block CG *improved* it further (1.174x on
    `saddle_graded`), because the removal is a win wherever conditioning is poor.
  - **The failure mode is the documented one and the Tikhonov floor does not address it.** On a
    graded mesh the two columns' search directions go nearly parallel, the shared subspace stops
    buying anything, and the iteration count goes **up** by 2.14x instead of down by 1.3x. A floor
    on the 2x2 Gram keeps the iteration finite; only deflation/restart would recover the rate.
  - **Why no gate was written, although a cheap predictor does exist.** The diagonal spread
    (`max(diag)/min(diag)` over the Jacobi diagonal already built for the preconditioner) separates
    the cases by three orders of magnitude — 1.17-1.24 on every winning system against 2422/4102 on
    the losing ones — so the plan's proposed predictor works. It was still declined: a predictor
    that is perfect is worth at most the +3 % best case, which does not pay for 309 lines, eight
    kernels and their compile time (§4.2). **A mechanism whose best case is a wash does not need a
    better gate; it needs removing.**
  - **The lesson that outlives it, and it is the important one.** `_BlockCg2` was validated on
    `saddle_small` and `saddle` and shipped without ever being run on `saddle_graded` — the fixture
    that exists *specifically* to be the ill-conditioned member of an otherwise identical pair
    (same vertex count, same face array, same boundary loops, worst aspect ratio 1.6 against 4 719).
    Ill-conditioning is exactly where block CG's documented failure mode lives, and §16.8's own
    round-11 notes had named it before the build started. **A mechanism validated on one member of
    a fixture pair built to isolate a variable must be measured on the other member before it
    ships** — `saddle`/`saddle_graded`, `sphere`/`tangle`, closed/open. The pair exists because the
    variable matters. That is §7.4's vacuity rule applied to a *benchmark* rather than a test, and
    it is now §9's rule too.
  - **The regression guard is deterministic and lives in `tests/test_linalg.py`**:
    `test_two_column_solve_costs_no_more_iterations_than_its_worst_column`, parametrized over a
    well- and an ill-conditioned shift, asserts that a two-column solve takes *exactly* the worst
    single column's iteration count. `_BatchedCg` satisfies that by construction; any mechanism that
    couples the columns does not. The old guard asserted block CG's count was strictly *lower* and
    would have caught this had it been parametrized over the ill-conditioned arm.

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
