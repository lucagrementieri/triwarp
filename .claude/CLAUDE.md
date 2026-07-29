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
- Cast `wp.tid()` to `int` explicitly when used as an array index: `f = int(wp.tid())`.
- Use `wp.launch(kernel=..., dim=..., inputs=[...], device=...)` for execution. Always forward the `device` from the input arrays.
- Array slicing is supported inside kernels: `faces[f * 3 : (f + 1) * 3]` produces a sub-array view.
- Use `wp.cast(expr, TargetType)` for explicit type conversions between Warp types.
- Prepend output argument names with out_ and put them at the end of the kernel signature after all the input arguments.

---

## 4. Python-Scope Wrappers

- Every kernel lives in a `kernels/` sub-module. Import it with an alias: `from triwarp.kernels import triangles as kernel_triangles`.
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

Inside kernels, keep using `wp.cast(expr, TargetType)` for scalar and vector conversions.

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
  `wp.indexedarray` view (see `repair.orient_faces`).
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

### Fallback references: libigl and potpourri3d

When `trimesh` has no equivalent function, use the `igl` Python package (bindings for the C++
reference mirrored under `reference/libigl/`) as the CPU reference instead — import as
`import igl`, name reference variables with an `_igl` suffix. Pass triwarp's flat face buffer
as `mesh_tm.faces` (`(n_faces, 3)` int array) to the igl function. Otherwise follow the same
comparison conventions (`np.array_equal`/`np.allclose`, inline `.numpy()`).

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

### Mesh fixtures (prefer over inline construction)

Reuse shared mesh fixtures from `tests/conftest.py` instead of building meshes in each test. Fixtures return `(mesh_tm: tm.Trimesh, mesh_wp: wp.Mesh)` via `tests.conversions.trimesh_to_warp`.

| Fixture | Use when |
|---------|----------|
| `icosahedron` | Default watertight solid; inside/outside, surface sampling, sign tests |
| `cave_cube` | Hollow / non-convex shell (boolean difference) |
| `hemisphere`, `half_torus` | Curved or open surfaces |

- **Do not** call `tm.creation.box()` or hand-roll `wp.Mesh(...)` in tests unless the case requires a bespoke degenerate mesh (e.g. empty faces, unreferenced vertices).
- When a simple cube would suffice, prefer **`icosahedron`** or **`cave_cube`** for richer geometry.
- Parametrize over multiple fixtures with `request.getfixturevalue(mesh_name)` when coverage should span mesh types (see `test_signed_distance_on_mesh_random`).
- Use `mesh_wp.device` (not the `device` fixture) for query-point allocation when a mesh fixture is already in scope.
- Edge-case tests (empty points/faces, single-triangle pathology) may still use minimal inline buffers.

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

**Do not check that input arrays share the same device.** Warp raises a clear error automatically when mismatched devices are used in `wp.launch` or array operations, so manual `if arr.device != device: raise ValueError(...)` guards are redundant. Omit them entirely.

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

BEFORE using an unfamiliar Warp builtin, sparse, or utils function, `grep` these files to confirm the exact name, signature, and scope rather than guessing. Each file lists its source URL (Warp 1.14.0) at the top — fetch it for full argument details or examples when the one-line description is insufficient.

---

## 10. Documentation (MkDocs + mkdocstrings)

Docs are built with **MkDocs Material** + **mkdocstrings** (`python` handler, `docstring_style: numpy`), configured in `mkdocs.yml`. `docs/gen_ref_pages.py` auto-generates one API reference page per public module under `triwarp/` on every build, including subpackages such as `triwarp/heat/` (named by dotted path, e.g. `heat.distance`; `triwarp/kernels/` is excluded) — a new module needs **no manual nav entry**, it appears automatically. Preview locally with `DISABLE_MKDOCS_2_WARNING=true uv run mkdocs serve`; validate with `DISABLE_MKDOCS_2_WARNING=true uv run mkdocs build --strict` (fails the build on any broken cross-reference or unresolved external inventory). The `DISABLE_MKDOCS_2_WARNING` prefix silences a promotional banner injected by the `properdocs` transitive dependency of `mkdocs-gen-files`/`mkdocs-literate-nav`/`mkdocs-section-index`.

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
- After editing docstrings, sanity-check with `grep -nE ':(func|attr|meth|class|data|mod):\`' triwarp/*.py` — it should return nothing.

---

## 11. Function Ordering Within a Module

`mkdocstrings` is configured with `members_order: source` (see `mkdocs.yml`), so **source order is the rendered docs order** — placement in the file is part of the public API's discoverability, not cosmetic.

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
