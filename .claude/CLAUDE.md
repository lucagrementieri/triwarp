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
- Call `.numpy()` on Warp output arrays before passing to NumPy comparison functions.
- Use `np.allclose(got, exp, rtol=1e-5, atol=1e-5)` for floating-point results (or `got_wp` / `exp_tm` with library suffixes).
- Use `np.array_equal(got, exp)` for boolean or integer results.
- Name variables with a suffix for the library: `_np` for NumPy/SciPy, `_tm` for Trimesh, `_wp` for Warp. In assertions prefer `got` / `exp` (e.g. `got_wp = ...`, `exp_tm = mesh_tm.face_adjacency_unshared`).

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
