# triwarp benchmarks

Performance benchmarks comparing `triwarp` against the CPU references **trimesh**, **libigl
(`igl`)**, **open3d** and **scipy**, built on
[pytest-benchmark](https://pytest-benchmark.readthedocs.io).

These are **not** collected by the normal test run (`pytest`'s `testpaths` is `tests/`); run them
by pointing pytest at this directory.

## Install

```bash
uv sync --group bench
```

## The rule: one group measures one axis, at 2–3 points

Face count is the *only* cost driver for a minority of functions in this package. Far more often
the driver is something a triangle count cannot express: how many connected components there are,
how many boundary loops and how long, the diameter of the adjacency graph, triangle aspect ratio,
vertex valence, ray depth, collision density, or a parameter that is not a mesh property at all.

So every benchmark group names one axis and takes 2–3 points along it. What is held *fixed* is as
much the point as what varies — that is what makes a timing spread attributable to a cause rather
than to "a different mesh". A group whose spread is ~1× is telling you its axis does not drive that
function, which is a useful answer too.

Two examples of what this buys, both measured on an RTX 5090:

| group | control | perturbed | spread |
|---|---|---|---|
| `combine.concatenate`, F = 81 920 fixed | 8 pieces **0.23 ms** | 512 pieces **5.7 ms** | **25×** |
| `graph.bfs`, V = 40 962 fixed | sphere Ø ≈ 130 **4.0 ms** | ribbon Ø = 20 481 **23 ms** | **5.8×** |

Neither is visible to a face-count sweep, and `bfs` still *loses to scipy* at the far end of its
axis (0.68 ms on the ribbon) even after the spread came down from 73×. `concatenate` came down from
48× (0.39 → 19.0 ms) by collapsing its per-piece index renumbering into one launch; what is left is
one `wp.copy` per input buffer, which Warp cannot batch — there is no gather across separate
allocations — so the residual slope is real and bounded below by the piece count.

Four more entries have already been retired from this table by the fixes they prompted, and all
four are kept in their modules as worked examples of what the axis rule is for:

- **`combine.split`** headed it at **262×** (2.6 ms → 669 ms, a 3× loss to trimesh at a thousand
  components). Commit `f57d3f0` batched its per-component compaction; it now runs 2.5 / 3.1 /
  9.4 ms against trimesh's 36.6 / 37.1 / 183 ms and open3d's 68.1 / 76.4 / 290 ms — a 3.7× spread
  and a 19-31× win at every point.
- **`validation.face_orientation_bits`** was **83×** (7.9 ms → 653 ms, an 83× loss to trimesh's
  `fix_winding`) while it propagated its Z2 bits one launch per graph level. Solving them with a
  parity-carrying union-find instead makes it depth-independent: **0.92 ms and 0.77 ms**, a win
  over trimesh at both ends. `repair.make_winding_consistent` inherited the fix.
- **`hole_filling.fill_holes_min_weight`** was *inverted*: 512 three-vertex holes cost 376 ms
  against 273 ms for two 512-vertex rims, i.e. the trivial case cost more than the `B³` one,
  because each hole paid its own readbacks, chord pass and span launches. Batching the interval DP
  across loops took it to **4.2 ms**, and dropped the two-rim point to 157 ms by running both rims
  in the same launches.
- **`creation.triangulate_polygon`** was **23×** across its resolution points (6.1 ms → 141 ms on a
  1024-point star) and an 80× loss to trimesh. The ear clipper is parallel; what was linear in the
  ring size was its *round count*, because competing ears were ranked by raw ring index and a star
  ring makes that rank suppress all but one ear per round. Ranking by a hash of the index gives
  **2.8 / 4.6 ms** and 30 rounds instead of 1 022.

## Mesh registries

Two registries, in [`meshes.py`](meshes.py), for two different questions.

### Scan meshes — the pure-`N` axis

Real scan meshes, read once with `meshio` (the same loader `triwarp.io` uses). Place the files in
`benchmarks/data/` (gitignored, local-only). Used **only** by the throughput groups: `triangles`,
`edges`, `reduce`, `grouping`, the `laplacian` entry kernels, `vertices.n_vertices` /
`mean_vertex_normals`, and the parts of `remesh` / `proximity` / `texture` / `neighbors` that
genuinely scale with size.

| category | faces | meshes |
|---|---|---|
| `small` | `<10k` | — |
| `medium` | `<100k` | bunny_decimated, bunny |
| `large` | `<1M` | dragon |
| `extralarge` | `<10M` | happy_buddha |
| `huge` | `≥10M` | lucy |

### Feature meshes — one control, one property each

Generated with `trimesh.creation` (never with `triwarp.creation`: a benchmark input must not depend
on the code under test). `sphere_med` = `icosphere(6)` is the control — **V = 40 962, F = 81 920**,
one watertight component, uniform valence 6, graph diameter ≈ 130, no boundary, aspect ratio ≈ 1.4.
Every other mesh perturbs exactly one of those while pinning `V` and/or `F` to it.

| name | V | F | pinned | perturbed property |
|---|---|---|---|---|
| `sphere_small` | 2 562 | 5 120 | — | clean small `N` |
| **`sphere_med`** | 40 962 | 81 920 | — | **control** |
| `sphere_large` | 163 842 | 327 680 | — | clean large `N` |
| `parts_64` | 41 088 | 81 920 | **F** | components 1 → 64 |
| `parts_1024` | 43 008 | 81 920 | **F** | components 1 → 1 024 |
| `ribbon_long` | 40 962 | 40 960 | **V** | graph diameter 130 → 20 481 |
| `fan_hub` | 40 962 | 81 920 | **V, F** | max valence 6 → 40 960 |
| `rim_long` | 131 072 | 131 072 | — | 2 boundary loops × 65 536 |
| `holes_many` | 40 962 | 81 408 | **V** | 512 boundary loops × 3 |
| `rim_short` | 1 024 | 1 024 | — | 2 loops × 512 (the `B³` DP scale) |
| `saddle_small` | 4 624 | 8 978 | — | small disk patch |
| `saddle` | 17 689 | 34 848 | — | well-conditioned disk patch |
| `saddle_graded` | 17 689 | 34 848 | **V, F, connectivity** | worst aspect ratio 1.6 → 4 719 |
| `hemisphere` | 20 737 | 41 088 | — | curved disk, 1 rim of 384 |
| `shells_8` | 20 496 | 40 960 | — | ray crossings 2 → 16 |
| `tangle_2` | 20 484 | 40 960 | — | self-intersection density |

Seven parametric builders produce all sixteen, and [`test_meshes.py`](test_meshes.py) asserts every
recorded count, the topology each mesh is chosen for, and the cross-mesh pinning invariants. **Run
it after touching `meshes.py`** — a silent change in `trimesh.creation` would otherwise leave the
benchmarks green while comparing meshes that differ in more than one way.

### Axes

`AXES` in [`meshes.py`](meshes.py) names each comparison; the control comes first, so a results
table reads left-to-right as "baseline, then the perturbation".

| axis | meshes | holds fixed |
|---|---|---|
| `scale` | sphere_small → med → large | — (the igl-safe `N` sweep) |
| `components` | sphere_med, parts_64, parts_1024 | `F` |
| `diameter` | sphere_med, ribbon_long | `V`, triangle quality |
| `valence` | sphere_med, fan_hub | `V` and `F` |
| `loops` | sphere_med, rim_long, holes_many | — (loop length vs loop count) |
| `loops_dp` | rim_short, holes_many | — (same contrast, at `B³`-survivable size) |
| `quality` | saddle, saddle_graded | `V`, `F`, connectivity |
| `patch` | saddle_small, saddle, hemisphere | disk topology |
| `polyline` | saddle_small, saddle, rim_long | — (longest loop 268 → 528 → 65 536) |
| `depth` | sphere_med, shells_8 | — |
| `overlap` | sphere_med, tangle_2 | — |
| `resolution` | *no meshes* | — (`test_creation`; see below) |

Select one with `@pytest.mark.benchaxis("components")`. `@pytest.mark.benchmeshes(...)` names meshes
directly for a one-off; both bypass `--size`, and both are a `UsageError` on a mesh-free benchmark.
A benchmark with neither marker gets the scan sweep.

Where a group's driver is a *parameter* rather than a mesh, it is a second
`pytest.mark.parametrize` layer with named ids and lands as extra rows in the same table:
`screened_poisson(depth)`, `rasterize_*(resolution)`, `blue_noise(radius)`,
`expand_vertex_mask(hops)`, `winding_number(|queries|)`, `simplify_polyline(tol)`,
`icp(initial misalignment)`, `solve_spd_columns(check_every)`, `neighbors(k / leaf_size /
grid_bins)`, and the defect counts in `test_repair`.

### Mesh-free benchmarks

`triwarp.creation` has no input mesh, so [`test_creation.py`](test_creation.py) takes the
`bench_lib` fixture instead of `bench_case` and is parametrized over libraries alone; its axis is
`resolution`, expressed as a plain `pytest.mark.parametrize` on `sections` / `subdivisions` /
`face_count`. Supporting that needed two harness changes: `BenchCase` derives from a mesh-free
`BenchLibrary` base, and this package overrides `pytest_benchmark_group_stats` — the plugin's own
version indexes `bench["params"]["mesh_name"]` directly and raises `KeyError` as soon as a mesh-free
case is collected under the default `group,param:mesh_name` grouping.

## Run

```bash
# Default: triwarp-cuda on all meshes; CPU references on scan meshes up to 'large'.
# triwarp-cpu is off by default when CUDA is available (pass --device=both to add it).
uv run pytest benchmarks/

# One module, GPU only:
uv run pytest benchmarks/test_combine.py --device=cuda

# Quick CPU-only smoke on the medium scan meshes:
uv run pytest benchmarks/test_edges.py --device=cpu --size=medium
```

A full default run is **~18 minutes** on an RTX 5090 (measured: 1 097 s across all 33 modules, one
process each, 999 cases plus 139 skipped). The four slowest modules are `test_reconstruction`
(235 s), `test_laplacian` (95 s), `test_proximity` (85 s, the `O(Q x F)` winding number) and
`test_smoothing` (70 s).

Only the first of those is not measuring triwarp: **190 of `test_reconstruction`'s 199 timed
seconds are open3d's CPU `create_from_point_cloud_poisson`** (7.5 s a call at depth 9 on `bunny`),
against 9.0 s for every triwarp case in the module combined. It is the largest single reference cost
in the suite and the reason that module doubled the 123 s recorded here previously; before trimming
anything in it, read the per-library split rather than the module total.

By default the harness prints **one comparison table per (function, mesh)** — each table lists the
libraries and any parameter points side by side — via `--benchmark-group-by=group,param:mesh_name`,
set automatically. Pass your own `--benchmark-group-by=...` to override.

## Flags

| flag | default | meaning |
|---|---|---|
| `--device` | `auto` | `triwarp` target(s): `auto`/`cpu`/`cuda`/`both`. `auto` = cuda if available, else cpu. The CPU references always run. |
| `--size` | `all` | comma-separated size categories for the **scan** sweep (`small,medium,large,extralarge,huge`). Naming a size also lifts the CPU cap for it. Has no effect on axis-driven groups. |
| `--cpu-max-size` | `large` | CPU-bound libraries skip scan meshes larger than this unless the size is named in `--size`. |

## Notes

- **GPU timing** is captured correctly: `wp.synchronize_device` runs inside the timed region and
  one warm-up round covers Warp kernel JIT compilation.
- **`BenchCase.run(fn, rounds=...)`** lowers the repeat count from the default 10 for the groups
  whose single call runs into hundreds of milliseconds — the hole-filling DP, `split` on a thousand
  components, an isotropic remesh, a Poisson reconstruction, the float64 CG solves. Ten rounds of
  those would dominate the suite's wall clock and their spread is wide enough that the extra samples
  buy nothing.
- **There is a ~340 µs host-side floor on every triwarp wrapper call** (allocation plus Warp's
  launch path; ~75 µs of it is the NumPy prologue). `test_creation::test_box` is the probe that
  measures it — `box` is a 12-triangle constant table, so it measures nothing else. Read absolute
  numbers against that floor: a group sitting at it across its whole axis is reporting launch
  overhead, not an algorithm. **Read the floor from a full-suite run**: whichever group runs first
  in a module absorbs that library's one-time initialization (52 ms for triwarp, 102 ms for trimesh,
  measured on `test_creation.py` alone).
- **A CPU reference can be the whole cost of a module.** Two were, before being capped: trimesh's
  `discrete_mean_curvature_measure` (one `cKDTree` ball query per point) took **40 s a call** on
  `sphere_med` and was 95% of `test_curvature`, and `tm.points.fit_line` took **22 s a call** on
  `bunny` and was 91% of `test_points`. Both are now capped at the smallest input that still
  establishes the ratio. When a module's wall clock looks wrong, get `--durations=10` before
  trimming anything — the reference is a more likely culprit than the code under test.
- **Read the per-table unit.** pytest-benchmark picks µs or ms *per table*, so a number copied out
  of one group is not comparable to one from another without checking the `Name (time in ...)`
  header. Cross-group ratios quoted in this repo's docstrings come from direct measurement scripts
  for exactly that reason.
- **`--device=cpu` runs should be one module per process.** Warp 1.15 CPU kernels can corrupt the
  heap, which shows up as an unrelated crash later in the session:
  ```bash
  for f in benchmarks/test_*.py; do uv run pytest "$f" --device=cpu -q; done
  ```
- **trimesh references** rebuild their `tm.Trimesh` *inside* the timed callable wherever the
  reference reads a cached property, so every round measures real work. open3d references that
  mutate idempotently do the same; `BenchCase.mesh_o3d` is only shared where the reference either
  does not touch its input or recomputes unconditionally.
- `edges_unique*` triwarp calls pass `n_vertices=` to avoid a host sync skewing GPU numbers.
- Every benchmark carries an explicit `benchlibs` marker, so adding a library kind to `LIBRARIES`
  never silently generates cases for modules that have no branch for it.
- **`test_texture` uses projected, non-injective UVs** (vertex `xy` normalized to the unit square)
  because the scan meshes carry no atlas and computing one would dominate the measurement. The
  rasterizer cost is the number of (triangle, covered pixel) pairs, which a projection reproduces
  faithfully; the module docstring explains why that is sound for timing but not for parity.

### Measured hazards — do not re-enable these without reading the numbers

Clean synthetic meshes **unblocked libigl** where it previously had to be skipped:
`igl.principal_curvature` segfaults on every scan mesh (they have non-manifold vertices and its
vertex-ring walk assumes manifoldness — a hard crash, not an exception) but runs on `sphere_med` in
0.48 s; `igl.heat_geodesics_precompute` raises `Precomputation failed.` on every scan mesh but
succeeds on all sixteen feature meshes; `igl.harmonic` / `igl.lscm` cannot factor the scan meshes'
cotangent systems but handle every mesh on the `patch` axis. So those three comparisons are now
drawn on the same meshes triwarp is measured on rather than on two tiny saddle patches.

Two hazards remain and are encoded as explicit skips:

| call | mesh | measured | handling |
|---|---|---|---|
| `igl.principal_curvature` | `fan_hub` | **110 s** (ring walk is quadratic in valence) | `fan_hub` never reaches igl curvature |
| `igl.principal_curvature` | `sphere_large` | ~2 s/call | igl capped at `sphere_med` |
| `open3d.is_watertight` | `sphere_med` | **13.6 s** (brute-force self-intersection scan) | `rounds=1` |
| `open3d.is_watertight` | `tangle_2` | 3.5 s — *faster on the harder mesh*, because the scan early-exits on the first hit | `rounds=1` |

Also: never construct a `wp.Mesh` with zero triangles on CUDA (it corrupts device state silently);
`test_meshes.py` asserts every feature mesh is non-empty.

## Reference coverage

`trimesh`, `igl`, `open3d` and `scipy` are all registered in `LIBRARIES` and used for **every**
benchmarked function that has a genuine equivalent — the point is to have independent
implementations to spot outliers against, not only to fill gaps. `open3d` is marked `cpu_bound`
even though the installed wheel is a CUDA build: the legacy `open3d.pipelines` / `open3d.geometry`
APIs used here are CPU-only (only `open3d.t` has GPU kernels). `scipy` covers the k-nearest and
radius searches (`spatial.KDTree`) and the graph traversals (`sparse.csgraph`), which is where
trimesh itself delegates.

| module | open3d reference |
|---|---|
| `test_creation` | `create_box`, `create_sphere` (a UV sphere, so it pairs with `uv_sphere`), `create_cylinder`, `create_cone`, `create_torus` |
| `test_registration` | `TransformationEstimationPointToPoint.compute_transformation`, `registration_icp` (point-to-point, point-to-plane, `TukeyLoss`) |
| `test_reconstruction` | `create_from_point_cloud_ball_pivoting`, `create_from_point_cloud_poisson` |
| `test_remesh` | `subdivide_midpoint` (`subdivide` only) |
| `test_smoothing` | `filter_smooth_laplacian` (`novol` only) |
| `test_sample` | `sample_points_poisson_disk` |
| `test_combine` | `cluster_connected_triangles` + `select_by_index` (`split` only) |
| `test_hole_filling` | `open3d.t.geometry.TriangleMesh.fill_holes` |
| `test_repair` | `remove_duplicated_triangles`, `remove_duplicated_vertices` |
| `test_validation` | `is_watertight` |
| `test_vertices` | `compute_vertex_normals` |
| `test_proximity` | `get_axis_aligned_bounding_box` (`aabb_bounds` only) |
| `test_points` | `PointCloud.estimate_normals` (`KDTreeSearchParamKNN`) |
| `test_distance` | `PointCloud.compute_point_cloud_distance` (the non-differentiable Chamfer / Hausdorff cases) |
| `test_convex` | `compute_convex_hull` (exact qhull vs the approximate support sweep) |

Modules with **no** open3d equivalent, and why, are documented in each module's docstring:
`test_edges` (no general edge list), `test_boundary` (no loop ordering), `test_grouping` and
`test_reduce` (array primitives), `test_parametrization` (no harmonic/LSCM/ARAP), `test_linalg` (no
iterative solver), `test_laplacian` (no cotangent or mass matrix — the smoothing filters build their
weights inline), `test_curvature` (no curvature estimation at all), `test_intersection` (no plane
section), `test_texture` (stores UVs but has no bake or resample), `test_polyline` (`LineSet` is
unordered segments with no length/resample/simplify), `test_geodesic` (no geodesic distance),
`test_selection` (no selection morphology), `test_mesh` (no caching container), `test_graph` (no
traversal over an abstract CSR), `test_neighbors` (no batched k-NN query), plus the individual
functions noted inline.

Modules with no baseline from **any** reference are `test_texture`, `test_polyline`, `test_reduce`
and `test_linalg` (plus `stitch*` in `test_combine` and the morphology groups in `test_selection`);
each docstring says which reference was considered and why it is not apples-to-apples. Those are
before/after self-comparisons.

Where the reference is not algorithmically identical, the module docstring says so — `test_repair`
(open3d's dedup is orientation-sensitive), `test_sample` (count- vs radius-parametrized),
`test_hole_filling` and `test_smoothing` (different algorithms for the same task), `test_convex`
(approximate vs exact).
