# triwarp benchmarks

Performance benchmarks comparing `triwarp` against the CPU references **trimesh**, **libigl
(`igl`)**, **open3d**, **scipy**, **potpourri3d** (geometry-central) and **pymeshlab**
(MeshLab / VCGlib), built on
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

The **pymeshlab** rows add roughly **six minutes** on top of that, spread over 26 modules; the
largest single contributions are `test_geodesic` (+40 s, its heat solver at `setup=full` rebuilds the
factorization every round), `test_proximity` (+3 s even capped at `bunny`), `test_combine` (+18 s, the
1 630 ms-per-call component split on `parts_1024`) and `test_curvature` (+9 s). Three groups are
explicitly capped for that reason — see the pymeshlab hazards below.

That measurement predates the six modules added with the potpourri3d port
(`test_halfedge`, `test_tangent`, `test_contour`, `test_tracing`, `test_vector_heat`,
`test_intrinsic`, `test_signed_heat`) and the potpourri3d rows in the existing ones. The three Phase-1 modules add ~10 s;
`test_vector_heat` and `test_tracing` add ~40 s between them (the reference factors two sparse systems
per case, and traces one ray per call); `test_signed_heat` adds ~15 s, most of it the reference's
1 s-per-call solver; and the potpourri3d rows add most of their cost to
`test_geodesic`, now 74 s for the module with its `heat_geodesic` group carrying three libraries at two
setup points each.

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
| `--device` | `auto` | `triwarp` target(s): `auto`/`cpu`/`cuda`/`both`. `auto` = cuda if available, else cpu. The CPU references (trimesh / igl / open3d / scipy / potpourri3d / pymeshlab) always run. |
| `--size` | `all` | comma-separated size categories for the **scan** sweep (`small,medium,large,extralarge,huge`). Naming a size also lifts the CPU cap for it. Has no effect on axis-driven groups. |
| `--cpu-max-size` | `large` | CPU-bound libraries (every reference, plus `triwarp-cpu`) skip scan meshes larger than this unless the size is named in `--size`. |

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

  **A full-suite ratio matrix (605 comparisons) found 24 rows that are entirely this floor**, and
  each now says so in its own docstring rather than reading as a defect: the `creation` revolution
  primitives and Platonic tables (`creation`'s module docstring covers them collectively),
  `icosphere` since its connectivity became closed-form, `vertices.n_vertices`,
  `proximity.aabb_bounds`, `points.point_plane_distance` and `remesh.cluster_decimate` at
  `bunny_decimated`. **Every one of them inverts further along its own axis** — `cluster_decimate`
  wins 76x at `dragon`, `n_vertices` 259x at `lucy` — so a floor row is a statement about the input
  size, not about the code, and optimizing one would mean removing allocations from correct code for
  microseconds nobody experiences. Below roughly 10³ elements, prefer to fix the *input size* of the
  benchmark over the function it measures.
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
- **pymeshlab rows build their `MeshSet` inside the timed callable** unless the filter is verified
  geometry-preserving, because almost every MeshLab filter mutates `current_mesh()` in place. See the
  pymeshlab hazards section for the build cost and the shared-MeshSet whitelist.
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

`potpourri3d` needs the same treatment for a different reason, and the boundary is not
manifoldness but *which* geometry-central mesh class a call builds. Measured:

| call | input | result |
|---|---|---|
| `cotan_laplacian`, `face_areas`, `vertex_areas` | anything | fine — pure numpy/scipy, no mesh built |
| `MeshHeatMethodDistanceSolver`, `marching_triangles` | non-manifold edges | fine — `SurfaceMesh` tolerates them |
| `pp3d.edges` | a mesh with an **unreferenced vertex** | `RuntimeError: GC_SAFETY_ASSERT FAILURE … unreferenced vertex`, on **every scan mesh** |
| `MeshVectorHeatSolver` | non-manifold | `RuntimeError: GC_SAFETY_ASSERT FAILURE … manifold_surface_mesh` |
| `MeshFastMarchingDistanceSolver` | non-manifold | `RuntimeError: handling of nonmanifold mesh not yet implemented` |

So potpourri3d rows live on the synthetic axes, which are manifold and fully referenced by
construction. Where the group's own axis is the scan sweep, the comparison moves to a sibling group
on the `scale` axis rather than being skipped — `edges_unique_manifold` is that case — the same move
that unblocked the libigl comparisons above.

Two more hazards remain and are encoded as explicit skips:

| call | mesh | measured | handling |
|---|---|---|---|
| `igl.principal_curvature` | `fan_hub` | **110 s** (ring walk is quadratic in valence) | `fan_hub` never reaches igl curvature |
| `igl.principal_curvature` | `sphere_large` | ~2 s/call | igl capped at `sphere_med` |
| `open3d.is_watertight` | `sphere_med` | **13.6 s** (brute-force self-intersection scan) | `rounds=1` |
| `open3d.is_watertight` | `tangle_2` | 3.5 s — *faster on the harder mesh*, because the scan early-exits on the first hit | `rounds=1` |
| `pp3d.marching_triangles` | `sphere_med`, 1 072-curve field | 932 ms/call | `rounds=3` |
| `MeshVectorHeatSolver` construction | `sphere_large` | 373 ms/call | `rounds=3` |

Also: never construct a `wp.Mesh` with zero triangles on CUDA (it corrupts device state silently);
`test_meshes.py` asserts every feature mesh is non-empty.

## Reference coverage

`trimesh`, `igl`, `open3d`, `scipy`, `potpourri3d` and `pymeshlab` are all registered in `LIBRARIES`
and used for
**every** benchmarked function that has a genuine equivalent — the point is to have independent
implementations to spot outliers against, not only to fill gaps. `open3d` is marked `cpu_bound`
even though the installed wheel is a CUDA build: the legacy `open3d.pipelines` / `open3d.geometry`
APIs used here are CPU-only (only `open3d.t` has GPU kernels). `scipy` covers the k-nearest and
radius searches (`spatial.KDTree`) and the graph traversals (`sparse.csgraph`), which is where
trimesh itself delegates. `potpourri3d` is CPU-only (geometry-central) and is the **only** reference
for the heat-method family, tangent spaces and isocontours. `pymeshlab` is CPU-only (MeshLab /
VCGlib) and is the **broadest** — it reaches 26 modules, more than any other single reference.

### The parity gate: a benchmarked reference must be a tested reference

Nothing in this directory asserts that two timed implementations compute the same thing — every
assert here is a shape, `isfinite` or count guard whose only job is to prove the timed work was not
optimised away. `tests/test_parity.py` closes that loop and **fails the default `pytest` run** when a
benchmarked `(group, library)` pair is neither tested nor exempted. Run
`uv run python -m tests.parity` for the full matrix.

The `benchmark(group=...)` name is the join key, which makes group names a **cross-suite API**:
renaming one breaks every `parity` marker in `tests/` that cites it.

Either a test in `tests/` claims the pair:

```python
@pytest.mark.parity("faces_to_edges", "trimesh")     # one marker per group, all its libraries listed
```

…or the benchmark function declares, at its own call site and in prose, why the two results are not
comparable:

```python
@pytest.mark.noparity("pymeshlab", oracle="trimesh", reason="D2 ...measured number... trimesh is "
                      "the oracle for this group, in tests/test_x.py::test_y.")
```

`reason` is mandatory and is checked for substance (≥ 40 chars, ≥ 6 words, no `"see above"`-class
filler). `oracle=` names the library that *is* the oracle, and the gate then requires **that** pair to
be covered — so an exemption is a checked claim rather than an escape hatch. Both markers take string
literals only; a computed argument is invisible to the static scan and is rejected. Nineteen of the
211 pairs are exempt; the category rubric (D1–D6) is in `.claude/CLAUDE.md` §6.

**Exemptions are for results that cannot be compared, not comparisons that are awkward.** Three rows
here were heading for exemptions on an 8%-of-displacement Laplacian disagreement until MeshLab's
actual smoothing stencil was recovered numerically; they are now exact class-B asserts. "Does strictly
less", "the output shape differs" and "the ordering differs" are transforms, not exemptions.

### pymeshlab

MeshLab exposes 281 filters. 61 of them map onto something triwarp already has, and they are what
these rows are drawn from. Eight of the groups they land in **had no reference of any kind** before:

| module | group | filter | what it settled |
|---|---|---|---|
| `test_linalg` | `min_quad_with_fixed` | `compute_scalar_by_scalar_harmonic_field_per_vertex` | its direct solve is **flat** (39.0 / 39.6 ms) across the `quality` axis where triwarp's CG goes 33.7 → 83.5, so the whole spread is iteration count |
| `test_selection` | `expand_vertex_mask` / `shrink_vertex_mask` | `apply_selection_dilatation` / `..._erosion` | slopes of 7.3× and 8.0× over 1 → 8 hops, confirming a hop is constant work |
| `test_proximity` | `signed_distance_on_mesh` | `compute_scalar_by_distance_from_another_mesh_per_vertex` | its per-query cost **grows** with the face count (20 / 63 / 768 µs) where triwarp's BVH does not |
| `test_smoothing` | `filter_taubin`, `filter_humphrey` (new groups) | `apply_coord_taubin_smoothing`, `apply_coord_hc_laplacian_smoothing` | two triwarp filters that were unbenchmarked |
| `test_remesh` | `isotropic_remesh` | `meshing_isotropic_explicit_remeshing` | 3.3× across `quality` against triwarp's **flat** 123 ms — the fixed-pass loop is not adapting |
| `test_reduce` | `median` | `get_scalar_statistics_per_vertex` | six statistics in one call, 2.5 / 11 / 110× |
| `test_vertices` | `average_onto_vertices` | `compute_scalar_transfer_face_to_vertex` | also *faster* on `fan_hub` — valence is not a hot spot, independently |
| `test_repair` | `remove_non_manifold_faces` | `meshing_repair_non_manifold_edges` | nothing else in the set removes non-manifold faces at all |

The rest are second or third independent implementations:

| module | pymeshlab reference |
|---|---|
| `test_boundary` | `compute_selection_from_mesh_border` (`boundary_edges` only — no loop ordering) |
| `test_combine` | `generate_splitting_by_connected_components` — one call for label *and* compaction, and a **39× spread** across the `components` axis against triwarp's 3.7× |
| `test_convex` | `generate_convex_hull` (qhull a third time, so it prices the wrapper rather than the algorithm) |
| `test_creation` | `create_cube`, `create_sphere` (an *icosphere*, so it pairs with `icosphere` where open3d's pairs with `uv_sphere`), `create_torus`, `create_annulus`, `create_cone` |
| `test_curvature` | `compute_curvature_principal_directions_per_vertex(method='Quadric Fitting')` and `compute_scalar_by_discrete_curvature_per_vertex` — the only reference that survives the whole `scale` axis, where trimesh is capped at `sphere_small` |
| `test_distance` | `get_hausdorff_distance` (one-directional, so both directions are timed) |
| `test_edges` | `get_geometric_measures()['avg_edge_length']` |
| `test_geodesic` | `compute_scalar_by_heat_geodesic_distance_from_selection_per_vertex` — the fourth heat-method implementation, and the only one whose amortized path is just "call it twice"; plus `..._geodesic_distance_from_given_point_...` as a non-PDE alternative |
| `test_graph` | `compute_selection_by_small_disconnected_components_per_face(nbfaceratio=0.0)` |
| `test_hole_filling` | `meshing_close_holes` (ear clipping, so **1.7×** across `loops_dp` against triwarp's 37× — the price of *not* running a `B³` DP) |
| `test_parametrization` | `compute_texcoord_parametrization_harmonic` / `..._least_squares_conformal_maps` — both wrap **libigl's own code**, so they price MeshLab's wrapper rather than a third algorithm |
| `test_points` | `compute_normal_for_point_clouds(k=)`, `compute_matrix_by_fitting_to_plane` |
| `test_reconstruction` | `generate_surface_reconstruction_ball_pivoting` (VCGlib's original BPA) and `..._screened_poisson` (Kazhdan's own code, the same one open3d wraps) |
| `test_registration` | `compute_matrix_by_icp_between_meshes` — the only reference the mesh-target `icp` group has |
| `test_remesh` | `meshing_surface_subdivision_midpoint(threshold=)` in `subdivide_to_size`, landing on the identical output face count |
| `test_sample` | `generate_sampling_poisson_disk(radius=)` — the only blue-noise reference that takes a *radius*, so the radius sweep maps for the first time |
| `test_smoothing` | `apply_coord_laplacian_smoothing_scale_dependent` (Desbrun's, = `filter_mut_dif_laplacian`) and `apply_coord_laplacian_smoothing` |
| `test_triangles` | `compute_normal_per_face`, `get_geometric_measures` (`shell_barycenter` = `centroid`) |
| `test_validation` | `get_topological_measures` + `compute_selection_by_self_intersections_per_face` — the same composition open3d does in **13.8 s** and MeshLab in 140.8 ms; plus `compute_selection_by_non_manifold_per_vertex` |
| `test_vertices` | `compute_normal_per_vertex` at `weightmode='Simple Average'` and `'By Area'` — one filter covering both normal groups |

#### Ported gaps: rows where pymeshlab is the *source*, not the reference

The 24 filters MeshLab exposed that triwarp did not have were ported (bucket B of
`plans/make-pymeshlab-a-first-class-reference.md`). Their benchmark rows read the other way round —
the pymeshlab filter is the thing being caught up with, and in every case the port is the newer code:

| group | module | pymeshlab source filter | measured |
|---|---|---|---|
| `face_quality` | `test_triangles` | `compute_scalar_by_aspect_ratio_per_face` | 34 µs against 445 µs, flat across `quality` on both sides |
| `outlier_probability` | `test_points` | `compute_selection_point_cloud_outliers` (LoOP) | 2.6 ms against 0.66 ms at `sphere_small` — the k-NN table dominates, see `test_neighbors` |
| `platonic_solids`, `grid`, `sphere_cap` | `test_creation` | `create_{tetrahedron,octahedron,dodecahedron,grid,sphere_cap}` | fixed-cost ties on the tables; 2.2 ms against 12.5 ms on a subdiv-6 cap |
| `filter_scalar_laplacian`, `saturate_scalar_gradient` | `test_smoothing` | `apply_scalar_{smoothing,saturation}_per_vertex` | 0.60 ms against 5.3 ms on a spike-seeded Lipschitz projection |
| `transfer_onto_vertices` | `test_vertices` | `transfer_attributes_per_vertex` | closest-point plus barycentric blend, against a serial closest-point walk |
| `cluster_decimate` | `test_remesh` | `meshing_decimation_clustering` | exact agreement with `open3d.simplify_vertex_clustering` on face *and* vertex count |
| `ambient_occlusion`, `shape_diameter` | `test_proximity` | `compute_scalar_ambient_occlusion`, `..._shape_diameter_function_per_vertex` | **9.3 ms against 662 ms** on `bunny` at 64 rays — the largest ratio in the suite |
| `flip_by_objective` | `test_remesh` | `meshing_edge_flip_by_planar_optimization` | 2.0 ms against 28.6 ms on `saddle_graded` |
| `bad_face_mask`, `remove_t_vertices` | `test_repair` | `compute_selection_bad_faces`, `meshing_remove_t_vertices` | 8.9 ms against 24.2 ms on `saddle_graded` |
| `crease_edges`, `cut_along_edges` | `test_seams` | `compute_selection_crease_per_edge`, `meshing_cut_along_crease_edges` | 1.3 ms against 51.6 ms on `sphere_med`, and **24 output vertices against 32** on a cut cube |
| `uv_seam_edges` | `test_seams` | `compute_selection_by_texture_seams_per_vertex` | **0.94 ms against 300 ms** on `sphere_large`; the filter returns only the seam *vertex set*, unioned with the boundary, so it does strictly less than the triwarp row (which also splits boundaries out and finds foldovers) |
| `filter_normals`, `filter_two_step`, `filter_unsharp_mask` | `test_smoothing` | `apply_normal_smoothing_per_face`, `apply_coord_two_steps_smoothing`, `apply_coord_unsharp_mask` | 8.3 ms against 124 ms; 0.34 ms against 51.8 ms |
| `resample_uniform` | `test_reconstruction` | `generate_resampled_uniform_mesh` | 13.5 ms against 292 ms on `bunny` at a 1% cell |
| `quadric_decimate` | `test_remesh` | `meshing_decimation_quadric_edge_collapse` | **the one port that is slower**: 88 / 346 ms against igl's 50 / 80 — see below |

Two of those rows are worth reading as findings rather than ratios:

- **`quadric_decimate` trades speed for quality.** Its Hausdorff error at 512 faces from a subdiv-4
  icosphere is 0.0147 against `igl.decimate`'s 0.0250 and Open3D's 0.0236, and it is 1.8–4.3× slower
  than igl. The cost is the *pass count*, not the per-pass work: one pass commits an independent set of
  roughly `candidates / valence` collapses, and each pass rebuilds the edge list, adjacency, quadrics
  and two radix sorts. A serial queue pays none of that per collapse.
- **`cut_along_edges` is faster when it cuts *more*.** Cutting every interior edge of `sphere_med`
  costs 1.34 ms against 1.63 ms for a quarter of them, because the corner graph then has no edges and
  the connected-components pass converges immediately instead of contracting `3F` nodes into `V`. So
  the components contraction is the cost in that group, not the ray/key work.

#### Hazards, all measured

- **The MeshSet build is the floor under every row**, at ~0.47 µs/vertex: 0.91 ms on 2 562 vertices,
  4.55 ms on 10 242, **17.1 ms on `bunny`**'s 35 947. Any row cheaper than that is reporting the
  build. It is a large share of the mutating rows — 16.4 of the 17.8 ms a clean
  `meshing_remove_duplicate_faces` costs, 35% of a ten-step Laplacian on `bunny` — so subtract it
  before quoting a ratio.
- **Almost every filter mutates `current_mesh()` in place**, so the MeshSet goes *inside* the timed
  callable (`BenchCase.new_meshset_pml`), the same precedent as the trimesh rebuilds and
  potpourri3d's in-callable solver construction. `BenchCase.meshset_pml` is the shared exception, for
  filters verified geometry-preserving: the `compute_scalar_*` / `compute_normal_*` family, and the
  selection filters — whose cost is *independent of how much is selected* (120 consecutive dilatations
  carrying `sphere_med` from 0.9% to 86% selected cost 0.86–1.21 ms each), which is what lets
  `test_selection` seed once outside the timed region and still read a clean per-hop cost.
- **`compute_curvature_principal_directions_per_vertex` and
  `meshing_decimation_quadric_edge_collapse` default to `autoclean=True`** and delete unreferenced
  vertices under you, so they cannot share a MeshSet even though they only write attributes.
- **Length parameters take a wrapper type.** `ml.PercentageValue(1)` is 1% of the bbox diagonal;
  `ml.PureValue(x)` is absolute (this version has no `AbsoluteValue`). Every row that sizes work by
  length passes `PureValue` fed from `bench_case.mean_edge` so both sides get the identical parameter.
- **Four filters have parameters that silently measure nothing**, and one of them is a value *we*
  chose. `meshing_close_holes(maxholesize=30)` closes *zero* of `rim_short`'s two 512-edge rims
  (0.66 ms); `get_hausdorff_distance(samplenum=8)` samples eight points out of the whole cloud;
  `generate_sampling_poisson_disk(radius=0%)` autoguesses a radius instead of using yours. And
  `generate_surface_reconstruction_ball_pivoting(clustering=0)` reconstructs **nothing** — 0 faces
  against 1 277 at MeshLab's 20% default, returning *faster* for it (9.6 ms against 2.7 ms), so the
  row read as a triwarp win while timing a failure. The clustering fraction is a seed-triangle
  spacing floor, not the optional merge-nearby-vertices post-pass it looks like. Each row overrides
  these and asserts the reference produced output. **Probe a parameter's zero before assuming it means
  "off".**
- **Pass counts are conventions, and mismatching them silently doubles a row's work.** MeshLab's
  `apply_coord_taubin_smoothing(stepsmoothnum=n)` runs `n` lambda-mu **pairs** where triwarp and
  trimesh do one half-step per `iterations` and alternate, so the row passes `_ITERATIONS // 2`;
  passing `_ITERATIONS` to both timed MeshLab doing twice the passes. The mapping is pinned exactly
  (5e-08) in `tests/test_smoothing.py::test_filter_taubin_matches_pymeshlab`.
- **MeshLab has two different uniform coordinate umbrellas.** `apply_coord_laplacian_smoothing` and
  `apply_coord_unsharp_mask` weight each neighbour by its shared-face count and include the vertex
  itself once (`1/(2d+1)` self, `2/(2d+1)` per neighbour on a closed mesh); `apply_coord_taubin_
  smoothing` uses the plain 1-ring mean. Neither is documented; both were recovered by solving
  least-squares for the per-vertex stencil over 12 random position sets on one connectivity (residual
  2e-16). The difference from the 1-ring mean is 8% of the displacement.
- **`compute_matrix_by_icp_between_meshes` does not move the vertices.** It writes the source layer's
  *transformation matrix*, so `vertex_matrix()` reads back byte-identical to the input and a converged
  registration looks like a total no-op; the answer is `transform_matrix()` /
  `transformed_vertex_matrix()`.
- **Non-manifold input is a hard boundary, not a slow path** — the same one that shapes the libigl and
  potpourri3d rows. `meshing_surface_subdivision_midpoint` raises `Mesh has some not 2 manifold faces,
  subdivision surfaces require manifoldness` on **every scan mesh**, so its reference row lives on the
  `scale` axis with `subdivide_to_size`; `generate_polyline_from_planar_section` raises
  `Failed to apply filter` on every scan mesh at any plane, and `test_intersection`'s groups are *all*
  on the scan sweep, so that row does not exist at all.
- **Two more preconditions raise rather than degrade.**
  `compute_texcoord_parametrization_{harmonic,lscm}` reject a **closed** mesh (a boundary loop is
  required); `compute_matrix_by_fitting_to_plane` raises `Cannot compute rotation: there is no
  selection` unless something is selected; and `compute_matrix_by_icp_between_meshes` needs **both**
  layers to carry faces.
- **`harm_function` is a no-op in pymeshlab 2025.7.** `compute_texcoord_parametrization_harmonic`
  returns **bit-identical** texture coordinates at `harm_function=1`, `2` and `3` (max deviation
  exactly 0.0) and identical timings, where libigl's own `k=2` costs 4.3× its `k=1`. So the harmonic
  order axis does not map and the pymeshlab row appears at `k=1` only — a row tracking triwarp's
  `k=2` would be silently reporting the `k=1` solve.
- **Selection morphology is face-based.** `apply_selection_dilatation` / `..._erosion` dilate the
  *face* set (VCGlib's loose vertex-from-face / face-from-vertex pair); handing them a vertex
  selection simply clears it. Seeding needs `compute_selection_by_condition_per_face`, or
  `compute_selection_transfer_vertex_to_face(inclusive=False)` from a vertex seed — `inclusive=True`,
  the default, selects only faces whose *every* vertex is selected.
- **`generate_surface_reconstruction_vcg` was tried and rejected**: it returns **zero faces** on an
  oriented point cloud at every voxel size probed (`PureValue` 0.02 / 0.05 / 0.10), reporting
  `Mesh Saved 'plymcout.ply': 0 vertices, 0 faces`. It reconstructs from all *visible* layers via a
  temporary `.vmi` file and did not produce geometry from a single cloud.
- **Three more silent-no-op parameters, found while porting bucket B.**
  `compute_scalar_by_shape_diameter_function_per_vertex`'s `cone_amplitude` is a **no-op**: the output
  is byte-identical at 90 and 120 degrees. `generate_resampled_uniform_mesh`'s `offset` as a
  `PercentageValue` runs from *full erosion* at 0% to full dilation at 100%, so its own 50% default is
  the **zero** offset and `PercentageValue(0)` erodes a unit sphere to radius 0.30 — pass
  `PureValue(0.0)` for an absolute zero. And `apply_normal_smoothing_per_face` exposes no parameters at
  all, so its row is a single pass against triwarp's twenty.
- **Two filters disagree with the port in ways that are not tolerances.**
  `apply_scalar_smoothing_per_vertex` smooths a *boundary* vertex along the boundary curve alone
  (dividing by 2 rather than by its degree), so the oracle in `tests/test_smoothing.py` runs on closed
  fixtures only. `apply_coord_two_steps_smoothing` at its own four defaults moves a noisy cube
  **further** from clean than the noise was (RMS 0.0244 against 0.0156) because its fitting step rounds
  the corners in — so that comparison asserts crease preservation, which both pass, and a one-sided
  RMS bound.
- **`bunny_decimated` is not edge-manifold** (150 edges with three or more faces), which is why
  `test_seams`' cut *and* `uv_seam_edges` groups run on the synthetic icospheres: both are defined
  through halfedge twins, and triwarp and MeshLab's `meshing_cut_along_crease_edges` alike reject
  such a mesh outright rather than degrading.
- **The registry meshes carry no UVs at all**, so `uv_seam_edges` synthesizes a per-corner spherical
  atlas (seamed at the `±π` longitude wrap) and feeds the *same* array to both sides through
  MeshLab's `w_tex_coords_matrix`. A continuous atlas would leave the seam set empty and a
  per-triangle one would make every interior edge a seam, so neither would time the compaction.
- **Several filters print to stdout regardless of `verbose=False`** (ICP's `Found N pairs`, the point-
  cloud normal estimator's `UG 34 34 34`, the VCG reconstructor's whole volume report). pytest's
  fd-level capture absorbs it; a bare script will not.
- **Three groups are capped** because the reference, not triwarp, is the cost:
  `signed_distance_on_mesh` at `bunny` (768 µs/query on dragon = 85 s a row),
  `get_geometric_measures` at `bunny` (1.12 s a call on dragon, in *two* modules), and the
  `apply_coord_*` smoothers at `bunny` alongside the trimesh rows.

### open3d — two hazards in the tensor API

Both were found while writing the parity asserts, and both fail *silently* rather than raising:

- **A tensor mesh must be held in a name.** Chaining
  `o3d.t.geometry.TriangleMesh.from_legacy(x).fill_holes()` lets the temporary be collected, and the
  result's `vertex["positions"]` then reads freed memory — observed as 2052.1 and 4.4e-41 in the
  leading rows. Bind the intermediate before calling the filter.
- **`fill_holes` winds its cap against the rest of the mesh.** A raw signed volume of its output is
  therefore meaningless: −1.06 on a hemisphere whose true sealed volume is 2.02. Run
  `trimesh.repair.fix_winding` (or triwarp's `make_winding_consistent`) before any volume or
  orientation read. Its cap triangulation is otherwise a valid `B − 2` fill over the existing
  vertices, matching triwarp's and MeshLab's counts exactly.

### potpourri3d

| module | potpourri3d reference |
|---|---|
| `test_geodesic` | `MeshHeatMethodDistanceSolver` (`use_robust=False`, the same discretization as triwarp's), and `MeshFastMarchingDistanceSolver` as a different algorithm for the same task |
| `test_contour` | `marching_triangles` — the only reference for isocontours of an arbitrary vertex field |
| `test_tracing` | `GeodesicTracer.trace_geodesic_from_vertex` — one ray per call, so its row is linear in the ray count by construction |
| `test_signed_heat` | `MeshSignedHeatSolver.compute_distance` — requires every curve segment inside one face, which is why the source curves are edge paths |
| `test_vector_heat` | `MeshVectorHeatSolver.{extend_scalar,transport_tangent_vectors,compute_log_map}` (`use_intrinsic_delaunay=False`) |
| `test_tangent` | `MeshVectorHeatSolver.get_tangent_frames` (`use_intrinsic_delaunay=False`) — an *upper bound*: the frames only come out of the solver's construction, which also factors two sparse systems |
| `test_laplacian` | `cotan_laplacian` (a numpy/scipy build, not C++) and `vertex_areas` (the barycentric lumped mass diagonal) |
| `test_triangles` | `face_areas` |
| `test_edges` | `edges` — in the `edges_unique_manifold` group only, see the hazard table |

`test_intrinsic` uses **libigl** (`cotmatrix_intrinsic`) rather than potpourri3d, which exposes
mollification only inside its heat solver. `test_halfedge` and `test_topology`'s subject matter has no
reference anywhere.

Two things shape every potpourri3d row:

- **Its solvers cache their factorizations**, so the solver is constructed **inside** the timed
  callable — that is where geometry-central does the work triwarp's per-call assembly does. Timing
  only `compute_*` would compare a back-substitution against a full iterative solve. Where triwarp
  has its own reusable precompute the amortized case is measured explicitly instead of argued about:
  `heat_geodesic` carries a `setup=full`/`setup=amortized` parameter layer, and on triwarp's side the
  amortized row is a real API path (`heat_operators` passed back through `heat_geodesic`).
- **Its defaults do more work than triwarp's.** `MeshHeatMethodDistanceSolver(use_robust=True)` and
  `MeshVectorHeatSolver(use_intrinsic_delaunay=True)` mollify and flip to an intrinsic Delaunay
  triangulation first. Every row here passes `False` so both sides discretize the same triangulation;
  the robust path becomes a parameter layer when triwarp grows one.

Measured against it on an RTX 5090: `marching_triangles` is **25x** faster at `sphere_med` and 59x at
`sphere_large`; `heat_geodesic` is 2.8x faster at `saddle` but **1.2x slower** at `saddle_graded`,
where triwarp's CG pays for the conditioning and geometry-central's direct solve does not; the
amortized solve is **19x slower** than potpourri3d's back-substitution at `sphere_small`, which is
the clearest statement in the suite of what an iterative solver costs per extra source set.

`heat_signed_distance` is **13x** faster than the reference on `sphere_med` (81 ms against 1 060 ms),
and its own axis answers a design question rather than a competitive one: a 370-segment source curve
costs *less* than a 6-segment one (71 vs 81 ms), because a source spread over the surface converges in
fewer conjugate-gradient iterations. Pinning the level set costs 4.7x an unconstrained solve
(80.6 against 17.3 ms).

The tangent-space and tracing groups repeat both halves of that story. `trace_geodesic_rays` is **flat
at 1.15 ms from 1 to 4 096 rays** against potpourri3d's 63 → 77 ms (55x → 67x), because one thread
traces one ray and the reference's API traces one ray per call. `log_map` is 6.8x faster at `saddle`
and only 2.6x at `saddle_graded` — triwarp's vector solve pays **2.7x** for the aspect ratio there,
the reference's factorization nothing. `heat_signed_distance_conditioning` shows the same at **3.1x**
(40.3 → 125.8 ms against a flat 525 → 519), which is the worst of the three because that method runs
three solves. `robust_laplacian` runs 9.5-22x faster than
`igl.cotmatrix_intrinsic` on the scan meshes, and `mollify_intrinsic` sits at 0.28 ms on `bunny`,
which is its two host readbacks and essentially nothing else. `intrinsic_delaunay` runs
1.15 / 1.20 / 2.18 ms over the `scale` axis against `igl.intrinsic_delaunay_cotmatrix`'s
7.2 / 58.9 / 260 ms — read with the caveat that igl's call assembles the matrix too, but the
crossover is real: one round of the parallel flip loop finding nothing to do costs about a
millisecond.

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

Modules with no baseline from **any** reference are `test_texture`, `test_polyline`, `test_reduce`,
`test_linalg` and `test_halfedge` (plus `stitch*` in `test_combine`, the morphology groups in
`test_selection`, and the two transport groups in `test_tangent`);
each docstring says which reference was considered and why it is not apples-to-apples. Those are
before/after self-comparisons.

Where the reference is not algorithmically identical, the module docstring says so — `test_repair`
(open3d's dedup is orientation-sensitive), `test_sample` (count- vs radius-parametrized),
`test_hole_filling` and `test_smoothing` (different algorithms for the same task), `test_convex`
(approximate vs exact).
