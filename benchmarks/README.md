# triwarp benchmarks

Performance benchmarks comparing `triwarp` against the CPU references **trimesh**, **libigl
(`igl`)** and **open3d** on real scan meshes, built on
[pytest-benchmark](https://pytest-benchmark.readthedocs.io).

These are **not** collected by the normal test run (`pytest`'s `testpaths` is `tests/`); run them
by pointing pytest at this directory.

## Install

```bash
uv sync --group bench
```

## Data

Place mesh files in `benchmarks/data/` (gitignored, local-only). The registry in
[`conftest.py`](conftest.py) expects: `bunny_decimated.ply`, `bunny.ply`, `dragon.ply`,
`happy_buddha.ply`, `lucy.ply`. Meshes are read once with `meshio` (the same loader
`triwarp.io` uses). Size categories by triangle count:

| category | faces | meshes |
|---|---|---|
| `small` | `<10k` | — |
| `medium` | `<100k` | bunny_decimated, bunny |
| `large` | `<1M` | dragon |
| `extralarge` | `<10M` | happy_buddha |
| `huge` | `≥10M` | lucy |

## Run

```bash
# Default: triwarp-cuda on all meshes; trimesh/igl/open3d on meshes up to 'large'.
# triwarp-cpu is off by default when CUDA is available (pass --device=both to add it).
uv run pytest benchmarks/

# Quick CPU-only smoke on the medium meshes:
uv run pytest benchmarks/test_edges.py --device=cpu --size=medium

# GPU-only, one mesh size:
uv run pytest benchmarks/test_edges.py --device=cuda --size=large
```

By default the harness prints **one comparison table per (function, mesh)** — each table lists the
libraries side by side — via `--benchmark-group-by=group,param:mesh_name`, set automatically. Pass
your own `--benchmark-group-by=...` to override (e.g. `group` to merge all meshes of a function
into a single table).

## Flags

| flag | default | meaning |
|---|---|---|
| `--device` | `auto` | `triwarp` target(s): `auto`/`cpu`/`cuda`/`both`. `auto` = cuda if CUDA is available, else cpu (triwarp-cpu is not timed alongside cuda; use `both` for that). The trimesh/igl/open3d baselines always run. |
| `--size` | `all` | comma-separated size categories to include (`small,medium,large,extralarge,huge`). Naming a size explicitly also lifts the CPU cap for it. |
| `--cpu-max-size` | `large` | CPU-bound libraries (`triwarp-cpu`, `trimesh`, `igl`, `open3d`) skip meshes larger than this unless the size is named in `--size`. |

## Notes

- **GPU timing** is captured correctly: `wp.synchronize_device` runs inside the timed region and
  one warm-up round covers Warp kernel JIT compilation.
- **trimesh references** use the pure `trimesh.geometry` / `trimesh.grouping` functions (not cached
  `Trimesh` properties) so every round measures real work.
- `edges_unique*` triwarp calls pass `n_vertices=` to avoid a host sync skewing GPU numbers.
- Every benchmark carries an explicit `benchlibs` marker, so adding a library kind to `LIBRARIES`
  never silently generates cases for modules that have no branch for it.
- **libigl is not safe on every registry mesh.** `igl.principal_curvature` *segfaults* on all of
  them (they have non-manifold vertices, which its vertex-ring walk assumes away) — a hard crash
  that takes the pytest process with it, so `test_curvature` draws that comparison on the synthetic
  saddle patches instead. `test_parametrization` documents a milder version of the same problem
  (igl's direct LDLT cannot factor the scan meshes' cotangent systems), and `test_geodesic` a third
  (`igl.heat_geodesics_precompute` raises `Precomputation failed.` on every scan mesh) — both also
  fall back to the saddle patches.
- **Polyline benchmarks are driven by boundary loops, not mesh geometry.** `test_polyline` uses the
  longest boundary loop of the *synthetic* meshes (the cylinder's `2**16`-vertex rim is the
  asymptotic case); the scan meshes' holes are a few vertices each and would measure only launch
  latency.
- **`test_texture` uses projected, non-injective UVs** (vertex `xy` normalized to the unit square)
  because the scan meshes carry no atlas and computing one would dominate the measurement. The
  rasterizer cost is the number of (triangle, covered pixel) pairs, which a projection reproduces
  faithfully; the module docstring explains why that is sound for timing but not for parity.

## open3d coverage

`open3d` is registered in `LIBRARIES` and used for **every** benchmarked function that has a
genuine equivalent, not only where trimesh/libigl are missing — the point is to have a third
independent implementation to spot outliers against. It is marked `cpu_bound` even though the
installed wheel is a CUDA build: the legacy `open3d.pipelines` / `open3d.geometry` APIs used here
are CPU-only (only `open3d.t` has GPU kernels). `BenchCase.mesh_o3d` gives a shared legacy mesh
built from the same NumPy source every other library gets.

| module | open3d reference |
|---|---|
| `test_registration` | `TransformationEstimationPointToPoint.compute_transformation`, `registration_icp` (point-to-point, point-to-plane, `TukeyLoss`) |
| `test_reconstruction` | `create_from_point_cloud_ball_pivoting`, `create_from_point_cloud_poisson` |
| `test_remesh` | `subdivide_midpoint` (`subdivide` only) |
| `test_smoothing` | `filter_smooth_laplacian` (`novol` only) |
| `test_sample` | `sample_points_poisson_disk` |
| `test_graph` | `cluster_connected_triangles` + `select_by_index` (`split` only) |
| `test_hole_filling` | `open3d.t.geometry.TriangleMesh.fill_holes` |
| `test_repair` | `remove_duplicated_triangles` |
| `test_validation` | `is_watertight` |
| `test_vertices` | `compute_vertex_normals` |
| `test_proximity` | `get_axis_aligned_bounding_box` (`aabb_bounds` only) |
| `test_points` | `PointCloud.estimate_normals` (`KDTreeSearchParamKNN`) |
| `test_convex` | `compute_convex_hull` (exact qhull vs the approximate support sweep) |

Modules with **no** open3d equivalent, and why, are documented in each module's docstring:
`test_edges` (no general edge list), `test_boundary` (no loop ordering), `test_grouping` (array
primitive), `test_parametrization` (no harmonic/LSCM/ARAP), `test_distance` (no autodiff),
`test_laplacian` (no cotangent or mass matrix — the smoothing filters build their weights inline),
`test_curvature` (no curvature estimation at all), `test_intersection` (no plane section; its
booleans need the `open3d.t` backend and a coupled remesh), `test_texture` (stores UVs but has no
bake or resample), `test_polyline` (`LineSet` is unordered segments with no length/resample/simplify),
`test_geodesic` (no geodesic distance), `test_reduce` (array primitive), plus the individual
functions noted inline (`winding_number`, `thickness`, `geodesic_ball`, `triangulate_point_cloud`,
`subdivide_to_size`, `flip_to_delaunay`, `isotropic_remesh`, `centroid`, `n_vertices`, `is_volume`,
`bfs`).

Modules with no baseline from **any** of the three references are `test_texture`, `test_polyline`
and `test_reduce` (plus `mesh_with_mesh` in `test_intersection`); each docstring says which
reference was considered and why it is not apples-to-apples. Those are before/after
self-comparisons.

Where the reference is not algorithmically identical, the module docstring says so — `test_repair`
(open3d's dedup is orientation-sensitive), `test_sample` (count- vs radius-parametrized),
`test_hole_filling` and `test_smoothing` (different algorithms for the same task).
