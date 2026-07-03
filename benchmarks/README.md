# triwarp benchmarks

Performance benchmarks comparing `triwarp` against the CPU references **trimesh** and **libigl
(`igl`)** on real scan meshes, built on [pytest-benchmark](https://pytest-benchmark.readthedocs.io).

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
# Default: triwarp-cuda on all meshes; triwarp-cpu/trimesh/igl on meshes up to 'large'.
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
| `--device` | `auto` | `triwarp` target(s): `auto`/`cpu`/`cuda`/`both`. `auto` = both if CUDA is available. The trimesh/igl baselines always run. |
| `--size` | `all` | comma-separated size categories to include (`small,medium,large,extralarge,huge`). Naming a size explicitly also lifts the CPU cap for it. |
| `--cpu-max-size` | `large` | CPU-bound libraries (`triwarp-cpu`, `trimesh`, `igl`) skip meshes larger than this unless the size is named in `--size`. |

## Notes

- **GPU timing** is captured correctly: `wp.synchronize_device` runs inside the timed region and
  one warm-up round covers Warp kernel JIT compilation.
- **trimesh references** use the pure `trimesh.geometry` / `trimesh.grouping` functions (not cached
  `Trimesh` properties) so every round measures real work.
- `edges_unique*` triwarp calls pass `n_vertices=` to avoid a host sync skewing GPU numbers.
- **open3d** is not benchmarked here — it has no edge-extraction equivalents. The harness
  (the `LIBRARIES` registry in `conftest.py`) is structured so it can be added for modules where
  it fits.
