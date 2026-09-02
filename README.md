<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/lucagrementieri/triwarp/main/docs/assets/logo-lockup-dark.svg">
  <img src="https://raw.githubusercontent.com/lucagrementieri/triwarp/main/docs/assets/logo-lockup.svg" alt="triwarp" height="72">
</picture>

[![PyPI](https://img.shields.io/pypi/v/triwarp)](https://pypi.org/project/triwarp/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://pypi.org/project/triwarp/)
[![Docs](https://img.shields.io/badge/docs-lucagrementieri.github.io%2Ftriwarp-blue)](https://lucagrementieri.github.io/triwarp/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-green)](#license)

**triwarp** is a GPU-first triangular mesh processing library built on
[NVIDIA Warp](https://github.com/NVIDIA/warp). It provides mesh geometry, connectivity,
queries, editing, and solvers as plain array-in / array-out functions backed by Warp kernels,
with a [trimesh](https://trimesh.org)-inspired API. Every function runs on CUDA when a GPU is
available and falls back to CPU otherwise — same code, same results.

## Highlights

- **One dependency.** The runtime depends on `warp-lang` alone; NumPy is only needed to move
  data in and out.
- **Broad coverage.** 50 public modules and over 400 functions spanning primitives, topology,
  repair, remeshing, spatial queries, discrete differential operators, geodesics, sampling,
  surface reconstruction, and registration.
- **Stays on the device.** Functions take Warp arrays and return Warp arrays, so pipelines
  compose without host round-trips; device placement follows the inputs.
- **Optional `Trimesh` object.** A frozen mesh class with lazily cached derived quantities
  (normals, adjacency, boundary loops, manifoldness and watertightness predicates, BVH) for
  when an object API is more convenient than free functions.
- **Fully typed** (`py.typed`), documented per module at
  <https://lucagrementieri.github.io/triwarp/>, and validated function-by-function against the
  established CPU geometry-processing libraries (see below).

## One GPU library instead of six

Mesh processing in Python has long meant stitching together several excellent — but mostly
CPU-bound and stylistically different — libraries. triwarp consolidates the functionality it needs
from each of them behind a single GPU-accelerated API:

| Replaces | For | In triwarp |
|---|---|---|
| [trimesh](https://github.com/mikedh/trimesh) | Mesh bookkeeping: edges, adjacency, boundary, validation, primitives, sampling, proximity | `edges`, `adjacency`, `boundary`, `validation`, `creation`, `sample`, `proximity`, the `Trimesh` class |
| [libigl](https://libigl.github.io/) ([Python bindings](https://github.com/libigl/libigl-python-bindings)) | Discrete differential geometry: cotangent Laplacians, mass matrices, curvature, parametrization, exact/heat geodesics | `laplacian`, `energies`, `curvature`, `parametrization`, `heat.distance` |
| [Open3D](https://www.open3d.org/) | Point clouds, registration, and surface reconstruction: ICP, screened Poisson, ball pivoting | `points`, `registration`, `reconstruction` |
| [potpourri3d](https://github.com/nmwsharp/potpourri3d) (geometry-central) | The heat-method family: vector heat, parallel transport, log maps, signed distance, tangent frames | `heat.vector`, `heat.signed`, `tangent_space` |
| [MeshLab](https://www.meshlab.net/) ([PyMeshLab](https://github.com/cnr-isti-vislab/PyMeshLab)) | Mesh editing filters: isotropic remeshing, decimation, smoothing, hole filling, uniform resampling | `remesh`, `smoothing`, `holes`, `repair` |
| [PyTorch3D](https://pytorch3d.org/) | Batched neighbour and Chamfer primitives, and the mesh regularization losses | `metrics`, `neighbors`, `registration`, `energies` |

These libraries are not runtime dependencies — they are **test oracles**. Every triwarp
function ships with a regression test comparing its output against the corresponding reference
implementation, and a parity gate in the test suite fails the build if a benchmarked
implementation pair is neither value-tested nor explicitly exempted with a written reason. When
triwarp and a reference disagree by definition rather than tolerance, the test says so and
documents the measured difference.

## Install

```bash
uv add triwarp
```

or with `pip`:

```bash
pip install triwarp
```

Requires Python ≥ 3.11 and `warp-lang` ≥ 1.16. A CUDA-capable GPU is recommended but not
required. Mesh file I/O via [meshio](https://github.com/nschloe/meshio) is an optional extra:
`pip install triwarp[io]`.

## Quickstart

```python
import triwarp as tw

# Parametric primitives allocate directly on the default device (CUDA if available).
vertices, faces = tw.creation.icosphere(subdivisions=4)

# Optional object API: derived quantities are computed on first access and cached.
mesh = tw.Trimesh(vertices, faces)
print(mesh.area)                  # 12.551...
print(mesh.is_watertight)         # True
print(mesh.euler_characteristic)  # 2
```

Everything the `Trimesh` class offers is also available as free functions over raw Warp
arrays, so pipelines stay on the device end to end:

```python
import numpy as np
import warp as wp

import triwarp as tw

vertices, faces = tw.creation.icosphere(subdivisions=4)

# Geodesic distance from vertex 0 via the heat method (two sparse CG solves, on-device).
sources = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=vertices.device)
distance = tw.heat.distance.heat_geodesic(vertices, faces, sources)

# Uniform, area-weighted surface sampling.
points, face_ids = tw.sample.sample_surface(vertices, faces, 10_000, seed=0)

# Feature-preserving isotropic remeshing (split / collapse / flip / smooth / reproject).
remeshed_vertices, remeshed_faces = tw.remesh.isotropic_remesh(
    vertices, faces, target_length=0.05
)
```

Interop with NumPy is a `wp.array(...)` / `.numpy()` pair away, and
`tw.mesh.Trimesh.from_warp_mesh` / `mesh.warp_mesh` bridge to Warp's own `wp.Mesh` BVH for ray
and proximity queries.

## What's inside

| Area | Modules | Highlights |
|---|---|---|
| **Primitives** | `creation` | Boxes, spheres, platonic solids, capsules, tori; extrusion, revolution, sweeps, 2D polygon triangulation |
| **Structure & topology** | `mesh`, `vertices`, `edges`, `triangles`, `boundary`, `adjacency`, `halfedge`, `tangent_space`, `homology`, `validation`, `totals`, `selection` | Edge/face adjacency, half-edge structure, boundary loops, manifold/watertight/orientability predicates, submesh selection |
| **Editing & repair** | `repair`, `holes`, `combine`, `remesh`, `smoothing`, `seams` | Hole filling, winding repair, isotropic remeshing, quadric and clustering decimation, Loop and midpoint subdivision, Laplacian/Taubin smoothing, seam cutting |
| **Queries & measures** | `proximity`, `neighbors`, `bounds`, `voxels`, `ray`, `distance`, `intersection`, `curvature`, `convex`, `visibility` | Closest point, signed distance, winding number, ray casting, BVH and hash-grid neighbor queries, chamfer distance (differentiable via `wp.Tape`), curvature measures, convex hulls, ambient occlusion |
| **Operators & fields** | `laplacian`, `energies`, `linalg`, `interpolation`, `parametrization` | Cotangent Laplacian, mass matrices, discrete energies, sparse conjugate-gradient solvers, harmonic and LSCM parametrization |
| **Geodesics & heat methods** | `geodesic_walk`, `heat.distance`, `heat.vector`, `heat.signed` | Surface walks, heat-method geodesic distance, vector heat / parallel transport / log maps, signed heat method |
| **Point clouds & registration** | `points`, `sample`, `reconstruction`, `registration` | Poisson-disk and blue-noise sampling, screened Poisson reconstruction, ball pivoting, marching cubes, Delaunay triangulation, Procrustes and ICP (point-to-point, point-to-plane) |
| **Curves** | `polyline` | Polyline resampling, simplification, and measures |
| **Attributes & I/O** | `texture`, `io` | Per-vertex/face attribute handling, meshio-backed mesh loading |
| **Arrays & infrastructure** | `array`, `reduce`, `grouping`, `graph`, `typing`, `constants` | GPU sort/scan/unique/group primitives, reductions, typed array aliases |

The full API reference, generated per module, lives at
<https://lucagrementieri.github.io/triwarp/>.

## Design

- **Arrays in, arrays out.** The core API is free functions over `wp.array` buffers — no
  mandatory mesh object, no hidden state. Faces are a flat `int32` index buffer, vertices a
  `wp.vec3` array, the same convention everywhere.
- **The device follows the data.** Every kernel launch and allocation inherits the device of
  its inputs; there are no global device switches.
- **Host syncs are budgeted.** Wrappers avoid device-to-host readbacks, and where one is
  unavoidable the reason is documented; bounds a caller already knows can be passed in to skip
  the inference.
- **Measured, not assumed.** Every public module has both a test file and a benchmark file;
  performance work lands only with a before/after measurement, and correctness is asserted on
  values, never just shapes.

## Development

```bash
git clone https://github.com/lucagrementieri/triwarp
cd triwarp
uv sync --all-groups            # runtime + dev + test + docs + bench dependencies

uv run pytest                   # regression tests against the reference libraries
uv run ruff format triwarp tests && uv run ruff check triwarp tests
uv run basedpyright             # type checking (0 errors expected)

DISABLE_MKDOCS_2_WARNING=true uv run mkdocs serve   # preview the docs locally
```

The test environment installs the reference stack (trimesh, libigl, Open3D, potpourri3d,
PyMeshLab, PyVista, MeshLib, PyMeshFix, PyTorch3D, and more) so the comparison suite runs in full.
Benchmarks live in `benchmarks/` and use `pytest-benchmark`; PyTorch3D is the one reference with
CUDA kernels of its own, so it is also the suite's only GPU-against-GPU comparison.

## Links

- **Documentation:** <https://lucagrementieri.github.io/triwarp/>
- **Source:** <https://github.com/lucagrementieri/triwarp>
- **Issues:** <https://github.com/lucagrementieri/triwarp/issues>
- **NVIDIA Warp:** <https://nvidia.github.io/warp/>

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion
in triwarp by you, as defined in the Apache-2.0 license, shall be dual licensed as above,
without any additional terms or conditions.
