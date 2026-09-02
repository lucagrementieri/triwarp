"""
Benchmark harness: mesh loading, CLI flags, parametrization and a GPU-safe timing fixture.

Run with ``uv run pytest benchmarks/`` (the default ``pytest`` run only collects ``tests/``).

The mesh registries themselves live in [`meshes.py`](meshes.py). There are two, because there are
two questions: the scan meshes sweep triangle count, and the synthetic *feature* meshes each
perturb one cost driver -- component count, boundary loop count or length, graph diameter,
triangle quality, vertex valence, ray depth, collision density -- while pinning the vertex and/or
face count to a shared control. A benchmark picks whichever is relevant:

- ``@pytest.mark.benchaxis("components")`` runs the named ``meshes.AXES`` comparison,
- ``@pytest.mark.benchmeshes("sphere_med", ...)`` names meshes directly for a one-off,
- neither runs the scan sweep, filtered by ``--size`` / ``--cpu-max-size``.

The rule the suite follows is **one group measures one axis, at 2-3 points**. A group that has no
mesh axis at all (``triwarp.creation``) takes ``bench_lib`` instead of ``bench_case`` and sizes its
work with a plain ``pytest.mark.parametrize``.

Flags
-----
``--device={auto,cpu,cuda,both}``
    Which ``triwarp`` targets to time. ``auto`` (default) uses cuda when CUDA is available,
    else falls back to cpu — ``triwarp-cpu`` is not timed alongside cuda by default. Pass
    ``cpu`` for cpu only or ``both`` to time both triwarp targets. The ``trimesh`` / ``igl`` /
    ``open3d`` / ``scipy`` / ``numpy`` / ``potpourri3d`` / ``pymeshlab`` / ``pyvista`` /
    ``meshlib`` / ``pymeshfix`` CPU baselines are always included, and
    ``pytorch3d-cuda`` whenever the installed pytorch3d carries a working CUDA extension --
    ``pytorch3d`` is the one reference with GPU kernels of its own, so it is *not* selected by
    ``--device``, which chooses among triwarp's targets.
``--size=<comma list | all>``
    Restrict meshes to these size categories (``small,medium,large,extralarge,huge``).
``--cpu-max-size=<category>``
    CPU-bound libraries (``triwarp-cpu``, ``trimesh``, ``igl``, ``open3d``, ``scipy``,
    ``numpy``, ``potpourri3d``, ``pymeshlab``, ``pyvista``, ``meshlib``, ``pymeshfix``) skip
    meshes larger than this unless the size was named explicitly in ``--size``. Default ``large``
    — so
    ``happy_buddha`` and ``lucy`` run GPU-only by default while
    ``bunny_decimated``/``bunny``/``dragon`` run on CPU.
"""

from __future__ import annotations

import functools
import operator
import warnings
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
import pytest
import warp as wp
from meshes import (
    ALL_MESHES_BY_NAME,
    AXES,
    BUILDERS,
    DATA_DIR,
    MESH_ORDER,
    MESHES,
    MESHES_BY_NAME,
    SIZE_ORDER,
    MeshSpec,
)

if TYPE_CHECKING:
    import open3d as o3d
    import pymeshlab as ml
    import pytorch3d.structures as p3d_structures
    import pyvista as pv
    from meshlib import mrmeshpy as mm
    from pymeshfix._meshfix import PyTMesh
    from pytest_benchmark.fixture import BenchmarkFixture

# Number of timed rounds and untimed warm-up rounds. The warm-up covers Warp kernel JIT
# compilation (first launch of each kernel) and CPU cache priming. Groups whose work runs into
# hundreds of milliseconds pass a smaller ``rounds`` to ``run`` instead.
_ROUNDS = 10
_WARMUP_ROUNDS = 1


class LibrarySpec(TypedDict):
    """A benchmarked library variant. ``device`` is ``None`` for the CPU-only references."""

    id: str
    kind: str
    device: str | None
    cpu_bound: bool


def skip_larger_than(bench_case: BenchCase, largest: str, reason: str = "") -> None:
    """
    Skip the current case when its mesh is larger than ``largest``.

    On the scan registry the comparison is registry order, which is triangle count. Off it -- the
    synthetic feature meshes -- it is a **vertex-count** comparison against the cap's own count,
    because a feature mesh is not on the size ladder and has no order to index.

    That second branch used to be a silent ``return``, on the reasoning that "every feature mesh is
    deliberately sized to run everywhere it is used, so a cap written for the scan sweep has nothing
    to say about them". That was a claim about the *reference set*, not about the meshes, and it
    expired when the first reference with a quadratic cost curve arrived: ``sphere_large``'s 163 842
    points are sized to run everywhere that "everywhere" is linear or tree-accelerated, and
    ``pytorch3d``'s brute-force ICP over them did not finish in 926 s. A cap is now honoured
    wherever it is written.
    """
    spec = ALL_MESHES_BY_NAME.get(bench_case.mesh_name)
    cap_spec = ALL_MESHES_BY_NAME.get(largest)
    if spec is None or cap_spec is None:
        return
    if bench_case.mesh_name in MESHES_BY_NAME and largest in MESHES_BY_NAME:
        larger = MESH_ORDER.index(bench_case.mesh_name) > MESH_ORDER.index(largest)
    else:
        larger = spec["n_vertices"] > cap_spec["n_vertices"]
    if larger:
        pytest.skip(reason or f"{bench_case.mesh_name} is larger than the {largest} cap")


# ``cpu_bound`` marks references that only ever run on the CPU (the baselines), so the harness
# can skip them on the largest meshes by default. ``open3d`` is marked CPU-bound even though the
# installed wheel is a CUDA build: the legacy ``open3d.pipelines`` / ``open3d.geometry`` APIs the
# baselines use are CPU-only (only the newer ``open3d.t`` tensor API has GPU kernels).
#
# ``scipy`` is a narrow baseline — it is only a *geometry* reference for the k-NN queries
# (``spatial.KDTree``), which is what ``tests/test_neighbors.py`` already validates against. Every
# benchmark carries an explicit ``benchlibs`` marker, so it generates cases only where a branch
# exists.
#
# ``potpourri3d`` (pybind11 bindings over geometry-central) is CPU-only and the only reference for
# the heat-method family; note that its solver objects cache their factorizations, so a benchmark
# must construct the solver *inside* the timed callable to measure the work triwarp does per call.
#
# ``pymeshlab`` (pybind11 bindings over MeshLab / VCGlib) is CPU-only and the broadest reference in
# the list -- 281 filters, of which 61 map onto something triwarp already has. Two things shape
# every row: almost every filter *mutates* ``current_mesh()`` in place, and building the ``MeshSet``
# costs ~0.47 us/vertex (17 ms on ``bunny``). So the MeshSet is built inside the timed callable via
# ``BenchCase.new_meshset_pml`` unless the filter is verified pure, and any row cheaper than the
# build cost is reporting the build. See ``BenchCase.new_meshset_pml`` for the full rule.
#
# ``pyvista`` (VTK 9.6 through its Python wrapper) is CPU-only and single-threaded, and it wraps the
# **same VTK as vedo** -- so a group carries one of the two, never both, or the ratio is comparing a
# library against itself. Three measured facts shape its rows. Nothing is cached: repeat calls
# recompute (``cell_quality`` 10.1 / 7.2 ms, ``decimate`` 210 / 201 ms back to back), so one
# ``PolyData`` may be shared across the pure family and no row is accidentally timing a cache hit.
# Almost everything returns a *new* object and leaves its input alone (``inplace=False`` is the
# default) -- verified on twelve filters, with ``edge_mask`` the one exception, which writes
# ``point_ind`` into the input. And the build is cheap, 0.057 us/vertex through
# ``PolyData.from_regular_faces`` (~8x cheaper than pymeshlab's 0.47 us/vertex), so ``mesh_pv`` is a
# shared cached property rather than a per-call rebuild and there is no "the row is reporting the
# build" caveat above ~0.2 ms.
#
# ``meshlib`` (pybind11 bindings over MeshLib's C++ core) is CPU-bound but, uniquely in this list,
# **multi-threaded** -- 143 OS threads measured live during ``findSelfCollidingTrianglesBS`` +
# ``computePerVertNormals`` on an 82k-face icosphere, where trimesh / igl / pymeshlab / pyvista are
# all effectively single-threaded. So a ``triwarp-cuda`` vs ``meshlib`` ratio is a fair fight in a
# way the other CPU ratios are not, and the other edge of the same knife is that a ``triwarp-cpu``
# row loses to it on any parallel op regardless of algorithm; section 13's "decide on the CUDA
# number" is what applies. ``meshlib.mrcudapy`` exists and is deliberately **not** used -- the plain
# ``mrmeshpy`` free functions are the reference, and mixing in a CUDA module would make the row
# incomparable with the other five. Almost every one of those free functions mutates its ``Mesh`` in
# place, so the accessor is ``BenchCase.new_mesh_ml()`` (the ``new_meshset_pml`` shape) and there is
# no cached property.
#
# ``pymeshfix`` (nanobind bindings over MeshFix / the TMesh kernel) is CPU-only and
# **single-threaded** -- measured on ``bunny``, ``select_intersecting_triangles`` runs 435 ms of
# wall clock against 435 ms of ``process_time``, a ratio of 1.00 -- so it belongs with trimesh /
# igl / pymeshlab / pyvista rather than with meshlib. It is also the *narrowest deep* reference
# here: nine bound algorithms, all of them repair, against MeshLib's 246.
#
# One measured fact decides every row: **the load is a large and often dominant share of it.**
# ``load_array`` is not a load but a connectivity repair, and a ``PyTMesh`` accepts exactly one load
# and one mutating call (a second ``load_array`` raises), so the build has to go inside the timed
# callable via ``BenchCase.new_tmesh_pmf()`` -- there is no cached form and no choice about it.
# Measured, load / operation: ``bunny_decimated`` 67.9 / 64.2 ms for
# ``select_intersecting_triangles``, 67.9 / 5.8 ms for ``fill_small_boundaries``, 67.9 / 6.7 ms for
# ``remove_smallest_components``; ``bunny`` 439.6 / 435.3, 439.6 / 51.4, 439.6 / 60.5. So a
# pymeshfix row is **only created where the operation is at least ~30 % of the round** -- the
# intersection family (49-50 %) and ``clean_from_arrays`` (68-73 %) qualify, the hole-fill (8-10 %)
# and component-removal (9-12 %) rows do not and carry
# ``pytest.mark.parity(..., benchmarked=False)`` with the ratio in the reason instead. Every row
# states its measured share in its docstring, and rows are capped at ``bunny``: ``dragon`` is
# seconds of load plus seconds of query per round.
#
# The shares for the three algorithms *not* timed above, so the next author does not re-measure them
# (same mesh, ``bunny_decimated``, best of 3, load 53.1 ms in that session):
# ``join_closest_components`` 900.7 ms round / 847.6 ms operation = **94.1 %**, ``clean`` 225.9 /
# 172.8 = **76.5 %**, ``strong_intersection_removal`` 174.4 / 121.3 = **69.5 %**,
# ``fix_connectivity`` 124.4 / 71.3 = **57.3 %**, ``strong_degeneracy_removal`` 65.4 / 12.3 =
# 18.8 %. Four of five clear the bar on cost -- and **a share is necessary, not sufficient.**
# ``fix_connectivity`` is the counter-example: 57.3 % of a round doing provably nothing, since
# ``load_array`` already ran it (measured, 8 372 V / 16 220 F / 86 boundaries in and out,
# unchanged). A row for it would price a no-op. ``clean`` and ``strong_intersection_removal`` are
# real work whose *result* is not comparable with any single triwarp function -- ``clean`` is
# ``strongDegeneracyRemoval`` + ``strongIntersectionRemoval`` iterated to a fixpoint and nothing
# else, where ``clean_from_arrays`` wraps it in component removal and two hole-filling passes.
#
# ``numpy`` is the narrowest baseline of all: it is only a reference for the *array primitives*
# (``triwarp.reduce``), where a host reduction over an already-resident NumPy buffer is the honest
# CPU floor. It is deliberately not a geometry reference — every other CPU baseline is already
# built on NumPy, so timing it against a geometry wrapper would measure nothing new.
#
# ``pytorch3d`` is the **only** reference here with CUDA kernels of its own, so it takes *two* rows
# the way triwarp does. The ``-cuda`` row is the point: a ``triwarp-cuda`` vs ``pytorch3d-cuda``
# ratio is the one GPU-against-GPU comparison in the suite, and it is not a foregone conclusion.
# pytorch3d's ``_C`` carries **no spatial structure on either device** -- no tree, no grid, just the
# pairwise loop -- so the ratio is a crossover rather than a constant. Measured on this box, min of
# 12 interleaved reps, uniform cloud in a 5-unit box, each side synchronized on its own stream and
# each building whatever index it builds inside the timed call:
#
#   | (p3d / triwarp-cuda)       | 20 000 points          | 200 000 points          |
#   |----------------------------|------------------------|-------------------------|
#   | ``knn_points(K=8)``        | 2.31 / 3.29 ms  0.70x  | 74.65 / 0.87 ms  85.5x  |
#   | ``chamfer_distance``       | 3.39 / 5.36 ms  0.63x  | 108.12 / 1.19 ms 90.5x  |
#   | ``ball_query(r=0.15)``     | 4.97 / 0.31 ms  16.1x  | 179.52 / 2.12 ms 84.6x  |
#   | ``sample_farthest_points`` | 14.35 / 5.43 ms 2.64x  | 104.93 / 35.99 ms 2.92x |
#
# **pytorch3d wins the k-NN and chamfer rows at 20 000 points** -- brute force with perfect
# coalescing beats a BVH descent while the whole problem still fits the device's bandwidth -- and
# loses them by ~90x at 200 000. So the neighbour and chamfer groups need the point count as an
# *axis*: a one-size row reports whichever side of the crossover it landed on, which is section
# 13's "a single operating point cannot show a trend".
#
# **Half of that crossover is triwarp's, not pytorch3d's, and the row must not be read as a
# statement about brute force.** pytorch3d is a clean quadratic (0.63, 2.26, 5.59, 20.18, 73.82 ms
# at 5 k / 20 k / 50 k / 100 k / 200 k); triwarp's ``query_nearest`` is **non-monotonic** over the
# same sweep at ``k = 8`` -- 0.837, 3.262, 7.567, **0.472**, 0.801 ms -- a 16x *drop* between 50 k
# and 100 k on the same box and extent. So the honest reading of the 20 000-point row is "triwarp is
# 7x off its own 100 000-point cost here", not "pytorch3d is faster".
#
# Two qualifications, both from re-running that sweep **interleaved across size and backend in one
# pre-warmed process** rather than sequentially, and both of which the earlier reading had wrong:
# it is a ``k >= 8`` effect and **does not exist at k = 1** (0.229, 0.236, 0.249, 0.269, 0.346 ms,
# flatly monotonic -- so it must not be read onto the ``*_k1`` rows); and it is **hash-grid
# specific** rather than "a stage the two structures share", because the ``bvh`` backend is
# monotonic over the identical sweep (0.695, 0.774, 0.854, 1.368, 2.023) and therefore beats the
# grid 1.2x / 4.2x / 8.9x at 5 k / 20 k / 50 k before losing 2.9x / 2.5x at 100 k / 200 k.
#
# The located mechanism: the grid's cell width *is* ``initial_radius``, and a row whose true k-th
# distance runs past ``_knn_widest_grid_radius(cell, n)`` abandons the walk for an exact linear scan
# of the cloud -- 50 000**2 tests in 7.567 ms is 3.3e11 tests/s, a scan and not a search. See memory
# ``hashgrid-nearest-radius-is-cubic``, and ``benchmarks/README.md`` for the full correction.
#
# **There is deliberately no ``pytorch3d-cpu`` row.** pytorch3d is registered for its CUDA kernels
# -- it is the suite's one GPU-against-GPU comparison -- and its host path is a reference
# implementation rather than a tuned one, so a ``-cpu`` row measures the same missing spatial index
# at Θ(N²) and adds nothing the other eleven CPU baselines do not already say better. The cost of
# keeping it was measured rather than assumed: ``knn_points`` 299.8 / 1 189.5 / 4 562.8 ms at 10 k /
# 20 k / 40 k points and ``chamfer_distance`` 557.8 / 2 319.1 / 9 077.1 ms -- 3.84-4.16x per
# doubling against the 4x a quadratic predicts, which extrapolates to ~3.5 s per ``knn`` round on
# ``bunny`` and ~9 minutes per round on ``dragon``. Two of those rows ran unbounded in round 9 and
# cost 1 h 43 min of wall clock plus two whole modules' JSON, because ``--benchmark-json`` is
# written at session end. So pytorch3d contributes its ``-cuda`` row or no row at all, and where the
# CUDA extension is missing the reference is simply absent -- ``_pytorch3d_cuda_available`` gates
# it, and nothing falls back to the host.
#
# Two things every pytorch3d row has to do. **Synchronize torch's stream** -- ``BenchCase.run``
# does it on the ``pytorch3d`` kind, because ``wp.synchronize_device`` synchronizes *Warp's* stream
# and torch launches are asynchronous, so without it a ``-cuda`` row times the launch and not the
# kernel (the 20 000-point chamfer row measured 3.28 ms synchronized). And **state the measured
# share**, on the pymeshfix precedent: pytorch3d's per-round overhead is the batched ``[None]``
# wrap plus the ``Meshes`` construction and its cached ``*_packed()`` derivations, and on a
# ``-cuda`` row also the host-to-device copy.
LIBRARIES: list[LibrarySpec] = [
    {"id": "triwarp-cpu", "kind": "triwarp", "device": "cpu", "cpu_bound": True},
    {"id": "triwarp-cuda", "kind": "triwarp", "device": "cuda:0", "cpu_bound": False},
    {"id": "trimesh", "kind": "trimesh", "device": None, "cpu_bound": True},
    {"id": "igl", "kind": "igl", "device": None, "cpu_bound": True},
    {"id": "open3d", "kind": "open3d", "device": None, "cpu_bound": True},
    {"id": "scipy", "kind": "scipy", "device": None, "cpu_bound": True},
    {"id": "numpy", "kind": "numpy", "device": None, "cpu_bound": True},
    {"id": "potpourri3d", "kind": "potpourri3d", "device": None, "cpu_bound": True},
    {"id": "pymeshlab", "kind": "pymeshlab", "device": None, "cpu_bound": True},
    {"id": "pyvista", "kind": "pyvista", "device": None, "cpu_bound": True},
    {"id": "meshlib", "kind": "meshlib", "device": None, "cpu_bound": True},
    {"id": "pymeshfix", "kind": "pymeshfix", "device": None, "cpu_bound": True},
    {"id": "pytorch3d-cuda", "kind": "pytorch3d", "device": "cuda:0", "cpu_bound": False},
]
LIBRARIES_BY_ID = {lib["id"]: lib for lib in LIBRARIES}


# ---------------------------------------------------------------------------
# mesh loading (shared numpy source; per-device warp buffers built from it)
# ---------------------------------------------------------------------------

_numpy_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_wp_cache: dict[tuple[str, str, str], wp.array] = {}
_mean_edge_cache: dict[str, float] = {}
_o3d_cache: dict[str, o3d.geometry.TriangleMesh] = {}
_pml_cache: dict[str, ml.MeshSet] = {}
_pv_cache: dict[str, pv.PolyData] = {}
_p3d_cache: dict[tuple[str, str], p3d_structures.Meshes] = {}


def _load_numpy(name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Read ``(vertices_f64, faces_i64)`` once with meshio, cached across the session.

    Uses the same ``meshio.read`` path as ``triwarp.io.load_mesh_data``; kept at NumPy level
    (float64 vertices / int64 faces) so igl and trimesh get their arrays directly and the warp
    buffers are built from the same source. Feature meshes are generated by ``meshes.BUILDERS``
    instead of read.
    """
    if name not in _numpy_cache:
        builder = BUILDERS.get(name)
        if builder is not None:
            vertices, faces = builder()
            _numpy_cache[name] = (
                np.ascontiguousarray(vertices, dtype=np.float64),
                np.ascontiguousarray(faces, dtype=np.int64),
            )
            return _numpy_cache[name]

        import meshio

        mesh = meshio.read(DATA_DIR / MESHES_BY_NAME[name]["filename"])
        faces = mesh.cells_dict.get("triangle")
        if faces is None or len(faces) == 0:
            raise ValueError(f"{name!r} has no triangle faces to benchmark.")
        vertices = np.ascontiguousarray(mesh.points, dtype=np.float64)
        _numpy_cache[name] = (vertices, np.ascontiguousarray(faces, dtype=np.int64))
    return _numpy_cache[name]


def _faces_wp(name: str, device: str) -> wp.array[wp.int32]:
    """Flat ``wp.int32`` face buffer on ``device``, cached per ``(name, device)``."""
    key = ("faces", name, device)
    if key not in _wp_cache:
        _, faces = _load_numpy(name)
        _wp_cache[key] = wp.array(
            np.ascontiguousarray(faces.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device
        )
    return _wp_cache[key]


def _mean_edge(name: str) -> float:
    """
    Mean undirected edge length of a mesh, from the shared NumPy source.

    Computed on the host so every library variant of a case gets the *same* value: benchmarks that
    size their work by edge length (remeshing targets, ball-pivoting radii) must not hand triwarp
    and its reference slightly different parameters.
    """
    if name not in _mean_edge_cache:
        vertices, faces = _load_numpy(name)
        triangles = vertices[faces]
        edges = np.concatenate(
            (
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 1],
                triangles[:, 0] - triangles[:, 2],
            )
        )
        _mean_edge_cache[name] = float(np.linalg.norm(edges, axis=1).mean())
    return _mean_edge_cache[name]


def _new_mesh_o3d(name: str) -> o3d.geometry.TriangleMesh:
    """
    Build a fresh legacy ``open3d.geometry.TriangleMesh`` from the shared NumPy source.

    Imported lazily (like ``meshio`` above) so the ~1 s open3d import is only paid by runs that
    actually include an open3d case.
    """
    import open3d as o3d

    vertices, faces = _load_numpy(name)
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(np.ascontiguousarray(faces, dtype=np.int32)),
    )


def _new_meshset_pml(name: str) -> ml.MeshSet:
    """
    Build a fresh single-mesh ``pymeshlab.MeshSet`` from the shared NumPy source.

    Imported lazily (like ``meshio`` and ``open3d`` above) so the pymeshlab import and its Qt
    dependencies are only paid by runs that include a pymeshlab case. MeshLab wants float64
    positions and takes the ``(n_faces, 3)`` index array as-is.
    """
    import pymeshlab as ml

    vertices, faces = _load_numpy(name)
    meshset = ml.MeshSet()
    meshset.add_mesh(ml.Mesh(vertices, faces))
    return meshset


def _new_mesh_pv(name: str) -> pv.PolyData:
    """
    Build a ``pyvista.PolyData`` from the shared NumPy source, float64 positions preserved.

    Imported lazily (like ``meshio``, ``open3d`` and ``pymeshlab`` above) so the pyvista/VTK import
    is only paid by runs that include a pyvista case. Goes through ``from_regular_faces`` rather
    than the padded ``[3, i, j, k]`` cell array: 0.184 ms against 2.34 ms on a 40 962-vertex mesh.
    """
    import pyvista as pv

    vertices, faces = _load_numpy(name)
    return pv.PolyData.from_regular_faces(vertices, np.ascontiguousarray(faces, dtype=np.int32))


def _new_mesh_p3d(name: str, device: str) -> p3d_structures.Meshes:
    """
    Build a ``pytorch3d.structures.Meshes`` on ``device`` from the shared NumPy source.

    Positions land as **float32** and faces as int64, which is what triwarp's ``wp.vec3`` /
    ``wp.int32`` buffers hold and what pytorch3d's own kernels want; a float64 ``Meshes`` keeps
    float64 through ``verts_packed()``, so matching the storage keeps a ratio comparing the same
    arithmetic on both sides.

    Imported lazily (like ``meshio``, ``open3d``, ``pymeshlab``, ``pyvista``, ``meshlib`` and
    ``pymeshfix`` above) so the ~4 s torch + pytorch3d import is only paid by runs that include a
    pytorch3d case.
    """
    import pytorch3d.structures as p3d_structures
    import torch

    vertices, faces = _load_numpy(name)
    return p3d_structures.Meshes(
        verts=[torch.as_tensor(np.ascontiguousarray(vertices, dtype=np.float32), device=device)],
        faces=[torch.as_tensor(np.ascontiguousarray(faces, dtype=np.int64), device=device)],
    )


def mesh_ml_from_numpy(vertices: np.ndarray, faces: np.ndarray) -> mm.Mesh:
    """
    Build a ``meshlib.mrmeshpy.Mesh`` from a NumPy pair, **vertices first**.

    The benchmark-side twin of ``tests.conversions.numpy_to_meshlib`` (the two suites do not import
    each other), and it exists for the same reason: ``mn.meshFromFacesVerts`` takes *faces* first,
    the reverse of every other builder here, and a swapped call raises nothing because the two
    arrays differ in shape only when the counts differ. Keeping that swap in one place per suite is
    the point. Use this for a row whose mesh is constructed rather than loaded; a row on a registry
    mesh wants ``BenchCase.new_mesh_ml()`` instead.

    Imported lazily (like ``meshio``, ``open3d``, ``pymeshlab`` and ``pyvista`` above) so the
    0.256 s meshlib import is only paid by runs that include a meshlib case. Both dtypes are
    permissive -- float64 positions and int64 indices go through as the loader holds them.
    """
    from meshlib import mrmeshnumpy as mn

    return mn.meshFromFacesVerts(
        np.ascontiguousarray(np.asarray(faces).reshape(-1, 3)), np.ascontiguousarray(vertices)
    )


def face_bitset_ml(mask_np: np.ndarray) -> mm.FaceBitSet:
    """
    Load a dense face mask into a ``FaceBitSet`` through the packed blocks, not a per-face loop.

    The benchmark-side twin of ``tests.conversions.numpy_to_meshlib_bitset`` (the two suites do not
    import each other). Three details that are each a silent wrong answer on their own:
    ``BitSet.fromBlocks`` takes ``uint64`` blocks and rejects a NumPy array (pass ``.tolist()``),
    ``bitorder="little"`` is not NumPy's default and is not optional, and ``fromBlocks`` rounds the
    size up to whole 64-bit blocks so the domain has to be restored afterwards.

    Measured against the per-face ``set()`` loop it replaces: 258x at 110 592 elements.
    """
    from meshlib import mrmeshpy as mm

    packed_np = np.packbits(np.ascontiguousarray(mask_np, dtype=bool), bitorder="little")
    packed_np = np.pad(packed_np, (0, (-packed_np.size) % 8)).view(np.uint64)
    bitset_ml = mm.BitSet.fromBlocks(mm.std_vector_unsigned_long(packed_np.tolist()))
    bitset_ml.resize(int(mask_np.size))
    return mm.FaceBitSet(bitset_ml)


def _new_mesh_ml(name: str) -> mm.Mesh:
    """Build a fresh ``meshlib.mrmeshpy.Mesh`` from the shared NumPy source."""
    return mesh_ml_from_numpy(*_load_numpy(name))


def tmesh_pmf_from_numpy(vertices: np.ndarray, faces: np.ndarray) -> PyTMesh:
    """
    Build a ``pymeshfix._meshfix.PyTMesh`` from a NumPy pair, **vertices first** and quiet.

    The benchmark-side twin of ``tests.conversions.numpy_to_pymeshfix`` (the two suites do not
    import each other). ``set_quiet(True)`` is not cosmetic: the kernel narrates to stderr and one
    of its messages is reported inverted, so a row that left it on would print
    "MeshFix could not fix everything" once per round on meshes it repaired successfully.

    ``load_array`` is itself a connectivity repair -- it renumbers and may change both counts -- so
    a row asserting anything about the result reads the counts off the ``PyTMesh``, never off the
    input arrays.

    Imported lazily (like ``meshio``, ``open3d``, ``pymeshlab``, ``pyvista`` and ``meshlib`` above)
    so the pymeshfix import is only paid by runs that include a pymeshfix case.
    """
    from pymeshfix import _meshfix

    tin_pmf = _meshfix.PyTMesh()
    tin_pmf.set_quiet(True)
    tin_pmf.load_array(
        np.ascontiguousarray(vertices, dtype=np.float64),
        np.ascontiguousarray(np.asarray(faces).reshape(-1, 3), dtype=np.int32),
    )
    return tin_pmf


def _new_tmesh_pmf(name: str) -> PyTMesh:
    """Build a fresh ``PyTMesh`` from the shared NumPy source."""
    return tmesh_pmf_from_numpy(*_load_numpy(name))


def points_torch_from_numpy(points_np: np.ndarray, device: str) -> Any:
    """
    Upload an ``(n, 3)`` cloud as the ``(1, n, 3)`` float32 tensor ``pytorch3d.ops`` wants.

    The benchmark-side twin of ``tests.conversions.points_to_torch`` (the two suites do not import
    each other). **Every** ``pytorch3d.ops`` entry point is batched with a leading minibatch axis,
    so a single cloud goes in as ``x[None]`` and its answer comes out as ``result[0]``; a bare
    ``(P, 3)`` handed to ``knn_points`` is silently read as ``(N=P, P1=3, D)`` and compares three
    points at full speed, which would read as pytorch3d being 1 000x faster than it is. Keeping the
    wrap in one place per suite is the point.

    The upload is **not free and it is inside every row**: measured 0.35 ms for 35 947 points to
    the host and 0.60 ms to ``cuda:0``, against a ``knn_points`` call of 2-3 ms at 20 000 points --
    so it is a real share of a small row and negligible on a large one. State the share in the
    group docstring, on the pymeshfix precedent.

    Imported lazily (like ``meshio``, ``open3d``, ``pymeshlab``, ``pyvista``, ``meshlib`` and
    ``pymeshfix`` above) so the ~4 s torch import is only paid by runs that include a pytorch3d
    case.
    """
    import torch

    return torch.as_tensor(
        np.ascontiguousarray(points_np, dtype=np.float32), device=device
    ).unsqueeze(0)


def _vertices_wp(name: str, device: str) -> wp.array[wp.vec3]:
    """``wp.vec3`` (float32) vertex buffer on ``device``, cached per ``(name, device)``."""
    key = ("verts", name, device)
    if key not in _wp_cache:
        vertices, _ = _load_numpy(name)
        _wp_cache[key] = wp.array(
            np.ascontiguousarray(vertices, dtype=np.float32), dtype=wp.vec3, device=device
        )
    return _wp_cache[key]


# ---------------------------------------------------------------------------
# flags & (mesh, library) parametrization
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    # --device is registered by the repo-root conftest.py: argparse rejects a duplicate option
    # string, and tests/conftest.py reads the same flag. See that file for why both suites' ``auto``
    # selects one device. Registering it here too made ``pytest tests benchmarks`` die with
    # ``conflicting option string: --device`` before collecting anything.
    group = parser.getgroup("triwarp-bench")
    group.addoption(
        "--size",
        action="store",
        default="all",
        help="comma-separated mesh size categories to include, or 'all' (default).",
    )
    group.addoption(
        "--cpu-max-size",
        action="store",
        default="large",
        choices=SIZE_ORDER,
        help="largest mesh size CPU-bound libraries run on by default (default: large).",
    )


def _selected_sizes(config: pytest.Config) -> tuple[set[str], bool]:
    """Return ``(sizes, explicit)`` — the requested size set and whether it was user-specified."""
    raw = str(config.getoption("--size")).strip().lower()
    if raw == "all":
        return set(SIZE_ORDER), False
    sizes = {s.strip() for s in raw.split(",") if s.strip()}
    unknown = sizes - set(SIZE_ORDER)
    if unknown:
        raise pytest.UsageError(f"unknown --size categories: {sorted(unknown)}")
    return sizes, True


@functools.cache
def _pytorch3d_cuda_available() -> bool:
    """
    Whether the installed pytorch3d has a **working** CUDA extension, by running one kernel.

    ``torch.cuda.is_available()`` is the wrong probe and this is not a hypothetical: a
    ``pipablepytorch3d`` wheel whose arch list stopped at ``sm_90`` returned ``True`` on this
    RTX 5090 and then failed every kernel with *"no kernel image is available"*. The other failure
    mode is a build that compiled the CPU extension only -- selected silently whenever
    ``setup.py`` finds no ``CUDA_HOME``, which is the normal state of a stock CI runner -- and it
    raises ``RuntimeError: Not compiled with GPU support.`` on a CUDA tensor. Both would otherwise
    turn a whole benchmark file's ``pytorch3d-cuda`` rows into errors, so the row is gated on the
    only probe that distinguishes them: launch a two-point ``knn_points`` and see.
    """
    import torch

    if not torch.cuda.is_available():
        return False
    import pytorch3d.ops as p3d_ops

    try:
        points_t = torch.zeros(1, 2, 3, device="cuda:0")
        p3d_ops.knn_points(points_t, points_t, K=1)
    except RuntimeError:
        return False
    return True


def _selected_libraries(config: pytest.Config) -> list[LibrarySpec]:
    device = str(config.getoption("--device"))
    cuda_available = wp.is_cuda_available()
    if device == "auto":
        # triwarp-cpu is disabled by default: run it only as a fallback when there is no CUDA
        # device. Pass --device=cpu or --device=both to time it explicitly.
        include_cpu, include_cuda = not cuda_available, cuda_available
    elif device == "cpu":
        include_cpu, include_cuda = True, False
    elif device == "cuda":
        include_cpu, include_cuda = False, True
    else:  # both
        include_cpu, include_cuda = True, True

    if include_cpu and cuda_available:
        # A ``triwarp-cpu`` row timed in a process where CUDA has been initialised is *inflated*, so
        # it is not merely a slow row -- it is a wrong one, and it reads as triwarp losing to the
        # CPU references. The cost behaves like a per-launch charge, so the factor scales with
        # launch count rather than with work, and both ends are measured:
        #
        #   - launch-light rows barely notice: ``edges_unique`` on bunny_decimated 11.22 against
        #     **8.76 ms** hidden (1.28x, siblings 1.40-1.41x), and ``marching_cubes`` only
        #     **1.20x / 1.04x** at 64/128 -- under that row's own noise, where ``meshlib-128``
        #     itself moved 7.61 to 12.48 ms between the two runs;
        #   - an iterative solver pays all of it: ``heat_signed_distance`` 50.57 s against
        #     **1.40 s** on one call, and ``tests/test_heat_signed.py --device=cpu`` 166.77 s
        #     against **8.53 s** (19.6x).
        #
        # Unchanged by ``warp.config.launch_array_access_mode`` (``RELAXED`` 50.34 s, ``CHECKED``
        # 49.77 s on the solver), so it is CUDA presence and not the launch guard. This is why the
        # default ``auto`` picks ``triwarp-cpu`` only when there is no CUDA device. A *warning*
        # rather than a refusal: a deliberate same-process CPU-vs-CUDA comparison is still a
        # legitimate thing to ask for, as long as the asker knows the CPU side is not publishable.
        warnings.warn(
            "triwarp-cpu rows are being timed in a CUDA-initialised process, which inflates them "
            "(measured 1.0-1.4x on launch-light ops, ~20x on an iterative solver) and makes them "
            "incomparable with the CPU references. Run 'uv run python benchmarks/devices.py' "
            "instead, which times the CPU target in a separate process with "
            'CUDA_VISIBLE_DEVICES="".',
            UserWarning,
            stacklevel=2,
        )

    selected = []
    for lib in LIBRARIES:
        if lib["id"] == "triwarp-cpu" and not include_cpu:
            continue
        if lib["id"] == "triwarp-cuda" and not (include_cuda and cuda_available):
            continue
        if lib["id"] == "pytorch3d-cuda" and not _pytorch3d_cuda_available():
            continue
        # Every other baseline is always included: --device selects among triwarp's targets, and
        # pytorch3d-cuda is gated on its own extension rather than on that flag.
        selected.append(lib)
    return selected


def _selected_meshes(config: pytest.Config, lib: LibrarySpec) -> list[MeshSpec]:
    sizes, explicit = _selected_sizes(config)
    cpu_max_rank = SIZE_ORDER.index(str(config.getoption("--cpu-max-size")))
    meshes = []
    for mesh in MESHES:
        if mesh["size"] not in sizes:
            continue
        # Cap CPU-bound libraries at --cpu-max-size unless the size was explicitly requested.
        if lib["cpu_bound"] and not explicit and SIZE_ORDER.index(mesh["size"]) > cpu_max_rank:
            continue
        meshes.append(mesh)
    return meshes


def _supported_kinds(metafunc: pytest.Metafunc) -> set[str]:
    marker = metafunc.definition.get_closest_marker("benchlibs")
    if marker is None:
        return {lib["kind"] for lib in LIBRARIES}
    return set(marker.args)


def _mesh_names(metafunc: pytest.Metafunc) -> tuple[str, ...] | None:
    """Explicit mesh names for a test from ``benchaxis`` / ``benchmeshes``, or ``None``."""
    axis = metafunc.definition.get_closest_marker("benchaxis")
    explicit = metafunc.definition.get_closest_marker("benchmeshes")
    if axis is not None and explicit is not None:
        raise pytest.UsageError(
            f"{metafunc.definition.nodeid}: benchaxis and benchmeshes are mutually exclusive."
        )
    if axis is not None:
        if len(axis.args) != 1 or axis.args[0] not in AXES:
            raise pytest.UsageError(
                f"benchaxis takes one name from {sorted(AXES)}, got {list(axis.args)}"
            )
        return AXES[axis.args[0]]
    return tuple(explicit.args) if explicit is not None else None


def _test_meshes(metafunc: pytest.Metafunc, lib: LibrarySpec) -> list[MeshSpec]:
    """Meshes for one test: the axis / explicit names, else the registry filtered by CLI flags."""
    names = _mesh_names(metafunc)
    if names is None:
        return _selected_meshes(metafunc.config, lib)
    meshes = []
    for name in names:
        spec = ALL_MESHES_BY_NAME.get(name)
        if spec is None:
            raise pytest.UsageError(f"unknown mesh name: {name!r}")
        meshes.append(spec)
    return meshes


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "library" not in metafunc.fixturenames:
        return
    supported = _supported_kinds(metafunc)
    libraries = [lib for lib in _selected_libraries(metafunc.config) if lib["kind"] in supported]

    if "mesh_name" not in metafunc.fixturenames:
        # Mesh-free benchmark (``bench_lib``): parametrize over libraries alone. The mesh registry
        # and its size filters have nothing to select here -- so a mesh marker on such a test would
        # be silently dropped, which is worth failing over rather than ignoring.
        if _mesh_names(metafunc) is not None:
            raise pytest.UsageError(
                f"{metafunc.definition.nodeid}: benchaxis / benchmeshes need the bench_case "
                "fixture; a bench_lib benchmark has no mesh to select."
            )
        metafunc.parametrize("library", [lib["id"] for lib in libraries], ids=None)
        return

    cases: list[tuple[str, str]] = []
    ids: list[str] = []
    for lib in libraries:
        for mesh in _test_meshes(metafunc, lib):
            cases.append((mesh["name"], lib["id"]))
            ids.append(f"{mesh['name']}-{lib['id']}")
    metafunc.parametrize(("mesh_name", "library"), cases, ids=ids)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "benchlibs(*kinds): library kinds "
        "(triwarp/trimesh/igl/open3d/scipy/numpy/potpourri3d/pymeshlab/pyvista/meshlib/"
        "pymeshfix/pytorch3d) a "
        "benchmark supports.",
    )
    config.addinivalue_line(
        "markers",
        "benchaxis(name): feature-mesh axis from meshes.AXES for a benchmark; bypasses --size.",
    )
    config.addinivalue_line(
        "markers",
        "benchmeshes(*names): mesh names for a benchmark (registry or feature); bypasses --size.",
    )
    # Declared here but enforced from tests/: the parity gate is a static scan so that it runs in
    # the default ``pytest`` invocation, which does not collect this directory at all. Validating
    # here as well would put the same rule in two places, and the weaker of the two would be the
    # one that only fires when somebody remembers to run the benchmarks.
    config.addinivalue_line(
        "markers",
        "noparity(kind, reason=..., oracle=...): a reference this benchmark times whose *result* "
        "is not comparable with triwarp's. reason= is required prose; oracle= names the library "
        "that is the correctness oracle instead, and must itself be covered. See tests/parity.py.",
    )
    # Default to one comparison table per (function, mesh): the ``group`` marker is the function
    # name and ``param:mesh_name`` splits by mesh, so each table lists the libraries side by side.
    # Only applied when the user did not pass their own ``--benchmark-group-by``.
    if getattr(config.option, "benchmark_group_by", "group") == "group":
        config.option.benchmark_group_by = "group,param:mesh_name"


@pytest.hookimpl(tryfirst=True)
def pytest_benchmark_group_stats(
    config: pytest.Config, benchmarks: list[Any], group_by: str
) -> list[tuple[str | None, list[Any]]]:
    """
    Group results like pytest-benchmark does, but tolerate a ``param:`` axis a case does not have.

    The plugin's own implementation indexes ``bench["params"][name]`` directly, so the default
    ``group,param:mesh_name`` grouping raises ``KeyError`` as soon as one mesh-free benchmark
    (``bench_lib``) is collected alongside the mesh-driven ones. Missing axes are skipped instead,
    which puts each mesh-free group under its ``group`` name alone.
    """
    del config
    groups: dict[str | None, list[Any]] = defaultdict(list)
    for bench in benchmarks:
        key: tuple[object, ...] = ()
        for grouping in group_by.split(","):
            if grouping == "group":
                key += (bench["group"],)
            elif grouping == "name":
                key += (bench["name"],)
            elif grouping == "func":
                key += (bench["name"].split("[")[0],)
            elif grouping == "fullname":
                key += (bench["fullname"],)
            elif grouping == "fullfunc":
                key += (bench["fullname"].split("[")[0],)
            elif grouping == "param":
                key += (bench["param"],)
            elif grouping.startswith("param:"):
                name = grouping[len("param:") :]
                params = bench["params"] or {}
                if name in params:
                    key += (f"{name}={params[name]}",)
            else:
                raise NotImplementedError(f"Unsupported grouping {group_by!r}.")
        groups[" ".join(str(part) for part in key if part) or None].append(bench)

    for grouped in groups.values():
        grouped.sort(key=operator.itemgetter("fullname" if "full" in group_by else "name"))
    return sorted(groups.items(), key=lambda pair: pair[0] or "")


# ---------------------------------------------------------------------------
# per-case fixture: bundles inputs + a GPU-safe timing call
# ---------------------------------------------------------------------------


class BenchLibrary:
    """One library variant plus a timed ``run``, with no mesh attached."""

    def __init__(self, library: str, benchmark: BenchmarkFixture) -> None:
        """Bind a library variant and the pytest-benchmark fixture."""
        self.library: LibrarySpec = LIBRARIES_BY_ID[library]
        self._benchmark = benchmark

    @property
    def kind(self) -> str:
        """Library family: one of the ``kind`` values in ``LIBRARIES``."""
        return self.library["kind"]

    @property
    def device(self) -> str | None:
        """Warp device for triwarp targets; ``None`` for CPU-only references."""
        return self.library["device"]

    @property
    def torch_device(self) -> str:
        """
        Torch device string for the ``pytorch3d`` row.

        Always ``"cuda:0"`` in practice: pytorch3d is registered for its CUDA kernels alone (see
        the ``LIBRARIES`` block) and ``_pytorch3d_cuda_available`` drops the reference entirely
        where the extension is missing, so there is no host row to fall back to. Warp's
        ``"cuda:0"`` spelling is a valid torch device string, so the entry is forwarded verbatim.
        The ``or "cpu"`` is the harmless floor for a non-pytorch3d row that reads this.
        """
        return self.device or "cpu"

    def run(
        self,
        fn: Callable[..., Any],
        *,
        rounds: int = _ROUNDS,
        setup: Callable[[], Any] | None = None,
    ) -> Any:
        """
        Time ``fn`` with pytest-benchmark, synchronising inside the timed region on CUDA.

        Warp launches are asynchronous, so ``wp.synchronize_device`` must sit inside the timed
        callable to capture real GPU compute; CPU targets/references skip it.

        ``rounds`` lowers the repeat count for the handful of groups whose single call runs into
        hundreds of milliseconds (the hole-filling DP, ``split`` on a thousand components, an
        isotropic remesh, a Poisson reconstruction). Ten rounds of those alone would dominate the
        suite's wall clock, and their spread is wide enough that the extra samples buy nothing.

        ``setup`` builds a **fresh input per round, outside the timed region**, and its return value
        is passed to ``fn`` as the single positional argument. Most references that mutate their
        input rebuild it *inside* the timed callable instead (``new_mesh_ml``, ``new_meshset_pml``),
        which is right when the build is a small fraction of the call -- and honest, since a caller
        pays it too. Use ``setup`` where it is not: the MeshLib dilation row loads its voxel bitset
        in 0.30 ms against a 0.22 ms ``expandVoxelsMask`` on ``bunny``, so folding the load in would
        report the load. ``pedantic`` runs ``setup`` before every round including the warmup, and
        forbids it above ``iterations=1`` -- which is what every row here already uses.

        **The sync is per-library, not per-device.** ``wp.synchronize_device`` synchronizes *Warp's*
        stream and says nothing about torch's, so a ``pytorch3d-cuda`` row synchronized the Warp way
        would time the launch and not the kernel -- the same class of error as section 8's
        launch-device hazard, in that it does not raise and simply reports a number that is far too
        good. That row therefore calls ``torch.cuda.synchronize()`` instead.

        **And a ``pytorch3d-cuda`` row releases torch's cached blocks when it finishes**, outside
        the timed region. torch's ``CUDACachingAllocator`` reserves device memory and returns it
        only on ``torch.cuda.empty_cache()``, so a large row leaves nothing for the Warp allocator
        that runs next: measured, a 14 M-vertex sparse assembly in torch made Warp fail to allocate
        **65 368 bytes** on a 32 GB device, and every ``triwarp-cuda`` row after it in that module
        died with it -- 16 rows, exit 1, and because the failures were all triwarp's the surviving
        cells *understated* the loss table. The release is per-row rather than per-module because
        the two allocators interleave at row granularity.

        This is the general guard and it turned out to be the *whole* fix: ``test_laplacian``'s
        ``lucy`` pytorch3d rows were then measured **uncapped** with it in place and the module runs
        107 passed / exit 0 with zero allocation failures, so the size cap that was going to
        accompany this was dropped rather than shipped -- it would have cost four real comparisons.
        """
        device = self.device
        needs_cuda_sync = device is not None and device.startswith("cuda")
        needs_torch_sync = needs_cuda_sync and self.kind == "pytorch3d"

        def target(*args: Any) -> Any:
            result = fn(*args)
            if needs_torch_sync:
                import torch

                torch.cuda.synchronize()
            elif needs_cuda_sync:
                wp.synchronize_device(device)
            return result

        if setup is None:
            result = self._benchmark.pedantic(
                target, rounds=rounds, warmup_rounds=_WARMUP_ROUNDS, iterations=1
            )
        else:

            def make_arguments() -> tuple[tuple[Any, ...], dict[str, Any]]:
                """Hand ``pedantic`` this round's freshly-built input as ``fn``'s only argument."""
                return (setup(),), {}

            result = self._benchmark.pedantic(
                target,
                setup=make_arguments,
                rounds=rounds,
                warmup_rounds=_WARMUP_ROUNDS,
                iterations=1,
            )

        if needs_torch_sync:
            import torch

            torch.cuda.empty_cache()
        return result


class BenchCase(BenchLibrary):
    """One ``(mesh, library)`` benchmark case: lazily-built inputs plus a timed ``run``."""

    def __init__(self, mesh_name: str, library: str, benchmark: BenchmarkFixture) -> None:
        """Bind a mesh name, its library variant and the pytest-benchmark fixture."""
        super().__init__(library, benchmark)
        self.mesh_name = mesh_name
        self.spec: MeshSpec = ALL_MESHES_BY_NAME[mesh_name]

    @property
    def faces_wp(self) -> wp.array[wp.int32]:
        """Flat ``wp.int32`` face buffer on this case's device (triwarp only)."""
        assert self.device is not None
        return _faces_wp(self.mesh_name, self.device)

    @property
    def vertices_wp(self) -> wp.array[wp.vec3]:
        """``wp.vec3`` vertex buffer on this case's device (triwarp only)."""
        assert self.device is not None
        return _vertices_wp(self.mesh_name, self.device)

    @property
    def vertices_np(self) -> np.ndarray:
        """Float64 ``(n_vertices, 3)`` vertices for the trimesh / igl / scipy references."""
        return _load_numpy(self.mesh_name)[0]

    @property
    def faces_np(self) -> np.ndarray:
        """Int64 ``(n_faces, 3)`` faces for the trimesh / igl references."""
        return _load_numpy(self.mesh_name)[1]

    @property
    def n_vertices(self) -> int:
        """Vertex count (the ``edges_unique`` hash base), from the spec rather than the buffer."""
        return self.spec["n_vertices"]

    @property
    def n_faces(self) -> int:
        """Triangle count, from the spec so it is available without building the mesh."""
        return self.spec["n_faces"]

    @property
    def mean_edge(self) -> float:
        """Mean undirected edge length; the natural scale for remeshing / reconstruction sizing."""
        return _mean_edge(self.mesh_name)

    @property
    def mesh_o3d(self) -> o3d.geometry.TriangleMesh:
        """
        Shared legacy ``open3d`` mesh, built once per mesh.

        Safe for references that either do not touch their input or recompute unconditionally
        (``subdivide_midpoint`` and ``filter_smooth_laplacian`` return new meshes;
        ``compute_vertex_normals`` overwrites but never caches). For a reference that mutates
        *idempotently* — ``remove_duplicated_triangles``, where rounds 2..n would find nothing left
        to do — construct the mesh inside the timed callable instead, the way the trimesh references
        rebuild their ``tm.Trimesh`` (see ``test_repair.py``).
        """
        if self.mesh_name not in _o3d_cache:
            _o3d_cache[self.mesh_name] = _new_mesh_o3d(self.mesh_name)
        return _o3d_cache[self.mesh_name]

    @property
    def mesh_pv(self) -> pv.PolyData:
        """
        Shared ``pyvista.PolyData``, built once per mesh.

        Safe to share, unlike ``meshset_pml``: pyvista caches nothing (a repeat ``cell_quality`` or
        ``decimate`` call recomputes in full, measured 10.1 / 7.2 ms and 210 / 201 ms) and almost
        every filter returns a *new* object rather than mutating -- ``inplace=False`` is the default
        throughout, so never pass ``inplace=True`` in a row. Two exceptions to build inside the
        timed callable instead: ``edge_mask``, which writes ``point_ind`` into its input, and any
        filter a row calls with ``inplace=True``.
        """
        if self.mesh_name not in _pv_cache:
            _pv_cache[self.mesh_name] = _new_mesh_pv(self.mesh_name)
        return _pv_cache[self.mesh_name]

    @property
    def mesh_p3d(self) -> p3d_structures.Meshes:
        """
        Shared ``pytorch3d.structures.Meshes``, built once per ``(mesh, torch device)``.

        Safe to share, and for a stronger reason than ``mesh_pv``: a ``Meshes`` is **immutable**
        and every ``pytorch3d.ops`` / ``pytorch3d.loss`` entry point is pure, so there is no
        freshness rule here at all -- the opposite end of the scale from ``new_meshset_pml``. What
        it *does* have is a cache of its own derived quantities (``verts_packed``, ``edges_packed``,
        ``faces_packed_to_edges_packed``, ``verts_normals_packed``), memoized on first request. So
        a row must decide which side of that it wants to time: reading an accessor for the first
        time inside the timed callable prices the derivation, and reading it again prices nothing.
        Warm the accessor outside the callable where the row names an operation rather than a
        derivation, and say which in the docstring.
        """
        key = (self.mesh_name, self.torch_device)
        if key not in _p3d_cache:
            _p3d_cache[key] = _new_mesh_p3d(*key)
        return _p3d_cache[key]

    @property
    def cloud_p3d(self) -> Any:
        """
        The mesh's own vertices as a ``pytorch3d.structures.Pointclouds`` on this row's device.

        The container the ``loss`` entry points take, where the ``ops`` ones take the bare batched
        tensor ``points_torch_from_numpy`` builds. Not cached: it is one tensor wrap over an array
        already on the device, and a row that wants it out of the timed region can hoist it itself.
        """
        import pytorch3d.structures as p3d_structures

        return p3d_structures.Pointclouds(
            points=[points_torch_from_numpy(self.vertices_np, self.torch_device)[0]]
        )

    def new_meshset_pml(self) -> ml.MeshSet:
        """
        Build a **fresh** single-mesh ``pymeshlab.MeshSet``; call this *inside* the timed callable.

        Almost every MeshLab filter mutates ``current_mesh()`` in place -- ``apply_coord_*`` moves
        vertices, ``meshing_*`` rewrites the topology, ``compute_*_per_vertex`` writes an attribute,
        and ``generate_*`` pushes a new mesh onto the set -- so a shared MeshSet would have rounds
        2..n measure a filter applied to its own output. That is the trimesh situation rather than
        the open3d one, so the build goes inside the timed region, exactly as ``test_repair.py``'s
        trimesh rows rebuild their ``tm.Trimesh`` and the potpourri3d rows construct their solver.

        Two filters were checked and are *not* idempotent even in geometry, because they default to
        ``autoclean=True`` and delete unreferenced vertices under you:
        ``compute_curvature_principal_directions_per_vertex`` and
        ``meshing_decimation_quadric_edge_collapse``.

        The build is **not free** and it is the floor under every pymeshlab row: ~0.47 us/vertex,
        measured at 0.91 ms on 2 562 vertices, 4.55 ms on 10 242 and **17.1 ms on ``bunny``**'s
        35 947. Any pymeshlab row cheaper than that floor is reporting the build and nothing else --
        read it that way rather than as a filter cost. Use ``meshset_pml`` for the narrow set of
        filters verified to leave the geometry alone.
        """
        return _new_meshset_pml(self.mesh_name)

    @property
    def meshset_pml(self) -> ml.MeshSet:
        """
        Return a shared ``pymeshlab.MeshSet``, built once per mesh -- geometry-preserving only.

        Two families qualify, both of which leave positions and topology untouched so that rounds
        2..n do the identical work:

        - the ``compute_scalar_*`` / ``compute_normal_*`` filters, which write one vertex or face
          attribute and read only positions (``compute_scalar_ambient_occlusion``,
          ``compute_scalar_by_volumetric_obscurance``,
          ``compute_scalar_by_shape_diameter_function_per_vertex``,
          ``compute_scalar_by_aspect_ratio_per_face``, ``compute_normal_per_vertex``);
        - the selection filters, which write only the per-element *selected* bit. These are not
          idempotent -- ``apply_selection_dilatation`` grows the set every call -- but their cost is
          independent of how much is selected, because each one is a full pass over the face set.
          Measured on ``sphere_med``: 120 consecutive dilatations taking the selection from 0.9% to
          86% of 81 920 faces cost **0.86-1.21 ms** each (median 0.93), and 120 erosions back down
          cost 1.18-1.23. So a group may seed the selection once, outside the timed callable, and
          still read a clean per-call cost.

        Because the state persists, a group that *depends* on the selection (the heat-geodesic
        source set, say) must establish it itself rather than inherit what a previous group left.

        Everything else -- including every ``meshing_*``, ``apply_coord_*`` and ``generate_*``
        filter -- must call ``new_meshset_pml`` instead.
        """
        if self.mesh_name not in _pml_cache:
            _pml_cache[self.mesh_name] = _new_meshset_pml(self.mesh_name)
        return _pml_cache[self.mesh_name]

    def new_mesh_ml(self) -> mm.Mesh:
        """
        Build a **fresh** ``meshlib.mrmeshpy.Mesh``; call this *inside* the timed callable.

        A method rather than a cached property, and for the ``new_meshset_pml`` reason: almost every
        MeshLib free function mutates its ``Mesh`` in place and returns something else -- ``relax``,
        ``fillHole``, ``decimateMesh``, ``remesh``, ``subdivideMesh``, ``fixMeshDegeneracies``,
        ``denoiseNormals``, ``expand`` and ``shrink`` all return a status, a count, an ``EdgeId`` or
        a bitset of *new* elements, never the mesh -- so a shared mesh would have rounds 2..n
        measure a filter applied to its own output.

        Two things a row using this must decide explicitly and state in its docstring:

        - **The AABB tree is lazily built, cached on the ``Mesh``, and worth 131x.** Measured on a
          40 962-vertex sphere: the first ``findProjection`` takes 2.741 ms and the second 21.0 us.
          A query row should therefore build the mesh *outside* the timed callable and pre-warm it
          with one throwaway query, so the row times the query rather than the tree -- which is
          what triwarp's ``wp.Mesh``-in-hand rows already do with their BVH. A row that rebuilds per
          round is timing the build.
        - A mutating call **invalidates** that tree, so the two decisions are not independent.
        """
        return _new_mesh_ml(self.mesh_name)

    def new_tmesh_pmf(self) -> PyTMesh:
        """
        Build a **fresh** ``pymeshfix._meshfix.PyTMesh``; call this *inside* the timed callable.

        A method rather than a cached property, and unlike ``new_mesh_ml`` there is no alternative:
        a ``PyTMesh`` accepts exactly **one** ``load_array`` (a second raises ``RuntimeError``) and
        every one of its nine algorithms mutates in place and returns a status, a count or an array
        rather than the mesh. So a shared object could not even be reloaded, let alone reused.

        The consequence is that **every pymeshfix row prices the load**, and the load is not small:
        67.9 ms on ``bunny_decimated`` and 439.6 ms on ``bunny``, against 64.2 / 435.3 ms for
        ``select_intersecting_triangles``, 5.8 / 51.4 ms for ``fill_small_boundaries`` and
        6.7 / 60.5 ms for ``remove_smallest_components``. A row whose operation is a small share of
        that total reports the load and reads as pymeshfix being an order of magnitude slower at
        the operation than it is -- so rows exist only above ~30 %, and each states its share. See
        the ``pymeshfix`` paragraph in the LIBRARIES comment block for the full rule.
        """
        return _new_tmesh_pmf(self.mesh_name)


@pytest.fixture
def bench_case(benchmark: BenchmarkFixture, mesh_name: str, library: str) -> BenchCase:
    return BenchCase(mesh_name, library, benchmark)


@pytest.fixture
def bench_lib(benchmark: BenchmarkFixture, library: str) -> BenchLibrary:
    """
    Time a benchmark that has no input mesh, e.g. the ``triwarp.creation`` generators.

    Requesting this instead of ``bench_case`` parametrizes over libraries alone: ``--device`` still
    selects the triwarp targets, but ``--size`` / ``--cpu-max-size`` and the ``benchmeshes`` marker
    have nothing to act on. Size the work with a plain ``pytest.mark.parametrize`` instead.
    """
    return BenchLibrary(library, benchmark)
