"""
Benchmark harness: mesh registry, CLI flags, parametrization and a GPU-safe timing fixture.

Run with ``uv run pytest benchmarks/`` (the default ``pytest`` run only collects ``tests/``).

Flags
-----
``--device={auto,cpu,cuda,both}``
    Which ``triwarp`` targets to time. ``auto`` (default) uses cuda when CUDA is available,
    else falls back to cpu — ``triwarp-cpu`` is not timed alongside cuda by default. Pass
    ``cpu`` for cpu only or ``both`` to time both triwarp targets. The ``trimesh`` / ``igl`` /
    ``open3d`` / ``scipy`` CPU baselines are always included.
``--size=<comma list | all>``
    Restrict meshes to these size categories (``small,medium,large,extralarge,huge``).
``--cpu-max-size=<category>``
    CPU-bound libraries (``triwarp-cpu``, ``trimesh``, ``igl``, ``open3d``, ``scipy``) skip meshes
    larger than this unless the size was named explicitly in ``--size``. Default ``large`` — so
    ``happy_buddha`` and ``lucy`` run GPU-only by default while
    ``bunny_decimated``/``bunny``/``dragon`` run on CPU.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
import pytest
import warp as wp

if TYPE_CHECKING:
    import open3d as o3d
    from pytest_benchmark.fixture import BenchmarkFixture

DATA_DIR = Path(__file__).parent / "data"

# Number of timed rounds and untimed warm-up rounds. The warm-up covers Warp kernel JIT
# compilation (first launch of each kernel) and CPU cache priming.
_ROUNDS = 10
_WARMUP_ROUNDS = 1

# Ordered from smallest to largest — used for size comparisons.
SIZE_ORDER = ["small", "medium", "large", "extralarge", "huge"]


def size_category(n_faces: int) -> str:
    """Classify a mesh by triangle count: <10k, <100k, <1M, <10M, else huge."""
    if n_faces < 10_000:
        return "small"
    if n_faces < 100_000:
        return "medium"
    if n_faces < 1_000_000:
        return "large"
    if n_faces < 10_000_000:
        return "extralarge"
    return "huge"


class MeshSpec(TypedDict):
    """A benchmark mesh: file name and (precomputed) triangle-count size category."""

    name: str
    filename: str
    n_faces: int
    size: str


class LibrarySpec(TypedDict):
    """A benchmarked library variant. ``device`` is ``None`` for the CPU-only references."""

    id: str
    kind: str
    device: str | None
    cpu_bound: bool


def _mesh(name: str, filename: str, n_faces: int) -> MeshSpec:
    return {"name": name, "filename": filename, "n_faces": n_faces, "size": size_category(n_faces)}


# ``heptoroid.ply`` is intentionally absent: it is a tristrip PLY that meshio cannot decode as
# triangle faces. Face counts are read from the PLY headers so a mesh can be skipped by size
# before it is ever loaded.
MESHES: list[MeshSpec] = [
    _mesh("bunny_decimated", "bunny_decimated.ply", 16_301),
    _mesh("bunny", "bunny.ply", 69_451),
    _mesh("dragon", "dragon.ply", 871_414),
    _mesh("happy_buddha", "happy_buddha.ply", 1_087_716),
    _mesh("lucy", "lucy.ply", 28_055_742),
]
MESHES_BY_NAME = {mesh["name"]: mesh for mesh in MESHES}
MESH_ORDER = [mesh["name"] for mesh in MESHES]

RIM_LONG = 1 << 16

# Grid resolutions for the saddle patches: ``k x k`` vertices give ``2 * (k - 1) ** 2`` faces.
SADDLE_SMALL, SADDLE_MEDIUM = 68, 133

SYNTHETIC_MESHES: list[MeshSpec] = [
    _mesh("synthetic_cylinder", "", 2 * RIM_LONG),
    _mesh("synthetic_saddle_small", "", 2 * (SADDLE_SMALL - 1) ** 2),
    _mesh("synthetic_saddle", "", 2 * (SADDLE_MEDIUM - 1) ** 2),
]
SYNTHETIC_MESHES_BY_NAME = {mesh["name"]: mesh for mesh in SYNTHETIC_MESHES}
ALL_MESHES_BY_NAME = {**MESHES_BY_NAME, **SYNTHETIC_MESHES_BY_NAME}


def skip_larger_than(bench_case: BenchCase, largest: str, reason: str = "") -> None:
    """Skip the current case when its mesh is larger than ``largest`` (registry order)."""
    if MESH_ORDER.index(bench_case.mesh_name) > MESH_ORDER.index(largest):
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
LIBRARIES: list[LibrarySpec] = [
    {"id": "triwarp-cpu", "kind": "triwarp", "device": "cpu", "cpu_bound": True},
    {"id": "triwarp-cuda", "kind": "triwarp", "device": "cuda:0", "cpu_bound": False},
    {"id": "trimesh", "kind": "trimesh", "device": None, "cpu_bound": True},
    {"id": "igl", "kind": "igl", "device": None, "cpu_bound": True},
    {"id": "open3d", "kind": "open3d", "device": None, "cpu_bound": True},
    {"id": "scipy", "kind": "scipy", "device": None, "cpu_bound": True},
]
LIBRARIES_BY_ID = {lib["id"]: lib for lib in LIBRARIES}


# ---------------------------------------------------------------------------
# mesh loading (shared numpy source; per-device warp buffers built from it)
# ---------------------------------------------------------------------------

_numpy_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_wp_cache: dict[tuple[str, str, str], wp.array] = {}
_mean_edge_cache: dict[str, float] = {}
_o3d_cache: dict[str, o3d.geometry.TriangleMesh] = {}


def _open_cylinder(rim: int) -> tuple[np.ndarray, np.ndarray]:
    """Open tube with two boundary rims of ``rim`` vertices each (``2 * rim`` triangles)."""
    angle = 2.0 * np.pi * np.arange(rim) / rim
    ring = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(rim)))
    vertices = np.vstack((ring, ring + np.array([0.0, 0.0, 1.0]))).astype(np.float32)
    j = np.arange(rim)
    k = (j + 1) % rim
    lower = np.column_stack((j, k, j + rim))
    upper = np.column_stack((j + rim, k, k + rim))
    return vertices, np.vstack((lower, upper)).astype(np.int32)


def _saddle_patch(k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    ``k x k`` regular grid lifted onto a saddle: ``2 * (k - 1) ** 2`` faces, one boundary loop.

    A disk-topology patch, which is what the parametrization solvers are actually for: every vertex
    has valence at most 6, the curvature is non-trivial (so the cotangent weights are not
    degenerate), and the geometry is generated from NumPy alone, so the face count is exact and
    independent of any mesh library's version.
    """
    axis = np.linspace(-1.0, 1.0, k)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    z = 0.35 * (x * x - 0.6 * y * y)
    vertices = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    i, j = np.meshgrid(np.arange(k - 1), np.arange(k - 1), indexing="ij")
    lower_left = (i * k + j).ravel()
    faces = np.vstack(
        (
            np.column_stack((lower_left, lower_left + k, lower_left + k + 1)),
            np.column_stack((lower_left, lower_left + k + 1, lower_left + 1)),
        )
    )
    return vertices, faces


_SYNTHETIC_BUILDERS = {
    "synthetic_cylinder": lambda: _open_cylinder(RIM_LONG),
    "synthetic_saddle_small": lambda: _saddle_patch(SADDLE_SMALL),
    "synthetic_saddle": lambda: _saddle_patch(SADDLE_MEDIUM),
}


def _load_numpy(name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Read ``(vertices_f64, faces_i64)`` once with meshio, cached across the session.

    Uses the same ``meshio.read`` path as ``triwarp.io.load_mesh_data``; kept at NumPy level
    (float64 vertices / int64 faces) so igl and trimesh get their arrays directly and the warp
    buffers are built from the same source. Synthetic meshes are generated instead of read.
    """
    if name not in _numpy_cache:
        builder = _SYNTHETIC_BUILDERS.get(name)
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
    group = parser.getgroup("triwarp-bench")
    group.addoption(
        "--device",
        action="store",
        default="auto",
        choices=["auto", "cpu", "cuda", "both"],
        help="triwarp target device(s) to benchmark (default: auto = cuda if available, else cpu).",
    )
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

    selected = []
    for lib in LIBRARIES:
        if lib["id"] == "triwarp-cpu" and not include_cpu:
            continue
        if lib["id"] == "triwarp-cuda" and not (include_cuda and cuda_available):
            continue
        # trimesh / igl / open3d / scipy baselines are always included.
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


def _test_meshes(metafunc: pytest.Metafunc, lib: LibrarySpec) -> list[MeshSpec]:
    """Meshes for one test: explicit ``benchmeshes`` names, else registry filtered by CLI flags."""
    marker = metafunc.definition.get_closest_marker("benchmeshes")
    if marker is not None:
        meshes = []
        for name in marker.args:
            spec = ALL_MESHES_BY_NAME.get(name)
            if spec is None:
                raise pytest.UsageError(f"unknown benchmeshes name: {name!r}")
            meshes.append(spec)
        return meshes
    return _selected_meshes(metafunc.config, lib)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "mesh_name" not in metafunc.fixturenames or "library" not in metafunc.fixturenames:
        return
    supported = _supported_kinds(metafunc)
    cases: list[tuple[str, str]] = []
    ids: list[str] = []
    for lib in _selected_libraries(metafunc.config):
        if lib["kind"] not in supported:
            continue
        for mesh in _test_meshes(metafunc, lib):
            cases.append((mesh["name"], lib["id"]))
            ids.append(f"{mesh['name']}-{lib['id']}")
    metafunc.parametrize(("mesh_name", "library"), cases, ids=ids)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "benchlibs(*kinds): library kinds (triwarp/trimesh/igl/open3d/scipy) a benchmark supports.",
    )
    config.addinivalue_line(
        "markers",
        "benchmeshes(*names): mesh names for a benchmark (registry or synthetic); bypasses --size.",
    )
    # Default to one comparison table per (function, mesh): the ``group`` marker is the function
    # name and ``param:mesh_name`` splits by mesh, so each table lists the libraries side by side.
    # Only applied when the user did not pass their own ``--benchmark-group-by``.
    if getattr(config.option, "benchmark_group_by", "group") == "group":
        config.option.benchmark_group_by = "group,param:mesh_name"


# ---------------------------------------------------------------------------
# per-case fixture: bundles inputs + a GPU-safe timing call
# ---------------------------------------------------------------------------


class BenchCase:
    """One ``(mesh, library)`` benchmark case: lazily-built inputs plus a timed ``run``."""

    def __init__(self, mesh_name: str, library: str, benchmark: BenchmarkFixture) -> None:
        """Bind a mesh name, its library variant and the pytest-benchmark fixture."""
        self.mesh_name = mesh_name
        self.library: LibrarySpec = LIBRARIES_BY_ID[library]
        self._benchmark = benchmark

    @property
    def kind(self) -> str:
        """Library family: ``triwarp`` / ``trimesh`` / ``igl`` / ``open3d`` / ``scipy``."""
        return self.library["kind"]

    @property
    def device(self) -> str | None:
        """Warp device for triwarp targets; ``None`` for CPU references."""
        return self.library["device"]

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
        """Vertex count (the ``edges_unique`` hash base)."""
        return int(_load_numpy(self.mesh_name)[0].shape[0])

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

    def run(self, fn: Callable[[], Any]) -> Any:
        """
        Time ``fn`` with pytest-benchmark, synchronising inside the timed region on CUDA.

        Warp launches are asynchronous, so ``wp.synchronize_device`` must sit inside the timed
        callable to capture real GPU compute; CPU targets/references skip it.
        """
        device = self.device
        needs_sync = device is not None and device.startswith("cuda")

        def target() -> Any:
            result = fn()
            if needs_sync:
                wp.synchronize_device(device)
            return result

        return self._benchmark.pedantic(
            target, rounds=_ROUNDS, warmup_rounds=_WARMUP_ROUNDS, iterations=1
        )


@pytest.fixture
def bench_case(benchmark: BenchmarkFixture, mesh_name: str, library: str) -> BenchCase:
    return BenchCase(mesh_name, library, benchmark)
