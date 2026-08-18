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
    ``open3d`` / ``scipy`` / ``potpourri3d`` / ``pymeshlab`` / ``pyvista`` / ``meshlib`` CPU
    baselines are always included.
``--size=<comma list | all>``
    Restrict meshes to these size categories (``small,medium,large,extralarge,huge``).
``--cpu-max-size=<category>``
    CPU-bound libraries (``triwarp-cpu``, ``trimesh``, ``igl``, ``open3d``, ``scipy``,
    ``potpourri3d``, ``pymeshlab``, ``pyvista``, ``meshlib``) skip meshes
    larger than this unless the size was named explicitly in ``--size``. Default ``large`` — so
    ``happy_buddha`` and ``lucy`` run GPU-only by default while
    ``bunny_decimated``/``bunny``/``dragon`` run on CPU.
"""

from __future__ import annotations

import operator
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
    import pyvista as pv
    from meshlib import mrmeshpy as mm
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
    Skip the current case when its mesh is larger than ``largest`` in scan-registry order.

    A no-op for the synthetic feature meshes: they are not on the size ladder, and every one of
    them is deliberately sized to run everywhere it is used, so a cap written for the scan sweep
    has nothing to say about them.
    """
    if bench_case.mesh_name not in MESHES_BY_NAME:
        return
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
# ``numpy`` is the narrowest baseline of all: it is only a reference for the *array primitives*
# (``triwarp.reduce``), where a host reduction over an already-resident NumPy buffer is the honest
# CPU floor. It is deliberately not a geometry reference — every other CPU baseline is already
# built on NumPy, so timing it against a geometry wrapper would measure nothing new.
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


def _new_mesh_ml(name: str) -> mm.Mesh:
    """
    Build a fresh ``meshlib.mrmeshpy.Mesh`` from the shared NumPy source.

    Imported lazily (like ``meshio``, ``open3d``, ``pymeshlab`` and ``pyvista`` above) so the
    0.256 s meshlib import is only paid by runs that include a meshlib case. ``meshFromFacesVerts``
    takes
    **faces first** -- the reverse of every other builder here -- and accepts float64 positions and
    int64 indices as the loader holds them.
    """
    from meshlib import mrmeshnumpy as mn

    vertices, faces = _load_numpy(name)
    return mn.meshFromFacesVerts(faces, vertices)


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
        # trimesh / igl / open3d / scipy / potpourri3d / pymeshlab / pyvista / meshlib baselines
        # are always included.
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
        "(triwarp/trimesh/igl/open3d/scipy/numpy/potpourri3d/pymeshlab/pyvista/meshlib) a "
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
        """Library family: triwarp, trimesh, igl, open3d, scipy, numpy, potpourri3d or pymeshlab."""
        return self.library["kind"]

    @property
    def device(self) -> str | None:
        """Warp device for triwarp targets; ``None`` for CPU references."""
        return self.library["device"]

    def run(self, fn: Callable[[], Any], *, rounds: int = _ROUNDS) -> Any:
        """
        Time ``fn`` with pytest-benchmark, synchronising inside the timed region on CUDA.

        Warp launches are asynchronous, so ``wp.synchronize_device`` must sit inside the timed
        callable to capture real GPU compute; CPU targets/references skip it.

        ``rounds`` lowers the repeat count for the handful of groups whose single call runs into
        hundreds of milliseconds (the hole-filling DP, ``split`` on a thousand components, an
        isotropic remesh, a Poisson reconstruction). Ten rounds of those alone would dominate the
        suite's wall clock, and their spread is wide enough that the extra samples buy nothing.
        """
        device = self.device
        needs_sync = device is not None and device.startswith("cuda")

        def target() -> Any:
            result = fn()
            if needs_sync:
                wp.synchronize_device(device)
            return result

        return self._benchmark.pedantic(
            target, rounds=rounds, warmup_rounds=_WARMUP_ROUNDS, iterations=1
        )


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
