from __future__ import annotations

from types import CodeType

import numpy as np
import pytest
import trimesh as tm
import warp as wp

import triwarp as tw
from tests.conversions import trimesh_to_warp, warp_to_trimesh

# Reject a launch whose array arguments do not live on the launch device. Warp's default is
# RELAXED, which passes the pointers straight through: a launch that forgets ``device=`` lands on
# the default CUDA device, reads the CPU arrays over HMM, returns the *right answer*, and then
# corrupts the host heap when those arrays are freed while the kernel is still running (measured:
# 20/20 aborts with a free and no sync, 0/20 with either). CHECKED does not catch it -- it
# validates addressability, which HMM genuinely provides. STRICT is the only mode that rejects a
# genuine cross-device argument, and no triwarp launch is intentionally cross-device. It is only
# half the guard: on a CUDA run an omitted ``device=`` resolves to the arrays' own device, so there
# is no mismatch to reject and only check 15's static scan sees it.
if hasattr(wp.config, "launch_array_access_mode"):  # warp >= 1.14
    wp.config.launch_array_access_mode = wp.config.LaunchArrayAccessMode.STRICT


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "parity(group, *libraries, benchmarked=..., reason=...): this test asserts triwarp agrees "
        "with each named reference library for the benchmark group of that name. The gate in "
        "tests/test_parity.py requires one of these (or a noparity exemption in benchmarks/) for "
        "every benchmarked pair. Pass benchmarked=False with a written reason= where the pair is "
        "compared here but deliberately not timed.",
    )
    # The cut is at 15 s, measured, and it is four tests. On a full CPU-only run of tests/ (432.6 s
    # total) the whole ``screened_poisson`` family is 342.0 s -- 79 % -- and these four alone are
    # 277.6 s (64 %). Each is one ~90 s depth-6 Poisson solve that costs under a second on CUDA, so
    # skipping the ``cpu`` half of them leaves ~155 s of CPU work and loses no claim: the answers
    # are device-independent, ``test_poisson_cpu_matches_cuda`` pins the two devices to each other
    # at depth 4, and eleven more Poisson tests still run on CPU at depth 5 (``_poisson_depth``).
    #
    # Read the seconds in the marker as an order of magnitude, not a contract. The same four
    # measured 259 s inside the full suite and 380 s as their own ``-k`` selection in one session --
    # 1.47x apart, with the ranking inverted -- so this box's CPU timings swing far too much for a
    # threshold to be re-derived by rerunning. What is stable is the shape: one depth-6 solve each.
    config.addinivalue_line(
        "markers",
        "slow_cpu(seconds): this test costs the stated measured seconds on the CPU device, so its "
        "``cpu`` parametrization is skipped unless --device=both. It still runs on CUDA, where the "
        "same test costs under a second. Reserved for the handful of tests that dominate a CPU "
        "run -- see the comment above for the measurements and why the cut sits where it does.",
    )


def _selected_devices(config: pytest.Config) -> list[str]:
    """
    Devices to parametrize the ``device`` fixture over, for **this process**.

    ``auto`` picks one device, matching ``benchmarks/conftest.py``. Both-device coverage is worth
    having -- it is what caught the ``warp.fem`` device leak in ``_screened_poisson_adaptive`` and
    the module-scope ``wp.array`` in ``test_grouping`` -- but it must not be bought *inside one
    process*, because **CPU work is ~36x slower once CUDA has been initialised**. Measured on one
    ``heat_signed_distance`` call, same mesh, same code, only ``CUDA_VISIBLE_DEVICES`` differing:

    ===============  =============  ==================
    launch mode      CUDA visible   ``CUDA_VISIBLE_DEVICES=""``
    ===============  =============  ==================
    ``STRICT``       50.57 s        **1.40 s**
    ``RELAXED``      50.34 s        **1.40 s**
    ``CHECKED``      49.77 s        --
    ===============  =============  ==================

    So it is CUDA *presence*, not section 8's launch-access guard, and the guard is free to stay
    ``STRICT``. In-process ``--device=both`` measured 717 s for the whole suite where the two
    passes run separately cost ~37.6 s + ~155 s; ``uv run python -m tests.devices`` is the runner
    that spawns them, and the CPU one sets ``CUDA_VISIBLE_DEVICES=""`` for exactly this reason.

    ``both`` stays meaningful and is not the slow trap it sounds like: it means "every device this
    process can see, and skip nothing". In a CUDA-hidden process that is precisely "all of CPU,
    including the ``slow_cpu`` tests", which is how the runner asks for a full CPU pass.
    """
    mode = str(config.getoption("--device"))
    if mode == "cuda":
        if not wp.is_cuda_available():
            raise pytest.UsageError("--device=cuda was requested but no CUDA device is available")
        return ["cuda:0"]
    if mode == "cpu":
        return ["cpu"]
    if mode == "auto":
        return ["cuda:0"] if wp.is_cuda_available() else ["cpu"]
    return ["cpu", "cuda:0"] if wp.is_cuda_available() else ["cpu"]


def _calls_getfixturevalue(code: CodeType) -> bool:
    """
    Whether this code object, or any code object nested in it, names ``getfixturevalue``.

    The recursion is the point. On Python 3.11 a comprehension compiles to its own code object, so
    ``[request.getfixturevalue(n) for n in names]`` puts the name in the *comprehension's*
    ``co_names`` and leaves the enclosing function's clean -- which is how a flat check silently
    missed ``test_combine.py::test_split_batched_matches_split`` and let it run single-device.
    (3.12 inlines comprehensions and would have hidden the bug the other way, on a future upgrade.)
    """
    if "getfixturevalue" in code.co_names:
        return True
    return any(
        _calls_getfixturevalue(const) for const in code.co_consts if isinstance(const, CodeType)
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """
    Parametrize ``device`` over the selected devices, so a test id names the device it ran on.

    ``metafunc.parametrize`` only reaches a fixture in the test's *static* closure, and 206 tests
    reach ``device`` only through ``request.getfixturevalue(mesh_name)`` -- a lookup by string that
    pytest cannot see at collection time, so those tests would silently keep running on one device
    (and, with no ``device`` fixture to fall back on, fail outright with ``fixture 'device' not
    found``). Appending to ``metafunc.fixturenames`` puts it in the closure anyway, which is what
    lets a mesh fixture resolved later pick up the parametrized value. Keyed off the *bytecode*
    rather than off ``request`` being requested, because several tests take ``request`` for other
    reasons and doubling those buys nothing.
    """
    if "device" not in metafunc.fixturenames and _calls_getfixturevalue(metafunc.function.__code__):
        metafunc.fixturenames.append("device")
    if "device" in metafunc.fixturenames:
        metafunc.parametrize(
            "device", _selected_devices(metafunc.config), ids=lambda name: name.replace(":", "")
        )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Drop the ``cpu`` parametrization of the ``slow_cpu`` tests unless --device=both."""
    if str(config.getoption("--device")) == "both":
        return
    for item in items:
        marker = item.get_closest_marker("slow_cpu")
        if marker is None:
            continue
        callspec = getattr(item, "callspec", None)
        if callspec is None or callspec.params.get("device") != "cpu":
            continue
        seconds = marker.args[0] if marker.args else "many"
        item.add_marker(
            pytest.mark.skip(
                reason=f"slow on CPU ({seconds} s measured); pass --device=both to run it"
            )
        )


@pytest.fixture
def device(request: pytest.FixtureRequest) -> str:
    """
    Fallback only: ``pytest_generate_tests`` parametrizes this name for every test that reaches it.

    Kept so a path that hook does not anticipate degrades to a single device instead of erroring
    with ``fixture 'device' not found``. If this body ever runs, some test is getting one device
    where it should be getting both -- which is a gap, not a failure, so it must not raise.
    """
    return _selected_devices(request.config)[-1]


@pytest.fixture
def icosahedron(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    icosahedron = tm.creation.icosahedron()
    icosahedron.apply_translation(translation=np.array([-1.0, 0.0, 2.0]))
    return icosahedron, trimesh_to_warp(icosahedron, device)


@pytest.fixture
def unit_box(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    """
    Build the unit cube: the sharp-featured convex solid, keyed to a right dihedral angle.

    Neither ``icosahedron`` nor ``cave_cube`` substitutes here -- a cube's 12 creases sit at exactly
    90 degrees with 6 flat face diagonals between them, which is what crease and seam tests are
    written against, and ``cave_cube`` is non-convex. Left untranslated, since several callers read
    coordinate signs to pick out one face.
    """
    box = tm.creation.box(extents=[1.0, 1.0, 1.0])
    return box, trimesh_to_warp(box, device)


@pytest.fixture
def icosphere(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    """
    Build a unit icosphere at ``subdivisions=3``: 642 vertices, 1 280 faces, closed and regular.

    The workhorse curved fixture, and the one whose absence was structural: section 6 says not to
    hand-roll a mesh when a fixture will do, but the only closed fixture was ``icosahedron`` at 12
    vertices -- too coarse for anything that needs curvature -- so 47 tests across 21 files built
    this by hand. ``subdivisions=3`` is where 26 of them clustered.

    Untranslated, unlike ``icosahedron``: a test that wants the origin off the centroid should move
    it itself, and several of the callers this replaced depend on the sphere being centred.

    Function-scoped like every fixture here, so mutating ``mesh_tm.vertices`` in a test is safe --
    see the note on session scoping in ``plans/better-tests.md`` for why it stays that way.

    See Also
    --------
    ``icosphere_coarse``
        The same sphere at ``subdivisions=2``, where another 13 of the call sites sat.
    """
    sphere = tm.creation.icosphere(subdivisions=3, radius=1.0)
    return sphere, trimesh_to_warp(sphere, device)


@pytest.fixture
def icosphere_coarse(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    """
    Build the same sphere at ``subdivisions=2``: 162 vertices, 320 faces.

    A quarter the faces of ``icosphere`` and still genuinely curved. Prefer it wherever the test's
    claim does not need the resolution -- a reference call that costs 100 ms on 1 280 faces is the
    difference between a fast suite and a slow one, and most of these comparisons are about
    correctness rather than about mesh size.
    """
    sphere = tm.creation.icosphere(subdivisions=2, radius=1.0)
    return sphere, trimesh_to_warp(sphere, device)


@pytest.fixture
def half_torus(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    torus = tm.creation.torus(major_radius=1.0, minor_radius=0.5)
    half_torus = torus.slice_plane(plane_origin=np.zeros(3), plane_normal=np.array([1.0, 0.0, 0.0]))
    scale = 1 + np.exp(-half_torus.vertices[:, 1])
    half_torus.vertices *= scale[:, None]
    half_torus.apply_translation(translation=np.array([-1.0, 0.0, 2.0]))
    return half_torus, trimesh_to_warp(half_torus, device)


@pytest.fixture
def torus(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    torus = tm.creation.torus(major_radius=1.0, minor_radius=0.4)
    return torus, trimesh_to_warp(torus, device)


@pytest.fixture
def genus_two(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    left = tm.creation.torus(major_radius=1.0, minor_radius=0.35)
    right = tm.creation.torus(major_radius=1.0, minor_radius=0.35)
    right.apply_translation(translation=np.array([1.8, 0.0, 0.0]))
    mesh = tm.boolean.union([left, right])
    return mesh, trimesh_to_warp(mesh, device)


@pytest.fixture
def cave_cube(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    mesh = tm.boolean.difference(
        [tm.creation.box(extents=[1.0, 1.0, 1.0]), tm.creation.box(extents=[0.1, 0.1, 0.1])]
    )
    return mesh, trimesh_to_warp(mesh, device)


@pytest.fixture
def hemisphere(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    sphere = tm.creation.icosphere(subdivisions=2, radius=1.0)
    hemisphere = sphere.slice_plane(
        plane_origin=np.zeros(3), plane_normal=np.array([0.0, 0.0, 1.0]), cap=False
    )
    hemisphere.merge_vertices()
    rotation = tm.transformations.rotation_matrix(
        np.deg2rad(45.0), direction=np.array([1.0, 1.0, 0.0])
    )
    rotation[:3, 3] = np.array([-1.0, 0.0, 2.0])
    hemisphere.apply_transform(rotation)
    return hemisphere, trimesh_to_warp(hemisphere, device)


@pytest.fixture
def boy_surface(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    """
    Boy's surface: closed, watertight and **non-orientable**, with Euler characteristic 1.

    The only fixture of its class. Every other closed mesh in this file is orientable with an even
    characteristic, so the ``False`` branch of ``is_orientable`` / ``face_orientation_bits`` and the
    impossible branch of ``make_winding_consistent`` are unreachable without it.
    """
    vertices_wp, faces_wp = tw.creation.parametric_surface("boy", device=device)
    mesh = warp_to_trimesh(vertices_wp, faces_wp)
    return mesh, trimesh_to_warp(mesh, device)


@pytest.fixture
def mobius(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    """Moebius band: non-orientable *with* a boundary — one loop of 78 edges, and χ = 0."""
    vertices_wp, faces_wp = tw.creation.parametric_surface("mobius", device=device)
    mesh = warp_to_trimesh(vertices_wp, faces_wp)
    return mesh, trimesh_to_warp(mesh, device)


@pytest.fixture
def bohemian_dome(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    """Build a closed genus-1 surface that intersects itself: watertight, orientable, χ = 0."""
    vertices_wp, faces_wp = tw.creation.parametric_surface("bohemian_dome", device=device)
    mesh = warp_to_trimesh(vertices_wp, faces_wp)
    return mesh, trimesh_to_warp(mesh, device)


@pytest.fixture
def sliver_patch(device: str) -> tuple[np.ndarray, np.ndarray, wp.array, wp.array]:
    """
    Build a patch with one near-zero-area triangle, thin enough to break the triangle inequality.

    Mollification and the robust Laplacian are only interesting on a mesh that needs them: the
    plain cotangent Laplacian returns NaN here, the robust one must not. Returned as both NumPy
    (``float64``, for the CPU references) and Warp (``float32``, where the inequality actually
    fails) so the two sides see the same mesh.
    """
    vertices_np = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1e-9, 0.0], [0.5, 1.0, 0.0]], dtype=np.float64
    )
    faces_np = np.array([[0, 1, 2], [0, 2, 3], [2, 1, 3]], dtype=np.int32)
    return (
        vertices_np,
        faces_np,
        wp.array(vertices_np.astype(np.float32), dtype=wp.vec3, device=device),
        wp.array(faces_np.reshape(-1), dtype=wp.int32, device=device),
    )
