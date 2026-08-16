from __future__ import annotations

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


@pytest.fixture
def device():
    if wp.is_cuda_available():
        return "cuda:0"
    return "cpu"


@pytest.fixture
def icosahedron(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    icosahedron = tm.creation.icosahedron()
    icosahedron.apply_translation(translation=np.array([-1.0, 0.0, 2.0]))
    return icosahedron, trimesh_to_warp(icosahedron, device)


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
