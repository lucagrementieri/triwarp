"""Regression tests for ``triwarp.smoothing`` against trimesh / igl (CPU reference)."""

from __future__ import annotations

import warnings

import igl
import numpy as np
import pytest
import scipy.sparse.linalg as spla
import trimesh as tm
import trimesh.smoothing as tms
import warp as wp

import triwarp as tw


def _skip_without_cuda(mesh_wp: wp.Mesh) -> None:
    if wp.get_device(mesh_wp.device).is_cpu:
        pytest.skip("implicit smoothing requires a CUDA device (warp.optim.linear.cg)")


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
def test_filter_laplacian_explicit(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    smoothed_wp = tw.smoothing.filter_laplacian(
        mesh_wp.points, mesh_wp.indices, iterations=8, volume_constraint=False
    )
    mesh_ref = mesh_tm.copy()
    tms.filter_laplacian(mesh_ref, iterations=8, volume_constraint=False)

    assert np.allclose(smoothed_wp.numpy(), mesh_ref.vertices, rtol=1e-5, atol=1e-5)


def test_filter_laplacian_volume_constraint(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = icosahedron

    smoothed_wp = tw.smoothing.filter_laplacian(
        mesh_wp.points, mesh_wp.indices, iterations=8, volume_constraint=True
    )
    mesh_ref = mesh_tm.copy()
    tms.filter_laplacian(mesh_ref, iterations=8, volume_constraint=True)

    assert np.allclose(smoothed_wp.numpy(), mesh_ref.vertices, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
def test_filter_humphrey(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    smoothed_wp = tw.smoothing.filter_humphrey(
        mesh_wp.points, mesh_wp.indices, alpha=0.1, beta=0.5, iterations=8
    )
    mesh_ref = mesh_tm.copy()
    tms.filter_humphrey(mesh_ref, alpha=0.1, beta=0.5, iterations=8)

    assert np.allclose(smoothed_wp.numpy(), mesh_ref.vertices, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
def test_filter_taubin(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    smoothed_wp = tw.smoothing.filter_taubin(
        mesh_wp.points, mesh_wp.indices, lamb=0.5, nu=0.53, iterations=9
    )
    mesh_ref = mesh_tm.copy()
    tms.filter_taubin(mesh_ref, lamb=0.5, nu=0.53, iterations=9)

    assert np.allclose(smoothed_wp.numpy(), mesh_ref.vertices, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube"])
def test_filter_mut_dif_laplacian_volume_constraint(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    # Watertight meshes only: the volume constraint inflates along vertex normals, so mesh.volume
    # must be meaningful.
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    smoothed_wp = tw.smoothing.filter_mut_dif_laplacian(
        mesh_wp.points, mesh_wp.indices, lamb=0.5, iterations=8, volume_constraint=True
    )
    mesh_ref = mesh_tm.copy()
    tms.filter_mut_dif_laplacian(mesh_ref, lamb=0.5, iterations=8, volume_constraint=True)

    assert np.allclose(smoothed_wp.numpy(), mesh_ref.vertices, rtol=1e-5, atol=1e-5)


def test_filter_mut_dif_laplacian_no_volume_constraint(
    hemisphere: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    # Open mesh, unconstrained path. Note: the per-vertex adil = 1/|N.(V - L.V)| reciprocal is
    # coupled globally through its mean, so on strongly-saddled meshes (e.g. half_torus) the filter
    # is chaotically sensitive to input precision (the float64 trimesh reference itself diverges by
    # ~1e-2 under a float32 input round-trip). The hemisphere has no such near-zero normal residual,
    # so float32 warp matches the float64 reference tightly.
    mesh_tm, mesh_wp = hemisphere

    smoothed_wp = tw.smoothing.filter_mut_dif_laplacian(
        mesh_wp.points, mesh_wp.indices, lamb=0.5, iterations=8, volume_constraint=False
    )
    mesh_ref = mesh_tm.copy()
    tms.filter_mut_dif_laplacian(mesh_ref, lamb=0.5, iterations=8, volume_constraint=False)

    assert np.allclose(smoothed_wp.numpy(), mesh_ref.vertices, rtol=1e-5, atol=1e-5)


def test_filter_laplacian_implicit(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = icosahedron
    _skip_without_cuda(mesh_wp)

    smoothed_wp = tw.smoothing.filter_laplacian(
        mesh_wp.points,
        mesh_wp.indices,
        lamb=0.5,
        iterations=6,
        implicit_time_integration=True,
        volume_constraint=False,
    )
    mesh_ref = mesh_tm.copy()
    tms.filter_laplacian(
        mesh_ref, lamb=0.5, iterations=6, implicit_time_integration=True, volume_constraint=False
    )

    assert np.allclose(smoothed_wp.numpy(), mesh_ref.vertices, rtol=1e-4, atol=1e-4)


def test_filter_implicit_fairing(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    # Implicit curvature flow is defined for closed meshes; on open boundaries the unconstrained
    # flow degrades boundary triangles and the conjugate-gradient solve diverges (igl's direct
    # solver tolerates it, Warp only offers CG), so the regression uses the watertight icosahedron.
    mesh_tm, mesh_wp = icosahedron
    _skip_without_cuda(mesh_wp)

    lamb = 0.1
    iterations = 6
    vertices_igl = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_igl = np.array(mesh_tm.faces, dtype=np.int64)
    for _ in range(iterations):
        stiffness_igl = igl.cotmatrix(vertices_igl, faces_igl)
        mass_igl = igl.massmatrix(vertices_igl, faces_igl, igl.MASSMATRIX_TYPE_BARYCENTRIC)
        system_igl = (mass_igl - lamb * stiffness_igl).tocsc()
        vertices_igl = spla.spsolve(system_igl, mass_igl.dot(vertices_igl))

    smoothed_wp = tw.smoothing.filter_implicit_fairing(
        mesh_wp.points, mesh_wp.indices, lamb=lamb, iterations=iterations
    )

    assert np.allclose(smoothed_wp.numpy(), vertices_igl, rtol=1e-4, atol=1e-4)


def test_filter_laplacian_pluggable_operator(half_torus: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = half_torus
    operator = tw.laplacian.laplacian(mesh_wp.points, mesh_wp.indices, equal_weight=False)

    smoothed_wp = tw.smoothing.filter_laplacian(
        mesh_wp.points,
        mesh_wp.indices,
        iterations=6,
        volume_constraint=False,
        laplacian_operator=operator,
    )
    mesh_ref = mesh_tm.copy()
    operator_tm = tms.laplacian_calculation(mesh_ref, equal_weight=False)
    tms.filter_laplacian(
        mesh_ref, iterations=6, volume_constraint=False, laplacian_operator=operator_tm
    )

    assert np.allclose(smoothed_wp.numpy(), mesh_ref.vertices, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
def test_filter_neighborhood_average(request: pytest.FixtureRequest, mesh_name: str) -> None:
    o3d = pytest.importorskip("open3d")
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    iterations = 5

    smoothed_wp = tw.smoothing.filter_neighborhood_average(
        mesh_wp.points, mesh_wp.indices, iterations=iterations
    )

    mesh_o3d = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh_tm.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(mesh_tm.faces, dtype=np.int32)),
    )
    mesh_o3d = mesh_o3d.filter_smooth_simple(number_of_iterations=iterations)

    assert np.allclose(smoothed_wp.numpy(), np.asarray(mesh_o3d.vertices), rtol=1e-5, atol=1e-5)


def test_filter_neighborhood_average_zero_iterations(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    _, mesh_wp = icosahedron
    smoothed_wp = tw.smoothing.filter_neighborhood_average(
        mesh_wp.points, mesh_wp.indices, iterations=0
    )
    assert np.array_equal(smoothed_wp.numpy(), mesh_wp.points.numpy())


def test_cpu_implicit_raises() -> None:
    vertices = wp.array(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        dtype=wp.vec3,
        device="cpu",
    )
    faces = wp.array([0, 1, 2], dtype=wp.int32, device="cpu")

    with pytest.raises(NotImplementedError):
        tw.smoothing.filter_laplacian(vertices, faces, iterations=2, implicit_time_integration=True)
    with pytest.raises(NotImplementedError):
        tw.smoothing.filter_implicit_fairing(vertices, faces, iterations=2)


def test_zero_iterations_returns_copy(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _, mesh_wp = icosahedron
    smoothed_wp = tw.smoothing.filter_taubin(mesh_wp.points, mesh_wp.indices, iterations=0)
    assert np.array_equal(smoothed_wp.numpy(), mesh_wp.points.numpy())


def test_empty_mesh(device: str) -> None:
    vertices = wp.empty(0, dtype=wp.vec3, device=device)
    faces = wp.empty(0, dtype=wp.int32, device=device)
    smoothed_wp = tw.smoothing.filter_taubin(vertices, faces, iterations=4)
    assert int(smoothed_wp.shape[0]) == 0


# ---------------------------------------------------------------------------
# Region smoothing solves vs MeshLib (positionVertsSmoothly / SharpBd)
# ---------------------------------------------------------------------------

_meshlib = pytest.importorskip("meshlib")
from meshlib import mrmeshnumpy as _mn  # noqa: E402
from meshlib import mrmeshpy as _mm  # noqa: E402


def _sphere_region(subdivisions: int = 2, z_cut: float = 0.5):
    sph = tm.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    vertices = sph.vertices.astype(np.float64)
    faces = sph.faces.astype(np.int32)
    free = vertices[:, 2] > z_cut
    return vertices, faces, free


def test_position_verts_smoothly_sharp_boundary_matches_meshlib(device: str):
    if wp.get_device(device).is_cpu:
        pytest.skip("region smoothing requires a CUDA device (warp.optim.linear.cg)")
    vertices_np, faces_np, free_np = _sphere_region()
    v_wp = wp.array(vertices_np, dtype=wp.vec3, device=device)
    f_wp = wp.array(faces_np.reshape(-1), dtype=wp.int32, device=device)
    free_wp = wp.array(free_np, dtype=wp.bool, device=device)

    result_wp = tw.smoothing.position_verts_smoothly_sharp_boundary(v_wp, f_wp, free_wp)

    mesh_ml = _mn.meshFromFacesVerts(faces_np, vertices_np.astype(np.float32))
    params_ml = _mm.PositionVertsSmoothlyParams()
    params_ml.region = _mn.vertBitSetFromBools(free_np)
    _mm.positionVertsSmoothlySharpBd(mesh_ml, params_ml)
    verts_ml = _mn.getNumpyVerts(mesh_ml)

    assert np.allclose(result_wp.numpy(), verts_ml, rtol=1e-5, atol=1e-5)
    assert np.array_equal(result_wp.numpy()[~free_np], v_wp.numpy()[~free_np])


@pytest.mark.parametrize("edge_weights", ["cotan", "unit"])
def test_position_verts_smoothly_matches_meshlib(device: str, edge_weights: str):
    if wp.get_device(device).is_cpu:
        pytest.skip("region smoothing requires a CUDA device (warp.optim.linear.cg)")
    vertices_np, faces_np, free_np = _sphere_region()
    v_wp = wp.array(vertices_np, dtype=wp.vec3, device=device)
    f_wp = wp.array(faces_np.reshape(-1), dtype=wp.int32, device=device)
    free_wp = wp.array(free_np, dtype=wp.bool, device=device)

    result_wp = tw.smoothing.position_verts_smoothly(v_wp, f_wp, free_wp, edge_weights=edge_weights)

    mesh_ml = _mn.meshFromFacesVerts(faces_np, vertices_np.astype(np.float32))
    ew_ml = _mm.EdgeWeights.Cotan if edge_weights == "cotan" else _mm.EdgeWeights.Unit
    _mm.positionVertsSmoothly(mesh_ml, _mn.vertBitSetFromBools(free_np), ew_ml, _mm.VertexMass.Unit)
    verts_ml = _mn.getNumpyVerts(mesh_ml)

    assert np.allclose(result_wp.numpy(), verts_ml, rtol=1e-4, atol=1e-4)
    assert np.array_equal(result_wp.numpy()[~free_np], v_wp.numpy()[~free_np])


def test_position_verts_sharp_boundary_dirichlet_residual(device: str):
    if wp.get_device(device).is_cpu:
        pytest.skip("region smoothing requires a CUDA device (warp.optim.linear.cg)")
    vertices_np, faces_np, free_np = _sphere_region()
    v_wp = wp.array(vertices_np, dtype=wp.vec3, device=device)
    f_wp = wp.array(faces_np.reshape(-1), dtype=wp.int32, device=device)
    free_wp = wp.array(free_np, dtype=wp.bool, device=device)

    result = tw.smoothing.position_verts_smoothly_sharp_boundary(v_wp, f_wp, free_wp).numpy()

    # Umbrella (unit-weight) residual: deg(v) * p_v - sum_neighbors p_d == 0 for every free vertex.
    adjacency: dict[int, list[int]] = {}
    for tri in faces_np:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            adjacency.setdefault(int(a), set()).add(int(b))
            adjacency.setdefault(int(b), set()).add(int(a))
    max_residual = 0.0
    for v in np.flatnonzero(free_np):
        neighbors = list(adjacency[int(v)])
        residual = len(neighbors) * result[v] - result[neighbors].sum(axis=0)
        max_residual = max(max_residual, float(np.linalg.norm(residual)))
    assert max_residual < 1e-4


def test_position_verts_smoothly_empty_region(device: str):
    vertices_np, faces_np, _ = _sphere_region()
    v_wp = wp.array(vertices_np, dtype=wp.vec3, device=device)
    f_wp = wp.array(faces_np.reshape(-1), dtype=wp.int32, device=device)
    empty = wp.zeros(len(vertices_np), dtype=wp.bool, device=device)
    result = tw.smoothing.position_verts_smoothly_sharp_boundary(v_wp, f_wp, empty)
    assert np.array_equal(result.numpy(), v_wp.numpy())


def test_filter_implicit_fairing_pins_the_boundary(hemisphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """Boundary vertices are held exactly; the interior is the part that moves."""
    _, mesh_wp = hemisphere
    _skip_without_cuda(mesh_wp)
    original_np = mesh_wp.points.numpy()
    boundary_np = tw.boundary.boundary_vertex_indices(mesh_wp.points, mesh_wp.indices).numpy()
    interior_np = np.setdiff1d(np.arange(len(original_np)), boundary_np)
    assert len(boundary_np) > 0  # the fixture must actually be open for this to mean anything

    smoothed_np = tw.smoothing.filter_implicit_fairing(
        mesh_wp.points, mesh_wp.indices, iterations=5
    ).numpy()

    assert np.array_equal(smoothed_np[boundary_np], original_np[boundary_np])
    assert np.abs(smoothed_np[interior_np] - original_np[interior_np]).max() > 1e-6


def test_filter_implicit_fairing_pinned_stays_stable_on_an_open_mesh(
    hemisphere: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Many passes on an open mesh converge instead of diverging.

    The unconstrained flow pulls the free boundary inward until the triangles there collapse, after
    which the system is effectively singular and the conjugate gradient runs to its iteration cap.
    Pinning makes each pass a Dirichlet problem over the interior, which is well posed however many
    times it is applied — so this asserts *no* non-convergence warning, not merely finiteness.
    """
    _, mesh_wp = hemisphere
    _skip_without_cuda(mesh_wp)
    extent = float(np.abs(mesh_wp.points.numpy()).max())

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        smoothed_np = tw.smoothing.filter_implicit_fairing(
            mesh_wp.points, mesh_wp.indices, iterations=25
        ).numpy()

    assert np.isfinite(smoothed_np).all()
    # Smoothing cannot inflate the patch beyond its pinned rim.
    assert np.abs(smoothed_np).max() <= extent * 1.01


def test_filter_implicit_fairing_pin_boundary_is_a_no_op_on_a_closed_mesh(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    A watertight mesh has no boundary, so the flag routes down the same unreduced solve either way.

    Compared at ``float64`` round-off rather than bitwise: the sparse mat-vec accumulates with
    atomics, so *any* two runs of this function differ in the last bits, flag or no flag.
    """
    _, mesh_wp = icosahedron
    _skip_without_cuda(mesh_wp)
    assert int(tw.boundary.boundary_vertex_indices(mesh_wp.points, mesh_wp.indices).shape[0]) == 0

    pinned_np = tw.smoothing.filter_implicit_fairing(
        mesh_wp.points, mesh_wp.indices, iterations=6, pin_boundary=True
    ).numpy()
    unpinned_np = tw.smoothing.filter_implicit_fairing(
        mesh_wp.points, mesh_wp.indices, iterations=6, pin_boundary=False
    ).numpy()

    assert np.allclose(pinned_np, unpinned_np, rtol=0, atol=1e-12)
