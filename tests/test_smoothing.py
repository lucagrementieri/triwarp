"""Regression tests for ``triwarp.smoothing`` against trimesh / igl (CPU reference)."""

from __future__ import annotations

import warnings

import igl
import numpy as np
import pymeshlab as ml
import pytest
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import scipy.sparse.linalg as spla
import trimesh as tm
import trimesh.smoothing as tms
import warp as wp
import warp.sparse as wps

import triwarp as tw
from tests.conversions import trimesh_to_open3d, trimesh_to_pymeshlab, trimesh_to_warp


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
@pytest.mark.parity("filter_humphrey", "trimesh")
def test_filter_humphrey(request: pytest.FixtureRequest, mesh_name: str) -> None:
    """
    HC filtering against trimesh, which is the only oracle this filter has.

    **pymeshlab is not a second oracle here**, recorded so it is not re-tried:
    ``apply_coord_hc_laplacian_smoothing`` implements the same Vollmer et al. paper but exposes no
    parameters at all -- no step count, no ``alpha``, no ``beta`` -- and its single pass matches
    ``filter_humphrey`` at *none* of the 8 x 11 x 11 ``(iterations, alpha, beta)`` combinations
    probed (best max-coordinate deviation 0.019 on a mesh carrying 0.016 of noise). It is a
    different formulation, not this one with other constants, and it is used only as a per-pass
    timing reference in ``benchmarks/test_smoothing.py``.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    smoothed_wp = tw.smoothing.filter_humphrey(
        mesh_wp.points, mesh_wp.indices, alpha=0.1, beta=0.5, iterations=8
    )
    mesh_ref = mesh_tm.copy()
    tms.filter_humphrey(mesh_ref, alpha=0.1, beta=0.5, iterations=8)

    assert np.allclose(smoothed_wp.numpy(), mesh_ref.vertices, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
@pytest.mark.parity("filter_taubin", "trimesh")
def test_filter_taubin(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)

    smoothed_wp = tw.smoothing.filter_taubin(
        mesh_wp.points, mesh_wp.indices, lamb=0.5, nu=0.53, iterations=9
    )
    mesh_ref = mesh_tm.copy()
    tms.filter_taubin(mesh_ref, lamb=0.5, nu=0.53, iterations=9)

    assert np.allclose(smoothed_wp.numpy(), mesh_ref.vertices, rtol=1e-5, atol=1e-5)


# --- MeshLab's two coordinate umbrellas -------------------------------------------------
#
# MeshLab does not use one Laplacian for its coordinate smoothers, it uses two, and which one a
# filter picks was recovered numerically rather than read off any documentation: one pass of each
# filter is a linear map, so running it over 12 random position sets on one connectivity and solving
# least-squares for the per-vertex stencil determines the weights exactly (residual 2e-16).
#
#   * ``apply_coord_laplacian_smoothing`` and ``apply_coord_unsharp_mask`` weight each neighbour by
#     the number of faces the edge to it shares and count the vertex *itself* once, so on a closed
#     mesh a degree-d vertex gets ``1 / (2d + 1)`` on itself and ``2 / (2d + 1)`` on each neighbour.
#     That is `_meshlab_umbrella` below, and it is why these two filters are NOT the plain 1-ring
#     mean -- the difference is 8% of the displacement, far too large to read as a tolerance.
#   * ``apply_coord_taubin_smoothing`` uses the plain 1-ring mean instead, which is triwarp's own
#     default operator.


def _meshlab_umbrella(mesh_tm: tm.Trimesh, device: str) -> wps.BsrMatrix[wp.float32]:
    """
    Assemble MeshLab's face-count-weighted umbrella as a row-stochastic operator.

    Each neighbour is weighted by the number of faces its edge shares (2 for an interior edge, 1 on
    a boundary) and the vertex itself by 1, then the row is normalized. Feeding this to triwarp's
    ``laplacian_operator=`` parameter is what makes the comparison class B rather than an exemption.
    """
    shared: dict[tuple[int, int], int] = {}
    for face_np in mesh_tm.faces:
        for start, end in ((0, 1), (1, 2), (2, 0)):
            key = (int(face_np[start]), int(face_np[end]))
            shared[key] = shared.get(key, 0) + 1
            shared[key[::-1]] = shared.get(key[::-1], 0) + 1

    n_vertices = mesh_tm.vertices.shape[0]
    row_sum_np = np.ones(n_vertices)
    for (row, _column), count in shared.items():
        row_sum_np[row] += count
    rows_np = np.arange(n_vertices, dtype=np.int32)
    columns_np = rows_np.copy()
    values_np = 1.0 / row_sum_np
    entries_np = np.array(list(shared.items()), dtype=object)
    neighbor_rows = np.array([row for (row, _column), _count in entries_np], dtype=np.int32)
    neighbor_columns = np.array([column for (_row, column), _count in entries_np], dtype=np.int32)
    neighbor_values = (
        np.array([count for _key, count in entries_np], dtype=np.float64)
        / row_sum_np[neighbor_rows]
    )

    return wps.bsr_from_triplets(
        n_vertices,
        n_vertices,
        wp.array(np.concatenate((rows_np, neighbor_rows)), dtype=wp.int32, device=device),
        wp.array(np.concatenate((columns_np, neighbor_columns)), dtype=wp.int32, device=device),
        wp.array(
            np.concatenate((values_np, neighbor_values)).astype(np.float32),
            dtype=wp.float32,
            device=device,
        ),
    )


def _noisy_icosphere() -> tm.Trimesh:
    """Build a closed, evenly tessellated mesh carrying 0.01 of noise for a smoother to remove."""
    mesh_tm = tm.creation.icosphere(subdivisions=2)
    mesh_tm.vertices = mesh_tm.vertices + np.random.default_rng(0).normal(
        0.0, 0.01, mesh_tm.vertices.shape
    )
    return mesh_tm


@pytest.mark.parametrize("iterations", [1, 3, 10])
@pytest.mark.parity("filter_laplacian_integration", "pymeshlab")
def test_filter_laplacian_matches_pymeshlab(device: str, iterations: int) -> None:
    """
    Class B: exact, once MeshLab's own umbrella is passed through ``laplacian_operator=``.

    Three named transforms. ``cotangentweight=False`` selects the uniform scheme (the benchmark
    passes it too); ``lamb=1.0`` matches MeshLab's full step, which replaces the vertex by the
    weighted mean rather than interpolating toward it; and the operator is
    [`_meshlab_umbrella`][tests.test_smoothing._meshlab_umbrella] rather than triwarp's default,
    because MeshLab's uniform Laplacian is *not* the 1-ring mean.

    Measured agreement **6.5e-08 / 1.2e-07 / 2.6e-07** at 1 / 3 / 10 passes -- float32 accumulation
    noise -- against **5.0e-03** at one pass with the plain 1-ring mean. So this is the assert that
    pins which of MeshLab's two umbrellas this filter uses; with the wrong one it fails by 4 orders
    of magnitude. This also exercises the pluggable-operator path the benchmark's
    ``filter_laplacian_integration`` group times.

    A closed fixture, because the ``1`` self-weight and per-face neighbour counts were solved for on
    one; the boundary generalization (a shared count of 1) is written but not asserted here.
    """
    mesh_tm = _noisy_icosphere()
    mesh_wp = trimesh_to_warp(mesh_tm, device)

    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.apply_coord_laplacian_smoothing(
        stepsmoothnum=iterations, cotangentweight=False, boundary=True
    )

    smoothed_wp = tw.smoothing.filter_laplacian(
        mesh_wp.points,
        mesh_wp.indices,
        lamb=1.0,
        iterations=iterations,
        volume_constraint=False,
        laplacian_operator=_meshlab_umbrella(mesh_tm, device),
    )
    assert np.allclose(
        smoothed_wp.numpy(), meshset_pml.current_mesh().vertex_matrix(), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("steps", [1, 3, 5])
@pytest.mark.parity("filter_taubin", "pymeshlab")
def test_filter_taubin_matches_pymeshlab(device: str, steps: int) -> None:
    """
    Class B: exact, once the *pass count* is mapped -- and the mapping is a factor of two.

    MeshLab's ``stepsmoothnum`` counts lambda-mu **pairs**, where triwarp follows trimesh and does
    one half-step per ``iterations``, alternating the shrinking and inflating passes. So
    ``iterations = 2 * stepsmoothnum`` is the named transform, and ``mu=-0.53`` is triwarp's
    ``nu=0.53`` (the sign is in the convention, not the value). Unlike the two filters above, this
    one uses the plain 1-ring mean, so triwarp's default operator is the right one.

    Measured **5.2e-08 / 4.4e-08 / 4.6e-08** at 1 / 3 / 5 MeshLab steps under the doubled count,
    against **2.7e-02 / 2.6e-02 / 2.6e-02** at the naive equal count -- a 5-orders-of-magnitude
    separation, so this assert is what holds the pass mapping in place.
    ``benchmarks/test_smoothing`` originally gave MeshLab ``stepsmoothnum=_ITERATIONS`` against
    triwarp's ``iterations=_ITERATIONS`` and so timed it doing twice the passes; that row now halves
    it.
    """
    mesh_tm = _noisy_icosphere()
    mesh_wp = trimesh_to_warp(mesh_tm, device)

    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.apply_coord_taubin_smoothing(lambda_=0.5, mu=-0.53, stepsmoothnum=steps)

    smoothed_wp = tw.smoothing.filter_taubin(
        mesh_wp.points, mesh_wp.indices, lamb=0.5, nu=0.53, iterations=2 * steps
    )
    assert np.allclose(
        smoothed_wp.numpy(), meshset_pml.current_mesh().vertex_matrix(), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("iterations", [1, 5])
@pytest.mark.parity("filter_sharpen", "pymeshlab")
def test_filter_sharpen_matches_pymeshlab(device: str, iterations: int) -> None:
    """
    Class B: exact under the same operator substitution as the Laplacian test above.

    See [`test_filter_laplacian_matches_pymeshlab`]
    [tests.test_smoothing.test_filter_laplacian_matches_pymeshlab] for how the operator is derived.

    ``apply_coord_unsharp_mask`` takes ``weight`` and ``iterations`` under triwarp's own names and
    meanings, and ``weightorig=1.0`` is its default (the original-position weight triwarp fixes at
    1), so the only transform is passing
    [`_meshlab_umbrella`][tests.test_smoothing._meshlab_umbrella] as the smoothing operator.

    Measured **1.4e-07** at both 1 and 5 iterations, against **1.5e-03 / 4.4e-03** with the plain
    1-ring mean -- so, like the Laplacian test, this doubles as the check on which umbrella the
    filter uses.
    """
    mesh_tm = _noisy_icosphere()
    mesh_wp = trimesh_to_warp(mesh_tm, device)

    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.apply_coord_unsharp_mask(weight=0.3, weightorig=1.0, iterations=iterations)

    sharpened_wp = tw.smoothing.filter_sharpen(
        mesh_wp.points,
        mesh_wp.indices,
        weight=0.3,
        iterations=iterations,
        laplacian_operator=_meshlab_umbrella(mesh_tm, device),
    )
    assert np.allclose(
        sharpened_wp.numpy(), meshset_pml.current_mesh().vertex_matrix(), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("mesh_name", ["icosahedron", "cave_cube"])
@pytest.mark.parity("filter_mut_dif_laplacian", "trimesh")
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


def test_filter_laplacian_implicit_duplicate_built_operator(
    half_torus: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    A duplicate-built operator gives the same result as its compact form.

    ``cotmatrix`` emits 12 triplets per face, so ``operator.nnz`` -- the triplet *capacity*
    ``bsr_from_triplets`` was handed -- overshoots ``nnz_sync()`` by ~3.4x. The implicit system used
    to size its ``wp.empty`` triplet buffers by ``nnz``, leaving that gap uninitialized for
    ``bsr_from_triplets`` to read back as triplets: out-of-range garbage indices are dropped
    silently, but any landing in ``[0, n)`` accumulate a junk value into a real entry. Measured
    ``‖values‖ = 1.1e13`` against a correct 84.3, and every vertex ``NaN`` end to end. Rebuilt
    sliced to ``nnz_sync()`` the same operator is compact, so the two must agree.
    """
    _, mesh_wp = half_torus
    # Two independent builds of the same operator. ``nnz_sync()`` repairs the stale ``nnz`` cache
    # *in place*, so measuring the capacity on one build would hand the filter a repaired matrix and
    # the test would pass whatever the implementation does -- the operator under test has to be a
    # build nothing has synced.
    unsynced = tw.laplacian.cotmatrix(mesh_wp.points, mesh_wp.indices, dtype=wp.float32)
    reference = tw.laplacian.cotmatrix(mesh_wp.points, mesh_wp.indices, dtype=wp.float32)
    capacity = int(reference.nnz)
    nnz = reference.nnz_sync()
    # Guard the guard: without a real capacity gap this test compares a matrix with itself.
    assert capacity > nnz
    compact = wps.bsr_from_triplets(
        int(reference.nrow),
        int(reference.ncol),
        reference.uncompress_rows()[:nnz],
        reference.columns[:nnz],
        reference.values[:nnz],
        prune_numerical_zeros=False,
    )

    # Random garbage is almost always out of range, and ``bsr_from_triplets`` drops an out-of-range
    # index silently -- which is exactly why the defect stayed invisible. Leaving plausible in-range
    # indices in the memory pool makes the unwritten tail reachable, so the assertions below have
    # something to catch.
    n_vertices = int(mesh_wp.points.shape[0])
    for _ in range(6):
        _junk = (
            wp.full(capacity + n_vertices, 7, dtype=wp.int32, device=mesh_wp.points.device),
            wp.full(capacity + n_vertices, 11, dtype=wp.int32, device=mesh_wp.points.device),
            wp.full(capacity + n_vertices, 1.0e9, dtype=wp.float64, device=mesh_wp.points.device),
        )
        del _junk
    wp.synchronize()

    smoothed_wp = tw.smoothing.filter_laplacian(
        mesh_wp.points,
        mesh_wp.indices,
        iterations=2,
        implicit_time_integration=True,
        volume_constraint=False,
        laplacian_operator=unsynced,
    )
    compact_wp = tw.smoothing.filter_laplacian(
        mesh_wp.points,
        mesh_wp.indices,
        iterations=2,
        implicit_time_integration=True,
        volume_constraint=False,
        laplacian_operator=compact,
    )

    assert np.all(np.isfinite(smoothed_wp.numpy()))
    assert np.allclose(smoothed_wp.numpy(), compact_wp.numpy(), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
def test_filter_neighborhood_average(request: pytest.FixtureRequest, mesh_name: str) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    iterations = 5

    smoothed_wp = tw.smoothing.filter_neighborhood_average(
        mesh_wp.points, mesh_wp.indices, iterations=iterations
    )

    mesh_o3d = trimesh_to_open3d(mesh_tm).filter_smooth_simple(number_of_iterations=iterations)

    assert np.allclose(smoothed_wp.numpy(), np.asarray(mesh_o3d.vertices), rtol=1e-5, atol=1e-5)


def test_filter_neighborhood_average_zero_iterations(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    _, mesh_wp = icosahedron
    smoothed_wp = tw.smoothing.filter_neighborhood_average(
        mesh_wp.points, mesh_wp.indices, iterations=0
    )
    assert np.array_equal(smoothed_wp.numpy(), mesh_wp.points.numpy())


def test_implicit_filters_cpu_match_cuda(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """
    Class A: both implicit filters give the CUDA answer on CPU.

    Pins the removal of the CUDA-only guard: ``warp.optim.linear.cg`` returned NaN on the Warp CPU
    device through 1.15, so these two refused to run there at all. Fixed in 1.16.
    """
    if not wp.is_cuda_available():
        pytest.skip("needs both devices to compare them")
    mesh_tm, _ = icosahedron

    for name, call in (
        (
            "filter_laplacian",
            lambda v, f: tw.smoothing.filter_laplacian(
                v, f, iterations=2, implicit_time_integration=True
            ),
        ),
        (
            "filter_implicit_fairing",
            lambda v, f: tw.smoothing.filter_implicit_fairing(v, f, iterations=2),
        ),
    ):
        positions = {}
        for device in ("cpu", "cuda:0"):
            mesh_wp = trimesh_to_warp(mesh_tm, device)
            positions[device] = call(mesh_wp.points, mesh_wp.indices).numpy()
        assert np.isfinite(positions["cpu"]).all(), name
        assert np.allclose(positions["cpu"], positions["cuda:0"], rtol=1e-5, atol=1e-5), name


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


def test_smooth_region_fixed_rim_matches_meshlib(device: str):
    if wp.get_device(device).is_cpu:
        pytest.skip("region smoothing requires a CUDA device (warp.optim.linear.cg)")
    vertices_np, faces_np, free_np = _sphere_region()
    v_wp = wp.array(vertices_np, dtype=wp.vec3, device=device)
    f_wp = wp.array(faces_np.reshape(-1), dtype=wp.int32, device=device)
    free_wp = wp.array(free_np, dtype=wp.bool, device=device)

    result_wp = tw.smoothing.smooth_region_fixed_rim(v_wp, f_wp, free_wp)

    mesh_ml = _mn.meshFromFacesVerts(faces_np, vertices_np.astype(np.float32))
    params_ml = _mm.PositionVertsSmoothlyParams()
    params_ml.region = _mn.vertBitSetFromBools(free_np)
    _mm.positionVertsSmoothlySharpBd(mesh_ml, params_ml)
    verts_ml = _mn.getNumpyVerts(mesh_ml)

    assert np.allclose(result_wp.numpy(), verts_ml, rtol=1e-5, atol=1e-5)
    assert np.array_equal(result_wp.numpy()[~free_np], v_wp.numpy()[~free_np])


@pytest.mark.parametrize("edge_weights", ["cotan", "unit"])
def test_smooth_region_matches_meshlib(device: str, edge_weights: str):
    if wp.get_device(device).is_cpu:
        pytest.skip("region smoothing requires a CUDA device (warp.optim.linear.cg)")
    vertices_np, faces_np, free_np = _sphere_region()
    v_wp = wp.array(vertices_np, dtype=wp.vec3, device=device)
    f_wp = wp.array(faces_np.reshape(-1), dtype=wp.int32, device=device)
    free_wp = wp.array(free_np, dtype=wp.bool, device=device)

    result_wp = tw.smoothing.smooth_region(v_wp, f_wp, free_wp, edge_weights=edge_weights)

    mesh_ml = _mn.meshFromFacesVerts(faces_np, vertices_np.astype(np.float32))
    ew_ml = _mm.EdgeWeights.Cotan if edge_weights == "cotan" else _mm.EdgeWeights.Unit
    _mm.positionVertsSmoothly(mesh_ml, _mn.vertBitSetFromBools(free_np), ew_ml, _mm.VertexMass.Unit)
    verts_ml = _mn.getNumpyVerts(mesh_ml)

    assert np.allclose(result_wp.numpy(), verts_ml, rtol=1e-4, atol=1e-4)
    assert np.array_equal(result_wp.numpy()[~free_np], v_wp.numpy()[~free_np])


def test_smooth_region_fixed_rim_dirichlet_residual(device: str):
    if wp.get_device(device).is_cpu:
        pytest.skip("region smoothing requires a CUDA device (warp.optim.linear.cg)")
    vertices_np, faces_np, free_np = _sphere_region()
    v_wp = wp.array(vertices_np, dtype=wp.vec3, device=device)
    f_wp = wp.array(faces_np.reshape(-1), dtype=wp.int32, device=device)
    free_wp = wp.array(free_np, dtype=wp.bool, device=device)

    result = tw.smoothing.smooth_region_fixed_rim(v_wp, f_wp, free_wp).numpy()

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


def test_smooth_region_empty_region(device: str):
    vertices_np, faces_np, _ = _sphere_region()
    v_wp = wp.array(vertices_np, dtype=wp.vec3, device=device)
    f_wp = wp.array(faces_np.reshape(-1), dtype=wp.int32, device=device)
    empty = wp.zeros(len(vertices_np), dtype=wp.bool, device=device)
    result = tw.smoothing.smooth_region_fixed_rim(v_wp, f_wp, empty)
    assert np.array_equal(result.numpy(), v_wp.numpy())


def test_filter_implicit_fairing_pins_the_boundary(hemisphere: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """Boundary vertices are held exactly; the interior is the part that moves."""
    _, mesh_wp = hemisphere
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
    assert int(tw.boundary.boundary_vertex_indices(mesh_wp.points, mesh_wp.indices).shape[0]) == 0

    pinned_np = tw.smoothing.filter_implicit_fairing(
        mesh_wp.points, mesh_wp.indices, iterations=6, pin_boundary=True
    ).numpy()
    unpinned_np = tw.smoothing.filter_implicit_fairing(
        mesh_wp.points, mesh_wp.indices, iterations=6, pin_boundary=False
    ).numpy()

    assert np.allclose(pinned_np, unpinned_np, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# Scalar-field filters vs pymeshlab (apply_scalar_smoothing / _saturation_per_vertex)
# ---------------------------------------------------------------------------


def _scalar_spike(mesh_tm: tm.Trimesh) -> np.ndarray:
    """Build a delta at vertex 0: the field with the steepest possible gradient, in float64."""
    values_np = np.zeros(mesh_tm.vertices.shape[0], dtype=np.float64)
    values_np[0] = 10.0
    return values_np


def _meshset_with_scalars(mesh_tm: tm.Trimesh, values_np: np.ndarray) -> ml.MeshSet:
    meshset_pml = ml.MeshSet()
    meshset_pml.add_mesh(
        ml.Mesh(
            np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64),
            np.ascontiguousarray(mesh_tm.faces, dtype=np.int32),
            v_scalar_array=np.ascontiguousarray(values_np, dtype=np.float64),
        )
    )
    return meshset_pml


@pytest.mark.parametrize("mesh_name", ["icosahedron", "torus", "cave_cube"])
@pytest.mark.parity("filter_scalar_laplacian", "pymeshlab")
def test_filter_scalar_laplacian_matches_pymeshlab(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    One full-step pass is exactly MeshLab's ``apply_scalar_smoothing_per_vertex``.

    **Closed fixtures only**, and deliberately so: VCG's ``VertexQualityLaplacian`` smooths a
    *boundary* vertex along the boundary curve alone — averaging only its two boundary neighbours
    and discarding the rest of its ring — while this port averages every ring for every vertex.
    Measured on ``hemisphere``, a boundary vertex two boundary-steps from a spike reads ``5.0``
    there against ``3.33`` here, because its divisor is 2 rather than its degree 3. That is a
    different operator, not a tolerance, so the oracle is only claimed where no vertex has a
    boundary.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    values_np = _scalar_spike(mesh_tm)

    meshset_pml = _meshset_with_scalars(mesh_tm, values_np)
    meshset_pml.apply_scalar_smoothing_per_vertex()

    values_wp = wp.array(values_np.astype(np.float32), dtype=wp.float32, device=mesh_wp.device)
    smoothed_wp = tw.smoothing.filter_scalar_laplacian(
        values_wp, mesh_wp.points, mesh_wp.indices, lamb=1.0, iterations=1
    )
    assert np.allclose(
        smoothed_wp.numpy(), meshset_pml.current_mesh().vertex_scalar_array(), rtol=1e-5, atol=1e-6
    )


def test_filter_scalar_laplacian_conserves_the_mean_on_a_closed_mesh(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    Repeated passes flatten the field without moving its total, on a mesh of uniform valence.

    The averaging operator is row-stochastic but not column-stochastic in general, so the sum is
    only conserved when every vertex has the same degree — which an icosahedron does (valence 5
    everywhere). That makes it the one fixture where this is an exact invariant.
    """
    mesh_tm, mesh_wp = icosahedron
    values_np = _scalar_spike(mesh_tm)
    values_wp = wp.array(values_np.astype(np.float32), dtype=wp.float32, device=mesh_wp.device)

    smoothed_wp = tw.smoothing.filter_scalar_laplacian(
        values_wp, mesh_wp.points, mesh_wp.indices, lamb=0.5, iterations=40
    )
    assert np.isclose(smoothed_wp.numpy().sum(), values_np.sum(), rtol=1e-4)
    assert smoothed_wp.numpy().std() < values_np.std()
    assert np.array_equal(values_wp.numpy(), values_np.astype(np.float32))  # input untouched


def test_filter_scalar_laplacian_zero_iterations(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = icosahedron
    values_np = _scalar_spike(mesh_tm).astype(np.float32)
    values_wp = wp.array(values_np, dtype=wp.float32, device=mesh_wp.device)
    out_wp = tw.smoothing.filter_scalar_laplacian(
        values_wp, mesh_wp.points, mesh_wp.indices, iterations=0
    )
    assert np.array_equal(out_wp.numpy(), values_np)


def test_filter_scalar_laplacian_length_mismatch(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    values_wp = wp.zeros(3, dtype=wp.float32, device=mesh_wp.device)
    with pytest.raises(ValueError, match="one entry per vertex"):
        tw.smoothing.filter_scalar_laplacian(values_wp, mesh_wp.points, mesh_wp.indices)


@pytest.mark.parametrize("threshold", [0.5, 1.0, 3.0])
@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
@pytest.mark.parity("saturate_scalar_gradient", "pymeshlab")
def test_saturate_scalar_gradient_matches_pymeshlab(
    request: pytest.FixtureRequest, mesh_name: str, threshold: float
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    values_np = _scalar_spike(mesh_tm)

    meshset_pml = _meshset_with_scalars(mesh_tm, values_np)
    meshset_pml.apply_scalar_saturation_per_vertex(gradientthr=threshold)

    values_wp = wp.array(values_np.astype(np.float32), dtype=wp.float32, device=mesh_wp.device)
    saturated_wp = tw.smoothing.saturate_scalar_gradient(
        values_wp, mesh_wp.points, mesh_wp.indices, threshold=threshold
    )
    assert np.allclose(
        saturated_wp.numpy(), meshset_pml.current_mesh().vertex_scalar_array(), rtol=1e-4, atol=1e-5
    )


def test_saturate_scalar_gradient_respects_the_bound(
    half_torus: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """After convergence no edge violates the cap, and nothing was raised."""
    mesh_tm, mesh_wp = half_torus
    threshold = 2.0
    rng = np.random.default_rng(7)
    values_np = rng.uniform(0.0, 5.0, size=mesh_tm.vertices.shape[0])
    values_wp = wp.array(values_np.astype(np.float32), dtype=wp.float32, device=mesh_wp.device)

    saturated_np = tw.smoothing.saturate_scalar_gradient(
        values_wp, mesh_wp.points, mesh_wp.indices, threshold=threshold
    ).numpy()
    assert (saturated_np <= values_np.astype(np.float32) + 1e-5).all()

    edges_np = mesh_tm.edges_unique
    lengths_np = np.linalg.norm(
        mesh_tm.vertices[edges_np[:, 0]] - mesh_tm.vertices[edges_np[:, 1]], axis=1
    )
    jumps_np = np.abs(saturated_np[edges_np[:, 0]] - saturated_np[edges_np[:, 1]])
    assert (jumps_np <= lengths_np / threshold + 1e-4).all()
    # Every minimum survives: the smallest value in the field is untouched.
    assert np.isclose(saturated_np.min(), values_np.min(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("n_sources", [1, 3])
def test_saturate_scalar_gradient_is_the_edge_graph_distance(
    icosahedron: tuple[tm.Trimesh, wp.Mesh], n_sources: int
) -> None:
    """
    Class A against ``scipy.sparse.csgraph.dijkstra``: seeded with zeros, this *is* the distance.

    The saturated field is the envelope ``min_u (q0(u) + d(u, v) / threshold)`` over the edge graph,
    so at ``threshold=1`` with ``q0`` zero on the sources and large elsewhere it is exactly the
    weighted multi-source shortest-path distance — one relaxation to convergence, the same
    computation a Dijkstra would do. The test exists to make that a checked fact rather than a
    reading of the formula, because it is the reason the package ships no ``graph.dijkstra``: a
    second implementation of this relaxation would be a near-duplicate.

    Both sides take the identical graph — trimesh's unique edges with Euclidean lengths — so the
    only difference is float32 against float64, measured at 7.1e-07 on ``icosphere(3)``.
    """
    mesh_tm, mesh_wp = icosahedron
    n_vertices = mesh_tm.vertices.shape[0]
    sources_np = np.arange(n_sources)

    # A source is 0 and everything else starts far above any reachable distance, so the envelope
    # can only be lowered onto the true distance.
    seeded_np = np.full(n_vertices, 1e6, dtype=np.float32)
    seeded_np[sources_np] = 0.0
    saturated_np = tw.smoothing.saturate_scalar_gradient(
        wp.array(seeded_np, dtype=wp.float32, device=mesh_wp.device),
        mesh_wp.points,
        mesh_wp.indices,
        threshold=1.0,
    ).numpy()

    edges_np = mesh_tm.edges_unique
    lengths_np = np.linalg.norm(
        mesh_tm.vertices[edges_np[:, 0]] - mesh_tm.vertices[edges_np[:, 1]], axis=1
    )
    both_np = np.concatenate([edges_np, edges_np[:, ::-1]])
    graph_sp = sp.coo_matrix(
        (np.concatenate([lengths_np, lengths_np]), (both_np[:, 0], both_np[:, 1])),
        shape=(n_vertices, n_vertices),
    ).tocsr()
    distance_sp = csgraph.dijkstra(graph_sp, indices=sources_np, min_only=True)

    # Anti-vacuity: the field has to have actually propagated, not stayed at its 1e6 seed.
    assert distance_sp.max() > 0.5
    assert saturated_np.max() < 1e5
    assert np.allclose(saturated_np, distance_sp, rtol=1e-5, atol=1e-5)


def test_saturate_scalar_gradient_invalid(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = icosahedron
    values_wp = wp.zeros(mesh_tm.vertices.shape[0], dtype=wp.float32, device=mesh_wp.device)
    with pytest.raises(ValueError, match="threshold must be positive"):
        tw.smoothing.saturate_scalar_gradient(
            values_wp, mesh_wp.points, mesh_wp.indices, threshold=0.0
        )
    with pytest.raises(ValueError, match="max_iterations must be non-negative"):
        tw.smoothing.saturate_scalar_gradient(
            values_wp, mesh_wp.points, mesh_wp.indices, max_iterations=-1
        )


# ---------------------------------------------------------------------------
# Feature-preserving smoothing vs pymeshlab (two-step / normal filtering / unsharp)
# ---------------------------------------------------------------------------


def _noisy_cube(scale: float = 0.01, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a subdivided unit cube plus noise: flat faces, 90-degree creases, and grit."""
    box_tm = tm.creation.box(extents=[1.0, 1.0, 1.0]).subdivide().subdivide().subdivide()
    clean_np = np.ascontiguousarray(box_tm.vertices, dtype=np.float64)
    faces_np = np.ascontiguousarray(box_tm.faces, dtype=np.int32)
    rng = np.random.default_rng(seed)
    noisy_np = clean_np + rng.normal(scale=scale, size=clean_np.shape)
    return clean_np, faces_np, noisy_np


def _dihedral_percentile(vertices_np: np.ndarray, faces_np: np.ndarray, percentile: float) -> float:
    """Percentile of the unsigned dihedral in degrees: how sharp the sharpest edges still are."""
    mesh_tm = tm.Trimesh(vertices_np, faces_np, process=False)
    return float(np.percentile(np.rad2deg(np.abs(mesh_tm.face_adjacency_angles)), percentile))


def _rms_error(vertices_np: np.ndarray, clean_np: np.ndarray) -> float:
    return float(np.sqrt(((vertices_np - clean_np) ** 2).sum(axis=1)).mean())


def test_filter_normals_are_unit_and_crease_gated(device: str) -> None:
    """The gate is the whole algorithm: at 0 degrees nothing averages, at 180 everything does."""
    clean_np, faces_np, noisy_np = _noisy_cube()
    vertices_wp = wp.array(noisy_np.astype(np.float32), dtype=wp.vec3, device=device)
    faces_wp = wp.array(
        np.ascontiguousarray(faces_np.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device
    )
    raw_wp, _areas = tw.triangles.face_normals_and_areas(vertices_wp, faces_wp)

    gated_wp = tw.smoothing.filter_normals(vertices_wp, faces_wp, iterations=20, threshold=60.0)
    assert np.allclose(np.linalg.norm(gated_wp.numpy(), axis=1), 1.0, rtol=1e-4, atol=1e-4)

    # Threshold 0 lets nothing average, so the field is the geometric one (up to renormalization).
    untouched_wp = tw.smoothing.filter_normals(vertices_wp, faces_wp, threshold=0.0)
    assert np.allclose(untouched_wp.numpy(), raw_wp.numpy(), rtol=1e-4, atol=1e-4)

    # Threshold 180 averages across the cube's edges too, so the normals collapse toward each other.
    isotropic_wp = tw.smoothing.filter_normals(vertices_wp, faces_wp, threshold=180.0)
    assert (
        np.abs(isotropic_wp.numpy() - raw_wp.numpy()).max()
        > np.abs(gated_wp.numpy() - raw_wp.numpy()).max()
    )
    assert clean_np.shape[0] > 0


def test_filter_normals_invalid(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    with pytest.raises(ValueError, match=r"threshold must be in \[0, 180\]"):
        tw.smoothing.filter_normals(mesh_wp.points, mesh_wp.indices, threshold=-1.0)


def test_filter_two_step_denoises_without_rounding_the_creases(device: str) -> None:
    """
    The claim that separates this filter from every other one in the module, tested both ways.

    On a noisy cube two-step must (a) get *closer* to the clean mesh than the noise was, and (b)
    leave the 90-degree creases at 90 degrees. Isotropic Laplacian smoothing fails both: measured,
    it moves the mesh **further** from clean (RMS 0.016 to 0.047, and 0.016 to 0.097 on the crease
    vertices alone) and drops the 95th-percentile dihedral from 88 to 25 degrees. Taubin, which
    exists to fix Laplacian shrinkage, still lands at 56 — shrinkage was never the problem here.
    """
    clean_np, faces_np, noisy_np = _noisy_cube()
    vertices_wp = wp.array(noisy_np.astype(np.float32), dtype=wp.vec3, device=device)
    faces_wp = wp.array(
        np.ascontiguousarray(faces_np.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device
    )

    two_step_np = tw.smoothing.filter_two_step(vertices_wp, faces_wp).numpy().astype(np.float64)
    laplacian_np = (
        tw.smoothing.filter_laplacian(vertices_wp, faces_wp, iterations=10)
        .numpy()
        .astype(np.float64)
    )

    assert _rms_error(two_step_np, clean_np) < _rms_error(noisy_np, clean_np)
    assert _rms_error(two_step_np, clean_np) < _rms_error(laplacian_np, clean_np)

    # Creases survive: the sharp end of the dihedral distribution stays where the clean cube has it.
    assert _dihedral_percentile(two_step_np, faces_np, 95.0) > 85.0
    assert _dihedral_percentile(laplacian_np, faces_np, 95.0) < 50.0

    # And the improvement is concentrated on the crease vertices, which is the point.
    on_edge_np = (np.abs(np.abs(clean_np) - 0.5) < 1e-6).sum(axis=1) >= 2
    assert on_edge_np.any()
    assert _rms_error(two_step_np[on_edge_np], clean_np[on_edge_np]) < _rms_error(
        noisy_np[on_edge_np], clean_np[on_edge_np]
    )


@pytest.mark.parity("filter_two_step", "pymeshlab")
def test_filter_two_step_matches_pymeshlab_on_crease_preservation(device: str) -> None:
    """
    ``apply_coord_two_steps_smoothing`` is the same two-stage scheme at the same four parameters.

    The measure both must pass is crease preservation, and both do: 95th-percentile dihedral 87
    degrees for MeshLab against this port's 90, where plain Laplacian smoothing collapses to 25.

    **Denoising is where they part company, and not in MeshLab's favour.** At the same four
    parameters its output is *further* from the clean cube than the noise was — RMS 0.0244 against
    the noise's 0.0156, against this port's 0.0114 — because its fitting step rounds the corners
    inward (a corner at ``(-0.5, -0.5, -0.5)`` comes back at ``(-0.43, -0.49, -0.42)``). So the
    assertion below is one-sided: this port must be at least as close to clean as MeshLab, not equal
    to it.
    """
    clean_np, faces_np, noisy_np = _noisy_cube()
    meshset_pml = ml.MeshSet()
    meshset_pml.add_mesh(ml.Mesh(noisy_np, faces_np))
    meshset_pml.apply_coord_two_steps_smoothing(
        stepsmoothnum=3, normalthr=60.0, stepnormalnum=20, stepfitnum=20
    )
    pml_np = np.ascontiguousarray(meshset_pml.current_mesh().vertex_matrix(), dtype=np.float64)

    vertices_wp = wp.array(noisy_np.astype(np.float32), dtype=wp.vec3, device=device)
    faces_wp = wp.array(
        np.ascontiguousarray(faces_np.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device
    )
    two_step_np = tw.smoothing.filter_two_step(vertices_wp, faces_wp).numpy().astype(np.float64)

    assert _rms_error(two_step_np, clean_np) <= _rms_error(pml_np, clean_np)
    assert _dihedral_percentile(pml_np, faces_np, 95.0) > 85.0
    assert _dihedral_percentile(two_step_np, faces_np, 95.0) > 85.0


def test_filter_two_step_leaves_a_flat_patch_alone(device: str) -> None:
    """A plane is a fixed point of both halves: filtered normals are already the geometric ones."""
    grid_vertices_wp, grid_faces_wp = tw.creation.grid(count=(12, 12), device=device)
    smoothed_wp = tw.smoothing.filter_two_step(grid_vertices_wp, grid_faces_wp)
    assert np.allclose(smoothed_wp.numpy(), grid_vertices_wp.numpy(), rtol=1e-5, atol=1e-5)


def test_filter_two_step_zero_iterations(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    _mesh_tm, mesh_wp = icosahedron
    out_wp = tw.smoothing.filter_two_step(mesh_wp.points, mesh_wp.indices, iterations=0)
    assert np.array_equal(out_wp.numpy(), mesh_wp.points.numpy())


def test_filter_two_step_empty(device: str) -> None:
    vertices_wp = wp.zeros(0, dtype=wp.vec3, device=device)
    faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    assert tw.smoothing.filter_two_step(vertices_wp, faces_wp).shape == (0,)


def test_filter_sharpen_amplifies_detail(device: str) -> None:
    """Sharpening inverts smoothing, so it must move the mesh *away* from its smooth self."""
    sphere_tm = tm.creation.icosphere(subdivisions=3)
    rng = np.random.default_rng(4)
    bumpy_np = np.asarray(sphere_tm.vertices) * (
        1.0 + rng.normal(scale=0.02, size=(sphere_tm.vertices.shape[0], 1))
    )
    vertices_wp = wp.array(bumpy_np.astype(np.float32), dtype=wp.vec3, device=device)
    faces_wp = wp.array(
        np.ascontiguousarray(sphere_tm.faces.reshape(-1), dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )

    smoothed_np = (
        tw.smoothing.filter_laplacian(
            vertices_wp, faces_wp, lamb=1.0, iterations=5, volume_constraint=False
        )
        .numpy()
        .astype(np.float64)
    )
    sharpened_np = (
        tw.smoothing.filter_sharpen(vertices_wp, faces_wp, weight=0.5, iterations=5)
        .numpy()
        .astype(np.float64)
    )
    detail_np = bumpy_np - smoothed_np
    # The sharpened mesh is the original plus half the detail, so it is exactly that much further
    # out.
    assert np.allclose(sharpened_np, bumpy_np + 0.5 * detail_np, rtol=1e-3, atol=1e-4)
    assert np.abs(sharpened_np - smoothed_np).max() > np.abs(bumpy_np - smoothed_np).max()


def test_filter_sharpen_zero_weight_is_the_identity(
    icosahedron: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    _mesh_tm, mesh_wp = icosahedron
    out_wp = tw.smoothing.filter_sharpen(mesh_wp.points, mesh_wp.indices, weight=0.0)
    assert np.allclose(out_wp.numpy(), mesh_wp.points.numpy(), rtol=1e-6, atol=1e-6)


def test_filter_sharpen_empty(device: str) -> None:
    vertices_wp = wp.zeros(0, dtype=wp.vec3, device=device)
    faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    assert tw.smoothing.filter_sharpen(vertices_wp, faces_wp).shape == (0,)
