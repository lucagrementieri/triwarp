"""
Regression tests for ``triwarp.vector_heat`` against potpourri3d (CPU reference).

Comparing tangent fields across libraries needs care in two places, and both are load-bearing here:

* **The source vector.** ``(1, 0)`` means "along *my* reference direction", and the two libraries
  choose different ones, so handing both the literal ``(1, 0)`` transports two different world
  vectors. Every comparison below converts the source vector into the reference library's frame
  first.
* **The result.** 2D components are meaningless across libraries; the fields are compared after
  expanding them to 3D through each library's own frames.
"""

from __future__ import annotations

import numpy as np
import potpourri3d as pp3d
import pytest
import warp as wp

import triwarp as tw

_MESHES = ["icosahedron", "cave_cube", "hemisphere", "half_torus"]


def _skip_on_cpu(device: str) -> None:
    if wp.get_device(device).is_cpu:
        pytest.skip("the vector heat method needs conjugate gradient, which Warp cannot run on CPU")


def _solver_pp(mesh_tm: object) -> pp3d.MeshVectorHeatSolver:
    return pp3d.MeshVectorHeatSolver(
        np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64),  # type: ignore[attr-defined]
        np.ascontiguousarray(mesh_tm.faces, dtype=np.int32),  # type: ignore[attr-defined]
        use_intrinsic_delaunay=False,
    )


def _frames(mesh_wp: wp.Mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(  # type: ignore[return-value]
        basis.numpy() for basis in tw.tangent.vertex_tangent_frames(mesh_wp.points, mesh_wp.indices)
    )


def _to_world(tangent: np.ndarray, basis_x: np.ndarray, basis_y: np.ndarray) -> np.ndarray:
    return tangent[:, 0:1] * basis_x + tangent[:, 1:2] * basis_y


# ---------------------------------------------------------------------------
# extend_scalar
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", _MESHES)
def test_extend_scalar_matches_potpourri3d(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    _skip_on_cpu(device)
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = len(mesh_tm.vertices)
    sources_np = np.array([0, n_vertices // 3, 2 * n_vertices // 3], dtype=np.int32)
    values_np = np.array([1.0, 2.0, 5.0])

    extended_wp = tw.vector_heat.extend_scalar(
        mesh_wp.points,
        mesh_wp.indices,
        wp.array(sources_np, dtype=wp.int32, device=mesh_wp.device),
        wp.array(values_np, dtype=wp.float64, device=mesh_wp.device),
    )
    extended_pp = np.asarray(
        _solver_pp(mesh_tm).extend_scalar(sources_np.tolist(), values_np.tolist())
    )

    # Nothing here depends on a frame, so this one is a direct comparison. The two libraries solve
    # the same system differently (conjugate gradient against a Cholesky factorization), which on
    # these fixtures costs at most 0.25% of the source range.
    assert np.allclose(extended_wp.numpy(), extended_pp, rtol=1e-2, atol=2e-2)
    # The extension interpolates: it never leaves the range of its sources.
    assert extended_wp.numpy().min() >= values_np.min() - 1e-6
    assert extended_wp.numpy().max() <= values_np.max() + 1e-6


def test_extend_scalar_single_source_is_constant(
    icosahedron: tuple[object, wp.Mesh], device: str
) -> None:
    _skip_on_cpu(device)
    _, mesh_wp = icosahedron
    extended = tw.vector_heat.extend_scalar(
        mesh_wp.points,
        mesh_wp.indices,
        wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=mesh_wp.device),
        wp.array(np.array([3.5]), dtype=wp.float64, device=mesh_wp.device),
    )
    # One source has nothing to blend against, so its value must fill the surface.
    assert np.allclose(extended.numpy(), 3.5, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# transport_tangent_vectors
# ---------------------------------------------------------------------------


def test_transport_on_a_flat_patch_is_constant(device: str) -> None:
    _skip_on_cpu(device)
    # Parallel transport across a plane is the identity, so the transported field must be one
    # constant
    # world vector. Away from the rim, where the angle sum is exactly 2*pi and the intrinsic
    # flattening is trivial, that is exact.
    size = 9
    grid_x, grid_y = np.meshgrid(np.linspace(0.0, 1.0, size), np.linspace(0.0, 1.0, size))
    vertices_np = np.stack([grid_x.ravel(), grid_y.ravel(), np.zeros(size * size)], axis=1)
    faces_np = np.array(
        [
            quad
            for row in range(size - 1)
            for column in range(size - 1)
            for quad in (
                [row * size + column, row * size + column + 1, (row + 1) * size + column + 1],
                [row * size + column, (row + 1) * size + column + 1, (row + 1) * size + column],
            )
        ],
        dtype=np.int32,
    )
    vertices_wp = wp.array(vertices_np.astype(np.float32), dtype=wp.vec3, device=device)
    faces_wp = wp.array(faces_np.reshape(-1), dtype=wp.int32, device=device)

    transported = tw.vector_heat.transport_tangent_vectors(
        vertices_wp,
        faces_wp,
        wp.array(np.array([size * size // 2], dtype=np.int32), dtype=wp.int32, device=device),
        wp.array(np.array([[1.0, 0.0]], dtype=np.float32), dtype=wp.vec2, device=device),
    )
    basis_x, basis_y, _ = (
        basis.numpy() for basis in tw.tangent.vertex_tangent_frames(vertices_wp, faces_wp)
    )
    world = _to_world(transported.numpy(), basis_x, basis_y)
    world /= np.linalg.norm(world, axis=1, keepdims=True)

    interior = ~tw.halfedge.vertex_one_rings(faces_wp, n_vertices=len(vertices_np))[2].numpy()
    reference = world[interior][0]
    assert np.allclose(world[interior] @ reference, 1.0, rtol=1e-4, atol=1e-4)
    # Magnitude is carried by the scalar extension, so it is preserved everywhere.
    assert np.allclose(
        np.linalg.norm(_to_world(transported.numpy(), basis_x, basis_y), axis=1),
        1.0,
        rtol=1e-4,
        atol=1e-4,
    )


# ``cave_cube`` is left out: potpourri3d returns NaN on it. Its quad faces are split by diagonals
# whose two opposite angles are both right angles, so those cotangent weights are exactly zero and
# geometry-central's direct factorization of the connection Laplacian fails ("factorization
# failed").
# It is covered by the disconnected-component test below instead.
@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere", "half_torus"])
def test_transport_tangent_vectors_matches_potpourri3d(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    _skip_on_cpu(device)
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    solver_pp = _solver_pp(mesh_tm)
    basis_x_pp, basis_y_pp, _ = (np.asarray(basis) for basis in solver_pp.get_tangent_frames())
    basis_x, basis_y, _ = _frames(mesh_wp)

    source = 0
    vector = np.array([[1.0, 0.0]], dtype=np.float32)
    # The same *world* vector for both libraries: triwarp's basis_x at the source, re-expressed in
    # potpourri3d's frame there.
    vector_pp = [
        [float(basis_x[source] @ basis_x_pp[source]), float(basis_x[source] @ basis_y_pp[source])]
    ]

    transported_wp = tw.vector_heat.transport_tangent_vectors(
        mesh_wp.points,
        mesh_wp.indices,
        wp.array(np.array([source], dtype=np.int32), dtype=wp.int32, device=mesh_wp.device),
        wp.array(vector, dtype=wp.vec2, device=mesh_wp.device),
    )
    transported_pp = np.asarray(solver_pp.transport_tangent_vectors([source], vector_pp))

    world_wp = _to_world(transported_wp.numpy(), basis_x, basis_y)
    world_pp = _to_world(transported_pp, basis_x_pp, basis_y_pp)
    assert np.allclose(np.linalg.norm(world_wp, axis=1), 1.0, rtol=5e-2, atol=5e-2)
    cosine = (world_wp * world_pp).sum(axis=1) / (
        np.linalg.norm(world_wp, axis=1) * np.linalg.norm(world_pp, axis=1) + 1e-12
    )
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    # Measured: 0.93 degrees median on ``hemisphere``, 0.03 on ``half_torus``. The *median* is the
    # statistic to use: on the cut locus (and on a 12-vertex icosahedron, most of which is cut
    # locus)
    # the transported direction is genuinely undefined and the two libraries disagree freely there.
    assert np.median(angle) < 2.0


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere"])
def test_transport_preserves_source_magnitudes(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    _skip_on_cpu(device)
    _, mesh_wp = request.getfixturevalue(mesh_name)
    magnitude = 2.5
    transported = tw.vector_heat.transport_tangent_vectors(
        mesh_wp.points,
        mesh_wp.indices,
        wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=mesh_wp.device),
        wp.array(
            np.array([[0.0, magnitude]], dtype=np.float32), dtype=wp.vec2, device=mesh_wp.device
        ),
    )
    # A single source's magnitude is extended as a constant, so every transported vector has it.
    assert np.allclose(np.linalg.norm(transported.numpy(), axis=1), magnitude, rtol=1e-4, atol=1e-4)


def test_transport_does_not_cross_components(
    cave_cube: tuple[object, wp.Mesh], device: str
) -> None:
    _skip_on_cpu(device)
    _, mesh_wp = cave_cube
    # ``cave_cube`` is a cube shell around a smaller cube shell: two components. Nothing can be
    # transported across the gap, so the cavity's vertices must come back at zero rather than with a
    # smeared value (potpourri3d returns NaN on this mesh; see the note above).
    transported = tw.vector_heat.transport_tangent_vectors(
        mesh_wp.points,
        mesh_wp.indices,
        wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=mesh_wp.device),
        wp.array(np.array([[1.0, 0.0]], dtype=np.float32), dtype=wp.vec2, device=mesh_wp.device),
    )

    magnitude = np.linalg.norm(transported.numpy(), axis=1)
    labels = tw.graph.connected_component_labels_from_edges(
        tw.edges.edges_unique(mesh_wp.indices)[0], int(mesh_wp.points.shape[0])
    ).numpy()
    reachable = labels == labels[0]
    assert np.isfinite(magnitude).all()
    assert np.allclose(magnitude[reachable], 1.0, rtol=1e-4, atol=1e-4)
    assert np.allclose(magnitude[~reachable], 0.0, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# log_map
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", _MESHES)
def test_log_map_radius_is_the_geodesic_distance(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    _skip_on_cpu(device)
    _, mesh_wp = request.getfixturevalue(mesh_name)
    sources_wp = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=mesh_wp.device)
    distance = tw.geodesic.heat_geodesic(mesh_wp.points, mesh_wp.indices, sources_wp).numpy()
    logarithm = tw.vector_heat.log_map(mesh_wp.points, mesh_wp.indices, 0).numpy()

    # By construction the radius *is* the distance field: this pins the assembly, not the accuracy.
    assert np.allclose(np.linalg.norm(logarithm, axis=1), distance, rtol=1e-4, atol=1e-4)


# Only the better-resolved fixtures: ``potpourri3d.compute_log_map`` raises "factorization failed"
# on
# ``cave_cube`` (zero cotangent weights, as above), and a 12-vertex icosahedron is too coarse for
# either library's log map to mean much.
@pytest.mark.parametrize("mesh_name", ["hemisphere", "half_torus"])
def test_log_map_matches_potpourri3d(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    _skip_on_cpu(device)
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    solver_pp = _solver_pp(mesh_tm)
    basis_x_pp, basis_y_pp, _ = (np.asarray(basis) for basis in solver_pp.get_tangent_frames())
    basis_x, _, _ = _frames(mesh_wp)

    logarithm_wp = tw.vector_heat.log_map(mesh_wp.points, mesh_wp.indices, 0).numpy()
    logarithm_pp = np.asarray(solver_pp.compute_log_map(0, "VectorHeat"))

    # Both maps live in the source vertex's tangent plane but measure angles from their own
    # reference
    # direction, so potpourri3d's has to be rotated into triwarp's before the two can be compared.
    cosine = float(basis_x[0] @ basis_x_pp[0])
    sine = float(basis_x[0] @ basis_y_pp[0])
    rotation = np.array([[cosine, sine], [-sine, cosine]])
    aligned_pp = logarithm_pp @ rotation.T

    scale = float(np.linalg.norm(mesh_tm.vertices.max(axis=0) - mesh_tm.vertices.min(axis=0)))
    error = np.linalg.norm(logarithm_wp - aligned_pp, axis=1)
    # The two constructions differ (this one reads the angle off the distance gradient), so they
    # agree
    # only to a few percent of the mesh scale, improving with resolution: measured 10% of the
    # bounding
    # diagonal on ``hemisphere`` (97 vertices) and 2.8% on ``half_torus`` (544).
    assert np.median(error) < 0.15 * scale


def test_log_map_is_zero_at_its_source(icosahedron: tuple[object, wp.Mesh], device: str) -> None:
    _skip_on_cpu(device)
    _, mesh_wp = icosahedron
    logarithm = tw.vector_heat.log_map(mesh_wp.points, mesh_wp.indices, 3).numpy()
    assert np.allclose(logarithm[3], 0.0, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# vector_heat_operators (the amortized path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere", "half_torus"])
def test_reused_operators_give_the_same_answer(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    _skip_on_cpu(device)
    _, mesh_wp = request.getfixturevalue(mesh_name)
    sources = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=mesh_wp.device)
    vectors = wp.array(
        np.array([[1.0, 0.0]], dtype=np.float32), dtype=wp.vec2, device=mesh_wp.device
    )
    operators = tw.vector_heat.vector_heat_operators(mesh_wp.points, mesh_wp.indices)

    # Reusing the operators must be an optimization and nothing else: the same assembly, so the same
    # matrices, so the same answer. Not *bit* for bit, though — conjugate gradient reduces with
    # atomics, so two identical solves can differ in their last bits. Measured here: transport agrees
    # to 2e-15, and the log map to 4e-7 absolute, which is float32 epsilon on its own output.
    for fresh, reused in (
        (
            tw.vector_heat.transport_tangent_vectors(
                mesh_wp.points, mesh_wp.indices, sources, vectors
            ),
            tw.vector_heat.transport_tangent_vectors(
                mesh_wp.points, mesh_wp.indices, sources, vectors, operators=operators
            ),
        ),
        (
            tw.vector_heat.log_map(mesh_wp.points, mesh_wp.indices, 0),
            tw.vector_heat.log_map(mesh_wp.points, mesh_wp.indices, 0, operators=operators),
        ),
    ):
        # Compared against the *field's* magnitude rather than per element: a component that is
        # near-zero in a field of size one carries no information about the solve's agreement.
        span = float(np.abs(fresh.numpy()).max())
        assert np.allclose(fresh.numpy(), reused.numpy(), rtol=0.0, atol=1e-5 * span)


def test_operators_fix_the_diffusion_time(icosahedron: tuple[object, wp.Mesh], device: str) -> None:
    _skip_on_cpu(device)
    _, mesh_wp = icosahedron
    sources = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=mesh_wp.device)
    vectors = wp.array(
        np.array([[1.0, 0.0]], dtype=np.float32), dtype=wp.vec2, device=mesh_wp.device
    )
    slow = tw.vector_heat.vector_heat_operators(mesh_wp.points, mesh_wp.indices, t=1.0)

    # ``t`` lives in the assembled system, so a bundle built with one ``t`` must win over the
    # argument rather than being silently re-derived. A wrong precedence here would not be subtle:
    # ``t`` differs by six orders of magnitude between the two.
    with_bundle = tw.vector_heat.transport_tangent_vectors(
        mesh_wp.points, mesh_wp.indices, sources, vectors, t=1e-6, operators=slow
    )
    direct = tw.vector_heat.transport_tangent_vectors(
        mesh_wp.points, mesh_wp.indices, sources, vectors, t=1.0
    )
    span = float(np.abs(direct.numpy()).max())
    assert np.allclose(with_bundle.numpy(), direct.numpy(), rtol=0.0, atol=1e-5 * span)


# ---------------------------------------------------------------------------
# tangent_to_world and edge cases
# ---------------------------------------------------------------------------


def test_tangent_to_world_reproduces_the_frames(
    icosahedron: tuple[object, wp.Mesh], device: str
) -> None:
    _, mesh_wp = icosahedron
    basis_x_wp, basis_y_wp, _ = tw.tangent.vertex_tangent_frames(mesh_wp.points, mesh_wp.indices)
    n_vertices = int(mesh_wp.points.shape[0])
    tangent = wp.array(
        np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (n_vertices, 1)),
        dtype=wp.vec2,
        device=mesh_wp.device,
    )
    world = tw.vector_heat.tangent_to_world(tangent, basis_x_wp, basis_y_wp)
    assert np.allclose(world.numpy(), basis_y_wp.numpy(), rtol=1e-6, atol=1e-6)


def test_vector_heat_empty(device: str) -> None:
    vertices_wp = wp.empty(0, dtype=wp.vec3, device=device)
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    empty_int = wp.empty(0, dtype=wp.int32, device=device)
    assert tw.vector_heat.extend_scalar(
        vertices_wp, faces_wp, empty_int, wp.empty(0, dtype=wp.float64, device=device)
    ).shape == (0,)
    assert tw.vector_heat.transport_tangent_vectors(
        vertices_wp, faces_wp, empty_int, wp.empty(0, dtype=wp.vec2, device=device)
    ).shape == (0,)
    assert tw.vector_heat.log_map(vertices_wp, faces_wp, 0).shape == (0,)
