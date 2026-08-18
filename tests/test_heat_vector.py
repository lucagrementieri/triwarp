"""
Regression tests for ``triwarp.heat.vector`` against potpourri3d (CPU reference).

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
import warp.sparse as wps

import triwarp as tw

_MESHES = ["icosahedron", "cave_cube", "hemisphere", "half_torus"]


def _solver_pp(mesh_tm: object) -> pp3d.MeshVectorHeatSolver:
    return pp3d.MeshVectorHeatSolver(
        np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64),  # type: ignore[attr-defined]
        np.ascontiguousarray(mesh_tm.faces, dtype=np.int32),  # type: ignore[attr-defined]
        use_intrinsic_delaunay=False,
    )


def _frames(mesh_wp: wp.Mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(  # type: ignore[return-value]
        basis.numpy()
        for basis in tw.tangent_space.vertex_tangent_frames(mesh_wp.points, mesh_wp.indices)
    )


def _to_world(tangent: np.ndarray, basis_x: np.ndarray, basis_y: np.ndarray) -> np.ndarray:
    return tangent[:, 0:1] * basis_x + tangent[:, 1:2] * basis_y


# ---------------------------------------------------------------------------
# extend_scalar
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", _MESHES)
@pytest.mark.parity("extend_scalar", "potpourri3d")
def test_extend_scalar_matches_potpourri3d(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = len(mesh_tm.vertices)
    sources_np = np.array([0, n_vertices // 3, 2 * n_vertices // 3], dtype=np.int32)
    values_np = np.array([1.0, 2.0, 5.0])

    extended_wp = tw.heat.vector.extend_scalar(
        mesh_wp.points,
        mesh_wp.indices,
        wp.array(sources_np, dtype=wp.int32, device=mesh_wp.device),
        wp.array(values_np, dtype=wp.float64, device=mesh_wp.device),
    )
    extended_pp = np.asarray(
        _solver_pp(mesh_tm).extend_scalar(sources_np.tolist(), values_np.tolist())
    )

    # Nothing here depends on a frame, so this one is a direct comparison. The two libraries solve
    # the same system differently (conjugate gradient against a Cholesky factorization).
    #
    # ``icosahedron`` and ``hemisphere`` agree to the last bit. ``half_torus`` has exactly one
    # vertex of 544 outside the band, at 0.065 on a source range of [1, 5]; the rest agree to a mean
    # of 3.3e-4. That one vertex is *not* a regression from the timestep convention, which is worth
    # recording because it looks like one: sweeping ``t`` around the default shows the mean error
    # has a clean minimum exactly at the ``mean_unique_edge_length`` value used here --
    # 0.00085 / 0.00059 / **0.00033** / 0.00051 / 0.00153 at t x 0.995 / 0.998 / 1.000 / 1.002 /
    # 1.011 of it -- so this is the timestep geometry-central uses, and the per-face average
    # (x1.011) is 4.6x worse on the mean. The *maximum* falls monotonically across that whole
    # sweep and so tracks nothing: it is one vertex in a steep part of the field that smooths out
    # as t grows.
    #
    # Hence a mean-and-fraction bound rather than a max-only one, and no widening of the band for
    # all 544 to absorb a single point.
    within_band = np.abs(extended_wp.numpy() - extended_pp) <= 2e-2 + 1e-2 * np.abs(extended_pp)
    assert within_band.mean() > 0.99
    assert np.abs(extended_wp.numpy() - extended_pp).max() < 0.1
    assert np.abs(extended_wp.numpy() - extended_pp).mean() < 1e-3
    # The extension interpolates: it never leaves the range of its sources.
    assert extended_wp.numpy().min() >= values_np.min() - 1e-6
    assert extended_wp.numpy().max() <= values_np.max() + 1e-6


def test_extend_scalar_single_source_is_constant(
    icosahedron: tuple[object, wp.Mesh], device: str
) -> None:
    _, mesh_wp = icosahedron
    extended = tw.heat.vector.extend_scalar(
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

    transported, _ = tw.heat.vector.transport_tangent_vectors(
        vertices_wp,
        faces_wp,
        wp.array(np.array([size * size // 2], dtype=np.int32), dtype=wp.int32, device=device),
        wp.array(np.array([[1.0, 0.0]], dtype=np.float32), dtype=wp.vec2, device=device),
    )
    basis_x, basis_y, _ = (
        basis.numpy() for basis in tw.tangent_space.vertex_tangent_frames(vertices_wp, faces_wp)
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
@pytest.mark.parity("transport_tangent_vectors", "potpourri3d")
@pytest.mark.parity("vector_heat_scale", "potpourri3d")
def test_transport_tangent_vectors_matches_potpourri3d(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
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

    transported_wp, _ = tw.heat.vector.transport_tangent_vectors(
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
    _, mesh_wp = request.getfixturevalue(mesh_name)
    magnitude = 2.5
    transported, _ = tw.heat.vector.transport_tangent_vectors(
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
    _, mesh_wp = cave_cube
    # ``cave_cube`` is a cube shell around a smaller cube shell: two components. Nothing can be
    # transported across the gap, so the cavity's vertices must come back at zero rather than with a
    # smeared value (potpourri3d returns NaN on this mesh; see the note above).
    transported, _ = tw.heat.vector.transport_tangent_vectors(
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

    # The outer shell's corner diagonally opposite the source is excluded, and not as a tolerance
    # dodge: it is a point of the cut locus with three-fold symmetry about the body diagonal, so the
    # three shortest paths deliver copies of the source vector 120 degrees apart whose sum is
    # exactly zero. Measured on this fixture: exactly 0.0 at diffusion times 1e-3, 1e-2 and 1e-1 (so
    # it is not short-time underflow), unchanged by rotating the source vector (so it is not a bad
    # input direction), and restored in proportion to a symmetry-breaking jitter of the vertices
    # (1.4e-04 at 1e-3, 1.2e-02 at 1e-1).
    #
    # What survives the cancellation is round-off, and the two devices round differently: CPU
    # returns exactly zero and is scaled to zero, CUDA returns 8.7e-09 of the field maximum and is
    # scaled to unit length. Neither is more correct, and no threshold can pick the round-off out --
    # ``half_torus`` resolves genuine directions down to 8.0e-10 of its maximum, below this noise.
    # ``test_transport_cancels_at_a_symmetric_cut_locus_point`` pins the cancellation itself.
    positions_np = mesh_wp.points.numpy()
    antipode = int(np.argmin(np.linalg.norm(positions_np + positions_np[0], axis=1)))
    assert reachable[antipode]
    resolvable = reachable & (np.arange(magnitude.shape[0]) != antipode)

    assert np.isfinite(magnitude).all()
    # Seven of the outer shell's eight corners, so the comparison below is not vacuous.
    assert resolvable.sum() == 7
    assert np.allclose(magnitude[resolvable], 1.0, rtol=1e-4, atol=1e-4)
    assert np.allclose(magnitude[~reachable], 0.0, rtol=1e-6, atol=1e-6)


def test_transport_cancels_at_a_symmetric_cut_locus_point(
    cave_cube: tuple[object, wp.Mesh], device: str
) -> None:
    """
    Class C: the transported direction at a symmetric cut-locus point is a cancellation.

    There is nothing to compare the answer against — the quantity *is* the cancellation — so the
    statistic is a stability one: how far the unit direction at the antipodal corner moves when the
    symmetry causing the cancellation is broken by a vertex jitter, against how far every other
    vertex of the same shell moves under the identical perturbation. It excludes "that corner is
    merely the farthest from the source", which would move it no more than its neighbours move.

    Margins, measured over five jitter seeds on both devices: the antipode moves 1.000 (CPU, where
    the unperturbed answer is the zero vector) to 1.99 (CUDA), against a 0.25 threshold — 4x. Every
    other vertex moves a median 0.0007-0.0034 and at most 0.0089, against a 0.02 threshold — 5.9x.
    The two populations are 112x apart at their closest.
    """
    _, mesh_wp = cave_cube
    sources_wp = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=mesh_wp.device)
    vectors_wp = wp.array(
        np.array([[1.0, 0.0]], dtype=np.float32), dtype=wp.vec2, device=mesh_wp.device
    )
    positions_np = mesh_wp.points.numpy()
    antipode = int(np.argmin(np.linalg.norm(positions_np + positions_np[0], axis=1)))

    def _directions(points_np: np.ndarray) -> np.ndarray:
        transported, _ = tw.heat.vector.transport_tangent_vectors(
            wp.array(points_np, dtype=wp.vec3, device=mesh_wp.device),
            mesh_wp.indices,
            sources_wp,
            vectors_wp,
        )
        norm = np.linalg.norm(transported.numpy(), axis=1, keepdims=True)
        return transported.numpy() / np.where(norm > 0.0, norm, 1.0)

    jitter = np.random.default_rng(20260810).normal(scale=1e-3, size=positions_np.shape)
    moved = np.linalg.norm(_directions(positions_np) - _directions(positions_np + jitter), axis=1)
    labels = tw.graph.connected_component_labels_from_edges(
        tw.edges.edges_unique(mesh_wp.indices)[0], int(mesh_wp.points.shape[0])
    ).numpy()
    others = (labels == labels[0]) & (np.arange(moved.shape[0]) != antipode)

    assert others.sum() == 7
    assert np.median(moved[others]) < 0.02
    assert moved[antipode] > 0.25


@pytest.mark.parametrize("scale", [1e-3, 1e5])
def test_transport_is_invariant_to_mesh_scale(
    hemisphere: tuple[object, wp.Mesh], scale: float, device: str
) -> None:
    """
    Class A: the same surface in different units transports to the same tangent field.

    Both fields the solver divides by — the direction field and the source indicator — carry the
    mesh's scale as ~1/scale^2, so the cutoff that decides "has this vanished?" has to be relative
    to the field. Against an absolute cutoff the failure is a *silent* zero field rather than an
    error: 43 of this fixture's 97 vertices came back at zero magnitude at scale 1e5, and 0 do now.
    """
    _, mesh_wp = hemisphere
    sources_wp = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=mesh_wp.device)
    vectors_wp = wp.array(
        np.array([[1.0, 0.0]], dtype=np.float32), dtype=wp.vec2, device=mesh_wp.device
    )
    unit, _ = tw.heat.vector.transport_tangent_vectors(
        mesh_wp.points, mesh_wp.indices, sources_wp, vectors_wp
    )
    rescaled, _ = tw.heat.vector.transport_tangent_vectors(
        wp.array(mesh_wp.points.numpy() * scale, dtype=wp.vec3, device=mesh_wp.device),
        mesh_wp.indices,
        sources_wp,
        vectors_wp,
    )

    # Non-vacuous on both sides: every vertex of the unit-scale field is resolved, so a zeroed
    # rescaled field cannot pass the comparison by matching zeros against zeros.
    assert np.allclose(np.linalg.norm(unit.numpy(), axis=1), 1.0, rtol=1e-4, atol=1e-4)
    assert np.allclose(rescaled.numpy(), unit.numpy(), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    ("mesh_name", "n_resolved"), [("cave_cube", 7), ("icosahedron", 11), ("hemisphere", 97)]
)
def test_transport_validity_mask_flags_the_unresolvable(
    request: pytest.FixtureRequest, mesh_name: str, n_resolved: int, device: str
) -> None:
    """
    Class A against resolved counts measured on both devices.

    The counts are the point of the mask: they are identical on CPU and CUDA (7 / 11 / 97) where
    the *vectors* are not, because at a cancelling vertex CPU returns the zero vector and CUDA
    returns a full-length one. ``hemisphere`` resolves every vertex, so the boolean assert is
    parametrized over inputs producing both answers rather than only the interesting one.
    """
    _, mesh_wp = request.getfixturevalue(mesh_name)
    sources_wp = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=mesh_wp.device)
    vectors_wp = wp.array(
        np.array([[1.0, 0.0]], dtype=np.float32), dtype=wp.vec2, device=mesh_wp.device
    )
    transported, resolved = tw.heat.vector.transport_tangent_vectors(
        mesh_wp.points, mesh_wp.indices, sources_wp, vectors_wp
    )

    assert resolved.numpy().sum() == n_resolved
    # A vanished vector is never called resolved. The converse fails, which is the next test.
    assert not ((np.linalg.norm(transported.numpy(), axis=1) == 0.0) & resolved.numpy()).any()


def test_transport_validity_mask_separates_the_cut_locus_from_the_unreached(
    cave_cube: tuple[object, wp.Mesh], device: str
) -> None:
    """
    Class A: on ``cave_cube`` the mask is exactly "reachable, and not the antipodal corner".

    Those are the two ways a direction fails to exist, and the vectors alone distinguish neither
    from an ordinary answer: the cavity's eight vertices and the antipode all read zero on CPU,
    while on CUDA the antipode reads *full length* in a direction that is pure round-off. The mask
    is the same array on both.
    """
    _, mesh_wp = cave_cube
    sources_wp = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=mesh_wp.device)
    vectors_wp = wp.array(
        np.array([[1.0, 0.0]], dtype=np.float32), dtype=wp.vec2, device=mesh_wp.device
    )
    _, resolved = tw.heat.vector.transport_tangent_vectors(
        mesh_wp.points, mesh_wp.indices, sources_wp, vectors_wp
    )

    positions_np = mesh_wp.points.numpy()
    antipode = int(np.argmin(np.linalg.norm(positions_np + positions_np[0], axis=1)))
    labels = tw.graph.connected_component_labels_from_edges(
        tw.edges.edges_unique(mesh_wp.indices)[0], int(mesh_wp.points.shape[0])
    ).numpy()
    expected = (labels == labels[0]) & (np.arange(len(labels)) != antipode)

    assert expected.sum() == 7
    assert np.array_equal(resolved.numpy(), expected)


# ---------------------------------------------------------------------------
# log_map
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", _MESHES)
def test_log_map_radius_is_the_geodesic_distance(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    _, mesh_wp = request.getfixturevalue(mesh_name)
    sources_wp = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=mesh_wp.device)
    distance = tw.heat.distance.heat_geodesic(mesh_wp.points, mesh_wp.indices, sources_wp).numpy()
    logarithm = tw.heat.vector.log_map(mesh_wp.points, mesh_wp.indices, 0).numpy()

    # By construction the radius *is* the distance field: this pins the assembly, not the accuracy.
    assert np.allclose(np.linalg.norm(logarithm, axis=1), distance, rtol=1e-4, atol=1e-4)


# Only the better-resolved fixtures: ``potpourri3d.compute_log_map`` raises "factorization failed"
# on
# ``cave_cube`` (zero cotangent weights, as above), and a 12-vertex icosahedron is too coarse for
# either library's log map to mean much.
@pytest.mark.parametrize("mesh_name", ["hemisphere", "half_torus"])
@pytest.mark.parity("log_map", "potpourri3d")
def test_log_map_matches_potpourri3d(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    solver_pp = _solver_pp(mesh_tm)
    basis_x_pp, basis_y_pp, _ = (np.asarray(basis) for basis in solver_pp.get_tangent_frames())
    basis_x, _, _ = _frames(mesh_wp)

    logarithm_wp = tw.heat.vector.log_map(mesh_wp.points, mesh_wp.indices, 0).numpy()
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
    _, mesh_wp = icosahedron
    logarithm = tw.heat.vector.log_map(mesh_wp.points, mesh_wp.indices, 3).numpy()
    assert np.allclose(logarithm[3], 0.0, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# vector_heat_operators (the amortized path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere", "half_torus"])
def test_reused_operators_give_the_same_answer(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    _, mesh_wp = request.getfixturevalue(mesh_name)
    sources = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=mesh_wp.device)
    vectors = wp.array(
        np.array([[1.0, 0.0]], dtype=np.float32), dtype=wp.vec2, device=mesh_wp.device
    )
    operators = tw.heat.vector.vector_heat_operators(mesh_wp.points, mesh_wp.indices)

    # Reusing the operators must be an optimization and nothing else: the same assembly, so the same
    # matrices, so the same answer. Not *bit* for bit, though — conjugate gradient reduces with
    # atomics, so two identical solves can differ in their last bits. Measured: transport agrees to
    # 2e-15, and the log map to 4e-7 absolute, which is float32 epsilon on its own output.
    for fresh, reused in (
        (
            tw.heat.vector.transport_tangent_vectors(
                mesh_wp.points, mesh_wp.indices, sources, vectors
            )[0],
            tw.heat.vector.transport_tangent_vectors(
                mesh_wp.points, mesh_wp.indices, sources, vectors, operators=operators
            )[0],
        ),
        (
            tw.heat.vector.log_map(mesh_wp.points, mesh_wp.indices, 0),
            tw.heat.vector.log_map(mesh_wp.points, mesh_wp.indices, 0, operators=operators),
        ),
    ):
        # Compared against the *field's* magnitude rather than per element: a component that is
        # near-zero in a field of size one carries no information about the solve's agreement.
        span = float(np.abs(fresh.numpy()).max())
        assert np.allclose(fresh.numpy(), reused.numpy(), rtol=0.0, atol=1e-5 * span)


def test_operators_fix_the_diffusion_time(icosahedron: tuple[object, wp.Mesh], device: str) -> None:
    _, mesh_wp = icosahedron
    sources = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=mesh_wp.device)
    vectors = wp.array(
        np.array([[1.0, 0.0]], dtype=np.float32), dtype=wp.vec2, device=mesh_wp.device
    )
    slow = tw.heat.vector.vector_heat_operators(mesh_wp.points, mesh_wp.indices, t=1.0)

    # ``t`` lives in the assembled system, so a bundle built with one ``t`` must win over the
    # argument rather than being silently re-derived. A wrong precedence here would not be subtle:
    # ``t`` differs by six orders of magnitude between the two.
    with_bundle, _ = tw.heat.vector.transport_tangent_vectors(
        mesh_wp.points, mesh_wp.indices, sources, vectors, t=1e-6, operators=slow
    )
    direct, _ = tw.heat.vector.transport_tangent_vectors(
        mesh_wp.points, mesh_wp.indices, sources, vectors, t=1.0
    )
    span = float(np.abs(direct.numpy()).max())
    assert np.allclose(with_bundle.numpy(), direct.numpy(), rtol=0.0, atol=1e-5 * span)


# ---------------------------------------------------------------------------
# diffuse_tangent_field
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere"])
def test_diffuse_tangent_field_solves_its_own_system(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Class A: the result satisfies ``(M + t L_connection) X = source`` to the solver's tolerance.

    This entry point is public because the *source* is where the vector-valued methods differ while
    the solve is shared, so what has to be pinned is the equation rather than any particular field.
    Applying the operator back to the answer is the direct check, and it is independent of the CG
    path that produced it. Measured residual 1.5e-09 against a unit source.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = int(mesh_tm.vertices.shape[0])  # type: ignore[attr-defined]
    vector_system, _scalar, _frames = tw.heat.vector.vector_heat_operators(
        mesh_wp.points, mesh_wp.indices
    )

    source_np = np.zeros((n_vertices, 2))
    source_np[0] = [1.0, 0.0]
    source_np[n_vertices // 3] = [0.0, -1.0]
    source_wp = wp.array(
        np.ascontiguousarray(source_np), dtype=wp.vec2d, device=mesh_wp.points.device
    )

    diffused_wp = tw.heat.vector.diffuse_tangent_field(vector_system, source_wp)

    # Non-trivial: diffusion reaches every vertex, so this is not solving for zero.
    assert np.all(np.linalg.norm(diffused_wp.numpy(), axis=1) > 0.0)
    residual_wp = wp.zeros(n_vertices, dtype=wp.vec2d, device=mesh_wp.points.device)
    wps.bsr_mv(vector_system, diffused_wp, residual_wp, alpha=1.0, beta=0.0)
    assert np.abs(residual_wp.numpy() - source_np).max() < 1e-7


def test_diffuse_tangent_field_empty(icosahedron: tuple[object, wp.Mesh]) -> None:
    """An empty source returns an empty field without entering the solver."""
    _mesh_tm, mesh_wp = icosahedron
    vector_system, _scalar, _frames = tw.heat.vector.vector_heat_operators(
        mesh_wp.points, mesh_wp.indices
    )

    diffused_wp = tw.heat.vector.diffuse_tangent_field(
        vector_system, wp.empty(0, dtype=wp.vec2d, device=mesh_wp.points.device)
    )

    assert diffused_wp.shape == (0,)


# ---------------------------------------------------------------------------
# tangent_to_world and edge cases
# ---------------------------------------------------------------------------


def test_tangent_to_world_reproduces_the_frames(
    icosahedron: tuple[object, wp.Mesh], device: str
) -> None:
    _, mesh_wp = icosahedron
    basis_x_wp, basis_y_wp, _ = tw.tangent_space.vertex_tangent_frames(
        mesh_wp.points, mesh_wp.indices
    )
    n_vertices = int(mesh_wp.points.shape[0])
    tangent = wp.array(
        np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (n_vertices, 1)),
        dtype=wp.vec2,
        device=mesh_wp.device,
    )
    world = tw.heat.vector.tangent_to_world(tangent, basis_x_wp, basis_y_wp)
    assert np.allclose(world.numpy(), basis_y_wp.numpy(), rtol=1e-6, atol=1e-6)


def test_vector_heat_empty(device: str) -> None:
    vertices_wp = wp.empty(0, dtype=wp.vec3, device=device)
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    empty_int = wp.empty(0, dtype=wp.int32, device=device)
    assert tw.heat.vector.extend_scalar(
        vertices_wp, faces_wp, empty_int, wp.empty(0, dtype=wp.float64, device=device)
    ).shape == (0,)
    empty_transported, empty_resolved = tw.heat.vector.transport_tangent_vectors(
        vertices_wp, faces_wp, empty_int, wp.empty(0, dtype=wp.vec2, device=device)
    )
    assert empty_transported.shape == (0,)
    assert empty_resolved.shape == (0,)
    assert tw.heat.vector.log_map(vertices_wp, faces_wp, 0).shape == (0,)
