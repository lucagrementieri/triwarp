"""
Regression tests for ``triwarp.signed_heat`` against potpourri3d (CPU reference).

Every curve here is a **vertex one-ring cycle**, for a reason that is easy to trip over:
``potpourri3d.MeshSignedHeatSolver`` rejects a curve whose consecutive points do not share a face
("Each curve segment must be contained within a single face"), so an arbitrary vertex list is
not a valid source for it. A one-ring cycle is the smallest curve that is genuinely edge-connected,
closed, *and* separating — which is what makes the sign meaningful — and it comes straight out of
[`vertex_one_rings`][triwarp.halfedge.vertex_one_rings].

The strongest check is not against the reference at all: the *magnitude* of the signed field has to
agree with the unsigned [`heat_geodesic`][triwarp.geodesic.heat_geodesic] distance to the same
curve, which is a completely different solve.
"""

from __future__ import annotations

import numpy as np
import potpourri3d as pp3d
import pytest
import trimesh as tm
import warp as wp

import triwarp as tw


def _skip_on_cpu(device: str) -> None:
    if wp.get_device(device).is_cpu:
        pytest.skip("the signed heat method needs conjugate gradient, which Warp cannot run on CPU")


def _one_ring_cycle(
    mesh_tm: tm.Trimesh, mesh_wp: wp.Mesh, which: int = 0
) -> tuple[int, np.ndarray]:
    """
    Return an interior vertex and the counter-clockwise cycle of its neighbours.

    The centre has to be an *interior* vertex: a boundary vertex's fan is open, so its neighbours do
    not close into a cycle and the last "segment" would be a chord across the surface rather than an
    edge — which both this method and the reference read as a different curve entirely.
    """
    offsets, ring, is_boundary = (
        array.numpy()
        for array in tw.halfedge.vertex_one_rings(mesh_wp.indices, n_vertices=len(mesh_tm.vertices))
    )
    interior = np.flatnonzero(~is_boundary)
    center = int(interior[which % len(interior)])
    faces_np = np.asarray(mesh_tm.faces)
    halfedges = ring[offsets[center] : offsets[center + 1]]
    return center, np.array([faces_np[h // 3][(h % 3 + 1) % 3] for h in halfedges], dtype=np.int32)


def _distance_pp(
    mesh_tm: tm.Trimesh, curve: np.ndarray, level_set_constraint: str = "ZeroSet"
) -> np.ndarray:
    solver_pp = pp3d.MeshSignedHeatSolver(
        np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64),
        np.ascontiguousarray(mesh_tm.faces, dtype=np.int32),
    )
    return np.asarray(
        solver_pp.compute_distance(
            [[(int(vertex), []) for vertex in curve]], level_set_constraint=level_set_constraint
        )
    )


# ---------------------------------------------------------------------------
# heat_signed_distance
# ---------------------------------------------------------------------------


# ``cave_cube`` and ``half_torus`` are left out, both for the same reason as in
# ``tests/test_vector_heat.py``: every one of their faces carries an edge whose two opposite angles
# are
# right angles, so that edge's cotangent weight is exactly zero. potpourri3d cannot even factor
# ``cave_cube`` (it returns NaN), and on ``half_torus`` the two libraries agree only to a
# correlation
# of 0.39 — both are solving a degenerate problem there and degrade differently. The structural
# tests
# below still cover ``half_torus``, and they pass.
@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere"])
def test_heat_signed_distance_matches_potpourri3d(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    _skip_on_cpu(device)
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    _, curve_np = _one_ring_cycle(mesh_tm, mesh_wp)
    curve_wp = wp.array(curve_np, dtype=wp.int32, device=mesh_wp.device)

    distance_wp = tw.signed_heat.heat_signed_distance(
        mesh_wp.points, mesh_wp.indices, curve_wp
    ).numpy()
    distance_pp = _distance_pp(mesh_tm, curve_np)

    # Same sign convention (a counter-clockwise curve encloses the positive side) and the same field
    # to a few percent of the mesh scale: measured 1.6% on ``icosahedron`` and 4.0% on
    # ``hemisphere``.
    # The correlation is the looser of the two statistics here -- 0.996 and 0.940 -- because this
    # curve sits one ring from the hemisphere's rim, where the reference's boundary handling and
    # this
    # one's diverge; taking the centre-most interior vertex instead gives 0.990.
    scale = float(np.linalg.norm(mesh_tm.vertices.max(axis=0) - mesh_tm.vertices.min(axis=0)))
    assert np.corrcoef(distance_wp, distance_pp)[0, 1] > 0.9
    assert np.abs(distance_wp - distance_pp).mean() < 0.05 * scale


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere"])
def test_signed_distance_magnitude_is_the_unsigned_distance(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    _skip_on_cpu(device)
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    _, curve_np = _one_ring_cycle(mesh_tm, mesh_wp)
    curve_wp = wp.array(curve_np, dtype=wp.int32, device=mesh_wp.device)

    signed = tw.signed_heat.heat_signed_distance(mesh_wp.points, mesh_wp.indices, curve_wp).numpy()
    unsigned = tw.geodesic.heat_geodesic(mesh_wp.points, mesh_wp.indices, curve_wp).numpy()

    # Two independent solves — a vector diffusion plus Poisson against a scalar diffusion plus
    # Poisson — that have to agree about *how far* the curve is, whatever they say about which side.
    # Measured 2.6% of the span on ``icosahedron`` and 9.9% on ``hemisphere``, where the unsigned
    # method's Neumann boundary and the signed field's behaviour at the rim pull apart.
    span = float(unsigned.max())
    assert np.abs(np.abs(signed) - unsigned).mean() < 0.15 * span


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere", "half_torus"])
def test_signed_distance_is_positive_inside_the_curve(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    _skip_on_cpu(device)
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    center, curve_np = _one_ring_cycle(mesh_tm, mesh_wp)
    curve_wp = wp.array(curve_np, dtype=wp.int32, device=mesh_wp.device)

    distance = tw.signed_heat.heat_signed_distance(
        mesh_wp.points, mesh_wp.indices, curve_wp
    ).numpy()

    # The ring encloses exactly one vertex, and the sign convention puts that side positive; every
    # other vertex is outside it and must come out negative.
    assert distance[center] > 0.0
    outside = np.ones(len(mesh_tm.vertices), dtype=bool)
    outside[curve_np] = False
    outside[center] = False
    assert (distance[outside] < 0.0).all()


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus"])
def test_reversing_the_curve_negates_the_field(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    _skip_on_cpu(device)
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    _, curve_np = _one_ring_cycle(mesh_tm, mesh_wp)

    forward = tw.signed_heat.heat_signed_distance(
        mesh_wp.points, mesh_wp.indices, wp.array(curve_np, dtype=wp.int32, device=mesh_wp.device)
    ).numpy()
    backward = tw.signed_heat.heat_signed_distance(
        mesh_wp.points,
        mesh_wp.indices,
        wp.array(curve_np[::-1].copy(), dtype=wp.int32, device=mesh_wp.device),
    ).numpy()

    # Orientation *is* the sign: nothing else about the source changed.
    assert np.allclose(forward, -backward, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus"])
def test_zero_set_constraint_pins_the_curve(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    _skip_on_cpu(device)
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    _, curve_np = _one_ring_cycle(mesh_tm, mesh_wp)
    curve_wp = wp.array(curve_np, dtype=wp.int32, device=mesh_wp.device)

    pinned = tw.signed_heat.heat_signed_distance(
        mesh_wp.points, mesh_wp.indices, curve_wp, level_set_constraint="zero_set"
    ).numpy()
    shifted = tw.signed_heat.heat_signed_distance(
        mesh_wp.points, mesh_wp.indices, curve_wp, level_set_constraint="none"
    ).numpy()

    # ``zero_set`` pins exactly; ``none`` only centres the curve's mean, so individual curve
    # vertices
    # sit slightly off zero.
    assert np.array_equal(pinned[curve_np], np.zeros(len(curve_np)))
    assert abs(shifted[curve_np].mean()) < 1e-9
    # Both agree away from the curve, where the constraint does not act.
    span = float(np.abs(pinned).max())
    assert np.abs(pinned - shifted).mean() < 0.1 * span


def test_multiple_curves_via_offsets(icosahedron: tuple[tm.Trimesh, wp.Mesh], device: str) -> None:
    _skip_on_cpu(device)
    mesh_tm, mesh_wp = icosahedron
    _, first = _one_ring_cycle(mesh_tm, mesh_wp, which=0)
    _, second = _one_ring_cycle(mesh_tm, mesh_wp, which=5)
    packed = wp.array(np.concatenate([first, second]), dtype=wp.int32, device=mesh_wp.device)
    offsets = wp.array(
        np.array([0, len(first), len(first) + len(second)], dtype=np.int32),
        dtype=wp.int32,
        device=mesh_wp.device,
    )

    both = tw.signed_heat.heat_signed_distance(
        mesh_wp.points, mesh_wp.indices, packed, offsets
    ).numpy()
    only_first = tw.signed_heat.heat_signed_distance(
        mesh_wp.points, mesh_wp.indices, wp.array(first, dtype=wp.int32, device=mesh_wp.device)
    ).numpy()

    # Two sources pin two zero sets, so the combined field differs from either alone but still
    # vanishes on both curves.
    assert np.array_equal(both[np.concatenate([first, second])], np.zeros(len(first) + len(second)))
    assert not np.allclose(both, only_first, rtol=1e-2, atol=1e-2)


def test_open_curve_still_changes_sign_across_itself(
    icosahedron: tuple[tm.Trimesh, wp.Mesh], device: str
) -> None:
    _skip_on_cpu(device)
    mesh_tm, mesh_wp = icosahedron
    _, curve_np = _one_ring_cycle(mesh_tm, mesh_wp)
    curve_np = curve_np[:3]
    curve_wp = wp.array(curve_np, dtype=wp.int32, device=mesh_wp.device)

    # An open curve has no inside, so the far field means nothing — but the method does not need a
    # closed curve to run, and the result must still be finite and vanish on the source.
    distance = tw.signed_heat.heat_signed_distance(
        mesh_wp.points, mesh_wp.indices, curve_wp, closed=False
    ).numpy()
    assert np.isfinite(distance).all()
    assert np.array_equal(distance[curve_np], np.zeros(len(curve_np)))
    assert distance.min() < 0.0 < distance.max()


def test_reused_operators_give_the_same_answer(
    half_torus: tuple[tm.Trimesh, wp.Mesh], device: str
) -> None:
    _skip_on_cpu(device)
    mesh_tm, mesh_wp = half_torus
    _, curve_np = _one_ring_cycle(mesh_tm, mesh_wp)
    curve_wp = wp.array(curve_np, dtype=wp.int32, device=mesh_wp.device)
    operators = tw.vector_heat.vector_heat_operators(mesh_wp.points, mesh_wp.indices)

    # This method assembles three matrices; reusing them must change nothing but the time taken.
    # Compared at solver precision, not exactly: conjugate gradient's reductions are not
    # bit-reproducible run to run.
    fresh = tw.signed_heat.heat_signed_distance(mesh_wp.points, mesh_wp.indices, curve_wp)
    reused = tw.signed_heat.heat_signed_distance(
        mesh_wp.points, mesh_wp.indices, curve_wp, operators=operators
    )
    span = float(np.abs(fresh.numpy()).max())
    assert np.allclose(fresh.numpy(), reused.numpy(), rtol=0.0, atol=1e-5 * span)


def test_invalid_level_set_constraint(icosahedron: tuple[tm.Trimesh, wp.Mesh], device: str) -> None:
    _, mesh_wp = icosahedron
    with pytest.raises(ValueError, match="level_set_constraint"):
        tw.signed_heat.heat_signed_distance(
            mesh_wp.points,
            mesh_wp.indices,
            wp.array(np.array([0, 1], dtype=np.int32), dtype=wp.int32, device=mesh_wp.device),
            level_set_constraint="Multiple",
        )


def test_heat_signed_distance_empty(device: str) -> None:
    vertices_wp = wp.empty(0, dtype=wp.vec3, device=device)
    faces_wp = wp.array(np.array([], dtype=np.int32), dtype=wp.int32, device=device)
    empty_int = wp.empty(0, dtype=wp.int32, device=device)
    assert tw.signed_heat.heat_signed_distance(vertices_wp, faces_wp, empty_int).shape == (0,)
