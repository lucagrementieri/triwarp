"""
Regression tests for ``triwarp.tangent_space`` against potpourri3d (CPU reference).

Tangent frames are only defined up to a rotation within the tangent plane — each library picks its
own reference halfedge — so the comparisons here are the gauge-invariant ones: normals directly,
frames through the rotation that relates them, and transport angles through the holonomy around each
face, which is what the connection Laplacian's phases encode independently of any frame choice.
"""

from __future__ import annotations

import numpy as np
import potpourri3d as pp3d
import pytest
import warp as wp

import triwarp as tw

_MESHES = ["icosahedron", "cave_cube", "hemisphere", "half_torus"]


def _vector_heat_solver_pp(mesh_tm: object) -> pp3d.MeshVectorHeatSolver:
    # ``use_intrinsic_delaunay=False`` so both sides discretize the same triangulation; potpourri3d
    # defaults to flipping to an intrinsic Delaunay triangulation first.
    return pp3d.MeshVectorHeatSolver(
        np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64),  # type: ignore[attr-defined]
        np.ascontiguousarray(mesh_tm.faces, dtype=np.int32),  # type: ignore[attr-defined]
        use_intrinsic_delaunay=False,
    )


# ---------------------------------------------------------------------------
# vertex_tangent_frames
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", _MESHES)
def test_vertex_tangent_frames_are_orthonormal(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    _, mesh_wp = request.getfixturevalue(mesh_name)
    basis_x_wp, basis_y_wp, normal_wp = tw.tangent_space.vertex_tangent_frames(
        mesh_wp.points, mesh_wp.indices
    )

    basis_x, basis_y, normal = basis_x_wp.numpy(), basis_y_wp.numpy(), normal_wp.numpy()
    assert np.allclose(np.linalg.norm(basis_x, axis=1), 1.0, rtol=1e-5, atol=1e-5)
    assert np.allclose(np.linalg.norm(basis_y, axis=1), 1.0, rtol=1e-5, atol=1e-5)
    assert np.allclose((basis_x * basis_y).sum(axis=1), 0.0, rtol=1e-5, atol=1e-5)
    assert np.allclose((basis_x * normal).sum(axis=1), 0.0, rtol=1e-5, atol=1e-5)
    # Right-handed: basis_x x basis_y points along the normal.
    assert np.allclose((np.cross(basis_x, basis_y) * normal).sum(axis=1), 1.0, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mesh_name", _MESHES)
def test_vertex_tangent_frames_match_potpourri3d(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    basis_x_wp, basis_y_wp, normal_wp = tw.tangent_space.vertex_tangent_frames(
        mesh_wp.points, mesh_wp.indices
    )
    basis_x_pp, basis_y_pp, normal_pp = (
        np.asarray(basis) for basis in _vector_heat_solver_pp(mesh_tm).get_tangent_frames()
    )

    # Normals are frame-independent, so they must agree outright.
    assert np.allclose(normal_wp.numpy(), normal_pp, rtol=1e-4, atol=1e-4)
    # The two tangent bases span the same plane and differ by a rotation about the normal, so the
    # components of basis_x in potpourri3d's basis are a unit (cos, sin) pair.
    basis_x = basis_x_wp.numpy()
    cosine = (basis_x * basis_x_pp).sum(axis=1)
    sine = (basis_x * basis_y_pp).sum(axis=1)
    assert np.allclose(np.hypot(cosine, sine), 1.0, rtol=1e-4, atol=1e-4)
    # ... and the same rotation carries basis_y onto potpourri3d's, i.e. the frames are right-handed
    # in the same orientation rather than mirrored.
    basis_y = basis_y_wp.numpy()
    assert np.allclose((basis_y * basis_x_pp).sum(axis=1), -sine, rtol=1e-4, atol=1e-4)
    assert np.allclose((basis_y * basis_y_pp).sum(axis=1), cosine, rtol=1e-4, atol=1e-4)


def test_vertex_tangent_frames_isolated_vertex(device: str) -> None:
    # An unreferenced vertex has no halfedge to take a reference direction from; the frame must
    # still be finite and orthogonal to its (zero-sum, then normalized) normal.
    vertices_wp = wp.array(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [5.0, 5.0, 5.0]]),
        dtype=wp.vec3,
        device=device,
    )
    faces_wp = wp.array(np.array([0, 1, 2], dtype=np.int32), dtype=wp.int32, device=device)
    basis_x_wp, basis_y_wp, _ = tw.tangent_space.vertex_tangent_frames(vertices_wp, faces_wp)

    assert np.isfinite(basis_x_wp.numpy()).all()
    assert np.allclose(np.linalg.norm(basis_y_wp.numpy(), axis=1), 1.0, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# halfedge_tangent_angles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mesh_name", _MESHES)
def test_halfedge_tangent_angles_span_the_rescaled_disk(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    n_vertices = len(mesh_tm.vertices)
    offsets_wp, ring_wp, is_boundary_wp = tw.halfedge.vertex_one_rings(
        mesh_wp.indices, n_vertices=n_vertices
    )
    angles = tw.tangent_space.halfedge_tangent_angles(
        mesh_wp.points, mesh_wp.indices, rings=(offsets_wp, ring_wp, is_boundary_wp)
    ).numpy()

    offsets, ring, is_boundary = offsets_wp.numpy(), ring_wp.numpy(), is_boundary_wp.numpy()
    corner_angles = tw.triangles.face_angles(mesh_wp.points, mesh_wp.indices).numpy()
    for vertex in range(n_vertices):
        ring_halfedges = ring[offsets[vertex] : offsets[vertex + 1]]
        if len(ring_halfedges) == 0:
            continue
        ring_angles = angles[ring_halfedges]
        corners = np.array([corner_angles[h // 3, h % 3] for h in ring_halfedges])
        full_turn = np.pi if is_boundary[vertex] else 2 * np.pi
        scale = full_turn / corners.sum()

        # The ring starts at the reference direction, and each step is the rescaled corner angle.
        assert ring_angles[0] == 0.0
        expected = scale * np.concatenate([[0.0], np.cumsum(corners)[:-1]])
        assert np.allclose(ring_angles, expected, rtol=1e-5, atol=1e-5)
        # The last corner closes the disk exactly: a half-turn at a boundary vertex, a full turn in
        # the interior, whatever the vertex's angle defect was.
        assert np.isclose(ring_angles[-1] + scale * corners[-1], full_turn, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# halfedge_transport_angles
# ---------------------------------------------------------------------------


# Only meshes without right-angle triangles: an edge whose two opposite angles are both 90 degrees
# has cotangent weight exactly zero, which erases its phase from the connection Laplacian entirely.
# In ``cave_cube`` and ``half_torus`` every face carries such an edge (both are quad grids split by
# a diagonal), so no entry is left to read a transport angle out of — a limit of the oracle, not of
# the computation. Those meshes are covered by the round-trip test below.
@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere"])
def test_halfedge_transport_angle_holonomy_matches_potpourri3d(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    rho = tw.tangent_space.halfedge_transport_angles(mesh_wp.points, mesh_wp.indices).numpy()

    # A single transport angle depends on both endpoints' reference directions, but the holonomy
    # around a face does not — the reference rotations cancel around a closed loop — so that is what
    # can be compared across libraries. potpourri3d's connection Laplacian holds
    # ``w_ij * exp(i * rho_ji)``; dividing by its real cotangent Laplacian, which carries the same
    # weights, leaves the phase (up to one global sign convention, which a triple product cubes).
    faces = np.asarray(mesh_tm.faces)
    vertices_np = np.ascontiguousarray(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.ascontiguousarray(mesh_tm.faces, dtype=np.int32)
    connection_pp = _vector_heat_solver_pp(mesh_tm).get_connection_laplacian().tocsr()
    cotangent_pp = pp3d.cotan_laplacian(vertices_np, faces_np).tocsr()

    corners = ((faces[:, 0], faces[:, 1]), (faces[:, 1], faces[:, 2]), (faces[:, 2], faces[:, 0]))
    weights = [np.asarray(cotangent_pp[row, column]).ravel() for row, column in corners]
    usable = np.logical_and.reduce([np.abs(weight) > 1e-9 for weight in weights])
    assert usable.any()

    phase_pp = np.ones(int(usable.sum()), dtype=complex)
    for (row, column), weight in zip(corners, weights, strict=True):
        phase_pp *= np.asarray(connection_pp[row, column]).ravel()[usable] / weight[usable]
    holonomy_pp = np.angle(phase_pp)
    holonomy_wp = (rho[0::3] + rho[1::3] + rho[2::3])[usable]

    assert np.allclose(
        np.angle(np.exp(1j * (holonomy_wp + holonomy_pp))), 0.0, rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize("mesh_name", _MESHES)
def test_halfedge_transport_angles_are_antisymmetric(
    request: pytest.FixtureRequest, mesh_name: str, device: str
) -> None:
    _, mesh_wp = request.getfixturevalue(mesh_name)
    twins = tw.halfedge.halfedge_twins(mesh_wp.indices).numpy()
    rho = tw.tangent_space.halfedge_transport_angles(mesh_wp.points, mesh_wp.indices).numpy()

    # Transporting a vector across an edge and back is the identity: rho_ij = -rho_ji (mod 2*pi).
    interior = np.flatnonzero(twins >= 0)
    round_trip = np.angle(np.exp(1j * (rho[interior] + rho[twins[interior]])))
    assert np.allclose(round_trip, 0.0, rtol=1e-5, atol=1e-5)
