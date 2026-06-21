"""Regression tests for ``triwarp.curvature`` against ``trimesh.curvature`` (CPU reference)."""

import heapq
from collections import deque

import igl
import numpy as np
import pytest
import trimesh as tm
import warp as wp

import triwarp as tw


def _geodesic_ball_neighborhoods_oracle(
    vertices_np: np.ndarray, faces_np: np.ndarray, radius: float, min_count: int = 6
) -> tuple[list[list[int]], np.ndarray]:
    """
    Host NumPy reference (the original ``_geodesic_ball_neighborhoods``) used as the oracle.

    Returns the per-vertex collected lists and the reference-neighbor array.
    """
    n = vertices_np.shape[0]
    adjacency: list[list[int]] = [[] for _ in range(n)]
    seen: set[tuple[int, int]] = set()
    for tri in faces_np:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            a, b = int(a), int(b)
            key = (a, b) if a < b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            adjacency[a].append(b)
            adjacency[b].append(a)
    reference = np.array(
        [min(adjacency[i]) if adjacency[i] else i for i in range(n)], dtype=np.int32
    )
    per_vertex: list[list[int]] = []
    for i in range(n):
        center = vertices_np[i]
        visited = {i}
        queue = deque([i])
        collected: list[int] = []
        extras: list[tuple[float, int]] = []
        while queue:
            current = queue.popleft()
            collected.append(current)
            for neighbor in adjacency[current]:
                if neighbor in visited:
                    continue
                distance = float(np.linalg.norm(center - vertices_np[neighbor]))
                if distance < radius:
                    queue.append(neighbor)
                elif len(collected) < min_count:
                    heapq.heappush(extras, (distance, neighbor))
                visited.add(neighbor)
        while extras and len(collected) < min_count:
            _, cand = heapq.heappop(extras)
            collected.append(cand)
            for neighbor in adjacency[cand]:
                if neighbor in visited:
                    continue
                distance = float(np.linalg.norm(center - vertices_np[neighbor]))
                heapq.heappush(extras, (distance, neighbor))
                visited.add(neighbor)
        per_vertex.append(collected)
    return per_vertex, reference


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus"])
def test_geodesic_ball_neighborhoods(mesh_name: str, request: pytest.FixtureRequest) -> None:
    """On-device geodesic balls match the NumPy/libigl oracle (per-row set equality)."""
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int32)

    vertices_wp = wp.array(vertices_np.astype(np.float32), dtype=wp.vec3, device=mesh_wp.device)
    faces_wp = wp.array(mesh_wp.indices, dtype=wp.int32, device=mesh_wp.device)

    radius = 3.0 * tw.edges.mean_edge_length(vertices_wp, faces_wp)
    neighbor_indices_wp, offsets_wp, reference_wp = tw.proximity.query_geodesic_ball(
        vertices_wp, faces_wp, radius
    )
    per_vertex_oracle, reference_oracle = _geodesic_ball_neighborhoods_oracle(
        vertices_np, faces_np, radius
    )

    assert np.array_equal(reference_wp.numpy(), reference_oracle)

    neighbor_indices = neighbor_indices_wp.numpy()
    offsets = offsets_wp.numpy()
    total = neighbor_indices.shape[0]
    n = vertices_np.shape[0]
    for i in range(n):
        start = int(offsets[i])
        end = int(offsets[i + 1]) if i + 1 < n else total
        neighbors_wp = {int(x) for x in neighbor_indices[start:end]}
        assert neighbors_wp == set(per_vertex_oracle[i]), f"vertex {i} neighborhood differs"


def test_geodesic_ball_neighborhoods_overflow_warns() -> None:
    """A neighborhood exceeding the fixed 512 cap clamps (does not crash) and warns."""
    # A subdivided icosphere has > 512 vertices; a radius covering the whole mesh makes every
    # vertex's geodesic ball the entire connected component, exceeding the fixed scratch capacity.
    mesh_tm = tm.creation.icosphere(subdivisions=4)
    assert mesh_tm.vertices.shape[0] > 512
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    vertices_wp = wp.array(
        np.array(mesh_tm.vertices, dtype=np.float32), dtype=wp.vec3, device=device
    )
    faces_wp = wp.array(np.array(mesh_tm.faces, dtype=np.int32).reshape(-1), device=device)
    radius = 100.0 * float(mesh_tm.scale)

    with pytest.warns(UserWarning, match="capacity breaches"):
        neighbor_indices_wp, offsets_wp, _ = tw.proximity.query_geodesic_ball(
            vertices_wp, faces_wp, radius
        )
    # Clamped, not crashed: every per-vertex count fits within the fixed capacity.
    counts = np.diff(np.append(offsets_wp.numpy(), neighbor_indices_wp.shape[0]))
    assert counts.max() <= 512


def test_principal_curvature(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """Curvature values against libigl reference on an icosahedron (frame-dependent path)."""
    mesh_tm, mesh_wp = icosahedron

    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int32)
    _, _, pv1_igl, pv2_igl, _ = igl.principal_curvature(vertices_np, faces_np, useKring=False)

    vertices_wp = wp.array(vertices_np.astype(np.float32), dtype=wp.vec3, device=mesh_wp.device)
    faces_wp = wp.array(mesh_wp.indices, dtype=wp.int32, device=mesh_wp.device)
    # frame_independent=False reproduces igl::principal_curvature's symmetrized shape operator.
    _, _, pv1_wp, pv2_wp = tw.curvature.principal_curvature(
        vertices_wp, faces_wp, frame_independent=False
    )

    assert np.allclose(pv1_wp.numpy(), pv1_igl, atol=1e-3, rtol=1e-3)
    assert np.allclose(pv2_wp.numpy(), pv2_igl, atol=1e-3, rtol=1e-3)


def test_principal_curvature_half_torus(half_torus: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """Curvature values and directions on a surface with spatially varying curvature."""
    mesh_tm, mesh_wp = half_torus

    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int32)
    pd1_igl, pd2_igl, pv1_igl, pv2_igl, bad_igl = igl.principal_curvature(
        vertices_np, faces_np, useKring=False
    )

    vertices_wp = wp.array(vertices_np.astype(np.float32), dtype=wp.vec3, device=mesh_wp.device)
    faces_wp = wp.array(mesh_wp.indices, dtype=wp.int32, device=mesh_wp.device)
    # frame_independent=False reproduces igl::principal_curvature's symmetrized shape operator.
    pd1_wp, pd2_wp, pv1_wp, pv2_wp = tw.curvature.principal_curvature(
        vertices_wp, faces_wp, frame_independent=False
    )

    # Exclude vertices igl marked bad (degenerate) and umbilics where PV1 ~ PV2 (dirs undefined)
    bad = np.array(bad_igl, dtype=np.int32)
    gap = np.abs(pv1_igl - pv2_igl)
    mask = np.ones(len(pv1_igl), dtype=bool)
    if len(bad) > 0:
        mask[bad] = False
    mask[gap < 1e-2] = False

    # Tolerance is relaxed relative to the icosahedron test: float32 input vs libigl float64,
    # plus slight radius difference from avg_edge_length rounding.
    assert np.allclose(pv1_wp.numpy()[mask], pv1_igl[mask], atol=5e-2, rtol=5e-2)
    assert np.allclose(pv2_wp.numpy()[mask], pv2_igl[mask], atol=5e-2, rtol=5e-2)
    # Directions defined up to sign — compare |cos angle| ≈ 1 at non-umbilic vertices
    pd1_dot = np.abs(np.einsum("ij,ij->i", pd1_wp.numpy()[mask], pd1_igl[mask]))
    pd2_dot = np.abs(np.einsum("ij,ij->i", pd2_wp.numpy()[mask], pd2_igl[mask]))
    assert np.allclose(pd1_dot, 1.0, atol=1e-1)
    assert np.allclose(pd2_dot, 1.0, atol=1e-1)


def test_principal_curvature_frame_independent(half_torus: tuple[tm.Trimesh, wp.Mesh]) -> None:
    """The default frame-independent Weingarten map stays similar to the libigl reference.

    ``frame_independent=True`` solves the true generalized eigenproblem (a surface invariant)
    rather than libigl's frame-dependent symmetrized operator. The two formulations share the
    trace of the shape operator, so the mean curvature ``(PV1 + PV2) / 2`` is preserved exactly;
    only the eigenvalue *spread* differs, and only appreciably at high-anisotropy vertices where
    ``PV1 - PV2`` is large. The bulk of vertices therefore stay close to libigl.
    """
    mesh_tm, mesh_wp = half_torus

    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int32)
    _, _, pv1_igl, pv2_igl, bad_igl = igl.principal_curvature(
        vertices_np, faces_np, useKring=False
    )

    vertices_wp = wp.array(vertices_np.astype(np.float32), dtype=wp.vec3, device=mesh_wp.device)
    faces_wp = wp.array(mesh_wp.indices, dtype=wp.int32, device=mesh_wp.device)
    # Default (frame_independent=True): true Weingarten map, independent of the tangent frame.
    _, _, pv1_wp, pv2_wp = tw.curvature.principal_curvature(vertices_wp, faces_wp)
    pv1_indep = pv1_wp.numpy()
    pv2_indep = pv2_wp.numpy()

    # Same masking as the frame-dependent test: drop degenerate and umbilic vertices.
    bad = np.array(bad_igl, dtype=np.int32)
    gap = np.abs(pv1_igl - pv2_igl)
    mask = np.ones(len(pv1_igl), dtype=bool)
    if len(bad) > 0:
        mask[bad] = False
    mask[gap < 1e-2] = False

    # Mean curvature (the shared trace invariant) must match libigl tightly.
    mean_indep = 0.5 * (pv1_indep + pv2_indep)
    mean_igl = 0.5 * (pv1_igl + pv2_igl)
    assert np.allclose(mean_indep[mask], mean_igl[mask], atol=5e-2, rtol=5e-2)

    # The principal values themselves stay close for the vast majority of vertices; genuine
    # divergence is confined to the few highest-anisotropy vertices.
    within_pv1 = np.abs(pv1_indep[mask] - pv1_igl[mask]) <= 5e-2 + 5e-2 * np.abs(pv1_igl[mask])
    within_pv2 = np.abs(pv2_indep[mask] - pv2_igl[mask]) <= 5e-2 + 5e-2 * np.abs(pv2_igl[mask])
    assert within_pv1.mean() > 0.95
    assert within_pv2.mean() > 0.95


def test_discrete_gaussian_curvature(hemisphere: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = hemisphere

    face_angles_tm = mesh_tm.face_angles
    points_tm = mesh_tm.vertices[:4]
    radius = 0.1
    gauss_curvature_tm = tm.curvature.discrete_gaussian_curvature_measure(
        mesh_tm, points_tm, radius
    )

    points_wp = wp.array(points_tm, dtype=wp.vec3, device=mesh_wp.device)
    vertices_wp = wp.array(mesh_tm.vertices, dtype=wp.vec3, device=mesh_wp.device)
    faces_wp = wp.array(mesh_wp.indices, dtype=wp.int32, device=mesh_wp.device)
    face_angles_wp = wp.array(face_angles_tm, dtype=wp.float32, device=mesh_wp.device)
    gauss_curvature_wp = tw.curvature.discrete_gaussian_curvature(
        points_wp, vertices_wp, faces_wp, face_angles_wp, radius
    )
    assert np.allclose(gauss_curvature_wp.numpy(), gauss_curvature_tm, rtol=1e-5, atol=1e-5)


def test_discrete_mean_curvature(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> None:
    mesh_tm, mesh_wp = icosahedron
    radius = 2.0
    points_tm = mesh_tm.vertices
    mean_curvature_tm = tm.curvature.discrete_mean_curvature_measure(mesh_tm, points_tm, radius)

    points_wp = wp.array(points_tm.astype(np.float32), dtype=wp.vec3, device=mesh_wp.device)
    vertices_wp = wp.array(
        mesh_tm.vertices.astype(np.float32), dtype=wp.vec3, device=mesh_wp.device
    )
    faces_wp = wp.array(mesh_wp.indices, dtype=wp.int32, device=mesh_wp.device)
    mean_curvature_wp = tw.curvature.discrete_mean_curvature(
        points_wp, vertices_wp, faces_wp, radius
    )
    assert np.allclose(mean_curvature_wp.numpy(), mean_curvature_tm, rtol=1e-5, atol=1e-5)
