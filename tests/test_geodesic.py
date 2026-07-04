"""Regression tests for ``triwarp.geodesic`` against igl (CPU reference)."""

from __future__ import annotations

import igl
import numpy as np
import pytest
import warp as wp

import triwarp as tw


def _heat_geodesic_igl(
    vertices_np: np.ndarray, faces_np: np.ndarray, sources_np: np.ndarray
) -> np.ndarray:
    data = igl.HeatGeodesicsData()
    igl.heat_geodesics_precompute(vertices_np, faces_np, data)
    return np.asarray(igl.heat_geodesics_solve(data, sources_np))


def _skip_on_cpu(device: str) -> None:
    # heat_geodesic needs a non-trivial CG solve, which warp.optim.linear.cg cannot do on the CPU
    # device (NaN) in Warp 1.14.0. The solver raises NotImplementedError there, so skip the
    # comparison tests when no CUDA device is available.
    if wp.get_device(device).is_cpu:
        pytest.skip(
            "heat_geodesic requires a CUDA device (warp CG is broken on CPU in Warp 1.14.0)"
        )


@pytest.mark.parametrize("mesh_name", ["icosahedron", "hemisphere"])
def test_heat_geodesic_matches_igl(
    request: pytest.FixtureRequest, device: str, mesh_name: str
) -> None:
    _skip_on_cpu(device)
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int64)
    sources_np = np.array([0], dtype=np.int64)
    sources_wp = wp.array(sources_np.astype(np.int32), dtype=wp.int32, device=mesh_wp.device)

    distance_igl = _heat_geodesic_igl(vertices_np, faces_np, sources_np)
    distance_wp = tw.geodesic.heat_geodesic(mesh_wp.points, mesh_wp.indices, sources_wp)

    assert np.allclose(distance_wp.numpy(), distance_igl, rtol=5e-2, atol=5e-2)


def test_heat_geodesic_multi_source_matches_igl(
    device: str, icosahedron: tuple[object, wp.Mesh]
) -> None:
    _skip_on_cpu(device)
    mesh_tm, mesh_wp = icosahedron
    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)  # type: ignore[attr-defined]
    faces_np = np.array(mesh_tm.faces, dtype=np.int64)  # type: ignore[attr-defined]
    sources_np = np.array([0, len(vertices_np) // 2], dtype=np.int64)
    sources_wp = wp.array(sources_np.astype(np.int32), dtype=wp.int32, device=mesh_wp.device)

    distance_igl = _heat_geodesic_igl(vertices_np, faces_np, sources_np)
    distance_wp = tw.geodesic.heat_geodesic(mesh_wp.points, mesh_wp.indices, sources_wp)

    assert np.allclose(distance_wp.numpy(), distance_igl, rtol=5e-2, atol=5e-2)


def test_heat_geodesic_approximates_exact(
    device: str, icosahedron: tuple[object, wp.Mesh]
) -> None:
    _skip_on_cpu(device)
    mesh_tm, mesh_wp = icosahedron
    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)  # type: ignore[attr-defined]
    faces_np = np.array(mesh_tm.faces, dtype=np.int64)  # type: ignore[attr-defined]
    n_vertices = len(vertices_np)
    empty = np.array([], dtype=np.int64)
    sources_np = np.array([0], dtype=np.int64)

    distance_exact = igl.exact_geodesic(
        vertices_np, faces_np, sources_np, empty, np.arange(n_vertices, dtype=np.int64), empty
    )
    sources_wp = wp.array(sources_np.astype(np.int32), dtype=wp.int32, device=mesh_wp.device)
    distance_wp = tw.geodesic.heat_geodesic(mesh_wp.points, mesh_wp.indices, sources_wp)

    # Heat method is an approximation of the true geodesic distance; a loose tolerance.
    assert np.allclose(distance_wp.numpy(), distance_exact, rtol=8e-2, atol=1e-1)


def test_heat_geodesic_source_is_zero_and_nonnegative(
    device: str, hemisphere: tuple[object, wp.Mesh]
) -> None:
    _skip_on_cpu(device)
    _, mesh_wp = hemisphere
    sources_np = np.array([0], dtype=np.int32)
    sources_wp = wp.array(sources_np, dtype=wp.int32, device=mesh_wp.device)

    distance = tw.geodesic.heat_geodesic(mesh_wp.points, mesh_wp.indices, sources_wp).numpy()

    assert np.all(distance >= -1e-6)
    assert np.allclose(distance[sources_np], 0.0, atol=1e-4)


def test_heat_geodesic_empty_faces(device: str) -> None:
    vertices = wp.array(
        np.zeros((4, 3), dtype=np.float32), dtype=wp.vec3, device=device
    )
    faces = wp.empty(0, dtype=wp.int32, device=device)
    sources = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=device)

    distance = tw.geodesic.heat_geodesic(vertices, faces, sources)

    assert distance.shape[0] == 4
    assert np.array_equal(distance.numpy(), np.zeros(4))


def test_heat_geodesic_empty_sources(icosahedron: tuple[object, wp.Mesh]) -> None:
    _, mesh_wp = icosahedron
    sources = wp.empty(0, dtype=wp.int32, device=mesh_wp.device)

    distance = tw.geodesic.heat_geodesic(mesh_wp.points, mesh_wp.indices, sources)

    assert np.array_equal(distance.numpy(), np.zeros(int(mesh_wp.points.shape[0])))


def test_heat_geodesic_cpu_solve_raises() -> None:
    # warp.optim.linear.cg produces NaN on the CPU device (Warp 1.14.0), so a solve on CPU must
    # fail loudly rather than return garbage.
    vertices = wp.array(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        dtype=wp.vec3,
        device="cpu",
    )
    faces = wp.array(np.array([0, 1, 2], dtype=np.int32), dtype=wp.int32, device="cpu")
    sources = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device="cpu")

    with pytest.raises(NotImplementedError):
        tw.geodesic.heat_geodesic(vertices, faces, sources)
