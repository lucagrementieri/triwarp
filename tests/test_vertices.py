"""Regression tests for ``triwarp.vertices`` against Trimesh (CPU reference)."""

import igl
import numpy as np
import pytest
import trimesh as tm
import warp as wp

import triwarp as tw
from tests.conversions import trimesh_to_open3d, trimesh_to_pymeshlab


@pytest.mark.parity("area_weighted_vertex_normals", "open3d", "pymeshlab")
@pytest.mark.parity("mean_vertex_normals", "pymeshlab")
def test_vertex_normal_weightings_match_open3d_and_pymeshlab(
    half_torus: tuple[tm.Trimesh, wp.Mesh],
) -> None:
    """
    The two unweighted-and-area weightings against the libraries that implement the same ones.

    Class A for all three, and they agree to **3.5e-7**, tighter than the 1e-5 tolerance, because
    ``compute_vertex_normals`` and ``compute_normal_per_vertex(weightmode="By Area")`` are the
    same scheme triwarp implements, and ``"Simple Average"`` is the unweighted one.

    **trimesh is deliberately absent**, and that is the finding worth recording:
    ``Trimesh.vertex_normals`` is *angle*-weighted, not area-weighted. It matches
    ``angle_weighted_vertex_normals`` to 4.3e-7 (see ``test_angle_weighted_vertex_normals``) and
    differs from the area-weighted answer by up to **0.072** on this fixture. The
    ``area_weighted_vertex_normals`` benchmark group timed it as though it were the same quantity;
    it is now exempted there with a redirect to open3d.
    """
    mesh_tm, mesh_wp = half_torus
    n_vertices = int(mesh_wp.points.shape[0])

    area_wp = tw.vertices.area_weighted_vertex_normals(n_vertices, mesh_wp.points, mesh_wp.indices)

    mesh_o3d = trimesh_to_open3d(mesh_tm)
    mesh_o3d.compute_vertex_normals()
    assert np.allclose(area_wp.numpy(), np.asarray(mesh_o3d.vertex_normals), rtol=1e-5, atol=1e-5)

    meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    meshset_pml.compute_normal_per_vertex(weightmode="By Area")
    normals_pml = meshset_pml.current_mesh().vertex_normal_matrix()
    assert np.allclose(area_wp.numpy(), normals_pml, rtol=1e-5, atol=1e-5)

    # The unweighted scheme, from the same filter under a different weightmode.
    face_normals_wp, _areas_wp = tw.triangles.face_normals_and_areas(
        mesh_wp.points, mesh_wp.indices
    )
    mean_wp = tw.vertices.mean_vertex_normals(n_vertices, mesh_wp.indices, face_normals_wp)
    mean_meshset_pml = trimesh_to_pymeshlab(mesh_tm)
    mean_meshset_pml.compute_normal_per_vertex(weightmode="Simple Average")
    mean_pml = mean_meshset_pml.current_mesh().vertex_normal_matrix()
    assert np.allclose(mean_wp.numpy(), mean_pml, rtol=1e-5, atol=1e-5)

    # The two weightings are genuinely different, so neither assert above is weightless.
    assert not np.allclose(area_wp.numpy(), mean_wp.numpy(), atol=1e-3)


@pytest.mark.parametrize("mesh_name", ["icosahedron", "half_torus", "hemisphere"])
@pytest.mark.parity("n_vertices", "trimesh")
def test_n_vertices_matches_the_index_maximum(
    request: pytest.FixtureRequest, mesh_name: str
) -> None:
    """
    Vertex count inferred from the face buffer, against the numpy formula the benchmark times.

    ``Trimesh`` has no uncached equivalent -- its vertex count comes from the array it was built
    with -- so the benchmark's "trimesh" row is the stand-in formula ``int(faces.max()) + 1``, and
    that is the reference here. The value is checked against the fixture's actual vertex count too,
    which is the part that would catch an off-by-one that the formula shares.
    """
    mesh_tm, mesh_wp = request.getfixturevalue(mesh_name)
    faces_np = mesh_tm.faces

    assert tw.vertices.n_vertices(mesh_wp.indices) == int(faces_np.max()) + 1
    assert tw.vertices.n_vertices(mesh_wp.indices) == len(mesh_tm.vertices)


def test_mean_vertex_normals(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = half_torus

    n_vertices = mesh_tm.vertices.shape[0]
    face_normals_tm = mesh_tm.face_normals
    vertex_normals_tm = tm.geometry.mean_vertex_normals(n_vertices, mesh_tm.faces, face_normals_tm)

    face_normals_wp = wp.array(face_normals_tm, dtype=wp.vec3, device=mesh_wp.device)
    vertex_normals_wp = tw.vertices.mean_vertex_normals(
        n_vertices, mesh_wp.indices, face_normals_wp
    )
    assert np.allclose(vertex_normals_wp.numpy(), vertex_normals_tm, rtol=1e-5, atol=1e-5)


def test_weighted_vertex_normals(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = half_torus

    n_vertices = mesh_tm.vertices.shape[0]
    face_normals_tm = mesh_tm.face_normals
    face_angles_tm = mesh_tm.face_angles
    vertex_normals_tm = tm.geometry.weighted_vertex_normals(
        n_vertices, mesh_tm.faces, face_normals_tm, face_angles_tm
    )

    face_normals_wp = wp.array(face_normals_tm, dtype=wp.vec3, device=mesh_wp.device)
    face_weights_wp = wp.array(face_angles_tm, dtype=wp.float32, device=mesh_wp.device)
    vertex_normals_wp = tw.vertices.weighted_vertex_normals(
        n_vertices, mesh_wp.indices, face_normals_wp, face_weights_wp
    )
    assert np.allclose(vertex_normals_wp.numpy(), vertex_normals_tm, rtol=1e-5, atol=1e-5)


def test_area_weighted_vertex_normals(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = half_torus

    n_vertices = mesh_tm.vertices.shape[0]
    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int32)
    vertex_normals_igl = igl.per_vertex_normals(
        vertices_np, faces_np, igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA
    )

    vertices_wp = wp.array(mesh_tm.vertices, dtype=wp.vec3, device=mesh_wp.device)
    vertex_normals_wp = tw.vertices.area_weighted_vertex_normals(
        n_vertices, vertices_wp, mesh_wp.indices
    )
    assert np.allclose(vertex_normals_wp.numpy(), vertex_normals_igl, rtol=1e-5, atol=1e-5)


def test_area_weighted_vertex_normals_precomputed(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = half_torus

    n_vertices = mesh_tm.vertices.shape[0]
    vertices_wp = wp.array(mesh_tm.vertices, dtype=wp.vec3, device=mesh_wp.device)

    face_normals_wp, face_areas_wp = tw.triangles.face_normals_and_areas(
        vertices_wp, mesh_wp.indices
    )
    vertex_normals_precomputed_wp = tw.vertices.area_weighted_vertex_normals(
        n_vertices,
        vertices_wp,
        mesh_wp.indices,
        face_normals=face_normals_wp,
        face_areas=face_areas_wp,
    )
    vertex_normals_wp = tw.vertices.area_weighted_vertex_normals(
        n_vertices, vertices_wp, mesh_wp.indices
    )
    assert np.allclose(
        vertex_normals_precomputed_wp.numpy(), vertex_normals_wp.numpy(), rtol=1e-5, atol=1e-5
    )


def test_angle_weighted_vertex_normals(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = half_torus

    n_vertices = mesh_tm.vertices.shape[0]
    vertex_normals_tm = tm.geometry.weighted_vertex_normals(
        n_vertices, mesh_tm.faces, mesh_tm.face_normals, mesh_tm.face_angles
    )

    vertices_wp = wp.array(mesh_tm.vertices, dtype=wp.vec3, device=mesh_wp.device)
    vertex_normals_wp = tw.vertices.angle_weighted_vertex_normals(
        n_vertices, vertices_wp, mesh_wp.indices
    )
    assert np.allclose(vertex_normals_wp.numpy(), vertex_normals_tm, rtol=1e-5, atol=1e-5)


def test_angle_weighted_vertex_normals_precomputed(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = half_torus

    n_vertices = mesh_tm.vertices.shape[0]
    vertices_wp = wp.array(mesh_tm.vertices, dtype=wp.vec3, device=mesh_wp.device)

    face_normals_wp, _ = tw.triangles.face_normals_and_areas(vertices_wp, mesh_wp.indices)
    face_angles_wp = tw.triangles.face_angles(vertices_wp, mesh_wp.indices)
    vertex_normals_precomputed_wp = tw.vertices.angle_weighted_vertex_normals(
        n_vertices,
        vertices_wp,
        mesh_wp.indices,
        face_normals=face_normals_wp,
        face_angles=face_angles_wp,
    )
    vertex_normals_wp = tw.vertices.angle_weighted_vertex_normals(
        n_vertices, vertices_wp, mesh_wp.indices
    )
    assert np.allclose(
        vertex_normals_precomputed_wp.numpy(), vertex_normals_wp.numpy(), rtol=1e-5, atol=1e-5
    )


def _compute_max_vertex_normals_np(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Nelson Max MWSELR vertex normals: (e1 x e2) / (||e1||^2 * ||e2||^2) per corner."""
    corner = np.arange(3)
    i0 = faces
    i1 = faces[:, np.roll(corner, -1)]
    i2 = faces[:, np.roll(corner, -2)]
    e1 = vertices[i1] - vertices[i0]
    e2 = vertices[i2] - vertices[i0]
    cross = np.cross(e1, e2)
    len_sq = np.sum(e1**2, axis=-1) * np.sum(e2**2, axis=-1)
    contrib = cross / np.where(len_sq[..., None] == 0, 1.0, len_sq[..., None])
    vertex_normals_np = np.zeros_like(vertices, dtype=np.float64)
    np.add.at(vertex_normals_np, i0.ravel(), contrib.reshape(-1, 3))
    norms = np.linalg.norm(vertex_normals_np, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vertex_normals_np / norms


def test_sine_and_edge_length_weighted_vertex_normals(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = half_torus

    n_vertices = mesh_tm.vertices.shape[0]
    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int32)
    vertex_normals_np = _compute_max_vertex_normals_np(vertices_np, faces_np)

    vertices_wp = wp.array(mesh_tm.vertices, dtype=wp.vec3, device=mesh_wp.device)
    vertex_normals_wp = tw.vertices.sine_and_edge_length_weighted_vertex_normals(
        n_vertices, vertices_wp, mesh_wp.indices
    )
    assert np.allclose(vertex_normals_wp.numpy(), vertex_normals_np, rtol=1e-5, atol=1e-5)

    face_normals_wp = wp.array(mesh_tm.face_normals, dtype=wp.vec3, device=mesh_wp.device)
    vertex_normals_explicit_wp = tw.vertices.sine_and_edge_length_weighted_vertex_normals(
        n_vertices, vertices_wp, mesh_wp.indices, face_normals=face_normals_wp
    )
    assert np.allclose(vertex_normals_explicit_wp.numpy(), vertex_normals_np, rtol=1e-5, atol=1e-5)


def test_vertex_defects(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = half_torus

    n_vertices = mesh_tm.vertices.shape[0]
    face_angles_tm = mesh_tm.face_angles
    vertex_defects_tm = tm.curvature.vertex_defects(mesh_tm)

    face_angles_wp = wp.array(face_angles_tm, dtype=wp.float32, device=mesh_wp.device)
    vertex_defects_wp = tw.vertices.vertex_defects(n_vertices, mesh_wp.indices, face_angles_wp)
    assert np.allclose(vertex_defects_wp.numpy(), vertex_defects_tm, rtol=1e-5, atol=1e-5)
