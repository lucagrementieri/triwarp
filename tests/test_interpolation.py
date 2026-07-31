"""Regression tests for ``triwarp.interpolation`` against igl (CPU reference)."""

import igl
import numpy as np
import pymeshlab as ml
import pytest
import trimesh as tm
import warp as wp

import triwarp as tw


def test_average_onto_faces(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = half_torus
    rng = np.random.default_rng(0)

    faces_np = np.array(mesh_tm.faces, dtype=np.int64)
    vertex_values_np = rng.uniform(size=mesh_tm.vertices.shape[0])
    face_values_igl = igl.average_onto_faces(faces_np, vertex_values_np)

    vertex_values_wp = wp.array(vertex_values_np, dtype=wp.float32, device=mesh_wp.device)
    face_values_wp = tw.interpolation.average_onto_faces(mesh_wp.indices, vertex_values_wp)
    assert np.allclose(face_values_wp.numpy(), face_values_igl, rtol=1e-5, atol=1e-5)


@pytest.mark.parity("average_onto_vertices", "pymeshlab")
def test_average_onto_vertices(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    """
    Class A against libigl, class B against MeshLab's face-to-vertex scalar transfer.

    ``compute_scalar_transfer_face_to_vertex`` reads the *face* scalar attribute and writes the
    *vertex* one rather than taking and returning arrays, so the transform is seeding
    ``f_scalar_array`` on the way in and reading ``vertex_scalar_array()`` on the way out.
    ``areaweight=False`` is load-bearing and is what the benchmark passes: its default weights each
    incident face by area, where this function takes the plain corner mean.
    """
    mesh_tm, mesh_wp = half_torus
    rng = np.random.default_rng(1)

    n_vertices = mesh_tm.vertices.shape[0]
    vertices_np = np.array(mesh_tm.vertices, dtype=np.float64)
    faces_np = np.array(mesh_tm.faces, dtype=np.int64)
    face_values_np = rng.uniform(size=mesh_tm.faces.shape[0])
    vertex_values_igl = igl.average_onto_vertices(vertices_np, faces_np, face_values_np)

    meshset_pml = ml.MeshSet()
    meshset_pml.add_mesh(
        ml.Mesh(
            np.ascontiguousarray(vertices_np),
            np.ascontiguousarray(mesh_tm.faces, dtype=np.int32),
            f_scalar_array=np.ascontiguousarray(face_values_np),
        )
    )
    meshset_pml.compute_scalar_transfer_face_to_vertex(areaweight=False)
    vertex_values_pml = np.asarray(meshset_pml.current_mesh().vertex_scalar_array())

    face_values_wp = wp.array(face_values_np, dtype=wp.float32, device=mesh_wp.device)
    vertex_values_wp = tw.interpolation.average_onto_vertices(
        n_vertices, mesh_wp.indices, face_values_wp
    )
    assert np.allclose(vertex_values_wp.numpy(), vertex_values_igl, rtol=1e-5, atol=1e-5)
    assert np.allclose(vertex_values_wp.numpy(), vertex_values_pml, rtol=1e-5, atol=1e-5)


def test_average_from_edges_onto_vertices(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = half_torus
    rng = np.random.default_rng(2)

    n_vertices = mesh_tm.vertices.shape[0]
    faces_np = np.array(mesh_tm.faces, dtype=np.int64)
    edges_igl, orientation_igl = igl.orient_halfedges(faces_np)
    edges_igl = np.asarray(edges_igl)
    orientation_igl = np.asarray(orientation_igl)
    n_unique_edges = int(edges_igl.max()) + 1
    edge_values_np = rng.uniform(size=n_unique_edges)
    vertex_values_igl = igl.average_from_edges_onto_vertices(
        faces_np, edges_igl, orientation_igl, edge_values_np
    )

    edges_wp = wp.array(edges_igl.astype(np.int32), dtype=wp.int32, device=mesh_wp.device)
    orientation_wp = wp.array(
        orientation_igl.astype(np.int32), dtype=wp.int32, device=mesh_wp.device
    )
    edge_values_wp = wp.array(edge_values_np, dtype=wp.float32, device=mesh_wp.device)
    vertex_values_wp = tw.interpolation.average_from_edges_onto_vertices(
        n_vertices, mesh_wp.indices, edges_wp, orientation_wp, edge_values_wp
    )
    assert np.allclose(vertex_values_wp.numpy(), vertex_values_igl, rtol=1e-5, atol=1e-5)


def _transfer_meshes(device: str) -> tuple[tm.Trimesh, tm.Trimesh]:
    """Build a fine source and a coarse target of one shape: the remesh/decimate situation."""
    del device
    return tm.creation.icosphere(subdivisions=3), tm.creation.icosphere(subdivisions=2)


@pytest.mark.parity("transfer_onto_vertices", "pymeshlab")
def test_transfer_onto_vertices_matches_pymeshlab(device: str):
    """``transfer_attributes_per_vertex`` with ``qualitytransfer`` is the same barycentric pull."""
    source_tm, target_tm = _transfer_meshes(device)
    values_np = np.ascontiguousarray(source_tm.vertices[:, 0] + 2.0, dtype=np.float64)

    meshset_pml = ml.MeshSet()
    meshset_pml.add_mesh(
        ml.Mesh(
            np.ascontiguousarray(source_tm.vertices, dtype=np.float64),
            np.ascontiguousarray(source_tm.faces, dtype=np.int32),
            v_scalar_array=values_np,
        )
    )
    meshset_pml.add_mesh(
        ml.Mesh(
            np.ascontiguousarray(target_tm.vertices, dtype=np.float64),
            np.ascontiguousarray(target_tm.faces, dtype=np.int32),
        )
    )
    meshset_pml.transfer_attributes_per_vertex(
        sourcemesh=0,
        targetmesh=1,
        qualitytransfer=True,
        colortransfer=False,
        upperbound=ml.PercentageValue(50),
    )
    meshset_pml.set_current_mesh(1)
    transferred_pml = meshset_pml.current_mesh().vertex_scalar_array()

    source_vertices_wp = wp.array(
        np.ascontiguousarray(source_tm.vertices, dtype=np.float32), dtype=wp.vec3, device=device
    )
    source_faces_wp = wp.array(
        np.ascontiguousarray(source_tm.faces.reshape(-1), dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )
    target_vertices_wp = wp.array(
        np.ascontiguousarray(target_tm.vertices, dtype=np.float32), dtype=wp.vec3, device=device
    )
    values_wp = wp.array(values_np.astype(np.float32), dtype=wp.float32, device=device)
    transferred_wp, distance_wp = tw.interpolation.transfer_onto_vertices(
        source_vertices_wp, source_faces_wp, values_wp, target_vertices_wp
    )
    assert np.allclose(transferred_wp.numpy(), transferred_pml, rtol=1e-4, atol=1e-4)
    assert np.isfinite(distance_wp.numpy()).all()


def test_transfer_onto_vertices_reproduces_a_linear_field(device: str):
    """A field that is linear on the source triangles must come back exactly, not blurred."""
    source_tm, target_tm = _transfer_meshes(device)
    direction_np = np.array([0.3, -0.6, 0.74])
    values_np = source_tm.vertices @ direction_np

    source_vertices_wp = wp.array(
        np.ascontiguousarray(source_tm.vertices, dtype=np.float32), dtype=wp.vec3, device=device
    )
    source_faces_wp = wp.array(
        np.ascontiguousarray(source_tm.faces.reshape(-1), dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )
    # Target vertices projected onto the source surface first, so "linear on the source" holds
    # exactly rather than up to the two spheres' radial gap.
    projected_wp, _distance, _face = tw.proximity.closest_point_on_mesh(
        source_vertices_wp,
        source_faces_wp,
        wp.array(
            np.ascontiguousarray(target_tm.vertices, dtype=np.float32), dtype=wp.vec3, device=device
        ),
    )
    values_wp = wp.array(values_np.astype(np.float32), dtype=wp.float32, device=device)
    transferred_wp, _distance = tw.interpolation.transfer_onto_vertices(
        source_vertices_wp, source_faces_wp, values_wp, projected_wp
    )
    assert np.allclose(
        transferred_wp.numpy(), projected_wp.numpy() @ direction_np, rtol=1e-4, atol=1e-4
    )


def test_transfer_onto_vertices_vec3_field(device: str):
    """The transfer is dtype-generic: a ``wp.vec3`` field (a normal, a colour) works unchanged."""
    source_tm, target_tm = _transfer_meshes(device)
    source_vertices_wp = wp.array(
        np.ascontiguousarray(source_tm.vertices, dtype=np.float32), dtype=wp.vec3, device=device
    )
    source_faces_wp = wp.array(
        np.ascontiguousarray(source_tm.faces.reshape(-1), dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )
    target_vertices_wp = wp.array(
        np.ascontiguousarray(target_tm.vertices, dtype=np.float32), dtype=wp.vec3, device=device
    )
    # Transferring the source *positions* must reproduce each target vertex's closest point.
    transferred_wp, _distance = tw.interpolation.transfer_onto_vertices(
        source_vertices_wp, source_faces_wp, source_vertices_wp, target_vertices_wp
    )
    closest_wp, _distance, _face = tw.proximity.closest_point_on_mesh(
        source_vertices_wp, source_faces_wp, target_vertices_wp
    )
    assert np.allclose(transferred_wp.numpy(), closest_wp.numpy(), rtol=1e-4, atol=1e-4)


def test_transfer_onto_vertices_misses_stay_zero(device: str):
    """A target beyond ``max_dist`` keeps the zero fill and reports ``inf``."""
    source_tm, _target_tm = _transfer_meshes(device)
    source_vertices_wp = wp.array(
        np.ascontiguousarray(source_tm.vertices, dtype=np.float32), dtype=wp.vec3, device=device
    )
    source_faces_wp = wp.array(
        np.ascontiguousarray(source_tm.faces.reshape(-1), dtype=np.int32),
        dtype=wp.int32,
        device=device,
    )
    values_wp = wp.array(
        np.ones(source_tm.vertices.shape[0], dtype=np.float32), dtype=wp.float32, device=device
    )
    targets_wp = wp.array(
        np.array([[0.0, 0.0, 1.0], [100.0, 0.0, 0.0]], dtype=np.float32),
        dtype=wp.vec3,
        device=device,
    )
    transferred_wp, distance_wp = tw.interpolation.transfer_onto_vertices(
        source_vertices_wp, source_faces_wp, values_wp, targets_wp, max_dist=0.1
    )
    assert np.isclose(transferred_wp.numpy()[0], 1.0, rtol=1e-5)
    assert transferred_wp.numpy()[1] == 0.0
    assert np.isinf(distance_wp.numpy()[1])


def test_transfer_onto_vertices_length_mismatch(device: str):
    source_vertices_wp = wp.zeros(4, dtype=wp.vec3, device=device)
    source_faces_wp = wp.array([0, 1, 2], dtype=wp.int32, device=device)
    values_wp = wp.zeros(3, dtype=wp.float32, device=device)
    targets_wp = wp.zeros(2, dtype=wp.vec3, device=device)
    with pytest.raises(ValueError, match="one entry per source vertex"):
        tw.interpolation.transfer_onto_vertices(
            source_vertices_wp, source_faces_wp, values_wp, targets_wp
        )


def test_transfer_onto_vertices_empty(device: str):
    source_vertices_wp = wp.zeros(0, dtype=wp.vec3, device=device)
    source_faces_wp = wp.empty(0, dtype=wp.int32, device=device)
    values_wp = wp.zeros(0, dtype=wp.float32, device=device)
    targets_wp = wp.zeros(3, dtype=wp.vec3, device=device)
    transferred_wp, distance_wp = tw.interpolation.transfer_onto_vertices(
        source_vertices_wp, source_faces_wp, values_wp, targets_wp
    )
    assert np.array_equal(transferred_wp.numpy(), np.zeros(3, dtype=np.float32))
    assert np.isinf(distance_wp.numpy()).all()
