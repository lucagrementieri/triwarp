from __future__ import annotations

import numpy as np
import pytest
import trimesh as tm
import warp as wp


def _trimesh_to_warp(mesh: tm.Trimesh, device: str) -> wp.Mesh:
    vertices = wp.array(np.ascontiguousarray(mesh.vertices, dtype=np.float32), dtype=wp.vec3, device=device)
    faces = wp.array(np.ascontiguousarray(mesh.faces.reshape(-1), dtype=np.int32), dtype=wp.int32, device=device)
    return wp.Mesh(points=vertices, indices=faces)


@pytest.fixture
def device():
    if wp.is_cuda_available():
        return "cuda:0"
    return "cpu"


@pytest.fixture
def icosahedron(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    icosahedron = tm.creation.icosahedron()
    icosahedron.apply_translation(translation=np.array([-1.0, 0.0, 2.0]))
    return icosahedron, _trimesh_to_warp(icosahedron, device)


@pytest.fixture
def half_torus(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    torus = tm.creation.torus(major_radius=1.0, minor_radius=0.5)
    half_torus = torus.slice_plane(plane_origin=np.zeros(3), plane_normal=np.array([1.0, 0.0, 0.0]))
    scale = 1 + np.exp(-half_torus.vertices[:, 1])
    half_torus.vertices *= scale[:, None]
    half_torus.apply_translation(translation=np.array([-1.0, 0.0, 2.0]))
    return half_torus, _trimesh_to_warp(half_torus, device)


@pytest.fixture
def hemisphere(device: str) -> tuple[tm.Trimesh, wp.Mesh]:
    sphere = tm.creation.icosphere(subdivisions=2, radius=1.0)
    hemisphere = sphere.slice_plane(plane_origin=np.zeros(3), plane_normal=np.array([0.0, 0.0, 1.0]), cap=False)
    rotation = tm.transformations.rotation_matrix(np.deg2rad(45.0), direction=np.array([1.0, 1.0, 0.0]))
    rotation[:3, 3] = np.array([-1.0, 0.0, 2.0])
    hemisphere.apply_transform(rotation)
    return hemisphere, _trimesh_to_warp(hemisphere, device)
