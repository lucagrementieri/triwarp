from __future__ import annotations

from pathlib import Path

import meshio
import numpy as np
import pytest
import warp as wp

import triwarp as tw


def _write_synthetic_mesh(path: Path) -> dict[str, np.ndarray]:
    """Write a tetrahedron surface with normals, colors and uv; return the source arrays."""
    vertices_np = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    faces_np = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
    normals_np = np.array(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=np.float32
    )
    colors_np = np.array(
        [[10, 20, 30], [200, 100, 50], [0, 0, 0], [255, 255, 255]], dtype=np.uint8
    )
    uv_np = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    point_data_mio = {
        "nx": normals_np[:, 0],
        "ny": normals_np[:, 1],
        "nz": normals_np[:, 2],
        "red": colors_np[:, 0],
        "green": colors_np[:, 1],
        "blue": colors_np[:, 2],
        "u": uv_np[:, 0],
        "v": uv_np[:, 1],
    }
    mesh_mio = meshio.Mesh(vertices_np, [("triangle", faces_np)], point_data=point_data_mio)
    meshio.write(str(path), mesh_mio)
    return {
        "vertices": vertices_np,
        "faces": faces_np,
        "vertex_normals": normals_np,
        "colors": colors_np,
        "uv": uv_np,
    }


def test_load_mesh_data_roundtrip(tmp_path, device):
    path = tmp_path / "mesh.ply"
    source = _write_synthetic_mesh(path)

    data_wp = tw.io.load_mesh_data(path, device=device)

    assert np.allclose(data_wp["vertices"].numpy(), source["vertices"], rtol=1e-5, atol=1e-5)
    assert np.array_equal(data_wp["faces"].numpy().reshape(-1, 3), source["faces"])
    assert np.allclose(
        data_wp["vertex_normals"].numpy(), source["vertex_normals"], rtol=1e-5, atol=1e-5
    )
    assert np.allclose(data_wp["uv"].numpy(), source["uv"], rtol=1e-5, atol=1e-5)
    assert np.allclose(
        data_wp["colors"].numpy(), source["colors"].astype(np.float32) / 255.0, rtol=1e-5, atol=1e-5
    )
    # meshio never computes face normals and none were written.
    assert "face_normals" not in data_wp


def test_load_mesh_returns_wp_mesh(tmp_path, device):
    path = tmp_path / "mesh.ply"
    source = _write_synthetic_mesh(path)

    mesh_wp = tw.io.load_mesh(path, device=device)

    assert isinstance(mesh_wp, wp.Mesh)
    assert np.allclose(mesh_wp.points.numpy(), source["vertices"], rtol=1e-5, atol=1e-5)
    assert np.array_equal(mesh_wp.indices.numpy().reshape(-1, 3), source["faces"])


def test_load_mesh_without_faces_raises(tmp_path, device):
    path = tmp_path / "cloud.ply"
    points_np = np.random.default_rng(0).random((10, 3)).astype(np.float64)
    meshio.write(str(path), meshio.Mesh(points_np, []))

    with pytest.raises(ValueError, match="no triangle faces"):
        tw.io.load_mesh(path, device=device)
