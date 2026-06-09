"""Regression tests for ``triwarp.sample`` vs ``trimesh.sample`` (CPU reference)."""

from __future__ import annotations

import numpy as np
import trimesh as tm
import warp as wp

import triwarp.sample as tw


def test_sample_surface(half_torus: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = half_torus
    count = 10_000
    n_faces = int(mesh_tm.faces.shape[0])
    face_idx_tm = tm.sample.sample_surface(mesh_tm, count, seed=0)[1]
    freq_tm = np.bincount(face_idx_tm, minlength=n_faces) / count

    face_idx_wp = tw.sample_surface(mesh_wp.points, mesh_wp.indices, count, seed=0)[1]
    freq_wp = np.bincount(face_idx_wp.numpy(), minlength=n_faces) / count

    freq_expected = mesh_tm.area_faces / mesh_tm.area_faces.sum()
    assert np.allclose(freq_wp, freq_expected, rtol=0.08, atol=0.008)
    assert np.allclose(freq_tm, freq_expected, rtol=0.08, atol=0.008)


def test_sample_surface_with_face_weights(icosahedron: tuple[tm.Trimesh, wp.Mesh]):
    mesh_tm, mesh_wp = icosahedron
    count = 10_000

    weights_np = np.arange(mesh_tm.faces.shape[0], dtype=np.float32)
    face_idx_tm = tm.sample.sample_surface(mesh_tm, count, face_weight=weights_np, seed=0)[1]

    weights_wp = wp.array(weights_np, dtype=wp.float32, device=mesh_wp.points.device)
    face_idx_wp = tw.sample_surface(
        mesh_wp.points, mesh_wp.indices, count, face_weight=weights_wp, seed=0
    )[1]

    n_faces = int(mesh_tm.faces.shape[0])
    freq_wp = np.bincount(face_idx_wp.numpy(), minlength=n_faces) / count
    freq_tm = np.bincount(face_idx_tm, minlength=n_faces) / count

    freq_expected = weights_np / weights_np.sum()
    assert np.allclose(freq_wp, freq_expected, rtol=0.07, atol=0.01)
    assert np.allclose(freq_tm, freq_expected, rtol=0.07, atol=0.01)
