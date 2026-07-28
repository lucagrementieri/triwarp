"""
Benchmarks for ``triwarp.distance``: the differentiable Chamfer loss and the plain metrics.

Cloud/mesh B is the same mesh translated by 5% of its bbox diagonal (untimed setup), so both
directions of every symmetric metric do real work.

Differentiable case
-------------------
``chamfer_mesh_to_mesh_loss`` carries ``requires_grad=True`` on both vertex buffers; each timed
round records a fresh tape, runs backward and zeroes the gradients, which is the real
optimization-loop cost. It is triwarp-only: open3d has no autodiff, so timing forward-only against
forward-plus-backward would be a misleading ratio rather than a useful baseline.

Non-differentiable cases
------------------------
``chamfer_points_to_points`` and ``hausdorff_points_to_points`` are each **two**
[`query_hashgrid_nearest`][triwarp.neighbors.query_hashgrid_nearest] calls at ``k=1`` plus a
reduction, so they are the direct measurement for a change to the k-NN kernel — the same kernel
that sits under all twelve of ``distance.py``'s nearest-neighbour call sites.

**open3d** is the reference: ``PointCloud.compute_point_cloud_distance`` returns exactly the forward
nearest-neighbour Euclidean distances (a serial ``KDTreeFlann`` search), from which both metrics
follow — Chamfer as the mean of the squares in each direction (the pytorch3d convention triwarp
uses), Hausdorff as the overall maximum. Both baselines therefore run the same two searches triwarp
does, and the host-side ``numpy`` reduction over the returned vector is inside the timed region
because open3d has no device-side equivalent to hide it behind.

trimesh and libigl have no point-cloud Chamfer/Hausdorff entry point (``igl.hausdorff`` is
mesh-to-mesh only and is already the documented reference for
[`hausdorff_mesh_to_mesh`][triwarp.distance.hausdorff_mesh_to_mesh] in ``tests/``), so neither
appears here.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import open3d as o3d
import pytest
import warp as wp
from conftest import BenchCase, skip_larger_than

import triwarp as tw
import triwarp.typing as twt

_TRANSLATION_FRACTION = 0.05

_grad_cache: dict[tuple[str, str], tuple] = {}
_cloud_np_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_cloud_wp_cache: dict[tuple[str, str], tuple] = {}
_cloud_o3d_cache: dict[str, tuple] = {}


def _clouds_np(bench_case: BenchCase) -> tuple[np.ndarray, np.ndarray]:
    """Build the ``(a, b)`` float64 clouds: the mesh vertices and a rigidly translated copy."""
    name = bench_case.mesh_name
    if name not in _cloud_np_cache:
        vertices = bench_case.vertices_np
        offset = _TRANSLATION_FRACTION * np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
        _cloud_np_cache[name] = (vertices, np.ascontiguousarray(vertices + offset))
    return _cloud_np_cache[name]


def _clouds_wp(bench_case: BenchCase) -> tuple[wp.array[wp.vec3], wp.array[wp.vec3]]:
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _cloud_wp_cache:
        cloud_a, cloud_b = _clouds_np(bench_case)
        _cloud_wp_cache[key] = tuple(
            wp.array(
                np.ascontiguousarray(c, dtype=np.float32), dtype=wp.vec3, device=bench_case.device
            )
            for c in (cloud_a, cloud_b)
        )
    return _cloud_wp_cache[key]


def _clouds_o3d(bench_case: BenchCase) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    name = bench_case.mesh_name
    if name not in _cloud_o3d_cache:
        _cloud_o3d_cache[name] = tuple(
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(c)) for c in _clouds_np(bench_case)
        )
    return _cloud_o3d_cache[name]


def _grad_inputs(bench_case: BenchCase) -> tuple:
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _grad_cache:
        vertices = np.ascontiguousarray(bench_case.vertices_np, dtype=np.float32)
        offset = 0.05 * np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
        vertices_a = wp.array(vertices, dtype=wp.vec3, device=bench_case.device, requires_grad=True)
        vertices_b = wp.array(
            vertices + np.float32(offset),
            dtype=wp.vec3,
            device=bench_case.device,
            requires_grad=True,
        )
        _grad_cache[key] = (vertices_a, vertices_b)
    return _grad_cache[key]


@pytest.mark.benchmark(group="chamfer_mesh_to_mesh_loss")
@pytest.mark.benchlibs("triwarp")
def test_chamfer_mesh_to_mesh_loss(bench_case: BenchCase) -> None:
    skip_larger_than(bench_case, "dragon")
    vertices_a, vertices_b = _grad_inputs(bench_case)
    faces = bench_case.faces_wp

    def run() -> twt.Array1dFloat32:
        tape = wp.Tape()
        loss = tw.distance.chamfer_mesh_to_mesh_loss(
            vertices_a, faces, vertices_b, faces, tape=tape
        )
        tape.backward(loss=cast(wp.array, loss))
        tape.zero()
        return loss

    loss = bench_case.run(run)
    assert loss.shape == (1,)


@pytest.mark.benchmark(group="chamfer_points_to_points")
@pytest.mark.benchlibs("triwarp", "open3d")
@pytest.mark.parametrize("single_directional", [True, False], ids=["oneway", "symmetric"])
def test_chamfer_points_to_points(bench_case: BenchCase, single_directional: bool) -> None:
    """
    Point-cloud Chamfer: ``k=1`` searches plus a mean-of-squares reduction.

    ``single_directional=False`` does not merely double the work -- it builds a *second*
    acceleration structure over the other cloud. So the symmetric row should be more than 2x the
    one-way row, and how much more is the build cost, which is otherwise invisible.
    """
    skip_larger_than(bench_case, "dragon")
    if bench_case.kind == "triwarp":
        cloud_a, cloud_b = _clouds_wp(bench_case)
        chamfer = bench_case.run(
            lambda: tw.distance.chamfer_points_to_points(
                cloud_a, cloud_b, single_directional=single_directional
            )
        )
    elif single_directional:
        cloud_a, cloud_b = _clouds_o3d(bench_case)
        chamfer = bench_case.run(
            lambda: float(
                np.square(np.asarray(cloud_a.compute_point_cloud_distance(cloud_b))).mean()
            )
        )
    else:
        cloud_a, cloud_b = _clouds_o3d(bench_case)
        chamfer = bench_case.run(
            lambda: float(
                np.square(np.asarray(cloud_a.compute_point_cloud_distance(cloud_b))).mean()
                + np.square(np.asarray(cloud_b.compute_point_cloud_distance(cloud_a))).mean()
            )
        )
    assert chamfer > 0.0


@pytest.mark.benchmark(group="hausdorff_points_to_points")
@pytest.mark.benchlibs("triwarp", "open3d")
def test_hausdorff_points_to_points(bench_case: BenchCase) -> None:
    """Symmetric point-cloud Hausdorff: the same two searches, reduced with ``max`` instead."""
    skip_larger_than(bench_case, "dragon")
    if bench_case.kind == "triwarp":
        cloud_a, cloud_b = _clouds_wp(bench_case)
        hausdorff = bench_case.run(lambda: tw.distance.hausdorff_points_to_points(cloud_a, cloud_b))
    else:
        cloud_a, cloud_b = _clouds_o3d(bench_case)
        hausdorff = bench_case.run(
            lambda: max(
                float(np.asarray(cloud_a.compute_point_cloud_distance(cloud_b)).max()),
                float(np.asarray(cloud_b.compute_point_cloud_distance(cloud_a)).max()),
            )
        )
    assert hausdorff > 0.0


@pytest.mark.benchmark(group="chamfer_points_to_mesh")
@pytest.mark.benchlibs("triwarp")
def test_chamfer_points_to_mesh(bench_case: BenchCase) -> None:
    """
    Cloud-to-surface Chamfer: an exact mesh query forward, a ``k=1`` cloud search backward.

    triwarp-only. open3d's closest-point-on-surface equivalent lives in the *tensor* API
    (``o3d.t.geometry.RaycastingScene.compute_distance``), which is a different implementation from
    the legacy ``o3d.geometry`` baselines the rest of this suite uses; mixing the two in one table
    would compare implementations, not libraries.
    """
    skip_larger_than(bench_case, "dragon")
    cloud = _clouds_wp(bench_case)[1]
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    chamfer = bench_case.run(lambda: tw.distance.chamfer_points_to_mesh(cloud, vertices, faces))
    assert chamfer > 0.0
