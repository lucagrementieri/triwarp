"""
Benchmark for the differentiable Chamfer loss (forward + backward under ``wp.Tape``).

Mesh B is the same mesh translated by 5% of its bbox diagonal (untimed setup). Both vertex
buffers carry ``requires_grad=True``; each timed round records a fresh tape, runs backward and
zeroes the gradients, which is the real optimization-loop cost.

No open3d case: its ``compute_point_cloud_distance`` covers the *forward* nearest-neighbour distance
only, and open3d has no autodiff, so it cannot produce the backward pass that dominates this
measurement. Timing forward-only against forward-plus-backward would be a misleading ratio rather
than a useful baseline.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import warp as wp
from conftest import BenchCase, skip_larger_than

import triwarp as tw
import triwarp.typing as twt

_grad_cache: dict[tuple[str, str], tuple] = {}


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
