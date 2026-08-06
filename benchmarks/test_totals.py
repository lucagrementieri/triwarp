"""
Benchmarks for ``triwarp.totals``: the whole-mesh reductions, which are readback-bound.

Every group here ends in a host-side value, so each pays at least one device-to-host crossing on top
of its device pass -- and at the sizes in the scan sweep that latency is a large share of the total.
That is the axis these rows measure: not the arithmetic, which is trivially parallel per face, but
how many crossings the *shape of the return type* forces. Read ``surface_centroid`` (two crossings)
against ``moments`` (three, for ten sums) -- the gap is nearly all latency, and it is why triwarp
*loses* the ``moments`` row to igl outright.

``get_geometric_measures`` is pymeshlab's reference for the centroid and it does *more*: one
read-only call returns ``shell_barycenter`` (the area-weighted centroid triwarp computes),
``barycenter`` (the plain vertex mean), the surface area, the mesh volume, the average edge length
and the inertia tensor. So that row is an **upper** bound on the centroid alone -- and the same
number appears as the ``mean_edge_length`` reference in [`test_edges.py`](test_edges.py), which is
worth knowing before reading either as a per-quantity cost. It is read-only, so it shares the
MeshSet.

``volume`` and ``euler_characteristic`` have **no groups**; see the known-gaps table in
``README.md``.
"""

from __future__ import annotations

import igl
import numpy as np
import pytest
import trimesh as tm
from conftest import BenchCase, skip_larger_than

import triwarp as tw


@pytest.mark.benchmark(group="surface_centroid")
@pytest.mark.benchlibs("triwarp", "trimesh", "pymeshlab")
def test_surface_centroid(bench_case: BenchCase) -> None:
    """
    The area-weighted shell centroid: one pass over the faces plus two host readbacks.

    Two reductions and therefore two crossings, which is the floor for a function whose return
    type is a host-side ``wp.vec3`` -- and the baseline ``moments`` below is read against, since
    that one pays three for ten sums. Also the row that covers *both* reduction kernels:
    ``centroid_tiled`` on CUDA and ``centroid_sliced`` on CPU, picked by
    ``_device.prefers_tiled_reduction``, measured 1.67x apart at 327k faces.
    """
    if bench_case.kind == "pymeshlab":
        # ``get_geometric_measures``' ``barycenter`` is the vertex mean and ``shell_barycenter`` the
        # area-weighted centroid triwarp computes; the call returns both plus the area, volume and
        # inertia tensor, so it is an upper bound rather than an equivalent. Capped at ``bunny``:
        # it costs 1.12 s a call on ``dragon``, for a ratio the two medium meshes already establish.
        skip_larger_than(bench_case, "bunny", "get_geometric_measures is 1.12 s a call on dragon")
        meshset_pml = bench_case.meshset_pml
        assert bench_case.run(meshset_pml.get_geometric_measures)["shell_barycenter"].shape == (3,)
        return
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        result = bench_case.run(lambda: tw.totals.surface_centroid(vertices, faces))
        assert np.isfinite(list(result)).all()
    else:  # numpy reference: the uncached formula behind ``trimesh.Trimesh.centroid``
        vertices, faces = bench_case.vertices_np, bench_case.faces_np

        def run() -> np.ndarray:
            triangles = vertices[faces]
            crosses = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            areas = 0.5 * np.linalg.norm(crosses, axis=1)
            return (triangles.mean(axis=1) * areas[:, None]).sum(axis=0) / areas.sum()

        result = bench_case.run(run)
        assert result.shape == (3,)


@pytest.mark.benchmark(group="moments")
@pytest.mark.benchlibs("triwarp", "igl", "trimesh")
def test_moments(bench_case: BenchCase) -> None:
    """
    Volume, centre of mass and inertia tensor: ten ``float64`` sums over the faces.

    The one row in this module that is **readback-bound rather than kernel-bound**, and deliberately
    so: all three returns are host-side values, so four device reductions are followed by three
    crossings that no amount of kernel work amortises. Compare it against ``centroid`` above, which
    pays two -- the gap is what the extra quantities cost, and it is nearly all latency.

    ``igl.moments`` returns the first moment un-normalised and the inertia already about the centre
    of mass; ``trimesh``'s ``mass_properties`` computes the same three from the same integrals on
    the host. Both are timed on the whole call, since neither exposes the integrals separately.

    **And triwarp loses this one**, which the readback account predicts and the numbers confirm: on
    ``bunny`` it reads **3.19 ms against igl's 1.23** (and trimesh's 41.1), because ten ``float64``
    sums over 69 451 faces is less work than three host crossings cost in latency. It is the
    clearest case in the suite of a row where the *shape of the API* -- three host-side scalars --
    sets the cost, not the arithmetic. A caller wanting only the volume should call
    [`volume`][triwarp.totals.volume], which pays one.
    """
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        volume, _center, inertia = bench_case.run(lambda: tw.totals.moments(vertices, faces))
        assert np.isfinite(volume)
        assert inertia.shape == (3, 3)
        return
    vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
    if bench_case.kind == "igl":
        volume_igl, first_igl, inertia_igl = bench_case.run(
            lambda: igl.moments(vertices_np, faces_np)
        )
        assert np.isfinite(volume_igl)
        assert np.asarray(first_igl).shape == (3,)
        assert np.asarray(inertia_igl).shape == (3, 3)
        return
    properties_tm = bench_case.run(
        lambda: tm.Trimesh(vertices_np, faces_np, process=False).mass_properties
    )
    assert np.asarray(properties_tm["inertia"]).shape == (3, 3)
