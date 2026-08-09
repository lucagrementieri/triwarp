"""
Benchmarks for ``triwarp.totals``: the whole-mesh reductions, which are readback-bound.

Every group here ends in a host-side value, so each pays at least one device-to-host crossing on top
of its device pass -- and at the sizes in the scan sweep that latency is a large share of the total.
That is the axis these rows measure: not the arithmetic, which is trivially parallel per face, but
how many crossings the *shape of the return type* forces. Read ``surface_centroid`` (two crossings)
against ``moments`` (four, for ten sums) -- the gap is nearly all latency.

What this axis is **not** is a licence to copy arrays. ``moments`` used to read its three ``vec3d``
integrand buffers back with ``.numpy().sum(axis=0)``, which is not a crossing per return value but
72 bytes *per face* moved to be added on the host, and it is why this module's docstring used to
record triwarp losing the row to igl outright. Four ``wp.utils.array_sum`` calls (it reduces a
``vec3d`` array componentwise, so no kernel was needed) leave the count of crossings unchanged at
four and the bytes crossed at 80: measured back-to-back in one process, ``moments`` went
**3.18 ms -> 0.376 ms on bunny (8.5x)** and **35.2 -> 0.84 on dragon (42x)**, which turns the row
into a win -- min-of-10 **443 us against igl's 806** on bunny and **5.32 ms against 9.04** on
dragon. It still loses ``bunny_decimated`` (367 us against 180), where four launches plus four
4-byte reads are the floor and igl's 16 k faces fit in cache. Read the **min** in this group, not
the median: with the igl and trimesh rows sharing the process the triwarp medians here run up to
18x its own min (measured 8 062 us against a 443 us min on bunny), the same instability
[`test_holes.py`](test_holes.py) documents for its DP rows.

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
    so: all three returns are host-side values, so four device reductions are followed by four
    crossings that no amount of kernel work amortises. Compare it against ``centroid`` above, which
    pays two -- the gap is what the extra quantities cost, and it is nearly all latency.

    ``igl.moments`` returns the first moment un-normalised and the inertia already about the centre
    of mass; ``trimesh``'s ``mass_properties`` computes the same three from the same integrals on
    the host. Both are timed on the whole call, since neither exposes the integrals separately.

    There is no open3d row, measured rather than assumed: ``get_volume`` validates before it
    integrates, and the validation is the same brute-force ``IsWatertight`` composition its
    ``is_watertight`` row times -- **13.8 s on a watertight 82k-face sphere** whose integral is
    microseconds. A row here would re-time ``is_watertight`` under this group's name (the same
    trap the validation module documents for ``is_volume``), and it raises outright on the
    non-watertight sweep meshes.

    **This row used to be a loss and no longer is**, and the reason is worth keeping: the old
    3.19 ms on ``bunny`` against igl's 1.23 was read as confirming the readback account, when four
    crossings of 80 bytes cannot cost 3 ms. Three of them were ``.numpy().sum(axis=0)`` over the
    per-face ``vec3d`` integrands -- 72 bytes per face, not per return -- so the row was timing a
    5 MB copy that the latency story made look expected. Now **443 us against igl's 806** on
    ``bunny`` and **5.32 ms against 9.04** on ``dragon`` (mins; see the module docstring on why the
    medians here are unusable), with ``bunny_decimated`` still lost at 367 us against 180 because
    four launches plus four 4-byte reads are the floor.

    The lesson the group is kept for: **a plausible cost model is not a measurement.** "Readback-
    bound" was true and still hid a bytes-moved bug for as long as nobody checked which of the two
    the number was. A caller wanting only the volume should still call
    [`volume`][triwarp.totals.volume], which pays one crossing.
    """
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        volume, _center, inertia = bench_case.run(lambda: tw.totals.moments(vertices, faces))
        assert np.isfinite(volume)
        assert np.asarray(inertia).reshape(3, 3).shape == (3, 3)
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
