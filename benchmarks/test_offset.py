"""
Benchmarks for ``triwarp.offset``.

One group, and its axis is the **lattice**, not the mesh. A level-set offset samples a signed
distance at every point of a regular grid and marches it, so its cost is
``resolution ** 3`` closest-point queries plus one marching-cubes pass -- the input's face count
enters only through the BVH descent each query pays. That is why the cell width is the parametrized
axis here and the mesh sweep is along for the ride: doubling the resolution is 8x the queries
whatever the mesh, and a row whose slope is not ~8x is telling you the BVH, not the lattice, is the
cost.

References
----------
**meshlib** ``offsetMesh`` is the same algorithm: an OpenVDB level set at a given ``voxelSize``,
marched back to a mesh. It is the fair row -- both sides sample a field on a lattice of the same
pitch, and ``tests/test_offset.py`` pins them to within half a cell of each other on the surface and
to within 5 % on the vertex count.

**pymeshlab** ``generate_resampled_uniform_mesh`` is MeshLab's offset and is timed at the same cell
size. Its ``offset`` parameter is passed as ``PureValue``, which is mandatory rather than stylistic:
as a ``PercentageValue`` it runs from full erosion at 0 % to full dilation at 100 %, so its own
default is the *zero* offset (CLAUDE.md section 6).

open3d has no offset, and neither does trimesh or igl: a level-set offset needs a signed distance
field on a lattice, and of the six CPU references only these two build one.
"""

from __future__ import annotations

import numpy as np
import pymeshlab as ml
import pytest
import warp as wp
from meshlib import mrmeshpy as mm

import triwarp as tw
from conftest import BenchCase, skip_larger_than

# Cell widths as a fraction of the bounding-box diagonal, and the offset distance as a multiple of
# the cell. The pair is a slope check: 1/64 to 1/128 is 8x the samples, and an offset of four cells
# is wide enough that the padding (``ceil(distance / cell) + 2``) is a real part of the lattice.
_CELL_DIVISORS = [64, 128]
_OFFSET_CELLS = 4.0


def _cell_and_offset(bench_case: BenchCase, divisor: int) -> tuple[float, float]:
    """Absolute cell width from the mesh's own bbox diagonal, and the offset that goes with it."""
    vertices_np = bench_case.vertices_np
    diagonal = float(np.linalg.norm(vertices_np.max(axis=0) - vertices_np.min(axis=0)))
    cell = diagonal / divisor
    return cell, _OFFSET_CELLS * cell


@pytest.mark.benchmark(group="offset_mesh")
@pytest.mark.benchlibs("triwarp", "meshlib", "pymeshlab")
@pytest.mark.parametrize("divisor", _CELL_DIVISORS)
def test_offset_mesh(bench_case: BenchCase, divisor: int) -> None:
    """
    Outward level-set offset at a matched cell width: field sampling plus one marching-cubes pass.

    Every side gets the identical absolute cell and the identical absolute offset, so the row is a
    comparison of two field-and-march implementations rather than of two parameter conventions. The
    lattice includes the padding the offset needs -- ``ceil(distance / cell) + 2`` cells on every
    side, or the level set is clipped by the boundary -- which at four cells of offset is a
    noticeable fraction of the box and is paid by triwarp's row alone (both references pad
    internally).

    Read the two divisors as a slope. 2x the resolution is **8x the samples**, and triwarp's row
    grows only **2.8x** -- the samples are independent queries, so a lattice this size does not
    saturate the GPU and the wall clock tracks occupancy rather than work. The reference rows grow
    2.2x (meshlib) and 1.6x (pymeshlab) for the same reason on their own cores.

    First measurement, medians on an RTX 5090, ``bunny`` at 1/64 and 1/128 of the diagonal:

    | | 1/64 | 1/128 |
    |---|---|---|
    | triwarp-cuda | **4.94 ms** | **13.99 ms** |
    | meshlib | 50.5 (10.2x) | 109.8 (7.8x) |
    | pymeshlab | 1 206 (244x) | 1 875 (134x) |

    Read meshlib's *median*, not its mean: its OpenVDB band build has a 49-347 ms spread here.
    """
    cell, distance = _cell_and_offset(bench_case, divisor)

    if bench_case.kind == "meshlib":
        skip_larger_than(bench_case, "bunny", "OpenVDB's level set is a single-threaded band build")
        mesh_ml = bench_case.new_mesh_ml()  # held in a name: MeshPart does not own it
        part_ml = mm.MeshPart(mesh_ml)
        parameters_ml = mm.OffsetParameters()
        parameters_ml.voxelSize = cell
        offset_ml = bench_case.run(lambda: mm.offsetMesh(part_ml, distance, parameters_ml))
        assert offset_ml.topology.numValidFaces() > 0
        return

    if bench_case.kind == "pymeshlab":
        skip_larger_than(bench_case, "bunny", "MeshLab's resampler is a serial dense pass")
        cell_pml = ml.PureValue(cell)
        offset_pml = ml.PureValue(distance)

        def resample_pml() -> int:
            meshset_pml = bench_case.new_meshset_pml()
            meshset_pml.generate_resampled_uniform_mesh(
                cellsize=cell_pml, offset=offset_pml, mergeclosevert=True
            )
            return meshset_pml.current_mesh().face_number()

        assert bench_case.run(resample_pml) > 0
        return

    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    offset_vertices, offset_faces = bench_case.run(
        lambda: tw.offset.offset_mesh(vertices, faces, distance, cell)
    )
    assert int(offset_faces.shape[0]) > 0
    assert int(offset_vertices.shape[0]) > 0


@pytest.mark.benchmark(group="thicken_mesh")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "meshlib")
def test_thicken_mesh(bench_case: BenchCase) -> None:
    """
    The topology-preserving shell: two vertex passes and one band, no lattice anywhere.

    Read this against ``offset_mesh`` above -- the two answer the same question and their costs are
    not comparable in kind. This one is linear in the *mesh* (normals, two displacements, a rim
    walk) and produces ``2F + 2E`` triangles; the level-set offset is cubic in the *lattice* and
    produces whatever the marching finds. So the interesting number is the ratio, and it is the
    argument for having both.

    meshlib's ``makeThickMesh`` is the same construction, and ``tests/test_offset.py`` shows just how
    same: identical counts, volumes equal to five decimals, and identical face sets under a vertex
    bijection. That makes this the cleanest row in the module -- two implementations of one
    algorithm, no parameter mapping in between. Its ``ThickenParams`` is built outside the timed
    callable and its mesh is fresh per round, since the call mutates nothing but the params object is
    an input.

    The thickness is 1 % of the bounding-box diagonal, which is small enough that neither side folds
    (the shell would self-intersect past the local curvature radius, and a shell that folds is not
    the same amount of work).

    First measurement, medians on an RTX 5090, and the reference is **capped at ``sphere_med``**
    because of what it shows: triwarp-cuda **0.78 / 0.70 / 1.03 ms** at ``sphere_small`` /
    ``sphere_med`` / ``sphere_large`` -- flat, because at these sizes it is three launches against
    the launch floor -- against meshlib's **6.5 ms / 266 ms / 4.21 s**. That is 8x, 382x and 4 000x,
    and the slope is the story rather than any single ratio: the reference is superlinear where this
    is flat, which for two implementations of the same construction means the cost is not the
    construction. Whatever else ``makeThickMesh`` does at 1 M faces, it is not what its own 6.5 ms at
    2 562 predicts.
    """
    thickness = 0.01 * float(
        np.linalg.norm(bench_case.vertices_np.max(axis=0) - bench_case.vertices_np.min(axis=0))
    )
    if bench_case.kind == "meshlib":
        # ``skip_larger_than`` is a no-op on the synthetic feature meshes, so the cap is by name,
        # as ``test_heat_distance.py``'s serial rows are.
        if bench_case.mesh_name == "sphere_large":
            pytest.skip("makeThickMesh is 4.2 s at 1 M faces: capped at sphere_med")
        parameters_ml = mm.ThickenParams()
        parameters_ml.insideOffset = thickness
        parameters_ml.outsideOffset = 0.0
        shell_ml = bench_case.run(
            lambda mesh: mm.makeThickMesh(mesh, parameters_ml), setup=bench_case.new_mesh_ml
        )
        assert shell_ml.topology.numValidFaces() > 0
        return

    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    shell_vertices, shell_faces = bench_case.run(
        lambda: tw.offset.thicken_mesh(vertices, faces, thickness)
    )
    assert int(shell_vertices.shape[0]) == 2 * bench_case.n_vertices
    assert int(shell_faces.shape[0]) >= 6 * bench_case.n_faces


@pytest.mark.benchmark(group="signed_distance_grid")
@pytest.mark.benchlibs("triwarp", "open3d")
@pytest.mark.parametrize("divisor", _CELL_DIVISORS)
def test_signed_distance_grid(bench_case: BenchCase, divisor: int) -> None:
    """
    The field alone, without the marching: ``resolution ** 3`` signed closest-point queries.

    Subtract this from ``offset_mesh`` above to price the extraction, which is the only reason both
    groups exist -- they run the identical lattice, so the difference is Warp's ``MarchingCubes``
    over it.

    open3d's ``RaycastingScene.compute_signed_distance`` is the same quantity, to 1.27e-07
    (``tests/test_proximity.py``), and it is the one reference here that batches: the lattice is
    built once outside the timed callable on both sides, so each row prices the queries. It is taken
    *from the function under test* rather than re-derived, so neither side gets a different lattice.

    First measurement, medians on an RTX 5090, ``bunny``: triwarp-cuda **4.48 / 8.53 ms** against
    open3d's **25.7 / 171.1** at 1/64 and 1/128, i.e. 5.7x and 20x. The two slopes are the whole
    row: open3d's 6.7x for 8x the samples is a saturated CPU doing the work, triwarp's 1.9x is a GPU
    that was not full at the coarser lattice. Subtracting these from ``offset_mesh`` prices the
    extraction: **0.28 ms** of marching at 1/64 against **5.3 ms** at 1/128, so past a certain
    resolution the offset is dominated by ``MarchingCubes`` rather than by the field.
    """
    cell, _distance = _cell_and_offset(bench_case, divisor)

    if bench_case.kind == "open3d":
        import open3d as o3d

        skip_larger_than(bench_case, "bunny", "the reference is a single-threaded BVH walk")
        # The lattice has to be the *same* lattice, so it is taken from the function under test
        # rather than re-derived here -- run once, untimed, on host copies, since ``vertices_wp`` is
        # a triwarp-only accessor and this row has no device.
        vertices_cpu = wp.array(
            np.ascontiguousarray(bench_case.vertices_np, dtype=np.float32),
            dtype=wp.vec3,
            device="cpu",
        )
        faces_cpu = wp.array(
            np.ascontiguousarray(bench_case.faces_np, dtype=np.int32).reshape(-1),
            dtype=wp.int32,
            device="cpu",
        )
        field_wp, box = tw.proximity.signed_distance_grid(vertices_cpu, faces_cpu, cell)
        samples_np = tw.voxels.grid_points(field_wp.shape, bounds=box, device="cpu").numpy()
        scene_o3d = o3d.t.geometry.RaycastingScene()
        scene_o3d.add_triangles(
            o3d.core.Tensor(np.ascontiguousarray(bench_case.vertices_np, dtype=np.float32)),
            o3d.core.Tensor(np.ascontiguousarray(bench_case.faces_np, dtype=np.uint32)),
        )
        queries_o3d = o3d.core.Tensor(samples_np.astype(np.float32))
        distance_o3d = bench_case.run(lambda: scene_o3d.compute_signed_distance(queries_o3d))
        assert distance_o3d.shape[0] == samples_np.shape[0]
        return

    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    field, box = bench_case.run(lambda: tw.proximity.signed_distance_grid(vertices, faces, cell))
    assert field.shape[0] >= 2
    assert float(box[1][0]) > float(box[0][0])
