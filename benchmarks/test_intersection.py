"""
Benchmarks for ``triwarp.intersection``.

Four functions on two different cost shapes:

* ``mesh_with_plane`` / ``slice_mesh_with_plane`` / ``clip_mesh_with_field`` — one pass over the
  faces, a per-vertex scalar, then a compaction of the (few) faces the level set actually crosses.
  Memory-bound and dominated by the full-mesh sweep, not by the segment count: a plane meets
  O(sqrt(n_faces)) triangles but every face is still classified. The last two run the *same* engine
  (the plane's signed distance is one such scalar), so their triwarp rows should track each other —
  measured 794 vs 859 µs on ``bunny`` — and a divergence means the shared path changed under one of
  them. ``clip_mesh_with_field``'s ``cap=True`` case is a different shape: the ``O(B^3)`` min-weight
  fill of the section loop dominates the clip by 60x (55 ms against 0.86 ms on ``bunny``), so read
  that case as a ``holes.fill_min_weight`` measurement on a long rim.
* ``mesh_with_mesh`` — the only quadratic-ish one. A ``wp.Mesh`` BVH is built over the smaller mesh
  and every triangle of the other supplies an AABB query, so the cost tracks the number of
  *candidate* pairs (capped per query triangle by ``max_triangle_collisions``) rather than the face
  count. This is the benchmark that moves when the separating-axis narrow phase changes.
* ``segments_with_plane`` — a pure ``wp.map`` over independent segments; the array-primitive
  baseline for the module.

References
----------
**trimesh** is the reference for three of the four: ``mesh_plane``, ``slice_faces_plane`` and
``plane_lines``. Its ``Trimesh`` is rebuilt inside the timed callable for ``mesh_plane`` because
``triangles`` / ``face_normals`` are cached properties that would make rounds 2..n measure only the
plane arithmetic. ``slice_faces_plane`` and ``plane_lines`` take raw arrays and need no rebuild.

**mesh_with_mesh has no CPU reference here.** trimesh's mesh-mesh intersection is not in
``trimesh.intersections`` at all — it routes through the optional ``python-fcl`` collision backend,
which reports *whether* pairs collide rather than returning the intersection curve, and is not a
declared dependency. open3d's boolean operations require the (also optional) ``open3d.t`` tensor
backend with a coupled remesh, so neither is an apples-to-apples baseline for "return the
intersection segments". triwarp is timed alone; the before/after delta is what this case is for.

**pymeshlab** has the right filter and cannot run it here. ``generate_polyline_from_planar_section``
does exactly what ``mesh_with_plane`` does and more (it *orders* the segments into a polyline), and
it works on the synthetic meshes -- but it raises ``PyMeshLabException: Failed to apply filter`` on
**every scan mesh**, at any ``planeaxis``, ``planeoffset`` or ``relativeto`` (probed on
``bunny_decimated`` with both ``'Z Axis'`` and ``'Custom Axis'``). That is the same non-manifold
boundary the libigl and potpourri3d references run into elsewhere in the suite, and this module's
groups are all on the scan sweep, so there is nowhere for the row to move. Recorded rather than
skipped, so it is not re-derived.

**libigl** has no plane-section or mesh-mesh intersection binding in the Python package, so it is
absent from the ``mesh_with_*`` and ``segments_with_plane`` groups. It *does* have the isocontour
operation, though -- an earlier version of this docstring claimed ``igl.ray_mesh_intersect`` was its
only intersection entry point, which was wrong: ``igl.isolines(V, F, S, vals)`` is precisely what
``marching_triangles`` computes, and it is a row in both of that function's groups.

**pyvista** (VTK 9.6) answers ``clip_mesh_with_field``'s uncapped case through
``PolyData.clip_scalar``, over the identical per-vertex field so the two do the same work.
``invert=False`` is passed explicitly: its default keeps the side *below* the value. Its capped
counterpart ``clip_closed_surface`` cannot be timed here at all — it validates the mesh first and
raises on any open edge, which every scan mesh has; the capped comparison therefore lives in
``tests/test_intersection.py`` on a closed synthetic mesh, and the capped benchmark case is
triwarp-only.

Caps
----
``mesh_with_mesh`` is capped at ``bunny``: the broad phase allocates
``max_triangle_collisions`` candidate slots per query triangle, so the pair buffer alone is
``16 * n_faces`` ints before the narrow phase filters it. ``clip_mesh_with_field``'s pyvista row is
capped there too, VTK's clip being a single-threaded per-cell sweep.

Geometry
--------
Every plane cuts through the middle of the mesh — the origin is the vertex-bounding-box centre and
the normal is a fixed off-axis direction — so the section is a full cross-section rather than a
near-miss that would exit early. ``mesh_with_mesh`` intersects the mesh with a copy of itself
translated by a fraction of its own extent, which guarantees a large, genuinely overlapping
intersection curve on every mesh.
"""

from __future__ import annotations

import igl
import numpy as np
import potpourri3d as pp3d
import pytest
import trimesh as tm
import warp as wp
from conftest import BenchCase, skip_larger_than

import triwarp as tw

# Off-axis so no cut is degenerate w.r.t. the (axis-aligned) scan-mesh geometry.
_PLANE_NORMAL = np.array([0.3, 0.8, 0.5])
_PLANE_NORMAL = _PLANE_NORMAL / np.linalg.norm(_PLANE_NORMAL)

# Self-intersection offsets, as fractions of the mesh bounding-box diagonal. ``mesh_with_mesh``
# costs the number of *actually overlapping* triangle pairs, not the face count, so the offset is
# the axis: a deep overlap intersects a broad band, a grazing one barely touches.
_SELF_OFFSET_FRACTIONS = [0.05, 0.60]

_shifted_cache: dict[tuple[str, str, float], wp.array[wp.vec3]] = {}


def _plane_origin(bench_case: BenchCase) -> np.ndarray:
    """Centre of the vertex bounding box — guarantees the plane cuts the mesh."""
    vertices = bench_case.vertices_np
    return 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))


def _shifted_vertices_wp(bench_case: BenchCase, offset_fraction: float) -> wp.array[wp.vec3]:
    """Translate the mesh's own vertices along the plane normal, to act as the second mesh."""
    key = (bench_case.mesh_name, str(bench_case.device), offset_fraction)
    if key not in _shifted_cache:
        vertices = bench_case.vertices_np
        diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
        shifted = vertices + offset_fraction * diagonal * _PLANE_NORMAL
        _shifted_cache[key] = wp.array(
            np.ascontiguousarray(shifted, dtype=np.float32), dtype=wp.vec3, device=bench_case.device
        )
    return _shifted_cache[key]


@pytest.mark.benchmark(group="mesh_with_plane")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_mesh_with_plane(bench_case: BenchCase) -> None:
    """Cross-section segments of a mid-mesh plane: a full face sweep plus a compaction."""
    origin = _plane_origin(bench_case)
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        normal = wp.vec3(*_PLANE_NORMAL.tolist())
        plane_origin = wp.vec3(*origin.tolist())
        lines = bench_case.run(
            lambda: tw.intersection.mesh_with_plane(vertices, faces, normal, plane_origin)
        )
        assert lines.shape[1] == 2
    else:  # rebuild inside: triangles / face_normals are cached Trimesh properties
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        lines_tm = bench_case.run(
            lambda: tm.intersections.mesh_plane(
                tm.Trimesh(vertices_np, faces_np, process=False), _PLANE_NORMAL, origin
            )
        )
        assert lines_tm.shape[1] == 2


@pytest.mark.benchmark(group="slice_mesh_with_plane")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_slice_mesh_with_plane(bench_case: BenchCase) -> None:
    """Keep the positive-normal half of the mesh: classify every face, then re-triangulate cuts."""
    origin = _plane_origin(bench_case)
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        normal = wp.vec3(*_PLANE_NORMAL.tolist())
        plane_origin = wp.vec3(*origin.tolist())
        new_vertices, new_faces = bench_case.run(
            lambda: tw.intersection.slice_mesh_with_plane(vertices, faces, normal, plane_origin)
        )
        assert int(new_faces.shape[0]) % 3 == 0
        assert new_vertices.shape[0] >= 0
    else:  # takes raw arrays, no Trimesh cache to defeat
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        sliced = bench_case.run(
            lambda: tm.intersections.slice_faces_plane(vertices_np, faces_np, _PLANE_NORMAL, origin)
        )
        assert sliced[1].shape[1] == 3


def _plane_field(bench_case: BenchCase) -> tuple[wp.array[wp.float32], np.ndarray]:
    """Build the cutting plane's signed distance as a per-vertex field, for both sides of a row."""
    vertices_np = bench_case.vertices_np
    field_np = (vertices_np - _plane_origin(bench_case)) @ _PLANE_NORMAL
    return (
        wp.array(
            np.ascontiguousarray(field_np, dtype=np.float32),
            dtype=wp.float32,
            device=bench_case.device,
        ),
        field_np,
    )


@pytest.mark.benchmark(group="clip_mesh_with_field")
@pytest.mark.benchlibs("triwarp", "pyvista")
@pytest.mark.parametrize("cap", [False, True])
def test_clip_mesh_with_field(bench_case: BenchCase, cap: bool) -> None:
    """
    The same classify-and-compact sweep as ``slice_mesh_with_plane``, driven by a per-vertex field.

    Timed against the plane's own signed distance so the work is identical to that group's and the
    two are directly comparable — the field costs one extra buffer read per vertex and saves the
    per-edge dot products. ``cap=True`` adds the rim weld (a position hash over the result) plus the
    ``O(B^3)`` min-weight fill of the section, which is why it is a separate case rather than a flag
    folded into one row: on a scan mesh the section loop is long and the fill, not the clip, is what
    is being measured.

    pyvista's counterpart is ``clip_scalar`` (``invert=False`` — its default keeps the *low* side).
    **The capped case has no reference row**: ``clip_closed_surface`` validates its input first and
    raises ``ValueError: This surface appears to be non-manifold`` on every scan mesh, all of which
    carry open edges, so there is nowhere for the row to move. It *is* compared, on a closed
    synthetic mesh, in ``tests/test_intersection.py``.
    """
    if bench_case.kind == "pyvista":
        if cap:
            pytest.skip("clip_closed_surface rejects a non-manifold input; every scan mesh is one")
        # VTK's clip is a single-threaded per-cell sweep, capped like the suite's other host-bound
        # references.
        skip_larger_than(bench_case, "bunny", "VTK's clip is a single-threaded per-cell sweep")
        mesh_pv = bench_case.mesh_pv
        mesh_pv.point_data["field"] = _plane_field(bench_case)[1]
        clipped_pv = bench_case.run(
            lambda: mesh_pv.clip_scalar(scalars="field", value=0.0, invert=False)
        )
        assert clipped_pv.n_faces > 0
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    field_wp = _plane_field(bench_case)[0]
    _new_vertices, new_faces = bench_case.run(
        lambda: tw.intersection.clip_mesh_with_field(vertices, faces, field_wp, cap=cap)
    )
    assert int(new_faces.shape[0]) % 3 == 0


@pytest.mark.benchmark(group="mesh_with_mesh")
@pytest.mark.benchlibs("triwarp")
@pytest.mark.parametrize("offset_fraction", _SELF_OFFSET_FRACTIONS, ids=["deep", "grazing"])
def test_mesh_with_mesh(bench_case: BenchCase, offset_fraction: float) -> None:
    """
    BVH broad phase plus the separating-axis narrow phase, against a translated self-copy.

    Cost is the number of overlapping triangle *pairs*, so the translation distance is the axis
    rather than the face count. The ``deep`` row shares most of its volume with the original and
    the ``grazing`` row barely touches it; the gap is the collision density, and the fixed
    ``max_triangle_collisions`` cap silently truncates once the broad phase saturates.
    """
    skip_larger_than(bench_case, "bunny", "broad phase allocates 16 candidate slots per triangle")
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    shifted = _shifted_vertices_wp(bench_case, offset_fraction)
    lines = bench_case.run(lambda: tw.intersection.mesh_with_mesh(vertices, faces, shifted, faces))
    assert lines.shape[1] == 2


@pytest.mark.benchmark(group="segments_with_plane")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_segments_with_plane(bench_case: BenchCase) -> None:
    """Batched segment-plane hits over the mesh's directed edges: one ``wp.map``, no adjacency."""
    origin = _plane_origin(bench_case)
    faces_np = bench_case.faces_np
    # One segment per face corner: (v0,v1) of every triangle, so the count scales with the mesh.
    start_np = bench_case.vertices_np[faces_np[:, 0]]
    end_np = bench_case.vertices_np[faces_np[:, 1]]
    if bench_case.kind == "triwarp":
        device = bench_case.device
        start = wp.array(
            np.ascontiguousarray(start_np, dtype=np.float32), dtype=wp.vec3, device=device
        )
        end = wp.array(np.ascontiguousarray(end_np, dtype=np.float32), dtype=wp.vec3, device=device)
        normal = wp.vec3(*_PLANE_NORMAL.tolist())
        plane_origin = wp.vec3(*origin.tolist())
        _points, valid = bench_case.run(
            lambda: tw.intersection.segments_with_plane(start, end, plane_origin, normal)
        )
        assert valid.shape[0] == start_np.shape[0]
    else:
        endpoints = np.stack((start_np, end_np))
        points_tm, valid_tm = bench_case.run(
            lambda: tm.intersections.plane_lines(origin, _PLANE_NORMAL, endpoints)
        )
        assert points_tm.shape[0] == valid_tm.sum()


# --- marching_triangles ---------------------------------------------------------------
_ISOVALUE = 0.1137
# potpourri3d runs into hundreds of milliseconds on the many-curve fields.
_ROUNDS = 3

# Field frequency -> (name, wavenumber). ``plane`` is the single-contour control.
_FIELDS = {"plane": 0, "wave12": 12, "wave40": 40}

_field_cache: dict[tuple[str, str], np.ndarray] = {}
_field_wp_cache: dict[tuple[str, str, str], wp.array] = {}


def _field_np(bench_case: BenchCase, field: str) -> np.ndarray:
    """Scalar field on this mesh, cached: it is an input, not part of the measurement."""
    key = (bench_case.mesh_name, field)
    if key not in _field_cache:
        vertices = bench_case.vertices_np
        wavenumber = _FIELDS[field]
        if wavenumber == 0:
            values = vertices[:, 2]
        else:
            values = (
                np.sin(wavenumber * vertices[:, 0])
                * np.cos(wavenumber * vertices[:, 1])
                * np.sin(wavenumber * vertices[:, 2])
            )
        _field_cache[key] = np.ascontiguousarray(values, dtype=np.float64)
    return _field_cache[key]


def _field_wp(bench_case: BenchCase, field: str) -> wp.array:
    """Return the same field as a device buffer."""
    key = (bench_case.mesh_name, field, str(bench_case.device))
    if key not in _field_wp_cache:
        _field_wp_cache[key] = wp.array(
            _field_np(bench_case, field), dtype=wp.float64, device=bench_case.device
        )
    return _field_wp_cache[key]


def _run_case(bench_case: BenchCase, field: str) -> None:
    """Extract one level set, in triwarp, potpourri3d or libigl."""
    if bench_case.kind == "igl":
        # igl returns a segment *soup* -- (points, segments, segment_values), no curve linkage --
        # so it does strictly less than triwarp and potpourri3d, both of which return linked
        # polylines. Read the row as the floor for the extraction without the linking.
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        values_np = _field_np(bench_case, field)
        points_igl, segments_igl, _values_igl = bench_case.run(
            lambda: igl.isolines(vertices_np, faces_np, values_np, np.array([_ISOVALUE])),
            rounds=_ROUNDS,
        )
        assert points_igl.shape[1] == 3
        assert segments_igl.shape[0] > 0
        return
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        values, n_vertices = _field_wp(bench_case, field), bench_case.n_vertices
        curves, _ = bench_case.run(
            lambda: tw.intersection.marching_triangles(
                vertices, faces, values, _ISOVALUE, n_vertices=n_vertices
            ),
            rounds=_ROUNDS,
        )
        assert len(curves) > 0
    else:
        vertices_np = bench_case.vertices_np
        faces_np = np.ascontiguousarray(bench_case.faces_np, dtype=np.int32)
        values_np = _field_np(bench_case, field)
        curves_pp = bench_case.run(
            lambda: pp3d.marching_triangles(vertices_np, faces_np, values_np, _ISOVALUE),
            rounds=_ROUNDS,
        )
        assert len(curves_pp) > 0


@pytest.mark.benchmark(group="marching_triangles")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "potpourri3d", "igl")
def test_marching_triangles(bench_case: BenchCase) -> None:
    """One long closed contour of a coordinate function, over the clean size sweep."""
    _run_case(bench_case, "plane")


@pytest.mark.benchmark(group="marching_triangles_curves")
@pytest.mark.benchmeshes("sphere_med")
@pytest.mark.benchlibs("triwarp", "potpourri3d", "igl")
@pytest.mark.parametrize("field", list(_FIELDS))
def test_marching_triangles_curves(bench_case: BenchCase, field: str) -> None:
    """One mesh, level sets from 1 to ~1 000 curves, to see whether linking cost shows up."""
    _run_case(bench_case, field)
