"""
Benchmarks for ``triwarp.adjacency``: the face-pair table and the two quantities derived from it.

``face_adjacency`` is the package's most reused topological query and the widest composition in it:
an edge build, a row hash, a radix sort over ``3F`` keys, a scan-and-compact, and a gather. Two of
those steps read back to the host -- the compaction's output size, which is data-dependent and
unavoidable, and the hash radix, which is *not* when the caller knows the vertex count. That second
one is the ``n_vertices`` axis below: same answer, one less serialisation point, and the gap
between the two ids is what that keyword buys.

``face_adjacency_unshared`` and ``face_adjacency_angles`` are each one launch over an existing
adjacency table, so they are timed with the table precomputed. Timing them from raw faces would
just re-measure ``face_adjacency`` three times.

trimesh is the host reference for all three. Its ``Trimesh`` properties are cached, so each round
builds a fresh mesh; ``benchmarks/test_mesh.py`` covers the warm-cache side under
``mesh_face_adjacency`` and is deliberately a separate group.
"""

from __future__ import annotations

import pytest
import trimesh as tm
from conftest import BenchCase

import triwarp as tw

_adjacency_cache: dict[tuple[str, str], tuple] = {}


def _adjacency(bench_case: BenchCase) -> tuple:
    """Precomputed ``(face_adjacency, face_adjacency_edges)`` for the derived-quantity groups."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _adjacency_cache:
        _adjacency_cache[key] = tw.adjacency.face_adjacency(
            bench_case.faces_wp, return_edges=True, n_vertices=bench_case.n_vertices
        )
    return _adjacency_cache[key]


@pytest.mark.benchmark(group="face_adjacency")
@pytest.mark.benchlibs("triwarp", "trimesh")
@pytest.mark.parametrize("known_radix", [False, True], ids=["inferred", "known_nv"])
def test_face_adjacency(bench_case: BenchCase, known_radix: bool) -> None:
    """
    Manifold face pairs, with the hash radix inferred against supplied.

    ``inferred`` pays a ``reduce.minmax`` over the edge rows and the host readback that ends it;
    ``known_nv`` passes the vertex count instead and skips both. The answer is identical either
    way, so the whole gap is that one serialisation point -- if it does not show, the keyword is
    not worth threading through callers.
    """
    if bench_case.kind == "triwarp":
        faces_wp = bench_case.faces_wp
        n_vertices = bench_case.n_vertices if known_radix else None
        adjacency = bench_case.run(
            lambda: tw.adjacency.face_adjacency(faces_wp, n_vertices=n_vertices)
        )
        assert adjacency.shape[1] == 2
        assert adjacency.shape[0] > 0
        return
    if known_radix:
        pytest.skip("trimesh has no radix-hint equivalent")
    vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
    adjacency_tm = bench_case.run(
        lambda: tm.Trimesh(vertices_np, faces_np, process=False).face_adjacency
    )
    assert len(adjacency_tm) > 0


@pytest.mark.benchmark(group="face_adjacency_unshared")
@pytest.mark.benchlibs("triwarp", "trimesh")
@pytest.mark.parametrize("tabled", [True, False], ids=["tabled", "from_faces"])
def test_face_adjacency_unshared(bench_case: BenchCase, tabled: bool) -> None:
    """
    The off-edge corner of each adjacent face, with and without the adjacency tables.

    ``tabled`` is one launch over a precomputed ``(face_adjacency, face_adjacency_edges)`` pair and
    is the floor for this operation. ``from_faces`` starts from the face buffer alone, which is what
    a caller who does not already hold those tables pays: the grouping is unavoidable, but the
    tables themselves are not -- the shared edge and both owning faces are recoverable from the
    grouped edge indices. The gap between this id and ``face_adjacency`` + ``tabled`` is what that
    saves.
    """
    if bench_case.kind == "triwarp":
        faces_wp = bench_case.faces_wp
        if tabled:
            adjacency, adjacency_edges = _adjacency(bench_case)
            unshared = bench_case.run(
                lambda: tw.adjacency.face_adjacency_unshared(
                    faces_wp, face_adjacency=adjacency, face_adjacency_edges=adjacency_edges
                )
            )
        else:
            n_vertices = bench_case.n_vertices
            unshared = bench_case.run(
                lambda: tw.adjacency.face_adjacency_unshared(faces_wp, n_vertices=n_vertices)
            )
        assert unshared.shape[1] == 2
        return
    if not tabled:
        pytest.skip("trimesh always builds the adjacency tables as cached properties")
    vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
    unshared_tm = bench_case.run(
        lambda: tm.Trimesh(vertices_np, faces_np, process=False).face_adjacency_unshared
    )
    assert len(unshared_tm) > 0


@pytest.mark.benchmark(group="face_adjacency_angles")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_face_adjacency_angles(bench_case: BenchCase) -> None:
    """Dihedral angle per adjacent pair, from a precomputed adjacency table and fresh normals."""
    if bench_case.kind == "triwarp":
        vertices_wp, faces_wp = bench_case.vertices_wp, bench_case.faces_wp
        adjacency, _ = _adjacency(bench_case)
        angles = bench_case.run(
            lambda: tw.adjacency.face_adjacency_angles(
                vertices_wp, faces_wp, face_adjacency=adjacency
            )
        )
        assert angles.shape[0] > 0
        return
    vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
    angles_tm = bench_case.run(
        lambda: tm.Trimesh(vertices_np, faces_np, process=False).face_adjacency_angles
    )
    assert len(angles_tm) > 0
