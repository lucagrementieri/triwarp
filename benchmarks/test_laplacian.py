"""
Benchmarks for ``triwarp.laplacian``.

Times the operator builders that every solver in the library sits on top of: the per-face
half-cotangent weights, the assembled cotangent stiffness matrix, the row-normalized umbrella
operator and the barycentric lumped mass matrix. Two things make this module worth its own file
even though [`test_parametrization.py`](test_parametrization.py) already times solvers that call
these functions: the assembly is a fixed cost paid by *every* solve, and the per-face weight kernels
are pure arithmetic, so a kernel-level change shows up here undiluted by a conjugate-gradient loop.

``cotmatrix_entries`` and ``cotmatrix_entries_intrinsic`` are the same arithmetic reached two ways —
from vertex positions, or from a caller-supplied ``(n_faces, 3)`` edge-length table. The intrinsic
variant is the cleaner measurement of the weight computation itself (no vertex gather, no
``sqrt``-from-positions), which is why both are timed.

References
----------
**libigl** is the reference for everything with a direct equivalent: ``igl.cotmatrix_entries`` (both
overloads), ``igl.cotmatrix`` and ``igl.massmatrix`` (``MASSMATRIX_TYPE_BARYCENTRIC``, the lumping
triwarp implements). **trimesh** is the reference for the umbrella operator —
``trimesh.smoothing.laplacian_calculation`` builds exactly the row-normalized 1-ring averaging
matrix ``laplacian`` returns, for both ``equal_weight`` settings.

``uniform_laplacian`` has no reference: libigl builds ``A - diag(rowsum(A))`` inline inside
``igl::harmonic`` rather than exposing it, and reassembling it here out of ``igl.adjacency_matrix``
plus scipy would time a hand-rolled composition rather than a library function. It is timed for
triwarp alone.

The operator family (``harmonic_integrated``, ``hessian_energy``, ``curved_hessian_energy``,
``crouzeix_raviart_*``) is igl-referenced throughout and sits on the **scale** axis rather than the
scan sweep: ``igl::crouzeix_raviart_*`` and ``igl::orient_halfedges`` (inside
``curved_hessian_energy``) assume edge-manifold input — igl asserts it, triwarp documents it as
undefined — and the scan meshes are not. Two structural notes on those rows: triwarp assembles
``harmonic_integrated`` by a triplet pass per power instead of ``bsr_mm`` (whose chained triple
product is nondeterministic — issue_report.md), and ``hessian_energy``'s per-vertex work is
quadratic in valence, which is harmless on the uniform-valence spheres but would dominate on
``fan_hub``. Measured (medians, 2026-08-05): every group's margin grows with size, from 1.7-13x at
``sphere_small`` to 35-86x at ``sphere_large`` — except ``crouzeix_raviart_massmatrix``, whose
triwarp side is a flat 0.18-0.21 ms host launch/alloc floor across the axis and therefore *loses*
the ``sphere_small`` point (0.206 ms against igl's 0.150) while winning ``sphere_large`` 72x.

**open3d** has no equivalent for any function in this module. Its smoothing filters
(``filter_smooth_laplacian`` / ``filter_smooth_taubin``, timed in
[`test_smoothing.py`](test_smoothing.py)) build their weights inside the per-iteration C++ loop and
never expose a matrix, and ``open3d.geometry`` has no cotangent or mass matrix at all.

What is inside the timed callable
---------------------------------
Everything the public function does, including the sparse assembly (``bsr_from_triplets`` for
triwarp, the COO→CSC build for igl/trimesh). The trimesh reference constructs its ``tm.Trimesh``
inside the timed region because ``laplacian_calculation`` reads the cached ``vertex_neighbors``
property, which would otherwise make rounds 2..n measure nothing but the sparse assembly.

The edge-length table for the intrinsic variant *is* precomputed and cached (it is the function's
input, not part of its work) — with ``igl.edge_lengths``, so triwarp and libigl consume
bit-identical values.
"""

from __future__ import annotations

import igl
import numpy as np
import potpourri3d as pp3d
import pytest
import trimesh as tm
import trimesh.smoothing as tms
import warp as wp
from conftest import BenchCase

import triwarp as tw
import triwarp.typing as twt

_edge_lengths_np_cache: dict[str, np.ndarray] = {}
_edge_lengths_wp_cache: dict[tuple[str, str], twt.Array2dFloat32] = {}


def _edge_lengths_np(bench_case: BenchCase) -> np.ndarray:
    """``(n_faces, 3)`` opposite-edge lengths from ``igl.edge_lengths``, cached per mesh."""
    name = bench_case.mesh_name
    if name not in _edge_lengths_np_cache:
        _edge_lengths_np_cache[name] = igl.edge_lengths(bench_case.vertices_np, bench_case.faces_np)
    return _edge_lengths_np_cache[name]


def _edge_lengths_wp(bench_case: BenchCase) -> twt.Array2dFloat32:
    """Upload the same table as a float32 ``(n_faces, 3)`` buffer on this case's device."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _edge_lengths_wp_cache:
        _edge_lengths_wp_cache[key] = twt.as_array2d_float32(
            wp.array(
                np.ascontiguousarray(_edge_lengths_np(bench_case), dtype=np.float32),
                dtype=wp.float32,
                device=bench_case.device,
            )
        )
    return _edge_lengths_wp_cache[key]


@pytest.mark.benchmark(group="cotmatrix_entries")
@pytest.mark.benchlibs("triwarp", "igl")
def test_cotmatrix_entries(bench_case: BenchCase) -> None:
    """Per-face half-cotangent weights from vertex positions (``igl::cotmatrix_entries``)."""
    n_faces = int(bench_case.faces_np.shape[0])
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        entries = bench_case.run(lambda: tw.laplacian.cotmatrix_entries(vertices, faces))
        assert entries.shape == (n_faces, 3)
    else:
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        entries_igl = bench_case.run(lambda: igl.cotmatrix_entries(vertices_np, faces_np))
        assert entries_igl.shape == (n_faces, 3)


@pytest.mark.benchmark(group="cotmatrix_entries_intrinsic")
@pytest.mark.benchlibs("triwarp", "igl")
def test_cotmatrix_entries_intrinsic(bench_case: BenchCase) -> None:
    """The same weights from a precomputed edge-length table: the weight arithmetic alone."""
    n_faces = int(bench_case.faces_np.shape[0])
    if bench_case.kind == "triwarp":
        edge_lengths = _edge_lengths_wp(bench_case)
        entries = bench_case.run(lambda: tw.laplacian.cotmatrix_entries_intrinsic(edge_lengths))
        assert entries.shape == (n_faces, 3)
    else:
        edge_lengths_np = _edge_lengths_np(bench_case)
        entries_igl = bench_case.run(lambda: igl.cotmatrix_entries(edge_lengths_np))
        assert entries_igl.shape == (n_faces, 3)


@pytest.mark.benchmark(group="cotmatrix")
@pytest.mark.benchlibs("triwarp", "igl", "potpourri3d")
def test_cotmatrix(bench_case: BenchCase) -> None:
    """Assembled cotangent stiffness matrix: weight kernel plus the sparse build."""
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        matrix = bench_case.run(lambda: tw.laplacian.cotmatrix(vertices, faces))
        assert matrix.nrow == n_vertices
    elif bench_case.kind == "potpourri3d":
        # potpourri3d assembles this one in Python (vectorized numpy into a scipy COO), not in
        # geometry-central, so this row measures a numpy + scipy build rather than C++.
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        matrix_pp = bench_case.run(lambda: pp3d.cotan_laplacian(vertices_np, faces_np))
        assert matrix_pp.shape == (n_vertices, n_vertices)
    else:
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        matrix_igl = bench_case.run(lambda: igl.cotmatrix(vertices_np, faces_np))
        assert matrix_igl.shape == (n_vertices, n_vertices)


@pytest.mark.benchmark(group="connection_laplacian")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp")
def test_connection_laplacian(bench_case: BenchCase) -> None:
    """Assembly only: the same triplets as ``cotmatrix`` with a rotation in every off-diagonal."""
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    matrix = bench_case.run(lambda: tw.laplacian.connection_laplacian(vertices, faces))
    assert matrix.nrow == bench_case.n_vertices


@pytest.mark.benchmark(group="laplacian_uniform")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_laplacian_uniform(bench_case: BenchCase) -> None:
    """Row-normalized 1-ring averaging operator with unit weights."""
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        matrix = bench_case.run(lambda: tw.laplacian.laplacian(vertices, faces, equal_weight=True))
        assert matrix.nrow == n_vertices
    else:  # rebuild inside: laplacian_calculation reads the cached vertex_neighbors property
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        matrix_tm = bench_case.run(
            lambda: tms.laplacian_calculation(
                tm.Trimesh(vertices_np, faces_np, process=False), equal_weight=True
            )
        )
        assert matrix_tm.shape == (n_vertices, n_vertices)


@pytest.mark.benchmark(group="laplacian_inverse_distance")
@pytest.mark.benchlibs("triwarp", "trimesh")
def test_laplacian_inverse_distance(bench_case: BenchCase) -> None:
    """The same operator with inverse-edge-length weights (the geometry-dependent branch)."""
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        matrix = bench_case.run(lambda: tw.laplacian.laplacian(vertices, faces, equal_weight=False))
        assert matrix.nrow == n_vertices
    else:
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        matrix_tm = bench_case.run(
            lambda: tms.laplacian_calculation(
                tm.Trimesh(vertices_np, faces_np, process=False), equal_weight=False
            )
        )
        assert matrix_tm.shape == (n_vertices, n_vertices)


@pytest.mark.benchmark(group="uniform_laplacian")
@pytest.mark.benchlibs("triwarp")
def test_uniform_laplacian(bench_case: BenchCase) -> None:
    """Combinatorial graph Laplacian ``A - diag(deg)`` (no library exposes an equivalent)."""
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    matrix = bench_case.run(lambda: tw.laplacian.uniform_laplacian(vertices, faces))
    assert matrix.nrow == bench_case.n_vertices


@pytest.mark.benchmark(group="mass_matrix_entries")
@pytest.mark.benchaxis("valence")
@pytest.mark.benchlibs("triwarp", "igl", "potpourri3d")
def test_mass_matrix_entries(bench_case: BenchCase) -> None:
    """Barycentric lumped mass per vertex: a scatter-add, so on the valence axis for contention."""
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        mass = bench_case.run(lambda: tw.laplacian.mass_matrix_entries(vertices, faces))
        assert mass.shape == (n_vertices,)
    elif bench_case.kind == "potpourri3d":
        # ``vertex_areas`` is one third of the incident face areas, i.e. exactly this diagonal;
        # it scatters with ``np.bincount`` per corner rather than atomics.
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        mass_pp = bench_case.run(lambda: pp3d.vertex_areas(vertices_np, faces_np))
        assert mass_pp.shape == (n_vertices,)
    else:
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        mass_igl = bench_case.run(
            lambda: igl.massmatrix(
                vertices_np, faces_np, igl.MASSMATRIX_TYPE_BARYCENTRIC
            ).diagonal()
        )
        assert mass_igl.shape == (n_vertices,)


@pytest.mark.benchmark(group="mass_matrix")
@pytest.mark.benchaxis("valence")
@pytest.mark.benchlibs("triwarp", "igl")
def test_mass_matrix(bench_case: BenchCase) -> None:
    """The same diagonal, assembled as a sparse matrix."""
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        matrix = bench_case.run(lambda: tw.laplacian.mass_matrix(vertices, faces))
        assert matrix.nrow == n_vertices
    else:
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        matrix_igl = bench_case.run(
            lambda: igl.massmatrix(vertices_np, faces_np, igl.MASSMATRIX_TYPE_BARYCENTRIC)
        )
        assert matrix_igl.shape == (n_vertices, n_vertices)


@pytest.mark.benchmark(group="robust_laplacian")
@pytest.mark.benchlibs("triwarp", "igl")
def test_robust_laplacian(bench_case: BenchCase) -> None:
    """Mollified edge lengths plus the intrinsic cotangent assembly."""
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        matrix = bench_case.run(lambda: tw.laplacian.robust_laplacian(vertices, faces))
        assert matrix.nrow == n_vertices
    else:  # igl's intrinsic overload, from the same length table (built on the host with numpy)
        triangles_np = bench_case.vertices_np[bench_case.faces_np]
        lengths_np = np.ascontiguousarray(
            np.stack(
                [
                    np.linalg.norm(triangles_np[:, 2] - triangles_np[:, 1], axis=1),
                    np.linalg.norm(triangles_np[:, 0] - triangles_np[:, 2], axis=1),
                    np.linalg.norm(triangles_np[:, 1] - triangles_np[:, 0], axis=1),
                ],
                axis=1,
            ),
            dtype=np.float64,
        )
        faces_np = np.ascontiguousarray(bench_case.faces_np, dtype=np.int64)
        matrix_igl = bench_case.run(lambda: igl.cotmatrix_intrinsic(lengths_np, faces_np))
        assert matrix_igl.shape == (n_vertices, n_vertices)


@pytest.mark.benchmark(group="mollify_intrinsic")
@pytest.mark.benchlibs("triwarp")
def test_mollify_intrinsic(bench_case: BenchCase) -> None:
    """The length table, the max-reduce and the two host readbacks it needs."""
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    lengths, _ = bench_case.run(lambda: tw.laplacian.mollify_intrinsic(vertices, faces))
    assert lengths.shape == (bench_case.n_faces, 3)


_field_cache: dict[tuple[str, str], wp.array] = {}


def _scalar_field_wp(bench_case: BenchCase) -> wp.array[wp.float64]:
    """Return the vertices' own z as a ``float64`` field: an *input*, cached per (mesh, device)."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _field_cache:
        _field_cache[key] = wp.array(
            np.ascontiguousarray(bench_case.vertices_np[:, 2], dtype=np.float64),
            dtype=wp.float64,
            device=bench_case.device,
        )
    return _field_cache[key]


_operator_inputs_np_cache: dict[str, tuple] = {}
_operator_inputs_wp_cache: dict[tuple[str, str], tuple] = {}
_edge_numbering_np_cache: dict[str, tuple] = {}
_edge_numbering_wp_cache: dict[tuple[str, str], tuple] = {}


def _laplacian_and_mass_np(bench_case: BenchCase) -> tuple:
    """Prebuild igl's cotangent Laplacian and barycentric mass, cached per mesh (the inputs)."""
    name = bench_case.mesh_name
    if name not in _operator_inputs_np_cache:
        _operator_inputs_np_cache[name] = (
            igl.cotmatrix(bench_case.vertices_np, bench_case.faces_np),
            igl.massmatrix(
                bench_case.vertices_np, bench_case.faces_np, igl.MASSMATRIX_TYPE_BARYCENTRIC
            ),
        )
    return _operator_inputs_np_cache[name]


def _laplacian_and_mass_wp(bench_case: BenchCase) -> tuple:
    """Triwarp's float64 Laplacian and mass diagonal, cached per (mesh, device)."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _operator_inputs_wp_cache:
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        _operator_inputs_wp_cache[key] = (
            tw.laplacian.cotmatrix(vertices, faces, dtype=wp.float64),
            tw.laplacian.mass_matrix_entries(vertices, faces, dtype=wp.float64),
        )
    return _operator_inputs_wp_cache[key]


@pytest.mark.benchmark(group="harmonic_integrated")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "igl")
def test_harmonic_integrated(bench_case: BenchCase) -> None:
    """
    The biharmonic operator ``L M^-1 L`` from a prebuilt Laplacian and mass (``k = 2``).

    Both sides consume cached, prebuilt inputs — the matrices *are* the function's arguments in
    both APIs — so the row times only the composition: igl's two Eigen sparse products against
    triwarp's counted-and-emitted triplet pass plus one ``bsr_from_triplets``.
    """
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "triwarp":
        laplacian, mass = _laplacian_and_mass_wp(bench_case)
        operator = bench_case.run(lambda: tw.laplacian.harmonic_integrated(laplacian, mass, k=2))
        assert operator.nrow == n_vertices
    else:
        laplacian_igl, mass_igl = _laplacian_and_mass_np(bench_case)
        operator_igl = bench_case.run(
            lambda: igl.harmonic_integrated_from_laplacian_and_mass(laplacian_igl, mass_igl, 2)
        )
        assert operator_igl.shape == (n_vertices, n_vertices)


@pytest.mark.benchmark(group="hessian_energy")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "igl")
def test_hessian_energy(bench_case: BenchCase) -> None:
    """
    The natural-boundary Hessian smoothness energy, geometry to assembled matrix.

    igl materializes the ``(9 F, V)`` stacked Hessian and squares it through Eigen sparse
    products; triwarp contracts the component pairs analytically and emits ``9 * valence^2``
    triplets per vertex. Uniform valence 6 on this axis keeps that quadratic cost benign.
    """
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        operator = bench_case.run(lambda: tw.laplacian.hessian_energy(vertices, faces))
        assert operator.nrow == n_vertices
    else:
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        operator_igl = bench_case.run(lambda: igl.hessian_energy(vertices_np, faces_np))
        assert operator_igl.shape == (n_vertices, n_vertices)


@pytest.mark.benchmark(group="curved_hessian_energy")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "igl")
def test_curved_hessian_energy(bench_case: BenchCase) -> None:
    """
    The Crouzeix-Raviart curved Hessian energy, geometry to assembled matrix.

    igl assembles four ``(2 E, ...)`` CR operators and chains four sparse products; triwarp emits
    each face's sandwiched 6x6 block directly (144 triplets per face) and never materializes an
    edge-indexed matrix.
    """
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        operator = bench_case.run(lambda: tw.laplacian.curved_hessian_energy(vertices, faces))
        assert operator.nrow == n_vertices
    else:
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        operator_igl = bench_case.run(lambda: igl.curved_hessian_energy(vertices_np, faces_np))
        assert operator_igl.shape == (n_vertices, n_vertices)


def _edge_numbering_np(bench_case: BenchCase) -> tuple:
    """Igl's ``unique_edge_map`` numbering, cached per mesh: the CR bindings' explicit input."""
    name = bench_case.mesh_name
    if name not in _edge_numbering_np_cache:
        unique_edge_map = igl.unique_edge_map(bench_case.faces_np)
        _edge_numbering_np_cache[name] = (unique_edge_map[1], unique_edge_map[2].ravel())
    return _edge_numbering_np_cache[name]


def _edge_numbering_wp(bench_case: BenchCase) -> tuple:
    """Triwarp's ``edges_unique`` numbering, cached per (mesh, device)."""
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _edge_numbering_wp_cache:
        _edge_numbering_wp_cache[key] = tw.edges.edges_unique(
            bench_case.faces_wp, n_vertices=bench_case.n_vertices
        )
    return _edge_numbering_wp_cache[key]


@pytest.mark.benchmark(group="crouzeix_raviart_cotmatrix")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "igl")
def test_crouzeix_raviart_cotmatrix(bench_case: BenchCase) -> None:
    """
    Edge-based CR stiffness matrix with the edge numbering prebuilt on both sides.

    ``igl.crouzeix_raviart_cotmatrix`` takes ``(E, EMAP)`` explicitly, so the numbering is the
    function's input, not its work; triwarp is handed its own precomputed ``edges_unique`` pair
    for the same reason. Each side uses its native numbering — the timing is unaffected, and the
    value-level correspondence is pinned in ``tests/test_laplacian.py``.
    """
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        unique_edges, edge_map = _edge_numbering_wp(bench_case)
        matrix = bench_case.run(
            lambda: tw.laplacian.crouzeix_raviart_cotmatrix(
                vertices, faces, unique_edges=unique_edges, edge_map=edge_map
            )
        )
        assert matrix.nrow == unique_edges.shape[0]
    else:
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        edges_igl, edge_map_igl = _edge_numbering_np(bench_case)
        matrix_igl = bench_case.run(
            lambda: igl.crouzeix_raviart_cotmatrix(vertices_np, faces_np, edges_igl, edge_map_igl)
        )
        assert matrix_igl.shape == (len(edges_igl), len(edges_igl))


@pytest.mark.benchmark(group="crouzeix_raviart_massmatrix")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "igl")
def test_crouzeix_raviart_massmatrix(bench_case: BenchCase) -> None:
    """The diagonal CR mass, same prebuilt-numbering convention as the cotmatrix row."""
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        unique_edges, edge_map = _edge_numbering_wp(bench_case)
        matrix = bench_case.run(
            lambda: tw.laplacian.crouzeix_raviart_massmatrix(
                vertices, faces, unique_edges=unique_edges, edge_map=edge_map
            )
        )
        assert matrix.nrow == unique_edges.shape[0]
    else:
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        edges_igl, edge_map_igl = _edge_numbering_np(bench_case)
        matrix_igl = bench_case.run(
            lambda: igl.crouzeix_raviart_massmatrix(vertices_np, faces_np, edges_igl, edge_map_igl)
        )
        assert matrix_igl.shape == (len(edges_igl), len(edges_igl))


@pytest.mark.benchmark(group="face_gradients")
@pytest.mark.benchlibs("triwarp", "igl")
@pytest.mark.parametrize("precomputed", [False, True], ids=["from_positions", "face_data"])
def test_face_gradients(bench_case: BenchCase, precomputed: bool) -> None:
    """
    The piecewise-linear gradient per face: three cross products, no connectivity, no solve.

    The pair of ids is the same amortization question the ``*_precomputed`` rows elsewhere ask:
    ``from_positions`` recomputes the per-face normals and areas, ``face_data`` is handed them. The
    gap is what a caller who already holds that pair saves -- and the heat solvers, which are the
    in-repo consumers, always do.

    **The two sides return different things and that is the comparison.** ``igl.grad(V, F)``
    *assembles* a sparse ``(3F, V)`` operator and never applies it; triwarp applies the same
    operator face by face and never materialises it. So igl's row is dominated by building 3F x 3
    triplets and triwarp's by reading the field, which is exactly the trade the two designs make --
    read it as "assemble once, apply many" against "apply directly", not as one side being faster at
    the same job. The values agree, through the matrix product, in ``tests/test_laplacian.py``.
    """
    if bench_case.kind == "igl":
        if precomputed:
            pytest.skip("igl assembles the operator; there is no face-data shortcut to pass it")
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        matrix_igl = bench_case.run(lambda: igl.grad(vertices_np, faces_np))
        assert matrix_igl.shape == (3 * bench_case.n_faces, bench_case.n_vertices)
        return
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    values = _scalar_field_wp(bench_case)
    if precomputed:
        normals, areas = tw.triangles.face_normals_and_areas(vertices, faces)
        gradients = bench_case.run(
            lambda: tw.laplacian.face_gradients(
                vertices, faces, values, face_normals=normals, face_areas=areas
            )
        )
    else:
        gradients = bench_case.run(lambda: tw.laplacian.face_gradients(vertices, faces, values))
    assert gradients.shape == (bench_case.n_faces,)
