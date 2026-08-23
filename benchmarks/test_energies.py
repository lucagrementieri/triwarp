"""
Benchmarks for ``triwarp.energies``: the quadratic forms assembled from a Laplacian.

Split out of [`test_laplacian.py`](test_laplacian.py) with the module. The distinction that makes it
its own file is what the timed callable *takes*: everything here consumes an already-assembled
operator, so its inputs are cached per mesh on both sides and the row measures the assembly of the
second-order form alone. ``test_laplacian.py`` times the first-order builders these are fed.

All five rows run the **scale** axis (``sphere_small`` -> ``sphere_med`` -> ``sphere_large``) rather
than the scan sweep, and the ratio against igl grows with size on every one of them, because igl's
side is Eigen sparse products where triwarp's is a fixed number of launches. libigl is the only
reference that exposes these operators at all -- no other library has a Crouzeix-Raviart pair, a
curved Hessian energy or an integrated k-harmonic form.

The LSCM operators (``lscm_hessian``, ``vector_area_matrix``) are **not** timed here: neither has an
igl binding of its own, and their cost is measured through the ``lscm`` solve in
[`test_parametrization.py`](test_parametrization.py), which is the only way a caller reaches them.
Recorded in ``README.md``.
"""

from __future__ import annotations

import igl
import pytest
import warp as wp

import triwarp as tw
from conftest import BenchCase

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


@pytest.mark.benchmark(group="k_harmonic")
@pytest.mark.benchaxis("scale")
@pytest.mark.benchlibs("triwarp", "igl")
def test_k_harmonic(bench_case: BenchCase) -> None:
    """
    The biharmonic operator ``L M^-1 L`` from a prebuilt Laplacian and mass (``k = 2``).

    Both sides consume cached, prebuilt inputs — the matrices *are* the function's arguments in
    both APIs — so the row times only the composition: igl's two Eigen sparse products against
    triwarp's counted-and-emitted triplet pass plus one ``bsr_from_triplets``.
    """
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "triwarp":
        laplacian, mass = _laplacian_and_mass_wp(bench_case)
        operator = bench_case.run(lambda: tw.energies.k_harmonic(laplacian, mass, k=2))
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
        operator = bench_case.run(lambda: tw.energies.hessian_energy(vertices, faces))
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
        operator = bench_case.run(lambda: tw.energies.curved_hessian_energy(vertices, faces))
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
            lambda: tw.energies.crouzeix_raviart_cotmatrix(
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
            lambda: tw.energies.crouzeix_raviart_massmatrix(
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
