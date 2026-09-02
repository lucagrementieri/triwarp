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

``graph_laplacian`` has no reference: libigl builds ``A - diag(rowsum(A))`` inline inside
``igl::harmonic`` rather than exposing it, and reassembling it here out of ``igl.adjacency_matrix``
plus scipy would time a hand-rolled composition rather than a library function. It is timed for
triwarp alone.

The operator family (``k_harmonic``, ``hessian_energy``, ``curved_hessian_energy``,
``crouzeix_raviart_*``) is igl-referenced throughout and sits on the **scale** axis rather than the
scan sweep: ``igl::crouzeix_raviart_*`` and ``igl::orient_halfedges`` (inside
``curved_hessian_energy``) assume edge-manifold input — igl asserts it, triwarp documents it as
undefined — and the scan meshes are not. Two structural notes on those rows: triwarp assembles
``k_harmonic`` by a triplet pass per power instead of ``bsr_mm`` (originally to avoid a suspected
``bsr_mm`` bug that turned out to be a triplet-capacity defect of our own, and kept because it
measures ~2x faster than ``bsr_mm`` through ``sphere_med`` — but note ``bsr_mm`` wins 0.89x at
``sphere_large`` on CUDA, so this row's margin is the one that would move first), and
``hessian_energy``'s per-vertex work is
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
import pytorch3d.ops as p3d_ops
import trimesh as tm
import trimesh.smoothing as tms
import warp as wp

import triwarp as tw
import triwarp.typing as twt
from conftest import BenchCase

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
        _edge_lengths_wp_cache[key] = twt.as_array2d(
            wp.array(
                np.ascontiguousarray(_edge_lengths_np(bench_case), dtype=np.float32),
                dtype=wp.float32,
                device=bench_case.device,
            ),
            wp.float32,
        )
    return _edge_lengths_wp_cache[key]


# **These rows run on ``lucy`` and are deliberately not capped**, which is worth stating because
# the obvious reading of round 9 says they should be. A ``lucy`` sparse assembly in torch is the
# single largest device allocation this suite makes -- 14 027 872 vertices -- and torch's
# ``CUDACachingAllocator`` reserves those blocks until an explicit ``empty_cache()``, which appeared
# nowhere in ``benchmarks/``: the module exited 1 with ``RuntimeError: Failed to allocate 65368
# bytes`` on a 32 GB card and **16 ``triwarp-cuda`` rows were lost**, all of them silently dropping
# out of the comparison and *understating* the loss table.
#
# The fix is the general one, in ``BenchCase.run``'s pytorch3d teardown, and a cap here was measured
# **unnecessary** rather than assumed: with the release in place and no cap at all, the module runs
# 107 passed / exit 0 with zero allocation failures. So the cap was dropped again -- it would have
# cost the four ``lucy`` comparisons for nothing. If a future pytorch3d row fails here it will be a
# row whose own *peak* does not fit, which no teardown can help and which wants a cap on that row.
def _packed_p3d(bench_case: BenchCase, *, edges: bool = False) -> tuple:
    """
    Return the ``(verts, faces)`` pair the ``ops`` assemblers take, or ``(verts, edges)``.

    Read outside every timed callable, deliberately. A ``Meshes`` memoizes each of these on first
    request, and ``edges_packed`` in particular is the unique-undirected-edge build that
    [`test_edges.py`](test_edges.py) times under ``edges_unique`` -- folding it into a Laplacian row
    would price two groups under one name and make the assembly look 2-3x its cost.

    **That reasoning is right and it used to be applied to one side only**, which is what made the
    edge-list groups the largest misreading in the suite. pytorch3d was handed ``edges_packed()``
    here while triwarp derived its own edge set *inside* the timed call, so the row compared an
    assembly against an assembly-plus-derivation -- and the derivation is the larger half, not a
    detail: ``edges_unique`` measures 0.62 / 0.60 / 2.01 / 2.13 / **73.50 ms** on the five scan
    meshes against a whole ``laplacian(equal_weight=False)`` call of 1.22 / 1.24 / 2.77 / 2.84 /
    **92.10**, i.e. **80 %** of the ``lucy`` row. Read like for like, the reported 8.61x at
    ``lucy`` is a **1.40x win** (10.697 + 118.507 against 92.101), and pytorch3d's own
    ``edges_packed()`` costs 118.5 ms -- 11x its timed row, and 1.29x triwarp's entire call.

    So the two edge-list groups now hand triwarp the same precomputed edges through
    ``laplacian``'s ``edges`` keyword (``_edges_unique_wp``), and both rows price the assembly
    alone. The derivation stays where it belongs, in ``edges_unique``'s own group.
    """
    mesh_p3d = bench_case.mesh_p3d
    return (mesh_p3d.verts_packed(), mesh_p3d.edges_packed() if edges else mesh_p3d.faces_packed())


_edges_wp_cache: dict[tuple[str, str], twt.Array2dInt32] = {}


def _edges_unique_wp(bench_case: BenchCase) -> twt.Array2dInt32:
    """
    Return the unique undirected edges, read outside the timed callable as pytorch3d's are.

    ``edges_packed()`` is memoized on the ``Meshes`` and read outside the row; this is the same
    quantity for the triwarp branch, so the two edge-list groups compare assembly against
    assembly. See ``_packed_p3d`` for why priced-once-elsewhere is the right convention and why
    applying it to one side only was worth ~89 ms of the round-9 loss table.
    """
    key = (bench_case.mesh_name, str(bench_case.device))
    if key not in _edges_wp_cache:
        _edges_wp_cache[key] = tw.edges.edges_unique(
            bench_case.faces_wp, n_vertices=bench_case.n_vertices
        )[0]
    return _edges_wp_cache[key]


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
@pytest.mark.benchlibs("triwarp", "igl", "potpourri3d", "pytorch3d")
def test_cotmatrix(bench_case: BenchCase) -> None:
    """
    Assembled cotangent stiffness matrix: weight kernel plus the sparse build.

    **pytorch3d**'s ``cot_laplacian`` is the fourth assembly here and the only one on the GPU, and
    its row is a **scope mismatch rather than a race** -- the same class as ``is_watertight``'s
    cached ``isClosed`` and ``split``'s label-only ``getAllComponents``. It does not assemble a
    matrix at all: it wraps ``3F`` entries as an *uncoalesced* ``sparse_coo_tensor`` and adds its
    transpose, so the duplicate ``(i, j)`` pairs are never summed and no diagonal is ever written.
    triwarp sorts, dedups and accumulates **12 triplets a face** into a CSR *with* its assembled
    row sum.

    Measured (min of 7 interleaved reps, RTX 5090), which is what settles it -- forcing the
    reference to actually coalesce closes the whole gap:

    | mesh | triwarp | pytorch3d | + ``.coalesce()`` |
    |---|---|---|---|
    | bunny_decimated | 0.526 ms | 0.514 (1.02x) | 0.589 -- **triwarp wins 1.12x** |
    | dragon | 3.363 | 0.728 (**4.62x**) | 3.042 -- **1.11x**, parity |

    And the structure confirms the mechanism rather than merely being consistent with it: on
    ``dragon`` the uncoalesced tensor holds 5 228 484 entries (``6F``), coalescing it gives
    2 618 512, and triwarp's ``nnz_sync()`` is 3 056 157 -- a difference of **437 645, exactly the
    vertex count**, which is the diagonal pytorch3d has none of. Its coalesced diagonal measures
    absmax **0**.

    So the reported 3.66-5.13x is not headroom, and the prebuilt-sparsity-pattern rewrite it
    invited (assemble into an ``edges_unique`` pattern instead of sorting triplets) is **declined
    on this measurement** rather than left as an open item. It also returns the lumped mass
    reciprocal alongside, which is what ``mass_matrix``'s pytorch3d row times from the same call:
    read those two as one call priced twice rather than as two independent measurements.
    """
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "pytorch3d":
        vertices_p3d, faces_p3d = _packed_p3d(bench_case)
        matrix_p3d, _ = bench_case.run(lambda: p3d_ops.cot_laplacian(vertices_p3d, faces_p3d))
        assert matrix_p3d.shape == (n_vertices, n_vertices)
        return
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


@pytest.mark.benchmark(group="laplacian_equal_weight")
@pytest.mark.benchlibs("triwarp", "trimesh", "pytorch3d")
def test_laplacian_equal_weight(bench_case: BenchCase) -> None:
    """
    Row-normalized 1-ring averaging operator with unit weights.

    **pytorch3d**'s ``ops.laplacian`` is the same operator with **-1** on the diagonal where
    triwarp writes 0 (pinned bit-exactly off the diagonal in
    ``tests/test_laplacian.py::test_laplacian_operators_match_pytorch3d``), and it is the only row
    here with GPU kernels. It takes the edge list rather than the faces, so ``edges_packed()`` is
    read outside the timed callable -- that derivation is what ``edges_unique`` times in
    [`test_edges.py`](test_edges.py), and folding it in would price two groups under one name.

    This group needs no matching precomputation on the triwarp side, and the reason is worth
    stating because it is *not* symmetry with the inverse-distance group: ``equal_weight=True``
    takes the ``symmetric=False`` branch, whose triplets come straight off ``faces_to_edges`` --
    it never calls ``edges_unique`` at all. That is also why ``[lucy]`` is 28.95 ms here against
    92.10 in the inverse-distance group and **wins** its row.
    """
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "pytorch3d":
        vertices_p3d, edges_p3d = _packed_p3d(bench_case, edges=True)
        matrix_p3d = bench_case.run(lambda: p3d_ops.laplacian(vertices_p3d, edges_p3d))
        assert matrix_p3d.shape == (n_vertices, n_vertices)
        return
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
@pytest.mark.benchlibs("triwarp", "trimesh", "pytorch3d")
def test_laplacian_inverse_distance(bench_case: BenchCase) -> None:
    """
    The same operator with inverse-edge-length weights (the geometry-dependent branch).

    **pytorch3d**'s ``norm_laplacian`` is this operator before its row normalization -- same
    ``1 / (||vi - vj|| + 1e-12)`` weight, same literal ``eps`` -- so it does strictly less work
    than triwarp's row by exactly one row-sum division.

    **The row-sum division is not what the ratio used to measure, though**, and saying it was is
    what let an 8.61x stand for five rounds. This is the ``symmetric`` branch, so triwarp derived
    the unique undirected edge set *inside* the timed call while pytorch3d was handed
    ``edges_packed()`` outside it -- and that derivation is up to **80 %** of the call, two orders
    of magnitude past a row-sum division. Both sides now take the same precomputed edges
    (``_edges_unique_wp`` / ``_packed_p3d``, which carries the numbers), so the ratio is the
    assembly and the division, which is what this sentence always claimed.
    """
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "pytorch3d":
        vertices_p3d, edges_p3d = _packed_p3d(bench_case, edges=True)
        matrix_p3d = bench_case.run(lambda: p3d_ops.norm_laplacian(vertices_p3d, edges_p3d))
        assert matrix_p3d.shape == (n_vertices, n_vertices)
        return
    if bench_case.kind == "triwarp":
        vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
        edges = _edges_unique_wp(bench_case)
        matrix = bench_case.run(
            lambda: tw.laplacian.laplacian(vertices, faces, equal_weight=False, edges=edges)
        )
        assert matrix.nrow == n_vertices
    else:
        vertices_np, faces_np = bench_case.vertices_np, bench_case.faces_np
        matrix_tm = bench_case.run(
            lambda: tms.laplacian_calculation(
                tm.Trimesh(vertices_np, faces_np, process=False), equal_weight=False
            )
        )
        assert matrix_tm.shape == (n_vertices, n_vertices)


@pytest.mark.benchmark(group="graph_laplacian")
@pytest.mark.benchlibs("triwarp")
def test_graph_laplacian(bench_case: BenchCase) -> None:
    """
    Combinatorial graph Laplacian ``A - diag(deg)``, timed for triwarp alone.

    Not because nothing computes it -- ``scipy.sparse.csgraph.laplacian`` is exactly this operator
    up to a sign (measured an exact match at ``0.0`` against a sign flip), and libigl assembles it
    inline inside ``igl::harmonic``. Neither is a *row*: scipy takes an adjacency matrix, which is
    not triwarp's input, and libigl does not bind the composition, so either row would time an
    assembly written here rather than a library function. The module docstring above carries the
    argument, and ``tests/test_parametrization.py::test_graph_laplacian_matches_igl`` is the
    correctness comparison the decline does not cost.
    """
    vertices, faces = bench_case.vertices_wp, bench_case.faces_wp
    matrix = bench_case.run(lambda: tw.laplacian.graph_laplacian(vertices, faces))
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
@pytest.mark.benchlibs("triwarp", "igl", "pytorch3d")
def test_mass_matrix(bench_case: BenchCase) -> None:
    """
    The same diagonal, assembled as a sparse matrix.

    **pytorch3d** has no separate mass-matrix entry point: ``cot_laplacian`` returns the lumped
    reciprocal ``inv_areas`` as its second value, three times triwarp's diagonal
    (``1 / inv_areas == 3 * M_ii``, measured 5.96e-08). So this row times the **same call** the
    ``cotmatrix`` group times and is an *upper* bound here rather than a race -- it prices the
    stiffness assembly as well. It is still worth the row: it is the only GPU one in the group, and
    the two rows together say what the shared call costs and what fraction of it either half is.
    """
    n_vertices = bench_case.n_vertices
    if bench_case.kind == "pytorch3d":
        vertices_p3d, faces_p3d = _packed_p3d(bench_case)
        _, inv_areas_p3d = bench_case.run(lambda: p3d_ops.cot_laplacian(vertices_p3d, faces_p3d))
        assert inv_areas_p3d.shape[0] == n_vertices
        return
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


@pytest.mark.benchmark(group="face_gradients")
@pytest.mark.benchlibs("triwarp", "igl", "pyvista")
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

    **pyvista applies it and then averages onto the points**: ``compute_derivative`` returns a
    per-*point* gradient for a point-data field (``preference='cell'`` does not move it), so its row
    carries a cell-to-point pass triwarp's does not. It is the closest thing here to triwarp's
    "apply directly" design, which is why it earns a row that igl's assembled operator cannot be
    compared against directly.
    """
    if bench_case.kind == "pyvista":
        if precomputed:
            pytest.skip("VTK recomputes the face geometry internally; nothing can be handed to it")
        mesh_pv = bench_case.mesh_pv
        mesh_pv.point_data["field"] = np.ascontiguousarray(bench_case.vertices_np[:, 2])
        gradient_pv = bench_case.run(
            lambda: mesh_pv.compute_derivative(scalars="field", gradient=True)
        )
        assert np.asarray(gradient_pv.point_data["gradient"]).shape == (bench_case.n_vertices, 3)
        return
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
