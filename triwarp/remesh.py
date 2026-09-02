"""
Changing a mesh's triangulation: subdividing it, coarsening it, and improving its triangle shapes.

Four families, in decreasing order of how much they rearrange:

- **Remeshing.** [`isotropic_remesh`][triwarp.remesh.isotropic_remesh] runs the Botsch-Kobbelt
  split / collapse / flip / smooth / reproject loop until every edge is near a target length. It is
  the only entry point here that does all four of the others' jobs at once.
- **Decimation.** [`quadric_decimate`][triwarp.remesh.quadric_decimate] collapses edges in
  quadric-error order to a target face count; [`cluster_decimate`][triwarp.remesh.cluster_decimate]
  instead welds each voxel of a uniform grid to a single vertex, which is far cheaper and far
  blunter.
- **Edge flipping**, which moves no vertex and changes no vertex count:
  [`flip_to_delaunay`][triwarp.remesh.flip_to_delaunay] toward the Delaunay criterion,
  [`flip_by_objective`][triwarp.remesh.flip_by_objective] toward a triangle-shape or flatness
  objective, and [`intrinsic_delaunay`][triwarp.remesh.intrinsic_delaunay] toward the *intrinsic*
  Delaunay triangulation, which flips the connectivity a Laplacian sees without touching the
  embedding.
- **Subdivision**, which only ever adds: [`subdivide`][triwarp.remesh.subdivide] (one-to-four
  splits), [`subdivide_loop`][triwarp.remesh.subdivide_loop] (Loop's approximating scheme),
  [`subdivide_to_size`][triwarp.remesh.subdivide_to_size] and
  [`subdivide_region_to_size`][triwarp.remesh.subdivide_region_to_size] (until every edge, or every
  edge of a face region, is under a target length).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal, NamedTuple, overload

import warp as wp
import warp.sparse as wps

import triwarp as tw
import triwarp.typing as twt
from triwarp._device import read_scalar, require_nonempty_mesh
from triwarp.constants import INT32_MAX, TOLERANCE_MOLLIFY
from triwarp.kernels import adjacency as kernel_adjacency
from triwarp.kernels import array as kernel_array
from triwarp.kernels import remesh as kernel_remesh
from triwarp.kernels import scatter as kernel_scatter
from triwarp.laplacian import mollify_intrinsic

# Callback that launches a predicate kernel filling ``out_flip``/``out_quad`` for one flip
# iteration. Supplied by each consumer of ``_flip_interior_edges`` (3D Delone / 2D incircle).
_LaunchCandidates = Callable[
    [
        twt.Array2dInt32,  # adjacency (m, 2)
        twt.Array2dInt32,  # adjacency_edges (m, 2)
        twt.Array2dInt32,  # unshared (m, 2)
        "wp.array[wp.uint64]",  # sorted edge keys
        "wp.uint64",  # key base (n_vertices)
        "wp.array[wp.bool]",  # out_flip (m,)
        twt.Array2dInt32,  # out_quad (m, 4)
    ],
    None,
]


class _EdgeIncidence(NamedTuple):
    """
    The unique undirected edges of one triangulation, and which faces meet along each of them.

    Everything the collapse passes and ``_classify`` need about edge topology, grouped once:
    ``unique_edges`` and ``inverse`` come straight from
    [`edges_unique`][triwarp.edges.edges_unique], and one scatter over ``inverse`` fills both
    ``face_count`` (1 on a boundary edge, 2 on an interior one) and ``faces``.
    """

    unique_edges: twt.Array2dInt32
    """``(m, 2)`` unique undirected vertex pairs, each row min-first."""
    inverse: wp.array[wp.int32]
    """Length ``3 * n_faces`` corner -> unique-edge map; corner ``c`` belongs to face ``c // 3``."""
    face_count: wp.array[wp.int32]
    """Length ``m`` face-corners per unique edge."""
    faces: twt.Array2dInt32
    """``(m, 2)`` incident face indices, the second column unwritten where ``face_count`` is 1."""


# Backstop on the independent-set rounds per geometry rebuild in ``quadric_decimate``.
#
# One round commits only a fraction of the scored candidates -- each winner locks the closed 1-rings
# of both endpoints, so a hashed-key round takes on the order of ``m / 50`` of them -- and the
# rebuild that follows is ~40 wrapper calls against a handful of launches for another round. So the
# pass loop runs rounds against the same scoring **until one finds nothing new**, which is the real
# stopping rule; this constant only bounds it. 6 and 8 produce byte-identical output on every
# fixture measured, i.e. saturation happens first, and a round that finds nothing costs ~8 launches.
_QUADRIC_ROUNDS = 8


def isotropic_remesh(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    target_length: float | wp.array[wp.float32] | None = None,
    iterations: int = 10,
    feature_angle: float = 30.0,
    split: bool = True,
    collapse: bool = True,
    swap: bool = True,
    smooth: bool = True,
    reproject: bool = True,
    max_deviation: float | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Isotropic explicit remeshing (Botsch-Kobbelt split / collapse / flip / smooth / reproject).

    GPU port of the classic incremental isotropic remesher (PyMeshLab's
    ``meshing_isotropic_explicit_remeshing``, vcglib ``IsotropicRemeshing``): each iteration drives
    all edge lengths toward ``target_length`` by (1) **splitting** every edge longer than
    ``4/3 * target_length`` at its midpoint (crack-free, reusing
    [`subdivide_to_size`][triwarp.remesh.subdivide_to_size]); (2) **collapsing** every edge shorter
    than ``4/5 * target_length`` (a parallel primitive with full 1-ring locking and a manifold
    link-condition guard); (3) **flipping** interior edges toward the ideal vertex valence (6
    interior, 4 boundary); (4) **tangentially smoothing** free vertices (area-equalizing Laplacian
    projected onto the tangent plane); and (5) **reprojecting** free vertices back onto the original
    surface. Feature and boundary structure is preserved: each vertex is classified FREE / CREASE
    (on a boundary loop or a dihedral crease sharper than ``feature_angle``) / CORNER (feature
    junction or endpoint), corners are frozen, crease vertices only move along their feature, and
    feature edges are never flipped.

    Inputs are cloned and never mutated.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions on the target device.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer. Under ``reproject``,
        the reference ``wp.Mesh`` aliases ``vertices`` and ``faces`` rather than copying them;
        do not mutate them for the duration of the call.
    target_length
        Desired edge length. A **scalar** is the uniform target, defaulting to ``1 %`` of the
        bounding-box diagonal. A ``(n_vertices,)`` ``wp.float32`` array is an **adaptive sizing
        field** over the *input* vertices: every stage then reads its own local target, so the
        result is fine where the field is small and coarse where it is large. This is the general
        form of PyMeshLab's ``adaptive`` flag — rather than deriving the field from curvature
        internally, the caller supplies it, which also covers a painted field, a distance-to-feature
        field, or a field carried from another mesh. Build a curvature-driven one from
        [`triwarp.curvature`][triwarp.curvature], or an interpolated one from
        [`interpolate_from_points`][triwarp.interpolation.interpolate_from_points].
    iterations
        Number of full remeshing passes.
    feature_angle
        Dihedral angle in **degrees** above which an interior edge is treated as a sharp feature
        (protected from flipping and collapsing across).
    split, collapse, swap, smooth, reproject
        Enable/disable each stage of the per-iteration pipeline.
    max_deviation
        Bound on how far the result may move off the input surface, in model units. At the end of
        every iteration each vertex further than this from the input surface is pulled straight back
        toward its own closest point until it is exactly this far, bounding the result's one-sided
        Hausdorff distance to the input. ``None`` (the default) leaves fidelity to ``reproject``
        alone, as before. This is PyMeshLab's ``checksurfdist`` / ``maxsurfdist`` pair as a single
        optional bound. See the Notes for how tightly the bound actually holds.

    Returns
    -------
    vertices : wp.array[wp.vec3]
        Remeshed vertex positions on ``vertices.device``.
    faces : wp.array[wp.int32]
        Flat ``3 * n_faces`` triangle index buffer.

    Raises
    ------
    ValueError
        If ``target_length`` is non-positive, a sizing field does not have one entry per vertex or
        holds a non-positive value, or ``max_deviation`` is non-positive.

    See Also
    --------
    [`subdivide_to_size`][triwarp.remesh.subdivide_to_size]
    [`split_edges`][triwarp.remesh.split_edges]
    [`flip_to_delaunay`][triwarp.remesh.flip_to_delaunay]

    Notes
    -----
    Hysteresis (split above ``4/3 t``, collapse below ``4/5 t``) keeps split and collapse from
    fighting. The collapse primitive guarantees manifoldness through the link condition but has no
    normal-flip guard in this version, relying on the reprojection step to keep free vertices on the
    original surface.

    An adaptive field is **re-sampled from the input surface** before each stage that reads it, with
    [`transfer_onto_vertices`][triwarp.interpolation.transfer_onto_vertices], rather than
    transported through the split / collapse operations. The field is a property of the input
    geometry, so re-sampling keeps it exact under an arbitrary sequence of operations where
    transport would accumulate error; the cost is one closest-point query per vertex per stage (two
    per iteration with both ``split`` and ``collapse`` on). Inside a single stage the field *is*
    transported, because there the correspondence is known exactly: a split midpoint takes the mean
    of the endpoints it splits, and a collapse compacts the bands alongside the vertices.

    A **constant** field is not quite the scalar path: measured on a remeshed ``icosphere(3)`` at 3
    iterations the two agree on the face buffer *exactly* (``np.array_equal``) and on positions to
    1.2e-05 on a mesh of extent 2.0. The gap is float rounding in the threshold alone — the array
    path forms ``4/3 * t`` per vertex in ``float32`` where the scalar path forms it in Python
    ``float64`` and narrows once — and the resampled constant itself is exact to 7.5e-09. Pass a
    scalar when the target is uniform; it is also one closest-point query per stage cheaper.

    ``max_deviation`` is a **positional bound applied per iteration**, not a per-operation rejection
    test: an individual collapse or flip is never vetoed for moving the surface too far, it is the
    accumulated vertex position that is corrected afterwards. A mesh whose *edges* must never sweep
    past the bound mid-iteration needs the flip stage's own gate as well
    ([`flip_to_delaunay`][triwarp.remesh.flip_to_delaunay] takes one).

    How tightly the bound holds is worth stating, because it is set by
    ``wp.mesh_query_point_no_sign`` rather than by this function. Against that query — the one the
    clamp is implemented with, and the one ``reproject`` has always used — the result is within the
    bound to **1.4e-07**. Against an independent ``float64`` query
    (``trimesh.proximity.closest_point``) on a remeshed ``icosphere(3)`` at 5 iterations with
    ``reproject=False``, whose unconstrained deviation is
    3.21e-03: a bound of 1.07e-03 measures 1.00x the bound, 3.21e-04 measures 1.01x, and 1.07e-04
    measures **1.37x** — the two queries disagree by up to 2.1e-05 in absolute terms (mean 1.6e-08,
    so it is a handful of vertices, and iterating the clamp does not converge further because Warp's
    answer is a fixed point). The bound therefore controls deviation proportionally — those three
    settings reduce it by 3.0x, 9.9x and 21.9x — but at a bound near Warp's own query accuracy it is
    approximate rather than hard. Ask for a bound comfortably above 2e-05 in model units, or scale
    the model up.

    The remaining limitation, stated because the parameter that used to advertise it is gone:
    PyMeshLab's ``selectedonly`` has no equivalent here, and a region-restricted refinement is
    [`subdivide_region_to_size`][triwarp.remesh.subdivide_region_to_size] rather than a mode of this
    function.
    """
    n_faces = int(faces.shape[0]) // 3
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    current_vertices = wp.clone(vertices)
    current_faces = wp.clone(faces)
    if n_faces == 0 or iterations <= 0:
        return current_vertices, current_faces

    diag = tw.bounds.enclosing_diagonal(vertices)
    sizing_input: wp.array[wp.float32] | None = None
    if target_length is None or isinstance(target_length, int | float):
        target = float(target_length) if target_length is not None else 0.01 * diag
        if target <= 0.0:
            raise ValueError(f"isotropic_remesh requires target_length > 0, got {target}.")
    else:
        field = target_length
        if int(field.shape[0]) != n_vertices:
            raise ValueError(
                f"isotropic_remesh requires one target_length per vertex ({n_vertices}), "
                f"got {int(field.shape[0])}."
            )
        # One readback, on a buffer the caller just built: a non-positive entry makes the split
        # stage diverge (every edge over-long), so it is worth catching here rather than at
        # ``max_iter``.
        smallest = float(tw.reduce.min(field))
        if smallest <= 0.0:
            raise ValueError(
                f"isotropic_remesh requires a positive target_length everywhere, got {smallest}."
            )
        target = float(tw.reduce.mean(field))
        sizing_input = field
    if max_deviation is not None and max_deviation <= 0.0:
        raise ValueError(f"isotropic_remesh requires max_deviation > 0, got {max_deviation}.")
    feature = wp.float32(math.radians(feature_angle))

    # Original surface, built once, for reprojecting free vertices and for the deviation bound
    # (never a 0-triangle mesh).
    original_mesh = None
    if reproject or max_deviation is not None or sizing_input is not None:
        require_nonempty_mesh(faces, "isotropic_remesh")
        # The mesh aliases the caller's buffers and is discarded here, so it needs no copy: the
        # loop below rebinds ``current_vertices`` / ``current_faces`` and never writes ``vertices``.
        original_mesh = wp.Mesh(points=vertices, indices=faces)
    query_radius = max(diag, 1.0)
    clamp_kernel = None
    if max_deviation is not None:
        # Hoisted out of the loop: the generated kernel is cached, but the per-call Python is not.
        clamp_kernel = wp.map(
            kernel_remesh.clamp_to_surface_band,
            current_vertices,
            wp.uint64(0),
            wp.float32(0.0),
            wp.float32(0.0),
            out=wp.empty_like(current_vertices),
            return_kernel=True,
        )

    for _ in range(iterations):
        # The sizing field is re-sampled from the *input* surface immediately before each stage that
        # reads it, because the stage before it changed the vertex set. Two closest-point passes per
        # iteration is the price of keeping the field exact; see this function's Notes.
        if split:
            _, high = _length_bands(
                _sizing_at(current_vertices, vertices, faces, sizing_input, query_radius),
                target,
                int(current_vertices.shape[0]),
                device,
            )
            current_vertices, current_faces = subdivide_to_size(
                current_vertices,
                current_faces,
                high if sizing_input is not None else 4.0 / 3.0 * target,
                max_iter=20,
            )
        if collapse:
            low, high = _length_bands(
                _sizing_at(current_vertices, vertices, faces, sizing_input, query_radius),
                target,
                int(current_vertices.shape[0]),
                device,
            )
            current_vertices, current_faces = _collapse_pass(
                current_vertices, current_faces, low, high, feature
            )
        if int(current_faces.shape[0]) == 0:
            break
        if swap:
            _valence_flip_pass(current_vertices, current_faces, feature)
        if smooth or reproject:
            codes, _boundary = _classify(current_vertices, current_faces, feature)
            if smooth:
                current_vertices = _smooth_pass(current_vertices, current_faces, codes)
            if reproject and original_mesh is not None:
                current_vertices = _reproject_pass(
                    current_vertices, codes, original_mesh, query_radius
                )
        if clamp_kernel is not None and original_mesh is not None:
            bounded = wp.empty_like(current_vertices)
            wp.launch(
                clamp_kernel,
                dim=int(current_vertices.shape[0]),
                inputs=[
                    current_vertices,
                    wp.uint64(original_mesh.id),
                    wp.float32(max_deviation),
                    wp.float32(query_radius),
                ],
                outputs=[bounded],
                device=device,
            )
            current_vertices = bounded

    return current_vertices, current_faces


def _sizing_at(
    positions: wp.array[wp.vec3],
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    sizing_input: wp.array[wp.float32] | None,
    query_radius: float,
) -> wp.array[wp.float32] | None:
    """
    Sample the input mesh's sizing field at ``positions``, via their closest points on that surface.

    ``None`` in, ``None`` out, so the uniform-target path costs nothing and the caller needs no
    branch of its own.
    """
    if sizing_input is None:
        return None
    return tw.interpolation.transfer_onto_vertices(
        vertices, faces, sizing_input, positions, max_dist=query_radius
    )[0]


def _length_bands(
    sizing: wp.array[wp.float32] | None, target: float, n_vertices: int, device: wp.DeviceLike
) -> tuple[wp.array[wp.float32], wp.array[wp.float32]]:
    """
    Per-vertex collapse-below and split-above length bands, from a sizing field or a uniform target.

    The Botsch-Kobbelt hysteresis (``4/5 t`` and ``4/3 t``) is applied per vertex so the uniform
    case is literally the constant field, letting the collapse kernel keep a single code path.
    """
    if sizing is None:
        low = wp.full(n_vertices, 4.0 / 5.0 * target, dtype=wp.float32, device=device)
        high = wp.full(n_vertices, 4.0 / 3.0 * target, dtype=wp.float32, device=device)
        return low, high
    low = wp.empty(n_vertices, dtype=wp.float32, device=device)
    high = wp.empty(n_vertices, dtype=wp.float32, device=device)
    wp.map(wp.mul, sizing, wp.float32(4.0 / 5.0), out=low)
    wp.map(wp.mul, sizing, wp.float32(4.0 / 3.0), out=high)
    return low, high


def _classify(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    feature: wp.float32,
    incidence: _EdgeIncidence | None = None,
) -> tuple[wp.array[wp.int32], wp.array[wp.bool]]:
    """
    Per-vertex FREE / CREASE / CORNER codes plus a boundary-vertex mask.

    A boundary edge or an interior edge sharper than ``feature`` counts as one incident feature
    edge; a vertex with zero is FREE, exactly two is CREASE (a smooth feature/boundary line), and
    anything else (a feature endpoint or a junction) is a frozen CORNER.

    Both questions are answered by **one launch** over ``incidence``, which the collapse passes have
    already built for their own scoring and pass in. That matters because the answer used to come
    from ``boundary.boundary_edges`` plus ``adjacency.face_adjacency``, each of which hashes, sorts
    and groups the same ``3 * n_faces`` edge rows the incidence was grouped from -- two of those
    three groupings were pure repetition, and they were 84 % of this function. Measured back to
    back on ``saddle`` with the incidence supplied: **1 059 -> 100 us, 10.6x**, which is
    ``quadric_decimate`` **1.31-1.36x** end to end (``saddle`` at ``target_ratio=0.1`` 148 -> 112
    ms, ``saddle_graded`` 193 -> 142) because the pass runs 35-45 times. ``_classify`` is also
    called up to seven times per ``isotropic_remesh`` iteration, which gains **1.27x** from the
    same change even though those callers have no incidence to hand and build their own.
    """
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    n_faces = int(faces.shape[0]) // 3
    codes = wp.zeros(n_vertices, dtype=wp.int32, device=device)
    boundary_vertex = wp.zeros(n_vertices, dtype=wp.bool, device=device)
    if n_faces == 0:
        return codes, boundary_vertex

    if incidence is None:
        incidence = _edge_incidence(faces, n_vertices)
    m = int(incidence.unique_edges.shape[0])
    if m == 0:
        return codes, boundary_vertex

    feature_count = wp.zeros(n_vertices, dtype=wp.int32, device=device)
    wp.launch(
        kernel_remesh.scatter_feature_edge_counts,
        dim=m,
        inputs=[
            vertices,
            faces,
            incidence.unique_edges,
            incidence.face_count,
            incidence.faces,
            feature,
            feature_count,
            boundary_vertex,
        ],
        device=device,
    )
    wp.map(kernel_remesh.finalize_vertex_codes, feature_count, out=codes)
    return codes, boundary_vertex


def _edge_incidence(faces: wp.array[wp.int32], n_vertices: int) -> _EdgeIncidence:
    """
    Group a triangulation's edge rows once, into unique edges and their incident faces.

    The face table costs one scatter over the corners on top of the ``edges_unique`` the collapse
    passes run anyway -- and it replaces the ``scatter.count_occurrences`` launch they used to make
    for the count alone, so it is very nearly free. It is what lets ``_classify`` skip a second and
    third grouping of the same rows; see ``scatter_edge_incidence`` in ``kernels/scatter.py``.
    """
    device = faces.device
    unique_edges, inverse = tw.edges.edges_unique(faces, n_vertices=n_vertices)
    m = int(unique_edges.shape[0])
    face_count = wp.zeros(m, dtype=wp.int32, device=device)
    edge_faces = twt.empty_2d((m, 2), wp.int32, device=device)
    if m > 0:
        wp.launch(
            kernel_scatter.scatter_edge_incidence,
            dim=int(inverse.shape[0]),
            inputs=[inverse, face_count, edge_faces],
            device=device,
        )
    return _EdgeIncidence(unique_edges, inverse, face_count, twt.as_array2d(edge_faces, wp.int32))


def _collapse_pass(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    low: wp.array[wp.float32],
    high: wp.array[wp.float32],
    feature: wp.float32,
    max_passes: int = 5,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Collapse short edges in parallel with 1-ring locking; returns compacted (vertices, faces).

    ``low`` and ``high`` are the per-vertex length bands, so one path serves both a uniform target
    and a sizing field. They are compacted alongside the vertices at the end of each pass rather
    than re-sampled, which keeps the whole loop free of closest-point queries.
    """
    device = vertices.device
    for _ in range(max_passes):
        n_vertices = int(vertices.shape[0])
        n_faces = int(faces.shape[0]) // 3
        if n_faces == 0:
            break

        # One edge grouping per pass, shared by the candidate scoring and ``_classify``.
        incidence = _edge_incidence(faces, n_vertices)
        unique_edges = incidence.unique_edges
        m = int(unique_edges.shape[0])
        if m == 0:
            break
        lengths = tw.edges.edges_unique_length(vertices, faces, unique_edges=unique_edges)

        codes, _boundary = _classify(vertices, faces, feature, incidence)
        csr = tw.graph.edges_to_csr(n_vertices, unique_edges)

        survivor = wp.full(m, -1, dtype=wp.int32, device=device)
        removed = wp.empty(m, dtype=wp.int32, device=device)
        target_pos = wp.empty(m, dtype=wp.vec3, device=device)
        wp.launch(
            kernel_remesh.collapse_candidates,
            dim=m,
            inputs=[
                unique_edges,
                lengths,
                vertices,
                codes,
                incidence.face_count,
                csr.offsets,
                csr.columns,
                low,
                high,
                survivor,
                removed,
                target_pos,
            ],
            device=device,
        )

        claim = wp.full(n_vertices, INT32_MAX, dtype=wp.int32, device=device)
        wp.launch(
            kernel_remesh.claim_collapses,
            dim=m,
            inputs=[survivor, removed, csr.offsets, csr.columns, claim],
            device=device,
        )
        remap = tw.array.arange(n_vertices, device)
        positions = wp.clone(vertices)
        count = wp.zeros(1, dtype=wp.int32, device=device)
        wp.launch(
            kernel_remesh.commit_collapses,
            dim=m,
            inputs=[
                survivor,
                removed,
                target_pos,
                csr.offsets,
                csr.columns,
                claim,
                remap,
                positions,
                count,
            ],
            device=device,
        )
        if int(read_scalar(count, 0)) == 0:
            break

        remapped = tw.array.gather(remap, faces)
        valid = wp.empty(n_faces, dtype=wp.bool, device=device)
        wp.launch(
            kernel_remesh.faces_with_distinct_indices,
            dim=n_faces,
            inputs=[remapped, valid],
            device=device,
        )
        kept = tw.array.flatnonzero(valid)
        faces = tw.array.gather(remapped.reshape((n_faces, 3)), kept).reshape(-1)
        vertices, faces, _, surviving = tw.repair.remove_unreferenced_vertices(
            positions, faces, return_inverse=True
        )
        # ``surviving`` is the new-to-old vertex map, so the bands follow the compaction exactly.
        low = tw.array.gather(low, surviving)
        high = tw.array.gather(high, surviving)

    return vertices, faces


def _valence_flip_pass(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], feature: wp.float32, max_iter: int = 10
) -> None:
    """Flip interior edges toward ideal valence (6 interior, 4 boundary); mutates ``faces``."""
    device = faces.device
    n_vertices = int(vertices.shape[0])
    _codes, boundary_vertex = _classify(vertices, faces, feature)

    def launch(adjacency, adjacency_edges, unshared, sorted_keys, key_base, out_flip, out_quad):
        # Valence is recomputed from the (in-place mutated) faces each pass to avoid staleness.
        valence = wp.zeros(n_vertices, dtype=wp.int32, device=device)
        unique_edges, _ = tw.edges.edges_unique(faces, n_vertices=n_vertices)
        wp.launch(
            kernel_scatter.count_occurrences_rows,
            dim=int(unique_edges.shape[0]),
            inputs=[unique_edges, valence],
            device=device,
        )
        wp.launch(
            kernel_remesh.valence_flip_candidates,
            dim=int(adjacency.shape[0]),
            inputs=[
                vertices,
                faces,
                adjacency,
                adjacency_edges,
                unshared,
                sorted_keys,
                key_base,
                valence,
                boundary_vertex,
                feature,
                out_flip,
                out_quad,
            ],
            device=device,
        )

    _flip_interior_edges(faces, n_vertices, launch, max_iter)


def _smooth_pass(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], codes: wp.array[wp.int32]
) -> wp.array[wp.vec3]:
    """
    One tangential Laplacian smoothing step over free vertices.

    This is the *unweighted* one-ring centroid, which is a documented gap rather than a choice: the
    area-equalizing form Botsch-Kobbelt specify (and which ``isotropic_remesh``'s Notes describe) is
    what actually removes anisotropy, and this one cannot -- on a regular graded grid every vertex
    already sits at the plain average of its neighbours, so the smoother is at a fixed point. See
    ``accumulate_one_ring`` for the measurement and the blocker.
    """
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    normals = tw.vertices.vertex_normals(vertices, faces)
    unique_edges, _ = tw.edges.edges_unique(faces, n_vertices=n_vertices)

    ring_sum = wp.zeros(n_vertices, dtype=wp.vec3, device=device)
    degree = wp.zeros(n_vertices, dtype=wp.int32, device=device)
    wp.launch(
        kernel_remesh.accumulate_one_ring,
        dim=int(unique_edges.shape[0]),
        inputs=[unique_edges, vertices, ring_sum, degree],
        device=device,
    )
    out_positions = wp.empty(n_vertices, dtype=wp.vec3, device=device)
    wp.map(
        kernel_remesh.tangential_smooth_step,
        vertices,
        codes,
        normals,
        ring_sum,
        degree,
        wp.float32(1.0),
        out=out_positions,
    )
    return out_positions


def _reproject_pass(
    vertices: wp.array[wp.vec3], codes: wp.array[wp.int32], original_mesh: wp.Mesh, max_dist: float
) -> wp.array[wp.vec3]:
    """Snap free vertices onto the closest point of the original surface."""
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    out_positions = wp.empty(n_vertices, dtype=wp.vec3, device=device)
    # ``wp.uint64(...)`` is required: a bare ``wp.Mesh.id`` is a Python int and ``wp.map`` would
    # infer ``int32`` for it (see tests/test_map_uniform_probe.py).
    wp.map(
        kernel_remesh.reproject_vertices,
        vertices,
        codes,
        wp.uint64(original_mesh.id),
        wp.float32(max_dist),
        out=out_positions,
    )
    return out_positions


def _flip_interior_edges(
    faces: wp.array[wp.int32], n_vertices: int, launch_candidates: _LaunchCandidates, max_iter: int
) -> int:
    """
    Repeatedly flip an independent set of interior edges until none is a candidate.

    ``faces`` is mutated in place. Each iteration rebuilds face adjacency, lets
    ``launch_candidates`` mark flippable edges (predicate-specific), then commits a
    conflict-free subset (no two committed flips touch a shared face or create the same new
    edge). Returns the total number of flips performed. The winding rewrite matches
    ``igl::flip_edge``.

    The per-pass topology is built by ``_FlipTopology`` on fixed buffers rather than by composing
    the public wrappers, which is a measured 1.5-2.6x on the whole call; see it for why.
    """
    device = faces.device
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0 or max_iter <= 0:
        return 0
    topology = _FlipTopology(faces, n_vertices)
    key_base = wp.uint64(n_vertices)
    count = wp.zeros(1, dtype=wp.int32, device=device)
    total = 0
    for _ in range(max_iter):
        m = topology.rebuild()
        if m == 0:
            break
        launch_candidates(
            topology.adjacency,
            topology.adjacency_edges,
            topology.unshared,
            topology.sorted_keys,
            key_base,
            topology.flip,
            topology.quad,
        )

        # Independent-set selection: a flip commits only if it wins both incident faces and the
        # hashed slot of its new edge (prevents two disjoint flips creating the same edge).
        topology.face_claim.fill_(INT32_MAX)
        topology.edge_claim.fill_(INT32_MAX)
        wp.launch(
            kernel_remesh.claim_flips,
            dim=m,
            inputs=[
                topology.flip,
                topology.quad,
                topology.adjacency,
                wp.int32(topology.edge_claim_mask),
                key_base,
                topology.face_claim,
                topology.edge_claim,
            ],
            device=device,
        )
        count.zero_()
        wp.launch(
            kernel_remesh.commit_flips,
            dim=m,
            inputs=[
                topology.flip,
                topology.quad,
                topology.adjacency,
                topology.face_claim,
                topology.edge_claim,
                wp.int32(topology.edge_claim_mask),
                key_base,
                faces,
                count,
            ],
            device=device,
        )
        n = int(read_scalar(count, 0))
        total += n
        if n == 0:
            break
    return total


class _FlipTopology:
    """
    Persistent per-pass working set of the parallel edge-flip loop.

    The flip loop reruns the same topology build 20-40 times on a face buffer whose *shape* never
    changes — a flip rewrites two triangles' corners and leaves the vertex, face and interior-edge
    counts alone — so composing the public wrappers pays for a fresh allocation chain every pass.
    Measured on ``delaunay_triangulation``: the composed form spent 442 us/pass in
    [`face_adjacency`][triwarp.adjacency.face_adjacency] and another 186 in
    [`sort_and_argsort`][triwarp.array.sort_and_argsort] out of 962 us of host time, against ~1.2 ms
    of device work for the *whole* 37-pass loop.

    Two things collapse here. Every buffer is allocated once and rewritten in place; and the pass
    sorted its edge keys **twice** — once inside ``face_adjacency``'s row grouping and once again
    for the "would this flip duplicate an existing edge" table the candidate predicates search —
    where one radix sort produces both. What remains is one key launch, one sort, a run-length
    mark, a scan and a single emit launch that writes the adjacency pairs, their shared-edge
    endpoints and the opposite apexes together.

    Measured back to back in one process, interleaved:
    ``delaunay_triangulation`` **2.61x** at 2 000 points (18.70 -> 7.16 ms) and **1.93x** at 20 000
    (36.14 -> 18.70), [`flip_to_delaunay`][triwarp.remesh.flip_to_delaunay] and
    [`flip_by_objective`][triwarp.remesh.flip_by_objective] 1.50-1.61x,
    [`isotropic_remesh`][triwarp.remesh.isotropic_remesh] 1.06x (its flip stage is a small share of
    the iteration). CPU agrees: 1.98-2.08x, 1.51-1.73x, 1.06x. The one row that does not gain is
    [`intrinsic_delaunay`][triwarp.remesh.intrinsic_delaunay], whose input is already near-Delaunay
    so the loop exits in one or two passes and the fixed setup is the whole cost -- 1.00x on CUDA
    and 0.92-0.98x on CPU.

    Every intermediate is byte-identical to the composed path: the keys match
    [`hash_indices_rows`][triwarp.grouping.hash_indices_rows] over
    [`faces_to_edges`][triwarp.edges.faces_to_edges] rows (see
    [`face_edge_keys`][triwarp.kernels.adjacency.face_edge_keys]), and the scan reproduces the
    ascending-key row order [`group`][triwarp.grouping.group] gets from its ``flatnonzero``
    compaction.

    Attributes
    ----------
    sorted_keys : wp.array[wp.uint64]
        Length ``3 * n_faces`` ascending undirected-edge keys of the current triangulation, which
        the candidate predicates binary-search for the duplicate-edge guard.
    adjacency : twt.Array2dInt32
        ``(m, 2)`` ascending face pairs sharing an interior edge.
    adjacency_edges : twt.Array2dInt32
        ``(m, 2)`` sorted endpoints of each shared edge, row-aligned with ``adjacency``.
    unshared : twt.Array2dInt32
        ``(m, 2)`` apex of each incident face opposite the shared edge.
    flip : wp.array[wp.bool]
        Length ``m`` candidate mask. Every predicate kernel opens by writing all of it, so it is
        deliberately not zeroed between passes.
    quad : twt.Array2dInt32
        ``(m, 4)`` flip quad ``(a, b, c, d)``, written only where ``flip`` is set.
    face_claim : wp.array[wp.int32]
        Length ``n_faces`` per-face winner of the independent-set round.
    edge_claim : wp.array[wp.int32]
        Open-addressed claim table over the new edges, one slot per hashed key.
    edge_claim_mask : int
        Power-of-two mask for ``edge_claim`` slots.
    """

    def __init__(self, faces: wp.array[wp.int32], n_vertices: int) -> None:
        """Allocate the fixed working set for the ``faces`` buffer the loop will mutate in place."""
        self._faces = faces
        self._device = faces.device
        self._n_faces = int(faces.shape[0]) // 3
        self._n_corners = self._n_faces * 3
        self._radix = wp.uint64(n_vertices)
        n = self._n_corners
        # ``radix_sort_pairs`` ping-pongs through the upper half of both buffers, so each is
        # double width and only ``[:n]`` is data. The sorted keys are the duplicate-edge table.
        self._keys = wp.empty(2 * n, dtype=wp.uint64, device=self._device)
        self._order = wp.empty(2 * n, dtype=wp.int32, device=self._device)
        self._starts = wp.empty(n, dtype=wp.int32, device=self._device)
        self._ranks = wp.empty(n, dtype=wp.int32, device=self._device)
        self._ranks_tail = self._ranks[n - 1 :]
        self.sorted_keys = self._keys[:n]
        self.face_claim = wp.empty(self._n_faces, dtype=wp.int32, device=self._device)
        self._rows = -1
        self._allocate_rows(0)

    def rebuild(self) -> int:
        """
        Regroup the mutated face buffer into interior-edge rows, and return how many there are.

        Every public attribute is rewritten; the returned row count is also the launch dimension
        for the candidate, claim and commit kernels.

        **It is launch-bound, not data-bound, and that is why the region flip pass does not scope
        it to the region.** Measured on icospheres, one rebuild, ``min`` of 7 on an RTX 5090: 0.161
        ms at 320 faces, 0.204 at 5 120, 0.248 at 81 920 and 0.316 at 327 680 -- a **1 024x** range
        of input for **1.96x** of cost, because the eight launches' marshalling dominates the radix
        sort. So the reading that suggests itself from ``_flip_region_faces``, where the sort covers
        the whole mesh while only edges with *both* faces in the region are flippable, is a much
        smaller number than it looks: on ``refill_region``'s ``dragon`` row the pass rebuilds 1 050
        555 interior-edge rows to serve a 4 314-face region (0.61 %), and scoping it to that region
        would save the difference between 0.454 ms per round and the ~0.16 ms floor -- about 0.15
        ms x 13 rounds on the largest mesh in the suite, and nothing at all on ``bunny`` (0.252 ms
        per round already). Restricting the *duplicate-edge* table would also need the region's
        vertex one-ring closure to stay exact, since a flip's new edge may already exist outside the
        region. Not worth it at that price; measure again if the floor moves.
        """
        n = self._n_corners
        wp.launch(
            kernel_adjacency.face_edge_keys,
            dim=self._n_faces,
            inputs=[self._faces, self._radix, self._keys],
            device=self._device,
        )
        wp.launch(
            kernel_array.init_sort_pair_indices,
            dim=2 * n,
            inputs=[wp.int32(n), wp.int32(-1), self._order],
            device=self._device,
        )
        wp.utils.radix_sort_pairs(self._keys, self._order, count=n)
        wp.launch(
            kernel_remesh.mark_edge_pair_starts,
            dim=n,
            inputs=[self._keys, wp.int32(n), self._starts],
            device=self._device,
        )
        # Inclusive, so the row count is one 4-byte tail read and the emit kernel's row is
        # ``ranks[i] - 1`` -- the contract ``array.flatnonzero`` uses for the same reason.
        wp.utils.array_scan(self._starts, out_array=self._ranks, inclusive=True)
        m = int(read_scalar(self._ranks_tail, 0))
        if m == 0:
            return 0
        if m != self._rows:
            # Never taken after the first pass on any mesh measured: the duplicate-edge guard in
            # ``_resolve_flip_quad_guarded`` is what keeps the exactly-two-corner edge count fixed.
            self._allocate_rows(m)
        wp.launch(
            kernel_remesh.emit_flip_topology,
            dim=n,
            inputs=[
                self._faces,
                self._order,
                self._starts,
                self._ranks,
                self.adjacency,
                self.adjacency_edges,
                self.unshared,
            ],
            device=self._device,
        )
        return m

    def _allocate_rows(self, m: int) -> None:
        """(Re)allocate the ``m``-row tables and the hashed edge-claim table sized from them."""
        self._rows = m
        self.adjacency = twt.empty_2d((m, 2), wp.int32, device=self._device)
        self.adjacency_edges = twt.empty_2d((m, 2), wp.int32, device=self._device)
        self.unshared = twt.empty_2d((m, 2), wp.int32, device=self._device)
        self.flip = wp.empty(m, dtype=wp.bool, device=self._device)
        self.quad = twt.empty_2d((m, 4), wp.int32, device=self._device)
        table = 1
        while table < 4 * m + 1:
            table <<= 1
        self.edge_claim = wp.empty(table, dtype=wp.int32, device=self._device)
        self.edge_claim_mask = table - 1


def cluster_decimate(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    voxel_size: float | None = None,
    contraction: Literal["average", "closest"] = "average",
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Decimate by snapping vertices to a uniform voxel grid and welding each cell to one vertex.

    The one decimation scheme that is *naturally* parallel: there is no priority queue and no
    sequential dependence anywhere, so the whole thing is a handful of kernel launches whatever the
    mesh size. Every vertex is binned into a cell of width ``voxel_size``, each occupied cell
    becomes a single output vertex, faces are remapped onto those, and the faces that collapsed
    (two or three corners in the same cell) or duplicated are dropped.

    Ports MeshLab's ``meshing_decimation_clustering`` and Open3D's ``simplify_vertex_clustering``;
    the grid is anchored half a cell below the bounding box, which is Open3D's convention, so both
    libraries produce the same cell assignment for the same ``voxel_size``. The vertex output alone
    is also MeshLab's ``generate_sampling_clustered_vertex``.

    !!! warning "This does not preserve topology"
        Two sheets of the surface that pass within ``voxel_size`` of each other get welded
        together, and a thin feature narrower than a cell disappears. That is the *point* of the
        algorithm — it is a resampling, not a simplification — but it means the result can be
        non-manifold even when the input is not. Use
        [`isotropic_remesh`][triwarp.remesh.isotropic_remesh] when the topology matters and
        [`quadric_decimate`][triwarp.remesh.quadric_decimate] when a face budget does.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions on the target device.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    voxel_size
        Cell width. Defaults to ``1 %`` of the bounding-box diagonal, matching MeshLab's
        ``threshold`` default of ``1 %``. Larger cells decimate harder.
    contraction
        How each cell picks its output position:

        - ``"average"`` (default) — the mean of the cell's vertices, which is Open3D's
          ``SimplificationContraction.Average``. Smooths slightly and cannot land off the input's
          convex hull.
        - ``"closest"`` — the input vertex nearest the cell centre, which is MeshLab's
          ``'Closest to center'`` sampling. Keeps every output vertex *on* the input surface, so it
          is the right choice when the positions must stay exact (ties break to the lowest index).

    Returns
    -------
    vertices : wp.array[wp.vec3]
        One position per occupied cell that still carries a face, compacted from index zero.
    faces : wp.array[wp.int32]
        Flat ``3 * n_faces`` triangle index buffer, free of collapsed and duplicate faces.

    Raises
    ------
    ValueError
        If ``voxel_size`` is not positive, or ``contraction`` is not one of the two names.

    See Also
    --------
    [`isotropic_remesh`][triwarp.remesh.isotropic_remesh]
    [`triwarp.repair.remove_duplicated_vertices`][triwarp.repair.remove_duplicated_vertices]
    [`triwarp.grouping.unique_faces`][triwarp.grouping.unique_faces]
    [`triwarp.voxels.voxel_down_sample`][triwarp.voxels.voxel_down_sample]

    Notes
    -----
    Cells whose every face collapsed are dropped from the output, where Open3D keeps them as
    unreferenced vertices. So the face counts agree exactly and the vertex counts can differ by the
    number of such cells — usually zero, and never in a way that changes the surface.

    The binning goes through [`triwarp.voxels.cell_indices`][triwarp.voxels.cell_indices], but the
    *dedup* deliberately stays on [`triwarp.grouping.unique_rows`][triwarp.grouping.unique_rows]
    rather than moving onto a NanoVDB grid, which was measured back to back and rejected. The grid
    dedups 2.6-3.0x faster as a stage (0.34 against 0.90 ms), but it numbers the clusters
    leaf-major, so keeping today's vertex order costs a restoring sort that gives the whole
    advantage back: end to end over three icospheres at two cell widths, the grid with its own
    ordering is 0.80-0.88x of today (12-20 % faster) and the grid with today's ordering is
    0.94-1.02x, i.e. inside the session drift. Neither clears the bar for changing a public output
    convention, and the 12-20 % is the ceiling because the dedup is only a quarter of the call —
    ``_cluster_positions``, the face remap, ``submesh_from_face_mask``, ``unique_faces`` and
    ``remove_unreferenced_vertices`` are untouched by it.
    """
    if contraction not in ("average", "closest"):
        raise ValueError(f"contraction must be 'average' or 'closest', got {contraction!r}")

    device = vertices.device
    n_vertices = int(vertices.shape[0])
    n_faces = int(faces.shape[0]) // 3
    if n_vertices == 0 or n_faces == 0:
        return wp.clone(vertices), wp.clone(faces)

    # One definition of "the default voxel grid for these points", shared with ``triwarp.voxels``
    # so the two modules cannot drift on the cell size or on Open3D's half-cell anchor.
    voxel_size, origin = tw.voxels.resolve_voxel_grid(
        vertices, voxel_size, caller="cluster_decimate"
    )
    cells = tw.voxels.cell_indices(vertices, voxel_size, origin=origin)
    _unique_cells, labels = tw.grouping.unique_rows(cells, return_inverse=True)
    n_clusters = int(tw.reduce.max(labels)) + 1

    cluster_vertices = _cluster_positions(
        vertices, labels, n_clusters, origin, voxel_size, contraction
    )
    remapped = tw.array.remap_indices(faces, labels)
    keep_mask = wp.empty(n_faces, dtype=wp.bool, device=device)
    wp.launch(
        kernel_remesh.faces_with_distinct_indices,
        dim=n_faces,
        inputs=[remapped, keep_mask],
        device=device,
    )
    kept_vertices, kept_faces = tw.selection.submesh_from_face_mask(
        cluster_vertices, remapped, keep_mask
    )
    # Welding can map two distinct input faces onto the same triple, which would leave a duplicated
    # face rather than a manifold one, so the dedup is part of the algorithm rather than polish.
    out_vertices, out_faces, _remap = tw.repair.remove_unreferenced_vertices(
        kept_vertices, tw.grouping.unique_faces(kept_faces)
    )
    return out_vertices, out_faces


def _cluster_positions(
    vertices: wp.array[wp.vec3],
    labels: wp.array[wp.int32],
    n_clusters: int,
    origin: wp.vec3,
    voxel_size: float,
    contraction: Literal["average", "closest"],
) -> wp.array[wp.vec3]:
    """One representative position per occupied cell, by cell mean or by nearest-to-centre."""
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    if contraction == "average":
        out = wp.zeros(n_clusters, dtype=wp.vec3, device=device)
        counts = wp.zeros(n_clusters, dtype=wp.int32, device=device)
        wp.launch(
            kernel_remesh.cluster_accumulate,
            dim=n_vertices,
            inputs=[labels, vertices, out, counts],
            device=device,
        )
        counts_f32 = tw.array.astype(counts, wp.float32)
        wp.map(wp.div, out, counts_f32, out=out)
        return out

    min_distance = wp.full(n_clusters, float("inf"), dtype=wp.float32, device=device)
    wp.launch(
        kernel_remesh.cluster_min_center_distance,
        dim=n_vertices,
        inputs=[labels, vertices, origin, wp.float32(voxel_size), min_distance],
        device=device,
    )
    representative = wp.full(n_clusters, n_vertices, dtype=wp.int32, device=device)
    wp.launch(
        kernel_remesh.cluster_pick_closest,
        dim=n_vertices,
        inputs=[labels, vertices, origin, wp.float32(voxel_size), min_distance, representative],
        device=device,
    )
    return tw.array.gather(vertices, representative)


@overload
def quadric_decimate(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    target_faces: int | None = ...,
    target_ratio: float | None = ...,
    feature_angle: float = ...,
    max_iter: int = ...,
    return_index: Literal[False] = False,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]: ...
@overload
def quadric_decimate(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    target_faces: int | None = ...,
    target_ratio: float | None = ...,
    feature_angle: float = ...,
    max_iter: int = ...,
    return_index: Literal[True],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32]]: ...
def quadric_decimate(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    target_faces: int | None = None,
    target_ratio: float | None = None,
    feature_angle: float = 30.0,
    max_iter: int = 100,
    return_index: bool = False,
) -> (
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
    | tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32], wp.array[wp.int32]]
):
    """
    Simplify to a target face count by quadric-error edge collapses (Garland-Heckbert).

    The decimation to reach for when the requirement is a **face budget** rather than an edge
    length: [`isotropic_remesh`][triwarp.remesh.isotropic_remesh] targets a length and
    [`cluster_decimate`][triwarp.remesh.cluster_decimate] a voxel size, and neither lets a caller
    ask for "this mesh at 10 % of its triangles". This does, and it is the method every comparable
    library exposes for the purpose (MeshLab's ``meshing_decimation_quadric_edge_collapse``,
    ``igl.decimate``, Open3D's ``simplify_quadric_decimation``).

    Each vertex accumulates the area-weighted plane quadrics of its incident faces; the cost of
    collapsing an edge is the residual of the summed quadric at its own minimizer, which is also
    where the surviving vertex is placed. Cheap collapses are the ones that barely move the surface,
    so the flat regions go first and the features last — the property that makes this the standard
    method.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions on the target device. Never mutated.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    target_faces
        Desired face count. Mutually exclusive with ``target_ratio``; exactly one must be given.
        Values at or above the input count return a copy.
    target_ratio
        Desired face count as a fraction of the input's, so ``0.1`` is MeshLab's usual "10 %". Must
        be in ``(0, 1]``.
    feature_angle
        Dihedral angle in **degrees** above which an edge is a feature. Feature and boundary
        structure is preserved exactly as in
        [`isotropic_remesh`][triwarp.remesh.isotropic_remesh]: corners are frozen, a crease vertex
        only collapses along its own feature, and a crease is never dragged off it.
    max_iter
        Cap on collapse passes. Each pass commits a conflict-free independent set, so a large
        reduction needs many; the loop also stops early once the target is met or a pass commits
        nothing.
    return_index
        If ``True``, also return the two provenance maps below, which is how a per-vertex or
        per-face attribute survives the decimation.

    Returns
    -------
    vertices : wp.array[wp.vec3]
        Simplified vertex positions on ``vertices.device``, compacted from index zero.
    faces : wp.array[wp.int32]
        Flat ``3 * n_faces`` triangle index buffer. The count is ``<= target_faces`` but not
        necessarily equal to it — see Notes.
    vertex_index : wp.array[wp.int32]
        Only when ``return_index`` is ``True``: length ``n_input_vertices``, the **output** vertex
        each input vertex ended up in, or ``-1`` for an input vertex that survives in no output face
        (one that was already unreferenced). Many-to-one, since that is what a collapse is, so it is
        the direction a scatter or a segmented reduction wants.
    face_index : wp.array[wp.int32]
        Only when ``return_index`` is ``True``: length ``n_output_faces``, the **input** face each
        output face came from -- the same output-to-input direction
        [`split_edges`][triwarp.remesh.split_edges] uses, so a per-face attribute follows through
        ``tw.array.gather``. A collapse only deletes faces and never creates one, so every output
        face has exactly one source.

    Raises
    ------
    ValueError
        If neither or both of ``target_faces`` / ``target_ratio`` is given, ``target_faces`` is
        negative, or ``target_ratio`` is outside ``(0, 1]``.

    See Also
    --------
    [`cluster_decimate`][triwarp.remesh.cluster_decimate]
    [`isotropic_remesh`][triwarp.remesh.isotropic_remesh]
    [`triwarp.repair.collapse_small_triangles`][triwarp.repair.collapse_small_triangles]

    Notes
    -----
    **This is a batched-parallel greedy method, not the textbook serial one, and the difference is
    visible in the output.** Textbook QEM pops one edge at a time from a global priority queue,
    which is inherently sequential. Here each pass scores every edge, ranks the candidates by cost,
    and commits the cheapest *independent set* of them — two collapses may commit together only if
    their closed 1-rings are disjoint. So the sequence of collapses differs from a serial run's and
    the resulting triangulation is not the same mesh, even though both are driven by the same
    metric. Compare the two by deviation from the input rather than by equality.

    In exchange the *quality* is competitive and then some: at 512 faces from a subdivision-4
    icosphere the two-sided Hausdorff distance to the input is **0.0133 here against
    ``igl.decimate``'s 0.0250 and Open3D's 0.0236**, and the ordering holds at every target tried.
    Committing an independent set spreads the error over the surface where draining a queue
    concentrates it, and a max-norm rewards that.

    A pass commits **several independent sets against one scoring**, not one. A single hashed-key
    round takes on the order of ``m / 50`` of the candidates, because each winner locks the closed
    1-rings of both its endpoints; the geometry rebuild that would otherwise follow is ~40 wrapper
    calls, which is 92 % of this function's wall clock. So the pass retires only the candidates the
    previous round's commits actually invalidated — those whose closed 1-rings touch a collapsed
    neighbourhood, for which the cached quadric, cost, target position, link condition and
    normal-flip verdict are the only things that went stale — and runs another round until one finds
    nothing new. Worth **1.3-1.8x**, and it *improves* the deviation above at two of three targets
    (the max-norm moves by ±20 % run to run on a tied fixture in any case; see below).

    That round loop then runs **entirely on device**, as one ``wp.capture_while`` graph — see
    ``_run_collapse_rounds``. It was 73 % of this function's 3 948 ``wp.launch`` calls while
    committing ~65 collapses a round, so it was almost pure host marshalling; capturing it took
    the 240 per-round readbacks with it. Measured against the four CPU references as a control in
    the same run (they moved 0.91-1.07x, i.e. noise): **167 -> 147 ms at ``saddle`` 0.1, 209 -> 189
    at ``saddle_graded`` 0.1**, and 1.06-1.07x at 0.5. That is **1.06-1.13x**, not the ~1.2x a
    launch count alone predicted — the round loop's share of the *clock* was smaller than its share
    of the launches, and what is left is the per-pass rebuild, 44 of them, each ~45 wrapper calls
    whose cost is Python rather than either launches or kernels.

    **The per-pass cost did not depend on the mesh**: 4.35 ms at 32 524 faces against 4.15 at
    3 484, a 1.05x range over a 9.3x range of size, so the wrapper calls per rebuild — not the
    kernels, and not the pass count — were the lever. The largest single one was ``_classify``
    re-deriving what the pass had already grouped, which is why the pass groups its edges once,
    into an ``_EdgeIncidence``, and hands it over: measured back to back, **148 -> 112 ms at
    ``saddle`` 0.1, 193 -> 142 at ``saddle_graded`` 0.1, 47 -> 36 at ``saddle`` 0.5**, i.e.
    **1.31-1.36x**.

    The end of that road is ``_DecimationBuffers``, which removes the pass's host readbacks
    entirely and replays the whole rebuild as one captured graph — a further **2.6-4.5x**
    (``saddle`` at 0.1: 115 -> 37 ms, ``capsule``: 95 -> 21), and the point at which this function
    became the fastest of the five implementations benchmarked rather than 3.1x behind
    ``pyvista``. Read that class before changing anything here: the pass now runs at a **fixed
    width** with its live sizes in a device array, which is what makes the replay legal.
    What is *not* the lever, measured: more rounds (the loop saturates at ~5.3 of its 8-round cap),
    and narrowing the fixed width (worth 1.01x on the device, since these kernels sit at the launch
    floor — it is kept only because it is worth 1.3-1.7x on **CPU**, which has no graph to replay).

    **``return_index`` costs nothing when it is off and next to nothing when it is on.** The two
    provenance maps are folded per pass by two launches and one copy against the pass's ~77, and the
    branch that adds them is evaluated when the pass is *issued*, so the captured graph a CUDA run
    replays does not even contain it. Measured interleaved on ``icosphere(5)``, medians of 7:
    **1.00-1.07x on CUDA** (36.6 -> 39.0 ms at 0.1, 18.19 -> 18.20 at 0.5) and within noise on CPU,
    where the two directions disagreed (0.92x and 1.06x) -- i.e. unmeasurable against this
    function's own run-to-run spread.

    Four consequences to plan around:

    - **The target is usually reached exactly, but is not guaranteed.** A pass is budgeted at half
      the remaining surplus (an interior collapse removes two faces), shared across its rounds, and
      the loop stops early when a pass can commit nothing — a mesh whose remaining edges all fail
      the link condition or the normal-flip guard cannot be reduced further at any ``max_iter``.
      Check the returned face count if it matters.
    - Every collapse is checked against a **normal-flip guard**: an incident face whose normal would
      turn by more than ~78 degrees vetoes it. That is what keeps the output free of the inverted,
      self-intersecting triangles an unguarded quadric method produces at high reduction ratios, and
      it is the usual reason a target is not reached.
    - The independent set is chosen under a **hashed** lock key rather than by cost rank. That looks
      like a detail and is not: on a structured mesh both the edge index and the quadric cost are
      spatially monotone fields, and a monotone key has one local minimum, so either of those keys
      commits a single collapse per pass. See ``scramble_index`` in ``kernels/remesh.py`` for the
      measured numbers.
    - **The output is not bit-reproducible on a mesh with tied costs, and never was.** The
      vertex-face incidence CSR is built by an atomic counting scatter, so a row's order varies run
      to run; where two candidate edges tie on cost, which one the sort keeps varies with it. On the
      ``saddle`` grid at 10 % this moves the two-sided Hausdorff between 0.21 and 0.77 across
      identical runs, so **treat the max-norm as a band, not a value**: the mean deviation is stable
      to three digits over the same runs (0.0144-0.0149). Compare a change to this function on the
      mean, or on many repeats.
    """
    n_faces = int(faces.shape[0]) // 3
    target = _resolve_decimation_target(target_faces, target_ratio, n_faces)

    if n_faces == 0 or target >= n_faces:
        kept_vertices, kept_faces = wp.clone(vertices), wp.clone(faces)
        if not return_index:
            return kept_vertices, kept_faces
        device = faces.device
        return (
            kept_vertices,
            kept_faces,
            tw.array.arange(int(vertices.shape[0]), device),
            tw.array.arange(n_faces, device),
        )

    buffers = _DecimationBuffers(
        vertices, faces, target, wp.float32(math.radians(feature_angle)), track_index=return_index
    )
    for _ in range(max_iter):
        if not buffers.run_pass():
            break
    out_vertices, out_faces, face_source = buffers.result()
    if not return_index:
        return out_vertices, out_faces
    return out_vertices, out_faces, buffers.vertex_index, face_source


class _DecimationBuffers:
    """
    The whole working set of [`quadric_decimate`][triwarp.remesh.quadric_decimate], allocated once.

    A decimation pass is **flat in the mesh size** -- 3.13 ms at 32 258 faces and 2.82 ms at 3 227,
    a 1.11x range over a 10x one -- because only 0.81 ms of it is device work and the rest is the
    ~77 launches and ~40 wrapper calls that rebuild the geometry. Two measurements say what to do
    about that. A wrapper chain issued costs 2.69x what the same chain costs replayed from a CUDA
    graph, because a replay pays for no Python at all; and running every pass at the **pass-0
    width** instead of the live one costs 1.01x, because the device time barely tracks the size.
    So the pass is issued once against fixed-capacity buffers and replayed for every pass after.

    The only thing that stopped that was the host readbacks -- ``edges_unique``, ``flatnonzero`` and
    ``remove_unreferenced_vertices`` each read a count back to size their own output, and a
    ``memcpy DtoH`` inside a capture is CUDA error 906. Each is replaced here by the scan it was
    reading, with the count left in ``state`` on the device. Everything else the pass calls --
    [`edges_to_csr`][triwarp.graph.edges_to_csr] (so ``warp.sparse.bsr_from_triplets``),
    [`vertex_face_adjacency`][triwarp.adjacency.vertex_face_adjacency],
    [`sort_and_argsort`][triwarp.array.sort_and_argsort], ``warp.utils.array_scan`` and the round
    loop's own nested ``wp.capture_while`` -- was measured to capture and replay correctly on
    Warp 1.17 and is used unchanged.

    Padding is carried by two sentinels rather than by a guard in every kernel: a **dummy vertex**
    at index ``n_vertices`` that every padded face corner and edge endpoint points at, and a
    **dummy edge slot** at index ``n_edges`` that every padded corner's ``inverse`` entry points at.
    See the kernel section in ``kernels/remesh.py`` for why that is enough.

    Attributes
    ----------
    state : wp.array[wp.int32]
        ``[n_faces, n_vertices, n_edges]``, the live prefix lengths. The one array the host reads,
        once per pass, to decide whether to run another.
    """

    def __init__(
        self,
        vertices: wp.array[wp.vec3],
        faces: wp.array[wp.int32],
        target: int,
        feature: wp.float32,
        *,
        track_index: bool = False,
    ) -> None:
        """Allocate at the input's size, which bounds every later pass, and seed the live counts."""
        device = faces.device
        self._device: wp.Device = device
        self._target = target
        self._feature = feature
        self._track_index = track_index
        self._graph = None
        self._passes = 0
        self._retain: list = []
        self.n_faces = int(faces.shape[0]) // 3
        self.n_vertices = int(vertices.shape[0])
        self.n_corners = 3 * self.n_faces
        # An edge collapse removes at least three undirected edges and adds none, so the pass-0
        # count bounds every later one. It is not known until the first grouping runs, so the
        # capacity is the structural bound instead; the first pass then narrows nothing.
        self.n_edges = self.n_corners

        # The dummy vertex lives one past the capacity, so every per-vertex buffer is one longer.
        v_cap = self.n_vertices + 1
        self.vertices = wp.zeros(v_cap, dtype=wp.vec3, device=device)
        wp.copy(self.vertices, vertices, count=self.n_vertices)
        self.faces = wp.empty(self.n_corners, dtype=wp.int32, device=device)
        wp.copy(self.faces, faces, count=self.n_corners)
        self.state = wp.zeros(3, dtype=wp.int32, device=device)
        self.state.assign([self.n_faces, self.n_vertices, 0])

        n = self.n_corners
        self._keys = wp.empty(2 * n, dtype=wp.uint64, device=device)
        self._order = wp.empty(2 * n, dtype=wp.int32, device=device)
        self._starts = wp.empty(n, dtype=wp.int32, device=device)
        self._ranks = wp.empty(n, dtype=wp.int32, device=device)
        self._inverse = wp.empty(n, dtype=wp.int32, device=device)
        self._unique_edges = twt.empty_2d((self.n_edges, 2), wp.int32, device=device)
        # One row longer than the edge capacity: the dummy slot every padded corner scatters into.
        self._edge_face_count = wp.zeros(self.n_edges + 1, dtype=wp.int32, device=device)
        self._edge_faces = twt.empty_2d((self.n_edges + 1, 2), wp.int32, device=device)
        self._positions = wp.empty(v_cap, dtype=wp.vec3, device=device)
        self._vertex_remap = wp.empty(v_cap, dtype=wp.int32, device=device)
        self._face_flags = wp.empty(self.n_faces, dtype=wp.int32, device=device)
        self._face_ranks = wp.empty(self.n_faces, dtype=wp.int32, device=device)
        self._vertex_flags = wp.empty(self.n_vertices, dtype=wp.int32, device=device)
        self._vertex_ranks = wp.empty(self.n_vertices, dtype=wp.int32, device=device)
        self._csr_rows = wp.empty(2 * self.n_edges, dtype=wp.int32, device=device)
        self._csr_columns = wp.empty(2 * self.n_edges, dtype=wp.int32, device=device)
        self._csr_values = wp.ones(2 * self.n_edges, dtype=wp.float32, device=device)
        self._half = wp.empty(1, dtype=wp.int32, device=device)
        self._surplus = wp.empty(1, dtype=wp.int32, device=device)
        self._count = wp.zeros(1, dtype=wp.int32, device=device)

        # Provenance, only when a caller asked for it: one entry per *input* vertex composed pass by
        # pass (``compose_vertex_index``), and a column beside the face buffer compacted with it
        # (``compact_face_provenance``). Both are fixed width -- the vertex map by construction, the
        # face column because the face buffer is -- so tracking them does not stop the pass being
        # captured; the scratch exists because the compaction cannot read and write one buffer.
        self.vertex_index = (
            tw.array.arange(self.n_vertices, device)
            if track_index
            else wp.empty(0, dtype=wp.int32, device=device)
        )
        self.face_source = (
            tw.array.arange(self.n_faces, device)
            if track_index
            else wp.empty(0, dtype=wp.int32, device=device)
        )
        self._face_source_scratch = (
            wp.empty(self.n_faces, dtype=wp.int32, device=device)
            if track_index
            else wp.empty(0, dtype=wp.int32, device=device)
        )

    def run_pass(self) -> bool:
        """
        Run one decimation pass, and report whether another is worth running.

        The first pass is issued -- it is the one that measures the true edge count, which the
        allocation could only bound structurally at ``3 * n_faces`` -- and the second is captured
        at that narrower width and replayed by every pass after it. The one host readback per pass
        is here rather than in the pass body, and covers both the "target reached" and the
        "nothing legal left to collapse" exits.

        Without a conditional-graph device there is nothing to replay, so the width is re-tightened
        every pass instead and the pass is issued: that keeps the CPU path tracking the live mesh
        rather than paying the pass-0 width forever.
        """
        counts = self.state.numpy()
        if int(counts[0]) <= self._target or int(counts[1]) == 0:
            return False
        if self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            self._tighten_edges(int(counts[2]))
            if self._passes > 0 and self._device.is_cuda and wp.is_conditional_graph_supported():
                with wp.ScopedCapture(self._device) as capture:
                    self._issue_pass()
                self._graph = capture.graph
                wp.capture_launch(self._graph)
            else:
                self._issue_pass()
        self._passes += 1
        return int(read_scalar(self._count, 0)) != 0

    def _tighten_edges(self, edges: int) -> None:
        """
        Narrow the edge-indexed buffers to a bound the previous pass measured.

        The allocation can only bound the unique-edge count by ``3 * n_faces``, which is 2x the
        true value on a closed mesh -- and every per-edge kernel, the candidate cost sort and each
        round's sort run at that width. A collapse removes at least three undirected edges and adds
        none, so the previous pass's count bounds this one's.
        """
        if edges == 0 or edges >= self.n_edges:
            return
        device = self._device
        self.n_edges = edges
        self._unique_edges = twt.empty_2d((edges, 2), wp.int32, device=device)
        # One row longer than the edge capacity: the dummy slot every padded corner scatters into.
        self._edge_face_count = wp.zeros(edges + 1, dtype=wp.int32, device=device)
        self._edge_faces = twt.empty_2d((edges + 1, 2), wp.int32, device=device)
        self._csr_rows = wp.empty(2 * edges, dtype=wp.int32, device=device)
        self._csr_columns = wp.empty(2 * edges, dtype=wp.int32, device=device)
        self._csr_values = wp.ones(2 * edges, dtype=wp.float32, device=device)

    def result(self) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]:
        """Copy the live prefixes out of the fixed buffers, which is the only place a size leaks."""
        counts = self.state.numpy()
        n_faces, n_vertices = int(counts[0]), int(counts[1])
        vertices = wp.empty(n_vertices, dtype=wp.vec3, device=self._device)
        faces = wp.empty(3 * n_faces, dtype=wp.int32, device=self._device)
        if n_vertices > 0:
            wp.copy(vertices, self.vertices, count=n_vertices)
        if n_faces > 0:
            wp.copy(faces, self.faces, count=3 * n_faces)
        face_source = wp.empty(0, dtype=wp.int32, device=self._device)
        if self._track_index:
            face_source = wp.empty(n_faces, dtype=wp.int32, device=self._device)
            if n_faces > 0:
                wp.copy(face_source, self.face_source, count=n_faces)
        return vertices, faces, face_source

    def _issue_pass(self) -> None:
        """
        Issue every launch of one pass, in order. Called once, when the graph is captured.

        Every array this creates is kept in ``self._retain``. ``warp``'s ``Graph`` references the
        *modules* a captured launch needs but **not its arrays**, so on the face of it an
        intermediate dropped when this returns has its memory recycled and the replay then writes
        into whatever took its place. That was probed and **could not be reproduced** -- dropping
        the list and then churning the mempool with buffers of the intermediates' exact shapes and
        dtypes still replays correctly, which matches CUDA's graph-memory model, where an
        allocation made *during* capture becomes a memory node the graph owns. The list is kept
        anyway because it costs a few object references and the guarantee is the driver's rather
        than Warp's; do not remove it on the grounds that the probe was clean.
        """
        device = self._device
        dummy = wp.int32(self.n_vertices)
        incidence = self._group_edges()

        codes, _boundary = _classify(self.vertices, self.faces, self._feature, incidence)
        wp.launch(kernel_remesh.freeze_dummy_vertex, dim=1, inputs=[dummy, codes], device=device)
        csr = self._edge_csr(incidence.unique_edges)
        quadrics = _vertex_quadrics(self.vertices, self.faces)
        vertex_faces, face_offsets = tw.adjacency.vertex_face_adjacency(
            self.faces, n_vertices=self.n_vertices + 1
        )

        survivor = wp.empty(self.n_edges, dtype=wp.int32, device=device)
        removed = wp.empty(self.n_edges, dtype=wp.int32, device=device)
        target_pos = wp.empty(self.n_edges, dtype=wp.vec3, device=device)
        cost = wp.empty(self.n_edges, dtype=wp.float32, device=device)
        wp.launch(
            kernel_remesh.quadric_collapse_candidates,
            dim=self.n_edges,
            inputs=[
                incidence.unique_edges,
                self.vertices,
                self.faces,
                quadrics,
                codes,
                incidence.face_count,
                csr.offsets,
                csr.columns,
                face_offsets,
                vertex_faces,
                survivor,
                removed,
                target_pos,
                cost,
            ],
            device=device,
        )

        # Two-stage selection, and both stages matter.
        #
        # Stage one narrows the field to the cheapest *half* of the candidate edges, which is what
        # makes the method quadric-driven. Stage two picks a maximal independent set from those,
        # locking each winner's closed 2-ring under a **hashed** key -- see ``scramble_index`` for
        # why the obvious keys (edge index, or the cost itself) both collapse to one winner a pass
        # on a structured mesh.
        _sorted_cost, order = tw.array.sort_and_argsort(cost)
        wp.launch(
            kernel_remesh.collapse_pass_budgets,
            dim=1,
            inputs=[wp.int32(self._target), self.state, self._half, self._surplus],
            device=device,
        )
        wp.launch(
            kernel_remesh.drop_collapses_past_budget,
            dim=self.n_edges,
            inputs=[order, self._half, survivor],
            device=device,
        )

        # One independent set is a *fraction* of the candidates, not all of them: each winner locks
        # the closed 1-rings of both endpoints, so a hashed-key round commits roughly ``m / 50`` of
        # them and the geometry then gets rebuilt for the next fraction. That rebuild is ~40 wrapper
        # calls and is what the pass count multiplies, so run several rounds against the *same*
        # scoring, retiring only the candidates the previous round's commits invalidated
        # (``drop_locked_candidates`` states the disjointness argument that makes this exact rather
        # than approximate).
        candidates = wp.clone(survivor)
        locked = wp.zeros(self.n_vertices + 1, dtype=wp.int32, device=device)
        min_key = wp.empty(self.n_vertices + 1, dtype=wp.int32, device=device)
        claim = wp.empty(self.n_vertices + 1, dtype=wp.int32, device=device)
        remap = tw.array.arange(self.n_vertices + 1, device)
        wp.copy(self._positions, self.vertices)
        self._count.zero_()

        round_scratch = _run_collapse_rounds(
            device,
            self.n_edges,
            csr,
            candidates,
            removed,
            cost,
            target_pos,
            survivor,
            locked,
            min_key,
            claim,
            remap,
            self._positions,
            self._count,
            self._surplus,
            capture=False,
        )
        compaction_scratch = self._compact(remap, dummy)
        self._retain = [
            incidence,
            codes,
            _boundary,
            csr,
            quadrics,
            face_offsets,
            vertex_faces,
            survivor,
            removed,
            target_pos,
            cost,
            _sorted_cost,
            order,
            candidates,
            locked,
            min_key,
            claim,
            remap,
            round_scratch,
            compaction_scratch,
        ]

    def _group_edges(self) -> _EdgeIncidence:
        """
        Group the live face corners into unique edges, with no host readback.

        The same answer as [`_edge_incidence`][triwarp.remesh._edge_incidence] -- and in the same
        ascending-key edge order, which the lock keys depend on -- from one radix sort instead of
        ``edges_unique``'s hash table, a compaction scan and a second sort. Padded corners carry a
        maximal sentinel key so they sort past every real one.
        """
        device = self._device
        n = self.n_corners
        wp.launch(
            kernel_remesh.pass_edge_keys,
            dim=self.n_faces,
            inputs=[self.faces, self.state, wp.uint64(self.n_vertices + 1), self._keys],
            device=device,
        )
        wp.launch(
            kernel_array.init_sort_pair_indices,
            dim=2 * n,
            inputs=[wp.int32(n), wp.int32(-1), self._order],
            device=device,
        )
        wp.utils.radix_sort_pairs(self._keys, self._order, count=n)
        wp.launch(
            kernel_remesh.mark_unique_edge_starts,
            dim=n,
            inputs=[self._keys, self.state, self._starts],
            device=device,
        )
        wp.utils.array_scan(self._starts, out_array=self._ranks, inclusive=True)
        wp.launch(
            kernel_remesh.emit_unique_edges,
            dim=n,
            inputs=[
                self.faces,
                self._order,
                self._starts,
                self._ranks,
                self.state,
                wp.int32(self.n_edges),
                self._unique_edges,
                self._inverse,
                self.state,
            ],
            device=device,
        )
        wp.launch(
            kernel_remesh.pad_unique_edge_tail,
            dim=self.n_edges,
            inputs=[self.state, wp.int32(self.n_vertices), self._unique_edges],
            device=device,
        )
        self._edge_face_count.zero_()
        wp.launch(
            kernel_scatter.scatter_edge_incidence,
            dim=n,
            inputs=[self._inverse, self._edge_face_count, self._edge_faces],
            device=device,
        )
        return _EdgeIncidence(
            self._unique_edges, self._inverse, self._edge_face_count, self._edge_faces
        )

    def _edge_csr(self, unique_edges: twt.Array2dInt32) -> wps.BsrMatrix[wp.Scalar]:
        """
        Vertex-vertex adjacency CSR over the live edges.

        The matrix [`edges_to_csr`][triwarp.graph.edges_to_csr] builds, split out only to send the
        padded rows out of range rather than to the dummy vertex; see
        [`edge_csr_triplets`][triwarp.kernels.remesh.edge_csr_triplets] for why that matters.
        """
        wp.launch(
            kernel_remesh.edge_csr_triplets,
            dim=self.n_edges,
            inputs=[unique_edges, self.state, self._csr_rows, self._csr_columns],
            device=self._device,
        )
        return wps.bsr_from_triplets(
            self.n_vertices + 1,
            self.n_vertices + 1,
            self._csr_rows,
            self._csr_columns,
            self._csr_values,
            prune_numerical_zeros=False,
        )

    def _compact(self, remap: wp.array[wp.int32], dummy: wp.int32) -> list[wp.array]:
        """
        Rebuild the face and vertex buffers in place, publishing both new counts to ``state``.

        The tail of ``quadric_decimate``'s pass with its three host readbacks removed: the face
        compaction's ``flatnonzero`` and both of
        [`remove_unreferenced_vertices`][triwarp.repair.remove_unreferenced_vertices]'s become the
        inclusive scans they were reading.

        Returns the arrays it allocated, for [`_issue_pass`][triwarp.remesh._DecimationBuffers]
        to keep alive across the capture.
        """
        device = self._device
        remapped = tw.array.gather(remap, self.faces)
        valid = wp.empty(self.n_faces, dtype=wp.bool, device=device)
        wp.launch(
            kernel_remesh.faces_with_distinct_indices,
            dim=self.n_faces,
            inputs=[remapped, valid],
            device=device,
        )
        wp.utils.array_cast(valid, self._face_flags)
        wp.utils.array_scan(self._face_flags, out_array=self._face_ranks, inclusive=True)
        wp.launch(
            kernel_remesh.compact_faces,
            dim=self.n_faces,
            inputs=[remapped, self._face_flags, self._face_ranks, dummy, self.faces, self.state],
            device=device,
        )

        referenced = wp.zeros(self.n_vertices + 1, dtype=wp.bool, device=device)
        wp.launch(
            kernel_scatter.mark_membership_mask,
            dim=self.n_corners,
            inputs=[self.faces, wp.int32(self.n_vertices + 1), referenced],
            device=device,
        )
        wp.utils.array_cast(referenced[: self.n_vertices], self._vertex_flags)
        wp.utils.array_scan(self._vertex_flags, out_array=self._vertex_ranks, inclusive=True)
        wp.launch(
            kernel_remesh.compact_vertices,
            dim=self.n_vertices,
            inputs=[
                self._positions,
                self._vertex_flags,
                self._vertex_ranks,
                self.vertices,
                self._vertex_remap,
                self.state,
            ],
            device=device,
        )
        wp.launch(
            kernel_remesh.apply_vertex_remap,
            dim=self.n_corners,
            inputs=[self._vertex_remap, dummy, self.faces],
            device=device,
        )
        if self._track_index:
            # Both maps fold *this* pass into the running answer, so they run after the two
            # compactions that produced ``_face_ranks`` and ``_vertex_remap``.
            wp.copy(self._face_source_scratch, self.face_source)
            wp.launch(
                kernel_remesh.compact_face_provenance,
                dim=self.n_faces,
                inputs=[
                    self._face_source_scratch,
                    self._face_flags,
                    self._face_ranks,
                    self.face_source,
                ],
                device=device,
            )
            wp.launch(
                kernel_remesh.compose_vertex_index,
                dim=self.n_vertices,
                inputs=[remap, self._vertex_remap, self.vertex_index],
                device=device,
            )
        return [remapped, valid, referenced]


def _run_collapse_rounds(
    device: wp.Device,
    m: int,
    csr: wps.BsrMatrix[wp.Scalar],
    candidates: wp.array[wp.int32],
    removed: wp.array[wp.int32],
    cost: wp.array[wp.float32],
    target_pos: wp.array[wp.vec3],
    survivor: wp.array[wp.int32],
    locked: wp.array[wp.int32],
    min_key: wp.array[wp.int32],
    claim: wp.array[wp.int32],
    remap: wp.array[wp.int32],
    positions: wp.array[wp.vec3],
    count: wp.array[wp.int32],
    surplus: wp.array[wp.int32],
    capture: bool = True,
) -> list[wp.array]:
    """
    Commit independent sets of collapses against one scoring, until a round finds nothing new.

    Everything the loop decides with lives in two small device arrays -- ``budget``, and
    ``round_state``, the shared round-loop state (``kernels/array.py``'s ``LOOP_ROUND`` /
    ``LOOP_CONDITION``) with a third slot appended for the previous round's commit count -- so
    the body holds no host readback and the whole loop is a single ``wp.capture_while`` graph.

    That is the point of the shape. A round is ~12 launches over an ``m`` that is tens of
    thousands wide, and it commits only ~65 collapses (measured on ``saddle_graded`` at
    ``target_ratio=0.1``: 45 passes x 5.3 rounds for 15 682 collapses), so the round loop was **73%
    of this function's 3 948 launches** and the host marshalling — not the kernels, which are 11% of
    the call — was the cost. Capturing costs about what issuing the same launches costs (measured
    1.03-1.13x, so the break-even is ~1.1 replays), and this replays 5.3 times per capture.

    Every round runs the identical body, which is what makes one captured graph enough:

    - ``drop_locked_candidates`` runs on the first round too, where ``locked`` is all-zero and it
      restores ``survivor`` from ``candidates`` unchanged.
    - ``lock_collapse_neighborhoods`` runs on the last round too, where it only writes per-pass
      scratch nobody reads again.
    - the budget's floor of one applies while ``count`` is still zero rather than on "round 0".
      Those differ only if a round commits nothing, and then the floor cannot manufacture a commit
      anyway — the budget only ever *trims* an independent set that is already chosen.

    A budget-exhausted pass therefore stops the same way a saturated one does: the budget goes to
    zero, ``drop_collapses_past_budget`` retires everything, nothing commits, and
    ``end_collapse_round`` sees no progress.

    Returns
    -------
    list of wp.array
        This loop's own scratch, returned only so that a caller which is itself capturing
        (``capture=False``) can keep it alive; see ``_DecimationBuffers._issue_pass`` for why that
        is defensive rather than known to be required.
    """
    budget = wp.zeros(1, dtype=wp.int32, device=device)
    round_state = wp.zeros(kernel_remesh.COLLAPSE_STATE_SIZE, dtype=wp.int32, device=device)
    wp.launch(kernel_remesh.reset_collapse_rounds, dim=1, inputs=[round_state], device=device)
    # Sort scratch, allocated here rather than inside the body: ``radix_sort_pairs`` wants
    # double-width key and payload buffers, and a captured graph replays the *same* pointers, so the
    # scratch cannot be allocated per round.
    sort_keys = wp.empty(2 * m, dtype=wp.float32, device=device)
    sort_values = wp.empty(2 * m, dtype=wp.int32, device=device)

    def round_body() -> None:
        wp.launch(
            kernel_remesh.begin_collapse_round,
            dim=1,
            inputs=[surplus, count, budget],
            device=device,
        )
        wp.launch(
            kernel_remesh.drop_locked_candidates,
            dim=m,
            inputs=[candidates, removed, csr.offsets, csr.columns, locked, survivor],
            device=device,
        )
        min_key.fill_(INT32_MAX)
        wp.launch(
            kernel_remesh.claim_collapse_key,
            dim=m,
            inputs=[survivor, removed, csr.offsets, csr.columns, min_key],
            device=device,
        )
        claim.fill_(INT32_MAX)
        wp.launch(
            kernel_remesh.claim_collapse_index,
            dim=m,
            inputs=[survivor, removed, csr.offsets, csr.columns, min_key, claim],
            device=device,
        )
        # Writes the winners' costs straight into the sort's key buffer (+inf elsewhere, so every
        # non-winner ranks last), which is why nothing is copied between here and the sort.
        wp.launch(
            kernel_remesh.mark_collapse_winners,
            dim=m,
            inputs=[
                survivor,
                removed,
                csr.offsets,
                csr.columns,
                min_key,
                claim,
                cost,
                survivor,
                sort_keys,
            ],
            device=device,
        )
        # The set is already independent, so dropping members of it keeps it independent.
        wp.launch(
            kernel_array.init_sort_pair_indices,
            dim=2 * m,
            inputs=[wp.int32(m), wp.int32(-1), sort_values],
            device=device,
        )
        wp.utils.radix_sort_pairs(sort_keys, sort_values, m)
        wp.launch(
            kernel_remesh.drop_collapses_past_budget,
            dim=m,
            inputs=[sort_values, budget, survivor],
            device=device,
        )
        wp.launch(
            kernel_remesh.commit_selected_collapses,
            dim=m,
            inputs=[survivor, removed, target_pos, remap, positions, count],
            device=device,
        )
        wp.launch(
            kernel_remesh.lock_collapse_neighborhoods,
            dim=m,
            inputs=[survivor, removed, csr.offsets, csr.columns, locked],
            device=device,
        )
        wp.launch(
            kernel_remesh.end_collapse_round,
            dim=1,
            inputs=[wp.int32(_QUADRIC_ROUNDS), count, round_state],
            device=device,
        )

    condition = round_state[kernel_array.LOOP_CONDITION : kernel_array.LOOP_CONDITION + 1]
    if capture and device.is_cuda and wp.is_conditional_graph_supported():
        with wp.ScopedCapture(device) as capture_scope:
            wp.capture_while(condition, round_body)
        wp.capture_launch(capture_scope.graph)
    else:
        # ``capture=False`` when the caller is already capturing: a conditional graph nests, so the
        # round loop becomes an inner ``while`` node of the pass graph rather than its own.
        wp.capture_while(condition, round_body)
    return [budget, round_state, sort_keys, sort_values]


def _resolve_decimation_target(
    target_faces: int | None, target_ratio: float | None, n_faces: int
) -> int:
    """Validate the mutually exclusive target arguments and reduce them to a face count."""
    if (target_faces is None) == (target_ratio is None):
        raise ValueError("pass exactly one of target_faces and target_ratio")
    if target_faces is not None:
        if target_faces < 0:
            raise ValueError(f"target_faces must be non-negative, got {target_faces}")
        return int(target_faces)
    assert target_ratio is not None
    if not 0.0 < target_ratio <= 1.0:
        raise ValueError(f"target_ratio must be in (0, 1], got {target_ratio}")
    return math.ceil(target_ratio * n_faces)


def _vertex_quadrics(vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]) -> wp.array[wp.mat44d]:
    """Area-weighted sum of the incident faces' plane quadrics at each vertex, in float64."""
    device = vertices.device
    quadrics = wp.zeros(int(vertices.shape[0]), dtype=wp.mat44d, device=device)
    wp.launch(
        kernel_remesh.accumulate_face_quadrics,
        dim=int(faces.shape[0]) // 3,
        inputs=[vertices, faces, quadrics],
        device=device,
    )
    return quadrics


def flip_to_delaunay(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    region: wp.array[wp.bool] | None = None,
    max_angle_change: float | None = None,
    max_deviation: float | None = None,
    critical_aspect_ratio: float = 1000.0,
    max_iter: int = 100,
) -> wp.array[wp.int32]:
    """
    Improve triangle quality by flipping interior edges toward the Delaunay criterion.

    For every interior edge whose two incident faces are both in ``region``, the shared diagonal
    is flipped when doing so satisfies the local Delone (empty-circumcircle) test — subject to an
    optional dihedral-angle-change gate and a surface-deviation gate, so the flips never distort
    the surface. Rim edges (with a face
    outside the region, or on the mesh boundary) are never flipped. Vertices, face count and
    region membership are unchanged; only the triangulation of the region is rewritten.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    region
        Optional length-``n_faces`` ``wp.bool`` mask; only edges interior to the ``True`` faces
        are flippable. ``None`` treats the whole mesh as flippable.
    max_angle_change
        Maximum dihedral-angle change (radians) a flip may introduce. ``None`` disables the gate.
    max_deviation
        Maximum surface deviation a flip may introduce. ``None`` disables the gate.
    critical_aspect_ratio
        Triangle aspect ratio above which the dihedral-angle gate is lifted, so degenerate
        triangles can still be repaired.
    max_iter
        Maximum number of parallel flip passes.

    Returns
    -------
    wp.array[wp.int32]
        Flat face buffer with the region re-triangulated, on ``faces.device`` (a copy; the
        input is not modified).

    See Also
    --------
    [`subdivide_region_to_size`][triwarp.remesh.subdivide_region_to_size]
    [`delaunay_triangulation`][triwarp.reconstruction.delaunay_triangulation]
    [`face_adjacency`][triwarp.adjacency.face_adjacency]
    """
    device = faces.device
    setup = _flip_setup(faces, region)
    if setup is None:
        return wp.clone(faces)
    out_faces, n_vertices, region_flags = setup

    mac = wp.float32(max_angle_change if max_angle_change is not None else float(2.0 * math.pi))
    mdsq = wp.float32(max_deviation * max_deviation if max_deviation is not None else 3.0e38)
    car = wp.float32(critical_aspect_ratio)

    def launch(adjacency, adjacency_edges, unshared, sorted_keys, key_base, out_flip, out_quad):
        wp.launch(
            kernel_remesh.delone_flip_candidates,
            dim=int(adjacency.shape[0]),
            inputs=[
                vertices,
                out_faces,
                adjacency,
                adjacency_edges,
                unshared,
                region_flags,
                sorted_keys,
                key_base,
                mac,
                mdsq,
                car,
                out_flip,
                out_quad,
            ],
            device=device,
        )

    _flip_interior_edges(out_faces, n_vertices, launch, max_iter)
    return out_faces


_OBJECTIVE_QUALITY_METRICS = ("radius_ratio", "area_max_side", "mean_ratio")


def flip_by_objective(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    objective: Literal["planarity", "curvature", "t_vertex"] = "planarity",
    region: wp.array[wp.bool] | None = None,
    planar_angle: float = 1.0,
    metric: Literal["radius_ratio", "area_max_side", "mean_ratio"] = "area_max_side",
    aspect_threshold: float = 40.0,
    max_iter: int = 100,
) -> wp.array[wp.int32]:
    """
    Flip interior edges to optimize triangle shape or surface flatness, instead of the Delone test.

    Runs the same parallel flip engine as [`flip_to_delaunay`][triwarp.remesh.flip_to_delaunay] —
    independent-set selection over the face adjacency, iterated until no edge is a candidate — with
    a different predicate at the front. That is the whole port: the machinery was already there, and
    these two objectives are what MeshLab's ``meshing_edge_flip_by_planar_optimization`` and
    ``meshing_edge_flip_by_curvature_optimization`` put in front of it.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions. Never modified — only the triangulation changes.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    objective
        Which predicate to flip on:

        - ``"planarity"`` (default) — **shape only, surface preserved.** A quad is eligible when its
          two triangles meet within ``planar_angle`` of flat, and its diagonal is flipped when doing
          so raises the ``metric`` score of the *worse* of the two triangles. Because the quad is
          near-planar to begin with, the rewrite is a retriangulation and not a deformation. This is
          the one to reach for after any operation that leaves thin triangles across a flat region.
        - ``"curvature"`` — **flatness, surface changed.** The diagonal is flipped whenever the
          other one bends less, i.e. whenever the dihedral angle across it is smaller. This *does*
          move the surface (it chooses between two interpolations of the same four points) and is
          what makes a coarse triangulation of a curved shape follow its principal directions.
          ``planar_angle`` and ``metric`` are ignored.
        - ``"t_vertex"`` — **slivers only.** A quad is eligible only when one of its two triangles
          is a sliver: its ``aspect_ratio`` (circumradius over twice the inradius) exceeds
          ``aspect_threshold``; the diagonal is then flipped if that improves the worse of the two.
          A T-vertex — a vertex sitting in the interior of a neighbouring edge — is exactly what
          produces such a sliver, which is why this is the repair for one; see
          [`flip_t_vertices`][triwarp.repair.flip_t_vertices] for the wrapper that says so.
          ``planar_angle`` and ``metric`` are ignored.
    region
        Optional length-``n_faces`` ``wp.bool`` mask; only edges interior to the ``True`` faces are
        flippable. ``None`` treats the whole mesh as flippable.
    planar_angle
        Planarity tolerance in **degrees** for ``objective="planarity"``: a quad whose dihedral
        exceeds it is left alone. MeshLab's ``pthreshold``, whose default of ``1`` is this one. Must
        be in ``[0, 180]``.
    metric
        Which [`face_quality`][triwarp.triangles.face_quality] measure ``objective="planarity"``
        maximizes. Only the three larger-is-better shape measures are accepted — ``"aspect_ratio"``
        runs the other way and ``"area"`` is not a shape measure at all. MeshLab's ``planartype``,
        whose default ``'area/max side'`` is this one.
    aspect_threshold
        Sliver threshold for ``objective="t_vertex"``: only a quad whose worse triangle has an
        ``aspect_ratio`` above this is eligible. MeshLab's ``meshing_remove_t_vertices`` threshold,
        whose default of ``40`` is this one. Must be positive.
    max_iter
        Maximum number of parallel flip passes. Each pass commits a conflict-free independent set,
        so a mesh needing many local rewrites needs several.

    Returns
    -------
    wp.array[wp.int32]
        Flat face buffer with the region re-triangulated, on ``faces.device`` (a copy; the input is
        not modified).

    Raises
    ------
    ValueError
        If ``objective`` or ``metric`` is unknown, ``planar_angle`` is outside ``[0, 180]``, or
        ``region`` has the wrong length.

    See Also
    --------
    [`flip_to_delaunay`][triwarp.remesh.flip_to_delaunay]
    [`triwarp.triangles.face_quality`][triwarp.triangles.face_quality]
    [`isotropic_remesh`][triwarp.remesh.isotropic_remesh]

    Notes
    -----
    A quad whose two diagonals score *equally* — every quad of a regular grid — would flip back and
    forth forever, one pass each way, so a flip must beat the incumbent by a relative ``1e-6``
    rather than merely tie it. That margin is what makes ``max_iter`` a safety net rather than the
    normal stopping condition.
    """
    if objective not in ("planarity", "curvature", "t_vertex"):
        raise ValueError(
            f"objective must be 'planarity', 'curvature' or 't_vertex', got {objective!r}"
        )
    if aspect_threshold <= 0.0:
        raise ValueError(f"aspect_threshold must be positive, got {aspect_threshold}")
    if metric not in _OBJECTIVE_QUALITY_METRICS:
        raise ValueError(
            f"metric must be one of {list(_OBJECTIVE_QUALITY_METRICS)}, got {metric!r}"
        )
    if not 0.0 <= planar_angle <= 180.0:
        raise ValueError(f"planar_angle must be in [0, 180] degrees, got {planar_angle}")

    device = faces.device
    setup = _flip_setup(faces, region)
    if setup is None:
        return wp.clone(faces)
    out_faces, n_vertices, region_flags = setup

    objective_flag = {
        "planarity": kernel_remesh.OBJECTIVE_PLANARITY,
        "curvature": kernel_remesh.OBJECTIVE_CURVATURE,
        "t_vertex": kernel_remesh.OBJECTIVE_T_VERTEX,
    }[objective]
    metric_flag = tw.triangles._QUALITY_METRICS[metric]
    # The gate is on the dihedral's cosine so the kernel needs no inverse trigonometry.
    planar_cos = wp.float32(math.cos(math.radians(planar_angle)))

    def launch(adjacency, adjacency_edges, unshared, sorted_keys, key_base, out_flip, out_quad):
        wp.launch(
            kernel_remesh.objective_flip_candidates,
            dim=int(adjacency.shape[0]),
            inputs=[
                vertices,
                out_faces,
                adjacency,
                adjacency_edges,
                unshared,
                region_flags,
                sorted_keys,
                key_base,
                objective_flag,
                metric_flag,
                planar_cos,
                wp.float32(aspect_threshold),
                out_flip,
                out_quad,
            ],
            device=device,
        )

    _flip_interior_edges(out_faces, n_vertices, launch, max_iter)
    return out_faces


def _flip_setup(
    faces: wp.array[wp.int32], region: wp.array[wp.bool] | None
) -> tuple[wp.array[wp.int32], int, wp.array[wp.int32]] | None:
    """
    Working face buffer, vertex count and per-face region flags shared by the two flip drivers.

    Both [`flip_to_delaunay`][triwarp.remesh.flip_to_delaunay] and
    [`flip_by_objective`][triwarp.remesh.flip_by_objective] flip in place on a clone of the input
    and gate every candidate on the same ``int32`` region mask, which is all ones when the caller
    named no region.

    Parameters
    ----------
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    region
        Optional length-``n_faces`` ``wp.bool`` mask restricting which faces may flip.

    Returns
    -------
    tuple[wp.array[wp.int32], int, wp.array[wp.int32]] | None
        ``(out_faces, n_vertices, region_flags)``, or ``None`` for an empty mesh.

    Raises
    ------
    ValueError
        If ``region`` is given and is not length ``n_faces``.
    """
    device = faces.device
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return None
    if region is not None and int(region.shape[0]) != n_faces:
        raise ValueError(f"region must have length n_faces={n_faces}, got {int(region.shape[0])}")

    if region is None:
        region_flags = wp.full(n_faces, wp.int32(1), dtype=wp.int32, device=device)
    else:
        region_flags = tw.array.astype(region, wp.int32)
    return wp.clone(faces), tw.array.index_domain_size(faces), region_flags


def intrinsic_delaunay(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    epsilon: float = TOLERANCE_MOLLIFY,
    max_iter: int = 100,
) -> tuple[wp.array[wp.int32], twt.Array2dFloat32, int]:
    """
    Retriangulate to the intrinsic Delaunay triangulation, without moving a vertex.

    Flips edges whose two opposite angles sum past ``pi`` — exactly the edges whose cotangent weight
    is negative — until none is left. The flip is *intrinsic*: the new edge is not a straight line
    in space but the geodesic across the two triangles, and its length comes from unfolding them
    into a plane and measuring the other diagonal. The surface, its vertices and its metric are all
    untouched; only which pairs of vertices count as connected changes, so every operator built from
    the result is a better-behaved operator for the *same* geometry.

    Its practical effect is that the cotangent weights all become non-negative, which is what a
    Laplacian needs to satisfy a maximum principle: no spurious extrema, no negative diffusion, far
    better conditioned solves on a badly-shaped mesh.

    Flips run in parallel rounds, each committing a conflict-free independent set (the same engine
    behind [`flip_to_delaunay`][triwarp.remesh.flip_to_delaunay]) with the edge-length table carried
    alongside the connectivity.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer. Not modified.
    epsilon
        Mollification margin applied before flipping, relative to the mean edge length: a degenerate
        triangle has no well-defined angles to test.
    max_iter
        Cap on the number of parallel flip rounds.

    Returns
    -------
    intrinsic_faces : wp.array[wp.int32]
        Length-``3 * n_faces`` connectivity of the intrinsic triangulation, over the same vertices.
    edge_lengths : twt.Array2dFloat32
        ``(n_faces, 3)`` intrinsic edge lengths for those faces, column ``e`` opposite corner ``e``.
    n_flips : int
        How many edges were flipped. Zero means the input was already intrinsically Delaunay.

    Notes
    -----
    A flip that would duplicate an existing edge is skipped rather than allowed to create a
    multi-edge, so a few non-Delaunay edges can survive on coarse meshes — geometry-central's
    signpost machinery represents those, this does not. The count is small in practice: on the
    fixtures used in the tests the result matches ``igl.intrinsic_delaunay_cotmatrix`` exactly.

    See Also
    --------
    [`robust_laplacian`][triwarp.laplacian.robust_laplacian]
    [`mollify_intrinsic`][triwarp.laplacian.mollify_intrinsic]
    [`flip_to_delaunay`][triwarp.remesh.flip_to_delaunay]
    """
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    n_faces = int(faces.shape[0]) // 3
    lengths, _ = mollify_intrinsic(vertices, faces, epsilon=epsilon)
    intrinsic_faces = wp.clone(faces)
    if n_faces == 0:
        return intrinsic_faces, lengths, 0

    total = 0
    for _ in range(max_iter):
        edges_sorted = tw.edges.faces_to_edges(intrinsic_faces, sorted=True)
        adjacency, adjacency_edges = tw.adjacency.face_adjacency(
            intrinsic_faces, edges_sorted, return_edges=True, n_vertices=n_vertices
        )
        n_interior = int(adjacency.shape[0])
        if n_interior == 0:
            break
        unshared = tw.adjacency.face_adjacency_unshared(intrinsic_faces, adjacency, adjacency_edges)
        keys = tw.grouping.hash_indices_rows(edges_sorted, max_index=n_vertices, validate=False)
        sorted_keys, _order = tw.array.sort_and_argsort(keys)

        flip = wp.zeros(n_interior, dtype=wp.bool, device=device)
        quad = twt.empty_2d((n_interior, 4), wp.int32, device=device)
        new_length = wp.empty(n_interior, dtype=wp.float32, device=device)
        wp.launch(
            kernel_remesh.intrinsic_delaunay_candidates,
            dim=n_interior,
            inputs=[
                intrinsic_faces,
                lengths,
                adjacency,
                adjacency_edges,
                unshared,
                sorted_keys,
                wp.uint64(n_vertices),
                flip,
                quad,
                new_length,
            ],
            device=device,
        )

        # Independent set: a flip commits only if it wins both incident faces and its new edge's
        # hashed slot, so no two committed flips share a face or invent the same edge.
        table = 1
        while table < 4 * n_interior + 1:
            table <<= 1
        face_claim = wp.full(n_faces, INT32_MAX, dtype=wp.int32, device=device)
        edge_claim = wp.full(table, INT32_MAX, dtype=wp.int32, device=device)
        wp.launch(
            kernel_remesh.claim_flips,
            dim=n_interior,
            inputs=[
                flip,
                quad,
                adjacency,
                wp.int32(table - 1),
                wp.uint64(n_vertices),
                face_claim,
                edge_claim,
            ],
            device=device,
        )
        # Lengths first: this pass needs the *old* connectivity to know which corner holds which
        # vertex, and ``commit_flips`` is about to overwrite it.
        wp.launch(
            kernel_remesh.update_flipped_lengths,
            dim=n_interior,
            inputs=[
                intrinsic_faces,
                flip,
                quad,
                adjacency,
                new_length,
                face_claim,
                edge_claim,
                wp.int32(table - 1),
                wp.uint64(n_vertices),
                lengths,
            ],
            device=device,
        )
        count = wp.zeros(1, dtype=wp.int32, device=device)
        wp.launch(
            kernel_remesh.commit_flips,
            dim=n_interior,
            inputs=[
                flip,
                quad,
                adjacency,
                face_claim,
                edge_claim,
                wp.int32(table - 1),
                wp.uint64(n_vertices),
                intrinsic_faces,
                count,
            ],
            device=device,
        )
        committed = int(read_scalar(count, 0))
        total += committed
        if committed == 0:
            break
    return intrinsic_faces, lengths, total


def subdivide(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32]
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]:
    """
    Subdivide a mesh by splitting every face into four triangles.

    Each triangle is split by placing a new vertex at the midpoint of each
    edge. The four child triangles share these midpoints and preserve the
    original winding order, matching [`trimesh.remesh.subdivide`][] exactly.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        Length-``3 * n_faces`` flat triangle index buffer.

    Returns
    -------
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
        ``(new_vertices, new_faces)`` on ``vertices.device``.

    See Also
    --------
    [`trimesh.remesh.subdivide`][]
    """
    device = vertices.device
    n_vertices = int(vertices.shape[0])

    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        return vertices, faces

    unique_edges, inverse = tw.edges.edges_unique(faces, n_vertices=n_vertices)
    n_unique = int(unique_edges.shape[0])

    # Compute midpoint vertex for each unique edge
    out_midpoints = wp.empty(n_unique, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_remesh.compute_midpoints,
        dim=n_unique,
        inputs=[vertices, unique_edges, out_midpoints],
        device=device,
    )

    new_vertices, _ = tw.array.pack_1d_arrays([vertices, out_midpoints])
    return new_vertices, _split_faces_four(faces, inverse, n_vertices)


@overload
def subdivide_loop(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    *,
    return_operator: Literal[False] = False,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]: ...
@overload
def subdivide_loop(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], *, return_operator: Literal[True]
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wps.BsrMatrix[wp.float32]]: ...
def subdivide_loop(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], *, return_operator: bool = False
) -> (
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
    | tuple[wp.array[wp.vec3], wp.array[wp.int32], wps.BsrMatrix[wp.float32]]
):
    """
    Subdivide a mesh with one pass of Loop subdivision.

    Same 1-to-4 split as [`subdivide`][triwarp.remesh.subdivide] -- identical face table, identical
    index layout -- but the positions are the smooth Loop stencils rather than midpoints, so the
    surface is *approximated* instead of interpolated: original vertices move, and repeated
    application converges to a C² limit surface (C¹ at irregular vertices).

    - **Odd (edge) vertices**, one per unique edge: ``3/8`` on each endpoint and ``1/8`` on each of
      the two vertices opposite the edge. A boundary edge takes the midpoint instead.
    - **Even (original) vertices**, interior: ``(1 - n * beta) * v + beta * sum(ring)`` with
      **Warren's** ``beta`` -- ``3/16`` at valence 3, ``3/(8n)`` above -- which is the variant
      ``igl.loop`` uses, rather than Loop's original trigonometric weight.
    - **Even vertices on a boundary**: ``3/4 * v`` plus ``1/8`` of each of the two neighbours along
      the boundary, with interior neighbours excluded, so a boundary curve subdivides identically
      from either side of a seam.

    The mesh should be edge-manifold. Where it is not, the stencils are not defined and the
    fallbacks are conservative rather than arbitrary: an edge with three or more incident faces
    takes the midpoint rule, and a vertex where one or three-plus boundary edges meet -- or one with
    no edges at all -- keeps its position.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        Length-``3 * n_faces`` flat triangle index buffer.
    return_operator
        If ``True``, also return the sparse interpolation operator ``P`` this pass applies, so that
        ``new_vertices == P @ vertices`` and **any** per-vertex attribute can be carried through the
        subdivision by the same matrix (see
        [`triwarp.interpolation.transfer_through_operator`][triwarp.interpolation.transfer_through_operator]).

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        Positions on ``vertices.device``: the ``n_vertices`` relocated originals first and then one
        vertex per unique edge, so the leading ``n_vertices`` rows are the input vertex set
        *displaced* -- unlike ``subdivide``, where that prefix is unchanged.
    new_faces : wp.array[wp.int32]
        Flat ``3 * 4 * n_faces`` triangle index buffer.
    operator : warp.sparse.BsrMatrix
        Only when ``return_operator`` is ``True``: the ``(n_vertices + n_edges, n_vertices)``
        ``float32`` matrix of Loop weights, one row per output vertex in the same layout as
        ``new_vertices``. Every row sums to 1.

    Notes
    -----
    **Why an operator rather than an index map.** Every other topology edit here reports provenance
    as one ``int32`` per output element, because each output comes from exactly one input. A Loop
    vertex does not: an odd vertex is an affine combination of four inputs and an even vertex of its
    whole 1-ring, so no index map can express it, and an attribute cannot otherwise be carried
    through this function at all. The operator is the honest form, it is the standard prolongation
    object (``kernels/algorithms/multigrid.py`` builds one for a different purpose), and it costs
    nothing unless asked for.

    The operator is assembled from the same three grids and through the same two weight functions
    (``kernels/remesh.loop_odd_weights`` / ``loop_even_weights``) that the position kernels use, so
    the two cannot drift onto different surfaces. It is *not* used to compute the positions -- those
    stay two direct kernels, since a ``bsr_from_triplets`` build plus a ``bsr_mv`` would make every
    caller pay for the matrix.

    Four launches over three grids -- faces, unique edges, vertices -- plus the shared topology.
    Neither stencil needs an ordered 1-ring: the valence and the ring sum come from an atomic pass
    over the *unique* edges, which is the deduplicated neighbour count ``igl::loop`` reads off a
    sorted adjacency list, and the two boundary neighbours are found as the ones joined by boundary
    edges rather than as the ends of that list.

    For several passes, call this repeatedly -- that is what ``igl.loop``'s ``number_of_subdivs``
    does internally, and each pass multiplies the face count by four.

    See Also
    --------
    [`subdivide`][triwarp.remesh.subdivide]
    [`subdivide_to_size`][triwarp.remesh.subdivide_to_size]
    ``igl.loop``
    """
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    n_faces = int(faces.shape[0]) // 3
    if n_faces == 0:
        if return_operator:
            # No faces means no edges and no relocation, so the pass is the identity.
            return vertices, faces, wps.bsr_identity(n_vertices, wp.float32, device=device)
        return vertices, faces

    unique_edges, inverse = tw.edges.edges_unique(faces, n_vertices=n_vertices)
    n_unique = int(unique_edges.shape[0])

    # How many faces each edge carries, and the sum of the vertices opposite it.
    edge_opposite_sum = wp.zeros(n_unique, dtype=wp.vec3, device=device)
    edge_face_count = wp.zeros(n_unique, dtype=wp.int32, device=device)
    wp.launch(
        kernel_remesh.loop_edge_opposites,
        dim=n_faces,
        inputs=[vertices, faces, inverse, edge_opposite_sum, edge_face_count],
        device=device,
    )

    valence = wp.zeros(n_vertices, dtype=wp.int32, device=device)
    ring_sum = wp.zeros(n_vertices, dtype=wp.vec3, device=device)
    boundary_count = wp.zeros(n_vertices, dtype=wp.int32, device=device)
    boundary_sum = wp.zeros(n_vertices, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_remesh.loop_vertex_rings,
        dim=n_unique,
        inputs=[
            vertices,
            unique_edges,
            edge_face_count,
            valence,
            ring_sum,
            boundary_count,
            boundary_sum,
        ],
        device=device,
    )

    # One buffer sized for its final use, written through two views: the relocated originals in the
    # prefix and the new edge vertices after them, which is the index layout `_split_faces_four`
    # assumes and the one ``igl.loop`` returns.
    new_vertices = wp.empty(n_vertices + n_unique, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_remesh.loop_even_positions,
        dim=n_vertices,
        inputs=[vertices, valence, ring_sum, boundary_count, boundary_sum],
        outputs=[new_vertices[:n_vertices]],
        device=device,
    )
    wp.launch(
        kernel_remesh.loop_odd_positions,
        dim=n_unique,
        inputs=[vertices, unique_edges, edge_opposite_sum, edge_face_count],
        outputs=[new_vertices[n_vertices:]],
        device=device,
    )
    new_faces = _split_faces_four(faces, inverse, n_vertices)
    if not return_operator:
        return new_vertices, new_faces
    return (
        new_vertices,
        new_faces,
        _loop_operator(
            faces, unique_edges, inverse, edge_face_count, valence, boundary_count, n_vertices
        ),
    )


def _loop_operator(
    faces: wp.array[wp.int32],
    unique_edges: twt.Array2dInt32,
    inverse: wp.array[wp.int32],
    edge_face_count: wp.array[wp.int32],
    valence: wp.array[wp.int32],
    boundary_count: wp.array[wp.int32],
    n_vertices: int,
) -> wps.BsrMatrix[wp.float32]:
    """
    Assemble one Loop pass as a sparse interpolation matrix, from the pass's own intermediates.

    Three launches over the three grids the positions come from -- vertices, unique edges, faces --
    writing into one triplet buffer. ``bsr_from_triplets`` sums coincident entries, which is what
    lets the odd rows' 3/8 endpoints (edge grid) and 1/8 wings (face grid) be emitted independently.
    """
    device = faces.device
    n_faces = int(faces.shape[0]) // 3
    n_unique = int(unique_edges.shape[0])
    edge_base = n_vertices
    face_base = edge_base + 4 * n_unique
    rows, cols, values = tw.array.triplet_buffers(face_base + 3 * n_faces, wp.float32, device)

    wp.launch(
        kernel_remesh.loop_even_self_triplets,
        dim=n_vertices,
        inputs=[valence, boundary_count, rows, cols, values],
        device=device,
    )
    wp.launch(
        kernel_remesh.loop_edge_triplets,
        dim=n_unique,
        inputs=[
            unique_edges,
            edge_face_count,
            valence,
            boundary_count,
            wp.int32(n_vertices),
            wp.int32(edge_base),
            rows,
            cols,
            values,
        ],
        device=device,
    )
    wp.launch(
        kernel_remesh.loop_opposite_triplets,
        dim=n_faces,
        inputs=[
            faces,
            inverse,
            edge_face_count,
            wp.int32(n_vertices),
            wp.int32(face_base),
            rows,
            cols,
            values,
        ],
        device=device,
    )
    return wps.bsr_from_triplets(
        n_vertices + n_unique, n_vertices, rows, cols, values, prune_numerical_zeros=False
    )


def _split_faces_four(
    faces: wp.array[wp.int32], inverse: wp.array[wp.int32], n_vertices: int
) -> wp.array[wp.int32]:
    """
    Build the 1-to-4 face table both uniform subdivisions share, from the per-corner edge map.

    New vertex ``n_vertices + e`` belongs to unique edge ``e``, and corner ``j`` of face ``f`` spans
    ``(fv[j], fv[j + 1])``, so shifting ``inverse`` by ``n_vertices`` is the whole index translation
    [`subdivide`][triwarp.remesh.subdivide] and [`subdivide_loop`][triwarp.remesh.subdivide_loop]
    need before the split -- the two differ only in where they put the new positions.
    """
    device = faces.device
    n_faces = int(faces.shape[0]) // 3
    mid_idx_flat = wp.empty(3 * n_faces, dtype=wp.int32, device=device)
    wp.map(wp.add, inverse, wp.int32(n_vertices), out=mid_idx_flat)

    out_new_faces = wp.empty(n_faces * 12, dtype=wp.int32, device=device)
    wp.launch(
        kernel_remesh.subdivide_faces,
        dim=n_faces,
        inputs=[faces, mid_idx_flat.reshape((n_faces, 3)), out_new_faces],
        device=device,
    )
    return out_new_faces


@overload
def subdivide_to_size(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    max_edge: float | wp.array[wp.float32],
    max_iter: int = 10,
    return_index: Literal[False] = False,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]: ...
@overload
def subdivide_to_size(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    max_edge: float | wp.array[wp.float32],
    max_iter: int = 10,
    *,
    return_index: Literal[True],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]: ...
def subdivide_to_size(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    max_edge: float | wp.array[wp.float32],
    max_iter: int = 10,
    return_index: bool = False,
) -> (
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
    | tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]
):
    """
    Subdivide a mesh until every edge is at most ``max_edge`` long.

    Every edge longer than ``max_edge`` is bisected at a single shared midpoint,
    so the two faces on either side stay in sync and a watertight input stays
    watertight — no T-junctions (cracks) are introduced. Faces already small
    enough are left untouched. Each pass splits every over-long edge once and
    re-triangulates the incident faces with per-face templates (1, 2, or 3 split
    edges; the 2-split quad is cut along its shorter diagonal), iterating until
    no edge exceeds the threshold, matching
    [`trimesh.remesh.subdivide_to_size`][].

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device.
    faces
        Length-``3 * n_faces`` flat triangle index buffer.
    max_edge
        Maximum length of any edge in the result. A **scalar** gives the uniform target; a
        ``(n_vertices,)`` ``wp.float32`` array is a per-vertex **sizing field**, and an edge's own
        target is then the mean of its two endpoints', so the refinement is fine where the field is
        small and coarse where it is large. The field is *extended* to each inserted midpoint as the
        mean of the endpoints it splits, so no resampling is needed between passes and a field that
        satisfies the target cannot be driven past it by a later pass.
    max_iter
        Maximum number of subdivision passes. A ``ValueError`` is raised if the
        mesh still has an over-long edge after this many passes.
    return_index
        If ``True``, also return the source face index of each output face.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        Refined vertex positions on ``vertices.device`` (original vertices first,
        then the inserted edge midpoints).
    new_faces : wp.array[wp.int32]
        Flat buffer of the refined faces.
    index : wp.array[wp.int32]
        Only returned when ``return_index`` is ``True``: length ``n_out_faces``,
        the index of the original face each output face was refined from.

    Raises
    ------
    ValueError
        If any edge is still longer than ``max_edge`` after ``max_iter`` passes.

    See Also
    --------
    [`subdivide`][triwarp.remesh.subdivide]
    [`trimesh.remesh.subdivide_to_size`][]
    """
    device = vertices.device
    sizing = max_edge if isinstance(max_edge, wp.array) else None
    max_edge_f = wp.float32(0.0) if sizing is not None else wp.float32(max_edge)

    current_vertices = vertices
    current_faces = faces
    n_faces = int(faces.shape[0]) // 3
    index = tw.array.arange(n_faces, device)

    if n_faces == 0:
        if return_index:
            return current_vertices, current_faces, index
        return current_vertices, current_faces

    for i in range(max_iter + 1):
        n_vertices = int(current_vertices.shape[0])

        unique_edges, inverse = tw.edges.edges_unique(current_faces, n_vertices=n_vertices)
        m = int(unique_edges.shape[0])
        lengths = tw.edges.edges_unique_length(
            current_vertices, current_faces, unique_edges=unique_edges
        )

        # Flag the edges that are longer than the target length.
        long_mask = wp.empty(m, dtype=wp.bool, device=device)
        if sizing is None:
            wp.map(kernel_array.greater, lengths, max_edge_f, out=long_mask)
        else:
            wp.launch(
                kernel_remesh.mark_edges_over_sizing_field,
                dim=m,
                inputs=[unique_edges, lengths, sizing, long_mask],
                device=device,
            )

        # A sizing field must grow with the vertex buffer, and it is *extended* rather than
        # re-sampled: a midpoint's target is the mean of the endpoints it splits, which is the same
        # value the edge was tested against, so a run of passes cannot drift the field. Computed
        # before the split because it reads the pre-split edge rows.
        next_sizing = (
            None if sizing is None else _extend_sizing_field(sizing, unique_edges, long_mask)
        )

        # ``index`` rides through the split rather than being gathered afterwards, and the "did
        # anything split" test reads the resulting face count rather than reducing the mask: a face
        # count strictly grows when a mask is non-empty and is unchanged when it is empty, so this
        # is exact and costs nothing. Both points are why the refactor to ``split_edges`` left the
        # launch count per pass identical to the inline version it replaced (CLAUDE.md section 13).
        new_vertices, new_faces, new_index = split_edges(
            current_vertices,
            current_faces,
            long_mask,
            unique_edges=unique_edges,
            inverse=inverse,
            index=index,
            return_index=True,
        )
        # Every edge is short enough: we are done.
        if int(new_faces.shape[0]) == int(current_faces.shape[0]):
            break
        # Ran out of passes with over-long edges still present.
        if i >= max_iter:
            raise ValueError("max_iter exceeded!")
        current_vertices, current_faces, index, sizing = (
            new_vertices,
            new_faces,
            new_index,
            next_sizing,
        )

    if return_index:
        return current_vertices, current_faces, index
    return current_vertices, current_faces


def _extend_sizing_field(
    sizing: wp.array[wp.float32], unique_edges: twt.Array2dInt32, split_mask: wp.array[wp.bool]
) -> wp.array[wp.float32]:
    """
    Append one sizing value per edge about to be split: the mean of the edge's two endpoints.

    Called before the split rather than after, because it needs the *pre-split* edge rows, and the
    value it writes is exactly the target the edge was just tested against — so a midpoint inherits
    the size that justified inserting it and repeated passes converge instead of drifting.
    """
    device = sizing.device
    offsets, n_split = tw.array.counts_to_offsets(tw.array.astype(split_mask, wp.int32))
    if n_split == 0:
        return sizing
    appended = wp.empty(n_split, dtype=wp.float32, device=device)
    wp.launch(
        kernel_remesh.fill_edge_mean_sizing,
        dim=int(unique_edges.shape[0]),
        inputs=[sizing, unique_edges, split_mask, offsets, appended],
        device=device,
    )
    extended, _ = tw.array.pack_1d_arrays([sizing, appended])
    return extended


def subdivide_region_to_size(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    region: wp.array[wp.bool],
    max_edge: float,
    max_iter: int = 10,
    max_splits: int | None = None,
    delaunay: bool = True,
    max_angle_change: float | None = math.pi / 6.0,
    max_deviation: float | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.bool]]:
    """
    Subdivide only a face region until its edges are at most ``max_edge`` long.

    The region-restricted form of [`subdivide_to_size`][triwarp.remesh.subdivide_to_size], and the
    refinement stage of the smooth-patch pipeline: every edge with at least one incident region
    face and length
    greater than ``max_edge`` is bisected, the incident faces are re-triangulated crack-free
    (the [`subdivide_to_size`][triwarp.remesh.subdivide_to_size] 1/2/3-split templates, so faces
    outside the region that touch a split edge stay watertight), and — unless disabled — a
    parallel Delaunay edge-flip pass ([`flip_to_delaunay`][triwarp.remesh.flip_to_delaunay])
    improves the region triangulation after each pass. New vertices are appended after the
    originals, so the caller derives the new-vertex set as the index range
    ``[len(vertices), len(new_vertices))``.

    Where a sequential implementation would drive this from a longest-edge-first priority queue,
    splitting is done in parallel passes; ``max_splits`` is honoured as a soft budget by keeping
    only the longest eligible edges of the pass that would exceed it.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    region
        Length-``n_faces`` ``wp.bool`` mask; only edges touching a ``True`` face are refined.
    max_edge
        Target maximum edge length inside the region.
    max_iter
        Maximum number of subdivision passes.
    max_splits
        Optional soft cap on the total number of edge splits. ``None`` keeps
        splitting until convergence and raises if ``max_iter`` is exhausted first.
    delaunay
        When ``True`` (default), interleave and finish with the Delaunay flip pass.
    max_angle_change
        Dihedral-angle-change gate (radians) for the flip pass (default 30°). ``None`` disables
        the gate.
    max_deviation
        Surface-deviation gate for the flip pass. ``None`` disables it.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        Original vertices followed by the inserted midpoints, on ``vertices.device``.
    new_faces : wp.array[wp.int32]
        Flat buffer of the refined faces.
    new_region : wp.array[wp.bool]
        Length ``n_out_faces`` region mask; child faces inherit their parent's membership.

    Raises
    ------
    ValueError
        If ``region`` length does not match the face count, or if over-long region edges remain
        after ``max_iter`` passes and ``max_splits`` is ``None``.

    See Also
    --------
    [`subdivide_to_size`][triwarp.remesh.subdivide_to_size]
    [`flip_to_delaunay`][triwarp.remesh.flip_to_delaunay]
    [`fill_smooth`][triwarp.holes.fill_smooth]
    """
    device = vertices.device
    n_faces = int(faces.shape[0]) // 3
    if int(region.shape[0]) != n_faces:
        raise ValueError(f"region must have length n_faces={n_faces}, got {int(region.shape[0])}")
    if n_faces == 0:
        return vertices, faces, region

    max_edge_f = wp.float32(max_edge)
    current_vertices = vertices
    current_faces = faces
    region_flags = tw.array.astype(region, wp.int32)
    splits_done = 0

    for i in range(max_iter + 1):
        n_faces = int(current_faces.shape[0]) // 3
        n_vertices = int(current_vertices.shape[0])

        unique_edges, inverse = tw.edges.edges_unique(current_faces, n_vertices=n_vertices)
        m = int(unique_edges.shape[0])
        lengths = tw.edges.edges_unique_length(
            current_vertices, current_faces, unique_edges=unique_edges
        )

        edge_in_region = wp.zeros(m, dtype=wp.bool, device=device)
        wp.launch(
            kernel_remesh.mark_region_edges,
            dim=3 * n_faces,
            inputs=[region_flags, inverse, edge_in_region],
            device=device,
        )
        long_mask = wp.empty(m, dtype=wp.bool, device=device)
        wp.map(kernel_remesh.long_region_edge, lengths, max_edge_f, edge_in_region, out=long_mask)

        flags = tw.array.astype(long_mask, wp.int32)
        offsets, n_long = tw.array.counts_to_offsets(flags)

        if n_long == 0:
            break
        if i >= max_iter:
            if max_splits is None:
                raise ValueError("max_iter exceeded!")
            break

        if max_splits is not None:
            remaining = max_splits - splits_done
            if remaining <= 0:
                break
            if n_long > remaining:
                long_mask = _keep_longest_edges(long_mask, lengths, remaining, m, device)
                wp.utils.array_cast(long_mask, flags)
                offsets, n_long = tw.array.counts_to_offsets(flags)

        midpoint_idx = wp.empty(m, dtype=wp.int32, device=device)
        wp.launch(
            kernel_remesh.build_midpoint_index,
            dim=m,
            inputs=[long_mask, offsets, wp.int32(n_vertices), midpoint_idx],
            device=device,
        )
        new_mid = wp.empty(n_long, dtype=wp.vec3, device=device)
        wp.launch(
            kernel_remesh.fill_edge_midpoints,
            dim=m,
            inputs=[current_vertices, unique_edges, long_mask, offsets, new_mid],
            device=device,
        )
        current_vertices, _ = tw.array.pack_1d_arrays([current_vertices, new_mid])

        face_mid = tw.array.gather(midpoint_idx, inverse).reshape((n_faces, 3))
        out_faces = twt.empty_2d((n_faces * 4, 3), wp.int32, device=device)
        out_valid = wp.empty(n_faces * 4, dtype=wp.bool, device=device)
        out_slot_index = wp.empty(n_faces * 4, dtype=wp.int32, device=device)
        wp.launch(
            kernel_remesh.emit_size_faces,
            dim=n_faces,
            inputs=[
                current_faces,
                face_mid,
                current_vertices,
                region_flags,
                out_faces,
                out_valid,
                out_slot_index,
            ],
            device=device,
        )
        kept = tw.array.flatnonzero(out_valid)
        current_faces = tw.array.gather(out_faces, kept).reshape(-1)
        region_flags = tw.array.gather(out_slot_index, kept)
        splits_done += n_long

        if delaunay:
            _flip_region_faces(
                current_vertices, current_faces, region_flags, max_angle_change, max_deviation, 8
            )

    if delaunay:
        _flip_region_faces(
            current_vertices, current_faces, region_flags, max_angle_change, max_deviation, 50
        )

    new_region = tw.array.astype(region_flags, wp.bool)
    return current_vertices, current_faces, new_region


def refine_region_to_density(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    region: wp.array[wp.bool],
    *,
    max_iter: int = 10,
    alpha: float = math.sqrt(2.0),
    delaunay: bool = True,
    max_angle_change: float | None = math.radians(30.0),
    max_deviation: float | None = None,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.bool]]:
    """
    Refine a face region until its sampling matches the surrounding mesh's, not a target length.

    The density-driven sibling of
    [`subdivide_region_to_size`][triwarp.remesh.subdivide_region_to_size], and the difference is the
    criterion rather than the mechanism. That one bisects every region edge longer than a single
    global ``max_edge``; this one gives each vertex a **scale attribute** -- the average length of
    the edges incident to it -- and splits a region triangle only while its own scale is coarse
    relative to its corners'. On a uniformly sampled neighbourhood the two agree; on a *graded* one
    they do not, because a single length cannot be right at both ends of the grading.

    The rule is Liepa's (see Notes). Writing ``sigma(v)`` for the scale attribute, ``c`` for a
    triangle's centroid and ``sigma(c)`` for the mean of its three corners' attributes, the triangle
    is split at ``c`` when

        ``alpha * |c - v_m| > sigma(c)``  and  ``alpha * sigma(c) > sigma(v_m)``

    holds for every corner ``m``. The first clause refines; the second is what makes the process
    *terminate at the surrounding sampling* rather than at a tolerance. A pass that splits nothing
    ends the loop.

    The split is a **1 -> 3 centroid split**, which is what makes this cheap in parallel: the new
    vertex is interior to the triangle and no edge is divided, so there is nothing to agree with the
    neighbours about and no crack-free template is needed -- unlike edge bisection, which is why
    ``subdivide_region_to_size`` carries the 1/2/3 split families. One pass is two prefix scans and
    one kernel.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    region
        Length-``n_faces`` ``wp.bool`` mask; only ``True`` faces are refined. Child faces inherit
        their parent's membership.
    max_iter
        Cap on refinement passes. Unlike ``subdivide_region_to_size`` this does **not** raise when
        the cap is reached: the criterion is a density match rather than a hard bound, so stopping
        early leaves a coarser patch and not a wrong one.
    alpha
        The criterion's constant, ``sqrt(2)`` in the paper. Larger refines further, smaller stops
        sooner; it is the only real tuning knob here.
    delaunay
        When ``True`` (default), run the parallel Delone edge-flip pass over the region after each
        split pass, which is the relaxation step the paper pairs with the criterion.
    max_angle_change
        Dihedral-angle-change gate for that flip pass (default 30 degrees). ``None`` disables the
        gate.
    max_deviation
        Surface-deviation gate for the flip pass. ``None`` disables it.

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        Positions with the centroid vertices appended after the originals, so the caller derives the
        new-vertex set as the index range ``[len(vertices), len(new_vertices))``.
    new_faces : wp.array[wp.int32]
        Flat buffer of the refined faces.
    new_region : wp.array[wp.bool]
        Length ``n_out_faces`` region mask.

    Raises
    ------
    ValueError
        If ``region`` length does not match the face count.

    See Also
    --------
    [`subdivide_region_to_size`][triwarp.remesh.subdivide_region_to_size]
        The same operation driven by a target edge length instead.
    [`smoothing.refine_and_smooth_region`][triwarp.smoothing.refine_and_smooth_region]
        Where the two criteria are selected between, and what the hole fillers reach through.
    [`holes.fill_smooth`][triwarp.holes.fill_smooth]

    Notes
    -----
    The criterion is section 3 of P. Liepa, *"Filling holes in meshes"*, Eurographics/ACM SIGGRAPH
    Symposium on Geometry Processing (2003).

    The scale attribute is computed **once**, from the mesh as given, and only *extended* as
    vertices are added -- a centroid inherits the mean of its parents'. That is the point of it: it
    carries the surrounding sampling inward across the patch instead of being re-measured from the
    increasingly fine triangles it is producing, which would never converge.
    """
    device = vertices.device
    n_faces = int(faces.shape[0]) // 3
    if int(region.shape[0]) != n_faces:
        raise ValueError(f"region must have length n_faces={n_faces}, got {int(region.shape[0])}")
    if n_faces == 0:
        return vertices, faces, region

    alpha_f = wp.float32(alpha)
    current_vertices = vertices
    current_faces = faces
    current_region = region
    scale = _vertex_scale_attribute(vertices, faces, region)

    for _ in range(max_iter):
        n_faces = int(current_faces.shape[0]) // 3
        split = wp.empty(n_faces, dtype=wp.int32, device=device)
        wp.launch(
            kernel_remesh.mark_density_splits,
            dim=n_faces,
            inputs=[current_vertices, current_faces, current_region, scale, alpha_f, split],
            device=device,
        )
        split_offsets, n_split = tw.array.counts_to_offsets(split)
        if n_split == 0:
            break

        counts = wp.empty(n_faces, dtype=wp.int32, device=device)
        wp.map(kernel_remesh.face_split_count, split, out=counts)
        face_offsets, n_out_faces = tw.array.counts_to_offsets(counts)

        positions = wp.empty(n_split, dtype=wp.vec3, device=device)
        new_scale = wp.empty(n_split, dtype=wp.float32, device=device)
        out_faces = wp.empty(3 * n_out_faces, dtype=wp.int32, device=device)
        out_region = wp.empty(n_out_faces, dtype=wp.bool, device=device)
        wp.launch(
            kernel_remesh.emit_density_splits,
            dim=n_faces,
            inputs=[
                current_vertices,
                current_faces,
                current_region,
                scale,
                split,
                split_offsets,
                face_offsets,
                wp.int32(int(current_vertices.shape[0])),
                positions,
                new_scale,
                out_faces,
                out_region,
            ],
            device=device,
        )
        current_vertices, _ = tw.array.pack_1d_arrays([current_vertices, positions])
        scale, _ = tw.array.pack_1d_arrays([scale, new_scale])
        current_faces = out_faces
        current_region = out_region

        if delaunay:
            _flip_region_faces(
                current_vertices,
                current_faces,
                tw.array.astype(current_region, wp.int32),
                max_angle_change,
                max_deviation,
                8,
            )

    return current_vertices, current_faces, current_region


def _vertex_scale_attribute(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], region: wp.array[wp.bool]
) -> wp.array[wp.float32]:
    """
    Liepa's per-vertex scale attribute: the mean incident edge length in the *surrounding* mesh.

    "Surrounding" is load-bearing, not a synonym for "whole": the edges counted are those of the
    faces **outside** ``region``. Including the region's own edges is the natural-looking mistake
    and it defeats the criterion -- a minimum-weight patch spans its rim with long chords, so a rim
    vertex that happens to carry two of them reads a scale several times its neighbourhood's, the
    ``alpha * sigma(c) > sigma(v_m)`` clause fails there, and the patch is left unrefined. Measured
    on a 24-edge rim: counting the patch chords splits **nothing** at ``alpha = sqrt(2)`` where
    excluding them inserts the expected vertices.

    Over the **unique** edge list, so an interior edge counts once at each endpoint rather than
    twice -- which is why this goes through ``scatter_unique_edges_sum_and_valence`` rather than the
    half-edge form beside it. A vertex with no surrounding edge at all keeps ``0`` (the guarded
    division rather than ``nan``), which makes its clause fail and leaves its triangles alone; when
    the region is the *whole* mesh there is no surrounding mesh to measure and every edge is
    counted instead, so the criterion degrades to the mesh's own average rather than to zero.
    """
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    complement = wp.empty(int(region.shape[0]), dtype=wp.bool, device=device)
    wp.map(kernel_array.mask_not, region, out=complement)
    outside = tw.array.flatnonzero(complement)
    surrounding = (
        faces
        if int(outside.shape[0]) == 0
        else tw.array.gather(faces.reshape((-1, 3)), outside).reshape(-1)
    )
    unique_edges, _inverse = tw.edges.edges_unique(surrounding, n_vertices=n_vertices)
    lengths = tw.edges.edges_unique_length(vertices, surrounding, unique_edges=unique_edges)

    total = wp.zeros(n_vertices, dtype=wp.float32, device=device)
    valence = wp.zeros(n_vertices, dtype=wp.float32, device=device)
    wp.launch(
        kernel_scatter.scatter_unique_edges_sum_and_valence,
        dim=int(unique_edges.shape[0]),
        inputs=[unique_edges, lengths, total, valence],
        device=device,
    )
    wp.map(kernel_array.divide_if_positive, total, valence, out=total)
    return total


@overload
def split_edges(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    split_mask: wp.array[wp.bool],
    split_positions: wp.array[wp.vec3] | None = None,
    *,
    unique_edges: twt.Array2dInt32 | None = None,
    inverse: wp.array[wp.int32] | None = None,
    index: wp.array[wp.int32] | None = None,
    return_index: Literal[False] = False,
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32]]: ...
@overload
def split_edges(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    split_mask: wp.array[wp.bool],
    split_positions: wp.array[wp.vec3] | None = None,
    *,
    unique_edges: twt.Array2dInt32 | None = None,
    inverse: wp.array[wp.int32] | None = None,
    index: wp.array[wp.int32] | None = None,
    return_index: Literal[True],
) -> tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]: ...
def split_edges(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    split_mask: wp.array[wp.bool],
    split_positions: wp.array[wp.vec3] | None = None,
    *,
    unique_edges: twt.Array2dInt32 | None = None,
    inverse: wp.array[wp.int32] | None = None,
    index: wp.array[wp.int32] | None = None,
    return_index: bool = False,
) -> (
    tuple[wp.array[wp.vec3], wp.array[wp.int32]]
    | tuple[wp.array[wp.vec3], wp.array[wp.int32], wp.array[wp.int32]]
):
    """
    Split a chosen set of edges in one crack-free pass, inserting one vertex per edge.

    The primitive the whole ``subdivide_*`` family is built from, exposed because the *choice* of
    edges and the *position* of the new vertex are the only things that differ between its members:
    [`subdivide_to_size`][triwarp.remesh.subdivide_to_size] iterates this with a length test and
    midpoints, and
    [`triwarp.intersection.split_mesh_with_plane`][triwarp.intersection.split_mesh_with_plane] calls
    it once with the edges a plane crosses and the crossing points. A caller with a different
    criterion — a curvature threshold, a paint selection, an isovalue — needs no new machinery.

    Crack-free means the new vertex of an edge is inserted **once** and both incident faces
    reference it, so a watertight input stays watertight and no T-junction is introduced. Each face
    is re-triangulated by how many of its three edges were split: 1 gives two triangles, 2 gives
    three (the quad cut along its shorter diagonal), 3 gives the regular 1-to-4 split, and 0 passes
    through unchanged.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions on the target device. Never mutated.
    faces
        Length-``3 * n_faces`` ``wp.int32`` flat triangle index buffer.
    split_mask
        ``(n_edges,)`` ``wp.bool`` mask over the **unique undirected edges** in
        [`edges_unique`][triwarp.edges.edges_unique] order, ``True`` for each edge to split.
    split_positions
        Where to put each new vertex, as a ``(n_split,)`` ``wp.vec3`` array indexed by the
        **exclusive scan of** ``split_mask`` — that is, in ascending unique-edge order among the
        flagged edges, which is where a kernel writing ``out[offsets[e]]`` naturally puts them.
        ``None`` uses each edge's midpoint.
    unique_edges, inverse
        The [`edges_unique`][triwarp.edges.edges_unique] pair for ``faces``, when the caller has
        already built it to compute ``split_mask``. Both must be given together; either being
        ``None`` rebuilds them.
    index
        ``(n_faces,)`` ``wp.int32`` per-face values to carry through the split: each output face
        receives the value of the input face it came from. ``None`` means the identity, so
        ``return_index`` then reports provenance into ``faces``. Passing the *previous* pass's index
        is how an iterated caller composes provenance without a gather per pass.
    return_index
        If ``True``, also return ``index`` resolved onto the output faces (provenance into ``faces``
        when ``index`` is ``None``).

    Returns
    -------
    new_vertices : wp.array[wp.vec3]
        Original vertices followed by the ``n_split`` inserted ones, in ascending edge order.
    new_faces : wp.array[wp.int32]
        Flat ``3 * m`` triangle index buffer for the refined mesh.
    index : wp.array[wp.int32]
        Only when ``return_index`` is ``True``: length ``m``, the index into the **input** ``faces``
        of the face each output face came from.

    Raises
    ------
    ValueError
        If ``split_mask`` does not have one entry per unique edge, ``split_positions`` does not have
        one entry per flagged edge, or ``index`` does not have one entry per face.

    See Also
    --------
    [`subdivide_to_size`][triwarp.remesh.subdivide_to_size]
    [`subdivide`][triwarp.remesh.subdivide]
    [`triwarp.intersection.split_mesh_with_plane`][triwarp.intersection.split_mesh_with_plane]
    [`triwarp.edges.edges_unique`][triwarp.edges.edges_unique]

    Examples
    --------
    Splitting *every* edge is the regular 1-to-4 subdivision, so the face count quadruples:

    ```python
    unique_edges, inverse = tw.edges.edges_unique(f)
    every_edge = wp.full(int(unique_edges.shape[0]), True, dtype=wp.bool, device=f.device)
    fine_v, fine_f = tw.remesh.split_edges(
        v, f, every_edge, unique_edges=unique_edges, inverse=inverse
    )
    print(int(fine_f.shape[0]) // 3 == 4 * (int(f.shape[0]) // 3))
    ```
    """
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    n_faces = int(faces.shape[0]) // 3
    # Bound in one expression rather than an ``if`` that reassigns the parameters, so the optional
    # annotations narrow for the type checker without an ``assert``.
    edges, corner_edge = (
        (unique_edges, inverse)
        if unique_edges is not None and inverse is not None
        else tw.edges.edges_unique(faces, n_vertices=n_vertices)
    )
    n_edges = int(edges.shape[0])
    if int(split_mask.shape[0]) != n_edges:
        raise ValueError(
            f"split_mask must have one entry per unique edge ({n_edges}), "
            f"got {int(split_mask.shape[0])}."
        )

    carried = index if index is not None else tw.array.arange(n_faces, device)
    if int(carried.shape[0]) != n_faces:
        raise ValueError(
            f"index must have one entry per face ({n_faces}), got {int(carried.shape[0])}."
        )

    # The exclusive scan both counts the split edges and assigns each one its new vertex slot, which
    # is the indexing ``split_positions`` is documented against.
    offsets, n_split = tw.array.counts_to_offsets(tw.array.astype(split_mask, wp.int32))
    if n_split == 0 or n_faces == 0:
        if return_index:
            return wp.clone(vertices), wp.clone(faces), carried
        return wp.clone(vertices), wp.clone(faces)

    if split_positions is None:
        new_points = wp.empty(n_split, dtype=wp.vec3, device=device)
        wp.launch(
            kernel_remesh.fill_edge_midpoints,
            dim=n_edges,
            inputs=[vertices, edges, split_mask, offsets, new_points],
            device=device,
        )
    else:
        if int(split_positions.shape[0]) != n_split:
            raise ValueError(
                f"split_positions must have one entry per flagged edge ({n_split}), "
                f"got {int(split_positions.shape[0])}."
            )
        new_points = split_positions

    # Appended before face emission so the new indices resolve against one buffer.
    new_vertices, _ = tw.array.pack_1d_arrays([vertices, new_points])

    new_index = wp.empty(n_edges, dtype=wp.int32, device=device)
    wp.launch(
        kernel_remesh.build_midpoint_index,
        dim=n_edges,
        inputs=[split_mask, offsets, wp.int32(n_vertices), new_index],
        device=device,
    )
    face_new = tw.array.gather(new_index, corner_edge).reshape((n_faces, 3))

    # Emit up to four triangles per face into fixed slots, then compact.
    out_faces = twt.empty_2d((n_faces * 4, 3), wp.int32, device=device)
    out_valid = wp.empty(n_faces * 4, dtype=wp.bool, device=device)
    out_slot_index = wp.empty(n_faces * 4, dtype=wp.int32, device=device)
    wp.launch(
        kernel_remesh.emit_size_faces,
        dim=n_faces,
        inputs=[faces, face_new, new_vertices, carried, out_faces, out_valid, out_slot_index],
        device=device,
    )

    kept = tw.array.flatnonzero(out_valid)
    new_faces = tw.array.gather(out_faces, kept).reshape(-1)
    if return_index:
        return new_vertices, new_faces, tw.array.gather(out_slot_index, kept)
    return new_vertices, new_faces


def _flip_region_faces(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    region_flags: wp.array[wp.int32],
    max_angle_change: float | None,
    max_deviation: float | None,
    max_iter: int,
) -> int:
    """Run the parallel Delone flip pass over the region, mutating ``faces`` in place."""
    device = faces.device
    n_vertices = int(vertices.shape[0])
    mac = wp.float32(max_angle_change if max_angle_change is not None else float(2.0 * math.pi))
    mdsq = wp.float32(max_deviation * max_deviation if max_deviation is not None else 3.0e38)
    car = wp.float32(1000.0)

    def launch(adjacency, adjacency_edges, unshared, sorted_keys, key_base, out_flip, out_quad):
        wp.launch(
            kernel_remesh.delone_flip_candidates,
            dim=int(adjacency.shape[0]),
            inputs=[
                vertices,
                faces,
                adjacency,
                adjacency_edges,
                unshared,
                region_flags,
                sorted_keys,
                key_base,
                mac,
                mdsq,
                car,
                out_flip,
                out_quad,
            ],
            device=device,
        )

    return _flip_interior_edges(faces, n_vertices, launch, max_iter)


def _keep_longest_edges(
    long_mask: wp.array[wp.bool],
    lengths: wp.array[wp.float32],
    remaining: int,
    m: int,
    device: wp.DeviceLike,
) -> wp.array[wp.bool]:
    """
    Keep only the ``remaining`` longest edges currently flagged in ``long_mask``.

    This used to read ``long_mask`` and ``lengths`` back in full and pick the top ``remaining`` with
    ``numpy.argsort``, moving ``2 * m`` elements across the bus in a loop that otherwise moves four
    bytes a pass. Sorting the eligible lengths on the device removes both readbacks and the upload
    that returned the answer, the same spelling
    [`sample_surface_poisson_disk`][triwarp.sample.sample_surface_poisson_disk]'s final round uses.

    Measured interleaved, ``min`` of 9: **3.5-19x on CUDA** over the range this branch actually sees
    (1.95 -> 0.56 ms at ``m = 104 288``, the first budgeted pass on ``bunny``; 11.9 -> 0.62 ms at
    ``m = 10^6``), worth **1.14-1.21x on the whole call**. The device sort has a ~0.5 ms floor and
    so loses below ``m ~ 20 000``, and in isolation it loses 1.7-3.6x on CPU -- but the CPU call is
    flat (0.98-1.00x) because the branch is a small share of it there, so nothing regresses at the
    call level. ``m`` is the unique-edge count and grows with the mesh, which is what makes the
    readback the wrong side of the trade.

    Ties are not ordered by contract on either path -- ``numpy.argsort``'s introsort is unstable and
    ``sort_and_argsort`` is a stable radix sort, and the budget is documented as soft and as keeping
    "the longest eligible edges", which every tie-break satisfies equally. Measured, they in fact
    agree: the selected set is identical on three constructed tie patterns (all-equal, a cut inside
    the middle of three tie runs, a cut inside one long tie run) and the output mesh is byte-equal
    on ``icosphere(6)`` and ``bunny``.
    """
    eligible = tw.array.flatnonzero(long_mask)
    # Ascending on the negated length is descending on the length, and ``sort_and_argsort`` is the
    # package's one radix-sort spelling. ``order`` is a view, so it is cloned dense before gathering
    # through it (Warp ignores an index array's stride).
    descending = wp.empty(int(eligible.shape[0]), dtype=wp.float32, device=device)
    wp.map(wp.neg, tw.array.gather(lengths, eligible), out=descending)
    _sorted, order = tw.array.sort_and_argsort(descending)
    keep = tw.array.gather(eligible, wp.clone(order[:remaining]))
    return tw.array.indices_to_mask(keep, m, device=device)
