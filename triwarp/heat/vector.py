"""
The vector heat method: extend scalars, transport tangent vectors, and build logarithmic maps.

Short-time diffusion is a good approximation of "carry this value away from its source along the
surface" (Sharp, Soliman & Crane 2019). Diffusing a *scalar* needs the ordinary cotangent Laplacian;
diffusing a *tangent vector* needs the connection Laplacian
([`connection_laplacian`][triwarp.laplacian.connection_laplacian]), because the two endpoints of an
edge measure directions from different reference directions and a difference between them only means
something after transporting one into the other's frame.

Every function here returns tangent vectors as ``wp.vec2`` in each vertex's own frame from
[`vertex_tangent_frames`][triwarp.tangent_space.vertex_tangent_frames]. Use
[`tangent_to_world`][triwarp.heat.vector.tangent_to_world] to get 3D vectors — and note that
comparing 2D components against another library's is meaningless, since each library picks its own
reference direction per vertex.

All three solvers need conjugate gradient and are therefore CUDA-only, like
[`heat_geodesic`][triwarp.heat.distance.heat_geodesic].
"""

from __future__ import annotations

import warp as wp
import warp.sparse as wps

import triwarp as tw
import triwarp.linalg as twl
from triwarp._device import require_cuda
from triwarp.heat.distance import HeatOperators, heat_geodesic, heat_operators
from triwarp.kernels.heat import vector as kernel_heat_vector
from triwarp.laplacian import connection_laplacian, mass_matrix_entries
from triwarp.tangent_space import vertex_tangent_frames

_CG_TOLERANCE = 1e-8


# ``HeatOperators`` is imported directly rather than reached through ``tw.heat.distance``: this is
# evaluated at module scope, and while ``triwarp.heat.__init__`` is still executing the ``heat``
# attribute does not yet exist on the ``triwarp`` module.
VectorHeatOperators = tuple[
    wps.BsrMatrix[wp.float64],
    HeatOperators,
    tuple[wp.array[wp.vec3], wp.array[wp.vec3], wp.array[wp.vec3]],
]
"""What [`vector_heat_operators`][triwarp.heat.vector.vector_heat_operators] returns: the vector
heat system, the scalar [`heat_operators`][triwarp.heat.distance.heat_operators], and the frames."""


def vector_heat_operators(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], t: float | None = None
) -> VectorHeatOperators:
    """
    Assemble everything the vector-valued solvers need before their solves.

    Three pieces, none of which depends on a source:

    1. the **vector heat system** ``M + t * L_connection``, whose ``2 x 2`` blocks act on tangent
       vectors ([`connection_laplacian`][triwarp.laplacian.connection_laplacian]);
    2. the scalar [`heat_operators`][triwarp.heat.distance.heat_operators], for the magnitude
       extension and the distance field the log map needs;
    3. the [`vertex_tangent_frames`][triwarp.tangent_space.vertex_tangent_frames] every 2D component
       is measured in.

    Pass the result back through any solver's ``operators=`` argument to skip the assembly — most of
    the cost on a coarse mesh, and all of it when the solve converges quickly. That is the split
    ``potpourri3d.MeshVectorHeatSolver`` gets from being an object; here it stays a plain tuple.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    t
        Diffusion time for both the vector and the scalar systems. When ``None``, defaults to the
        squared mean edge length.

    Returns
    -------
    vector_system : warp.sparse.BsrMatrix
        ``M + t * L_connection`` in ``float64`` with ``wp.mat22d`` blocks.
    scalar : tuple
        The [`heat_operators`][triwarp.heat.distance.heat_operators] bundle for the same ``t``.
    frames : tuple[wp.array[wp.vec3], wp.array[wp.vec3], wp.array[wp.vec3]]
        ``(basis_x, basis_y, normal)`` per vertex.

    See Also
    --------
    [`transport_tangent_vectors`][triwarp.heat.vector.transport_tangent_vectors]
    [`log_map`][triwarp.heat.vector.log_map]
    [`heat_signed_distance`][triwarp.heat.signed.heat_signed_distance]
    """
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    if t is None:
        h = tw.edges.mean_edge_length(vertices, faces)
        t = h * h

    connection = connection_laplacian(vertices, faces)
    mass = mass_matrix_entries(vertices, faces, dtype=wp.float64)
    mass_blocks = wp.empty(n_vertices, dtype=wp.mat22d, device=device)
    if n_vertices > 0:
        wp.launch(
            kernel_heat_vector.block_mass, dim=n_vertices, inputs=[mass, mass_blocks], device=device
        )
    vector_system = wps.bsr_axpy(
        x=connection, y=wps.bsr_diag(diag=mass_blocks), alpha=float(t), beta=1.0
    )
    return vector_system, heat_operators(vertices, faces, t), vertex_tangent_frames(vertices, faces)


def extend_scalar(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    sources: wp.array[wp.int32],
    values: wp.array[wp.float64],
    t: float | None = None,
    operators: tw.heat.distance.HeatOperators | None = None,
) -> wp.array[wp.float64]:
    """
    Extend values from a few source vertices over the whole surface by nearest-source interpolation.

    Diffuses the values and an indicator of where they came from for the same short time, then
    divides one by the other. The ratio is what makes the result interpolate rather than decay: both
    numerator and denominator fall off away from the sources at the same rate, so their quotient
    stays close to the value of the nearest source, and blends smoothly where two sources compete.
    Matches ``potpourri3d.MeshVectorHeatSolver.extend_scalar``.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    sources
        ``(n_sources,)`` ``wp.int32`` source vertex indices.
    values
        ``(n_sources,)`` ``wp.float64`` value carried by each source.
    t
        Diffusion time; defaults to the squared mean edge length.
    operators
        Optional precomputed [`heat_operators`][triwarp.heat.distance.heat_operators] for this mesh.

    Returns
    -------
    wp.array[wp.float64]
        ``(n_vertices,)`` extended field on ``vertices.device``.

    Raises
    ------
    NotImplementedError
        On the CPU device: the diffusion is a conjugate-gradient solve, which
        ``warp.optim.linear.cg``
        cannot do on the CPU in Warp 1.14-1.15.

    See Also
    --------
    [`transport_tangent_vectors`][triwarp.heat.vector.transport_tangent_vectors]
    [`heat_geodesic`][triwarp.heat.distance.heat_geodesic]
    """
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    n_sources = int(sources.shape[0])
    if n_vertices == 0 or int(faces.shape[0]) == 0 or n_sources == 0:
        return wp.zeros(n_vertices, dtype=wp.float64, device=device)
    require_cuda(device, "extend_scalar")

    if operators is None:
        operators = heat_operators(vertices, faces, t)
    heat_system = operators[0]

    indicator = wp.zeros(n_vertices, dtype=wp.float64, device=device)
    weighted = wp.zeros(n_vertices, dtype=wp.float64, device=device)
    wp.launch(
        kernel_heat_vector.seed_source_scalars,
        dim=n_sources,
        inputs=[sources, values, indicator, weighted],
        device=device,
    )

    diffused_indicator = _solve_scalar(heat_system, indicator, n_vertices, device)
    diffused_values = _solve_scalar(heat_system, weighted, n_vertices, device)
    extended = wp.empty(n_vertices, dtype=wp.float64, device=device)
    wp.map(kernel_heat_vector.divide_positive, diffused_values, diffused_indicator, out=extended)
    return extended


def transport_tangent_vectors(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    sources: wp.array[wp.int32],
    vectors: wp.array[wp.vec2],
    t: float | None = None,
    operators: VectorHeatOperators | None = None,
) -> wp.array[wp.vec2]:
    """
    Parallel-transport tangent vectors from a few source vertices to every vertex.

    Three solves, following the vector heat method: the connection Laplacian diffuses the source
    vectors (which preserves their *directions* well but smears their magnitudes), while a scalar
    extension of the source magnitudes supplies the length. The result at each vertex is the source
    vector carried along the shortest path to it — the field a "drag this arrow across the surface"
    tool needs. Matches ``potpourri3d.MeshVectorHeatSolver.transport_tangent_vectors``.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    sources
        ``(n_sources,)`` ``wp.int32`` source vertex indices.
    vectors
        ``(n_sources,)`` tangent vectors, each in *its own source vertex's* frame.
    t
        Diffusion time; defaults to the squared mean edge length. Ignored when ``operators`` is
        given, which already fixes it.
    operators
        Optional precomputed
        [`vector_heat_operators`][triwarp.heat.vector.vector_heat_operators] for this mesh: they
        depend on the mesh alone, so passing them back skips the assembly on every call after the
        first.

    Returns
    -------
    wp.array[wp.vec2]
        ``(n_vertices,)`` transported vectors, each in that vertex's own frame. No frames need to be
        passed in: the components come out in the canonical frames of
        [`vertex_tangent_frames`][triwarp.tangent_space.vertex_tangent_frames] by construction (see
        [`connection_laplacian`][triwarp.laplacian.connection_laplacian]).

    Raises
    ------
    NotImplementedError
        On the CPU device (conjugate gradient; see
        [`extend_scalar`][triwarp.heat.vector.extend_scalar]).

    See Also
    --------
    [`log_map`][triwarp.heat.vector.log_map]
    [`connection_laplacian`][triwarp.laplacian.connection_laplacian]
    [`tangent_to_world`][triwarp.heat.vector.tangent_to_world]
    """
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    n_sources = int(sources.shape[0])
    if n_vertices == 0 or int(faces.shape[0]) == 0 or n_sources == 0:
        return wp.zeros(n_vertices, dtype=wp.vec2, device=device)
    require_cuda(device, "transport_tangent_vectors")

    if operators is None:
        operators = vector_heat_operators(vertices, faces, t)
    vector_system, scalar, _ = operators

    direction = _diffuse_from_sources(vector_system, sources, vectors, n_vertices, device)

    magnitudes = wp.empty(n_sources, dtype=wp.float64, device=device)
    wp.map(wp.length, _as_vec2d(vectors), out=magnitudes)
    extended = extend_scalar(vertices, faces, sources, magnitudes, operators=scalar)

    scaled = wp.empty(n_vertices, dtype=wp.vec2d, device=device)
    wp.map(kernel_heat_vector.scale_to_magnitude, direction, extended, out=scaled)
    transported = wp.empty(n_vertices, dtype=wp.vec2, device=device)
    wp.map(kernel_heat_vector.to_vec2, scaled, out=transported)
    return transported


def log_map(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    source: int,
    t: float | None = None,
    operators: VectorHeatOperators | None = None,
) -> wp.array[wp.vec2]:
    """
    Logarithmic map: every vertex's position in the source vertex's tangent plane.

    ``log_map(...)[v]`` is the 2D point in the *source's* frame whose length is the geodesic
    distance to ``v``, and whose direction is the initial direction of the geodesic that reaches
    ``v``. It is the inverse of the exponential map
    [`trace_geodesic_from_vertex`][triwarp.tracing.trace_geodesic_from_vertex]
    computes, and the standard way to lay out a local coordinate patch around a point.

    Assembled from two fields that are each cheap: the distance to the source
    ([`heat_geodesic`][triwarp.heat.distance.heat_geodesic]) gives the radius, and the source's
    reference direction parallel-transported outwards gives the angle — at any vertex the angle
    between that transported direction and the outward radial direction is exactly the angle at
    which the connecting geodesic left the source, because transport along that geodesic preserves
    it. This is
    the ``VectorHeat`` strategy in ``potpourri3d.MeshVectorHeatSolver.compute_log_map``; its
    ``AffineLocal`` and ``AffineAdaptive`` strategies solve a small dense problem per vertex and are
    deliberately not ported.

    Parameters
    ----------
    vertices
        ``(n_vertices,)`` mesh vertex positions.
    faces
        Length-``3 * n_faces`` ``wp.int32`` triangle index buffer.
    source
        Index of the vertex the map is centred on.
    t
        Diffusion time; defaults to the squared mean edge length. Ignored when ``operators`` is
        given.
    operators
        Optional precomputed
        [`vector_heat_operators`][triwarp.heat.vector.vector_heat_operators]. Pass the same bundle
        used elsewhere when the frames matter: the *angles* this function returns are measured from
        the source's ``basis_x``.

    Returns
    -------
    wp.array[wp.vec2]
        ``(n_vertices,)`` log-map coordinates in the source vertex's frame; ``(0, 0)`` at the
        source.
        On the cut locus — the antipode of a closed surface, where geodesics from the source arrive
        from every side — there is no direction to report, and the entry keeps the correct magnitude
        with an arbitrary angle.

    Raises
    ------
    NotImplementedError
        On the CPU device (conjugate gradient; see
        [`extend_scalar`][triwarp.heat.vector.extend_scalar]).

    See Also
    --------
    [`transport_tangent_vectors`][triwarp.heat.vector.transport_tangent_vectors]
    [`trace_geodesic_from_vertex`][triwarp.tracing.trace_geodesic_from_vertex]
    """
    device = vertices.device
    n_vertices = int(vertices.shape[0])
    if n_vertices == 0 or int(faces.shape[0]) == 0:
        return wp.zeros(n_vertices, dtype=wp.vec2, device=device)
    require_cuda(device, "log_map")

    if operators is None:
        operators = vector_heat_operators(vertices, faces, t)
    vector_system, scalar, frames = operators
    basis_x, basis_y, _ = frames

    sources = wp.array([source], dtype=wp.int32, device=device)
    # The source's own reference direction, transported outwards: this is the "which way was x?"
    # field the angle is measured against.
    reference = wp.array([[1.0, 0.0]], dtype=wp.vec2, device=device)
    transported = wp.empty(n_vertices, dtype=wp.vec2, device=device)
    wp.map(
        kernel_heat_vector.to_vec2,
        _diffuse_from_sources(vector_system, sources, reference, n_vertices, device),
        out=transported,
    )

    # Radial direction: the unit gradient of the distance field, averaged onto vertices and
    # expressed in each vertex's frame.
    distance = heat_geodesic(vertices, faces, sources, operators=scalar)
    _, _, _, normals, areas = scalar
    n_faces = int(faces.shape[0]) // 3
    face_gradient = wp.empty(n_faces, dtype=wp.vec3d, device=device)
    wp.launch(
        kernel_heat_vector.face_gradient_unit,
        dim=n_faces,
        inputs=[vertices, faces, normals, areas, distance, face_gradient],
        device=device,
    )
    vertex_gradient = wp.zeros(n_vertices, dtype=wp.vec3, device=device)
    wp.launch(
        kernel_heat_vector.scatter_face_field_to_vertices,
        dim=n_faces,
        inputs=[faces, areas, face_gradient, vertex_gradient],
        device=device,
    )
    radial = wp.empty(n_vertices, dtype=wp.vec2, device=device)
    wp.launch(
        kernel_heat_vector.world_to_tangent_unit,
        dim=n_vertices,
        inputs=[vertex_gradient, basis_x, basis_y, radial],
        device=device,
    )

    logarithm = wp.empty(n_vertices, dtype=wp.vec2, device=device)
    wp.launch(
        kernel_heat_vector.log_map_from_angles,
        dim=n_vertices,
        inputs=[radial, transported, distance, logarithm],
        device=device,
    )
    return logarithm


def tangent_to_world(
    tangent: wp.array[wp.vec2], basis_x: wp.array[wp.vec3], basis_y: wp.array[wp.vec3]
) -> wp.array[wp.vec3]:
    """
    Expand per-vertex tangent vectors into 3D using their frames.

    The only way to compare a tangent field against another library's: each library measures 2D
    components from its own reference direction, but ``a * basis_x + b * basis_y`` is the same 3D
    vector either way.

    Parameters
    ----------
    tangent
        ``(n_vertices,)`` tangent vectors in each vertex's frame.
    basis_x, basis_y
        The frames those components refer to, from
        [`vertex_tangent_frames`][triwarp.tangent_space.vertex_tangent_frames].

    Returns
    -------
    wp.array[wp.vec3]
        ``(n_vertices,)`` world-space vectors on ``tangent.device``.

    See Also
    --------
    [`vertex_tangent_frames`][triwarp.tangent_space.vertex_tangent_frames]
    """
    world = wp.empty(int(tangent.shape[0]), dtype=wp.vec3, device=tangent.device)
    wp.map(kernel_heat_vector.tangent_to_world, tangent, basis_x, basis_y, out=world)
    return world


def _diffuse_from_sources(
    system: wps.BsrMatrix[wp.float64],
    sources: wp.array[wp.int32],
    vectors: wp.array[wp.vec2],
    n_vertices: int,
    device: wp.DeviceLike,
) -> wp.array[wp.vec2d]:
    """Seed a tangent field at the source vertices, then diffuse it."""
    field = wp.zeros(n_vertices, dtype=wp.vec2d, device=device)
    wp.launch(
        kernel_heat_vector.seed_source_vectors,
        dim=int(sources.shape[0]),
        inputs=[sources, _as_vec2d(vectors), field],
        device=device,
    )
    return diffuse_tangent_field(system, field)


def diffuse_tangent_field(
    system: wps.BsrMatrix[wp.float64], source: wp.array[wp.vec2d]
) -> wp.array[wp.vec2d]:
    """
    Short-time diffusion of a tangent-vector field: solve ``(M + t L_connection) X = source``.

    Public because the source term is where the vector-valued methods differ from one another — a
    handful of vertices for parallel transport, a whole splatted curve for
    [`heat_signed_distance`][triwarp.heat.signed.heat_signed_distance] — while the solve is the same
    for all of them.

    Only the *directions* of the result carry meaning: magnitudes decay away from the source, and
    every caller replaces them, either with a scalar extension or by normalizing outright.

    Parameters
    ----------
    system
        The vector heat system from
        [`vector_heat_operators`][triwarp.heat.vector.vector_heat_operators].
    source
        ``(n_vertices,)`` ``wp.vec2d`` right-hand side, in each vertex's own tangent frame.

    Returns
    -------
    wp.array[wp.vec2d]
        ``(n_vertices,)`` diffused field on ``source.device``.

    See Also
    --------
    [`vector_heat_operators`][triwarp.heat.vector.vector_heat_operators]
    [`transport_tangent_vectors`][triwarp.heat.vector.transport_tangent_vectors]
    """
    n_vertices = int(source.shape[0])
    diffused = wp.zeros(n_vertices, dtype=wp.vec2d, device=source.device)
    if n_vertices == 0:
        return diffused
    twl.solve_spd(system, source, diffused, tol=_CG_TOLERANCE)
    return diffused


def _solve_scalar(
    system: wps.BsrMatrix[wp.float64],
    right_hand_side: wp.array[wp.float64],
    n_vertices: int,
    device: wp.DeviceLike,
) -> wp.array[wp.float64]:
    """Diffuse one scalar right-hand side through an already-assembled heat system."""
    solution = wp.zeros(n_vertices, dtype=wp.float64, device=device)
    twl.solve_spd(system, right_hand_side, solution, tol=_CG_TOLERANCE)
    return solution


def _as_vec2d(vectors: wp.array[wp.vec2]) -> wp.array[wp.vec2d]:
    """Widen a tangent field to float64, the precision the diffusion solves run in."""
    widened = wp.empty(int(vectors.shape[0]), dtype=wp.vec2d, device=vectors.device)
    wp.map(kernel_heat_vector.to_vec2d, vectors, out=widened)
    return widened
