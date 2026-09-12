import warp as wp

from triwarp.constants import TOLERANCE_ZERO_CONSTANT
from triwarp.kernels.array import OverloadTable
from triwarp.kernels.triangles import triangle_cross


@wp.kernel
def face_crosses(
    vertices: wp.array[wp.vec3], faces: wp.array[wp.int32], out_cross: wp.array[wp.vec3]
) -> None:
    f = wp.int32(wp.tid())
    out_cross[f] = triangle_cross(vertices, faces, f)


@wp.kernel
def max_vertex_normal_weights(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    unit_face_normals: wp.bool,
    out_weights: wp.array2d[wp.float32],
) -> None:
    f = wp.int32(wp.tid())
    cross_len = wp.length(triangle_cross(vertices, faces, f))
    base = f * wp.int32(3)
    for c in range(3):
        i0 = faces[base + c]
        i1 = faces[base + (c + 1) % 3]
        i2 = faces[base + (c + 2) % 3]
        inv_denom = max_corner_inverse_edge_length_sq(vertices, i0, i1, i2)
        if unit_face_normals:
            out_weights[f, c] = cross_len * inv_denom
        else:
            out_weights[f, c] = inv_denom


@wp.func
def max_corner_inverse_edge_length_sq(
    vertices: wp.array[wp.vec3], i0: wp.int32, i1: wp.int32, i2: wp.int32
) -> wp.float32:
    """Per-corner MWSELR factor ``1 / (||e1||^2 * ||e2||^2)``."""
    e1 = vertices[i1] - vertices[i0]
    e2 = vertices[i2] - vertices[i0]
    denom = wp.length_sq(e1) * wp.length_sq(e2)
    if denom > TOLERANCE_ZERO_CONSTANT:
        return wp.float32(1.0) / denom
    return wp.float32(0.0)


@wp.kernel
def normalize_accumulated_rows(sums: wp.array2d[wp.Float], out_normals: wp.array[wp.vec3]) -> None:
    """
    Unit-normalize each row of an ``(n, 3)`` accumulator into a ``wp.vec3``.

    The tail of ``vertices._accumulate_and_normalize``, and the reason its accumulator can be
    ``float64`` while its answer is ``float32``: it narrows and normalizes in one pass, where the
    ``float32`` accumulator it replaced could reach the same answer with a zero-copy
    ``wp.utils.array_cast`` reinterpretation plus a ``wp.map(wp.normalize, ...)``. One launch
    instead of two, so the wider accumulator costs nothing here.

    Generic over the accumulator's precision, and paired with
    ``kernels.scatter.scatter_sum_vec`` / ``scatter_weighted_sum_vec``, which are generic over the
    same thing -- so the pair takes a second precision as a registration row rather than as a
    second copy of either kernel. The three components are carried as scalars rather than assembled
    into a vector because there is no rank-3 vector type to name generically in kernel scope, and
    the length is the same arithmetic either way.

    A zero row -- an unreferenced vertex, or a fan whose contributions cancel -- comes back as the
    zero vector, which is ``wp.normalize``'s own answer for one (its ``kEps`` is 0) and is the
    contract the callers document.
    """
    i = wp.int32(wp.tid())
    x = sums[i, 0]
    y = sums[i, 1]
    z = sums[i, 2]
    length = wp.sqrt(x * x + y * y + z * z)
    if length > sums.dtype(0.0):
        x = x / length
        y = y / length
        z = z / length
    out_normals[i] = wp.vec3(wp.float32(x), wp.float32(y), wp.float32(z))


# The accumulator precision ``vertices._accumulate_and_normalize`` allocates, and the only one
# registered: float64, so that the order the scatter's atomics pick cannot reach the answer
# (``kernels.scatter.atomic_add_vec3`` carries the measurement). Mirrors
# ``kernels/scatter._VECTOR_ACCUMULATOR_DTYPES``, which must list the same set -- the two kernels
# are launched back to back on one buffer.
_ACCUMULATOR_DTYPES = (wp.float64,)

NORMALIZE_ACCUMULATED_ROWS: OverloadTable


def _register_overloads() -> None:
    """Instantiate every concrete overload of this module's generic kernels."""
    global NORMALIZE_ACCUMULATED_ROWS
    NORMALIZE_ACCUMULATED_ROWS = OverloadTable(
        normalize_accumulated_rows,
        {d: [wp.array2d[d], wp.array[wp.vec3]] for d in _ACCUMULATOR_DTYPES},
    )


_register_overloads()
