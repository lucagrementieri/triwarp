import warp as wp


@wp.func
def loop_size(
    loop_starts: wp.array[wp.int32], total: wp.int32, n_loops: wp.int32, i: wp.int32
) -> wp.int32:
    # ``loop_starts`` is the exclusive scan of the loop sizes; the last loop ends at ``total``.
    end = total
    if i + 1 < n_loops:
        end = loop_starts[i + 1]
    return end - loop_starts[i]


@wp.kernel
def fan_faces(
    flat_loops: wp.array[wp.int32],
    loop_starts: wp.array[wp.int32],
    total: wp.int32,
    n_loops: wp.int32,
    out_faces: wp.array[wp.int32],
) -> None:
    ell = int(wp.tid())
    o = loop_starts[ell]
    s = loop_size(loop_starts, total, n_loops, ell)
    # This loop contributes s - 2 fan triangles; earlier loops occupy o - 2 * ell of them.
    base = o - 2 * ell
    for k in range(1, s - 1):
        t = base + (k - 1)
        out_faces[3 * t + 0] = flat_loops[o]
        out_faces[3 * t + 1] = flat_loops[o + k + 1]
        out_faces[3 * t + 2] = flat_loops[o + k]


@wp.kernel
def cone_faces(
    flat_loops: wp.array[wp.int32],
    loop_starts: wp.array[wp.int32],
    total: wp.int32,
    n_loops: wp.int32,
    n_vertices: wp.int32,
    out_faces: wp.array[wp.int32],
) -> None:
    ell = int(wp.tid())
    o = loop_starts[ell]
    s = loop_size(loop_starts, total, n_loops, ell)
    apex = n_vertices + ell
    # This loop contributes s cone triangles; the cone base equals o (scan of the loop sizes).
    for j in range(s):
        t = o + j
        nxt = o + (j + 1) % s
        out_faces[3 * t + 0] = apex
        out_faces[3 * t + 1] = flat_loops[nxt]
        out_faces[3 * t + 2] = flat_loops[o + j]


@wp.kernel
def loop_centroids(
    vertices: wp.array[wp.vec3],
    flat_loops: wp.array[wp.int32],
    loop_starts: wp.array[wp.int32],
    total: wp.int32,
    n_loops: wp.int32,
    out_centroids: wp.array[wp.vec3],
) -> None:
    ell = int(wp.tid())
    o = loop_starts[ell]
    s = loop_size(loop_starts, total, n_loops, ell)
    acc = wp.vec3(0.0, 0.0, 0.0)
    for j in range(s):
        acc = acc + vertices[flat_loops[o + j]]
    out_centroids[ell] = acc * (1.0 / wp.float32(s))


# --- Boundary-to-boundary zippering (``triangulate_boundaries`` / ``stitch``) ---------------


@wp.func
def _wrap(i: wp.int32, n: wp.int32) -> wp.int32:
    # Positive modulo: ``%`` follows C++11 semantics (sign of the dividend).
    return ((i % n) + n) % n


@wp.func
def searchsorted_right(edge: wp.array[wp.int32], n: wp.int32, value: wp.int32) -> wp.int32:
    # Number of entries in the non-decreasing ``edge[0:n]`` that are ``<= value``
    # (``numpy.searchsorted(..., side="right")``), via binary search.
    lo = int(0)  # noqa: UP018, RUF046 — int() declares a mutable Warp dynamic variable
    hi = int(n)
    while lo < hi:
        mid = (lo + hi) // 2
        if edge[mid] <= value:
            lo = mid + 1
        else:
            hi = mid
    return lo


@wp.kernel
def flip_loop(loop_a: wp.array[wp.int32], n_a: wp.int32, out_flipped: wp.array[wp.int32]) -> None:
    i = int(wp.tid())
    out_flipped[i] = loop_a[n_a - 1 - i]


@wp.kernel
def boundary_perimeters(
    a_pos: wp.array[wp.vec3],
    b_pos: wp.array[wp.vec3],
    n_a: wp.int32,
    out_perimeters: wp.array2d[wp.float32],
) -> None:
    i, j = wp.tid()
    edge_start = a_pos[i]
    edge_end = a_pos[_wrap(i + 1, n_a)]
    b = b_pos[j]
    out_perimeters[i, j] = wp.length(edge_start - b) + wp.length(edge_end - b)


@wp.kernel
def row_argmin(
    perimeters: wp.array2d[wp.float32],
    m_b: wp.int32,
    out_col: wp.array[wp.int32],
    out_val: wp.array[wp.float32],
) -> None:
    i = int(wp.tid())
    best_col = int(0)  # noqa: UP018, RUF046 — int() declares a mutable Warp dynamic variable
    best_val = perimeters[i, 0]
    for j in range(1, m_b):
        v = perimeters[i, j]
        if v < best_val:
            best_val = v
            best_col = j
    out_col[i] = best_col
    out_val[i] = best_val


@wp.kernel
def global_argmin(
    col_min: wp.array[wp.int32],
    val_min: wp.array[wp.float32],
    n_a: wp.int32,
    out_shift: wp.array[wp.int32],
) -> None:
    # Single-thread reduction: out_shift = (shift_a, shift_b).
    best_row = int(0)  # noqa: UP018, RUF046 — int() declares a mutable Warp dynamic variable
    best_val = val_min[0]
    for i in range(1, n_a):
        v = val_min[i]
        if v < best_val:
            best_val = v
            best_row = i
    out_shift[0] = best_row
    out_shift[1] = col_min[best_row]


@wp.kernel
def rolled_edge_map(
    col_min: wp.array[wp.int32],
    shift_a: wp.int32,
    shift_b: wp.int32,
    n_a: wp.int32,
    m_b: wp.int32,
    out_edge: wp.array[wp.int32],
) -> None:
    # Per-edge B vertex after rolling both loops so the global-min pair is first
    # (``argmin(roll(roll(perimeters, -shift_a, 0), -shift_b, 1), axis=1)``).
    i = int(wp.tid())
    out_edge[i] = _wrap(col_min[_wrap(i + shift_a, n_a)] - shift_b, m_b)


@wp.kernel
def resolve_corrections(
    perimeters: wp.array2d[wp.float32],
    unsorted_indices: wp.array[wp.int32],
    next_indices: wp.array[wp.int32],
    n_corrections: wp.int32,
    row_roll: wp.int32,
    col_roll: wp.int32,
    n_a: wp.int32,
    m_b: wp.int32,
    out_edge: wp.array[wp.int32],
) -> None:
    # Single-thread sequential correction: force ``out_edge`` non-decreasing by re-picking, for
    # each unsorted edge, the B vertex minimizing the perimeter within the bracket of its stable
    # neighbours. ``out_edge`` has length ``n_a + 1`` with the sentinel ``out_edge[n_a] == m_b``.
    # ``perimeters`` is the unrolled matrix, indexed through the running ``row_roll``/``col_roll``.
    for k in range(n_corrections):
        idx = unsorted_indices[k]
        lo = out_edge[idx - 1]
        hi = out_edge[next_indices[k]]
        if hi > m_b - 1:
            hi = m_b - 1
        row = _wrap(idx + row_roll, n_a)
        best_col = lo
        best_val = perimeters[row, _wrap(lo + col_roll, m_b)]
        for c in range(lo + 1, hi + 1):
            v = perimeters[row, _wrap(c + col_roll, m_b)]
            if v < best_val:
                best_val = v
                best_col = c
        out_edge[idx] = best_col


@wp.kernel
def rolled_loop_a(
    flipped_loop_a: wp.array[wp.int32],
    row_roll: wp.int32,
    n_a: wp.int32,
    out_loop: wp.array[wp.int32],
) -> None:
    i = int(wp.tid())
    out_loop[i] = flipped_loop_a[_wrap(i + row_roll, n_a)]


@wp.kernel
def rolled_loop_b(
    loop_b: wp.array[wp.int32],
    col_roll: wp.int32,
    vertex_offset: wp.int32,
    m_b: wp.int32,
    out_loop: wp.array[wp.int32],
) -> None:
    j = int(wp.tid())
    out_loop[j] = loop_b[_wrap(j + col_roll, m_b)] + vertex_offset


@wp.kernel
def bridge_a_faces(
    roll_loop_a: wp.array[wp.int32],
    roll_loop_b: wp.array[wp.int32],
    edge: wp.array[wp.int32],
    n_a: wp.int32,
    out_faces: wp.array[wp.int32],
) -> None:
    i = int(wp.tid())
    out_faces[3 * i + 0] = roll_loop_a[i]
    out_faces[3 * i + 1] = roll_loop_a[_wrap(i + 1, n_a)]
    out_faces[3 * i + 2] = roll_loop_b[edge[i]]


@wp.kernel
def bridge_b_faces(
    roll_loop_a: wp.array[wp.int32],
    roll_loop_b: wp.array[wp.int32],
    edge: wp.array[wp.int32],
    n_a: wp.int32,
    m_b: wp.int32,
    out_faces: wp.array[wp.int32],
) -> None:
    j = int(wp.tid())
    apex = roll_loop_a[searchsorted_right(edge, n_a, j) % n_a]
    # The B edge is reversed (``fliplr``) so the bridge winding matches mesh B's faces.
    out_faces[3 * j + 0] = roll_loop_b[_wrap(j + 1, m_b)]
    out_faces[3 * j + 1] = roll_loop_b[j]
    out_faces[3 * j + 2] = apex
