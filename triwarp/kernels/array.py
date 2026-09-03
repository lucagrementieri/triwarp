from typing import Any

import warp as wp

from triwarp.constants import INT32_MAX_CONSTANT

# Slot table for the **device-side round loop**: one zero-initialized ``wp.array[wp.int32]`` that a
# ``dim=1`` kernel updates at the end of each round, so ``wp.capture_while`` can drive the rounds
# with no host readback. Slot 0 counts rounds -- read against a cap, which is what bounds a loop
# whose progress test could otherwise stall -- and slot 1 is the condition; the wrapper hands
# ``wp.capture_while`` the view ``state[LOOP_CONDITION : LOOP_CONDITION + 1]``.
#
# The condition is written by a **plain store**, never an atomic: it is one address taking one
# value, so there is nothing to serialize even where every thread of a wide launch may write it
# (``polyline.rdp_split_spans``). **The condition must be seeded non-zero before the loop starts**:
# ``wp.capture_while`` evaluates it *before* the first round, so a plain ``wp.zeros`` state runs
# zero rounds. Seed it with ``wp.array([0, 1])`` / ``assign([0, 1])``, or from the same ``dim=1``
# kernel that resets the rest of the pass (``remesh.reset_collapse_rounds``).
#
# Four loops share this: the level-synchronous Ramer-Douglas-Peucker and the ear-clipping rounds in
# ``kernels/polyline.py``, the conjugate gradient's iteration test, and ``kernels/remesh.py``'s
# collapse rounds -- which needs a third slot for the previous round's commit count and **appends**
# it, so the shared two keep their numbers. ``kernels/algorithms/bfs.py`` deliberately does not:
# its seven slots are a frontier window (``start``, ``tail``) rather than a round counter, so slot 0
# does not mean the same thing and renumbering it would buy a coincidence of indices, not a shared
# convention.
LOOP_ROUND = wp.constant(wp.int32(0))
LOOP_CONDITION = wp.constant(wp.int32(1))
LOOP_STATE_SIZE = 2


@wp.func
def sort3(a: wp.Scalar, b: wp.Scalar, c: wp.Scalar) -> tuple[wp.Scalar, wp.Scalar, wp.Scalar]:
    if a > b:
        a, b = b, a
    if b > c:
        b, c = c, b
    if a > b:
        a, b = b, a
    return a, b, c


@wp.func
def sign_with_tolerance(value: wp.Float, tolerance: wp.Float) -> wp.int32:
    # Sign of ``value`` with a dead zone of half-width ``tolerance`` around zero, which reads 0.
    #
    # **The dead zone is an argument and not a constant on purpose, and every pipeline that uses
    # this twice must use the same one both times.** A plane cut asks the question in two places --
    # which side is each *vertex* on, and does the plane cross each *edge*'s interior -- and the two
    # answers have to agree or the classifier flags a face the edge pass does not split. This lived
    # for a while as two functions, a fixed-``TOLERANCE_MERGE`` ``tolerance_sign`` for the
    # classifiers and this one for the edge mask, which put the coupling beyond the reach of a
    # reader of either: ``intersection.split_faces_along_field`` had to hardcode ``TOLERANCE_MERGE``
    # at its edge mask to match a classifier whose dead zone was invisible from the call site.
    #
    # The classifiers pass ``TOLERANCE_MERGE_CONSTANT`` because their public entry points
    # (``slice_mesh_with_plane``, ``clip_mesh_with_field``, ``split_faces_along_field``) expose no
    # tolerance and, per CLAUDE.md section 14, should not grow one until a caller needs it;
    # ``split_mesh_with_plane`` documents a ``tolerance=`` and passes it through. Both spellings are
    # now visible at the call site, which is the whole point.
    if value < -tolerance:
        return wp.int32(-1)
    if value > tolerance:
        return wp.int32(1)
    return wp.int32(0)


@wp.func
def wrap_index(i: wp.int32, n: wp.int32) -> wp.int32:
    # Positive modulo: ``%`` follows C++11 semantics (sign of the dividend).
    return ((i % n) + n) % n


@wp.func
def loop_next_slot(
    loop_id: wp.array[wp.int32],
    loop_starts: wp.array[wp.int32],
    loop_sizes: wp.array[wp.int32],
    slot: wp.int32,
) -> wp.int32:
    # Given a flat slot in a packed array of *closed* loops, the slot of the next element around
    # **its own** loop -- so the last element of a loop wraps to that loop's first and not into the
    # next loop's.
    #
    # Named because five kernels across ``boundary`` and ``holes`` had written this arithmetic out,
    # in three spellings that a duplicate scan keying on statement text cannot connect: with the
    # size in a local, with the size inlined into the ``wrap_index`` call, and with ``begin`` /
    # ``size`` / ``next_slot`` in place of ``o`` / ``b``. Every one of them is one edge of a rim,
    # and getting the wrap wrong silently joins two different holes.
    begin = loop_starts[loop_id[slot]]
    return begin + wrap_index(slot - begin + 1, loop_sizes[loop_id[slot]])


@wp.func
def update_argmin(
    best_value: wp.ref[wp.float32], best_index: wp.ref[wp.int32], value: wp.float32, index: wp.int32
):
    # Running min-with-index update in place. Callers must be compiled with
    # ``enable_backward=False`` (``wp.ref`` helpers have no adjoint). Concrete ``float32``:
    # ``wp.ref[wp.Scalar]`` generics do not instantiate through Warp 1.17 -- re-probed there,
    # still a ``WarpCodegenError`` at kernel parse ("Couldn't find function overload") -- so float64
    # sites keep a hand-written loop; the index/tag stays ``int32``.
    if value < best_value:
        best_value = value
        best_index = index  # noqa: F841 — writes through the wp.ref parameter


@wp.func
def update_argmax(
    best_value: wp.ref[wp.float32], best_index: wp.ref[wp.int32], value: wp.float32, index: wp.int32
):
    # Running max-with-index mirror of ``update_argmin`` (same ``enable_backward=False`` rule).
    if value > best_value:
        best_value = value
        best_index = index  # noqa: F841 — writes through the wp.ref parameter


@wp.func
def update_argmax_lowest_index(
    best_value: wp.ref[wp.float32], best_index: wp.ref[wp.int32], value: wp.float32, index: wp.int32
):
    # argmax with a lowest-index tie-break (matches the label-vote rule in ``rasterize_labels``).
    if value > best_value or (value == best_value and index < best_index):
        best_value = value
        best_index = index


@wp.func
def update_argmin_pair(
    best_value: wp.ref[wp.float32],
    first: wp.ref[wp.int32],
    second: wp.ref[wp.int32],
    value: wp.float32,
    a: wp.int32,
    b: wp.int32,
):
    # Running argmin carrying a pair of associated indices (shortest-edge endpoints).
    if value < best_value:
        best_value = value
        first = a  # noqa: F841 — writes through the wp.ref parameter
        second = b  # noqa: F841 — writes through the wp.ref parameter


@wp.func
def tile_argmin(value: wp.Float, index: wp.int32) -> tuple[wp.Float, wp.int32]:
    # The block-cooperative counterpart of ``update_argmin``: given one candidate per lane, the
    # smallest value over the block and the **lowest index among the lanes attaining it**. Every
    # lane holds the same pair afterwards, so any lane may store it.
    #
    # The two-stage form is what makes the winner independent of *which* lane saw it, and that is
    # the whole reason this is one function rather than three: a block reduction hands back a value
    # and not the lane that held it, so recovering the index is a second reduction with a tie-break,
    # and the three sites that need it (the hole-filling DP's apex choice, ball pivoting's pivot
    # search, ``proximity``'s straggler faces) were writing that rule out by hand in three
    # spellings. A duplicated *decision rule* diverges silently where duplicated arithmetic only
    # reads badly -- and it had: one of the three carried a trailing ``INT32_MAX -> -1`` fixup that
    # cannot fire, since at least one lane always attains the minimum and therefore contributes its
    # own index, and when no lane found anything every lane holds the caller's sentinel already.
    #
    # No ``wp.ref``, so unlike ``update_argmin`` this imposes no ``enable_backward=False`` on its
    # callers. Verified generic on Warp 1.17 at ``float32`` and ``float64``, on both devices, at
    # ``block_dim`` 1 / 32 / 64 / 256 -- including the CPU device, where ``wp.launch_tiled`` runs
    # one lane per block and both tiles hold that lane's own pair.
    block_value = wp.tile_min(wp.tile(value))[0]
    attained = wp.where(value == block_value, index, INT32_MAX_CONSTANT)
    return block_value, wp.tile_min(wp.tile(attained))[0]


@wp.func
def cross2(a: Any, b: Any) -> wp.Float:
    # 2D cross product (signed parallelogram area). Generic so float32 and float64 call sites share
    # one definition.
    return a[0] * b[1] - a[1] * b[0]


@wp.func
def to_vec3d(v: wp.vec3) -> wp.vec3d:
    return wp.vec3d(wp.float64(v[0]), wp.float64(v[1]), wp.float64(v[2]))


@wp.func
def to_vec3(v: wp.vec3d) -> wp.vec3:
    return wp.vec3(wp.float32(v[0]), wp.float32(v[1]), wp.float32(v[2]))


@wp.func
def to_vec2d(v: wp.vec2) -> wp.vec2d:
    return wp.vec2d(wp.float64(v[0]), wp.float64(v[1]))


@wp.func
def to_vec2(v: wp.vec2d) -> wp.vec2:
    return wp.vec2(wp.float32(v[0]), wp.float32(v[1]))


@wp.func
def lift_vec2(p: wp.vec2, z: wp.float32) -> wp.vec3:
    # 2D point -> 3D at a fixed height: trimesh's ``util.stack_3D`` plus a z offset.
    #
    # The fifth of the vec-conversion family above, and it was written **twice** -- once in
    # ``kernels/creation.py`` with this signature and once in ``kernels/proximity.py`` with the
    # height hardcoded to zero -- which is the collision ``.claude/CLAUDE.md`` section 4 warns
    # about as a hypothetical: ``wp.map``'s cache is keyed by the *unqualified* function name plus
    # the input dtypes, so two same-named ops fork one generated module. The two arities kept it
    # from being a wrong answer, and a warp-debug log of one suite run showed what it did cost --
    # ``Module map_lift_vec2`` loading at two distinct hashes on ``cuda:0``.
    #
    # ``z`` stays a parameter because ``creation.extrude_triangulation`` genuinely lifts to a
    # height; the four zero-lifting call sites pass ``wp.float32(0.0)`` explicitly, which all but
    # one of them already did.
    return wp.vec3(p[0], p[1], z)


@wp.func
def square_scalar(value: wp.Scalar) -> wp.Scalar:
    return value * value


@wp.func
def sqrt_abs(value: wp.Float) -> wp.Float:
    # ``sqrt(|x|)``, the per-row scaling the algebraic-multigrid strength test takes its geometric
    # mean from: ``|A_ij| >= theta sqrt(A_ii) sqrt(A_jj)`` is then a product and no square root runs
    # per edge. The magnitude is taken because a Laplacian written negative-semi-definite has a
    # negative diagonal.
    return wp.sqrt(wp.abs(value))


@wp.func
def inverse_or_one(value: wp.Float) -> wp.Float:
    # The Jacobi preconditioner's reciprocal, with a **zero diagonal mapped to 1 rather than to
    # infinity** -- what ``warp.optim.linear.preconditioner(m, "diag")`` does, and necessary because
    # such a row contributes nothing and must not poison the whole vector with a NaN.
    return wp.where(value != type(value)(0.0), type(value)(1.0) / value, type(value)(1.0))


@wp.func
def divide_if_positive(value: wp.Float, divisor: wp.Float) -> wp.Float:
    # Guarded division: leave ``value`` unchanged when ``divisor <= 0``.
    if divisor > type(divisor)(0.0):
        return value / divisor
    return value


@wp.kernel
def init_range(out_indices: wp.array[wp.Int]) -> None:
    i = wp.int32(wp.tid())
    out_indices[i] = i


@wp.kernel
def init_range_step(step: wp.Int, out_indices: wp.array[wp.Int]) -> None:
    i = wp.int32(wp.tid())
    out_indices[i] = i * step


@wp.kernel
def init_sort_pair_indices(n: wp.Int, fill_value: wp.Int, out_indices: wp.array[wp.Int]) -> None:
    i = wp.int32(wp.tid())
    if i < n:
        out_indices[i] = i
    else:
        out_indices[i] = fill_value


@wp.kernel
def init_repeat_index(repeats: wp.Int, out_indices: wp.array[wp.Int]) -> None:
    i = wp.int32(wp.tid())
    out_indices[i] = i // repeats


@wp.kernel
def segment_owner_labels(offsets: wp.array[wp.int32], out_owner: wp.array[wp.int32]) -> None:
    # For every element of a packed ragged array, which segment it belongs to -- the ragged
    # counterpart of ``init_repeat_index``, whose segments are all one width. ``offsets`` is the
    # exclusive scan of the segment sizes with the total appended, so this is launched over the
    # *segment* count and each thread writes its own label across its own span.
    #
    # One thread per segment rather than one per element (a binary search into ``offsets``) is the
    # right shape for the two callers here for opposite reasons, and both are worth knowing before
    # reaching for it. ``linalg`` expands CSR row offsets back to one row index per entry so a
    # pruned pattern can be rebuilt through ``bsr_from_triplets``: the rows are many and short, and
    # the alternative costs a search per non-zero. ``geodesic_walk`` labels each packed loop
    # position with its loop: the loops are few and short, so this beats a search *and* needs no
    # readback of the offsets. Where the segments are few but enormous, the per-element form would
    # win instead -- nothing in the tree is in that regime.
    segment = wp.int32(wp.tid())
    for slot in range(offsets[segment], offsets[segment + 1]):
        out_owner[slot] = segment


@wp.kernel
def random_priorities(seed: wp.int32, out_priority: wp.array[wp.uint32]) -> None:
    # A total order on the elements, drawn once for the whole run rather than per round. Every
    # multi-round selection that breaks ties by priority -- blue-noise dart throwing, the
    # maximal-independent-set aggregation -- reads the same order in every round, which is what
    # makes the loop a deterministic function of ``seed`` alone. Drawing per round would make the
    # answer depend on how many rounds the input happened to need.
    i = wp.int32(wp.tid())
    out_priority[i] = wp.randu(wp.rand_init(seed, i))


@wp.kernel
def gather_1d_skip_negative(
    indices: wp.array[wp.int32], table: wp.array[wp.int32], out_gathered: wp.array[wp.int32]
) -> None:
    tid = wp.int32(wp.tid())
    index = indices[tid]
    if index < wp.int32(0):
        out_gathered[tid] = index
    else:
        out_gathered[tid] = table[index]


@wp.kernel
def sort_rows_insertion(data: wp.array2d[wp.Scalar]) -> None:
    # One thread per row, in-place insertion sort across the row. For the narrow rows this library
    # actually sorts (vertex pairs, triangle corners) that is 1-3 register comparisons, versus a
    # segmented radix sort whose fixed per-segment cost dominates completely at these widths.
    row = wp.int32(wp.tid())
    width = data.shape[1]
    for i in range(1, width):
        value = data[row, i]
        j = i - 1
        while j >= 0 and data[row, j] > value:
            data[row, j + 1] = data[row, j]
            j = j - 1
        data[row, j + 1] = value


@wp.kernel
def gather_vec_skip_negative(
    source: wp.array[wp.vec3], index: wp.array[wp.int32], out_gathered: wp.array[wp.vec3]
) -> None:
    # Gather vectors by index, writing a zero vector wherever the index is negative
    # (missing-correspondence sentinel).
    i = wp.int32(wp.tid())
    f = index[i]
    if f >= 0:
        out_gathered[i] = source[f]
    else:
        out_gathered[i] = wp.vec3(0.0, 0.0, 0.0)


@wp.func
def shifted_index(value: wp.Scalar, offset: wp.Scalar) -> wp.int32:
    # Position of ``value`` in a table anchored at ``offset``. The subtraction happens in the
    # value's own dtype, which is exact for every dtype ``array.isin`` reaches this with: it widens
    # sub-32-bit dtypes first (so the span cannot overflow the type) and only takes the table path
    # when the span is small (so a 64-bit difference cannot overflow either).
    return wp.int32(value - offset)


@wp.kernel
def isin_lookup_sorted(
    elements: wp.array[wp.Scalar], sorted_test: wp.array[wp.Scalar], out_mask: wp.array[wp.bool]
) -> None:
    tid = wp.int32(wp.tid())
    out_mask[tid] = binary_search_sorted_contains(sorted_test, elements[tid])


@wp.func
def masked_at(mask: wp.array[wp.bool], index: wp.int32) -> wp.bool:
    # The mask's value at ``index``, reading ``False`` for an index outside it rather than off the
    # end of the buffer. The guard is what lets a *membership* mask stand in for an ``isin`` over
    # the equivalent index list: ``isin`` answers ``False`` for a value it has never seen, so a
    # malformed buffer carrying an out-of-range index keeps the answer it had instead of turning
    # into an out-of-bounds read. Two compares on a memory-bound lookup.
    if index < 0 or index >= mask.shape[0]:
        return False
    return mask[index]


@wp.func
def mask_not(a: wp.bool) -> wp.bool:
    # Mask complement -- for a caller holding the region to *delete* and needing the one to keep,
    # among others. Warp exposes no ``logical_not`` builtin (``wp.invert`` is the bitwise
    # complement, which is wrong for a ``wp.bool``), so this one-liner is what ``wp.map`` needs,
    # and it is the tree's only spelling of it: a second copy under a second name lived in
    # ``kernels/selection.py`` for a week, carrying this same paragraph.
    return not a


@wp.func
def mask_and(a: wp.bool, b: wp.bool) -> wp.bool:
    return a and b


@wp.func
def mask_and_not(a: wp.bool, b: wp.bool) -> wp.bool:
    return a and not b


@wp.func
def complement_flag(a: wp.bool) -> wp.int32:
    # ``1`` where the mask is False, ``0`` where it is True: the scan input for an inverted
    # ``mask_to_index_map`` (free/interior DOFs of a fixed-vertex mask, for instance).
    return wp.where(a, wp.int32(0), wp.int32(1))


@wp.func
def nonzero_flag(value: wp.Scalar) -> wp.int32:
    # ``1`` for any non-zero value, ``0`` otherwise: the scan input that lets ``flatnonzero``
    # accept integer and float arrays as well as masks. A bool array does not need this -- Warp's
    # ``wp.Scalar`` does not instantiate for ``wp.bool``, and ``wp.utils.array_cast`` already
    # produces exactly 0/1 for one, which is why the wrapper keeps that path separate.
    return wp.where(value != type(value)(0), wp.int32(1), wp.int32(0))


# The comparison family below is the tree's spelling for a thresholding ``wp.map``, and it is worth
# saying so here because it kept being re-spelled: ``seams.crease_edge_mask``,
# ``smoothing.is_spike_defect`` and ``heat/vector.is_resolved`` were each a private ``a > b`` under
# a domain name while ``greater`` already had six adopters. What those three carried that was worth
# keeping was never the comparison -- it was *which quantity* and *which threshold* their caller
# chose, and that argument now sits in the wrapper that chooses it, where a user of the public
# function reads it. Reach for a named predicate when it computes something (``is_positive_finite``,
# ``is_close_scalar``); reach for these when it is a comparison.
@wp.func
def greater(a: wp.Scalar, b: wp.Scalar) -> wp.bool:
    return a > b


@wp.func
def greater_equal(a: wp.Scalar, b: wp.Scalar) -> wp.bool:
    return a >= b


@wp.func
def less(a: wp.Scalar, b: wp.Scalar) -> wp.bool:
    return a < b


@wp.func
def less_equal(a: wp.Scalar, b: wp.Scalar) -> wp.bool:
    return a <= b


@wp.func
def equal(a: wp.Scalar, b: wp.Scalar) -> wp.bool:
    return a == b


@wp.func
def not_equal(a: wp.Scalar, b: wp.Scalar) -> wp.bool:
    return a != b


@wp.func
def is_positive_finite(value: wp.Float) -> wp.bool:
    return value > type(value)(0) and wp.isfinite(value)


@wp.func
def value_if_positive_finite(value: wp.Float) -> wp.Float:
    # The masked half of a "mean over the positive finite entries" reduction: the excluded entries
    # contribute an exact zero to the sum, so one plain reduction over this and one over the
    # companion mask give the numerator and the denominator without a compaction pass.
    if value > type(value)(0) and wp.isfinite(value):
        return value
    return type(value)(0)


@wp.func
def is_close_scalar(a: wp.Float, b: wp.Float, rtol: wp.Float, atol: wp.Float) -> wp.bool:
    # Generic over the caller's float precision, so ``allclose`` works on float16/32/64 from one
    # definition. The tolerances must arrive at that same precision -- see the wrapper.
    return wp.abs(a - b) <= atol + rtol * wp.abs(b)


@wp.func
def is_close_vec3(a: wp.vec3, b: wp.vec3, rtol: wp.float32, atol: wp.float32) -> wp.bool:
    close = True
    for k in range(3):
        if wp.abs(a[k] - b[k]) > atol + rtol * wp.abs(b[k]):
            close = False
    return close


@wp.func
def array_shift_insert(array: wp.array[wp.Scalar], value: wp.Scalar, index: wp.int32) -> None:
    for i in range(array.shape[0] - 1, index, -1):
        array[i] = array[i - 1]
    array[index] = value


@wp.func
def binary_search_index(values: wp.array[wp.Scalar], value: wp.Scalar) -> wp.int32:
    """First index i with values[i] > value, or len(values) (numpy searchsorted side='right')."""
    n = values.shape[0]
    left = wp.int32(0)
    right = n - 1
    result = n
    while left <= right:
        mid = (left + right) // 2
        if values[mid] > value:
            result = mid
            right = mid - 1
        else:
            left = mid + 1
    return wp.int32(result)


@wp.func
def binary_search_index_left(values: wp.array[wp.Scalar], value: wp.Scalar) -> wp.int32:
    """First index i with values[i] >= value, or len(values) (numpy searchsorted side='left')."""
    # ``wp.lower_bound`` is the same search, but it clamps its result to ``n - 1``
    # (``warp/native/array.h``), so a value past the last element reads back as the last index
    # instead of ``n``. Both callers (``graph.component_segment_bounds`` probes one past the
    # highest component key, ``holes.rim_opposite_from_table`` probes edges absent from the
    # table) do query past the end, so the fix-up is mandatory, not defensive.
    n = values.shape[0]
    index = wp.lower_bound(values, value)
    if index == n - 1 and values[n - 1] < value:
        index = n
    return wp.int32(index)


@wp.func
def binary_search_sorted_contains(values: wp.array[wp.Scalar], value: wp.Scalar) -> wp.bool:
    # ``wp.lower_bound``'s clamp to ``n - 1`` is harmless here: a value past the end lands on the
    # last element, which then compares unequal. ``and`` short-circuits in kernel scope, so the
    # element read is skipped on an empty array.
    n = values.shape[0]
    index = wp.lower_bound(values, value)
    return index < n and values[index] == value


@wp.kernel
def map_sorted_inverse(
    data: wp.array[wp.Scalar], sorted_unique: wp.array[wp.Scalar], out_inverse: wp.array[wp.int32]
) -> None:
    i = wp.int32(wp.tid())
    out_inverse[i] = binary_search_index(sorted_unique, data[i]) - wp.int32(1)


@wp.kernel
def mark_rows_present(
    rows: wp.array2d[wp.int32], queries: wp.array2d[wp.int32], out_present: wp.array[wp.bool]
) -> None:
    # Which of a handful of index rows occur in a table of them, decided on device so the caller
    # never reads back a buffer that scales with the mesh. Every thread that matches stores ``True``
    # into its query's slot, so the race is benign and no atomic is needed; ``out_present`` arrives
    # zeroed. Deliberately a scan rather than a hash or a sort: the query count is a handful, the
    # table is read once, and there is nothing to amortize a structure over.
    row, query = wp.tid()
    for column in range(queries.shape[1]):
        if rows[row, column] != queries[query, column]:
            return
    out_present[query] = True


# Concrete overloads, registered at import -- rationale in ``triwarp/kernels/reduce.py``, rule in
# CLAUDE.md section 4. Measured over the suite: 12 overloads created across **13** module loads,
# and this module is imported by 25 kernel modules and 15 wrappers, so its rebuilds are felt widely.
#
# The ``init_*`` kernels fill an index buffer and every caller in the package allocates that buffer
# ``wp.int32``; ``wp.Int`` in their annotation is the template, not a menu. The two search kernels
# take the caller's *key* dtype, whose surface is the one ``sortable_dtype`` maps onto.
_INDEX_DTYPES = (wp.int32,)
_KEY_DTYPES = (wp.int32, wp.int64, wp.uint32, wp.uint64)


def _register_overloads() -> None:
    """Instantiate every concrete overload of this module's generic kernels."""
    for dtype in _INDEX_DTYPES:
        wp.overload(init_range, [wp.array[dtype]])
        wp.overload(init_range_step, [dtype, wp.array[dtype]])
        wp.overload(init_repeat_index, [dtype, wp.array[dtype]])
        wp.overload(init_sort_pair_indices, [dtype, dtype, wp.array[dtype]])
    for dtype in _KEY_DTYPES:
        wp.overload(isin_lookup_sorted, [wp.array[dtype], wp.array[dtype], wp.array[wp.bool]])
        wp.overload(map_sorted_inverse, [wp.array[dtype], wp.array[dtype], wp.array[wp.int32]])
    # ``sort_rows_insertion`` sorts a rank-2 table in place; ``unique_rows`` and the hashing paths
    # that reach it build that table in the caller's dtype.
    for dtype in (wp.int32, wp.float32, wp.float64):
        wp.overload(sort_rows_insertion, [wp.array2d[dtype]])


_register_overloads()


@wp.func
def lowbias32(x: wp.uint32) -> wp.uint32:
    # The ``lowbias32`` finalizer: a **bijection** on uint32 whose output is decorrelated from its
    # input. Two properties, and the tree needs both -- each of its callers needed one of them and
    # wrote the mixer out for itself.
    #
    # *Bijective*, so distinct inputs never collide: a priority drawn as ``lowbias32(index)`` is a
    # strict total order on the indices with no tie to break, which is what
    # ``polyline.ear_outranks`` relies on (its index tiebreak is dead code kept only in case the
    # mixer is ever swapped).
    #
    # *Decorrelating*, and this is load-bearing for any parallel independent-set pass. ``edges_
    # unique`` orders edges lexicographically by endpoint index, which on a structured mesh is
    # spatially *monotone*, and a monotone key field has essentially one local minimum -- so a
    # min-key lock commits a single winner per pass however many candidates there are. Measured on
    # ``saddle_graded``: locking by raw edge index yields **1** winner out of 51 546 candidates, and
    # locking by quadric cost yields 23 (that field is smoothly graded there, so it is monotone
    # too). Hashing the index restores the expected ~candidates/valence.
    #
    # For a random order that a caller can *vary*, use ``random_priorities`` instead: it takes a
    # seed. This one is a pure function of the index, so it needs no state and is reproducible
    # across runs and across launches -- which is why the collapse pass can recompute a lock key in
    # a later kernel instead of storing it.
    x = (x ^ (x >> wp.uint32(16))) * wp.uint32(0x7FEB352D)
    x = (x ^ (x >> wp.uint32(15))) * wp.uint32(0x846CA68B)
    return x ^ (x >> wp.uint32(16))


@wp.func
def pack_edge_key(u: wp.int32, v: wp.int32, base: wp.uint64) -> wp.uint64:
    """Key of undirected edge (u, v); matches ``pack_indices`` for a sorted 2-index row."""
    lo = wp.uint64(wp.uint32(wp.min(u, v)))
    hi = wp.uint64(wp.uint32(wp.max(u, v)))
    return lo + hi * base


@wp.func
def pack_farthest_key(distance_sq: wp.float32, index: wp.int32) -> wp.int64:
    # One int64 whose ``wp.atomic_max`` is "largest distance, lowest index on a tie". The IEEE-754
    # bits of a non-negative float increase monotonically with the value, so the high half orders
    # by distance; the low half stores ``index`` complemented within int32 so that a *smaller*
    # index compares *larger*. Both halves are non-negative, so the whole key is, which is what
    # makes ``-1`` a sentinel below every real candidate. ``unpack_ranked_index`` inverts the
    # low half.
    distance_bits = wp.uint64(wp.uint32(wp.cast(distance_sq, wp.int32)))
    rank = wp.uint64(wp.uint32(wp.int32(2147483647) - index))
    return wp.int64((distance_bits << wp.uint64(32)) | rank)


@wp.func
def pack_ranked_key(value: wp.int32, index: wp.int32) -> wp.int64:
    # The integer sibling of ``pack_farthest_key``: one int64 whose ``wp.atomic_max`` is "largest
    # value, lowest index on a tie". ``value`` must be non-negative (a count, a size, a degree), so
    # the high half orders by it directly with no bit trick; the low half stores ``index``
    # complemented within int32 so that a *smaller* index compares *larger*. Both halves are
    # non-negative, so ``-1`` is a sentinel below every real candidate.
    #
    # Comparing this key is also how a caller applies the result without a readback: recomputing
    # ``pack_ranked_key(value, index)`` in a second kernel and testing it against the reduced
    # maximum identifies the winner on the device. Where the host does need the index,
    # ``unpack_ranked_index`` inverts the low half.
    return wp.int64(
        (wp.uint64(wp.uint32(value)) << wp.uint64(32))
        | wp.uint64(wp.uint32(wp.int32(2147483647) - index))
    )


@wp.func
def unpack_ranked_index(key: wp.int64) -> wp.int32:
    # The index out of a ``pack_farthest_key`` / ``pack_ranked_key`` key -- the inverse of the
    # complement both use in their low half, so one decoder serves both.
    #
    # It lives here, beside its packers, because a pack/unpack pair in two modules is a pair that
    # cannot be read: this was in ``kernels/points.py`` while both packers were here, three hundred
    # lines from either. Note the alternative a caller may prefer:
    # ``repair.mark_largest_group_mask`` never unpacks at all, it *recomputes* the key and tests it
    # against the reduced maximum, keeping the winner on the device instead of pulling it back to
    # pick a row. Unpack when the host needs the index (a greedy loop's next seed); recompute when
    # only the device does.
    return wp.int32(2147483647) - wp.int32(wp.uint32(wp.uint64(key) & wp.uint64(4294967295)))


@wp.func
def pack_nearest_key(distance: wp.float32, index: wp.int32) -> wp.int64:
    # The ``min``-ordered twin of ``pack_farthest_key``: one int64 whose minimum is "smallest
    # distance, lowest index on a tie". Same monotone IEEE-754 high half (valid because a distance
    # is non-negative), but the index is stored plainly rather than complemented, because a
    # *smaller* index must now compare *smaller*. The key stays non-negative, so it needs no
    # sentinel.
    distance_bits = wp.uint64(wp.uint32(wp.cast(distance, wp.int32)))
    return wp.int64((distance_bits << wp.uint64(32)) | wp.uint64(wp.uint32(index)))


@wp.func
def lattice_position(
    lower: wp.vec3, step: wp.vec3, i: wp.int32, j: wp.int32, k: wp.int32
) -> wp.vec3:
    # World position of node ``(i, j, k)`` of a dense lattice whose node 0 sits at ``lower``.
    #
    # A **node** lattice, not a cell-centre one: there is no half-step shift, so a lattice of
    # ``dims`` nodes with ``step = extent / (dims - 1)`` spans its box inclusively at both ends.
    # Both callers want that -- ``reconstruction``'s signed-distance sample grid, in the row-major
    # order ``wp.MarchingCubes`` expects, and ``voxels.lattice`` -- and neither is a candidate for
    # ``wp.volume_index_to_world``, the spelling ``kernels/voxels.py``'s module docstring records
    # as preferred: both run *before* any ``wp.Volume`` exists, so there is no volume id to pass.
    # That is the same un-convertible half of the split that docstring names. Getting this wrong is
    # a rigid half-diagonal offset, which is exactly the failure mode a half-step convention has.
    #
    # It differs from the two kernels that share it only in the *destination's rank* -- a flat
    # row-major buffer against a ``wp.array3d`` -- so each keeps its own indexing and shares the
    # arithmetic, per section 3's "factor the family, not the pair".
    return lower + wp.cw_mul(step, wp.vec3(wp.float32(i), wp.float32(j), wp.float32(k)))


@wp.func
def trilinear_cell(coordinate: wp.vec3, shape: wp.vec3i) -> tuple[wp.vec3i, wp.vec3]:
    # Decompose a continuous lattice coordinate into its base corner and the three fractional
    # offsets, clamping the corner so the ``+1`` reads of a trilinear stencil stay in range. Exact
    # integer and subtraction work, which is why the three callers share it safely: extracting it
    # reorders nothing and a float32 result cannot drift.
    #
    # A degenerate axis (``shape[k] < 2``) collapses to corner 0 with fraction 0, so a single-slice
    # lattice reads and writes that slice rather than indexing out of bounds.
    limit = wp.vec3i(wp.max(shape[0] - 2, 0), wp.max(shape[1] - 2, 0), wp.max(shape[2] - 2, 0))
    base = wp.vec3i(
        wp.clamp(wp.int32(wp.floor(coordinate[0])), 0, limit[0]),
        wp.clamp(wp.int32(wp.floor(coordinate[1])), 0, limit[1]),
        wp.clamp(wp.int32(wp.floor(coordinate[2])), 0, limit[2]),
    )
    return base, wp.vec3(
        wp.clamp(coordinate[0] - wp.float32(base[0]), 0.0, 1.0),
        wp.clamp(coordinate[1] - wp.float32(base[1]), 0.0, 1.0),
        wp.clamp(coordinate[2] - wp.float32(base[2]), 0.0, 1.0),
    )


@wp.func
def trilinear_weight(
    fractions: wp.vec3, offset_x: wp.int32, offset_y: wp.int32, offset_z: wp.int32
) -> wp.float32:
    # Weight of one of the eight corners a ``trilinear_cell`` decomposition addresses. The eight
    # sum to exactly 1, which is what makes ``scatter.splat_grid_trilinear`` and
    # ``interpolation.sample_grid_trilinear`` transposes of each other -- and, measured, what makes
    # a splatted density sum to the point count exactly.
    return (
        wp.where(offset_x == 0, 1.0 - fractions[0], fractions[0])
        * wp.where(offset_y == 0, 1.0 - fractions[1], fractions[1])
        * wp.where(offset_z == 0, 1.0 - fractions[2], fractions[2])
    )
