from typing import Any

import warp as wp

from triwarp.kernels.array import binary_search_sorted_contains
from triwarp.kernels.grouping import pack_edge_key


@wp.func
def halfedge_next(h: wp.int32) -> wp.int32:
    # Next halfedge inside the same face, under the ``h = 3 * f + k`` convention of
    # ``triwarp.halfedge``: index arithmetic, no structure to look up.
    return h - h % 3 + (h + 1) % 3


@wp.kernel
def corner_union_edges(
    faces: wp.array[wp.int32],
    twins: wp.array[wp.int32],
    marked_keys: wp.array[wp.uint64],
    key_base: wp.uint64,
    out_edges: wp.array2d[wp.int32],
    out_count: wp.array[wp.int32],
) -> None:
    # Two graph edges per *uncut* interior mesh edge, joining the two face corners that meet at each
    # of its two endpoints. Connected components of that graph are exactly the copies each vertex
    # needs: a vertex whose whole fan is uncut stays one vertex, and every marked edge crossing the
    # fan splits it.
    #
    # Both endpoints matter, and getting only one of them wrong is silent: the fan around a vertex
    # is then connected by half its edges and every vertex splits into two.
    #
    # ``h < twin`` emits each undirected edge once. For the halfedge ``h: u -> v`` with twin
    # ``t: v -> u``, the corners at ``u`` are ``h`` and ``next(t)``, and the corners at ``v`` are
    # ``next(h)`` and ``t`` -- the twin runs the other way, so its *following* halfedge is the one
    # starting where ``h`` does.
    h = int(wp.tid())
    twin = twins[h]
    if twin < 0 or twin < h:
        return
    if binary_search_sorted_contains(
        marked_keys, pack_edge_key(faces[h], faces[halfedge_next(h)], key_base)
    ):
        return
    slot = wp.atomic_add(out_count, 0, 2)
    out_edges[slot, 0] = h
    out_edges[slot, 1] = halfedge_next(twin)
    out_edges[slot + 1, 0] = halfedge_next(h)
    out_edges[slot + 1, 1] = twin


@wp.kernel
def scatter_corner_values(
    faces: wp.array[wp.int32],
    corner_index: wp.array[wp.int32],
    values: wp.array[Any],
    out_values: wp.array[Any],
) -> None:
    # Position (or any per-vertex attribute) of each output copy, gathered through the corner it
    # came from. Every corner in a component writes the *same* value, so the race is benign by
    # construction and no atomics are needed.
    h = int(wp.tid())
    out_values[corner_index[h]] = values[faces[h]]


@wp.kernel
def crease_edge_mask(
    adjacency_angles: wp.array[wp.float32], threshold: wp.float32, out_mask: wp.array[wp.bool]
) -> None:
    k = int(wp.tid())
    out_mask[k] = adjacency_angles[k] > threshold
