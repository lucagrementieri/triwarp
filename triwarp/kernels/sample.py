import math

import warp as wp

# Golden angle in radians: pi * (3 - sqrt(5)) ~ 2.399963. Successive multiples of this
# angle place points on the Fibonacci lattice, the most uniform simple spiral on a sphere.
GOLDEN_ANGLE = wp.constant(wp.float32(math.pi * (3.0 - math.sqrt(5.0))))


@wp.kernel
def fibonacci_sphere(count: wp.int32, out_directions: wp.array[wp.vec3]) -> None:
    i = int(wp.tid())
    count_f = wp.float32(count)
    # z descends uniformly through (-1, 1); the offset 0.5 centers the samples.
    z = 1.0 - 2.0 * (wp.float32(i) + 0.5) / count_f
    radius = wp.sqrt(wp.max(0.0, 1.0 - z * z))
    theta = GOLDEN_ANGLE * wp.float32(i)
    out_directions[i] = wp.vec3(radius * wp.cos(theta), radius * wp.sin(theta), z)


@wp.kernel
def fibonacci_hemisphere(count: wp.int32, out_directions: wp.array[wp.vec3]) -> None:
    i = int(wp.tid())
    count_f = wp.float32(count)
    # z descends uniformly through (0, 1): positive-z hemisphere only.
    z = 1.0 - (wp.float32(i) + 0.5) / count_f
    radius = wp.sqrt(wp.max(0.0, 1.0 - z * z))
    theta = GOLDEN_ANGLE * wp.float32(i)
    out_directions[i] = wp.vec3(radius * wp.cos(theta), radius * wp.sin(theta), z)


@wp.kernel
def sample_surface(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    cdf: wp.array[wp.float32],
    seed: int,
    out_points: wp.array[wp.vec3],
    out_face_indices: wp.array[wp.int32],
) -> None:
    tid = wp.tid()
    state = wp.rand_init(seed, tid)
    fi = int(wp.sample_cdf(state, cdf))

    v0 = vertices[faces[fi * 3]]
    v1 = vertices[faces[fi * 3 + 1]]
    v2 = vertices[faces[fi * 3 + 2]]

    uv = wp.sample_triangle(state)
    w = 1.0 - uv.x - uv.y

    out_face_indices[tid] = fi
    out_points[tid] = v0 * w + v1 * uv.x + v2 * uv.y


@wp.kernel
def signed_tet_volumes(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    center: wp.vec3,
    out_volumes: wp.array[wp.float32],
) -> None:
    fi = int(wp.tid())
    v0 = vertices[faces[fi * 3]] - center
    v1 = vertices[faces[fi * 3 + 1]] - center
    v2 = vertices[faces[fi * 3 + 2]] - center
    out_volumes[fi] = wp.dot(v0, wp.cross(v1, v2)) / 6.0


@wp.kernel
def sample_volume_tet(
    vertices: wp.array[wp.vec3],
    faces: wp.array[wp.int32],
    center: wp.vec3,
    cdf: wp.array[wp.float32],
    seed: int,
    out_points: wp.array[wp.vec3],
) -> None:
    tid = int(wp.tid())
    state = wp.rand_init(seed, tid)
    fi = int(wp.sample_cdf(state, cdf))

    v0 = vertices[faces[fi * 3]]
    v1 = vertices[faces[fi * 3 + 1]]
    v2 = vertices[faces[fi * 3 + 2]]

    # Uniform sampling in tet (center, v0, v1, v2) via order statistics of
    # 3 U(0,1) samples. Sorted values s1 ≤ s2 ≤ s3 give spacings
    # (s1, s2-s1, s3-s2, 1-s3) as barycentric coords for (center, v0, v1, v2):
    # P = center*(1-s3) + v0*s1 + v1*(s2-s1) + v2*(s3-s2)
    a = wp.randf(state)
    b = wp.randf(state)
    c = wp.randf(state)

    s1 = wp.min(a, wp.min(b, c))
    s3 = wp.max(a, wp.max(b, c))
    s2 = a + b + c - s1 - s3

    out_points[tid] = center * (1.0 - s3) + v0 * s1 + v1 * (s2 - s1) + v2 * (s3 - s2)


@wp.func
def _poisson_edge_weight(d: wp.float32, r_max: wp.float32, r_min: wp.float32, alpha: wp.float32) -> wp.float32:
    d_eff = wp.max(d, r_min)
    return wp.pow(wp.float32(1.0) - d_eff / r_max, alpha)


@wp.kernel
def compute_poisson_weights(
    nbr_indices: wp.array[wp.int32],
    nbr_dists: wp.array[wp.float32],
    offsets: wp.array[wp.int32],
    alive: wp.array[wp.int32],
    r_max: wp.float32,
    r_min: wp.float32,
    alpha: wp.float32,
    out_weights: wp.array[wp.float32],
) -> None:
    i = int(wp.tid())
    if alive[i] == 0:
        out_weights[i] = wp.float32(0.0)
        return
    start = int(offsets[i])
    end = int(offsets[i + 1])
    w = wp.float32(0.0)
    for k in range(start, end):
        j = int(nbr_indices[k])
        if j == i or alive[j] == 0:
            continue
        w += _poisson_edge_weight(nbr_dists[k], r_max, r_min, alpha)
    out_weights[i] = w


@wp.kernel
def find_local_maxima(
    weights: wp.array[wp.float32],
    alive: wp.array[wp.int32],
    nbr_indices: wp.array[wp.int32],
    offsets: wp.array[wp.int32],
    out_is_max: wp.array[wp.int32],
) -> None:
    i = int(wp.tid())
    if alive[i] == 0:
        out_is_max[i] = 0
        return
    wi = wp.max(weights[i], wp.float32(0.0))
    start = int(offsets[i])
    end = int(offsets[i + 1])
    is_max = int(1)
    for k in range(start, end):
        j = int(nbr_indices[k])
        if j == i or alive[j] == 0:
            continue
        if wp.max(weights[j], wp.float32(0.0)) > wi:
            is_max = int(0)
            break
    out_is_max[i] = is_max


@wp.kernel
def apply_deletions(
    deleted_mask: wp.array[wp.int32],
    alive: wp.array[wp.int32],
) -> None:
    i = int(wp.tid())
    if deleted_mask[i] == 1:
        alive[i] = 0


@wp.kernel
def subtract_deleted_contributions(
    deleted_mask: wp.array[wp.int32],
    nbr_indices: wp.array[wp.int32],
    nbr_dists: wp.array[wp.float32],
    offsets: wp.array[wp.int32],
    alive: wp.array[wp.int32],
    r_max: wp.float32,
    r_min: wp.float32,
    alpha: wp.float32,
    weights: wp.array[wp.float32],
) -> None:
    i = int(wp.tid())
    if deleted_mask[i] == 0:
        return
    start = int(offsets[i])
    end = int(offsets[i + 1])
    for k in range(start, end):
        j = int(nbr_indices[k])
        if j == i or alive[j] == 0:
            continue
        contribution = _poisson_edge_weight(nbr_dists[k], r_max, r_min, alpha)
        wp.atomic_add(weights, j, -contribution)
