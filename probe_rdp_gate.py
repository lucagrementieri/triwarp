"""Gate the level-synchronous Ramer-Douglas-Peucker against the recursive kernel it replaces.

Breadth-first and depth-first evaluation of the same recursion tree accept the same points, so
this comparison is an exact one -- ``np.array_equal`` on the accepted index set, not a tolerance.
The oracle is the single-thread kernel from ``HEAD``, transcribed verbatim, so the two run in one
process against one input.

Also prices the two against each other, interleaved under one clock state, and measures the
recursion depth so the round count in the docstring's table can be checked rather than assumed.
"""

import sys
import time

import numpy as np
import warp as wp

sys.path.insert(0, "benchmarks")

import triwarp as tw
from triwarp.kernels.array import update_argmax
from triwarp.kernels.polyline import RDP_LINE_EPS, line_squared_distance


@wp.kernel(enable_backward=False)
def rdp_keep_mask_oracle(
    polyline: wp.array[wp.vec3],
    stol: wp.float32,
    stack: wp.array[wp.int32],
    out_keep: wp.array[wp.bool],
) -> None:
    # Verbatim from HEAD:triwarp/kernels/polyline.py -- the recursion this replaces.
    n = polyline.shape[0]
    stack[0] = 0
    stack[1] = n - 1
    top = wp.int32(1)
    while top > 0:
        top -= 1
        ixs = stack[2 * top + 0]
        ixe = stack[2 * top + 1]
        sdmax = wp.float32(0.0)
        ixc = wp.int32(-1)
        if ixe - ixs > 1:
            seg = polyline[ixe] - polyline[ixs]
            sdes = wp.length_sq(seg)
            for k in range(ixs + 1, ixe):
                sd = wp.float32(0.0)
                if sdes <= RDP_LINE_EPS:
                    dvec = polyline[k] - polyline[ixs]
                    sd = wp.length_sq(dvec)
                else:
                    sd = line_squared_distance(polyline[k], polyline[ixs], polyline[ixe], sdes)
                update_argmax(sdmax, ixc, sd, k)
        if sdmax <= stol:
            for k in range(ixs + 1, ixe):
                out_keep[k] = False
        else:
            stack[2 * top + 0] = ixs
            stack[2 * top + 1] = ixc
            top += 1
            stack[2 * top + 0] = ixc
            stack[2 * top + 1] = ixe
            top += 1


def simplify_oracle(polyline: wp.array, tol: float) -> np.ndarray:
    """The recursive form's accepted index set."""
    device = polyline.device
    n = int(polyline.shape[0])
    keep = wp.full(n, True, dtype=wp.bool, device=device)
    stack = wp.empty(max(2 * n, 2), dtype=wp.int32, device=device)
    wp.launch(
        rdp_keep_mask_oracle,
        dim=1,
        inputs=[polyline, wp.float32(tol * tol), stack, keep],
        device=device,
    )
    return tw.array.flatnonzero(keep).numpy()


def rdp_rounds(points: np.ndarray, tol: float) -> int:
    """Host-side level-synchronous walk, for the round count alone."""
    spans, rounds = [(0, len(points) - 1)], 0
    while spans:
        rounds += 1
        nxt = []
        for i, j in spans:
            if j <= i + 1:
                continue
            a, b = points[i], points[j]
            ab = b - a
            length = np.linalg.norm(ab)
            seg = points[i + 1 : j]
            d = (
                np.linalg.norm(seg - a, axis=1)
                if length < 1e-30
                else np.linalg.norm(np.cross(seg - a, ab / length), axis=1)
            )
            k = int(np.argmax(d))
            if d[k] > tol:
                m = i + 1 + k
                nxt.extend([(i, m), (m, j)])
        spans = nxt
    return rounds


def boundary_polyline(name: str) -> np.ndarray:
    from meshes import BUILDERS

    verts_np, faces_np = BUILDERS[name]()
    vertices = wp.array(np.ascontiguousarray(verts_np, np.float32), dtype=wp.vec3, device="cpu")
    faces = wp.array(
        np.ascontiguousarray(faces_np.reshape(-1), np.int32), dtype=wp.int32, device="cpu"
    )
    loops = tw.boundary.boundary_loops(vertices, faces)
    longest = max(loops, key=lambda loop: int(loop.shape[0]))
    return np.ascontiguousarray(verts_np[longest.numpy()], np.float32)


def spiral(n: int, turns: float) -> np.ndarray:
    """The adversarial input: an ever-growing radius peels one point per level."""
    t = np.linspace(0.0, turns * 2.0 * np.pi, n)
    radius = np.linspace(0.05, 1.0, n)
    return np.ascontiguousarray(
        np.stack([radius * np.cos(t), radius * np.sin(t), np.zeros_like(t)], axis=1), np.float32
    )


def time_call(fn, reps: int) -> tuple[float, float]:
    """(median, min) milliseconds, with the device synchronized inside the timed region."""
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        wp.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return float(np.median(samples)), float(np.min(samples))


CASES: list[tuple[str, np.ndarray, list[float]]] = []


def build_cases() -> None:
    rim = boundary_polyline("rim_long")
    saddle = boundary_polyline("saddle")
    saddle_small = boundary_polyline("saddle_small")
    for name, pts in (("rim_long", rim), ("saddle", saddle), ("saddle_small", saddle_small)):
        mean_seg = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).mean())
        CASES.append((name, pts, [1e-3 * mean_seg, 1e-1 * mean_seg]))
    CASES.append(("spiral_4096", spiral(4096, 10.0), [1e-3, 1e-2]))
    rng = np.random.default_rng(20260824)
    CASES.append(("random_2000", rng.standard_normal((2000, 3)).astype(np.float32), [0.05, 0.5]))
    # Degenerate classes the recursion's own guards exist for.
    CASES.append(("collinear_512", np.stack(
        [np.linspace(0, 1, 512), np.zeros(512), np.zeros(512)], axis=1).astype(np.float32),
        [1e-6, 0.1]))
    duplicated = np.repeat(spiral(64, 2.0), 4, axis=0)  # every point four times: zero-length chords
    CASES.append(("duplicated_256", duplicated, [1e-4, 0.05]))
    for n in (0, 1, 2, 3, 4):
        CASES.append((f"tiny_{n}", spiral(max(n, 1), 1.0)[:n], [0.01]))


def main() -> None:
    build_cases()
    devices = ["cuda:0", "cpu"] if wp.get_cuda_device_count() else ["cpu"]
    failures = 0
    print(f"{'case':16s} {'n':>7s} {'tol':>10s} {'dev':>6s} {'kept':>7s} {'rounds':>7s}  verdict")
    for name, points, tolerances in CASES:
        for tol in tolerances:
            rounds = rdp_rounds(points.astype(np.float64), tol) if len(points) > 2 else 0
            for device in devices:
                polyline = wp.array(points, dtype=wp.vec3, device=device)
                expected = simplify_oracle(polyline, tol)
                # A fresh upload: the oracle kernel writes nothing to the polyline, but the
                # level-synchronous form must not be handed a buffer the oracle's launch touched.
                polyline_new = wp.array(points, dtype=wp.vec3, device=device)
                _, indices = tw.polyline.polyline_simplify(polyline_new, tol)
                got = indices.numpy()
                ok = np.array_equal(got, expected)
                failures += not ok
                verdict = "identical" if ok else f"DIFFER exp={len(expected)} got={len(got)}"
                print(
                    f"{name:16s} {len(points):7d} {tol:10.3e} {device:>6s} "
                    f"{len(got):7d} {rounds:7d}  {verdict}"
                )
    print(f"\n{'PASS' if not failures else f'{failures} MISMATCHES'}: keep-set equality\n")

    if not wp.get_cuda_device_count():
        return
    print("=== interleaved A/B, median/min ms, device sync inside the timed region ===")
    print(f"{'case':16s} {'n':>7s} {'tol':>10s} {'dev':>6s} "
          f"{'recursive':>18s} {'level-sync':>18s} {'speedup':>8s}")
    timed = [
        ("rim_long", 0), ("rim_long", 1), ("spiral_4096", 0), ("saddle", 0), ("saddle_small", 0),
    ]
    by_name = {name: (pts, tols) for name, pts, tols in CASES}
    for device in ("cuda:0", "cpu"):
        for name, which in timed:
            points, tolerances = by_name[name]
            tol = tolerances[which]
            polyline = wp.array(points, dtype=wp.vec3, device=device)
            reps = 3 if len(points) > 20000 else 20
            simplify_oracle(polyline, tol)  # warm the module and the pool
            tw.polyline.polyline_simplify(polyline, tol)
            old_samples, new_samples = [], []
            for _ in range(reps):  # interleaved, so one clock state serves both
                old_samples.append(time_call(lambda: simplify_oracle(polyline, tol), 1))
                new_samples.append(
                    time_call(lambda: tw.polyline.polyline_simplify(polyline, tol), 1)
                )
            old_med, old_min = float(np.median([s[0] for s in old_samples])), min(
                s[1] for s in old_samples
            )
            new_med, new_min = float(np.median([s[0] for s in new_samples])), min(
                s[1] for s in new_samples
            )
            print(
                f"{name:16s} {len(points):7d} {tol:10.3e} {device:>6s} "
                f"{old_med:9.2f}/{old_min:<8.2f} {new_med:9.2f}/{new_min:<8.2f} "
                f"{old_med / new_med:7.2f}x"
            )


if __name__ == "__main__":
    main()
