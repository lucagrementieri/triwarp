"""Find the level-synchronous RDP's real worst case, and firm up the speedup numbers.

Two questions the docstring's table asserts and this measures instead:

1. Is a spiral the adversarial input? The claim is that its farthest point sits next to an
   endpoint at every level, giving ``O(n)`` depth. A monotone spiral does not obviously do
   that -- the farthest point from a long chord is nearer the middle -- so several candidate
   shapes are tried and their depths reported.
2. Are the speedups what the table says? Re-measured with more repetitions than the first pass.
"""

import sys
import time

import numpy as np
import warp as wp

sys.path.insert(0, "benchmarks")

import triwarp as tw
from probe_rdp_gate import boundary_polyline, simplify_oracle, spiral


def rdp_depth(points: np.ndarray, tol: float) -> tuple[int, int]:
    """(rounds, kept) for the level-synchronous schedule, on the host."""
    spans, rounds, kept = [(0, len(points) - 1)], 0, 2
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
                kept += 1
                nxt.extend([(i, m), (m, j)])
        spans = nxt
    return rounds, kept


def power_curve(n: int, alpha: float) -> np.ndarray:
    """``y = x**alpha`` on ``[0, 1]``: for large alpha the argmax hugs the *start* of any chord."""
    x = np.linspace(0.0, 1.0, n)
    return np.ascontiguousarray(np.stack([x, x**alpha, np.zeros_like(x)], axis=1), np.float32)


def geometric_staircase(n: int) -> np.ndarray:
    """Interior points crowding one endpoint geometrically, each just outside the last chord."""
    x = 1.0 - 0.5 ** np.arange(n, dtype=np.float64)
    y = 0.5 ** np.arange(n, dtype=np.float64)
    return np.ascontiguousarray(np.stack([x, y, np.zeros_like(x)], axis=1), np.float32)


def sawtooth_decay(n: int) -> np.ndarray:
    """A zigzag whose teeth shrink geometrically, so each level peels exactly one tooth."""
    x = np.linspace(0.0, 1.0, n)
    y = (0.5 ** np.arange(n, dtype=np.float64)) * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    return np.ascontiguousarray(np.stack([x, y, np.zeros_like(x)], axis=1), np.float32)


def main() -> None:
    n = 4096
    print("=== recursion depth by shape (n = 4096), tol = 1e-3 of the coordinate range ===")
    print(f"{'shape':22s} {'rounds':>7s} {'kept':>7s} {'log2(n)':>8s} {'depth/n':>8s}")
    shapes = {
        "spiral_10turn": spiral(n, 10.0),
        "spiral_100turn": spiral(n, 100.0),
        "power_curve_a=8": power_curve(n, 8.0),
        "power_curve_a=64": power_curve(n, 64.0),
        "geometric_stair": geometric_staircase(n),
        "sawtooth_decay": sawtooth_decay(n),
        "rim_long[:4096]": boundary_polyline("rim_long")[:n],
    }
    worst_name, worst_rounds = "", 0
    for name, pts in shapes.items():
        extent = float(np.ptp(pts.reshape(-1, 3), axis=0).max())
        rounds, kept = rdp_depth(pts.astype(np.float64), 1e-3 * extent)
        if rounds > worst_rounds:
            worst_name, worst_rounds = name, rounds
        print(f"{name:22s} {rounds:7d} {kept:7d} {np.log2(n):8.1f} {rounds / n:8.3f}")
    print(f"\ndeepest shape: {worst_name} at {worst_rounds} rounds "
          f"({worst_rounds / n:.1%} of n)\n")

    print("=== interleaved A/B, 40 reps (8 above 20k points), median/min ms, CUDA ===")
    print(f"{'case':22s} {'n':>7s} {'recursive':>18s} {'level-sync':>18s} {'speedup':>8s}")
    rim = boundary_polyline("rim_long")
    rim_mean = float(np.linalg.norm(np.diff(rim, axis=0), axis=1).mean())
    saddle = boundary_polyline("saddle")
    saddle_mean = float(np.linalg.norm(np.diff(saddle, axis=0), axis=1).mean())
    small = boundary_polyline("saddle_small")
    small_mean = float(np.linalg.norm(np.diff(small, axis=0), axis=1).mean())
    rows = [
        ("rim_long fine", rim, 1e-3 * rim_mean),
        ("rim_long coarse", rim, 1e-1 * rim_mean),
        ("saddle", saddle, 1e-3 * saddle_mean),
        ("saddle_small", small, 1e-3 * small_mean),
        (worst_name, shapes[worst_name], 1e-3 * float(
            np.ptp(shapes[worst_name].reshape(-1, 3), axis=0).max())),
    ]
    for name, points, tol in rows:
        polyline = wp.array(points, dtype=wp.vec3, device="cuda:0")
        reps = 8 if len(points) > 20000 else 40
        simplify_oracle(polyline, tol)
        tw.polyline.polyline_simplify(polyline, tol)
        wp.synchronize()
        old, new = [], []
        for _ in range(reps):
            start = time.perf_counter()
            simplify_oracle(polyline, tol)
            wp.synchronize()
            old.append((time.perf_counter() - start) * 1e3)
            start = time.perf_counter()
            tw.polyline.polyline_simplify(polyline, tol)
            wp.synchronize()
            new.append((time.perf_counter() - start) * 1e3)
        print(
            f"{name:22s} {len(points):7d} "
            f"{np.median(old):9.2f}/{np.min(old):<8.2f} "
            f"{np.median(new):9.2f}/{np.min(new):<8.2f} "
            f"{np.median(old) / np.median(new):7.2f}x"
        )


if __name__ == "__main__":
    main()
