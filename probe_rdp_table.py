"""Produce the final speedup table in one session, both devices, for the docstring.

One process, one clock state, A and B interleaved per repetition, medians reported with minima
(``.claude/CLAUDE.md`` section 13). The accepted set is asserted identical in every cell here too,
so the table cannot be read as a speedup on a different answer.
"""

import sys
import time

import numpy as np
import warp as wp

sys.path.insert(0, "benchmarks")

import triwarp as tw
from probe_rdp_gate import boundary_polyline, simplify_oracle, spiral


def measure(points: np.ndarray, tol: float, device: str, reps: int) -> tuple[float, float, float, float, int]:
    polyline = wp.array(points, dtype=wp.vec3, device=device)
    expected = simplify_oracle(polyline, tol)
    got = tw.polyline.polyline_simplify(polyline, tol)[1].numpy()
    assert np.array_equal(got, expected), f"keep set differs on {device}"
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
    return (
        float(np.median(old)), float(np.min(old)),
        float(np.median(new)), float(np.min(new)), len(got),
    )


def main() -> None:
    rim = boundary_polyline("rim_long")
    saddle = boundary_polyline("saddle")
    small = boundary_polyline("saddle_small")

    def mean_seg(pts: np.ndarray) -> float:
        return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).mean())

    deep = spiral(4096, 100.0)  # the deepest shape found: 204 rounds, 5.0% of n
    rows = [
        ("rim_long", rim, 1e-3 * mean_seg(rim)),
        ("rim_long coarse", rim, 1e-1 * mean_seg(rim)),
        ("spiral, 100 turns", deep, 1e-3 * float(np.ptp(deep.reshape(-1, 3), axis=0).max())),
        ("saddle", saddle, 1e-3 * mean_seg(saddle)),
        ("saddle_small", small, 1e-3 * mean_seg(small)),
    ]
    for device in ("cuda:0", "cpu"):
        print(f"\n=== {device} : median/min ms over interleaved reps ===")
        print(f"{'case':20s} {'n':>7s} {'kept':>7s} {'recursive':>18s} "
              f"{'level-sync':>18s} {'ratio':>8s}")
        for name, points, tol in rows:
            reps = 12 if len(points) > 20000 else 40
            old_med, old_min, new_med, new_min, kept = measure(points, tol, device, reps)
            print(
                f"{name:20s} {len(points):7d} {kept:7d} "
                f"{old_med:9.2f}/{old_min:<8.2f} {new_med:9.2f}/{new_min:<8.2f} "
                f"{old_med / new_med:7.2f}x"
            )


if __name__ == "__main__":
    main()
