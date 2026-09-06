"""
Render `benchmarks/aggregate.py` cells as comparison histograms, each library named by its logo.

One PNG pair (light + dark) per cell, each competing library identified by its own logo rather
than a generic categorical color:

    uv run python benchmarks/plot_comparison.py <json_dir> --out docs/assets/benchmarks/
    uv run python benchmarks/plot_comparison.py <json_dir> --out docs/assets/benchmarks/ --hero
    uv run python benchmarks/plot_comparison.py <json_dir> --out /tmp/charts --group cotmatrix

Design, and why (plans/release.md section 4.3 has the full reasoning; this is the summary that
matters for reading the code below):

- **Identity is carried by the logo + a muted text label, not by hue.** A logo already
  disambiguates *which* library a bar is; burning a categorical color per bar on top of that would
  be decoration competing with the one thing color should be doing here.
- **Color's one job on this chart is "is this triwarp."** The subject bar (triwarp-cuda, or
  triwarp-cpu if that is the only triwarp row in the cell) takes the site's own brand accent; every
  reference bar takes one shared muted gray. This is a two-value status encoding, not an 8-hue
  categorical one -- it does not need the dataviz skill's categorical CVD/chroma validator (which
  is built for genuine multi-hue identity palettes and will flag an intentionally achromatic gray
  as "reads as gray"; that is the point of a de-emphasis color, not a defect). What *was* run
  through the validator is the accent-vs-gray pair itself, in both modes -- see `_PALETTE` below
  for the result and the one accepted deviation.
- **Horizontal bars**, sorted ascending by time (fastest at the top), each with a direct value
  label at its tip -- a static image has no hover layer, so the number has to be readable without
  one.
- **Log scale only when the cell's own max/min ratio warrants it** (> `LOG_SCALE_RATIO`), and the
  chart's own subtitle says which axis it used, so it is never ambiguous. Direct labels stay
  regardless of scale, so the log axis's usual "can't recover the value" weakness does not apply.
- **A markdown table ships beside every chart** with the same numbers, for search/copy and as the
  accessibility "table view" the dataviz skill's non-negotiables ask for.
- **Never a silent drop.** A cell with no comparable reference, or a requested hero/group cell
  missing from the data, is reported to stderr and skipped -- not quietly absent from the output.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this runs in CI and over SSH, never against a display
import matplotlib.pyplot as plt
import warp as wp
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).parent))
import aggregate
from assets.logos import registry

# The threshold a cell's max/min ratio is compared against to choose log over linear -- see the
# module docstring. CLAUDE.md's own measured ranges (85-260x between the fastest and slowest
# library on one operation) are exactly the case a linear axis hides.
LOG_SCALE_RATIO = 15.0

# One curated cell per area, picked from a real benchmark run for being both visually clean (a
# clear ranking, no all-declined rows) and broad (topology, DDG, geodesics, reconstruction,
# editing, measures) -- see plans/release.md section 4.4. Keys are exactly the `CellKey` shape
# `aggregate.load` produces: (module, group, mesh_name, rest).
HERO_CELLS: list[aggregate.CellKey] = [
    ("test_edges", "faces_to_edges", "dragon", ()),
    ("test_laplacian", "cotmatrix", "dragon", ()),
    ("test_heat", "heat_geodesic", "sphere_small", (("setup", "full"),)),
    ("test_remesh", "quadric_decimate", "saddle_graded", (("target_ratio", "0.1"),)),
    ("test_reconstruction", "ball_pivoting", "bunny", ()),
    ("test_bounds", "aabb", "bunny_decimated", ()),
]


class _Palette:
    """
    One accent-vs-muted pair per mode, run through the dataviz skill's `validate_palette.js`.

    Light: `#4f8000` / `#9a9a9a` on `#ffffff` -- both clear the categorical CVD and normal-vision
    floors (worst pair dE 17.0 CVD / 20.6 normal); the gray's own contrast (2.74:1) sits in the
    documented WARN band, which is legal because every bar carries a direct value label (the
    required relief channel).

    Dark: `#76b900` / `#7a7a7a` on `#14161b` -- CVD and contrast both pass outright. The
    validator's lightness-band check flags the accent (L=0.713, wants <=0.67) -- accepted rather
    than fixed, because darkening it further drifts from the actual brand green
    (`docs/stylesheets/extra.css`'s `--md-primary-fg-color`/`--md-accent-fg-color` ramp) for a
    check whose purpose (banding across an 8-slot categorical ramp) does not apply to a single
    accent color.
    """

    def __init__(
        self, surface: str, accent: str, muted: str, text: str, text_muted: str, grid: str
    ) -> None:
        self.surface = surface
        self.accent = accent
        self.muted = muted
        self.text = text
        self.text_muted = text_muted
        self.grid = grid


LIGHT = _Palette(
    surface="#ffffff",
    accent="#4f8000",
    muted="#9a9a9a",
    text="#15200a",
    text_muted="#555555",
    grid="#e4e4e4",
)
DARK = _Palette(
    surface="#14161b",
    accent="#76b900",
    muted="#7a7a7a",
    text="#eef0ea",
    text_muted="#a9a9a9",
    grid="#2a2d33",
)


def _slugify(text: str) -> str:
    """Turn a cell label into a filesystem-safe stem."""
    text = text.lower().replace("[", "-").replace("]", "").replace(" ", "-")
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    return re.sub(r"-{2,}", "-", text).strip("-")


def _format_duration(seconds: float) -> str:
    """Format one duration in whichever unit reads most naturally for its own magnitude."""
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def _rounded_hbar(ax, y: float, width: float, height: float, color: str) -> None:
    """Draw one horizontal bar with a rounded tip and a square baseline (mark spec: 4px radius)."""
    radius = min(height * 0.4, width * 0.08) if width > 0 else 0.0
    ax.add_patch(
        FancyBboxPatch(
            (0, y - height / 2),
            width,
            height,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            linewidth=0,
            facecolor=color,
            mutation_aspect=1,
        )
    )


#  A `png/*.png` source is 96x96 raw pixels with no DPI metadata, so `OffsetImage` treats each
# pixel as one point at zoom=1 -- `_LOGO_ZOOM` is chosen so the on-figure glyph reads at roughly
# the same cap-height as the 9pt label text beside it (96 * 0.15 ≈ 14.4pt).
_LOGO_ZOOM = 0.15
_LOGO_OFFSET_PT = -16  # logo's right edge, points from the axis
_LABEL_OFFSET_WITH_LOGO_PT = -34  # label's right edge, points from the axis, clear of the logo
_LABEL_OFFSET_ALONE_PT = -10


def _place_logo_and_label(ax, y: int, library_id: str, palette: _Palette) -> None:
    """Draw the library's logo (if it has one) and its text label to the left of row `y`."""
    label = registry.display_label(library_id)
    logo = registry.logo_path(library_id)
    if logo is not None and logo.exists():
        image = OffsetImage(plt.imread(logo), zoom=_LOGO_ZOOM)
        box = AnnotationBbox(
            image,
            (0.0, y),
            xycoords=("axes fraction", "data"),
            xybox=(_LOGO_OFFSET_PT, 0),
            boxcoords="offset points",
            frameon=False,
            box_alignment=(1.0, 0.5),
            annotation_clip=False,
        )
        ax.add_artist(box)
        label_offset = _LABEL_OFFSET_WITH_LOGO_PT
    else:
        label_offset = _LABEL_OFFSET_ALONE_PT
    ax.annotate(
        label,
        xy=(0.0, y),
        xycoords=("axes fraction", "data"),
        xytext=(label_offset, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=9,
        color=palette.text_muted,
        annotation_clip=False,
    )


def render_cell(
    key: aggregate.CellKey, libs: dict[str, float], out_dir: Path, provenance: str
) -> bool:
    """Render one `(light, dark)` PNG pair plus a markdown table for one benchmark cell."""
    if "triwarp-cuda" in libs:
        subject_id = "triwarp-cuda"
    elif "triwarp-cpu" in libs:
        subject_id = "triwarp-cpu"
    else:
        print(f"  ! {aggregate.cell_label(key)}: no triwarp row, skipped", file=sys.stderr)
        return False
    references = {lib: t for lib, t in libs.items() if not registry.is_subject(lib)}
    if not references:
        print(f"  ! {aggregate.cell_label(key)}: no reference library, skipped", file=sys.stderr)
        return False

    rows = sorted(libs.items(), key=lambda item: item[1])  # fastest first
    values_ms = [seconds * 1e3 for _, seconds in rows]
    use_log = (max(values_ms) / min(values_ms)) > LOG_SCALE_RATIO if min(values_ms) > 0 else False

    stem = _slugify(aggregate.cell_label(key))
    table_lines = [
        "| Library | Median |",
        "|---|---|",
        *(
            f"| {registry.display_label(lib)} | {_format_duration(seconds)} |"
            for lib, seconds in rows
        ),
    ]
    (out_dir / f"{stem}.md").write_text("\n".join(table_lines) + "\n")

    for mode, palette in (("light", LIGHT), ("dark", DARK)):
        fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(rows) + 1.3), dpi=200)
        fig.patch.set_facecolor(palette.surface)
        ax.set_facecolor(palette.surface)

        for i, (lib, seconds) in enumerate(rows):
            width = seconds * 1e3
            color = palette.accent if lib == subject_id else palette.muted
            _rounded_hbar(ax, i, width, height=0.62, color=color)
            ax.text(
                width * (1.02 if not use_log else 1.06),
                i,
                _format_duration(seconds),
                va="center",
                ha="left",
                fontsize=9,
                color=palette.text,
            )
            _place_logo_and_label(ax, i, lib, palette)

        ax.set_ylim(-0.6, len(rows) - 0.4)
        # Patches added via `ax.add_patch` (the rounded bars) do not feed autoscale the way a
        # real `ax.barh` call would, so both bounds are set explicitly here -- an unset right
        # bound silently left every bar at matplotlib's default (0, 1) axes view once, which
        # rendered every bar the same width regardless of its actual value.
        if use_log:
            ax.set_xscale("log")
            ax.set_xlim(min(values_ms) * 0.5, max(values_ms) * 1.6)
        else:
            ax.set_xlim(0, max(values_ms) * 1.22)
        ax.invert_yaxis()  # fastest at the top
        ax.set_yticks([])
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(palette.grid)
        ax.tick_params(axis="x", colors=palette.text_muted, labelsize=8)
        ax.grid(axis="x", color=palette.grid, linewidth=1, zorder=0)
        ax.set_axisbelow(True)

        scale_note = "log scale" if use_log else "linear scale"
        ax.set_title(aggregate.cell_label(key), color=palette.text, fontsize=12, loc="left", pad=14)
        ax.text(
            0.0,
            1.02,
            f"median wall time, lower is better — {scale_note}",
            transform=ax.transAxes,
            color=palette.text_muted,
            fontsize=8.5,
        )
        fig.text(0.01, 0.01, provenance, color=palette.text_muted, fontsize=7)

        fig.subplots_adjust(left=0.30, right=0.96, top=0.86, bottom=0.14)
        fig.savefig(out_dir / f"{stem}-{mode}.png", facecolor=fig.get_facecolor())
        plt.close(fig)
    return True


def _provenance(json_dir: Path) -> str:
    """One caption line: hardware/Warp version (this process) plus the run's own commit/date."""
    device_name = "CPU"
    if wp.is_cuda_available():
        device_name = wp.get_device("cuda:0").name
    commit, when = "unknown", "unknown date"
    for path in sorted(json_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        commit = data.get("commit_info", {}).get("id", commit)[:7]
        when = data.get("datetime", when)[:10]
        break
    return f"{device_name} · warp-lang {wp.config.version} · commit {commit} · {when}"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json_dir", type=Path, help="directory of --benchmark-json output files")
    ap.add_argument("--out", type=Path, required=True, help="directory to write charts into")
    ap.add_argument(
        "--group", action="append", default=None, help="render only this group (repeatable)"
    )
    ap.add_argument(
        "--hero", action="store_true", help="render only the curated HERO_CELLS, not every cell"
    )
    args = ap.parse_args(argv)

    cells, modules, _suspects = aggregate.load(str(args.json_dir))
    print(f"loaded {len(modules)} modules, {len(cells)} cells from {args.json_dir}")
    provenance = _provenance(args.json_dir)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.hero:
        targets = []
        for key in HERO_CELLS:
            if key not in cells:
                label = aggregate.cell_label(key)
                print(
                    f"  ! hero cell {label} not found in {args.json_dir}, skipped", file=sys.stderr
                )
                continue
            targets.append((key, cells[key]))
    elif args.group:
        wanted = set(args.group)
        targets = [(k, libs) for k, libs in cells.items() if k[1] in wanted]
        found_groups = {k[1] for k, _ in targets}
        for missing in wanted - found_groups:
            print(f"  ! group {missing!r} not found in {args.json_dir}", file=sys.stderr)
    else:
        targets = list(cells.items())

    rendered = sum(render_cell(key, libs, args.out, provenance) for key, libs in targets)
    print(f"wrote {rendered} chart(s) (light+dark PNG + a table each) to {args.out}")


if __name__ == "__main__":
    main()
