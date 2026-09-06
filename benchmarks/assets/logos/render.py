"""
Rasterize each vendored library mark in ``svg/`` to a fixed-size, square, transparent PNG.

Run once (or after touching a source SVG or ``SIZE``):

    uv run python benchmarks/assets/logos/render.py

``registry.py`` and ``plot_comparison.py`` only ever read the PNGs this writes -- neither needs
``cairosvg`` at chart-render time, only this script does, and only when regenerating.

Every source mark under ``svg/`` is either exactly square already (the common case: a project's
own icon-only mark, as opposed to its wordmark) or, for triwarp's own lockup, wider than it is
tall. Rather than distort the wide ones to fit a square, this renders each at its native aspect
ratio scaled to fit inside ``SIZE x SIZE`` and centers it on a transparent square canvas -- so
every PNG this writes is ``SIZE x SIZE`` regardless of the source's own proportions, which is the
property a chart placing these as fixed-size axis glyphs actually needs.

A neutral mid-gray chip is composited *behind* every mark, baked into the PNG rather than left to
the chart to add -- one mark (PyTorch3D's icon) is drawn in pure white with no dark element at
all, which is legible on that project's own dark navy site and invisible against
`plot_comparison.py`'s light-mode white chart surface with no chip behind it. A chip sized and
colored consistently for every mark, not special-cased for that one, is what keeps the chart
script's placement logic identical for every library regardless of what background each project
originally designed its mark against.
"""

from __future__ import annotations

import io
from pathlib import Path

import cairosvg
from PIL import Image, ImageDraw

# 96px source, ~2x an anticipated ~48px on-chart display size, so the glyph stays crisp on a
# retina/HiDPI figure export (`dpi=200`+, which `plot_comparison.py` uses throughout).
SIZE = 96

# A light, low-chroma chip -- legible against both this project's light (#ffffff) and dark
# (#14161b) chart surfaces, and neutral enough not to compete with any of the marks' own colors.
CHIP_COLOR = (216, 216, 216, 235)
CHIP_MARGIN = 4  # px of chip visible past the mark's own bounding box, before its rounding

HERE = Path(__file__).parent
SVG_DIR = HERE / "svg"
PNG_DIR = HERE / "png"


def render_one(svg_path: Path, png_path: Path, size: int = SIZE) -> None:
    """Rasterize one SVG to a `size x size` transparent PNG, on a chip, centered and fitted."""
    # Render oversized on the long axis, then fit -- asking cairosvg for the exact output size
    # directly would stretch a non-square source instead of letter/pillar-boxing it.
    raw = cairosvg.svg2png(url=str(svg_path), output_width=size * 4, output_height=size * 4)
    rendered = Image.open(io.BytesIO(raw)).convert("RGBA")
    rendered.thumbnail((size, size), Image.LANCZOS)

    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    chip_box = (CHIP_MARGIN, CHIP_MARGIN, size - CHIP_MARGIN, size - CHIP_MARGIN)
    ImageDraw.Draw(canvas).rounded_rectangle(chip_box, radius=size * 0.18, fill=CHIP_COLOR)
    offset = ((size - rendered.width) // 2, (size - rendered.height) // 2)
    canvas.paste(rendered, offset, rendered)
    canvas.save(png_path)


def main() -> None:
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    svg_paths = sorted(SVG_DIR.glob("*.svg"))
    if not svg_paths:
        raise SystemExit(f"no .svg sources found under {SVG_DIR}")
    for svg_path in svg_paths:
        png_path = PNG_DIR / f"{svg_path.stem}.png"
        render_one(svg_path, png_path)
        print(f"  {svg_path.name} -> {png_path.relative_to(HERE.parent.parent)}")


if __name__ == "__main__":
    main()
