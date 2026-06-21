#!/usr/bin/env python
"""Report the installed Warp version vs. the version stamped in the reference files.

Usage: python reference/warp_api/warp_version.py

Helps decide whether the API reference under reference/warp_api/ needs
regenerating after a Warp upgrade (see REGENERATE.md).
"""

import re
from pathlib import Path

STAMP_RE = re.compile(r"\(Warp ([0-9][^)]*)\)")
HERE = Path(__file__).parent


def installed_version() -> str:
    try:
        import warp as wp

        return wp.config.version
    except Exception as exc:  # warp not installed / import failure
        return f"<unavailable: {exc}>"


def stamped_version(md_path: Path) -> str | None:
    match = STAMP_RE.search(md_path.read_text())
    return match.group(1) if match else None


def main() -> None:
    installed = installed_version()
    print(f"Installed Warp version: {installed}")
    print(f"Reference files (in {HERE}):")
    for md_path in sorted(HERE.glob("*.md")):
        stamped = stamped_version(md_path)
        if stamped is None:
            continue
        flag = "" if stamped == installed else "  <-- stale, regenerate"
        print(f"  {md_path.name:<16} stamped Warp {stamped}{flag}")


if __name__ == "__main__":
    main()
