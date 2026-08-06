#!/usr/bin/env python
"""Report the installed Warp version vs. the version stamped in the reference files.

Usage: python reference/warp_api/warp_version.py

Helps decide whether the API reference under reference/warp_api/ needs
regenerating after a Warp upgrade (see REGENERATE.md).

The stamps track the *docs* version each file was transcribed from, which is
allowed to run ahead of the installed ``warp-lang`` while an upgrade is in
flight -- that is reported separately from a genuinely stale file.
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


def version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version))


def main() -> None:
    installed = installed_version()
    print(f"Installed Warp version: {installed}")
    print(f"Reference files (in {HERE}):")
    for md_path in sorted(HERE.glob("*.md")):
        stamped = stamped_version(md_path)
        if stamped is None:
            continue
        if stamped == installed:
            flag = ""
        elif not installed[:1].isdigit():
            flag = "  <-- cannot compare, Warp not importable"
        elif version_key(stamped) > version_key(installed):
            flag = "  <-- ahead of the installed warp-lang (upgrade in flight)"
        else:
            flag = "  <-- stale, regenerate"
        print(f"  {md_path.name:<16} stamped Warp {stamped}{flag}")


if __name__ == "__main__":
    main()
