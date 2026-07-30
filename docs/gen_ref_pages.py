"""Generate API reference pages for every triwarp submodule, grouped by theme."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
src = root / "triwarp"

# Curated theme groups. Order within a group matters: it is the docs nav order for that
# section. Every public module under triwarp/ — including subpackages such as triwarp/heat/,
# named by its dotted path relative to the package, and excluding triwarp/kernels/,
# __init__.py and private "_*.py" modules — must appear in exactly one group below. The guard
# at the bottom fails the build otherwise, so adding a new module without classifying it here
# is caught immediately rather than silently falling back to a flat alphabetical list.
SECTIONS: dict[str, list[str]] = {
    "Primitives": ["creation"],
    "Mesh structure & topology": [
        "mesh",
        "vertices",
        "edges",
        "triangles",
        "boundary",
        "adjacency",
        "halfedge",
        "tangent_space",
        "homology",
        "validation",
        "selection",
    ],
    "Mesh editing & repair": [
        "repair",
        "hole_filling",
        "combine",
        "remesh",
        "smoothing",
        "seams",
    ],
    "Queries & measures": [
        "proximity",
        "neighbors",
        "bounds",
        "ray",
        "distance",
        "tracing",
        "intersection",
        "curvature",
        "convex",
        "shading",
    ],
    "Operators & fields": ["laplacian", "linalg", "interpolation", "parametrization"],
    "Heat-method solvers": ["heat.distance", "heat.vector", "heat.signed"],
    "Point clouds & registration": ["points", "sample", "reconstruction", "registration"],
    "Curves": ["polyline"],
    "Attributes & I/O": ["texture", "io"],
    "Arrays & infrastructure": ["array", "reduce", "grouping", "graph", "typing", "constants"],
}

listed = {module_name for modules in SECTIONS.values() for module_name in modules}
actual = {
    ".".join(path.relative_to(src).with_suffix("").parts)
    for path in src.rglob("*.py")
    if path.name != "__init__.py"
    and not path.name.startswith("_")
    # Kernel modules are the Warp DSL and are deliberately undocumented. Testing the first path
    # component (rather than a substring) keeps a future triwarp/<pkg>/kernels.py documentable.
    and path.relative_to(src).parts[0] != "kernels"
}
if listed != actual:
    unmapped = actual - listed
    stale = listed - actual
    raise SystemExit(
        "gen_ref_pages: SECTIONS is out of sync with the modules under triwarp/ — "
        f"unmapped modules (add to a section): {sorted(unmapped)}; "
        f"stale entries (module no longer exists): {sorted(stale)}"
    )

for section, modules in SECTIONS.items():
    for module_name in modules:
        module_path = src.joinpath(*module_name.split(".")).with_suffix(".py")
        doc_path = Path("api", f"{module_name}.md")
        nav[("API Reference", section, module_name)] = doc_path.as_posix()

        with mkdocs_gen_files.open(doc_path, "w") as fd:
            print(f"::: triwarp.{module_name}", file=fd)

        mkdocs_gen_files.set_edit_path(doc_path, module_path.relative_to(root))

with mkdocs_gen_files.open("SUMMARY.md", "w") as nav_file:
    nav_file.write("* [Home](index.md)\n")
    nav_file.writelines(nav.build_literate_nav())
