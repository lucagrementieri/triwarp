"""
Generate API reference pages for every triwarp submodule, grouped by theme.

This is a standalone pre-build step, not a plugin hook: run it before building or serving the
docs. It writes real files under ``docs/`` -- ``docs/api/<module>.md`` and ``docs/SUMMARY.md``,
both gitignored build output -- where it previously handed them to ``gen-files``' virtual
filesystem.

Running it is not optional and the failure mode is quiet: Zensical does not support
``gen-files`` and ignores an unsupported plugin entry without a warning, so a build run without
this script first produces a site with no API reference at all. Only ``zensical build --strict``
reports it, as one unresolved cross-reference per API symbol referenced from a guide page.
"""

import shutil
from pathlib import Path

# ``mkdocs_gen_files`` is imported as a plain library, for ``Nav`` alone -- it is deliberately not
# registered in ``mkdocs.yml``'s ``plugins:`` list, and must not be. ``Nav`` is a self-contained
# nav-tree builder with no dependency on the plugin runtime, so reusing it keeps the
# SECTIONS-to-nav bullet nesting identical to what the ``gen-files`` build produced.
import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
docs_root = Path(__file__).parent
src = root / "triwarp"

# Curated theme groups. Order within a group matters: it is the docs nav order for that
# section. Every public module under triwarp/ — excluding triwarp/kernels/, __init__.py and
# private "_*.py" modules — must appear in exactly one group below, and the guard at the bottom
# fails the build otherwise, so adding a new module without classifying it here is caught
# immediately rather than silently falling back to a flat alphabetical list. The package is flat
# today; the guard walks rglob and names a module by its dotted path, so a future subpackage
# would surface as an unmapped dotted name rather than be skipped.
SECTIONS: dict[str, list[str]] = {
    "Primitives & creation": ["creation"],
    "Mesh structure & topology": [
        "mesh",
        "vertices",
        "edges",
        "triangles",
        "halfedge",
        "adjacency",
        "boundary",
        "selection",
        "validation",
        "homology",
        "tangent_space",
    ],
    # ``visibility`` sits here rather than under a query heading: ``ambient_occlusion``,
    # ``shape_diameter`` and ``thickness`` are per-point scalar shape descriptors of the same kind
    # as ``curvature``'s, which is how pymeshlab files them (``compute_scalar_*``).
    "Measures & shape descriptors": ["measures", "curvature", "bounds", "visibility"],
    # ``levelset`` last in the editing section: it is the one member whose output topology is not
    # the input's -- a level-set offset resamples the surface rather than moving it -- so a reader
    # meets the vertex-preserving edits first. It also holds ``marching_cubes``, which every
    # implicit-surface pipeline in the package ends in.
    "Mesh editing & repair": [
        # ``transform`` first: moving a mesh rigidly is the simplest edit in the section and the
        # one every other member leaves invariant, so a reader meets it before the edits that
        # change the surface. Its natural partner is ``registration``, which *fits* the matrices
        # this module applies; that pairing is carried by See Also in both directions rather than
        # by the shelving, since a module may appear in exactly one section.
        "transform",
        "repair",
        "holes",
        "combine",
        "remesh",
        "smoothing",
        "seams",
        "levelset",
    ],
    "Spatial queries": ["proximity", "ray", "neighbors", "intersection", "metrics"],
    # ``voxels`` is shelved with ``points`` / ``sample`` rather than with the spatial queries:
    # ``points.farthest_point_sample`` and ``voxels.voxel_down_sample`` are the two point-cloud
    # down-samplers and a reader looking for one should meet the other. The *code* stays split --
    # one returns indices and the other needs the whole grid machinery.
    "Point clouds, voxels & reconstruction": [
        "points",
        "sample",
        "voxels",
        "reconstruction",
        "registration",
    ],
    # ``energies`` immediately after ``laplacian``: the two halves of one subject, split because
    # 1 346 lines cannot carry both in one source order (which is the docs order). A reader
    # arriving from libigl looks for ``cotmatrix`` and ``crouzeix_raviart_*`` together, and the
    # adjacency plus the See Also in both directions is what replaces that.
    "Operators & solvers": ["laplacian", "energies", "linalg", "interpolation", "parametrization"],
    # ``geodesic_walk`` first: a direct combinatorial walk is the simpler thing, and section 11
    # orders a section by expected frequency of use. It is listed here rather than under
    # "Spatial queries" so the package's two geodesic entry points -- the walk and
    # ``heat.heat_geodesic`` -- are shelved together; a module may appear in exactly one section,
    # so widening this one is the fix and moving ``heat_geodesic`` is not.
    "Geodesics & heat-method solvers": ["geodesic_walk", "heat"],
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

# Clear the previous run's output before writing. The ``gen-files`` plugin held these pages in a
# virtual filesystem that started empty on every build; real files do not, so without this a
# module that is renamed or deleted leaves its old page behind as an orphan nothing links to.
api_root = docs_root / "api"
shutil.rmtree(api_root, ignore_errors=True)
api_root.mkdir(parents=True)

for section, modules in SECTIONS.items():
    for module_name in modules:
        doc_path = Path("api", f"{module_name}.md")
        nav[("API Reference", section, module_name)] = doc_path.as_posix()
        (docs_root / doc_path).write_text(f"::: triwarp.{module_name}\n")

# The narrative/guide pages are hand-written under docs/ (not generated), so they're listed here
# by hand rather than discovered -- this is the one place literate-nav's ordering is authored
# rather than derived from SECTIONS above. Keep this list and the files under docs/ in sync: a
# page added to one without the other is either unreachable from the nav or a 404 in this file.
GUIDE_PAGES: list[tuple[str, str] | tuple[str, str, list[tuple[str, str]]]] = [
    ("Getting started", "getting-started.md"),
    ("Concepts", "concepts.md"),
    (
        "Cookbook",
        "cookbook/index.md",
        [
            ("Cleaning and remeshing a rough mesh", "cookbook/clean-and-remesh.md"),
            ("Point cloud to watertight surface", "cookbook/point-cloud-to-surface.md"),
            ("Geodesic distance fields", "cookbook/geodesic-distance.md"),
            ("Aligning two scans", "cookbook/align-two-scans.md"),
        ],
    ),
    (
        "Migrating from another library",
        "migrating-from/index.md",
        [
            ("trimesh", "migrating-from/trimesh.md"),
            ("libigl", "migrating-from/igl.md"),
            ("Open3D", "migrating-from/open3d.md"),
            ("MeshLab / PyMeshLab", "migrating-from/meshlab.md"),
            ("potpourri3d", "migrating-from/potpourri3d.md"),
            ("PyTorch3D", "migrating-from/pytorch3d.md"),
        ],
    ),
    ("Performance", "performance.md"),
    ("Benchmarks", "benchmarks.md"),
]

with (docs_root / "SUMMARY.md").open("w") as nav_file:
    nav_file.write("* [Home](index.md)\n")
    for entry in GUIDE_PAGES:
        title, path, *rest = entry
        nav_file.write(f"* [{title}]({path})\n")
        for child_title, child_path in rest[0] if rest else []:
            nav_file.write(f"    * [{child_title}]({child_path})\n")
    nav_file.writelines(nav.build_literate_nav())
