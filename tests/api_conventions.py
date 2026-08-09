"""
Static scan of the public API's shape: names, summaries, file layout and module boundaries.

Eight conventions the package holds to, each one a defect class that was actually found rather than
an aesthetic preference. They are checked by an ``ast`` scan of ``triwarp/`` (excluding
``kernels/``, ``__init__.py`` and private ``_*.py`` modules) plus a listing of ``tests/`` and
``benchmarks/``, and [`tests/test_api_conventions.py`](test_api_conventions.py) fails the default
test run on any violation:

1. **A one-line summary says what the function returns, not which C++ call it wraps.** mkdocstrings
   renders the summary as the entry in the module's API index, so a library name there turns the
   index into a table of bindings. Attribution is wanted -- in a ``Notes`` block or a ``See Also``,
   one line down.
2. **A ``*_mask`` producer returns a boolean array.** The suffix is a family, and a member that
   returns something else makes the family unreadable. A function that *consumes* a mask (it takes a
   ``*mask`` parameter) is named for its input and is exempt.
3. **A module summary does not end in "(Warp)" or "on NVIDIA Warp".** The whole package is Warp.
4. **Every module has a test file and a benchmark file named for it**, and every ``test_*.py`` in
   either suite corresponds to a module -- ``tests/test_heat_distance.py`` for
   ``triwarp/heat/distance.py``, dots becoming underscores.
5. **A private name stays inside its module.** A ``_helper`` imported across a module boundary is a
   function that should have been public, and the alias-on-import (``import _x as x``) is the tell.
6. **Two modules do not export the same public name**, outside a written allowlist.
7. **A top-level kernel module is named for the public module it backs**, and vice versa
   (``.claude/CLAUDE.md`` section 4). This is what stops a wrapper module from being created while
   its kernels are left behind under the old name.
8. **A private helper is defined below its first caller** (``.claude/CLAUDE.md`` section 11's
   stepdown rule), so a reader never jumps backward to a definition they have not met. The 50 sites
   that predate the check are an explicit, staleness-checked debt list, not an exemption.

Why a static scan rather than importing ``triwarp``
---------------------------------------------------
Importing would make the verdict depend on Warp's module cache and on which optional dependencies
resolve, and would say nothing about files (checks 4 and 7) at all. A scan reads the tree as
written, so its answer is the same in every environment and in every pytest invocation.
"""

from __future__ import annotations

import ast
import functools
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parent
_PACKAGE_DIR = _REPO_ROOT / "triwarp"
_KERNELS_DIR = _PACKAGE_DIR / "kernels"
_BENCHMARKS_DIR = _REPO_ROOT / "benchmarks"

# --- check 1 -----------------------------------------------------------------------------------

# Reference libraries whose name in a *summary* line is the defect. Spelled as a single alternation
# so the allowlist below is the only escape, and it is per-module rather than per-name.
_LIBRARY_IN_SUMMARY = re.compile(
    r"MeshLib|MeshLab|libigl|pymeshlab|igl::|geometry-central|potpourri|VCG|promesh|trimesh\."
)

# ``mesh.Trimesh`` genuinely mirrors ``trimesh.Trimesh``, and that is load-bearing: the property
# names (``.centroid``, ``.volume``, ``.euler_characteristic``) are chosen to match, so a reader
# needs to be told at the top. No other module may name a reference library in a summary.
_SUMMARY_LIBRARY_ALLOWLIST: dict[str, frozenset[str]] = {"mesh": frozenset({"trimesh."})}

# --- check 3 -----------------------------------------------------------------------------------

_WARP_SUFFIX = re.compile(r"\(Warp\)|on NVIDIA Warp")

# --- check 4 -----------------------------------------------------------------------------------

# Suite files that deliberately mirror no module.
_EXTRA_TEST_FILES = frozenset(
    {
        "parity",  # the cross-suite parity gate
        "api_conventions",  # this gate
        "map_uniform_probe",  # a Warp-behaviour probe, not a module's coverage
    }
)
_EXTRA_BENCHMARK_FILES = frozenset({"meshes"})  # mesh-fixture invariants, nothing timed

# Modules with nothing to test or nothing to time, each for a stated reason.
_MODULES_WITHOUT_TESTS = frozenset({"constants"})  # module-level ``wp.constant`` values only
_MODULES_WITHOUT_BENCHMARKS = frozenset(
    {
        "constants",  # no callables
        "typing",  # annotations and rank/dtype guards; nothing with a runtime cost
        "homology",  # no reference library exposes a homology basis to time against
        "io",  # meshio round-trips, i.e. a benchmark of meshio
        "ray",  # every ray query is timed through proximity's BVH groups
    }
)

# --- check 5 -----------------------------------------------------------------------------------

# Cross-module private imports this package accepts, as (importing module, ``owner._name``) pairs
# with the reason each one is not the defect the check is looking for. Anything else fires.
#
# The defect the check exists for is a private *operation* with callers in two modules -- a function
# that was written where it was first needed and should have been public
# (``proximity._default_mesh_query_max_dist``, four importers, one of them aliasing it back to a
# public-looking name). What survives below is the two things that are not that: a string-to-enum
# lookup table shared by a wrapper and its dispatcher, and a dtype conversion that would be a public
# name whose whole content is ``wp.array(..., dtype=...)``.
_PRIVATE_IMPORT_ALLOWLIST: dict[tuple[str, str], str] = {
    ("reduce", "array._sorted_copy"): (
        "reduce.median needs a sorted copy, and array.sort_and_argsort is the public form -- which "
        "also builds the permutation median throws away"
    ),
    ("combine", "holes._PackedLoops"): (
        "the flat-plus-sizes loop representation both dynamic programs consume. It is a data "
        "type, not an operation -- publishing it would make an internal layout part of the API, "
        "and every public entry point on both sides takes and returns plain arrays"
    ),
    ("combine", "holes._EdgeTable"): (
        "the rim-opposite-vertex lookup the dihedral term of both metrics needs; shared for the "
        "same reason as the metric table below, so the single-hole cap and the two-rim band cannot "
        "disagree about what an edge's opposite vertex is"
    ),
    ("combine", "holes._BAD_TRIANGULATION_METRIC"): (
        "the host-side twin of the kernels' BAD_METRIC sentinel, so the two DP tables are "
        "initialised to the same unusable value; a second copy would be a second constant to keep "
        "in step with the kernel's"
    ),
    ("combine", "holes._STITCH_METRIC_IDS"): (
        "the metric-name-to-kernel-flag table, shared so the two entry points validate the same "
        "spellings; a public copy would be a second source of truth for one enum"
    ),
    ("combine", "holes._patch_mask"): (
        "marks which faces a fill added, by index range -- bookkeeping over another function's "
        "return convention rather than a mesh operation"
    ),
    ("combine", "holes._mean_rim_edge_length"): (
        "the default target edge length for a patch, derived from the rim it was fitted to; it is "
        "one caller's default, not a measurement anyone would reach for"
    ),
    ("remesh", "triangles._QUALITY_METRICS"): (
        "the same string-to-kernel-flag table as the stitch one above, for face_quality's metric "
        "argument, so remesh's objective names cannot drift from triangles.face_quality's"
    ),
    ("reconstruction", "remesh._flip_interior_edges"): (
        "ball pivoting flips its own output's interior edges; every public remesh entry point "
        "rebuilds connectivity as well, which reconstruction must not do to a fresh triangulation"
    ),
}

# --- check 6 -----------------------------------------------------------------------------------

# Duplicate public names that are correct in both vocabularies. ``array.concatenate`` /
# ``array.split`` mirror ``numpy.concatenate`` / ``numpy.split``, while ``combine.concatenate``
# mirrors ``trimesh.util.concatenate`` and ``combine.split`` mirrors ``trimesh.Trimesh.split``;
# none is reachable unqualified and the naming rule requires both spellings of each.
_DUPLICATE_NAME_ALLOWLIST: dict[str, frozenset[str]] = {
    "concatenate": frozenset({"array", "combine"}),
    "split": frozenset({"array", "combine"}),
}

# --- check 7 -----------------------------------------------------------------------------------

# Kernel-side libraries that back no single public module.
_SHARED_KERNEL_MODULES = frozenset({"predicates", "scatter"})

# Public modules that launch no kernel of their own and so have no kernel module.
_MODULES_WITHOUT_KERNELS = frozenset({"constants", "homology", "io", "mesh", "typing"})

# --- check 8 ------------------------------------------------------------------------------------

# Private helpers that sit above their first caller today. CLAUDE.md section 11 says a helper must
# never appear above the caller it serves, but the package drifted off that rule wholesale before
# the check existed: these 50 sites across 15 modules are internally consistent in doing the
# opposite, and reordering fifteen files at once is a diff nobody can review. So they are an
# explicit debt list rather than a silent exemption -- **entries come out, they do not go in**.
# Drain one whenever you are editing its module for another reason; a *new* helper must be placed
# correctly, which is exactly what this check now enforces.
_HELPER_ORDER_ALLOWLIST: dict[str, frozenset[str]] = {
    "array": frozenset({"_sorted_copy"}),
    "combine": frozenset({"_closest_loop_pair", "_longest_increasing_subsequence"}),
    "creation": frozenset({"_icosphere_face_table"}),
    "distance": frozenset(
        {
            "_chamfer",
            "_distances_mesh_to_mesh",
            "_distances_points_to_mesh",
            "_distances_points_to_points",
            "_empty_chamfer",
            "_hausdorff",
            "_launch_nn_term",
            "_launch_surface_term",
            "_maybe_taped",
            "_reduce",
            "_reduction_scale",
            "_square",
            "_validate_diff_reduction",
            "_validate_point_reduction",
            "_zero_loss",
        }
    ),
    "holes": frozenset(
        {
            "_hole_loops",
            "_mean_rim_edge_length",
            "_patch_mask",
            "_run_hole_dp",
            "_traceback_triangles",
            "_unpack_loops",
        }
    ),
    "io": frozenset({"_import_meshio"}),
    "ray": frozenset({"_validate_ray_inputs"}),
    "reconstruction": frozenset(
        {"_bpa_wave", "_lexicographic_triangulation", "_orient2d", "_repeated_oriented_triangles"}
    ),
    "reduce": frozenset(
        {
            "_launch_axis_scalar",
            "_launch_global_bool_tiled",
            "_launch_global_scalar_tiled",
            "_validate_scalar_array",
        }
    ),
    "registration": frozenset(
        {
            "_identity_mat44",
            "_is_mesh_target",
            "_resolve_initial",
            "_robust_scale_from_residuals",
            "_target_index",
        }
    ),
    "remesh": frozenset({"_flip_region_faces"}),
    "sample": frozenset({"_dart_throw_blue_noise"}),
    "smoothing": frozenset(
        {
            "_apply_operator",
            "_apply_volume_constraint",
            "_boundary_verts_mask",
            "_edge_weight_matrix",
        }
    ),
    "texture": frozenset({"_check_uv_in_range"}),
    "typing": frozenset({"_shape_2d", "_shape_3d"}),
}


@dataclass(frozen=True)
class PublicFunction:
    """A module-level ``def`` whose name does not start with an underscore."""

    module: str
    name: str
    lineno: int
    summary: str
    returns: str | None
    parameters: tuple[str, ...]

    @property
    def site(self) -> str:
        """Render as ``path:lineno`` so editors and terminals can jump to it."""
        return f"triwarp/{self.module.replace('.', '/')}.py:{self.lineno}"


@dataclass(frozen=True)
class PublicModule:
    """One documented wrapper module: its dotted name, its summary and its public functions."""

    name: str
    path: str
    summary: str
    functions: tuple[PublicFunction, ...]
    private_imports: tuple[tuple[str, int], ...]
    """``("owner._name", lineno)`` for every private name reached across a module boundary."""

    early_helpers: tuple[tuple[str, int, int], ...]
    """``(name, def_lineno, first_caller_lineno)`` for each private helper defined above its
    first caller."""


@dataclass
class PackageScan:
    """Everything the checks below read, gathered in one pass over ``triwarp/``."""

    modules: dict[str, PublicModule] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def functions(self) -> list[PublicFunction]:
        """Every public function in the package, in module then source order."""
        return [function for module in self.modules.values() for function in module.functions]


def _summary(node: ast.Module | ast.FunctionDef) -> str:
    """Collapse a docstring's first paragraph onto one line, or ``""`` when there is none."""
    docstring = ast.get_docstring(node) or ""
    return " ".join(docstring.strip().split("\n\n")[0].split())


def _dotted(node: ast.expr) -> tuple[str, ...]:
    """Resolve a ``Name``/``Attribute`` chain to its dotted parts, or ``()`` for anything else."""
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return ()
    parts.append(current.id)
    return tuple(reversed(parts))


def _private_imports(tree: ast.Module, module: str) -> list[tuple[str, int]]:
    """
    Every private name this module reaches for across a module boundary.

    Two spellings reach one: ``from triwarp.x import _y`` (which ``ray.py`` used to alias straight
    back to a public-looking name) and the attribute form ``tw.x._y``. Both are collected as
    ``"x._y"`` so the allowlist is keyed the same way regardless of which was used.
    """
    found: dict[tuple[str, int], None] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("triwarp"):
            owner = node.module.removeprefix("triwarp").lstrip(".")
            if owner in ("", module) or owner.startswith("kernels"):
                continue
            for alias in node.names:
                if alias.name.startswith("_"):
                    found[(f"{owner}.{alias.name}", node.lineno)] = None
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            parts = _dotted(node)
            if len(parts) == 3 and parts[0] == "tw" and parts[1] != module:
                found[(f"{parts[1]}.{parts[2]}", node.lineno)] = None
    return sorted(found)


def _early_helpers(tree: ast.Module) -> list[tuple[str, int, int]]:
    """
    Private module-level helpers whose ``def`` precedes their first reference in the same module.

    A reference inside the helper's own body is skipped, so recursion and a self-referencing
    closure do not count as callers. A helper with no caller at all is not reported here -- check 5
    already covers the cross-module case, and a genuinely dead helper is a different defect.
    """
    definitions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name.startswith("_")
        and not node.name.startswith("__")
    }
    first_use: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
            continue
        definition = definitions.get(node.id)
        if definition is None:
            continue
        if definition.lineno <= node.lineno <= (definition.end_lineno or definition.lineno):
            continue
        first_use[node.id] = min(first_use.get(node.id, node.lineno), node.lineno)
    return sorted(
        (name, definition.lineno, first_use[name])
        for name, definition in definitions.items()
        if name in first_use and definition.lineno < first_use[name]
    )


@functools.cache
def scan_package() -> PackageScan:
    """Read every public wrapper module under ``triwarp/``, skipping ``kernels/`` and ``_*.py``."""
    scan = PackageScan()
    for path in sorted(_PACKAGE_DIR.rglob("*.py")):
        relative = path.relative_to(_PACKAGE_DIR)
        if relative.parts[0] == "kernels" or path.name == "__init__.py":
            continue
        if any(part.startswith("_") for part in relative.with_suffix("").parts):
            continue
        module = ".".join(relative.with_suffix("").parts)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as error:
            scan.errors.append(f"triwarp/{relative.as_posix()}:{error.lineno or 0}: {error.msg}")
            continue

        functions = tuple(
            PublicFunction(
                module=module,
                name=node.name,
                lineno=node.lineno,
                summary=_summary(node),
                returns=ast.unparse(node.returns) if node.returns is not None else None,
                parameters=tuple(
                    argument.arg
                    for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
                ),
            )
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
        )
        scan.modules[module] = PublicModule(
            name=module,
            path=f"triwarp/{relative.as_posix()}",
            summary=_summary(tree),
            functions=functions,
            private_imports=tuple(_private_imports(tree, module)),
            early_helpers=tuple(_early_helpers(tree)),
        )
    return scan


def library_in_summary_problems() -> list[str]:
    """Check 1: a public summary line names a reference library."""
    problems: list[str] = []
    for module in scan_package().modules.values():
        allowed = _SUMMARY_LIBRARY_ALLOWLIST.get(module.name, frozenset())
        for where, summary in (
            (f"{module.path}:1 <module>", module.summary),
            *((f"{item.site} {item.name}", item.summary) for item in module.functions),
        ):
            match = _LIBRARY_IN_SUMMARY.search(summary)
            if match is not None and match.group() not in allowed:
                problems.append(
                    f"{where}: summary names {match.group()!r} -- say what it returns and move the "
                    f"attribution to Notes or See Also: {summary!r}"
                )
    return problems


def mask_return_problems() -> list[str]:
    """Check 2: a ``*_mask`` producer whose return annotation is not a boolean array."""
    problems: list[str] = []
    for function in scan_package().functions:
        if not function.name.endswith("_mask"):
            continue
        # A function taking a mask is named for its *input* (``submesh_from_face_mask``), so the
        # suffix says nothing about what it returns.
        if any(parameter.endswith("mask") for parameter in function.parameters):
            continue
        if function.returns is not None and "wp.bool" in function.returns:
            continue
        problems.append(
            f"{function.site} {function.name}: returns {function.returns} -- every other member of "
            "the *_mask family returns wp.array[wp.bool]; rename it out of the family"
        )
    return problems


def warp_suffix_problems() -> list[str]:
    """Check 3: a module summary advertising the framework the whole package is built on."""
    return [
        f"{module.path}:1 <module>: summary carries a Warp suffix; the whole package is Warp: "
        f"{module.summary!r}"
        for module in scan_package().modules.values()
        if _WARP_SUFFIX.search(module.summary)
    ]


def coverage_location_problems() -> list[str]:
    """Check 4: a suite file that mirrors no module, or a module with no file in a suite."""
    modules = {name.replace(".", "_") for name in scan_package().modules}
    problems: list[str] = []
    for directory, extras, exempt in (
        (_TESTS_DIR, _EXTRA_TEST_FILES, _MODULES_WITHOUT_TESTS),
        (_BENCHMARKS_DIR, _EXTRA_BENCHMARK_FILES, _MODULES_WITHOUT_BENCHMARKS),
    ):
        if not directory.is_dir():
            continue
        suite = directory.name
        stems = {path.stem.removeprefix("test_") for path in directory.glob("test_*.py")}
        for stem in sorted(stems - modules - extras):
            problems.append(
                f"{suite}/test_{stem}.py: mirrors no module -- name it for the one it covers "
                "(dots become underscores, so triwarp/heat/distance.py is test_heat_distance.py)"
            )
        for module in sorted(modules - stems - {name.replace(".", "_") for name in exempt}):
            problems.append(
                f"triwarp/{module.replace('_', '/')}.py: no {suite}/test_{module}.py -- "
                "coverage is per module"
            )
    return problems


def private_import_problems() -> list[str]:
    """Check 5: a private name reached across a module boundary."""
    problems: list[str] = []
    for module in scan_package().modules.values():
        for target, lineno in module.private_imports:
            if (module.name, target) in _PRIVATE_IMPORT_ALLOWLIST:
                continue
            problems.append(
                f"{module.path}:{lineno}: reaches {target} in another module -- a helper with "
                "callers in two modules is a public function, not a private one"
            )
    return problems


def duplicate_name_problems() -> list[str]:
    """Check 6: one public name exported by two modules."""
    homes: dict[str, set[str]] = defaultdict(set)
    for function in scan_package().functions:
        homes[function.name].add(function.module)
    return [
        f"{name!r} is public in {', '.join(sorted(modules))} -- two functions with one name are "
        "one name too few; say what each returns"
        for name, modules in sorted(homes.items())
        if len(modules) > 1 and modules != _DUPLICATE_NAME_ALLOWLIST.get(name)
    ]


def kernel_module_problems() -> list[str]:
    """Check 7: a top-level kernel module and its public module must be named for each other."""
    public = {path.stem for path in _PACKAGE_DIR.glob("*.py") if path.stem != "__init__"}
    public -= {stem for stem in public if stem.startswith("_")}
    kernels = {path.stem for path in _KERNELS_DIR.glob("*.py") if path.stem != "__init__"}
    problems = [
        f"triwarp/kernels/{stem}.py: no triwarp/{stem}.py -- a top-level kernel module is named "
        "for the public module it backs"
        for stem in sorted(kernels - public - _SHARED_KERNEL_MODULES)
    ]
    problems += [
        f"triwarp/{stem}.py: no triwarp/kernels/{stem}.py -- a moved wrapper takes its kernels "
        "with it"
        for stem in sorted(public - kernels - _MODULES_WITHOUT_KERNELS)
    ]
    return problems


def helper_order_problems() -> list[str]:
    """Check 8: a private helper defined above its first caller (the stepdown rule)."""
    problems: list[str] = []
    for module in scan_package().modules.values():
        allowed = _HELPER_ORDER_ALLOWLIST.get(module.name, frozenset())
        for name, def_line, use_line in module.early_helpers:
            if name in allowed:
                continue
            problems.append(
                f"{module.path}:{def_line}: {name!r} is defined above its first caller "
                f"(line {use_line}) -- a private helper goes immediately after the public function "
                "that calls it, or after its last caller when several do"
            )
    stale = [
        f"{module.path}: {name!r} is in _HELPER_ORDER_ALLOWLIST but is no longer out of order -- "
        "drop the entry, the debt list only shrinks"
        for module in scan_package().modules.values()
        for name in sorted(_HELPER_ORDER_ALLOWLIST.get(module.name, frozenset()))
        if name not in {helper[0] for helper in module.early_helpers}
    ]
    return problems + stale
