"""
Static scan of the public API's shape: names, summaries, file layout and module boundaries.

Nineteen conventions the package holds to, each one a defect class that was actually found rather
than an aesthetic preference. They are checked by an ``ast`` scan of ``triwarp/`` (excluding
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
   either suite corresponds to a module -- ``tests/test_holes.py`` for ``triwarp/holes.py``. The
   package is flat, so the mapping is the module name and nothing else; the dotted-path transform
   this used to carry existed for ``triwarp/heat/``, the one subpackage, and went with it.
5. **A private name stays inside its module.** A ``_helper`` imported across a module boundary is a
   function that should have been public, and the alias-on-import (``import _x as x``) is the tell.
6. **Two modules do not export the same public name**, outside a written allowlist.
7. **A top-level kernel module is named for the public module it backs**, and vice versa
   (``.claude/CLAUDE.md`` section 4). This is what stops a wrapper module from being created while
   its kernels are left behind under the old name.
8. **A private helper is defined below its first caller** (``.claude/CLAUDE.md`` section 11's
   stepdown rule), so a reader never jumps backward to a definition they have not met. The 49 sites
   that predated the check were a staleness-checked debt list, now drained; the single remaining
   entry is a permanent exemption, a helper called at module scope to build a constant.
9. **A Warp-version claim names a version at least as new as the installed ``warp-lang``.** This is
   the one check that reads outside ``triwarp/``, and it exists because an upgrade left twelve
   workarounds citing Warp 1.13-1.15 for a year. The local API mirrors carry a version stamp, so a
   stale *mirror* is catchable; nothing caught a stale *justification*. Unlike checks 1-8 this one
   also
   scans ``kernels/``, where five of those twelve lived -- and ``tests/`` and ``benchmarks/``, which
   is where the rot ran deepest. It scanned neither until eleven ``warp.optim.linear.cg`` skips had
   survived the 1.16 fix that made CPU ``cg`` converge, two of them naming versions 1.14-1.15 in
   the anchored spelling this check is built to read. A skip is worse than a stale comment: the
   comment
   misinforms, the skip silently deletes coverage, and on a box with CUDA the deleted branch is the
   one nobody runs. The check abstains on ``.md`` -- it reads Python prose blocks -- so a claim in
   ``benchmarks/README.md`` still needs a human.
10. **A Python-scope allocation names the device it allocates on.** ``wp.zeros`` / ``empty`` /
    ``ones`` / ``full`` / ``array`` without ``device=`` land on Warp's *current* device, not on the
    device of the arrays they are about to be used with. The suite never catches it, because a test
    runs with its arrays' device as the current device and the omitted argument then resolves
    correctly by accident; it surfaces only under ``wp.ScopedDevice``. Found twice now, in two
    different call families, so it is mechanical from here.
11. **A public function that raises documents a ``Raises`` block.** Only a *direct* ``raise`` in the
    function's own body counts: 42 public functions delegate their validation to a helper that
    raises, which is correct and is not scanned. The tell that this was drift rather than a policy
    was that in five of the eleven sites the very next function in the same file documented its own
    raise, and in one of them a *private* helper did while its public caller did not.
12. **A fenced ``python`` docstring example runs.** Two of the package's four examples raised when
    executed -- both by calling a Warp array where the code had written a NumPy expression -- and
    neither ``ast.parse`` nor any reviewer had noticed, because an example is documentation nobody
    executes. This one is the odd member of the family: the scan only *extracts* the blocks, and
    [`tests/test_api_conventions.py`](test_api_conventions.py) runs them against a mesh fixture.
13. **A kernel output argument is named ``out_*`` and sits at the end of the signature**
    (``.claude/CLAUDE.md`` section 3). Like check 9 this one scans ``kernels/``, which the others
    exclude: the first full sweep of the kernel tree found eight genuine outputs wearing plain
    names (four of them literally ``out``, the prefix without the name), three read-only inputs
    wearing the prefix, and two argument classes the convention had no spelling for -- in-place
    arguments and scratch / persistent-state buffers, now exempted in section 3 and carried here
    as ``_KERNEL_OUTPUT_ALLOWLIST``.
14. **An array annotation is subscript-style** -- ``wp.array[T]``, not the pre-1.12
    ``wp.array(dtype=T)`` (``.claude/CLAUDE.md`` section 2). Both forms work, so the old one simply
    accumulated: 176 annotations against 1 644, all of them in the three newest large kernel
    modules, and one file carrying both. Restricted to *annotation* positions, which is what lets
    it scan the whole package -- ``wp.array(dtype=T)`` is a legal allocation expression at Python
    scope and only an annotation makes it the stale spelling.
15. **A ``wp.launch`` / ``wp.launch_tiled`` names the device it launches on.** The memory-safety
    guard of the family: an omitted ``device=`` resolves to Warp's *current* device, and with CPU
    arrays that runs the kernel on ``cuda:0`` over host pointers, returns the right answer, and
    corrupts the heap when those arrays are freed mid-kernel. This is the half of the guard that
    carries the load -- ``conftest.py``'s ``STRICT`` mode only bites when the arrays are *not* on
    the launch device, so on a CUDA run the omission is invisible to it.
16. **A cast inside a kernel is spelled ``wp.int32`` / ``wp.float32``, never bare ``int`` /
    ``float``** (``.claude/CLAUDE.md`` section 3). They are the same builtins under a different
    name, with one asymmetry that matters: ``float(...)`` is a *hard compile error* inside a
    ``wp.Float``-generic function, so it silently forecloses genericising that function -- which
    runs against section 14's "prefer dtype-generic ``@wp.func``s". The tree carried 329
    ``int(wp.tid())`` alongside 640 ``wp.int32(...)``, and 46 sites in 41 kernels used *both* on
    the same local. Like checks 9, 13 and 14 this one scans ``kernels/``.
17. **An integer division inside a kernel is spelled ``//``, never ``/``** (``.claude/CLAUDE.md``
    section 5). On integers the two are the *same* operation in Warp -- both truncate toward zero,
    where CPython's ``//`` floors -- so this is legibility: ``/`` on two ``int32``s reads as real
    division and truncates only because the operands happen to be integers. A check rather than an
    edit because the defect recurred after the rule was written, and because the scan found four
    sites a textual pass had missed. It types an operand only *by declaration*, which is what makes
    it safe to run on float-heavy code.
18. **A kernel-scope argument or return is annotated in Warp's types** -- ``wp.bool`` /
    ``wp.int32`` / ``wp.float32``, not the bare Python names (``.claude/CLAUDE.md`` section 2).
    The third member of the 16/17 family, and the same story: Warp resolves both spellings to the
    same types, so only a scan keeps them from coexisting. The tree carried 11 ``-> bool`` against
    46 ``-> wp.bool`` plus 12 bare parameters, the newest written the day after the pass that
    converted the last batch of casts. It reads ``@wp.kernel`` / ``@wp.func`` signatures only,
    because a kernel *factory* is ordinary Python whose ``int`` parameters are correct.
19. **A test comparing against a reference library says which class the comparison is**
    (``.claude/CLAUDE.md`` section 6). The second check that reads ``tests/`` rather than
    ``triwarp/``, and it is here because the convention has decayed twice: the lowercase ``class b``
    spelling went 21 -> 0 -> 9, invisible to the grep section 6 prescribes because a human reads it
    the same. It accepts all four label phrases the suite uses, keys on ``ast.Assert`` so a fixture
    unpack is not a hit, and leaves ``_np`` out because it marks inputs as often as oracles. It
    checks that a label is *present*, never that it is the right one.

Why a static scan rather than importing ``triwarp``
---------------------------------------------------
Importing would make the verdict depend on Warp's module cache and on which optional dependencies
resolve, and would say nothing about files (checks 4 and 7) at all. A scan reads the tree as
written, so its answer is the same in every environment and in every pytest invocation. Check 9
needs the installed Warp version and takes it from ``importlib.metadata`` rather than
``warp.config.version``, so even that one imports nothing. Check 12 is the single exception and it
is deliberate: an example's defect is a *runtime* one, so nothing short of running it finds it, and
the extraction half stays here so the execution half has no parsing to do.
"""

from __future__ import annotations

import ast
import functools
import re
from collections import defaultdict
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as installed_version
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
        "io",  # meshio round-trips, i.e. a benchmark of meshio
    }
)

# --- check 5 -----------------------------------------------------------------------------------

# Cross-module private imports this package accepts, as (importing module, ``owner._name``) pairs
# with the reason each one is not the defect the check is looking for. Anything else fires.
#
# The defect the check exists for is a private *operation* with callers in two modules -- a function
# that was written where it was first needed and should have been public
# (``proximity._default_mesh_query_max_dist``, four importers, one of them aliasing it back to a
# public-looking name). What survives below is the three things that are not that: a sorted copy
# whose public form does extra work, a string-to-kernel-flag table shared so two modules cannot
# disagree about one enum's spellings, and one pass a caller must skip.
#
# Six further entries used to sit here, all ``combine`` -> ``holes``, and none of them was ever an
# exemption worth keeping: they existed because the minimum-weight interval DP that fills one hole
# is the same DP that stitches two rims, and the two halves lived in different modules. Moving the
# stitch family into ``holes`` deleted all six at once. A cluster of entries sharing one (importer,
# owner) pair is that shape of defect -- read it as a misplaced family before writing the seventh.
_PRIVATE_IMPORT_ALLOWLIST: dict[tuple[str, str], str] = {
    ("reduce", "array._sorted_copy"): (
        "reduce.median needs a sorted copy, and array.sort_and_argsort is the public form -- which "
        "also builds the permutation median throws away"
    ),
    ("remesh", "triangles._QUALITY_METRICS"): (
        "a string-to-kernel-flag table, for face_quality's metric argument, so remesh's objective "
        "names cannot drift from triangles.face_quality's"
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
_MODULES_WITHOUT_KERNELS = frozenset({"constants", "io", "mesh", "typing"})

# --- check 8 ------------------------------------------------------------------------------------

# Private helpers that sit above their first caller today. CLAUDE.md section 11 says a helper must
# never appear above the caller it serves, and the package had drifted off that rule wholesale
# before the check existed -- 49 sites across 15 modules, carried here as an explicit debt list
# rather than a silent exemption. That list has now been drained: one entry remains, and it is not
# debt but a permanent structural exemption. **Entries come out, they do not go in** -- a new helper
# must be placed correctly, which is what this check enforces.
#
# Two things the sweep learned, both of which the check itself caught:
#
# - Reordering has to consider *every* private helper in a module, not only the flagged ones.
#   Moving a flagged helper below a callee it uses strands that callee: ``holes._loop_perimeters``
#   was compliant before, because its caller sat above it, and became a fresh violation.
# - A name can appear several times at module scope, since ``@overload`` stubs precede their
#   implementation. Verifying a "pure move" by comparing functions keyed on name silently collapses
#   those duplicates, so the check on the *reorder* has to compare the multiset.
# --- check 9 ------------------------------------------------------------------------------------

# A Warp version claim, and the spelling is the convention: the word **Warp** immediately before
# the number. Anything looser is unusable here -- this package writes measured ratios in the same
# shape ("within 1.25x of best", "1.06 ms", "1.13x on CUDA at both sizes"), and a bare ``1.N`` token
# matched 30 of them against 3 real claims when this check was first written. So a version claim
# says "Warp 1.16", never "through 1.16" with the word three lines up.
_WARP_VERSION_CLAIM = re.compile(r"\bWarp\s+1\.(\d+)(?:\.(\d+))?\b")

# Version claims that deliberately record history rather than describe the installed Warp. Keyed by
# ``(module, "1.x")``; the value is the reason, and it is where the re-verification goes -- so the
# next upgrade reads a list of claims to re-run instead of a grep to invent. Modules under
# ``triwarp/`` are keyed by their dotted name, everything else by its path
# (``tests.api_conventions``, ``benchmarks.test_creation``).
_WARP_VERSION_ALLOWLIST: dict[tuple[str, str], str] = {
    ("graph", "1.15"): (
        "deliberate history: names the version the CPU heap corruption was measured on so the "
        "1.16 fix beside it has something to be a fix *of*"
    ),
    ("tests.api_conventions", "1.13"): (
        "deliberate history: this check's own rationale, naming the range of versions the twelve "
        "stale workarounds cited -- the thing it was written to stop"
    ),
    ("benchmarks.test_creation", "1.15"): (
        "deliberate history: a measurement stamp on a recorded benchmark table, naming the Warp "
        "the numbers below it were taken on. Re-stamping without re-measuring would be a lie"
    ),
}

# Directories check 9 scans, and the prefix each one's allowlist key carries. ``triwarp/`` keeps the
# bare dotted module name it has always used; the two suites are prefixed so a key stays unambiguous
# when a test file and a package module share a stem.
_WARP_VERSION_SCAN_ROOTS: tuple[tuple[Path, str], ...] = (
    (_PACKAGE_DIR, ""),
    (_TESTS_DIR, "tests."),
    (_BENCHMARKS_DIR, "benchmarks."),
)

# --- check 10 -----------------------------------------------------------------------------------

# The Python-scope allocators that take a ``device`` keyword. ``wp.clone`` and the ``*_like``
# family inherit the source array's device and so cannot get this wrong.
_ALLOCATORS = frozenset({"array", "empty", "full", "ones", "zeros"})

# Allocations that deliberately fall back to Warp's current device, keyed by ``(module, call)``
# with the call spelled exactly as ``ast.unparse`` renders it. The value is the reason, and the
# docstring of the function it sits in has to say the same thing -- an undocumented fallback is
# the defect, not the fallback itself.
_ALLOCATION_DEVICE_ALLOWLIST: dict[tuple[str, str], str] = {
    ("combine", "wp.empty(0, dtype=wp.vec3)"): (
        "concatenate([]) has no input to take a device from, so the current device is the only "
        "answer available; its Returns block says so"
    ),
    ("combine", "wp.empty(0, dtype=wp.int32)"): (
        "the face half of the same empty return, for the same reason"
    ),
}

_HELPER_ORDER_ALLOWLIST: dict[str, frozenset[str]] = {
    # Not debt: ``_icosphere_face_table`` is called at *module* scope (line 179) to build the
    # ``_ICOSPHERE_FACE_TABLE`` constant, so it has no function caller to sit below and moving it
    # under its use is a NameError at import. This is the one permanent entry; the other 48 sites
    # this list carried were reordered in the section 4.4 sweep.
    "creation": frozenset({"_icosphere_face_table"})
}

# --- check 15 -----------------------------------------------------------------------------------

# The Python-scope launchers that take a ``device`` keyword.
_LAUNCHERS = frozenset({"launch", "launch_tiled"})

# Launches that deliberately fall back to Warp's current device, keyed by ``(module, kernel)``.
# Empty on purpose: every one of the package's launches forwards a device, and a new entry needs a
# reason a reader can check, because the failure mode is silent memory corruption on CPU runs.
_LAUNCH_DEVICE_ALLOWLIST: dict[tuple[str, str], str] = {}

# --- check 13 -----------------------------------------------------------------------------------

# Kernel-scope calls whose first argument is a buffer the kernel writes.
_KERNEL_WRITE_CALLS = frozenset(
    {
        "atomic_add",
        "atomic_sub",
        "atomic_min",
        "atomic_max",
        "atomic_cas",
        "atomic_exch",
        "tile_store",
        "tile_atomic_add",
    }
)

# Kernel arguments that are written without the ``out_`` prefix, or that legitimately follow an
# ``out_`` argument, keyed by ``(kernel module, kernel)``. Two exemption classes, both written into
# ``.claude/CLAUDE.md`` section 3:
#
# - **In-place**: the argument is both the input and the result -- an ``out_`` prefix would misread
#   as write-only. ``sort_rows_insertion(data)``, ``orient_ccw(points2d)``,
#   ``offset_packed_faces(faces)``, the hole-filling DP tables (read at smaller spans, written at
#   the current one) and ``accumulate_cost(acc)``, which reads the packed accumulator's weight-sum
#   slot while atomically adding into its cost slot.
# - **Scratch / persistent state**: caller-allocated working memory carried across launches --
#   cursors, stacks, open-addressing tables, the ear-clipping ring, ``ball_pivoting``'s
#   persistent front. Neither an input nor the answer, so the name says what the buffer holds
#   (``cursor``, ``front_out``, ``new_src``) rather than wearing a prefix that promises a result.
_KERNEL_OUTPUT_ALLOWLIST: dict[tuple[str, str], frozenset[str]] = {
    # in-place
    ("array", "sort_rows_insertion"): frozenset({"data"}),
    # ``neighbors`` is sorted in place: this kernel only orders the two slots each vertex already
    # holds, so it is both the input and the result and ``out_`` would read as write-only.
    ("boundary", "sort_boundary_neighbor_slots"): frozenset({"neighbors"}),
    ("combine", "offset_packed_faces"): frozenset({"faces"}),
    ("holes", "fill_dp_span"): frozenset({"dp", "prev"}),
    ("holes", "fill_dp_span_tiled"): frozenset({"dp", "prev"}),
    ("polyline", "orient_ccw"): frozenset({"points2d"}),
    # ``quadric_decimate``'s provenance column, folded one pass at a time: the array is the previous
    # pass's answer *and* this pass's, so it is in place and ``out_`` would read as write-only.
    ("remesh", "compose_vertex_index"): frozenset({"index"}),
    ("registration", "accumulate_cost"): frozenset({"acc"}),
    # scratch / persistent state
    ("adjacency", "scatter_vertex_faces"): frozenset({"cursor"}),
    # ``min_distance_sq`` is the farthest-point sampler's running distance-to-the-chosen-set:
    # caller-allocated scratch the persistent block initializes and folds down one selection per
    # round. The answer is ``out_selected``; this buffer is the sweep's state.
    ("points", "farthest_point_sample_block"): frozenset({"min_distance_sq"}),
    # ``slot_count`` is the per-vertex atomic cursor picking which of the two neighbour slots each
    # scattered edge lands in -- the ``scatter_vertex_faces`` case exactly, under a name that says
    # what it holds. The answer is ``out_neighbors``.
    ("boundary", "scatter_boundary_neighbors"): frozenset({"slot_count"}),
    # The hull sweep's working set: the boundary polygon it carries between insertions, the buffer
    # it rebuilds that polygon into, and the per-boundary-edge orientations of one insertion.
    # Caller-allocated because the sweep is one thread over an ``n``-sized problem, so none of the
    # three can be a kernel local -- but none is an input or the answer either, so ``out_`` would
    # misread.
    ("reconstruction", "lexicographic_triangulation"): frozenset(
        {"boundary", "boundary_next", "orientations"}
    ),
    ("algorithms.ball_pivoting", "begin_wave"): frozenset({"counters"}),
    # ``edges`` is absent deliberately: this kernel mutates the table only through
    # ``register_face_edge``, and check 13 does not follow writes into a called ``@wp.func``.
    ("algorithms.ball_pivoting", "commit_triangles"): frozenset({"counters", "point_used"}),
    ("algorithms.ball_pivoting", "end_wave"): frozenset({"counters"}),
    # ``edges`` is the open-addressing edge table (``BpaEdgeTable``): persistent state carried
    # across every wave, mutated in place by the pivot and commit kernels and read by both. It is
    # the ``front_out`` case one level up -- neither an input nor the answer -- so it keeps the name
    # of what it holds. Check 13 sees writes through its fields since ``_subscript_base`` resolves
    # ``edges.count[slot] = 1`` to ``edges``.
    ("algorithms.ball_pivoting", "pivot_front_edges"): frozenset(
        {"counters", "edges", "front_out"}
    ),
    ("algorithms.ball_pivoting", "rehash_edges"): frozenset(
        {"new_cand", "new_count", "new_opp", "new_src", "new_state", "new_tgt"}
    ),
    ("energies", "scatter_edge_halfedges"): frozenset({"cursor"}),
    ("selection", "open_dual_edges_and_seeds"): frozenset({"cursor"}),
    ("repair", "emit_degree3_replacement"): frozenset({"cursor"}),
    ("repair", "emit_straighten_faces"): frozenset({"cursor"}),
    # A running minimum other threads publish and read: neither an input nor the answer, and the
    # per-face results are the ``out_`` arguments beside it. ``counter`` and ``overflow`` are the
    # work list the capped first pass hands the tiled second one -- scratch, not the answer.
    ("proximity", "face_to_mesh_distance"): frozenset({"global_best_sq", "counter", "overflow"}),
    ("proximity", "face_to_mesh_distance_tiled"): frozenset({"global_best_sq"}),
    ("grouping", "hash_insert"): frozenset({"slot_counts"}),
    # ``values`` is the matrix whose rows this scales -- input and result in the same buffer, since
    # the prolongation smoother's ``-w D^-1 (A P0)`` is a row scaling of a product that has just
    # been built and is not needed unscaled.
    ("algorithms.multigrid", "scale_rows"): frozenset({"values"}),
    ("polyline", "clip_selected"): frozenset({"active", "left", "right"}),
    ("polyline", "init_ring"): frozenset({"active", "left", "right"}),
    # The level-synchronous Ramer-Douglas-Peucker round. ``span_lo`` / ``span_hi`` are each point's
    # current span, rewritten in place to the child span it belongs to at the next level;
    # ``state`` is the ``wp.capture_while`` loop's own [levels run, condition] pair, which is the
    # ``ear_loop_continue`` case in the same module. Neither is an input and neither is the answer
    # -- that is ``out_keep``.
    ("polyline", "rdp_split_spans"): frozenset({"span_lo", "span_hi", "state"}),
    ("sample", "subtract_deleted_contributions"): frozenset({"weights"}),
    ("visibility", "shape_diameter"): frozenset({"scratch"}),
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
    raises: tuple[tuple[int, str], ...]
    """``(lineno, exception)`` for each ``raise`` in this function's own body, nested functions
    excluded."""

    documents_raises: bool
    """Whether the docstring carries a numpydoc ``Raises`` section header."""

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


def _direct_raises(node: ast.FunctionDef) -> list[tuple[int, str]]:
    """
    Every ``raise`` in this function's own body, with the exception it names.

    A nested ``def``, ``lambda`` or ``class`` is not descended into -- its raises belong to *it*,
    and a closure's failure mode is its caller's docstring only by coincidence. A bare ``raise``
    re-raises something already in flight and names nothing, so it is skipped.
    """
    found: list[tuple[int, str]] = []
    stack: list[ast.AST] = list(ast.iter_child_nodes(node))
    while stack:
        current = stack.pop()
        if isinstance(current, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda):
            continue
        if isinstance(current, ast.Raise) and current.exc is not None:
            exception = current.exc.func if isinstance(current.exc, ast.Call) else current.exc
            found.append((current.lineno, ast.unparse(exception)))
        stack.extend(ast.iter_child_nodes(current))
    return sorted(found)


_RAISES_SECTION = re.compile(r"^[ \t]*Raises[ \t]*\n[ \t]*-{5,}[ \t]*$", re.MULTILINE)


def _documents_raises(node: ast.FunctionDef) -> bool:
    """Whether the docstring has a numpydoc ``Raises`` header, underline and all."""
    return _RAISES_SECTION.search(ast.get_docstring(node, clean=False) or "") is not None


def _bare_allocations(tree: ast.Module) -> list[tuple[str, int]]:
    """
    Python-scope ``wp.<allocator>(...)`` calls that name no ``device``.

    A ``**kwargs`` splat could be carrying one, so a call with one is not reported -- the check
    would rather miss a forwarded device than fire on a call it cannot read.
    """
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in _ALLOCATORS or _dotted(node.func)[:1] != ("wp",):
            continue
        if any(keyword.arg in ("device", None) for keyword in node.keywords):
            continue
        found.append((ast.unparse(node), node.lineno))
    return sorted(found, key=lambda item: item[1])


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
                raises=tuple(_direct_raises(node)),
                documents_raises=_documents_raises(node),
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
    modules = set(scan_package().modules)
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
                f"{suite}/test_{stem}.py: mirrors no module -- name it for the one it covers"
            )
        for module in sorted(modules - stems - exempt):
            problems.append(
                f"triwarp/{module}.py: no {suite}/test_{module}.py -- coverage is per module"
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


def installed_warp_version() -> str | None:
    """Report the installed ``warp-lang`` version, or ``None`` when the distribution is absent."""
    try:
        return installed_version("warp-lang")
    except PackageNotFoundError:
        return None


def _prose_blocks(source: str) -> list[tuple[int, str]]:
    """
    ``(lineno, text)`` for each run of consecutive comment lines and each string literal.

    Newlines are preserved inside a block so a caller can recover the exact line from a match
    offset. String literals cover every docstring without walking the tree, and a version claim
    inside a non-docstring string is still a claim.
    """
    blocks: list[tuple[int, str]] = []
    run_start, run_lines = 0, []
    for lineno, line in enumerate(source.splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            if not run_lines:
                run_start = lineno
            run_lines.append(stripped.lstrip("#"))
            continue
        if run_lines:
            blocks.append((run_start, "\n".join(run_lines)))
            run_lines = []
    if run_lines:
        blocks.append((run_start, "\n".join(run_lines)))

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return blocks
    blocks.extend(
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )
    return blocks


def warp_version_problems() -> list[str]:
    """Check 9: a Warp-version claim naming a version older than the installed ``warp-lang``."""
    installed = installed_warp_version()
    if installed is None:  # nothing to compare against; the check abstains rather than guesses
        return []
    installed_minor = int(installed.split(".")[1])

    problems: list[str] = []
    seen: set[tuple[str, str]] = set()
    for root, prefix in _WARP_VERSION_SCAN_ROOTS:
        if not root.is_dir():  # neither suite ships in the wheel
            continue
        for path in sorted(root.rglob("*.py")):
            module = path.relative_to(root).with_suffix("").as_posix().replace("/", ".")
            module = prefix + module.removesuffix(".__init__").lstrip(".")
            for lineno, text in _prose_blocks(path.read_text()):
                for match in _WARP_VERSION_CLAIM.finditer(text):
                    line = lineno + text.count("\n", 0, match.start())
                    minor = int(match.group(1))
                    if minor >= installed_minor:
                        continue
                    key = (module, f"1.{minor}")
                    if key in _WARP_VERSION_ALLOWLIST:
                        seen.add(key)
                        continue
                    problems.append(
                        f"{path.relative_to(_REPO_ROOT)}:{line}: names {match.group(0)}, "
                        f"older than the installed warp-lang {installed} -- re-probe the claim "
                        "against the installed version and re-stamp it, or add a "
                        "_WARP_VERSION_ALLOWLIST entry recording why it deliberately names history"
                    )
    problems.extend(
        f"_WARP_VERSION_ALLOWLIST entry {key!r} matches nothing in the scanned tree -- drop it"
        for key in sorted(_WARP_VERSION_ALLOWLIST)
        if key not in seen and int(key[1].split(".")[1]) < installed_minor
    )
    return problems


def allocation_device_problems() -> list[str]:
    """
    Check 10: a Python-scope allocation that does not name the device it allocates on.

    Walks the whole package rather than ``scan_package``'s public subset -- private ``_*.py``
    modules allocate too, and a buffer landing on the wrong device is not a question about the
    API's shape. ``kernels/`` is excluded because a kernel allocates nothing at Python scope.
    """
    problems: list[str] = []
    seen: set[tuple[str, str]] = set()
    for path in sorted(_PACKAGE_DIR.rglob("*.py")):
        relative = path.relative_to(_PACKAGE_DIR)
        if relative.parts[0] == "kernels":
            continue
        module = ".".join(relative.with_suffix("").parts).removesuffix(".__init__")
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue  # test_package_scan_is_discoverable reports the parse failure
        for call, lineno in _bare_allocations(tree):
            key = (module, call)
            if key in _ALLOCATION_DEVICE_ALLOWLIST:
                seen.add(key)
                continue
            problems.append(
                f"triwarp/{relative.as_posix()}:{lineno}: {call} has no device= -- it lands on "
                "Warp's *current* device, which is the input's only by accident under the test "
                "suite; forward the device of the arrays it will be used with"
            )
    problems.extend(
        f"_ALLOCATION_DEVICE_ALLOWLIST entry {key!r} matches nothing in triwarp/ -- drop it"
        for key in sorted(_ALLOCATION_DEVICE_ALLOWLIST)
        if key not in seen
    )
    return problems


def undocumented_raise_problems() -> list[str]:
    """
    Check 11: a public function with a direct ``raise`` and no ``Raises`` block.

    One direction only. The reverse -- a ``Raises`` block with no direct raise -- is the *correct*
    shape for the 44 functions that delegate validation to a shared guard, and scanning it would
    need an allowlist longer than the check.
    """
    return [
        f"{function.site} {function.name}: raises "
        f"{', '.join(sorted({exception for _, exception in function.raises}))} at line(s) "
        f"{', '.join(str(lineno) for lineno, _ in function.raises)} but documents no Raises block"
        for function in scan_package().functions
        if function.raises and not function.documents_raises
    ]


# --- check 12 -----------------------------------------------------------------------------------

_PYTHON_FENCE = re.compile(r"^(?P<indent>[ \t]*)```python[ \t]*$", re.MULTILINE)


@dataclass(frozen=True)
class DocstringExample:
    """One fenced ``python`` block lifted out of a docstring, dedented and ready to ``exec``."""

    site: str
    code: str

    @property
    def is_sketch(self) -> bool:
        """
        Whether the block is a deliberate outline: a bare ``...`` standing in for real code.

        Read as an ``Ellipsis`` *statement* rather than by matching the text, so a trailing comment
        (``...  # rewrite rhs in place``) still counts and a genuine ``...`` inside an expression
        does not.
        """
        try:
            tree = ast.parse(self.code)
        except SyntaxError:
            return False
        return any(
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and node.value.value is Ellipsis
            for node in ast.walk(tree)
        )


def docstring_examples() -> list[DocstringExample]:
    """
    Every fenced ``python`` block in ``triwarp/``, dedented to column zero.

    Read off the raw source rather than off docstring nodes: a block's *line number* is what makes
    a failure reportable, and ``ast`` gives the docstring's line, not the fence's. ``kernels/`` is
    excluded -- it has no fenced blocks, and kernel-scope code could not be executed as written.
    """
    examples: list[DocstringExample] = []
    for path in sorted(_PACKAGE_DIR.rglob("*.py")):
        relative = path.relative_to(_PACKAGE_DIR)
        if relative.parts[0] == "kernels":
            continue
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        for match in _PYTHON_FENCE.finditer(source):
            indent = match.group("indent")
            fence = source.count("\n", 0, match.start())  # 0-based index of the ```python line
            body: list[str] = []
            for line in lines[fence + 1 :]:
                if line.strip() == "```":
                    break
                body.append(line.removeprefix(indent))
            examples.append(
                DocstringExample(
                    site=f"triwarp/{relative.as_posix()}:{fence + 1}", code="\n".join(body) + "\n"
                )
            )
    return examples


# --- check 13 -----------------------------------------------------------------------------------


def _is_kernel(node: ast.FunctionDef) -> bool:
    """Whether ``node`` is decorated ``@wp.kernel`` or ``@wp.kernel(...)``."""
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if (
            isinstance(target, ast.Attribute)
            and target.attr == "kernel"
            and isinstance(target.value, ast.Name)
            and target.value.id == "wp"
        ):
            return True
    return False


def _subscript_base(node: ast.expr) -> str | None:
    """
    Resolve a (possibly nested) subscript target to its bare name, or ``None``.

    A ``@wp.struct`` field counts as its struct: ``edges.count[slot] = 1`` resolves to ``edges``.
    Without that, bundling a kernel's buffers into a struct would make every write through them
    invisible to check 13 -- which is not hypothetical, it is what happened the first time
    ``ball_pivoting``'s edge table was bundled.
    """
    while isinstance(node, (ast.Subscript, ast.Attribute)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _written_parameters(node: ast.FunctionDef) -> set[str]:
    """
    Parameters the kernel body writes *directly*.

    A subscript store, or the first argument of a ``wp.atomic_*`` / ``wp.tile_store`` /
    ``wp.tile_atomic_add`` call, in either case through a ``@wp.struct`` field as well as directly.
    A write that happens inside a called ``@wp.func`` (e.g. through a ``wp.ref`` parameter) is
    invisible here, so this check can under-report but never false-positive.
    """
    parameters = {arg.arg for arg in node.args.args}
    written: set[str] = set()
    for sub in ast.walk(node):
        targets: list[ast.expr] = []
        if isinstance(sub, ast.Assign):
            targets = list(sub.targets)
        elif isinstance(sub, ast.AugAssign):
            targets = [sub.target]
        for target in targets:
            if isinstance(target, ast.Subscript):
                base = _subscript_base(target)
                if base in parameters:
                    written.add(base)
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr in _KERNEL_WRITE_CALLS
            and isinstance(sub.func.value, ast.Name)
            and sub.func.value.id == "wp"
            and sub.args
            and _subscript_base(sub.args[0]) in parameters
        ):
            base = _subscript_base(sub.args[0])
            assert base is not None
            written.add(base)
    return written


def kernel_output_naming_problems() -> list[str]:
    """
    Check 13: a kernel output argument without the ``out_`` prefix, or not at the signature's end.

    ``.claude/CLAUDE.md`` section 3's naming rule for ``triwarp/kernels/``, with its two written
    exemptions (in-place arguments and scratch / persistent-state buffers) carried by
    ``_KERNEL_OUTPUT_ALLOWLIST``. Both directions are checked: a *written* argument must wear the
    prefix, and nothing without the prefix may follow the first argument that wears it.
    """
    problems: list[str] = []
    seen: set[tuple[tuple[str, str], str]] = set()
    for path in sorted(_KERNELS_DIR.rglob("*.py")):
        module = ".".join(path.relative_to(_KERNELS_DIR).with_suffix("").parts)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue  # test_package_scan_is_discoverable reports the parse failure
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or not _is_kernel(node):
                continue
            allowed = _KERNEL_OUTPUT_ALLOWLIST.get((module, node.name), frozenset())
            site = f"{path.relative_to(_REPO_ROOT)}:{node.lineno}"
            for name in sorted(_written_parameters(node)):
                if name.startswith("out_"):
                    continue
                if name in allowed:
                    seen.add(((module, node.name), name))
                    continue
                problems.append(
                    f"{site} {node.name}: writes parameter '{name}', which is neither "
                    "out_-prefixed nor an allowlisted in-place / scratch argument"
                )
            past_outputs = False
            for arg in node.args.args:
                if arg.arg.startswith("out_"):
                    past_outputs = True
                elif past_outputs:
                    if arg.arg in allowed:
                        seen.add(((module, node.name), arg.arg))
                        continue
                    problems.append(
                        f"{site} {node.name}: input parameter '{arg.arg}' follows an out_ "
                        "argument -- outputs go last"
                    )
    problems.extend(
        f"_KERNEL_OUTPUT_ALLOWLIST entry {(*key, name)!r} matches nothing in kernels/ -- drop it"
        for key, names in sorted(_KERNEL_OUTPUT_ALLOWLIST.items())
        for name in sorted(names)
        if (key, name) not in seen
    )
    return problems


# --- check 14 -----------------------------------------------------------------------------------


def _annotations(tree: ast.Module) -> list[ast.expr]:
    """Every annotation expression in a module: parameters, returns, and annotated assignments."""
    found: list[ast.expr] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arguments = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            found.extend(arg.annotation for arg in arguments if arg.annotation is not None)
            if node.returns is not None:
                found.append(node.returns)
        elif isinstance(node, ast.AnnAssign):
            found.append(node.annotation)
    return found


def array_annotation_style_problems() -> list[str]:
    """
    Check 14: an array annotation spelled ``wp.array(dtype=T)`` rather than ``wp.array[T]``.

    ``.claude/CLAUDE.md`` section 2's subscript style, restricted to *annotation* positions so the
    scan can cover the whole package: at Python scope ``wp.array(dtype=T)`` is also a legal
    allocation expression, and only in an annotation is it the pre-1.12 spelling.

    Both forms work, which is why the old one survived in the three newest large kernel modules
    (``algorithms/ball_pivoting.py``, ``reconstruction.py``, ``remesh.py``) long after the
    convention settled -- 176 annotations against 1 644 in the current style, and ``remesh.py``
    mixed the two inside one file, which is the state that leaves a reader unsure which is current.
    """
    problems: list[str] = []
    for path in sorted(_PACKAGE_DIR.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue  # test_package_scan_is_discoverable reports the parse failure
        for annotation in _annotations(tree):
            for node in ast.walk(annotation):
                if not isinstance(node, ast.Call):
                    continue
                target = ".".join(_dotted(node.func))
                if target.startswith(("wp.array", "warp.array")):
                    problems.append(
                        f"{path.relative_to(_REPO_ROOT)}:{node.lineno} annotation "
                        f"'{ast.unparse(node)}' uses the pre-1.12 call style -- write it as "
                        f"{target}[...] instead"
                    )
    return problems


# --- check 15 -----------------------------------------------------------------------------------


def launch_device_problems() -> list[str]:
    """
    Check 15: a ``wp.launch`` / ``wp.launch_tiled`` that does not name the device it launches on.

    An omitted ``device=`` resolves to Warp's *current* device, which is ``cuda:0`` whenever CUDA is
    present. When the arrays are on the CPU the launch is not rejected and not wrong: Warp permits
    it by design on a system whose GPU can address host memory (``RELAXED`` and ``CHECKED`` both
    pass it), and the kernel returns the correct answer. What it does not get is ordering, so the
    host buffers are freed at scope exit while the kernel is still reading them, and the process
    aborts later inside glibc -- measured on the reduced reproducer at 20/20 aborts with a free and
    no synchronize against 0/20 with either. ``vertices.mean_vertex_normals`` shipped this way.

    This is the static half of the guard and it is the half that carries the load. The runtime half
    (``tests/conftest.py`` sets ``LaunchArrayAccessMode.STRICT``) only bites when the arrays are not
    on the launch device, so on a CUDA run -- what CI and this box do -- the default device *is* the
    arrays' device and the omission stays invisible. Only the scan sees it there.
    """
    problems: list[str] = []
    seen: set[tuple[str, str]] = set()
    for path in sorted(_PACKAGE_DIR.rglob("*.py")):
        relative = path.relative_to(_PACKAGE_DIR)
        module = ".".join(relative.with_suffix("").parts).removesuffix(".__init__")
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue  # test_package_scan_is_discoverable reports the parse failure
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in _LAUNCHERS or _dotted(node.func)[:1] != ("wp",):
                continue
            if any(keyword.arg in ("device", None) for keyword in node.keywords):
                continue  # a **kwargs splat may be forwarding one; miss it rather than misfire
            kernel = ast.unparse(node.args[0]) if node.args else node.func.attr
            key = (module, kernel)
            if key in _LAUNCH_DEVICE_ALLOWLIST:
                seen.add(key)
                continue
            problems.append(
                f"triwarp/{relative.as_posix()}:{node.lineno}: wp.{node.func.attr}({kernel}, ...) "
                "has no device= -- it launches on Warp's *current* device, so with CPU arrays it "
                "runs the kernel on cuda:0 over host pointers, returns the right answer, and "
                "corrupts the host heap when those arrays are freed mid-kernel; forward the device "
                "of the input arrays"
            )
    problems.extend(
        f"_LAUNCH_DEVICE_ALLOWLIST entry {key!r} matches nothing in triwarp/ -- drop it"
        for key in sorted(_LAUNCH_DEVICE_ALLOWLIST)
        if key not in seen
    )
    return problems


# --- check 16 -----------------------------------------------------------------------------------

_BUILTIN_CASTS = frozenset({"int", "float"})


def _kernel_scope_functions(tree: ast.Module) -> list[ast.FunctionDef]:
    """Every ``@wp.kernel`` / ``@wp.func`` in a module, whose body is Warp's DSL and not Python."""
    found: list[ast.FunctionDef] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        decorators = " ".join(ast.unparse(decorator) for decorator in node.decorator_list)
        if "wp.kernel" in decorators or "wp.func" in decorators:
            found.append(node)
    return found


def builtin_cast_problems() -> list[str]:
    """
    Check 16: a bare ``int(...)`` / ``float(...)`` inside a ``@wp.kernel`` or ``@wp.func`` body.

    ``.claude/CLAUDE.md`` section 3: ``wp.int32`` / ``wp.float32`` is the tree's only cast spelling.
    ``int`` and ``float`` are the same Warp builtins under a different name -- ``int(x)`` compiles
    only because Warp writes an unconditional ``#define int(x) cast_int(x)`` into every generated
    module header, ``wp::int(x)`` not being valid C++ -- and the generated code is identical.

    The asymmetry that makes this a rule rather than a preference is ``float``: inside a
    ``wp.Float``-generic function ``total / float(count)`` does not narrow silently, it *fails to
    parse* (``Input types must be the same, got ['float64', 'float32']``). So every bare ``float()``
    is an unannounced decision that its function will never be generic, against section 14's
    "prefer dtype-generic ``@wp.func``s". Where the function is or could be generic the spelling is
    ``type(x)(...)``, the ``kernels/predicates.py`` convention.

    Scoped to kernel-scope bodies because at Python scope ``int(...)`` / ``float(...)`` are the
    ordinary builtins and entirely correct -- ``wp.constant(wp.float32(float("nan")))`` at module
    scope in ``kernels/texture.py`` is a Python call and is not a violation.
    """
    problems: list[str] = []
    for path in sorted(_PACKAGE_DIR.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue  # test_package_scan_is_discoverable reports the parse failure
        for function in _kernel_scope_functions(tree):
            for node in ast.walk(function):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                    continue
                if node.func.id not in _BUILTIN_CASTS:
                    continue
                replacement = "wp.int32" if node.func.id == "int" else "wp.float32"
                problems.append(
                    f"{path.relative_to(_REPO_ROOT)}:{node.lineno} {function.name} casts with "
                    f"'{ast.unparse(node)}' -- write {replacement}(...) instead, or type(x)(...) "
                    "if the enclosing function is or could be dtype-generic"
                )
    return problems


# --- check 17 -----------------------------------------------------------------------------------

_INT_SCALARS = frozenset(
    {
        "wp.int8",
        "wp.int16",
        "wp.int32",
        "wp.int64",
        "wp.uint8",
        "wp.uint16",
        "wp.uint32",
        "wp.uint64",
        "wp.Int",
    }
)

# Binary operators that keep an integer integral. ``Div`` is deliberately absent: its result is the
# thing under test, so admitting it would let one unflagged division launder its operands into a
# second.
_INT_PRESERVING = (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.BitAnd, ast.BitOr)


def _int_array_dtype(annotation: str) -> bool:
    """Whether a ``wp.array[...]`` annotation names a concrete integer dtype."""
    inside = annotation.partition("[")[2].rpartition("]")[0]
    return inside.split(",")[0].strip() in _INT_SCALARS


def _int_module_constants(tree: ast.Module) -> set[str]:
    """Module-level ``wp.constant(wp.int32(...))`` names, which kernels read as plain integers."""
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        value = node.value
        if not isinstance(target, ast.Name) or not isinstance(value, ast.Call):
            continue
        if _dotted(value.func) != ("wp", "constant") or not value.args:
            continue
        inner = value.args[0]
        if isinstance(inner, ast.Call) and ".".join(_dotted(inner.func)) in _INT_SCALARS:
            names.add(target.id)
    return names


def _int_typed_names(function: ast.FunctionDef, constants: set[str]) -> tuple[set[str], set[str]]:
    """
    Names inside a kernel-scope body that are integers *by declaration*, and integer arrays.

    Declaration, not inference: a module-level integer ``wp.constant``, an annotated parameter, or
    a local whose right-hand side is a ``wp.tid()``, an integer constructor, a ``.shape[...]``, or
    an integer-preserving expression over names already known. Anything the scan cannot type this
    way stays untyped, so a division involving it is never reported -- the check misses rather than
    misfires.
    """
    scalars: set[str] = set(constants)
    arrays: set[str] = set()
    for argument in function.args.args:
        if argument.annotation is None:
            continue
        annotation = ast.unparse(argument.annotation)
        if annotation in _INT_SCALARS:
            scalars.add(argument.arg)
        elif annotation.startswith("wp.array") and _int_array_dtype(annotation):
            arrays.add(argument.arg)
    # Two passes so an assignment reached before its operands were typed still resolves; the
    # dependency chains here are short and a third pass has never added a name.
    for _ in range(2):
        for node in ast.walk(function):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if isinstance(target, ast.Tuple):
                # ``i, j = wp.tid()``: every element of a multi-dimensional thread index is an int.
                value = node.value
                if isinstance(value, ast.Call) and _dotted(value.func) == ("wp", "tid"):
                    scalars.update(
                        element.id for element in target.elts if isinstance(element, ast.Name)
                    )
                continue
            if isinstance(target, ast.Name) and _is_int_expression(node.value, scalars, arrays):
                scalars.add(target.id)
    return scalars, arrays


def _is_int_expression(node: ast.expr, scalars: set[str], arrays: set[str]) -> bool:
    """Whether ``node`` is an integer by declaration, given the names already typed."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, int) and not isinstance(node.value, bool)
    if isinstance(node, ast.Name):
        return node.id in scalars
    if isinstance(node, ast.Call):
        dotted = _dotted(node.func)
        return ".".join(dotted) in _INT_SCALARS or dotted == ("wp", "tid")
    if isinstance(node, ast.Subscript):
        if isinstance(node.value, ast.Attribute) and node.value.attr == "shape":
            return True
        return isinstance(node.value, ast.Name) and node.value.id in arrays
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _is_int_expression(node.operand, scalars, arrays)
    if isinstance(node, ast.BinOp) and isinstance(node.op, _INT_PRESERVING):
        return _is_int_expression(node.left, scalars, arrays) and _is_int_expression(
            node.right, scalars, arrays
        )
    return False


def integer_division_problems() -> list[str]:
    """
    Check 17: an integer ``/`` inside a ``@wp.kernel`` or ``@wp.func`` body.

    ``.claude/CLAUDE.md`` section 5: on integers Warp's ``/`` and ``//`` are the *same* operation --
    both truncate toward zero, unlike CPython's ``//``, which floors. So this is a legibility rule
    and not a correctness one: ``/`` on two ``int32``s reads as real division and truncates only
    because the operands happen to be integers, which means a reader has to recover both types
    before they know what the line does. Every dividend in this package is a non-negative index,
    where the two conventions coincide; the hazard the spelling creates is *porting* such a line
    between host Python and kernel scope, where the answer changes silently for a negative
    dividend.

    Why a check rather than a one-off edit: the third pass converted eight sites and wrote the rule
    into ``CLAUDE.md``, and the next module written after it reintroduced two
    (``algorithms/multigrid.py``'s ``column = t / n_rows`` and ``column = t / stride``). That is
    section 14's bar for a new check -- the same defect found twice -- and the same failure mode
    check 16 exists to prevent for casts.

    The scan is deliberately conservative, because the cost of a false positive is that the first
    person it annoys disables it. An operand counts as an integer only *by declaration*: an
    annotated ``wp.int*`` / ``wp.uint*`` / ``wp.Int`` parameter, an element of a ``wp.array`` whose
    annotated dtype is one of those, an integer literal, a ``.shape[...]``, ``wp.tid()``, an
    integer constructor, or an integer-preserving expression over those. A ``wp.Scalar``-generic
    parameter, a float, and anything whose type comes from a call the scan cannot see are all left
    untyped, so their divisions are never reported.
    """
    problems: list[str] = []
    for path in sorted(_PACKAGE_DIR.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue  # test_package_scan_is_discoverable reports the parse failure
        constants = _int_module_constants(tree)
        for function in _kernel_scope_functions(tree):
            scalars, arrays = _int_typed_names(function, constants)
            for node in ast.walk(function):
                if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
                    continue
                if not _is_int_expression(node.left, scalars, arrays):
                    continue
                if not _is_int_expression(node.right, scalars, arrays):
                    continue
                problems.append(
                    f"{path.relative_to(_REPO_ROOT)}:{node.lineno} {function.name} divides "
                    f"integers with '/' in '{ast.unparse(node)}' -- write // instead, which is the "
                    "same operation on integers and says so"
                )
    return problems


# --- check 18 -----------------------------------------------------------------------------------

# Bare Python annotations with a Warp equivalent. ``str`` is deliberately absent: no kernel-scope
# argument can be one, so a ``str`` annotation is proof the function is a Python-scope factory and
# not kernel code at all.
_BARE_ANNOTATIONS = {"bool": "wp.bool", "int": "wp.int32", "float": "wp.float32"}


def _annotation_nodes(function: ast.FunctionDef) -> list[tuple[str, ast.expr]]:
    """Every annotation in a signature, paired with the argument name (``->`` for the return)."""
    arguments = function.args
    every = [
        *arguments.posonlyargs,
        *arguments.args,
        *([arguments.vararg] if arguments.vararg else []),
        *arguments.kwonlyargs,
        *([arguments.kwarg] if arguments.kwarg else []),
    ]
    found = [(argument.arg, argument.annotation) for argument in every if argument.annotation]
    if function.returns is not None:
        found.append(("->", function.returns))
    return found


def bare_annotation_problems() -> list[str]:
    """
    Check 18: a bare ``bool`` / ``int`` / ``float`` annotation in a kernel-scope signature.

    ``.claude/CLAUDE.md`` section 2: kernel arguments and returns are spelled in Warp's types. Warp
    resolves the bare names to the same ones, so like checks 16 and 17 this is legibility rather
    than correctness -- and like them, that is exactly why nothing but a scan holds it. The tree
    carried 11 ``-> bool`` against 46 ``-> wp.bool``, five of them predating the fourth kernel pass
    and the newest written the day after it, which is section 14's bar for a check: the same defect
    found twice, in code written after the rule.

    The distinction that makes it sound is that it reads *only* ``@wp.kernel`` / ``@wp.func``
    signatures. A kernel **factory** is ordinary Python and its parameters are correctly plain --
    ``reduce.blocks_1d(n: int) -> int`` and ``neighbors._bvh_nearest_row_kernel(row_size: int,
    name: str)`` compute a launch geometry and a kernel name at import time, and annotating them
    ``wp.int32`` would be a lie about where they run. Those are not kernel-scope functions, so they
    are never scanned; ``str`` is left out of the mapping for the same reason, as a further tell.
    """
    problems: list[str] = []
    for path in sorted(_PACKAGE_DIR.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue  # test_package_scan_is_discoverable reports the parse failure
        for function in _kernel_scope_functions(tree):
            for name, annotation in _annotation_nodes(function):
                if not isinstance(annotation, ast.Name):
                    continue
                replacement = _BARE_ANNOTATIONS.get(annotation.id)
                if replacement is None:
                    continue
                where = "return" if name == "->" else f"argument '{name}'"
                problems.append(
                    f"{path.relative_to(_REPO_ROOT)}:{annotation.lineno} {function.name} annotates "
                    f"its {where} '{annotation.id}' -- write {replacement} instead"
                )
    return problems


# --- check 19 -----------------------------------------------------------------------------------

# The suffixes ``.claude/CLAUDE.md`` section 6 assigns to reference libraries. ``_np`` is
# deliberately absent: section 6 gives it to "NumPy/SciPy" and ``tests/parity.py`` counts it, which
# is right there because a ``parity`` marker has already declared that a second implementation was
# consulted -- but in the suite at large ``_np`` marks *inputs* at least as often as oracles.
# Measured: adding it takes this scan from 523 comparison tests to 783 and from 0 problems to 290.
#
# ``_gl`` is moderngl's, and this table has to be edited alongside ``tests/parity.py``'s
# ``_LIBRARY_SUFFIXES`` -- the two encode the same convention independently, so a reference added
# to one and not the other is enforced by half the gate. Measured clean: the only ``*_gl`` names in
# the suite are ``image_gl``, ``covered_gl`` and ``class_gl``, all moderngl's.
_REFERENCE_SUFFIXES = ("_tm", "_igl", "_pp", "_pml", "_o3d", "_pv", "_ml", "_pmf", "_gl")

# The four phrases the suite uses to label a comparison, all four in good standing. ``Class [ABCD]``
# and ``Not a library comparison`` are section 6's named labels; ``Not a parity assert`` and
# ``Triwarp against triwarp`` are its triwarp-against-triwarp family. A gate accepting only the
# first two would fail 14 correct tests and the author's fix would be to reword good docstrings.
_COMPARISON_LABELS = re.compile(
    r"Class [ABCD]\b|Not a library comparison|Not a parity assert|[Tt]riwarp against triwarp"
)


def _asserted_reference_names(function: ast.FunctionDef) -> set[str]:
    """Reference-suffixed names read by an ``assert`` in this function."""
    found: set[str] = set()
    for statement in ast.walk(function):
        if not isinstance(statement, ast.Assert):
            continue
        for node in ast.walk(statement):
            if isinstance(node, ast.Name) and node.id.endswith(_REFERENCE_SUFFIXES):
                found.add(node.id)
    return found


def comparison_label_problems() -> list[str]:
    """
    Check 19: a test comparing against a reference library says which class the comparison is.

    ``.claude/CLAUDE.md`` section 6 asks for the label and explains what each class means; this only
    checks that one of the four phrases is present. It cannot check that the label is the *right*
    one -- that stays a review question, as section 14 says of every naming rule -- and it must not
    try: a ``_tm`` name inside an ``assert`` is not proof of an oracle.
    ``test_split_single_component`` compares ``split``'s output against ``mesh_tm.vertices``, the
    *input* mesh, which is a round trip; and in
    ``test_split_faces_along_field_positive_side_is_the_clip`` the ``_tm`` names are triwarp results
    run through ``warp_to_trimesh``. Both are correctly labelled and neither is a library
    comparison.

    Two decisions make it fire only on real omissions. It keys on ``ast.Assert`` rather than on the
    whole function body, because a fixture unpack ``mesh_tm, mesh_wp = icosphere`` names a ``_tm``
    variable in every mesh test -- keying on the body takes this from 0 problems to 120. And it
    accepts all four label phrases rather than section 6's two headline ones, for the reason given
    at ``_COMPARISON_LABELS``.

    It exists because the convention has decayed twice. Round 2 of the test-suite rescan took the
    lowercase ``class b`` spelling from 21 to 0, and 9 more had accumulated by round 4 -- invisible
    to the grep section 6 prescribes, since a human reads them the same. The other defect it closes
    is rarer and worse: ``test_fill_min_weight_matches_meshlib`` carried a ``parity`` marker,
    compared a class-C statistic against MeshLib, and had **no docstring at all**, which ruff cannot
    see because ``D103`` is in the ignore list.
    """
    problems: list[str] = []
    for path in sorted(_TESTS_DIR.glob("test_*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue  # test_package_scan_is_discoverable reports the parse failure
        for function in ast.walk(tree):
            if not isinstance(function, ast.FunctionDef) or not function.name.startswith("test_"):
                continue
            names = _asserted_reference_names(function)
            if not names:
                continue
            docstring = ast.get_docstring(function) or ""
            if _COMPARISON_LABELS.search(docstring):
                continue
            reason = "has no docstring" if not docstring else "carries no class label"
            problems.append(
                f"{path.relative_to(_REPO_ROOT)}:{function.lineno} {function.name} {reason} "
                f"but asserts on {', '.join(sorted(names)[:3])} -- write 'Class A'/'Class B'/"
                f"'Class C'/'Class D', 'Not a library comparison' or 'Not a parity assert'"
            )
    return problems
