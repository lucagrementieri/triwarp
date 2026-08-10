"""
The convention gate: the public API's names, summaries and file layout, checked statically.

Fourteen conventions, one test each so the failing test's *name* says which one was broken. The scan
and the reasoning behind each rule live in [`tests/api_conventions.py`](api_conventions.py); this
file is only the pytest surface -- with one exception, the docstring-example test, whose whole
point is that a static read cannot find what is wrong with an example.

Deliberately not parametrized over modules or functions: that would add hundreds of always-green
items to every run, and a rule that stopped matching anything would silently lose its check instead
of failing. The example test *is* parametrized, because there are four of them and the failure has
to name which one.
"""

from __future__ import annotations

import pytest
import trimesh as tm
import warp as wp

import triwarp as tw
from tests.api_conventions import (
    DocstringExample,
    allocation_device_problems,
    array_annotation_style_problems,
    coverage_location_problems,
    docstring_examples,
    duplicate_name_problems,
    helper_order_problems,
    installed_warp_version,
    kernel_module_problems,
    kernel_output_naming_problems,
    library_in_summary_problems,
    mask_return_problems,
    private_import_problems,
    scan_package,
    undocumented_raise_problems,
    warp_suffix_problems,
    warp_version_problems,
)


def _fail(headline: str, problems: list[str]) -> None:
    """Fail with a counted headline and one indented line per problem, without a traceback."""
    if not problems:
        return
    body = "\n".join(f"  {problem}" for problem in sorted(problems))
    # pytrace=False: the traceback runs through the scanner and says nothing useful, and pytest
    # truncates long assertion explanations in non-verbose runs -- which is exactly when the full
    # list is wanted.
    pytest.fail(f"{len(problems)} {headline}\n{body}", pytrace=False)


def test_package_scan_is_discoverable() -> None:
    """
    Guard the scan itself: a silently empty scan would make all seven checks vacuously green.

    ``triwarp/`` ships in the wheel, so unlike the parity gate there is no packaged-tree case where
    it can legitimately be missing.
    """
    scan = scan_package()
    _fail("module(s) could not be parsed:", scan.errors)
    assert scan.modules, "triwarp/ has modules but the scan found none"
    assert scan.functions, "triwarp/ has public functions but the scan found none"


def test_summaries_do_not_name_a_reference_library() -> None:
    """
    A one-line summary says what the function returns, not which C++ call it wraps.

    mkdocstrings renders the summary as the entry in the module's API index, so a library name there
    turns the index into a table of bindings -- the same defect ``.claude/CLAUDE.md`` section 14
    already forbids one level in, where it rules that a function is named after what it returns.
    Attribution is wanted and stays: one line down, in ``Notes`` or ``See Also``.
    """
    _fail("public summary line(s) naming a reference library:", library_in_summary_problems())


def test_mask_producers_return_boolean_arrays() -> None:
    """
    A ``*_mask`` producer returns ``wp.array[wp.bool]``.

    The suffix is a family, and a member returning something else -- ``filter_unsharp_mask``
    returned vertex *positions* -- makes the whole family unreadable to anyone scanning it. A
    function that consumes a mask is named for its input and is exempt.
    """
    _fail("*_mask function(s) that do not return a boolean array:", mask_return_problems())


def test_module_summaries_do_not_advertise_warp() -> None:
    """A module summary does not end in "(Warp)" or "on NVIDIA Warp": the whole package is Warp."""
    _fail("module summary(ies) carrying a redundant Warp suffix:", warp_suffix_problems())


def test_coverage_lives_beside_its_module() -> None:
    """
    Every module has a ``tests/`` and a ``benchmarks/`` file named for it, and vice versa.

    ``.claude/CLAUDE.md`` section 14's "coverage is per module" rule, made mechanical. The failure
    it catches is a function's tests drifting into a neighbour's file, which is invisible until
    somebody goes looking for them -- and a suite file whose module has been renamed out from under
    it.
    """
    _fail("coverage file(s) not named for their module:", coverage_location_problems())


def test_private_names_stay_in_their_module() -> None:
    """
    A private helper with callers in two modules is a public function that was written in a hurry.

    ``proximity._default_mesh_query_max_dist`` had four importers, one of which aliased it straight
    back to a public-looking name on import -- the same fact stated as a workaround. The allowlist
    in the scanner carries a written reason per entry.
    """
    _fail("cross-module private import(s):", private_import_problems())


def test_public_names_are_unique_across_modules() -> None:
    """
    Two modules do not export the same public name, outside a written allowlist.

    ``points.centroid`` and ``triangles.centroid`` differed in what they meant (point mean against
    area-weighted surface centroid) *and* in whether calling them synced the device. The allowlist
    holds ``concatenate``, where both spellings are required by the vocabularies they mirror.
    """
    _fail("duplicated public name(s):", duplicate_name_problems())


def test_kernel_modules_are_named_for_their_wrapper() -> None:
    """
    ``triwarp/kernels/<module>.py`` backs ``triwarp/<module>.py``, one to one.

    ``.claude/CLAUDE.md`` section 4's rule. It is what stops a wrapper module from being created or
    renamed while its kernels are left behind under the old name -- the half of a move that compiles
    fine and is therefore easy to skip. ``predicates`` and ``scatter`` are the shared kernel-side
    libraries that back no single module; sub-packages mirror a folder and are not checked here.
    """
    _fail("kernel/wrapper module name mismatch(es):", kernel_module_problems())


def test_private_helpers_follow_their_callers() -> None:
    """
    A private helper is defined below the public function that calls it (the stepdown rule).

    ``.claude/CLAUDE.md`` section 11: a reader should never need to jump backward to a definition
    they have not been introduced to yet. ``_HELPER_ORDER_ALLOWLIST`` carried the 49 sites that
    predated this check as an explicit debt list rather than a silent exemption, and the staleness
    half of it did its job: the list is now drained to a single permanent entry, a helper called at
    module scope to build a constant, which has no caller to sit below.
    """
    _fail("private helper(s) above their first caller:", helper_order_problems())


def test_warp_version_claims_are_not_stale() -> None:
    """
    No comment or docstring blames a Warp version older than the installed ``warp-lang``.

    The defect this exists for: the 1.16 upgrade re-stamped all five ``reference/warp_api/``
    mirrors -- ``warp_version.py`` makes that checkable -- and left twelve *code* justifications
    citing bugs in 1.13-1.15, none re-probed. Six of those bugs were still real and one was not,
    and nothing in the tree could tell them apart. Unlike the other eight checks this one scans
    ``kernels/`` too, since five of the twelve lived there.

    ``_WARP_VERSION_ALLOWLIST`` is for a claim that deliberately records history, and the entry
    carries the reason -- so the next upgrade inherits a list of claims to re-run.
    """
    if installed_warp_version() is None:
        pytest.skip("warp-lang is not installed, so there is no version to compare against")
    _fail("stale Warp-version claim(s):", warp_version_problems())


def test_allocations_name_their_device() -> None:
    """
    Every Python-scope ``wp.zeros``/``empty``/``ones``/``full``/``array`` names a ``device``.

    Without it the buffer lands on Warp's *current* device rather than the device of the arrays it
    is about to be used with, and the suite cannot see the difference: a test runs with its arrays'
    device current, so the omitted argument resolves correctly by accident.
    ``array.index_sparse`` raised only under ``wp.ScopedDevice("cpu")`` with ``cuda:0`` inputs.
    """
    _fail("allocation(s) without device=:", allocation_device_problems())


def test_kernel_outputs_are_named_and_placed() -> None:
    """
    A kernel argument the kernel writes is named ``out_*``, and every ``out_*`` argument is last.

    ``.claude/CLAUDE.md`` section 3's rule, checked from both sides after the first full sweep of
    ``kernels/`` found eight outputs wearing plain names (four literally ``out``) and three
    read-only inputs wearing the prefix (``claim_collapses`` read a *prior* kernel's outputs under
    their producer's names). In-place arguments and scratch / persistent-state buffers are exempt
    by section 3 and listed in ``_KERNEL_OUTPUT_ALLOWLIST``, which is staleness-checked.
    """
    _fail("kernel output-naming violation(s):", kernel_output_naming_problems())


def test_array_annotations_are_subscript_style() -> None:
    """
    An array annotation reads ``wp.array[T]``, never the pre-1.12 ``wp.array(dtype=T)``.

    ``.claude/CLAUDE.md`` section 2. Both spellings compile, so nothing but a check stops the old
    one from coming back with the next large module: it survived in ``algorithms/ball_pivoting.py``
    (98 of the 176), ``reconstruction.py`` and ``remesh.py`` long after the convention settled, and
    ``remesh.py`` carried both styles at once -- which is the state that leaves a reader unsure
    which one is current.
    """
    _fail("call-style array annotation(s):", array_annotation_style_problems())


def test_public_functions_document_what_they_raise() -> None:
    """
    A public function with a ``raise`` in its own body documents a ``Raises`` block.

    ``.claude/CLAUDE.md`` section 14's "docstring, signature and body must agree", from the side
    where the body says more than the docstring. Delegated validation is not scanned -- 42 public
    functions correctly document a ``Raises`` their shared guard performs.
    """
    _fail("public function(s) raising without a Raises block:", undocumented_raise_problems())


@pytest.fixture
def example_namespace(icosahedron: tuple[tm.Trimesh, wp.Mesh]) -> dict[str, object]:
    """
    Bind every name the package's docstring examples use to a real object on the fixture's device.

    Every free name an example needs lives here. A new example that reaches for a name this
    namespace does not carry fails with a ``NameError``, which is the intended outcome: an example
    the suite cannot run is exactly the state that let two broken ones survive.
    """
    _, mesh_wp = icosahedron
    device = mesh_wp.device
    vertices, faces = mesh_wp.points, mesh_wp.indices
    queries = wp.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.5, 0.0]], dtype=wp.vec3, device=device
    )
    neighbor_idx, neighbor_distance = tw.neighbors.query_bvh_nearest(vertices, vertices, 4)
    return {
        "tw": tw,
        "wp": wp,
        "warp_mesh": mesh_wp,
        "v": vertices,
        "f": faces,
        "pts": queries,
        "origins": queries,
        "directions": wp.array(
            [[1.0, 0.0, 0.0]] * int(queries.shape[0]), dtype=wp.vec3, device=device
        ),
        "neighbor_idx": neighbor_idx,
        "neighbor_distance": neighbor_distance,
    }


@pytest.mark.parametrize(
    "example", [pytest.param(item, id=item.site) for item in docstring_examples()]
)
def test_docstring_examples_run(
    example: DocstringExample, example_namespace: dict[str, object]
) -> None:
    """
    Every fenced ``python`` block in a docstring executes against a real mesh.

    An example is the one piece of documentation that can be checked by running it, and both
    failures this test was written for were runtime ones invisible to ``ast.parse``:
    ``points.outlier_probability`` fed a NumPy bool array to ``array.flatnonzero``, and
    ``ray.contains_points`` compared a ``wp.array`` with ``0.0``, which Warp's Python-scope arrays
    do not support. Blocks containing a bare ``...`` are deliberate outlines and are skipped.
    """
    if example.is_sketch:
        pytest.skip(f"{example.site} is a deliberate sketch (bare ...), not runnable code")
    try:
        exec(compile(example.code, example.site, "exec"), dict(example_namespace))
    except Exception as error:  # noqa: BLE001
        pytest.fail(
            f"{example.site}: the documented example raises "
            f"{type(error).__name__}: {error}\n{example.code}",
            pytrace=False,
        )
