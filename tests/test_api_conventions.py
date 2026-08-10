"""
The convention gate: the public API's names, summaries and file layout, checked statically.

Nine conventions, one test each so the failing test's *name* says which one was broken. The scan
and the reasoning behind each rule live in [`tests/api_conventions.py`](api_conventions.py); this
file is only the pytest surface.

Deliberately not parametrized over modules or functions: that would add hundreds of always-green
items to every run, and a rule that stopped matching anything would silently lose its check instead
of failing.
"""

from __future__ import annotations

import pytest

from tests.api_conventions import (
    coverage_location_problems,
    duplicate_name_problems,
    helper_order_problems,
    installed_warp_version,
    kernel_module_problems,
    library_in_summary_problems,
    mask_return_problems,
    private_import_problems,
    scan_package,
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
    they have not been introduced to yet. ``_HELPER_ORDER_ALLOWLIST`` carries the 50 sites that
    predate this check, as an explicit debt list rather than a silent exemption -- it is also
    checked for staleness, so an entry that gets fixed has to be removed rather than left to rot.
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
