"""The permitting-ground specification is checked, not re-read.

``axor_core.kernel.grounds.GROUNDS`` lists every input that can turn a denial into
an allow, with its issuer, check, behaviour on absence and what it does not
override. These tests hold that list to the code: each named mechanism imports,
each named test exists, and the table in the theorem document is the one rendered
from the list.
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from axor_core.kernel.grounds import GROUNDS, STATUSES, render_markdown

ROOT = Path(__file__).resolve().parents[2]
BEGIN, END = "<!-- grounds:begin -->", "<!-- grounds:end -->"


def _resolve(ref: str) -> object:
    module, _, attr = ref.partition(":")
    obj: object = importlib.import_module(module)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def _test_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    return {
        n.name for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test_")
    }


@pytest.mark.parametrize("ground", GROUNDS, ids=lambda g: g.name[:40])
def test_every_ground_is_fully_stated(ground):
    assert ground.status in STATUSES
    for text in (ground.name, ground.issuer, ground.permits, ground.check,
                 ground.on_absence, ground.does_not_override, ground.premise):
        assert text.strip()
    if ground.status == "premise":
        # a ground core cannot check must say what it rests on
        assert ground.premise != "—"
    assert ground.pinned_by, "a ground no test exercises is a claim, not a specification"
    if ground.status != "inadmissible":
        assert ground.verified_by


@pytest.mark.parametrize("ground", GROUNDS, ids=lambda g: g.name[:40])
def test_every_named_mechanism_exists(ground):
    for ref in ground.verified_by:
        assert _resolve(ref) is not None, ref


@pytest.mark.parametrize("ground", GROUNDS, ids=lambda g: g.name[:40])
def test_every_pinning_test_exists(ground):
    for ref in ground.pinned_by:
        path, _, name = ref.partition("::")
        assert name in _test_names(ROOT / path), ref


def test_names_are_unique():
    names = [g.name for g in GROUNDS]
    assert len(names) == len(set(names))


def test_the_theorem_table_is_rendered_from_the_specification():
    doc = (ROOT / "docs" / "kernel-theorem.md").read_text()
    assert BEGIN in doc and END in doc
    table = doc.split(BEGIN, 1)[1].split(END, 1)[0].strip()
    assert table == render_markdown().strip(), (
        "docs/kernel-theorem.md §3b is out of date — regenerate it from "
        "axor_core.kernel.grounds.render_markdown()"
    )


def test_attestation_is_not_presented_as_a_checked_contract():
    by_name = {g.name: g for g in GROUNDS}
    contract = by_name["Consumption contract (T4 for a guarded field)"]
    attested = by_name["Consumption attestation"]
    assert contract.status == "enforced" and attested.status == "premise"
    assert "none" in attested.check
