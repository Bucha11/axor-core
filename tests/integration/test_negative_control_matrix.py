"""Every row of the negative-control matrix shows the control pattern, and the
document is the measured table."""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.negative_controls import SCENARIOS, path_registration_control, render_markdown

DOC = Path(__file__).resolve().parents[2] / "docs" / "negative-controls.md"


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.condition)
def test_removing_the_condition_produces_the_effect_restoring_it_removes_it(scenario):
    present, removed, restored = scenario.run(True), scenario.run(False), scenario.run(True)
    assert removed.violations > 0, "the condition is not load-bearing in this scenario"
    assert present.violations == 0 and restored.violations == 0
    assert present.legit == present.legit_total, "the condition blocks the legitimate call"


def test_the_path_case_is_decided_at_registration():
    refused, leaked = path_registration_control()
    assert refused and leaked > 0


def test_the_document_is_the_measured_table():
    doc = DOC.read_text()
    table = doc.split("<!-- matrix:begin -->", 1)[1].split("<!-- matrix:end -->", 1)[0]
    assert table.strip() == render_markdown().strip(), (
        "docs/negative-controls.md is stale — regenerate with `python -m tests.negative_controls`"
    )
