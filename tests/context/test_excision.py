"""Context excision is a governance action: authority-gated, operator-in-loop."""
from __future__ import annotations

import pytest

from axor_core.context.excision import ExcisionError, apply_excision
from axor_core.contracts.context import ContextFragment
from axor_core.contracts.degradation import GovernanceAuthority


def _frag(mark: str, content: str = "x") -> ContextFragment:
    return ContextFragment(kind="tool_result", content=content, token_estimate=1,
                           source="web", taint_mark=mark)


_FRAGS = [_frag("CAN-A"), _frag("CAN-B"), ContextFragment(
    kind="fact", content="task", token_estimate=1, source="raw_task")]  # untainted, no mark

_AUTO = GovernanceAuthority(authority_id="policy-1", authority_type="automated_policy",
                            reason_code="context_repair")
_HUMAN = GovernanceAuthority(authority_id="op-7", authority_type="human_operator",
                             reason_code="context_repair")


def test_automated_policy_removes_auto_defers_escalate() -> None:
    r = apply_excision(_FRAGS, auto_excise=["CAN-A"], escalate=["CAN-B"], authority=_AUTO)
    assert r.excised == ("CAN-A",)
    assert r.deferred == ("CAN-B",)                       # operator-only — left in place
    marks = {f.taint_mark for f in r.repaired_fragments}
    assert "CAN-A" not in marks and "CAN-B" in marks      # B survives automated repair


def test_human_operator_removes_both() -> None:
    r = apply_excision(_FRAGS, auto_excise=["CAN-A"], escalate=["CAN-B"], authority=_HUMAN)
    assert set(r.excised) == {"CAN-A", "CAN-B"}
    assert r.deferred == ()
    assert all(f.taint_mark not in ("CAN-A", "CAN-B") for f in r.repaired_fragments)
    assert len(r.repaired_fragments) == 1                 # only the untainted fact remains


def test_invalid_authority_raises() -> None:
    with pytest.raises(ExcisionError):
        apply_excision(_FRAGS, auto_excise=["CAN-A"],
                       authority=GovernanceAuthority(authority_id="", authority_type="automated_policy",
                                                     reason_code="r"))
    with pytest.raises(ExcisionError):
        apply_excision(_FRAGS, auto_excise=["CAN-A"],
                       authority=GovernanceAuthority(authority_id="x", authority_type="attacker",
                                                     reason_code="r"))


def test_untainted_fragments_untouched() -> None:
    r = apply_excision(_FRAGS, auto_excise=["CAN-A", "CAN-B"], authority=_HUMAN)
    assert any(f.kind == "fact" for f in r.repaired_fragments)  # the legitimate fact stays
