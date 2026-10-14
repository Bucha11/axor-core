"""Revision 10: above the unattended ceiling a call needs a ground — a covering grant for
CONSEQUENTIAL, L3/L4 only for CATASTROPHIC; a non-grant sink above it is refused."""
from __future__ import annotations

from axor_core.contracts.canonical import ConsequenceClass as C
from axor_core.config import GovernanceConfig
from axor_core.governor import ToolCallGovernor
from axor_core.kernel.confirm import ScriptedConfirmer

BASE = dict(
    egress_sinks={"send_money"},
    integrity_sinks={"delete_file", "create_file", "post_note"},
    driving_args={"send_money": ["recipient", "amount"], "delete_file": ["file_id"],
                  "create_file": ["filename"], "post_note": ["target"]},
    grant_args={"send_money": {"recipient": "iban", "amount": "money"},
                "delete_file": {"file_id": "id"}},
    consequence_overrides={"send_money": C.CONSEQUENTIAL, "delete_file": C.CATASTROPHIC,
                           "create_file": C.REVERSIBLE, "post_note": C.CONSEQUENTIAL},
    max_unattended_consequence=C.REVERSIBLE,
    integrity_default="context",
    integrity_origins="request-only",
)
IBAN = "GB29NWBK60161331926819"


def _gov(request, **kw):
    g = ToolCallGovernor(**{**BASE, **kw})
    g.register_task(request)
    g.grants.from_request(request)
    return g


def test_consequential_needs_a_grant():
    assert not _gov("pay the bill").evaluate("send_money", {"recipient": IBAN, "amount": 10}).allowed
    d = _gov(f"send 10 to {IBAN}").evaluate("send_money", {"recipient": IBAN, "amount": 10})
    assert d.allowed and d.grant_source == "L1"


def test_reversible_is_unattended():
    assert _gov("make notes.txt").evaluate("create_file", {"filename": "notes.txt"}).allowed


def test_over_ceiling_without_grant_args_is_refused():
    d = _gov("post to board").evaluate("post_note", {"target": "board"})
    assert not d.allowed and d.category == "consequence_gate"


def test_catastrophic_not_by_request_literal():
    d = _gov("delete file 13").evaluate("delete_file", {"file_id": "13"})
    assert not d.allowed


def test_catastrophic_by_confirmation():
    g = _gov("delete file 13", confirmer=ScriptedConfirmer(lambda a, c: True))
    d = g.evaluate("delete_file", {"file_id": "13"})
    assert d.allowed and d.grant_source == "L4"


def test_default_ceiling_unchanged_for_consequential():
    kw = {k: v for k, v in BASE.items() if k != "max_unattended_consequence"}
    g = ToolCallGovernor(**kw)
    g.register_task("post to board")
    assert g.evaluate("post_note", {"target": "board"}).allowed


def test_config_key():
    cfg = GovernanceConfig.from_dict({"max_unattended_consequence": "reversible"})
    assert cfg.as_governor_kwargs()["max_unattended_consequence"] is C.REVERSIBLE
