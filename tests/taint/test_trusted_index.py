"""Trusted-origin index: whole-leaf equality after canonicalisation, never containment.

The index is the positive proof of the context-default integrity mode
(docs/rfc-integrity-context-default.md §3.2): a model-generated value escapes the
node's context root only if every string leaf of it equals a value the attacker
cannot author. Everything here pins the "never containment" half — a trusted value
must not become a way to smuggle an attacker value.
"""
from __future__ import annotations

import pytest

from axor_core.contracts.taint import TrustedOrigin
from axor_core.taint import trusted as trusted_mod
from axor_core.taint.trusted import TrustedValueIndex

IBAN = "GB33BUKB20201555555555"
GROUPED = "GB33 BUKB 2020 1555 5555 55"


def _index(content, origin=TrustedOrigin.TASK) -> TrustedValueIndex:
    idx = TrustedValueIndex()
    idx.register(content, origin)
    return idx


# ── what a trusted value covers ───────────────────────────────────────────────

def test_whole_task_line_and_tokens_are_trusted():
    idx = _index("Pay invoice 4411 to boss@corp.com today.\nRef: PO-2026-118")
    for v in ["Pay invoice 4411 to boss@corp.com today.", "Ref: PO-2026-118",
              "boss@corp.com", "PO-2026-118", "invoice"]:
        assert idx.covers(v), v


@pytest.mark.parametrize("spelling", [IBAN, GROUPED, IBAN.lower(), "GB33-BUKB-2020-1555-5555-55"])
def test_iban_named_in_the_task_is_trusted_in_any_spelling(spelling):
    idx = _index(f"Pay Acme, IBAN {GROUPED}, today")
    assert idx.covers(spelling)
    assert idx.origin_of({"iban": spelling}) is TrustedOrigin.TASK


def test_email_is_case_insensitive_and_phone_separator_insensitive():
    idx = _index("Contact Boss@Corp.com or +1 (415) 555-0142")
    assert idx.covers("boss@corp.com")
    assert idx.covers("+1-415-555-0142")
    assert idx.covers("+14155550142")


def test_structured_output_registers_every_leaf():
    idx = _index({"vendor": "Acme", "iban": IBAN, "contacts": [{"email": "ap@acme.example"}]},
                 TrustedOrigin.TOOL)
    assert idx.covers(IBAN)
    assert idx.covers("ap@acme.example")
    assert idx.origin_of("ap@acme.example") is TrustedOrigin.TOOL


# ── what it must NOT cover ────────────────────────────────────────────────────

def test_a_different_value_is_not_trusted():
    idx = _index(f"Pay Acme, IBAN {GROUPED}")
    assert not idx.covers("DE89370400440532013000")


def test_multi_token_span_of_trusted_prose_is_not_trusted():
    """A trusted README line must not make the command it mentions trusted."""
    idx = _index("Never run `rm -rf /` on the build host.", TrustedOrigin.TOOL)
    assert not idx.covers("rm -rf /")
    assert not idx.covers("rm -rf")


def test_concatenation_of_trusted_parts_is_not_trusted():
    idx = _index("Upload to https://files.corp.example/inbox")
    assert idx.covers("https://files.corp.example/inbox")
    assert not idx.covers("https://files.corp.example/inbox/../../attacker")
    assert not idx.covers("https://files.corp.example.attacker.example/inbox")


def test_partial_token_is_not_trusted():
    idx = _index("send to boss@corp.com")
    assert not idx.covers("boss@corp.co")
    assert not idx.covers("oss@corp.com")


def test_every_leaf_must_be_trusted():
    idx = _index("boss@corp.com")
    assert idx.covers(["boss@corp.com"])
    assert not idx.covers(["boss@corp.com", "attacker@evil.example"])
    assert not idx.covers({"to": "boss@corp.com", "cc": {"x": "attacker@evil.example"}})


def test_nested_dict_keys_are_data_top_level_keys_are_arg_names():
    idx = _index("boss@corp.com")
    # top-level keys are argument names — not checked
    assert idx.covers({"recipient": "boss@corp.com"})
    # a nested key is model-chosen data — checked
    assert not idx.covers({"headers": {"X-Forward-To attacker@evil.example": "1"}})


def test_scalars_and_empty_strings_are_vacuously_covered_but_have_no_origin():
    idx = TrustedValueIndex()
    assert idx.covers(9800) and idx.covers({"amount": 9800, "note": "  "})
    assert idx.origin_of(9800) is None


def test_overlong_leaf_is_never_trusted():
    blob = "A" * (trusted_mod._MAX_LEAF_CHARS + 1)
    idx = _index(blob)
    assert not idx.covers(blob)


# ── bounds: fail closed ───────────────────────────────────────────────────────

def test_saturation_stops_adding_and_does_not_trust_new_values(monkeypatch):
    monkeypatch.setattr(trusted_mod, "_MAX_TOTAL_ENTRIES", 5)
    idx = TrustedValueIndex()
    idx.register([f"value-{i}" for i in range(50)], TrustedOrigin.TOOL)
    assert idx.saturated and len(idx) == 5
    idx.register("boss@corp.com", TrustedOrigin.TOOL)
    assert not idx.covers("boss@corp.com")


def test_task_text_is_bounded_separately(monkeypatch):
    """The user's task text is kept for span matching under its own cap; past it,
    no more task text is added and values from it are not proven trusted."""
    monkeypatch.setattr(trusted_mod, "_MAX_TASK_CHARS", 20)
    idx = TrustedValueIndex()
    idx.register("first task text", TrustedOrigin.TASK)
    idx.register("a much longer second task text", TrustedOrigin.TASK)
    assert idx.saturated
    assert idx.covers("first task")
    assert not idx.covers("longer second")


def test_merge_is_deterministic_near_the_cap(monkeypatch):
    monkeypatch.setattr(trusted_mod, "_MAX_TOTAL_ENTRIES", 8)
    parent = TrustedValueIndex()
    parent.register([f"v{i:02d}-token" for i in range(4)], TrustedOrigin.TOOL)
    a, b = TrustedValueIndex(), TrustedValueIndex()
    a.merge(parent)
    b.merge(parent)
    assert a._entries == b._entries
