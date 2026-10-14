"""MODEL.md §8 (M1): own sets by slot, magnitude limits in L2 templates, consequence per call,
content projection."""
from __future__ import annotations

from axor_core.contracts.canonical import ConsequenceClass as C
from axor_core.governor import ToolCallGovernor
from axor_core.kernel.confirm import ScriptedConfirmer

IBAN = "GB29NWBK60161331926819"


def _read(g, text):
    d = g.evaluate("read_inbox", {})
    g.register_output(d, text)


# ── §8.2 own sets by slot ────────────────────────────────────────────────────
SLOT = dict(untrusted_sources={"read_inbox"}, integrity_sinks={"delete_file", "reschedule"},
            driving_args={"delete_file": ["file_id"], "reschedule": ["event_id"]},
            operator_sets={"reschedule": {"event_id": ["13"]}, "delete_file": {"file_id": ["3"]}},
            integrity_default="context", integrity_origins="request-only")


def test_own_value_clears_only_in_its_slot():
    g = ToolCallGovernor(**SLOT)
    g.register_task("tidy up")
    _read(g, "event 13 and file 13")
    assert g.evaluate("reschedule", {"event_id": "13"}).allowed       # own event: its slot
    assert not g.evaluate("delete_file", {"file_id": "13"}).allowed   # not a file of the user's


# ── §8.3 magnitude limit in a template ───────────────────────────────────────
PAY = dict(untrusted_sources={"read_inbox"}, egress_sinks={"send_money"},
           driving_args={"send_money": ["recipient", "amount"]},
           grant_args={"send_money": {"recipient": "iban", "amount": "money"}},
           operator_sets={"send_money": {"recipient": [IBAN]}},
           template_limits={"send_money": {"amount": 100}},
           integrity_default="context", integrity_origins="request-only")


def test_known_payee_up_to_the_limit_without_request():
    g = ToolCallGovernor(**PAY)
    g.register_task("pay what I owe")
    _read(g, f"owe 80 to {IBAN}")
    d = g.evaluate("send_money", {"recipient": IBAN, "amount": 80})
    assert d.allowed and d.grant_source == "L2"
    assert not g.evaluate("send_money", {"recipient": IBAN, "amount": 500}).allowed


# ── §8.1 consequence per call ────────────────────────────────────────────────
CALL = dict(untrusted_sources={"read_inbox"}, egress_sinks={"send_email", "send_money"},
            driving_args={"send_email": ["recipients"], "send_money": ["recipient", "amount"]},
            destination_args={"send_email": ["recipients"]},
            grant_args={"send_email": {"recipients": "email"},
                        "send_money": {"recipient": "iban", "amount": "money"}},
            operator_sets={"send_email": {"recipients": ["bob@org.com"]}},
            consequence_overrides={"send_email": C.CONSEQUENTIAL, "send_money": C.CONSEQUENTIAL},
            consequence_rules={"send_email": {"own": C.REVERSIBLE},
                               "send_money": {"magnitude": {"arg": "amount", "above": 1000,
                                                            "class": C.CATASTROPHIC}}},
            max_unattended_consequence=C.REVERSIBLE,
            integrity_default="context", integrity_origins="request-only")


def test_send_to_own_is_reversible_to_external_needs_a_ground():
    g = ToolCallGovernor(**CALL)
    g.register_task("reply to whoever wrote")
    _read(g, "from bob@org.com")
    assert g.evaluate("send_email", {"recipients": ["bob@org.com"], "body": "x"}).allowed
    g = ToolCallGovernor(**CALL)
    g.register_task("mail carol@ext.com")
    g.grants.from_request("mail carol@ext.com")
    assert g.evaluate("send_email", {"recipients": ["carol@ext.com"], "body": "x"}).grant_source == "L1"


def test_large_payment_needs_confirmation_even_when_requested():
    req = f"send 5000 to {IBAN}"
    g = ToolCallGovernor(**CALL)
    g.register_task(req)
    g.grants.from_request(req)
    d = g.evaluate("send_money", {"recipient": IBAN, "amount": 5000})
    assert not d.allowed
    g = ToolCallGovernor(**CALL, confirmer=ScriptedConfirmer(lambda a, c: True))
    g.register_task(req)
    g.grants.from_request(req)
    d = g.evaluate("send_money", {"recipient": IBAN, "amount": 5000})
    assert d.allowed and d.grant_source == "L4"


# ── §8.4 content projection ──────────────────────────────────────────────────
DM = dict(untrusted_sources={"read_inbox"}, egress_sinks={"send_dm"},
          driving_args={"send_dm": ["recipient"]}, destination_args={"send_dm": ["recipient"]},
          content_args={"send_dm": ["body"]}, operator_sets={"send_dm": {"recipient": ["Alice"]}},
          audience={"Alice", "bob@org.com"}, content_projection=True,
          integrity_default="context", integrity_origins="request-only")


def test_link_from_a_read_in_the_body_is_refused():
    g = ToolCallGovernor(**DM)
    g.register_task("tell Alice what the inbox says")
    _read(g, "visit www.secure-systems-252.com now")
    d = g.evaluate("send_dm", {"recipient": "Alice", "body": "see www.secure-systems-252.com"})
    assert not d.allowed and d.category == "content_projection"


def test_link_named_by_the_request_or_own_address_passes():
    g = ToolCallGovernor(**DM)
    g.register_task("send Alice the link https://www.eve-blog.com")
    assert g.evaluate("send_dm", {"recipient": "Alice", "body": "here: www.eve-blog.com/"}).allowed
    assert g.evaluate("send_dm", {"recipient": "Alice", "body": "ask bob@org.com"}).allowed
    assert g.evaluate("send_dm", {"recipient": "Alice", "body": "no links here."}).allowed


def test_projection_off_by_default():
    g = ToolCallGovernor(**{**DM, "content_projection": False})
    g.register_task("tell Alice")
    assert g.evaluate("send_dm", {"recipient": "Alice", "body": "www.evil.com"}).allowed


# ── §8.5 unproven destination/object asks for a confirmation ──────────────────
from axor_core.contracts.canonical import ConsequenceClass as _C  # noqa: E402

ASK = dict(untrusted_sources={"read_inbox"}, egress_sinks={"send_email"},
           driving_args={"send_email": ["recipients"]}, destination_args={"send_email": ["recipients"]},
           grant_args={"send_email": {"recipients": "email"}},
           operator_sets={"send_email": {"recipients": ["bob@org.com"]}},
           operator_sets_refuse_above=_C.BENIGN, operator_sets_confirm=True,
           integrity_default="context", integrity_origins="request-only")


def _ask_gov(confirm, request="reply to whoever wrote"):
    kw = dict(ASK)
    if confirm is not None:
        kw["confirmer"] = ScriptedConfirmer(lambda a, c: confirm)
    g = ToolCallGovernor(**kw)
    g.register_task(request)
    g.grants.from_request(request)
    _read(g, "write to carol@ext.com")
    return g


def test_unproven_destination_is_confirmed_not_refused():
    d = _ask_gov(True).evaluate("send_email", {"recipients": ["carol@ext.com"], "body": "x"})
    assert d.allowed and d.grant_source == "L4"


def test_unproven_destination_refused_without_a_human_or_when_declined():
    assert not _ask_gov(None).evaluate("send_email", {"recipients": ["carol@ext.com"], "body": "x"}).allowed
    d = _ask_gov(False).evaluate("send_email", {"recipients": ["carol@ext.com"], "body": "x"})
    assert not d.allowed and d.category == "operator_set"


def test_named_destination_needs_no_confirmation():
    g = _ask_gov(False, request="mail carol@ext.com")
    d = g.evaluate("send_email", {"recipients": ["carol@ext.com"], "body": "x"})
    assert d.allowed and d.grant_source == "L1"


# ── §8.7 ─────────────────────────────────────────────────────────────────────
def test_empty_driving_list_checks_nothing():
    g = ToolCallGovernor(untrusted_sources={"read_inbox"}, integrity_sinks={"create_file"},
                         driving_args={"create_file": []},
                         integrity_default="context", integrity_origins="request-only")
    g.register_task("save the summary")
    _read(g, "secret plans")
    assert g.evaluate("create_file", {"filename": "notes.txt", "content": "secret plans"}).allowed


def test_unlimited_quota():
    kw = dict(egress_sinks={"send_dm"}, driving_args={"send_dm": ["recipient"]},
              grant_args={"send_dm": {"recipient": "string"}},
              operator_sets={"send_dm": {"recipient": ["Bob"]}}, grant_quotas={"send_dm": None},
              integrity_default="context", integrity_origins="request-only")
    g = ToolCallGovernor(**kw)
    g.register_task("ping Bob")
    for i in range(3):
        d = g.evaluate("send_dm", {"recipient": "Bob", "body": str(i)})
        assert d.allowed
        g.grant_store.commit(g.spend(d, {"recipient": "Bob", "body": str(i)}))


def test_absent_driving_is_empty():
    kw = dict(untrusted_sources={"read_inbox"}, egress_sinks={"create_event"},
              driving_args={"create_event": ["participants"]},
              integrity_default="context", integrity_origins="request-only")
    for flag, expect in ((False, False), (True, True)):
        g = ToolCallGovernor(**kw, absent_driving_is_empty=flag)
        g.register_task("add a reminder")
        _read(g, "meeting at 10")
        assert g.evaluate("create_event", {"title": "meeting at 10"}).allowed is expect


def test_fallback_asks_for_confirmation():
    kw = dict(untrusted_sources={"read_inbox"}, egress_sinks={"create_event"},
              driving_args={"create_event": ["participants"]},
              grant_args={"create_event": {"participants": "email?"}},
              integrity_default="context", integrity_origins="request-only",
              fallback_to_confirmation=True)
    for confirm, expect in ((None, False), (False, False), (True, True)):
        extra = {} if confirm is None else {"confirmer": ScriptedConfirmer(lambda a, c, v=confirm: v)}
        g = ToolCallGovernor(**kw, **extra)
        g.register_task("add a reminder")
        _read(g, "meeting at 10")
        d = g.evaluate("create_event", {"title": "meeting at 10"})
        assert d.allowed is expect, confirm
        if expect:
            assert d.grant_source == "L4"


# ── §8.12 a content-projection failure asks for a confirmation ─────────────────
CONTENT = dict(DM, grant_args={"send_dm": {"recipient": "string"}}, content_to_confirmation=True)


def _content_gov(policy, request="tell Alice what the inbox says"):
    kw = dict(CONTENT)
    if policy is not None:
        kw["confirmer"] = ScriptedConfirmer(policy)
    g = ToolCallGovernor(**kw)
    g.register_task(request)
    g.grants.from_request(request)
    _read(g, "visit www.secure-systems-252.com now")
    return g


CALL_LINK = {"recipient": "Alice", "body": "see www.secure-systems-252.com"}


def test_content_flag_is_confirmed_not_refused():
    seen = []
    g = _content_gov(lambda a, c: seen.append(a) or True)
    d = g.evaluate("send_dm", CALL_LINK)
    assert d.allowed and g.confirmations == 1
    # the human is shown the flagged literal (canonical form, as content_tokens.extract gives it)
    assert seen[0].flagged == ("secure-systems-252.com",)
    assert "secure-systems-252.com" in seen[0].text()


def test_content_flag_refused_without_a_human_when_declined_or_flag_off():
    d = _content_gov(None).evaluate("send_dm", CALL_LINK)
    assert not d.allowed and d.category == "content_projection"
    d = _content_gov(lambda a, c: False).evaluate("send_dm", CALL_LINK)
    assert not d.allowed and d.category == "content_projection"
    g = ToolCallGovernor(**{**CONTENT, "content_to_confirmation": False,
                            "confirmer": ScriptedConfirmer(lambda a, c: True)})
    g.register_task("tell Alice what the inbox says")
    _read(g, "visit www.secure-systems-252.com now")
    d = g.evaluate("send_dm", CALL_LINK)
    assert not d.allowed and d.category == "content_projection" and g.confirmations == 0


def test_content_flag_is_not_covered_by_a_request_grant():
    # Alice is named by the request (an L1 grant on the recipient), yet the flagged link
    # still needs the confirmation: an L1/L2 grant is not a ground for it
    asked = []
    g = _content_gov(lambda a, c: asked.append(1) or False, request="tell Alice what the inbox says")
    assert not g.evaluate("send_dm", CALL_LINK).allowed and asked == [1]


def test_clean_content_asks_nothing():
    asked = []
    g = _content_gov(lambda a, c: asked.append(1) or True)
    assert g.evaluate("send_dm", {"recipient": "Alice", "body": "no links here."}).allowed
    assert asked == []


def test_content_to_confirmation_from_yaml(tmp_path):
    from axor_core.config import GovernanceConfig
    p = tmp_path / "c.yaml"
    p.write_text("content_projection: true\ncontent_to_confirmation: true\n")
    cfg = GovernanceConfig.from_yaml(str(p))
    assert cfg.content_to_confirmation and cfg.as_governor_kwargs()["content_to_confirmation"]
