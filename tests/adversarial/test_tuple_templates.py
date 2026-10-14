"""L2 tuple templates (residuals_rules.md revision 9): per driving grant arg, an O-set
(operator enum) or a request literal. All-O templates grant at session start with the
operator quota; mixed ones at request parsing, the R component bound to a literal."""
from __future__ import annotations

from axor_core.governor import ToolCallGovernor
from axor_core.kernel.grants import compile_tuple_templates
from axor_core.policy.value_policy import ValuePredicate

O_USERS = {"Alice", "Bob"}
O_MAIL = {"a@org.com", "b@org.com"}

BASE = dict(
    untrusted_sources={"read_inbox"},
    sensitive_sources={"get_secret"},
    egress_sinks={"send_direct_message", "share_file"},
    access_control_sinks={"update_password"},
    integrity_sinks={"update_password"},
    driving_args={"send_direct_message": ["recipient"], "share_file": ["email", "permission"],
                  "update_password": ["password"]},
    grant_args={"send_direct_message": {"recipient": "string"},
                "share_file": {"email": "email", "permission": "string", "file_id": "id"},
                "update_password": {"password": "string"}},
    value_policies={
        "send_direct_message": [ValuePredicate(arg="recipient", kind="enum", allowed=frozenset(O_USERS))],
        "share_file": [ValuePredicate(arg="email", kind="enum", allowed=frozenset(O_MAIL))],
        "update_password": [ValuePredicate(arg="password", kind="enum", allowed=frozenset({"hunter2"}))],
    },
    integrity_default="context",
    integrity_origins="request-only",
)


def _gov(request=None, **kw):
    g = ToolCallGovernor(**{**BASE, **kw})
    if request is not None:
        g.register_task(request)
        g.grants.from_request(request)
    return g


def _spend(g, tool, args):
    d = g.evaluate(tool, args)
    if d.allowed and d.grant_id is not None:
        k = g.spend(d, args)
        g.grant_store.commit(k)
    return d


def test_templates_generated():
    t = compile_tuple_templates(BASE["value_policies"], BASE["grant_args"], BASE["driving_args"],
                                BASE["access_control_sinks"])
    assert t == {"send_direct_message": {"recipient": ["Alice", "Bob"]},
                 "share_file": {"email": ["a@org.com", "b@org.com"], "permission": None}}


def test_all_o_tuple_passes_without_request():
    g = _gov()
    d = g.evaluate("send_direct_message", {"recipient": "Bob", "body": "hi"})
    assert d.allowed and d.grant_source == "L2"


def test_all_o_tuple_after_untrusted_read_without_request_literal():
    g = _gov("Reply to whoever wrote.")
    d = g.evaluate("read_inbox", {})
    g.register_output(d, "Bob: lunch?")
    assert g.evaluate("send_direct_message", {"recipient": "Bob", "body": "yes"}).allowed


def test_mixed_needs_the_request_literal_in_the_r_component():
    args = {"email": "a@org.com", "permission": "r", "file_id": "7"}
    assert not _gov().evaluate("share_file", args).allowed           # no request: no grant
    assert not _gov("Share the report.").evaluate("share_file", args).allowed  # 'r' not in request
    g = _gov("Share file 7 with permission r.")
    d = g.evaluate("share_file", args)
    assert d.allowed and d.grant_source == "L2"


def test_value_outside_o_and_request_is_refused():
    g = _gov("Reply to whoever wrote.")
    assert not g.evaluate("send_direct_message", {"recipient": "Mallory", "body": "x"}).allowed
    assert not g.evaluate("share_file", {"email": "evil@x.com", "permission": "r", "file_id": "7"}).allowed


def test_quota_is_spent():
    g = _gov()  # default quota 1
    assert _spend(g, "send_direct_message", {"recipient": "Alice", "body": "1"}).allowed
    assert not g.evaluate("send_direct_message", {"recipient": "Bob", "body": "2"}).allowed
    g = _gov(grant_quotas={"send_direct_message": 2})
    assert _spend(g, "send_direct_message", {"recipient": "Alice", "body": "1"}).allowed
    assert _spend(g, "send_direct_message", {"recipient": "Bob", "body": "2"}).allowed
    assert not g.evaluate("send_direct_message", {"recipient": "Alice", "body": "3"}).allowed


def test_floor_and_access_control_unchanged():
    g = _gov()
    d = g.evaluate("get_secret", {})
    g.register_output(d, "passport 123456789")
    assert not g.evaluate("send_direct_message", {"recipient": "Bob", "body": "x"}).allowed
    assert not _gov().evaluate("update_password", {"password": "hunter2"}).allowed
