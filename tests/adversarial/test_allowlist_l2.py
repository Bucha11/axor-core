"""Operator enum allowlists on a grant sink's driving grant args compile into an L2
template (same sink, same arg, set = allowlist, unlimited uses). Supersession, L1, L4,
the floor and access control are unchanged."""
from __future__ import annotations

from axor_core.governor import ToolCallGovernor
from axor_core.kernel.grants import compile_allowlist_templates
from axor_core.policy.value_policy import ValuePredicate

IBAN_OK = "GB29NWBK60161331926819"
IBAN_BAD = "US133000000121212121212"
REQ = "Pay the bill in the file."

BASE = dict(
    untrusted_sources={"read_file"},
    sensitive_sources={"get_user_info"},
    egress_sinks={"send_money", "send_email"},
    access_control_sinks={"update_password"},
    integrity_sinks={"update_password"},
    driving_args={"send_money": ["recipient"], "send_email": ["recipients", "cc"],
                  "update_password": ["password"]},
    grant_args={"send_money": {"recipient": "iban", "amount": "money"},
                "send_email": {"recipients": "email", "cc": "email?"},
                "update_password": {"password": "string"}},
    integrity_default="context",
    integrity_origins="request-only",
)


def _vp(**allow):
    return {tool: [ValuePredicate(arg=a, kind="enum", allowed=frozenset(v)) for a, v in args.items()]
            for tool, args in allow.items()}


def _gov(value_policies=None, **kw):
    g = ToolCallGovernor(**{**BASE, "value_policies": value_policies or {}, **kw})
    g.register_task(REQ)
    g.grants.from_request(REQ)
    return g


def _read(g, text):
    d = g.evaluate("read_file", {"file_path": "bill.txt"})
    g.register_output(d, text)


def test_without_allowlist_read_recipient_has_no_grant():
    g = _gov()
    _read(g, f"Pay 98.70 to {IBAN_OK}")
    assert not g.evaluate("send_money", {"recipient": IBAN_OK, "amount": 98.7}).allowed


def test_allowlisted_value_passes_under_grant():
    g = _gov(_vp(send_money={"recipient": [IBAN_OK]}))
    _read(g, f"Pay 98.70 to {IBAN_OK}")
    d = g.evaluate("send_money", {"recipient": IBAN_OK, "amount": 98.7})
    assert d.allowed and d.grant_source == "L2"


def test_value_outside_allowlist_is_refused():
    g = _gov(_vp(send_money={"recipient": [IBAN_OK]}))
    _read(g, f"Pay 98.70 to {IBAN_BAD}")
    assert not g.evaluate("send_money", {"recipient": IBAN_BAD, "amount": 98.7}).allowed


def test_one_entry_per_sink_and_optional_kept():
    g = _gov(_vp(send_email={"recipients": ["a@x.com"], "cc": ["b@x.com"]}))
    _read(g, "write to a@x.com")
    assert g.evaluate("send_email", {"recipients": ["a@x.com"], "body": "x"}).allowed  # cc absent
    assert g.evaluate("send_email", {"recipients": ["a@x.com"], "cc": ["b@x.com"], "body": "x"}).allowed
    assert not g.evaluate("send_email", {"recipients": ["a@x.com"], "cc": ["evil@y.com"], "body": "x"}).allowed


def test_allowlist_does_not_lift_the_floor():
    g = _gov(_vp(send_money={"recipient": [IBAN_OK]}))
    d = g.evaluate("get_user_info", {})
    g.register_output(d, "passport 123456789")
    assert not g.evaluate("send_money", {"recipient": IBAN_OK, "amount": 1}).allowed


def test_allowlist_does_not_lift_access_control():
    vp = _vp(update_password={"password": ["hunter2"]})
    assert compile_allowlist_templates(vp, BASE["grant_args"], BASE["driving_args"],
                                       BASE["access_control_sinks"]) == {}
    g = _gov(vp)
    assert not g.evaluate("update_password", {"password": "hunter2"}).allowed


def test_compile_skips_non_driving_and_non_enum():
    vp = {"send_money": [ValuePredicate(arg="amount", kind="enum", allowed=frozenset({"1"})),
                         ValuePredicate(arg="recipient", kind="numeric_range", lo=0, hi=1)]}
    assert compile_allowlist_templates(vp, BASE["grant_args"], BASE["driving_args"]) == {}
