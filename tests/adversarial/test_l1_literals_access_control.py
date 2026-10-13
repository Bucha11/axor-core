"""L1 literal discipline and the access-control class.

* An untyped (``string``) L1 literal comes only from a fragment the user set apart —
  quotes or backticks; typed literals (IBAN, e-mail, amount, id, path) are unchanged.
* A credential change (an access-control sink) never gets an L1 or L2 grant: its
  authority is L3 (structured) or L4 (confirmation) only.
"""
from __future__ import annotations

import pytest

from axor_core.config import GovernanceConfig
from axor_core.governor import ToolCallGovernor
from axor_core.kernel.confirm import ScriptedConfirmer
from axor_core.kernel.request_literals import literal_values

REQ = ("Can you please pay the bill 'bill-december-2023.txt' for me? Also update my password "
       "to `Hunter2!` and send 50 to GB29NWBK60161331926819, cc bob@x.com, Alice")


def test_string_only_from_quotes_and_backticks():
    assert literal_values(REQ, "string") == ["bill-december-2023.txt", "Hunter2!"]
    assert "Can" not in literal_values(REQ, "string") and "Alice" not in literal_values(REQ, "string")


def test_typed_literals_unchanged():
    assert literal_values(REQ, "iban") == ["GB29NWBK60161331926819"]
    assert literal_values(REQ, "email") == ["bob@x.com"]
    assert "50" in literal_values(REQ, "money")
    assert "bill-december-2023.txt" in literal_values("pay bill-december-2023.txt", "path")


GA = {"update_password": {"password": "string"}, "send_money": {"recipient": "iban"}}


def _gov(**kw):
    g = ToolCallGovernor(grant_args=GA, egress_sinks={"send_money"}, integrity_sinks={"update_password"},
                         driving_args={"send_money": ["recipient"], "update_password": ["password"]}, **kw)
    g.register_task(REQ)
    g.grants.from_request(REQ)
    return g


def test_without_access_control_any_request_word_is_a_password():
    """revision 4: a string arg takes any word-bounded request fragment — so without
    the access-control class a password could be any word of the request. That is why
    a credential change must be an access-control sink (next tests)."""
    g = _gov()
    assert g.evaluate("update_password", {"password": "can"}).allowed


def test_access_control_sink_gets_no_l1():
    g = _gov(access_control_sinks={"update_password"})
    for word in ("can", "Alice", "Hunter2!", "bill", "please pay"):
        assert not g.evaluate("update_password", {"password": word}).allowed
    assert not any(x.sink == "update_password" for x in g.grants.issued)
    d = g.evaluate("update_password", {"password": "Hunter2!"})
    assert not d.allowed and d.category == "grant_gate"
    # other sinks keep their L1 grants
    assert g.evaluate("send_money", {"recipient": "GB29NWBK60161331926819"}).allowed


def test_access_control_l4_and_l3():
    g = _gov(access_control_sinks={"update_password"}, confirmer=ScriptedConfirmer(lambda a, c: True))
    d = g.evaluate("update_password", {"password": "Hunter2!"})
    assert d.allowed and d.grant_source == "L4"
    g2 = _gov(access_control_sinks={"update_password"})
    g2.grants.from_structured("update_password", {"password": "Hunter2!"})
    assert g2.evaluate("update_password", {"password": "Hunter2!"}).allowed


def test_access_control_refuses_templates_and_quotas():
    with pytest.raises(ValueError):
        ToolCallGovernor(grant_args=GA, access_control_sinks={"update_password"},
                         grant_templates={"update_password": [{"password": "x"}]})
    with pytest.raises(ValueError):
        ToolCallGovernor(grant_args=GA, access_control_sinks={"update_password"},
                         grant_quotas={"update_password": 2})
    with pytest.raises(ValueError):
        ToolCallGovernor(grant_args=GA, access_control_sinks={"reset_token"})
    cfg = GovernanceConfig.from_dict({"grant_args": GA, "access_control_sinks": ["update_password"]})
    assert cfg.as_governor_kwargs()["access_control_sinks"] == {"update_password"}
    # no grants declared: nothing to mint, the declaration is accepted
    GovernanceConfig.from_dict({"access_control_sinks": ["update_password"]})
