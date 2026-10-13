"""floor_scope: what the confidentiality floor holds while a secret read is outstanding.

addressee (default) holds the egress sinks only; shared_stores adds the stores the
operator declared visible to third parties; all_mutating adds every integrity sink.
"""
from __future__ import annotations

import pytest

from axor_core.config import GovernanceConfig
from axor_core.governor import ToolCallGovernor

BASE = dict(
    sensitive_sources={"get_user_info"},
    egress_sinks={"send_email"},
    integrity_sinks={"append_to_file", "create_file"},
    driving_args={"send_email": ["recipients"], "append_to_file": ["file_id"], "create_file": ["filename"]},
    integrity_default="context",
    integrity_origins="any-trusted",
)
REQ = "append the notes to file 7 and to file 9, create notes.txt, mail bob@x.com"


def _gov(**kw):
    g = ToolCallGovernor(**{**BASE, **kw})
    g.register_task(REQ)
    return g


def _after_secret(g):
    d = g.evaluate("get_user_info", {})
    assert d.allowed
    g.register_output(d, "passport 123456789, card 4111111111111111")
    return g


@pytest.mark.parametrize("scope,shared,expect", [
    ("addressee", None, {"append7": True, "append9": True, "create": True}),
    ("shared_stores", {"append_to_file": {"file_id": ["7"]}}, {"append7": False, "append9": True, "create": True}),
    ("all_mutating", None, {"append7": False, "append9": False, "create": False}),
])
def test_scope_under_outstanding_secret(scope, shared, expect):
    g = _after_secret(_gov(floor_scope=scope, shared_stores=shared))
    got = {
        "append7": g.evaluate("append_to_file", {"file_id": "7", "content": "x"}).allowed,
        "append9": g.evaluate("append_to_file", {"file_id": "9", "content": "x"}).allowed,
        "create": g.evaluate("create_file", {"filename": "notes.txt", "content": "x"}).allowed,
    }
    assert got == expect
    # the egress sink is held in every scope
    assert not g.evaluate("send_email", {"recipients": ["bob@x.com"], "body": "x"}).allowed


@pytest.mark.parametrize("scope", ["addressee", "shared_stores", "all_mutating"])
def test_no_secret_no_floor(scope):
    shared = {"append_to_file": {"file_id": ["7"]}} if scope == "shared_stores" else None
    g = _gov(floor_scope=scope, shared_stores=shared)
    assert g.evaluate("append_to_file", {"file_id": "7", "content": "x"}).allowed
    assert g.evaluate("send_email", {"recipients": ["bob@x.com"], "body": "x"}).allowed


def test_always_store_and_denial_names_the_scope():
    g = _after_secret(_gov(floor_scope="shared_stores", shared_stores={"create_file": "always"}))
    d = g.evaluate("create_file", {"filename": "notes.txt", "content": "x"})
    assert not d.allowed and "floor's scope" in d.reason


def test_config_parsing_fails_closed():
    ok = GovernanceConfig.from_dict({"floor_scope": "shared_stores",
                                     "shared_stores": {"append_to_file": {"file_id": ["7"]}}})
    kw = ok.as_governor_kwargs()
    assert kw["floor_scope"] == "shared_stores" and kw["shared_stores"]["append_to_file"]["file_id"] == ["7"]
    for bad in ({"floor_scope": "everything"},
                {"floor_scope": "shared_stores", "shared_stores": {"x": "sometimes"}},
                {"shared_stores": {"x": "always"}}):
        with pytest.raises(ValueError):
            GovernanceConfig.from_dict(bad)
    with pytest.raises(ValueError):
        ToolCallGovernor(floor_scope="nope")
