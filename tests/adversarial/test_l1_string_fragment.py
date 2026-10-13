"""revision 4: an L1 string value is a word-bounded fragment of the request (case-
insensitive), never a typed literal; uses are counted per distinct value."""
from __future__ import annotations

from axor_core.governor import ToolCallGovernor
from axor_core.kernel.request_literals import is_request_fragment, request_fragments

REQ = ("Invite Dora to Slack and add her to the general channel; tell Bob Smith the news. "
       "Her email is dora@gmail.com, meeting on 2024-05-19, file notes.txt.")


def test_fragment_rule():
    assert is_request_fragment(REQ, "Dora") and is_request_fragment(REQ, "dora")       # case-insensitive
    assert is_request_fragment(REQ, "bob   smith")                                       # whitespace collapsed
    assert is_request_fragment(REQ, "general channel")
    assert not is_request_fragment(REQ, "Dor")                                           # not on a word boundary
    assert not is_request_fragment(REQ, "eneral")
    assert not is_request_fragment(REQ, "Eve")                                           # not in the request
    assert not is_request_fragment(REQ, "")
    # a typed literal never fills a string slot, even when it is in the request
    for typed in ("dora@gmail.com", "2024-05-19", "notes.txt", "19"):
        assert not is_request_fragment(REQ, typed)


def test_fragment_count_excludes_typed():
    frags = request_fragments("pay 12 to Bob", max_words=None)
    assert "bob" in frags and "to bob" in frags and "12" not in frags


GA = {"send_direct_message": {"recipient": "string"},
      "add_user_to_channel": {"user": "string", "channel": "string"},
      "invite_user_to_slack": {"user": "string", "user_email": "email"},
      "update_password": {"password": "string"}}


def _gov(**kw):
    g = ToolCallGovernor(grant_args=GA, egress_sinks={"send_direct_message", "add_user_to_channel",
                                                      "invite_user_to_slack"},
                         integrity_sinks={"update_password"},
                         driving_args={"send_direct_message": ["recipient"], "update_password": ["password"],
                                       "add_user_to_channel": ["user", "channel"],
                                       "invite_user_to_slack": ["user", "user_email"]}, **kw)
    g.register_task(REQ)
    g.grants.from_request(REQ)
    return g


def _run(g, tool, args):
    d = g.evaluate(tool, args)
    if d.allowed and d.grant_id is not None:
        g.grant_store.commit(g.spend(d, args))
    return d.allowed


def test_unquoted_names_now_grant():
    g = _gov()
    assert _run(g, "invite_user_to_slack", {"user": "Dora", "user_email": "dora@gmail.com"})
    assert _run(g, "add_user_to_channel", {"user": "dora", "channel": "general"})
    assert not _run(g, "send_direct_message", {"recipient": "Eve"})


def test_typed_value_not_in_string_slot():
    g = _gov()
    assert not _run(g, "send_direct_message", {"recipient": "dora@gmail.com"})
    assert not _run(g, "send_direct_message", {"recipient": "2024-05-19"})


def test_uses_per_distinct_value():
    g = _gov()
    assert _run(g, "send_direct_message", {"recipient": "Dora"})
    assert _run(g, "send_direct_message", {"recipient": "Bob Smith"})    # another value: its own use
    assert not _run(g, "send_direct_message", {"recipient": "dora"})     # same value again: quota 1 spent


def test_access_control_password_any_word_refused():
    g = _gov(access_control_sinks={"update_password"})
    for w in ("Dora", "general", "the news", "Slack"):
        assert not g.evaluate("update_password", {"password": w}).allowed
