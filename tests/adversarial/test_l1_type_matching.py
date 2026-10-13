"""L1 type matching: a request literal binds to a grant argument only of its own kind."""
from __future__ import annotations

from axor_core.governor import ToolCallGovernor
from axor_core.kernel.request_literals import literal_values

REQ = ("Create an event 'Lunch' at 12:00 on 2024-05-19, invite 'sarah.connor@gmail.com', "
       "move event 42, pay 12.00 to GB29NWBK60161331926819, see www.x.com, file 'bill-2023.txt'")


def test_literal_types_by_syntax():
    assert literal_values(REQ, "email") == ["sarah.connor@gmail.com"]
    assert literal_values(REQ, "string") == ["Lunch", "bill-2023.txt"]  # quoted, untyped
    assert literal_values(REQ, "datetime") == ["2024-05-19"]
    assert literal_values(REQ, "id") == ["42"]                            # bare integers only
    assert "12.00" in literal_values(REQ, "money") and "12.00" not in literal_values(REQ, "id")
    assert literal_values(REQ, "url") == ["www.x.com"]
    assert literal_values(REQ, "path") == ["bill-2023.txt"]
    # a clock time is not an integer; a quoted e-mail is not a string
    assert "12" not in literal_values(REQ, "id") and "00" not in literal_values(REQ, "number")
    assert "sarah.connor@gmail.com" not in literal_values(REQ, "string")


GA = {
    "create_calendar_event": {"start_time": "datetime", "location": "string?", "participants": "email?"},
    "cancel_calendar_event": {"event_id": "id"},
    "share_file": {"file_id": "id", "email": "email", "permission": "string"},
}


def _gov():
    g = ToolCallGovernor(grant_args=GA, egress_sinks={"create_calendar_event", "share_file"},
                         integrity_sinks={"cancel_calendar_event"},
                         driving_args={"create_calendar_event": ["participants"], "share_file": ["email"],
                                       "cancel_calendar_event": ["event_id"]})
    g.register_task(REQ)
    g.grants.from_request(REQ)
    return g


def test_no_cross_type_binding():
    g = _gov()
    # an e-mail or quoted text never fills a time slot; a date or e-mail never an id
    assert not g.evaluate("create_calendar_event", {"start_time": "sarah.connor@gmail.com"}).allowed
    assert not g.evaluate("create_calendar_event", {"start_time": "Lunch"}).allowed
    assert not g.evaluate("cancel_calendar_event", {"event_id": "2024-05-19"}).allowed
    assert not g.evaluate("cancel_calendar_event", {"event_id": "sarah.connor@gmail.com"}).allowed
    assert not g.evaluate("cancel_calendar_event", {"event_id": "12.00"}).allowed
    assert not g.evaluate("share_file", {"file_id": "42", "email": "sarah.connor@gmail.com",
                                         "permission": "sarah.connor@gmail.com"}).allowed


def test_same_type_binding_still_works():
    g = _gov()
    assert g.evaluate("cancel_calendar_event", {"event_id": "42"}).allowed
    assert g.evaluate("create_calendar_event", {"start_time": "2024-05-19", "location": "Lunch",
                                                "participants": ["sarah.connor@gmail.com"]}).allowed
    assert g.evaluate("share_file", {"file_id": "42", "email": "sarah.connor@gmail.com",
                                     "permission": "Lunch"}).allowed
