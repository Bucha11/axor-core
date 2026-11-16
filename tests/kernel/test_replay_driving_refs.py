"""A counterfactual that declares a tool's driving args narrows the replay to
them on records that carry ``arg_refs`` — as the live gate narrows its check
(``driving_subset``). An email whose BODY quotes an untrusted read and whose
recipient is clean passes under "only `to` decides"; a recipient that came
from the read is still refused."""
from __future__ import annotations

import json

from axor_core.kernel.events import event_from_json_line
from axor_core.kernel.replay import KernelConfig, replay

WEB = {"sources": ["web"], "sensitive": False}


def _line(seq: int, kind: str, verdict: str | None, **payload: object) -> str:
    return json.dumps({"schema_version": "1.0", "seq": seq, "node_id": "n", "kind": kind,
                       "ts": "2026-11-16T00:00:00Z", "causal_root": None, "gate": None,
                       "verdict": verdict, "payload": payload})


def _events(refs: dict[str, str]) -> list:
    return [event_from_json_line(x) for x in (
        _line(0, "tool_call", "pass", tool="read_inbox", args={}),
        _line(1, "tool_result", None, tool="read_inbox", value_ref="v_mail", root=WEB),
        _line(2, "tool_call", "deny", tool="send_email",
              args={"to": "boss@corp.example", "body": "a summary"}, arg_refs=refs,
              driving_root=WEB),
    )]


def _send(events: list, **cfg: object) -> str:
    config = KernelConfig(egress_sinks=frozenset({"send_email"}), **cfg)
    step = next(s for s in replay(events, config).steps
                if s.event.payload.get("tool") == "send_email")
    return step.reevaluated_verdict.value


def test_the_body_does_not_decide_when_only_to_does() -> None:
    body = _events({"body": "v_mail"})
    assert _send(body) == "deny"                            # every arg decides
    assert _send(body, driving_args={"send_email": frozenset({"to"})}) == "pass"


def test_a_recipient_from_the_read_is_still_refused() -> None:
    both = _events({"body": "v_mail", "to": "v_mail"})
    assert _send(both, driving_args={"send_email": frozenset({"to"})}) == "deny"


def test_declared_args_absent_from_the_call_fall_back_to_every_ref() -> None:
    body = _events({"body": "v_mail"})
    assert _send(body, driving_args={"send_email": frozenset({"cc"})}) == "deny"
