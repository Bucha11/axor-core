"""Versioned event schema — the trace contract shared by every consumer.

One event per JSONL line. Single source of truth for the runtime adapter, the
proxy recorder, backend storage, control-plane telemetry, and replay. Frontend
TS types are generated from these dataclasses (JSON Schema -> ts) on the
platform side — never hand-maintained.

Stdlib only: axor-core ships with zero required dependencies, so the schema is
plain frozen dataclasses with explicit dict/JSON codecs. Platform consumers
that want validation wrap these in ``pydantic.TypeAdapter`` — the types here
are the single source either way.

Payload conventions by kind (the schema keeps ``payload`` open; producers and
replay agree on these keys):

- TOOL_CALL: ``{"tool", "args", "arg_refs": {arg: value_ref}, "normalized":
  {NormalizedIntent fields}, "driving_root": {"sources": [...], "sensitive"},
  "arg_provenance": {arg: {"sources": [...], "sensitive"}} — OPTIONAL, and NOT
  ``arg_refs``: refs are opaque value ids the fold looks up in tainted_refs and
  the subgraph walks as edges, while this is a per-argument taint summary a
  producer with no ref vocabulary (the synchronous governor) can still record,
  "floor_active": bool}`` — ``verdict`` on the event is the recorded overall
  gate verdict for this call; ``gate`` is the denying gate's category if denied.
- TOOL_RESULT: ``{"tool", "status", "value_ref", "root": {"sources": [...],
  "sensitive"}}`` — registers ``value_ref`` in the taint fold.
- FAULT_INJECTED: ``{"tool", "mode", "canary"}``
- CLAIM: ``{"text", "claims"}`` (hosted persistence may hash ``text``)
- DENIAL: ``{"reason", "category"}`` with ``gate`` set
- FACT: the :class:`Fact` fields (see ``fact_to_payload``)
- CONTEXT_EXCISION: ``{"refs": [...], "provenance": {ref: "runtime"|
  "operator_config"}, "operator", "reason"}`` — refs stop influencing future
  derivations; already-derived values keep their taint (spec 8.2.1)
- OPERATOR_INTERVENTION / INJECTION_CONSUMED / STATE_APPLIED / HEARTBEAT:
  control-plane bookkeeping (protocol note sections 4-5)

Multi-agent kinds (spec v2 Ch.4 — labels ride in message envelopes; structure
derives from traced spawn events, never from self-report):

- NODE_SPAWNED: ``{"child_id", "parent_id", "depth", "edge_kind":
  "delegation"}`` — emitted at the PARENT; replay reconstructs the tree shape
  from these, the platform derives topology from the same events.
- MESSAGE_SENT: ``{"to", "edge_kind": "delegation"|"lateral"|"peer",
  "msg_id", "value_ref", "carried": {"root": {"sources": [...],
  "sensitive"}}}`` — the message is a sink at the sender; ``verdict`` is the
  sender-side gate verdict, ``gate`` the denying category. A DENY here is
  containment at the source (spec v2 Ch.2).
- MESSAGE_RECEIVED: ``{"from", "edge_kind", "msg_id", "value_ref",
  "carried": {"root": ...}}`` — the message is a source at the receiver; the
  fold registers ``value_ref`` under the CARRIED root (labels travel with
  values, spec v2 Ch.1 §1 — intra-federation labels are data, not claims).
  ``msg_id`` stitches cross-node causality (no global clock, Ch.4 §5).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum

from axor_core.kernel.errors import SchemaVersionError

SCHEMA_VERSION = "1.0"


class EventKind(str, Enum):
    INTENT = "intent"
    GATE_EVAL = "gate_eval"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FAULT_INJECTED = "fault_injected"
    CLAIM = "claim"
    DENIAL = "denial"
    FACT = "fact"  # feeds the degradation recompute (incl. operator_attestation)
    CONTEXT_EXCISION = "context_excision"  # spec 8.2.1; folded deterministically
    OPERATOR_INTERVENTION = "operator_intervention"  # marks the run `intervened`
    STATE_APPLIED = "state_applied"  # desired-state version ack
    INJECTION_CONSUMED = "injection_consumed"
    INJECTION_REFUSED = "injection_refused"
    EXCISION_REFUSED = "excision_refused"
    HEARTBEAT = "heartbeat"
    # Multi-agent (spec v2 Ch.4). Additive; absent from any size-1 trace.
    NODE_SPAWNED = "node_spawned"
    MESSAGE_SENT = "message_sent"
    MESSAGE_RECEIVED = "message_received"


class Verdict(str, Enum):
    PASS = "pass"
    DENY = "deny"


@dataclass(frozen=True)
class Event:
    """A single trace line. Append-only; never mutated after emission.

    ``ts`` is ISO 8601 and informational — ordering is by ``seq`` (monotonic
    per node). ``causal_root`` is the value-ref this event is about, when any.
    """

    seq: int
    node_id: str
    kind: EventKind
    ts: str
    schema_version: str = SCHEMA_VERSION
    causal_root: str | None = None
    gate: str | None = None
    verdict: Verdict | None = None
    payload: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Fact:
    """Degradation-machine input. Attestations are facts too (spec 8.1.1).

    ``severity`` indexes :class:`~axor_core.contracts.degradation.DegradationLevel`
    (0=NORMAL .. 4=TERMINAL). ``covers`` lists fact_ids an attestation spans;
    ``revokes`` names an attestation fact_id being revoked.
    """

    fact_id: str
    fact_type: str
    severity: int = 0
    covers: tuple[str, ...] = ()
    revokes: str | None = None
    operator: str | None = None
    reason: str | None = None
    sig: str | None = None  # ed25519; required for operator-originated facts


def event_to_json_line(event: Event) -> str:
    d = asdict(event)
    d["kind"] = event.kind.value
    if event.verdict is not None:
        d["verdict"] = event.verdict.value
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def event_from_json_line(line: str) -> Event:
    d = json.loads(line)
    version = d.get("schema_version", "")
    if version.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
        raise SchemaVersionError(
            f"event schema {version!r} has unknown major; kernel is {SCHEMA_VERSION}"
        )
    return Event(
        seq=d["seq"],
        node_id=d["node_id"],
        kind=EventKind(d["kind"]),
        ts=d["ts"],
        schema_version=version,
        causal_root=d.get("causal_root"),
        gate=d.get("gate"),
        verdict=Verdict(d["verdict"]) if d.get("verdict") else None,
        payload=d.get("payload") or {},
    )


def fact_to_payload(fact: Fact) -> dict:
    return asdict(fact)


def fact_from_payload(payload: dict) -> Fact:
    return Fact(
        fact_id=payload["fact_id"],
        fact_type=payload["fact_type"],
        severity=int(payload.get("severity", 0)),
        covers=tuple(payload.get("covers", ())),
        revokes=payload.get("revokes"),
        operator=payload.get("operator"),
        reason=payload.get("reason"),
        sig=payload.get("sig"),
    )
