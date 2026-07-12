"""The causal subgraph — the multi-agent EvidenceCase core (spec v2 Ch.3).

``causal_subgraph`` walks causal-root provenance BACKWARD from an anchored
claim/denial and returns the minimal subgraph whose events causally contribute
to the discrepancy — the causes, not the org chart. A 40-node tree with a
3-node causal chain yields a 3-node case.

Pure kernel (Rule 0): derived from the trace deterministically, computed on
case open, never stored (decision v2-12). A size-1 result — the anchor as the
only node, no edges — IS the v0.13 case; renderers branch on subgraph size.

Roles per node (one node can hold several):
  origin     — a fault landed here
  conduit    — propagated through (taint carried, not laundered)
  container  — denied propagation here
  anchor     — the claim reached a consequence here
"""
from __future__ import annotations

from collections.abc import Sequence

from axor_core.kernel.events import Event, EventKind, Verdict

ROLE_ORIGIN = "origin"
ROLE_CONDUIT = "conduit"
ROLE_CONTAINER = "container"
ROLE_ANCHOR = "anchor"


def causal_subgraph(
    events: Sequence[Event], anchor_node: str, anchor_seq: int
) -> dict:
    """Backward provenance walk from the anchor event.

    The anchor is where the discrepancy surfaced — an export attempt, a claim,
    a returned answer (one case per consequence, decision v2-10). Returns a
    JSON-able dict; raises ValueError when the anchor event does not exist.
    """
    by_node: dict[str, list[Event]] = {}
    for e in events:
        by_node.setdefault(e.node_id, []).append(e)
    for seq_events in by_node.values():
        seq_events.sort(key=lambda e: e.seq)

    anchor_ev = next(
        (e for e in by_node.get(anchor_node, ())
         if e.seq == anchor_seq), None,
    )
    if anchor_ev is None:
        raise ValueError(f"no event seq={anchor_seq} at node {anchor_node!r}")

    # msg_id -> (sent event, received event): the cross-node stitch (Ch.4 §5).
    sends: dict[str, Event] = {}
    receives: dict[str, Event] = {}
    for e in events:
        mid = e.payload.get("msg_id")
        if not mid:
            continue
        if e.kind is EventKind.MESSAGE_SENT:
            sends[str(mid)] = e
        elif e.kind is EventKind.MESSAGE_RECEIVED:
            receives[str(mid)] = e

    roles: dict[str, set[str]] = {anchor_node: {ROLE_ANCHOR}}
    edges: list[dict] = []
    contained_at: list[dict] = []
    fault_origin: dict | None = None
    case_seqs: dict[str, set[int]] = {anchor_node: {anchor_seq}}

    # Containment at the anchor itself: a denied consequence is the positive-
    # space case (verdict CONTAINED, decision v2-11).
    if anchor_ev.verdict is Verdict.DENY:
        roles[anchor_node].add(ROLE_CONTAINER)
        contained_at.append({
            "from": anchor_node,
            "to": str(anchor_ev.payload.get("tool")
                      or anchor_ev.payload.get("to") or "export"),
            "kind": "export" if anchor_ev.kind is EventKind.TOOL_CALL else "message",
            "gate": anchor_ev.gate,
            "seq": anchor_ev.seq,
        })

    def driving_refs(e: Event) -> list[str]:
        arg_refs = e.payload.get("arg_refs") or {}
        if arg_refs:
            return [str(r) for r in arg_refs.values()]
        ref = e.payload.get("value_ref") or e.causal_root
        return [str(ref)] if ref else []

    # Frontier of (node, ref) pairs still to be explained.
    frontier: list[tuple[str, str]] = [
        (anchor_node, r) for r in driving_refs(anchor_ev)
    ]
    # For a claim/denial with no refs, fall back to the node's last received
    # or derived value — the claim was assembled from what arrived.
    if not frontier:
        for e in reversed(by_node[anchor_node]):
            if e.seq >= anchor_seq:
                continue
            if e.kind in (EventKind.MESSAGE_RECEIVED, EventKind.TOOL_RESULT):
                ref = e.payload.get("value_ref") or e.causal_root
                if ref:
                    frontier.append((anchor_node, str(ref)))
                    break

    seen: set[tuple[str, str]] = set()
    while frontier:
        node, ref = frontier.pop()
        if (node, ref) in seen:
            continue
        seen.add((node, ref))
        node_events = by_node.get(node, [])

        # Producer of `ref` at `node`: latest registration before it was used.
        producer: Event | None = None
        for e in reversed(node_events):
            if e.kind in (EventKind.MESSAGE_RECEIVED, EventKind.TOOL_RESULT) and (
                str(e.payload.get("value_ref") or e.causal_root or "") == ref
            ):
                producer = e
                break
        if producer is None:
            continue
        case_seqs.setdefault(node, set()).add(producer.seq)

        if producer.kind is EventKind.MESSAGE_RECEIVED:
            mid = str(producer.payload.get("msg_id") or "")
            sent = sends.get(mid)
            sender = str(producer.payload.get("from") or
                         (sent.node_id if sent else ""))
            if sender:
                roles.setdefault(sender, set()).add(ROLE_CONDUIT)
                edge = {
                    "from": sender,
                    "to": node,
                    "kind": str(producer.payload.get("edge_kind", "lateral")),
                    "carried": producer.payload.get("carried") or {},
                    "gate_verdict": (sent.verdict.value
                                     if sent and sent.verdict else None),
                    "msg_id": mid or None,
                }
                if edge not in edges:
                    edges.append(edge)
                # Continue the walk at the sender.
                frontier.append((sender, ref))
                if sent is not None:
                    case_seqs.setdefault(sender, set()).add(sent.seq)
        elif producer.kind is EventKind.TOOL_RESULT:
            tool = producer.payload.get("tool")
            # A fault injected into this tool at this node = the origin.
            for e in node_events:
                if (e.kind is EventKind.FAULT_INJECTED
                        and e.payload.get("tool") == tool
                        and e.seq <= producer.seq):
                    roles.setdefault(node, set()).add(ROLE_ORIGIN)
                    if fault_origin is None:
                        fault_origin = {"node_id": node, "seq": e.seq}
                    case_seqs.setdefault(node, set()).add(e.seq)
            # The producing call's own inputs continue the chain (derivation
            # stays intra-node; only message hops become subgraph edges).
            for e in reversed(node_events):
                if (e.kind is EventKind.TOOL_CALL
                        and e.seq < producer.seq
                        and e.payload.get("tool") == tool):
                    for up_ref in (e.payload.get("arg_refs") or {}).values():
                        frontier.append((node, str(up_ref)))
                    case_seqs.setdefault(node, set()).add(e.seq)
                    break

    # A node that only relayed (roles == {conduit}) stays conduit; the anchor
    # never doubles as conduit of its own claim.
    roles.get(anchor_node, set()).discard(ROLE_CONDUIT)

    nodes = [
        {"node_id": nid, "roles": sorted(rs),
         "seqs": sorted(case_seqs.get(nid, ()))}
        for nid, rs in sorted(roles.items())
    ]
    return {
        "anchor": {"node_id": anchor_node, "seq": anchor_seq},
        "nodes": nodes,
        "edges": edges,
        "fault_origin": fault_origin,
        "contained_at": contained_at or None,
        "federation_scope": (
            "inter" if any(e["kind"] == "peer" for e in edges) else "intra"
        ),
    }
