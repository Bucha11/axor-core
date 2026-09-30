"""Pure boundary-gate predicates for inter-node messages (spec v2 Ch.4 §2).

A boundary is an edge between two nodes; the gate runs at the SENDER on the
send (message-as-sink) and at the RECEIVER on the receipt (message-as-source).
Two independent local evaluations — no distributed transaction, no shared
state. The verdict-deciding code lives here, in the pure kernel, shared by the
runtime (axor_core.node.messaging) and replay (Rule 0).

Intra-federation semantics (spec v2 Ch.1 §1): labels are DATA — they travel
with values and are folded intact at the receiver. Being internal grants no
permissions; it only preserves label fidelity. Hop count cleans nothing.
"""
from __future__ import annotations

from axor_core.contracts.degradation import DegradationLevel
from axor_core.policy.gates import GateDecision
from axor_core.taint.causal_root import CausalRoot

EDGE_DELEGATION = "delegation"
EDGE_LATERAL = "lateral"
EDGE_PEER = "peer"
EDGE_KINDS = frozenset({EDGE_DELEGATION, EDGE_LATERAL, EDGE_PEER})


def evaluate_message_send(
    sender_level: DegradationLevel,
    root: CausalRoot,
    edge_kind: str,
    *,
    peer_declared: bool = False,
) -> GateDecision | None:
    """Sender-side gate: the message is a sink. None = pass.

    - Unknown edge kind: deny (fail closed, same posture as an unclassified
      sink).
    - LOCKED/TERMINAL sender: deny — a locked node admits only read/escalate;
      sending a value laterally would be an effect.
    - PEER edge to an undeclared peer: deny (spec v2 Ch.1 §2 — undeclared =
      L0 = untrusted export destination; establishing the channel at all
      requires a declaration). Declared-peer gating (levels, discounts) is the
      inter-federation ladder — layered on top of this predicate, never
      replacing it.
    - Intra edges (delegation/lateral) carry labels intact; taint does NOT
      deny the send — it rides with the value and denies at the next sink
      that matters. That is carried-taint containment, not laundering.
    """
    if edge_kind not in EDGE_KINDS:
        return GateDecision(
            reason=f"message: unknown edge kind '{edge_kind}' (undeclared = denied)",
            category="message_gate",
        )
    if sender_level >= DegradationLevel.LOCKED:
        return GateDecision(
            reason=(
                f"degradation: level {sender_level.name} admits no message sends"
            ),
            category="degradation",
        )
    if edge_kind == EDGE_PEER and not peer_declared:
        return GateDecision(
            reason="message: peer edge to an undeclared peer (undeclared = L0, denied)",
            category="message_gate",
        )
    return None


def fold_carried_root(root: CausalRoot | None) -> CausalRoot:
    """Receiver-side fold of a carried root, intra-federation: labels are data
    and arrive intact — the carried root IS the local root. A missing root
    fails closed to an untrusted cross-process re-mint (a message that lost
    its labels must not arrive clean)."""
    if root is None:
        return CausalRoot.cross_process_in()
    return root
