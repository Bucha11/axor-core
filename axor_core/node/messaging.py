"""Lateral / delegation message transport with labels in the envelope.

Runtime orchestration around the pure kernel predicates (spec v2 Ch.4):
labels ride in the message envelope; the gate runs at the sender on the send
and at the receiver on the receipt — two local evaluations, never a joint
computation. Networked intra edges are signed with federation keys —
signature proves integrity and origin, it does NOT trigger re-derivation
(same authority on both ends, spec v2 Ch.1 §1). Intra lateral edges go
direct, never through the plane (decision v2-5).

Nothing in this module decides a verdict: that is
:func:`axor_core.kernel.messaging.evaluate_message_send` and
:func:`axor_core.kernel.messaging.fold_carried_root` (Rule 0).
"""
from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace

from axor_core.contracts.degradation import DegradationLevel
from axor_core.federation.ladder import (
    InboundVerdict,
    LabelAssertion,
    PeerDeclaration,
    receive_foreign,
)
from axor_core.federation.signing import Signer, Verifier
from axor_core.kernel.events import Event, EventKind, Verdict
from axor_core.kernel.messaging import (
    EDGE_KINDS,
    EDGE_PEER,
    evaluate_message_send,
    fold_carried_root,
)
from axor_core.policy.gates import GateDecision
from axor_core.taint.causal_root import CausalRoot
from axor_core.taint.engine import TaintEngine
from axor_core.taint.fingerprint import content_fingerprint


class MessageDenied(Exception):
    """Sender-side gate denied the send; the message was never delivered."""

    def __init__(self, decision: GateDecision) -> None:
        super().__init__(decision.reason)
        self.decision = decision


class MessageRejected(Exception):
    """Receiver-side rejection: a signed envelope failed verification. A forged
    or tampered envelope is an attack, not a downgrade — never folded."""


@dataclass(frozen=True)
class MessageEnvelope:
    """A value in transit between two nodes, labels riding with it.

    ``root`` is the value's causal root AT THE SENDER — carried intact across
    intra edges (labels are data inside one federation). ``value_ref`` is the
    stable ref both trace events use, so replay and the causal-subgraph walk
    stitch send→receive without content access.
    """

    msg_id: str
    from_node: str
    to_node: str
    edge_kind: str
    value: object
    root: CausalRoot
    value_ref: str = ""
    signature: bytes = b""
    algorithm: str = ""

    def signed_payload(self) -> bytes:
        """Canonical byte string the federation signature covers. Binds
        identity, edge, value hash and labels — an envelope cannot be detached
        from its value or have its labels stripped without breaking the sig."""
        src = ",".join(sorted(s.value for s in self.root.sources))
        vh = content_fingerprint(self.value)
        return (
            f"{self.msg_id}\x1f{self.from_node}\x1f{self.to_node}"
            f"\x1f{self.edge_kind}\x1f{vh}\x1f{src}\x1f{int(self.root.sensitive)}"
        ).encode()


def make_envelope(
    sender: TaintEngine,
    from_node: str,
    to_node: str,
    edge_kind: str,
    value: object,
    *,
    value_ref: str | None = None,
    signer: Signer | None = None,
) -> MessageEnvelope:
    """Build an envelope for ``value``, deriving its labels from the sender's
    own per-value ledger (the value's ACTUAL provenance — clean stays clean,
    tainted stays tainted; the envelope never edits labels)."""
    env = MessageEnvelope(
        msg_id=f"msg_{uuid.uuid4().hex[:12]}",
        from_node=from_node,
        to_node=to_node,
        edge_kind=edge_kind,
        value=value,
        root=sender.derive_value(value),
        value_ref=value_ref or f"v_{content_fingerprint(value)[:12]}",
    )
    if signer is not None:
        env = replace(
            env,
            signature=signer.sign(env.signed_payload()),
            algorithm=signer.algorithm,
        )
    return env


def _root_payload(root: CausalRoot) -> dict:
    return {
        "sources": sorted(s.value for s in root.sources),
        "sensitive": root.sensitive,
    }


class InMemoryMessageBus:
    """Direct in-process transport for intra-federation edges.

    Transport is irrelevant to semantics (spec v2 Ch.1 §1): the same gate
    pipeline runs for an in-process hop as for a networked one. The bus emits
    kernel-schema trace events on both ends via the ``emit`` callback so the
    platform's topology and the causal-subgraph walk see every hop.
    """

    def __init__(
        self,
        emit: Callable[[Event], None] | None = None,
        *,
        verifier_for: Callable[[str], Verifier | None] | None = None,
        peer_declared: Callable[[str], bool] | None = None,
        peer_declaration_for: Callable[[str], PeerDeclaration | None] | None = None,
    ) -> None:
        self._emit = emit or (lambda e: None)
        self._verifier_for = verifier_for
        # Which foreign peers are DECLARED in operator config (spec v2 Ch.1
        # §3): undeclared = L0 = the send gate fails closed on peer edges.
        # peer_declaration_for supplies the full declaration for the INBOUND
        # ladder (receive_foreign); when it is set, peer_declared defaults to
        # "has a declaration".
        self._declaration_for = peer_declaration_for or (lambda peer: None)
        self._peer_declared = peer_declared or (
            (lambda peer: self._declaration_for(peer) is not None)
            if peer_declaration_for is not None else (lambda peer: False)
        )
        self._nodes: dict[str, tuple[TaintEngine, Callable[[], DegradationLevel]]] = {}
        self._inboxes: dict[str, list[MessageEnvelope]] = {}
        self._seq: dict[str, int] = {}

    def register(
        self,
        node_id: str,
        taint: TaintEngine,
        level_of: Callable[[], DegradationLevel] | None = None,
    ) -> None:
        self._nodes[node_id] = (taint, level_of or (lambda: DegradationLevel.NORMAL))
        self._inboxes.setdefault(node_id, [])

    def _next_seq(self, node_id: str) -> int:
        self._seq[node_id] = self._seq.get(node_id, 0) + 1
        return self._seq[node_id]

    def _event(
        self,
        node_id: str,
        kind: EventKind,
        env: MessageEnvelope,
        verdict: Verdict,
        *,
        peer_field: str,
        gate: str | None = None,
        reason: str | None = None,
    ) -> Event:
        payload = {
            peer_field: env.to_node if peer_field == "to" else env.from_node,
            "edge_kind": env.edge_kind,
            "msg_id": env.msg_id,
            "value_ref": env.value_ref,
            "carried": {"root": _root_payload(env.root)},
        }
        if reason:
            payload["reason"] = reason
        return Event(
            seq=self._next_seq(node_id),
            node_id=node_id,
            kind=kind,
            ts=f"t:{time.time():.6f}",
            causal_root=env.value_ref,
            gate=gate,
            verdict=verdict,
            payload=payload,
        )

    def send(self, env: MessageEnvelope) -> None:
        """Sender-side gate, then delivery. Raises MessageDenied on a gate
        deny (containment at the source — the message is never sent)."""
        if env.edge_kind not in EDGE_KINDS:
            # Still traced: an attempted send over an undeclared edge is an
            # event, not a silent no-op.
            pass
        _, level_of = self._nodes[env.from_node]
        # A peer edge is "declared" when its FOREIGN end is declared in our
        # config — outbound that is the destination, inbound the origin.
        deny = evaluate_message_send(
            level_of(), env.root, env.edge_kind,
            peer_declared=(
                self._peer_declared(env.to_node)
                or self._peer_declared(env.from_node)
            ),
        )
        if deny is not None:
            self._emit(
                self._event(
                    env.from_node, EventKind.MESSAGE_SENT, env, Verdict.DENY,
                    peer_field="to", gate=deny.category, reason=deny.reason,
                )
            )
            raise MessageDenied(deny)
        self._emit(
            self._event(
                env.from_node, EventKind.MESSAGE_SENT, env, Verdict.PASS,
                peer_field="to",
            )
        )
        self._deliver(env)

    def _deliver(self, env: MessageEnvelope) -> None:
        """Receiver side: verify origin (when signed), fold carried labels
        into the receiver's ledger, record the receipt. Gates on USE of the
        value run in the receiver's own pipeline — receipt is a source, not a
        permission."""
        if env.signature and self._verifier_for is not None:
            verifier = self._verifier_for(env.from_node)
            if verifier is None or env.algorithm != verifier.algorithm or (
                not verifier.verify(env.signed_payload(), env.signature)
            ):
                raise MessageRejected(
                    f"forged or tampered envelope from {env.from_node!r}"
                )
        taint, _ = self._nodes[env.to_node]
        evidence: tuple[str, ...] = ()
        if env.edge_kind == EDGE_PEER:
            # INTER-federation inbound (spec v2 Ch.1 §2): the carried root is
            # a CLAIM, not data — re-derive through the trust ladder. The
            # foreign root is kept as an opaque forensic ref in the event
            # payload; the LOCAL root is what registers.
            verdict = self._receive_peer(env)
            folded = verdict.root
            evidence = verdict.evidence
        else:
            # INTRA: labels are data, carried intact. Monotone fold: a value
            # returning to its origin (cycle A→B→A) re-mints the union of what
            # it already carried and what it carries now — the cycle cannot
            # launder (resolved by causal-root identity, not hop count).
            folded = fold_carried_root(env.root)
        prior = taint.derive_value(env.value)
        if prior.is_tainted or prior.sensitive:
            folded = CausalRoot.mint(prior, folded)
        taint.register_value(env.value, folded)
        self._inboxes[env.to_node].append(env)
        received = self._event(
            env.to_node, EventKind.MESSAGE_RECEIVED, env, Verdict.PASS,
            peer_field="from",
        )
        if env.edge_kind == EDGE_PEER:
            received.payload["foreign_root"] = _root_payload(env.root)
            received.payload["local_root"] = _root_payload(folded)
            if evidence:
                received.payload["ladder_evidence"] = list(evidence)
        self._emit(received)

    def _receive_peer(self, env: MessageEnvelope) -> InboundVerdict:
        declaration = self._declaration_for(env.from_node)
        assertion: LabelAssertion | None = None
        if declaration is not None and env.signature:
            # The envelope's own signature doubles as the L2 label assertion:
            # it covers identity, edge, value hash AND the labels — exactly
            # the claim the ladder discounts.
            assertion = LabelAssertion(
                peer_id=env.from_node,
                message_class="default",
                sources=tuple(sorted(s.value for s in env.root.sources)),
                sensitive=env.root.sensitive,
                payload=env.signed_payload(),
                signature=env.signature,
            )
        return receive_foreign(
            value_sensitive_hint=False,
            declaration=declaration,
            assertion=assertion,
        )

    def inbox(self, node_id: str) -> list[MessageEnvelope]:
        return list(self._inboxes.get(node_id, ()))
