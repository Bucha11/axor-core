"""The federated-value transport type.

When a tool result (or a child node's output) actually came from ANOTHER agent, the
producer wraps it as a FederatedValue carrying the peer's provenance receipt. The
kernel recognises the wrapper at ingress and routes (value, receipt, peer) through
the FederationGateway to decide the value's local provenance (restore at L2,
re-mint untrusted at L1, or reject a forged receipt) before the value is used.

A bare value (no wrapper) is handled exactly as before — federation is purely
additive and opt-in.
"""

from __future__ import annotations

from dataclasses import dataclass

from axor_core.federation.receipt import FederationReceipt


@dataclass(frozen=True)
class FederatedValue:
    """A value received from a peer agent, with its provenance receipt."""
    value: object
    receipt: FederationReceipt | None = None
    peer_id: str | None = None
