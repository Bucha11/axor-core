"""Reference transport for cross-process A2A federation.

The kernel decides TRUST once a value+receipt arrive; it does not move bytes. This
module supplies the small, transport-agnostic glue so a runnable cross-process A2A
exists out of the box, and so a real transport (HTTP/gRPC/queue) only has to
implement one method:

  • receipt_to_dict / receipt_from_dict — JSON-safe (de)serialization of a receipt
    (the signature is hex-encoded). This is the only wire-format concern.
  • to_wire / from_wire — envelope a (value, receipt, peer_id) for the wire and
    rebuild a FederatedValue on the other side.
  • PeerTransport — the one-method interface a real transport implements:
        async def call(peer_id, request) -> wire message (a dict).
  • InMemoryPeerNetwork — a working in-process transport for tests/demos: peers
    register a responder + identity + provenance engine; call() runs the responder,
    mints a receipt for the result, and returns a wire message that has been
    round-tripped through (de)serialization, exercising the full path.
  • peer_tool — builds a ToolHandler that delegates to a peer over a transport and
    returns the FederatedValue the kernel routes through the gateway.

Values crossing the wire must be JSON-serialisable (strings, numbers, plain dicts).
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

from axor_core.capability.executor import ToolHandler
from axor_core.federation.receipt import (
    FederationReceipt,
    LocalIdentity,
    mint_output_receipt,
)
from axor_core.federation.value import FederatedValue


# ── wire (de)serialization ─────────────────────────────────────────────────────

def receipt_to_dict(receipt: FederationReceipt) -> dict:
    """JSON-safe dict for a receipt (signature hex-encoded)."""
    return {
        "peer_id": receipt.peer_id,
        "kernel_version": receipt.kernel_version,
        "domain": receipt.domain,
        "value_hash": receipt.value_hash,
        "algorithm": receipt.algorithm,
        "sources": list(receipt.sources),
        "sensitive": receipt.sensitive,
        "nonce": receipt.nonce,
        "expires_at": receipt.expires_at,
        "signature": receipt.signature.hex(),
    }


def receipt_from_dict(d: dict) -> FederationReceipt:
    """Rebuild a receipt from its JSON-safe dict."""
    return FederationReceipt(
        peer_id=d["peer_id"],
        kernel_version=d["kernel_version"],
        domain=d["domain"],
        value_hash=d["value_hash"],
        algorithm=d.get("algorithm", ""),
        sources=tuple(d.get("sources", ())),
        sensitive=bool(d.get("sensitive", False)),
        nonce=d.get("nonce", ""),
        expires_at=float(d.get("expires_at", 0.0)),
        signature=bytes.fromhex(d["signature"]),
    )


def to_wire(value: object, receipt: FederationReceipt, peer_id: str) -> dict:
    """Envelope a value + receipt for the wire."""
    return {"value": value, "receipt": receipt_to_dict(receipt), "peer_id": peer_id}


def from_wire(message: dict) -> FederatedValue:
    """Rebuild a FederatedValue from a wire envelope."""
    return FederatedValue(
        value=message["value"],
        receipt=receipt_from_dict(message["receipt"]),
        peer_id=message.get("peer_id"),
    )


# ── transport interface + in-memory reference ─────────────────────────────────

@runtime_checkable
class PeerTransport(Protocol):
    """The one method a real transport (HTTP/gRPC/queue) implements: send a request
    to `peer_id` and return the peer's wire message (a dict from `to_wire`)."""

    async def call(self, peer_id: str, request: object) -> dict:
        ...


class _PeerEndpoint:
    __slots__ = ("identity", "engine", "responder")

    def __init__(self, identity, engine, responder):
        self.identity = identity
        self.engine = engine
        self.responder = responder


class InMemoryPeerNetwork:
    """A working in-process transport for tests and demos. Peers register a
    responder coroutine (request -> value), their LocalIdentity, and the engine that
    holds their per-value provenance. call() runs the responder, mints a receipt for
    the result, and returns a wire message round-tripped through serialization."""

    def __init__(self) -> None:
        self._peers: dict[str, _PeerEndpoint] = {}

    def register(
        self,
        identity: LocalIdentity,
        engine: Any,
        responder: Callable[[object], Awaitable[object]],
    ) -> None:
        self._peers[identity.peer_id] = _PeerEndpoint(identity, engine, responder)

    async def call(self, peer_id: str, request: object) -> dict:
        peer = self._peers[peer_id]
        value = await peer.responder(request)
        receipt = mint_output_receipt(peer.engine, value, peer.identity)
        # Round-trip through JSON so the in-memory path exercises the same wire
        # format a real transport would.
        import json
        return json.loads(json.dumps(to_wire(value, receipt, peer.identity.peer_id)))


def peer_tool(name: str, transport: PeerTransport, peer_id: str) -> ToolHandler:
    """Build a ToolHandler that delegates to `peer_id` over `transport` and returns
    the FederatedValue the kernel will route through the federation gateway. Swap
    `transport` for an HTTP/gRPC implementation to go cross-host with no other
    change."""

    class _PeerTool(ToolHandler):
        @property
        def name(self) -> str:
            return name

        async def execute(self, args: dict[str, Any]) -> Any:
            message = await transport.call(peer_id, args)
            return from_wire(message)

    return _PeerTool()
