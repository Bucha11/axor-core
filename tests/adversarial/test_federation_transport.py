"""Reference transport: a value + receipt cross a (serialized) wire and the kernel
makes the trust decision on the far side. Covers the JSON round-trip, an end-to-end
cross-process A2A call through the in-memory network + a session, and a forging peer
being denied."""

from __future__ import annotations

import pytest

from axor_core import GovernedSession
from axor_core.capability import CapabilityExecutor
from axor_core.contracts.invokable import Invokable
from axor_core.contracts.policy import ExecutionPolicy, ExportMode, ToolPolicy
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind
from axor_core.contracts.taint import TaintSource
from axor_core.federation import (
    FederationGateway,
    FederationPeer,
    HmacSigner,
    InMemoryPeerNetwork,
    LocalIdentity,
    mint_receipt,
    peer_tool,
    receipt_from_dict,
    receipt_to_dict,
    verify_receipt,
)
from axor_core.taint.causal_root import CausalRoot
from axor_core.taint.engine import TaintEngine

pytestmark = pytest.mark.adversarial

KEY = b"shared"
KERNEL = "axor-core/4.12"
DOMAIN = "org"


def _id(peer_id="peerB", key=KEY):
    return LocalIdentity(peer_id, KERNEL, DOMAIN, HmacSigner(key))


def _peer(peer_id="peerB", key=KEY):
    return FederationPeer(peer_id, HmacSigner(key), KERNEL, DOMAIN)


def _gateway():
    return FederationGateway(peers={"peerB": _peer()},
                            compatible_kernels={KERNEL}, federated_domains={DOMAIN})


# ── serialization ──────────────────────────────────────────────────────────────

def test_receipt_survives_json_round_trip():
    val = "a value B attests"
    receipt = mint_receipt(val, CausalRoot.external_read(TaintSource.WEB), _id())
    rebuilt = receipt_from_dict(receipt_to_dict(receipt))
    # a rebuilt receipt still verifies and still carries the labels
    assert verify_receipt(val, rebuilt, _peer()) is True
    assert rebuilt.sources == receipt.sources and rebuilt.sensitive == receipt.sensitive


def test_tampering_with_serialized_signature_fails_verification():
    val = "x"
    d = receipt_to_dict(mint_receipt(val, CausalRoot.constant(), _id()))
    d["signature"] = ("00" * (len(d["signature"]) // 2))   # zeroed signature
    assert verify_receipt(val, receipt_from_dict(d), _peer()) is False


# ── end-to-end cross-process A2A through the in-memory network ─────────────────

def _agent(tool: str) -> Invokable:
    class _A(Invokable):
        async def stream(self, env):
            yield ExecutorEvent(kind=ExecutorEventKind.TOOL_USE,
                                payload={"tool": tool, "args": {}}, node_id=env.node_id)
            yield ExecutorEvent(kind=ExecutorEventKind.STOP,
                                payload={"usage": {}}, node_id=env.node_id)
    return _A()


def _policy(tool: str) -> ExecutionPolicy:
    return ExecutionPolicy(name="c", tool_policy=ToolPolicy(extra_allowed=(tool,)),
                           export_mode=ExportMode.SUMMARY)


async def _run(network, tool, gateway):
    cap = CapabilityExecutor()
    cap.register(peer_tool(tool, network, "peerB"))
    sess = GovernedSession(executor=_agent(tool), capability_executor=cap,
                           federation_gateway=gateway)
    result = await sess.run("ask the peer", policy=_policy(tool))
    return sess, result


@pytest.mark.asyncio
async def test_e2e_trusted_peer_value_restored():
    CLEAN = "clean config B computed"
    net = InMemoryPeerNetwork()
    net.register(_id(), TaintEngine(), lambda req: _coro(CLEAN))
    sess, result = await _run(net, "ask_b", _gateway())
    assert result.output.endswith("computed") or CLEAN in result.output
    # restored as clean — a later A sink driven by it is not tainted
    assert sess._taint_engine.derive_value(CLEAN).is_tainted is False


@pytest.mark.asyncio
async def test_e2e_peer_tainted_value_preserved():
    WEB = "web content B fetched for us"
    b_engine = TaintEngine()
    b_engine.register_value(WEB, CausalRoot.external_read(TaintSource.WEB))  # B read it from web
    net = InMemoryPeerNetwork()
    net.register(_id(), b_engine, lambda req: _coro(WEB))
    sess, _ = await _run(net, "ask_b", _gateway())
    assert sess._taint_engine.derive_value(WEB).is_tainted is True   # taint crosses the wire


@pytest.mark.asyncio
async def test_e2e_forging_peer_is_denied():
    # The peer signs with a key our gateway does not trust → the receipt fails
    # verification on arrival → the value is rejected, never fed to the agent.
    net = InMemoryPeerNetwork()
    net.register(_id(key=b"WRONG"), TaintEngine(), lambda req: _coro("attacker data"))
    sess, result = await _run(net, "ask_b", _gateway())
    assert "denied" in result.output


async def _coro(value):
    return value
