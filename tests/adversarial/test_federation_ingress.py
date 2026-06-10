"""Federation ingress through the intent loop: a tool that delegated to a peer
returns a FederatedValue; the kernel routes it through the gateway before the value
is used. A valid receipt restores trust (and is unwrapped to the plain value); a
forged receipt is rejected as a denial; without a gateway the wrapper is inert."""

from __future__ import annotations

from typing import Any

import pytest

from axor_core.capability.executor import CapabilityExecutor, ToolHandler
from axor_core.contracts.cancel import make_token
from axor_core.contracts.context import ContextView, LineageSummary
from axor_core.contracts.envelope import (
    Capabilities,
    ExecutionEnvelope,
    ExportContract,
)
from axor_core.contracts.policy import ExecutionPolicy, ExportMode, ToolPolicy
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind
from axor_core.contracts.taint import TaintSource
from axor_core.federation import (
    FederatedValue,
    FederationGateway,
    FederationPeer,
    HmacSigner,
    LocalIdentity,
    mint_receipt,
)
from axor_core.node.intent_loop import IntentLoop
from axor_core.taint.causal_root import CausalRoot
from axor_core.taint.engine import TaintEngine

pytestmark = pytest.mark.adversarial

KEY = b"peer-shared-key"
KERNEL = "axor-core/4.12"
DOMAIN = "trusted.example"
PEER_VALUE = "a result the peer computed cleanly"


def _identity(key=KEY):
    return LocalIdentity("peerB", KERNEL, DOMAIN, HmacSigner(key))


def _gateway():
    peer = FederationPeer("peerB", HmacSigner(KEY), KERNEL, DOMAIN)
    return FederationGateway(peers={"peerB": peer},
                             compatible_kernels={KERNEL}, federated_domains={DOMAIN})


class _PeerTool(ToolHandler):
    """Returns a FederatedValue with a receipt signed by `signing_key`."""

    def __init__(self, signing_key=KEY, value=PEER_VALUE, root=None):
        self._k = signing_key
        self._v = value
        self._root = root or CausalRoot.constant()

    @property
    def name(self) -> str:
        return "ask_peer"

    async def execute(self, args: dict[str, Any]) -> Any:
        receipt = mint_receipt(self._v, self._root, _identity(self._k))
        return FederatedValue(self._v, receipt, "peerB")


def _env() -> ExecutionEnvelope:
    policy = ExecutionPolicy(name="t", tool_policy=ToolPolicy(extra_allowed=("ask_peer",)))
    lineage = LineageSummary(node_id="n1", parent_id=None, depth=0,
                             ancestry_ids=[], inherited_restrictions=[])
    ctx = ContextView(node_id="n1", working_summary="t", visible_fragments=[],
                      active_constraints=[], lineage=lineage, token_count=0,
                      compression_ratio=1.0)
    caps = Capabilities(
        allowed_tools=frozenset({"ask_peer"}), allow_children=False,
        allow_nested_children=False, allow_context_expansion=False,
        allow_export=True, allow_mutation=True, max_child_depth=0,
    )
    return ExecutionEnvelope(
        node_id="n1", task="t", context=ctx, policy=policy, capabilities=caps,
        export_contract=ExportContract(mode=ExportMode.FULL,
                                       allowed_fields=frozenset(["output"]),
                                       max_export_tokens=1024),
        lineage=lineage, cancel_token=make_token(),
    )


async def _resolve(loop, env, tool="ask_peer"):
    ev = ExecutorEvent(kind=ExecutorEventKind.TOOL_USE,
                       payload={"tool": tool, "args": {}, "tool_use_id": "u"},
                       node_id=env.node_id)
    return await loop._resolve_tool_intent(ev, env)


def _loop(handler, gateway=None):
    ex = CapabilityExecutor()
    ex.register(handler)
    return IntentLoop(capability_executor=ex, trace_events=[],
                      taint_engine=TaintEngine(), federation_gateway=gateway)


@pytest.mark.asyncio
async def test_valid_receipt_unwraps_and_restores_trust():
    eng_loop = _loop(_PeerTool(), gateway=_gateway())
    r = await _resolve(eng_loop, _env())
    assert r.approved is True
    assert r.result == PEER_VALUE                          # unwrapped to plain value
    # restored clean → a later sink driven by it is NOT integrity-tainted
    assert eng_loop._taint_engine.derive_value(PEER_VALUE).is_tainted is False


@pytest.mark.asyncio
async def test_peer_tainted_provenance_is_preserved():
    root = CausalRoot.external_read(TaintSource.WEB)
    eng_loop = _loop(_PeerTool(value="web data from peer", root=root), gateway=_gateway())
    env = _env()
    r = await _resolve(eng_loop, env)
    assert r.approved is True
    assert eng_loop._taint_engine.derive_value("web data from peer").is_tainted is True


@pytest.mark.asyncio
async def test_forged_receipt_is_rejected_as_denial():
    eng_loop = _loop(_PeerTool(signing_key=b"WRONG-KEY"), gateway=_gateway())
    r = await _resolve(eng_loop, _env())
    assert r.approved is False
    assert r.result.get("category") == "federation_gate"


@pytest.mark.asyncio
async def test_without_gateway_wrapper_is_inert():
    # No federation configured → the FederatedValue is treated as an ordinary tool
    # result (not unwrapped, not gated). Federation is purely opt-in.
    eng_loop = _loop(_PeerTool(), gateway=None)
    r = await _resolve(eng_loop, _env())
    assert r.approved is True
    assert isinstance(r.result, FederatedValue)
