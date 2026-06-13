"""ValueProvenance — the public trust-model interface. Enforcement runs through
ANY implementation, proving the boundary is the contract, not a concrete engine."""

from __future__ import annotations

import pytest

from axor_core.contracts.cancel import make_token
from axor_core.contracts.context import ContextView, LineageSummary
from axor_core.contracts.envelope import Capabilities, ExecutionEnvelope, ExportContract
from axor_core.contracts.policy import ExecutionPolicy, ExportMode, ToolPolicy
from axor_core.contracts.provenance import ValueProvenance
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind
from axor_core.contracts.taint import TaintSource
from axor_core.node.intent_loop import IntentLoop
from axor_core.taint.causal_root import CausalRoot
from axor_core.taint.engine import TaintEngine


def test_taint_engine_satisfies_contract():
    assert isinstance(TaintEngine(), ValueProvenance)


class _FakeTrustModel:
    def __init__(self, flagged): self._flagged = flagged
    def register_value(self, content, root): pass
    def derive_value(self, value):
        return (CausalRoot.external_read(TaintSource.WEB)
                if any(f in str(value) for f in self._flagged) else CausalRoot.constant())
    def inherit_value_ledger(self, parent): pass   # contract: backends support inheritance
    # The confidentiality floor is contract-mandated: the kernel calls it directly
    # for the egress decision, so a backend must provide it (this fake models no
    # secret reads, so the floor is never armed).
    def confidentiality_floor_active(self): return False


def test_custom_trust_model_satisfies_contract():
    assert isinstance(_FakeTrustModel(["x"]), ValueProvenance)


def _env():
    pol = ExecutionPolicy(name="t", tool_policy=ToolPolicy(allow_read=True, allow_write=True))
    ln = LineageSummary(node_id="n1", parent_id=None, depth=0, ancestry_ids=[], inherited_restrictions=[])
    ctx = ContextView(node_id="n1", working_summary="t", visible_fragments=[], active_constraints=[],
                      lineage=ln, token_count=0, compression_ratio=1.0)
    caps = Capabilities(allowed_tools=frozenset({"write"}), allow_children=False, allow_nested_children=False,
                        allow_context_expansion=False, allow_export=False, allow_mutation=True, max_child_depth=0)
    return ExecutionEnvelope(node_id="n1", task="t", context=ctx, policy=pol, capabilities=caps,
                             export_contract=ExportContract(mode=ExportMode.RESTRICTED, allowed_fields=frozenset(), max_export_tokens=0),
                             lineage=ln, cancel_token=make_token())


async def _drive(loop, env, args):
    async def _stream():
        yield ExecutorEvent(kind=ExecutorEventKind.TOOL_USE,
                            payload={"tool": "write", "args": args, "tool_use_id": "t"}, node_id=env.node_id)
        yield ExecutorEvent(kind=ExecutorEventKind.STOP, payload={"usage": {}}, node_id=env.node_id)
    out = None
    async for ev in loop.run(_stream(), env):
        if ev.kind == ExecutorEventKind.TEXT and "approved" in ev.payload:
            out = ev.payload
    return out


@pytest.mark.asyncio
async def test_enforcement_runs_through_any_value_provenance(cap_executor):
    loop = IntentLoop(capability_executor=cap_executor, trace_events=[],
                      taint_engine=_FakeTrustModel(["EVIL_PAYLOAD_marker"]))
    denied = await _drive(loop, _env(), {"path": "/etc/evil.txt", "content": "run EVIL_PAYLOAD_marker"})
    assert denied["approved"] is False
    assert denied["tool_result"].get("category") == "taint_enforcement"
    allowed = await _drive(loop, _env(), {"path": "/etc/ok.txt", "content": "clean content"})
    assert allowed["tool_result"].get("category") != "taint_enforcement"
