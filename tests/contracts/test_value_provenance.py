"""ValueProvenance — the public trust-model interface. Enforcement runs through
ANY implementation, proving the boundary is the contract, not a concrete engine."""

from __future__ import annotations

import pytest

from axor_core.capability.executor import CapabilityExecutor, ToolHandler
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


# ── K4 invariants under a SECOND, deliberately-different trust model ──────────────
#
# The contract test above shows enforcement *runs* through a custom backend. These
# two pin the actual theorem (docs/kernel-theorem.md): the decision is a pure
# function of whatever projection the trust model emits, and a deny is fully
# mediated — under a trust model whose tainting RULE differs from the reference
# content-derivation engine. That converts "any trust model" from argued-from-the-
# interface (one Protocol, one implementer) into demonstrated on ≥2 instances.


class _ParityTrustModel:
    """A trust model with a different labelling logic from the reference engine:
    it taints a value iff its string form contains a digit. Arbitrary on purpose —
    the point is that the GATE does not care *how* the causal_root was computed, only
    that the decision factors through it (O1). Still emits the K3 projection type
    (CausalRoot), as the interface requires."""

    def register_value(self, content, root): pass

    def derive_value(self, value):
        return (CausalRoot.external_read(TaintSource.WEB)
                if any(c.isdigit() for c in str(value)) else CausalRoot.constant())

    def inherit_value_ledger(self, parent): pass

    def confidentiality_floor_active(self): return False


@pytest.mark.asyncio
async def test_noninterference_through_alt_trust_model(cap_executor):
    """π(x₁)=π(x₂) ⟹ allow(x₁)=allow(x₂): two DIFFERENT raw values the alt model maps
    to the same (tainted) projection get the same decision; a value it maps to clean
    flips it. The decision depends only on the emitted projection, not on raw bytes."""
    def loop():
        return IntentLoop(capability_executor=cap_executor, trace_events=[],
                          taint_engine=_ParityTrustModel())

    # Two distinct raw contents, both → tainted projection (both contain a digit),
    # into the same high-risk sink (write outside workdir). Equal projection.
    a = await _drive(loop(), _env(), {"path": "/etc/alpha.txt", "content": "alpha7"})
    b = await _drive(loop(), _env(), {"path": "/etc/bravo.txt", "content": "zulu9"})
    assert a["approved"] is b["approved"] is False
    assert a["tool_result"].get("category") == "taint_enforcement"
    assert b["tool_result"].get("category") == "taint_enforcement"

    # A value the alt model maps to CLEAN (no digit) → the projection differs → the
    # decision flips. This is what proves taint_enforcement keyed on the projection,
    # not on the raw content (both contents are equally "untrusted" text).
    clean = await _drive(loop(), _env(), {"path": "/etc/charlie.txt", "content": "no digits here"})
    assert clean["tool_result"].get("category") != "taint_enforcement"


@pytest.mark.asyncio
async def test_complete_mediation_through_alt_trust_model():
    """O3 under the alt model: a denied write never reaches the handler."""
    class RecordingWrite(ToolHandler):
        def __init__(self): self.calls = []
        @property
        def name(self): return "write"
        async def execute(self, args):
            self.calls.append(args)
            return "written"

    handler = RecordingWrite()
    ex = CapabilityExecutor()
    ex.register(handler)
    loop = IntentLoop(capability_executor=ex, trace_events=[], taint_engine=_ParityTrustModel())

    denied = await _drive(loop, _env(), {"path": "/etc/evil.txt", "content": "payload9"})
    assert denied["approved"] is False
    assert handler.calls == [], "denied write reached the executor — mediation breached"
