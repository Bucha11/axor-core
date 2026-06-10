"""Advisory adjudicator: sees only a projection of an intent (never raw content),
memoizes verdicts by the projection's hash, and may only tighten a decision
(an existing hard-deny is never overridden into an allow)."""

from __future__ import annotations

from typing import Any

import pytest

from axor_core.capability.executor import CapabilityExecutor, ToolHandler
from axor_core.contracts.cancel import make_token
from axor_core.contracts.canonical import CanonicalizedIntent
from axor_core.contracts.context import ContextView, LineageSummary
from axor_core.contracts.envelope import (
    Capabilities,
    ExecutionEnvelope,
    ExportContract,
)
from axor_core.contracts.policy import ExecutionPolicy, ExportMode, ToolPolicy
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind
from axor_core.kernel.adjudicator import (
    AdjudicationVerdict,
    MemoizingAdjudicator,
    projection_hash,
)
from axor_core.node.canonicalizer import IntentCanonicalizer
from axor_core.node.intent_loop import IntentLoop
from axor_core.node.normalizer import IntentNormalizer
from axor_core.contracts.intent import Intent, IntentKind
from axor_core.taint.engine import TaintEngine

pytestmark = pytest.mark.adversarial


class _CountingAdjudicator:
    """Advises DENY based on a caller-supplied predicate; counts how often it's
    queried and records that it only ever sees a projection (no raw content)."""

    def __init__(self, deny_when=lambda p: False) -> None:
        self.calls = 0
        self.seen: list[CanonicalizedIntent] = []
        self._deny_when = deny_when

    def adjudicate(self, projection: CanonicalizedIntent) -> AdjudicationVerdict:
        self.calls += 1
        self.seen.append(projection)
        return (AdjudicationVerdict.ADVISE_DENY if self._deny_when(projection)
                else AdjudicationVerdict.ABSTAIN)


def _projection(tool="bash", args=None) -> CanonicalizedIntent:
    ni = IntentNormalizer().normalize(
        Intent(kind=IntentKind.TOOL_CALL, payload={"tool": tool, "args": args or {}},
               node_id="n", sequence=1)
    )
    return IntentCanonicalizer().canonicalize(ni, args or {})


# ── unit: memoization + tightening-only behaviour ─────────────────────────────

def test_equal_projection_equal_verdict_queried_once():
    inner = _CountingAdjudicator(deny_when=lambda p: True)
    adj = MemoizingAdjudicator(inner)
    p = _projection("bash", {"cmd": "ls"})
    v1 = adj.verdict(p)
    v2 = adj.verdict(p)
    assert v1 == v2 == AdjudicationVerdict.ADVISE_DENY
    assert inner.calls == 1                      # memoized by projection hash


def test_projection_hash_stable_and_equal_for_equal_projection():
    p1 = _projection("bash", {"cmd": "ls"})
    p2 = _projection("bash", {"cmd": "ls"})
    assert p1 == p2
    assert projection_hash(p1) == projection_hash(p2)


def test_apply_never_overrides_hard_deny():
    inner = _CountingAdjudicator(deny_when=lambda p: False)  # would advise allow
    adj = MemoizingAdjudicator(inner)
    p = _projection()
    # kernel hard-denied → stays denied regardless of advice; advice not even queried
    assert adj.apply(p, kernel_allowed=False) is False
    assert inner.calls == 0


def test_apply_tightens_an_allow_to_deny():
    adj = MemoizingAdjudicator(_CountingAdjudicator(deny_when=lambda p: True))
    p = _projection()
    assert adj.apply(p, kernel_allowed=True) is False


def test_raising_adjudicator_abstains_does_not_block():
    class _Boom:
        def adjudicate(self, projection):
            raise RuntimeError("oracle down")
    adj = MemoizingAdjudicator(_Boom())
    # advisory layer must not break the loop: an error abstains, allow stands.
    assert adj.apply(_projection(), kernel_allowed=True) is True


# ── integration: wired into the intent loop on the would-approve path ─────────

def _env(tool="write") -> ExecutionEnvelope:
    policy = ExecutionPolicy(name="t", tool_policy=ToolPolicy(allow_write=True))
    lineage = LineageSummary(node_id="n1", parent_id=None, depth=0,
                             ancestry_ids=[], inherited_restrictions=[])
    ctx = ContextView(node_id="n1", working_summary="t", visible_fragments=[],
                      active_constraints=[], lineage=lineage, token_count=0,
                      compression_ratio=1.0)
    caps = Capabilities(
        allowed_tools=frozenset({tool}), allow_children=False,
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


class _Handler(ToolHandler):
    def __init__(self, name): self._n = name
    @property
    def name(self): return self._n
    async def execute(self, args: dict[str, Any]): return "ok"


async def _resolve(loop, env, tool, args):
    ev = ExecutorEvent(kind=ExecutorEventKind.TOOL_USE,
                       payload={"tool": tool, "args": args, "tool_use_id": "u"},
                       node_id=env.node_id)
    return await loop._resolve_tool_intent(ev, env)


@pytest.mark.asyncio
async def test_adjudicator_denies_an_otherwise_approved_intent():
    ex = CapabilityExecutor(); ex.register(_Handler("write"))
    adj = _CountingAdjudicator(deny_when=lambda p: True)
    loop = IntentLoop(capability_executor=ex, trace_events=[],
                      taint_engine=TaintEngine(), adjudicator=adj)
    r = await _resolve(loop, _env(), "write", {"path": "/work/x", "content": "hi"})
    assert r.approved is False
    assert r.result.get("category") == "adjudicator"
    assert adj.seen and isinstance(adj.seen[0], CanonicalizedIntent)  # projection-only


@pytest.mark.asyncio
async def test_adjudicator_abstain_leaves_intent_approved():
    ex = CapabilityExecutor(); ex.register(_Handler("write"))
    adj = _CountingAdjudicator(deny_when=lambda p: False)
    loop = IntentLoop(capability_executor=ex, trace_events=[],
                      taint_engine=TaintEngine(), adjudicator=adj)
    r = await _resolve(loop, _env(), "write", {"path": "/work/x", "content": "hi"})
    assert r.approved is True
