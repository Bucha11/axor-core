"""Call-bound authority — an approval authorises ONE call, and only while it holds.

The question an approver used to be handed was (tool_use_id, tool, paths,
max_ops): a human answering it authorised a TOOL, for N uses. They cannot have
authorised *this transfer*, because neither the recipient nor the amount was in
the question. Binding on the projection hash would not fix it either — the
canonicalizer strips raw strings, so two transfers to different recipients share a
projection.

A CALL-scoped approval therefore carries the call and is bound by
``call_digest`` (sink + authority scope + every argument). The binding is to a
*call representation*, not to an effect: the same call against changed state can
produce a different effect, so one-shot-ness is not enough. Four properties are
pinned here, each one a way an authorisation could outlive what was approved:

  1. an argument replaced after the approval → the evidence does not fit;
  2. the confidentiality floor armed while the approver was deciding → the call is
     refused and the authorisation is discarded, not banked;
  3. two concurrent calls holding one authorisation → at most one executes;
  4. a tool in ``grantable_tools`` but not in current rights → unreachable before
     a grant, reachable within it, unreachable again once spent.

Plus the baseline: the default posture is unchanged, the approver is asked last,
and every escalation ceiling still bounds the path.
"""
from __future__ import annotations

import asyncio
import dataclasses

import pytest

from axor_core.capability.executor import CapabilityExecutor, ToolHandler
from axor_core.contracts.approval import ApprovalRequest, AuthorityScope
from axor_core.contracts.canonical import ConsequenceClass
from axor_core.contracts.cancel import CancelReason, make_token
from axor_core.contracts.context import ContextView, LineageSummary
from axor_core.contracts.envelope import Capabilities, ExecutionEnvelope, ExportContract
from axor_core.contracts.policy import (
    EscalationPolicy,
    ExecutionPolicy,
    ExportMode,
    ToolPolicy,
)
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind
from axor_core.contracts.taint import TaintSource
from axor_core.contracts.trace import TraceEventKind
from axor_core.node.escalation import (
    EscalationManager,
    _CallEvidence,
    _GrantedEscalation,
)
from axor_core.node.intent_loop import IntentLoop
from axor_core.policy.effect import call_digest
from axor_core.taint.causal_root import CausalRoot
from axor_core.taint.engine import TaintEngine

_TOOL = "send_money"
_NODE = "n1"
_A = {"recipient": "acme", "amount": 100}
_B = {"recipient": "attacker", "amount": 100}


class _Send(ToolHandler):
    def __init__(self) -> None:
        self.calls: list[dict] = []

    @property
    def name(self) -> str:
        return _TOOL

    async def execute(self, args: dict):
        await asyncio.sleep(0)  # a real await, so concurrent calls interleave
        self.calls.append(dict(args))
        return "sent"


class _Zap(ToolHandler):
    @property
    def name(self) -> str:
        return "zap"

    async def execute(self, args: dict):
        return "zapped"


class _Approver:
    """Records every question and answers from a fixed script."""

    def __init__(self, answer=True, during=None) -> None:
        self.seen: list[ApprovalRequest] = []
        self._answer = answer
        self._during = during

    async def __call__(self, request: ApprovalRequest) -> bool:
        self.seen.append(request)
        await asyncio.sleep(0)
        if self._during is not None:
            self._during(request)      # the world moves while we decide
        return self._answer(request) if callable(self._answer) else self._answer


def _envelope(*, call_approval: bool, grantable=(_TOOL,), max_escalations=3,
              rights=(_TOOL,), tools=(_TOOL,)) -> ExecutionEnvelope:
    """``rights`` is A₀ (callable now); ``tools`` is what the policy's tool_policy
    carries; ``grantable`` is G (the delegation ceiling)."""
    policy = ExecutionPolicy(
        name="t",
        tool_policy=ToolPolicy(allow_read=True, extra_allowed=tuple(tools)),
        # send_money is an unknown sink → CONSEQUENTIAL; a REVERSIBLE ceiling puts
        # it over the unattended ceiling, so it needs a governance gate.
        max_unattended_consequence=ConsequenceClass.REVERSIBLE,
        escalation_policy=EscalationPolicy(
            allow_escalation=True,
            grantable_tools=tuple(grantable),
            require_human=True,
            max_escalations=max_escalations,
            call_approval=call_approval,
        ),
    )
    lineage = LineageSummary(
        node_id=_NODE, parent_id=None, depth=0, ancestry_ids=[], inherited_restrictions=[]
    )
    ctx = ContextView(
        node_id=_NODE, working_summary="t", visible_fragments=[], active_constraints=[],
        lineage=lineage, token_count=0, compression_ratio=1.0,
    )
    caps = Capabilities(
        allowed_tools=frozenset(rights), allow_children=False,
        allow_nested_children=False, allow_context_expansion=False,
        allow_export=False, allow_mutation=True, max_child_depth=0,
    )
    return ExecutionEnvelope(
        node_id=_NODE, task="t", context=ctx, policy=policy, capabilities=caps,
        export_contract=ExportContract(
            mode=ExportMode.RESTRICTED, allowed_fields=frozenset(), max_export_tokens=0
        ),
        lineage=lineage, cancel_token=make_token(),
    )


def _loop(approver, handler=None, *, taint=None, egress=frozenset()):
    ex = CapabilityExecutor()
    ex.register(handler or _Send())
    ex.register(_Zap())
    return IntentLoop(
        ex, [], escalation_callback=approver,
        taint_engine=taint or TaintEngine(), egress_sinks=egress,
    )


def _call(args: dict, tool: str = _TOOL) -> ExecutorEvent:
    return ExecutorEvent(
        kind=ExecutorEventKind.TOOL_USE,
        payload={"tool": tool, "args": args, "tool_use_id": "tu1"},
        node_id=_NODE,
    )


def _escalation_naming(tool: str, args: dict) -> ExecutorEvent:
    return ExecutorEvent(
        kind=ExecutorEventKind.TOOL_USE,
        payload={
            "tool": "escalate_policy",
            "tool_use_id": "esc-1",
            "args": {"tool": tool, "reason": "one call", "paths": [], "args": args},
        },
        node_id=_NODE,
    )


# ── baseline: the default posture does not change ────────────────────────────


@pytest.mark.asyncio
async def test_over_ceiling_call_is_denied_when_the_knob_is_off():
    approver = _Approver()
    loop = _loop(approver)
    resolved = await loop._resolve_tool_intent(_call(_A), _envelope(call_approval=False))
    assert not resolved.approved
    assert "CONSEQUENTIAL" in resolved.reason
    assert approver.seen == []  # nobody is asked anything


@pytest.mark.asyncio
async def test_call_approval_authorises_exactly_this_call():
    approver = _Approver(True)
    handler = _Send()
    loop = _loop(approver, handler)
    resolved = await loop._resolve_tool_intent(_call(_A), _envelope(call_approval=True))

    assert resolved.approved
    assert handler.calls == [_A]
    assert len(approver.seen) == 1
    req = approver.seen[0]
    assert req.scope is AuthorityScope.CALL
    assert dict(req.args) == _A                       # the approver saw the call
    assert req.call_digest == call_digest(_TOOL, _A, scope=_NODE)
    assert req.max_ops == 1


@pytest.mark.asyncio
async def test_a_second_identical_call_gets_its_own_question():
    approver = _Approver(True)
    loop = _loop(approver)
    env = _envelope(call_approval=True)
    assert (await loop._resolve_tool_intent(_call(_A), env)).approved
    assert (await loop._resolve_tool_intent(_call(dict(_A)), env)).approved
    assert len(approver.seen) == 2


@pytest.mark.asyncio
async def test_a_denied_approval_denies_the_call():
    approver = _Approver(False)
    handler = _Send()
    loop = _loop(approver, handler)
    resolved = await loop._resolve_tool_intent(_call(_A), _envelope(call_approval=True))
    assert not resolved.approved
    assert handler.calls == []


# ── 1. an argument replaced after the approval ───────────────────────────────


@pytest.mark.asyncio
async def test_evidence_does_not_fit_a_call_with_a_replaced_argument():
    """The agent names the call it wants (ESCALATE_POLICY carrying `args`), gets
    one authorisation — and then calls with a different recipient. The evidence
    does not fit, and with the just-in-time knob off nothing else can authorise
    the swap."""
    approver = _Approver(True)
    handler = _Send()
    loop = _loop(approver, handler)
    env = _envelope(call_approval=False)

    granted = await loop._escalation.grant_from_intent(
        _escalation_naming(_TOOL, _A), env, loop._trace_events
    )
    assert granted["scope"] == "call"

    swapped = await loop._resolve_tool_intent(_call(_B), env)
    assert not swapped.approved
    assert handler.calls == []
    # the call that WAS authorised still goes through
    assert (await loop._resolve_tool_intent(_call(_A), env)).approved
    assert handler.calls == [_A]


@pytest.mark.asyncio
async def test_the_handler_receives_the_argument_set_that_was_approved():
    """The approved set is frozen before the await: an approver (or anything else
    holding the payload) that mutates the arguments while the answer is pending
    cannot change what executes."""
    payload_args = dict(_A)
    handler = _Send()

    def _mutate(request: ApprovalRequest) -> None:
        payload_args["recipient"] = "attacker"   # same dict the intent carries

    approver = _Approver(True, during=_mutate)
    loop = _loop(approver, handler)
    resolved = await loop._resolve_tool_intent(
        _call(payload_args), _envelope(call_approval=True)
    )
    assert resolved.approved
    assert handler.calls == [_A]                  # the approved set, not the mutated one
    assert dict(approver.seen[0].args) == _A


# ── 2. the floor arms while the approver is deciding ─────────────────────────


@pytest.mark.asyncio
async def test_a_floor_armed_during_the_wait_refuses_the_call():
    taint = TaintEngine()
    handler = _Send()

    def _read_a_secret(request: ApprovalRequest) -> None:
        # a concurrent read registers a sensitive value: the confidentiality floor
        # is now up, and this sink is an egress sink
        taint.register_value(
            "sk-live-9999999999", CausalRoot.external_read(TaintSource.FILE, sensitive=True)
        )

    approver = _Approver(True, during=_read_a_secret)
    loop = _loop(approver, handler, taint=taint, egress={_TOOL})
    resolved = await loop._resolve_tool_intent(_call(_A), _envelope(call_approval=True))

    assert not resolved.approved
    assert "approval was outstanding" in resolved.reason
    assert handler.calls == []
    # and the approval is NOT banked for a later attempt under conditions nobody
    # approved: the next call is asked afresh
    assert loop._escalation.covers(_TOOL, _A, scope=_NODE) is False


@pytest.mark.asyncio
async def test_a_cancellation_during_the_wait_refuses_the_call():
    env = _envelope(call_approval=True)
    handler = _Send()

    def _cancel(request: ApprovalRequest) -> None:
        env.cancel_token.cancel(CancelReason.USER_ABORT)

    loop = _loop(_Approver(True, during=_cancel), handler)
    resolved = await loop._resolve_tool_intent(_call(_A), env)
    assert not resolved.approved
    assert "cancelled" in resolved.reason
    assert handler.calls == []


# ── 3. two concurrent calls, one authorisation ───────────────────────────────


@pytest.mark.asyncio
async def test_two_concurrent_calls_cannot_both_spend_one_authorisation():
    approver = _Approver(True)
    handler = _Send()
    loop = _loop(approver, handler)
    env = _envelope(call_approval=False)

    await loop._escalation.grant_from_intent(
        _escalation_naming(_TOOL, _A), env, loop._trace_events
    )
    first, second = await asyncio.gather(
        loop._resolve_tool_intent(_call(dict(_A)), env),
        loop._resolve_tool_intent(_call(dict(_A)), env),
    )
    assert [first.approved, second.approved].count(True) == 1
    assert len(handler.calls) == 1
    loser = first if not first.approved else second
    assert "already been spent" in loser.reason or "CONSEQUENTIAL" in loser.reason


@pytest.mark.asyncio
async def test_a_concurrent_question_about_the_same_call_is_not_asked_twice():
    """Two in-flight approvals for one call could mint two authorisations and would
    put the same question to the approver twice."""
    approver = _Approver(True)
    handler = _Send()
    loop = _loop(approver, handler)
    env = _envelope(call_approval=True)

    results = await asyncio.gather(
        loop._resolve_tool_intent(_call(dict(_A)), env),
        loop._resolve_tool_intent(_call(dict(_A)), env),
    )
    assert [r.approved for r in results].count(True) == 1
    assert len(approver.seen) == 1
    assert len(handler.calls) == 1
    loser = next(r for r in results if not r.approved)
    assert "already outstanding" in loser.reason


# ── 4. a grantable tool that is not a current right ──────────────────────────


@pytest.mark.asyncio
async def test_a_grantable_tool_is_unreachable_until_granted_and_after_spending():
    """A₀ (now) / G (grantable) / A_t (in force) kept apart: the tool is in the
    policy's delegation ceiling but not in the capability set, so it is denied
    before the grant, allowed for the one authorised call, and denied again
    afterwards."""
    approver = _Approver(True)
    handler = _Send()
    loop = _loop(approver, handler)
    env = _envelope(call_approval=False, rights=("read",), grantable=(_TOOL,))

    before = await loop._resolve_tool_intent(_call(_A), env)
    assert not before.approved
    assert "not in capabilities" in before.reason

    granted = await loop._escalation.grant_from_intent(
        _escalation_naming(_TOOL, _A), env, loop._trace_events
    )
    assert granted.get("granted") is True, granted

    during = await loop._resolve_tool_intent(_call(dict(_A)), env)
    assert during.approved
    assert handler.calls == [_A]

    after = await loop._resolve_tool_intent(_call(dict(_A)), env)
    assert not after.approved
    assert len(handler.calls) == 1


# ── the question is asked last; the ceilings still bound the path ────────────


@pytest.mark.asyncio
async def test_a_call_another_gate_refuses_never_reaches_the_approver():
    approver = _Approver(True)
    loop = _loop(approver)
    resolved = await loop._resolve_tool_intent(
        _call({"url": "http://169.254.169.254/latest/meta-data/"}, tool="zap"),
        _envelope(call_approval=True, grantable=("zap",), rights=("zap",), tools=("zap",)),
    )
    assert not resolved.approved
    assert approver.seen == []


@pytest.mark.asyncio
async def test_grantable_tools_bounds_the_call_path():
    approver = _Approver(True)
    loop = _loop(approver)
    resolved = await loop._resolve_tool_intent(
        _call(_A), _envelope(call_approval=True, grantable=("something_else",))
    )
    assert not resolved.approved
    assert "grantable_tools" in resolved.reason
    assert approver.seen == []


@pytest.mark.asyncio
async def test_max_escalations_bounds_the_call_path():
    approver = _Approver(True)
    loop = _loop(approver)
    env = _envelope(call_approval=True, max_escalations=1)
    assert (await loop._resolve_tool_intent(_call(_A), env)).approved
    resolved = await loop._resolve_tool_intent(_call(_B), env)
    assert not resolved.approved
    assert "max escalations" in resolved.reason


@pytest.mark.asyncio
async def test_no_callback_denies():
    loop = _loop(None)
    resolved = await loop._resolve_tool_intent(_call(_A), _envelope(call_approval=True))
    assert not resolved.approved
    assert "no approval callback" in resolved.reason


@pytest.mark.asyncio
async def test_the_authorisation_is_traced_with_its_scope_and_binding():
    approver = _Approver(True)
    loop = _loop(approver)
    await loop._resolve_tool_intent(_call(_A), _envelope(call_approval=True))
    granted = [
        e for e in loop._trace_events if e.kind == TraceEventKind.ESCALATION_GRANTED
    ]
    assert len(granted) == 1
    assert granted[0].scope == "call"
    assert granted[0].call_digest == call_digest(_TOOL, _A, scope=_NODE)
    assert granted[0].auto_approved is False


# ── unit: what call evidence covers ──────────────────────────────────────────


def test_covers_is_call_aware():
    mgr = EscalationManager()
    digest = call_digest(_TOOL, _A, scope=_NODE)
    mgr._call_evidence[digest] = _CallEvidence(tool=_TOOL, digest=digest)
    assert mgr.covers(_TOOL, _A, scope=_NODE) is True
    assert mgr.covers(_TOOL, _B, scope=_NODE) is False
    # another authority scope cannot spend this session's evidence
    assert mgr.covers(_TOOL, _A, scope="other-node") is False
    # no call in hand → only class authority counts
    assert mgr.covers(_TOOL) is False


def test_a_class_grant_still_covers_the_tool():
    mgr = EscalationManager()
    mgr._granted_escalations[_TOOL] = _GrantedEscalation(
        tool=_TOOL, paths=[], ops_remaining=2
    )
    assert mgr.covers(_TOOL) is True
    assert mgr.covers(_TOOL, _B) is True


def test_claim_is_single_use_and_reports_which_case():
    mgr = EscalationManager()
    digest = call_digest(_TOOL, _A, scope=_NODE)
    mgr._call_evidence[digest] = _CallEvidence(tool=_TOOL, digest=digest)
    assert mgr.claim_call(_TOOL, _A, scope=_NODE) is True    # spent here
    assert mgr.claim_call(_TOOL, _A, scope=_NODE) is False   # already spent
    assert mgr.claim_call(_TOOL, _B, scope=_NODE) is None    # nothing to spend
