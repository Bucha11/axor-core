"""Effect-bound authority — an approval authorises ONE call, not a class of calls.

The old question put to an approver was ``(tool_use_id, tool, paths, max_ops)``: a
human answering it authorised a tool, for N uses, under some path roots. It could
not have authorised *this transfer*, because neither the recipient nor the amount
was in the question — and the grant that followed could not tell one call from
another. Binding on the projection hash would not have fixed it either: the
canonicalizer strips raw strings, so two transfers to different recipients share a
projection.

So an EFFECT-scoped approval carries the call and is bound to its effect digest,
one use. What is pinned here:

  * the default posture does not change — with the knob off, an over-ceiling call
    is denied exactly where it was before;
  * an approval covers its own call and nothing else: not a second identical call
    (that gets its own question), not a different one;
  * the question is asked LAST, so an approver never renders the arguments of a
    call another gate would have refused;
  * every ceiling that bounds a requested escalation bounds this path too.
"""
from __future__ import annotations

import dataclasses

import pytest

from axor_core.capability.executor import CapabilityExecutor, ToolHandler
from axor_core.contracts.approval import ApprovalRequest, AuthorityScope
from axor_core.contracts.canonical import ConsequenceClass
from axor_core.contracts.cancel import make_token
from axor_core.contracts.context import ContextView, LineageSummary
from axor_core.contracts.envelope import Capabilities, ExecutionEnvelope, ExportContract
from axor_core.contracts.policy import (
    EscalationPolicy,
    ExecutionPolicy,
    ExportMode,
    ToolPolicy,
)
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind
from axor_core.contracts.trace import TraceEventKind
from axor_core.node.escalation import EscalationManager, _GrantedEscalation
from axor_core.node.intent_loop import IntentLoop
from axor_core.policy.effect import effect_digest

_TOOL = "send_money"


class _Send(ToolHandler):
    def __init__(self) -> None:
        self.calls: list[dict] = []

    @property
    def name(self) -> str:
        return _TOOL

    async def execute(self, args: dict):
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

    def __init__(self, answer=True) -> None:
        self.seen: list[ApprovalRequest] = []
        self._answer = answer

    async def __call__(self, request: ApprovalRequest) -> bool:
        self.seen.append(request)
        if callable(self._answer):
            return self._answer(request)
        return self._answer


def _envelope(*, effect_approval: bool, grantable=(_TOOL,), max_escalations=3,
              tools=(_TOOL,)) -> ExecutionEnvelope:
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
            effect_approval=effect_approval,
        ),
    )
    lineage = LineageSummary(
        node_id="n1", parent_id=None, depth=0, ancestry_ids=[], inherited_restrictions=[]
    )
    ctx = ContextView(
        node_id="n1", working_summary="t", visible_fragments=[], active_constraints=[],
        lineage=lineage, token_count=0, compression_ratio=1.0,
    )
    caps = Capabilities(
        allowed_tools=frozenset({*tools, "zap"}), allow_children=False,
        allow_nested_children=False, allow_context_expansion=False,
        allow_export=False, allow_mutation=True, max_child_depth=0,
    )
    return ExecutionEnvelope(
        node_id="n1", task="t", context=ctx, policy=policy, capabilities=caps,
        export_contract=ExportContract(
            mode=ExportMode.RESTRICTED, allowed_fields=frozenset(), max_export_tokens=0
        ),
        lineage=lineage, cancel_token=make_token(),
    )


def _loop(approver, handler=None):
    ex = CapabilityExecutor()
    ex.register(handler or _Send())
    ex.register(_Zap())
    return IntentLoop(ex, [], escalation_callback=approver)


def _call(args: dict, tool: str = _TOOL) -> ExecutorEvent:
    return ExecutorEvent(
        kind=ExecutorEventKind.TOOL_USE,
        payload={"tool": tool, "args": args, "tool_use_id": "tu1"},
        node_id="n1",
    )


_A = {"recipient": "acme", "amount": 100}
_B = {"recipient": "attacker", "amount": 100}


# ── the default posture is unchanged ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_over_ceiling_call_is_denied_when_the_knob_is_off():
    approver = _Approver()
    loop = _loop(approver)
    resolved = await loop._resolve_tool_intent(_call(_A), _envelope(effect_approval=False))
    assert not resolved.approved
    assert "CONSEQUENTIAL" in resolved.reason
    assert approver.seen == []  # nobody is asked anything


# ── an approval authorises its own effect ────────────────────────────────────


@pytest.mark.asyncio
async def test_effect_approval_authorises_exactly_this_call():
    approver = _Approver(True)
    handler = _Send()
    loop = _loop(approver, handler)
    resolved = await loop._resolve_tool_intent(_call(_A), _envelope(effect_approval=True))

    assert resolved.approved
    assert handler.calls == [_A]
    assert len(approver.seen) == 1
    req = approver.seen[0]
    assert req.scope is AuthorityScope.EFFECT
    assert dict(req.args) == _A                       # the approver saw the call
    assert req.effect_digest == effect_digest(_TOOL, _A)
    assert req.max_ops == 1


@pytest.mark.asyncio
async def test_a_second_identical_call_gets_its_own_question():
    """One approval, one effect. A repeat is a new effect occurrence and is asked
    again — the grant is burned by the call it was issued for."""
    approver = _Approver(True)
    loop = _loop(approver)
    env = _envelope(effect_approval=True)
    assert (await loop._resolve_tool_intent(_call(_A), env)).approved
    assert (await loop._resolve_tool_intent(_call(dict(_A)), env)).approved
    assert len(approver.seen) == 2


@pytest.mark.asyncio
async def test_an_approval_does_not_cover_a_different_effect():
    """Approve the transfer to acme; the swap to another recipient is a different
    effect, asked separately, and denied here."""
    approver = _Approver(lambda r: r.args.get("recipient") == "acme")
    handler = _Send()
    loop = _loop(approver, handler)
    env = _envelope(effect_approval=True)

    assert (await loop._resolve_tool_intent(_call(_A), env)).approved
    resolved = await loop._resolve_tool_intent(_call(_B), env)
    assert not resolved.approved
    assert "not approved" in resolved.reason
    assert handler.calls == [_A]                      # the swap never executed
    assert len(approver.seen) == 2


@pytest.mark.asyncio
async def test_a_denied_approval_denies_the_call():
    approver = _Approver(False)
    handler = _Send()
    loop = _loop(approver, handler)
    resolved = await loop._resolve_tool_intent(_call(_A), _envelope(effect_approval=True))
    assert not resolved.approved
    assert handler.calls == []
    assert len(approver.seen) == 1


# ── the question is asked last ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_call_another_gate_refuses_never_reaches_the_approver():
    """An approver has to render the call's arguments, which are untrusted content.
    Asking last means it is only ever shown a call every structural gate admits —
    here the SSRF gate refuses an internal-metadata destination first."""
    approver = _Approver(True)
    loop = _loop(approver)
    resolved = await loop._resolve_tool_intent(
        _call({"url": "http://169.254.169.254/latest/meta-data/"}, tool="zap"),
        _envelope(effect_approval=True, grantable=("zap",), tools=("zap",)),
    )
    assert not resolved.approved
    assert approver.seen == []


# ── every escalation ceiling still bounds this path ──────────────────────────


@pytest.mark.asyncio
async def test_grantable_tools_bounds_the_effect_path():
    approver = _Approver(True)
    loop = _loop(approver)
    resolved = await loop._resolve_tool_intent(
        _call(_A), _envelope(effect_approval=True, grantable=("something_else",))
    )
    assert not resolved.approved
    assert "grantable_tools" in resolved.reason
    assert approver.seen == []


@pytest.mark.asyncio
async def test_max_escalations_bounds_the_effect_path():
    approver = _Approver(True)
    loop = _loop(approver)
    env = _envelope(effect_approval=True, max_escalations=1)
    assert (await loop._resolve_tool_intent(_call(_A), env)).approved
    resolved = await loop._resolve_tool_intent(_call(_B), env)
    assert not resolved.approved
    assert "max escalations" in resolved.reason


@pytest.mark.asyncio
async def test_no_callback_denies():
    loop = _loop(None)
    resolved = await loop._resolve_tool_intent(_call(_A), _envelope(effect_approval=True))
    assert not resolved.approved
    assert "no approval callback" in resolved.reason


@pytest.mark.asyncio
async def test_the_grant_is_traced_with_its_scope_and_binding():
    approver = _Approver(True)
    loop = _loop(approver)
    await loop._resolve_tool_intent(_call(_A), _envelope(effect_approval=True))
    granted = [
        e for e in loop._trace_events if e.kind == TraceEventKind.ESCALATION_GRANTED
    ]
    assert len(granted) == 1
    assert granted[0].scope == "effect"
    assert granted[0].effect_digest == effect_digest(_TOOL, _A)
    assert granted[0].auto_approved is False


# ── unit: what an effect-bound grant covers ──────────────────────────────────


def test_covers_is_effect_aware():
    mgr = EscalationManager()
    mgr._granted_escalations[_TOOL] = _GrantedEscalation(
        tool=_TOOL, paths=[], ops_remaining=1, effect_digest=effect_digest(_TOOL, _A)
    )
    assert mgr.covers(_TOOL, _A) is True
    assert mgr.covers(_TOOL, _B) is False
    # No call in hand → only class-scoped authority counts.
    assert mgr.covers(_TOOL) is False


def test_a_class_grant_still_covers_the_tool():
    mgr = EscalationManager()
    mgr._granted_escalations[_TOOL] = _GrantedEscalation(
        tool=_TOOL, paths=[], ops_remaining=2
    )
    assert mgr.covers(_TOOL) is True
    assert mgr.covers(_TOOL, _B) is True


def test_an_effect_bound_grant_is_not_evidence_for_another_call_but_denies_nothing():
    """A bound authorisation is additive evidence for its own effect, never a new
    restriction on the tool: a different call falls through to ordinary policy
    instead of being denied by someone else's grant."""
    mgr = EscalationManager()
    mgr._granted_escalations[_TOOL] = _GrantedEscalation(
        tool=_TOOL, paths=[], ops_remaining=1, effect_digest=effect_digest(_TOOL, _A)
    )
    decision, pending = mgr.evaluate(_TOOL, _B)
    assert decision is None and pending is None
