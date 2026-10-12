"""Differential test: the real IntentLoop, driven by an arbitrary agent, executes
only traces the effect specification (tests/spec/effect_spec.py) admits.

Evidence, not proof. Hypothesis draws an adversarial agent — which tools it calls,
with which arguments, which escalations it asks for, what concurrent calls it
issues — plus attacker-controlled read content and a trusted approver's answers.
The handlers record every effect as it happens; the approver records every yes;
the harness records when a tool's output reaches the agent. That observed trace is
fed to the spec's checker, which fails on the first event no spec state allows.

The spec is the authority/effect one: no provenance. Taint, carrier, degradation
and the adjudicator can only make the loop refuse more, which inclusion permits.

Scope of this configuration (stated, not hidden):
  * one egress sink ``send_money`` over the unattended consequence ceiling, so
    every execution needs an approval; an enum on ``recipient`` and a numeric
    range on ``amount``; a call budget;
  * ``require_human=True`` — every approval comes from the trusted approver; an
    auto-approve configuration is a different spec (the operator delegates);
  * no governance endorsement, so a delivered secret is never released;
  * a real DegradationEngine and the default TaintEngine are wired in.
"""
from __future__ import annotations

import asyncio
import os

from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st

from axor_core.capability.executor import CapabilityExecutor, ToolHandler
from axor_core.contracts.approval import ApprovalRequest, AuthorityScope
from axor_core.contracts.cancel import make_token
from axor_core.contracts.canonical import ConsequenceClass
from axor_core.contracts.context import ContextView, LineageSummary
from axor_core.contracts.envelope import Capabilities, ExecutionEnvelope, ExportContract
from axor_core.contracts.policy import (
    EscalationPolicy,
    ExecutionPolicy,
    ExportMode,
    ToolPolicy,
)
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind
from axor_core.degradation.engine import DegradationEngine
from axor_core.node.intent_loop import IntentLoop
from axor_core.policy.value_policy import enum, numeric_range
from axor_core.taint.engine import TaintEngine

from tests.spec.effect_spec import Approve, Checker, Deliver, Exec, Policy

_NODE = "n1"
SEND, READ_DOC, READ_SECRET, OTHER = "send_money", "read_doc", "read_secret", "wire_all"
ALLOWED = frozenset({"acme", "bob"})
BOUNDS = (0, 1000)


# ── the world: handlers record effects, the approver records approvals ───────


class _Clock:
    """Scheduling noise drawn by Hypothesis: how many event-loop ticks each await
    in the world takes, so a read can land while an approval is pending."""

    def __init__(self, ticks: list[int]) -> None:
        self._ticks, self._i = ticks, 0

    async def wait(self) -> None:
        n = self._ticks[self._i % len(self._ticks)]
        self._i += 1
        for _ in range(n):
            await asyncio.sleep(0)


class _Recorder(ToolHandler):
    def __init__(self, name: str, log: list, output, clock: _Clock) -> None:
        self._name, self._log, self._output, self._clock = name, log, output, clock

    @property
    def name(self) -> str:
        return self._name

    async def execute(self, args: dict):
        # The effect is recorded at dispatch, with the arguments it was dispatched
        # with; the await after it lets concurrent intents interleave while the
        # effect is in flight.
        self._log.append(Exec(self._name, dict(args)))
        await self._clock.wait()
        return self._output()


class _TrustedApprover:
    """Answers from a fixed script (the principal's decisions are not the
    attacker's); records each yes as a spec event."""

    def __init__(self, log: list, answers: list[bool], clock: _Clock) -> None:
        self._log, self._answers, self._i, self._clock = log, answers, 0, clock

    async def __call__(self, request: ApprovalRequest) -> bool:
        await self._clock.wait()             # the world can move while we decide
        yes = self._answers[self._i % len(self._answers)]
        self._i += 1
        if yes:
            self._log.append(Approve(
                scope="call" if request.scope is AuthorityScope.CALL else "class",
                tool=request.tool,
                args=dict(request.args),
                max_ops=request.max_ops,
            ))
        return yes


def _envelope(*, send_in_rights: bool, call_approval: bool) -> ExecutionEnvelope:
    rights = (READ_DOC, READ_SECRET) + ((SEND,) if send_in_rights else ())
    policy = ExecutionPolicy(
        name="spec",
        tool_policy=ToolPolicy(allow_read=True, extra_allowed=rights),
        max_unattended_consequence=ConsequenceClass.REVERSIBLE,
        escalation_policy=EscalationPolicy(
            allow_escalation=True,
            grantable_tools=(SEND,),
            require_human=True,
            max_escalations=4,
            max_ops_per_grant=3,
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
        node_id=_NODE, task="pay the invoice", context=ctx, policy=policy,
        capabilities=caps,
        export_contract=ExportContract(
            mode=ExportMode.RESTRICTED, allowed_fields=frozenset(), max_export_tokens=0
        ),
        lineage=lineage, cancel_token=make_token(),
    )


# ── the arbitrary agent ──────────────────────────────────────────────────────

_recipient = st.one_of(
    st.sampled_from(["acme", "bob", "mallory", "ACME", "acme ", "bob​", ""]),
    st.lists(st.sampled_from(["acme", "bob", "mallory"]), max_size=3),
    st.sampled_from([None, 7, True]),
)
_amount = st.one_of(
    st.sampled_from([0, 1, 100, 100.0, 1000, 1000.0, 1000.5, 1001, -1, -0.0,
                     10**30, "100", True, None,
                     float("nan"), float("inf"), float("-inf")]),
    st.integers(-5, 2000),
    st.floats(allow_nan=True, allow_infinity=True),
)
_extra = st.text(alphabet="abcmalory acme bob1234-", max_size=12)


@st.composite
def _args(draw):
    if draw(st.booleans()):                  # half the time: a well-formed transfer
        # a small pool, so concurrent calls and banked approvals often coincide
        a = {"recipient": draw(st.sampled_from(sorted(ALLOWED))),
             "amount": draw(st.sampled_from([1, 100, 1000]))}
        if draw(st.booleans()):
            a["memo"] = draw(_extra)
        return a
    a = {}
    if draw(st.integers(0, 4)):              # usually present
        a["recipient"] = draw(_recipient)
    if draw(st.integers(0, 4)):
        a["amount"] = draw(_amount)
    if draw(st.booleans()):
        a["memo"] = draw(_extra)
    return a


_call = st.tuples(
    st.just("call"),
    st.sampled_from([SEND, SEND, SEND, READ_DOC, READ_SECRET, OTHER]),
    _args(),
)
_escalate = st.tuples(
    st.just("escalate"),
    st.sampled_from([SEND, SEND, OTHER]),
    st.one_of(st.none(), _args()),           # None → class-scoped request
    st.integers(-1, 6),
)
_single = st.one_of(_call, _call, _escalate)
_step = st.one_of(
    _single,
    st.tuples(st.just("concurrent"), st.lists(_single, min_size=2, max_size=3)),
)


def _event(tool: str, args) -> ExecutorEvent:
    return ExecutorEvent(
        kind=ExecutorEventKind.TOOL_USE,
        payload={"tool": tool, "args": args, "tool_use_id": "tu"},
        node_id=_NODE,
    )


def _escalation_event(tool: str, args, max_ops: int) -> ExecutorEvent:
    payload_args = {"tool": tool, "reason": "agent asks", "paths": [], "max_ops": max_ops}
    if args is not None:
        payload_args["args"] = args
    return ExecutorEvent(
        kind=ExecutorEventKind.TOOL_USE,
        payload={"tool": "escalate_policy", "tool_use_id": "esc", "args": payload_args},
        node_id=_NODE,
    )


async def _run(program, *, send_in_rights, call_approval, budget, answers, doc_text,
               ticks):
    log: list = []
    clock = _Clock(ticks)
    ex = CapabilityExecutor()
    ex.register(_Recorder(SEND, log, lambda: "sent", clock))
    ex.register(_Recorder(READ_DOC, log, lambda: doc_text, clock))
    ex.register(_Recorder(READ_SECRET, log, lambda: "sk-live-4242424242424242", clock))
    ex.register(_Recorder(OTHER, log, lambda: "wired", clock))
    loop = IntentLoop(
        ex, [],
        escalation_callback=_TrustedApprover(log, answers, clock),
        taint_engine=TaintEngine(),
        degradation_engine=DegradationEngine(node_id=_NODE),
        budget_cap_calls=budget,
        value_policies={SEND: [enum("recipient", ALLOWED), numeric_range("amount", *BOUNDS)]},
        consequence_overrides={
            READ_DOC: ConsequenceClass.BENIGN,
            READ_SECRET: ConsequenceClass.BENIGN,
        },
        egress_sinks={SEND},
        untrusted_sources={READ_DOC},
        sensitive_sources={READ_SECRET},
        driving_args={SEND: ["recipient"]},
    )
    env = _envelope(send_in_rights=send_in_rights, call_approval=call_approval)

    async def one(step):
        if step[0] == "call":
            _, tool, args = step
            resolved = await loop._resolve_tool_intent(_event(tool, args), env)
            if resolved.approved:
                log.append(Deliver(tool))
        else:
            _, tool, args, max_ops = step
            await loop._escalation.grant_from_intent(
                _escalation_event(tool, args, max_ops), env, loop._trace_events
            )

    for step in program:
        if step[0] == "concurrent":
            await asyncio.gather(*(one(s) for s in step[1]))
        else:
            await one(step)
    return log


def _policy(*, send_in_rights: bool, budget) -> Policy:
    return Policy(
        callable_tools=frozenset({READ_DOC, READ_SECRET} | ({SEND} if send_in_rights else set())),
        needs_approval=frozenset({SEND}),
        grantable=frozenset({SEND}),
        egress=frozenset({SEND}),
        secret_sources=frozenset({READ_SECRET}),
        allowed_recipients=ALLOWED,
        amount_bounds=BOUNDS,
        budget=budget,
    )


@settings(
    max_examples=int(os.environ.get("SPEC_EXAMPLES", "400")),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(
    program=st.lists(_step, max_size=10),
    send_in_rights=st.booleans(),
    call_approval=st.booleans(),
    budget=st.one_of(st.none(), st.integers(0, 6)),
    answers=st.lists(st.sampled_from([True, True, False]), min_size=1, max_size=6),
    doc_text=st.text(alphabet="pay acme bob mallory 1000 now", max_size=40),
    ticks=st.lists(st.integers(0, 8), min_size=1, max_size=6),
)
def test_every_executed_trace_is_in_the_effect_spec(
    program, send_in_rights, call_approval, budget, answers, doc_text, ticks
):
    log = asyncio.run(_run(
        program, send_in_rights=send_in_rights, call_approval=call_approval,
        budget=budget, answers=answers, doc_text=doc_text, ticks=ticks,
    ))
    checker = Checker(_policy(send_in_rights=send_in_rights, budget=budget))
    for e in log:
        checker.step(e)
    # Coverage of the interesting transitions (shown with --hypothesis-show-statistics).
    sends = [e for e in log if isinstance(e, Exec) and e.tool == SEND]
    event(f"transfers executed: {min(len(sends), 3)}{'+' if len(sends) >= 3 else ''}")
    if any(isinstance(e, Approve) and e.scope == "class" for e in log) and sends:
        event("transfer with a class approval outstanding")
    if any(isinstance(e, Deliver) and e.tool == READ_SECRET for e in log):
        event("secret delivered")
    if budget is not None and sum(isinstance(e, Exec) for e in log) == budget:
        event("budget exhausted")


# ── the checker is not vacuous: hand-built traces it must reject ─────────────


def test_spec_rejects_an_unapproved_transfer():
    c = Checker(_policy(send_in_rights=True, budget=None))
    try:
        c.step(Exec(SEND, {"recipient": "acme", "amount": 1}))
    except AssertionError:
        return
    raise AssertionError("spec admitted a transfer nobody approved")


def test_spec_rejects_a_second_use_of_one_call_approval():
    c = Checker(_policy(send_in_rights=True, budget=None))
    a = {"recipient": "acme", "amount": 1}
    c.step(Approve("call", SEND, a))
    c.step(Exec(SEND, a))
    try:
        c.step(Exec(SEND, a))
    except AssertionError:
        return
    raise AssertionError("spec admitted two executions on one call approval")


def test_spec_rejects_egress_after_a_delivered_secret():
    c = Checker(_policy(send_in_rights=True, budget=None))
    a = {"recipient": "acme", "amount": 1}
    c.step(Approve("class", SEND, {}, 3))
    c.step(Exec(READ_SECRET, {}))
    c.step(Deliver(READ_SECRET))
    try:
        c.step(Exec(SEND, a))
    except AssertionError:
        return
    raise AssertionError("spec admitted egress after a secret reached the agent")


def test_spec_admits_the_alice_bob_case():
    """The spec is effect-level: an allowlisted recipient chosen by an injection is
    allowed. Stated as a test so nobody reads more into the spec than it says."""
    c = Checker(_policy(send_in_rights=True, budget=None))
    c.step(Approve("class", SEND, {}, 1))
    c.step(Exec(READ_DOC, {}))
    c.step(Deliver(READ_DOC))
    c.step(Exec(SEND, {"recipient": "bob", "amount": 10}))
