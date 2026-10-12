"""The authority state machine, driven through random sequences of real moves.

What the earlier sweep pins is static: one composition step never returns an
authority wider than its trusted operand. That is not the claim a system with
lawful recovery needs. This module pins the dynamic one:

    A_t ⊆ B  at every step, under any interleaving of the four moves

with the moves themselves typed (`axor_core.kernel.authority.TransitionKind`) and
each step classified: a narrowing must narrow, a restoration must stay inside the
ceiling and name a grantable tool, spending a call authorisation must not widen
anything, and only a trusted event may move the ceiling. A step with no legal
reading fails with the reason.

The state machine holds **no copy** of the authority situation. Every view is
read back from the live objects — the policy, the degradation engine's narrowed
policy, the capability resolver, the escalation manager's own tables — through
`axor_core.node.authority_probe`. A violation it reports is therefore a violation
in the objects the enforcement path uses, not in a model of them.
"""
from __future__ import annotations

import asyncio

import pytest
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from axor_core.contracts.approval import ApprovalRequest
from axor_core.contracts.cancel import make_token
from axor_core.contracts.canonical import ConsequenceClass
from axor_core.contracts.context import ContextView, LineageSummary
from axor_core.contracts.degradation import DegradationLevel
from axor_core.contracts.envelope import Capabilities, ExecutionEnvelope, ExportContract
from axor_core.contracts.policy import (
    EscalationPolicy,
    ExecutionPolicy,
    ExportMode,
    ToolPolicy,
)
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind
from axor_core.degradation.engine import DegradationEngine
from axor_core.kernel.authority import (
    CONTRACTING,
    check,
    EXTENDING,
    KERNEL_INTERNAL_INTENTS,
    AuthorityView,
    TransitionKind,
    classify,
    violations,
)
from axor_core.node.authority_probe import authority_view
from axor_core.node.escalation import EscalationManager
from axor_core.policy import PolicyComposer
from axor_core.policy.effect import call_digest

_NODE = "n1"
_GRANTABLE = ("write", "bash")
_CALLS = {
    "write": {"path": "/w/a.py", "content": "x"},
    "bash": {"command": "ls"},
}


async def _always_yes(request: ApprovalRequest) -> bool:
    return True


def _policy(*, grantable=_GRANTABLE, allow_write=False,
            ceiling=ConsequenceClass.CONSEQUENTIAL, paths=()) -> ExecutionPolicy:
    return ExecutionPolicy(
        name="base",
        tool_policy=ToolPolicy(allow_read=True, allow_write=allow_write),
        allowed_paths=tuple(paths),
        max_unattended_consequence=ceiling,
        escalation_policy=EscalationPolicy(
            allow_escalation=True,
            grantable_tools=tuple(grantable),
            require_human=True,
            max_escalations=50,
            max_ops_per_grant=5,
        ),
    )


def _envelope(policy: ExecutionPolicy) -> ExecutionEnvelope:
    lineage = LineageSummary(
        node_id=_NODE, parent_id=None, depth=0, ancestry_ids=[], inherited_restrictions=[]
    )
    ctx = ContextView(
        node_id=_NODE, working_summary="t", visible_fragments=[], active_constraints=[],
        lineage=lineage, token_count=0, compression_ratio=1.0,
    )
    return ExecutionEnvelope(
        node_id=_NODE, task="t", context=ctx, policy=policy,
        capabilities=Capabilities(
            allowed_tools=frozenset({"read"}), allow_children=False,
            allow_nested_children=False, allow_context_expansion=False,
            allow_export=False, allow_mutation=True, max_child_depth=0,
        ),
        export_contract=ExportContract(
            mode=ExportMode.RESTRICTED, allowed_fields=frozenset(), max_export_tokens=0
        ),
        lineage=lineage, cancel_token=make_token(),
    )


def _escalation_event(tool: str, call_args: dict | None = None) -> ExecutorEvent:
    args: dict = {"tool": tool, "reason": f"need {tool}", "paths": [], "max_ops": 2}
    if call_args is not None:
        args["args"] = call_args
    return ExecutorEvent(
        kind=ExecutorEventKind.TOOL_USE,
        payload={"tool": "escalate_policy", "tool_use_id": "esc", "args": args},
        node_id=_NODE,
    )


class AuthorityMachine(RuleBasedStateMachine):
    """Real engines; the model is only the expected KIND of each move."""

    def __init__(self) -> None:
        super().__init__()
        self.base = _policy()
        self.active = self.base          # what adaptive narrowing has left of it
        self.degradation = DegradationEngine()
        self.composer = PolicyComposer()
        self.escalation = EscalationManager(
            escalation_callback=_always_yes, driving_args={}
        )
        self.minted: dict[str, dict] = {}   # tool → the call args we authorised
        self.view = self._read()
        self.seen: set[TransitionKind] = set()

    # ── reading the live objects, never a copy ───────────────────────────────

    def _read(self) -> AuthorityView:
        effective = self.degradation.apply_to_policy(self.active, None)
        return authority_view(
            base_policy=self.base,
            effective_policy=effective,
            escalation=self.escalation,
            degradation=self.degradation,
        )

    def _step(self, declared: TransitionKind) -> None:
        """Declare what this move was and have it checked against the states.

        A move that did not happen (a denied escalation, a tighten below the
        current level) leaves the situation untouched, which is IDLE — so the
        declaration is accepted when either it or IDLE holds, and never when
        neither does.
        """
        before, after = self.view, self._read()
        errors = check(before, after, declared)
        if errors and before == after:
            errors = check(before, after, TransitionKind.IDLE)
            declared = TransitionKind.IDLE
        assert not errors, f"{declared.value}: {errors}"
        self.view = after
        self.seen.add(declared)

    # ── the four moves ───────────────────────────────────────────────────────

    @rule(level=st.sampled_from([
        DegradationLevel.CAUTIOUS, DegradationLevel.RESTRICTED, DegradationLevel.LOCKED,
    ]))
    def degrade(self, level):
        self.degradation.tighten(level, reason="signal")
        self._step(TransitionKind.NARROW)

    @rule(write=st.booleans(), ceiling=st.sampled_from(list(ConsequenceClass)))
    def reclassify(self, write, ceiling):
        """Adaptive re-classification: a fresh classifier policy is composed under
        the one in force, so it can only narrow."""
        candidate = _policy(allow_write=write, ceiling=ceiling)
        self.active = self.composer.apply_parent_restrictions(candidate, self.active)
        self._step(TransitionKind.NARROW)

    @rule(tool=st.sampled_from(_GRANTABLE))
    def grant_class(self, tool):
        asyncio.run(
            self.escalation.grant_from_intent(
                _escalation_event(tool), _envelope(self.active), []
            )
        )
        self._step(TransitionKind.RESTORE_CLASS)

    @rule(tool=st.sampled_from(_GRANTABLE))
    def mint_call(self, tool):
        call = dict(_CALLS[tool])
        result = asyncio.run(
            self.escalation.grant_from_intent(
                _escalation_event(tool, call), _envelope(self.active), []
            )
        )
        if result.get("granted"):
            self.minted[tool] = call
        self._step(TransitionKind.RESTORE_CALL)

    @rule(tool=st.sampled_from(_GRANTABLE))
    def spend_call(self, tool):
        call = self.minted.get(tool)
        if call is None:
            return
        self.escalation.claim_call(tool, call, scope=_NODE)
        self._step(TransitionKind.SPEND)

    @rule(tool=st.sampled_from(_GRANTABLE))
    def spend_twice(self, tool):
        """A spent authorisation cannot be spent again — and the second attempt
        must not change the authority situation at all."""
        call = self.minted.get(tool)
        if call is None:
            return
        self.escalation.claim_call(tool, call, scope=_NODE)
        first = self._read()
        again = self.escalation.claim_call(tool, call, scope=_NODE)
        assert again in (False, None), again
        assert self._read() == first
        self.view = first
        self.seen.add(TransitionKind.SPEND)

    @rule(grantable=st.sampled_from([(), ("write",), _GRANTABLE]),
          ceiling=st.sampled_from(list(ConsequenceClass)))
    def set_ceiling(self, grantable, ceiling):
        """A trusted event moves B. In this runtime it coincides with a turn
        boundary: the IntentLoop — and with it the EscalationManager — is
        constructed inside GovernedNode.run, so authority does not outlive the
        turn whose ceiling it was granted under. The scope is therefore
        re-created here, which is what keeps A_t ⊆ B across a ceiling that drops.
        (If authority is ever hoisted to session scope, lowering the ceiling will
        have to retract authority above it — the case asserted in
        test_a_ceiling_that_drops_under_live_authority_is_caught below.)"""
        self.base = _policy(grantable=grantable, ceiling=ceiling)
        self.active = self.base
        self.degradation = DegradationEngine()
        self.escalation = EscalationManager(
            escalation_callback=_always_yes, driving_args={}
        )
        self.minted.clear()
        self._step(TransitionKind.SET_CEILING)

    # ── the invariant, checked after every single step ───────────────────────

    @invariant()
    def containment_holds(self):
        v = self._read()
        assert not violations(v), violations(v)

    @invariant()
    def degradation_adds_only_kernel_internal_names(self):
        """The one declared carve-out, asserted rather than trusted: at LOCKED the
        degradation engine adds `escalate` / `escalate_policy` to the tool set.
        Those are intent kinds the loop handles itself, never governed sinks. If a
        real tool ever appears there, this trips."""
        effective = self.degradation.apply_to_policy(self.active, None)
        added = frozenset(effective.tool_policy.extra_allowed or ()) - frozenset(
            self.active.tool_policy.extra_allowed or ()
        )
        assert added <= KERNEL_INTERNAL_INTENTS, sorted(added)


AuthorityMachine.TestCase.settings = settings(
    max_examples=120,
    stateful_step_count=24,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
TestAuthorityMachine = AuthorityMachine.TestCase


# ── the classification itself, on hand-built pairs ───────────────────────────
#
# The machine above exercises the legal moves. These pin that the classifier
# actually REFUSES the illegal ones — otherwise a green state machine would only
# mean the moves never happened to be illegal.


def _view(**kw) -> AuthorityView:
    base = dict(
        ceiling_tools=frozenset({"read", "write"}),
        grantable_tools=frozenset({"write"}),
        ceiling_consequence=ConsequenceClass.CONSEQUENTIAL,
        in_force_tools=frozenset({"read"}),
        in_force_consequence=ConsequenceClass.REVERSIBLE,
    )
    base.update(kw)
    return AuthorityView(**base)


def test_authority_in_force_outside_the_ceiling_is_a_violation():
    v = _view(in_force_tools=frozenset({"read", "bash"}))
    assert any("outside the ceiling" in e for e in violations(v))


def test_in_force_consequence_above_the_ceiling_is_a_violation():
    v = _view(in_force_consequence=ConsequenceClass.CATASTROPHIC)
    assert any("exceeds the ceiling" in e for e in violations(v))


def test_unrestricted_paths_under_a_restricted_ceiling_is_a_violation():
    v = _view(ceiling_paths=("/w",), in_force_paths=())
    assert any("unrestricted" in e for e in violations(v))
    assert violations(_view(ceiling_paths=("/w",), in_force_paths=("/w/sub",))) == []
    assert any(
        "not within" in e
        for e in violations(_view(ceiling_paths=("/w",), in_force_paths=("/etc",)))
    )


def test_a_restoration_must_name_a_grantable_tool():
    """Inside the ceiling but in neither A₀ nor G: `search` is callable by nobody
    under this policy, so no grant may conjure it. (The ceiling is held constant —
    a moved ceiling is a SET_CEILING, a different move entirely.)"""
    wide = frozenset({"read", "write", "search"})
    before = _view(ceiling_tools=wide)
    after = _view(
        ceiling_tools=wide,
        in_force_tools=frozenset({"read", "search"}),
        class_grants=frozenset({"search"}),
    )
    kind, errors = classify(before, after)
    assert kind is TransitionKind.RESTORE_CLASS
    assert any("neither in force nor grantable" in e for e in errors)


def test_a_restoration_inside_the_ceiling_is_legal():
    before = _view()
    after = _view(
        in_force_tools=frozenset({"read", "write"}), class_grants=frozenset({"write"})
    )
    kind, errors = classify(before, after)
    assert kind is TransitionKind.RESTORE_CLASS and errors == []


def test_spending_may_not_widen_what_is_in_force():
    pair = ("write", "d1")
    wide = frozenset({"read", "write", "bash"})
    before = _view(
        ceiling_tools=wide,
        in_force_tools=frozenset({"read", "write"}),
        unspent_calls=frozenset({pair}),
    )
    after = _view(
        ceiling_tools=wide,
        in_force_tools=frozenset({"read", "write", "bash"}),
        spent_calls=frozenset({pair}),
    )
    kind, errors = classify(before, after)
    assert kind is TransitionKind.SPEND
    assert any("widened" in e for e in errors)


def test_a_spent_authorisation_cannot_become_unspent():
    pair = ("write", "d1")
    before = _view(unspent_calls=frozenset({pair}))
    after = _view(spent_calls=frozenset({pair}), unspent_calls=frozenset({pair}))
    kind, errors = classify(before, after)
    assert kind is TransitionKind.SPEND
    assert errors  # both-states violation and/or the unspent-again error


def test_authority_appearing_with_no_record_and_no_ceiling_change_is_illegal():
    """The shape a silent widening would have: what is in force grew, nothing
    authorised it, the ceiling did not move."""
    before = _view()
    after = _view(in_force_tools=frozenset({"read", "write"}))
    _kind, errors = classify(before, after)
    assert any("no authority record" in e for e in errors)


def test_a_ceiling_that_drops_under_live_authority_is_caught():
    """Authority is per-turn in this runtime, so the case cannot arise today. The
    state machine detects it if that ever changes: a ceiling that no longer covers
    a live grant is a SET_CEILING carrying a containment violation."""
    before = _view(
        in_force_tools=frozenset({"read", "write"}), class_grants=frozenset({"write"})
    )
    after = _view(
        ceiling_tools=frozenset({"read"}),
        grantable_tools=frozenset(),
        in_force_tools=frozenset({"read", "write"}),
        class_grants=frozenset({"write"}),
    )
    kind, errors = classify(before, after)
    assert kind is TransitionKind.SET_CEILING
    assert any("outside the ceiling" in e for e in errors)


def test_the_move_sets_are_disjoint_and_named():
    assert EXTENDING & CONTRACTING == frozenset()
    assert TransitionKind.SET_CEILING not in EXTENDING | CONTRACTING


@pytest.mark.parametrize("name", ["escalate", "escalate_policy"])
def test_kernel_internal_intents_are_not_governed_sinks(name):
    """They are the carve-out the containment check excludes; they are intent
    kinds, not tools, and the degradation engine adds them at LOCKED."""
    v = _view(in_force_tools=frozenset({"read", name}))
    assert violations(v) == []


# ── the machine is not vacuously green ───────────────────────────────────────


def test_a_scripted_run_exercises_every_move_kind():
    """Hypothesis explores interleavings; this pins that each move is reachable at
    all, so a green machine cannot mean "the interesting moves never happened"."""
    m = AuthorityMachine()
    m.grant_class(tool="write")
    m.mint_call(tool="bash")
    m.spend_call(tool="bash")
    m.degrade(level=DegradationLevel.RESTRICTED)
    m.reclassify(write=False, ceiling=ConsequenceClass.BENIGN)
    m.set_ceiling(grantable=(), ceiling=ConsequenceClass.BENIGN)
    assert m.seen >= {
        TransitionKind.RESTORE_CLASS,
        TransitionKind.RESTORE_CALL,
        TransitionKind.SPEND,
        TransitionKind.NARROW,
        TransitionKind.SET_CEILING,
    }, sorted(k.value for k in m.seen)


def test_the_machine_catches_authority_granted_outside_the_delegation_ceiling():
    """The net has teeth: a bug that put a grant in the table for a tool the
    operator never declared grantable is caught by the invariant, not by anyone
    noticing. (Written straight into the table — going through
    grant_from_intent is exactly the path that refuses it.)"""
    m = AuthorityMachine()
    before = m._read()
    from axor_core.node.escalation import _GrantedEscalation

    m.escalation._granted_escalations["search"] = _GrantedEscalation(
        tool="search", paths=[], ops_remaining=1
    )
    after = m._read()
    assert any("outside the ceiling" in e for e in violations(after))
    assert check(before, after, TransitionKind.RESTORE_CLASS)


def test_the_machine_catches_a_call_authorisation_for_a_non_grantable_tool():
    m = AuthorityMachine()
    before = m._read()
    from axor_core.node.escalation import _CallEvidence

    digest = call_digest("search", {"q": "x"}, scope=_NODE)
    m.escalation._call_evidence[digest] = _CallEvidence(tool="search", digest=digest)
    after = m._read()
    errors = check(before, after, TransitionKind.RESTORE_CALL)
    assert any("outside the ceiling" in e for e in errors), errors


def test_a_degradation_level_takes_back_a_granted_tool():
    """A grant extends what POLICY allows; a degradation level is the session's own
    tightening and is not a policy question. So at LOCKED the granted tool is not
    in force — asked of the engine, which is what the enforcement path asks too."""
    m = AuthorityMachine()
    m.grant_class(tool="write")
    assert "write" in m._read().in_force_tools

    m.degradation.tighten(DegradationLevel.LOCKED, reason="signal")
    after = m._read()
    assert "write" not in after.in_force_tools
    assert check(m.view, after, TransitionKind.NARROW) == []
