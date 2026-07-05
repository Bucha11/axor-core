"""Desired state (control-plane lattice) and the folded governance state.

The lattice rules here are the protocol's merge semantics (protocol note v0.2,
section 3): versioned last-write-wins with ``stopped`` absorbing. Budget
narrowing-only is enforced where the gates live — the adapter — not here and
not in backend validation: a compromised backend must not be able to widen.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from axor_core.contracts.degradation import DegradationLevel
from axor_core.kernel.events import Fact
from axor_core.taint.causal_root import CausalRoot


@dataclass(frozen=True)
class Injection:
    """Operator prompt injection: single-shot, at-most-once by id (protocol 4)."""

    injection_id: str
    text: str
    reason: str
    operator: str
    sig: str


@dataclass(frozen=True)
class Excision:
    """Context excision (self-heal): one-shot like injection (protocol 4a).

    ``target_refs`` are the causal-root refs of the context segments to remove.
    The provenance guard (spec 8.2.1) is applied by the adapter before consuming:
    a target with operator-config provenance refuses the whole excision.
    """

    excision_id: str
    target_refs: tuple[str, ...]
    reason: str
    operator: str
    sig: str


@dataclass(frozen=True)
class DesiredState:
    """Control-plane target posture for one node. LWW by version.

    ``stopped`` is absorbing — merge() keeps later writes' version but their
    effects are dropped; the adapter reports them as noop_absorbed.
    ``budget_cap_calls`` is decrease-only AT THE ADAPTER (protocol, section 3).
    """

    version: int
    stopped: bool = False
    paused: bool = False
    budget_cap_calls: int | None = None
    pending_injection: Injection | None = None
    pending_excision: Excision | None = None

    def merge(self, newer: "DesiredState") -> "DesiredState":
        if newer.version <= self.version:
            return self
        if self.stopped:
            return DesiredState(version=newer.version, stopped=True)
        return newer


def excision_refused_refs(
    excision: Excision, provenance: dict[str, str]
) -> tuple[str, ...]:
    """The provenance guard (spec 8.2.1), as a pure decision.

    Returns the target refs whose recorded provenance is operator config; a
    non-empty result means the WHOLE excision must be refused (no partial,
    silently-narrower heal). Unknown provenance is treated as operator config —
    fail closed: deleting a restriction is widening via deletion.
    """
    return tuple(
        ref
        for ref in excision.target_refs
        if provenance.get(ref, "operator_config") == "operator_config"
    )


@dataclass
class GovernanceState:
    """Everything the scrubber shows per step; folded from events by replay().

    ``tainted_refs`` maps value refs to their causal roots as registered by
    TOOL_RESULT events (plus counterfactual synthetic taint). ``excised_refs``
    stop contributing to *new* derivations after a CONTEXT_EXCISION event;
    values already derived keep their taint (spec 8.2.1).
    """

    level: DegradationLevel = DegradationLevel.NORMAL
    tainted_refs: dict[str, CausalRoot] = field(default_factory=dict)
    excised_refs: set[str] = field(default_factory=set)
    floor_active: bool = False
    budget_spent_calls: int = 0
    # Cost is the sum of per-tool weights of approved calls (spec §15). Kept
    # alongside the call count so a budget can cap either dimension — an
    # operator-declared, deterministic cost model that replays identically.
    budget_spent_cost: float = 0.0
    facts: dict[str, Fact] = field(default_factory=dict)
    consumed_injection_ids: set[str] = field(default_factory=set)

    def snapshot(self) -> "GovernanceState":
        return GovernanceState(
            level=self.level,
            tainted_refs=dict(self.tainted_refs),
            excised_refs=set(self.excised_refs),
            floor_active=self.floor_active,
            budget_spent_calls=self.budget_spent_calls,
            budget_spent_cost=self.budget_spent_cost,
            facts=dict(self.facts),
            consumed_injection_ids=set(self.consumed_injection_ids),
        )
