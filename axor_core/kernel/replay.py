"""Deterministic replay & counterfactuals (spec, section 13).

``replay()`` folds an event sequence into per-step :class:`GovernanceState`
and — given a :class:`KernelConfig` — re-evaluates every recorded tool call
through the SAME pure gate predicates the runtime uses
(:mod:`axor_core.policy.gates`). Two implementations of the pipeline would
eventually diverge and counterfactual replay would silently lie; importing the
one implementation is the whole point (architecture rule 0).

Soundness (the first-divergence rule, spec 13.2): counterfactual re-evaluation
is sound only up to the first step where the counterfactual verdict differs
from the recorded one. Steps past divergence are still folded but flagged
``hypothetical`` — callers render them as such and MUST NOT score them.

Adjudicator exception: the adjudicator is the one gate with a model inside.
Replay NEVER re-runs it — its recorded verdict stands; if a counterfactual
changes the inputs that reached it, that step is a divergence point by
definition. All gates re-evaluated here are pure.

Taint-flow conventions (kept deliberately explicit):

- A TOOL_RESULT event with ``payload.value_ref`` registers that ref's causal
  root in the fold; a TOOL_CALL that lists ``payload.arg_refs`` has its driving
  root re-derived by minting the roots of the referenced values. This is what
  makes "inject synthetic taint at step N" and "excise this segment" flow
  through the rest of the trace.
- A TOOL_CALL without ``arg_refs`` replays its recorded ``driving_root`` as-is
  (the recorder didn't provide enough structure to re-derive; fidelity follows
  the trace, spec 13.4).
- Excised refs contribute nothing to derivations after the CONTEXT_EXCISION
  event; values already derived keep their taint (spec 8.2.1).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from axor_core.contracts.anomaly import NormalizedIntent
from axor_core.contracts.canonical import ConsequenceClass
from axor_core.contracts.degradation import DegradationLevel
from axor_core.kernel.degradation import compute_level
from axor_core.kernel.events import (
    SCHEMA_VERSION,
    Event,
    EventKind,
    Fact,
    Verdict,
    fact_from_payload,
)
from axor_core.kernel.errors import SchemaVersionError
from axor_core.kernel.state import GovernanceState
from axor_core.policy.gates import (
    GateDecision,
    carrier_gate,
    consequence_gate,
    driving_subset,
    integrity_superseded_by_decidable,
    positional_gate,
    ssrf_gate,
    taint_gate,
    value_policy_gate,
)
from axor_core.taint.causal_root import CausalRoot, TaintSource

# Tools always admitted at LOCKED/TERMINAL — mirrors the runtime engine's
# _LOCKED_ALLOWED_TOOLS (axor_core/degradation/engine.py).
LOCKED_ALLOWED_TOOLS = frozenset({"read", "escalate", "escalate_policy"})

# Severity a counterfactual denial contributes to the degradation recompute,
# by gate category. Per-value/channel gates quarantine (RESTRICTED); the rest
# are soft signals (CAUTIOUS). Mirrors the runtime engine's fact table.
_CF_DENIAL_SEVERITY = {
    "taint_enforcement": int(DegradationLevel.RESTRICTED),
    "carrier_gate": int(DegradationLevel.RESTRICTED),
    "ssrf_gate": int(DegradationLevel.RESTRICTED),
}
_CF_DEFAULT_SEVERITY = int(DegradationLevel.CAUTIOUS)


@dataclass(frozen=True)
class KernelConfig:
    """The replayable projection of governance config the gates consume.

    Produced by the Config Builder (spec 11) and by hand; a plain value object
    so a counterfactual is "same trace, different KernelConfig".
    """

    allowed_tools: frozenset[str] | None = None  # None = capability not evaluated
    egress_sinks: frozenset[str] = frozenset()
    imperative_sinks: frozenset[str] = frozenset()
    positional_sinks: frozenset[str] = frozenset()
    value_policies: dict = field(default_factory=dict)  # tool -> [ValuePredicate]
    driving_args: dict = field(default_factory=dict)  # tool -> frozenset[str]
    consequence_overrides: dict = field(default_factory=dict)
    max_unattended_consequence: ConsequenceClass = ConsequenceClass.CONSEQUENTIAL
    budget_cap_calls: int | None = None
    # Cost budget (spec §15): a cap on the summed per-tool weights of approved
    # calls. tool_weights is the operator-declared weight table; a tool absent
    # from it costs default_tool_weight. Deterministic → replay-compatible.
    budget_cap_cost: float | None = None
    tool_weights: dict = field(default_factory=dict)  # tool -> float
    default_tool_weight: float = 1.0
    # Counterfactual: these value refs arrive tainted at registration.
    synthetic_taint_refs: frozenset[str] = frozenset()

    def weight_of(self, tool: str) -> float:
        """The operator-declared cost weight of one call to `tool`."""
        return float(self.tool_weights.get(tool, self.default_tool_weight))


@dataclass(frozen=True)
class ReplayStep:
    event: Event
    state: GovernanceState
    recorded_verdict: Verdict | None
    reevaluated_verdict: Verdict | None
    deny: GateDecision | None
    hypothetical: bool


@dataclass(frozen=True)
class ReplayResult:
    steps: tuple[ReplayStep, ...]
    first_divergence: int | None  # index into steps, None = no divergence


def check_schema(events: Sequence[Event]) -> None:
    major = SCHEMA_VERSION.split(".")[0]
    for e in events:
        if e.schema_version.split(".")[0] != major:
            raise SchemaVersionError(
                f"event seq={e.seq}: schema {e.schema_version}, kernel {SCHEMA_VERSION}"
            )


def normalized_from_payload(tool: str, payload: dict) -> NormalizedIntent:
    n = payload.get("normalized") or {}
    return NormalizedIntent(
        tool=tool,
        operation=n.get("operation", "other"),
        target_kind=n.get("target_kind", "workdir"),
        destination_kind=n.get("destination_kind", "none"),
        provenance=n.get("provenance", "user"),
        reads_secret_like_data=bool(n.get("reads_secret_like_data", False)),
        writes_outside_workdir=bool(n.get("writes_outside_workdir", False)),
        executes_generated_code=bool(n.get("executes_generated_code", False)),
        after_external_read=bool(n.get("after_external_read", False)),
        after_secret_access=bool(n.get("after_secret_access", False)),
        data_flow=n.get("data_flow", "none"),
    )


def root_from_payload(d: dict | None) -> CausalRoot:
    if not d:
        return CausalRoot.constant()
    sources = frozenset(TaintSource(s) for s in d.get("sources", ()))
    return CausalRoot(sources=sources, sensitive=bool(d.get("sensitive", False)))


def _derive_driving_root(
    payload: dict, state: GovernanceState, config: KernelConfig
) -> CausalRoot:
    arg_refs = payload.get("arg_refs") or {}
    if not arg_refs:
        return root_from_payload(payload.get("driving_root"))
    roots = []
    for ref in arg_refs.values():
        if ref in state.excised_refs:
            continue  # excision removes future influence (spec 8.2.1)
        if ref in state.tainted_refs:
            roots.append(state.tainted_refs[ref])
        if ref in config.synthetic_taint_refs:
            roots.append(CausalRoot.cross_process_in())
    return CausalRoot.mint(*roots) if roots else CausalRoot.constant()


def evaluate_call(
    tool: str,
    args: dict,
    normalized: NormalizedIntent,
    driving_root: CausalRoot,
    floor_active: bool,
    config: KernelConfig,
    state: GovernanceState,
) -> GateDecision | None:
    """The replay gate cascade — pure-gate subset of the IntentLoop order.

    Capability, degradation admission and budget wrap the shared predicates the
    same way the runtime does; everything from consequence_gate down IS the
    runtime's code (axor_core.policy.gates), not a copy of it.
    """
    if config.allowed_tools is not None and tool not in config.allowed_tools:
        return GateDecision(
            reason=f"capability: tool '{tool}' is not in the capability table",
            category="capability",
        )
    if state.level >= DegradationLevel.LOCKED and tool not in LOCKED_ALLOWED_TOOLS:
        return GateDecision(
            reason=(
                f"degradation: level {state.level.name} admits only "
                f"{sorted(LOCKED_ALLOWED_TOOLS)}"
            ),
            category="degradation",
        )
    if (
        config.budget_cap_calls is not None
        and state.budget_spent_calls >= config.budget_cap_calls
    ):
        return GateDecision(
            reason=(
                f"budget: call cap {config.budget_cap_calls} exhausted "
                f"({state.budget_spent_calls} spent)"
            ),
            category="budget",
        )
    if (
        config.budget_cap_cost is not None
        and state.budget_spent_cost + config.weight_of(tool) > config.budget_cap_cost
    ):
        return GateDecision(
            reason=(
                f"budget: cost cap {config.budget_cap_cost} exhausted "
                f"({state.budget_spent_cost} spent, "
                f"+{config.weight_of(tool)} for '{tool}')"
            ),
            category="budget",
        )
    deny = consequence_gate(
        tool,
        normalized.operation,
        config.max_unattended_consequence,
        overrides=config.consequence_overrides,
    )
    if deny:
        return deny
    deny = value_policy_gate(tool, args, config.value_policies)
    if deny:
        return deny
    deny = ssrf_gate(tool, normalized)
    if deny:
        return deny
    deny = positional_gate(tool, args, config.positional_sinks)
    if deny:
        return deny
    drivers = config.driving_args.get(tool)
    driving = driving_subset(args, drivers)
    deny = carrier_gate(tool, driving, normalized, driving_root, config.imperative_sinks)
    if deny:
        return deny
    superseded = integrity_superseded_by_decidable(
        tool, args, config.driving_args, config.value_policies
    )
    return taint_gate(
        tool,
        normalized,
        driving_root,
        floor_active,
        egress_sinks=config.egress_sinks,
        integrity_superseded=superseded,
    )


def replay(
    events: Sequence[Event], config: KernelConfig | None = None
) -> ReplayResult:
    """Fold events; with a config this is a counterfactual re-evaluation.

    Without a config (scrubber mode) recorded verdicts are echoed and only the
    state fold runs. With a config, every TOOL_CALL is re-gated; the first
    recorded-vs-reevaluated mismatch is the divergence point.
    """
    check_schema(events)
    state = GovernanceState()
    steps: list[ReplayStep] = []
    first_divergence: int | None = None

    for event in events:
        recorded = event.verdict
        reevaluated: Verdict | None = None
        deny: GateDecision | None = None

        if event.kind is EventKind.TOOL_CALL and config is not None:
            tool = str(event.payload.get("tool", ""))
            args = event.payload.get("args") or {}
            normalized = normalized_from_payload(tool, event.payload)
            driving_root = _derive_driving_root(event.payload, state, config)
            floor = state.floor_active or bool(event.payload.get("floor_active"))
            deny = evaluate_call(
                tool, args, normalized, driving_root, floor, config, state
            )
            reevaluated = Verdict.DENY if deny else Verdict.PASS
            if (
                first_divergence is None
                and recorded is not None
                and reevaluated is not recorded
            ):
                first_divergence = len(steps)
            if deny is not None and reevaluated is not recorded:
                # A counterfactual denial is a new fact for the recompute.
                fact = Fact(
                    fact_id=f"cf_{event.seq}",
                    fact_type="counterfactual_denial",
                    severity=_CF_DENIAL_SEVERITY.get(
                        deny.category, _CF_DEFAULT_SEVERITY
                    ),
                    reason=deny.reason,
                )
                state.facts[fact.fact_id] = fact
                state.level = compute_level(state.facts)

        effective = reevaluated if config is not None else recorded

        if event.kind is EventKind.TOOL_CALL and effective is not Verdict.DENY:
            state.budget_spent_calls += 1
            if config is not None:
                state.budget_spent_cost += config.weight_of(
                    str(event.payload.get("tool", ""))
                )
        elif event.kind is EventKind.TOOL_RESULT:
            ref = event.payload.get("value_ref") or event.causal_root
            if ref:
                root = root_from_payload(event.payload.get("root"))
                if config is not None and ref in config.synthetic_taint_refs:
                    root = CausalRoot.mint(root, CausalRoot.cross_process_in())
                if root.is_tainted or root.sensitive:
                    state.tainted_refs[str(ref)] = root
                if root.sensitive:
                    state.floor_active = True
        elif event.kind is EventKind.FACT:
            fact = fact_from_payload(event.payload)
            state.facts[fact.fact_id] = fact
            state.level = compute_level(state.facts)
        elif event.kind is EventKind.CONTEXT_EXCISION:
            state.excised_refs.update(
                str(r) for r in event.payload.get("refs", ())
            )
        elif event.kind is EventKind.INJECTION_CONSUMED:
            injection_id = event.payload.get("id")
            if injection_id:
                state.consumed_injection_ids.add(str(injection_id))

        steps.append(
            ReplayStep(
                event=event,
                state=state.snapshot(),
                recorded_verdict=recorded,
                reevaluated_verdict=reevaluated,
                deny=deny,
                hypothetical=(
                    first_divergence is not None and len(steps) > first_divergence
                ),
            )
        )

    return ReplayResult(steps=tuple(steps), first_divergence=first_divergence)
