"""Sink-projection registration validated by the faithfulness classifier.

A sink declares, per argument, the admissible projection it consumes (a codomain
kind + a consumption mode). The kernel then decides the faithfulness obligation via
``kernel.decidability.classify``:

  • DECIDABLE  — the argument may be guarded by a value-policy PREDICATE, discharged
                 by a decision procedure (verify_enum / verify_bounded_numeric).
  • FUZZ_REQ   — the argument is a rich-syntax field (path / interpreted string /
                 carrier over free text); it CANNOT be soundly discharged by a
                 predicate and must instead ride the fuzz obligation and/or the
                 positional (instruction-incomplete) gate.

This makes the classifier load-bearing: ``validate_value_policies`` rejects a
configuration that tries to guard a fuzz-required field with a decidable predicate
(the silent misconfiguration that would give false assurance).
"""

from __future__ import annotations

from axor_core.kernel.decidability import (
    CodomainKind,
    ConsumptionMode,
    DecidabilityVerdict,
    classify,
    is_decidable,
)
from axor_core.policy.value_policy import ValuePredicate

# Each value-predicate kind implies the projection it consumes. Both are in the
# DECIDABLE class by construction; this map is what `validate_value_policies`
# checks against the faithfulness classifier, and it is the single place a new
# predicate kind must declare its projection (or it fails closed as unknown).
_PREDICATE_PROJECTION: dict[str, tuple[CodomainKind, ConsumptionMode]] = {
    "numeric_range": (CodomainKind.BOUNDED_NUMERIC, ConsumptionMode.NUMERIC),
    "enum": (CodomainKind.ENUM, ConsumptionMode.CASE_SPLIT),
}


def predicate_is_decidable(pred: ValuePredicate) -> bool:
    """True iff the predicate's implied projection makes faithfulness decidable."""
    proj = _PREDICATE_PROJECTION.get(pred.kind)
    if proj is None:
        return False  # unknown predicate kind → not provably decidable
    return is_decidable(*proj)


def validate_value_policies(
    policies: "dict[str, list[ValuePredicate]] | None",
) -> list[str]:
    """Validate operator-registered value policies against the faithfulness rule.

    Returns a list of configuration errors (empty == valid). A predicate is an
    error when its projection does not make faithfulness decidable: a decision
    procedure cannot soundly discharge a rich-syntax (fuzz-required) field, so
    guarding one that way is false assurance and must be surfaced, not silently
    accepted.
    """
    errors: list[str] = []
    if not policies:
        return errors
    for sink, preds in policies.items():
        for pred in preds:
            proj = _PREDICATE_PROJECTION.get(pred.kind)
            if proj is None:
                errors.append(
                    f"sink {sink!r} arg {pred.arg!r}: unknown predicate kind "
                    f"{pred.kind!r} — projection undeclared, cannot classify faithfulness"
                )
                continue
            result = classify(*proj)
            if result.verdict != DecidabilityVerdict.DECIDABLE_PASS:
                errors.append(
                    f"sink {sink!r} arg {pred.arg!r}: projection "
                    f"{proj[0].value}/{proj[1].value} is {result.verdict.value} — "
                    "a value predicate cannot discharge a fuzz-required field"
                )
    return errors


def validate_egress_allowlists(
    egress_sinks: "frozenset[str] | set[str] | None",
    policies: "dict[str, list[ValuePredicate]] | None",
) -> list[str]:
    """STRICT-mode obligation: every declared egress sink must carry a destination
    allowlist (an ``enum`` value predicate).

    Rationale: the per-value taint gate on an egress sink is content-derivation —
    sound in the deny direction but with a documented paraphrase residual. An
    ``enum`` allowlist on the destination is content-blind and provenance-
    independent (membership, not derivation), so it closes that residual. In STRICT
    mode we refuse to ship an egress sink that relies on the leaky gate alone:
    returns one error per egress sink without an enum predicate (empty == valid).
    """
    errors: list[str] = []
    sinks = frozenset(egress_sinks or ())
    pol = policies or {}
    for sink in sorted(sinks):
        preds = pol.get(sink, [])
        # Require a NON-EMPTY enum: an enum with an empty allowlist denies every
        # value, so it would satisfy a presence-only check while permanently
        # blocking the sink — structure without substance is not an allowlist.
        if not any(p.kind == "enum" and len(p.allowed) > 0 for p in preds):
            errors.append(
                f"egress sink {sink!r} has no allowlist: STRICT mode requires an "
                f"enum value_policy with at least one allowed destination (the sound, "
                f"paraphrase-proof control); content-derivation alone is not enough"
            )
    return errors


# Kernel-internal intents, not data-flow tools — always exempt from role checks.
_ROLE_EXEMPT = frozenset({"spawn_child", "escalate_policy"})


def _classified_tools(
    *,
    untrusted_sources: "frozenset[str] | set[str] | None" = None,
    sensitive_sources: "frozenset[str] | set[str] | None" = None,
    egress_sinks: "frozenset[str] | set[str] | None" = None,
    positional_sinks: "frozenset[str] | set[str] | None" = None,
    benign_tools: "frozenset[str] | set[str] | None" = None,
    value_policies: "dict[str, list[ValuePredicate]] | None" = None,
) -> frozenset[str]:
    """The set of tools that carry an explicit data-flow role (any taxonomy set, a
    value policy, explicitly benign, or kernel-exempt)."""
    return (
        frozenset(untrusted_sources or ())
        | frozenset(sensitive_sources or ())
        | frozenset(egress_sinks or ())
        | frozenset(positional_sinks or ())
        | frozenset(benign_tools or ())
        | frozenset((value_policies or {}).keys())
        | _ROLE_EXEMPT
    )


def tool_is_classified(
    tool: str,
    *,
    untrusted_sources: "frozenset[str] | set[str] | None" = None,
    sensitive_sources: "frozenset[str] | set[str] | None" = None,
    egress_sinks: "frozenset[str] | set[str] | None" = None,
    positional_sinks: "frozenset[str] | set[str] | None" = None,
    benign_tools: "frozenset[str] | set[str] | None" = None,
    value_policies: "dict[str, list[ValuePredicate]] | None" = None,
) -> bool:
    """True iff ``tool`` has an explicit data-flow role. Used for the per-call
    STRICT obligation on the governor/loop paths, which (unlike GovernedSession) do
    not know the full tool universe at construction time, so they enforce the same
    completeness rule lazily — denying an unclassified tool the moment it is used."""
    return tool in _classified_tools(
        untrusted_sources=untrusted_sources,
        sensitive_sources=sensitive_sources,
        egress_sinks=egress_sinks,
        positional_sinks=positional_sinks,
        benign_tools=benign_tools,
        value_policies=value_policies,
    )


def validate_role_completeness(
    allowed_tools: "frozenset[str] | set[str]",
    *,
    untrusted_sources: "frozenset[str] | set[str] | None" = None,
    sensitive_sources: "frozenset[str] | set[str] | None" = None,
    egress_sinks: "frozenset[str] | set[str] | None" = None,
    positional_sinks: "frozenset[str] | set[str] | None" = None,
    benign_tools: "frozenset[str] | set[str] | None" = None,
    value_policies: "dict[str, list[ValuePredicate]] | None" = None,
) -> list[str]:
    """STRICT-mode obligation: every callable tool has an explicit data-flow role.

    The source side of the taxonomy has a silent default: a tool that is neither
    declared nor matched by the normalizer's heuristics is treated as a *clean*
    read and registers no provenance. So forgetting to mark a secret-reading tool
    means its output is never tainted and the confidentiality floor never arms —
    the symmetric foot-gun to a missing egress allowlist. STRICT closes it by
    refusing any tool whose role is not explicitly declared.

    A tool is "classified" when it appears in any taxonomy set (untrusted /
    sensitive / egress / positional), has a value policy, or is explicitly declared
    benign (a trusted read whose output need not be tainted). Returns one error per
    unclassified tool (empty == valid). ``spawn_child`` and ``escalate_policy`` are
    kernel-internal intents, not data-flow tools, and are always exempt.
    """
    classified = _classified_tools(
        untrusted_sources=untrusted_sources,
        sensitive_sources=sensitive_sources,
        egress_sinks=egress_sinks,
        positional_sinks=positional_sinks,
        benign_tools=benign_tools,
        value_policies=value_policies,
    )
    unclassified = sorted(frozenset(allowed_tools) - classified)
    if not unclassified:
        return []
    return [
        f"tool {tool!r} has no declared data-flow role: STRICT mode requires every "
        f"tool to be classified (untrusted_source / sensitive_source / egress_sink / "
        f"positional_sink / value_policy) or explicitly benign_tools — an "
        f"unclassified read defaults to clean and silently arms no floor"
        for tool in unclassified
    ]


def field_obligation(kind: CodomainKind, mode: ConsumptionMode) -> str:
    """Operator-facing helper: 'predicate' if a sink field can be guarded by a
    decidable value predicate, else 'fuzz' (must ride the fuzz/positional path)."""
    return "predicate" if is_decidable(kind, mode) else "fuzz"
