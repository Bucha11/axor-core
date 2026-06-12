"""The stateless gate predicates, shared by every enforcement path.

These are the content-blind and per-value gates of the kernel's decision
sequence — consequence, value policies, SSRF, positional admission, carrier, and
per-value taint with the confidentiality floor. Each is a pure function: it takes
the already-computed facts about a call (its normalized form, the driving
argument's provenance, the active ceiling/taxonomy) and returns either ``None``
(this gate passes) or a :class:`GateDecision` (deny, with a reason and a coarse
category).

They live here, in one place, so the streaming :class:`~axor_core.node.intent_loop.IntentLoop`
and the synchronous :class:`~axor_core.governor.ToolCallGovernor` cannot drift:
both orchestrate these exact predicates, each adding its own surrounding state
(capability, degradation, leases, execution) and side effects (trace, telemetry).
A gate's *logic* exists once; the orchestration differs.

Nothing here computes provenance or reads the taint engine — the caller derives
``driving_root`` and ``floor_active`` and passes them in, so this module stays a
pure decision layer with no engine dependency.
"""
from __future__ import annotations

from dataclasses import dataclass

from axor_core.contracts.anomaly import NormalizedIntent
from axor_core.contracts.canonical import ConsequenceClass
from axor_core.contracts.taint import Carrier
from axor_core.policy.consequence import consequence_class
from axor_core.policy.sinks import is_imperative_sink
from axor_core.policy.value_policy import check_value_policies
from axor_core.security.carrier import classify_carrier
from axor_core.taint.causal_root import CausalRoot

# Internal destinations no agent should reach unless policy explicitly allows it.
INTERNAL_TARGETS = ("cloud_metadata", "private_network", "docker_socket")
# Destinations that constitute egress of data out of the trust boundary.
EXFIL_DESTINATIONS = ("cloud_metadata", "private_network", "external_domain")


def driving_subset(args: dict, keys: "frozenset[str] | set[str] | list[str] | None") -> dict:
    """The portion of a call's arguments the per-value taint decision keys on.

    By default the whole argument blob drives the decision (safe but coarse: a
    tainted *body* makes a ``send_email`` look tainted even when the recipient is
    trusted). When an operator declares a sink's *driving arguments* — the fields
    that actually carry the destination / instruction (e.g. ``to`` for an email,
    ``task`` for a spawn) — the taint check narrows to just those, so untrusted
    *content* flowing to a *trusted* destination is no longer over-blocked.

    Soundness boundary — narrowing trades coverage for precision, and it is sound
    only when the driving field is itself constrained:
      • The confidentiality floor is unaffected (content-blind, session-wide), so a
        *secret* still cannot leak through a non-driving field.
      • An attacker-chosen *destination* is still caught: it lands in a driving
        field, which the gate does check.
      • What narrowing gives up is the integrity/carrier check on *non-driving*
        fields, so untrusted (non-secret) content can reach the sink through, e.g.,
        an email ``body`` while ``to`` is clean. That is acceptable only when the
        driving destination is allowlisted (the recipient cannot be attacker-
        chosen) — which STRICT mode enforces via
        :func:`~axor_core.kernel.registration.validate_driving_arg_allowlists`.
        In production mode without an allowlist it is a documented residual.

    Fail-safe: if driving keys are declared but none are present in this call, fall
    back to the whole blob rather than silently un-gating, so a destination smuggled
    into a *wholly undeclared* field still gets caught. (This does not cover the
    case above, where a declared driving key IS present and clean while a different
    field carries the content — hence the allowlist obligation in STRICT.)"""
    if not keys:
        return args
    subset = {k: args[k] for k in keys if k in args}
    return subset if subset else args


@dataclass(frozen=True)
class GateDecision:
    """A gate's deny. ``reason`` is the operator-facing explanation; ``category``
    is the coarse label recorded in the trace (e.g. ``taint_enforcement``)."""

    reason: str
    category: str


def consequence_gate(
    tool_name: str,
    operation: str | None,
    ceiling: ConsequenceClass,
    overrides: dict | None = None,
    has_governance_gate: bool = False,
) -> GateDecision | None:
    """Content-blind action-class gate. Deny if the sink's irreversibility exceeds
    the unattended ceiling and no governance/human gate (escalation or lease)
    covers it."""
    cls = consequence_class(tool_name, operation=operation, overrides=overrides)
    if cls <= ceiling:
        return None
    if has_governance_gate:
        return None
    return GateDecision(
        reason=(
            f"consequence gate: sink '{tool_name}' is {cls.name}, exceeding the "
            f"unattended ceiling {ceiling.name}; a governance/human gate is required"
        ),
        category="consequence_gate",
    )


def value_policy_gate(
    tool_name: str, args: dict, value_policies: dict | None
) -> GateDecision | None:
    """Operator-registered decidable predicates over an argument projection
    (amount in range, target in an allowed set)."""
    reason = check_value_policies(tool_name, args, value_policies or {})
    if reason is None:
        return None
    return GateDecision(reason=reason, category="value_policy")


def ssrf_gate(tool_name: str, normalized: NormalizedIntent) -> GateDecision | None:
    """Content-blind, taint-independent block on internal destinations (cloud
    metadata, private network, docker socket)."""
    if (
        normalized.target_kind in INTERNAL_TARGETS
        or normalized.destination_kind in ("cloud_metadata", "private_network")
    ):
        return GateDecision(
            reason=(
                f"ssrf gate: '{tool_name}' targets an internal destination "
                f"({normalized.target_kind}/{normalized.destination_kind}) — "
                f"blocked independent of taint"
            ),
            category="ssrf_gate",
        )
    return None


def positional_gate(
    tool_name: str, args: dict, positional_sinks: frozenset[str] | set[str]
) -> GateDecision | None:
    """For a sink declared instruction-incomplete, admit only an
    instruction-incomplete carrier — independent of content-derivation."""
    if tool_name in positional_sinks and classify_carrier(args) == Carrier.FREE_TEXT:
        return GateDecision(
            reason=(
                f"positional gate: '{tool_name}' is a declared "
                f"instruction-incomplete sink; its driving value is FREE_TEXT "
                f"(non-positional) — admitted only via an instruction-incomplete "
                f"carrier, independent of content-derivation"
            ),
            category="positional_gate",
        )
    return None


def carrier_gate(
    tool_name: str,
    args: dict,
    normalized: NormalizedIntent,
    driving_root: CausalRoot,
    imperative_sinks: frozenset[str] | set[str] = frozenset(),
) -> GateDecision | None:
    """An untrusted FREE_TEXT value reaching an instruction-following sink is the
    imperative channel."""
    imperative = is_imperative_sink(tool_name, normalized) or tool_name in imperative_sinks
    if driving_root.is_tainted and imperative and classify_carrier(args) == Carrier.FREE_TEXT:
        return GateDecision(
            reason=(
                f"carrier gate: untrusted FREE_TEXT value into the "
                f"instruction-following sink '{tool_name}' (imperative channel)"
            ),
            category="carrier_gate",
        )
    return None


def taint_gate(
    tool_name: str,
    normalized: NormalizedIntent,
    driving_root: CausalRoot,
    floor_active: bool,
    egress_sinks: frozenset[str] | set[str] = frozenset(),
) -> GateDecision | None:
    """Per-value taint: integrity (untrusted-derived value into a high-risk
    operation) plus the confidentiality floor (egress while a secret read is
    outstanding). ``egress_sinks`` is the operator's declaration of which tools
    exfiltrate; it complements the normalizer's destination classification."""
    exfil = (
        tool_name in egress_sinks
        or normalized.destination_kind in EXFIL_DESTINATIONS
    )
    integrity_risk = driving_root.is_tainted and (
        normalized.writes_outside_workdir
        or normalized.executes_generated_code
        or exfil
    )
    confidentiality_risk = exfil and floor_active
    if not (integrity_risk or confidentiality_risk):
        return None
    axis = (
        "confidentiality (egress under the sound floor — a secret read is "
        "outstanding; release requires governance endorsement)"
        if confidentiality_risk
        else "integrity (untrusted-derived value into a high-risk operation)"
    )
    return GateDecision(
        reason=(
            f"taint enforcement (per-value): the driving argument of "
            f"'{tool_name}' carries a tainted/sensitive value — {axis}"
        ),
        category="taint_enforcement",
    )
