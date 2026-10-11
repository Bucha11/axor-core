"""The monitor's observation language, read off the gate signatures.

This is the load-bearing definition of the second result. A monitor verdict on a
call `c` reached on history `h` is a function of a *projection* of `c` and `h` —
and the projection is not something this experiment invents: it is exactly the
set of arguments the kernel's gates are called with in
``ToolCallGovernor.evaluate`` (``axor_core/governor.py``), i.e.

    0. STRICT role completeness  — f(tool, config)
    1. consequence_gate          — tool, normalized.operation, ceiling, overrides
    2. value_policy_gate         — per-predicate satisfaction over args
    3. ssrf_gate                 — normalized.target_kind / destination_kind
    4. positional_gate           — tool in positional_sinks, carrier class of args
    5. carrier_gate              — imperative-sink-ness, driving_root.is_tainted,
                                   carrier class of args
    6. taint_gate                — tool roles, normalized destination/write/exec
                                   flags, driving_root (sources, sensitive),
                                   floor_active, enum-supersession coverage

:func:`projection` assembles that tuple. Everything *else* about the call — the
particular identifier, the amount, the message body, which record of a class the
argument names — is outside it, and that is the whole content of the second
result.

The projection is a *claim about this implementation*, so the experiment does not
take it on faith: :func:`pairs.run_suite` checks on every measured pair that
equal projections got equal verdicts, and reports any counterexample as a failed
obligation rather than hiding it. An inequality in the other direction (different
projections, same verdict) is expected and carries no information.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from axor_core.contracts.intent import Intent, IntentKind
from axor_core.governor import ToolCallGovernor
from axor_core.policy.gates import (
    EXFIL_DESTINATIONS,
    INTERNAL_TARGETS,
    integrity_superseded_by_decidable,
)
from axor_core.policy.provenance import derive_driving_root, source_tokens
from axor_core.policy.sinks import is_imperative_sink
from axor_core.security.carrier import classify_carrier
from axor_core.policy.consequence import consequence_class


@dataclass(frozen=True)
class Verdict:
    """What the monitor decided — the quantity Theorem 2 is about."""

    allowed: bool
    category: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"allowed": self.allowed, "category": self.category, "reason": self.reason}


def decide(governor: ToolCallGovernor, tool: str, args: dict[str, Any]) -> Verdict:
    """The real verdict from the real governor. ``evaluate`` is pure (it does not
    touch the ledger), which is what lets the original call and a substitution be
    judged from *the same* governor state — the fixed history Theorem 2 needs."""
    d = governor.evaluate(tool, dict(args))
    return Verdict(allowed=bool(d.allowed), category=d.category, reason=d.reason)


def projection(governor: ToolCallGovernor, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    """π(c, h) — everything the gates read about this call, and nothing else.

    Reaches into the governor's private configuration on purpose: the point is to
    read the projection off the *deployed* monitor rather than off a model of it.
    """
    intent = Intent(
        kind=IntentKind.TOOL_CALL,
        payload={"tool": tool, "args": dict(args)},
        node_id="",
        sequence=0,
    )
    normalized = governor._normalizer.normalize(intent)
    taint = governor._taint
    driving = governor._driving_args.get(tool)
    root = derive_driving_root(
        taint, dict(args), driving, integrity_sink=tool in governor._integrity_sinks,
    )
    # Per-predicate satisfaction: the *outcome* of each operator predicate, not the
    # value it was computed on. A denial reason names the offending value, but the
    # verdict depends only on satisfied/not — which is why two different allowlist
    # members are indistinguishable here.
    preds = (governor._value_policies or {}).get(tool) or []
    pred_outcomes = [
        {
            "arg": getattr(p, "arg", None),
            "kind": getattr(p, "kind", None),
            "satisfied": p.check(dict(args)) is None,
        }
        for p in preds
    ]
    return {
        "tool": tool,
        "operation": normalized.operation,
        "consequence": consequence_class(
            tool, operation=normalized.operation,
            overrides=governor._consequence_overrides, strict=governor._strict,
        ).name,
        "internal_target": normalized.target_kind in INTERNAL_TARGETS,
        "exfil_destination": normalized.destination_kind in EXFIL_DESTINATIONS,
        "destination_kind": normalized.destination_kind,
        "writes_outside_workdir": bool(normalized.writes_outside_workdir),
        "executes_generated_code": bool(normalized.executes_generated_code),
        "imperative_sink": bool(
            is_imperative_sink(tool, normalized) or tool in governor._imperative_sinks
        ),
        "carrier": classify_carrier(dict(args)).value,
        "value_predicates": pred_outcomes,
        "driving_sources": sorted(source_tokens(root.sources)),
        "driving_sensitive": bool(root.sensitive),
        "driving_tainted": bool(root.is_tainted),
        "floor_active": bool(taint.confidentiality_floor_active()),
        # The supersession rule needs *all* driving args present in the call, so the
        # set of argument names (not their values) is part of what the gates read.
        "arg_names": sorted(args or {}),
        "integrity_superseded": bool(
            integrity_superseded_by_decidable(
                tool, dict(args), governor._driving_args, governor._value_policies
            )
        ),
    }


def pi_key(pi: dict[str, Any]) -> str:
    """A comparable key for a projection."""
    return json.dumps(pi, sort_keys=True, default=str)


def pi_diff(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Which observation components differ — the answer to "what does the monitor
    see that separates this pair?" when a substitution *is* refused."""
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
