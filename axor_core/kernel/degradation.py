"""Degradation as pure recompute (spec decision #9).

``level = max(severity(uncovered facts))``. A fact is covered iff an unrevoked
attestation spans it. No transition table, no partial-descent ambiguity:
coverage changes, the function re-evaluates. Monotone over the fact sequence
(facts only accumulate; attestations are facts too); the level itself may
descend as coverage grows.

This is the trust-model form of degradation used by replay and the control
plane. The runtime :class:`~axor_core.degradation.engine.DegradationEngine`
remains the in-process enforcement machine; its transitions are recorded as
facts, which makes the two agree on the trace. Full runtime adoption of the
recompute is tracked as follow-up work — until then the engine's levels and
this recompute coincide on the recorded fact stream by construction.
"""
from __future__ import annotations

from collections.abc import Mapping

from axor_core.contracts.degradation import DegradationLevel
from axor_core.kernel.events import Fact

ATTESTATION_FACT_TYPE = "operator_attestation"


def covered_fact_ids(facts: Mapping[str, Fact]) -> frozenset[str]:
    """Fact ids spanned by an unrevoked attestation."""
    revoked = {f.revokes for f in facts.values() if f.revokes is not None}
    covered: set[str] = set()
    for f in facts.values():
        if f.fact_type == ATTESTATION_FACT_TYPE and f.fact_id not in revoked:
            covered.update(f.covers)
    return frozenset(covered)


def compute_level(facts: Mapping[str, Fact]) -> DegradationLevel:
    """Pure. Runtime path after every fact append; replay path — same code."""
    covered = covered_fact_ids(facts)
    worst = max(
        (
            f.severity
            for f in facts.values()
            if f.fact_type != ATTESTATION_FACT_TYPE and f.fact_id not in covered
        ),
        default=0,
    )
    return DegradationLevel(min(worst, int(DegradationLevel.TERMINAL)))
