"""Preset protection profiles — the product-facing surface.

A user wraps their agent and picks ONE profile (+ optional workspace and a small
`danger` table for custom tools). A profile is a named bundle of existing knobs
(ExecutionMode, consequence ceiling, escalation, isolation, watcher) — no new
mechanism. `profile` is a parameter on GovernedSession, not a new entry method.
"""

from __future__ import annotations

from dataclasses import dataclass

from axor_core.contracts.canonical import ConsequenceClass
from axor_core.contracts.mode import ExecutionMode
from axor_core.contracts.policy import EscalationPolicy


@dataclass(frozen=True)
class Profile:
    name: str
    mode: ExecutionMode
    consequence_ceiling: ConsequenceClass | None  # lower = stricter; None → policy default
    require_isolation: bool
    escalation_policy: EscalationPolicy | None
    attach_watcher: bool
    # Sinks the operator declares instruction-incomplete, admitted by positional
    # carrier rather than content-derivation. Opt-in; never an instruction-complete
    # (exec-class) sink. Empty by default — a profile knob.
    positional_sinks: frozenset[str] = frozenset()


_HUMAN_ESCALATION = EscalationPolicy(
    allow_escalation=True, require_human=True, max_escalations=3, max_ops_per_grant=10
)
_AUTO_ESCALATION = EscalationPolicy(
    allow_escalation=True, require_human=False, max_escalations=5, max_ops_per_grant=20
)


PROFILES: dict[str, Profile] = {
    "observe": Profile("observe", ExecutionMode.OBSERVE, ConsequenceClass.CATASTROPHIC,
                       False, _AUTO_ESCALATION, True),
    "balanced": Profile("balanced", ExecutionMode.PRODUCTION, ConsequenceClass.CONSEQUENTIAL,
                        False, _HUMAN_ESCALATION, True),
    "strict": Profile("strict", ExecutionMode.STRICT, ConsequenceClass.REVERSIBLE,
                      True, _HUMAN_ESCALATION, True),
    "dev": Profile("dev", ExecutionMode.LIBRARY, ConsequenceClass.CATASTROPHIC,
                   False, _AUTO_ESCALATION, False),
}

DEFAULT_PROFILE = "balanced"


def resolve_profile(profile: "str | Profile") -> Profile:
    if isinstance(profile, Profile):
        return profile
    try:
        return PROFILES[profile]
    except KeyError:
        raise ValueError(
            f"unknown profile {profile!r}; choose one of {sorted(PROFILES)} or pass a Profile"
        ) from None
