"""axor_core.kernel — the pure decision layer (Ring 0 of Ring 0).

Two families live here:

- The decidability spine the gates rely on (``adjudicator``, ``decidability``,
  ``registration``) — the theory layer that keeps runtime gates sound.
- The trace/replay kernel (``events``, ``state``, ``degradation``, ``replay``)
  — the versioned event schema, the control-plane desired-state lattice, the
  degradation recompute, and the deterministic fold. Imported by BOTH the
  runtime adapter (enforcement) and the platform replay engine; architecture
  rule 0 — shared code, not a shared package.

Everything here is stdlib-only and free of I/O; the purity is pinned by
.importlinter (Ring 0) and tests/kernel/test_kernel_purity.py.
"""
from axor_core.kernel.degradation import compute_level, covered_fact_ids
from axor_core.kernel.errors import InvariantViolation, KernelError, SchemaVersionError
from axor_core.kernel.jcs import CanonicalizationError, canonicalize
from axor_core.kernel.events import (
    SCHEMA_VERSION,
    Event,
    EventKind,
    Fact,
    Verdict,
    event_from_json_line,
    event_to_json_line,
    fact_from_payload,
    fact_to_payload,
)
from axor_core.kernel.state import (
    DesiredState,
    Excision,
    GovernanceState,
    Injection,
    excision_refused_refs,
)

# Replay is exported lazily (PEP 562): it imports axor_core.policy.gates, and
# the gates' value-policy layer imports axor_core.kernel.decidability — an
# eager import here would make that a cycle through this package __init__.
_REPLAY_EXPORTS = {"KernelConfig", "ReplayResult", "ReplayStep", "evaluate_call", "replay"}


def __getattr__(name: str):
    if name in _REPLAY_EXPORTS:
        from axor_core.kernel import replay as _replay

        return getattr(_replay, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SCHEMA_VERSION",
    "Event",
    "EventKind",
    "Fact",
    "Verdict",
    "event_from_json_line",
    "event_to_json_line",
    "fact_from_payload",
    "fact_to_payload",
    "compute_level",
    "covered_fact_ids",
    "DesiredState",
    "Excision",
    "GovernanceState",
    "Injection",
    "excision_refused_refs",
    "KernelConfig",
    "ReplayResult",
    "ReplayStep",
    "evaluate_call",
    "replay",
    "KernelError",
    "SchemaVersionError",
    "InvariantViolation",
    "canonicalize",
    "CanonicalizationError",
]
