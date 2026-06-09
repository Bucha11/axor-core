"""Consequence axis (TM3.1) — deterministic, content-blind sink classification.

`consequence_class(sink)` is the fourth sink-policy projection: how irreversible
the *action* is, independent of its arguments' content or provenance. It is a
**deterministic table lookup keyed on the sink's structural type** (the tool
name, K2; optionally the operation enum), assigned by the operator at sink
registration. It reads the *type of the call, never the arguments' content*, so
it passes T0 (non-interpreting producer) and K3.5 (codomain is a finite enum).

This is what closes the OpenClaw class (X5): a `shutdown`/`restart_gateway`
driven entirely by a trusted user is invisible to the provenance axes
(integrity/confidentiality) — there is nothing to taint — but is CATASTROPHIC by
its action class, so the consequence axis gates it without reading content.

The table is operator-extensible. Unknown sinks default to CONSEQUENTIAL, which
sits at the default unattended ceiling (ExecutionPolicy.max_unattended_consequence
= CONSEQUENTIAL) — i.e. allowed unattended unless an operator lowers the ceiling.
Only sinks classified CATASTROPHIC are gated by default. The honest structural
FP (a benign admin restart gated identically to a malicious one) is accepted by
K0 — the projection is structural and cannot see the semantic difference.
"""

from __future__ import annotations

from axor_core.contracts.canonical import ConsequenceClass

# Keyed on the sink's structural type (tool name, lower-cased). Operator-set at
# registration; this is the coarse default ring.
_CONSEQUENCE_TABLE: dict[str, ConsequenceClass] = {
    # CATASTROPHIC — irreversible infrastructure / power-state / data-destruction.
    "shutdown": ConsequenceClass.CATASTROPHIC,
    "restart": ConsequenceClass.CATASTROPHIC,
    "restart_gateway": ConsequenceClass.CATASTROPHIC,
    "reboot": ConsequenceClass.CATASTROPHIC,
    "poweroff": ConsequenceClass.CATASTROPHIC,
    "power_state_change": ConsequenceClass.CATASTROPHIC,
    "factory_reset": ConsequenceClass.CATASTROPHIC,
    "delete_volume": ConsequenceClass.CATASTROPHIC,
    "drop_database": ConsequenceClass.CATASTROPHIC,
    "wipe": ConsequenceClass.CATASTROPHIC,
    # CONSEQUENTIAL — real-world side effects that are hard (not impossible) to undo.
    "bash": ConsequenceClass.CONSEQUENTIAL,
    "shell": ConsequenceClass.CONSEQUENTIAL,
    "execute": ConsequenceClass.CONSEQUENTIAL,
    "run": ConsequenceClass.CONSEQUENTIAL,
    "transfer": ConsequenceClass.CONSEQUENTIAL,
    "send": ConsequenceClass.CONSEQUENTIAL,
    "deploy": ConsequenceClass.CONSEQUENTIAL,
    "spawn_child": ConsequenceClass.CONSEQUENTIAL,
    # REVERSIBLE — local mutations recoverable from VCS / backups.
    "write": ConsequenceClass.REVERSIBLE,
    "edit": ConsequenceClass.REVERSIBLE,
    "multiedit": ConsequenceClass.REVERSIBLE,
    # BENIGN — observation only.
    "read": ConsequenceClass.BENIGN,
    "search": ConsequenceClass.BENIGN,
    "grep": ConsequenceClass.BENIGN,
    "list": ConsequenceClass.BENIGN,
    "glob": ConsequenceClass.BENIGN,
}

# Optional refinement keyed on the operation enum (NormalizedIntent.operation).
# Lets a generic sink name (e.g. "bash") escalate when its operation is known to
# be power-state-changing. Still content-blind (reads the operation enum, not args).
_OPERATION_OVERRIDE: dict[str, ConsequenceClass] = {
    "power_state_change": ConsequenceClass.CATASTROPHIC,
}

# Unknown sinks: coarse default. Sits at the default ceiling (allowed unattended);
# operators wanting fail-closed lower ExecutionPolicy.max_unattended_consequence.
_DEFAULT = ConsequenceClass.CONSEQUENTIAL


def consequence_class(
    sink: str,
    operation: str | None = None,
    overrides: "dict[str, ConsequenceClass] | None" = None,
) -> ConsequenceClass:
    """Return the ConsequenceClass for a sink (tool name), optionally refined by
    the operation enum. Deterministic; reads no argument content.

    `overrides` is an operator table for custom sinks (the `danger=` profile knob),
    taking precedence over the built-in defaults.
    """
    key = (sink or "").lower()
    if overrides and key in overrides:
        base = overrides[key]
    else:
        base = _CONSEQUENCE_TABLE.get(key, _DEFAULT)
    if operation is not None:
        override = _OPERATION_OVERRIDE.get(operation)
        if override is not None and override > base:
            return override
    return base
