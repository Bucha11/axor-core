"""Consequence axis — deterministic, content-blind sink classification.

`consequence_class(sink)` is the sink-policy projection that measures how
irreversible the *action* is, independent of its arguments' content or
provenance. It is a deterministic table lookup keyed on the sink's structural
type (the tool name; optionally the operation enum), assigned by the operator at
sink registration. It reads the *type of the call, never the arguments'
content*, so it never interprets governed data and its result is a finite enum.

This is what catches a destructive infrastructure action under trusted
provenance: a `shutdown`/`restart_gateway` driven entirely by a trusted user is
invisible to the provenance axes (integrity/confidentiality) — there is nothing
to taint — but is CATASTROPHIC by its action class, so the consequence axis
gates it without reading content.

The table is operator-extensible. Unknown sinks default to CONSEQUENTIAL, which
sits at the default unattended ceiling (ExecutionPolicy.max_unattended_consequence
= CONSEQUENTIAL) — i.e. allowed unattended unless an operator lowers the ceiling.
Only sinks classified CATASTROPHIC are gated by default. The honest structural
false positive (a benign admin restart gated identically to a malicious one) is
accepted: the projection is structural and cannot see the semantic difference.

Exact-name lookup alone is defeated by a rename: ``shutdown_server`` or
``drop_table`` is not a table key and fell to the CONSEQUENTIAL default, i.e.
unattended. Two layers close that:

- **Name-token heuristic (every mode, tightening only).** An unknown name is split
  into tokens (``shutdown_server`` / ``shutdownServer`` → ``shutdown``,
  ``server``); a catastrophic verb (``shutdown``, ``wipe``, ...) or a destructive
  verb paired with an infrastructure object (``drop`` + ``table``, ``restart`` +
  ``server``) raises it to CATASTROPHIC. It can only raise a class, never lower
  one, and never overrides the table or an operator's explicit class. It is a
  heuristic — a name that avoids the vocabulary still passes — so it is not what
  STRICT relies on.
- **STRICT: unknown means CATASTROPHIC.** Under STRICT a sink with no explicit
  class (neither a table key nor an operator override) is CATASTROPHIC: it needs a
  human/operator gate. ``GovernedSession`` fails construction for a registered
  tool without one (``kernel.registration.validate_consequence_completeness``),
  the same way it does for a tool without a data-flow role.
"""

from __future__ import annotations

import re

from axor_core.contracts.canonical import ConsequenceClass

# Keyed on the sink's structural type (tool name, lower-cased). Operator-set at
# registration; these are the coarse defaults.
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
# be power-state-changing. Still content-blind: reads the operation enum, not args.
_OPERATION_OVERRIDE: dict[str, ConsequenceClass] = {
    "power_state_change": ConsequenceClass.CATASTROPHIC,
}

# Unknown sinks: coarse default. Sits at the default ceiling (allowed unattended);
# operators wanting fail-closed lower ExecutionPolicy.max_unattended_consequence.
_DEFAULT = ConsequenceClass.CONSEQUENTIAL

# Kernel-internal intents are not tools an operator registers; they are never
# "unclassified" under STRICT.
_KERNEL_INTERNAL = frozenset({"spawn_child", "escalate_policy", "escalate"})

# Name-token heuristic (tightening only). A catastrophic verb on its own, or a
# destructive verb together with an infrastructure object, anywhere in the name.
_CATASTROPHIC_TOKENS = frozenset({"shutdown", "poweroff", "reboot", "wipe"})
_DESTRUCTIVE_VERBS = frozenset({
    "drop", "delete", "destroy", "truncate", "purge", "erase", "format", "nuke",
})
_INFRA_OBJECTS = frozenset({
    "database", "databases", "db", "table", "tables", "schema", "volume", "disk",
    "bucket", "cluster", "namespace", "keyspace", "partition", "all",
})
_RESTART_VERBS = frozenset({"restart", "reset", "power"})
_RESTART_OBJECTS = frozenset({
    "server", "host", "machine", "gateway", "node", "system", "cluster", "device",
    "off", "cycle", "state", "factory",
})
_CAMEL = re.compile(r"([a-z0-9])([A-Z])")
_SPLIT = re.compile(r"[^a-z0-9]+")


def _name_tokens(sink: str) -> frozenset[str]:
    return frozenset(t for t in _SPLIT.split(_CAMEL.sub(r"\1_\2", sink or "").lower()) if t)


def _heuristic_class(sink: str) -> ConsequenceClass | None:
    """CATASTROPHIC if the name's tokens say so, else None. Structural: reads only
    the tool name."""
    tokens = _name_tokens(sink)
    if (
        tokens & _CATASTROPHIC_TOKENS
        or (tokens & _DESTRUCTIVE_VERBS and tokens & _INFRA_OBJECTS)
        or (tokens & _RESTART_VERBS and tokens & _RESTART_OBJECTS)
        or {"rm", "rf"} <= tokens
        or {"factory", "reset"} <= tokens
    ):
        return ConsequenceClass.CATASTROPHIC
    return None


def is_consequence_classified(
    sink: str, overrides: "dict[str, ConsequenceClass] | None" = None,
) -> bool:
    """True iff the sink has an explicit class: an operator override or a built-in
    table key (kernel-internal intents count as classified). The name heuristic
    does not classify — it only tightens."""
    key = (sink or "").lower()
    return (
        key in _KERNEL_INTERNAL
        or key in _CONSEQUENCE_TABLE
        or bool(overrides and key in overrides)
    )


def consequence_class(
    sink: str,
    operation: str | None = None,
    overrides: "dict[str, ConsequenceClass] | None" = None,
    *,
    strict: bool = False,
) -> ConsequenceClass:
    """Return the ConsequenceClass for a sink (tool name), optionally refined by
    the operation enum. Deterministic; reads no argument content.

    `overrides` is an operator table for custom sinks (the `danger=` profile knob),
    taking precedence over the built-in defaults and the name heuristic. A sink
    with no explicit class gets the default, raised by the name heuristic; under
    ``strict`` it is CATASTROPHIC.
    """
    key = (sink or "").lower()
    if overrides and key in overrides:
        base = overrides[key]
    elif key in _CONSEQUENCE_TABLE:
        base = _CONSEQUENCE_TABLE[key]
    elif key in _KERNEL_INTERNAL:
        base = _DEFAULT
    elif strict:
        base = ConsequenceClass.CATASTROPHIC
    else:
        base = _DEFAULT
        heuristic = _heuristic_class(sink)
        if heuristic is not None and heuristic > base:
            base = heuristic
    if operation is not None:
        override = _OPERATION_OVERRIDE.get(operation)
        if override is not None and override > base:
            return override
    return base
