"""Deciding from a RECORDED trace, with the kernel's own predicates.

A recorded trace is not a live call. The Control Plane and the Lab both need to
ask "what would the kernel decide here?" over events that were written down
earlier — for export convertibility, for replay, for a counterfactual. Both used
to answer it by reimplementing the taint floor, which is the thing Rule 0 exists
to forbid, and which drifts the moment the kernel changes.

They do not have to. The gate is already a pure predicate over two objects:

    taint_gate(tool, NormalizedIntent, CausalRoot, floor_active, ...)

``CausalRoot`` is labels — ``{sources, sensitive}``, ``is_tainted`` is simply
"carries any external source". A recorded trace stores exactly that per value.
What a recorded trace does NOT store is content, and it does not need to: the
content-derivation work belongs to the LEDGER (``ToolCallGovernor`` building
``CausalRoot``s as a session runs), and by the time an event is written the
ledger's answer is already in it.

So this module is the seam: recorded JSON in, kernel objects out, and the real
gate decides. Nothing here interprets governance.

**Reconstruction fails closed.** A field the recording does not carry is not
defaulted to its permissive value — ``writes_outside_workdir`` absent does not
mean False, it means unknown, and an unknown that could flip a DENY into an
ALLOW is refused (:class:`IncompleteRecord`). A caller that would rather skip
the record than fail must catch it; it must never be able to get a verdict out
of a trace that cannot support one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from axor_core.contracts.anomaly import NormalizedIntent
from axor_core.contracts.taint import TaintSource
from axor_core.taint.causal_root import CausalRoot

__all__ = [
    "DECISIVE_NORMALIZED_FIELDS",
    "ContextRecord",
    "IncompleteRecord",
    "causal_root_from_record",
    "context_driving_root",
    "context_record",
    "normalized_from_record",
]


class IncompleteRecord(ValueError):
    """The recorded event does not carry what a verdict depends on."""

    def __init__(self, missing: list[str], where: str = "") -> None:
        self.missing = list(missing)
        at = f" at {where}" if where else ""
        super().__init__(
            f"recorded intent{at} is missing {sorted(missing)} — these decide a "
            f"taint verdict, and absent is not False. Re-record with the full "
            f"normalized block, or skip this event; it cannot be judged."
        )


# The fields `taint_gate` actually reads off a NormalizedIntent. Absent, each of
# them could turn a recorded DENY into a replayed ALLOW, so each is required.
DECISIVE_NORMALIZED_FIELDS = (
    "destination_kind",
    "writes_outside_workdir",
    "executes_generated_code",
)

# Everything else a NormalizedIntent declares. The gate does not read these, so
# a record may omit them; they are filled with the neutral value a structural
# projection has when nothing was observed.
_NEUTRAL: dict[str, Any] = {
    "operation": "other",
    "target_kind": "workdir",
    "provenance": "unknown",
    "reads_secret_like_data": False,
    "after_external_read": False,
    "after_secret_access": False,
    "data_flow": "none",
}


def normalized_from_record(
    tool: str, normalized: dict[str, Any] | None, *, where: str = ""
) -> NormalizedIntent:
    """Rebuild the structural projection a recorded call carried.

    Raises :class:`IncompleteRecord` when a field the gate decides on is absent.
    """
    record = dict(normalized or {})
    missing = [f for f in DECISIVE_NORMALIZED_FIELDS if f not in record]
    if missing:
        raise IncompleteRecord(missing, where)
    fields = {**_NEUTRAL, **record}
    return NormalizedIntent(
        tool=tool,
        operation=str(fields["operation"]),
        target_kind=str(fields["target_kind"]),
        destination_kind=str(fields["destination_kind"]),
        provenance=str(fields["provenance"]),
        reads_secret_like_data=bool(fields["reads_secret_like_data"]),
        writes_outside_workdir=bool(fields["writes_outside_workdir"]),
        executes_generated_code=bool(fields["executes_generated_code"]),
        after_external_read=bool(fields["after_external_read"]),
        after_secret_access=bool(fields["after_secret_access"]),
        data_flow=str(fields["data_flow"]),
    )


def causal_root_from_record(root: dict[str, Any] | None) -> CausalRoot:
    """Rebuild a value's provenance from its recorded root.

    An unrecognised source name is kept as ``UNKNOWN_EXTERNAL`` rather than
    dropped: forgetting a source is the direction that turns tainted into
    trusted, and over-tainting is the safe one (the causal-root algebra takes
    the union for exactly this reason).
    """
    record = dict(root or {})
    sources = set()
    for name in record.get("sources") or ():
        try:
            sources.add(TaintSource(str(name)))
        except ValueError:
            sources.add(TaintSource.UNKNOWN_EXTERNAL)
    return CausalRoot(sources=frozenset(sources), sensitive=bool(record.get("sensitive")))


# ── context-default integrity (docs/rfc-integrity-context-default.md) ─────────


@dataclass(frozen=True)
class ContextRecord:
    """What a ``integrity_default == "context"`` verdict turned on, from the record.

    The recorded ``driving_root`` is ``driving_carried ⊔ (context_root if any
    driving arg is not a trusted value)``. Keeping the parts lets a consumer
    re-derive it for a different set of driving args, or a different context.
    """

    context_root: CausalRoot
    driving_carried: CausalRoot
    trusted_args: frozenset[str]


def context_record(payload: dict[str, Any], *, where: str = "") -> ContextRecord | None:
    """The context-mode parts of a recorded TOOL_CALL, or ``None`` for a legacy
    (``"clean"``) record.

    Fails closed: a record that says it was decided in context mode but lacks a
    part raises :class:`IncompleteRecord`. Every argument must state whether it
    was a trusted value — absent is not True, since True would drop the context
    root and could turn a recorded DENY into an ALLOW.
    """
    if payload.get("integrity_default") != "context":
        return None
    missing = [k for k in ("context_root", "driving_carried") if k not in payload]
    provenance = payload.get("arg_provenance")
    if not isinstance(provenance, dict):
        missing.append("arg_provenance")
        provenance = {}
    for name in payload.get("args") or {}:
        if "trusted" not in (provenance.get(name) or {}):
            missing.append(f"arg_provenance.{name}.trusted")
    if missing:
        raise IncompleteRecord(missing, where)
    return ContextRecord(
        context_root=causal_root_from_record(payload["context_root"]),
        driving_carried=causal_root_from_record(payload["driving_carried"]),
        trusted_args=frozenset(
            name for name, entry in provenance.items()
            if isinstance(entry, dict) and entry.get("trusted") is True
        ),
    )


def context_driving_root(
    record: ContextRecord,
    args: dict[str, Any],
    drivers: "frozenset[str] | set[str] | list[str] | None",
    *,
    carried: CausalRoot | None = None,
    extra_context: CausalRoot | None = None,
) -> CausalRoot:
    """Re-derive a context-mode driving root, the way ``TaintEngine`` does.

    ``drivers`` selects the driving args exactly as the gate does
    (:func:`~axor_core.policy.gates.driving_subset`). ``carried`` replaces the
    recorded ``driving_carried`` when the caller re-derived it (from value refs,
    under synthetic taint or excision). ``extra_context`` joins more untrusted
    sources into the recorded context root — a counterfactual can only add to
    what the model was shown, never remove it.
    """
    from axor_core.policy.gates import driving_subset

    base = record.driving_carried if carried is None else carried
    context = record.context_root
    if extra_context is not None:
        context = CausalRoot.mint(context, extra_context)
    if not context.is_tainted:
        return base
    names = driving_subset(dict(args or {}), drivers).keys()
    if all(name in record.trusted_args for name in names):
        return base
    return CausalRoot.mint(base, CausalRoot(sources=context.sources))
