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

from typing import Any

from axor_core.contracts.anomaly import NormalizedIntent
from axor_core.contracts.taint import TaintSource
from axor_core.taint.causal_root import CausalRoot

__all__ = [
    "DECISIVE_NORMALIZED_FIELDS",
    "IncompleteRecord",
    "causal_root_from_record",
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
