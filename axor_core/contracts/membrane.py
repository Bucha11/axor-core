from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from axor_core.contracts.context import ContextView
    from axor_core.contracts.taint import TaintState


class SummarizationOutcome(str, Enum):
    SUMMARIZE_WITHOUT_CLEARANCE = "summarize_without_clearance"
    SUMMARIZE_AND_PRESERVE_TAINT = "summarize_and_preserve_taint"
    SUMMARIZE_AND_CLEAR_TAINT_WITH_AUTHORITY = "summarize_and_clear_taint_with_authority"


@dataclass(frozen=True)
class ClearanceEvent:
    """
    Auditable record of a taint clearance that occurred during summarization.

    All audit fields are required. Worker-initiated summarization cannot
    produce a ClearanceEvent — only operator-authority pathways can.
    """
    membrane_id: str
    source_taint_set: frozenset
    operator_authority: str
    timestamp: float = field(default_factory=time.time)
    resulting_taint_state: str = "clean"
    reason_code: str = ""
    audit_id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass(frozen=True)
class SummarizationResult:
    """
    Output of SummarizationMembrane.summarize().

    summary:         compressed representation of the context view
    outcome:         what happened to taint during summarization
    clearance_event: populated only when outcome is CLEAR_TAINT_WITH_AUTHORITY
    """
    summary: str
    outcome: SummarizationOutcome = SummarizationOutcome.SUMMARIZE_AND_PRESERVE_TAINT
    clearance_event: ClearanceEvent | None = None

    def __post_init__(self) -> None:
        if self.outcome == SummarizationOutcome.SUMMARIZE_AND_CLEAR_TAINT_WITH_AUTHORITY:
            if self.clearance_event is None:
                raise ValueError(
                    "ClearanceEvent is required when outcome is CLEAR_TAINT_WITH_AUTHORITY"
                )


@runtime_checkable
class SummarizationMembrane(Protocol):
    """
    Boundary between tainted context and summarized output.

    Worker-initiated summarization MUST default to SUMMARIZE_AND_PRESERVE_TAINT.
    Only operator-authority pathways may produce CLEAR_TAINT_WITH_AUTHORITY.
    """

    async def summarize(
        self,
        context_view: "ContextView",
        taint_state: "TaintState",
        authority: str = "",
    ) -> SummarizationResult:
        """
        Summarize context_view respecting taint_state.

        authority: operator authority token for clearance; empty = no clearance authority.
        """
        ...
