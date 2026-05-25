from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DenialResponse:
    """
    Coarse, worker-facing denial signal.

    Workers receive only this — never layer scores, thresholds,
    trigger features, taint history, or raw policy reasons.
    Full detail is in the trace collector (operator-only access).

    status:           always "denied"
    coarse_category:  one of tool_denied|spawn_denied|export_denied|governance_error
    opaque_decision_id: immutable audit reference; links to trace entry
    """
    status: str
    coarse_category: str
    opaque_decision_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])

    def to_tool_result(self) -> dict:
        """Structured representation returned to executor as tool result."""
        return {
            "error": self.status,
            "category": self.coarse_category,
            "decision_id": self.opaque_decision_id,
        }
