"""AdmissionController — the seam the control plane steers through.

Enforcement is local and in-process; the control plane is an advisory overlay
that never enters the decision path (spec 12.0). This Protocol is the entire
surface it touches on the runtime: the IntentLoop asks, at the intent boundary
only, whether it may proceed. A paused node holds here (completing its current
intent first, never mid-effect); a stopped node returns False and the loop
winds down exactly like a cancellation.

Pure contract, no imports — the implementation (axor_core.plane.PlaneAdmission)
lives in Ring 1, the kernel/contract layer stays free of it.
"""
from __future__ import annotations

from typing import Protocol


class AdmissionController(Protocol):
    async def await_admission(self, node_id: str) -> bool:
        """Called at the IntentLoop boundary before each intent is evaluated.

        Return True to admit the next intent. Block (await) while the node is
        paused, returning True once resumed. Return False to stop the loop —
        the node finishes/aborts cleanly and admits no further intents.
        Must be safe to call when the plane is disconnected: a best-effort
        overlay never blocks governance, so a disconnected controller admits.
        """
        ...
