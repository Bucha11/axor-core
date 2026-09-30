"""AdmissionController — the seam the control plane steers through.

Enforcement is local and in-process; the control plane is an advisory overlay
that never enters the decision path (spec 12.0). This Protocol is the entire
surface it touches on the runtime: the IntentLoop asks, at the intent boundary
only, whether it may proceed. A paused node holds here (completing its current
intent first, never mid-effect); a stopped node returns False and the loop
winds down exactly like a cancellation.

Pure contract, no imports — and the implementation is not in this package at
all. ``PlaneAdmission``, and every other plane-specific primitive (the protocol
session, the outbound transport, the trace→event projection), lives in
**axor-wrap** (``axor_wrap.plane``), which depends on axor-core rather than the
other way round. What stays here is only what the kernel reasons over
regardless of whether a plane is ever attached: this contract, the desired-state
lattice and its provenance guard (:mod:`axor_core.kernel.state`), the canonical
byte form commands are signed over (:mod:`axor_core.kernel.jcs`), and the event
schema (:mod:`axor_core.kernel.events`).

That is the packaging expression of spec 12.0: a kernel that *cannot import* a
plane client cannot grow a dependency on one, so "the plane never enters the
decision path" is enforced by the dependency graph, not by review. It is also
why axor-core has zero required dependencies and no network surface.
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
