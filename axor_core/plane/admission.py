"""PlaneAdmission — drives an IntentLoop from a PlaneSession's posture.

Implements the AdmissionController contract over a :class:`PlaneSession` whose
state is updated out-of-band by the plane client's SSE subscription. Pausing
holds here (the node has already finished its current intent — the boundary is
between intents); stopping returns False so the loop winds down like a
cancellation. A disconnected plane never blocks: the session simply stays in
its last posture, and with no pause/stop set the node runs under local config.
"""
from __future__ import annotations

import asyncio

from axor_core.plane.session import PlaneSession


class PlaneAdmission:
    def __init__(self, session: PlaneSession, poll_interval: float = 0.05) -> None:
        self._session = session
        self._poll = poll_interval
        # Pulsed by notify() when the session posture changes, so a paused loop
        # resumes promptly instead of waiting out the poll interval.
        self._changed = asyncio.Event()

    def notify(self) -> None:
        """Signal that the session posture changed (call after applying a
        delta). Safe to call from the client's SSE task."""
        self._changed.set()

    async def await_admission(self, node_id: str) -> bool:
        while True:
            if self._session.stopped:
                return False
            if not self._session.paused:
                return True
            self._changed.clear()
            try:
                await asyncio.wait_for(self._changed.wait(), self._poll)
            except TimeoutError:
                pass  # re-poll: posture may have changed without a notify()
