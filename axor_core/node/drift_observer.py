from __future__ import annotations

import logging

log = logging.getLogger("axor.drift_observer")


class BehavioralDriftWatcher:
    """Behavioral-drift watcher — telemetry only, strictly non-enforcing.

    Detection is kept structurally separate from enforcement. This watcher holds
    NO reference to the TaintEngine, degradation state, or any other governance
    object: letting a probabilistic, out-of-band observation reach the live
    allow/deny decision is exactly what this layer must never be able to do.
    A drift signal is recorded as telemetry for an operator or for offline
    cross-session analysis, and nothing more.
    """

    async def on_drift(self, session_id: str, agent_id: str, action: str) -> None:
        # Watcher: observe and report only — no enforcement-state to mutate.
        log.info(
            "behavioral drift observed (watcher, non-enforcing): "
            "session=%s agent=%s action=%s",
            session_id, agent_id, action,
        )
