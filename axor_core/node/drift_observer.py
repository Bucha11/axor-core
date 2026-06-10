from __future__ import annotations

import logging

log = logging.getLogger("axor.drift_observer")


class TaintEngineDriftObserver:
    """Behavioral-drift observer — this probe is a WATCHER only.

    Detection is kept strictly separate from enforcement. A drift signal does
    NOT mutate the enforcing TaintEngine or degradation state: letting a
    probabilistic, out-of-band observation gate a live session is exactly what
    this layer must not do. The signal is recorded as telemetry for an operator
    or governance actor to consume, and may inform offline cross-session
    analysis, but it never affects the live allow/deny decision.

    Constructed with an optional TaintEngine for backward-compatible wiring; the
    engine is intentionally left untouched.
    """

    def __init__(self, taint_engine=None) -> None:
        self._taint = taint_engine  # retained for wiring compat; not mutated

    async def on_drift(self, session_id: str, agent_id: str, action: str) -> None:
        # Watcher: observe and report only — no enforcement-state mutation.
        log.info(
            "behavioral drift observed (watcher, non-enforcing): "
            "session=%s agent=%s action=%s",
            session_id, agent_id, action,
        )
