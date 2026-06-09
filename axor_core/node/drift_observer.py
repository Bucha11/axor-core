from __future__ import annotations

import logging

log = logging.getLogger("axor.drift_observer")


class TaintEngineDriftObserver:
    """Implements BehavioralDriftObserver — **probe is a WATCHER**.

    v4.12 register separation: probe (behavioral drift) is detection, never
    enforcement. A drift signal therefore does NOT mutate the enforcing
    TaintEngine / degradation (that would let a probabilistic, out-of-band
    counterfactual gate the live session — TM7 forbids it). It is recorded as
    telemetry; an operator / governance actor consumes it and it may inform the
    offline sentinel cross-session prior, but it does not gate `allow`.

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
