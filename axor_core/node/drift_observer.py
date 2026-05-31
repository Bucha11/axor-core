from __future__ import annotations

import logging

from axor_core.contracts.taint import TaintScope, TaintSource
from axor_core.taint.engine import TaintEngine

log = logging.getLogger("axor.drift_observer")

# Taint source used for behavioral drift — distinct from external content taint
# so sentinel and trace can distinguish the origin.
_DRIFT_TAINT_SOURCE = TaintSource.UNKNOWN_EXTERNAL


class TaintEngineDriftObserver:
    """
    Implements BehavioralDriftObserver.

    Translates axor-probe behavioral drift signals into TaintEngine propagation.
    The resulting taint is picked up by:
      - IntentLoop taint enforcement (blocks high-risk operations on tainted sessions)
      - SessionSummary.had_taint=True at session end (fed to SentinelCycle.run_once)

    Both ELEVATED_REVIEW and RESTRICTED_MODE propagate SESSION-scoped taint so that
    IntentLoop enforcement and the sentinel had_taint path both activate.
    Callers that want softer ELEVATED_REVIEW handling should guard notify_behavioral_drift
    at the call site rather than expecting scope differentiation here — TaintState's
    SESSION default makes sub-session scope propagation a no-op.
    """

    def __init__(self, taint_engine: TaintEngine) -> None:
        self._taint = taint_engine

    async def on_drift(self, session_id: str, agent_id: str, action: str) -> None:
        self._taint.propagate(_DRIFT_TAINT_SOURCE, TaintScope.SESSION)
        log.info(
            "behavioral drift signal applied: session=%s agent=%s action=%s",
            session_id, agent_id, action,
        )
