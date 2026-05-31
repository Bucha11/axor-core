from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class BehavioralDriftObserver(Protocol):
    """
    Receives behavioral drift signals from axor-probe.

    axor-core defines this protocol; axor-probe calls it via its own
    structurally-compatible CoreDriftSink protocol.
    Core never imports axor-probe — dependency direction is strictly one-way (P-34).

    Wiring:
        observer = TaintEngineDriftObserver(session._taint_engine)
        # pass observer to ProbePipeline integrations via CoreDriftSink callback
        await notify_core(drift_signal, observer)

    action values mirror DriftAction.value:
        "elevated_review"  — repeated pattern; propagates NODE-scoped taint
        "restricted_mode"  — strong longitudinal signal; propagates SESSION-scoped taint
    """

    async def on_drift(self, session_id: str, agent_id: str, action: str) -> None:
        """
        Called when axor-probe detects drift at or above ELEVATED_REVIEW.

        Must not raise — failures must be caught and logged by the caller.
        session_id: axor-core session identifier (matches GovernedSession.session_id())
        agent_id:   agent identifier from the RuntimeEvent
        action:     "elevated_review" | "restricted_mode"
        """
        ...
