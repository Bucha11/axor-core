"""A stateful trajectory observer: the stove has been on too long.

This is the canonical "stateful trajectory" risk class — not a property of any one
call or value, but of the session's history of calls *and their results*. It cannot
be a content-blind per-call gate or a ``tool -> class`` table: it needs to read the
stove's reported state and remember it across calls.

`StoveOnTooLongObserver` watches the result of a `check_stove` call. When the stove
reports on for longer than a threshold, it returns a `TrajectorySignal` that tightens
the session to LOCKED — freezing the risky surface and leaving only read + escalate,
so the next step has to go through a human/operator gate (who can authorise a
remediation or intervene). It only ever *tightens*; it never authorises an action,
because "on too long" is a domain heuristic, not a sound structural fact.

A household agent would register one instance per session:

    session = GovernedSession(..., trajectory_observers=[StoveOnTooLongObserver()])
"""
from __future__ import annotations

from typing import Any

from axor_core.contracts.degradation import DegradationLevel
from axor_core.contracts.trajectory import TrajectorySignal


class StoveOnTooLongObserver:
    """Tighten to LOCKED once the stove has been reported on beyond a threshold.

    Reads ``check_stove`` results of the shape ``{"on": bool, "minutes": int}``.
    A ``turn_off_stove`` call clears the condition (so a later check starts fresh).
    Stateful and session-scoped; one instance per session.
    """

    def __init__(self, threshold_minutes: int = 30) -> None:
        self._threshold = threshold_minutes
        self._fired = False  # don't re-tighten every step once locked

    def observe(self, tool: str, args: dict, result: Any) -> "TrajectorySignal | None":
        if tool == "turn_off_stove":
            self._fired = False
            return None
        if tool != "check_stove" or not isinstance(result, dict):
            return None
        if self._fired:
            return None
        if result.get("on") and int(result.get("minutes", 0)) > self._threshold:
            self._fired = True
            return TrajectorySignal(
                target_level=DegradationLevel.LOCKED,
                reason=(
                    f"stove on for {result.get('minutes')}min (> {self._threshold}) "
                    f"with no intervention — surface frozen, escalate to a human"
                ),
            )
        return None
