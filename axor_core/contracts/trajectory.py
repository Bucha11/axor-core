"""Stateful trajectory observers — domain pressure that lives in the session.

Some risks are not a property of a single call or a single value, but of the
*session's trajectory*: a stove that has been on too long with no `turn_off`, a
patient whose metric is not improving after a treatment, an agent that keeps
retrying a failing action. These cannot be a content-blind per-call gate or a
``tool -> class`` table — they need state across calls and they read tool *results*.

A ``TrajectoryObserver`` is a stateful, domain-supplied object fed every executed
``(tool, args, result)``. When its accumulated state crosses a domain threshold it
returns a :class:`TrajectorySignal` that **tightens** the session's degradation
level. Two deliberate constraints, both load-bearing for honesty:

- **It can only tighten, never allow.** A trajectory predicate reads world-state
  observations (the stove's minutes, the patient's metric) — a domain heuristic, not
  a sound structural fact. So it rides the same rule as all detection: it may raise
  degradation (narrow the surface, force a human gate on the next risky action), it
  can never authorise one. Wiring it as a hard deny would put a heuristic on the
  enforcement path.
- **It is owned by the domain/agent developer**, who alone knows that "stove" and
  "patient metric" mean. It is code with state, not configuration — which is exactly
  why this class of risk is an extension point and not a YAML knob.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from axor_core.contracts.degradation import DegradationLevel


@dataclass(frozen=True)
class TrajectorySignal:
    """A request to tighten the session to at least ``target_level``. Monotone: the
    engine never lowers a level, so a signal below the current level is a no-op."""

    target_level: DegradationLevel
    reason: str


@runtime_checkable
class TrajectoryObserver(Protocol):
    """A stateful observer of the session trajectory. One instance per session; it
    accumulates state across calls and may tighten degradation."""

    def observe(self, tool: str, args: dict, result: Any) -> "TrajectorySignal | None":
        """Called after every executed tool call with its arguments and result.
        Return a signal to tighten degradation, or ``None``."""
        ...
