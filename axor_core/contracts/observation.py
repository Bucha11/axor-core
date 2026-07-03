from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class SessionContextView:
    """
    Read-only structural snapshot of a session's context, pushed to ContextTap
    observers on each governance-path context event (one per GovernedSession.run
    turn).

    This is an observation contract, not an enforcement input: taps receive the
    view after the turn's governance decisions are already made, and nothing a
    tap does can reach the allow/deny path. Consumers (axor-probe) match this
    shape structurally without importing axor-core (P-34) — field renames here
    are breaking changes for them.

    context_window is provider-shaped message dicts ({"role": ..., "content":
    ...}) derived from the session's shaped ContextFragments — never raw
    executor history. taint_canaries are the distinct ContextFragment.taint_mark
    tokens live in the context; a probe can check whether the agent leaks one
    into its output (the clean shadow never holds them).
    """
    session_id: str
    agent_id: str
    timestamp: float
    turn_index: int
    token_count: int
    context_window: tuple[dict[str, object], ...]
    system_prompt_hash: str          # hash only — never plain text
    taint_active: bool
    external_read_count: int         # externally-sourced values registered this session
    taint_canaries: tuple[str, ...] = field(default_factory=tuple)


@runtime_checkable
class ContextTap(Protocol):
    """
    Receives SessionContextView events from a GovernedSession.

    axor-core defines this protocol; axor-probe implements it structurally
    (CoreContextTap) without importing axor-core — dependency direction is
    strictly one-way (P-34).

    on_context_event MUST return promptly and must not block the governance
    path: schedule any expensive work out-of-band (e.g. asyncio.create_task).
    The session catches and logs tap exceptions — a failing observer never
    disturbs execution — but it cannot defend against a tap that blocks.
    """

    async def on_context_event(self, view: SessionContextView) -> None:
        """Called after each run() turn with the current context view."""
        ...
