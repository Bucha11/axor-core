"""Neutral live-context observation seam (the governance HOT path).

axor-core EMITS a read-only view of a node's context window on each context
build; an external monitor (axor-probe) attaches as a *structural* implementation
of ``ContextTap`` to receive it and build drift snapshots. Core never imports the
monitor (the one-way dependency rule): it defines the vocabulary here and fires
the tap from ``node/wrapper.py`` via ``node/context_observation.emit_context_view``.

This is the COLD seam's sibling: the per-session-close audit seam
(``SessionAuditRecord`` / ``SessionSink``, consumed by axor-sentinel) already
lives in ``contracts/session.py`` and is fired from ``GovernedSession``. This
module is only the hot, per-turn context half, which had no contract.

The payload is raw facts (message-shaped ``context_window``, counts, hashes); the
consumer derives its own structural buckets (the ``CanonicalizedContextSummary``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class SessionContextView:
    """Read-only view of a node's context window at one governance event.

    Emitted on the hot path, so a ``ContextTap`` must return promptly.
    ``context_window`` is message-shaped (``{"role": ..., "content": ...}`` dicts)
    so a consumer can replay it directly. Structural buckets (sensitivity, etc.)
    are derived by the consumer — core hands over raw facts only.
    """
    session_id: str
    agent_id: str
    timestamp: float
    turn_index: int
    token_count: int
    context_window: tuple[dict[str, Any], ...]
    system_prompt_hash: str
    taint_active: bool
    external_read_count: int
    # Canary tokens marking tainted fragments (ContextFragment.taint_mark),
    # propagated through compression. A behavioral monitor (axor-probe) checks
    # whether the agent leaks one of these into its output — the structural
    # disclosure signal for a live health check.
    taint_canaries: tuple[str, ...] = ()


@runtime_checkable
class ContextTap(Protocol):
    """Receiver of live per-turn context events, implemented structurally by an
    external monitor (e.g. axor-probe). Core never imports the implementor.

    Contract: ``on_context_event`` runs on the governance HOT path. It MUST return
    promptly and MUST NOT raise. Heavy work is scheduled out-of-band by the tap;
    core additionally swallows any exception (see
    ``node/context_observation.emit_context_view``) so a misbehaving tap can never
    disturb the governance path.
    """
    async def on_context_event(self, view: SessionContextView) -> None: ...
