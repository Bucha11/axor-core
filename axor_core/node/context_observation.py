"""Map a built ContextView into a SessionContextView and fire the context tap.

The glue between the internal context representation (``contracts/context.py``
``ContextView`` / ``ContextFragment``) and the neutral observation seam
(``contracts/observation.py`` ``SessionContextView`` / ``ContextTap``). Kept out of
``node/wrapper.py`` so the pure mapping is unit-testable without a node run.
"""
from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Sequence

from axor_core.contracts.context import ContextView
from axor_core.contracts.observation import ContextTap, SessionContextView

log = logging.getLogger("axor.observation")


def _frame_hash(constraints: tuple[str, ...]) -> str:
    """Stable digest of the governance frame (active constraints) — a proxy for
    'which policy regime this context was assembled under'."""
    return hashlib.sha256("|".join(constraints).encode()).hexdigest()[:16]


def to_session_context_view(
    context: ContextView,
    *,
    session_id: str,
    agent_id: str,
    timestamp: float,
    system_prompt_hash: str,
) -> SessionContextView:
    """Pure mapping ContextView → SessionContextView (deterministic, no I/O).

    Derivations from the visible fragments:
      * ``context_window`` — each fragment as a replayable message dict; a
        ``tool_result`` fragment becomes role "tool", everything else "user".
      * ``turn_index``     — the latest ``fragment.turn`` seen (0 if unknown).
      * ``taint_active``   — any fragment carries a taint canary (``taint_mark``).
      * ``external_read_count`` — number of ``tool_result`` fragments.
    """
    fragments = context.visible_fragments
    return SessionContextView(
        session_id=session_id,
        agent_id=agent_id,
        timestamp=timestamp,
        turn_index=max((f.turn for f in fragments), default=0),
        token_count=context.token_count,
        context_window=tuple(
            {"role": "tool" if f.kind == "tool_result" else "user", "content": f.content}
            for f in fragments
        ),
        system_prompt_hash=system_prompt_hash,
        taint_active=any(f.taint_mark is not None for f in fragments),
        external_read_count=sum(1 for f in fragments if f.kind == "tool_result"),
        taint_canaries=tuple(f.taint_mark for f in fragments if f.taint_mark),
    )


async def emit_context_view(
    taps: "Sequence[ContextTap] | ContextTap | None",
    context: ContextView,
    *,
    session_id: str,
    agent_id: str = "",
) -> None:
    """Fire one context observation, fail-safe. No-op when no tap is attached.

    Supplies the ambient fields (timestamp, governance-frame hash), maps the
    context once, and awaits each tap. A tap is contracted to return promptly;
    we also swallow and log any exception per tap so a misbehaving observer can
    never disturb the governance hot path nor starve the other taps.

    Accepts a single tap or a sequence (GovernedSession's public parameter is
    ``context_taps: list``).
    """
    if taps is None:
        return
    tap_list: tuple["ContextTap", ...] = (
        (taps,) if hasattr(taps, "on_context_event") else tuple(taps)  # type: ignore[arg-type]
    )
    if not tap_list:
        return
    view = to_session_context_view(
        context,
        session_id=session_id,
        agent_id=agent_id,
        timestamp=time.time(),
        system_prompt_hash=_frame_hash(tuple(context.active_constraints)),
    )
    for tap in tap_list:
        try:
            await tap.on_context_event(view)
        except Exception:
            log.warning(
                "context tap %r raised on the hot path — swallowed",
                type(tap).__name__, exc_info=True,
            )
