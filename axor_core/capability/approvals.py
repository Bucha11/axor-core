"""Ready-made escalation approval callbacks.

An ``EscalationCallback`` is the human/operator half of capability-on-demand:
presets whose tool surface is narrower than the full set allow escalation with
``require_human=True``, so nothing is granted until a callback approves the
(tool, paths, max_ops) request. Without a callback every such escalation is
auto-denied (fail-closed).

Signature (see :data:`axor_core.node.intent_loop.EscalationCallback`):

    async (tool_use_id: str, tool: str, paths: list[str], max_ops: int) -> bool

Two implementations ship in core:

- :class:`AllowlistEscalationApprover` — deterministic operator policy:
  approve requests that fall inside a statically configured allowlist.
  No human in the loop at grant time; the operator's authority is the
  configuration itself.
- :func:`console_escalation_callback` — interactive human approval on a TTY;
  denies when no interactive terminal is attached (fail-closed for headless
  deployments).

Both only *answer* the approval question. TTL, max-use, flood-guard and path
enforcement of the resulting grant stay in :class:`EscalationManager` /
``LeaseValidator`` — an over-permissive callback still cannot exceed the
policy's ``escalation_policy`` bounds.
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import PurePosixPath

log = logging.getLogger("axor.capability.approvals")


def _path_within(candidate: str, prefixes: tuple[str, ...]) -> bool:
    """True if candidate resolves under one of the allowed prefixes.
    Pure lexical containment on normalized segments — no filesystem access,
    and ``..`` segments are rejected outright so a crafted path cannot
    escape a prefix it lexically starts with."""
    parts = PurePosixPath(candidate).parts
    if ".." in parts:
        return False
    for prefix in prefixes:
        if parts[: len(PurePosixPath(prefix).parts)] == PurePosixPath(prefix).parts:
            return True
    return False


class AllowlistEscalationApprover:
    """Operator-policy approver: approve escalations inside a static allowlist.

    Args:
        allowed_tools: tool name → maximum ``max_ops`` approvable per grant.
            A requested ``max_ops`` above the cap denies the request (the
            agent may retry with a smaller ask) rather than silently
            shrinking the grant.
        allowed_path_prefixes: when non-empty, every requested path must fall
            under one of these prefixes, and a request with NO paths is
            denied — an unconfined grant is broader than a confined
            configuration allows (fail-closed).

    Every decision is logged so grants remain attributable to the operator
    configuration that produced them.
    """

    def __init__(
        self,
        allowed_tools: dict[str, int],
        allowed_path_prefixes: tuple[str, ...] | list[str] = (),
    ) -> None:
        if not allowed_tools:
            raise ValueError(
                "AllowlistEscalationApprover with no allowed_tools would deny "
                "everything — omit the callback instead"
            )
        bad = {t: cap for t, cap in allowed_tools.items() if cap <= 0}
        if bad:
            raise ValueError(f"allowed_tools caps must be positive: {bad}")
        self._allowed_tools = dict(allowed_tools)
        self._path_prefixes = tuple(allowed_path_prefixes)

    async def __call__(
        self, tool_use_id: str, tool: str, paths: list[str], max_ops: int
    ) -> bool:
        cap = self._allowed_tools.get(tool)
        if cap is None:
            log.info("escalation denied by allowlist: tool %r not allowed", tool)
            return False
        if max_ops > cap:
            log.info(
                "escalation denied by allowlist: tool %r max_ops %d > cap %d",
                tool, max_ops, cap,
            )
            return False
        if self._path_prefixes:
            if not paths:
                log.info(
                    "escalation denied by allowlist: tool %r requested no path "
                    "restriction but approver confines to %r",
                    tool, self._path_prefixes,
                )
                return False
            outside = [p for p in paths if not _path_within(p, self._path_prefixes)]
            if outside:
                log.info(
                    "escalation denied by allowlist: tool %r paths %r outside %r",
                    tool, outside, self._path_prefixes,
                )
                return False
        log.info(
            "escalation approved by allowlist: tool %r paths %r max_ops %d",
            tool, paths, max_ops,
        )
        return True


async def console_escalation_callback(
    tool_use_id: str, tool: str, paths: list[str], max_ops: int
) -> bool:
    """Interactive human approval on the controlling terminal.

    Denies when stdin is not a TTY — a headless deployment must configure an
    operator-policy approver (or none) explicitly, never inherit an approval
    prompt nobody can answer.
    """
    stdin = getattr(sys, "stdin", None)
    try:
        interactive = stdin is not None and stdin.isatty()
    except (AttributeError, ValueError):
        interactive = False
    if not interactive:
        log.info(
            "escalation denied: console approval requested for tool %r but "
            "stdin is not a TTY", tool,
        )
        return False

    scope = f"paths {paths!r}" if paths else "no path restriction"
    prompt = (
        f"[axor] escalation request: grant tool {tool!r} ({scope}, "
        f"max {max_ops} ops)? [y/N] "
    )
    # input() blocks; keep the event loop (and any concurrent governance
    # work) responsive while the human decides.
    answer = await asyncio.to_thread(input, prompt)
    return answer.strip().lower() in ("y", "yes")
