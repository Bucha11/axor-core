"""Ready-made escalation approval callbacks.

An ``EscalationCallback`` is the human/operator half of capability-on-demand:
presets whose tool surface is narrower than the full set allow escalation with
``require_human=True``, so nothing is granted until a callback approves the
(tool, paths, max_ops) request. Without a callback every such escalation is
auto-denied (fail-closed).

Signature (see :data:`axor_core.node.intent_loop.EscalationCallback`):

    async (request: ApprovalRequest) -> bool

The request carries its :class:`~axor_core.contracts.approval.AuthorityScope`:

* CLASS — "grant this tool, under these path roots, for N ops". The answer is
  authority over a *class* of calls, which is all a request made before the call
  exists can be.
* EFFECT — "allow THIS call", with its arguments and an ``effect_digest``. The
  resulting grant is one-shot and matches only a call with that digest.

An EFFECT request's ``args`` are raw, attacker-influenceable values: an approver
renders them as data and never follows them as instruction. They reach an approver
only after every structural gate has already admitted the call.

The legacy four-argument signature still works (it is adapted with a
DeprecationWarning) but can only be asked CLASS-scoped questions.

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
policy's ``escalation_policy`` bounds. A callback that raises is treated as a
denial by ``EscalationManager`` (fail-closed exception boundary).
"""
from __future__ import annotations

import asyncio
import logging
import sys

from axor_core.capability.lease_validator import extract_path_arg
from axor_core.contracts.approval import ApprovalRequest
from axor_core.security.paths import paths_within

log = logging.getLogger("axor.capability.approvals")


class AllowlistEscalationApprover:
    """Operator-policy approver: approve escalations inside a static allowlist.

    Args:
        allowed_tools: tool name → maximum ``max_ops`` approvable per grant.
            A requested ``max_ops`` above the cap denies the request (the
            agent may retry with a smaller ask) rather than silently
            shrinking the grant.
        allowed_path_prefixes: when non-empty, every requested path must fall
            under one of these roots, and a request with NO paths is denied —
            an unconfined grant is broader than a confined configuration
            allows (fail-closed). Containment uses the same canonical model
            as lease enforcement (``axor_core.security.paths``): symlinks and
            ``..`` are resolved against the real filesystem, so a link inside
            a root cannot mint a new allowed root outside it.
        unconfined_tools: tools whose calls carry no extractable path argument
            (e.g. ``bash`` — its args are a command string, not a file path).
            A path-restricted lease would deny every call of such a tool, so
            path confinement CANNOT be meaningfully applied to it. Tools
            listed here are grantable only WITHOUT a path restriction: their
            requests must carry an empty ``paths`` (approved unrestricted,
            bounded by max_ops/TTL), and a paths-carrying request is denied.
            Listing a tool here is an explicit operator statement that its
            grants are not path-confined — omit the tool entirely if that is
            not acceptable.

    Every decision is logged so grants remain attributable to the operator
    configuration that produced them.
    """

    def __init__(
        self,
        allowed_tools: dict[str, int],
        allowed_path_prefixes: tuple[str, ...] | list[str] = (),
        unconfined_tools: tuple[str, ...] | list[str] = (),
    ) -> None:
        if not allowed_tools:
            raise ValueError(
                "AllowlistEscalationApprover with no allowed_tools would deny "
                "everything — omit the callback instead"
            )
        bad = {t: cap for t, cap in allowed_tools.items() if cap <= 0}
        if bad:
            raise ValueError(f"allowed_tools caps must be positive: {bad}")
        unknown = set(unconfined_tools) - set(allowed_tools)
        if unknown:
            raise ValueError(
                f"unconfined_tools not present in allowed_tools: {sorted(unknown)}"
            )
        self._allowed_tools = dict(allowed_tools)
        self._path_prefixes = tuple(allowed_path_prefixes)
        self._unconfined_tools = frozenset(unconfined_tools)

    async def __call__(self, request: ApprovalRequest) -> bool:
        if request.is_effect_scoped:
            return self._approve_effect(request)
        return self._approve_class(
            request.tool, list(request.paths), request.max_ops
        )

    def _approve_effect(self, request: ApprovalRequest) -> bool:
        """One concrete call. The path confinement is checked against the call's
        OWN path argument — strictly more precise than a class-scoped grant, where
        only the requested roots are known."""
        tool = request.tool
        cap = self._allowed_tools.get(tool)
        if cap is None:
            log.info("effect approval denied by allowlist: tool %r not allowed", tool)
            return False
        path = extract_path_arg(dict(request.args))
        if tool in self._unconfined_tools:
            # Declared not path-confinable: approvable only without a path to
            # confine, exactly as for a class grant.
            if path:
                log.info(
                    "effect approval denied by allowlist: tool %r is unconfined-only "
                    "but the call carries path %r", tool, path,
                )
                return False
        elif self._path_prefixes:
            if not path:
                log.info(
                    "effect approval denied by allowlist: tool %r call carries no "
                    "path but approver confines to %r", tool, self._path_prefixes,
                )
                return False
            if not paths_within([path], self._path_prefixes):
                log.info(
                    "effect approval denied by allowlist: tool %r path %r does not "
                    "resolve within %r", tool, path, self._path_prefixes,
                )
                return False
        log.info(
            "effect approval granted by allowlist: tool %r effect %s (signature %s)",
            tool, request.effect_digest[:12], list(request.effect_signature),
        )
        return True

    def _approve_class(self, tool: str, paths: list[str], max_ops: int) -> bool:
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
        if tool in self._unconfined_tools:
            # A path-restricted grant for this tool would be unusable (its
            # calls expose no path to check), so only a path-free request is
            # approvable — deny a paths-carrying one and let the agent retry
            # without paths.
            if paths:
                log.info(
                    "escalation denied by allowlist: tool %r is unconfined-only "
                    "but requested paths %r (retry without paths)",
                    tool, paths,
                )
                return False
        elif self._path_prefixes:
            if not paths:
                log.info(
                    "escalation denied by allowlist: tool %r requested no path "
                    "restriction but approver confines to %r",
                    tool, self._path_prefixes,
                )
                return False
            # Canonical containment (symlink/.. resolving) — the SAME model
            # LeaseValidator enforces with, so approval and enforcement cannot
            # disagree about what "inside the workspace" means.
            if not paths_within(paths, self._path_prefixes):
                log.info(
                    "escalation denied by allowlist: tool %r paths %r do not "
                    "resolve within %r",
                    tool, paths, self._path_prefixes,
                )
                return False
        log.info(
            "escalation approved by allowlist: tool %r paths %r max_ops %d",
            tool, paths, max_ops,
        )
        return True


async def console_escalation_callback(request: ApprovalRequest) -> bool:
    """Interactive human approval on the controlling terminal.

    Denies when stdin is not a TTY — a headless deployment must configure an
    operator-policy approver (or none) explicitly, never inherit an approval
    prompt nobody can answer. A terminal error mid-prompt (EOF, closed
    stream) also denies.
    """
    tool, paths, max_ops = request.tool, list(request.paths), request.max_ops
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

    if request.is_effect_scoped:
        # The arguments are untrusted content. They are printed with !r so the
        # human sees them as data (quoted, escaped) rather than as a line of text
        # that could imitate the prompt itself.
        shown = ", ".join(f"{k}={v!r}" for k, v in sorted(request.args.items()))
        prompt = (
            f"[axor] approve THIS call?\n"
            f"       tool:   {tool!r}"
            + (f"  (class: {request.consequence})" if request.consequence else "")
            + f"\n       args:   {shown}\n"
            f"       effect: {request.effect_digest[:16]} (one use only)\n"
            f"  [y/N] "
        )
    else:
        scope = f"paths {paths!r}" if paths else "no path restriction"
        prompt = (
            f"[axor] escalation request: grant tool {tool!r} ({scope}, "
            f"max {max_ops} ops)? [y/N] "
        )
    # input() blocks; keep the event loop (and any concurrent governance
    # work) responsive while the human decides.
    try:
        answer = await asyncio.to_thread(input, prompt)
    except (EOFError, OSError, ValueError):
        log.info("escalation denied: console prompt failed for tool %r", tool)
        return False
    return answer.strip().lower() in ("y", "yes")
