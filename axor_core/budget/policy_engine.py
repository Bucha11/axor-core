from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from axor_core.budget.estimator import BudgetEstimator
from axor_core.budget.tracker import BudgetTracker
from axor_core.contracts.context import ContextView
from axor_core.contracts.envelope import ExecutionEnvelope
from axor_core.contracts.policy import (
    ExecutionPolicy,
    CompressionMode,
    ChildMode,
    ExportMode,
)


class OptimizationAction(str, Enum):
    PROCEED          = "proceed"           # execute as-is
    COMPRESS_CONTEXT = "compress_context"  # tighten compression before proceeding
    DENY_CHILD       = "deny_child"        # deny spawn_child — budget pressure
    RESTRICT_EXPORT  = "restrict_export"   # downgrade export mode


@dataclass(frozen=True)
class OptimizationDecision:
    action: OptimizationAction
    reason: str
    suggested_compression: CompressionMode | None = None
    suggested_export: ExportMode | None = None


class BudgetPolicyEngine:
    """
    Real-time optimizer. Fires on every execution event.

    Principle: minimum sufficient for quality — not a hard cap.

    The engine does not decompose tasks.
    The engine does not block execution.
    The engine suggests what to tighten on the next step.

    Three moments it fires:
        on_intent_arrived()    — before envelope is built for this intent
        on_result_arrived()    — after tool result, before writing to context
        on_child_requested()   — before child node is created

    Each returns an OptimizationDecision that the node acts on.
    """

    # Token thresholds for optimization signals
    # These are session-level totals, not per-node caps
    _COMPRESS_THRESHOLD   = 0.60   # at 60% → suggest compression
    _DENY_CHILD_THRESHOLD = 0.80   # at 80% → deny new children
    _RESTRICT_EXPORT      = 0.90   # at 90% → downgrade export mode
    _HARD_STOP_THRESHOLD  = 0.95   # at 95% → cancel active node

    def __init__(
        self,
        tracker: BudgetTracker,
        estimator: BudgetEstimator,
        soft_limit: int | None = None,
    ) -> None:
        self._tracker   = tracker
        self._estimator = estimator
        # soft_limit is advisory — not a hard cap
        # None means no budget guidance, always PROCEED
        self._soft_limit = soft_limit

    def on_intent_arrived(
        self,
        envelope: ExecutionEnvelope,
        tool_count: int,
    ) -> OptimizationDecision:
        """
        Before executing an intent — should we tighten context?
        """
        if self._soft_limit is None:
            return _proceed("no soft limit set")

        spent  = self._tracker.total_tokens()
        ratio  = spent / self._soft_limit

        # hard stop — cancel the active node via token
        if ratio >= self._HARD_STOP_THRESHOLD:
            from axor_core.contracts.cancel import CancelReason
            envelope.cancel_token.cancel(
                CancelReason.BUDGET_EXHAUSTED,
                detail=f"spent {ratio:.0%} of soft limit",
            )
            return OptimizationDecision(
                action=OptimizationAction.PROCEED,
                reason=f"hard stop triggered at {ratio:.0%} — cancel token fired",
            )

        if ratio >= self._COMPRESS_THRESHOLD:
            headroom = self._estimator.compression_headroom(
                envelope.context, envelope.policy
            )
            if headroom > 0.2:
                return OptimizationDecision(
                    action=OptimizationAction.COMPRESS_CONTEXT,
                    reason=f"spent {ratio:.0%} of soft limit — compress context",
                    suggested_compression=CompressionMode.AGGRESSIVE,
                )

        return _proceed(f"spent {ratio:.0%} of soft limit — ok")

    def on_result_arrived(
        self,
        node_id: str,
        result_token_estimate: int,
        policy: ExecutionPolicy,
    ) -> OptimizationDecision:
        """
        After tool result arrives — should we compress before writing to context?
        Result compression happens here, not in the executor.
        """
        if self._soft_limit is None:
            return _proceed("no soft limit set")

        projected = self._tracker.total_tokens() + result_token_estimate
        ratio = projected / self._soft_limit

        if ratio >= self._RESTRICT_EXPORT:
            return OptimizationDecision(
                action=OptimizationAction.RESTRICT_EXPORT,
                reason=f"projected {ratio:.0%} of soft limit — restrict export",
                suggested_export=ExportMode.SUMMARY,
            )

        if ratio >= self._COMPRESS_CONTEXT:
            return OptimizationDecision(
                action=OptimizationAction.COMPRESS_CONTEXT,
                reason=f"projected {ratio:.0%} — compress result before context write",
                suggested_compression=CompressionMode.AGGRESSIVE,
            )

        return _proceed(f"projected {ratio:.0%} — ok")

    def on_child_requested(
        self,
        parent_envelope: ExecutionEnvelope,
        child_task: str,
    ) -> OptimizationDecision:
        """
        Before spawning a child node — is the budget healthy enough?

        Does not block the agent's decision to spawn.
        Returns DENY_CHILD only under severe budget pressure.
        The node may still override this if task quality requires it.
        """
        if self._soft_limit is None:
            return _proceed("no soft limit set")

        spent = self._tracker.total_tokens()
        ratio = spent / self._soft_limit

        if ratio >= self._DENY_CHILD_THRESHOLD:
            return OptimizationDecision(
                action=OptimizationAction.DENY_CHILD,
                reason=f"spent {ratio:.0%} of soft limit — deny child to preserve budget",
            )

        # also check if the child's context slice will be sufficient
        slice_tokens = self._estimator.estimate_child_slice_tokens(
            parent_context=parent_envelope.context,
            fraction=parent_envelope.policy.child_context_fraction,
        )
        sufficient = self._estimator.is_slice_sufficient(
            child_task=child_task,
            slice_token_estimate=slice_tokens,
            parent_context=parent_envelope.context,
        )

        if not sufficient:
            # slice too small for quality — suggest compression to free up headroom
            return OptimizationDecision(
                action=OptimizationAction.COMPRESS_CONTEXT,
                reason="child context slice may be insufficient — compress parent context first",
                suggested_compression=CompressionMode.AGGRESSIVE,
            )

        return _proceed(f"child approved — spent {ratio:.0%}, slice sufficient")

    @property
    def _COMPRESS_CONTEXT(self) -> float:
        return self._COMPRESS_THRESHOLD


def _proceed(reason: str) -> OptimizationDecision:
    return OptimizationDecision(
        action=OptimizationAction.PROCEED,
        reason=reason,
    )
