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


@dataclass(frozen=True)
class BudgetThresholds:
    """
    Fractions of soft_limit at which the policy engine fires actions.

    Defaults are heuristics, not measurements. Tune for your workload:
      - Long research tasks with broad context: lower compress, lower hard_stop
        (e.g. 0.50 / 0.70 / 0.85 / 0.95) so headroom is preserved earlier.
      - Interactive REPL with short turns: raise everything
        (e.g. 0.70 / 0.85 / 0.95 / 0.99) so a chatty session isn't throttled.

    Order invariant (validated): compress < deny_child < restrict_export < hard_stop.
    """
    compress: float      = 0.60
    deny_child: float    = 0.80
    restrict_export: float = 0.90
    hard_stop: float     = 0.95

    def __post_init__(self) -> None:
        ordered = (self.compress, self.deny_child, self.restrict_export, self.hard_stop)
        if not all(0.0 < v <= 1.0 for v in ordered):
            raise ValueError("budget thresholds must be in (0, 1]")
        if not (ordered[0] < ordered[1] < ordered[2] < ordered[3]):
            raise ValueError(
                "budget thresholds must be strictly increasing: "
                f"compress={self.compress} < deny_child={self.deny_child} "
                f"< restrict_export={self.restrict_export} < hard_stop={self.hard_stop}"
            )


class BudgetPolicyEngine:
    """
    Real-time optimizer. Fires on every execution event.

    Principle: minimum sufficient for quality. Most thresholds are advisory —
    the engine *suggests* compression/restriction and the node decides how to
    react. The single exception is `hard_stop`: when the spent ratio crosses
    that threshold inside `on_intent_arrived()`, the engine fires
    `cancel_token.cancel(BUDGET_EXHAUSTED)` to terminate the active node
    immediately. Set `BudgetThresholds(hard_stop=1.0)` to disable the
    hard-stop and make the engine purely advisory.

    The engine does not decompose tasks.

    Three moments it fires:
        on_intent_arrived()    — before envelope is built for this intent.
                                 May fire `cancel_token` (the only blocking action).
        on_result_arrived()    — after tool result, before writing to context.
                                 Always advisory.
        on_child_requested()   — before child node is created.
                                 Always advisory.

    Each returns an OptimizationDecision that the node acts on.

    Thresholds are configurable via the `thresholds` constructor arg.
    The defaults (0.60 / 0.80 / 0.90 / 0.95) are starting heuristics, not
    measurements — see BudgetThresholds docstring for tuning guidance.
    """

    def __init__(
        self,
        tracker: BudgetTracker,
        estimator: BudgetEstimator,
        soft_limit: int | None = None,
        thresholds: BudgetThresholds | None = None,
    ) -> None:
        self._tracker   = tracker
        self._estimator = estimator
        # soft_limit is advisory — not a hard cap
        # None means no budget guidance, always PROCEED
        self._soft_limit = soft_limit
        self._thresholds = (
            thresholds if thresholds is not None else BudgetThresholds()
        )

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
        if ratio >= self._thresholds.hard_stop:
            from axor_core.contracts.cancel import CancelReason
            envelope.cancel_token.cancel(
                CancelReason.BUDGET_EXHAUSTED,
                detail=f"spent {ratio:.0%} of soft limit",
            )
            return OptimizationDecision(
                action=OptimizationAction.PROCEED,
                reason=f"hard stop triggered at {ratio:.0%} — cancel token fired",
            )

        if ratio >= self._thresholds.compress:
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

        if ratio >= self._thresholds.restrict_export:
            return OptimizationDecision(
                action=OptimizationAction.RESTRICT_EXPORT,
                reason=f"projected {ratio:.0%} of soft limit — restrict export",
                suggested_export=ExportMode.SUMMARY,
            )

        if ratio >= self._thresholds.compress:
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

        if ratio >= self._thresholds.deny_child:
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

    def record_child_tokens(
        self,
        node_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        tool_tokens: int = 0,
        context_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
    ) -> None:
        """Record token usage from a completed child node into the tracker."""
        self._tracker.record(
            node_id=node_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_tokens=tool_tokens,
            context_tokens=context_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
        )


def _proceed(reason: str) -> OptimizationDecision:
    return OptimizationDecision(
        action=OptimizationAction.PROCEED,
        reason=reason,
    )
