from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from axor_core.budget.estimator import BudgetEstimator
from axor_core.budget.tracker import BudgetTracker
from axor_core.contracts.envelope import ExecutionEnvelope
from axor_core.contracts.policy import (
    ExecutionPolicy,
    CompressionMode,
    ExportMode,
)


class OptimizationAction(str, Enum):
    PROCEED          = "proceed"
    COMPRESS_CONTEXT = "compress_context"
    DENY_CHILD       = "deny_child"
    RESTRICT_EXPORT  = "restrict_export"
    HARD_STOP        = "hard_stop"


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

    Order invariant: compress < deny_child < restrict_export < hard_stop.
    """
    compress: float        = 0.60
    deny_child: float      = 0.80
    restrict_export: float = 0.90
    hard_stop: float       = 0.95

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


# Callback type for threshold events.
# Receives (threshold_name, ratio) where threshold_name is one of:
# "compress", "deny_child", "restrict_export", "hard_stop"
ThresholdCallback = Callable[[str, float], None]


class BudgetPolicyEngine:
    """
    Real-time optimizer. Fires on every execution event.

    Principle: minimum sufficient for quality.

    Three moments it fires:
        on_intent_arrived()    — before envelope is built for this intent.
        on_result_arrived()    — after tool result.
        on_child_requested()   — before child node is created.

    It also exposes:
        on_threshold_crossed() — subscribe to threshold crossing events.
        suggest_tier_shift()   — advisory hint for adaptive routing.
    """

    def __init__(
        self,
        tracker: BudgetTracker,
        estimator: BudgetEstimator,
        soft_limit: int | None = None,
        thresholds: BudgetThresholds | None = None,
    ) -> None:
        self._tracker    = tracker
        self._estimator  = estimator
        self._soft_limit = soft_limit
        self._thresholds = thresholds if thresholds is not None else BudgetThresholds()
        self._threshold_callbacks: list[ThresholdCallback] = []
        # Track the last threshold bucket we fired for, to avoid duplicate events.
        self._last_threshold_fired: str = ""

    # ── Threshold subscription API ───────────────────────────────────────────────

    def on_threshold_crossed(
        self,
        callback: ThresholdCallback,
    ) -> Callable[[], None]:
        """
        Register a callback fired when spend/cap ratio crosses a threshold.

        The callback receives (threshold_name: str, ratio: float).
        Each threshold fires at most once per monotonic increase
        (i.e., it does not re-fire if spend drops and rises again).

        Returns a no-arg callable that removes the subscription.
        """
        self._threshold_callbacks.append(callback)

        def _remove() -> None:
            try:
                self._threshold_callbacks.remove(callback)
            except ValueError:
                pass

        return _remove

    def suggest_tier_shift(self) -> int:
        """
        Advisory hint for adaptive routing.

        Returns:
            -1  shift to a cheaper/lower tier (budget pressure)
             0  hold current tier
            +1  can afford a better tier (budget comfortable)

        The hint is based on the current spend/cap ratio relative to thresholds.
        Callers should use this as a signal, not a mandate.
        """
        if self._soft_limit is None:
            return 0
        ratio = self._tracker.total_tokens() / self._soft_limit
        if ratio >= self._thresholds.restrict_export:
            return -1
        if ratio >= self._thresholds.compress:
            return -1
        if ratio < self._thresholds.compress * 0.5:
            return 1
        return 0

    # ── Existing API ─────────────────────────────────────────────────────────────────

    def on_intent_arrived(
        self,
        envelope: ExecutionEnvelope,
        tool_count: int,
    ) -> OptimizationDecision:
        if self._soft_limit is None:
            return _proceed("no soft limit set")

        spent = self._tracker.total_tokens()
        ratio = spent / self._soft_limit

        self._maybe_notify_threshold(ratio)

        if ratio >= self._thresholds.hard_stop:
            from axor_core.contracts.cancel import CancelReason
            envelope.cancel_token.cancel(
                CancelReason.BUDGET_EXHAUSTED,
                detail=f"spent {ratio:.0%} of soft limit",
            )
            return OptimizationDecision(
                action=OptimizationAction.HARD_STOP,
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
        if self._soft_limit is None:
            return _proceed("no soft limit set")

        spent = self._tracker.total_tokens()
        ratio = spent / self._soft_limit

        if ratio >= self._thresholds.deny_child:
            return OptimizationDecision(
                action=OptimizationAction.DENY_CHILD,
                reason=f"spent {ratio:.0%} of soft limit — deny child to preserve budget",
            )

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
            return OptimizationDecision(
                action=OptimizationAction.COMPRESS_CONTEXT,
                reason="child context slice may be insufficient — compress parent context first",
                suggested_compression=CompressionMode.AGGRESSIVE,
            )

        return _proceed(f"child approved — spent {ratio:.0%}, slice sufficient")

    def register_node(
        self,
        node_id: str,
        parent_id: str | None,
        depth: int,
        tier: str | None = None,
    ) -> None:
        """Register a node's lineage so depth/subtree budget accounting is correct.
        Delegates to the tracker; idempotent."""
        self._tracker.register_node(node_id, parent_id, depth, tier)

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
        self._tracker.record(
            node_id=node_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_tokens=tool_tokens,
            context_tokens=context_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
        )

    # ── Private ─────────────────────────────────────────────────────────────────────

    def _maybe_notify_threshold(self, ratio: float) -> None:
        """Fire threshold callbacks once per threshold bucket crossing."""
        if not self._threshold_callbacks or self._soft_limit is None:
            return

        # Determine which bucket we're in (highest crossed wins)
        name = ""
        if ratio >= self._thresholds.hard_stop:
            name = "hard_stop"
        elif ratio >= self._thresholds.restrict_export:
            name = "restrict_export"
        elif ratio >= self._thresholds.deny_child:
            name = "deny_child"
        elif ratio >= self._thresholds.compress:
            name = "compress"

        if name and name != self._last_threshold_fired:
            self._last_threshold_fired = name
            for cb in self._threshold_callbacks:
                try:
                    cb(name, ratio)
                except Exception:
                    pass


def _proceed(reason: str) -> OptimizationDecision:
    return OptimizationDecision(
        action=OptimizationAction.PROCEED,
        reason=reason,
    )
