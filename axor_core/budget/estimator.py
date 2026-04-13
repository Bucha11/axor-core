from __future__ import annotations

from axor_core.contracts.context import ContextView
from axor_core.contracts.policy import ExecutionPolicy, ContextMode, CompressionMode


# Rough token cost of a tool definition in the envelope
_TOOL_DEFINITION_TOKENS = 150

# Threshold below which a context slice is considered insufficient
_MIN_SUFFICIENT_TOKENS = 200


class BudgetEstimator:
    """
    Estimates token costs before execution and checks slice sufficiency.

    Core does not decompose tasks — the agent does.
    Estimator only answers:
        - How many tokens will this envelope cost?
        - Is this context slice sufficient for the subtask?

    Never makes decisions. Only provides estimates to policy_engine.py.
    """

    def estimate_envelope_tokens(
        self,
        task: str,
        context: ContextView,
        policy: ExecutionPolicy,
        tool_count: int,
    ) -> int:
        """
        Estimate total input tokens for one envelope before execution.

        Breakdown:
            task prompt tokens
          + context view tokens
          + tool definition tokens
          + system prompt overhead
        """
        task_tokens     = len(task) // 4
        context_tokens  = context.token_count
        tool_tokens     = tool_count * _TOOL_DEFINITION_TOKENS
        overhead        = 200   # system prompt, formatting

        return task_tokens + context_tokens + tool_tokens + overhead

    def estimate_child_slice_tokens(
        self,
        parent_context: ContextView,
        fraction: float,
    ) -> int:
        """
        Estimate how many tokens a child's context slice will cost.
        fraction = policy.child_context_fraction
        """
        return int(parent_context.token_count * fraction)

    def is_slice_sufficient(
        self,
        child_task: str,
        slice_token_estimate: int,
        parent_context: ContextView,
    ) -> bool:
        """
        Check whether a context slice is sufficient for the child's subtask.

        Heuristic: slice must contain at least _MIN_SUFFICIENT_TOKENS
        and at least some fraction of the parent's relevant fragments.

        Core does not block execution if insufficient —
        it records this in trace so policy_engine can adjust.
        """
        if slice_token_estimate < _MIN_SUFFICIENT_TOKENS:
            return False

        # if parent has relevant fragments for child task keywords
        # check that at least some are included in the slice
        child_keywords = set(child_task.lower().split())
        relevant_fragments = [
            f for f in parent_context.visible_fragments
            if any(kw in f.content.lower() for kw in child_keywords)
        ]

        if relevant_fragments:
            # slice should cover at least one relevant fragment
            covered_tokens = sum(f.token_estimate for f in relevant_fragments)
            return slice_token_estimate >= min(covered_tokens, _MIN_SUFFICIENT_TOKENS)

        return True  # no specific relevant fragments — slice is fine

    def compression_headroom(
        self,
        context: ContextView,
        policy: ExecutionPolicy,
    ) -> float:
        """
        How much more can we compress this context if needed?
        Returns 0.0 (no headroom) to 1.0 (can compress fully).

        Used by policy_engine to decide whether to compress
        before a child spawn.
        """
        match policy.compression_mode:
            case CompressionMode.LIGHT:
                # already light compression — room to go more aggressive
                return 0.7
            case CompressionMode.BALANCED:
                return 0.4
            case CompressionMode.AGGRESSIVE:
                # already maximally compressed — no headroom
                return 0.1
