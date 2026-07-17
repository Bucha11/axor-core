"""PlanComposer — merge/expand ExecutionPlans within a ResourceBudget.

Unlike authority composition, plan composition is NOT monotonic: a plan may
widen (more context, less compression, deeper decomposition hint) as well as
narrow — but only inside the operator's ResourceBudget ceilings. Widening a
plan is not a capability escalation: it changes what the run may spend,
never what it may effect.
"""
from __future__ import annotations

import dataclasses

from axor_core.contracts.planning import ExecutionPlan, ResourceBudget
from axor_core.contracts.policy import ContextMode

_CONTEXT_ORDER = [ContextMode.MINIMAL, ContextMode.MODERATE, ContextMode.BROAD]


class PlanComposer:
    """Budget-bounded plan adjustment."""

    def expand(
        self,
        base: ExecutionPlan,
        budget: ResourceBudget,
        *,
        requested_context_mode: str | None = None,
        requested_child_depth: int | None = None,
        additional_token_reservation: int | None = None,
        reason: str = "",
    ) -> tuple[ExecutionPlan, list[str]]:
        """Apply a requested expansion, clamped to the budget.

        Returns (new_plan, constraints) — `constraints` lists every requested
        widening the budget refused (telemetry: PLAN_CONSTRAINED_BY_BUDGET).
        Malformed or negative requests are ignored fail-closed: they can
        never shrink a ceiling or smuggle non-planning fields.
        """
        changes: dict = {}
        constraints: list[str] = []

        if requested_context_mode is not None:
            try:
                mode = ContextMode(requested_context_mode)
            except ValueError:
                constraints.append(f"unknown context_mode {requested_context_mode!r}")
            else:
                changes["context_mode"] = mode

        if requested_child_depth is not None:
            try:
                depth = int(requested_child_depth)
            except (TypeError, ValueError):
                depth = -1
            if depth < 0:
                constraints.append("child_depth must be a non-negative integer")
            else:
                if budget.max_children is not None and depth > budget.max_children:
                    constraints.append(
                        f"child_depth {depth} > budget.max_children {budget.max_children}"
                    )
                    depth = budget.max_children
                changes["suggested_child_depth"] = depth

        if additional_token_reservation is not None:
            try:
                extra = int(additional_token_reservation)
            except (TypeError, ValueError):
                extra = -1
            if extra < 0:
                constraints.append("token reservation must be a non-negative integer")
            else:
                total = (base.token_reservation or 0) + extra
                cap = budget.max_token_reservation
                if cap is not None and total > cap:
                    constraints.append(
                        f"token_reservation {total} > budget cap {cap}"
                    )
                    total = cap
                changes["token_reservation"] = total

        if not changes:
            return base, constraints
        new_plan = dataclasses.replace(
            base, **changes, source="plan_expansion", name=base.name,
        )
        return new_plan, constraints
