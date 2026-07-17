"""Execution planning — the advisory half of governance.

Maps task classification (TaskSignal) to an ExecutionPlan. Everything in
this package is optimization-only: it must never construct or modify
authority (import contract `planning-non-authoritative`).
"""
from axor_core.planning.planner import ExecutionPlanner, HeuristicExecutionPlanner
from axor_core.planning import presets as plan_presets

__all__ = ["ExecutionPlanner", "HeuristicExecutionPlanner", "plan_presets"]
