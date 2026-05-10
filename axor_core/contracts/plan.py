from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlanStep:
    """
    A single mutation action captured during Phase 1 (planning).

    Phantom-executed in Phase 1 (never touches disk).
    Real-executed in Phase 2 only if approved=True.

    phantom_result: what the agent was told the tool returned during Phase 1.
    This keeps the agent's subsequent planning coherent — it believes writes
    succeeded and can plan follow-up steps that depend on them.
    """
    index: int
    tool: str
    args: dict[str, Any]
    phantom_result: Any
    approved: bool = True

    def summary(self) -> str:
        if self.tool == "write":
            path = self.args.get("path", "?")
            n = len(self.args.get("content", ""))
            return f'write("{path}", <{n} chars>)'
        if self.tool == "bash":
            cmd = self.args.get("command", "?")[:60]
            return f'bash("{cmd}")'
        if self.tool == "delete":
            return f'delete("{self.args.get("path", "?")}")'
        return f'{self.tool}({self.args})'


@dataclass
class AgentPlan:
    """
    The full mutation plan produced by Phase 1.

    Contains only mutation steps (write, bash, delete) — read and search
    intents are not recorded because they have no effect on disk state.

    Use .display() to show the plan to a human for approval.
    Use .to_context_fragment() to inject it into Phase 2's pinned context.
    """
    task: str
    steps: list[PlanStep]
    phase1_node_id: str
    phase1_tokens: int = 0

    def approved_steps(self) -> list[PlanStep]:
        return [s for s in self.steps if s.approved]

    def approve(self, indices: list[int]) -> None:
        approved_set = set(indices)
        for step in self.steps:
            step.approved = step.index in approved_set

    def approve_all(self) -> None:
        for step in self.steps:
            step.approved = True

    def mutation_count(self) -> int:
        return len(self.steps)

    def to_context_fragment(self) -> str:
        approved = self.approved_steps()
        if not approved:
            return "APPROVED EXECUTION PLAN: no mutation steps approved."

        lines = [
            "APPROVED EXECUTION PLAN — follow exactly, no deviations:",
            "",
        ]
        for step in approved:
            lines.append(f"  [{step.index}] {step.summary()}")

        lines += [
            "",
            "Rules for Phase 2 execution:",
            "- Call write, bash, delete only for steps listed above, in order.",
            "- Do not add, skip, or reorder steps.",
            "- Read and search tools are unrestricted.",
        ]
        return "\n".join(lines)

    def display(self) -> str:
        if not self.steps:
            return f'Plan for "{self.task[:72]}"\nNo mutation steps.'

        lines = [
            f'Plan for "{self.task[:72]}"',
            f'{len(self.steps)} step(s)  |  phase-1 cost: {self.phase1_tokens} tokens',
            "",
        ]
        for step in self.steps:
            mark = "[approved]" if step.approved else "[skipped] "
            lines.append(f"  {mark}  {step.index}. {step.summary()}")
        return "\n".join(lines)
