from __future__ import annotations

from axor_core.contracts.context import LineageSummary
from axor_core.contracts.envelope import ExecutionEnvelope
from axor_core.contracts.intent import Intent
from axor_core.contracts.policy import ExportMode, ExecutionPolicy
from axor_core.contracts.trace import ChildSpawnedEvent, TraceEventKind
from axor_core.errors.exceptions import ChildNotAllowedError, SpawnValidationError
from axor_core.node.intent_loop import IntentLoop


def inherit_degradation(parent_engine, child_node_id: str):
    """Per-node degradation at spawn (spec v2 Ch.4 section 3): the child gets a
    FRESH engine seeded at max(parent level, NORMAL) — a derived posture, not a
    blank one. Narrow-or-preserve: a CAUTIOUS parent cannot spawn a NORMAL
    child to escape its own restriction (spawn-laundering, the tree analog of
    lateral laundering, closed). The child engine is independent afterwards —
    no shared governance state (decision v2-13)."""
    from axor_core.contracts.degradation import DegradationLevel
    from axor_core.degradation.engine import DegradationEngine

    child = DegradationEngine(node_id=child_node_id)
    parent_level = parent_engine.state.level
    if parent_level > DegradationLevel.NORMAL:
        child.tighten(
            parent_level,
            reason="spawn inheritance: child starts at parent's level "
            "(narrow-or-preserve, spec v2 Ch.4)",
            trigger_intent="spawn",
        )
    return child

# ExportMode restrictiveness order — higher index = more restrictive.
# FILTERED (output+metadata, 4096 tokens) is less restrictive than SUMMARY (output only, 1024 tokens).
_EXPORT_MODE_ORDER = [ExportMode.FULL, ExportMode.FILTERED, ExportMode.SUMMARY, ExportMode.RESTRICTED]


def _export_mode_rank(mode: ExportMode) -> int:
    try:
        return _EXPORT_MODE_ORDER.index(mode)
    except ValueError:
        return 0


def _validate_child_policy(
    child_policy: ExecutionPolicy,
    parent_policy: ExecutionPolicy,
    child_depth: int,
) -> None:
    """
    Validate that the child policy does not exceed the parent policy ceiling.

    Raises SpawnValidationError (not AssertionError) so validation survives
    python -O optimisation mode.
    """
    # tools ⊆ parent tools: child cannot enable capabilities the parent lacks
    parent_tp = parent_policy.tool_policy
    child_tp = child_policy.tool_policy
    if child_tp.allow_write and not parent_tp.allow_write:
        raise SpawnValidationError("child requests allow_write but parent forbids it")
    if child_tp.allow_bash and not parent_tp.allow_bash:
        raise SpawnValidationError("child requests allow_bash but parent forbids it")
    if child_tp.allow_spawn and not parent_tp.allow_spawn:
        raise SpawnValidationError("child requests allow_spawn but parent forbids it")
    # extra_allowed: child cannot claim tools not in parent's extra_allowed
    parent_extra = set(parent_tp.extra_allowed)
    child_extra = set(child_tp.extra_allowed)
    excess = child_extra - parent_extra
    if excess:
        raise SpawnValidationError(
            f"child requests extra_allowed tools not available in parent: {sorted(excess)}"
        )
    # export_mode: child must be at least as restrictive as parent
    if _export_mode_rank(child_policy.export_mode) < _export_mode_rank(parent_policy.export_mode):
        raise SpawnValidationError(
            f"child export_mode={child_policy.export_mode.value!r} is less restrictive "
            f"than parent export_mode={parent_policy.export_mode.value!r}"
        )
    # child depth < parent remaining depth
    if child_depth > parent_policy.max_child_depth:
        raise SpawnValidationError(
            f"child depth {child_depth} exceeds parent max_child_depth {parent_policy.max_child_depth}"
        )


class ChildSpawner:
    """
    Governed child-node creation.

    Children are never raw agents.
    Every child is a GovernedNode with a derived governance envelope.

    Child creation pipeline:
        spawn_child intent arrives
          → intent_loop.resolve_spawn_intent()   (policy check)
          → derive child lineage                 (depth, ancestry)
          → derive child policy                  (parent restrictions applied)
          → derive child context slice           (fraction of parent context)
          → build child envelope
          → return child node id + envelope

    The actual GovernedNode construction happens in wrapper.py.
    ChildSpawner only produces the ingredients.
    """

    def __init__(self) -> None:
        pass

    def prepare_child(
        self,
        spawn_intent: Intent,
        parent_envelope: ExecutionEnvelope,
        intent_loop: IntentLoop,
        trace_events: list,
    ) -> tuple[str, LineageSummary]:
        """
        Validate and prepare everything needed to construct a child GovernedNode.

        Returns (child_task, child_lineage).
        Raises ChildNotAllowedError or MaxDepthExceededError if denied.

        Child policy is NOT derived here — it is selected fresh by the child's
        own GovernedNode.run() via TaskAnalyzer + PolicySelector, then ceilinged
        against parent_policy via PolicyComposer.compose(parent_policy=...).
        _validate_child_policy is called there against the actual derived policy.
        """
        from axor_core.node.envelope import _new_node_id

        decision = intent_loop.resolve_spawn_intent(spawn_intent, parent_envelope)

        if not decision.kind.value == "approve":
            raise ChildNotAllowedError(reason=decision.reason)

        child_task    = spawn_intent.payload.get("task", parent_envelope.task)
        child_node_id = _new_node_id()
        child_depth   = parent_envelope.lineage.depth + 1

        child_lineage = LineageSummary(
            node_id=child_node_id,
            parent_id=parent_envelope.node_id,
            depth=child_depth,
            ancestry_ids=[
                *parent_envelope.lineage.ancestry_ids,
                parent_envelope.node_id,
            ],
            inherited_restrictions=list(
                parent_envelope.lineage.inherited_restrictions
            ),
        )

        # record spawn in trace
        trace_events.append(
            ChildSpawnedEvent(
                kind=TraceEventKind.CHILD_SPAWNED,
                node_id=parent_envelope.node_id,
                sequence=len(trace_events),
                child_node_id=child_node_id,
                child_depth=child_depth,
                context_fraction=parent_envelope.policy.child_context_fraction,
            )
        )

        return child_task, child_lineage
