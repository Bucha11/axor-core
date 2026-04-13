from __future__ import annotations

from axor_core.contracts.context import LineageSummary
from axor_core.contracts.envelope import ExecutionEnvelope
from axor_core.contracts.intent import Intent
from axor_core.contracts.policy import ExecutionPolicy
from axor_core.contracts.trace import ChildSpawnedEvent, TraceEventKind
from axor_core.errors.exceptions import ChildNotAllowedError, MaxDepthExceededError
from axor_core.node.intent_loop import IntentLoop
from axor_core.policy.composer import PolicyComposer


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
        self._composer = PolicyComposer()

    def prepare_child(
        self,
        spawn_intent: Intent,
        parent_envelope: ExecutionEnvelope,
        intent_loop: IntentLoop,
        trace_events: list,
    ) -> tuple[str, ExecutionPolicy, LineageSummary]:
        """
        Validate and prepare everything needed to construct a child GovernedNode.

        Returns (child_task, child_policy, child_lineage).
        Raises ChildNotAllowedError or MaxDepthExceededError if denied.
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

        # child policy — parent restrictions always applied
        child_policy = self._composer.apply_parent_restrictions(
            child_policy=parent_envelope.policy,
            parent_policy=parent_envelope.policy,
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

        return child_task, child_policy, child_lineage
