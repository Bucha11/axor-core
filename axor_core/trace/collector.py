from __future__ import annotations

import threading
from axor_core.contracts.trace import (
    DecisionTrace,
    TraceEvent,
    TraceEventKind,
    TraceConfig,
    AnonymizedTraceRecord,
    TokensSpentEvent,
    PolicyAdjustedEvent,
)
from axor_core.contracts.policy import TaskSignal


class TraceCollector:
    """
    Collects governance decisions across the entire node lineage.

    Not flat logging — each event belongs to a node which belongs
    to a lineage tree. The collector maintains this structure.

    One collector per GovernedSession — shared across all nodes.
    Thread-safe — child nodes may record concurrently.

    Responsibilities:
        - accept events from any node in the tree
        - assign global sequence numbers (ordering across nodes)
        - build DecisionTrace per node
        - expose lineage-aware queries
        - optionally persist to disk (TraceConfig.local_only)
        - optionally produce AnonymizedTraceRecord for cloud training
    """

    def __init__(self, config: TraceConfig | None = None) -> None:
        self._config  = config or TraceConfig()
        self._lock    = threading.Lock()
        self._seq     = 0
        self._traces: dict[str, DecisionTrace] = {}

    def register_node(
        self,
        node_id: str,
        parent_id: str | None,
        depth: int,
        policy_name: str,
    ) -> None:
        with self._lock:
            self._traces[node_id] = DecisionTrace(
                node_id=node_id,
                parent_id=parent_id,
                depth=depth,
                policy_name=policy_name,
            )

    def record(self, event: TraceEvent) -> None:
        """
        Record a governance event for a node.
        Assigns global sequence number — ordering is preserved across nodes.
        """
        with self._lock:
            self._seq += 1
            import dataclasses
            stamped = dataclasses.replace(event, sequence=self._seq)

            if stamped.node_id not in self._traces:
                # node not pre-registered — create minimal trace
                self._traces[stamped.node_id] = DecisionTrace(
                    node_id=stamped.node_id,
                    parent_id=None,
                    depth=0,
                    policy_name="unknown",
                )

            self._traces[stamped.node_id].events.append(stamped)

    def record_many(self, events: list[TraceEvent]) -> None:
        for event in events:
            self.record(event)

    # ── Queries ────────────────────────────────────────────────────────────────

    def get_trace(self, node_id: str) -> DecisionTrace | None:
        with self._lock:
            return self._traces.get(node_id)

    def all_traces(self) -> list[DecisionTrace]:
        with self._lock:
            return list(self._traces.values())

    def lineage_traces(self, node_id: str) -> list[DecisionTrace]:
        """All traces in the ancestry chain: root → node."""
        with self._lock:
            result = []
            current_id: str | None = node_id
            seen = set()
            while current_id and current_id not in seen:
                seen.add(current_id)
                trace = self._traces.get(current_id)
                if trace:
                    result.append(trace)
                    current_id = trace.parent_id
                else:
                    break
            return list(reversed(result))  # root first

    def total_tokens(self) -> int:
        """Total tokens across all nodes — delegates to TokensSpentEvent."""
        with self._lock:
            total = 0
            for trace in self._traces.values():
                spent_events = [
                    e for e in trace.events
                    if isinstance(e, TokensSpentEvent)
                ]
                if spent_events:
                    # last TokensSpentEvent has cumulative total
                    total += spent_events[-1].output_tokens + spent_events[-1].input_tokens
            return total

    def had_policy_adjustments(self) -> bool:
        with self._lock:
            return any(
                trace.had_policy_adjustment
                for trace in self._traces.values()
            )

    def denied_intents(self) -> list[TraceEvent]:
        with self._lock:
            return [
                event
                for trace in self._traces.values()
                for event in trace.events
                if event.kind == TraceEventKind.INTENT_DENIED
            ]

    def child_count(self) -> int:
        with self._lock:
            return sum(
                1 for trace in self._traces.values()
                if trace.parent_id is not None
            )

    # ── Training data ──────────────────────────────────────────────────────────

    def to_anonymized_record(
        self,
        node_id: str,
        signal: TaskSignal,
        classifier_used: str,
        confidence: float,
        input_embedding: list[float],
    ) -> AnonymizedTraceRecord | None:
        """
        Produce an anonymized record for cloud classifier training.
        Only called when TraceConfig.training_opt_in=True.

        No raw input. No code. No file content. Embedding only.
        """
        if not self._config.training_opt_in:
            return None

        trace = self.get_trace(node_id)
        if not trace:
            return None

        return AnonymizedTraceRecord(
            input_embedding=input_embedding,
            signal_chosen=signal,
            classifier_used=classifier_used,
            confidence=confidence,
            tokens_spent=sum(
                (e.input_tokens + e.output_tokens)
                for e in trace.events
                if isinstance(e, TokensSpentEvent)
            ),
            policy_adjusted=trace.had_policy_adjustment,
        )

    # ── Persistence ────────────────────────────────────────────────────────────

    def persist(self, node_id: str) -> None:
        """
        Write trace to disk if config allows.
        Only called when TraceConfig.local_only=True (default).
        Raw inputs are never written unless persist_inputs=True.
        """
        if not self._config.local_only:
            return

        trace = self.get_trace(node_id)
        if not trace:
            return

        import json
        import os
        import time

        trace_dir = os.path.expanduser(self._config.trace_dir)
        os.makedirs(trace_dir, exist_ok=True)

        path = os.path.join(trace_dir, f"{node_id}_{int(time.time())}.json")

        record = {
            "node_id":     trace.node_id,
            "parent_id":   trace.parent_id,
            "depth":       trace.depth,
            "policy_name": trace.policy_name,
            "total_tokens": trace.total_tokens,
            "had_policy_adjustment": trace.had_policy_adjustment,
            "event_count": len(trace.events),
            "events": [
                {
                    "kind":     e.kind.value,
                    "sequence": e.sequence,
                    # raw input omitted unless persist_inputs=True
                    "payload": _safe_payload(e, self._config.persist_inputs),
                }
                for e in trace.events
            ],
        }

        with open(path, "w") as f:
            json.dump(record, f, indent=2, default=str)


def _safe_payload(event: TraceEvent, persist_inputs: bool) -> dict:
    """Strip raw input from payload unless persist_inputs is enabled."""
    payload = dict(event.payload)
    if not persist_inputs:
        payload.pop("raw_input", None)
        payload.pop("input", None)
        payload.pop("content", None)
    return payload
