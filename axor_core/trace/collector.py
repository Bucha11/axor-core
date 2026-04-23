from __future__ import annotations

import dataclasses
import json
import os
import threading
import time
from pathlib import Path
from typing import IO

from axor_core.contracts.trace import (
    DecisionTrace,
    TraceEvent,
    TraceEventKind,
    TraceConfig,
    AnonymizedTraceRecord,
    TokensSpentEvent,
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
        - stream events to JSONL per session when persist_to_disk is on
        - optionally produce AnonymizedTraceRecord for cloud training
    """

    def __init__(
        self,
        config: TraceConfig | None = None,
        session_id: str | None = None,
    ) -> None:
        self._config     = config or TraceConfig()
        self._session_id = session_id or f"session_{int(time.time()*1000)}"
        self._lock       = threading.Lock()
        self._seq        = 0
        self._traces: dict[str, DecisionTrace] = {}

        self._file: IO[str] | None = None
        self._file_path: Path | None = None
        self._closed = False

        if self._persistence_enabled:
            self._ensure_trace_dir()
            self._cleanup_old_files()

    @property
    def _persistence_enabled(self) -> bool:
        return self._config.local_only and self._config.persist_to_disk

    # ── Filesystem setup ───────────────────────────────────────────────────────

    def _trace_dir(self) -> Path:
        return Path(os.path.expanduser(self._config.trace_dir))

    def _ensure_trace_dir(self) -> None:
        self._trace_dir().mkdir(parents=True, exist_ok=True)

    def _cleanup_old_files(self) -> None:
        days = self._config.retention_days
        if days <= 0:
            return
        cutoff = time.time() - days * 86_400
        try:
            for path in self._trace_dir().glob("*.jsonl"):
                try:
                    if path.stat().st_mtime < cutoff:
                        path.unlink()
                except OSError:
                    # best effort — a racing writer or permission quirk
                    # should not block collector startup
                    continue
        except OSError:
            return

    def _open_if_needed(self) -> None:
        if self._file is not None or not self._persistence_enabled or self._closed:
            return
        self._file_path = self._trace_dir() / f"{self._session_id}.jsonl"
        # line-buffered so a crash after record() preserves whole lines
        self._file = self._file_path.open("a", encoding="utf-8", buffering=1)

    # ── Lineage bookkeeping ────────────────────────────────────────────────────

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
        Streams to JSONL when persistence is enabled.
        """
        with self._lock:
            self._seq += 1
            stamped = dataclasses.replace(event, sequence=self._seq)

            if stamped.node_id not in self._traces:
                self._traces[stamped.node_id] = DecisionTrace(
                    node_id=stamped.node_id,
                    parent_id=None,
                    depth=0,
                    policy_name="unknown",
                )

            self._traces[stamped.node_id].events.append(stamped)
            self._write_event(stamped)

    def record_many(self, events: list[TraceEvent]) -> None:
        for event in events:
            self.record(event)
        self.flush()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _write_event(self, event: TraceEvent) -> None:
        """Append one event to the session JSONL. Caller holds self._lock."""
        if not self._persistence_enabled or self._closed:
            return
        self._open_if_needed()
        if self._file is None:
            return
        trace = self._traces.get(event.node_id)
        line = {
            "session_id": self._session_id,
            "node_id":    event.node_id,
            "parent_id":  trace.parent_id if trace else None,
            "depth":      trace.depth if trace else 0,
            "policy":     trace.policy_name if trace else "unknown",
            "sequence":   event.sequence,
            "kind":       event.kind.value,
            "event":      _event_to_dict(event, self._config.persist_inputs),
            "ts":         time.time(),
        }
        self._file.write(json.dumps(line, default=str) + "\n")

    def flush(self) -> None:
        """Force buffered writes to disk. Safe to call when persistence is off."""
        with self._lock:
            if self._file is not None:
                try:
                    self._file.flush()
                    os.fsync(self._file.fileno())
                except (OSError, ValueError):
                    # fileno() raises ValueError on closed file; fsync may fail
                    # on non-regular files. Neither should crash a session close.
                    pass

    def close(self) -> None:
        """Close the JSONL writer. Idempotent."""
        with self._lock:
            if self._file is not None:
                try:
                    self._file.flush()
                    os.fsync(self._file.fileno())
                except (OSError, ValueError):
                    pass
                try:
                    self._file.close()
                except OSError:
                    pass
                self._file = None
            self._closed = True

    def trace_file_path(self) -> Path | None:
        """Path to the JSONL file for this session, if persistence is enabled."""
        if not self._persistence_enabled:
            return None
        return self._trace_dir() / f"{self._session_id}.jsonl"

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
        input_embedding: list[float] | None,
        fingerprint_kind: str = "",
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
            fingerprint_kind=fingerprint_kind,
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


# ── Serialization whitelist ───────────────────────────────────────────────────
#
# Fail-closed: any field not listed is dropped on persistence. New fields added
# to events are opaque to the sink until explicitly allowlisted, so sensitive
# additions never leak by default.

_KIND_ALLOWED_FIELDS: dict[TraceEventKind, set[str]] = {
    TraceEventKind.SIGNAL_CHOSEN: {
        "confidence", "classifier", "scores", "signal",
    },
    TraceEventKind.POLICY_ADJUSTED: {
        "original_signal", "adjusted_signal", "reason",
        "tokens_spent_before_adjustment",
    },
    TraceEventKind.INTENT_DENIED: {"intent_kind", "reason"},
    TraceEventKind.CHILD_SPAWNED: {
        "child_node_id", "child_depth", "context_fraction",
    },
    TraceEventKind.TOKENS_SPENT: {
        "input_tokens", "output_tokens", "tool_tokens",
        "context_tokens", "cumulative",
    },
    TraceEventKind.COMMAND_ROUTED: {
        "command_name", "command_class", "allowed",
    },
    TraceEventKind.PLUGIN_DENIED: {
        "plugin_name", "denied_item", "reason",
    },
    TraceEventKind.CANCELLED: {"reason", "detail", "completed_intents"},
}

# For events that carry a generic payload dict, whitelist per kind.
_KIND_ALLOWED_PAYLOAD: dict[TraceEventKind, set[str]] = {
    TraceEventKind.POLICY_CHOSEN: {
        "policy_name", "context_mode", "child_mode", "export_mode",
        "max_child_depth",
    },
    TraceEventKind.CONTEXT_COMPRESSED: {
        "before_tokens", "after_tokens", "compression_ratio",
    },
    TraceEventKind.EXTENSION_LOADED: {"name", "kind", "source"},
}

# TaskSignal fields safe to persist. raw_input is gated by persist_inputs.
_SIGNAL_SAFE_FIELDS = {
    "complexity", "nature", "estimated_scope",
    "requires_children", "requires_mutation", "domain",
}


def _sanitize_signal(signal: object, persist_inputs: bool) -> object:
    """Filter a TaskSignal-like dict to safe fields."""
    if not isinstance(signal, dict):
        return signal
    out = {k: v for k, v in signal.items() if k in _SIGNAL_SAFE_FIELDS}
    if persist_inputs and "raw_input" in signal:
        out["raw_input"] = signal["raw_input"]
    return out


def _event_to_dict(event: TraceEvent, persist_inputs: bool) -> dict:
    """
    Serialize a TraceEvent to a plain dict suitable for json.dumps.

    Whitelist-based: only fields explicitly allowed for the event kind are
    included. Nested TaskSignal values are further stripped. `raw_input`
    fields are included only when `persist_inputs` is True.
    """
    raw = dataclasses.asdict(event)
    raw.pop("kind", None)
    raw.pop("node_id", None)
    raw.pop("sequence", None)
    raw_payload = raw.pop("payload", None) or {}

    out: dict = {}

    for key in _KIND_ALLOWED_FIELDS.get(event.kind, set()):
        if key not in raw:
            continue
        val = raw[key]
        if key in ("signal", "original_signal", "adjusted_signal"):
            val = _sanitize_signal(val, persist_inputs)
        out[key] = val

    payload_allowed = _KIND_ALLOWED_PAYLOAD.get(event.kind, set())
    if payload_allowed and isinstance(raw_payload, dict):
        filtered = {k: v for k, v in raw_payload.items() if k in payload_allowed}
        if filtered:
            out["payload"] = filtered

    if persist_inputs and event.kind == TraceEventKind.SIGNAL_CHOSEN:
        raw_input = raw.get("raw_input", "")
        if raw_input:
            out["raw_input"] = raw_input

    return out
