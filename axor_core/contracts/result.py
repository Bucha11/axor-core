from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Executor Stream Events ─────────────────────────────────────────────────────

class ExecutorEventKind(str, Enum):
    """
    Events emitted by Invokable.stream().
    The intent_loop processes each event before it reaches the outside.
    """
    TEXT      = "text"       # text output from executor
    TOOL_USE  = "tool_use"   # executor requests a tool → becomes Intent
    STOP      = "stop"       # executor finished
    ERROR     = "error"      # executor error


@dataclass(frozen=True)
class ExecutorEvent:
    """
    A single event in the executor stream.

    tool_use events are intercepted by intent_loop
    before they execute — they become Intents.

    text events pass through the export filter.

    stop triggers governed result construction.
    """
    kind: ExecutorEventKind
    payload: dict[str, Any]
    node_id: str


# ── Execution Result ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExecutionResult:
    """
    Normalized governed result from a node.

    This is never raw executor output.
    It has passed through export contract enforcement.

    output:         what the node is allowed to export
    export_payload: structured export for parent node consumption
    token_usage:    actual tokens spent (for budget/tracker.py)
    metadata:       governance metadata (policy used, node_id, depth)
    """
    node_id: str
    output: str
    export_payload: dict[str, Any]
    token_usage: TokenUsage
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TokenUsage:
    """
    Actual token consumption for one node execution.
    Reported to budget/tracker.py after execution completes.

    cache_creation_input_tokens / cache_read_input_tokens:
        Anthropic prompt-cache accounting. cache_creation tokens are billed
        at 1.25x normal input price; cache_read tokens at 0.1x. Anthropic
        reports these as separate counters from input_tokens, so sum them with
        input_tokens when calculating full processed input volume.
    """
    input_tokens: int
    output_tokens: int
    tool_tokens: int        # tokens spent on tool definitions in envelope
    context_tokens: int     # tokens in ContextView
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total(self) -> int:
        return (
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
            + self.output_tokens
        )
