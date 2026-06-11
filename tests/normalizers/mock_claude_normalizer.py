from __future__ import annotations

"""
Test-only copies of ClaudeNormalizer and StreamNormalizer.

These are extracted from axor-claude so that axor-core tests do not import
adapter packages. The logic is identical to the production code; both classes
depend exclusively on axor-core types.
"""

import json
from dataclasses import dataclass, field
from typing import Any

from axor_core.contracts.anomaly import NormalizedIntent
from axor_core.contracts.intent import Intent, IntentKind
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind
from axor_core.errors.exceptions import NormalizerError, UnknownProviderFormatError
from axor_core.policy.normalizer import IntentNormalizer

_PROVIDER = "anthropic"


# ── ClaudeNormalizer ───────────────────────────────────────────────────────────

class ClaudeNormalizer:
    """Test-only copy of axor_claude.normalizer.ClaudeNormalizer."""

    def __init__(self, workdir: str | None = None) -> None:
        self._normalizer = IntentNormalizer(workdir=workdir)

    def normalize(self, raw_event: Any) -> NormalizedIntent:
        tool_name, args = self._extract(raw_event)
        intent = Intent(
            kind=IntentKind.TOOL_CALL,
            payload={"tool": tool_name, "args": args},
            node_id="",
        )
        return self._normalizer.normalize(intent)

    def _extract(self, raw_event: Any) -> tuple[str, dict]:
        if isinstance(raw_event, dict):
            if "_axor_parse_error" in raw_event:
                raise NormalizerError(
                    _PROVIDER,
                    f"event contains parse error marker: {raw_event['_axor_parse_error']}",
                )
            tool = raw_event.get("tool")
            args = raw_event.get("args", {})
            if not tool:
                raise UnknownProviderFormatError(_PROVIDER, "dict missing 'tool' key")
            if not isinstance(args, dict):
                raise NormalizerError(_PROVIDER, "'args' must be a dict")
            return str(tool), args

        name = getattr(raw_event, "name", None)
        if name is not None:
            input_data = getattr(raw_event, "input", {}) or {}
            if not isinstance(input_data, dict):
                raise NormalizerError(_PROVIDER, "ToolUseBlock.input must be a dict")
            return str(name), input_data

        raise UnknownProviderFormatError(_PROVIDER, type(raw_event).__name__)


# ── StreamNormalizer ───────────────────────────────────────────────────────────

@dataclass
class _TextBlock:
    index: int
    text: str = ""

    def apply_delta(self, delta) -> None:
        if hasattr(delta, "text"):
            self.text += delta.text


@dataclass
class _ToolUseBlock:
    index: int
    tool_use_id: str
    name: str
    input_json: str = ""

    def apply_delta(self, delta) -> None:
        if hasattr(delta, "partial_json"):
            self.input_json += delta.partial_json

    def parse_args(self) -> dict[str, Any]:
        if not self.input_json.strip():
            return {}
        try:
            parsed = json.loads(self.input_json)
        except json.JSONDecodeError as exc:
            return {
                "_axor_parse_error": f"invalid JSON in tool input: {exc.msg}",
                "_raw": self.input_json[:4000],
            }
        if not isinstance(parsed, dict):
            return {
                "_axor_parse_error": f"tool_use input is not a JSON object (got {type(parsed).__name__})",
                "_raw": self.input_json[:4000],
            }
        return parsed


@dataclass
class StreamNormalizer:
    """Test-only copy of axor_claude.events.StreamNormalizer."""

    node_id: str
    _blocks: dict[int, _TextBlock | _ToolUseBlock] = field(default_factory=dict)
    _input_tokens: int = 0
    _output_tokens: int = 0
    _cache_creation_input_tokens: int = 0
    _cache_read_input_tokens: int = 0

    def process(self, sdk_event) -> list[ExecutorEvent]:
        event_type = getattr(sdk_event, "type", None)

        match event_type:
            case "message_start":
                msg = getattr(sdk_event, "message", None)
                if msg:
                    usage = getattr(msg, "usage", None)
                    if usage:
                        self._input_tokens = getattr(usage, "input_tokens", 0)
                        self._cache_creation_input_tokens = getattr(
                            usage, "cache_creation_input_tokens", 0
                        ) or 0
                        self._cache_read_input_tokens = getattr(
                            usage, "cache_read_input_tokens", 0
                        ) or 0
                return []

            case "content_block_start":
                block = getattr(sdk_event, "content_block", None)
                index = getattr(sdk_event, "index", 0)
                if block is None:
                    return []
                block_type = getattr(block, "type", None)
                if block_type == "text":
                    self._blocks[index] = _TextBlock(index=index)
                elif block_type == "tool_use":
                    self._blocks[index] = _ToolUseBlock(
                        index=index,
                        tool_use_id=getattr(block, "id", ""),
                        name=getattr(block, "name", ""),
                    )
                return []

            case "content_block_delta":
                index = getattr(sdk_event, "index", 0)
                delta = getattr(sdk_event, "delta", None)
                block = self._blocks.get(index)
                if block and delta:
                    block.apply_delta(delta)
                return []

            case "content_block_stop":
                index = getattr(sdk_event, "index", 0)
                block = self._blocks.pop(index, None)
                if block is None:
                    return []
                if isinstance(block, _TextBlock) and block.text:
                    return [ExecutorEvent(
                        kind=ExecutorEventKind.TEXT,
                        payload={"text": block.text},
                        node_id=self.node_id,
                    )]
                if isinstance(block, _ToolUseBlock):
                    return [ExecutorEvent(
                        kind=ExecutorEventKind.TOOL_USE,
                        payload={
                            "tool":        block.name,
                            "args":        block.parse_args(),
                            "tool_use_id": block.tool_use_id,
                        },
                        node_id=self.node_id,
                    )]
                return []

            case "message_delta":
                usage = getattr(sdk_event, "usage", None)
                if usage:
                    self._output_tokens = getattr(usage, "output_tokens", 0)
                    self._cache_creation_input_tokens = max(
                        self._cache_creation_input_tokens,
                        getattr(usage, "cache_creation_input_tokens", 0) or 0,
                    )
                    self._cache_read_input_tokens = max(
                        self._cache_read_input_tokens,
                        getattr(usage, "cache_read_input_tokens", 0) or 0,
                    )
                return []

            case "message_stop":
                return [ExecutorEvent(
                    kind=ExecutorEventKind.STOP,
                    payload={
                        "usage": {
                            "input_tokens":  self._input_tokens,
                            "output_tokens": self._output_tokens,
                            "tool_tokens":   0,
                            "cache_creation_input_tokens": self._cache_creation_input_tokens,
                            "cache_read_input_tokens":     self._cache_read_input_tokens,
                        }
                    },
                    node_id=self.node_id,
                )]

            case _:
                return []
