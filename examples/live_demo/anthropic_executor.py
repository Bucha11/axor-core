"""A real Anthropic (Claude) executor for axor-core.

This is a *minimal, dependency-free* `Invokable` that drives Claude's native
tool-use loop against the Anthropic Messages API and feeds every tool call
through the kernel before it executes. It exists to prove the governance path
end-to-end against a real model, not to be a production adapter (the full
adapter lives in axor-claude).

How it plugs into the kernel
----------------------------
The kernel drives the executor's async generator: it pulls one event, resolves
it, and only then asks for the next event. So when `stream()` yields a
`TOOL_USE` event and the generator resumes, the governed result is already
sitting in the result bus. The executor exposes `get_bus()`; the node wrapper
sees that method, registers a callback, and pushes each governed tool result
into the bus instead of yielding it back as a TEXT event. That callback is the
only contract between this adapter and the kernel — the kernel never learns
anything Anthropic-specific.

No SDK: the call is a raw `urllib` POST run in a worker thread, which keeps the
example honest about the core's zero-dependency design (and sidesteps an SDK
connection issue in this sandbox).
"""
from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import Any, AsyncIterator

from axor_core.contracts.envelope import ExecutionEnvelope
from axor_core.contracts.invokable import Invokable
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind

try:
    import anyio  # noqa: F401
    _TO_THREAD = None
except Exception:  # pragma: no cover - anyio optional
    _TO_THREAD = None

import asyncio

_API_URL = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"


class ToolResultBus:
    """A one-slot-per-call mailbox the kernel pushes governed results into.

    The node wrapper detects `get_bus()` and calls `push(tool_use_id, result)`
    after governing each tool call. The executor reads the slot back the instant
    its `yield` resumes — see the module docstring for why that ordering holds.
    """

    def __init__(self) -> None:
        self._slots: dict[str, Any] = {}

    def push(self, tool_use_id: str, result: Any) -> None:
        self._slots[tool_use_id] = result

    def take(self, tool_use_id: str) -> Any:
        return self._slots.pop(tool_use_id, None)


def _stringify_tool_result(result: Any) -> str:
    """Render a governed tool result as the string Claude expects back."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        # A denial comes back as {"error": "denied", "category": ..., ...}.
        if result.get("error"):
            return (
                "GOVERNANCE_DENIED: this action was refused by the execution "
                f"kernel (category={result.get('category')}). Do not retry it; "
                "explain to the user that the action was blocked."
            )
        return json.dumps(result)
    return str(result)


class AnthropicExecutor(Invokable):
    """Drives Claude's tool-use loop, one governed tool call at a time."""

    def __init__(
        self,
        tools: list[dict[str, Any]],
        *,
        model: str = "claude-haiku-4-5-20251001",
        max_turns: int = 8,
        max_tokens: int = 1024,
        system: str | None = None,
        api_key: str | None = None,
        on_text: Any = None,
    ) -> None:
        self._tools = tools
        self._model = model
        self._max_turns = max_turns
        self._max_tokens = max_tokens
        self._system = system
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._bus = ToolResultBus()
        self._on_text = on_text  # optional callback(str) for live narration

    # The kernel looks for this to wire the result callback.
    def get_bus(self) -> ToolResultBus:
        return self._bus

    def _post(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": messages,
            "tools": self._tools,
        }
        if self._system:
            body["system"] = self._system
        req = urllib.request.Request(
            _API_URL,
            data=json.dumps(body).encode(),
            headers={
                "content-type": "application/json",
                "x-api-key": self._api_key,
                "anthropic-version": _API_VERSION,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:  # surface API errors usefully
            detail = exc.read().decode(errors="replace")
            raise RuntimeError(f"Anthropic API {exc.code}: {detail}") from exc

    async def stream(
        self, envelope: ExecutionEnvelope
    ) -> AsyncIterator[ExecutorEvent]:
        node_id = envelope.node_id
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": envelope.task}
        ]
        last_usage: dict[str, Any] = {}

        for _turn in range(self._max_turns):
            response = await asyncio.to_thread(self._post, messages)
            last_usage = response.get("usage", {}) or {}
            content = response.get("content", []) or []
            messages.append({"role": "assistant", "content": content})

            tool_results: list[dict[str, Any]] = []
            for block in content:
                btype = block.get("type")
                if btype == "text":
                    text = block.get("text", "")
                    if self._on_text is not None and text:
                        self._on_text(text)
                    yield ExecutorEvent(
                        kind=ExecutorEventKind.TEXT,
                        payload={"text": text},
                        node_id=node_id,
                    )
                elif btype == "tool_use":
                    tool_use_id = block.get("id", "")
                    # Hand the call to the kernel. When this yield resumes, the
                    # governed result is already in the bus.
                    yield ExecutorEvent(
                        kind=ExecutorEventKind.TOOL_USE,
                        payload={
                            "tool": block.get("name", ""),
                            "args": block.get("input", {}) or {},
                            "tool_use_id": tool_use_id,
                        },
                        node_id=node_id,
                    )
                    governed = self._bus.take(tool_use_id)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": _stringify_tool_result(governed),
                        }
                    )

            if response.get("stop_reason") == "tool_use" and tool_results:
                messages.append({"role": "user", "content": tool_results})
                continue

            # No more tools requested — the model is done.
            yield ExecutorEvent(
                kind=ExecutorEventKind.STOP,
                payload={"usage": _usage_to_tokens(last_usage)},
                node_id=node_id,
            )
            return

        # Hit the turn cap.
        yield ExecutorEvent(
            kind=ExecutorEventKind.STOP,
            payload={"usage": _usage_to_tokens(last_usage)},
            node_id=node_id,
        )


def _usage_to_tokens(usage: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "tool_tokens": 0,
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
        "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
    }
