"""AgentDojo adapter for axor-core.

Two pipeline elements:

- :class:`RawAnthropicLLM` — an AgentDojo ``BasePipelineElement`` that drives a
  real Claude model through raw ``urllib`` (the Anthropic SDK cannot connect from
  this sandbox; raw POST works, and it keeps the example SDK-free). It produces
  the same ``ChatAssistantMessage`` / ``FunctionCall`` shapes AgentDojo's own
  ``AnthropicLLM`` does, so it is a drop-in LLM.

- :class:`GovernedToolsExecutor` — replaces AgentDojo's ``ToolsExecutor``. Before
  each tool call runs, it asks an axor :class:`~axor_core.governor.ToolCallGovernor`
  whether the call is permitted. A denied call never touches the real tool; the
  model gets a governance-denial tool result and continues. Allowed calls run
  normally, and their output is registered in the governor's per-value ledger so a
  later sink carrying attacker-controlled content is gated.

The governor is configured with the suite's tool taxonomy (which tools are
untrusted-data sources, which are egress sinks) — that declaration is how an
operator deploys axor over a real tool set, exactly as it would in production.
"""
from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from collections.abc import Sequence

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionCall, FunctionsRuntime
from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    ChatToolResultMessage,
    text_content_block_from_string,
)

from axor_core.governor import ToolCallGovernor

_API_URL = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"


# ── Raw-urllib LLM element ──────────────────────────────────────────────────────

def _messages_to_anthropic(messages: Sequence[ChatMessage]):
    """Convert AgentDojo chat messages to (system, anthropic_messages) as plain
    dicts suitable for a raw API POST."""
    system = None
    out: list[dict] = []
    for msg in messages:
        role = msg["role"]
        if role == "system":
            system = msg["content"][0]["content"]
        elif role == "user":
            text = "\n".join(b["content"] for b in msg["content"] if b.get("content"))
            out.append({"role": "user", "content": [{"type": "text", "text": text}]})
        elif role == "assistant":
            blocks: list[dict] = []
            for b in msg["content"] or []:
                if b["type"] == "text" and b["content"]:
                    blocks.append({"type": "text", "text": b["content"]})
            for tc in msg["tool_calls"] or []:
                blocks.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.function,
                    "input": dict(tc.args),
                })
            if blocks:
                out.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            is_error = msg["error"] is not None
            text = msg["error"] if is_error else (
                msg["content"][0]["content"] if msg["content"] else ""
            )
            out.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": msg["tool_call_id"],
                "content": [{"type": "text", "text": text or ""}],
                "is_error": is_error,
            }]})
    # Merge consecutive user messages (Anthropic requires alternating roles).
    merged: list[dict] = []
    for m in out:
        if merged and merged[-1]["role"] == m["role"] == "user":
            merged[-1]["content"].extend(m["content"])
        else:
            merged.append(m)
    return system, merged


class RawAnthropicLLM(BasePipelineElement):
    def __init__(self, model: str = "claude-haiku-4-5-20251001", max_tokens: int = 1024) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.name = f"raw-anthropic-{model}"
        self._key = os.environ.get("ANTHROPIC_API_KEY", "")

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(
            _API_URL,
            data=json.dumps(body).encode(),
            headers={
                "content-type": "application/json",
                "x-api-key": self._key,
                "anthropic-version": _API_VERSION,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Anthropic API {exc.code}: {exc.read().decode(errors='replace')}") from exc

    def query(self, query, runtime, env=EmptyEnv(), messages=[], extra_args={}):
        system, anthropic_messages = _messages_to_anthropic(messages)
        tools = [
            {
                "name": f.name,
                "description": f.description,
                "input_schema": f.parameters.model_json_schema(),
            }
            for f in runtime.functions.values()
        ]
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": anthropic_messages,
            "tools": tools,
            "temperature": 0.0,
        }
        if system:
            body["system"] = system
        resp = self._post(body)

        content = resp.get("content", []) or []
        text_blocks = [
            text_content_block_from_string(b["text"])
            for b in content if b.get("type") == "text"
        ]
        tool_calls = [
            FunctionCall(id=b["id"], function=b["name"], args=b.get("input", {}) or {})
            for b in content if b.get("type") == "tool_use"
        ]
        assistant = ChatAssistantMessage(
            role="assistant",
            content=text_blocks,
            tool_calls=tool_calls if resp.get("stop_reason") == "tool_use" else None,
        )
        return query, runtime, env, [*messages, assistant], extra_args


# ── Governed tool executor ──────────────────────────────────────────────────────

class GovernedToolsExecutor(BasePipelineElement):
    """Runs the last assistant message's tool calls, each gated by axor first.

    Pass a *factory* that builds a fresh :class:`ToolCallGovernor` — the harness
    reuses one executor across tasks, but each task needs its own per-value ledger,
    so a new governor is created when a task's conversation starts (detected by the
    message list resetting to its first tool turn).
    """

    def __init__(self, governor_factory) -> None:
        self._make_governor = governor_factory
        self._governor: ToolCallGovernor | None = None
        self._seen_ids: set[str] = set()
        self.denied_count = 0
        self.denials: list[str] = []

    def _governor_for(self, messages) -> ToolCallGovernor:
        # New task: the very first tool-execution turn has exactly one assistant
        # message with tool calls and no prior tool results. Reset the ledger then.
        has_prior_tool_result = any(m["role"] == "tool" for m in messages[:-1])
        if self._governor is None or not has_prior_tool_result:
            self._governor = self._make_governor()
        return self._governor

    def query(self, query, runtime, env=EmptyEnv(), messages=[], extra_args={}):
        if not messages or messages[-1]["role"] != "assistant":
            return query, runtime, env, messages, extra_args
        tool_calls = messages[-1]["tool_calls"]
        if not tool_calls:
            return query, runtime, env, messages, extra_args

        governor = self._governor_for(messages)
        results: list[ChatToolResultMessage] = []
        for tool_call in tool_calls:
            decision = governor.evaluate(tool_call.function, dict(tool_call.args))
            if not decision.allowed:
                self.denied_count += 1
                self.denials.append(f"{tool_call.function}: {decision.category}")
                results.append(ChatToolResultMessage(
                    role="tool",
                    content=[text_content_block_from_string("")],
                    tool_call_id=tool_call.id,
                    tool_call=tool_call,
                    error=(
                        "Blocked by the execution governance kernel "
                        f"({decision.category}). This action was refused and must "
                        "not be retried."
                    ),
                ))
                continue

            tool_result, error = runtime.run_function(env, tool_call.function, tool_call.args)
            if error is None:
                governor.register_output(decision, tool_result_to_str(tool_result))
            results.append(ChatToolResultMessage(
                role="tool",
                content=[text_content_block_from_string(tool_result_to_str(tool_result))],
                tool_call_id=tool_call.id,
                tool_call=tool_call,
                error=error,
            ))
        return query, runtime, env, [*messages, *results], extra_args
