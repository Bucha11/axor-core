from __future__ import annotations

"""
Mock OpenRouter ProviderNormalizer for cross-provider parity tests.

OpenRouter proxies OpenAI-compatible tool call format. This mock
normalizer also handles streaming partial tool call accumulation.

Does NOT exist as a production adapter — tests only.
"""

import json
from typing import Any

from axor_core.contracts.anomaly import NormalizedIntent
from axor_core.contracts.intent import Intent, IntentKind
from axor_core.errors.exceptions import NormalizerError, UnknownProviderFormatError
from axor_core.node.normalizer import IntentNormalizer

_PROVIDER = "openrouter"


class MockOpenRouterNormalizer:
    """
    Test-only normalizer for OpenRouter (OpenAI-compat) tool call format.

    Handles accumulated streaming tool call format:
      {"tool": str, "args": dict}  — post-accumulation form
    and OpenAI-compat object:
      {"type": "function", "function": {"name": str, "arguments": str}}
    """

    def __init__(self, workdir: str | None = None) -> None:
        self._intent_normalizer = IntentNormalizer(workdir=workdir)

    def normalize(self, raw_event: Any) -> NormalizedIntent:
        tool_name, args = self._extract(raw_event)
        intent = Intent(
            kind=IntentKind.TOOL_CALL,
            payload={"tool": tool_name, "args": args},
            node_id="",
        )
        return self._intent_normalizer.normalize(intent)

    def _extract(self, raw_event: Any) -> tuple[str, dict]:
        if not isinstance(raw_event, dict):
            raise UnknownProviderFormatError(_PROVIDER, type(raw_event).__name__)
        # Accumulated streaming form (same as ClaudeNormalizer dict form)
        if "tool" in raw_event and "args" in raw_event:
            args = raw_event["args"]
            if not isinstance(args, dict):
                raise NormalizerError(_PROVIDER, f"args must be dict, got {type(args).__name__}")
            return str(raw_event["tool"]), dict(args)
        # OpenAI-compat object form
        if raw_event.get("type") == "function":
            fn = raw_event.get("function", {})
            name = fn.get("name")
            arguments_str = fn.get("arguments", "{}")
            if not name:
                raise NormalizerError(_PROVIDER, "missing function.name")
            try:
                args = json.loads(arguments_str) if arguments_str else {}
            except json.JSONDecodeError as exc:
                raise NormalizerError(_PROVIDER, f"invalid JSON: {exc}") from exc
            if not isinstance(args, dict):
                raise NormalizerError(_PROVIDER, "function.arguments must be a JSON object")
            return str(name), args
        raise UnknownProviderFormatError(
            _PROVIDER,
            f"unrecognised format keys: {list(raw_event.keys())}",
        )
