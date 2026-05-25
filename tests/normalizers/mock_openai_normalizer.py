from __future__ import annotations

"""
Mock OpenAI ProviderNormalizer for cross-provider parity tests.

Simulates the normalizer that a real axor-openai adapter would provide.
Handles:
  - Standard single tool call dict: {"id": str, "type": "function", "function": {"name": str, "arguments": str}}
  - Parallel tool calls: list of the above

Does NOT exist as a production adapter — tests only.
"""

import json
from typing import Any

from axor_core.contracts.anomaly import NormalizedIntent
from axor_core.contracts.intent import Intent, IntentKind
from axor_core.errors.exceptions import NormalizerError, UnknownProviderFormatError
from axor_core.node.normalizer import IntentNormalizer

_PROVIDER = "openai"


class MockOpenAINormalizer:
    """
    Test-only normalizer for OpenAI tool call format.

    Handles single and parallel tool calls.
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

    def normalize_parallel(self, raw_events: list[Any]) -> list[NormalizedIntent]:
        """Normalise each parallel tool call independently."""
        return [self.normalize(e) for e in raw_events]

    def _extract(self, raw_event: Any) -> tuple[str, dict]:
        if not isinstance(raw_event, dict):
            raise UnknownProviderFormatError(_PROVIDER, type(raw_event).__name__)
        if raw_event.get("type") != "function":
            raise UnknownProviderFormatError(_PROVIDER, str(raw_event.get("type")))
        fn = raw_event.get("function", {})
        name = fn.get("name")
        arguments_str = fn.get("arguments", "{}")
        if not name:
            raise NormalizerError(_PROVIDER, "tool call missing function.name")
        try:
            args = json.loads(arguments_str) if arguments_str else {}
        except json.JSONDecodeError as exc:
            raise NormalizerError(_PROVIDER, f"invalid JSON in function.arguments: {exc}") from exc
        if not isinstance(args, dict):
            raise NormalizerError(_PROVIDER, "function.arguments must be a JSON object")
        return str(name), args
