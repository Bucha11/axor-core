from __future__ import annotations

"""
Test-only minimal implementations of AxorMiddleware and AxorToolWrapper.

These replicate the governance behavior under test without importing
langchain, langchain_core, or pydantic — keeping axor-core tests
self-contained with zero adapter dependencies.
"""

import logging
from typing import Any, Callable

from axor_core.contracts.anomaly import NormalizedIntent
from axor_core.contracts.intent import Intent, IntentKind
from axor_core.errors.exceptions import NormalizerError, UnknownProviderFormatError
from axor_core.policy.normalizer import IntentNormalizer

_log = logging.getLogger("axor.langchain")


# ── LangChainNormalizer (copy from axor_langchain.normalizer) ─────────────────

class LangChainNormalizer:
    """Test-only copy of axor_langchain.normalizer.LangChainNormalizer."""

    _PROVIDER = "langchain"

    def __init__(self, workdir: str | None = None) -> None:
        self._intent_normalizer = IntentNormalizer(workdir=workdir)

    def normalize(self, raw_event: Any) -> NormalizedIntent:
        tool_name, args = self._extract_tool_and_args(raw_event)
        intent = Intent(
            kind=IntentKind.TOOL_CALL,
            payload={"tool": tool_name, "args": args},
            node_id="",
        )
        return self._intent_normalizer.normalize(intent)

    def _extract_tool_and_args(self, raw_event: Any) -> tuple[str, dict]:
        if not isinstance(raw_event, dict):
            raise UnknownProviderFormatError(self._PROVIDER, type(raw_event).__name__)
        if "tool" in raw_event:
            tool_name = raw_event["tool"]
            args = raw_event.get("tool_input", raw_event.get("args", {}))
            if not isinstance(args, dict):
                raise NormalizerError(
                    self._PROVIDER,
                    f"tool_input must be a dict, got {type(args).__name__}",
                )
            return str(tool_name), args
        if "name" in raw_event:
            tool_name = raw_event["name"]
            args = raw_event.get("args", {})
            if not isinstance(args, dict):
                raise NormalizerError(
                    self._PROVIDER,
                    f"args must be a dict, got {type(args).__name__}",
                )
            return str(tool_name), args
        raise UnknownProviderFormatError(
            self._PROVIDER,
            f"dict missing 'tool' or 'name' key — got {list(raw_event.keys())}",
        )


# ── AxorMiddleware ─────────────────────────────────────────────────────────────

class AxorMiddleware:
    """
    Test-only minimal AxorMiddleware.

    Implements only the governance behaviors exercised by axor-core tests:
      - _warn_callback_only_once(): emits a warning when used in callback-only mode.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._wrap_tools_called = False

    def _warn_callback_only_once(self) -> None:
        """Warn once when used without wrap_tools — callback-only mode."""
        if not self._wrap_tools_called:
            _log.warning(
                "AxorCallbackHandler provides observability only — "
                "use AxorToolWrapper for enforcement"
            )
            self._wrap_tools_called = True


# ── AxorToolWrapper ────────────────────────────────────────────────────────────

class AxorToolWrapper:
    """
    Test-only minimal AxorToolWrapper.

    Implements only the governance enforcement behavior:
      - _run(): evaluates policy_fn before calling the wrapped tool.
        Returns "tool_denied: <reason>" when policy denies; otherwise
        delegates to the wrapped tool's _run().
    """

    def __init__(
        self,
        tool: Any,
        policy_fn: Callable | None = None,
        **kwargs: Any,
    ) -> None:
        self.name = getattr(tool, "name", "unknown")
        self.description = getattr(tool, "description", "")
        self._wrapped = tool
        self._policy_fn = policy_fn
        self._normalizer = LangChainNormalizer()

    def _evaluate(self, tool_input: dict) -> str | None:
        if self._policy_fn is None:
            return None
        try:
            normalized = self._normalizer.normalize(
                {"tool": self.name, "args": tool_input}
            )
        except Exception:
            return "normalizer_error: tool blocked due to normalization failure"
        return self._policy_fn(normalized)

    def _run(self, *args: Any, **kwargs: Any) -> str:
        tool_input = kwargs if kwargs else ({"input": args[0]} if args else {})
        denial = self._evaluate(tool_input)
        if denial:
            return f"tool_denied: {denial}"
        return self._wrapped._run(*args, **kwargs)

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        tool_input = kwargs if kwargs else ({"input": args[0]} if args else {})
        denial = self._evaluate(tool_input)
        if denial:
            return f"tool_denied: {denial}"
        return await self._wrapped._arun(*args, **kwargs)
