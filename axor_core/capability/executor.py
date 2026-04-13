from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Awaitable

from axor_core.contracts.envelope import Capabilities
from axor_core.contracts.intent import Intent, IntentKind
from axor_core.errors.exceptions import ToolNotAllowedError, ToolNotFoundError


class ToolHandler(ABC):
    """
    Contract for executing a single named tool.

    Implemented by adapters — not by core.
    Core only knows that a tool has a name and can be called.

    axor-claude registers handlers for: read, write, bash, search
    Extensions register handlers for their custom tools.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    async def execute(self, args: dict[str, Any]) -> Any:
        ...


# Callback type: (tool_name, args, result) → None
PostExecuteCallback = Callable[[str, dict[str, Any], Any], Awaitable[None]]


class CapabilityExecutor:
    """
    Executes approved tool intents.

    Bridge between intent resolution and actual tool execution.
    Only executes tools that:
      1. Are registered (handler exists)
      2. Are in the node's Capabilities (policy approved)

    Post-execute callbacks allow subsystems to observe results
    without coupling to CapabilityExecutor directly.

    ContextManager registers a callback to:
      - cache file contents after read tool executes
      - memoize tool results for repeat call prevention
      - ingest symbols from read files into SymbolTable

    Called by node/intent_loop.py after an intent is approved.
    Never called directly by executors.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, ToolHandler] = {}
        self._post_callbacks: list[PostExecuteCallback] = []

    def register(self, handler: ToolHandler) -> None:
        """Register a tool handler. Called by adapters at session setup."""
        self._handlers[handler.name] = handler

    def register_post_callback(self, callback: PostExecuteCallback) -> None:
        """
        Register a callback fired after every successful tool execution.
        Callbacks receive (tool_name, args, result).
        Used by ContextManager to observe tool results without tight coupling.
        """
        self._post_callbacks.append(callback)

    def registered_tools(self) -> frozenset[str]:
        return frozenset(self._handlers.keys())

    async def execute(
        self,
        intent: Intent,
        capabilities: Capabilities,
    ) -> Any:
        """
        Execute a tool_call intent against registered handlers.

        Raises:
            ToolNotAllowedError  — tool not in capabilities (policy denied)
            ToolNotFoundError    — tool in capabilities but no handler registered
        """
        if intent.kind != IntentKind.TOOL_CALL:
            raise ValueError(
                f"CapabilityExecutor only handles TOOL_CALL, got {intent.kind}"
            )

        tool_name: str = intent.payload.get("tool", "")
        tool_args: dict[str, Any] = intent.payload.get("args", {})

        if tool_name not in capabilities.allowed_tools:
            raise ToolNotAllowedError(
                tool=tool_name,
                allowed=set(capabilities.allowed_tools),
            )

        handler = self._handlers.get(tool_name)
        if handler is None:
            raise ToolNotFoundError(tool=tool_name)

        result = await handler.execute(tool_args)

        # fire post-execute callbacks — ContextManager observes here
        for callback in self._post_callbacks:
            try:
                await callback(tool_name, tool_args, result)
            except Exception:
                pass  # callbacks must never break execution

        return result
