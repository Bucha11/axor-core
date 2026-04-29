from __future__ import annotations


class AxorError(Exception):
    """Base for all axor_core exceptions."""


# ── Policy ─────────────────────────────────────────────────────────────────────

class PolicyError(AxorError):
    """Policy selection or composition failed."""


# ── Intent ─────────────────────────────────────────────────────────────────────

class IntentDeniedError(AxorError):
    """An intent was denied by the node."""
    def __init__(self, kind: str, reason: str) -> None:
        self.kind = kind
        self.reason = reason
        super().__init__(f"Intent '{kind}' denied: {reason}")


class IntentResolutionError(AxorError):
    """Intent resolution encountered an unexpected error."""


# ── Capability ─────────────────────────────────────────────────────────────────

class ToolNotAllowedError(AxorError):
    """Tool is not in the node's Capabilities."""
    def __init__(self, tool: str, allowed: set[str]) -> None:
        self.tool = tool
        self.allowed = allowed
        super().__init__(
            f"Tool '{tool}' is not allowed. Allowed tools: {sorted(allowed)}"
        )


class ToolNotFoundError(AxorError):
    """Tool is in Capabilities but no handler is registered."""
    def __init__(self, tool: str) -> None:
        self.tool = tool
        super().__init__(
            f"No handler registered for tool '{tool}'. "
            "Register a ToolHandler via CapabilityExecutor.register()."
        )


# ── Federation ─────────────────────────────────────────────────────────────────

class ChildNotAllowedError(AxorError):
    """spawn_child intent denied — children not allowed by policy."""
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Child node creation denied: {reason}")


class MaxDepthExceededError(AxorError):
    """spawn_child intent denied — would exceed max_child_depth."""
    def __init__(self, current: int, max_depth: int) -> None:
        self.current = current
        self.max_depth = max_depth
        super().__init__(
            f"Max child depth exceeded: current={current}, max={max_depth}"
        )


# ── Context ────────────────────────────────────────────────────────────────────

class ContextError(AxorError):
    """Context subsystem error."""


class ContextExpansionDeniedError(AxorError):
    """expand_context intent denied — context expansion not allowed by policy."""


# ── Export ─────────────────────────────────────────────────────────────────────

class ExportDeniedError(AxorError):
    """Export intent denied by export contract."""
    def __init__(self, mode: str, reason: str) -> None:
        self.mode = mode
        self.reason = reason
        super().__init__(f"Export denied (mode={mode}): {reason}")


# ── Extensions ─────────────────────────────────────────────────────────────────

class ExtensionSanitizationError(AxorError):
    """Extension failed sanitization and cannot be loaded."""
    def __init__(self, name: str, reason: str) -> None:
        self.name = name
        self.reason = reason
        super().__init__(f"Extension '{name}' failed sanitization: {reason}")


# ── Budget ─────────────────────────────────────────────────────────────────────

class BudgetExceededError(AxorError):
    """
    Hard token budget would be exceeded by the next operation.

    Raised by adapters / middleware when a model call's projected token
    cost would push cumulative spend past `hard_token_limit`. Stops the
    agent loop instead of silently overspending.

    Carries:
      • spent: tokens already consumed
      • projected: tokens the next call would add
      • limit: configured hard cap
    """
    def __init__(self, spent: int, projected: int, limit: int) -> None:
        self.spent = spent
        self.projected = projected
        self.limit = limit
        super().__init__(
            f"Budget exceeded: {spent} spent + {projected} projected "
            f"> {limit} hard limit"
        )
