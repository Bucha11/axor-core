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


# ── Normalizer ─────────────────────────────────────────────────────────────────

class NormalizerError(AxorError):
    """Provider tool call is recognised but malformed or incomplete."""
    def __init__(self, provider: str, reason: str) -> None:
        self.provider = provider
        self.reason = reason
        super().__init__(f"Normalizer error [{provider}]: {reason}")


class UnknownProviderFormatError(NormalizerError):
    """Provider emitted a tool call format not recognised by any normalizer."""
    def __init__(self, provider: str, event_type: str) -> None:
        self.event_type = event_type
        super().__init__(
            provider=provider,
            reason=f"unknown event type '{event_type}' — execution denied",
        )


# ── Governance bypass ──────────────────────────────────────────────────────────

class GovernanceBypassError(AxorError):
    """Raised when executor stream() is called outside an active governance_context() in PRODUCTION/STRICT mode."""
    def __init__(self, detail: str = "") -> None:
        msg = "Direct executor call bypasses governance"
        if detail:
            msg = f"{msg}: {detail}"
        super().__init__(msg)


class IsolationRequiredError(AxorError):
    """Raised when process isolation is required but the capability executor is in-process.

    In PRODUCTION/STRICT with require_isolation (or AXOR_REQUIRE_ISOLATION=1), an
    untrusted agent must execute tools out-of-process (DaemonCapabilityClient).
    An in-process CapabilityExecutor offers no hard boundary against a
    compromised agent process.
    """
    def __init__(self, detail: str = "") -> None:
        msg = "process isolation required but capability executor is in-process"
        if detail:
            msg = f"{msg}: {detail}"
        super().__init__(msg)


# ── Taint ──────────────────────────────────────────────────────────────────────

class TaintClearanceError(AxorError):
    """Worker attempted to clear taint — only governance may do this."""
    def __init__(self, reason: str = "") -> None:
        msg = "Taint clearance may only be initiated by governance"
        if reason:
            msg = f"{msg}: {reason}"
        super().__init__(msg)


# ── Spawn validation ───────────────────────────────────────────────────────────

class SpawnValidationError(AxorError):
    """Child spawn failed policy-ceiling or taint-inheritance validation."""
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Spawn validation failed: {reason}")


# ── Budget ─────────────────────────────────────────────────────────────────────

class DaemonUnavailableError(AxorError):
    """AxorDaemon is not reachable. Fail-closed: execution stops."""
    def __init__(self, socket_path: str, reason: str = "") -> None:
        self.socket_path = socket_path
        msg = f"AxorDaemon unavailable at {socket_path}"
        if reason:
            msg = f"{msg}: {reason}"
        super().__init__(msg)


class DaemonRejectedError(AxorError):
    """Daemon refused the session (mode mismatch, policy violation, etc.)."""
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Daemon rejected session: {reason}")


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


# ── Degradation ────────────────────────────────────────────────────────────────

class SessionTerminatedError(AxorError):
    """Session reached TERMINAL degradation level; no further intents accepted."""
    def __init__(self, reason: str = "") -> None:
        msg = "Session is terminated (DegradationLevel.TERMINAL)"
        if reason:
            msg = f"{msg}: {reason}"
        super().__init__(msg)


class DegradationClearanceError(AxorError):
    """Worker attempted to lower degradation level — only governance may do this."""
    def __init__(self, reason: str = "") -> None:
        msg = "Degradation clearance may only be initiated by governance"
        if reason:
            msg = f"{msg}: {reason}"
        super().__init__(msg)
