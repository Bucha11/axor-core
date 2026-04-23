"""
axor_core.contracts
───────────────────
Pure shared contracts and models.
No business logic. No provider imports. No side effects.

Everything in the system depends on these.
Nothing here depends on anything else in axor_core.
"""

from axor_core.contracts.invokable import Invokable
from axor_core.contracts.cancel import CancelToken, CancelReason, make_token
from axor_core.contracts.context import (
    ContextView,
    ContextFragment,
    LineageSummary,
    RawExecutionState,
)
from axor_core.contracts.policy import (
    TaskSignal,
    TaskComplexity,
    TaskNature,
    SignalClassifier,
    ExecutionPolicy,
    PolicyDecision,
    PolicyDecisionKind,
    ContextMode,
    CompressionMode,
    ExportMode,
    ChildMode,
    ToolPolicy,
)
from axor_core.contracts.envelope import (
    ExecutionEnvelope,
    Capabilities,
    ExportContract,
)
from axor_core.contracts.intent import (
    Intent,
    IntentKind,
    ResolvedIntent,
)
from axor_core.contracts.result import (
    ExecutionResult,
    ExecutorEvent,
    ExecutorEventKind,
    TokenUsage,
)
from axor_core.contracts.command import (
    SlashCommand,
    CommandClass,
    CommandResult,
)
from axor_core.contracts.extension import (
    ExtensionFragment,
    ExtensionTool,
    ExtensionCommand,
    ExtensionHook,
    ExtensionBundle,
    ExtensionLoader,
)
from axor_core.contracts.trace import (
    TraceEvent,
    TraceEventKind,
    DecisionTrace,
    TraceConfig,
    AnonymizedTraceRecord,
    SignalChosenEvent,
    PolicyAdjustedEvent,
    IntentDeniedEvent,
    ChildSpawnedEvent,
    TokensSpentEvent,
    CommandRoutedEvent,
    PluginDeniedEvent,
    CancelledEvent,
    Embedder,
    TelemetrySink,
)

__all__ = [
    # invokable
    "Invokable",
    # cancel
    "CancelToken", "CancelReason", "make_token",
    # context
    "ContextView", "ContextFragment", "LineageSummary", "RawExecutionState",
    # policy
    "TaskSignal", "TaskComplexity", "TaskNature", "SignalClassifier",
    "ExecutionPolicy", "PolicyDecision", "PolicyDecisionKind",
    "ContextMode", "CompressionMode", "ExportMode", "ChildMode", "ToolPolicy",
    # envelope
    "ExecutionEnvelope", "Capabilities", "ExportContract",
    # intent
    "Intent", "IntentKind", "ResolvedIntent",
    # result
    "ExecutionResult", "ExecutorEvent", "ExecutorEventKind", "TokenUsage",
    # command
    "SlashCommand", "CommandClass", "CommandResult",
    # extension
    "ExtensionFragment", "ExtensionTool", "ExtensionCommand",
    "ExtensionHook", "ExtensionBundle", "ExtensionLoader",
    # trace
    "TraceEvent", "TraceEventKind", "DecisionTrace", "TraceConfig",
    "AnonymizedTraceRecord", "SignalChosenEvent", "PolicyAdjustedEvent",
    "IntentDeniedEvent", "ChildSpawnedEvent", "TokensSpentEvent",
    "CommandRoutedEvent", "PluginDeniedEvent", "CancelledEvent",
    "Embedder", "TelemetrySink",
]
