from __future__ import annotations

from axor_core.contracts.policy import ExecutionPolicy, TaskSignal, PolicyDecisionKind
from axor_core.contracts.trace import (
    TraceEvent,
    TraceEventKind,
    SignalChosenEvent,
    PolicyAdjustedEvent,
    IntentDeniedEvent,
    ChildSpawnedEvent,
    TokensSpentEvent,
    CommandRoutedEvent,
    PluginDeniedEvent,
)


def signal_chosen(
    node_id: str,
    raw_input: str,
    signal: TaskSignal,
    confidence: float,
    classifier: str,
) -> SignalChosenEvent:
    return SignalChosenEvent(
        kind=TraceEventKind.SIGNAL_CHOSEN,
        node_id=node_id,
        sequence=0,   # stamped by collector
        raw_input=raw_input,
        signal=signal,
        confidence=confidence,
        classifier=classifier,
    )


def policy_chosen(node_id: str, policy: ExecutionPolicy) -> TraceEvent:
    return TraceEvent(
        kind=TraceEventKind.POLICY_CHOSEN,
        node_id=node_id,
        sequence=0,
        payload={
            "policy_name":    policy.name,
            "context_mode":   policy.context_mode.value,
            "child_mode":     policy.child_mode.value,
            "export_mode":    policy.export_mode.value,
            "max_child_depth": policy.max_child_depth,
        },
    )


def policy_adjusted(
    node_id: str,
    original: TaskSignal,
    adjusted: TaskSignal,
    reason: str,
    tokens_before: int,
) -> PolicyAdjustedEvent:
    """
    Most valuable training signal.
    Records that the initial classification was wrong and how much it cost.
    """
    return PolicyAdjustedEvent(
        kind=TraceEventKind.POLICY_ADJUSTED,
        node_id=node_id,
        sequence=0,
        original_signal=original,
        adjusted_signal=adjusted,
        reason=reason,
        tokens_spent_before_adjustment=tokens_before,
    )


def intent_denied(node_id: str, intent_kind: str, reason: str) -> IntentDeniedEvent:
    return IntentDeniedEvent(
        kind=TraceEventKind.INTENT_DENIED,
        node_id=node_id,
        sequence=0,
        intent_kind=intent_kind,
        reason=reason,
    )


def child_spawned(
    node_id: str,
    child_node_id: str,
    child_depth: int,
    context_fraction: float,
) -> ChildSpawnedEvent:
    return ChildSpawnedEvent(
        kind=TraceEventKind.CHILD_SPAWNED,
        node_id=node_id,
        sequence=0,
        child_node_id=child_node_id,
        child_depth=child_depth,
        context_fraction=context_fraction,
    )


def tokens_spent(
    node_id: str,
    input_tokens: int,
    output_tokens: int,
    tool_tokens: int,
    context_tokens: int,
    cumulative: int,
) -> TokensSpentEvent:
    return TokensSpentEvent(
        kind=TraceEventKind.TOKENS_SPENT,
        node_id=node_id,
        sequence=0,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tool_tokens=tool_tokens,
        context_tokens=context_tokens,
        cumulative=cumulative,
    )


def command_routed(
    node_id: str,
    command_name: str,
    command_class: str,
    allowed: bool,
) -> CommandRoutedEvent:
    return CommandRoutedEvent(
        kind=TraceEventKind.COMMAND_ROUTED,
        node_id=node_id,
        sequence=0,
        command_name=command_name,
        command_class=command_class,
        allowed=allowed,
    )


def plugin_denied(
    node_id: str,
    plugin_name: str,
    denied_item: str,
    reason: str,
) -> PluginDeniedEvent:
    return PluginDeniedEvent(
        kind=TraceEventKind.PLUGIN_DENIED,
        node_id=node_id,
        sequence=0,
        plugin_name=plugin_name,
        denied_item=denied_item,
        reason=reason,
    )


def context_compressed(
    node_id: str,
    before_tokens: int,
    after_tokens: int,
    compression_ratio: float,
) -> TraceEvent:
    return TraceEvent(
        kind=TraceEventKind.CONTEXT_COMPRESSED,
        node_id=node_id,
        sequence=0,
        payload={
            "before_tokens":     before_tokens,
            "after_tokens":      after_tokens,
            "compression_ratio": compression_ratio,
        },
    )


def extension_loaded(
    node_id: str,
    name: str,
    kind: str,   # "fragment" | "tool" | "command" | "hook"
    source: str,
) -> TraceEvent:
    return TraceEvent(
        kind=TraceEventKind.EXTENSION_LOADED,
        node_id=node_id,
        sequence=0,
        payload={"name": name, "kind": kind, "source": source},
    )
