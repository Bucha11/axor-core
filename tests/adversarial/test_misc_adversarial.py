"""
Adversarial tests: miscellaneous adversarial scenarios — Step 27.

Six adversarial scenarios not covered by the dedicated test files:
  1. Worker self-taint-clear → TaintClearanceError
  2. Direct executor bypass in PRODUCTION mode → GovernanceBypassError
  3. LangChain callback-only mode warns; wrapper mode enforces denial
  4. Unknown provider format → execution denied (NormalizerError/UnknownProviderFormatError)
  5. Incomplete streaming tool call → no TOOL_USE event before content_block_stop
  6. Cross-provider parity: Claude / mock-OpenAI / mock-OpenRouter produce
     identical NormalizedIntent for the same semantic intent
     (3 providers × 5 intent classes = 15 assertions)
"""
from __future__ import annotations

import sys
import logging
import pytest


# ── 1. Worker self-taint-clear → TaintClearanceError ────────────────────────

def test_worker_self_taint_clear_raises():
    """Worker calling attempt_clear_by_worker() always raises TaintClearanceError."""
    from axor_core.contracts.taint import TaintSource
    from axor_core.errors.exceptions import TaintClearanceError
    from axor_core.taint.causal_root import CausalRoot
    from axor_core.taint.engine import TaintEngine

    engine = TaintEngine(node_id="test")
    engine.register_value("WEB_VAL_aabbccddeeff", CausalRoot.external_read(TaintSource.WEB))

    with pytest.raises(TaintClearanceError, match="worker"):
        engine.attempt_clear_by_worker()

    # Per-value provenance must survive the failed worker clear attempt.
    assert engine.derive_value("WEB_VAL_aabbccddeeff").is_tainted


# ── 2. Direct executor bypass in PRODUCTION mode → GovernanceBypassError ─────

def test_production_mode_executor_bypass_raises():
    """
    Calling LockedExecutor.stream() outside a governance_context() in
    PRODUCTION mode raises GovernanceBypassError.
    """
    from unittest.mock import MagicMock
    from axor_core.capability.locked import LockedExecutor
    from axor_core.contracts.mode import ExecutionMode
    from axor_core.errors.exceptions import GovernanceBypassError

    inner = MagicMock()
    locked = LockedExecutor(executor=inner, mode=ExecutionMode.PRODUCTION)

    envelope = MagicMock()
    with pytest.raises(GovernanceBypassError):
        locked.stream(envelope)

    # Inner executor must not have been called
    inner.stream.assert_not_called()


def test_governance_context_allows_executor_call():
    """
    Calling LockedExecutor.stream() inside governance_context() must not raise.
    """
    from unittest.mock import MagicMock
    from axor_core.capability.locked import LockedExecutor, governance_context
    from axor_core.contracts.mode import ExecutionMode

    inner = MagicMock()
    locked = LockedExecutor(executor=inner, mode=ExecutionMode.PRODUCTION)
    envelope = MagicMock()

    with governance_context():
        locked.stream(envelope)

    inner.stream.assert_called_once_with(envelope)


# ── 3. LangChain callback-only mode warns; wrapper enforces ───────────────────

def test_langchain_callback_only_mode_emits_warning(caplog):
    """
    Using AxorMiddleware in callback-only mode (without wrap_tools) logs a
    warning that governance enforcement is not guaranteed.
    """
    from tests.normalizers.mock_langchain import AxorMiddleware

    middleware = AxorMiddleware()

    with caplog.at_level(logging.WARNING, logger="axor.langchain"):
        # Trigger the callback-only warning path directly
        middleware._warn_callback_only_once()

    # A warning about callback-only mode must be emitted
    callback_warnings = [
        r for r in caplog.records
        if "callback" in r.message.lower() or "observability" in r.message.lower()
    ]
    assert len(callback_warnings) >= 1, (
        "middleware in callback-only mode must warn that enforcement is not guaranteed"
    )


def test_langchain_wrapper_enforces_denial():
    """
    AxorToolWrapper with a deny policy blocks the tool even when called directly.

    This confirms that wrapping (not callbacks) is the enforcement mechanism.
    """
    from tests.normalizers.mock_langchain import AxorToolWrapper
    from unittest.mock import MagicMock

    inner_tool = MagicMock()
    inner_tool.name = "bash"
    inner_tool.description = "run bash"

    def deny_all(normalized_intent):
        return "tool denied by policy"

    wrapper = AxorToolWrapper(tool=inner_tool, policy_fn=deny_all)

    result = wrapper._run(command="echo hello")
    assert "tool_denied" in result
    inner_tool._run.assert_not_called()


# ── 4. Unknown provider format → execution denied ─────────────────────────────

def test_unknown_provider_format_raises():
    """
    Feeding an unrecognised event format to a normalizer raises
    UnknownProviderFormatError — execution is denied before it begins.
    """
    sys.path.insert(0, "axor-core")
    from axor_core.errors.exceptions import UnknownProviderFormatError
    from tests.normalizers.mock_openai_normalizer import MockOpenAINormalizer

    normalizer = MockOpenAINormalizer()
    with pytest.raises(UnknownProviderFormatError):
        normalizer.normalize({"bad_format": True, "no_type_field": "present"})


def test_unknown_format_denied_via_claude_normalizer():
    """
    ClaudeNormalizer rejects events that don't match any expected format.
    """
    sys.path.insert(0, "axor-claude")
    from axor_core.errors.exceptions import UnknownProviderFormatError
    from tests.normalizers.mock_claude_normalizer import ClaudeNormalizer

    normalizer = ClaudeNormalizer()
    with pytest.raises(UnknownProviderFormatError):
        normalizer.normalize(42)  # not a dict or ToolUseBlock


# ── 5. Incomplete streaming tool call → no TOOL_USE before content_block_stop ─

def test_incomplete_streaming_no_tool_use_event():
    """
    StreamNormalizer must not emit a TOOL_USE event until content_block_stop.

    An adversary cannot trigger tool execution by sending partial chunks.
    """
    sys.path.insert(0, "axor-claude")
    from tests.normalizers.mock_claude_normalizer import StreamNormalizer
    from dataclasses import dataclass
    from axor_core.contracts.result import ExecutorEventKind

    @dataclass
    class FakeContentBlockStart:
        type: str = "content_block_start"
        index: int = 0
        content_block: object = None

        def __post_init__(self):
            if self.content_block is None:
                @dataclass
                class _Block:
                    type: str = "tool_use"
                    id: str = "call_123"
                    name: str = "bash"
                self.content_block = _Block()

    @dataclass
    class FakeDelta:
        type: str = "input_json_delta"
        partial_json: str = '{"command": "echo'  # incomplete JSON

    @dataclass
    class FakeContentBlockDelta:
        type: str = "content_block_delta"
        index: int = 0
        delta: object = None

        def __post_init__(self):
            if self.delta is None:
                self.delta = FakeDelta()

    normalizer = StreamNormalizer(node_id="test")

    # Start a tool_use block
    events_start = normalizer.process(FakeContentBlockStart())
    assert not any(e.kind == ExecutorEventKind.TOOL_USE for e in events_start), (
        "content_block_start must not emit TOOL_USE"
    )

    # Send an incomplete delta (partial JSON)
    events_delta = normalizer.process(FakeContentBlockDelta())
    assert not any(e.kind == ExecutorEventKind.TOOL_USE for e in events_delta), (
        "content_block_delta must not emit TOOL_USE — tool call not complete yet"
    )


# ── 6. Cross-provider parity ──────────────────────────────────────────────────

def test_cross_provider_parity():
    """
    The same semantic intent fed through Claude, mock-OpenAI, and mock-OpenRouter
    normalizers must produce identical NormalizedIntent for governance purposes.

    5 intent classes × 3 providers = 15 assertions.
    """
    sys.path.insert(0, "axor-claude")
    from tests.normalizers.mock_claude_normalizer import ClaudeNormalizer
    from tests.normalizers.mock_openai_normalizer import MockOpenAINormalizer
    from tests.normalizers.mock_openrouter_normalizer import MockOpenRouterNormalizer

    claude = ClaudeNormalizer()
    openai = MockOpenAINormalizer()
    openrouter = MockOpenRouterNormalizer()

    # Each entry: (intent_class_label, claude_event, openai_event, openrouter_event)
    INTENT_CASES = [
        (
            "file_read",
            {"tool": "Read", "args": {"path": "/src/main.py"}},
            {"type": "function", "function": {"name": "Read", "arguments": '{"path": "/src/main.py"}'}},
            {"tool": "Read", "args": {"path": "/src/main.py"}},
        ),
        (
            "file_write",
            {"tool": "Write", "args": {"path": "/src/out.py", "content": "x"}},
            {"type": "function", "function": {"name": "Write", "arguments": '{"path": "/src/out.py", "content": "x"}'}},
            {"tool": "Write", "args": {"path": "/src/out.py", "content": "x"}},
        ),
        (
            "shell",
            {"tool": "Bash", "args": {"command": "ls"}},
            {"type": "function", "function": {"name": "Bash", "arguments": '{"command": "ls"}'}},
            {"tool": "Bash", "args": {"command": "ls"}},
        ),
        (
            "network",
            {"tool": "WebFetch", "args": {"url": "https://example.com"}},
            {"type": "function", "function": {"name": "WebFetch", "arguments": '{"url": "https://example.com"}'}},
            {"tool": "WebFetch", "args": {"url": "https://example.com"}},
        ),
        (
            "search",
            {"tool": "WebSearch", "args": {"query": "test"}},
            {"type": "function", "function": {"name": "WebSearch", "arguments": '{"query": "test"}'}},
            {"tool": "WebSearch", "args": {"query": "test"}},
        ),
    ]

    for intent_class, claude_evt, openai_evt, openrouter_evt in INTENT_CASES:
        ni_claude = claude.normalize(claude_evt)
        ni_openai = openai.normalize(openai_evt)
        ni_openrouter = openrouter.normalize(openrouter_evt)

        # All three must agree on operation, target_kind, and key booleans
        assert ni_claude.operation == ni_openai.operation == ni_openrouter.operation, (
            f"{intent_class}: operation mismatch: "
            f"claude={ni_claude.operation!r} openai={ni_openai.operation!r} "
            f"openrouter={ni_openrouter.operation!r}"
        )
        assert ni_claude.target_kind == ni_openai.target_kind == ni_openrouter.target_kind, (
            f"{intent_class}: target_kind mismatch"
        )
        assert (
            ni_claude.reads_secret_like_data
            == ni_openai.reads_secret_like_data
            == ni_openrouter.reads_secret_like_data
        ), f"{intent_class}: reads_secret_like_data mismatch"
