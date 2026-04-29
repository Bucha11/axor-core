"""Tests for node/intent_loop and node/export."""
from __future__ import annotations

import asyncio
import pytest
from axor_core.contracts.cancel import make_token, CancelReason
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind
from axor_core.contracts.trace import TraceEventKind
from axor_core.node.intent_loop import IntentLoop
from axor_core.node.export import ExportFilter
from axor_core.contracts.policy import ExportMode
from axor_core.capability.executor import CapabilityExecutor, ToolHandler


async def make_stream(*events):
    for e in events:
        yield e

async def make_slow_stream(*events, delay=0.05):
    for e in events:
        await asyncio.sleep(delay)
        yield e


class MockRead(ToolHandler):
    @property
    def name(self): return "read"
    async def execute(self, args): return f"contents of {args.get('path','?')}"


class TestIntentLoop:

    @pytest.fixture
    def cap_exec(self):
        ex = CapabilityExecutor()
        ex.register(MockRead())
        return ex

    @pytest.mark.asyncio
    async def test_approved_tool_executes(self, cap_exec, make_envelope):
        envelope = make_envelope()
        trace = []
        loop = IntentLoop(cap_exec, trace, current_depth=0)

        events = [
            ExecutorEvent(kind=ExecutorEventKind.TOOL_USE,
                payload={"tool": "read", "args": {"path": "main.py"}}, node_id="n1"),
            ExecutorEvent(kind=ExecutorEventKind.TEXT,
                payload={"text": "done"}, node_id="n1"),
            ExecutorEvent(kind=ExecutorEventKind.STOP,
                payload={"usage": {}}, node_id="n1"),
        ]

        collected = []
        async for e in loop.run(make_stream(*events), envelope):
            collected.append(e)

        # tool_use replaced by tool result TEXT event
        text_events = [e for e in collected if e.kind == ExecutorEventKind.TEXT]
        assert len(text_events) == 2  # tool result + "done"
        assert "contents of main.py" in str(text_events[0].payload)

    @pytest.mark.asyncio
    async def test_denied_tool_returns_denial_result(self, cap_exec, make_envelope):
        envelope = make_envelope()
        trace = []
        loop = IntentLoop(cap_exec, trace, current_depth=0)

        events = [
            ExecutorEvent(kind=ExecutorEventKind.TOOL_USE,
                payload={"tool": "bash", "args": {"cmd": "ls"}}, node_id="n1"),
            ExecutorEvent(kind=ExecutorEventKind.STOP,
                payload={"usage": {}}, node_id="n1"),
        ]

        collected = []
        async for e in loop.run(make_stream(*events), envelope):
            collected.append(e)

        # denial recorded in trace
        denied = [e for e in trace if e.kind == TraceEventKind.INTENT_DENIED]
        assert len(denied) == 1
        assert "bash" in denied[0].intent_kind or "bash" in denied[0].reason

    @pytest.mark.asyncio
    async def test_cancel_stops_stream(self, cap_exec, make_envelope):
        token = make_token()
        envelope = make_envelope(cancel_token=token)
        trace = []
        loop = IntentLoop(cap_exec, trace, current_depth=0)

        events = [
            ExecutorEvent(kind=ExecutorEventKind.TEXT, payload={"text": "a"}, node_id="n1"),
            ExecutorEvent(kind=ExecutorEventKind.TEXT, payload={"text": "b"}, node_id="n1"),
            ExecutorEvent(kind=ExecutorEventKind.TEXT, payload={"text": "c"}, node_id="n1"),
            ExecutorEvent(kind=ExecutorEventKind.STOP, payload={"usage": {}}, node_id="n1"),
        ]

        collected = []

        async def cancel_soon():
            await asyncio.sleep(0.08)
            token.cancel(CancelReason.USER_ABORT, detail="test")

        async def collect():
            async for e in loop.run(make_slow_stream(*events), envelope):
                collected.append(e)

        await asyncio.gather(cancel_soon(), collect())

        # did not collect all 3 text events
        assert len(collected) < 3
        cancelled_events = [e for e in trace if e.kind == TraceEventKind.CANCELLED]
        assert len(cancelled_events) == 1
        assert cancelled_events[0].reason == CancelReason.USER_ABORT.value

    @pytest.mark.asyncio
    async def test_cancel_calls_aclose_on_executor_stream(self, cap_exec, make_envelope):
        """Regression: cancellation must close the executor stream so adapter
        resources (HTTP/streaming contexts) are released, not leaked.
        """
        token = make_token()
        envelope = make_envelope(cancel_token=token)
        trace = []
        loop = IntentLoop(cap_exec, trace, current_depth=0)

        closed = {"flag": False}

        class TrackingStream:
            def __init__(self):
                self._events = iter([
                    ExecutorEvent(kind=ExecutorEventKind.TEXT, payload={"text": "a"}, node_id="n1"),
                    ExecutorEvent(kind=ExecutorEventKind.TEXT, payload={"text": "b"}, node_id="n1"),
                    ExecutorEvent(kind=ExecutorEventKind.STOP, payload={"usage": {}}, node_id="n1"),
                ])

            def __aiter__(self):
                return self

            async def __anext__(self):
                await asyncio.sleep(0.05)
                try:
                    return next(self._events)
                except StopIteration:
                    raise StopAsyncIteration

            async def aclose(self):
                closed["flag"] = True

        stream = TrackingStream()

        async def cancel_soon():
            await asyncio.sleep(0.07)
            token.cancel(CancelReason.USER_ABORT)

        async def collect():
            async for _ in loop.run(stream, envelope):
                pass

        await asyncio.gather(cancel_soon(), collect())
        assert closed["flag"] is True

    @pytest.mark.asyncio
    async def test_unexpected_tool_exception_is_logged_and_denied(
        self, make_envelope, caplog,
    ):
        """Regression: programming bugs in handlers used to vanish silently as
        denials. They still convert to denial (so the conversation continues),
        but a full traceback is logged so the bug is visible.
        """
        class BoomHandler(ToolHandler):
            @property
            def name(self): return "read"
            async def execute(self, args):
                raise RuntimeError("internal handler bug")

        cap_exec = CapabilityExecutor()
        cap_exec.register(BoomHandler())

        envelope = make_envelope()
        trace = []
        loop = IntentLoop(cap_exec, trace, current_depth=0)

        events = [
            ExecutorEvent(
                kind=ExecutorEventKind.TOOL_USE,
                payload={"tool": "read", "tool_use_id": "x", "args": {"path": "/a"}},
                node_id="n1",
            ),
            ExecutorEvent(kind=ExecutorEventKind.STOP, payload={"usage": {}}, node_id="n1"),
        ]

        collected = []
        with caplog.at_level("ERROR", logger="axor.intent_loop"):
            async for e in loop.run(make_stream(*events), envelope):
                collected.append(e)

        # Logged the unexpected exception with traceback.
        assert any("internal handler bug" in r.message for r in caplog.records)
        # Conversation still produced a denial event for the model to see.
        denied = [e for e in trace if e.kind == TraceEventKind.INTENT_DENIED]
        assert len(denied) == 1
        assert "RuntimeError" in denied[0].reason

    @pytest.mark.asyncio
    async def test_token_usage_recorded_in_trace(self, cap_exec, make_envelope):
        envelope = make_envelope()
        trace = []
        loop = IntentLoop(cap_exec, trace, current_depth=0)

        events = [
            ExecutorEvent(kind=ExecutorEventKind.STOP,
                payload={"usage": {"input_tokens": 200, "output_tokens": 100, "tool_tokens": 30}},
                node_id="n1"),
        ]

        async for _ in loop.run(make_stream(*events), envelope):
            pass

        from axor_core.contracts.trace import TokensSpentEvent
        spent = [e for e in trace if isinstance(e, TokensSpentEvent)]
        assert len(spent) == 1
        assert spent[0].input_tokens == 200
        assert spent[0].output_tokens == 100

    @pytest.mark.asyncio
    async def test_cache_tokens_propagated_to_trace(self, cap_exec, make_envelope):
        envelope = make_envelope()
        trace = []
        loop = IntentLoop(cap_exec, trace, current_depth=0)

        events = [
            ExecutorEvent(kind=ExecutorEventKind.STOP,
                payload={"usage": {
                    "input_tokens": 200, "output_tokens": 100, "tool_tokens": 0,
                    "cache_creation_input_tokens": 1000,
                    "cache_read_input_tokens":     8000,
                }},
                node_id="n1"),
        ]

        async for _ in loop.run(make_stream(*events), envelope):
            pass

        from axor_core.contracts.trace import TokensSpentEvent
        spent = [e for e in trace if isinstance(e, TokensSpentEvent)]
        assert len(spent) == 1
        assert spent[0].cache_creation_input_tokens == 1000
        assert spent[0].cache_read_input_tokens == 8000

    @pytest.mark.asyncio
    async def test_cache_tokens_default_zero_when_absent(self, cap_exec, make_envelope):
        # Older models / providers without cache fields → defaults stay 0.
        envelope = make_envelope()
        trace = []
        loop = IntentLoop(cap_exec, trace, current_depth=0)

        events = [
            ExecutorEvent(kind=ExecutorEventKind.STOP,
                payload={"usage": {"input_tokens": 100, "output_tokens": 50}},
                node_id="n1"),
        ]

        async for _ in loop.run(make_stream(*events), envelope):
            pass

        from axor_core.contracts.trace import TokensSpentEvent
        spent = [e for e in trace if isinstance(e, TokensSpentEvent)]
        assert spent[0].cache_creation_input_tokens == 0
        assert spent[0].cache_read_input_tokens == 0


class TestExportFilter:

    @pytest.fixture
    def export_filter(self):
        return ExportFilter()

    def test_summary_mode_returns_output_only(self, export_filter, make_envelope):
        from axor_core.contracts.result import TokenUsage
        envelope = make_envelope()
        result = export_filter.apply(
            raw_output="full output text",
            raw_payload={"output": "full output text", "reasoning": "internal"},
            envelope=envelope,
            token_usage=TokenUsage(100, 50, 10, 20),
        )
        assert result.output == "full output text"
        assert "reasoning" not in result.export_payload

    def test_restricted_mode_returns_empty(self, export_filter, make_envelope, focused_policy):
        from axor_core.contracts.result import TokenUsage
        from axor_core.contracts.policy import ExportMode
        import dataclasses
        restricted_policy = dataclasses.replace(focused_policy, export_mode=ExportMode.RESTRICTED)
        envelope = make_envelope(policy=restricted_policy)
        result = export_filter.apply(
            raw_output="secret output",
            raw_payload={},
            envelope=envelope,
            token_usage=TokenUsage(0, 0, 0, 0),
        )
        assert result.output == ""
        assert result.export_payload == {}

    def test_truncation_applied_in_summary_mode(self, export_filter, make_envelope):
        from axor_core.contracts.result import TokenUsage
        envelope = make_envelope()
        long_output = "x" * 10000
        result = export_filter.apply(
            raw_output=long_output,
            raw_payload={},
            envelope=envelope,
            token_usage=TokenUsage(0, 0, 0, 0),
        )
        assert len(result.output) < len(long_output)
        assert "truncated" in result.output
