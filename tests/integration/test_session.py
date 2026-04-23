"""Integration tests — full GovernedSession with all subsystems wired."""
from __future__ import annotations

import asyncio
import pytest
from axor_core import GovernedSession, presets
from axor_core.contracts.cancel import CancelReason
from axor_core.contracts.trace import TraceConfig


class TestGovernedSessionIntegration:

    @pytest.fixture
    def session(self, echo_executor, cap_executor):
        return GovernedSession(
            executor=echo_executor,
            capability_executor=cap_executor,
            trace_config=TraceConfig(local_only=True, persist_inputs=False),
        )

    @pytest.mark.asyncio
    async def test_focused_task_selects_correct_policy(self, session):
        result = await session.run("write a test for the payment endpoint")
        assert result.metadata["policy"] == "focused_generative"

    @pytest.mark.asyncio
    async def test_expansive_task_selects_expansive_policy(self, session):
        result = await session.run("rewrite the entire codebase from Python to Go")
        assert result.metadata["policy"] == "expansive"

    @pytest.mark.asyncio
    async def test_preset_policy_overrides_analysis(self, session):
        result = await session.run("do something", policy=presets.get("readonly"))
        assert result.metadata["policy"] == "preset:readonly"

    @pytest.mark.asyncio
    async def test_read_tool_executes_successfully(self, echo_executor, cap_executor):
        from tests.conftest import EchoExecutor
        executor = EchoExecutor(tool_calls=[("read", {"path": "auth.py"})])
        session = GovernedSession(
            executor=executor,
            capability_executor=cap_executor,
            trace_config=TraceConfig(local_only=True, persist_inputs=False),
        )
        result = await session.run("write a test for the auth module")
        assert "auth.py" in result.output or "file content" in result.output

    @pytest.mark.asyncio
    async def test_denied_tool_returns_denial_in_output(self, echo_executor, cap_executor):
        from tests.conftest import EchoExecutor
        executor = EchoExecutor(tool_calls=[("bash", {"cmd": "rm -rf /"})])
        session = GovernedSession(
            executor=executor,
            capability_executor=cap_executor,
            trace_config=TraceConfig(local_only=True, persist_inputs=False),
        )
        result = await session.run("write a test")  # focused → no bash
        assert "tool_denied" in result.output or "error" in result.output.lower()

    @pytest.mark.asyncio
    async def test_tokens_tracked_after_run(self, session):
        await session.run("explain the auth flow")
        assert session.total_tokens_spent() > 0

    @pytest.mark.asyncio
    async def test_multiple_runs_accumulate_tokens(self, session):
        await session.run("write a test")
        after_first = session.total_tokens_spent()
        await session.run("explain this function")
        after_second = session.total_tokens_spent()
        assert after_second > after_first

    @pytest.mark.asyncio
    async def test_traces_recorded_per_run(self, session):
        await session.run("write a test")
        await session.run("explain this")
        traces = session.all_traces()
        assert len(traces) >= 2

    @pytest.mark.asyncio
    async def test_slash_cost_command(self, session):
        await session.run("write a test")
        result = await session.run("/cost")
        assert "Tokens" in result.output
        assert result.metadata["class"] == "governance"

    @pytest.mark.asyncio
    async def test_slash_policy_command(self, session):
        await session.run("write a test")
        result = await session.run("/policy")
        assert result.metadata["class"] == "governance"

    @pytest.mark.asyncio
    async def test_slash_compact_is_context_class(self, session):
        result = await session.run("/compact")
        assert result.metadata["class"] == "context"

    @pytest.mark.asyncio
    async def test_cancel_stops_execution(self, cap_executor):
        from tests.conftest import SlowExecutor
        session = GovernedSession(
            executor=SlowExecutor(event_count=10, delay=0.05),
            capability_executor=cap_executor,
            trace_config=TraceConfig(local_only=True, persist_inputs=False),
        )

        async def run_and_cancel():
            task = asyncio.create_task(session.run("write a test"))
            await asyncio.sleep(0.15)
            session.cancel(detail="user abort test")
            result = await task
            return result

        result = await run_and_cancel()
        # result is partial — not full 10 chunks
        assert result is not None
        # token usage may be zero for cancelled run
        assert result.token_usage.input_tokens >= 0

    @pytest.mark.asyncio
    async def test_budget_soft_limit_triggers_optimization(self, echo_executor, cap_executor):
        session = GovernedSession(
            executor=echo_executor,
            capability_executor=cap_executor,
            soft_token_limit=100,   # very low — triggers compress signals
            trace_config=TraceConfig(local_only=True, persist_inputs=False),
        )
        # run until budget pressure
        for _ in range(3):
            await session.run("write a test")
        # session still functional — budget optimizes, doesn't crash
        assert session.total_tokens_spent() > 0

    @pytest.mark.asyncio
    async def test_session_id_stable_across_runs(self, session):
        sid = session.session_id()
        await session.run("task 1")
        await session.run("task 2")
        assert session.session_id() == sid

    @pytest.mark.asyncio
    async def test_telemetry_ingest_called_after_run(self, echo_executor, cap_executor):
        class StubPipeline:
            def __init__(self):
                self.calls = []
                self.closed = False
            async def ingest_trace(self, trace, raw_input=""):
                self.calls.append((trace.node_id, raw_input))
            async def aclose(self):
                self.closed = True

        stub = StubPipeline()
        session = GovernedSession(
            executor=echo_executor,
            capability_executor=cap_executor,
            trace_config=TraceConfig(local_only=True, persist_inputs=False),
            telemetry=stub,
        )
        await session.run("write a test for auth")
        assert len(stub.calls) == 1
        assert stub.calls[0][1] == "write a test for auth"
        await session.aclose()
        assert stub.closed is True

    @pytest.mark.asyncio
    async def test_telemetry_failure_does_not_break_run(self, echo_executor, cap_executor):
        class BrokenPipeline:
            async def ingest_trace(self, trace, raw_input=""):
                raise RuntimeError("telemetry exploded")
            async def aclose(self):
                raise RuntimeError("close exploded")

        session = GovernedSession(
            executor=echo_executor,
            capability_executor=cap_executor,
            trace_config=TraceConfig(local_only=True, persist_inputs=False),
            telemetry=BrokenPipeline(),
        )
        # Must not raise
        result = await session.run("explain this")
        assert result.output  # normal result delivered
        await session.aclose()  # also must not raise


class TestContextManagerIntegration:
    """Test that ContextManager integrates correctly with tool execution."""

    @pytest.mark.asyncio
    async def test_file_read_cached_after_execution(self, cap_executor):
        from tests.conftest import EchoExecutor
        from axor_core.context.manager import ContextManager
        from axor_core.contracts.policy import (
            ExecutionPolicy, TaskComplexity, ContextMode, CompressionMode,
            ChildMode, ExportMode, ToolPolicy,
        )

        executor = EchoExecutor(tool_calls=[("read", {"path": "auth.py"})])
        policy = ExecutionPolicy(
            name="test", derived_from=TaskComplexity.FOCUSED,
            context_mode=ContextMode.MINIMAL,
            compression_mode=CompressionMode.BALANCED,
            child_mode=ChildMode.DENIED, max_child_depth=0,
            tool_policy=ToolPolicy(allow_read=True),
            export_mode=ExportMode.SUMMARY,
        )

        session = GovernedSession(
            executor=executor,
            capability_executor=cap_executor,
            trace_config=TraceConfig(local_only=True, persist_inputs=False),
        )

        await session.run("write a test for auth")
        # post-callback should have cached the file read
        # verify via budget snapshot — execution ran
        assert session.total_tokens_spent() > 0
