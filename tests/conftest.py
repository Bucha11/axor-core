"""
Shared fixtures for axor_core tests.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

import pytest
from axor_core.capability.executor import CapabilityExecutor, ToolHandler
from axor_core.contracts.cancel import make_token
from axor_core.contracts.context import (
    ContextFragment,
    ContextView,
    LineageSummary,
    RawExecutionState,
)
from axor_core.contracts.envelope import (
    Capabilities,
    ExecutionEnvelope,
    ExportContract,
)
from axor_core.contracts.invokable import Invokable
from axor_core.contracts.policy import (
    ChildMode,
    CompressionMode,
    ContextMode,
    ExecutionPolicy,
    ExportMode,
    TaskComplexity,
    ToolPolicy,
)
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind

# ── Mock executor ──────────────────────────────────────────────────────────────


class EchoExecutor(Invokable):
    """Executor that echoes policy name and emits configurable tool calls."""

    def __init__(
        self, tool_calls: list[tuple[str, dict]] | None = None
    ) -> None:
        self.tool_calls = tool_calls or []
        self.last_envelope: ExecutionEnvelope | None = None

    async def stream(
        self, envelope: ExecutionEnvelope
    ) -> AsyncIterator[ExecutorEvent]:
        self.last_envelope = envelope
        for tool_name, args in self.tool_calls:
            yield ExecutorEvent(
                kind=ExecutorEventKind.TOOL_USE,
                payload={"tool": tool_name, "args": args},
                node_id=envelope.node_id,
            )
        yield ExecutorEvent(
            kind=ExecutorEventKind.TEXT,
            payload={"text": f"policy={envelope.policy.name}"},
            node_id=envelope.node_id,
        )
        yield ExecutorEvent(
            kind=ExecutorEventKind.STOP,
            payload={
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "tool_tokens": 20,
                }
            },
            node_id=envelope.node_id,
        )


class SlowExecutor(Invokable):
    """Executor that emits events with asyncio delays — for cancel testing."""

    def __init__(self, event_count: int = 5, delay: float = 0.05) -> None:
        self.event_count = event_count
        self.delay = delay

    async def stream(
        self, envelope: ExecutionEnvelope
    ) -> AsyncIterator[ExecutorEvent]:
        for i in range(self.event_count):
            await asyncio.sleep(self.delay)
            yield ExecutorEvent(
                kind=ExecutorEventKind.TEXT,
                payload={"text": f"chunk_{i}"},
                node_id=envelope.node_id,
            )
        yield ExecutorEvent(
            kind=ExecutorEventKind.STOP,
            payload={
                "usage": {
                    "input_tokens": 50,
                    "output_tokens": 25,
                    "tool_tokens": 0,
                }
            },
            node_id=envelope.node_id,
        )


# ── Mock tool handlers ─────────────────────────────────────────────────────────


class MockReadHandler(ToolHandler):
    def __init__(self, content: str = "file content") -> None:
        self.content = content
        self.calls: list[dict] = []

    @property
    def name(self) -> str:
        return "read"

    async def execute(self, args: dict[str, Any]) -> Any:
        self.calls.append(args)
        return self.content


class MockBashHandler(ToolHandler):
    def __init__(self, output: str = "ok") -> None:
        self.output = output

    @property
    def name(self) -> str:
        return "bash"

    async def execute(self, args: dict[str, Any]) -> Any:
        return self.output


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _axor_trace_dir(tmp_path, monkeypatch):
    """Redirect trace JSONL output into pytest tmp_path so tests never touch ~/.axor."""
    monkeypatch.setenv("AXOR_TRACE_DIR", str(tmp_path / "traces"))


@pytest.fixture
def read_handler():
    return MockReadHandler()


@pytest.fixture
def cap_executor(read_handler):
    ex = CapabilityExecutor()
    ex.register(read_handler)
    return ex


@pytest.fixture
def echo_executor():
    return EchoExecutor()


@pytest.fixture
def focused_policy():
    return ExecutionPolicy(
        name="focused_generative",
        derived_from=TaskComplexity.FOCUSED,
        context_mode=ContextMode.MINIMAL,
        compression_mode=CompressionMode.BALANCED,
        child_mode=ChildMode.DENIED,
        max_child_depth=0,
        tool_policy=ToolPolicy(allow_read=True, allow_write=True),
        export_mode=ExportMode.SUMMARY,
    )


@pytest.fixture
def expansive_policy():
    return ExecutionPolicy(
        name="expansive",
        derived_from=TaskComplexity.EXPANSIVE,
        context_mode=ContextMode.BROAD,
        compression_mode=CompressionMode.LIGHT,
        child_mode=ChildMode.ALLOWED,
        max_child_depth=3,
        tool_policy=ToolPolicy(
            allow_read=True,
            allow_write=True,
            allow_bash=True,
            allow_spawn=True,
        ),
        export_mode=ExportMode.FULL,
        child_context_fraction=0.6,
    )


@pytest.fixture
def root_lineage():
    return LineageSummary(
        node_id="node_test_root",
        parent_id=None,
        depth=0,
        ancestry_ids=[],
        inherited_restrictions=[],
    )


@pytest.fixture
def raw_state():
    return RawExecutionState(
        task="write a test for the payment endpoint",
        session_id="sess_test",
        parent_export=None,
        session_state={},
        memory_fragments=[],
        lineage=None,
    )


@pytest.fixture
def make_envelope(focused_policy, root_lineage):
    """Factory for test envelopes."""

    def _make(policy=None, cancel_token=None):
        p = policy or focused_policy
        ctx = ContextView(
            node_id=root_lineage.node_id,
            working_summary="test task",
            visible_fragments=[
                ContextFragment(
                    kind="fact",
                    content="test",
                    token_estimate=10,
                    source="test",
                )
            ],
            active_constraints=[],
            lineage=root_lineage,
            token_count=10,
            compression_ratio=1.0,
        )
        caps = Capabilities(
            allowed_tools=frozenset(["read", "write"]),
            allow_children=False,
            allow_nested_children=False,
            allow_context_expansion=False,
            allow_export=True,
            allow_mutation=True,
            max_child_depth=0,
        )
        export = ExportContract(
            mode=p.export_mode,
            allowed_fields=frozenset(["output"]),
            max_export_tokens=1024,
        )
        return ExecutionEnvelope(
            node_id=root_lineage.node_id,
            task="test task",
            context=ctx,
            policy=p,
            capabilities=caps,
            export_contract=export,
            lineage=root_lineage,
            cancel_token=cancel_token or make_token(),
        )

    return _make
