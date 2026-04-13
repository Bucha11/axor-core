"""Tests for capability/resolver and capability/executor."""
from __future__ import annotations

import pytest
from axor_core.capability import CapabilityResolver, CapabilityExecutor, ToolHandler
from axor_core.contracts.intent import Intent, IntentKind
from axor_core.contracts.policy import (
    ExecutionPolicy, TaskComplexity, ContextMode, CompressionMode,
    ChildMode, ExportMode, ToolPolicy,
)
from axor_core.errors import ToolNotAllowedError, ToolNotFoundError


def make_policy(**kwargs) -> ExecutionPolicy:
    defaults = dict(
        name="test", derived_from=TaskComplexity.FOCUSED,
        context_mode=ContextMode.MINIMAL,
        compression_mode=CompressionMode.BALANCED,
        child_mode=ChildMode.DENIED, max_child_depth=0,
        tool_policy=ToolPolicy(allow_read=True),
        export_mode=ExportMode.SUMMARY,
    )
    defaults.update(kwargs)
    return ExecutionPolicy(**defaults)


class TestCapabilityResolver:

    @pytest.fixture
    def resolver(self):
        return CapabilityResolver()

    def test_read_allowed(self, resolver):
        policy = make_policy(tool_policy=ToolPolicy(allow_read=True))
        caps = resolver.resolve(policy)
        assert "read" in caps.allowed_tools

    def test_write_denied_by_default(self, resolver):
        policy = make_policy(tool_policy=ToolPolicy(allow_read=True, allow_write=False))
        caps = resolver.resolve(policy)
        assert "write" not in caps.allowed_tools

    def test_spawn_requires_both_allow_spawn_and_child_mode(self, resolver):
        # allow_spawn=True but child_mode=DENIED → no spawn
        policy = make_policy(
            tool_policy=ToolPolicy(allow_spawn=True),
            child_mode=ChildMode.DENIED,
        )
        caps = resolver.resolve(policy)
        assert "spawn_child" not in caps.allowed_tools
        assert caps.allow_children is False

    def test_spawn_allowed_when_both_set(self, resolver):
        policy = make_policy(
            tool_policy=ToolPolicy(allow_spawn=True),
            child_mode=ChildMode.ALLOWED,
            max_child_depth=2,
        )
        caps = resolver.resolve(policy)
        assert "spawn_child" in caps.allowed_tools
        assert caps.allow_children is True

    def test_extra_denied_wins_over_allow(self, resolver):
        policy = make_policy(
            tool_policy=ToolPolicy(
                allow_bash=True,
                extra_denied=("bash",),
            )
        )
        caps = resolver.resolve(policy)
        assert "bash" not in caps.allowed_tools

    def test_context_expansion_blocked_in_minimal_mode(self, resolver):
        policy = make_policy(context_mode=ContextMode.MINIMAL)
        caps = resolver.resolve(policy)
        assert caps.allow_context_expansion is False

    def test_context_expansion_allowed_in_moderate_mode(self, resolver):
        policy = make_policy(context_mode=ContextMode.MODERATE)
        caps = resolver.resolve(policy)
        assert caps.allow_context_expansion is True


class TestCapabilityExecutor:

    @pytest.fixture
    def executor_with_read(self):
        ex = CapabilityExecutor()
        ex.register(MockRead())
        return ex

    @pytest.mark.asyncio
    async def test_executes_allowed_tool(self, executor_with_read, make_envelope):
        envelope = make_envelope()
        intent = Intent(
            kind=IntentKind.TOOL_CALL,
            payload={"tool": "read", "args": {"path": "test.py"}},
            node_id="n1",
        )
        result = await executor_with_read.execute(intent, envelope.capabilities)
        assert result == "mock file content"

    @pytest.mark.asyncio
    async def test_raises_not_allowed_for_unlisted_tool(self, executor_with_read, make_envelope):
        envelope = make_envelope()
        intent = Intent(
            kind=IntentKind.TOOL_CALL,
            payload={"tool": "bash", "args": {}},
            node_id="n1",
        )
        with pytest.raises(ToolNotAllowedError) as exc_info:
            await executor_with_read.execute(intent, envelope.capabilities)
        assert exc_info.value.tool == "bash"

    @pytest.mark.asyncio
    async def test_raises_not_found_for_unregistered_tool(self, make_envelope):
        from axor_core.contracts.envelope import Capabilities
        ex = CapabilityExecutor()
        # capabilities say write is allowed but no handler registered
        caps = make_envelope().capabilities
        caps_with_write = Capabilities(
            allowed_tools=frozenset(["write"]),
            allow_children=False, allow_nested_children=False,
            allow_context_expansion=False, allow_export=True,
            allow_mutation=True, max_child_depth=0,
        )
        intent = Intent(
            kind=IntentKind.TOOL_CALL,
            payload={"tool": "write", "args": {}},
            node_id="n1",
        )
        with pytest.raises(ToolNotFoundError):
            await ex.execute(intent, caps_with_write)

    @pytest.mark.asyncio
    async def test_post_callback_fires(self, executor_with_read, make_envelope):
        observed = []

        async def callback(tool, args, result):
            observed.append((tool, args, result))

        executor_with_read.register_post_callback(callback)
        envelope = make_envelope()
        intent = Intent(
            kind=IntentKind.TOOL_CALL,
            payload={"tool": "read", "args": {"path": "x.py"}},
            node_id="n1",
        )
        await executor_with_read.execute(intent, envelope.capabilities)
        assert len(observed) == 1
        assert observed[0][0] == "read"
        assert observed[0][1] == {"path": "x.py"}

    @pytest.mark.asyncio
    async def test_callback_failure_does_not_break_execution(self, make_envelope):
        ex = CapabilityExecutor()
        ex.register(MockRead())

        async def bad_callback(tool, args, result):
            raise RuntimeError("callback failed")

        ex.register_post_callback(bad_callback)
        envelope = make_envelope()
        intent = Intent(
            kind=IntentKind.TOOL_CALL,
            payload={"tool": "read", "args": {}},
            node_id="n1",
        )
        # should not raise despite bad callback
        result = await ex.execute(intent, envelope.capabilities)
        assert result == "mock file content"


class MockRead(ToolHandler):
    @property
    def name(self): return "read"
    async def execute(self, args): return "mock file content"
