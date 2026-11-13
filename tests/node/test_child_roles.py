"""Child roles and parallel spawn: a parent names a role, the operator decides
what the role runs on; a batch of children is gated child by child and run
together."""
from __future__ import annotations

import asyncio
import dataclasses
from typing import AsyncIterator

import pytest

from axor_core import ChildRole, GovernedSession, presets
from axor_core.capability.executor import CapabilityExecutor, ToolHandler
from axor_core.contracts.invokable import Invokable
from axor_core.contracts.policy import ToolPolicy
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind
from axor_core.contracts.trace import ChildSpawnedEvent, TraceConfig
from tests.conftest import EchoExecutor


class _Named(Invokable):
    """A child executor that says who it is, optionally after one tool call,
    and remembers the policy it was handed."""

    def __init__(self, name: str, tool: tuple[str, dict] | None = None,
                 delay: float = 0.0, log: list | None = None) -> None:
        self.name, self.tool, self.delay = name, tool, delay
        self.log = log if log is not None else []
        self.policies: list = []

    async def stream(self, envelope) -> AsyncIterator[ExecutorEvent]:
        self.policies.append(envelope.policy)
        self.log.append(("start", self.name))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.tool is not None:
            yield ExecutorEvent(kind=ExecutorEventKind.TOOL_USE,
                                payload={"tool": self.tool[0], "args": self.tool[1]},
                                node_id=envelope.node_id)
        yield ExecutorEvent(kind=ExecutorEventKind.TEXT,
                            payload={"text": f"I am {self.name}"},
                            node_id=envelope.node_id)
        self.log.append(("end", self.name))
        yield ExecutorEvent(kind=ExecutorEventKind.STOP, payload={"usage": {}},
                            node_id=envelope.node_id)


class _Write(ToolHandler):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def name(self) -> str:
        return "write"

    async def execute(self, args) -> str:
        self.calls += 1
        return "written"


class _Recording(EchoExecutor):
    """Parent that records the tool results it is handed back."""

    def __init__(self, tool_calls) -> None:
        super().__init__(tool_calls=tool_calls)
        self.results: list = []


def _federated(**over):
    return dataclasses.replace(presets.get("federated"), **over)


async def _run(parent_calls, roles, *, policy=None, default_child=None,
               max_parallel=None, cap=None):
    parent = EchoExecutor(tool_calls=parent_calls)
    sess = GovernedSession(
        executor=parent,
        capability_executor=cap or CapabilityExecutor(),
        child_executor=default_child or _Named("default"),
        trace_config=TraceConfig(local_only=True, persist_inputs=False),
        child_roles=roles,
        max_parallel_children=max_parallel,
    )
    result = await sess.run("orchestrate", policy=policy or _federated(allow_model_switch=True))
    return sess, result


def _spawned(sess) -> list[ChildSpawnedEvent]:
    return [e for t in sess.all_traces() for e in t.events
            if isinstance(e, ChildSpawnedEvent)]


def _denials(sess) -> list[str]:
    return [str(getattr(e, "reason", "")) for t in sess.all_traces() for e in t.events
            if type(e).__name__ == "IntentDeniedEvent"]


@pytest.mark.asyncio
async def test_a_role_runs_its_child_on_the_role_s_executor_and_says_so():
    codex = _Named("codex")
    sess, _ = await _run(
        [("spawn_child", {"task": "read the auth module", "role": "reader"})],
        [ChildRole("reader", executor=codex, label="codex/luna")])
    assert len(codex.policies) == 1
    [ev] = _spawned(sess)
    assert ev.role == "reader" and ev.executor_label == "codex/luna"


@pytest.mark.asyncio
async def test_no_role_keeps_the_default_child_executor():
    default = _Named("default")
    sess, _ = await _run([("spawn_child", {"task": "x"})], [], default_child=default)
    assert len(default.policies) == 1
    [ev] = _spawned(sess)
    assert ev.role == "" and ev.executor_label == ""


@pytest.mark.asyncio
async def test_an_unknown_role_is_denied_not_guessed():
    default = _Named("default")
    sess, _ = await _run([("spawn_child", {"task": "x", "role": "admin"})],
                         [ChildRole("reader", executor=_Named("codex"))],
                         default_child=default)
    assert default.policies == [] and _spawned(sess) == []
    assert any("unknown child role 'admin'" in r for r in _denials(sess))


@pytest.mark.asyncio
async def test_switching_models_needs_the_authority_to():
    codex = _Named("codex")
    sess, _ = await _run([("spawn_child", {"task": "x", "role": "reader"})],
                         [ChildRole("reader", executor=codex, label="codex/luna")],
                         policy=_federated(allow_model_switch=False))
    assert codex.policies == []
    assert any("does not allow a model switch" in r for r in _denials(sess))


@pytest.mark.asyncio
async def test_a_role_on_the_parent_s_own_model_needs_no_switch():
    default = _Named("default")
    no_write = _federated(tool_policy=ToolPolicy(allow_read=True, allow_write=False,
                                                  allow_bash=False, allow_spawn=False))
    sess, _ = await _run([("spawn_child", {"task": "x", "role": "reader"})],
                         [ChildRole("reader", policy=no_write)],
                         policy=_federated(allow_model_switch=False),
                         default_child=default)
    assert len(default.policies) == 1
    assert default.policies[0].tool_policy.allow_write is False


@pytest.mark.asyncio
async def test_a_reader_role_cannot_write_even_when_its_model_tries():
    write = _Write()
    cap = CapabilityExecutor()
    cap.register(write)
    reader = _Named("codex", tool=("write", {"path": "a.py", "content": "x"}))
    read_only = _federated(tool_policy=ToolPolicy(allow_read=True, allow_write=False,
                                                   allow_bash=False, allow_spawn=False))
    await _run([("spawn_child", {"task": "look around", "role": "reader"})],
               [ChildRole("reader", executor=reader, label="codex/luna", policy=read_only)],
               cap=cap)
    assert write.calls == 0


@pytest.mark.asyncio
async def test_a_role_cannot_lift_the_parent_s_ceiling():
    write = _Write()
    cap = CapabilityExecutor()
    cap.register(write)
    writer = _Named("opus", tool=("write", {"path": "a.py", "content": "x"}))
    parent_read_only = _federated(
        allow_model_switch=True,
        tool_policy=ToolPolicy(allow_read=True, allow_write=False, allow_bash=False,
                               allow_spawn=True))
    await _run([("spawn_child", {"task": "implement", "role": "implementer"})],
               [ChildRole("implementer", executor=writer, policy=_federated())],
               policy=parent_read_only, cap=cap)
    assert write.calls == 0


@pytest.mark.asyncio
async def test_a_batch_runs_its_children_together_and_reports_each():
    log: list = []
    roles = [ChildRole("reader", executor=_Named("codex", delay=0.05, log=log),
                       label="codex/luna")]
    children = [{"task": f"read part {i}", "role": "reader"} for i in range(5)]
    parent = _Recording([("spawn_child", {"children": children})])
    sess = GovernedSession(
        executor=parent, capability_executor=CapabilityExecutor(),
        child_executor=_Named("default"),
        trace_config=TraceConfig(local_only=True, persist_inputs=False),
        child_roles=roles)
    await sess.run("fan out", policy=_federated(allow_model_switch=True))
    starts = [i for i, (what, _) in enumerate(log) if what == "start"]
    first_end = next(i for i, (what, _) in enumerate(log) if what == "end")
    # all five started before any finished: they ran concurrently
    assert len(starts) == 5 and max(starts) < first_end
    assert len(_spawned(sess)) == 5


@pytest.mark.asyncio
async def test_one_bad_child_does_not_sink_its_siblings():
    codex = _Named("codex")
    children = [{"task": "a", "role": "reader"}, {"task": "b", "role": "ghost"},
                {"task": "c", "role": "reader"}]
    sess, _ = await _run([("spawn_child", {"children": children})],
                         [ChildRole("reader", executor=codex, label="codex/luna")])
    assert len(codex.policies) == 2
    assert any("unknown child role 'ghost'" in r for r in _denials(sess))


@pytest.mark.asyncio
async def test_a_batch_wider_than_allowed_is_refused_whole():
    codex = _Named("codex")
    children = [{"task": str(i), "role": "reader"} for i in range(4)]
    sess, _ = await _run([("spawn_child", {"children": children})],
                         [ChildRole("reader", executor=codex)], max_parallel=3)
    assert codex.policies == []
    assert any("at most 3 run in parallel" in r for r in _denials(sess))


@pytest.mark.asyncio
async def test_a_batch_counts_against_the_total_spawn_cap():
    from axor_core.node.intent_loop import IntentLoop

    codex = _Named("codex")
    children = [{"task": str(i), "role": "reader"} for i in range(3)]
    parent = EchoExecutor(tool_calls=[("spawn_child", {"children": children})])
    sess = GovernedSession(
        executor=parent, capability_executor=CapabilityExecutor(),
        child_executor=_Named("default"),
        trace_config=TraceConfig(local_only=True, persist_inputs=False),
        child_roles=[ChildRole("reader", executor=codex)])
    original = IntentLoop.__init__

    def capped(self, *a, **kw):
        kw["max_total_spawns"] = 2
        original(self, *a, **kw)

    IntentLoop.__init__ = capped
    try:
        await sess.run("fan out", policy=_federated(allow_model_switch=True))
    finally:
        IntentLoop.__init__ = original
    assert len(codex.policies) == 2
    assert any("spawn limit" in r for r in _denials(sess))


def test_roles_are_plain_names_and_unique():
    from axor_core.contracts.roles import role_roster

    with pytest.raises(ValueError):
        ChildRole("not a name")
    with pytest.raises(ValueError):
        role_roster([ChildRole("a"), ChildRole("a")])
    assert set(role_roster([ChildRole("a"), ChildRole("b")])) == {"a", "b"}
