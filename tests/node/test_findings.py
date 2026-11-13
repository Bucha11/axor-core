"""Findings: a reader's answer is checked against the files it actually read."""
from __future__ import annotations

import dataclasses
import json
from typing import AsyncIterator

import pytest

from axor_core import ChildRole, GovernedSession, presets
from axor_core.capability.executor import CapabilityExecutor, ToolHandler
from axor_core.contracts.invokable import Invokable
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind
from axor_core.contracts.trace import TraceConfig, TraceEventKind
from axor_core.node.findings import parse_findings, verify_findings
from tests.conftest import EchoExecutor

ANSWER = {"findings": [
    {"claim": "tokens are compared with ==", "file": "src/auth.py", "line": 42},
    {"claim": "billing retries forever", "file": "src/billing.py", "line": 7},
    {"claim": "the code is generally fine"},
]}


def test_a_finding_is_verified_only_by_a_file_the_child_read() -> None:
    got = verify_findings(json.dumps(ANSWER), [("read", {"path": "./src/auth.py"})])
    marks = [(f.get("file"), f["verified"]) for f in got["findings"]]
    assert marks == [("src/auth.py", True), ("src/billing.py", False), (None, False)]
    assert got["verified"] == 1 and got["unverified"] == 2
    assert "never read src/billing.py" in got["findings"][1]["why"]
    assert "names no file" in got["findings"][2]["why"]


def test_a_search_over_a_directory_backs_files_inside_it() -> None:
    got = verify_findings(json.dumps(ANSWER), [("grep", {"pattern": "==", "path": "src"})])
    assert [f["verified"] for f in got["findings"]] == [True, True, False]
    # a search with no path searched the working directory
    got = verify_findings(json.dumps(ANSWER), [("search", {"query": "retry"})])
    assert got["findings"][1]["verified"] is True


def test_a_search_elsewhere_backs_nothing() -> None:
    got = verify_findings(json.dumps(ANSWER), [("grep", {"pattern": "x", "path": "docs"})])
    assert got["verified"] == 0


def test_findings_are_found_in_fences_and_prose() -> None:
    fenced = "Here is what I found:\n```json\n" + json.dumps(ANSWER) + "\n```\nDone."
    assert len(parse_findings(fenced)) == 3
    assert parse_findings("no structure at all") is None


def test_an_unstructured_answer_vouches_for_nothing() -> None:
    got = verify_findings("auth looks fine to me", [("read", {"path": "src/auth.py"})])
    assert got["findings"] == [] and got["unstructured"] == "auth looks fine to me"


# ── through a real spawn ──────────────────────────────────────────────────────

class _Files(ToolHandler):
    @property
    def name(self) -> str:
        return "read"

    async def execute(self, args) -> str:
        if args.get("path") != "src/auth.py":
            raise FileNotFoundError(args.get("path"))
        return "def check(a, b): return a == b"


class _Reader(Invokable):
    """Reads auth.py, tries billing.py (which does not exist), then reports
    findings about both."""

    async def stream(self, envelope) -> AsyncIterator[ExecutorEvent]:
        for path in ("src/auth.py", "src/billing.py"):
            yield ExecutorEvent(kind=ExecutorEventKind.TOOL_USE,
                                payload={"tool": "read", "args": {"path": path}},
                                node_id=envelope.node_id)
        yield ExecutorEvent(kind=ExecutorEventKind.TEXT,
                            payload={"text": json.dumps(ANSWER)},
                            node_id=envelope.node_id)
        yield ExecutorEvent(kind=ExecutorEventKind.STOP, payload={"usage": {}},
                            node_id=envelope.node_id)


@pytest.mark.asyncio
async def test_a_reader_role_hands_its_parent_checked_findings() -> None:
    cap = CapabilityExecutor()
    cap.register(_Files())
    sess = GovernedSession(
        executor=EchoExecutor(tool_calls=[("spawn_child", {"task": "audit auth",
                                                            "role": "reader"})]),
        capability_executor=cap,
        trace_config=TraceConfig(local_only=True, persist_inputs=False),
        child_roles=[ChildRole("reader", executor=_Reader(), label="codex/luna",
                               output="findings")],
    )
    await sess.run("audit", policy=dataclasses.replace(presets.get("federated"),
                                                       allow_model_switch=True))
    done = [e for t in sess.all_traces() for e in t.events
            if e.kind == TraceEventKind.CHILD_COMPLETED]
    assert done and done[0].payload["findings_verified"] == 1
    # billing.py: the read FAILED, so it was never read
    assert done[0].payload["findings_unverified"] == 2


def test_a_role_output_is_text_or_findings() -> None:
    with pytest.raises(ValueError):
        ChildRole("reader", output="yaml")


def test_reads_through_the_shell_count() -> None:
    calls = [("bash", {"command": "sed -n 30,60p src/auth.py | nl"}),
             ("bash", {"command": "rg -n retry lib/ && ls docs"})]
    answer = {"findings": [
        {"claim": "a", "file": "src/auth.py"},
        {"claim": "b", "file": "lib/billing/retry.py"},
        {"claim": "c", "file": "docs/x.md"},
        {"claim": "d", "file": "src/other.py"},
    ]}
    got = verify_findings(json.dumps(answer), calls)
    assert [f["verified"] for f in got["findings"]] == [True, True, True, False]


def test_a_shell_command_that_reads_nothing_backs_nothing() -> None:
    got = verify_findings(json.dumps({"findings": [{"claim": "a", "file": "src/a.py"}]}),
                          [("bash", {"command": "echo src/a.py > /dev/null"})])
    assert got["verified"] == 0


class _OutOfLoop(Invokable):
    """A CLI agent: its tool calls ran in its own process, governed by hooks,
    so it reports them instead of streaming them."""

    def __init__(self, sources=()) -> None:
        self.sources = sources
        self.asked: list[str] = []

    async def stream(self, envelope) -> AsyncIterator[ExecutorEvent]:
        yield ExecutorEvent(kind=ExecutorEventKind.TEXT,
                            payload={"text": json.dumps(ANSWER)},
                            node_id=envelope.node_id)
        yield ExecutorEvent(kind=ExecutorEventKind.STOP, payload={"usage": {}},
                            node_id=envelope.node_id)

    def executed_calls(self, node_id):
        self.asked.append(node_id)
        return [("bash", {"command": "sed -n 1,80p src/auth.py"})]

    def output_provenance(self, node_id):
        from axor_core.contracts.taint import TaintSource
        from axor_core.taint.causal_root import CausalRoot

        return CausalRoot(sources=frozenset(self.sources)) if self.sources else None


async def _run_out_of_loop(executor):
    from axor_core.federation import FederationGateway

    sess = GovernedSession(
        executor=EchoExecutor(tool_calls=[("spawn_child", {"task": "audit auth",
                                                            "role": "reader"})]),
        capability_executor=CapabilityExecutor(),
        trace_config=TraceConfig(local_only=True, persist_inputs=False),
        child_roles=[ChildRole("reader", executor=executor, output="findings")],
        federation_gateway=FederationGateway(peers={}, compatible_kernels=set(),
                                             federated_domains=set()),
    )
    await sess.run("audit", policy=dataclasses.replace(presets.get("federated"),
                                                       allow_model_switch=True))
    return sess


@pytest.mark.asyncio
async def test_an_out_of_loop_executor_reports_what_it_ran() -> None:
    executor = _OutOfLoop()
    sess = await _run_out_of_loop(executor)
    done = [e for t in sess.all_traces() for e in t.events
            if e.kind == TraceEventKind.CHILD_COMPLETED]
    assert executor.asked and done[0].payload["findings_verified"] == 1
    # it read nothing untrusted: its output arrives clean
    assert not sess._taint_engine.derive_value(json.dumps(ANSWER)).is_tainted


@pytest.mark.asyncio
async def test_what_it_read_taints_what_it_returns() -> None:
    from axor_core.contracts.taint import TaintSource

    sess = await _run_out_of_loop(_OutOfLoop(sources=[TaintSource.WEB]))
    assert sess._taint_engine.derive_value(json.dumps(ANSWER)).is_tainted


@pytest.mark.asyncio
async def test_a_host_can_read_what_the_ledger_knows_of_a_value() -> None:
    from axor_core.contracts.taint import TaintSource

    sess = await _run_out_of_loop(_OutOfLoop(sources=[TaintSource.WEB]))
    root = sess.provenance_of(json.dumps(ANSWER))
    assert root.is_tainted and TaintSource.WEB in root.sources
    assert not sess.provenance_of("something nobody returned").is_tainted
