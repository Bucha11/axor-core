"""
Ready-made escalation approval callbacks (axor_core.capability.approvals).

The callback only answers the approval question; grant bounds (TTL,
max-use, flood guard) stay in EscalationManager. Everything here must
fail closed: unknown tool, over-cap ops, out-of-prefix path, unconfined
request against a confined approver, and a console prompt with no TTY
all deny.
"""
from __future__ import annotations

import dataclasses
import sys

import pytest

from axor_core.capability.approvals import (
    AllowlistEscalationApprover,
    console_escalation_callback,
)
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind
from axor_core.contracts.policy import EscalationPolicy
from axor_core.node.escalation import EscalationManager


# ── AllowlistEscalationApprover ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_allowlist_approves_within_caps():
    approver = AllowlistEscalationApprover({"write": 10, "bash": 5})
    assert await approver("t1", "write", [], 10) is True
    assert await approver("t1", "bash", [], 1) is True


@pytest.mark.asyncio
async def test_allowlist_denies_unknown_tool():
    approver = AllowlistEscalationApprover({"write": 10})
    assert await approver("t1", "bash", [], 1) is False


@pytest.mark.asyncio
async def test_allowlist_denies_over_cap_ops():
    approver = AllowlistEscalationApprover({"write": 5})
    assert await approver("t1", "write", [], 6) is False


@pytest.mark.asyncio
async def test_allowlist_path_confinement():
    approver = AllowlistEscalationApprover(
        {"write": 10}, allowed_path_prefixes=("/workspace",)
    )
    assert await approver("t1", "write", ["/workspace/src/a.py"], 1) is True
    assert await approver("t1", "write", ["/etc/passwd"], 1) is False
    # one bad path poisons the whole request
    assert await approver("t1", "write", ["/workspace/a", "/etc/x"], 1) is False


@pytest.mark.asyncio
async def test_allowlist_confined_approver_denies_unconfined_request():
    approver = AllowlistEscalationApprover(
        {"write": 10}, allowed_path_prefixes=("/workspace",)
    )
    # no paths = unrestricted grant, broader than the confinement allows
    assert await approver("t1", "write", [], 1) is False


@pytest.mark.asyncio
async def test_allowlist_rejects_dotdot_escape():
    approver = AllowlistEscalationApprover(
        {"write": 10}, allowed_path_prefixes=("/workspace",)
    )
    assert await approver("t1", "write", ["/workspace/../etc/passwd"], 1) is False


def test_allowlist_constructor_rejects_empty_and_nonpositive():
    with pytest.raises(ValueError):
        AllowlistEscalationApprover({})
    with pytest.raises(ValueError):
        AllowlistEscalationApprover({"write": 0})


# ── console_escalation_callback ─────────────────────────────────────────────────

class _FakeStdin:
    def __init__(self, tty: bool) -> None:
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


@pytest.mark.asyncio
async def test_console_denies_without_tty(monkeypatch):
    monkeypatch.setattr(sys, "stdin", _FakeStdin(tty=False))
    assert await console_escalation_callback("t1", "write", [], 5) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("answer,expected", [
    ("y", True), ("YES", True), ("n", False), ("", False), ("later", False),
])
async def test_console_parses_answer(monkeypatch, answer, expected):
    monkeypatch.setattr(sys, "stdin", _FakeStdin(tty=True))
    monkeypatch.setattr("builtins.input", lambda _prompt: answer)
    assert await console_escalation_callback("t1", "write", [], 5) is expected


# ── end-to-end: approver drives a require_human grant ───────────────────────────

def _escalation_event(tool: str, max_ops: int = 3) -> ExecutorEvent:
    return ExecutorEvent(
        kind=ExecutorEventKind.TOOL_USE,
        payload={
            "tool": "escalate_policy",
            "tool_use_id": "esc-1",
            "args": {"tool": tool, "reason": "need write access", "paths": [],
                     "max_ops": max_ops},
        },
        node_id="node-1",
    )


@pytest.mark.asyncio
async def test_allowlist_approver_grants_require_human_escalation(make_envelope):
    env = make_envelope()
    env = dataclasses.replace(
        env,
        policy=dataclasses.replace(
            env.policy,
            escalation_policy=EscalationPolicy(
                allow_escalation=True,
                grantable_tools=("write",),
                require_human=True,
            ),
        ),
    )
    mgr = EscalationManager(
        escalation_callback=AllowlistEscalationApprover({"write": 10})
    )
    result = await mgr.grant_from_intent(_escalation_event("write"), env, [])
    assert result.get("granted") is True
    assert mgr.covers("write") is True


@pytest.mark.asyncio
async def test_allowlist_approver_denies_out_of_list_escalation(make_envelope):
    env = make_envelope()
    env = dataclasses.replace(
        env,
        policy=dataclasses.replace(
            env.policy,
            escalation_policy=EscalationPolicy(
                allow_escalation=True,
                grantable_tools=("write", "bash"),
                require_human=True,
            ),
        ),
    )
    mgr = EscalationManager(
        escalation_callback=AllowlistEscalationApprover({"write": 10})
    )
    result = await mgr.grant_from_intent(_escalation_event("bash"), env, [])
    assert result.get("granted") is not True
    assert mgr.covers("bash") is False
