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
async def test_allowlist_rejects_dotdot_escape(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    approver = AllowlistEscalationApprover(
        {"write": 10}, allowed_path_prefixes=(str(workspace),)
    )
    escape = str(workspace / ".." / "etc" / "passwd")
    assert await approver("t1", "write", [escape], 1) is False


@pytest.mark.asyncio
async def test_allowlist_rejects_symlink_escape(tmp_path):
    """A symlink inside the workspace must not mint an allowed root outside it.
    Approval uses the same canonical (symlink-resolving) containment as lease
    enforcement, so /workspace/link -> /etc is 'outside', not 'inside'."""
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    link = workspace / "link"
    link.symlink_to(outside)

    approver = AllowlistEscalationApprover(
        {"write": 10}, allowed_path_prefixes=(str(workspace),)
    )
    assert await approver("t1", "write", [str(link)], 1) is False
    assert await approver("t1", "write", [str(link / "passwd")], 1) is False


@pytest.mark.asyncio
async def test_allowlist_accepts_symlink_resolving_inside(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    link = workspace / "alias"
    link.symlink_to(workspace / "src")

    approver = AllowlistEscalationApprover(
        {"write": 10}, allowed_path_prefixes=(str(workspace),)
    )
    assert await approver("t1", "write", [str(link / "a.py")], 1) is True


# ── unconfined tools (bash-style: no extractable path in calls) ─────────────────

@pytest.mark.asyncio
async def test_unconfined_tool_granted_only_without_paths(tmp_path):
    approver = AllowlistEscalationApprover(
        {"write": 10, "bash": 5},
        allowed_path_prefixes=(str(tmp_path),),
        unconfined_tools=("bash",),
    )
    # bash: approvable path-free even though the approver is path-confined
    assert await approver("t1", "bash", [], 3) is True
    # a paths-carrying bash request would produce an unusable lease — denied
    assert await approver("t1", "bash", [str(tmp_path)], 3) is False
    # write still requires confinement
    assert await approver("t1", "write", [], 1) is False


def test_allowlist_constructor_rejects_empty_and_nonpositive():
    with pytest.raises(ValueError):
        AllowlistEscalationApprover({})
    with pytest.raises(ValueError):
        AllowlistEscalationApprover({"write": 0})
    with pytest.raises(ValueError):
        AllowlistEscalationApprover({"write": 5}, unconfined_tools=("bash",))


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


@pytest.mark.asyncio
async def test_console_denies_on_terminal_error(monkeypatch):
    """EOF mid-prompt (closed stream, ^D) is a denial, not an exception."""
    monkeypatch.setattr(sys, "stdin", _FakeStdin(tty=True))

    def _eof(_prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", _eof)
    assert await console_escalation_callback("t1", "write", [], 5) is False


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
async def test_crashing_callback_is_a_denial_with_trace(make_envelope):
    """A raising approver must become ESCALATION_DENIED with a trace event —
    never an exception escaping grant_from_intent."""
    from axor_core.contracts.trace import TraceEventKind

    async def _boom(tool_use_id, tool, paths, max_ops) -> bool:
        raise RuntimeError("operator callback exploded")

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
    mgr = EscalationManager(escalation_callback=_boom)
    trace: list = []
    result = await mgr.grant_from_intent(_escalation_event("write"), env, trace)

    assert result.get("granted") is not True
    assert "callback failed" in result.get("reason", "")
    assert mgr.covers("write") is False
    denied = [e for e in trace if e.kind == TraceEventKind.ESCALATION_DENIED]
    assert len(denied) == 1


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
