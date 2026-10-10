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
from axor_core.contracts.approval import ApprovalRequest, AuthorityScope
from axor_core.node.escalation import EscalationManager, adapt_escalation_callback
from axor_core.policy.effect import effect_digest


def _class_req(tool: str, paths: list[str], max_ops: int) -> ApprovalRequest:
    """A CLASS-scoped question, the shape every pre-existing case asks."""
    return ApprovalRequest(
        tool_use_id="t1", tool=tool, scope=AuthorityScope.CLASS,
        paths=tuple(paths), max_ops=max_ops,
    )


def _effect_req(tool: str, args: dict, **kw) -> ApprovalRequest:
    """An EFFECT-scoped question about one concrete call."""
    return ApprovalRequest(
        tool_use_id="t1", tool=tool, scope=AuthorityScope.EFFECT, max_ops=1,
        args=args, effect_digest=effect_digest(tool, args), **kw,
    )



# ── AllowlistEscalationApprover ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_allowlist_approves_within_caps():
    approver = AllowlistEscalationApprover({"write": 10, "bash": 5})
    assert await approver(_class_req("write", [], 10)) is True
    assert await approver(_class_req("bash", [], 1)) is True


@pytest.mark.asyncio
async def test_allowlist_denies_unknown_tool():
    approver = AllowlistEscalationApprover({"write": 10})
    assert await approver(_class_req("bash", [], 1)) is False


@pytest.mark.asyncio
async def test_allowlist_denies_over_cap_ops():
    approver = AllowlistEscalationApprover({"write": 5})
    assert await approver(_class_req("write", [], 6)) is False


@pytest.mark.asyncio
async def test_allowlist_path_confinement():
    approver = AllowlistEscalationApprover(
        {"write": 10}, allowed_path_prefixes=("/workspace",)
    )
    assert await approver(_class_req("write", ["/workspace/src/a.py"], 1)) is True
    assert await approver(_class_req("write", ["/etc/passwd"], 1)) is False
    # one bad path poisons the whole request
    assert await approver(_class_req("write", ["/workspace/a", "/etc/x"], 1)) is False


@pytest.mark.asyncio
async def test_allowlist_confined_approver_denies_unconfined_request():
    approver = AllowlistEscalationApprover(
        {"write": 10}, allowed_path_prefixes=("/workspace",)
    )
    # no paths = unrestricted grant, broader than the confinement allows
    assert await approver(_class_req("write", [], 1)) is False


@pytest.mark.asyncio
async def test_allowlist_rejects_dotdot_escape(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    approver = AllowlistEscalationApprover(
        {"write": 10}, allowed_path_prefixes=(str(workspace),)
    )
    escape = str(workspace / ".." / "etc" / "passwd")
    assert await approver(_class_req("write", [escape], 1)) is False


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
    assert await approver(_class_req("write", [str(link)], 1)) is False
    assert await approver(_class_req("write", [str(link / "passwd")], 1)) is False


@pytest.mark.asyncio
async def test_allowlist_accepts_symlink_resolving_inside(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    link = workspace / "alias"
    link.symlink_to(workspace / "src")

    approver = AllowlistEscalationApprover(
        {"write": 10}, allowed_path_prefixes=(str(workspace),)
    )
    assert await approver(_class_req("write", [str(link / "a.py")], 1)) is True


# ── unconfined tools (bash-style: no extractable path in calls) ─────────────────

@pytest.mark.asyncio
async def test_unconfined_tool_granted_only_without_paths(tmp_path):
    approver = AllowlistEscalationApprover(
        {"write": 10, "bash": 5},
        allowed_path_prefixes=(str(tmp_path),),
        unconfined_tools=("bash",),
    )
    # bash: approvable path-free even though the approver is path-confined
    assert await approver(_class_req("bash", [], 3)) is True
    # a paths-carrying bash request would produce an unusable lease — denied
    assert await approver(_class_req("bash", [str(tmp_path)], 3)) is False
    # write still requires confinement
    assert await approver(_class_req("write", [], 1)) is False


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
    assert await console_escalation_callback(_class_req("write", [], 5)) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("answer,expected", [
    ("y", True), ("YES", True), ("n", False), ("", False), ("later", False),
])
async def test_console_parses_answer(monkeypatch, answer, expected):
    monkeypatch.setattr(sys, "stdin", _FakeStdin(tty=True))
    monkeypatch.setattr("builtins.input", lambda _prompt: answer)
    assert await console_escalation_callback(_class_req("write", [], 5)) is expected


@pytest.mark.asyncio
async def test_console_denies_on_terminal_error(monkeypatch):
    """EOF mid-prompt (closed stream, ^D) is a denial, not an exception."""
    monkeypatch.setattr(sys, "stdin", _FakeStdin(tty=True))

    def _eof(_prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", _eof)
    assert await console_escalation_callback(_class_req("write", [], 5)) is False


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

    async def _boom(request) -> bool:
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


# ── effect-scoped approval ────────────────────────────────────────────────────
#
# A CLASS answer authorises a tool; an EFFECT answer authorises one call. For the
# allowlist approver that means the confinement check moves from the requested
# roots to the call's OWN path argument — strictly more precise, since a class
# grant cannot know which path a later call will carry.


@pytest.mark.asyncio
async def test_effect_request_is_confined_by_the_calls_own_path():
    approver = AllowlistEscalationApprover({"write": 1}, ["/workspace"])
    assert await approver(_effect_req("write", {"path": "/workspace/a.py"})) is True
    assert await approver(_effect_req("write", {"path": "/etc/passwd"})) is False


@pytest.mark.asyncio
async def test_effect_request_without_a_path_denies_under_confinement():
    """Same fail-closed stance as a class grant with no paths: a confined approver
    does not hand out an unconfined authority."""
    approver = AllowlistEscalationApprover({"write": 1}, ["/workspace"])
    assert await approver(_effect_req("write", {"content": "x"})) is False


@pytest.mark.asyncio
async def test_effect_request_for_an_unconfined_tool_rejects_a_path():
    approver = AllowlistEscalationApprover({"bash": 1}, [], ["bash"])
    assert await approver(_effect_req("bash", {"command": "ls"})) is True
    assert await approver(_effect_req("bash", {"path": "/workspace/x"})) is False


@pytest.mark.asyncio
async def test_effect_request_still_needs_the_tool_in_the_allowlist():
    approver = AllowlistEscalationApprover({"write": 1})
    assert await approver(_effect_req("bash", {"command": "ls"})) is False


@pytest.mark.asyncio
async def test_unconfined_approver_takes_any_path_but_only_for_listed_tools():
    approver = AllowlistEscalationApprover({"write": 1})
    assert await approver(_effect_req("write", {"path": "/anywhere/x"})) is True


@pytest.mark.asyncio
async def test_console_effect_prompt_shows_the_call(monkeypatch, capsys):
    """The human must see WHAT is being approved: the arguments and the digest.
    The arguments are untrusted content, so they are quoted (!r), never pasted raw
    into the prompt line."""
    monkeypatch.setattr(sys, "stdin", _FakeStdin(tty=True))
    seen: list[str] = []

    def _input(prompt: str) -> str:
        seen.append(prompt)
        return "y"

    monkeypatch.setattr("builtins.input", _input)
    req = _effect_req("send_money", {"recipient": "acme", "amount": 500},
                      consequence="transfer")
    assert await console_escalation_callback(req) is True
    assert len(seen) == 1
    assert "'acme'" in seen[0] and "500" in seen[0]
    assert req.effect_digest[:16] in seen[0]
    assert "one use only" in seen[0]


# ── legacy callback protocol ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_legacy_callback_is_adapted_for_class_questions():
    seen: list[tuple] = []

    async def legacy(tool_use_id, tool, paths, max_ops) -> bool:
        seen.append((tool_use_id, tool, tuple(paths), max_ops))
        return True

    with pytest.warns(DeprecationWarning, match="ApprovalRequest"):
        adapted = adapt_escalation_callback(legacy)
    assert await adapted(_class_req("write", ["/w"], 3)) is True
    assert seen == [("t1", "write", ("/w",), 3)]


@pytest.mark.asyncio
async def test_current_protocol_callback_is_not_adapted():
    async def current(request) -> bool:
        return True

    assert adapt_escalation_callback(current) is current
    assert adapt_escalation_callback(None) is None


@pytest.mark.asyncio
async def test_effect_approval_refuses_a_legacy_callback(make_envelope):
    """A four-argument callback has nowhere to receive the call's arguments, so it
    cannot answer "did the principal order THIS". Downgrading the question to a
    class-scoped one would take a tool-wide answer as authority for this effect —
    refuse instead."""
    async def legacy(tool_use_id, tool, paths, max_ops) -> bool:
        return True

    env = make_envelope()
    env = dataclasses.replace(
        env,
        policy=dataclasses.replace(
            env.policy,
            escalation_policy=EscalationPolicy(
                allow_escalation=True, grantable_tools=("read",),
                require_human=True, effect_approval=True,
            ),
        ),
    )
    with pytest.warns(DeprecationWarning):
        mgr = EscalationManager(escalation_callback=legacy)
    reason = await mgr.grant_for_effect(
        tool="read", tool_args={"path": "/w/a"}, envelope=env,
        tool_use_id="t1", consequence="read", denial_reason="over ceiling",
        trace_events=[],
    )
    assert reason is not None and "legacy signature" in reason
    assert mgr.covers("read", {"path": "/w/a"}) is False


# ── an ESCALATE_POLICY intent that names the call it is for ───────────────────
#
# The agent can ask for authority over a tool (CLASS, the original shape) or for
# authority to make one specific call (EFFECT). The second is a narrowing of the
# first: the approver is asked a concrete question, and the grant is one-shot and
# bound. No posture change — the agent opts in by naming the call.


def _effect_escalation_event(tool: str, call_args: dict, max_ops: int = 9) -> ExecutorEvent:
    return ExecutorEvent(
        kind=ExecutorEventKind.TOOL_USE,
        payload={
            "tool": "escalate_policy",
            "tool_use_id": "esc-1",
            "args": {"tool": tool, "reason": "one transfer", "paths": [],
                     "max_ops": max_ops, "args": call_args},
        },
        node_id="node-1",
    )


def _human_env(make_envelope, tool="write"):
    env = make_envelope()
    return dataclasses.replace(
        env,
        policy=dataclasses.replace(
            env.policy,
            escalation_policy=EscalationPolicy(
                allow_escalation=True, grantable_tools=(tool,), require_human=True
            ),
        ),
    )


@pytest.mark.asyncio
async def test_escalation_naming_a_call_is_effect_scoped_and_one_shot(make_envelope):
    seen: list[ApprovalRequest] = []

    async def approver(request: ApprovalRequest) -> bool:
        seen.append(request)
        return True

    call = {"path": "/w/a.py", "content": "x"}
    mgr = EscalationManager(escalation_callback=approver,
                            driving_args={"write": ["path"]})
    result = await mgr.grant_from_intent(
        _effect_escalation_event("write", call), _human_env(make_envelope), []
    )

    assert result["granted"] is True
    assert result["scope"] == "effect"
    assert result["max_ops"] == 1                 # one-shot, whatever was asked
    assert result["effect_digest"] == effect_digest("write", call)
    assert len(seen) == 1
    assert seen[0].scope is AuthorityScope.EFFECT
    assert dict(seen[0].args) == call
    assert seen[0].effect_signature == ("path",)  # as the operator declared
    # and it covers that call only
    assert mgr.covers("write", call) is True
    assert mgr.covers("write", {"path": "/w/other.py", "content": "x"}) is False


@pytest.mark.asyncio
async def test_escalation_without_a_call_stays_class_scoped(make_envelope):
    seen: list[ApprovalRequest] = []

    async def approver(request: ApprovalRequest) -> bool:
        seen.append(request)
        return True

    mgr = EscalationManager(escalation_callback=approver)
    result = await mgr.grant_from_intent(
        _escalation_event("write", max_ops=3), _human_env(make_envelope), []
    )
    assert result["granted"] is True
    assert result["scope"] == "class"
    assert result["max_ops"] == 3
    assert seen[0].scope is AuthorityScope.CLASS
    assert dict(seen[0].args) == {}
    assert mgr.covers("write") is True            # the whole tool, as before


@pytest.mark.asyncio
async def test_a_legacy_callback_can_still_answer_a_class_escalation(make_envelope):
    async def legacy(tool_use_id, tool, paths, max_ops) -> bool:
        return True

    with pytest.warns(DeprecationWarning):
        mgr = EscalationManager(escalation_callback=legacy)
    result = await mgr.grant_from_intent(
        _escalation_event("write"), _human_env(make_envelope), []
    )
    assert result["granted"] is True and result["scope"] == "class"
