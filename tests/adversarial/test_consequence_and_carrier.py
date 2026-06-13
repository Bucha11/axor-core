"""Operation-aware consequence gating, carrier sanitization, and value-policy
wiring.

Covers: dangerous-looking values (paths, URLs, non-finite numbers) classify as
free text while closed structured forms do not; destructive shell operations
(shutdown, reboot, disk wipes) are gated by action class even when they look
benign to provenance checks; and value policies configured on a session reach the
node that enforces them.
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock

import pytest

from axor_core.capability.executor import CapabilityExecutor, ToolHandler
from axor_core.contracts.cancel import make_token
from axor_core.contracts.context import ContextView, LineageSummary
from axor_core.contracts.envelope import (
    Capabilities,
    ExecutionEnvelope,
    ExportContract,
)
from axor_core.contracts.policy import ExecutionPolicy, ExportMode, ToolPolicy
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind
from axor_core.node.intent_loop import IntentLoop
from axor_core.security.carrier import classify_carrier
from axor_core.contracts.taint import Carrier
from axor_core.taint.engine import TaintEngine

pytestmark = pytest.mark.adversarial


# ── carrier sanitization ──────────────────────────────────────────────────────

@pytest.mark.parametrize("value", [
    "/etc/passwd", "../../secret", "http://attacker.example/x",
    "https://x.y/z", "localhost:8080", "a b c", math.inf, math.nan,
    {"x": math.inf}, {"path": "/etc/shadow"}, "{\"x\": Infinity}",
])
def test_dangerous_forms_are_free_text(value):
    assert classify_carrier(value) == Carrier.FREE_TEXT


@pytest.mark.parametrize("value", [
    42, True, None, "svc-1_abc", "1.2.3", {"action": "deploy", "v": "1.0"},
])
def test_closed_forms_unchanged(value):
    assert classify_carrier(value) != Carrier.FREE_TEXT


# ── operation-aware consequence (destructive bash commands) ────────────────────

class _Handler(ToolHandler):
    def __init__(self, name: str, output: Any) -> None:
        self._n, self._o = name, output

    @property
    def name(self) -> str:
        return self._n

    async def execute(self, args: dict[str, Any]) -> Any:
        return self._o


def _env(tool="bash") -> ExecutionEnvelope:
    policy = ExecutionPolicy(name="t", tool_policy=ToolPolicy(allow_bash=True))
    lineage = LineageSummary(node_id="n1", parent_id=None, depth=0,
                             ancestry_ids=[], inherited_restrictions=[])
    ctx = ContextView(node_id="n1", working_summary="t", visible_fragments=[],
                      active_constraints=[], lineage=lineage, token_count=0,
                      compression_ratio=1.0)
    caps = Capabilities(
        allowed_tools=frozenset({tool}), allow_children=False,
        allow_nested_children=False, allow_context_expansion=False,
        allow_export=True, allow_mutation=True, max_child_depth=0,
    )
    return ExecutionEnvelope(
        node_id="n1", task="t", context=ctx, policy=policy, capabilities=caps,
        export_contract=ExportContract(mode=ExportMode.FULL,
                                       allowed_fields=frozenset(["output"]),
                                       max_export_tokens=1024),
        lineage=lineage, cancel_token=make_token(),
    )


async def _resolve(loop, env, tool, args):
    ev = ExecutorEvent(kind=ExecutorEventKind.TOOL_USE,
                       payload={"tool": tool, "args": args, "tool_use_id": "x"},
                       node_id=env.node_id)
    return await loop._resolve_tool_intent(ev, env)


def _loop() -> IntentLoop:
    ex = CapabilityExecutor()
    ex.register(_Handler("bash", "ok"))
    return IntentLoop(capability_executor=ex, trace_events=[], taint_engine=TaintEngine())


@pytest.mark.asyncio
async def test_power_state_bash_is_gated():
    # `bash shutdown` looks benign to the provenance checks but is catastrophic by
    # action class, so the operation classifier escalates it past the unattended
    # ceiling and denies it.
    r = await _resolve(_loop(), _env(), "bash", {"cmd": "shutdown -h now"})
    assert not r.approved
    assert r.result.get("category") == "consequence_gate"


@pytest.mark.asyncio
async def test_ordinary_bash_still_passes():
    r = await _resolve(_loop(), _env(), "bash", {"cmd": "ls -la"})
    assert r.approved


@pytest.mark.parametrize("cmd", [
    "reboot", "sudo poweroff", "systemctl halt", "mkfs.ext4 /dev/sda1",
    "dd if=/dev/zero of=/dev/sda",
])
@pytest.mark.asyncio
async def test_power_state_variants_gated(cmd):
    r = await _resolve(_loop(), _env(), "bash", {"cmd": cmd})
    assert not r.approved


# ── value_policies wired through GovernedSession ───────────────────────────────

def test_session_forwards_value_policies_to_node():
    from axor_core.worker.session import GovernedSession
    from axor_core.policy.value_policy import enum

    pols = {"deploy": [enum("target", ["staging", "prod"])]}
    sess = GovernedSession(
        executor=MagicMock(),
        capability_executor=CapabilityExecutor(),
        value_policies=pols,
    )
    assert sess._value_policies == pols
    node = sess._make_node(sess._context_manager)
    assert node._value_policies == pols
