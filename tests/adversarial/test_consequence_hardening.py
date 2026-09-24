"""Consequence axis against renames.

The axis looked a class up by the tool's exact name, and an unknown name fell to
CONSEQUENTIAL — unattended at the default ceiling. ``shutdown`` was gated;
``shutdown_server``, ``drop_table`` or ``wipe_disk`` ran unattended. Two layers:

  * a name-token heuristic in every mode — tightening only, never over the table
    or an operator's explicit class;
  * STRICT: a tool with no explicit class (table key or operator override) is
    CATASTROPHIC at run time on every path (governor, IntentLoop, replay), and a
    STRICT session with such a registered tool fails to construct.
"""
from __future__ import annotations

import pytest

from axor_core import GovernedSession, ToolCallGovernor
from axor_core.capability.executor import CapabilityExecutor, ToolHandler
from axor_core.contracts.canonical import ConsequenceClass
from axor_core.contracts.mode import ExecutionMode
from axor_core.contracts.trace import TraceConfig
from axor_core.kernel.events import Event, EventKind, Verdict
from axor_core.kernel.registration import validate_consequence_completeness
from axor_core.kernel.replay import KernelConfig, replay
from axor_core.policy.consequence import consequence_class, is_consequence_classified
from tests.conftest import EchoExecutor

pytestmark = pytest.mark.adversarial

C = ConsequenceClass


# ── heuristic (every mode) ────────────────────────────────────────────────────

@pytest.mark.parametrize("name", [
    "shutdown_server", "server_shutdown", "shutdownServer", "RebootHost",
    "poweroff_node", "wipe_disk", "drop_table", "delete_database", "truncate_tables",
    "destroy_cluster", "purge_bucket", "restart_gateway_now", "restart_server",
    "factory_reset_device", "rm_rf", "delete_all_files",
])
def test_destructive_names_are_catastrophic(name):
    assert consequence_class(name) is C.CATASTROPHIC


@pytest.mark.parametrize("name, expected", [
    ("read_file", C.CONSEQUENTIAL),        # unknown, nothing destructive: default
    ("delete_file", C.CONSEQUENTIAL),      # destructive verb, but no infra object
    ("restart_timer", C.CONSEQUENTIAL),    # restart of a non-infra object
    ("send_email", C.CONSEQUENTIAL),
    ("read", C.BENIGN),                    # table key wins over the default
    ("write", C.REVERSIBLE),
])
def test_heuristic_does_not_fire_on_ordinary_names(name, expected):
    assert consequence_class(name) is expected


def test_heuristic_never_overrides_an_explicit_class():
    """An operator who declares a class owns it — even one the name would raise."""
    assert consequence_class("wipe_scratch", overrides={"wipe_scratch": C.REVERSIBLE}) is C.REVERSIBLE


def test_heuristic_does_not_classify_for_strict():
    assert not is_consequence_classified("shutdown_server")
    assert is_consequence_classified("shutdown")
    assert is_consequence_classified("send_email", {"send_email": C.CONSEQUENTIAL})


def test_renamed_destructive_tool_is_gated_by_the_governor():
    d = ToolCallGovernor().evaluate("shutdown_server", {})
    assert d.allowed is False and d.category == "consequence_gate"


# ── STRICT: unclassified means catastrophic ───────────────────────────────────

def test_strict_classes_every_unknown_tool_catastrophic():
    assert consequence_class("exfiltrate_blob", strict=True) is C.CATASTROPHIC
    assert consequence_class("send_email", strict=True) is C.CATASTROPHIC
    assert consequence_class("send_email", overrides={"send_email": C.CONSEQUENTIAL},
                             strict=True) is C.CONSEQUENTIAL
    assert consequence_class("read", strict=True) is C.BENIGN          # table key
    assert consequence_class("spawn_child", strict=True) is C.CONSEQUENTIAL  # internal


def _strict_governor(**kw) -> ToolCallGovernor:
    return ToolCallGovernor(untrusted_sources={"send_report"}, require_tool_roles=True, **kw)


def test_strict_governor_refuses_an_unclassified_tool_and_says_why():
    d = _strict_governor().evaluate("send_report", {})
    assert d.allowed is False and d.category == "consequence_gate"
    assert "no declared consequence class" in d.reason


def test_strict_governor_admits_the_same_tool_once_classified():
    d = _strict_governor(consequence_overrides={"send_report": C.CONSEQUENTIAL}).evaluate(
        "send_report", {})
    assert d.allowed


@pytest.mark.asyncio
async def test_strict_intent_loop_gates_an_unclassified_tool():
    from tests.adversarial.test_unknown_sink_posture import _env
    from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind
    from axor_core.node.intent_loop import IntentLoop
    from axor_core.taint.engine import TaintEngine

    class _H(ToolHandler):
        @property
        def name(self): return "custom_unknown_sink"
        async def execute(self, args): return "ok"

    async def resolve(**kw):
        ex = CapabilityExecutor()
        ex.register(_H())
        loop = IntentLoop(capability_executor=ex, trace_events=[], taint_engine=TaintEngine(),
                          benign_tools={"custom_unknown_sink"}, **kw)
        ev = ExecutorEvent(kind=ExecutorEventKind.TOOL_USE,
                           payload={"tool": "custom_unknown_sink", "args": {}, "tool_use_id": "u"},
                           node_id="n1")
        return await loop._resolve_tool_intent(ev, _env(C.CONSEQUENTIAL))

    assert (await resolve()).approved                       # non-STRICT: default class
    strict = await resolve(require_tool_roles=True)
    assert not strict.approved
    assert strict.result.get("category") == "consequence_gate"
    classified = await resolve(require_tool_roles=True,
                               consequence_overrides={"custom_unknown_sink": C.BENIGN})
    assert classified.approved


def test_replay_keeps_parity_with_the_strict_gate():
    """A recorded PASS of an unclassified tool re-gates as a DENY under the STRICT
    consequence posture — the same decision the runtime gate makes."""
    event = Event(seq=1, node_id="n", kind=EventKind.TOOL_CALL, ts="1", verdict=Verdict.PASS,
                  payload={"tool": "send_report", "args": {},
                           "normalized": {"destination_kind": "none",
                                          "writes_outside_workdir": False,
                                          "executes_generated_code": False},
                           "driving_root": {"sources": [], "sensitive": False}})
    assert replay([event], KernelConfig()).first_divergence is None
    strict = replay([event], KernelConfig(strict_consequence=True))
    assert strict.steps[0].reevaluated_verdict is Verdict.DENY
    assert strict.steps[0].deny.category == "consequence_gate"


# ── STRICT session construction ───────────────────────────────────────────────

def test_validate_consequence_completeness():
    errs = validate_consequence_completeness(
        {"read", "send_email", "get_time"}, {"get_time": C.BENIGN})
    assert len(errs) == 1 and "send_email" in errs[0]


def _strict_session(tools, **kw):
    class _H(ToolHandler):
        def __init__(self, n): self._n = n
        @property
        def name(self): return self._n
        async def execute(self, args): return "x"

    cap = CapabilityExecutor()
    for t in tools:
        cap.register(_H(t))
    return GovernedSession(
        executor=EchoExecutor(), capability_executor=cap,
        trace_config=TraceConfig(local_only=True, persist_inputs=False),
        mode=ExecutionMode.STRICT, **kw,
    )


def test_strict_session_refuses_a_registered_tool_without_a_class():
    with pytest.raises(ValueError, match="strict consequence completeness"):
        _strict_session(["get_time"], benign_tools={"get_time"})


def test_strict_session_constructs_once_every_tool_is_classified():
    assert _strict_session(["get_time", "read"], benign_tools={"get_time", "read"},
                           danger={"get_time": C.BENIGN}) is not None
