"""Unit tests for EvalGovernedToolsExecutor fault layering (fake governor/runtime)."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(__file__))

from agentdojo.functions_runtime import FunctionCall  # noqa: E402

import eval_deprivation as ed  # noqa: E402


class _Governor:
    def __init__(self, allow=True, category="approved"):
        self.allow = allow
        self.category = category
        self.registered: list[str] = []

    def evaluate(self, fn, args):
        return SimpleNamespace(allowed=self.allow, category=self.category)

    def register_output(self, decision, output):
        self.registered.append(output)


class _Runtime:
    def __init__(self, result="REAL_RESULT", error=None):
        self.result = result
        self.error = error
        self.calls: list[str] = []

    def run_function(self, env, fn, args):
        self.calls.append(fn)
        return self.result, self.error


def _assistant(fn: str, args=None):
    return {"role": "assistant", "content": [],
            "tool_calls": [FunctionCall(id="1", function=fn, args=args or {})]}


def _run(executor, runtime, fn="get_balance"):
    out = executor.query("q", runtime, env=None, messages=[_assistant(fn)], extra_args={})
    messages = out[3]
    return messages[-1]  # the appended tool-result message


def test_silent_fail_replaces_result_and_records():
    gov = _Governor()
    ex = ed.EvalGovernedToolsExecutor(lambda: gov, fault_spec={"get_balance": "silent_fail"})
    rt = _Runtime(result="$500")
    res = _run(ex, rt, "get_balance")
    assert res["error"] == ed._SILENT_FAIL_ERROR
    assert res["content"][0]["content"] == ""        # no real data surfaced
    assert rt.calls == ["get_balance"]               # real tool was invoked...
    assert gov.registered == []                      # ...but not registered into taint ledger
    assert ex.fault_records == [ed.FaultRecord("get_balance", "silent_fail")]


def test_non_target_tool_passes_through():
    gov = _Governor()
    ex = ed.EvalGovernedToolsExecutor(lambda: gov, fault_spec={"get_balance": "silent_fail"})
    rt = _Runtime(result="msg list")
    res = _run(ex, rt, "read_inbox")                 # not in fault_spec
    assert res["error"] is None
    assert res["content"][0]["content"] == "msg list"
    assert gov.registered == ["msg list"]
    assert ex.fault_records == []


def test_denied_call_is_governance_denial_not_faulted():
    gov = _Governor(allow=False, category="taint_enforcement")
    ex = ed.EvalGovernedToolsExecutor(lambda: gov, fault_spec={"send_money": "silent_fail"})
    rt = _Runtime()
    res = _run(ex, rt, "send_money")
    assert "governance kernel" in res["error"]
    assert "taint_enforcement" in res["error"]
    assert rt.calls == []                            # denied → real tool never ran
    assert ex.denied_count == 1
    assert ex.fault_records == []                    # denial is not a fault injection


def test_no_fault_spec_is_plain_governed_executor():
    gov = _Governor()
    ex = ed.EvalGovernedToolsExecutor(lambda: gov)   # no faults
    rt = _Runtime(result="data")
    res = _run(ex, rt, "get_balance")
    assert res["error"] is None
    assert res["content"][0]["content"] == "data"
    assert ex.fault_records == []
