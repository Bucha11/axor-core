"""The capability gate is authoritative.

Enforcement is a pure structural gate. The capability check
(`_evaluate_tool_intent`) denies an unauthorized tool outright; there is no
probabilistic layer in enforcement that could override that decision.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from axor_core.contracts.policy import PolicyDecisionKind, ExecutionPolicy, ToolPolicy
from axor_core.contracts.envelope import Capabilities
from axor_core.contracts.intent import Intent, IntentKind
from axor_core.node.intent_loop import IntentLoop


def _intent(tool_name: str) -> Intent:
    return Intent(kind=IntentKind.TOOL_CALL,
                  payload={"tool": tool_name, "args": {}, "tool_use_id": "tid"}, node_id="test")


def _envelope_without_tool():
    from axor_core.contracts.envelope import ExecutionEnvelope, ExportContract
    from axor_core.contracts.context import ContextView, LineageSummary
    from axor_core.contracts.cancel import make_token
    from axor_core.contracts.policy import ExportMode
    policy = ExecutionPolicy(name="restrictive", tool_policy=ToolPolicy(allow_read=False, allow_write=False))
    ln = LineageSummary(node_id="n1", parent_id=None, depth=0, ancestry_ids=[], inherited_restrictions=[])
    ctx = ContextView(node_id="n1", working_summary="t", visible_fragments=[], active_constraints=[],
                      lineage=ln, token_count=0, compression_ratio=1.0)
    caps = Capabilities(allowed_tools=frozenset(), allow_children=False, allow_nested_children=False,
                        allow_context_expansion=False, allow_export=False, allow_mutation=False, max_child_depth=0)
    return ExecutionEnvelope(node_id="n1", task="t", context=ctx, policy=policy, capabilities=caps,
                             export_contract=ExportContract(mode=ExportMode.RESTRICTED, allowed_fields=frozenset(), max_export_tokens=0),
                             lineage=ln, cancel_token=make_token())


def test_capability_gate_denies_unauthorized_tool():
    loop = IntentLoop(capability_executor=MagicMock(), trace_events=[])
    assert loop._evaluate_tool_intent(_intent("bash"), _envelope_without_tool())[0].kind == PolicyDecisionKind.DENY


def test_capability_gate_is_authoritative_for_any_tool():
    loop = IntentLoop(capability_executor=MagicMock(), trace_events=[])
    assert loop._evaluate_tool_intent(_intent("execute_code"), _envelope_without_tool())[0].kind == PolicyDecisionKind.DENY
