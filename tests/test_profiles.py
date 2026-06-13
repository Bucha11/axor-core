"""Preset profiles — product surface via the existing entry (no new method)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from axor_core.contracts.canonical import ConsequenceClass
from axor_core.contracts.mode import ExecutionMode
from axor_core.contracts.policy import ExecutionPolicy
from axor_core.policy.composer import PolicyComposer
from axor_core.profiles import PROFILES, resolve_profile


def test_resolve_known_and_unknown():
    for name in ("observe", "balanced", "strict", "dev"):
        assert resolve_profile(name).name == name
    assert resolve_profile(PROFILES["strict"]) is PROFILES["strict"]
    with pytest.raises(ValueError):
        resolve_profile("nope")


def test_overlay_imposes_ceiling_and_workspace():
    comp = PolicyComposer(consequence_ceiling=ConsequenceClass.REVERSIBLE, allowed_paths=("/proj",))
    out = comp.compose(ExecutionPolicy(name="t"), [])
    assert out.max_unattended_consequence == ConsequenceClass.REVERSIBLE
    assert out.allowed_paths == ("/proj",)


def test_overlay_absent_is_passthrough():
    base = ExecutionPolicy(name="t")
    assert PolicyComposer().compose(base, []).max_unattended_consequence == base.max_unattended_consequence


def test_profile_param_expands_into_session_knobs():
    from axor_core.worker.session import GovernedSession
    s = GovernedSession(executor=MagicMock(), capability_executor=MagicMock(),
                        profile="dev", danger={"transfer_money": ConsequenceClass.CATASTROPHIC},
                        workspace="/proj")
    assert s._mode == ExecutionMode.LIBRARY
    assert s._consequence_overrides == {"transfer_money": ConsequenceClass.CATASTROPHIC}
    assert s._composer._consequence_ceiling == ConsequenceClass.CATASTROPHIC
    assert s._overlay_allowed_paths == ("/proj",)


def test_danger_override_gates_custom_tool(cap_executor):
    from axor_core.node.intent_loop import IntentLoop
    from axor_core.contracts.cancel import make_token
    from axor_core.contracts.context import ContextView, LineageSummary
    from axor_core.contracts.envelope import Capabilities, ExecutionEnvelope, ExportContract
    from axor_core.contracts.policy import ExportMode, ToolPolicy

    def _env():
        pol = ExecutionPolicy(name="t", tool_policy=ToolPolicy(allow_read=True, allow_write=True))
        ln = LineageSummary(node_id="n1", parent_id=None, depth=0, ancestry_ids=[], inherited_restrictions=[])
        ctx = ContextView(node_id="n1", working_summary="t", visible_fragments=[], active_constraints=[],
                          lineage=ln, token_count=0, compression_ratio=1.0)
        caps = Capabilities(allowed_tools=frozenset({"transfer_money"}), allow_children=False,
                            allow_nested_children=False, allow_context_expansion=False, allow_export=False,
                            allow_mutation=True, max_child_depth=0)
        return ExecutionEnvelope(node_id="n1", task="t", context=ctx, policy=pol, capabilities=caps,
                                 export_contract=ExportContract(mode=ExportMode.RESTRICTED, allowed_fields=frozenset(), max_export_tokens=0),
                                 lineage=ln, cancel_token=make_token())

    plain = IntentLoop(capability_executor=cap_executor, trace_events=[])
    assert plain._check_consequence("transfer_money", _env()) is None  # default CONSEQUENTIAL, within ceiling
    armed = IntentLoop(capability_executor=cap_executor, trace_events=[],
                       consequence_overrides={"transfer_money": ConsequenceClass.CATASTROPHIC})
    assert armed._check_consequence("transfer_money", _env()) is not None  # now gated
