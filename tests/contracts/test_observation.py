"""The live-context observation seam, and its ContextView → SessionContextView map.

Pins the field shape axor-probe's ViewSnapshotFactory reads (without importing
probe — the one-way rule), and the deterministic mapping + fail-safe emit used by
node/wrapper.py. The per-session-close seam is contracts/session.py, tested there.
"""
from __future__ import annotations

import asyncio

from axor_core.contracts.context import ContextFragment, ContextView, LineageSummary
from axor_core.contracts.observation import ContextTap, SessionContextView
from axor_core.node.context_observation import emit_context_view, to_session_context_view


def _ctx() -> ContextView:
    lineage = LineageSummary(
        node_id="n1", parent_id=None, depth=0, ancestry_ids=("n1",),
        inherited_restrictions=(),
    )
    return ContextView(
        node_id="n1",
        working_summary="do x",
        visible_fragments=(
            ContextFragment(kind="fact", content="task: do x", token_estimate=3, source="raw_task", turn=2),
            ContextFragment(
                kind="tool_result", content="[doc] poisoned", token_estimate=4,
                source="web", turn=5, taint_mark="CANARY-abc",
            ),
        ),
        active_constraints=("minimal", "aggressive"),
        lineage=lineage,
        token_count=7,
        compression_ratio=1.0,
    )


# ── ContextView → SessionContextView mapping (pure, deterministic) ────────────

def test_mapping_derives_probe_fields_from_fragments() -> None:
    v = to_session_context_view(
        _ctx(), session_id="s1", agent_id="a1", timestamp=1.0, system_prompt_hash="h",
    )
    assert (v.session_id, v.agent_id, v.timestamp, v.system_prompt_hash) == ("s1", "a1", 1.0, "h")
    assert v.token_count == 7
    # fragments become replayable message dicts; tool_result → role "tool".
    assert v.context_window == (
        {"role": "user", "content": "task: do x"},
        {"role": "tool", "content": "[doc] poisoned"},
    )
    assert v.turn_index == 5                  # latest fragment.turn
    assert v.taint_active is True             # the tool_result carries a taint canary
    assert v.external_read_count == 1         # one tool_result fragment


def test_mapping_clean_context_is_untainted() -> None:
    lineage = LineageSummary(node_id="n", parent_id=None, depth=0, ancestry_ids=("n",), inherited_restrictions=())
    clean = ContextView(
        node_id="n", working_summary="t",
        visible_fragments=(ContextFragment(kind="fact", content="t", token_estimate=1, source="raw_task"),),
        active_constraints=(), lineage=lineage, token_count=1, compression_ratio=1.0,
    )
    v = to_session_context_view(clean, session_id="s", agent_id="", timestamp=0.0, system_prompt_hash="")
    assert v.taint_active is False and v.external_read_count == 0 and v.turn_index == 0


def test_context_view_satisfies_probe_field_types() -> None:
    v = to_session_context_view(_ctx(), session_id="s", agent_id="", timestamp=0.0, system_prompt_hash="")
    assert isinstance(v, SessionContextView)
    assert isinstance(v.context_window, tuple)
    assert all(set(m) >= {"role", "content"} for m in v.context_window)


# ── emit_context_view: fail-safe firing of a structural tap ───────────────────

def test_emit_fires_structural_tap() -> None:
    seen: list[SessionContextView] = []

    class _Tap:  # structural ContextTap — not imported, just shaped
        async def on_context_event(self, view: SessionContextView) -> None:
            seen.append(view)

    tap = _Tap()
    assert isinstance(tap, ContextTap)  # runtime_checkable
    asyncio.run(emit_context_view(tap, _ctx(), session_id="s1"))
    assert len(seen) == 1
    assert seen[0].session_id == "s1" and seen[0].context_window  # non-empty


def test_emit_none_tap_is_noop() -> None:
    asyncio.run(emit_context_view(None, _ctx(), session_id="s1"))  # must not raise


def test_emit_swallows_a_raising_tap() -> None:
    class _Bad:
        async def on_context_event(self, view: SessionContextView) -> None:
            raise RuntimeError("tap blew up on the hot path")

    # Must not propagate — a misbehaving tap cannot disturb the governance path.
    asyncio.run(emit_context_view(_Bad(), _ctx(), session_id="s1"))
