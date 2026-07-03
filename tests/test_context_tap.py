"""Core → Probe observation seam: ContextTap / SessionContextView.

The seam is observe-only: taps get a structural post-turn snapshot, tap
failures never disturb the governance path, and the view is built from shaped
ContextFragments — never raw executor history. axor-probe's CoreContextTap
matches ContextTap structurally without importing axor-core (P-34), so the
field set pinned here is the cross-repo contract.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from axor_core.capability.executor import CapabilityExecutor
from axor_core.contracts.context import ContextFragment
from axor_core.contracts.observation import ContextTap, SessionContextView
from axor_core.contracts.taint import TaintSource
from axor_core.taint.causal_root import CausalRoot
from axor_core.worker.session import GovernedSession


class _RecordingTap:
    def __init__(self) -> None:
        self.views: list[SessionContextView] = []

    async def on_context_event(self, view: SessionContextView) -> None:
        self.views.append(view)


class _FailingTap:
    async def on_context_event(self, view: SessionContextView) -> None:
        raise RuntimeError("boom")


def _session(*taps) -> GovernedSession:
    cap = MagicMock(spec=CapabilityExecutor)
    cap.register_post_callback = MagicMock()
    return GovernedSession(
        executor=MagicMock(),
        capability_executor=cap,
        context_taps=list(taps) or None,
    )


def test_recording_tap_satisfies_protocol():
    assert isinstance(_RecordingTap(), ContextTap)


async def test_tap_receives_structural_view():
    tap = _RecordingTap()
    session = _session(tap)
    session._context_manager.pin_fragment(ContextFragment(
        kind="skill", content="system rules", token_estimate=10,
        source="agent:personality",
    ))
    session._context_manager.ingest_fragments([ContextFragment(
        kind="fact", content="user fact", token_estimate=5, source="task",
        taint_mark="AXOR_CANARY_deadbeef",
    )])
    session._taint_engine.register_value(
        "external payload", CausalRoot.external_read(TaintSource.WEB)
    )

    await session._notify_context_taps()

    assert len(tap.views) == 1
    view = tap.views[0]
    assert view.session_id == session.session_id()
    assert view.taint_active is True
    assert view.external_read_count == 1
    assert view.taint_canaries == ("AXOR_CANARY_deadbeef",)
    assert view.token_count == 15
    # Fragment kinds map onto provider message roles; content is the shaped
    # fragment content, never raw history.
    assert {"role": "system", "content": "system rules"} in view.context_window
    assert {"role": "user", "content": "user fact"} in view.context_window


async def test_view_carries_probe_expected_fields():
    # The exact attribute set axor-probe's _SessionContextViewLike declares.
    tap = _RecordingTap()
    session = _session(tap)
    await session._notify_context_taps()
    view = tap.views[0]
    for attr in (
        "session_id", "agent_id", "timestamp", "turn_index", "token_count",
        "context_window", "system_prompt_hash", "taint_active",
        "external_read_count", "taint_canaries",
    ):
        assert hasattr(view, attr), attr


async def test_failing_tap_is_swallowed_and_others_still_fire():
    recorder = _RecordingTap()
    session = _session(_FailingTap(), recorder)
    await session._notify_context_taps()  # no raise
    assert len(recorder.views) == 1


async def test_no_taps_is_noop():
    session = _session()
    await session._notify_context_taps()  # no raise, no work
