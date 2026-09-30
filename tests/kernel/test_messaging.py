"""Kernel messaging: pure boundary-gate predicates and the multi-node fold.

Spec v2 Ch.4 — labels ride in envelopes, gates run locally on both edge ends,
no shared state; Ch.1 §1 — intra-federation labels are data, carried intact;
hop count cleans nothing.
"""
from __future__ import annotations

import json

from axor_core.contracts.degradation import DegradationLevel
from axor_core.kernel.events import Event, EventKind, Verdict, event_from_json_line
from axor_core.kernel.messaging import (
    EDGE_LATERAL,
    EDGE_PEER,
    evaluate_message_send,
    fold_carried_root,
)
from axor_core.kernel.replay import KernelConfig, replay, replay_tree
from axor_core.taint.causal_root import CausalRoot

# ── send gate ──────────────────────────────────────────────────────────────────


def test_send_passes_for_normal_lateral() -> None:
    assert evaluate_message_send(
        DegradationLevel.NORMAL, CausalRoot.cross_process_in(), EDGE_LATERAL
    ) is None


def test_send_carries_taint_instead_of_denying_intra() -> None:
    # Intra edges carry labels; a tainted value may travel — it denies at the
    # sink that matters, not at the internal hop (carried, not laundered).
    assert evaluate_message_send(
        DegradationLevel.CAUTIOUS, CausalRoot.cross_process_in(), EDGE_LATERAL
    ) is None


def test_send_denied_at_locked() -> None:
    deny = evaluate_message_send(
        DegradationLevel.LOCKED, CausalRoot.constant(), EDGE_LATERAL
    )
    assert deny is not None and deny.category == "degradation"


def test_send_denied_unknown_edge_kind() -> None:
    deny = evaluate_message_send(
        DegradationLevel.NORMAL, CausalRoot.constant(), "carrier-pigeon"
    )
    assert deny is not None and deny.category == "message_gate"


def test_send_denied_undeclared_peer_fail_closed() -> None:
    deny = evaluate_message_send(
        DegradationLevel.NORMAL, CausalRoot.constant(), EDGE_PEER
    )
    assert deny is not None and "undeclared" in deny.reason


def test_send_declared_peer_passes_the_base_predicate() -> None:
    assert evaluate_message_send(
        DegradationLevel.NORMAL, CausalRoot.constant(), EDGE_PEER,
        peer_declared=True,
    ) is None


# ── carried-root fold ──────────────────────────────────────────────────────────


def test_fold_missing_root_fails_closed_to_untrusted() -> None:
    folded = fold_carried_root(None)
    assert folded.is_tainted and not folded.sensitive


def test_fold_carried_root_arrives_intact() -> None:
    root = CausalRoot.cross_process_in()
    assert fold_carried_root(root) is root


# ── multi-node fold (replay) ───────────────────────────────────────────────────


def _ev(seq: int, node: str, kind: str, **payload) -> Event:
    verdict = payload.pop("verdict", None)
    return Event(
        seq=seq, node_id=node, kind=EventKind(kind), ts=f"seq:{seq}",
        verdict=Verdict(verdict) if verdict else None, payload=payload,
    )


def _tree_events() -> list[Event]:
    """scraper (fault, tainted result) → researcher (lateral) → orchestrator;
    orchestrator's export drives off the carried ref."""
    return [
        # scraper: a web-tainted tool result
        _ev(0, "scraper", "tool_call", verdict="pass", tool="web_search", args={}),
        _ev(1, "scraper", "tool_result", tool="web_search", value_ref="v_fab",
            root={"sources": ["web"], "sensitive": False}),
        _ev(2, "scraper", "message_sent", verdict="pass", to="researcher",
            edge_kind="lateral", msg_id="m1", value_ref="v_fab",
            carried={"root": {"sources": ["web"], "sensitive": False}}),
        # researcher: receives, folds carried taint, forwards
        _ev(0, "researcher", "message_received", **{"from": "scraper"},
            edge_kind="lateral", msg_id="m1", value_ref="v_fab",
            carried={"root": {"sources": ["web"], "sensitive": False}}),
        _ev(1, "researcher", "message_sent", verdict="pass", to="orch",
            edge_kind="delegation", msg_id="m2", value_ref="v_fab",
            carried={"root": {"sources": ["web"], "sensitive": False}}),
        # orchestrator: receives, then tries to export the carried value
        _ev(0, "orch", "message_received", **{"from": "researcher"},
            edge_kind="delegation", msg_id="m2", value_ref="v_fab",
            carried={"root": {"sources": ["web"], "sensitive": False}}),
        _ev(1, "orch", "tool_call", verdict="pass", tool="slack_post",
            args={"text": "…"}, arg_refs={"text": "v_fab"},
            normalized={"destination_kind": "external_domain"}),
    ]


def test_message_received_folds_carried_taint() -> None:
    per_node = replay_tree(_tree_events())
    for node in ("scraper", "researcher", "orch"):
        final = per_node[node].steps[-1].state
        assert "v_fab" in final.tainted_refs, f"{node} lost the carried taint"
        assert final.tainted_refs["v_fab"].is_tainted


def test_carried_taint_denies_export_under_config() -> None:
    """Containment, kernel-grounded: the orchestrator's export re-gates to
    DENY because its driving ref carries the taint minted two nodes away."""
    config = KernelConfig(egress_sinks=frozenset({"slack_post"}))
    per_node = replay_tree(_tree_events(), config)
    export_step = per_node["orch"].steps[-1]
    assert export_step.reevaluated_verdict is Verdict.DENY
    assert export_step.deny is not None
    assert export_step.deny.category == "taint_enforcement"


def test_sensitive_carried_root_arms_receiver_floor() -> None:
    events = [
        _ev(0, "b", "message_received", **{"from": "a"}, edge_kind="lateral",
            msg_id="m", value_ref="v_sec",
            carried={"root": {"sources": ["file"], "sensitive": True}}),
    ]
    result = replay(events)
    assert result.steps[-1].state.floor_active is True


def test_message_without_carried_root_folds_untrusted() -> None:
    events = [
        _ev(0, "b", "message_received", **{"from": "a"}, edge_kind="lateral",
            msg_id="m", value_ref="v_x", carried={}),
    ]
    result = replay(events)
    assert result.steps[-1].state.tainted_refs["v_x"].is_tainted


def test_cycle_refolds_monotonically() -> None:
    """A→B→A: the value returns to its origin; labels re-fold by union — the
    cycle cannot launder, taint only accumulates (Ch.4 open item)."""
    events = [
        _ev(0, "a", "tool_result", tool="t", value_ref="v_c",
            root={"sources": ["web"], "sensitive": False}),
        _ev(1, "a", "message_received", **{"from": "b"}, edge_kind="lateral",
            msg_id="m2", value_ref="v_c",
            carried={"root": {"sources": [], "sensitive": False}}),
    ]
    result = replay(events)
    final = result.steps[-1].state
    # The returning "clean" claim did not wash the existing web taint.
    assert final.tainted_refs["v_c"].is_tainted


def test_replay_tree_size1_is_exactly_replay() -> None:
    """The size-1 degeneracy (spec v2 header): one node, no edges — identical
    result to the v0.13 single-trace fold."""
    single = [
        _ev(0, "solo", "tool_call", verdict="pass", tool="read", args={}),
        _ev(1, "solo", "tool_result", tool="read", value_ref="v_1",
            root={"sources": ["web"], "sensitive": False}),
    ]
    tree = replay_tree(single)
    flat = replay(single)
    assert list(tree.keys()) == ["solo"]
    assert tree["solo"] == flat


def test_tree_fold_is_deterministic() -> None:
    a = replay_tree(_tree_events(), KernelConfig(egress_sinks=frozenset({"slack_post"})))
    b = replay_tree(_tree_events(), KernelConfig(egress_sinks=frozenset({"slack_post"})))
    assert a == b


def test_events_round_trip_json() -> None:
    for e in _tree_events():
        from axor_core.kernel.events import event_to_json_line

        assert event_from_json_line(event_to_json_line(e)) == e


def test_node_spawned_folds_as_structure_only() -> None:
    events = [
        _ev(0, "parent", "node_spawned", child_id="c1", parent_id="parent",
            depth=1, edge_kind="delegation"),
    ]
    result = replay(events)
    s = result.steps[-1].state
    assert not s.tainted_refs and s.level is DegradationLevel.NORMAL
