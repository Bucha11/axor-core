"""The causal-subgraph walk (spec v2 Ch.3): minimal causes, roles, anchoring,
containment, size-1 degeneracy."""
from __future__ import annotations

import pytest
from axor_core.kernel.events import Event, EventKind, Verdict
from axor_core.kernel.subgraph import causal_subgraph


def _ev(seq: int, node: str, kind: str, **payload) -> Event:
    verdict = payload.pop("verdict", None)
    gate = payload.pop("gate", None)
    return Event(
        seq=seq, node_id=node, kind=EventKind(kind), ts=f"seq:{seq}",
        gate=gate, verdict=Verdict(verdict) if verdict else None, payload=payload,
    )


WEB = {"root": {"sources": ["web"], "sensitive": False}}


def _tree() -> list[Event]:
    """The demo story: fault at scraper → fabrication carried up two hops →
    export denied at the orchestrator. Plus an UNRELATED fourth node."""
    return [
        _ev(0, "scraper", "fault_injected", tool="web_search", mode="silent_fail"),
        _ev(1, "scraper", "tool_call", verdict="pass", tool="web_search",
            args={}, arg_refs={}),
        _ev(2, "scraper", "tool_result", tool="web_search", value_ref="v_fab",
            root=WEB["root"]),
        _ev(3, "scraper", "message_sent", verdict="pass", to="research",
            edge_kind="delegation", msg_id="m1", value_ref="v_fab", carried=WEB),
        _ev(0, "research", "message_received", **{"from": "scraper"},
            edge_kind="delegation", msg_id="m1", value_ref="v_fab", carried=WEB),
        _ev(1, "research", "tool_call", verdict="pass", tool="summarize",
            args={}, arg_refs={"text": "v_fab"}),
        _ev(2, "research", "tool_result", tool="summarize", value_ref="v_sum",
            root=WEB["root"]),
        _ev(3, "research", "message_sent", verdict="pass", to="orch",
            edge_kind="delegation", msg_id="m2", value_ref="v_sum", carried=WEB),
        _ev(0, "orch", "message_received", **{"from": "research"},
            edge_kind="delegation", msg_id="m2", value_ref="v_sum", carried=WEB),
        _ev(1, "orch", "tool_call", verdict="deny", gate="taint_enforcement",
            tool="slack_post", args={}, arg_refs={"text": "v_sum"}),
        # unrelated node — must NOT appear in the case
        _ev(0, "writer", "tool_call", verdict="pass", tool="format", args={}),
    ]


def test_walk_finds_the_three_causal_nodes_not_the_org_chart() -> None:
    sub = causal_subgraph(_tree(), "orch", 1)
    ids = {n["node_id"] for n in sub["nodes"]}
    assert ids == {"scraper", "research", "orch"}  # writer excluded


def test_roles() -> None:
    sub = causal_subgraph(_tree(), "orch", 1)
    roles = {n["node_id"]: set(n["roles"]) for n in sub["nodes"]}
    assert "origin" in roles["scraper"]
    assert "conduit" in roles["research"]
    assert {"anchor", "container"} <= roles["orch"]  # denied at the anchor


def test_fault_origin_and_containment() -> None:
    sub = causal_subgraph(_tree(), "orch", 1)
    assert sub["fault_origin"] == {"node_id": "scraper", "seq": 0}
    assert sub["contained_at"] and sub["contained_at"][0]["gate"] == "taint_enforcement"
    assert sub["federation_scope"] == "intra"


def test_edges_are_the_message_hops_with_carried_labels() -> None:
    sub = causal_subgraph(_tree(), "orch", 1)
    hops = {(e["from"], e["to"]) for e in sub["edges"]}
    assert hops == {("scraper", "research"), ("research", "orch")}
    assert all(e["carried"]["root"]["sources"] == ["web"] for e in sub["edges"])
    assert all(e["gate_verdict"] == "pass" for e in sub["edges"])


def test_size1_degenerates_to_anchor_only() -> None:
    """One node, no edges — exactly the v0.13 case; renderers branch on size."""
    events = [
        _ev(0, "solo", "fault_injected", tool="web_search", mode="silent_fail"),
        _ev(1, "solo", "tool_call", verdict="pass", tool="web_search",
            args={}, arg_refs={}),
        _ev(2, "solo", "tool_result", tool="web_search", value_ref="v_f",
            root=WEB["root"]),
        _ev(3, "solo", "claim", text="all good"),
    ]
    sub = causal_subgraph(events, "solo", 3)
    assert [n["node_id"] for n in sub["nodes"]] == ["solo"]
    assert sub["edges"] == []
    assert sub["federation_scope"] == "intra"


def test_missing_anchor_raises() -> None:
    with pytest.raises(ValueError):
        causal_subgraph(_tree(), "orch", 99)


def test_deterministic() -> None:
    assert causal_subgraph(_tree(), "orch", 1) == causal_subgraph(_tree(), "orch", 1)
