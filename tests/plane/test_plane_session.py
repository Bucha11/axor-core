"""PlaneSession: the adapter-side semantics that keep a compromised backend
harmless (protocol v0.2 sections 3-6, 8)."""
from __future__ import annotations

import json

import pytest

from axor_core.plane.session import PlaneSession

cryptography = pytest.importorskip("cryptography")
from cryptography.hazmat.primitives.asymmetric import ed25519  # noqa: E402


def _keypair() -> tuple[ed25519.Ed25519PrivateKey, str]:
    key = ed25519.Ed25519PrivateKey.generate()
    from cryptography.hazmat.primitives import serialization

    pub = key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    return key, pub.hex()


def _sign(key: ed25519.Ed25519PrivateKey, node_id: str, version: int,
          body: dict, ts: str) -> str:
    message = json.dumps(
        {"node_id": node_id, "version": version, "body": body, "timestamp": ts},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()
    return key.sign(message).hex()


def test_snapshot_applies_and_stale_versions_are_noops() -> None:
    s = PlaneSession(node_id="n0")
    assert s.apply_snapshot(3, {"paused": True}).kind == "applied"
    assert s.paused
    assert s.apply_snapshot(2, {"paused": False}).kind == "noop_stale"
    assert s.paused


def test_stopped_is_absorbing() -> None:
    s = PlaneSession(node_id="n0")
    s.apply_snapshot(1, {"stopped": True})
    effect = s.apply_snapshot(2, {"paused": True})
    assert effect.kind == "noop_absorbed"
    assert not s.paused and s.stopped
    assert not s.admit_intent()


def test_signed_delta_verified_against_local_keys() -> None:
    key, pub = _keypair()
    s = PlaneSession(node_id="n0", operator_pubkeys={"op": pub})
    ts = "2026-07-05T00:00:00Z"
    sig = _sign(key, "n0", 1, {"paused": True}, ts)
    assert s.apply_delta(1, {"paused": True}, "op", ts, sig).kind == "applied"
    assert s.paused


def test_forged_delta_dropped_and_reported() -> None:
    _, pub = _keypair()
    rogue, _ = _keypair()
    s = PlaneSession(node_id="n0", operator_pubkeys={"op": pub})
    ts = "t"
    sig = _sign(rogue, "n0", 1, {"stopped": True}, ts)
    effect = s.apply_delta(1, {"stopped": True}, "op", ts, sig)
    assert effect.kind == "sig_invalid"
    assert not s.stopped
    assert any(o["payload"]["result"] == "sig_invalid" for o in s.outbox)


def test_unknown_operator_is_sig_invalid() -> None:
    _, pub = _keypair()
    s = PlaneSession(node_id="n0", operator_pubkeys={"op": pub})
    effect = s.apply_delta(1, {"paused": True}, "op_rogue", "t", "00" * 64)
    assert effect.kind == "sig_invalid"


def test_budget_widening_rejected_at_adapter() -> None:
    s = PlaneSession(node_id="n0", local_budget_cap=100)
    assert s.apply_snapshot(1, {"budget_cap_calls": 50}).kind == "applied"
    assert s.budget_cap_calls == 50
    effect = s.apply_snapshot(2, {"budget_cap_calls": 500})
    assert effect.kind == "rejected_widening"
    assert s.budget_cap_calls == 50  # a compromised backend cannot widen


def test_injection_at_most_once_and_test_bench_gated() -> None:
    inj = {"id": "inj_1", "text": "probe recovery", "reason": "r",
           "operator": "op", "sig": "s"}
    prod = PlaneSession(node_id="n0", test_bench=False)
    prod.apply_snapshot(1, {"pending_injection": inj})
    assert prod.take_pending_injection() is None  # decision 5: prod refuses
    assert any(o["kind"] == "injection_refused" for o in prod.outbox)

    bench = PlaneSession(node_id="n0", test_bench=True)
    bench.apply_snapshot(1, {"pending_injection": inj})
    taken = bench.take_pending_injection()
    assert taken is not None and taken.text == "probe recovery"
    # replayed snapshot after reconnect: same id is a no-op
    bench.apply_snapshot(2, {"pending_injection": inj})
    assert bench.take_pending_injection() is None
    assert sum(1 for o in bench.outbox
               if o["kind"] == "injection_consumed") == 1


def test_injection_refused_when_stopped() -> None:
    inj = {"id": "inj_9", "text": "x", "reason": "r", "operator": "op", "sig": "s"}
    s = PlaneSession(node_id="n0", test_bench=True)
    s.apply_snapshot(1, {"pending_injection": inj})
    s.apply_snapshot(2, {"stopped": True})  # stop AFTER the injection is pending
    assert s.take_pending_injection() is None  # absorbed by the stopped lattice
    refusal = next(o for o in s.outbox if o["kind"] == "injection_refused")
    assert refusal["payload"]["reason"] == "stopped"
    assert "inj_9" not in s.consumed_ids  # refused, not consumed


def test_excision_provenance_guard_refuses_whole_heal() -> None:
    exc = {"id": "exc_1", "target_refs": ["v1", "v2"], "reason": "drift",
           "operator": "op", "sig": "s"}
    s = PlaneSession(node_id="n0", test_bench=True)
    s.apply_snapshot(1, {"pending_excision": exc})
    taken = s.take_pending_excision({"v1": "runtime", "v2": "operator_config"})
    assert taken is None  # no partial, silently-narrower heal
    refusal = next(o for o in s.outbox if o["kind"] == "excision_refused")
    assert refusal["payload"]["refs"] == ["v2"]

    s.apply_snapshot(2, {"pending_excision": exc})
    taken = s.take_pending_excision({"v1": "runtime", "v2": "runtime"})
    assert taken is not None
    emitted = next(o for o in s.outbox if o["kind"] == "context_excision")
    assert emitted["payload"]["refs"] == ["v1", "v2"]


def test_heartbeat_carries_applied_version_and_consumed_ids() -> None:
    s = PlaneSession(node_id="n0", test_bench=True)
    s.apply_snapshot(4, {"pending_injection": {"id": "i1", "text": "x",
                                               "reason": "r", "operator": "o",
                                               "sig": "s"}})
    s.take_pending_injection()
    hb = s.heartbeat(level="CAUTIOUS", budget_remaining=7)
    assert hb["payload"]["applied_version"] == 4
    assert hb["payload"]["consumed_injection_ids"] == ["i1"]
    assert hb["payload"]["level"] == "CAUTIOUS"
