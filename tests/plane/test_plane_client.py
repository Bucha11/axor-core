"""PlaneClient transport: telemetry durability and the heartbeat loop.

The point of these tests is protocol §5 — the telemetry direction owns
durability. A backend that blinks must not cost the operator an ack.
"""
from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("httpx")

from axor_core.plane.client import PlaneClient  # noqa: E402
from axor_core.plane.session import PlaneSession  # noqa: E402

# Port 1 refuses instantly — a fast, deterministic "backend is down".
DEAD_BACKEND = "http://127.0.0.1:1"


def _session_with_ack() -> PlaneSession:
    s = PlaneSession(node_id="n0")
    s.outbox.append({"kind": "state_applied", "payload": {"version": 1, "result": "x"}})
    return s


def test_spool_roundtrip(tmp_path) -> None:  # noqa: ANN001
    client = PlaneClient(DEAD_BACKEND, PlaneSession(node_id="n0"),
                         spool_path=str(tmp_path / "spool.jsonl"))
    events = [{"kind": "a", "payload": {"x": 1}}, {"kind": "b", "payload": {}}]
    client._save_spool(events)
    assert client._load_spool() == events
    client._save_spool([])  # empty clears the file
    assert client._load_spool() == []


async def test_flush_keeps_outbox_durable_on_failure(tmp_path) -> None:  # noqa: ANN001
    spool = str(tmp_path / "spool.jsonl")
    session = _session_with_ack()
    client = PlaneClient(DEAD_BACKEND, session, spool_path=spool)
    await client.flush()  # backend down — must not raise
    # The outbox was drained, but the ack is not lost: it is on the spool.
    assert session.outbox == []
    spooled = client._load_spool()
    assert len(spooled) == 1 and spooled[0]["kind"] == "state_applied"


async def test_flush_without_spool_holds_events_in_memory() -> None:
    session = _session_with_ack()
    client = PlaneClient(DEAD_BACKEND, session)  # no spool → in-memory only
    await client.flush()  # backend down
    # With no disk spool the unsent ack goes back to the front of the outbox,
    # so the next flush retries it — it is not silently dropped.
    assert [e["kind"] for e in session.outbox] == ["state_applied"]


async def test_heartbeat_loop_ticks_and_stops(monkeypatch) -> None:  # noqa: ANN001
    session = PlaneSession(node_id="n0")
    client = PlaneClient(DEAD_BACKEND, session, heartbeat_period=0.01)
    calls: list[str] = []

    async def fake_flush(**kwargs) -> None:  # noqa: ANN003
        calls.append(kwargs.get("level", "NORMAL"))

    monkeypatch.setattr(client, "flush", fake_flush)
    stop = asyncio.Event()

    async def run() -> None:
        await client.heartbeat_loop(stop, level_fn=lambda: "CAUTIOUS")

    task = asyncio.create_task(run())
    await asyncio.sleep(0.05)
    stop.set()
    await task
    assert calls and all(c == "CAUTIOUS" for c in calls)
