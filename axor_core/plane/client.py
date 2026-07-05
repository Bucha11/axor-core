"""PlaneClient: the outbound-only transport around :class:`PlaneSession`.

The adapter dials out (protocol section 1): an SSE subscription for desired
state and batched telemetry POSTs. Zero listening sockets on user
infrastructure. Channel loss is survivable by construction: enforcement is
local, commands are best-effort, and an optional ``hold_on_disconnect`` pauses
the node locally after a grace period (adapter config — the backend can
neither cause nor prevent it, protocol section 7).

Requires the ``plane`` extra (httpx): axor-core itself stays dependency-free.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Callable

from axor_core.plane.session import PlaneSession

HEARTBEAT_PERIOD = 10.0  # protocol section 9: static T=10s, stale=3T


def _httpx():  # noqa: ANN202 - module import guard
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "the plane client needs httpx — install the plane extra: "
            "pip install 'axor-core[plane]'"
        ) from exc
    return httpx


class PlaneClient:
    def __init__(
        self,
        backend_url: str,
        session: PlaneSession,
        run_id: str | None = None,
        on_fact: Callable[[dict], None] | None = None,
        hold_on_disconnect_after: float | None = None,
        heartbeat_period: float = HEARTBEAT_PERIOD,
    ) -> None:
        self._base = backend_url.rstrip("/")
        self.session = session
        self._run_id = run_id or session.node_id
        self._on_fact = on_fact
        self._hold_after = hold_on_disconnect_after
        self._heartbeat_period = heartbeat_period
        self._seq = 0

    # ── telemetry ─────────────────────────────────────────────────────────────

    async def flush(
        self, extra_events: list[dict] | None = None,
        level: str = "NORMAL", budget_remaining: int | None = None,
    ) -> None:
        """POST outbox + heartbeat as one idempotent batch."""
        httpx = _httpx()
        events = list(self.session.outbox) + list(extra_events or [])
        self.session.outbox.clear()
        events.append(self.session.heartbeat(level, budget_remaining))
        lines = [self._line(e) for e in events]
        async with httpx.AsyncClient(timeout=15.0) as client:
            await client.post(
                f"{self._base}/v1/plane/{self.session.node_id}/telemetry",
                json={"run_id": self._run_id, "events": lines},
                headers={"Idempotency-Key": uuid.uuid4().hex},
            )
        for e in events:
            if e["kind"] in ("injection_consumed", "context_excision"):
                key = ("pending_injection" if e["kind"] == "injection_consumed"
                       else "pending_excision")
                async with httpx.AsyncClient(timeout=15.0) as client:
                    await client.post(
                        f"{self._base}/v1/plane/{self.session.node_id}/consumed",
                        json={"key": key},
                    )

    def _line(self, event: dict) -> dict:
        from datetime import UTC, datetime

        seq = self._seq
        self._seq += 1
        return {
            "schema_version": "1.0",
            "seq": seq,
            "node_id": self.session.node_id,
            "kind": event["kind"],
            "ts": datetime.now(UTC).isoformat(),
            "causal_root": None,
            "gate": None,
            "verdict": None,
            "payload": event.get("payload", {}),
        }

    # ── downstream subscription ───────────────────────────────────────────────

    async def run(self, stop: asyncio.Event) -> None:
        """Subscribe to desired state until `stop` is set. Reconnects with
        backoff; fail-continue under local config on channel loss."""
        httpx = _httpx()
        backoff = 1.0
        while not stop.is_set():
            try:
                async with httpx.AsyncClient(timeout=None) as client, client.stream(
                    "GET", f"{self._base}/v1/plane/{self.session.node_id}/desired"
                ) as response:
                    backoff = 1.0
                    await self._consume_sse(response, stop)
            except httpx.HTTPError:
                if self._hold_after is not None:
                    try:
                        await asyncio.wait_for(stop.wait(), self._hold_after)
                        return
                    except TimeoutError:
                        self.session.paused = True  # hold_on_disconnect opted in
                else:
                    try:
                        await asyncio.wait_for(stop.wait(), backoff)
                        return
                    except TimeoutError:
                        backoff = min(backoff * 2, 30.0)

    async def _consume_sse(self, response, stop: asyncio.Event) -> None:  # noqa: ANN001
        event_name = ""
        data_lines: list[str] = []
        async for raw in response.aiter_lines():
            if stop.is_set():
                return
            line = raw.rstrip("\n")
            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
            elif line == "" and data_lines:
                self._dispatch(event_name, json.loads("\n".join(data_lines)))
                event_name, data_lines = "", []

    def _dispatch(self, event_name: str, data: dict) -> None:
        if event_name == "snapshot":
            self.session.apply_snapshot(int(data["version"]), data.get("state", {}))
        elif event_name == "delta":
            self.session.apply_delta(
                int(data["version"]), data.get("delta", data.get("state", {})),
                data.get("operator", ""), data.get("timestamp", ""),
                data.get("sig", ""),
            )
        elif event_name == "fact" and self._on_fact is not None:
            self._on_fact(data.get("fact", {}))
