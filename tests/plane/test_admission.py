"""PlaneAdmission drives the IntentLoop boundary: pause holds, stop winds down,
absent/disconnected plane always admits (advisory overlay, spec 12.0)."""
from __future__ import annotations

import asyncio

import pytest

from axor_core.plane.admission import PlaneAdmission
from axor_core.plane.session import PlaneSession


async def test_running_session_admits() -> None:
    adm = PlaneAdmission(PlaneSession(node_id="n0"))
    assert await adm.await_admission("n0") is True


async def test_stopped_session_returns_false() -> None:
    s = PlaneSession(node_id="n0")
    s.apply_snapshot(1, {"stopped": True})
    adm = PlaneAdmission(s)
    assert await adm.await_admission("n0") is False


async def test_paused_holds_until_resumed() -> None:
    s = PlaneSession(node_id="n0")
    s.apply_snapshot(1, {"paused": True})
    adm = PlaneAdmission(s, poll_interval=0.01)

    async def resume_soon() -> None:
        await asyncio.sleep(0.03)
        s.apply_snapshot(2, {"paused": False})
        adm.notify()

    resumer = asyncio.create_task(resume_soon())
    admitted = await asyncio.wait_for(adm.await_admission("n0"), timeout=1.0)
    await resumer
    assert admitted is True


async def test_paused_then_stopped_returns_false() -> None:
    s = PlaneSession(node_id="n0")
    s.apply_snapshot(1, {"paused": True})
    adm = PlaneAdmission(s, poll_interval=0.01)

    async def stop_soon() -> None:
        await asyncio.sleep(0.03)
        s.apply_snapshot(2, {"stopped": True})
        adm.notify()

    stopper = asyncio.create_task(stop_soon())
    admitted = await asyncio.wait_for(adm.await_admission("n0"), timeout=1.0)
    await stopper
    assert admitted is False


async def test_notify_resumes_without_waiting_full_interval() -> None:
    s = PlaneSession(node_id="n0")
    s.apply_snapshot(1, {"paused": True})
    adm = PlaneAdmission(s, poll_interval=10.0)  # long poll; notify must wake it

    async def resume_now() -> None:
        s.apply_snapshot(2, {"paused": False})
        adm.notify()

    resumer = asyncio.create_task(resume_now())
    admitted = await asyncio.wait_for(adm.await_admission("n0"), timeout=0.5)
    await resumer
    assert admitted is True
