"""Spawn as federation: an in-process child is a same-kernel peer. With federation
on, the parent restores the child's ACTUAL per-value provenance for its output (a
secret the child read and returned re-arms the parent's egress floor); with
federation off, the child output is conservatively re-minted untrusted (integrity
only), so the floor does not fire."""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from axor_core import GovernedSession, presets
from axor_core.capability.executor import CapabilityExecutor, ToolHandler
from axor_core.contracts.invokable import Invokable
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind
from axor_core.contracts.trace import TraceConfig
from axor_core.federation import FederationGateway
from tests.conftest import EchoExecutor

pytestmark = pytest.mark.adversarial

SECRET = "SECRET_API_TOKEN_abcdef0123456789"


class _SecretReader(ToolHandler):
    @property
    def name(self) -> str:
        return "read"

    async def execute(self, args) -> str:
        return SECRET


class _ChildReadsAndReturnsSecret(Invokable):
    """Child agent: reads a secret file, then returns the secret as its output."""

    async def stream(self, envelope) -> AsyncIterator[ExecutorEvent]:
        yield ExecutorEvent(kind=ExecutorEventKind.TOOL_USE,
                            payload={"tool": "read", "args": {"path": ".env"}},
                            node_id=envelope.node_id)
        yield ExecutorEvent(kind=ExecutorEventKind.TEXT,
                            payload={"text": SECRET},   # surfaces the secret as output
                            node_id=envelope.node_id)
        yield ExecutorEvent(kind=ExecutorEventKind.STOP,
                            payload={"usage": {}}, node_id=envelope.node_id)


def _cap() -> CapabilityExecutor:
    ex = CapabilityExecutor()
    ex.register(_SecretReader())
    return ex


async def _run_with_spawn(*, federation: bool):
    # Parent agent: spawns one child, then stops.
    parent = EchoExecutor(tool_calls=[("spawn_child", {"task": "fetch the token"})])
    gateway = FederationGateway(peers={}, compatible_kernels=set(), federated_domains=set()) \
        if federation else None
    sess = GovernedSession(
        executor=parent,
        capability_executor=_cap(),
        child_executor=_ChildReadsAndReturnsSecret(),
        trace_config=TraceConfig(local_only=True, persist_inputs=False),
        federation_gateway=gateway,
    )
    # federated preset allows children
    await sess.run("delegate to a child", policy=presets.get("federated"))
    return sess


@pytest.mark.asyncio
async def test_federation_on_restores_child_secret_provenance():
    sess = await _run_with_spawn(federation=True)
    # the child's secret was surfaced in its output → its sensitive provenance is
    # restored in the parent ledger → the parent's egress floor is armed. This is
    # the per-value precision a blanket untrusted re-mint cannot express.
    assert sess._taint_engine.confidentiality_floor_active() is True


@pytest.mark.asyncio
async def test_federation_off_remints_child_output_untrusted_nonsensitive():
    sess = await _run_with_spawn(federation=False)
    # conservative default: child output is re-minted untrusted — integrity-tainted
    # but NOT sensitive (the parent cannot see the child's reads), so the
    # confidentiality floor does NOT fire.
    assert sess._taint_engine.confidentiality_floor_active() is False
