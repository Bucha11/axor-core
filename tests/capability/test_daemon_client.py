from __future__ import annotations

import asyncio

import pytest

from axor_core.capability.daemon_client import DaemonCapabilityClient
from axor_core.contracts.envelope import Capabilities
from axor_core.contracts.intent import Intent, IntentKind
from axor_core.errors.exceptions import DaemonUnavailableError


class _HangingWriter:
    def write(self, data: bytes) -> None:
        self.data = data

    async def drain(self) -> None:
        return None


def _caps() -> Capabilities:
    return Capabilities(
        allowed_tools=frozenset({"read"}),
        allow_children=False,
        allow_nested_children=False,
        allow_context_expansion=False,
        allow_export=True,
        allow_mutation=False,
        max_child_depth=0,
    )


@pytest.mark.asyncio
async def test_execute_times_out_when_daemon_never_responds() -> None:
    client = DaemonCapabilityClient(socket_path="/tmp/axor-test.sock", call_timeout=0.01)
    client._connected = True
    client._reader = asyncio.StreamReader()
    client._writer = _HangingWriter()
    intent = Intent(
        kind=IntentKind.TOOL_CALL,
        payload={"tool": "read", "args": {"path": "x.py"}},
        node_id="n",
    )

    with pytest.raises(DaemonUnavailableError, match="tool call timed out"):
        await client.execute(intent, _caps())
