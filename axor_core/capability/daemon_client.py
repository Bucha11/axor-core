"""
DaemonCapabilityClient — drop-in replacement for CapabilityExecutor.

Forwards tool calls to a running AxorDaemon over a Unix socket.
Tool implementations live only in the daemon process, not here.

Drop-in usage:

    from axor_core.capability.daemon_client import DaemonCapabilityClient

    cap = DaemonCapabilityClient(socket_path="~/.axor/daemon.sock")
    session = GovernedSession(executor=llm, capability_executor=cap)

The daemon must be started separately:

    python -m axor_daemon start

Fail-closed: if the daemon is unreachable, execute() raises DaemonUnavailableError
and execution stops. It never silently degrades to direct execution.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any, Callable, Awaitable

from axor_core.contracts.daemon import encode_message, read_message, PROTOCOL_VERSION
from axor_core.contracts.envelope import Capabilities
from axor_core.contracts.intent import Intent, IntentKind
from axor_core.capability.session_grant import issue_grant
from axor_core.errors.exceptions import (
    DaemonUnavailableError,
    DaemonRejectedError,
    ToolNotAllowedError,
)

PostExecuteCallback = Callable[[str, dict[str, Any], Any], Awaitable[None]]


class DaemonCapabilityClient:
    """
    Capability executor that delegates all tool execution to AxorDaemon.

    Exposes the same interface as CapabilityExecutor so it drops in
    anywhere CapabilityExecutor is accepted without further changes.
    """

    # Marker read by GovernedSession: tool execution happens out-of-process,
    # so a compromised agent process cannot reach tool implementations directly.
    is_process_isolated: bool = True

    def __init__(
        self,
        socket_path: str = "~/.axor/daemon.sock",
        mode: str = "library",
        connect_timeout: float = 3.0,
        call_timeout: float = 30.0,
    ) -> None:
        self._socket_path = os.path.expanduser(socket_path)
        self._mode = mode
        self._connect_timeout = connect_timeout
        self._call_timeout = call_timeout

        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._connected = False
        self._post_callbacks: list[PostExecuteCallback] = []
        self._lock = asyncio.Lock()

    # ── Public interface (matches CapabilityExecutor) ──────────────────────────

    def register_post_callback(self, callback: PostExecuteCallback) -> None:
        """Post-execute callbacks fire locally with the result returned by daemon."""
        self._post_callbacks.append(callback)

    def registered_tools(self) -> frozenset[str]:
        """
        Returns empty set — tool registrations live in the daemon.
        Callers that need the daemon's tool list should use list_tools() instead.
        """
        return frozenset()

    async def execute(self, intent: Intent, capabilities: Capabilities) -> Any:
        """
        Forward an approved tool intent to the daemon for execution.

        Raises:
            DaemonUnavailableError  — daemon not reachable (fail-closed)
            DaemonRejectedError     — daemon rejected the session on connect
            ToolNotAllowedError     — daemon denied the tool call
        """
        if intent.kind != IntentKind.TOOL_CALL:
            raise ValueError(
                f"DaemonCapabilityClient only handles TOOL_CALL, got {intent.kind}"
            )

        tool_name: str = intent.payload.get("tool", "")
        tool_args: dict[str, Any] = intent.payload.get("args", {})

        async with self._lock:
            return await self._execute_locked(tool_name, tool_args, capabilities)

    async def _execute_locked(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        capabilities: Capabilities,
    ) -> Any:
        await self._ensure_connected()

        call_id = uuid.uuid4().hex
        request = {
            "type": "tool_call",
            "call_id": call_id,
            "tool": tool_name,
            "args": tool_args,
            "allowed_tools": sorted(capabilities.allowed_tools),
        }
        # Attach a signed session grant when a session key is configured. The
        # daemon then derives the session ceiling from the grant, not from the
        # plaintext allowed_tools above (which a rogue same-user process could
        # forge). No key → field omitted → daemon falls back to legacy trust.
        grant = issue_grant(capabilities.allowed_tools)
        if grant is not None:
            request["grant"] = grant

        try:
            self._writer.write(encode_message(request))
            response = await asyncio.wait_for(
                self._send_and_read_response(),
                timeout=self._call_timeout,
            )
        except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError) as exc:
            self._connected = False
            raise DaemonUnavailableError(self._socket_path, str(exc)) from exc
        except asyncio.TimeoutError as exc:
            self._connected = False
            raise DaemonUnavailableError(
                self._socket_path,
                f"tool call timed out after {self._call_timeout}s",
            ) from exc

        if response.get("type") != "tool_result":
            raise DaemonUnavailableError(
                self._socket_path,
                f"unexpected response type: {response.get('type')!r}",
            )
        if response.get("call_id") != call_id:
            raise DaemonUnavailableError(
                self._socket_path,
                "response call_id mismatch",
            )

        if response["decision"] == "denied":
            raise ToolNotAllowedError(
                tool=tool_name,
                allowed=set(capabilities.allowed_tools),
            )

        result = response.get("result")

        for cb in self._post_callbacks:
            try:
                await cb(tool_name, tool_args, result)
            except Exception:
                pass

        return result

    async def aclose(self) -> None:
        if self._writer is not None:
            try:
                self._writer.write(encode_message({"type": "bye"}))
                await self._writer.drain()
            except Exception:
                pass
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
        self._reader = None
        self._writer = None
        self._connected = False

    # ── Private ────────────────────────────────────────────────────────────────

    async def _send_and_read_response(self) -> dict[str, Any]:
        await self._writer.drain()
        return await read_message(self._reader)

    async def _ensure_connected(self) -> None:
        if self._connected:
            return

        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_unix_connection(self._socket_path),
                timeout=self._connect_timeout,
            )
        except (FileNotFoundError, ConnectionRefusedError, OSError) as exc:
            raise DaemonUnavailableError(self._socket_path, str(exc)) from exc
        except asyncio.TimeoutError:
            raise DaemonUnavailableError(
                self._socket_path, "connection timed out"
            )

        # handshake
        hello = {"type": "hello", "mode": self._mode}
        try:
            self._writer.write(encode_message(hello))
            await self._writer.drain()
            ack = await asyncio.wait_for(
                read_message(self._reader), timeout=self._connect_timeout
            )
        except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError,
                asyncio.TimeoutError) as exc:
            raise DaemonUnavailableError(self._socket_path, str(exc)) from exc

        server_version = ack.get("v")
        if server_version != PROTOCOL_VERSION:
            raise DaemonRejectedError(
                f"protocol version mismatch: client={PROTOCOL_VERSION} server={server_version}"
            )
        if ack.get("type") == "rejected":
            raise DaemonRejectedError(ack.get("reason", "unknown"))
        if ack.get("type") != "ready":
            raise DaemonUnavailableError(
                self._socket_path,
                f"unexpected handshake response: {ack.get('type')!r}",
            )

        self._connected = True
