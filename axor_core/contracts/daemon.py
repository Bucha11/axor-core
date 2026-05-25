"""
Wire protocol for AxorDaemon ↔ DaemonCapabilityClient.

Framing: 4-byte big-endian unsigned int (message length) + JSON bytes.

All messages carry "v": PROTOCOL_VERSION. Server rejects clients with a
mismatched version at handshake time so the error is explicit, not silent.

Connection lifecycle:
  1. Client connects to Unix socket
  2. Client → {"v": 1, "type": "hello", "mode": "library"|"production"|"strict"}
  3. Server → {"v": 1, "type": "ready"} or {"v": 1, "type": "rejected", "reason": str}
  4. Per tool call:
     Client → {"v": 1, "type": "tool_call", "call_id": str, "tool": str,
                "args": {...}, "allowed_tools": [...]}
     Server → {"v": 1, "type": "tool_result", "call_id": str,
                "decision": "approved"|"denied",
                "result": <any>, "denial_reason": str|None}
  5. Client → {"v": 1, "type": "bye"} on close (best-effort, not required)

No external dependencies — stdlib json + struct only.
"""

from __future__ import annotations

import json
import struct
from typing import Any

PROTOCOL_VERSION = 1

_HEADER = struct.Struct(">I")
_MAX_MESSAGE_BYTES = 8 * 1024 * 1024  # 8 MB hard cap


def encode_message(payload: dict[str, Any]) -> bytes:
    payload = {"v": PROTOCOL_VERSION, **payload}
    body = json.dumps(payload, default=_json_default).encode()
    return _HEADER.pack(len(body)) + body


async def read_message(reader) -> dict[str, Any]:
    header = await reader.readexactly(_HEADER.size)
    n = _HEADER.unpack(header)[0]
    if n > _MAX_MESSAGE_BYTES:
        raise ValueError(f"Daemon message too large: {n} bytes")
    body = await reader.readexactly(n)
    return json.loads(body)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (frozenset, set)):
        return sorted(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)!r}")
