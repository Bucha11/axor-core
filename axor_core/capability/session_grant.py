"""Signed session grants for the daemon capability boundary.

The daemon used to trust the ``allowed_tools`` list a client sent on every
request. Any process running as the same OS user that could open the socket
could therefore claim a wider session ceiling (still bounded by the operator
policy, but wider than the legitimate session).

A session grant is a short-lived HMAC token, issued by the holder of
``AXOR_DAEMON_SESSION_KEY`` (the governed agent / core), that states the allowed
tools and an expiry. The daemon verifies it server-side and derives the session
ceiling from the *grant*, not from the bare client claim. A process without the
key cannot forge a grant.

Token format: ``base64url(json_body) + "." + hmac_hex``  where json_body is
``{"allowed_tools": [...], "exp": <unix>, "policy_hash": "..."}``.

Threat-model note: the key lives in the process environment, which is readable by
the same user via /proc/<pid>/environ. This raises the bar from "open the socket"
to "read another process's environment" (often blocked by hidepid) — it is not a
cryptographic boundary against a fully-compromised same-user context.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Sequence

SESSION_KEY_ENV = "AXOR_DAEMON_SESSION_KEY"
_DEFAULT_TTL = 300.0


def _key(explicit: bytes | None = None) -> bytes | None:
    if explicit is not None:
        return explicit
    raw = os.environ.get(SESSION_KEY_ENV, "")
    return raw.encode() if raw else None


def session_key_configured() -> bool:
    return _key() is not None


def issue_grant(
    allowed_tools: Sequence[str],
    *,
    ttl_seconds: float = _DEFAULT_TTL,
    policy_hash: str = "",
    key: bytes | None = None,
) -> str | None:
    """Issue a signed grant. Returns None when no key is configured."""
    k = _key(key)
    if k is None:
        return None
    body = {
        "allowed_tools": sorted(set(allowed_tools)),
        "exp": time.time() + ttl_seconds,
        "policy_hash": policy_hash,
    }
    raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    sig = hmac.new(k, raw, hashlib.sha256).hexdigest()
    return base64.urlsafe_b64encode(raw).decode() + "." + sig


def verify_grant(token: str | None, *, key: bytes | None = None) -> dict | None:
    """Verify a grant token. Returns the decoded body or None if invalid/expired."""
    k = _key(key)
    if k is None or not token:
        return None
    try:
        b64, sig = token.split(".", 1)
        raw = base64.urlsafe_b64decode(b64.encode())
    except (ValueError, base64.binascii.Error):
        return None
    expected = hmac.new(k, raw, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        body = json.loads(raw)
    except ValueError:
        return None
    if float(body.get("exp", 0)) < time.time():
        return None
    return body
