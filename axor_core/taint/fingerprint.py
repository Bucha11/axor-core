"""Canonical content fingerprint (Ring 0).

A stable, order-independent hash of a value's content, used to identify a specific
secret read for the confidentiality floor and to bind a federation receipt to its
value. Kept here (Ring 0) so both the kernel's taint engine and the runtime's
federation layer share one definition; federation re-exports it as ``value_hash``.

Avoids ``repr`` — which is order-unstable for dicts and non-injective for objects
with a custom/constant ``__repr__`` (two distinct values could share a fingerprint).
"""
from __future__ import annotations

import hashlib
import json


def content_fingerprint(value: object) -> str:
    """Canonical sha256 hex of a value's content.

    Strings/bytes hash directly; everything else goes through canonical, sorted
    JSON, falling back to a type-tagged repr only for the genuinely unserialisable
    (where the type tag at least prevents a cross-type collision)."""
    if isinstance(value, bytes):
        material = b"b:" + value
    elif isinstance(value, str):
        material = b"s:" + value.encode()
    else:
        try:
            material = b"j:" + json.dumps(
                value, sort_keys=True, separators=(",", ":"), default=str
            ).encode()
        except (TypeError, ValueError):
            material = f"r:{type(value).__module__}.{type(value).__qualname__}:{value!r}".encode()
    return hashlib.sha256(material).hexdigest()
