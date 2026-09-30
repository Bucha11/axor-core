r"""JCS — JSON Canonicalization Scheme (RFC 8785), for signed plane payloads.

The control plane signs the canonical bytes of a command/fact so a compromised
backend cannot forge one (protocol §6). Both sides — the adapter that verifies
and the backend that (defensively) checks — MUST agree on the exact bytes, so
the canonicalizer lives here, once, in the pure kernel, and both import it.

Scope: governance payloads carry only objects, arrays, strings, integers,
booleans and null — never floats (an IEEE-754 amount has no place in a signed
command, and refusing them sidesteps RFC 8785 §3.2.2.3's number-formatting
subtleties). Within that subset this is a conformant RFC 8785 serializer:

- object members sorted by the **UTF-16 code units** of their keys (§3.2.3),
  not Python's default code-point order — they differ for non-BMP keys;
- strings escaped with exactly the mandatory set (§3.2.2.2): ``\"`` ``\\``
  ``\b`` ``\f`` ``\n`` ``\r`` ``\t`` and ``\u00xx`` for the remaining C0
  controls, every other character emitted literally as UTF-8;
- integers rendered in their shortest form (no plus, no leading zeros);
- no insignificant whitespace.

Stdlib only — fits the kernel purity contract.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from axor_core.kernel.errors import KernelError


class CanonicalizationError(KernelError):
    """Value cannot be canonicalized (float present, or a non-string key)."""


# Mandatory two-char escapes (RFC 8785 §3.2.2.2 / RFC 8259).
_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\b": "\\b",
    "\f": "\\f",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}


def _escape_string(s: str) -> str:
    out = ['"']
    for ch in s:
        esc = _ESCAPES.get(ch)
        if esc is not None:
            out.append(esc)
        elif ch < "\x20":
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)  # literal UTF-8 for everything else
    out.append('"')
    return "".join(out)


def _utf16_key(key: str) -> tuple[int, ...]:
    """Sort key = the sequence of UTF-16 code units (RFC 8785 §3.2.3).

    Identical to code-point order for BMP characters; differs only for
    non-BMP keys (which become surrogate pairs). Encoding big-endian and
    reading 2 bytes at a time yields exactly the code-unit sequence.
    """
    raw = key.encode("utf-16-be")
    return tuple(int.from_bytes(raw[i : i + 2], "big") for i in range(0, len(raw), 2))


def _serialize(value: object) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return _escape_string(value)
    if isinstance(value, bool):  # pragma: no cover - handled above
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        raise CanonicalizationError("floats are not permitted in signed payloads")
    if isinstance(value, Mapping):
        for k in value:
            if not isinstance(k, str):
                raise CanonicalizationError("object keys must be strings")
        items = [
            f"{_escape_string(k)}:{_serialize(value[k])}"
            for k in sorted(value.keys(), key=_utf16_key)
        ]
        return "{" + ",".join(items) + "}"
    if isinstance(value, Sequence):  # list/tuple (str already handled)
        return "[" + ",".join(_serialize(v) for v in value) + "]"
    raise CanonicalizationError(f"unserializable type {type(value).__name__}")


def canonicalize(value: object) -> bytes:
    """Return the RFC 8785 canonical UTF-8 bytes of a float-free JSON value."""
    return _serialize(value).encode("utf-8")
