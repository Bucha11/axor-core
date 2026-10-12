"""Opaque handles over canonicalized values (Ring 0).

A grant constrains a driving argument to an exact value. Comparing raw strings would
make the grant as weak as its spelling: ``"GB29 NWBK 6016 1331 9268 19"`` and
``"gb29nwbk60161331926819"`` are one account. So a value is first CANONICALIZED by
its declared kind, then interned as a HANDLE — an opaque id that names the canonical
value and records where it came from.

Two rules carry the security argument:

  * A handle's trust comes from its origin, never from its value. Minting through
    ``mint_trusted`` takes a ``TrustedChannel`` (request literal, operator template,
    structured request, human confirmation); a tool read mints through
    ``mint_untrusted``. Nothing reachable from ``register_output`` can mint trusted.
  * Merging is exact. A read value whose CANONICAL form equals a trusted handle's
    resolves to that trusted handle (the attacker gains nothing: it is the value the
    user already authorized). A look-alike — one character off, a different
    encoding — canonicalizes differently and gets a fresh handle that no grant names.

Canonicalizers are deterministic and structural (T0: no model, no network). A
canonicalizer bug is a false-ALLOW bug, the same class as a normalizer bug; they are
kept small for that reason.
"""

from __future__ import annotations

import enum
import hashlib
import posixpath
import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Callable


class TrustedChannel(enum.Enum):
    """Where an authority-bearing value may come from (the L1–L4 sources)."""

    REQUEST = "L1"        # a literal in the user's request text
    OPERATOR = "L2"       # an operator template / enum
    STRUCTURED = "L3"     # a typed intent the application passed (not text)
    CONFIRMATION = "L4"   # a human confirmed this exact value


class Origin(enum.Enum):
    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"


# ── canonicalizers ────────────────────────────────────────────────────────────

_WS = re.compile(r"\s+")
_ZW = re.compile("[​-‏⁠﻿]")


def _text(v: Any) -> str:
    s = unicodedata.normalize("NFKC", str(v))
    return _ZW.sub("", s).strip()


def canon_iban(v: Any) -> str:
    return _WS.sub("", _text(v)).upper()


def canon_email(v: Any) -> str:
    return _WS.sub("", _text(v)).lower()


def canon_money(v: Any) -> str:
    if isinstance(v, bool):
        raise ValueError("a boolean is not an amount")
    s = _text(v).replace(",", "")
    s = re.sub(r"^[^\d\-.]+|[^\d.]+$", "", s)  # currency sign / code around the number
    try:
        d = Decimal(s)
    except InvalidOperation as exc:
        raise ValueError(f"not an amount: {v!r}") from exc
    if not d.is_finite():
        raise ValueError(f"not a finite amount: {v!r}")
    return str(d.quantize(Decimal("0.01")))


def canon_number(v: Any) -> str:
    if isinstance(v, bool):
        raise ValueError("a boolean is not a number")
    try:
        d = Decimal(_text(v))
    except InvalidOperation as exc:
        raise ValueError(f"not a number: {v!r}") from exc
    if not d.is_finite():
        raise ValueError(f"not a finite number: {v!r}")
    return format(d.normalize(), "f")


def canon_path(v: Any) -> str:
    s = _text(v).replace("\\", "/")
    return posixpath.normpath(s) if s else s


def canon_id(v: Any) -> str:
    return _text(v)


def canon_string(v: Any) -> str:
    return _WS.sub(" ", _text(v)).casefold()


CANONICALIZERS: dict[str, Callable[[Any], str]] = {
    "iban": canon_iban,
    "email": canon_email,
    "money": canon_money,
    "number": canon_number,
    "path": canon_path,
    "id": canon_id,
    "string": canon_string,
}

# Kinds a grant may constrain. Free text (a message body, a subject) is rich syntax:
# exact-match on it is meaningless authority and is refused at registration.
GRANTABLE_KINDS = frozenset(CANONICALIZERS)


def split_kind(kind: str) -> tuple[str, bool]:
    """``"email?"`` -> ("email", True): a trailing ``?`` marks an OPTIONAL grant arg —
    a call may omit it (absence cannot carry a value anywhere), but a value that IS
    passed must be covered like any other."""
    return (kind[:-1], True) if kind.endswith("?") else (kind, False)


def canonicalize(kind: str, value: Any) -> str:
    kind = split_kind(kind)[0]
    try:
        fn = CANONICALIZERS[kind]
    except KeyError:
        raise ValueError(f"unknown handle kind {kind!r}; known: {sorted(CANONICALIZERS)}") from None
    return fn(value)


# ── the table ─────────────────────────────────────────────────────────────────

HANDLE_PREFIX = "h:"


@dataclass(frozen=True)
class Handle:
    id: str
    kind: str
    canonical: str
    raw: str
    origin: Origin
    channel: TrustedChannel | None = None

    @property
    def trusted(self) -> bool:
        return self.origin is Origin.TRUSTED


def is_handle_ref(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(HANDLE_PREFIX)


class HandleTable:
    """Session-scoped interning of canonical values.

    Single-writer and lock-free (Ring 0 has no threading); the process that hosts
    the grant store serializes access to it."""

    def __init__(self, salt: str = "") -> None:
        self._by_id: dict[str, Handle] = {}
        # (kind, canonical) -> the trusted handle for it, if one was ever minted
        self._trusted: dict[tuple[str, str], Handle] = {}
        self._untrusted: dict[tuple[str, str], Handle] = {}
        self._salt = salt

    def _new_id(self, kind: str, canonical: str, origin: Origin) -> str:
        h = hashlib.sha256(f"{self._salt}|{origin.value}|{kind}|{canonical}".encode()).hexdigest()
        return f"{HANDLE_PREFIX}{h[:16]}"

    def mint_trusted(self, value: Any, kind: str, channel: TrustedChannel) -> Handle:
        if not isinstance(channel, TrustedChannel):
            raise TypeError("mint_trusted needs a TrustedChannel")
        c = canonicalize(kind, value)
        got = self._trusted.get((kind, c))
        if got is not None:
            return got
        h = Handle(self._new_id(kind, c, Origin.TRUSTED), kind, c, str(value),
                   Origin.TRUSTED, channel)
        self._trusted[(kind, c)] = h
        self._by_id[h.id] = h
        return h

    def mint_untrusted(self, value: Any, kind: str) -> Handle:
        """A value surfaced by a read. Resolves to the trusted handle on an exact
        canonical match; otherwise a fresh untrusted handle no grant can name."""
        c = canonicalize(kind, value)
        got = self._trusted.get((kind, c)) or self._untrusted.get((kind, c))
        if got is not None:
            return got
        h = Handle(self._new_id(kind, c, Origin.UNTRUSTED), kind, c, str(value),
                   Origin.UNTRUSTED)
        self._untrusted[(kind, c)] = h
        self._by_id[h.id] = h
        return h

    def get(self, handle_id: str) -> Handle | None:
        return self._by_id.get(handle_id)

    def trusted_for(self, kind: str, value: Any) -> Handle | None:
        try:
            c = canonicalize(kind, value)
        except ValueError:
            return None
        return self._trusted.get((kind, c))

    def resolve(self, value: Any, kind: str) -> str | None:
        """The canonical value an argument denotes: a handle ref resolves through the
        table (unknown ref -> None), a raw value is canonicalized."""
        if is_handle_ref(value):
            h = self.get(value)
            return h.canonical if h is not None and h.kind == kind else None
        try:
            return canonicalize(kind, value)
        except ValueError:
            return None

    def raw_value(self, handle_id: str) -> str | None:
        h = self.get(handle_id)
        return h.raw if h is not None else None

    def snapshot(self) -> tuple:
        return tuple(sorted((h.id, h.kind, h.canonical, h.origin.value) for h in self._by_id.values()))
