"""References (MODEL.md §9): target identifiers of untrusted origin are shown to the agent only as
references; the agent passes the reference, the kernel resolves it and decides on its origin.

A reference is (kind, value, origin). Kinds: email, iban, url (in any string of a tool output) and
id (the value of an id field). The agent sees ``<kind>_from_untrusted_read_hidden_<4 hex>``, or, for
an id field whose value is an integer, a unique integer in [10 000 000, 99 999 999]. Values the
request names (origin R) and own entities (origin own) are shown raw. The table is per session and
never leaves the kernel / daemon.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
URL = re.compile(r"(?:https?://|www\.)[^\s<>\"'),]+")
NAME = re.compile(r"\b(?:email|iban|url|id|secret)_from_untrusted_read_hidden_[0-9a-f]{4}\b")
ID_KEYS = ("id", "id_", "file_id", "event_id", "email_id")
ID_LINE = re.compile(r"^(?P<lead>\s*-?\s*)(?P<key>" + "|".join(ID_KEYS) + r"):[ \t]*(?P<val>\S.*?)\s*$")
FIELD_LINE = re.compile(r"^\s*-?\s*(?P<key>[A-Za-z_][\w]*):[ \t]*(?P<val>\S.*?)\s*$")
SENDER_LINE = re.compile(r"^\s*-?\s*(?:sender|from):[ \t]*(?P<val>\S.*?)\s*$")
INT_RANGE = (10_000_000, 100_000_000)
_TRAIL = ".;:!?"

R, OWN, SENDER, SECRET, UNTRUSTED = "R", "own", "named_sender", "secret", "untrusted"
HIDDEN = (SENDER, SECRET, UNTRUSTED)


@dataclass
class Reference:
    name: Any  # str name, or an int handle
    kind: str
    value: Any
    origin: str


@dataclass
class ReferenceTable:
    """One session's references. ``requests``: the user's request texts (origin R); ``own``: the
    entity model's own values (origin own)."""

    own: frozenset[str] = frozenset()
    seed: int | None = None
    # §9.11: fields whose value is a secret (a credential, a document / payment number)
    secret_fields: frozenset[str] = frozenset()
    requests: list[str] = field(default_factory=list)
    by_name: dict[Any, Reference] = field(default_factory=dict)
    by_value: dict[tuple[str, str], Reference] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # names come from a keyed hash of a counter: deterministic for a seed (tests), unguessable
        # without it (the governor seeds each session from the OS); the kernel stays pure
        self._counter = 0
        self.own = frozenset(str(v).casefold() for v in self.own)

    def _draw(self, n: int) -> int:
        self._counter += 1
        h = hashlib.sha256(f"{self.seed}|{self._counter}".encode()).digest()
        return int.from_bytes(h[:8], "big") % n

    # ── origins ──────────────────────────────────────────────────────────────
    def is_request(self, value: Any) -> bool:
        v = str(value).casefold()
        return len(v) > 2 and any(v in r.casefold() for r in self.requests)

    def is_own(self, value: Any) -> bool:
        return str(value).casefold() in self.own

    def shown_raw(self, value: Any) -> bool:
        return self.is_request(value) or self.is_own(value)

    def hidden_values(self) -> set[str]:
        return {str(r.value).casefold() for r in self.by_name.values()}

    # ── minting (tool output → agent view) ───────────────────────────────────
    def _ref(self, kind: str, value: Any, origin: str, integer: bool) -> Any:
        key = (kind, str(value))
        if key in self.by_value:  # the first origin wins for the session
            return self.by_value[key].name
        while True:
            if integer:
                name: Any = INT_RANGE[0] + self._draw(INT_RANGE[1] - INT_RANGE[0])
            else:
                name = f"{kind}_from_untrusted_read_hidden_{self._draw(16 ** 4):04x}"
            if name not in self.by_name:
                break
        ref = Reference(name, kind, value, origin)
        self.by_name[name] = ref
        self.by_value[key] = ref
        return name

    def _origin(self, item_named_sender: bool, sensitive: bool) -> str:
        # §9.11: a sensitive source no longer makes an identifier secret (secret = declared fields)
        return SENDER if item_named_sender else UNTRUSTED

    def _mint_text(self, s: str, origin: str) -> str:
        def sub(kind: str):
            def f(m: re.Match) -> str:
                raw = m.group(0)
                v = raw.rstrip(_TRAIL) if kind == "url" else raw
                if self.shown_raw(v):
                    return raw
                return str(self._ref(kind, v, origin, False)) + raw[len(v):]
            return f
        s = EMAIL.sub(sub("email"), s)
        s = IBAN.sub(sub("iban"), s)
        return URL.sub(sub("url"), s)

    def mint(self, text: str, *, sensitive: bool = False) -> str:
        """The agent's view of a tool output (a YAML / plain text string)."""
        if not isinstance(text, str) or not text:
            return text
        # §9.11: secret fields first — their values become secret references wherever they appear
        secrets: dict[str, Any] = {}
        if self.secret_fields:
            for line in text.split("\n"):
                m = FIELD_LINE.match(line)
                if m and m.group("key") in self.secret_fields:
                    v = _unquote(m.group("val"))
                    if v and not self.is_request(v):
                        secrets[v] = self._ref("secret", v, SECRET, False)
            for v in sorted(secrets, key=len, reverse=True):
                text = text.replace(v, str(secrets[v]))
        out = []
        for item in _items(text):
            sender = next((m.group("val") for m in map(SENDER_LINE.match, item) if m), None)
            named = bool(sender) and self.is_request(_unquote(sender))
            origin = self._origin(named, sensitive)
            for line in item:
                m = ID_LINE.match(line)
                if m:
                    raw = m.group("val")
                    v = _unquote(raw)
                    if not self.shown_raw(v):
                        integer = raw.isdigit()
                        name = self._ref("id", int(v) if integer else v, origin, integer)
                        line = f"{m.group('lead')}{m.group('key')}: {name}"
                    out.append(line)
                    continue
                out.append(self._mint_text(line, origin))
        return "\n".join(out)

    # ── resolution (agent call → values, with origins) ───────────────────────
    def resolve(self, args: Any, id_arg: str | None = None) -> tuple[Any, list[Reference], list[Any]]:
        """(resolved value, references used, unknown reference-shaped tokens)."""
        used: list[Reference] = []
        unknown: list[Any] = []

        def go(x: Any, key: str | None) -> Any:
            if isinstance(x, dict):
                return {k: go(v, k) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [go(v, key) for v in x]
            if isinstance(x, bool) or x is None:
                return x
            if isinstance(x, int):
                if key in ID_KEYS and INT_RANGE[0] <= x < INT_RANGE[1]:
                    r = self.by_name.get(x)
                    if r is None:
                        unknown.append(x)
                        return x
                    used.append(r)
                    return r.value
                return x
            if isinstance(x, str):
                if x in self.by_name:
                    used.append(self.by_name[x])
                    return self.by_name[x].value

                def f(m: re.Match) -> str:
                    r = self.by_name.get(m.group(0))
                    if r is None:
                        unknown.append(m.group(0))
                        return m.group(0)
                    used.append(r)
                    return str(r.value)
                return NAME.sub(f, x)
            return x
        return go(args, id_arg), used, unknown


def _unquote(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
        return v[1:-1]
    return v


def _items(text: str) -> Iterable[list[str]]:
    """A YAML list output split into its items (a line starting with '- ' at column 0 opens one);
    any other text is one item."""
    lines = text.split("\n")
    items: list[list[str]] = []
    for ln in lines:
        if ln.startswith("- ") or not items:
            items.append([ln])
        else:
            items[-1].append(ln)
    return items
