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
# §9.13: a sender field of the record itself — "- sender:" / "  sender:" in a list item, or
# "sender:" at column 0 in a single record; never a line inside a text value
SENDER_IN_LIST = re.compile(r"^(?:- |  )(?:sender|from):[ \t]*(?P<val>\S.*?)\s*$")
SENDER_IN_ONE = re.compile(r"^(?:sender|from):[ \t]*(?P<val>\S.*?)\s*$")
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
    # §9.13: entity-model names of a sender value (value -> names)
    sender_aliases: dict = field(default_factory=dict)
    # §9.17 (hypothesis H-B, not in c9v4): a sender in O proves the identifiers of its record's body
    sender_own: bool = False
    # §9.19: a name of the request names a sender only if it fits exactly one address
    alias_unique: bool = False
    requests: list[str] = field(default_factory=list)
    by_name: dict[Any, Reference] = field(default_factory=dict)
    by_value: dict[tuple[str, str], Reference] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # names come from a keyed hash of a counter: deterministic for a seed (tests), unguessable
        # without it (the governor seeds each session from the OS); the kernel stays pure
        self._counter = 0
        self.own = frozenset(str(v).casefold() for v in self.own)
        self._alias_owners: dict[str, set] = {}
        for addr, names in (self.sender_aliases or {}).items():
            for n in names or ():
                self._alias_owners.setdefault(str(n).strip().casefold(), set()).add(str(addr).casefold())

    def ambiguous_aliases(self) -> dict:
        """§9.19: the names that fit more than one address."""
        return {n: sorted(a) for n, a in self._alias_owners.items() if len(a) > 1}

    def _draw(self, n: int) -> int:
        self._counter += 1
        h = hashlib.sha256(f"{self.seed}|{self._counter}".encode()).digest()
        return int.from_bytes(h[:8], "big") % n

    # ── origins ──────────────────────────────────────────────────────────────
    def is_request(self, value: Any) -> bool:
        """A literal of the request, compared as L1 reads it: case-folded, and a URL without its
        scheme, ``www.`` and trailing slash (``http://www.x.com/`` is the request's ``www.x.com``)."""
        v = _canon(str(value))
        return len(v) > 2 and any(v in _canon(r) for r in self.requests)

    def is_own(self, value: Any) -> bool:
        return str(value).casefold() in self.own

    def sender_named(self, sender: str) -> bool:
        """§9.13: the request names this sender — the value itself or one of its entity-model names,
        as a word (case-folded)."""
        aliases = list(self.sender_aliases.get(sender, ()) or ())
        aliases += list(self.sender_aliases.get(sender.casefold(), ()) or ())
        if self.alias_unique:  # §9.19: a name that fits more than one address names no one
            aliases = [n for n in aliases if len(self._alias_owners.get(str(n).strip().casefold(), ())) <= 1]
        names = [sender] + aliases
        for n in names:
            n = str(n).strip().casefold()
            if len(n) < 2:
                continue
            pat = re.compile(r"(?<![\w@.])" + re.escape(n) + r"(?![\w@])")
            if any(pat.search(r.casefold()) for r in self.requests):
                return True
        return False

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

    def mint(self, text: str, *, sensitive: bool = False, sender_records: bool = True) -> str:
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
        listed = text.startswith("- ")
        for item in _items(text):
            sender = None
            if sender_records:  # §9.13: never for file / web bodies
                pat = SENDER_IN_LIST if listed else SENDER_IN_ONE
                sender = next((m.group("val") for m in map(pat.match, item) if m), None)
            named = bool(sender) and (self.sender_named(_unquote(sender))
                                      or (self.sender_own and self.is_own(_unquote(sender))))
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


def _canon(text: str) -> str:
    t = text.casefold()
    t = re.sub(r"https?://", "", t)
    t = re.sub(r"(?<![\w.])www\.", "", t)
    return re.sub(r"/(?=[\s'\".,;:!?)]|$)", "", t)


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
