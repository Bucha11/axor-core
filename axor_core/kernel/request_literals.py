"""Lit(R): the typed literals a user's request fixes (Ring 0, source L1).

A deterministic, regex-only extractor. It reads the user's own text — never a tool
output, never model text — and returns the values the user literally wrote, each
tagged with the handle kinds it may stand for. These are the only values an L1 grant
can name; anything the request does not spell out (an amount computed from a file,
a recipient found in a transaction list) is outside L1 by construction and falls to
a template (L2), a structured request (L3) or a confirmation (L4).

T0: no model, no network. Over-extraction only widens L1 grants to values the user
wrote; it can never add a value the user did not write.

Typed kinds (iban, email, url, money, number, id, path) are recognised by their syntax.
The untyped kind ``string`` is taken ONLY from a fragment the user set apart —
quotes or backticks: an untyped word anywhere in the request ("can", a file name in
passing, a capitalised word) is not an authority-bearing literal.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class Literal:
    value: str
    kinds: frozenset[str]
    span: tuple[int, int]


_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]){10,30}\b")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_URL = re.compile(r"\bhttps?://[^\s'\"<>]+|\bwww\.[^\s'\"<>]+")
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_MONEY = re.compile(
    r"(?:[$€£¥]\s?)?(?<![\w.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w.])|"
    r"(?:[$€£¥]\s?)?(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])"
)
_FILE = re.compile(r"(?<![\w/.-])[\w./-]+\.[A-Za-z0-9]{1,6}\b")
_QUOTED = re.compile(
    r"'([^'\n]{1,200})'|\"([^\"\n]{1,200})\"|“([^”\n]{1,200})”|‘([^’\n]{1,200})’|`([^`\n]{1,200})`")
# marker kind: the literal was set apart by the user (quotes / backticks)
QUOTED = "quoted"

_NAME = re.compile(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b")


def extract(text: str) -> list[Literal]:
    """Every typed literal in ``text``, in order of appearance. Overlaps are kept:
    ``'bill-2023.txt'`` is both a quoted string and a path."""
    if not isinstance(text, str) or not text:
        return []
    t = unicodedata.normalize("NFKC", text)
    out: list[Literal] = []

    def add(m: re.Match, kinds: set[str], group: int = 0) -> None:
        v = m.group(group)
        if v:
            out.append(Literal(v, frozenset(kinds), m.span(group)))

    for m in _IBAN.finditer(t):
        add(m, {"iban", "id", "string"})
    for m in _EMAIL.finditer(t):
        add(m, {"email", "string"})
    for m in _URL.finditer(t):
        # a URL is typed by its syntax (scheme or www.); sentence punctuation after
        # it is not part of it
        v = m.group(0)
        trimmed = v.rstrip(".,;:!?)")
        if trimmed:
            out.append(Literal(trimmed, frozenset({"url", "string", "id"}), (m.start(), m.start() + len(trimmed))))
    for m in _DATE.finditer(t):
        add(m, {"string", "id"})
    taken = [lit.span for lit in out]
    for m in _MONEY.finditer(t):
        s, e = m.span()
        if any(a <= s < b or a < e <= b for a, b in taken):
            continue  # a digit run inside an IBAN / date / email is not an amount
        add(m, {"money", "number", "id"})
    for m in _FILE.finditer(t):
        if "@" in m.group(0) or _DATE.fullmatch(m.group(0)):
            continue
        if re.fullmatch(r"-?\d+(?:\.\d+)?", m.group(0)):
            continue
        add(m, {"path", "string", "id"})
    for m in _QUOTED.finditer(t):
        g = next(i for i in range(1, 6) if m.group(i) is not None)
        add(m, {"string", "path", "id", QUOTED}, g)
    for m in _NAME.finditer(t):
        add(m, {"string"})
    return out


def literal_values(text: str, kind: str) -> list[str]:
    """The literals of one kind, deduplicated, in order. ``string`` (untyped) comes
    only from quoted / backticked fragments."""
    seen: dict[str, None] = {}
    for lit in extract(text):
        if kind not in lit.kinds:
            continue
        if kind == "string" and QUOTED not in lit.kinds:
            continue
        seen.setdefault(lit.value, None)
    return list(seen)
