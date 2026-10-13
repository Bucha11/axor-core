"""Lit(R): the typed literals a user's request fixes (Ring 0, source L1).

A deterministic, regex-only extractor. It reads the user's own text — never a tool
output, never model text — and returns the values the user literally wrote, each
tagged with the handle kinds it may stand for. These are the only values an L1 grant
can name; anything the request does not spell out (an amount computed from a file,
a recipient found in a transaction list) is outside L1 by construction and falls to
a template (L2), a structured request (L3) or a confirmation (L4).

T0: no model, no network. Over-extraction only widens L1 grants to values the user
wrote; it can never add a value the user did not write.

Every literal is typed by its syntax and binds only to a grant argument of that kind
(the L1 type-matching rule): iban, email, url, date/datetime, money/number, id (bare
integers only), path, and ``string`` — taken ONLY from a fragment the user set apart
(quotes or backticks) that is not itself a typed literal. An untyped word anywhere in
the request ("can", a capitalised name) is not an authority-bearing literal.
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
_DATETIME = re.compile(r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}\b")
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_MONEY = re.compile(
    r"(?:[$€£¥]\s?)?(?<![\w.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w.])|"
    r"(?:[$€£¥]\s?)?(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])"
)
_INT = re.compile(r"\d+")
_TIME = re.compile(r"\b\d{1,2}:\d{2}\b")  # a clock time: consumed, no type of its own
_FILE = re.compile(r"(?<![\w/.-])[\w./-]+\.[A-Za-z0-9]{1,6}\b")
_QUOTED = re.compile(
    r"'([^'\n]{1,200})'|\"([^\"\n]{1,200})\"|“([^”\n]{1,200})”|‘([^’\n]{1,200})’|`([^`\n]{1,200})`")
_TYPED_WHOLE = (_IBAN, _EMAIL, _URL, _DATETIME, _DATE, _MONEY)


def extract(text: str) -> list[Literal]:
    """Every typed literal in ``text``, in order of appearance, each with the types
    its SYNTAX gives it (residuals_rules revision 3, the L1 type-matching rule):

    IBAN -> iban; e-mail -> email; URL -> url; ``YYYY-MM-DD`` -> date, datetime;
    ``YYYY-MM-DD HH:MM`` -> datetime; an amount with decimals / currency / separators ->
    money, number; a bare integer -> money, number, id; a file-name-like token -> path;
    a quoted / backticked fragment that is not itself a typed literal -> string.

    A grant argument takes a literal only of its own kind, so a quoted e-mail never
    fills a time slot and a date never fills an id."""
    if not isinstance(text, str) or not text:
        return []
    t = unicodedata.normalize("NFKC", text)
    out: list[Literal] = []

    def add(v: str, span: tuple[int, int], kinds: set[str]) -> None:
        if v:
            out.append(Literal(v, frozenset(kinds), span))

    for m in _IBAN.finditer(t):
        add(m.group(0), m.span(), {"iban"})
    for m in _EMAIL.finditer(t):
        add(m.group(0), m.span(), {"email"})
    for m in _URL.finditer(t):
        # sentence punctuation after a URL is not part of it
        v = m.group(0).rstrip(".,;:!?)")
        add(v, (m.start(), m.start() + len(v)), {"url"})
    for m in _DATETIME.finditer(t):
        add(m.group(0), m.span(), {"datetime"})
    for m in _DATE.finditer(t):
        add(m.group(0), m.span(), {"date", "datetime"})
    taken = [lit.span for lit in out] + [m.span() for m in _TIME.finditer(t)]
    for m in _MONEY.finditer(t):
        s, e = m.span()
        if any(a <= s < b or a < e <= b for a, b in taken):
            continue  # a digit run inside an IBAN / date / email / URL is not an amount
        kinds = {"money", "number"}
        if _INT.fullmatch(m.group(0)):
            kinds.add("id")  # only a bare integer is id-like
        add(m.group(0), m.span(), kinds)
    for m in _FILE.finditer(t):
        v = m.group(0)
        if "@" in v or _DATE.fullmatch(v) or re.fullmatch(r"-?\d+(?:\.\d+)?", v):
            continue
        if v.lower().startswith(("www.", "http")) or any(a <= m.start() < b for a, b in taken):
            continue  # part of a URL / e-mail, not a file name
        add(v, m.span(), {"path"})
    for m in _QUOTED.finditer(t):
        g = next(i for i in range(1, 6) if m.group(i) is not None)
        v = m.group(g).strip()
        if any(rx.fullmatch(v) for rx in _TYPED_WHOLE):
            continue  # a quoted typed literal keeps only its own type (found above)
        add(v, m.span(g), {"string"})
    return out


def literal_values(text: str, kind: str) -> list[str]:
    """The literals of one kind (by syntax, see :func:`extract`), deduplicated, in order."""
    seen: dict[str, None] = {}
    for lit in extract(text):
        if kind in lit.kinds:
            seen.setdefault(lit.value, None)
    return list(seen)
