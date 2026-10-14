"""Content projection (MODEL.md §8.4): addresses extracted from free text by syntax only —
URL, e-mail, IBAN — in a canonical form per kind. No model, no meaning: a token is what the
grammar below matches, nothing else."""
from __future__ import annotations

import re

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
# a URL: a scheme, or www., or a dotted host ending in an alphabetic TLD; an optional path
_URL = re.compile(
    r"(?:(?:https?://)|(?:www\.))[^\s<>\"')]+"
    r"|\b(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}(?:/[^\s<>\"')]*)?")
_TRAIL = ".,;:!?)'\"]}>"


def canonical(kind: str, token: str) -> str:
    t = token.strip().rstrip(_TRAIL)
    if kind == "email":
        return t.lower()
    if kind == "iban":
        return t.replace(" ", "").upper()
    t = re.sub(r"^https?://", "", t, flags=re.I)
    t = re.sub(r"^www\.", "", t, flags=re.I)
    host, _, path = t.partition("/")
    return (host.lower() + ("/" + path if path else "")).rstrip("/")


def extract(text: object) -> set[tuple[str, str]]:
    """{(kind, canonical)} for every URL / e-mail / IBAN in ``text`` (strings in lists and
    dicts included)."""
    out: set[tuple[str, str]] = set()
    if isinstance(text, dict):
        for v in text.values():
            out |= extract(v)
        return out
    if isinstance(text, (list, tuple, set, frozenset)):
        for v in text:
            out |= extract(v)
        return out
    if not isinstance(text, str) or not text:
        return out
    emails = list(_EMAIL.finditer(text))
    for m in emails:
        out.add(("email", canonical("email", m.group(0))))
    masked = text
    for m in emails:  # an e-mail's domain is not a URL
        masked = masked[:m.start()] + " " * (m.end() - m.start()) + masked[m.end():]
    for m in _IBAN.finditer(masked):
        out.add(("iban", canonical("iban", m.group(0))))
    for m in _URL.finditer(masked):
        c = canonical("url", m.group(0))
        if c and "." in c.split("/")[0]:
            out.add(("url", c))
    return out
