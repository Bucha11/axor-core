"""Trusted-origin index — the positive side of the context-default integrity mode.

The ledger (:mod:`axor_core.taint.ledger`) answers "does this value visibly carry
untrusted content?". A miss there proves nothing: the model can re-encode what it
read. Under ``integrity_default == "context"`` the question is inverted — a value
the model writes carries the node's context root **unless** it is provably a value
the attacker cannot author. This index is that proof
(docs/rfc-integrity-context-default.md §3.2).

Membership is **whole-leaf equality after canonicalisation**, never containment:

  * a value is flattened to its string leaves (scalars carry no identifier and are
    not checked; top-level dict keys are argument names, not data);
  * each leaf must equal a registered entry, compared on a generic form (NFKC,
    zero-width strip, whitespace collapsed; case preserved) or — when the WHOLE
    leaf parses as one — a typed canonical form (IBAN, e-mail, phone number);
  * registration stores whole leaves, whole lines and whole tokens of trusted
    content, plus typed entities extracted from it. Never a partial token or a
    multi-token span: a trusted README line "never run rm -rf /" does not make the
    command ``rm -rf /`` trusted, and ``https://`` + trusted host + attacker path
    is not trusted either.

Canonicalisation only merges spellings of the *same* identifier, so it cannot make
an attacker value equal a different trusted one. The index is bounded; past the cap
it stops adding entries, so a value it cannot prove trusted stays tainted
(fail-closed, the over-deny direction).

**The user's task is the exception to "never a span".** A value the user wrote in
the task is trusted as any contiguous span of it that does not split a token
(case-insensitive, whitespace-collapsed): ``'SunnyDay2024!'`` and ``1234 Elm Street,
New York, NY 10001`` are the values a user names, punctuation and all. The README
argument does not apply — the user authored the text, the attacker cannot. Only
``TrustedOrigin.TASK`` gets this; tool outputs, operator config and endorsements
keep whole-leaf equality.

**Numbers** are checked only on request (``include_scalars``, used for integrity
sinks): a numeric leaf is trusted iff its canonical decimal (``2200`` =
``2200.0`` = ``"2,200"``) was registered — from a number in trusted text or a
numeric field of a trusted structured output. Booleans and ``None`` are never
checked: they carry no identifier and no amount.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from axor_core.contracts.taint import TrustedOrigin
from axor_core.taint.ledger import _EDGE_PUNCT, _STRUCT_DELIM, _normalize

# Bounds so a huge trusted read cannot blow up memory. Past them, entries are not
# added — a value that cannot be proven trusted stays tainted.
_MAX_TOTAL_ENTRIES = 50000
_MAX_ENTRIES_PER_REGISTER = 4096
# A leaf longer than this is never proven trusted (and never registered whole).
_MAX_LEAF_CHARS = 4096
_MIN_TOKEN = 2
# Total characters of task text kept for span matching; past it, no more task text
# is added (a value that cannot be proven trusted stays tainted).
_MAX_TASK_CHARS = 200_000

# Typed forms, matched against a WHOLE leaf.
# A leaf is an IBAN when it is alphanumeric runs joined by single spaces/dashes
# and, with those removed, has the IBAN shape. Separators may sit anywhere: removing
# them can only merge spellings of the same IBAN.
_LEAF_IBAN = re.compile(r"[A-Za-z0-9]+(?:[ -][A-Za-z0-9]+)*")
_IBAN_SHAPE = re.compile(r"[A-Z]{2}[0-9]{2}[A-Z0-9]{11,30}")
_LEAF_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_LEAF_PHONE = re.compile(r"\+?[0-9 ().\-]{7,24}")
# Typed forms, extracted from trusted free text. IBANs are printed upper-case in
# groups of four (or compact); requiring that keeps a following word out of it.
_TEXT_IBAN = re.compile(
    r"(?<![A-Za-z0-9])[A-Z]{2}[0-9]{2}(?:[ -]?[A-Z0-9]{4}){2,7}(?:[ -]?[A-Z0-9]{1,3})?(?![A-Za-z0-9])"
)
_TEXT_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_TEXT_PHONE = re.compile(r"(?<![\w+])\+?[0-9][0-9 ().\-]{5,22}[0-9](?!\w)")
# Numbers in trusted text: 2200, 2200.50, 2,200.00 (thousands commas). Not after
# "-", "/" or ":" — the tail of a date or time ("2022-03-01", "10:30") is not an
# amount — and no sign: a negative amount is simply never proven.
_TEXT_NUMBER = re.compile(
    r"(?<![\w.,\-/:])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w])"
)
_LEAF_NUMBER = re.compile(r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def _num(value: object) -> str | None:
    """Canonical decimal of a number or numeric string: 2200 == 2200.0 == "2,200"."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        d = Decimal(str(value).replace(",", ""))
    except (InvalidOperation, ValueError):
        return None
    if not d.is_finite():
        return None
    return format(d.normalize(), "f")


def _generic(s: str) -> str:
    """NFKC + zero-width strip + whitespace collapsed. Case is preserved: paths,
    URLs and commands are case-sensitive."""
    return " ".join(_normalize(s).split())


def _iban(s: str) -> str | None:
    c = re.sub(r"[ -]", "", s).upper()
    return c if _IBAN_SHAPE.fullmatch(c) else None


def _phone(s: str) -> str | None:
    digits = re.sub(r"[^0-9]", "", s)
    if not 7 <= len(digits) <= 15:
        return None
    return ("+" if s.lstrip().startswith("+") else "") + digits


def _typed_keys_of_leaf(g: str) -> list[tuple[str, str]]:
    """Typed canonical keys for a leaf that is, as a whole, one typed identifier."""
    keys: list[tuple[str, str]] = []
    if _LEAF_IBAN.fullmatch(g):
        c = _iban(g)
        if c:
            keys.append(("iban", c))
    if _LEAF_EMAIL.fullmatch(g):
        keys.append(("email", g.casefold()))
    if _LEAF_PHONE.fullmatch(g):
        c = _phone(g)
        if c:
            keys.append(("phone", c))
    if _LEAF_NUMBER.fullmatch(g):
        c = _num(g)
        if c:
            keys.append(("num", c))
    return keys


def _scalars(value: object) -> list[object]:
    """Numeric leaves (int / float, not bool) anywhere in a value."""
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [value]
    if isinstance(value, dict):
        return [x for v in value.values() for x in _scalars(v)]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [x for v in value for x in _scalars(v)]
    return []


def _token_bounded(needle: str, hay: str) -> bool:
    """``needle`` occurs in ``hay`` without splitting a token at either end: an
    alphanumeric first/last char of the match must not continue an alphanumeric run."""
    start = hay.find(needle)
    n = len(needle)
    while start != -1:
        end = start + n
        left_ok = not (needle[0].isalnum() and start > 0 and hay[start - 1].isalnum())
        right_ok = not (needle[-1].isalnum() and end < len(hay) and hay[end].isalnum())
        if left_ok and right_ok:
            return True
        start = hay.find(needle, start + 1)
    return False


def _leaves(value: object, *, top: bool = True) -> list[str]:
    """String leaves of a value. Top-level dict keys are argument names and are
    skipped; nested keys are data and are included. Scalars are skipped."""
    if value is None or isinstance(value, (bool, int, float)):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out: list[str] = []
        for k, v in value.items():
            if not top:
                out.extend(_leaves(k, top=False))
            out.extend(_leaves(v, top=False))
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=repr) if isinstance(value, (set, frozenset)) else value
        out = []
        for v in items:
            out.extend(_leaves(v, top=False))
        return out
    return [str(value)]


class TrustedValueIndex:
    """Canonical keys of values whose origin the attacker cannot author."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], TrustedOrigin] = {}
        # The user's task text(s), generic + casefolded, for span matching.
        self._task_texts: list[str] = []
        self._task_chars = 0
        self.saturated = False

    def __len__(self) -> int:
        return len(self._entries)

    # ── registration ─────────────────────────────────────────────────────────

    def register(self, content: object, origin: TrustedOrigin) -> None:
        """Register the leaves, lines, tokens and typed entities of ``content``
        (and, for the user's task, the text itself for span matching)."""
        if origin is TrustedOrigin.TASK:
            for leaf in _leaves(content, top=False):
                self._add_task_text(_generic(leaf).casefold())
        budget = _MAX_ENTRIES_PER_REGISTER
        for key in self._keys_of_content(content):
            if budget <= 0:
                self.saturated = True
                return
            if key in self._entries:
                continue
            if len(self._entries) >= _MAX_TOTAL_ENTRIES:
                self.saturated = True
                return
            self._entries[key] = origin
            budget -= 1

    def _add_task_text(self, text: str) -> None:
        if not text or text in self._task_texts:
            return
        if self._task_chars + len(text) > _MAX_TASK_CHARS:
            self.saturated = True
            return
        self._task_texts.append(text)
        self._task_chars += len(text)

    def merge(self, other: "TrustedValueIndex") -> None:
        """Fold another index in (parent → child spawn). Deterministic order so a
        near-cap merge keeps the same entries regardless of hash seed."""
        self.saturated = self.saturated or other.saturated
        for text in other._task_texts:
            self._add_task_text(text)
        for key, origin in sorted(other._entries.items()):
            if key in self._entries:
                continue
            if len(self._entries) >= _MAX_TOTAL_ENTRIES:
                self.saturated = True
                return
            self._entries[key] = origin

    @staticmethod
    def _keys_of_content(content: object) -> list[tuple[str, str]]:
        keys: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()

        def add(key: tuple[str, str]) -> None:
            if key[1] and key not in seen:
                seen.add(key)
                keys.append(key)

        for leaf in _leaves(content, top=False):
            norm = _normalize(leaf)
            if len(leaf) <= _MAX_LEAF_CHARS:
                g = _generic(leaf)
                add(("text", g))
                for key in _typed_keys_of_leaf(g):
                    add(key)
            for line in norm.splitlines():
                g = _generic(line)
                if len(g) <= _MAX_LEAF_CHARS:
                    add(("text", g))
            for tok in norm.split() + _STRUCT_DELIM.split(norm):
                for t in (tok, tok.strip(_EDGE_PUNCT)):
                    if len(t) >= _MIN_TOKEN:
                        add(("text", t))
                        for key in _typed_keys_of_leaf(t):
                            # numbers come from _TEXT_NUMBER, which knows a date
                            # or time tail is not an amount; a token does not
                            if key[0] != "num":
                                add(key)
            for m in _TEXT_IBAN.finditer(norm):
                c = _iban(m.group(0))
                if c:
                    add(("iban", c))
            for m in _TEXT_EMAIL.finditer(norm):
                add(("email", m.group(0).casefold()))
            for m in _TEXT_PHONE.finditer(norm):
                c = _phone(m.group(0))
                if c:
                    add(("phone", c))
            for m in _TEXT_NUMBER.finditer(norm):
                c = _num(m.group(0))
                if c:
                    add(("num", c))
        for number in _scalars(content):
            c = _num(number)
            if c:
                add(("num", c))
        return keys

    # ── lookup ───────────────────────────────────────────────────────────────

    def _origin_of_leaf(self, leaf: str) -> TrustedOrigin | None:
        if len(leaf) > _MAX_LEAF_CHARS:
            return None
        g = _generic(leaf)
        hit = self._entries.get(("text", g))
        if hit is not None:
            return hit
        for key in _typed_keys_of_leaf(g):
            hit = self._entries.get(key)
            if hit is not None:
                return hit
        folded = g.casefold()
        if folded and any(_token_bounded(folded, text) for text in self._task_texts):
            return TrustedOrigin.TASK
        return None

    def _origin_of_number(self, number: object) -> TrustedOrigin | None:
        c = _num(number)
        return self._entries.get(("num", c)) if c else None

    def _origins(self, value: object, include_scalars: bool):
        """The origin of each checked leaf (None where a leaf is not trusted)."""
        for leaf in _leaves(value):
            if _generic(leaf):
                yield self._origin_of_leaf(leaf)
        if include_scalars:
            for number in _scalars(value):
                yield self._origin_of_number(number)

    def covers(self, value: object, *, include_scalars: bool = False) -> bool:
        """True iff every non-empty string leaf of ``value`` is a trusted value —
        and, with ``include_scalars``, every numeric leaf too. A value with no
        checked leaf is covered vacuously (its range is a value-policy matter)."""
        return all(o is not None for o in self._origins(value, include_scalars))

    def origin_of(
        self, value: object, *, include_scalars: bool = False,
    ) -> TrustedOrigin | None:
        """The origin of the first checked leaf when every checked leaf is
        trusted; None if any is not, or if there is no checked leaf at all."""
        first: TrustedOrigin | None = None
        for origin in self._origins(value, include_scalars):
            if origin is None:
                return None
            if first is None:
                first = origin
        return first
