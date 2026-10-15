"""Description grants (MODEL.md §9.15): a target the request describes rather than names.

A description is an expression over the STRUCTURED output of a declared trusted read (fields the
platform sets — a list element, a record field, a number), never text:

* ``select(entity, name <op> literal)`` — "the channel starting with External";
* ``argmax / argmin(entity, field)`` — "the channel with the fewest messages", "the largest file";
* ``members(entity, name = literal)`` — "the same participants as 'Introductory meeting'";
* ``email-of(name)`` — "her email address" (the nearest preceding name that is one contact).

The phrase templates are one fixed library (not per task). The kernel computes the value from the
session's structured outputs; a tie, several matches or a missing field gives no value. The caller
(the governor) grants exactly the computed value for the entity's declared targets.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

# ── the grammar (MODEL.md §9.15a (c), complete) ─────────────────────────────
NOUNS: dict[str, str] = {
    r"channels?": "channel",
    r"users?|members?": "user",
    r"contacts?": "contact",
    r"events?|meetings?": "event",
    r"files?|documents?": "file",
    r"hotels?": "hotel",
    r"restaurants?": "restaurant",
    r"(?:car\s+)?rental\s+compan(?:y|ies)|car\s+rentals?": "car_rental",
}
FIELDS: dict[str, str] = {
    r"messages?|message\s+counts?|posts": "messages",
    r"users|members|people|participants": "users",
    r"ratings?|rated|score": "rating",
    r"prices?|costs?": "price",
    r"size|space|storage": "size",
}
MAX_WORDS = r"most|highest|largest|biggest|greatest|best|top|maximum"
MIN_WORDS = r"fewest|least|lowest|smallest|minimum"
SUP_ADJ: dict[str, tuple[str, str]] = {
    r"largest|biggest": ("argmax", "size"),
    r"smallest": ("argmin", "size"),
    r"cheapest": ("argmin", "price"),
    r"most\s+expensive|priciest": ("argmax", "price"),
    r"quietest|least\s+active": ("argmin", "messages"),
    r"most\s+active|busiest": ("argmax", "messages"),
    r"(?:best|highest|top)(?:-|\s+)rated": ("argmax", "rating"),
}
DET = r"(?:a|an|the|that|this|which)"
_GRAMMAR_WORDS = {"a", "an", "the", "that", "this", "which", "same", "most", "least", "new"}


def _noun_rx(name: str = "noun") -> str:
    return f"(?P<{name}>" + "|".join(f"(?:{k})" for k in sorted(NOUNS, key=len, reverse=True)) + r")\b"


_NOUN = _noun_rx()
_FIELD = "(?P<field>" + "|".join(f"(?:{k})" for k in FIELDS) + r")\b"
_LIT = r"(?P<lit>'[^']+'|\"[^\"]+\"|#[\w.-]+|[\w.@-]+)"
_W = r"(?:\S+\s+)"
_ADJ = "(?P<adj>" + "|".join(f"(?:{k})" for k in SUP_ADJ) + ")"

G1 = re.compile(r"\b" + _NOUN + r"(?:\s+\S+){0,4}?\s+(?:with|that\s+has|that\s+have|which\s+has|which\s+have|"
                r"having|has|have)\s+(?:the\s+)?(?P<sup>" + MAX_WORDS + "|" + MIN_WORDS
                + r")\s+(?:number\s+of\s+|total\s+|amount\s+of\s+|count\s+of\s+)?" + _FIELD, re.I)
G2 = re.compile(r"\b(?:" + DET + r"\s+)?" + _ADJ + r"\s+" + _W + r"{0,2}?" + _NOUN, re.I)
G3 = re.compile(r"\b" + _NOUN + r"(?:\s+\S+){0,4}?\s+(?:rated|ranked)\s+(?:the\s+)?(?:best|highest|top)\b", re.I)
_SEL = (r"(?P<op>named|called|titled|(?:(?:whose\s+name|that|which)\s+)?(?:starts|starting|begins|beginning)"
        r"\s+with|(?:(?:whose\s+name|that|which)\s+)?(?:contains|containing))")
G4 = re.compile(r"\b" + _NOUN + r"\s+" + _SEL + r"\s+" + _LIT, re.I)
G4_QUOTED = re.compile(r"(?P<lit>'[^']+'|\"[^\"]+\")\s+" + _NOUN, re.I)
G4_HASH = re.compile(r"(?<![\w&])#(?P<lit>[A-Za-z][\w.-]*)")
G4_WORD = re.compile(r"\bthe\s+(?P<lit>[A-Za-z][\w-]*)\s+" + _NOUN, re.I)
_NAME = r"[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*){0,2}"
G5_OF = re.compile(r"\b(?:e-?mail)(?:\s+address)?\s+(?:of|for|belonging\s+to)\s+(?P<name>" + _NAME + ")")
G5_POSS = re.compile(r"(?P<name>" + _NAME + r")'s\s+e-?mail\b")
G5_PRON = re.compile(r"\b(?:his|her|their)\s+e-?mail\b", re.I)
_TAIL = r"\s+(?:the\s+)?(?:" + _NOUN + r"\s+)?" + _LIT + r"(?:\s+" + _noun_rx("noun2") + ")?"
G6 = re.compile(r"\b(?:all\s+users|all\s+members|all\s+participants|all\s+attendees|members|participants|users|"
                r"attendees|invitees|everyone|everybody|people)(?:\s+who\s+(?:was|were|is|are))?\s+(?:of|in|from|at)"
                + _TAIL, re.I)
G6_WHO = re.compile(r"\b(?:everyone|everybody|those|the\s+people)\s+who\s+(?:attended|joined|was\s+in|were\s+in)"
                    + _TAIL, re.I)
G6_SAME = re.compile(r"\bsame\s+(?:participants|members|users|attendees|people)\s+as" + _TAIL, re.I)
_CAPS = re.compile(_NAME)


def _entity(noun: str) -> str:
    for k, v in NOUNS.items():
        if re.fullmatch(k, noun, re.I):
            return v
    return ""


def _field(word: str) -> str:
    for k, v in FIELDS.items():
        if re.fullmatch(k, word, re.I):
            return v
    return ""


def _unq(lit: str) -> str:
    lit = lit.strip()
    if len(lit) >= 2 and lit[0] == lit[-1] and lit[0] in "'\"":
        return lit[1:-1]
    return lit.lstrip("#").rstrip(".,;:!?")


def _c(x: Any) -> str:
    return str(x).strip().casefold()


@dataclass(frozen=True)
class Description:
    op: str  # argmax | argmin | select | members | email_of
    entity: str
    field: str = ""
    mode: str = ""  # select: eq | starts | contains | word
    lit: str = ""
    names: tuple = ()  # email_of: candidate names, nearest first
    phrase: str = ""
    span: tuple = (0, 0)


# ── (a) whole-phrase coverage ───────────────────────────────────────────────
_RESTRICT = re.compile(r"\b(?:under|below|less\s+than|more\s+than|at\s+most|at\s+least|up\s+to|no\s+more\s+than|"
                       r"within|budget|budget-friendly|affordable|cheap|cheaper|break\s+the\s+bank|per\s+person|"
                       r"per\s+day|per\s+night|euros?|eur|dollars?|usd)\b|[€$]", re.I)
_FILTER = re.compile(r"\b(?:options|serves|serve|serving|offers|offering|open|opens|available|new|different|"
                     r"another|other\s+than|except|without|vegan|vegetarian|gluten-free|cuisine|food|type\s+of|"
                     r"kind\s+of)\b", re.I)
_MONTH = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_NOT_A_CONSTRAINT = re.compile(
    r"\d{4}-\d{2}-\d{2}|\b\d{1,2}:\d{2}\b|\b\d{1,2}\s*(?:am|pm)\b|\b\d+(?:st|nd|rd|th)\b|"
    r"\b" + _MONTH + r"\s+\d{1,2}\b|\b\d{1,2}\s+" + _MONTH + r"\b|"
    r"\b\d+(?:\.\d+)?(?:\s+|-)(?:days?|nights?|hours?|weeks?|people|persons?|meals?|times?)\b|'[^']*'|\"[^\"]*\"", re.I)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def _sentences(request: str) -> list[tuple[int, int]]:
    out, start = [], 0
    for m in re.finditer(r"[.?!](?=\s|$)|\n", request):
        out.append((start, m.end()))
        start = m.end()
    out.append((start, len(request)))
    return [s for s in out if request[s[0]:s[1]].strip()]


def _clause_span(request: str, span: tuple) -> tuple[int, int]:
    from axor_core.kernel.grants import _CLAUSE_END
    start, end = 0, len(request)
    for m in _CLAUSE_END.finditer(request):
        if m.end() <= span[0]:
            start = m.end()
        elif m.start() >= span[1]:
            end = m.start()
            break
    return start, end


def _unparsed(text: str, own: tuple, argmax: bool, entity: str) -> str | None:
    """The first unparsed constraint of a scope text (the description's own phrase blanked out)."""
    a, b = own
    t = text[:a] + " " * (b - a) + text[b:] if b > a else text
    for rx in (_RESTRICT, _FILTER):
        m = rx.search(t)
        if m:
            return m.group(0)
    blank = _NOT_A_CONSTRAINT.sub(lambda m: " " * len(m.group(0)), t)
    m = _NUMBER.search(blank)
    if m:
        return m.group(0)
    if argmax:  # a filter adjective: a capitalized word right before a noun form, not sentence-initial
        noun = "|".join(k for k, v in NOUNS.items() if v == entity)
        for m in re.finditer(r"(?<=\S)(\s+)([A-Z][\w-]*)\s+(?:" + noun + r")\b", t):
            if m.group(2) not in ("I",):
                return m.group(2)
    return None


def covered(request: str, d: Description) -> str | None:
    """(a): None if the grammar covers the whole phrase; else the unparsed constraint found."""
    if d.op in ("argmax", "argmin"):
        noun = "|".join(k for k, v in NOUNS.items() if v == d.entity)
        for s, e in _sentences(request):
            if not (s <= d.span[0] < e) and not re.search(r"\b(?:" + noun + r")\b", request[s:e], re.I):
                continue
            own = (max(d.span[0] - s, 0), max(min(d.span[1], e) - s, 0)) if s <= d.span[0] < e else (0, 0)
            bad = _unparsed(request[s:e], own, True, d.entity)
            if bad:
                return bad
        return None
    s, e = _clause_span(request, d.span)
    return _unparsed(request[s:e], (d.span[0] - s, d.span[1] - s), False, d.entity)


def parse(request: str) -> list[Description]:
    """The descriptions a request carries, by the fixed grammar."""
    out: list[Description] = []
    nouns = set(NOUNS.values())

    def add(op, ent, m, **kw):
        out.append(Description(op, ent, phrase=m.group(0), span=(m.start(), m.end()), **kw))

    for m in G1.finditer(request):
        e, f = _entity(m.group("noun")), _field(m.group("field"))
        if e and f:
            add("argmax" if re.fullmatch(MAX_WORDS, m.group("sup"), re.I) else "argmin", e, m, field=f)
    for m in G2.finditer(request):
        a = re.sub(r"\s+", " ", m.group("adj"))
        op, f = next(v for k, v in SUP_ADJ.items() if re.fullmatch(k, a, re.I))
        add(op, _entity(m.group("noun")), m, field=f)
    for m in G3.finditer(request):
        add("argmax", _entity(m.group("noun")), m, field="rating")
    for m in G4.finditer(request):
        lit = _unq(m.group("lit"))
        op = m.group("op").casefold()
        mode = "eq" if op in ("named", "called", "titled") else "contains" if "contain" in op else "starts"
        if lit and _entity(lit) not in nouns and lit.casefold() not in _GRAMMAR_WORDS:
            add("select", _entity(m.group("noun")), m, mode=mode, lit=lit)
    for m in G4_QUOTED.finditer(request):
        add("select", _entity(m.group("noun")), m, mode="eq", lit=_unq(m.group("lit")))
    for m in G4_HASH.finditer(request):
        add("select", "channel", m, mode="eq", lit=m.group("lit").rstrip(".,;:!?"))
    for m in G4_WORD.finditer(request):
        w = m.group("lit")
        if (w.casefold() in _GRAMMAR_WORDS or _entity(w) or _field(w)
                or any(re.fullmatch(k, w, re.I) for k in SUP_ADJ)
                or re.fullmatch(MAX_WORDS + "|" + MIN_WORDS, w, re.I)):
            continue
        add("select", _entity(m.group("noun")), m, mode="word", lit=w)
    for rx in (G5_OF, G5_POSS):
        for m in rx.finditer(request):
            add("email_of", "contact", m, names=(m.group("name"),))
    for m in G5_PRON.finditer(request):
        cands = []
        for c in _CAPS.finditer(request[:m.start()]):
            # every capitalized word sequence and its sub-sequences, nearest first
            ws = c.group(0).split()
            for i in range(len(ws)):
                for j in range(len(ws), i, -1):
                    cands.append((c.end(), " ".join(ws[i:j])))
        cands.sort(key=lambda t: -t[0])
        add("email_of", "contact", m, names=tuple(dict.fromkeys(n for _, n in cands)))
    for rx in (G6, G6_WHO, G6_SAME):
        for m in rx.finditer(request):
            lit = _unq(m.group("lit"))
            noun = m.group("noun") or m.group("noun2")
            if not lit or _entity(lit) in nouns or lit.casefold() in _GRAMMAR_WORDS:
                continue
            add("members", _entity(noun) if noun else "", m, lit=lit)
    seen, res = set(), []
    for d in out:
        k = (d.op, d.entity, d.field, d.mode, d.lit, d.names, d.span)
        if k not in seen and (d.entity or d.op == "members"):
            seen.add(k)
            res.append(d)
    return res


# ── the session's structured sources ───────────────────────────────────────
_PATTERNS: dict[str, re.Pattern] = {
    "rating": re.compile(r"^\s*Rating:\s*(-?\d+(?:\.\d+)?)\s*$"),
    "price_range": re.compile(r"^\s*Price range:\s*(-?\d+(?:\.\d+)?)\s*-\s*-?\d+(?:\.\d+)?\s*$"),
}


def _num(x: Any) -> float | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(str(x).strip())
    except ValueError:
        return None


@dataclass
class _Ent:
    value: Any = None
    fields: dict = field(default_factory=dict)
    members: list | None = None
    conflict: bool = False


@dataclass
class DescriptionTable:
    """``sources``: tool -> [spec]; a spec: {entity, shape: names|records|keyed|count|members,
    complete?, key?, name?, value?, fields?, members?, field?, pattern?}. ``is_request(value)``,
    ``is_own(value)``: the origin tests the source-argument rule uses."""

    sources: dict
    is_request: Callable[[Any], bool]
    is_own: Callable[[Any], bool] = lambda v: False
    obs: list = field(default_factory=list)  # (tool, args, spec, structure)

    def observe(self, tool: str, args: dict, value: Any) -> bool:
        specs = self.sources.get(tool) or []
        if isinstance(specs, dict):
            specs = [specs]
        took = False
        for spec in specs:
            if _shape_ok(spec, value):
                self.obs.append((tool, dict(args or {}), spec, value))
                took = True
        return took

    # entities of one kind, built from the counted observations
    def _arg_ok(self, spec: dict, a: str, v: Any, listed: set) -> bool:
        vals = v if isinstance(v, (list, tuple)) else [v]
        for x in vals:
            if x is None or x == "" or self.is_request(x) or self.is_own(x):
                continue
            if a == spec.get("key") and _c(x) in listed:
                continue  # the key names an entity of a counted complete list
            return False
        return True

    def _counted(self, entity: str) -> list:
        listed: set = set()
        firsts = []
        for o in self.obs:
            tool, args, spec, val = o
            if spec["entity"] != entity:
                continue
            if spec["shape"] in ("names", "records") and all(
                    self._arg_ok(spec, a, v, set()) for a, v in args.items()):
                firsts.append(o)
                if spec.get("complete"):
                    listed |= {_c(n) for n in _names(spec, val)}
        rest = [o for o in self.obs if o[2]["entity"] == entity and o[2]["shape"] not in ("names", "records")
                and all(self._arg_ok(o[2], a, v, listed) for a, v in o[1].items())]
        return firsts + rest

    def entities(self, entity: str) -> tuple[dict, list]:
        """(name -> _Ent, [complete lists of names])."""
        ents: dict[str, _Ent] = {}
        lists: list[list[str]] = []

        def ent(n: Any) -> _Ent:
            return ents.setdefault(_c(n), _Ent())

        def setv(e: _Ent, v: Any) -> None:
            if e.value is not None and _c(e.value) != _c(v):
                e.conflict = True
            e.value = v

        for tool, args, spec, val in self._counted(entity):
            sh = spec["shape"]
            if sh == "names":
                for n in val:
                    setv(ent(n), n)
                if spec.get("complete"):
                    lists.append([str(n) for n in val])
            elif sh == "records":
                nk, vk = spec.get("name", "name"), spec.get("value", spec.get("name", "name"))
                for r in val:
                    if nk not in r:
                        continue
                    e = ent(r[nk])
                    if vk in r:
                        setv(e, r[vk])
                    for f in spec.get("fields") or ():
                        x = _num(r.get(f))
                        if x is not None:
                            e.fields[f] = x
                    mk = spec.get("members")
                    if mk and isinstance(r.get(mk), list):
                        e.members = list(r[mk])
                if spec.get("complete"):
                    lists.append([str(r[nk]) for r in val if nk in r])
            elif sh == "keyed":
                pat = _PATTERNS.get(spec.get("pattern") or "")
                for n, x in val.items():
                    if pat is not None:
                        first = str(x).split("\n", 1)[0]
                        m = pat.match(first)
                        x = m.group(1) if m else None
                    x = _num(x)
                    if x is not None:
                        ent(n).fields[spec["field"]] = x
            elif sh == "count":
                k = args.get(spec["key"])
                if isinstance(k, str):
                    ent(k).fields[spec["field"]] = float(len(val))
            elif sh == "members":
                k = args.get(spec["key"])
                if isinstance(k, str):
                    ent(k).members = list(val)
        return ents, lists

    def compute(self, d: Description, entity_of_members: Iterable[str] = ()) -> tuple[list, bool] | None:
        """(values, members?) or None (no value: missing, a tie, several matches)."""
        if d.op in ("argmax", "argmin"):
            ents, lists = self.entities(d.entity)
            vals = []
            for lst in lists:
                xs = [(n, ents.get(_c(n), _Ent()).fields.get(d.field)) for n in lst]
                if not xs or any(x is None for _, x in xs):
                    continue  # a member of the list lacks the field: nothing computed
                best = (max if d.op == "argmax" else min)(x for _, x in xs)
                win = [n for n, x in xs if x == best]
                if len(win) != 1:
                    continue  # a tie
                e = ents[_c(win[0])]
                if e.conflict or e.value is None:
                    continue
                vals.append(e.value)
            return (list(dict.fromkeys(vals)), False) if vals else None
        if d.op == "select":
            ents, _ = self.entities(d.entity)
            lit = _c(d.lit)
            if d.mode == "word":  # "the External channel": equals, else the one named word + '_' / digit
                hit = [e for n, e in ents.items() if n == lit] or [
                    e for n, e in ents.items() if re.fullmatch(re.escape(lit) + r"[_\d][\w-]*", n)]
            else:
                hit = [e for n, e in ents.items()
                       if (n == lit if d.mode == "eq" else n.startswith(lit) if d.mode == "starts" else lit in n)]
            if len(hit) != 1 or hit[0].conflict or hit[0].value is None:
                return None
            return [hit[0].value], False
        if d.op == "members":
            kinds = [d.entity] if d.entity else list(entity_of_members)
            found = []
            for k in kinds:
                ents, _ = self.entities(k)
                e = ents.get(_c(d.lit))
                if e is not None and e.members:
                    found.append(e.members)
            if len(found) != 1:
                return None
            return list(found[0]), True
        if d.op == "email_of":
            ents, _ = self.entities("contact")
            for n in d.names:  # the nearest name that is exactly one contact
                e = ents.get(_c(n))
                if e is not None:
                    if e.conflict or e.value is None:
                        return None
                    return [e.value], False
            return None
        return None


def _names(spec: dict, val: Any) -> list:
    if spec["shape"] == "names":
        return list(val)
    nk = spec.get("name", "name")
    return [r[nk] for r in val if isinstance(r, dict) and nk in r]


def _shape_ok(spec: dict, v: Any) -> bool:
    """A structure of the declared shape (anything else — free text — is never a source)."""
    sh = spec.get("shape")
    scalar = (str, int, float)
    if sh in ("names", "members"):
        return isinstance(v, list) and all(isinstance(x, scalar) and not isinstance(x, bool) for x in v)
    if sh in ("records", "count"):
        return isinstance(v, list) and all(isinstance(x, dict) for x in v)
    if sh == "keyed":
        return isinstance(v, dict) and all(isinstance(k, str) and isinstance(x, scalar) for k, x in v.items())
    return False
