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

# ── the phrase-template library (MODEL.md §9.15, complete) ──────────────────
NOUNS: dict[str, str] = {
    r"channels?": "channel",
    r"users?|members?": "user",
    r"contacts?": "contact",
    r"events?|meetings?": "event",
    r"files?|documents?": "file",
    r"hotels?": "hotel",
    r"restaurants?": "restaurant",
    r"(?:car\s+)?rental\s+compan(?:y|ies)": "car_rental",
}
FIELDS: dict[str, str] = {
    r"messages?": "messages",
    r"users|members|people": "users",
    r"ratings?|rated": "rating",
    r"prices?|cost": "price",
    r"size": "size",
}
MAX_WORDS = r"most|highest|largest|biggest|greatest|best|top"
MIN_WORDS = r"fewest|least|lowest|smallest"

def _noun_rx(name: str = "noun") -> str:
    return f"(?P<{name}>" + "|".join(f"(?:{k})" for k in sorted(NOUNS, key=len, reverse=True)) + r")\b"


_NOUN = _noun_rx()
_FIELD = "(?P<field>" + "|".join(f"(?:{k})" for k in FIELDS) + r")\b"
_LIT = r"(?P<lit>'[^']+'|\"[^\"]+\"|[\w.@-]+)"
_W = r"(?:\S+\s+)"

T1 = re.compile(r"\b" + _NOUN + r"(?:\s+\S+){0,4}?\s+(?:with|that\s+has|having|which\s+has)\s+(?:the\s+)?"
                r"(?P<sup>" + MAX_WORDS + "|" + MIN_WORDS + r")\s+(?:number\s+of\s+|total\s+|amount\s+of\s+)?"
                + _FIELD, re.I)
T1_RATED = re.compile(r"\b(?:highest|best|top)[- ]rated\s+" + _W + r"{0,2}?" + _NOUN, re.I)
T1_ADJ = re.compile(r"\bthe\s+(?P<adj>largest|biggest|smallest|cheapest|most\s+expensive)\s+" + _W + r"{0,2}?"
                    + _NOUN, re.I)
T2 = re.compile(r"\b" + _NOUN + r"\s+(?P<op>named|called|titled|starting\s+with|beginning\s+with|"
                r"that\s+starts\s+with|containing|that\s+contains|whose\s+name\s+contains)\s+" + _LIT, re.I)
T2_QUOTED = re.compile(r"(?P<lit>'[^']+'|\"[^\"]+\")\s+" + _NOUN, re.I)
_NAME = r"[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*){0,2}"
T3_OF = re.compile(r"\be-?mail(?:\s+address)?\s+of\s+(?P<name>" + _NAME + ")")
T3_POSS = re.compile(r"(?P<name>" + _NAME + r")'s\s+e-?mail\b")
T3_PRON = re.compile(r"\b(?:his|her|their)\s+e-?mail\b", re.I)
T4 = re.compile(r"\b(?:members|participants|users|everyone|everybody|all\s+users|all\s+members|"
                r"all\s+participants)\s+(?:of|in|from)\s+(?:the\s+)?(?:" + _NOUN + r"\s+)?" + _LIT
                + r"(?:\s+" + _noun_rx("noun2") + ")?", re.I)
T4_SAME = re.compile(r"\bsame\s+(?:participants|members|users|attendees)\s+as\s+(?:the\s+)?(?:" + _NOUN
                     + r"\s+)?" + _LIT + r"(?:\s+" + _noun_rx("noun2") + ")?", re.I)
_CAPS = re.compile(_NAME)
_SELECT_OP = {"named": "eq", "called": "eq", "titled": "eq", "starting with": "starts",
              "beginning with": "starts", "that starts with": "starts", "containing": "contains",
              "that contains": "contains", "whose name contains": "contains"}


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
    return lit.rstrip(".,;:!?")


def _c(x: Any) -> str:
    return str(x).strip().casefold()


@dataclass(frozen=True)
class Description:
    op: str  # argmax | argmin | select | members | email_of
    entity: str
    field: str = ""
    mode: str = ""  # select: eq | starts | contains
    lit: str = ""
    names: tuple = ()  # email_of: candidate names, nearest first
    phrase: str = ""


def parse(request: str) -> list[Description]:
    """The descriptions a request carries, by the fixed template library."""
    out: list[Description] = []
    nouns = set(NOUNS.values())
    for m in T1.finditer(request):
        e, f = _entity(m.group("noun")), _field(m.group("field"))
        if e and f:
            op = "argmax" if re.fullmatch(MAX_WORDS, m.group("sup"), re.I) else "argmin"
            out.append(Description(op, e, f, phrase=m.group(0)))
    for m in T1_RATED.finditer(request):
        out.append(Description("argmax", _entity(m.group("noun")), "rating", phrase=m.group(0)))
    for m in T1_ADJ.finditer(request):
        a = re.sub(r"\s+", " ", m.group("adj").casefold())
        op, f = {"largest": ("argmax", "size"), "biggest": ("argmax", "size"), "smallest": ("argmin", "size"),
                 "cheapest": ("argmin", "price"), "most expensive": ("argmax", "price")}[a]
        out.append(Description(op, _entity(m.group("noun")), f, phrase=m.group(0)))
    for m in T2.finditer(request):
        lit = _unq(m.group("lit"))
        if lit and _entity(lit) not in nouns and lit.casefold() not in ("the", "a", "an"):
            op = _SELECT_OP[re.sub(r"\s+", " ", m.group("op").casefold())]
            out.append(Description("select", _entity(m.group("noun")), mode=op, lit=lit, phrase=m.group(0)))
    for m in T2_QUOTED.finditer(request):
        out.append(Description("select", _entity(m.group("noun")), mode="eq", lit=_unq(m.group("lit")),
                               phrase=m.group(0)))
    for rx in (T3_OF, T3_POSS):
        for m in rx.finditer(request):
            out.append(Description("email_of", "contact", names=(m.group("name"),), phrase=m.group(0)))
    for m in T3_PRON.finditer(request):
        before = request[:m.start()]
        cands = []
        for c in _CAPS.finditer(before):
            # every capitalized word sequence and its sub-sequences, nearest first
            ws = c.group(0).split()
            for i in range(len(ws)):
                for j in range(len(ws), i, -1):
                    cands.append((c.end(), " ".join(ws[i:j])))
        cands.sort(key=lambda t: -t[0])
        out.append(Description("email_of", "contact", names=tuple(dict.fromkeys(n for _, n in cands)),
                               phrase=m.group(0)))
    for rx in (T4, T4_SAME):
        for m in rx.finditer(request):
            lit = _unq(m.group("lit"))
            noun = m.group("noun") or m.group("noun2")
            if not lit or _entity(lit) in nouns or lit.casefold() in ("the", "a", "an"):
                continue
            ent = _entity(noun) if noun else ""
            out.append(Description("members", ent, lit=lit, phrase=m.group(0)))
    return [d for d in dict.fromkeys(out) if d.entity or d.op == "members"]


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
