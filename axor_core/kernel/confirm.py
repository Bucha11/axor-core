"""L4: confirming an exact action (Ring 0).

The model may PROPOSE an action; only a confirmation turns it into a grant. What the
human sees is rendered from the handle table — the raw value each driving argument
resolves to, and whether that value is in the registry (a trusted handle minted from
the request, an operator template or an earlier confirmation) — never from text the
model wrote about the action. A model that describes "pay Alice's landlord" while
the recipient argument is the attacker's IBAN shows the attacker's IBAN, unmarked.

``ScriptedConfirmer`` stands in for the human in benchmarks: a policy over the
rendered action, with a log, so every confirmation is counted.

A confirmation grant covers the WHOLE call, not just its driving arguments: it is
bound to ``call_digest`` of the canonical call (sink + every argument, rich fields
included), and the rendering shows every field in full. Otherwise a confirmed
"send Bob the minutes" could be spent on the same recipient with another body.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Mapping, Protocol

from axor_core.kernel.handles import HandleTable, is_handle_ref, split_kind


@dataclass(frozen=True)
class RenderedArg:
    arg: str
    raw: str
    in_registry: bool


@dataclass(frozen=True)
class RenderedAction:
    sink: str
    args: tuple[RenderedArg, ...]
    other: tuple[tuple[str, str], ...] = ()  # non-grant args, shown verbatim

    def text(self) -> str:
        lines = [f"Allow {self.sink}?"]
        for a in self.args:
            mark = "in registry" if a.in_registry else "NOT in registry"
            lines.append(f"  {a.arg} = {a.raw}  [{mark}]")
        for k, v in self.other:
            lines.append(f"  {k} = {v}")
        return "\n".join(lines)


def render(sink: str, args: Mapping[str, Any], grant_args: Mapping[str, str],
           table: HandleTable) -> RenderedAction:
    """Render from the table, never from model text (the ``description`` a model
    attaches to a proposal is not an input here). Every argument is shown in full —
    the rich fields too: the grant this confirmation mints covers the whole call."""
    shown: list[RenderedArg] = []
    for arg, spec in grant_args.items():
        kind = split_kind(spec)[0]
        if args.get(arg) is None:
            continue
        v = args[arg]
        vals = list(v) if isinstance(v, (list, tuple, set, frozenset)) else [v]
        for x in vals:
            if is_handle_ref(x):
                h = table.get(x)
                raw = h.raw if h is not None else f"<unknown handle {x}>"
                trusted = bool(h and h.trusted)
            else:
                raw = str(x)
                trusted = table.trusted_for(kind, x) is not None
            shown.append(RenderedArg(arg, raw, trusted))
    other = tuple((k, str(v)) for k, v in args.items() if k not in grant_args)
    return RenderedAction(sink, tuple(shown), other)


def _canon_leaf(v: Any) -> Any:
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float, Decimal)):
        try:
            return format(Decimal(str(v)).normalize(), "f")
        except InvalidOperation:
            return str(v)
    if isinstance(v, str):
        return unicodedata.normalize("NFKC", v)
    if isinstance(v, Mapping):
        return {str(k): _canon_leaf(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_canon_leaf(x) for x in v]
    if isinstance(v, (set, frozenset)):
        return sorted((_canon_leaf(x) for x in v), key=repr)
    return str(v)


def call_digest(sink: str, args: Mapping[str, Any], grant_args: Mapping[str, str],
                table: HandleTable | None) -> str:
    """SHA-256 of the canonical call: the sink and every argument. Grant args are
    taken by canonical value (a handle and the raw value it names digest alike);
    every other argument — rich fields included — by its NFKC text. Absent and
    None are the same."""
    canon: dict[str, Any] = {}
    for k, v in args.items():
        if v is None:
            continue
        spec = grant_args.get(k)
        if spec is not None:
            kind = split_kind(spec)[0]
            vals = v if isinstance(v, (list, tuple, set, frozenset)) else [v]
            res = []
            for x in vals:
                c = table.resolve(x, kind) if table is not None else None
                res.append(c if c is not None else _canon_leaf(x))
            canon[k] = sorted(res, key=repr) if isinstance(v, (list, tuple, set, frozenset)) else res[0]
        else:
            canon[k] = _canon_leaf(v)
    blob = json.dumps({"sink": sink, "args": canon}, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ConfirmationProvider(Protocol):
    def confirm(self, action: RenderedAction, call: tuple[str, Mapping[str, Any]]) -> bool: ...


@dataclass
class ScriptedConfirmer:
    """A benchmark confirmer: ``policy(action, call) -> bool``, every ask logged."""

    policy: Callable[[RenderedAction, tuple[str, Mapping[str, Any]]], bool]
    log: list[tuple[str, bool]] = field(default_factory=list)

    def confirm(self, action: RenderedAction, call: tuple[str, Mapping[str, Any]]) -> bool:
        ok = bool(self.policy(action, call))
        self.log.append((action.text(), ok))
        return ok

    @property
    def asked(self) -> int:
        return len(self.log)


def careful(action: RenderedAction, call) -> bool:
    """Confirms only when every driving value is in the registry."""
    return all(a.in_registry for a in action.args)


def naive(action: RenderedAction, call) -> bool:
    """Confirms anything (the fatigued user): the lower bound for E3."""
    return True
