"""L4: confirming an exact action (Ring 0).

The model may PROPOSE an action; only a confirmation turns it into a grant. What the
human sees is rendered from the handle table — the raw value each driving argument
resolves to, and whether that value is in the registry (a trusted handle minted from
the request, an operator template or an earlier confirmation) — never from text the
model wrote about the action. A model that describes "pay Alice's landlord" while
the recipient argument is the attacker's IBAN shows the attacker's IBAN, unmarked.

``ScriptedConfirmer`` stands in for the human in benchmarks: a policy over the
rendered action, with a log, so every confirmation is counted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    attaches to a proposal is not an input here)."""
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
