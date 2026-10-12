"""Consumption contracts — what a sink's handler actually does with each argument.

T4 (effective codomain = nominal codomain) was stated per *predicate*: an ``enum``
is membership in a finite set, a ``numeric_range`` is a comparison, both decidable.
That shows the CHECK is exact. It does not show that the property checked is the
one the handler consumes. Between "``recipient`` ∈ allowlist" and "the transfer
went to an allowlisted account" sit three things the predicate never saw:

  * **presence** — a predicate passes on an absent argument, and the handler may
    then use its own default (a sweep of the whole balance, a default account);
  * **representation** — the handler may parse, coerce or truncate what it gets
    (``float("1e3")``, ``int(999.9)``), so the value checked is not the value used;
  * **interpretation** — a path is resolved against a filesystem, a URL or a
    command is parsed: equal strings can produce different effects, so exact
    membership on the string says nothing about the effect.

A consumption contract closes the first two and refuses the third. Per sink it
names every field the handler reads and the field's consumption kind:

  =================  ============================  ==================================
  kind               consumed as                   decidable predicate admissible?
  =================  ============================  ==================================
  NUMBER             a typed number, numerically   numeric_range, enum, keyed value
  FLAG               a boolean, as a case split    enum, tuple_enum
  IDENTIFIER         one exact identifier          enum, tuple_enum, keyed key
  IDENTIFIER_LIST    a list of exact identifiers   enum, tuple_enum, keyed key
  PATH               handed to a path resolver     no — fuzz obligation
  INTERPRETED        parsed by an interpreter      no — fuzz obligation
  TEXT               opaque content                no — fuzz obligation
  =================  ============================  ==================================

and and what the handler does when the field is absent (refuse, a declared default,
or not consumed). The kernel then does two things with it:

1. **Registration** (:func:`validate_consumption_contracts`) — a predicate is
   checked against the contract of the *field* it guards, not only against its own
   kind: an ``enum`` on a PATH field is refused (exact membership on a string the
   resolver interprets is not faithfulness), a guard on a field the contract does
   not name is refused (the check would decide a field the handler is not
   declared to read).
2. **Preparation** (:func:`prepare_args`) — the call's arguments are put into the
   one representation the contract describes (closed field set, declared defaults
   materialised, the declared numeric conversion applied) BEFORE any gate runs.
   That prepared set is what every gate reads and what the handler receives, so
   the value checked and the value consumed are the same object.

What this does NOT establish: that the handler honours its contract. A contract is
the operator's statement about the handler; the kernel makes the check and the
dispatch agree with it, and makes a predicate that the contract shows cannot be
faithful unregistrable. For a sink with no contract, the conformance of handler to
predicate is a premise — under STRICT the operator must state it explicitly
(``consumption_attested``), and :func:`conformance_obligations` reports which
guarded arguments rest on a contract, on an attestation, or on nothing.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from types import MappingProxyType
from typing import Iterable, Mapping

from axor_core.kernel.decidability import (
    CodomainKind,
    ConsumptionMode,
    DecidabilityVerdict,
    classify,
)


class FieldKind(str, Enum):
    """How a handler consumes one argument."""

    NUMBER = "number"
    FLAG = "flag"
    IDENTIFIER = "identifier"
    IDENTIFIER_LIST = "identifier_list"
    PATH = "path"
    INTERPRETED = "interpreted"
    TEXT = "text"


# The consumption mode each field kind implies (decidability.ConsumptionMode).
_MODE_OF: dict[FieldKind, ConsumptionMode] = {
    FieldKind.NUMBER: ConsumptionMode.NUMERIC,
    FieldKind.FLAG: ConsumptionMode.CASE_SPLIT,
    FieldKind.IDENTIFIER: ConsumptionMode.CASE_SPLIT,
    FieldKind.IDENTIFIER_LIST: ConsumptionMode.CASE_SPLIT,
    FieldKind.PATH: ConsumptionMode.PATH_RESOLVE,
    FieldKind.INTERPRETED: ConsumptionMode.INTERPRET,
    FieldKind.TEXT: ConsumptionMode.INTERPRET,
}

# Field kinds each predicate role may guard. A role is what the predicate reads
# from that argument: a closed-set membership ("member") or a numeric comparison
# ("range"). An enum over numbers is a case split and admissible; a range over an
# identifier is not (the comparison would decide nothing the handler uses).
_ADMISSIBLE: dict[str, frozenset[FieldKind]] = {
    "member": frozenset({
        FieldKind.IDENTIFIER, FieldKind.IDENTIFIER_LIST, FieldKind.NUMBER, FieldKind.FLAG,
    }),
    "range": frozenset({FieldKind.NUMBER}),
}
_CODOMAIN_OF_ROLE: dict[str, CodomainKind] = {
    "member": CodomainKind.ENUM,
    "range": CodomainKind.BOUNDED_NUMERIC,
}

# The only string → number conversion the kernel performs: a plain decimal.
# Exponents, separators, whitespace, "nan"/"inf" — everything a handler's own
# float()/Decimal() would accept and the predicate would not have seen — is
# refused rather than guessed at.
_DECIMAL = re.compile(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?")


class _Missing:
    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "MISSING"


MISSING: object = _Missing()


@dataclass(frozen=True)
class FieldContract:
    """One argument, as its handler consumes it.

    What the handler does when the argument is ABSENT must be stated, because a
    predicate passes on an absent value — so exactly one of:

    ``required=True`` — absent ⇒ the call is refused. The default unless one of
                         the next two is given.
    ``default=<v>``    — the handler uses ``v``; it is materialised into the
                         prepared call, so the gates see it.
    ``unused_when_absent=True`` — the handler does not consume the field at all
                         when it is absent (an update that leaves it unchanged).
                         This is the operator's statement about the handler; a
                         handler that in fact applies a default of its own makes
                         the contract false, and no check here can see that.

    ``null_is_absent`` — an explicit ``None`` means "absent" (optional fields
                         only), as in ``update(recipient=None)``.
    ``integral``       — NUMBER only: the handler consumes an int; a non-integral
                         number is refused (never truncated behind the check's back).
    ``decimal_string`` — NUMBER only: a plain decimal string is converted to a
                         number here, so the handler never parses it itself.
    """

    kind: FieldKind
    required: "bool | None" = None
    default: object = MISSING
    unused_when_absent: bool = False
    null_is_absent: bool = False
    integral: bool = False
    decimal_string: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.kind, FieldKind):
            object.__setattr__(self, "kind", FieldKind(self.kind))
        has_default = self.default is not MISSING
        if self.required is None:
            object.__setattr__(
                self, "required", not (has_default or self.unused_when_absent)
            )
        if self.required and (has_default or self.unused_when_absent or self.null_is_absent):
            raise ValueError(
                "a required field takes no default / unused_when_absent / null_is_absent"
            )
        if not self.required and has_default == self.unused_when_absent:
            raise ValueError(
                "an optional field must state what the handler does when it is absent: "
                "exactly one of default=<value> or unused_when_absent=True"
            )
        if (self.integral or self.decimal_string) and self.kind is not FieldKind.NUMBER:
            raise ValueError(f"integral/decimal_string apply to NUMBER fields, not {self.kind.value}")


@dataclass(frozen=True)
class SinkContract:
    """Every field a sink's handler reads. ``closed`` (the default) refuses an
    argument the contract does not name: a handler that also reads an alias
    (``to`` next to ``recipient``) would otherwise act on a field no predicate saw."""

    fields: Mapping[str, FieldContract] = field(default_factory=dict)
    closed: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.fields, MappingProxyType):
            object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))


def mode_of(kind: FieldKind) -> ConsumptionMode:
    return _MODE_OF[kind]


# ── preparation ──────────────────────────────────────────────────────────────


def _prepare_value(name: str, value: object, fc: FieldContract) -> "tuple[object, str | None]":
    """The value in the representation the contract names, or a refusal."""
    kind = fc.kind
    if kind is FieldKind.NUMBER:
        num = value
        if isinstance(num, str) and fc.decimal_string:
            if not _DECIMAL.fullmatch(num):
                return None, f"'{name}'={value!r} is not a plain decimal number"
            num = int(num) if "." not in num else float(num)
        if isinstance(num, bool) or not isinstance(num, Real):
            return None, f"'{name}'={value!r} is not a number"
        if isinstance(num, float) and not math.isfinite(num):
            return None, f"'{name}'={value!r} is not finite"
        if fc.integral and not isinstance(num, int):
            if not (isinstance(num, float) and num.is_integer()):
                return None, f"'{name}'={value!r} is not an integer (it would be truncated)"
            num = int(num)
        return num, None
    if kind is FieldKind.FLAG:
        if not isinstance(value, bool):
            return None, f"'{name}'={value!r} is not a boolean"
        return value, None
    if kind is FieldKind.IDENTIFIER:
        if not isinstance(value, str):
            return None, f"'{name}'={value!r} is not a single identifier string"
        return value, None
    if kind is FieldKind.IDENTIFIER_LIST:
        if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
            return None, f"'{name}'={value!r} is not a list of identifier strings"
        return list(value), None
    # PATH / INTERPRETED / TEXT: carried as the string the handler gets.
    if not isinstance(value, str):
        return None, f"'{name}'={value!r} is not a string"
    return value, None


def prepare_args(args: Mapping, contract: SinkContract) -> "tuple[dict | None, str | None]":
    """Put a call's arguments into the contract's representation.

    Returns ``(prepared, None)`` or ``(None, reason)``. Idempotent:
    ``prepare_args(prepared, c) == (prepared, None)``, which is what lets the
    replay kernel re-run it on a recorded call and reach the same verdict.
    """
    args = dict(args or {})
    if contract.closed:
        extra = sorted(set(args) - set(contract.fields))
        if extra:
            return None, (
                f"argument(s) {extra} are not in the sink's consumption contract — the "
                f"handler could act on a field no predicate saw"
            )
    out: dict = {k: v for k, v in args.items() if k not in contract.fields}
    for name, fc in contract.fields.items():
        absent = name not in args or (fc.null_is_absent and args[name] is None)
        if absent:
            if fc.required:
                return None, f"required argument '{name}' is absent"
            if fc.default is MISSING:
                continue  # unused_when_absent: the handler does not consume it
            value = fc.default
        else:
            value = args[name]
        prepared, reason = _prepare_value(name, value, fc)
        if reason is not None:
            return None, reason
        out[name] = prepared
    return out, None


def prepare_call(
    tool: str,
    args: Mapping,
    contracts: "Mapping[str, SinkContract] | None",
) -> "tuple[dict, str | None]":
    """Prepare one call. A tool with no contract passes through unchanged."""
    contract = (contracts or {}).get(tool)
    if contract is None:
        return dict(args or {}), None
    prepared, reason = prepare_args(args, contract)
    if reason is not None:
        return dict(args or {}), f"consumption contract of '{tool}': {reason}"
    return prepared, None


# ── registration ─────────────────────────────────────────────────────────────


def predicate_field_roles(pred: object) -> "list[tuple[str, str]]":
    """(arg, role) for every argument a predicate reads; role is "member" or
    "range". Unknown kinds yield nothing (registration refuses them elsewhere)."""
    kind = getattr(pred, "kind", "")
    if kind == "enum":
        return [(pred.arg, "member")]
    if kind == "numeric_range":
        return [(pred.arg, "range")]
    if kind == "tuple_enum":
        return [(a, "member") for a in pred.args]
    if kind == "keyed_range" and len(pred.args) == 2:
        return [(pred.args[0], "member"), (pred.args[1], "range")]
    return []


def validate_consumption_contracts(
    value_policies: "Mapping[str, Iterable] | None",
    contracts: "Mapping[str, SinkContract] | None",
    attested: "Iterable[str] | None" = None,
    *,
    strict: bool = False,
) -> list[str]:
    """Check every predicate against the contract of the field it guards.

    Errors (empty == valid):
      * a guarded argument the contract does not name;
      * a predicate role the field's consumption kind does not admit — for a PATH /
        INTERPRETED / TEXT field this is the decidability classifier's
        FUZZ_REQUIRED verdict, now reached through a real field;
      * a declared default that does not prepare under its own field contract;
      * a sink both contracted and attested (one ground per sink);
      * under ``strict``: a guarded sink with neither a contract nor an
        attestation — the conformance premise must be stated, not implied.
    """
    errors: list[str] = []
    pols = value_policies or {}
    cons = contracts or {}
    att = frozenset(attested or ())

    for sink, contract in cons.items():
        for name, fc in contract.fields.items():
            if not fc.required and fc.default is not MISSING:
                _, reason = _prepare_value(name, fc.default, fc)
                if reason is not None:
                    errors.append(f"sink {sink!r} field {name!r}: declared default fails "
                                  f"its own contract — {reason}")

    for sink in sorted(att & set(cons)):
        errors.append(
            f"sink {sink!r} is both contracted and attested — declare one ground: the "
            f"contract (checked) or the attestation (a stated premise)"
        )

    for sink in sorted(pols):
        preds = list(pols[sink] or ())
        if not preds:
            continue
        contract = cons.get(sink)
        if contract is None:
            if strict and sink not in att:
                errors.append(
                    f"sink {sink!r} carries value predicates but no consumption contract: "
                    f"STRICT requires either field_contracts[{sink!r}] (checked against "
                    f"the predicates and applied to every call) or listing it in "
                    f"consumption_attested (the operator's explicit statement that the "
                    f"handler consumes the guarded fields as the predicates assume)"
                )
            continue
        for pred in preds:
            for arg, role in predicate_field_roles(pred):
                fc = contract.fields.get(arg)
                if fc is None:
                    errors.append(
                        f"sink {sink!r}: predicate {pred.kind!r} guards {arg!r}, which the "
                        f"consumption contract does not name — the check would decide a "
                        f"field the handler is not declared to read"
                    )
                    continue
                if fc.kind not in _ADMISSIBLE[role]:
                    verdict = classify(_CODOMAIN_OF_ROLE[role], _MODE_OF[fc.kind])
                    why = (
                        f"faithfulness for ({_CODOMAIN_OF_ROLE[role].value}, "
                        f"{_MODE_OF[fc.kind].value}) is a fuzz obligation, not a "
                        f"decision procedure"
                        if verdict.verdict is DecidabilityVerdict.FUZZ_REQUIRED
                        else f"a {role} check on a {fc.kind.value} field decides "
                             f"nothing the handler consumes"
                    )
                    errors.append(
                        f"sink {sink!r} field {arg!r}: predicate {pred.kind!r} cannot "
                        f"guard a {fc.kind.value} field — {why}"
                    )
                    continue
    return errors


@dataclass(frozen=True)
class ConformanceObligation:
    """What a guarded argument's faithfulness rests on.

    ``ground`` is ``"contract"`` (checked at registration, applied per call),
    ``"attested"`` (an explicit operator premise) or ``"unverified"`` (nothing —
    possible only outside STRICT)."""

    sink: str
    arg: str
    predicate: str
    ground: str


def conformance_obligations(
    value_policies: "Mapping[str, Iterable] | None",
    contracts: "Mapping[str, SinkContract] | None",
    attested: "Iterable[str] | None" = None,
) -> list[ConformanceObligation]:
    """One row per (sink, guarded arg, predicate): the report the claim needs —
    which T4 instances are established by a contract, and which are premises."""
    cons = contracts or {}
    att = frozenset(attested or ())
    rows: list[ConformanceObligation] = []
    for sink in sorted(value_policies or {}):
        ground = "contract" if sink in cons else ("attested" if sink in att else "unverified")
        for pred in (value_policies or {})[sink] or ():
            for arg, _role in predicate_field_roles(pred):
                rows.append(ConformanceObligation(sink, arg, pred.kind, ground))
    return rows


def parse_field_contracts(raw: object) -> "dict[str, SinkContract]":
    """Build contracts from a config mapping; fails closed on anything malformed.

    ``{sink: {fields: {name: {kind, required?, default?, unused_when_absent?,
    null_is_absent?, integral?, decimal_string?}}, closed?: bool}}``
    """
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("field_contracts must be a mapping of tool -> contract")
    out: dict[str, SinkContract] = {}
    for sink, spec in raw.items():
        if not isinstance(spec, Mapping):
            raise ValueError(f"field_contracts[{sink!r}] must be a mapping")
        extra = set(spec) - {"fields", "closed"}
        if extra:
            raise ValueError(f"field_contracts[{sink!r}]: unknown key(s) {sorted(extra)}")
        fields_raw = spec.get("fields")
        if not isinstance(fields_raw, Mapping) or not fields_raw:
            raise ValueError(f"field_contracts[{sink!r}] needs a non-empty 'fields' mapping")
        closed = spec.get("closed", True)
        if not isinstance(closed, bool):
            raise ValueError(f"field_contracts[{sink!r}].closed must be a boolean")
        fields: dict[str, FieldContract] = {}
        for name, f in fields_raw.items():
            if not isinstance(f, Mapping) or "kind" not in f:
                raise ValueError(f"field_contracts[{sink!r}].{name}: needs a 'kind'")
            fextra = set(f) - {
                "kind", "required", "default", "unused_when_absent", "null_is_absent",
                "integral", "decimal_string",
            }
            if fextra:
                raise ValueError(
                    f"field_contracts[{sink!r}].{name}: unknown key(s) {sorted(fextra)}"
                )
            try:
                kind = FieldKind(f["kind"])
            except ValueError:
                raise ValueError(
                    f"field_contracts[{sink!r}].{name}: unknown kind {f['kind']!r}; "
                    f"expected one of {[k.value for k in FieldKind]}"
                ) from None
            required = f.get("required")
            flags = {
                k: f.get(k, False)
                for k in ("unused_when_absent", "null_is_absent", "integral", "decimal_string")
            }
            if (required is not None and not isinstance(required, bool)) or not all(
                isinstance(v, bool) for v in flags.values()
            ):
                raise ValueError(
                    f"field_contracts[{sink!r}].{name}: required / unused_when_absent / "
                    f"null_is_absent / integral / decimal_string must be booleans"
                )
            try:
                fields[str(name)] = FieldContract(
                    kind=kind, required=required, default=f.get("default", MISSING),
                    **flags,
                )
            except ValueError as exc:
                raise ValueError(f"field_contracts[{sink!r}].{name}: {exc}") from None
        out[str(sink)] = SinkContract(fields=fields, closed=closed)
    return out
