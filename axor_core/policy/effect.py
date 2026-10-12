"""Two separate mechanisms, deliberately not one.

**The call digest** (:func:`call_digest`) answers *"is this the call that was
authorised?"* It covers the sink, the authority scope it was issued in, and
**every** argument of the call — not the declared signature below, so a
mis-declared declaration cannot widen what an authorisation covers. It is
emphatically a binding to one *call representation*, not to an effect: the same
call made against changed external state can produce a different effect, which is
why the kernel re-checks the state-dependent gates after an approval and consumes
the authorisation atomically with execution (see
:meth:`axor_core.node.escalation.EscalationManager.grant_for_call`). The digest is
never an input to a gate — only an equality key between an authorisation and the
call it was issued for.

**The effect signature** (:func:`effect_signature`, :func:`validate_effect_coverage`)
answers a different question: *which parameters determine the effect this sink
produces, and how completely does the policy constrain them?* An ``enum`` of
recipients and a range of amounts constrain the PRODUCT ``R × [lo, hi]`` — not the
relation between them. A policy can be fully decidable and still not pin the
effect; the coverage check reports that instead of leaving it to be discovered.
The signature is reported to an approver (as the operator's statement of what
matters for this sink) and used for nothing else.

The signature is not a new operator knob: it is ``driving_args``, the per-sink
declaration the integrity and carrier gates already key on
([governance-model](../../docs/governance-model.md) §12). Undeclared, it falls
back to every argument present in the call.

Nothing here reads provenance or content semantics: a digest is a hash of values,
not an interpretation of them.
"""

from __future__ import annotations

import hashlib
import json
from typing import Iterable, Mapping, Sequence

# Marker for a signature argument the call does not carry. Distinct from any
# value an argument could hold, so "absent" and "present but null" differ.
_ABSENT = {"__axor__": "absent"}


def effect_signature(
    tool: str,
    args: Mapping[str, object] | None,
    driving_args: Mapping[str, Iterable[str]] | None = None,
) -> tuple[str, ...]:
    """The argument names that determine this call's effect, sorted.

    Declared ``driving_args`` for the sink win — including names the call does not
    carry, so a *missing* driver changes the digest rather than being skipped.
    Undeclared, every present argument is in the signature (fail-closed: any
    change to any argument invalidates an approval bound to it).
    """
    declared = (driving_args or {}).get(tool)
    if declared:
        return tuple(sorted(set(declared)))
    return tuple(sorted((args or {}).keys()))


def _canonical(value: object) -> object:
    """A JSON-stable, type-tagged form of one argument value.

    Type-tagged so ``1`` and ``"1"`` are different effects, and deliberately NOT
    normalised (no case-folding, no NFKC, no whitespace stripping): normalisation
    would make two *different* identifiers share a digest, which is the one
    direction that must never collide. Any difference in the raw value is a
    different effect.
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return [type(value).__name__, value]
    if isinstance(value, Mapping):
        return ["map", {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}]
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=repr) if isinstance(value, (set, frozenset)) else list(value)
        return [type(value).__name__, [_canonical(v) for v in items]]
    return ["repr", repr(value)]


def call_digest(
    tool: str,
    args: Mapping[str, object] | None,
    *,
    scope: str = "",
    driving_args: Mapping[str, Iterable[str]] | None = None,
) -> str:
    """A deterministic digest of (sink, authority scope, every argument).

    Equality of digests means "this is the call that was authorised" — nothing
    more is inferred from it, and in particular not "the same effect will
    happen": external state moves, so the authorisation's *currency* is a
    separate obligation (re-check + atomic consumption), not something a hash can
    carry.

    ``scope`` binds the authorisation to the context it was issued in (the
    session / node), so evidence cannot travel to another session even if the same
    call is made there. Callers pass the session or node identity; an empty scope
    is a process-local authorisation and is only correct where no other scope can
    reach the grant table.

    Every argument is covered, including ones outside the declared
    ``driving_args``: over-binding only narrows what an authorisation covers, and
    it removes any dependence on the declaration being right. ``driving_args`` is
    accepted only for the deliberately NARROWER digest used in tests and tooling
    that want the signature-scoped key; it is not what authorisations are bound
    with.
    """
    names = effect_signature(tool, args, driving_args)
    a = args or {}
    payload = {
        "tool": tool,
        "scope": scope,
        "signature": list(names),
        "values": {n: _canonical(a[n]) if n in a else _ABSENT for n in names},
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def validate_effect_coverage(
    driving_args: Mapping[str, Iterable[str]] | None,
    value_policies: "Mapping[str, Sequence] | None",
    *,
    require_relational: bool = False,
) -> list[str]:
    """How completely the registered predicates describe each sink's effect.

    Returns one message per gap (empty == complete). Advisory by construction: it
    reports what the policy does NOT pin, and never relaxes a gate.

      * an argument in the sink's effect signature that no predicate constrains —
        the effect can vary freely along that axis under an approved call;
      * with ``require_relational``, a signature of two or more arguments
        constrained only by independent per-argument predicates: that is the
        product of the constraints, not the relation between them (an allowlisted
        recipient with an in-range amount, in any pairing).
    """
    from axor_core.policy.value_policy import constrained_args, relational_arg_groups

    errors: list[str] = []
    pol = value_policies or {}
    for sink in sorted((driving_args or {}).keys()):
        signature = tuple(sorted(set((driving_args or {})[sink] or ())))
        if not signature:
            continue
        preds = list(pol.get(sink, ()) or ())
        covered = constrained_args(preds)
        for missing in [s for s in signature if s not in covered]:
            errors.append(
                f"sink {sink!r}: effect signature arg {missing!r} is constrained by "
                f"no value predicate — an authorised call may vary it freely"
            )
        if require_relational and len(signature) > 1:
            groups = relational_arg_groups(preds)
            if not any(set(signature) <= g for g in groups):
                errors.append(
                    f"sink {sink!r}: effect signature {list(signature)} is constrained "
                    f"only per-argument, so the policy admits the PRODUCT of the "
                    f"constraints, not the relation between them — use "
                    f"tuple_enum/keyed_range to pin the pairing"
                )
    return errors
