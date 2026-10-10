"""Effect signature and effect digest — which arguments make the effect WHAT IT IS.

A value predicate constrains an argument. An *effect* is the tuple of arguments
that decides what actually happens: a transfer is not "some recipient" plus "some
amount", it is this recipient receiving this amount. Two things in the kernel need
that tuple named rather than implied:

  * **Binding an approval to an effect.** A grant keyed on the sink (or on the
    projection hash) covers a *class*: the canonicalizer deliberately strips raw
    strings, so two transfers to different recipients share a projection and a
    class-scoped approval would authorise either. Binding needs a digest over the
    values that define the effect, so an approval of one call cannot be spent on
    another (see :mod:`axor_core.capability.approvals`).
  * **Saying how precisely a policy describes an effect.** An ``enum`` of
    recipients and a range of amounts constrain the PRODUCT
    ``R × [lo, hi]`` — not the relation between them. A policy can be fully
    decidable and still not pin the effect; :func:`validate_effect_coverage`
    reports that gap instead of leaving it to be discovered.

The signature is not a new operator knob: it is ``driving_args``, the per-sink
declaration the integrity and carrier gates already key on
([governance-model](../../docs/governance-model.md) §12). Undeclared, the
signature falls back to *every* argument present in the call — the fail-closed
direction, since over-binding only narrows what an approval covers.

Nothing here reads provenance or content semantics: the digest is a hash of
values, not an interpretation of them, and it is never an input to a gate — only
an equality key between an authorisation and the call it was issued for.
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


def effect_digest(
    tool: str,
    args: Mapping[str, object] | None,
    driving_args: Mapping[str, Iterable[str]] | None = None,
) -> str:
    """A deterministic digest of (sink, signature, the signature's values).

    Equality of digests means "the same effect was authorised"; nothing else is
    inferred from it. The signature itself is hashed too, so changing a sink's
    ``driving_args`` invalidates authorisations issued under the old declaration
    rather than silently widening them.
    """
    names = effect_signature(tool, args, driving_args)
    a = args or {}
    payload = {
        "tool": tool,
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
