"""Deterministic carrier classifier (TM1, T0).

`classify_carrier` projects a value onto the imperative-channel lattice
ENDORSED ⊏ CLOSED_SCHEMA ⊏ FREE_TEXT. It is **deterministic and structural** —
it reads the *form* of the value, never a model, never the value's "meaning"
(T0). A model classifier would let the projection be steered by the governed
content (breaks K4/T0); this one cannot.

Rules (fail-closed — anything ambiguous is ⊤ = FREE_TEXT):

  • scalar (None / int / float / bool)            → ENDORSED   (form carries no instruction)
  • a string / structure that parses *fully* into a closed schema whose every
    leaf is a scalar or a bounded identifier        → CLOSED_SCHEMA
  • anything containing free text                  → FREE_TEXT

Note (consumer-relativity, K3.5): a number is instruction-free *as a carrier*; it
can still be instruction-complete *by use* if a consumer decodes it into an
opcode/index. Carrier only kills the imperative channel of the *form* (TM1); the
data-flow channel is causal_root (TM2).
"""

from __future__ import annotations

import json
import re

from axor_core.contracts.taint import Carrier

# A "bounded identifier": short, no whitespace, no punctuation that a downstream
# interpreter treats as syntax. Conservative on purpose.
_BOUNDED_IDENT = re.compile(r"^[A-Za-z0-9_.:\-/]{1,64}$")


def _is_bounded_identifier(s: str) -> bool:
    return bool(_BOUNDED_IDENT.match(s))


def _classify_leaf(value: object) -> Carrier:
    if value is None or isinstance(value, (int, float, bool)):
        return Carrier.ENDORSED
    if isinstance(value, str):
        return Carrier.CLOSED_SCHEMA if _is_bounded_identifier(value) else Carrier.FREE_TEXT
    return Carrier.FREE_TEXT


def _classify_structure(obj: object) -> Carrier:
    """Worst (highest) carrier over all leaves of a parsed structure."""
    if isinstance(obj, dict):
        items: list[object] = list(obj.keys()) + list(obj.values())
    elif isinstance(obj, (list, tuple)):
        items = list(obj)
    else:
        return _classify_leaf(obj)

    worst = Carrier.ENDORSED
    for item in items:
        if isinstance(item, (dict, list, tuple)):
            c = _classify_structure(item)
        else:
            c = _classify_leaf(item)
        # A structure of identifiers is CLOSED_SCHEMA, not ENDORSED (it has form).
        if c == Carrier.ENDORSED:
            c = Carrier.CLOSED_SCHEMA
        if c == Carrier.FREE_TEXT:
            return Carrier.FREE_TEXT
        worst = c
    return worst


def classify_carrier(value: object) -> Carrier:
    """Deterministic, structural carrier classification (T0). Total function;
    fail-closed to FREE_TEXT for anything not provably lower-capacity.
    """
    if value is None or isinstance(value, (int, float, bool)):
        return Carrier.ENDORSED
    if isinstance(value, (dict, list, tuple)):
        return _classify_structure(value)
    if isinstance(value, str):
        s = value.strip()
        # Try to interpret as a closed (JSON) schema first.
        if s and s[0] in "{[":
            try:
                parsed = json.loads(s)
            except (ValueError, TypeError):
                return Carrier.FREE_TEXT
            return _classify_structure(parsed)
        # A JSON scalar literal (number / bool / null) is endorsed by form.
        try:
            parsed = json.loads(s)
        except (ValueError, TypeError):
            parsed = s
        if isinstance(parsed, (int, float, bool)) or parsed is None:
            return Carrier.ENDORSED
        # A bare bounded identifier is a closed token; anything else is free text.
        return Carrier.CLOSED_SCHEMA if _is_bounded_identifier(s) else Carrier.FREE_TEXT
    return Carrier.FREE_TEXT
