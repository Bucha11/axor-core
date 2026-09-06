"""The contract schemas, owned by the kernel and shipped inside it.

Three artifacts cross product boundaries in this ecosystem: a recorded trace, a
tool manifest, and the Lab → Control Plane deploy package. Until now none of them
had a single owner. `trace` and `tool-manifest` were documented as
"axor-core-owned" while the only real definitions lived downstream in the Lab —
which kept a directory literally named `_shared_from_axor_core` holding narrower
stubs of files this package did not have. `cp-deploy` had no schema anywhere: its
producer and consumer were written independently in two repositories, and had
already drifted apart without anyone noticing.

They live here because this package is the one dependency every consumer already
has. `axor_core.contracts` has always held the shared vocabulary as Python types;
these are the same vocabulary in the serialization that crosses a process.

Validation uses the subset validator next door — not the `jsonschema` package.
The schemas are written to the subset it implements, and this package stays
dependency-free.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

__all__ = ["SCHEMA_NAMES", "SchemaInvalid", "load", "validate"]

SCHEMA_NAMES = ("trace", "tool-manifest", "cp-deploy")


class SchemaInvalid(ValueError):
    """An artifact failed validation. `errors` lists every problem, not the first.

    A validator that stops at the first error makes a caller fix one field, run
    again, and find the next — the contract these schemas describe is that a
    rejection is complete.
    """

    def __init__(self, schema_name: str, errors: list[str]) -> None:
        self.schema_name = schema_name
        self.errors = list(errors)
        super().__init__(f"{schema_name}: " + "; ".join(errors))


def load(name: str) -> dict[str, Any]:
    """One schema by name, e.g. ``load("cp-deploy")``."""
    if name not in SCHEMA_NAMES:
        raise KeyError(f"unknown schema {name!r}; have {list(SCHEMA_NAMES)}")
    text = (
        resources.files(__package__)
        .joinpath(f"{name}.schema.json")
        .read_text("utf-8")
    )
    return json.loads(text)


def validate(name: str, document: Any) -> list[str]:  # noqa: ANN401 - untrusted input
    """Every way `document` violates schema `name`, or an empty list.

    Returns rather than raises, so a caller can merge these into its own reason
    list — a consumer of one of these artifacts almost always has checks the
    schema cannot express (a storage key's width, a hash recomputed over the
    payload) and must report them together.
    """
    from ._subset_validator import validate_against  # noqa: PLC0415

    return validate_against(document, name, {n: load(n) for n in SCHEMA_NAMES})
