"""Shared conformance vectors, shipped inside the kernel package.

A vector file is only shared if every consumer can read the SAME one. Reaching
into another product's checkout is not that — it breaks the moment a consumer is
installed from a wheel, which is how the Lab and the Control Plane consume this
kernel. So the vectors travel with the package and are loaded by import.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

__all__ = ["TAINT_FLOOR", "load"]


def load(name: str) -> dict[str, Any]:
    """Read one vector document by name (without the .json suffix)."""
    text = resources.files(__package__).joinpath(f"{name}.json").read_text("utf-8")
    return json.loads(text)


def TAINT_FLOOR() -> dict[str, Any]:  # noqa: N802 - a named document, not a constant
    """The taint-floor vectors: what the gate decides, for every consumer."""
    return load("taint_floor")
