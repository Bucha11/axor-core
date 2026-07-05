"""Kernel error types. Stdlib-only, like everything under axor_core.kernel."""
from __future__ import annotations


class KernelError(Exception):
    """Base for all kernel errors."""


class SchemaVersionError(KernelError):
    """Trace schema major version is unknown; refuse rather than best-effort parse."""


class InvariantViolation(KernelError):
    """An internal kernel invariant was broken; always a bug, never user input."""
