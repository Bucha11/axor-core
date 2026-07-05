"""Control-plane client — the adapter side of the advisory overlay.

The plane never enters the decision path (spec 12.0): everything here is
best-effort steering around local enforcement. :class:`PlaneSession` holds the
governance-correct semantics (signature verification against LOCAL operator
keys, version monotonicity, lattice merge, narrowing-only budget, at-most-once
injection/excision with the provenance guard). :class:`PlaneClient` is the
thin I/O wrapper (SSE out-dial + telemetry POST) behind the optional ``plane``
extra — axor-core keeps zero required dependencies.
"""
from axor_core.plane.session import AppliedEffect, PlaneSession

__all__ = ["AppliedEffect", "PlaneSession"]
