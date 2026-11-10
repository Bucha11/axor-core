"""ControlDirectives — the operator's one-shot effects, applied at a boundary.

`AdmissionController` lets the control plane hold or stop a node and nothing
else. Every other operator command — cut a context fragment the probe
localizer blamed, inject a test-bench fragment, lower the call budget — was
delivered to the node, verified, queued in the adapter's session, and then
never consumed: nothing in the kernel asked for it. A heal reported "excised"
over a context nobody had touched, and a budget cap tightened a number the
running loop never read again.

This is the kernel's side of that seam, under the same rule as admission
(spec 12.0): a Protocol with no imports of any plane. The adapter implements
it (axor-wrap's `PlaneAdmission`, over the session that verified each
command's signature); the kernel asks at its own boundaries and applies what
it is given with its own primitives — `context.excision.apply_excision`, the
context manager, the call cap. The plane still cannot widen anything: an
excision only removes, an injection is test-bench only (the adapter refuses
it otherwise), and a budget cap can only lower the cap the node started with.

Replan is deliberately absent: what "re-plan" means is framework-specific, and
a directive the kernel cannot apply honestly is one it should not accept.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from axor_core.contracts.degradation import GovernanceAuthority


@dataclass(frozen=True)
class ExcisionDirective:
    """Remove these context fragments, identified by their taint mark (the key
    the probe localizer and `apply_excision` share). The authority is the
    operator whose signed command this is — built by the adapter, which is the
    party that verified the signature, never by the worker path."""
    excision_id: str
    target_refs: tuple[str, ...]
    authority: GovernanceAuthority


@dataclass(frozen=True)
class InjectionDirective:
    """Add this text to context assembly for exactly one turn (test bench)."""
    injection_id: str
    text: str
    operator: str = ""


class ControlDirectives(Protocol):
    def take_excision(self, provenance: Mapping[str, str]) -> ExcisionDirective | None:
        """At most once per excision. `provenance` maps each fragment key the
        node currently holds to "operator_config" or "runtime"; an excision
        naming operator config is refused whole by the adapter (spec 8.2.1)."""
        ...

    def take_injection(self) -> InjectionDirective | None:
        """At most once per injection; None outside test bench."""
        ...

    def budget_cap_calls(self) -> int | None:
        """The operator's current call cap, or None for none. The kernel
        enforces the LOWER of this and its own configured cap."""
        ...
