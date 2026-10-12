"""ApprovalRequest — what an approver is asked to authorise, and at what scope.

An approval is *evidence*: the kernel accepts it as the governance gate for an
action it would otherwise deny. Evidence is only worth what it is bound to, and
the question an approver used to be handed — ``(tool_use_id, tool, paths,
max_ops)`` — bound to a **class**: "this tool, under these path roots, N times".
A human answering it cannot have authorised *this transfer*, because the amount
and the recipient were not in the question. Nor could the grant that followed
distinguish one call from another.

So scope is explicit:

* :attr:`AuthorityScope.CLASS` — the old shape, and the only one available when
  the request is made before the call exists (an ``ESCALATE_POLICY`` intent that
  names no arguments). Covers every call of the tool inside the grant's bounds,
  for ``max_ops`` uses.
* :attr:`AuthorityScope.CALL` — bound to one call by
  :func:`~axor_core.policy.effect.call_digest` (the sink, the authority scope it
  was issued in, and every argument). The request carries the arguments, so the
  approver sees what will happen; the resulting evidence is one-shot and matches
  only a call with the same digest.

**The binding is to a call, not to an effect.** The same call against changed
external state can produce a different effect, so a digest cannot by itself keep
an authorisation current. Three obligations do, and they live with the mechanism
that consumes the evidence, not here:

1. the state-dependent gates are re-checked after the answer comes back (the
   confidentiality floor, degradation, budget, cancellation);
2. the evidence is claimed atomically with the decision to execute, so two
   concurrent calls cannot both spend it;
3. the argument set that was hashed is the one the handler receives.

**What an approval does and does not buy.** A CALL-scoped approval is a one-shot
ground to pass the *consequence* axis for that call. It is not authority over the
other axes: taint, carrier, SSRF, value policies and capability all still have to
pass on their own, before the question is even asked. Standing rules about which
argument *combinations* are permissible are a policy matter
(``tuple_enum`` / ``keyed_range``), and an approval neither needs nor replaces
them.

An approval request's ``args`` are raw, attacker-influenceable values: an
approver renders them as data and never follows them as instruction. Quoting
helps a human read them as data; it does not prove they will not act on text
inside them. A human approver remains a *trusted party* in the issuing of
authority, and that trust is a premise of this mechanism, not a result of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping


class AuthorityScope(str, Enum):
    CLASS = "class"  # the tool, within bounds, for max_ops uses
    CALL = "call"    # one call, bound by its call digest, one use


@dataclass(frozen=True)
class ApprovalRequest:
    """The question put to an ``EscalationCallback``.

    ``args`` and ``call_digest`` are populated only for a CALL-scoped request;
    ``paths`` / ``max_ops`` only carry meaning for a CLASS-scoped one.
    """

    tool_use_id: str
    tool: str
    scope: AuthorityScope = AuthorityScope.CLASS
    paths: tuple[str, ...] = field(default_factory=tuple)
    max_ops: int = 1
    reason: str = ""
    # CALL scope only.
    args: Mapping[str, object] = field(default_factory=dict)
    call_digest: str = ""
    # The arguments the operator declared effect-determining for this sink
    # (driving_args) — the operator's statement of what matters here, shown to an
    # approver and used for nothing else. The digest covers every argument
    # regardless, so an empty or wrong signature cannot widen the binding.
    effect_signature: tuple[str, ...] = field(default_factory=tuple)
    # The action class the consequence gate computed, for an approver that wants
    # to treat an irreversible action differently. A name, not a number.
    consequence: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.args, MappingProxyType):
            object.__setattr__(self, "args", MappingProxyType(dict(self.args)))

    @property
    def is_call_scoped(self) -> bool:
        return self.scope == AuthorityScope.CALL
