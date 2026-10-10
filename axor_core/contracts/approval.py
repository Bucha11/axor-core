"""ApprovalRequest — what an approver is asked to authorise, and at what scope.

An approval is *evidence*: the kernel accepts it as the governance gate for an
action it would otherwise deny. Evidence is only worth what it is bound to, and
the previous question handed to an approver — ``(tool_use_id, tool, paths,
max_ops)`` — bound to a **class**: "this tool, under these path roots, N times".
A human answering it cannot have authorised *this transfer*, because the amount
and the recipient were not in the question. Nor could the grant that followed
distinguish one call from another.

So scope is explicit:

* :attr:`AuthorityScope.CLASS` — the old shape, still the only one available when
  the request is made before the call exists (an ``ESCALATE_POLICY`` intent that
  names no arguments). Covers every call of the tool inside the grant's bounds,
  for ``max_ops`` uses.
* :attr:`AuthorityScope.EFFECT` — bound to one call by
  :func:`~axor_core.policy.effect.effect_digest`. The request carries the
  arguments, so the approver sees what will happen; the resulting grant is
  one-shot and matches only a call with the same digest. A different recipient,
  a different amount, a different anything is a different effect and is not
  covered.

The arguments on an EFFECT request are raw values, by necessity: an approver that
cannot see them cannot answer "did the principal order THIS". They are untrusted
content — an approver must render them as data, never execute, follow or
interpret them as instruction. That is the honest cost of effect-level authority,
and it is why the kernel runs every other gate FIRST and asks last: an approver
is only ever shown a call that the structural gates already admit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping


class AuthorityScope(str, Enum):
    CLASS = "class"    # the tool, within bounds, for max_ops uses
    EFFECT = "effect"  # one call, bound by effect digest, one use


@dataclass(frozen=True)
class ApprovalRequest:
    """The question put to an ``EscalationCallback``.

    ``args`` and ``effect_digest`` are populated only for an EFFECT-scoped
    request; ``paths`` / ``max_ops`` only carry meaning for a CLASS-scoped one.
    """

    tool_use_id: str
    tool: str
    scope: AuthorityScope = AuthorityScope.CLASS
    paths: tuple[str, ...] = field(default_factory=tuple)
    max_ops: int = 1
    reason: str = ""
    # EFFECT scope only.
    args: Mapping[str, object] = field(default_factory=dict)
    effect_digest: str = ""
    # The arguments the operator declared effect-determining for this sink
    # (driving_args), for an approver that wants to show or check them. Empty
    # means undeclared — the digest then covers every argument of the call.
    effect_signature: tuple[str, ...] = field(default_factory=tuple)
    # The action class the consequence gate computed, for an approver that wants
    # to treat an irreversible action differently. A name, not a number.
    consequence: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.args, MappingProxyType):
            object.__setattr__(self, "args", MappingProxyType(dict(self.args)))

    @property
    def is_effect_scoped(self) -> bool:
        return self.scope == AuthorityScope.EFFECT
