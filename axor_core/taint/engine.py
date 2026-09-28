from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from axor_core.contracts.taint import INTEGRITY_DEFAULTS, TrustedOrigin
from axor_core.taint.causal_root import CausalRoot
from axor_core.taint.fingerprint import content_fingerprint
from axor_core.taint.ledger import ValueTaintLedger
from axor_core.taint.trusted import TrustedValueIndex
from axor_core.contracts.degradation import GovernanceAuthority
from axor_core.contracts.trace import (
    TaintClearanceAttemptedEvent,
    TaintClearedEvent,
    TraceEvent,
    TraceEventKind,
)
from axor_core.errors.exceptions import TaintClearanceError

log = logging.getLogger("axor.taint")

# Cap on distinct outstanding sensitive reads tracked by fingerprint. Past it the
# floor goes STICKY (forced active) instead of growing the map without bound — a
# flood of distinct secret reads bounds memory while failing CLOSED (the floor
# stays up). Only governance (clear_by_governance) resets it.
_MAX_OUTSTANDING_SECRETS = 4096

if TYPE_CHECKING:
    pass

# Authority types permitted to clear/endorse governance state. "worker" (and
# anything outside this set) is rejected — workers must never clear their own taint.
_VALID_GOVERNANCE_AUTHORITY_TYPES = frozenset({
    "human_operator",
    "automated_policy",
    "trusted_boundary",
})


def _is_valid_governance_authority(authority: GovernanceAuthority) -> bool:
    """A governance authority is valid when it is an unforgeable GovernanceAuthority
    capability (constructible only by the host, never materialised from a worker's
    tool-call args) carrying a non-empty principal, a non-empty reason, and an
    authority_type from the allowed set. The TYPE is the capability — a worker
    cannot fabricate one from JSON/strings — and the field checks reject a blank or
    worker-labelled instance."""
    return bool(
        isinstance(authority, GovernanceAuthority)
        and authority.authority_id
        and authority.reason_code
        and authority.authority_type in _VALID_GOVERNANCE_AUTHORITY_TYPES
    )


class TaintEngine:
    """
    Per-value taint tracker. Implements the ValueProvenance contract.

    Enforcement is per-value: a sink decides on the driving argument's own
    causal root (content-derivation ledger), not a session-wide flag. There is no
    session-taint state — provenance lives on values, released by governance
    endorsement (per value) or cleared wholesale.

    Integrity default for a model-generated value that carries no registered
    untrusted fragment (``integrity_default``):

    - ``"clean"`` (legacy): it derives ``constant()`` — trusted. A re-encoded copy
      of an untrusted identifier therefore passes the integrity gate.
    - ``"context"``: it carries this node's context root (every untrusted source
      registered so far) unless the trusted-origin index proves it equals a value
      the attacker cannot author (docs/rfc-integrity-context-default.md). Re-encoding
      no longer helps an attacker: any spelling that is not literally a trusted
      value is tainted. Confidentiality is unchanged in both modes.

    Thread-safety: not thread-safe. Each session has its own instance.
    """

    def __init__(self, node_id: str = "", integrity_default: str = "clean") -> None:
        if integrity_default not in INTEGRITY_DEFAULTS:
            raise ValueError(
                f"unknown integrity_default {integrity_default!r}; expected one of "
                f"{sorted(INTEGRITY_DEFAULTS)}"
            )
        self._node_id = node_id
        self._integrity_default = integrity_default
        self._pending_events: list[TraceEvent] = []
        self._ledger = ValueTaintLedger()
        # Context-default integrity. The context root is the join of every
        # untrusted source this node's model has been shown (every tainted value
        # registered here is a tool output / memory / child output / message the
        # model reads). Integrity sources only: confidentiality stays on the floor.
        # Per NODE, not per session — a session-wide root would taint every node
        # once any one of them read untrusted data. The trusted index is the
        # positive proof that lets a value escape the context root.
        self._context_root = CausalRoot.constant()
        self._trusted = TrustedValueIndex()
        # Session-wide SHADOW (observe-only, for the density comparison):
        # "has any tainted / any sensitive value ever been registered this session?"
        # This is what a coarse session-scoped model would gate on; it never feeds
        # `allow`, it only lets the density meter compare the session-wide flag
        # against per-value tracking honestly.
        self._session_any_tainted = False
        self._session_any_sensitive = False
        # Confidentiality floor (sound). An IDENTITY-BOUND registry of outstanding
        # sensitive reads: fingerprint(secret) -> count of un-released reads of THAT
        # exact secret. While non-empty the session is egress-restricted — soundly,
        # on the FACT of the read, content-blind (a paraphrased/re-encoded secret
        # cannot escape, unlike the per-value derivation gate). Keyed on a whole-
        # content fingerprint, NOT on the leaky ≥12-char ledger, so: (a) a secret
        # shorter than the ledger segment minimum still arms AND can be released
        # (the ledger can't represent it, the fingerprint can); (b) endorsing a
        # DIFFERENT value that merely shares a fragment cannot lift another secret's
        # floor (different fingerprint). This decouples the floor from the ledger's
        # derive(), removing the count/ledger desync.
        self._outstanding: dict[str, int] = {}
        # Sticky fail-closed flag: set once the outstanding map hits the cap, it
        # forces the floor active regardless of the map, so a flood of distinct
        # secret reads cannot grow memory without bound and endorsing the few tracked
        # secrets cannot lower the floor while untracked ones are still outstanding.
        self._floor_saturated = False

    # ── Per-value provenance (ValueProvenance) ────────────────────────────────

    @property
    def integrity_default(self) -> str:
        """``"clean"`` (legacy) or ``"context"`` — see the class docstring."""
        return self._integrity_default

    def register_value(self, content: object, root: CausalRoot) -> None:
        """Record that a value with the given causal_root produced this content."""
        self._ledger.register(content, root)
        if root.is_tainted:
            self._session_any_tainted = True
            self._context_root = CausalRoot.mint(
                self._context_root, CausalRoot(sources=root.sources)
            )
        if root.sensitive:
            self._session_any_sensitive = True
            # Arm the floor on the READ fact, keyed by the secret's fingerprint —
            # regardless of whether the ledger stored a fragment (sub-threshold
            # secrets still count, and remain releasable by fingerprint).
            fp = content_fingerprint(content)
            if fp in self._outstanding or len(self._outstanding) < _MAX_OUTSTANDING_SECRETS:
                self._outstanding[fp] = self._outstanding.get(fp, 0) + 1
            elif not self._floor_saturated:
                # Cap reached on a NEW secret: do not grow the map; go sticky so the
                # floor stays up for this (untracked) read. Fail-closed.
                self._floor_saturated = True
                log.warning(
                    "confidentiality floor saturated at %d distinct outstanding "
                    "secrets — floor forced active until governance clears it",
                    _MAX_OUTSTANDING_SECRETS,
                )

    def confidentiality_floor_active(self) -> bool:
        """Sound egress floor: True while a secret read is outstanding (not
        governance-released). An ENFORCEMENT input — unlike session_shadow. Once the
        outstanding map saturates it stays True (sticky) until governance clears."""
        return self._floor_saturated or bool(self._outstanding)

    def session_shadow(self) -> tuple[bool, bool]:
        """(any_tainted, any_sensitive) for the session-wide shadow model.

        Observe-only, for the density comparison. NOT an enforcement input —
        `allow` is per-value; this only exists to measure what a coarse
        session-scoped model would have done.
        """
        return (self._session_any_tainted, self._session_any_sensitive)

    def derive_value(self, value: object, *, include_scalars: bool = False) -> CausalRoot:
        """Per-value causal root of `value`.

        The ledger match attributes the untrusted/sensitive sources the value
        visibly carries. In ``"clean"`` mode that is the whole answer — a value with
        no match is trusted. In ``"context"`` mode, once this node's context holds
        untrusted data, a value is additionally joined with the context root unless
        every string leaf of it is a registered trusted value — and, with
        ``include_scalars`` (an integrity sink's driving args), every number too.
        """
        matched = self._ledger.derive(value)
        if self._integrity_default != "context" or not self._context_root.is_tainted:
            return matched
        if self._trusted.covers(value, include_scalars=include_scalars):
            return matched
        return CausalRoot.mint(matched, self._context_root)

    # ── Context-default integrity (ContextProvenance) ─────────────────────────

    def derive_carried(self, value: object) -> CausalRoot:
        """Only what ``value`` visibly carries (the ledger match), without the
        context root — the legacy label, in either mode. For a decision whose
        recipient provably inherits this node's context root, where joining it
        again would only duplicate a gate the recipient already runs."""
        return self._ledger.derive(value)

    def register_trusted(self, content: object, origin: TrustedOrigin) -> None:
        """Record values of ``content`` as having an origin the attacker cannot
        author. Only the kernel's own wiring and the host call this — a worker has
        no path to it (it is not a tool, and tool args never reach it)."""
        self._trusted.register(content, TrustedOrigin(origin))

    def context_root(self) -> CausalRoot:
        """Join of the untrusted sources this node's model has been shown."""
        return self._context_root

    def trusted_origin(self, value: object) -> TrustedOrigin | None:
        """The origin that makes ``value`` trusted, or None."""
        return self._trusted.origin_of(value)

    def is_trusted(self, value: object, *, include_scalars: bool = False) -> bool:
        """Whether ``value`` escapes the context root: every string leaf (and, with
        ``include_scalars``, every number) is a trusted value; vacuously true for a
        value with no checked leaf."""
        return self._trusted.covers(value, include_scalars=include_scalars)

    def inherit_value_ledger(self, parent: "TaintEngine") -> None:
        """Inherit the parent's per-value provenance into this (child) engine so
        the child's per-value gate sees values the parent marked tainted/sensitive."""
        self._ledger.merge(parent._ledger)
        # Context-default integrity: the child's model is launched from the
        # parent's context (its task is written by the parent's model), so it
        # starts with the parent's context root and never in a weaker mode. The
        # parent's trusted values (its user task, operator config, trusted tool
        # outputs) are still attacker-independent in the child.
        parent_root = getattr(parent, "_context_root", None)
        if parent_root is not None:
            self._context_root = CausalRoot.mint(self._context_root, parent_root)
        parent_trusted = getattr(parent, "_trusted", None)
        if parent_trusted is not None:
            self._trusted.merge(parent_trusted)
        if getattr(parent, "_integrity_default", "clean") == "context":
            self._integrity_default = "context"
        # Inherit the session-wide shadow too, so child density measurement is
        # comparable to the parent's (observe-only).
        self._session_any_tainted = self._session_any_tainted or parent._session_any_tainted
        self._session_any_sensitive = (
            self._session_any_sensitive or parent._session_any_sensitive
        )
        # Inherit the confidentiality floor: a child of a session that read a secret
        # is egress-restricted too (else the child is a floor bypass). Merge per
        # fingerprint so distinct secrets are not conflated and re-inheritance does
        # not double-count a secret the child already carries.
        for fp, count in parent._outstanding.items():
            self._outstanding[fp] = max(self._outstanding.get(fp, 0), count)
        # A saturated parent floor is inherited sticky (else the child is a bypass).
        self._floor_saturated = self._floor_saturated or parent._floor_saturated

    def drain_events(self) -> list[TraceEvent]:
        """Return and clear pending trace events for the trace collector."""
        events = list(self._pending_events)
        self._pending_events.clear()
        return events

    # ── Governance release ────────────────────────────────────────────────────

    def attempt_clear_by_worker(self) -> None:
        """Workers may not clear taint. Always raises TaintClearanceError."""
        self._pending_events.append(TaintClearanceAttemptedEvent(
            kind=TraceEventKind.TAINT_CLEARANCE_ATTEMPTED,
            node_id=self._node_id,
            sequence=len(self._pending_events),
            attempted_by="worker",
        ))
        raise TaintClearanceError(
            "worker attempted to clear taint — only governance may do this"
        )

    def clear_by_governance(self, authority: GovernanceAuthority) -> None:
        """Clear ALL per-value provenance under governance authority.

        Requires an unforgeable :class:`GovernanceAuthority` capability — a worker
        cannot materialise one from its tool-call args, so this path is not
        worker-reachable, and the object's fields are validated besides. For
        releasing a single value, use :meth:`endorse_value`.
        """
        if not _is_valid_governance_authority(authority):
            self._pending_events.append(TaintClearanceAttemptedEvent(
                kind=TraceEventKind.TAINT_CLEARANCE_ATTEMPTED,
                node_id=self._node_id,
                sequence=len(self._pending_events),
                attempted_by=getattr(authority, "authority_id", "") or "unknown",
            ))
            raise TaintClearanceError(
                "taint clearance rejected: requires a valid GovernanceAuthority "
                f"capability (got {authority!r})"
            )
        self._ledger = ValueTaintLedger()
        self._session_any_tainted = False
        self._session_any_sensitive = False
        # Governance attests the context is released; the trusted index is kept —
        # its values were attacker-independent before the clearance and still are.
        self._context_root = CausalRoot.constant()
        self._outstanding = {}
        self._floor_saturated = False
        self._pending_events.append(TaintClearedEvent(
            kind=TraceEventKind.TAINT_CLEARED,
            node_id=self._node_id,
            sequence=len(self._pending_events),
            cleared_by=authority.authority_id,
            authority_type=authority.authority_type,
            reason_code=authority.reason_code,
            audit_id=authority.audit_id,
        ))

    def endorse_value(self, content: object, authority: GovernanceAuthority) -> int:
        """Governed structural release of one specific value.

        Removes the value's fragments from the per-value ledger so ``derive_value``
        no longer flags it, AND lifts the confidentiality floor for THAT exact
        secret (by fingerprint). Attests release of this value/lineage (schema/
        transform/bounded use), NOT a semantic "safe" judgement, and NOT the whole
        session. Requires an unforgeable :class:`GovernanceAuthority`. Returns
        ledger fragments released.
        """
        if not _is_valid_governance_authority(authority):
            raise TaintClearanceError(
                "endorsement rejected: requires a valid GovernanceAuthority "
                f"capability (got {authority!r})"
            )
        # Lift the floor for THIS exact secret, identity-bound by fingerprint —
        # governance names the value it releases. A different value (even one that
        # shares a ledger fragment) has a different fingerprint and cannot lift
        # another secret's floor; a sub-threshold secret the ledger never stored is
        # still releasable here. This is what keeps the floor and the ledger from
        # desynchronising — the floor is released by identity, not by derive().
        self._outstanding.pop(content_fingerprint(content), None)
        removed = self._ledger.unregister(content)
        # Context mode: endorsement is positive declassification — the endorsed
        # value becomes a trusted value, so a later sink carrying it is not tainted
        # by the context root either. (In "clean" mode unregistering already
        # suffices; registering is harmless there and keeps the modes consistent.)
        self._trusted.register(content, TrustedOrigin.ENDORSED)
        self._pending_events.append(TaintClearedEvent(
            kind=TraceEventKind.TAINT_CLEARED,
            node_id=self._node_id,
            sequence=len(self._pending_events),
            cleared_by=authority.authority_id,
            authority_type=authority.authority_type,
            reason_code=f"endorsement:{authority.reason_code}",
            audit_id=authority.audit_id,
        ))
        return removed
