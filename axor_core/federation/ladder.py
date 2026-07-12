"""Inter-federation trust ladder (spec v2 Ch.1 §2) — foreign labels are CLAIMS.

This module is the spec-v2 alignment of federation semantics for peers under a
DIFFERENT operator keyset. The existing :mod:`axor_core.federation.gateway`
restore path ("trust the peer's labels") is only legitimate where the keyset is
ours — intra-federation, where labels are data. Across keysets, declaration
buys DISCOUNT, never label authority (decision v2-2):

  L0 (default, undeclared)  full taint on everything inbound; the peer is an
                            untrusted export destination outbound.
  L1 (authenticated)        identity verified — reputation can accrue to a
                            stable peer. Inbound taint UNCHANGED: L1 buys
                            attribution, not trust.
  L2 (federated agreement)  the peer's signed label assertions are accepted as
                            EVIDENCE with a policy-declared discount — bounded
                            steps per (peer, message class), never to clean,
                            never affecting the confidentiality floor.
  + governance_attested     the peer proves it runs under axor-core (signed
                            kernel version + config hash): trust in MECHANISM,
                            higher declared discounts — still not authority.

Critical sinks ignore discounts entirely (decision v2-3):
:func:`effective_root_for_sink` returns the undiscounted root for a critical
sink, making criticality non-negotiable by the peer.

A forged assertion falls to L0 handling and the fall is EVIDENCED (an event,
not a silent downgrade) — spec v2 Ch.1 eval scenarios.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from axor_core.contracts.taint import TaintSource
from axor_core.federation.signing import Verifier
from axor_core.taint.causal_root import CausalRoot

L0 = "l0"
L1 = "l1"
L2 = "l2"
_LEVELS = (L0, L1, L2)


@dataclass(frozen=True)
class PeerDeclaration:
    """Operator-declared foreign peer (config-resident, like a sink).

    Undeclared peers do not get a declaration object at all — they ARE L0.
    ``discount_classes`` lists the message classes the L2 discount applies to;
    ``governance_attested`` marks a verified kernel+config-hash attestation
    (fact of governance only, decision v2-6).
    """

    peer_id: str
    level: str = L0
    verifier: Verifier | None = None
    discount_classes: frozenset[str] = frozenset()
    governance_attested: bool = False

    def __post_init__(self) -> None:
        if self.level not in _LEVELS:
            raise ValueError(f"unknown trust level {self.level!r}")
        if self.level in (L1, L2) and self.verifier is None:
            raise ValueError(f"{self.level} requires a verifier (authenticated identity)")


@dataclass(frozen=True)
class LabelAssertion:
    """A peer's signed claim about a value's labels (L2 evidence envelope).
    Native-protocol-only (decision v2-4); MCP-as-A2A channels never carry one."""

    peer_id: str
    message_class: str
    sources: tuple[str, ...]
    sensitive: bool
    payload: bytes  # canonical bytes the signature covers
    signature: bytes


@dataclass(frozen=True)
class InboundVerdict:
    """The result of receiving a foreign value: the local root to register,
    the undiscounted root (critical sinks use this), and the evidence trail."""

    root: CausalRoot
    full_root: CausalRoot
    level: str
    discounted: bool
    evidence: tuple[str, ...] = field(default_factory=tuple)


def _full_taint(sensitive: bool) -> CausalRoot:
    base = CausalRoot.cross_process_in()
    if sensitive:
        return CausalRoot(sources=base.sources, sensitive=True)
    return base


def receive_foreign(
    value_sensitive_hint: bool,
    declaration: PeerDeclaration | None,
    assertion: LabelAssertion | None = None,
) -> InboundVerdict:
    """Derive the local root for a value arriving from a FOREIGN keyset.

    The foreign causal_root is never grafted onto ours — a local root is
    minted; whatever the peer asserted rides along only as bounded discount
    evidence. ``value_sensitive_hint`` is OUR channel-level knowledge (e.g.
    the channel is declared secret-bearing); a peer's "not sensitive" claim
    can never clear it (the floor is not negotiable by the peer).
    """
    full = _full_taint(value_sensitive_hint)
    if declaration is None:
        return InboundVerdict(full, full, L0, False, ("undeclared_peer_l0",))
    if declaration.level == L0 or assertion is None:
        return InboundVerdict(full, full, declaration.level, False)

    # L1/L2: the assertion must verify against the declared identity. A forged
    # assertion FALLS to L0 handling, evidenced — never silently accepted,
    # never a hard crash that loses the value's audit trail.
    if declaration.verifier is None or not declaration.verifier.verify(
        assertion.payload, assertion.signature
    ):
        return InboundVerdict(
            full, full, L0, False,
            (f"assertion_forged_fell_to_l0:{declaration.peer_id}",),
        )

    if declaration.level == L1:
        # Attribution, not trust: identity verified, taint unchanged.
        return InboundVerdict(full, full, L1, False, ("l1_attribution_only",))

    # L2: bounded discount for declared message classes only.
    if assertion.message_class not in declaration.discount_classes:
        return InboundVerdict(
            full, full, L2, False,
            (f"class_not_discounted:{assertion.message_class}",),
        )
    asserted = frozenset(
        TaintSource(s) for s in assertion.sources
        if s in TaintSource._value2member_map_
    )
    # Never to clean: an empty asserted source set keeps the untrusted re-mint
    # source. Never affecting the confidentiality floor: our sensitivity hint
    # (and the peer's own sensitive=True) survive; sensitive=False from the
    # peer cannot clear our hint.
    sources = asserted if asserted else frozenset({TaintSource.UNKNOWN_EXTERNAL})
    sensitive = value_sensitive_hint or assertion.sensitive
    discounted = CausalRoot(sources=sources, sensitive=sensitive)
    evidence = ("l2_discount_applied",) + (
        ("governance_attested",) if declaration.governance_attested else ()
    )
    return InboundVerdict(discounted, full, L2, True, evidence)


@dataclass(frozen=True)
class GovernanceAttestation:
    """A peer's signed proof it runs under axor-core: kernel version + config
    hash (fact of governance only, decision v2-6 — the hash pins
    accountability, never semantics; foreign configs are not parsed)."""

    peer_id: str
    kernel_version: str
    config_hash: str
    signature: bytes

    def payload(self) -> bytes:
        return (
            f"governance{self.peer_id}{self.kernel_version}"
            f"{self.config_hash}"
        ).encode()


@dataclass(frozen=True)
class ChannelGrant:
    """The result of establishing a peer channel (protocol v0.2 §6a):
    the EFFECTIVE level the channel is pinned at, and why."""

    peer_id: str
    level: str
    governance_attested: bool
    config_hash: str | None
    evidence: tuple[str, ...] = field(default_factory=tuple)


def establish_channel(
    declaration: PeerDeclaration | None,
    *,
    transport: str = "native",
    attestation: GovernanceAttestation | None = None,
) -> ChannelGrant:
    """Channel establishment verifies the peer declaration from local config
    (protocol v0.2 §6a). Undeclared = L0. MCP-as-A2A is pinned L0/L1 at
    establishment (decision v2-4) — the L2 assertion envelope arrives with the
    native protocol only. ``governance_attested`` requires a VALID signed
    kernel+config-hash attestation; verification is repeated on config-hash
    change by re-establishing (the grant records the hash it verified)."""
    if declaration is None:
        return ChannelGrant("", L0, False, None, ("undeclared_peer_l0",))

    level = declaration.level
    evidence: list[str] = []
    if transport == "mcp" and level == L2:
        level = L1
        evidence.append("mcp_transport_pinned_l1")

    attested = False
    config_hash: str | None = None
    if declaration.governance_attested:
        if (
            attestation is not None
            and declaration.verifier is not None
            and attestation.peer_id == declaration.peer_id
            and declaration.verifier.verify(
                attestation.payload(), attestation.signature
            )
        ):
            attested = True
            config_hash = attestation.config_hash
            evidence.append("governance_attestation_verified")
        else:
            # A declared-attested peer that cannot prove it: the channel still
            # establishes at the declared level, but WITHOUT the attested
            # standing (lower discounts apply) — and the failure is evidenced.
            evidence.append("governance_attestation_failed")
    return ChannelGrant(
        declaration.peer_id, level, attested, config_hash, tuple(evidence)
    )


def effective_root_for_sink(
    verdict: InboundVerdict, sink_is_critical: bool
) -> CausalRoot:
    """Critical sinks ignore L2 discounts entirely (decision v2-3): a discount
    that could reach a critical sink would make criticality negotiable by the
    peer. Pure predicate — shared by runtime and replay (Rule 0)."""
    return verdict.full_root if sink_is_critical else verdict.root
