"""PlaneSession: protocol v0.2 semantics, no I/O.

Everything that makes a compromised backend harmless lives here, adapter-side
(protocol sections 3-6, 8):

- Commands are verified end-to-end against operator pubkeys from LOCAL config;
  the channel's own auth is irrelevant to integrity.
- Version monotonicity: a delta with ``version <= applied_version`` is a no-op.
- Lattice: ``stopped`` absorbs later effects (reported ``noop_absorbed``).
- Budget caps are decrease-only AT THE ADAPTER (``rejected_widening``).
- Injections and excisions are one-shots, at-most-once by id; excisions pass
  the kernel provenance guard first (operator-config segments refuse the whole
  excision — deletion of a restriction is widening via deletion).

The host runtime polls :meth:`admit_intent` at the IntentLoop boundary and
drains :meth:`take_pending_injection` / :meth:`take_pending_excision` during
context assembly. Outbound acks accumulate in :attr:`outbox` for the transport
to flush (telemetry direction owns durability, protocol section 5).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from axor_core.kernel.jcs import canonicalize
from axor_core.kernel.state import Excision, Injection, excision_refused_refs


@dataclass(frozen=True)
class AppliedEffect:
    """What applying one downstream message did — mirrored upstream."""

    kind: str  # applied | noop_absorbed | noop_stale | rejected_widening | sig_invalid
    detail: str = ""


def _canonical(value: dict) -> bytes:
    """RFC 8785 canonical bytes for a signed payload — the one implementation
    the backend also uses (axor_core.kernel.jcs), so both sides agree byte for
    byte and floats are rejected consistently."""
    return canonicalize(value)


def _verify_ed25519(pubkey_hex: str, message: bytes, sig_hex: str) -> bool:
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "plane command verification needs the `cryptography` backend — "
            "install the plane extra: pip install 'axor-core[plane]'"
        ) from exc
    try:
        key = ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(pubkey_hex))
        key.verify(bytes.fromhex(sig_hex), message)
        return True
    except (InvalidSignature, ValueError):
        return False


@dataclass
class PlaneSession:
    """Adapter-local plane state for one node."""

    node_id: str
    # Operator pubkeys come from LOCAL adapter config, never over the channel
    # (protocol section 6) — a compromised backend must not swap keys.
    operator_pubkeys: dict[str, str] = field(default_factory=dict)
    test_bench: bool = False
    local_budget_cap: int | None = None

    applied_version: int = 0
    paused: bool = False
    stopped: bool = False
    budget_cap_calls: int | None = None
    _pending_injection: Injection | None = None
    _pending_excision: Excision | None = None
    _pending_replan: dict | None = None
    consumed_ids: set[str] = field(default_factory=set)
    outbox: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.budget_cap_calls = self.local_budget_cap

    # ── downstream application ────────────────────────────────────────────────

    def apply_snapshot(self, version: int, state: dict) -> AppliedEffect:
        """Snapshot on (re)subscribe: LWW makes replay trivially correct."""
        return self._apply(version, state, signed=False)

    def apply_delta(
        self, version: int, delta: dict, operator: str, timestamp: str, sig: str
    ) -> AppliedEffect:
        """A pushed command. Signature is verified against local keys over the
        canonical (node_id, version, delta, timestamp) — protocol section 6."""
        if self.operator_pubkeys:
            pubkey = self.operator_pubkeys.get(operator)
            message = _canonical({
                "node_id": self.node_id, "version": version,
                "body": delta, "timestamp": timestamp,
            })
            if pubkey is None or not _verify_ed25519(pubkey, message, sig):
                effect = AppliedEffect("sig_invalid", f"operator={operator!r}")
                self._report(effect, version)
                return effect
        return self._apply(version, delta, signed=True)

    def _apply(self, version: int, state: dict, signed: bool) -> AppliedEffect:
        if version <= self.applied_version:
            return AppliedEffect("noop_stale", f"version {version} already applied")
        self.applied_version = version

        if self.stopped:
            # Absorbing: accept the version, drop the effects.
            effect = AppliedEffect("noop_absorbed", "node is stopped")
            self._report(effect, version)
            return effect

        if "stopped" in state:
            self.stopped = bool(state["stopped"])
        if "paused" in state:
            self.paused = bool(state["paused"])

        if "budget_cap_calls" in state and state["budget_cap_calls"] is not None:
            requested = int(state["budget_cap_calls"])
            current = self.budget_cap_calls
            if current is not None and requested > current:
                # Narrowing-only, enforced HERE — not in backend validation.
                effect = AppliedEffect(
                    "rejected_widening",
                    f"cap {requested} > local cap {current}",
                )
                self._report(effect, version)
                return effect
            self.budget_cap_calls = requested

        if state.get("replan"):
            # Replan is a one-shot operator directive (not a lattice field): "drop
            # the current plan and reconsider" surfaced to the runtime at the next
            # intent boundary. Idempotent by id like injection/excision, so a
            # replayed snapshot after reconnect does not re-fire it.
            rp = state["replan"]
            self._pending_replan = {
                "id": str(rp.get("id", "")) if isinstance(rp, dict) else "",
                "reason": str(rp.get("reason", "")) if isinstance(rp, dict) else "",
                "operator": str(rp.get("operator", "")) if isinstance(rp, dict) else "",
            }
        if state.get("pending_injection"):
            inj = state["pending_injection"]
            self._pending_injection = Injection(
                injection_id=str(inj.get("id", inj.get("injection_id", ""))),
                text=str(inj.get("text", "")),
                reason=str(inj.get("reason", "")),
                operator=str(inj.get("operator", "")),
                sig=str(inj.get("sig", "")),
            )
        if state.get("pending_excision"):
            exc = state["pending_excision"]
            self._pending_excision = Excision(
                excision_id=str(exc.get("id", exc.get("excision_id", ""))),
                target_refs=tuple(exc.get("target_refs", ())),
                reason=str(exc.get("reason", "")),
                operator=str(exc.get("operator", "")),
                sig=str(exc.get("sig", "")),
            )

        effect = AppliedEffect("applied")
        self._report(effect, version)
        return effect

    # ── IntentLoop boundary ───────────────────────────────────────────────────

    def admit_intent(self) -> bool:
        """Polled at the IntentLoop boundary only — never mid-effect."""
        return not (self.paused or self.stopped)

    def take_pending_injection(self) -> Injection | None:
        """At-most-once by id (protocol section 4). Returns the injection to
        add to context assembly for exactly one turn, or None."""
        inj = self._pending_injection
        if inj is None:
            return None
        self._pending_injection = None
        if inj.injection_id in self.consumed_ids:
            return None
        if self.stopped:
            # A stopped node absorbs later effects (the lattice, spec 8) — an
            # injection is one of them. Refuse it rather than mutating a context
            # the node will never act on again.
            self.outbox.append({
                "kind": "injection_refused",
                "payload": {"id": inj.injection_id, "reason": "stopped"},
            })
            return None
        if not self.test_bench:
            self.outbox.append({
                "kind": "injection_refused",
                "payload": {"id": inj.injection_id, "reason": "not test-bench"},
            })
            return None
        self.consumed_ids.add(inj.injection_id)
        self.outbox.append({
            "kind": "injection_consumed", "payload": {"id": inj.injection_id},
        })
        return inj

    def take_pending_excision(
        self, provenance: dict[str, str]
    ) -> Excision | None:
        """One-shot excision behind the kernel provenance guard (spec 8.2.1):
        any operator-config target refuses the WHOLE excision."""
        exc = self._pending_excision
        if exc is None:
            return None
        self._pending_excision = None
        if exc.excision_id in self.consumed_ids:
            return None
        refused = excision_refused_refs(exc, provenance)
        if refused:
            self.outbox.append({
                "kind": "excision_refused",
                "payload": {
                    "id": exc.excision_id,
                    "reason": "operator_config_provenance",
                    "refs": list(refused),
                },
            })
            return None
        self.consumed_ids.add(exc.excision_id)
        self.outbox.append({
            "kind": "context_excision",
            "payload": {
                "id": exc.excision_id,
                "refs": list(exc.target_refs),
                "provenance": {r: provenance.get(r, "runtime") for r in exc.target_refs},
                "operator": exc.operator,
                "reason": exc.reason,
            },
        })
        return exc

    def take_pending_replan(self) -> dict | None:
        """One-shot replan directive for the runtime to consume at the intent
        boundary (at most once by id). A stopped node absorbs it. Returns the
        directive ({id, reason, operator}) or None."""
        rp = self._pending_replan
        if rp is None:
            return None
        self._pending_replan = None
        rid = rp.get("id", "")
        if rid and rid in self.consumed_ids:
            return None
        if self.stopped:
            self.outbox.append({
                "kind": "replan_refused",
                "payload": {"id": rid, "reason": "stopped"},
            })
            return None
        if rid:
            self.consumed_ids.add(rid)
        self.outbox.append({"kind": "replan_applied", "payload": {"id": rid}})
        return rp

    # ── upstream ──────────────────────────────────────────────────────────────

    def heartbeat(self, level: str, budget_remaining: int | None) -> dict:
        return {
            "kind": "heartbeat",
            "payload": {
                "applied_version": self.applied_version,
                "consumed_injection_ids": sorted(self.consumed_ids),
                "level": level,
                "budget_remaining": budget_remaining,
            },
        }

    def _report(self, effect: AppliedEffect, version: int) -> None:
        if effect.kind != "applied":
            self.outbox.append({
                "kind": "state_applied",
                "payload": {"version": version, "result": effect.kind,
                            "detail": effect.detail},
            })
