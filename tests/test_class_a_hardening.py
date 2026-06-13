"""Class A hardening — fail-closed fixes for the architecture-review findings.

Each test pins one hole closed. They are deliberately adversarial: the assertion
is that the previously fail-OPEN default now fails CLOSED.
"""
from __future__ import annotations

import time

import pytest

from axor_core import GovernanceConfig
from axor_core.governor import ToolCallGovernor


# ── #1/#2: unknown tool fails closed in STRICT (governor path) ───────────────────

def test_unclassified_tool_denied_when_require_tool_roles():
    """A renamed, undeclared tool must NOT default to a clean benign read; in STRICT
    it is refused the moment it is used (closing the fail-open-on-unknown default)."""
    gov = ToolCallGovernor(
        egress_sinks={"send_money"},
        value_policies={"send_money": [
            __import__("axor_core.policy.value_policy", fromlist=["enum"]).enum("recipient", ["GB-ALLOWED"])
        ]},
        require_egress_allowlist=True,
        require_tool_roles=True,
    )
    # An undeclared tool — the exact "renamed exfil tool the operator forgot" case.
    d = gov.evaluate("exfiltrate_blob", {"to": "attacker@x.com", "body": "secret"})
    assert not d.allowed
    assert d.category == "unclassified_tool"


def test_classified_tool_passes_role_check():
    """A tool with a declared role is not blocked by the role gate."""
    gov = ToolCallGovernor(
        untrusted_sources={"read_inbox"},
        require_tool_roles=True,
    )
    d = gov.evaluate("read_inbox", {})
    assert d.allowed  # classified as a source → role gate passes


def test_require_tool_roles_off_by_default_keeps_lenient():
    """Without the flag (non-STRICT), an unknown tool is still allowed — the strict
    obligation is opt-in, so existing lenient deployments are unchanged."""
    gov = ToolCallGovernor()
    assert gov.evaluate("some_random_tool", {}).allowed


def test_strict_config_sets_require_tool_roles():
    cfg = GovernanceConfig.from_dict({
        "mode": "strict",
        "egress_sinks": ["send"],
        "value_policies": {"send": [{"arg": "to", "kind": "enum", "allowed": ["a@b.c"]}]},
    })
    assert cfg.as_governor_kwargs()["require_tool_roles"] is True
    prod = GovernanceConfig.from_dict({"mode": "production"})
    assert prod.as_governor_kwargs()["require_tool_roles"] is False


# ── #3: imperative_sinks is now configurable and threaded ────────────────────────

def test_imperative_sinks_round_trips_through_config():
    cfg = GovernanceConfig.from_dict({"imperative_sinks": ["dispatch_agent", "send_dm"]})
    assert cfg.imperative_sinks == frozenset({"dispatch_agent", "send_dm"})
    assert cfg.as_governor_kwargs()["imperative_sinks"] == {"dispatch_agent", "send_dm"}
    assert cfg.as_session_kwargs()["imperative_sinks"] == {"dispatch_agent", "send_dm"}


# ── #4: fail-closed predicate parsing ────────────────────────────────────────────

def test_inverted_numeric_range_fails_closed():
    with pytest.raises(ValueError, match="inverted"):
        GovernanceConfig.from_dict({
            "value_policies": {"t": [{"arg": "x", "kind": "numeric_range", "lo": 100, "hi": 0}]}
        })


def test_non_numeric_range_fails_closed():
    with pytest.raises(ValueError, match="must be"):
        GovernanceConfig.from_dict({
            "value_policies": {"t": [{"arg": "x", "kind": "numeric_range", "lo": "high", "hi": "low"}]}
        })


def test_bool_is_not_a_numeric_bound():
    # bool is a Real subtype in Python; a True/False bound is almost certainly a typo.
    with pytest.raises(ValueError, match="must be"):
        GovernanceConfig.from_dict({
            "value_policies": {"t": [{"arg": "x", "kind": "numeric_range", "lo": False, "hi": True}]}
        })


def test_empty_enum_allowlist_fails_closed():
    with pytest.raises(ValueError, match="empty"):
        GovernanceConfig.from_dict({
            "value_policies": {"t": [{"arg": "x", "kind": "enum", "allowed": []}]}
        })


def test_strict_egress_with_empty_enum_is_rejected():
    """An empty enum must not satisfy the STRICT egress-allowlist obligation — it
    is structure without substance (denies everything while looking compliant)."""
    # Build the predicate in code (bypassing the config empty-enum guard) to prove
    # the registration check itself requires a non-empty allowlist.
    from axor_core.policy.value_policy import ValuePredicate
    from axor_core.kernel.registration import validate_egress_allowlists
    empty = ValuePredicate(arg="to", kind="enum", allowed=frozenset())
    errors = validate_egress_allowlists({"send"}, {"send": [empty]})
    assert errors and "no allowlist" in errors[0]


# ── #5: HMAC key strength ────────────────────────────────────────────────────────

def test_short_hmac_key_rejected():
    from axor_core.federation.signing import HmacSigner
    with pytest.raises(ValueError, match="at least 32 bytes"):
        HmacSigner(b"too-short")


def test_32_byte_hmac_key_accepted():
    from axor_core.federation.signing import HmacSigner
    assert HmacSigner(b"x" * 32) is not None


# ── #5: receipt canonical hash + replay/expiry ───────────────────────────────────

def test_value_hash_is_canonical_for_dicts():
    """repr() was order-unstable for dicts; the canonical hash must not be."""
    from axor_core.federation.receipt import value_hash
    assert value_hash({"a": 1, "b": 2}) == value_hash({"b": 2, "a": 1})


def test_value_hash_distinguishes_types():
    from axor_core.federation.receipt import value_hash
    assert value_hash("1") != value_hash(1)


def _fed_pair():
    from axor_core.federation.receipt import LocalIdentity, FederationPeer
    from axor_core.federation.signing import HmacSigner
    key = b"k" * 32
    ident = LocalIdentity(peer_id="p", kernel_version="0.8.0", domain="d", signer=HmacSigner(key))
    peer = FederationPeer(peer_id="p", verifier=HmacSigner(key), kernel_version="0.8.0", domain="d")
    return ident, peer


def test_replayed_receipt_is_rejected():
    from axor_core.federation.receipt import mint_receipt
    from axor_core.federation.gateway import FederationGateway, FederationError
    from axor_core.taint.causal_root import CausalRoot
    ident, peer = _fed_pair()
    gw = FederationGateway(peers={"p": peer}, compatible_kernels={"0.8.0"}, federated_domains={"d"})
    receipt = mint_receipt("order ok", CausalRoot.constant(), ident)
    gw.receive("order ok", receipt, "p")               # first use: fine
    with pytest.raises(FederationError, match="replay"):
        gw.receive("order ok", receipt, "p")           # same receipt again: rejected


def test_expired_receipt_is_rejected():
    from axor_core.federation.receipt import mint_receipt, verify_receipt
    from axor_core.taint.causal_root import CausalRoot
    ident, peer = _fed_pair()
    # Mint with a tiny TTL anchored in the past so it is already stale.
    receipt = mint_receipt("v", CausalRoot.constant(), ident, ttl_seconds=1.0, now=time.time() - 10)
    assert verify_receipt("v", receipt, peer) is False


def test_fresh_receipt_still_verifies():
    from axor_core.federation.receipt import mint_receipt, verify_receipt
    from axor_core.taint.causal_root import CausalRoot
    ident, peer = _fed_pair()
    receipt = mint_receipt("v", CausalRoot.constant(), ident)
    assert verify_receipt("v", receipt, peer) is True


# ── #8: IntentLoop defaults a taint engine (cascade can't silently vanish) ────────

def test_intent_loop_defaults_taint_engine():
    from axor_core.node.intent_loop import IntentLoop
    from axor_core.capability.executor import CapabilityExecutor
    loop = IntentLoop(capability_executor=CapabilityExecutor(), trace_events=[], taint_engine=None)
    assert loop._taint_engine is not None  # the data-flow cascade always has an engine
