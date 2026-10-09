"""Declarative GovernanceConfig — parsing, fail-closed validation, and wiring."""
from __future__ import annotations

import os

import pytest

from axor_core import GovernanceConfig, GovernedSession
from axor_core.capability.executor import CapabilityExecutor
from axor_core.contracts.canonical import ConsequenceClass
from axor_core.contracts.mode import ExecutionMode
from axor_core.contracts.trace import TraceConfig
from tests.conftest import EchoExecutor

_EXAMPLE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "examples", "config", "governance.yaml"
)


# ── parsing ───────────────────────────────────────────────────────────────────

def test_from_dict_minimal():
    cfg = GovernanceConfig.from_dict({"mode": "production", "egress_sinks": ["send"]})
    assert cfg.mode == ExecutionMode.PRODUCTION
    assert cfg.egress_sinks == frozenset({"send"})


def test_from_dict_full_roundtrip():
    cfg = GovernanceConfig.from_dict({
        "mode": "strict",
        "untrusted_sources": ["search"],
        "sensitive_sources": ["read_creds"],
        "egress_sinks": ["send_email"],
        "positional_sinks": ["set_thermostat"],
        "benign_tools": ["get_time"],
        "value_policies": {
            "send_email": [{"arg": "to", "kind": "enum", "allowed": ["a@b.com"]}],
            "transfer": [{"arg": "amount", "kind": "numeric_range", "lo": 0, "hi": 100}],
        },
        "consequence_overrides": {"drop_db": "catastrophic"},
    })
    assert cfg.value_policies["send_email"][0].kind == "enum"
    assert cfg.value_policies["transfer"][0].kind == "numeric_range"
    assert cfg.consequence_overrides["drop_db"] == ConsequenceClass.CATASTROPHIC


def test_example_yaml_loads():
    cfg = GovernanceConfig.from_yaml(_EXAMPLE)
    assert cfg.mode == ExecutionMode.STRICT
    assert "send_email" in cfg.egress_sinks
    assert "read_credentials" in cfg.sensitive_sources
    # every egress sink in the example has an enum allowlist
    for sink in cfg.egress_sinks:
        assert any(p.kind == "enum" for p in cfg.value_policies.get(sink, []))


# ── fail-closed validation ────────────────────────────────────────────────────

def test_unknown_top_level_key_fails_closed():
    with pytest.raises(ValueError, match="unknown governance config key"):
        GovernanceConfig.from_dict({"egress_sink": ["send"]})  # typo: missing 's'


def test_unknown_mode_fails_closed():
    with pytest.raises(ValueError, match="unknown mode"):
        GovernanceConfig.from_dict({"mode": "ultra"})


def test_unknown_predicate_kind_fails_closed():
    with pytest.raises(ValueError, match="unknown predicate kind"):
        GovernanceConfig.from_dict({
            "value_policies": {"t": [{"arg": "x", "kind": "regex", "pattern": ".*"}]}
        })


def test_enum_without_allowed_fails_closed():
    with pytest.raises(ValueError, match="needs an 'allowed' list"):
        GovernanceConfig.from_dict({
            "value_policies": {"t": [{"arg": "x", "kind": "enum"}]}
        })


def test_unknown_consequence_class_fails_closed():
    with pytest.raises(ValueError, match="unknown class"):
        GovernanceConfig.from_dict({"consequence_overrides": {"t": "apocalyptic"}})


def test_unknown_predicate_field_fails_closed():
    with pytest.raises(ValueError, match="unknown enum field"):
        GovernanceConfig.from_dict({
            "value_policies": {"t": [
                {"arg": "x", "kind": "enum", "allowed": ["a"], "typo_field": 1}
            ]}
        })


def test_agentdojo_suite_configs_load_and_build_governors():
    """The benchmark's per-suite YAMLs are real GovernanceConfigs and construct a
    working ToolCallGovernor via as_governor_kwargs (the second enforcement path)."""
    import glob

    from axor_core.governor import ToolCallGovernor

    config_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "examples", "agentdojo", "config"
    )
    paths = sorted(glob.glob(os.path.join(config_dir, "*.yaml")))
    assert len(paths) >= 4  # banking, slack, travel, workspace
    for path in paths:
        cfg = GovernanceConfig.from_yaml(path)
        assert cfg.untrusted_sources and cfg.egress_sinks
        governor = ToolCallGovernor(**cfg.as_governor_kwargs())
        assert governor is not None


def test_require_egress_allowlist_is_declarable_on_its_own():
    """STRICT's two obligations are separable, and the YAML can say so.

    The origin posture is STRICT tool roles WITHOUT the enum-allowlist obligation:
    the integrity/origin axis decides each driving value, and declaring an enum
    there would be a different control (a closed codomain triggers decidable
    supersession, skipping the integrity check the posture exists to measure).
    That used to live only as an imperative override in the benchmark runner, so
    `as_governor_kwargs()` built a configuration nobody runs — which is exactly
    how the four origin taxonomies came to fail
    `test_agentdojo_suite_configs_load_and_build_governors`.
    """
    origin = GovernanceConfig.from_dict({
        "mode": "strict",
        "egress_sinks": ["send"],
        "require_egress_allowlist": False,
    })
    kwargs = origin.as_governor_kwargs()
    assert kwargs["require_tool_roles"] is True       # STRICT's other obligation stays
    assert kwargs["require_egress_allowlist"] is False

    # production can opt IN, too — the flag is a declaration, not a mode alias.
    opted_in = GovernanceConfig.from_dict({
        "mode": "production",
        "egress_sinks": ["send"],
        "require_egress_allowlist": True,
    })
    assert opted_in.as_governor_kwargs()["require_egress_allowlist"] is True

    # omitted = the mode's default
    assert GovernanceConfig.from_dict({
        "mode": "strict", "egress_sinks": ["send"],
        "value_policies": {"send": [{"arg": "to", "kind": "enum", "allowed": ["a@b.c"]}]},
    }).as_governor_kwargs()["require_egress_allowlist"] is True


def test_require_egress_allowlist_rejects_a_non_boolean():
    import pytest as _pytest

    with _pytest.raises(ValueError, match="require_egress_allowlist"):
        GovernanceConfig.from_dict({"mode": "strict", "require_egress_allowlist": "no"})


def test_as_governor_kwargs_strict_maps_to_allowlist_obligation():
    strict = GovernanceConfig.from_dict({
        "mode": "strict",
        "egress_sinks": ["send"],
        "value_policies": {"send": [{"arg": "to", "kind": "enum", "allowed": ["a@b.c"]}]},
    })
    assert strict.as_governor_kwargs()["require_egress_allowlist"] is True
    prod = GovernanceConfig.from_dict({"mode": "production", "egress_sinks": ["send"]})
    assert prod.as_governor_kwargs()["require_egress_allowlist"] is False


# ── wiring into a session ─────────────────────────────────────────────────────

def test_from_config_builds_session():
    cfg = GovernanceConfig.from_dict({
        "mode": "strict",
        "egress_sinks": ["send_email"],
        "value_policies": {
            "send_email": [{"arg": "to", "kind": "enum", "allowed": ["a@b.com"]}]
        },
        "driving_args": {"send_email": ["to"]},
    })
    # empty capability executor → STRICT role-completeness check is skipped
    session = GovernedSession.from_config(
        EchoExecutor(), CapabilityExecutor(), cfg,
        trace_config=TraceConfig(local_only=True, persist_inputs=False),
    )
    assert session is not None


def test_from_config_strict_egress_without_allowlist_fails():
    cfg = GovernanceConfig.from_dict({"mode": "strict", "egress_sinks": ["send_email"]})
    with pytest.raises(ValueError, match="strict egress allowlist"):
        GovernedSession.from_config(
            EchoExecutor(), CapabilityExecutor(), cfg,
            trace_config=TraceConfig(local_only=True, persist_inputs=False),
        )


# ── federation (A2A) — keys by reference only, fail closed ────────────────────────

def _fed_cfg(monkeypatch):
    monkeypatch.setenv("FED_OURS", b"our-shared-secret-key".ljust(32, b"x").hex())
    monkeypatch.setenv("FED_PEER", b"peer-shared-secret-ab".ljust(32, b"x").hex())
    return GovernanceConfig.from_dict({
        "federation": {
            "compatible_kernels": ["0.8.0"],
            "federated_domains": ["payments.corp"],
            "identity": {
                "peer_id": "billing", "domain": "payments.corp", "kernel_version": "0.8.0",
                "algorithm": "hmac-sha256", "shared_key_env": "FED_OURS",
            },
            "peers": [
                {"peer_id": "fulfilment", "domain": "payments.corp", "kernel_version": "0.8.0",
                 "algorithm": "hmac-sha256", "shared_key_env": "FED_PEER"},
            ],
        }
    })


def test_federation_builds_gateway_and_identity(monkeypatch):
    cfg = _fed_cfg(monkeypatch)
    assert cfg.federation_gateway is not None
    assert cfg.federation_identity is not None
    assert "federation_gateway" in cfg.as_session_kwargs()


def test_federation_ingress_restores_known_peer(monkeypatch):
    cfg = _fed_cfg(monkeypatch)
    from axor_core.federation.receipt import LocalIdentity, mint_receipt
    from axor_core.federation.signing import HmacSigner
    from axor_core.taint.causal_root import CausalRoot
    peer = LocalIdentity(
        peer_id="fulfilment", kernel_version="0.8.0", domain="payments.corp",
        signer=HmacSigner(b"peer-shared-secret-ab".ljust(32, b"x")),
    )
    receipt = mint_receipt("order ok", CausalRoot.constant(), peer)
    _root, level = cfg.federation_gateway.receive("order ok", receipt, "fulfilment")
    assert level.name == "L2"  # provenance restored for a trusted, compatible peer


def test_federation_missing_key_fails_closed():
    with pytest.raises(ValueError, match="unset or empty"):
        GovernanceConfig.from_dict({"federation": {"peers": [
            {"peer_id": "x", "domain": "d", "kernel_version": "0.8.0",
             "algorithm": "hmac-sha256", "shared_key_env": "DEFINITELY_UNSET_VAR"}
        ]}})


def test_federation_inline_key_is_impossible():
    # there is no field to inline a key — an attempt is an unknown-field error
    with pytest.raises(ValueError, match="unknown field"):
        GovernanceConfig.from_dict({"federation": {"peers": [
            {"peer_id": "x", "domain": "d", "kernel_version": "0.8.0",
             "algorithm": "hmac-sha256", "shared_key": "deadbeef"}
        ]}})


def test_federation_unknown_algorithm_fails_closed():
    with pytest.raises(ValueError, match="unknown algorithm"):
        GovernanceConfig.from_dict({"federation": {"peers": [
            {"peer_id": "x", "domain": "d", "kernel_version": "0.8.0",
             "algorithm": "rsa-9000", "shared_key_env": "X"}
        ]}})


# ── integrity_default (context-default integrity, RFC step 2) ─────────────────

@pytest.mark.parametrize("mode, expected", [
    ("library", "clean"), ("production", "clean"), ("strict", "context"),
])
def test_integrity_default_follows_the_mode(mode, expected):
    """Unset, the mode decides: STRICT runs context-default integrity (RFC step 6),
    library/production keep the legacy default."""
    from axor_core import ToolCallGovernor
    cfg = GovernanceConfig.from_dict({"mode": mode})
    assert cfg.integrity_default is None
    gov = ToolCallGovernor(**cfg.as_governor_kwargs())
    assert gov._taint.integrity_default == expected
    sess = GovernedSession.from_config(
        EchoExecutor(), CapabilityExecutor(), cfg,
        trace_config=TraceConfig(local_only=True, persist_inputs=False),
    )
    assert sess._taint_engine.integrity_default == expected


def test_explicit_integrity_default_wins_over_the_mode():
    from axor_core import ToolCallGovernor
    cfg = GovernanceConfig.from_dict({"mode": "strict", "integrity_default": "clean"})
    assert ToolCallGovernor(**cfg.as_governor_kwargs())._taint.integrity_default == "clean"


def test_integrity_default_context_reaches_session_and_governor():
    from axor_core import ToolCallGovernor
    cfg = GovernanceConfig.from_dict({"integrity_default": "context"})
    gov = ToolCallGovernor(**cfg.as_governor_kwargs())
    assert gov._taint.integrity_default == "context"
    sess = GovernedSession.from_config(
        EchoExecutor(), CapabilityExecutor(), cfg,
        trace_config=TraceConfig(local_only=True, persist_inputs=False),
    )
    assert sess._taint_engine.integrity_default == "context"


def test_unknown_integrity_default_fails_closed():
    with pytest.raises(ValueError, match="integrity_default"):
        GovernanceConfig.from_dict({"integrity_default": "strict"})
