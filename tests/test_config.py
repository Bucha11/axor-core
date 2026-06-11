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
    os.path.dirname(os.path.dirname(__file__)), "examples", "governance.yaml"
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


# ── wiring into a session ─────────────────────────────────────────────────────

def test_from_config_builds_session():
    cfg = GovernanceConfig.from_dict({
        "mode": "strict",
        "egress_sinks": ["send_email"],
        "value_policies": {
            "send_email": [{"arg": "to", "kind": "enum", "allowed": ["a@b.com"]}]
        },
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
