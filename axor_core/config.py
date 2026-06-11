"""Declarative governance configuration.

Instead of passing the data-flow taxonomy as a spray of keyword arguments, an
operator can describe it once in a YAML file and load it:

    from axor_core import GovernanceConfig, GovernedSession

    config = GovernanceConfig.from_yaml("governance.yaml")
    session = GovernedSession.from_config(executor, cap_executor, config)

See ``examples/governance.yaml`` for a fully-commented template.

Design notes:

- This module is pure and dependency-free; YAML parsing is lazy (``from_yaml``
  imports ``yaml`` only when called — install the ``config`` extra: ``pip install
  axor-core[config]``). ``from_dict`` works with any already-parsed mapping, so a
  caller that prefers TOML/JSON/env can build the dict however it likes.
- It **fails closed on an unknown key**. A typo in a security config that silently
  disabled a rule would be the worst kind of foot-gun, so an unrecognised field —
  top-level or inside a predicate — raises rather than being ignored.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from axor_core.contracts.canonical import ConsequenceClass
from axor_core.contracts.mode import ExecutionMode
from axor_core.policy.value_policy import ValuePredicate, enum, numeric_range

# Recognised top-level keys. Anything else is a config error (fail closed).
_KNOWN_KEYS = frozenset({
    "mode", "workspace", "profile",
    "untrusted_sources", "sensitive_sources", "egress_sinks",
    "positional_sinks", "benign_tools",
    "value_policies", "consequence_overrides",
})
_CONSEQUENCE_BY_NAME = {c.name.lower(): c for c in ConsequenceClass}


@dataclass(frozen=True)
class GovernanceConfig:
    """The operator's declarative governance description.

    Every field maps to a ``GovernedSession`` keyword argument; ``as_session_kwargs``
    produces exactly that dict.
    """

    mode: ExecutionMode = ExecutionMode.LIBRARY
    workspace: str | None = None
    profile: str | None = None
    untrusted_sources: frozenset[str] = frozenset()
    sensitive_sources: frozenset[str] = frozenset()
    egress_sinks: frozenset[str] = frozenset()
    positional_sinks: frozenset[str] = frozenset()
    benign_tools: frozenset[str] = frozenset()
    # tool name -> list of decidable predicates over its arguments
    value_policies: dict[str, list[ValuePredicate]] = field(default_factory=dict)
    # tool name -> action-class override (raise/lower how irreversible it is)
    consequence_overrides: dict[str, ConsequenceClass] = field(default_factory=dict)

    # ── construction ─────────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GovernanceConfig":
        """Build from an already-parsed mapping. Fails closed on unknown keys."""
        if not isinstance(data, dict):
            raise ValueError(f"governance config must be a mapping, got {type(data).__name__}")
        unknown = set(data) - _KNOWN_KEYS
        if unknown:
            raise ValueError(
                f"unknown governance config key(s): {sorted(unknown)} — "
                f"a typo must fail closed, not silently disable a rule. "
                f"Known keys: {sorted(_KNOWN_KEYS)}"
            )

        mode_raw = data.get("mode", "library")
        try:
            mode = ExecutionMode(mode_raw)
        except ValueError:
            raise ValueError(
                f"unknown mode {mode_raw!r}; expected one of "
                f"{[m.value for m in ExecutionMode]}"
            )

        return cls(
            mode=mode,
            workspace=data.get("workspace"),
            profile=data.get("profile"),
            untrusted_sources=_as_set(data.get("untrusted_sources"), "untrusted_sources"),
            sensitive_sources=_as_set(data.get("sensitive_sources"), "sensitive_sources"),
            egress_sinks=_as_set(data.get("egress_sinks"), "egress_sinks"),
            positional_sinks=_as_set(data.get("positional_sinks"), "positional_sinks"),
            benign_tools=_as_set(data.get("benign_tools"), "benign_tools"),
            value_policies=_parse_value_policies(data.get("value_policies")),
            consequence_overrides=_parse_consequence_overrides(
                data.get("consequence_overrides")
            ),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "GovernanceConfig":
        """Load from a YAML file. Requires the ``config`` extra (PyYAML)."""
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "GovernanceConfig.from_yaml needs PyYAML — install the config extra: "
                "pip install 'axor-core[config]'"
            ) from exc
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    # ── output ───────────────────────────────────────────────────────────────────

    def as_session_kwargs(self) -> dict[str, Any]:
        """The keyword arguments to splat into ``GovernedSession(...)``."""
        kwargs: dict[str, Any] = {
            "mode": self.mode,
            "untrusted_sources": set(self.untrusted_sources),
            "sensitive_sources": set(self.sensitive_sources),
            "egress_sinks": set(self.egress_sinks),
            "positional_sinks": set(self.positional_sinks),
            "benign_tools": set(self.benign_tools),
            "value_policies": dict(self.value_policies),
            # GovernedSession names the consequence-override table `danger`.
            "danger": dict(self.consequence_overrides),
        }
        if self.workspace is not None:
            kwargs["workspace"] = self.workspace
        if self.profile is not None:
            kwargs["profile"] = self.profile
        return kwargs


# ── parsing helpers (each fails closed on a malformed entry) ─────────────────────

def _as_set(value: Any, field_name: str) -> frozenset[str]:
    if value is None:
        return frozenset()
    if not isinstance(value, (list, tuple, set)):
        raise ValueError(f"{field_name} must be a list of tool names, got {type(value).__name__}")
    return frozenset(str(v) for v in value)


def _parse_value_policies(raw: Any) -> dict[str, list[ValuePredicate]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("value_policies must be a mapping of tool -> [predicate, ...]")
    out: dict[str, list[ValuePredicate]] = {}
    for tool, preds in raw.items():
        if not isinstance(preds, list):
            raise ValueError(f"value_policies[{tool!r}] must be a list of predicates")
        out[str(tool)] = [_parse_predicate(tool, p) for p in preds]
    return out


def _parse_predicate(tool: Any, p: Any) -> ValuePredicate:
    if not isinstance(p, dict):
        raise ValueError(f"value_policies[{tool!r}]: each predicate must be a mapping")
    arg = p.get("arg")
    kind = p.get("kind")
    if not arg or not kind:
        raise ValueError(f"value_policies[{tool!r}]: predicate needs 'arg' and 'kind'")
    if kind == "enum":
        allowed = p.get("allowed")
        if not isinstance(allowed, (list, tuple, set)):
            raise ValueError(
                f"value_policies[{tool!r}].{arg}: enum predicate needs an 'allowed' list"
            )
        extra = set(p) - {"arg", "kind", "allowed"}
        if extra:
            raise ValueError(f"value_policies[{tool!r}].{arg}: unknown enum field(s) {sorted(extra)}")
        return enum(str(arg), [str(a) for a in allowed])
    if kind == "numeric_range":
        if "lo" not in p or "hi" not in p:
            raise ValueError(
                f"value_policies[{tool!r}].{arg}: numeric_range predicate needs 'lo' and 'hi'"
            )
        extra = set(p) - {"arg", "kind", "lo", "hi"}
        if extra:
            raise ValueError(f"value_policies[{tool!r}].{arg}: unknown numeric_range field(s) {sorted(extra)}")
        return numeric_range(str(arg), p["lo"], p["hi"])
    raise ValueError(
        f"value_policies[{tool!r}].{arg}: unknown predicate kind {kind!r} "
        f"(expected 'enum' or 'numeric_range')"
    )


def _parse_consequence_overrides(raw: Any) -> dict[str, ConsequenceClass]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("consequence_overrides must be a mapping of tool -> class name")
    out: dict[str, ConsequenceClass] = {}
    for tool, name in raw.items():
        key = str(name).lower()
        if key not in _CONSEQUENCE_BY_NAME:
            raise ValueError(
                f"consequence_overrides[{tool!r}]: unknown class {name!r}; expected one of "
                f"{sorted(_CONSEQUENCE_BY_NAME)}"
            )
        out[str(tool)] = _CONSEQUENCE_BY_NAME[key]
    return out
