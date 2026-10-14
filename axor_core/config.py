"""Declarative governance configuration.

Instead of passing the data-flow taxonomy as a spray of keyword arguments, an
operator can describe it once in a YAML file and load it:

    from axor_core import GovernanceConfig, GovernedSession

    config = GovernanceConfig.from_yaml("governance.yaml")
    session = GovernedSession.from_config(executor, cap_executor, config)

See ``examples/config/governance.yaml`` for a fully-commented template.

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
from numbers import Real
from typing import Any

from axor_core.contracts.canonical import ConsequenceClass
from axor_core.contracts.mode import ExecutionMode
from axor_core.contracts.taint import INTEGRITY_DEFAULTS, INTEGRITY_ORIGIN_MODES
from axor_core.policy.value_policy import (
    ValuePredicate,
    enum,
    keyed_range,
    numeric_range,
    tuple_enum,
)

# Recognised top-level keys. Anything else is a config error (fail closed).
_KNOWN_KEYS = frozenset({
    "mode", "workspace", "profile",
    "untrusted_sources", "sensitive_sources", "egress_sinks",
    "positional_sinks", "imperative_sinks", "integrity_sinks", "benign_tools",
    "value_policies", "consequence_overrides",
    "driving_args",
    "integrity_default",
    "integrity_origins",
    "federation",
    "grant_args",
    "operator_trusted",
    "max_unattended_consequence",
    "floor_mode", "audience", "destination_args", "content_args",
    "operator_sets", "operator_sets_refuse_above",
    "template_limits", "consequence_rules", "content_projection", "operator_sets_confirm",
    "absent_driving_is_empty", "fallback_to_confirmation", "content_to_confirmation",
    "grant_templates",
    "bounded_damage",
    "grant_quotas",
    "floor_scope",
    "shared_stores",
    "access_control_sinks",
    "sink_origins",
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
    imperative_sinks: frozenset[str] = frozenset()
    # State-changing sinks whose driving args the attacker must not choose
    # (a password / profile update, a role grant): integrity only, no floor.
    integrity_sinks: frozenset[str] = frozenset()
    benign_tools: frozenset[str] = frozenset()
    # tool name -> list of decidable predicates over its arguments
    value_policies: dict[str, list[ValuePredicate]] = field(default_factory=dict)
    # tool name -> argument names the taint decision keys on (whole-args by default)
    driving_args: dict[str, list[str]] = field(default_factory=dict)
    # tool name -> action-class override (raise/lower how irreversible it is)
    consequence_overrides: dict[str, ConsequenceClass] = field(default_factory=dict)
    # "clean" (legacy) or "context": whether a model-generated value that carries
    # no registered untrusted fragment is trusted, or carries the node's context
    # root unless it is a trusted value (docs/rfc-integrity-context-default.md).
    # None = the mode's default ("context" under strict, "clean" otherwise),
    # resolved by the session / governor it is splatted into.
    integrity_default: "str | None" = None
    # Which TrustedOrigins clear a value at an integrity sink; resolved downstream.
    integrity_origins: str = "any-trusted"
    # Built opt-in A2A objects (None when no `federation:` section). The gateway is
    # the receive side (which peer values to trust); the identity is the send side
    # (used by a transport adapter to mint our outgoing receipts).
    federation_gateway: "Any | None" = None
    federation_identity: "Any | None" = None
    # Task-scoped grants: sink -> {arg: handle kind}; operator templates (L2);
    # sink -> [args] whose numeric limit is a bounded-damage guarantee.
    grant_args: dict[str, dict[str, str]] = field(default_factory=dict)
    grant_templates: dict[str, list[dict]] = field(default_factory=dict)
    bounded_damage: dict[str, list[str]] = field(default_factory=dict)
    # Operator rule: uses per L1 (request-literal) grant, per sink (default 1).
    grant_quotas: dict[str, int] = field(default_factory=dict)
    # Confidentiality floor scope (addressee | shared_stores | all_mutating) and the
    # operator's declaration of stores visible to third parties.
    floor_scope: str = "addressee"
    shared_stores: dict[str, Any] = field(default_factory=dict)
    # Credential changes: grants only from L3 (structured) or L4 (confirmation).
    access_control_sinks: frozenset[str] = frozenset()
    # Per-sink integrity-origin mode (request-only | any-trusted | none).
    sink_origins: dict[str, str] = field(default_factory=dict)
    # Operator-vouched values (origin OPERATOR, not an allowlist: refuses nothing).
    operator_trusted: frozenset[str] = frozenset()
    # Unattended consequence ceiling; above it a call needs a grant (a ground).
    max_unattended_consequence: ConsequenceClass = ConsequenceClass.CONSEQUENTIAL
    # Confidentiality floor: "session" (any egress held while a secret is outstanding)
    # or "audience" (a destination outside the audience needs a ground).
    floor_mode: str = "session"
    audience: frozenset[str] = frozenset()
    destination_args: dict[str, list[str]] = field(default_factory=dict)
    # Per sink, the arguments that can carry a secret (an empty list: none).
    content_args: dict[str, list[str]] = field(default_factory=dict)
    # Operator-declared sets per (sink, arg): OPERATOR origin + tuple-template O-sets;
    # unlike an enum value policy, they refuse nothing.
    operator_sets: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    # Above this consequence class an O-set refuses a value neither in it nor named by
    # the request (None: operator sets refuse nothing).
    operator_sets_refuse_above: "ConsequenceClass | None" = None
    # M1: magnitude limits in L2 tuple templates; per-call consequence rules; content projection
    template_limits: dict[str, dict[str, float]] = field(default_factory=dict)
    consequence_rules: dict[str, dict] = field(default_factory=dict)
    content_projection: bool = False
    # §8.5: an unproven destination/object asks for a confirmation instead of refusing
    operator_sets_confirm: bool = False
    # §8.8: no declared driving argument passed = nothing to check (no whole-blob fallback)
    absent_driving_is_empty: bool = False
    # §8.10: the fallback's integrity failure asks for a confirmation instead of refusing
    fallback_to_confirmation: bool = False
    # MODEL.md §8.12: a content-projection failure asks for a confirmation of the exact
    # call (with a confirmer and grant args); refused as before otherwise
    content_to_confirmation: bool = False

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

        integrity_default = data.get("integrity_default")
        if integrity_default is not None and integrity_default not in INTEGRITY_DEFAULTS:
            raise ValueError(
                f"unknown integrity_default {integrity_default!r}; expected one of "
                f"{sorted(INTEGRITY_DEFAULTS)}"
            )

        integrity_origins = data.get("integrity_origins", "any-trusted")
        if integrity_origins not in INTEGRITY_ORIGIN_MODES:
            raise ValueError(
                f"unknown integrity_origins {integrity_origins!r}; expected one of "
                f"{sorted(INTEGRITY_ORIGIN_MODES)}"
            )

        return cls(
            mode=mode,
            integrity_default=integrity_default,
            integrity_origins=integrity_origins,
            workspace=data.get("workspace"),
            profile=data.get("profile"),
            untrusted_sources=_as_set(data.get("untrusted_sources"), "untrusted_sources"),
            sensitive_sources=_as_set(data.get("sensitive_sources"), "sensitive_sources"),
            egress_sinks=_as_set(data.get("egress_sinks"), "egress_sinks"),
            positional_sinks=_as_set(data.get("positional_sinks"), "positional_sinks"),
            imperative_sinks=_as_set(data.get("imperative_sinks"), "imperative_sinks"),
            integrity_sinks=_as_set(data.get("integrity_sinks"), "integrity_sinks"),
            benign_tools=_as_set(data.get("benign_tools"), "benign_tools"),
            access_control_sinks=_as_set(data.get("access_control_sinks"), "access_control_sinks"),
            sink_origins=_parse_sink_origins(data.get("sink_origins")),
            operator_trusted=_as_set(data.get("operator_trusted"), "operator_trusted"),
            floor_mode=str(data.get("floor_mode") or "session"),
            audience=_as_set(data.get("audience"), "audience"),
            destination_args=_parse_driving_args(data.get("destination_args")),
            content_args={str(k): [str(a) for a in (v or [])]
                          for k, v in (data.get("content_args") or {}).items()},
            operator_sets={str(t): {str(a): [str(x) for x in (vs or [])] for a, vs in (args or {}).items()}
                           for t, args in (data.get("operator_sets") or {}).items()},
            operator_sets_refuse_above=(
                _parse_consequence_overrides({"_": data["operator_sets_refuse_above"]})["_"]
                if data.get("operator_sets_refuse_above") is not None else None),
            template_limits={str(t): {str(a): float(v) for a, v in (x or {}).items()}
                             for t, x in (data.get("template_limits") or {}).items()},
            consequence_rules=_parse_consequence_rules(data.get("consequence_rules")),
            content_projection=bool(data.get("content_projection", False)),
            operator_sets_confirm=bool(data.get("operator_sets_confirm", False)),
            absent_driving_is_empty=bool(data.get("absent_driving_is_empty", False)),
            fallback_to_confirmation=bool(data.get("fallback_to_confirmation", False)),
            content_to_confirmation=bool(data.get("content_to_confirmation", False)),
            **({"max_unattended_consequence": _parse_consequence_overrides(
                {"_": data["max_unattended_consequence"]})["_"]}
               if data.get("max_unattended_consequence") is not None else {}),
            value_policies=_parse_value_policies(data.get("value_policies")),
            driving_args=_parse_driving_args(data.get("driving_args")),
            consequence_overrides=_parse_consequence_overrides(
                data.get("consequence_overrides")
            ),
            **_parse_federation(data.get("federation")),
            **_parse_grants(data),
            **_parse_floor(data),
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
            "imperative_sinks": set(self.imperative_sinks),
            "integrity_sinks": set(self.integrity_sinks),
            "benign_tools": set(self.benign_tools),
            "value_policies": dict(self.value_policies),
            "driving_args": dict(self.driving_args),
            # GovernedSession names the consequence-override table `danger`.
            "danger": dict(self.consequence_overrides),
            "integrity_default": self.integrity_default,
            "integrity_origins": self.integrity_origins,
        }
        if self.workspace is not None:
            kwargs["workspace"] = self.workspace
        if self.profile is not None:
            kwargs["profile"] = self.profile
        if self.federation_gateway is not None:
            kwargs["federation_gateway"] = self.federation_gateway
        return kwargs

    def as_governor_kwargs(self) -> dict[str, Any]:
        """The keyword arguments to splat into ``ToolCallGovernor(...)`` — the
        synchronous enforcement path. Same declaration, same gate engine. The
        governor has no mode knob; STRICT maps to its fail-closed
        ``require_egress_allowlist`` construction obligation. Session-only fields
        (workspace, profile, benign_tools, federation) are not part of the
        per-call gate sequence and are omitted."""
        return {
            "untrusted_sources": set(self.untrusted_sources),
            "sensitive_sources": set(self.sensitive_sources),
            "egress_sinks": set(self.egress_sinks),
            "positional_sinks": set(self.positional_sinks),
            "imperative_sinks": set(self.imperative_sinks),
            "integrity_sinks": set(self.integrity_sinks),
            "benign_tools": set(self.benign_tools),
            "value_policies": dict(self.value_policies),
            "driving_args": dict(self.driving_args),
            "consequence_overrides": dict(self.consequence_overrides),
            "require_egress_allowlist": self.mode is ExecutionMode.STRICT,
            "require_tool_roles": self.mode is ExecutionMode.STRICT,
            "integrity_default": self.integrity_default,
            "integrity_origins": self.integrity_origins,
            "grant_args": dict(self.grant_args),
            "grant_templates": dict(self.grant_templates),
            "operator_trusted": set(self.operator_trusted),
            "max_unattended_consequence": self.max_unattended_consequence,
            "floor_mode": self.floor_mode,
            "audience": set(self.audience),
            "destination_args": dict(self.destination_args),
            "content_args": dict(self.content_args),
            "operator_sets": {t: dict(a) for t, a in self.operator_sets.items()},
            "operator_sets_refuse_above": self.operator_sets_refuse_above,
            "template_limits": {t: dict(a) for t, a in self.template_limits.items()},
            "consequence_rules": dict(self.consequence_rules),
            "content_projection": self.content_projection,
            "operator_sets_confirm": self.operator_sets_confirm,
            "absent_driving_is_empty": self.absent_driving_is_empty,
            "fallback_to_confirmation": self.fallback_to_confirmation,
            "content_to_confirmation": self.content_to_confirmation,
            "bounded_damage": dict(self.bounded_damage),
            "grant_quotas": dict(self.grant_quotas),
            "floor_scope": self.floor_scope,
            "shared_stores": dict(self.shared_stores),
            "access_control_sinks": set(self.access_control_sinks),
            "sink_origins": dict(self.sink_origins),
        }


# ── parsing helpers (each fails closed on a malformed entry) ─────────────────────

def _parse_sink_origins(value: Any) -> dict[str, str]:
    from axor_core.policy.gates import validate_sink_origins
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("sink_origins must be a mapping of tool -> request-only | any-trusted | none")
    errs = validate_sink_origins(value)
    if errs:
        raise ValueError("; ".join(errs))
    return {str(k): str(v) for k, v in value.items()}


def _parse_floor(data: dict[str, Any]) -> dict[str, Any]:
    from axor_core.policy.gates import FLOOR_SCOPES, validate_shared_stores
    scope = data.get("floor_scope", "addressee")
    if scope not in FLOOR_SCOPES:
        raise ValueError(f"unknown floor_scope {scope!r}; expected one of {sorted(FLOOR_SCOPES)}")
    stores = data.get("shared_stores") or {}
    if not isinstance(stores, dict):
        raise ValueError("shared_stores must be a mapping of tool -> 'always' | {arg: [ids]}")
    errors = validate_shared_stores(stores)
    if errors:
        raise ValueError("; ".join(errors))
    # the addressed floor (floor_mode: audience) reads shared_stores under any scope
    if stores and scope != "shared_stores" and data.get("floor_mode") != "audience":
        raise ValueError("shared_stores is declared but floor_scope is not 'shared_stores'")
    return {"floor_scope": scope, "shared_stores": dict(stores)}


def _as_set(value: Any, field_name: str) -> frozenset[str]:
    if value is None:
        return frozenset()
    if not isinstance(value, (list, tuple, set)):
        raise ValueError(f"{field_name} must be a list of tool names, got {type(value).__name__}")
    return frozenset(str(v) for v in value)


def _parse_grants(data: dict[str, Any]) -> dict[str, Any]:
    """grant_args / grant_templates / bounded_damage, validated at load (fail closed)."""
    from axor_core.kernel.registration import validate_grant_config

    ga = data.get("grant_args") or {}
    gt = data.get("grant_templates") or {}
    bd = data.get("bounded_damage") or {}
    gq = data.get("grant_quotas") or {}
    if not isinstance(gq, dict):
        raise ValueError("grant_quotas must be a mapping of tool -> positive integer")
    if not isinstance(ga, dict) or not all(isinstance(v, dict) for v in ga.values()):
        raise ValueError("grant_args must be a mapping of tool -> {arg: kind}")
    if not isinstance(gt, dict) or not all(isinstance(v, list) for v in gt.values()):
        raise ValueError("grant_templates must be a mapping of tool -> [ {arg: value}, ... ]")
    if not isinstance(bd, dict) or not all(isinstance(v, (list, tuple)) for v in bd.values()):
        raise ValueError("bounded_damage must be a mapping of tool -> [arg name, ...]")
    grant_args = {str(t): {str(a): str(k) for a, k in v.items()} for t, v in ga.items()}
    templates = {str(t): [dict(e) for e in v] for t, v in gt.items()}
    bounded = {str(t): [str(a) for a in v] for t, v in bd.items()}
    quotas = {str(t): n for t, n in gq.items()}
    acs = _as_set(data.get("access_control_sinks"), "access_control_sinks")
    errors = validate_grant_config(grant_args, templates, bounded, quotas, acs)
    if errors:
        raise ValueError("grant config: " + "; ".join(errors))
    return {"grant_args": grant_args, "grant_templates": templates, "bounded_damage": bounded,
            "grant_quotas": quotas}


def _parse_driving_args(raw: Any) -> dict[str, list[str]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("driving_args must be a mapping of tool -> [arg name, ...]")
    out: dict[str, list[str]] = {}
    for tool, names in raw.items():
        if not isinstance(names, (list, tuple)):
            raise ValueError(f"driving_args[{tool!r}] must be a list of argument names")
        out[str(tool)] = [str(n) for n in names]
    return out


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


_RELATIONAL_KINDS = frozenset({"tuple_enum", "keyed_range"})


def _parse_relational_predicate(tool: Any, p: dict, kind: str) -> ValuePredicate:
    """Parse a relational predicate (``args`` instead of ``arg``).

    Same fail-closed-at-load discipline as the per-arg kinds: an empty combination
    set or range table denies every call while still satisfying a presence-only
    allowlist obligation, and a non-numeric or inverted bound would defer to a
    TypeError on the enforcement path.
    """
    args = p.get("args")
    if not isinstance(args, (list, tuple)) or len(args) < 2:
        raise ValueError(
            f"value_policies[{tool!r}]: {kind} predicate needs 'args' with at least "
            f"two argument names (it constrains a tuple, not one value)"
        )
    args = [str(a) for a in args]
    if len(set(args)) != len(args):
        raise ValueError(f"value_policies[{tool!r}]: {kind} 'args' has duplicates: {args}")

    if kind == "tuple_enum":
        extra = set(p) - {"kind", "args", "allowed"}
        if extra:
            raise ValueError(
                f"value_policies[{tool!r}].{args}: unknown tuple_enum field(s) {sorted(extra)}"
            )
        allowed = p.get("allowed")
        if not isinstance(allowed, (list, tuple)) or len(allowed) == 0:
            raise ValueError(
                f"value_policies[{tool!r}].{args}: tuple_enum needs a non-empty "
                f"'allowed' list of combinations"
            )
        combos = []
        for combo in allowed:
            if not isinstance(combo, (list, tuple)) or len(combo) != len(args):
                raise ValueError(
                    f"value_policies[{tool!r}].{args}: each tuple_enum combination "
                    f"must list exactly {len(args)} values, got {combo!r}"
                )
            combos.append(tuple(str(v) for v in combo))
        return tuple_enum(args, combos)

    # keyed_range
    extra = set(p) - {"kind", "args", "ranges"}
    if extra:
        raise ValueError(
            f"value_policies[{tool!r}].{args}: unknown keyed_range field(s) {sorted(extra)}"
        )
    if len(args) != 2:
        raise ValueError(
            f"value_policies[{tool!r}].{args}: keyed_range takes exactly two args "
            f"(key, value)"
        )
    ranges = p.get("ranges")
    if not isinstance(ranges, dict) or not ranges:
        raise ValueError(
            f"value_policies[{tool!r}].{args}: keyed_range needs a non-empty 'ranges' "
            f"mapping of key -> [lo, hi]"
        )
    parsed: dict[object, tuple] = {}
    for key, bounds in ranges.items():
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(
                f"value_policies[{tool!r}].{args}: range for {key!r} must be [lo, hi]"
            )
        lo, hi = bounds
        if isinstance(lo, bool) or isinstance(hi, bool) or not isinstance(lo, (int, float)) \
                or not isinstance(hi, (int, float)):
            raise ValueError(
                f"value_policies[{tool!r}].{args}: range bounds for {key!r} must be "
                f"numbers, got lo={lo!r} hi={hi!r}"
            )
        if lo > hi:
            raise ValueError(
                f"value_policies[{tool!r}].{args}: inverted range for {key!r} "
                f"(lo={lo} > hi={hi}) rejects every value"
            )
        parsed[str(key)] = (lo, hi)
    return keyed_range(args[0], args[1], parsed)


def _parse_predicate(tool: Any, p: Any) -> ValuePredicate:
    if not isinstance(p, dict):
        raise ValueError(f"value_policies[{tool!r}]: each predicate must be a mapping")
    arg = p.get("arg")
    kind = p.get("kind")
    if kind in _RELATIONAL_KINDS:
        return _parse_relational_predicate(tool, p, str(kind))
    if not arg or not kind:
        raise ValueError(f"value_policies[{tool!r}]: predicate needs 'arg' and 'kind'")
    if kind == "enum":
        allowed = p.get("allowed")
        if not isinstance(allowed, (list, tuple, set)):
            raise ValueError(
                f"value_policies[{tool!r}].{arg}: enum predicate needs an 'allowed' list"
            )
        if len(allowed) == 0:
            # A vacuous allowlist denies every value — almost always a typo (a YAML
            # anchor that resolved to nothing), and it would silently satisfy the
            # STRICT egress-allowlist obligation while permanently self-DoSing the
            # sink. Fail closed at load instead.
            raise ValueError(
                f"value_policies[{tool!r}].{arg}: enum 'allowed' is empty — an empty "
                "allowlist denies every value; declare at least one member or remove it"
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
        lo, hi = p["lo"], p["hi"]
        # Bounds must be numbers and ordered, checked at LOAD — otherwise a
        # non-numeric bound defers to a TypeError on the enforcement hot path, and
        # an inverted range (lo>hi) silently rejects every value at runtime.
        if not isinstance(lo, Real) or isinstance(lo, bool) \
                or not isinstance(hi, Real) or isinstance(hi, bool):
            raise ValueError(
                f"value_policies[{tool!r}].{arg}: numeric_range 'lo'/'hi' must be "
                f"numbers, got lo={lo!r}, hi={hi!r}"
            )
        if lo > hi:
            raise ValueError(
                f"value_policies[{tool!r}].{arg}: numeric_range is inverted "
                f"(lo={lo} > hi={hi}) — it would reject every value"
            )
        return numeric_range(str(arg), lo, hi)
    raise ValueError(
        f"value_policies[{tool!r}].{arg}: unknown predicate kind {kind!r} "
        f"(expected 'enum', 'numeric_range', 'tuple_enum' or 'keyed_range')"
    )


# ── federation (A2A) ─────────────────────────────────────────────────────────────
#
# Policy is declarative; KEY MATERIAL IS NEVER INLINE. Every key is referenced by an
# environment variable (hex-encoded) or a file path (raw bytes), and resolved at load
# time. A shared HMAC secret or an ed25519 private key in a committed YAML would be a
# disaster, so the parser accepts only references and fails closed when one is missing.

_FED_KEYS = frozenset({"compatible_kernels", "federated_domains", "peers", "identity"})
_PEER_KEYS = frozenset({
    "peer_id", "domain", "kernel_version", "algorithm",
    "public_key_env", "public_key_file", "shared_key_env", "shared_key_file",
})
_IDENTITY_KEYS = frozenset({
    "peer_id", "domain", "kernel_version", "algorithm",
    "private_key_env", "private_key_file", "shared_key_env", "shared_key_file",
})


def _resolve_key(spec: dict, prefix: str, ctx: str) -> bytes | None:
    """Resolve one key from ``<prefix>_env`` (hex) or ``<prefix>_file`` (raw bytes).
    Returns None if neither is present; raises if a reference is present but empty,
    missing, or malformed."""
    import os
    env_name = spec.get(f"{prefix}_env")
    file_name = spec.get(f"{prefix}_file")
    if env_name and file_name:
        raise ValueError(f"{ctx}: give only one of {prefix}_env / {prefix}_file")
    if env_name:
        raw = os.environ.get(env_name)
        if not raw:
            raise ValueError(f"{ctx}: env {env_name!r} is unset or empty (keys are never inline)")
        try:
            return bytes.fromhex(raw.strip())
        except ValueError as exc:
            raise ValueError(f"{ctx}: env {env_name!r} is not valid hex") from exc
    if file_name:
        try:
            with open(os.path.expanduser(file_name), "rb") as f:
                data = f.read()
        except OSError as exc:
            raise ValueError(f"{ctx}: cannot read key file {file_name!r}: {exc}") from exc
        if not data:
            raise ValueError(f"{ctx}: key file {file_name!r} is empty")
        return data
    return None


def _build_verifier(spec: dict, ctx: str):
    """Build a peer Verifier from its declared algorithm + referenced key."""
    from axor_core.federation.signing import Ed25519Verifier, HmacSigner
    algo = spec.get("algorithm", "hmac-sha256")
    if algo == "hmac-sha256":
        key = _resolve_key(spec, "shared_key", ctx)
        if key is None:
            raise ValueError(f"{ctx}: hmac peer needs shared_key_env or shared_key_file")
        return HmacSigner(key)  # symmetric — the signer verifies too
    if algo == "ed25519":
        key = _resolve_key(spec, "public_key", ctx)
        if key is None:
            raise ValueError(f"{ctx}: ed25519 peer needs public_key_env or public_key_file")
        return Ed25519Verifier(key)
    raise ValueError(f"{ctx}: unknown algorithm {algo!r} (expected hmac-sha256 or ed25519)")


def _build_signer(spec: dict, ctx: str):
    """Build our own Signer (the send side) from algorithm + referenced PRIVATE key."""
    from axor_core.federation.signing import Ed25519Signer, HmacSigner
    algo = spec.get("algorithm", "hmac-sha256")
    if algo == "hmac-sha256":
        key = _resolve_key(spec, "shared_key", ctx)
        if key is None:
            raise ValueError(f"{ctx}: hmac identity needs shared_key_env or shared_key_file")
        return HmacSigner(key)
    if algo == "ed25519":
        key = _resolve_key(spec, "private_key", ctx)
        if key is None:
            raise ValueError(f"{ctx}: ed25519 identity needs private_key_env or private_key_file")
        return Ed25519Signer(key)
    raise ValueError(f"{ctx}: unknown algorithm {algo!r} (expected hmac-sha256 or ed25519)")


def _require_str(spec: dict, key: str, ctx: str) -> str:
    val = spec.get(key)
    if not val:
        raise ValueError(f"{ctx}: missing required field {key!r}")
    return str(val)


def _parse_federation(raw: Any) -> dict[str, Any]:
    """Build the FederationGateway (and optional LocalIdentity) from the config.
    Returns kwargs for GovernanceConfig; empty when there is no federation section."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("federation must be a mapping")
    unknown = set(raw) - _FED_KEYS
    if unknown:
        raise ValueError(f"unknown federation key(s): {sorted(unknown)}; known: {sorted(_FED_KEYS)}")

    from axor_core.federation.gateway import FederationGateway
    from axor_core.federation.receipt import FederationPeer, LocalIdentity

    peers: dict[str, Any] = {}
    for i, p in enumerate(raw.get("peers") or []):
        if not isinstance(p, dict):
            raise ValueError(f"federation.peers[{i}] must be a mapping")
        extra = set(p) - _PEER_KEYS
        if extra:
            raise ValueError(f"federation.peers[{i}]: unknown field(s) {sorted(extra)}")
        ctx = f"federation.peers[{i}]"
        pid = _require_str(p, "peer_id", ctx)
        peers[pid] = FederationPeer(
            peer_id=pid,
            verifier=_build_verifier(p, ctx),
            kernel_version=_require_str(p, "kernel_version", ctx),
            domain=_require_str(p, "domain", ctx),
        )

    gateway = FederationGateway(
        peers=peers,
        compatible_kernels=_as_set(raw.get("compatible_kernels"), "federation.compatible_kernels"),
        federated_domains=_as_set(raw.get("federated_domains"), "federation.federated_domains"),
    )

    identity = None
    id_spec = raw.get("identity")
    if id_spec is not None:
        if not isinstance(id_spec, dict):
            raise ValueError("federation.identity must be a mapping")
        extra = set(id_spec) - _IDENTITY_KEYS
        if extra:
            raise ValueError(f"federation.identity: unknown field(s) {sorted(extra)}")
        identity = LocalIdentity(
            peer_id=_require_str(id_spec, "peer_id", "federation.identity"),
            kernel_version=_require_str(id_spec, "kernel_version", "federation.identity"),
            domain=_require_str(id_spec, "domain", "federation.identity"),
            signer=_build_signer(id_spec, "federation.identity"),
        )

    return {"federation_gateway": gateway, "federation_identity": identity}


def _parse_consequence_rules(raw: Any) -> dict[str, dict]:
    """tool -> {"own": class, "magnitude": {"arg": name, "above": number, "class": class}}."""
    out: dict[str, dict] = {}
    for tool, rule in (raw or {}).items():
        r: dict = {}
        if rule.get("own") is not None:
            r["own"] = _parse_consequence_overrides({"_": rule["own"]})["_"]
        m = rule.get("magnitude")
        if m:
            r["magnitude"] = {"arg": str(m["arg"]), "above": float(m["above"]),
                              "class": _parse_consequence_overrides({"_": m["class"]})["_"]}
        out[str(tool)] = r
    return out


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
