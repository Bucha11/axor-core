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
    "driving_args",
    "federation",
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
    # tool name -> argument names the taint decision keys on (whole-args by default)
    driving_args: dict[str, list[str]] = field(default_factory=dict)
    # tool name -> action-class override (raise/lower how irreversible it is)
    consequence_overrides: dict[str, ConsequenceClass] = field(default_factory=dict)
    # Built opt-in A2A objects (None when no `federation:` section). The gateway is
    # the receive side (which peer values to trust); the identity is the send side
    # (used by a transport adapter to mint our outgoing receipts).
    federation_gateway: "Any | None" = None
    federation_identity: "Any | None" = None

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
            driving_args=_parse_driving_args(data.get("driving_args")),
            consequence_overrides=_parse_consequence_overrides(
                data.get("consequence_overrides")
            ),
            **_parse_federation(data.get("federation")),
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
            "driving_args": dict(self.driving_args),
            # GovernedSession names the consequence-override table `danger`.
            "danger": dict(self.consequence_overrides),
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
            "value_policies": dict(self.value_policies),
            "driving_args": dict(self.driving_args),
            "consequence_overrides": dict(self.consequence_overrides),
            "require_egress_allowlist": self.mode is ExecutionMode.STRICT,
        }


# ── parsing helpers (each fails closed on a malformed entry) ─────────────────────

def _as_set(value: Any, field_name: str) -> frozenset[str]:
    if value is None:
        return frozenset()
    if not isinstance(value, (list, tuple, set)):
        raise ValueError(f"{field_name} must be a list of tool names, got {type(value).__name__}")
    return frozenset(str(v) for v in value)


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
