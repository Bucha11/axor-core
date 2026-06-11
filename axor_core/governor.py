"""Synchronous per-tool-call governor.

The streaming :class:`IntentLoop` is the kernel's primary enforcement path, but
it is built for an executor that *yields* events. Many integrations — a
LangChain tool wrapper, a benchmark harness, any framework that owns its own
agent loop — instead have a plain "the model wants to call tool T with args A;
may it?" question, answered synchronously, with the framework executing the tool
itself.

:class:`ToolCallGovernor` answers exactly that. It runs the same content-blind
and per-value gates the IntentLoop runs — consequence, value policies, the
SSRF/internal-destination gate, positional admission, the carrier/imperative
gate, and the per-value taint check with the confidentiality floor — over a
single ``(tool_name, args)`` pair, and tracks tool outputs in the same per-value
ledger so a later sink carrying untrusted or secret content is gated.

It deliberately does **not** cover capability/lease/degradation: those are
policy-and-session concerns the host framework owns (it decides which tools
exist and whether the session is degraded). What this governs is the data-flow
and action-class core that a tool registry alone cannot express.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from axor_core.contracts.anomaly import NormalizedIntent
from axor_core.contracts.canonical import ConsequenceClass
from axor_core.contracts.intent import Intent, IntentKind
from axor_core.contracts.taint import TaintSource
from axor_core.node.normalizer import IntentNormalizer
from axor_core.policy.gates import (
    GateDecision,
    carrier_gate,
    consequence_gate,
    positional_gate,
    ssrf_gate,
    taint_gate,
    value_policy_gate,
)
from axor_core.policy.sinks import INSTRUCTION_COMPLETE_SINKS
from axor_core.taint.causal_root import CausalRoot
from axor_core.taint.engine import TaintEngine


@dataclass
class GovernanceDecision:
    """The outcome of governing one tool call.

    ``allowed`` is the only thing most callers need; ``reason`` and ``category``
    explain a denial (category is the same coarse label the IntentLoop records,
    e.g. ``taint_enforcement``, ``consequence_gate``, ``carrier_gate``).
    """

    allowed: bool
    reason: str = "approved"
    category: str = "approved"
    # The normalized intent for this call, kept so register_output can reuse it
    # without advancing the normalizer's provenance state twice.
    _normalized: NormalizedIntent | None = field(default=None, repr=False)


class ToolCallGovernor:
    """Runs the kernel's per-call data-flow and action-class gates synchronously.

    Hold one instance per agent session: it carries the per-value taint ledger
    and the confidentiality floor across the session's tool calls.

    Typical use::

        gov = ToolCallGovernor()
        decision = gov.evaluate(tool_name, args)
        if not decision.allowed:
            return denial(decision.reason)
        output = run_the_tool(tool_name, args)
        gov.register_output(decision, output)
    """

    def __init__(
        self,
        *,
        positional_sinks: "set[str] | frozenset[str] | None" = None,
        value_policies: dict | None = None,
        consequence_overrides: dict | None = None,
        max_unattended_consequence: ConsequenceClass = ConsequenceClass.CONSEQUENTIAL,
        untrusted_sources: "set[str] | frozenset[str] | None" = None,
        sensitive_sources: "set[str] | frozenset[str] | None" = None,
        egress_sinks: "set[str] | frozenset[str] | None" = None,
        imperative_sinks: "set[str] | frozenset[str] | None" = None,
        node_id: str = "",
    ) -> None:
        self._positional_sinks = frozenset(positional_sinks or ())
        illegal = {
            s for s in self._positional_sinks if s.lower() in INSTRUCTION_COMPLETE_SINKS
        }
        if illegal:
            raise ValueError(
                "instruction-complete sinks cannot be declared positional: "
                f"{sorted(illegal)} — their codomain admits instructions by "
                "definition; they must stay on the content-derivation path."
            )
        self._value_policies = dict(value_policies or {})
        self._consequence_overrides = dict(consequence_overrides or {})
        self._ceiling = max_unattended_consequence
        # Deployment tool taxonomy. axor's normalizer recognises generic tool
        # names (read/write/bash/curl/...); a real deployment renames its tools
        # (send_money, get_webpage, read_inbox, ...). Declaring those tools' roles
        # here is how an operator tells the kernel which calls produce untrusted
        # data and which are egress sinks — the normalizer heuristics still apply
        # for any tool not declared. A declared role takes precedence.
        self._untrusted_sources = frozenset(untrusted_sources or ())
        self._sensitive_sources = frozenset(sensitive_sources or ())
        self._egress_sinks = frozenset(egress_sinks or ())
        self._imperative_sinks = frozenset(imperative_sinks or ())
        self._normalizer = IntentNormalizer()
        self._taint = TaintEngine(node_id=node_id)

    # ── decision ────────────────────────────────────────────────────────────────

    def evaluate(self, tool_name: str, args: dict[str, Any]) -> GovernanceDecision:
        """Decide whether this tool call may execute. Pure — no side effects on
        the ledger (call :meth:`register_output` after a tool actually runs)."""
        intent = Intent(
            kind=IntentKind.TOOL_CALL,
            payload={"tool": tool_name, "args": args},
            node_id="",
            sequence=0,
        )
        normalized = self._normalizer.normalize(intent)

        def _deny(gd: GateDecision) -> GovernanceDecision:
            return GovernanceDecision(
                allowed=False, reason=gd.reason, category=gd.category,
                _normalized=normalized,
            )

        # 1. consequence — content-blind action-class gate. (No lease/escalation
        #    in the governor, so no governance-gate exception.)
        gd = consequence_gate(
            tool_name, normalized.operation, self._ceiling, self._consequence_overrides
        )
        if gd is not None:
            return _deny(gd)

        # 2. value policies — operator-registered decidable predicates.
        gd = value_policy_gate(tool_name, args, self._value_policies)
        if gd is not None:
            return _deny(gd)

        # 3. SSRF / internal-destination gate.
        gd = ssrf_gate(tool_name, normalized)
        if gd is not None:
            return _deny(gd)

        driving_root = self._taint.derive_value(args)

        # 4. positional admission — for declared instruction-incomplete sinks.
        gd = positional_gate(tool_name, args, self._positional_sinks)
        if gd is not None:
            return _deny(gd)

        # 5. carrier / imperative-channel gate.
        gd = carrier_gate(tool_name, args, normalized, driving_root, self._imperative_sinks)
        if gd is not None:
            return _deny(gd)

        # 6. per-value taint — integrity + confidentiality floor.
        gd = taint_gate(
            tool_name, normalized, driving_root,
            self._taint.confidentiality_floor_active(), self._egress_sinks,
        )
        if gd is not None:
            return _deny(gd)

        return GovernanceDecision(allowed=True, _normalized=normalized)

    # ── ledger ────────────────────────────────────────────────────────────────

    def register_output(
        self, decision: GovernanceDecision, output: Any
    ) -> None:
        """Record an executed tool's output in the per-value ledger.

        Mirrors the IntentLoop's registration: external/web reads taint the
        produced value untrusted; secret/system reads also mark it sensitive,
        which arms the confidentiality floor. A clean read registers nothing.
        Call this only for calls that actually executed.
        """
        ni = decision._normalized
        tool_name = ni.tool if ni is not None else ""
        # A declared role wins over the normalizer heuristic.
        if tool_name in self._sensitive_sources:
            root = CausalRoot.external_read(TaintSource.FILE, sensitive=True)
        elif tool_name in self._untrusted_sources:
            root = CausalRoot.external_read(TaintSource.WEB)
        elif ni is None:
            return
        elif (
            ni.target_kind in ("external_url", "cloud_metadata", "docker_socket")
            or ni.operation == "network_request"
        ):
            root = CausalRoot.external_read(TaintSource.WEB)
        elif (
            ni.target_kind in ("secret", "system_path")
            or ni.reads_secret_like_data
            or ni.writes_outside_workdir
        ):
            sensitive = ni.target_kind == "secret" or ni.reads_secret_like_data
            root = CausalRoot.external_read(TaintSource.FILE, sensitive=sensitive)
        else:
            return
        self._taint.register_value(output, root)

    # ── introspection ────────────────────────────────────────────────────────

    def confidentiality_floor_active(self) -> bool:
        """True once a secret read has armed the egress floor this session."""
        return self._taint.confidentiality_floor_active()
