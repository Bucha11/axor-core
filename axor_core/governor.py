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
from axor_core.policy.normalizer import IntentNormalizer
from axor_core.policy.gates import (
    GateDecision,
    carrier_gate,
    consequence_gate,
    driving_subset,
    integrity_superseded_by_decidable,
    positional_gate,
    ssrf_gate,
    taint_gate,
    value_policy_gate,
)
from axor_core.policy.sinks import INSTRUCTION_COMPLETE_SINKS
from axor_core.policy.provenance import output_root
from axor_core.kernel.registration import (
    validate_driving_arg_allowlists,
    validate_egress_allowlists,
    tool_is_classified,
)
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
        benign_tools: "set[str] | frozenset[str] | None" = None,
        driving_args: "dict[str, list[str]] | None" = None,
        require_egress_allowlist: bool = False,
        require_tool_roles: bool = False,
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
        # Explicitly-benign reads (trusted output that need not be tainted). Kept so
        # the per-call STRICT role check can tell "declared benign" from "forgot to
        # classify" — without it, both would fail open to a clean read.
        self._benign_tools = frozenset(benign_tools or ())
        # STRICT role obligation, enforced LAZILY per call: the governor never sees
        # the full tool universe (it is asked one (tool, args) at a time), so it
        # cannot validate completeness at construction the way GovernedSession does.
        # Instead it denies any unclassified tool the moment it is used — closing the
        # fail-open default where a renamed, undeclared tool normalises to a benign
        # no-op and slips past every gate.
        self._require_tool_roles = require_tool_roles
        # Per-sink driving arguments — the fields the taint decision keys on. Empty
        # = whole-args (safe, coarse). Declaring them narrows to the destination /
        # instruction field so untrusted content to a trusted destination is not
        # over-blocked.
        self._driving_args = {k: frozenset(v) for k, v in (driving_args or {}).items()}
        # STRICT obligation: every egress sink must carry a destination allowlist
        # (an enum value_policy) — the sound, paraphrase-proof control. Fail closed
        # at construction rather than ship an egress sink on content-derivation alone.
        if require_egress_allowlist:
            errors = validate_egress_allowlists(self._egress_sinks, self._value_policies)
            errors += validate_driving_arg_allowlists(
                self._egress_sinks, self._driving_args, self._value_policies
            )
            if errors:
                raise ValueError("strict egress allowlist: " + "; ".join(errors))
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

        # 0. STRICT role completeness (lazy): an unclassified tool fails closed
        #    rather than defaulting to a clean benign read. This is the fix for the
        #    fail-open-on-unknown-tool default — without an explicit role the kernel
        #    cannot know a renamed tool exfiltrates or reads a secret.
        if self._require_tool_roles and not tool_is_classified(
            tool_name,
            untrusted_sources=self._untrusted_sources,
            sensitive_sources=self._sensitive_sources,
            egress_sinks=self._egress_sinks,
            positional_sinks=self._positional_sinks,
            benign_tools=self._benign_tools,
            value_policies=self._value_policies,
        ):
            return GovernanceDecision(
                allowed=False,
                reason=(
                    f"tool {tool_name!r} has no declared data-flow role; STRICT mode "
                    "refuses an unclassified tool (it would default to a clean read "
                    "and arm no floor). Declare it as a source/sink/positional/"
                    "value_policy or explicitly benign."
                ),
                category="unclassified_tool",
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

        driving_root = self._taint.derive_value(
            driving_subset(args, self._driving_args.get(tool_name))
        )

        # 4. positional admission — for declared instruction-incomplete sinks.
        gd = positional_gate(tool_name, args, self._positional_sinks)
        if gd is not None:
            return _deny(gd)

        # 5. carrier / imperative-channel gate.
        gd = carrier_gate(tool_name, args, normalized, driving_root, self._imperative_sinks)
        if gd is not None:
            return _deny(gd)

        # 6. per-value taint — integrity + confidentiality floor. A sink whose
        # driving args are fully guarded by satisfied decidable predicates carries
        # its integrity axis there (a stronger, content-blind control), so the
        # content-taint integrity check is superseded; the floor still applies.
        gd = taint_gate(
            tool_name, normalized, driving_root,
            self._taint.confidentiality_floor_active(), self._egress_sinks,
            integrity_superseded=integrity_superseded_by_decidable(
                tool_name, args, self._driving_args, self._value_policies
            ),
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
        # Shared arming map — identical to the IntentLoop's, so the two paths cannot
        # drift on what a tool's output taints.
        root = output_root(
            tool_name, ni,
            untrusted_sources=self._untrusted_sources,
            sensitive_sources=self._sensitive_sources,
        )
        if root is None:
            return  # clean read — nothing to register
        self._taint.register_value(output, root)

    # ── introspection ────────────────────────────────────────────────────────

    def confidentiality_floor_active(self) -> bool:
        """True once a secret read has armed the egress floor this session."""
        return self._taint.confidentiality_floor_active()
