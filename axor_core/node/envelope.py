from __future__ import annotations

import uuid
from axor_core.contracts.context import ContextView, LineageSummary
from axor_core.contracts.envelope import (
    ExecutionEnvelope,
    ExportContract,
)
from axor_core.contracts.authority import AuthorityPolicy
from axor_core.contracts.planning import ExecutionPlan
from axor_core.contracts.policy import ContextMode, ExecutionPolicy, ExportMode
from axor_core.policy.legacy import split_legacy_policy
from axor_core.contracts.extension import ExtensionTool
from axor_core.capability.resolver import CapabilityResolver


_EXPORT_ALLOWED_FIELDS: dict[ExportMode, frozenset[str]] = {
    ExportMode.FULL:       frozenset({"output", "reasoning", "tool_results", "metadata"}),
    ExportMode.FILTERED:   frozenset({"output", "metadata"}),
    ExportMode.SUMMARY:    frozenset({"output"}),
    ExportMode.RESTRICTED: frozenset(),
}

_EXPORT_MAX_TOKENS: dict[ExportMode, int | None] = {
    ExportMode.FULL:       None,
    ExportMode.FILTERED:   4096,
    ExportMode.SUMMARY:    1024,
    ExportMode.RESTRICTED: 0,
}


class EnvelopeBuilder:
    """
    Constructs ExecutionEnvelope from all subsystem outputs.

    Called by GovernedNode before invoking the executor.
    The executor receives the envelope — nothing else.

    Steps:
        1. Resolve capabilities from policy + extension tools
        2. Derive export contract from policy
        3. Assemble envelope
    """

    def __init__(self) -> None:
        self._resolver = CapabilityResolver()

    def build(
        self,
        task: str,
        context: ContextView,
        policy: ExecutionPolicy,
        lineage: LineageSummary,
        extension_tools: list[ExtensionTool] | None = None,
        node_id: str | None = None,
        parent_metadata: dict | None = None,
        cancel_token=None,
        authority: AuthorityPolicy | None = None,
        plan: ExecutionPlan | None = None,
    ) -> ExecutionEnvelope:
        from axor_core.contracts.cancel import make_token
        node_id = node_id or _new_node_id()

        # Authority/plan split: capabilities derive from the trusted half;
        # the one planning-driven capability flag (context expansion) is
        # computed HERE — the capability layer never reads ExecutionPlan
        # (import contract `authority-plan-separation`).
        if authority is None or plan is None:
            authority, plan = split_legacy_policy(policy)
        capabilities = self._resolver.resolve(
            authority,
            extension_tools or [],
            allow_context_expansion=plan.context_mode != ContextMode.MINIMAL,
        )
        export_contract = self._derive_export_contract(authority.export_policy.max_mode)

        return ExecutionEnvelope(
            node_id=node_id,
            task=task,
            context=context,
            policy=policy,
            capabilities=capabilities,
            export_contract=export_contract,
            lineage=lineage,
            cancel_token=cancel_token or make_token(),
            parent_metadata=parent_metadata or {},
            authority=authority,
            plan=plan,
        )

    def _derive_export_contract(self, mode: ExportMode) -> ExportContract:
        return ExportContract(
            mode=mode,
            allowed_fields=_EXPORT_ALLOWED_FIELDS[mode],
            max_export_tokens=_EXPORT_MAX_TOKENS[mode],
        )


def _new_node_id() -> str:
    return f"node_{uuid.uuid4().hex[:12]}"
