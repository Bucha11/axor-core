from __future__ import annotations

from axor_core.contracts.envelope import ExportContract, ExecutionEnvelope
from axor_core.contracts.policy import ExportMode
from axor_core.contracts.result import ExecutionResult, TokenUsage
from axor_core.errors.exceptions import ExportDeniedError


class ExportFilter:
    """
    Applies export contract to the executor's output.

    Nodes do not export arbitrary executor output.
    They export governed output shaped by ExportContract.

    Called by GovernedNode after the intent loop completes.
    The result that leaves the node has passed through here.
    """

    def apply(
        self,
        raw_output: str,
        raw_payload: dict,
        envelope: ExecutionEnvelope,
        token_usage: TokenUsage,
    ) -> ExecutionResult:
        contract = envelope.export_contract

        match contract.mode:
            case ExportMode.RESTRICTED:
                # nothing leaves
                export_payload = {}
                output = ""

            case ExportMode.SUMMARY:
                # output only, truncated if needed
                output = self._truncate(raw_output, contract.max_export_tokens)
                export_payload = {"output": output}

            case ExportMode.FILTERED:
                # allowed fields only
                export_payload = {
                    k: v for k, v in raw_payload.items()
                    if k in contract.allowed_fields
                }
                output = export_payload.get("output", raw_output)
                if contract.max_export_tokens:
                    output = self._truncate(output, contract.max_export_tokens)

            case ExportMode.FULL:
                output = raw_output
                export_payload = raw_payload

            case _:
                raise ExportDeniedError(
                    mode=str(contract.mode),
                    reason=f"unknown export mode: {contract.mode}",
                )

        return ExecutionResult(
            node_id=envelope.node_id,
            output=output,
            export_payload=export_payload,
            token_usage=token_usage,
            metadata={
                "policy": envelope.policy.name,
                "export_mode": contract.mode,
                "depth": envelope.lineage.depth,
            },
        )

    def _truncate(self, text: str, max_tokens: int | None) -> str:
        if max_tokens is None or max_tokens == 0:
            return text
        # rough approximation: 1 token ≈ 4 chars
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n[truncated by export contract]"
