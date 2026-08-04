"""The shared provenance-arming map — the counterpart to the deny gates.

Where :mod:`axor_core.policy.gates` is the *decision* half of the per-value model
(does this call's driving value carry taint?), this is the *arming* half: when a
tool's output comes back, what causal_root does it introduce? Both the streaming
:class:`~axor_core.node.intent_loop.IntentLoop` and the synchronous
:class:`~axor_core.governor.ToolCallGovernor` must answer this identically — a
divergence here silently desynchronises what the two paths consider tainted — so
the mapping lives once, here, and both register their outputs through it.

Pure: it maps (tool name, normalized intent, declared roles) → a ``CausalRoot`` (or
``None`` for a clean read that arms nothing). It does not touch the taint engine;
the caller registers the returned root.
"""
from __future__ import annotations

from axor_core.contracts.anomaly import NormalizedIntent
from axor_core.contracts.taint import TaintSource
from axor_core.taint.causal_root import CausalRoot


def output_root(
    tool_name: str,
    normalized: NormalizedIntent | None,
    *,
    untrusted_sources: "frozenset[str] | set[str]",
    sensitive_sources: "frozenset[str] | set[str]",
) -> CausalRoot | None:
    """The causal_root a tool's output introduces, or ``None`` for a clean read.

    A deployment's declared role wins over the normalizer heuristic: a tool named
    as a sensitive source reads secrets (untrusted + sensitive, arming the floor);
    a tool named as an untrusted source reads attacker-influenceable data
    (untrusted). For any other tool the normalizer's structural classification
    decides — an external/network read is untrusted; a secret/system read is
    untrusted + sensitive; everything else is clean and arms nothing.
    """
    if tool_name in sensitive_sources:
        return CausalRoot.external_read(TaintSource.FILE, sensitive=True)
    if tool_name in untrusted_sources:
        return CausalRoot.external_read(TaintSource.WEB)
    if normalized is None:
        return None
    if (
        normalized.target_kind in ("external_url", "cloud_metadata", "docker_socket")
        or normalized.operation == "network_request"
    ):
        return CausalRoot.external_read(TaintSource.WEB)
    if (
        normalized.target_kind in ("secret", "system_path")
        or normalized.reads_secret_like_data
        or normalized.writes_outside_workdir
    ):
        sensitive = (
            normalized.target_kind == "secret" or normalized.reads_secret_like_data
        )
        return CausalRoot.external_read(TaintSource.FILE, sensitive=sensitive)
    return None


def source_tokens(sources: object) -> list[str]:
    """Taint sources as their declared string values (`web`, `mcp`, …).

    ``str()`` on a ``(str, Enum)`` member yields ``TaintSource.WEB`` — a Python
    implementation detail, and these tokens go into recorded governance
    artifacts that other languages read and hash.
    """
    return sorted(getattr(s, "value", None) or str(s) for s in sources or ())


def call_payload(
    tool_name: str,
    args: "dict[str, object]",
    *,
    taint: object,
    driving_args: "dict[str, frozenset[str]] | dict[str, set[str]] | None" = None,
) -> "dict[str, object]":
    """The provenance a tool-call verdict was reached on.

    The shape the kernel event schema documents for TOOL_CALL: ``arg_refs``,
    ``driving_args``, ``driving_root``, ``floor_active``.

    Shared by BOTH ways of wrapping an agent — ``ToolCallGovernor`` (the
    synchronous per-call path) and ``IntentLoop`` (the async streaming path) —
    because they are two entry points to one kernel, not two kernels. Each
    computed this provenance to reach a verdict and then recorded only the tool
    name, so every consumer had to re-derive it from raw arguments, which is how
    a second taint ledger gets built downstream. Living in one function is what
    stops the two paths from drifting into two vocabularies.
    """
    from axor_core.policy.gates import driving_subset

    driving = (driving_args or {}).get(tool_name)
    arg_refs: dict[str, object] = {}
    for name, value in (args or {}).items():
        root = taint.derive_value(value)  # type: ignore[attr-defined]
        arg_refs[name] = {
            "sources": source_tokens(root.sources),
            "sensitive": bool(root.sensitive),
        }
    driving_root = taint.derive_value(driving_subset(args or {}, driving))  # type: ignore[attr-defined]
    return {
        "tool": tool_name,
        "arg_refs": arg_refs,
        "driving_args": sorted(driving) if driving else [],
        "driving_root": {
            "sources": source_tokens(driving_root.sources),
            "sensitive": bool(driving_root.sensitive),
        },
        "floor_active": bool(taint.confidentiality_floor_active()),  # type: ignore[attr-defined]
    }
