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
