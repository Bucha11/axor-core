"""Shared origin-only (no-grants) ablation, used by BOTH harnesses so the deployment
is byte-identical:

  * the AgentDojo bench  — run_agentdojo.make_governor, gated by AXOR_BENCH_NO_GRANTS=1
  * the Ajar/ROPE path   — axor_ajar's ``no_grants`` policy

Harness-only; NOT a kernel rule. The kernel (axor_core) stays frozen — this lives in
harness code and only transforms a GovernanceConfig before it is handed to the governor.

It strips every TASK-SCOPED GRANT SOURCE (L1 request literals, L2 operator/tuple
templates, L4 confirmer grants) together with the description / delegated / list grounds
that feed them, and restores the default unattended consequence ceiling. Then no
grant_source / grant_gate decision can arise — only the NON-grant gates decide:
origin (references / reference_origin), consequence, and operator-set refusal.

``operator_sets`` and ``value_policies`` are deliberately NOT touched: the operator-set
refusal gate is part of "origin-only" as defined across the paper (origin + handles +
operator-set refusal), not a grant.
"""
from __future__ import annotations

import dataclasses

from axor_core.config import GovernanceConfig
from axor_core.contracts.canonical import ConsequenceClass


def strip_grants(cfg: GovernanceConfig) -> GovernanceConfig:
    """Return ``cfg`` with all task-scoped grant machinery removed (origin-only)."""
    return dataclasses.replace(
        cfg,
        grant_args={},
        grant_templates={},
        grant_quotas={},
        description_sources={},
        description_targets={},
        delegated_sources={},
        list_union=False,
        list_args={},
        max_unattended_consequence=ConsequenceClass.CONSEQUENTIAL,
    )
