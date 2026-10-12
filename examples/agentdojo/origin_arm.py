"""The ORIGIN arm of the AgentDojo experiments — what it is, as code.

The ``*_origin.yaml`` configs are an **origin-only baseline**, mirroring ROPE's
PROMPT/SOURCED markers through Axor's integrity axis (axor-eval
``experiments/rope_bridge``). They are deliberately NOT a STRICT arm:

  * STRICT obligations it DOES carry: role completeness (every tool has a declared
    data-flow role) and a complete consequence table (fail-closed on an
    unclassified tool) — applied here, not by ``mode:``;
  * STRICT obligations it does NOT carry: the egress destination allowlist (and so
    no value predicates, no consumption contracts, no enum supersession). The
    origin axis — ``integrity_default: context`` — decides every driving value.

So the guarantees of this arm are: integrity by origin (context root, trusted
origins), the confidentiality floor on declared sensitive sources, and the
consequence ceiling. Not claimed: an allowlisted destination, magnitude bounds,
or T4 for any argument. The YAMLs declare ``mode: production`` because that is
what the arm's obligations are; :func:`origin_arm_kwargs` then adds exactly the
two STRICT obligations above. It has no AgentDojo import, so it can be tested
without the benchmark.
"""
from __future__ import annotations

from typing import Iterable


def origin_arm_kwargs(kw: dict, tool_names: Iterable[str]) -> dict:
    """Governor kwargs for the origin arm, from a config's ``as_governor_kwargs``
    and the suite's full tool universe."""
    from axor_core.contracts.canonical import ConsequenceClass as _C

    out = dict(kw)
    out["require_tool_roles"] = True
    out["require_egress_allowlist"] = False
    # A COMPLETE per-tool consequence table (mirrors rope_bridge
    # AxorOriginBootstrap._consequence): every tool is BENIGN unless it is a
    # declared egress/integrity sink, pinned at the default unattended ceiling
    # (CONSEQUENTIAL) so the origin axis decides. An explicit YAML override wins.
    sinks = set(out.get("egress_sinks") or ()) | set(out.get("integrity_sinks") or ())
    auto = {t: (_C.CONSEQUENTIAL if t in sinks else _C.BENIGN) for t in tool_names}
    auto.update(out.get("consequence_overrides") or {})
    out["consequence_overrides"] = auto
    return out
