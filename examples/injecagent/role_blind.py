"""InjecAgent with roles assigned without the benchmark's attacker/user split.

``config/injecagent.yaml`` takes tool roles from InjecAgent's ground truth (which
tools are user tools, which are attacker tools). Here every tool of
``data/tools.json`` is classified by its name alone: a tool whose leading action
verb is a read verb is an untrusted source; every other tool is an egress sink
whose driving arguments are all its parameters. The consequence classes are the
same role-blind, verb-based ones (``build_config.consequence_of``). The
direct-harm cases are then replayed as in ``govern_injecagent.account_dh``.

Reported: refusals over the 510 direct-harm cases, and the classification errors
the labels would have hidden — user tools classified as sinks (each a false
positive on a benign call after an untrusted read) and attacker tools classified
as reads (each a missed sink).

Run: INJECAGENT_DIR=<clone> python examples/injecagent/role_blind.py
"""
from __future__ import annotations

import collections
import json
import os
import re

import govern_injecagent as G
from build_config import _READ_VERBS, consequence_of
from axor_core.config import GovernanceConfig
from axor_core.governor import ToolCallGovernor

_WORD = re.compile(r"[A-Z][a-z]+|[a-z]+")


def _is_read(tool: str, toolkit: str) -> bool:
    words = [w.lower() for w in _WORD.findall(tool[len(toolkit):] or tool)]
    return bool(words) and words[0] in _READ_VERBS


def role_blind_config() -> "tuple[GovernanceConfig, dict]":
    with open(os.path.join(G.INJECAGENT_DIR, "data", "tools.json")) as f:
        kits = json.load(f)
    reads, sinks, driving, cons = set(), set(), {}, {}
    for kit in kits:
        for t in kit["tools"]:
            name = kit["toolkit"] + t["name"]
            params = [p["name"] for p in t.get("parameters", [])]
            c = consequence_of(name)
            if c:
                cons[name.lower()] = c  # the consequence table is keyed lower-case
            if _is_read(name, kit["toolkit"]):
                reads.add(name)
                cons.setdefault(name.lower(), "benign")
            else:
                sinks.add(name)
                if params:
                    driving[name] = params
    cfg = GovernanceConfig.from_dict({
        "mode": "production", "integrity_default": "context",
        "untrusted_sources": sorted(reads), "egress_sinks": sorted(sinks),
        "driving_args": driving, "consequence_overrides": cons,
    })
    return cfg, {"reads": reads, "sinks": sinks}


def main() -> None:
    cfg, roles = role_blind_config()
    with open(os.path.join(G.INJECAGENT_DIR, "data", "test_cases_dh_base.json")) as f:
        items = json.load(f)
    params = G._tool_params()
    driving = {k: list(v) for k, v in (cfg.driving_args or {}).items()}
    G._governor = lambda consequence=True: ToolCallGovernor(**cfg.as_governor_kwargs())
    verdicts = collections.Counter()
    for item in items:
        r = G.govern_dh_schema(item, driving, params)
        verdicts[r["reason"] if r["denied"] else "allow"] += 1
    user_tools = {i["User Tool"] for i in items}
    attacker_tools = {i["Attacker Tools"][0] for i in items}
    print({"cases": len(items), "verdicts": dict(verdicts),
           "user_tools": len(user_tools),
           "user_tools_classified_as_sinks": sorted(user_tools & roles["sinks"]),
           "attacker_tools": len(attacker_tools),
           "attacker_tools_classified_as_reads": sorted(attacker_tools & roles["reads"])})


if __name__ == "__main__":
    main()
