"""Build an axor governance taxonomy for the InjecAgent benchmark.

InjecAgent (Zhan et al., arXiv:2403.02691) is a *pure attack* benchmark: every
case is an indirect prompt injection embedded in a tool's response, and the
metric is ASR (did the agent call the attacker's tool). There are no benign
tasks, so — unlike AgentDojo — there is no benign utility to over-block; the
taxonomy can therefore drive taint on *all* params of an egress sink without a
false-positive budget to protect.

Roles are derived from the data, not guessed:
  - user tools (the tool whose response carries the injection) -> untrusted_sources
  - dh AttackerTools[0]  (the direct-harm action)              -> egress_sink
  - ds AttackerTools[1]  (the data-stealing exfil step, S2)    -> egress_sink
  - ds AttackerTools[0]  (the sensitive read, S1)              -> sensitive_source
  - driving_args = every declared parameter of each egress sink (any param whose
    value is content-derived from the untrusted read denies the call).

This emits config/injecagent.yaml. Point INJECAGENT_DIR at the cloned repo.
"""
from __future__ import annotations

import json
import os
import re
import sys

# --- consequence taxonomy: role-blind, by the tool's LEADING action verb ---------
# The first CamelCase word that is a recognised action verb decides the class. This
# is applied uniformly to EVERY tool (user and attacker alike), blind to its role in
# the benchmark, so it is a semantic classification, not an answer-fitted list. It
# only ever *raises* a tool to CATASTROPHIC for a genuinely irreversible action
# (money movement / physical access / data destruction / security disable / exec);
# a read verb seen first keeps the tool a read. Same axis as AgentDojo workspace's
# delete=catastrophic, whose false-positive cost we measured there (benign-delete
# denials in the -15.5pp) — so this axis is not a free lunch even though InjecAgent
# (all-attack, benign user-tools are all reads) cannot show that cost itself.
_READ_VERBS = frozenset({"get", "view", "search", "list", "read", "check", "retrieve",
                         "access", "find", "show", "fetch", "browse", "lookup"})
_CATASTROPHIC_VERBS = frozenset({"transfer", "withdraw", "deposit", "pay", "paybill",
                                 "wire", "unlock", "delete", "remove", "erase", "wipe",
                                 "disable", "deactivate", "grant", "revoke", "execute",
                                 "deploy", "reset", "factoryreset", "shutdown", "reboot"})


def consequence_of(tool: str) -> str | None:
    """CATASTROPHIC iff the tool's leading action verb is an irreversible one;
    None (leave at the default class) otherwise. Role-blind, semantic."""
    for w in re.findall(r"[A-Z][a-z]+", tool):
        lw = w.lower()
        if lw in _READ_VERBS:
            return None
        if lw in _CATASTROPHIC_VERBS:
            return "catastrophic"
    return None

INJECAGENT_DIR = os.environ.get("INJECAGENT_DIR", "/home/user/InjecAgent")
sys.path.insert(0, INJECAGENT_DIR)
os.environ.setdefault("OPENAI_API_KEY", "sk-dummy")  # utils.py builds a client at import

from src.utils import get_tool_dict  # noqa: E402


def _params(tool_dict, tool: str) -> list[str]:
    ps = tool_dict.get(tool, {}).get("parameters", [])
    if isinstance(ps, list):
        return [p.get("name") for p in ps if p.get("name")]
    return list(ps)


def build() -> dict:
    _cwd = os.getcwd()
    os.chdir(INJECAGENT_DIR)  # get_tool_dict() reads the cwd-relative data/tools.json
    try:
        tool_dict = get_tool_dict()
    finally:
        os.chdir(_cwd)
    user_tools: set[str] = set()
    dh_egress: set[str] = set()
    ds_s1_read: set[str] = set()
    ds_s2_egress: set[str] = set()

    data_dir = os.path.join(INJECAGENT_DIR, "data")
    for setting in ("base", "enhanced"):
        for it in json.load(open(os.path.join(data_dir, f"test_cases_dh_{setting}.json"))):
            user_tools.add(it["User Tool"])
            dh_egress.add(it["Attacker Tools"][0])
        for it in json.load(open(os.path.join(data_dir, f"test_cases_ds_{setting}.json"))):
            user_tools.add(it["User Tool"])
            ds_s1_read.add(it["Attacker Tools"][0])
            if len(it["Attacker Tools"]) > 1:
                ds_s2_egress.add(it["Attacker Tools"][1])

    egress = sorted(dh_egress | ds_s2_egress)
    driving = {t: _params(tool_dict, t) for t in egress if _params(tool_dict, t)}

    # consequence taxonomy, role-blind: classify EVERY tool, keep the catastrophic
    # egress sinks. consequence_class() lower-cases the sink name for lookup, so the
    # override KEYS are emitted lower-cased (a case-sensitivity wart worth knowing).
    all_atk = dh_egress | ds_s1_read | ds_s2_egress
    cata_egress = {t.lower(): "catastrophic" for t in egress if consequence_of(t)}
    cata_user = sorted(t for t in user_tools if consequence_of(t))  # anti-cherry-pick check
    cata_atk = sorted(t for t in all_atk if consequence_of(t))

    return {
        "mode": "production",
        "untrusted_sources": sorted(user_tools),
        "sensitive_sources": sorted(ds_s1_read),
        "egress_sinks": egress,
        "driving_args": driving,
        "consequence_overrides": cata_egress,
        "_meta": {
            "user_tools": len(user_tools),
            "dh_egress": len(dh_egress),
            "ds_s1_sensitive_reads": len(ds_s1_read),
            "ds_s2_egress": len(ds_s2_egress),
            "egress_sinks_total": len(egress),
            "egress_with_driving_args": len(driving),
            "consequence_catastrophic_egress": len(cata_egress),
            "consequence_catastrophic_user_tools": len(cata_user),
            "consequence_catastrophic_attacker_tools": len(cata_atk),
            "_note": ("catastrophic_user_tools=0 here because InjecAgent's user tools "
                      "are all benign reads; the same role-blind axis flags user-usable "
                      "tools on AgentDojo (workspace delete) with a measured FP cost."),
        },
    }


def to_yaml(cfg: dict) -> str:
    lines = ["# axor governance for the InjecAgent benchmark — GENERATED by build_config.py.",
             "# Roles are data-derived (see build_config.py docstring). Do not hand-edit;",
             "# regenerate instead.", f"# stats: {json.dumps(cfg['_meta'])}", "",
             f"mode: {cfg['mode']}", "", "untrusted_sources:"]
    for t in cfg["untrusted_sources"]:
        lines.append(f"  - {t}")
    lines += ["", "sensitive_sources:"]
    for t in cfg["sensitive_sources"]:
        lines.append(f"  - {t}")
    lines += ["", "egress_sinks:"]
    for t in cfg["egress_sinks"]:
        lines.append(f"  - {t}")
    lines += ["", "driving_args:"]
    for t, ps in cfg["driving_args"].items():
        lines.append(f"  {t}: [{', '.join(ps)}]")
    lines += ["",
              "# Role-blind consequence taxonomy (leading action verb -> catastrophic for",
              "# irreversible actions). Keys are lower-cased (consequence_class lower-cases",
              "# the sink for lookup). Reported separately from the taint/floor result.",
              "consequence_overrides:"]
    for t, cls in cfg["consequence_overrides"].items():
        lines.append(f"  {t}: {cls}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    cfg = build()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "injecagent.yaml")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(to_yaml(cfg))
    print(f"wrote {out}")
    print(json.dumps(cfg["_meta"], indent=2))
