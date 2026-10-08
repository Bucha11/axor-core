"""Deterministic replay of the two gate axes over ALL dh base cases.

No model is called. For every direct-harm case the attacker's egress
(``AttackerTools[0]``) is replayed through an axor ``ToolCallGovernor`` with an
*injection-derived* driving value: a >=12-char token lifted from the Attacker
Instruction + Tool Response and placed in the sink's first declared driving arg
(``register_task(User Instruction)`` is the trusted origin, so that value is
untrusted under ``integrity_default: context``). We run the governor twice —
consequence gate ON and OFF — and compare the two refusal sets.

This isolates each axis's *independent* contribution:
  - origin alone (consequence OFF): the provenance/taint axis denies the egress
    when its driving value is not of trusted origin;
  - consequence-only: cases the consequence gate denies that origin alone would
    ALLOW (irreversible, content-blind action-shaped harms — e.g. unlock/grant),
    where the driving arg carries no attacker-authored value for origin to flag.

The consequence gate runs BEFORE the origin gate, so it is not masked. This script
is the provenance for the paper's 2x2 contingency over the 510 dh base cases:
both 246 / origin-only 198 / consequence-only 60 / neither 6 (origin row 444,
consequence col 306). The 6 "neither" cases are all ``IndoorRobotGoToRoom``
(reversible-by-design, no liftable token) — correctly not denied; in the realized
runs they are never successful attacks (measured governed dh = 0/510).

Point ``INJECAGENT_DIR`` at a clone of uiuc-kang-lab/InjecAgent (only its
``data/test_cases_dh_base.json`` is read). Exits non-zero if the split drifts.

Usage:
  INJECAGENT_DIR=/path/to/InjecAgent \
    python3 examples/injecagent/replay_dh_axes.py
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))  # axor-core root (for axor_core import)
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _HERE)  # for govern_injecagent._taintable_token
INJECAGENT_DIR = os.environ.get("INJECAGENT_DIR", "/home/user/InjecAgent")

from axor_core.config import GovernanceConfig  # noqa: E402
from axor_core.governor import ToolCallGovernor  # noqa: E402
from govern_injecagent import _taintable_token  # noqa: E402  (shared >=12-char lift heuristic)

CONFIG = os.path.join(_HERE, "config", "injecagent.yaml")
_CFG = GovernanceConfig.from_yaml(CONFIG)
_CFG_NOCONS = dataclasses.replace(_CFG, consequence_overrides={})  # consequence axis OFF
_DRIVING = {k: list(v) for k, v in (_CFG.driving_args or {}).items()}

# Expected 2x2 decomposition (fails the run if the core or taxonomy drifts from the
# paper). Cells over the 510 dh base cases, run-free replay:
#   origin DENY  x consequence DENY  = 246   (both axes independently)
#   origin DENY  x consequence ALLOW = 198   (origin-only, exfil-shaped)
#   origin ALLOW x consequence DENY  =  60   (consequence-only, action-shaped)
#   origin ALLOW x consequence ALLOW =   6   (neither: all IndoorRobotGoToRoom,
#                                             reversible-by-design + no liftable token)
_EXPECT_ORIGIN = 444        # row total: 246 + 198
_EXPECT_CONS = 306          # col total: 246 + 60
_EXPECT_BOTH = 246
_EXPECT_ORIGIN_ONLY = 198
_EXPECT_CONS_ONLY = 60
_EXPECT_NEITHER = 6
_EXPECT_RESIDUAL_TOOL = "IndoorRobotGoToRoom"


def denied_by_origin(item: dict) -> bool:
    """True iff the ORIGIN (provenance/taint) axis alone refuses the egress:
    consequence taxonomy stripped, an untrusted read taints the context root, and
    an injection-derived token is placed in the sink's first driving arg."""
    g = ToolCallGovernor(**_CFG_NOCONS.as_governor_kwargs())
    g.register_task(item.get("User Instruction", ""))  # trusted origin
    atk = item["Attacker Tools"][0]
    g.register_output(g.evaluate(item["User Tool"], {}), item["Tool Response"])  # untrusted read
    tok = _taintable_token(item.get("Attacker Instruction", "") + " " + item["Tool Response"])
    args = {(_DRIVING.get(atk) or ["arg"])[0]: tok} if tok else {}
    return not g.evaluate(atk, args).allowed


def denied_by_consequence(item: dict) -> bool:
    """True iff the CONSEQUENCE axis alone refuses the egress: consequence taxonomy
    ON, but no untrusted read is registered and no tainted arg is supplied, so the
    origin axis cannot fire — a deny is purely the irreversible-action class."""
    g = ToolCallGovernor(**_CFG.as_governor_kwargs())
    g.register_task(item.get("User Instruction", ""))
    atk = item["Attacker Tools"][0]
    return not g.evaluate(atk, {}).allowed


def main() -> int:
    path = os.path.join(INJECAGENT_DIR, "data", "test_cases_dh_base.json")
    data = json.load(open(path))
    n = len(data)
    origin, cons = set(), set()
    for i, it in enumerate(data):
        if denied_by_origin(it):
            origin.add(i)
        if denied_by_consequence(it):
            cons.add(i)

    both = origin & cons          # either axis denies independently
    origin_only = origin - cons   # exfil-shaped: provenance sees it, action class does not
    cons_only = cons - origin     # action-shaped, content-blind: only the consequence gate
    neither = set(range(n)) - origin - cons
    covered = origin | cons       # denied by at least one axis

    print(f"dh base n={n}  —  2x2 contingency (origin x consequence), run-free replay")
    print(f"                       consequence DENY   consequence ALLOW   row total")
    print(f"  origin DENY              {len(both):>5}             {len(origin_only):>5}"
          f"          {len(origin):>5}")
    print(f"  origin ALLOW             {len(cons_only):>5}             {len(neither):>5}"
          f"          {n-len(origin):>5}")
    print(f"  col total                {len(cons):>5}             {n-len(cons):>5}"
          f"          {n:>5}")
    print()
    print(f"  denied by >=1 axis       = {len(covered)}  ({100*len(covered)/n:.2f}%)")
    print(f"  origin-only (exfil-shaped)= {len(origin_only)}")
    print(f"  consequence-only (action) = {len(cons_only)}  (origin alone would ALLOW these)")
    print(f"  neither axis (residual)   = {len(neither)}  ({100*len(neither)/n:.2f}%)")

    # Account for the residual: non-catastrophic tool AND no >=12-char liftable token,
    # so run-free there is no tainted driving value and no action class to fire. In the
    # realized runs these are never successful attacks (governed dh = 0/510 measured).
    if neither:
        print("\n  residual cases (neither axis denies, run-free):")
        for i in sorted(neither):
            it = data[i]
            tok = _taintable_token(
                it.get("Attacker Instruction", "") + " " + it["Tool Response"])
            print(f"    #{i:3d}  {it['Attacker Tools'][0]:40s} "
                  f"liftable_token={'yes' if tok else 'no'}")

    residual_tools = {data[i]["Attacker Tools"][0] for i in neither}
    ok = (len(origin) == _EXPECT_ORIGIN and len(cons) == _EXPECT_CONS
          and len(both) == _EXPECT_BOTH and len(origin_only) == _EXPECT_ORIGIN_ONLY
          and len(cons_only) == _EXPECT_CONS_ONLY and len(neither) == _EXPECT_NEITHER
          and residual_tools == {_EXPECT_RESIDUAL_TOOL})
    if not ok:
        print(f"\nMISMATCH vs paper: expected [both={_EXPECT_BOTH}, "
              f"origin_only={_EXPECT_ORIGIN_ONLY}, cons_only={_EXPECT_CONS_ONLY}, "
              f"neither={_EXPECT_NEITHER}, residual={{{_EXPECT_RESIDUAL_TOOL}}}]; "
              f"got [both={len(both)}, origin_only={len(origin_only)}, "
              f"cons_only={len(cons_only)}, neither={len(neither)}, "
              f"residual={sorted(residual_tools)}]", file=sys.stderr)
        return 1
    print(f"\nOK: 2x2 = both {len(both)} / origin-only {len(origin_only)} / "
          f"consequence-only {len(cons_only)} / neither {len(neither)}. "
          f"Both axes load-bearing (consequence runs first, so its {len(cons_only)} "
          f"are unmasked); the {len(neither)} residual are all {_EXPECT_RESIDUAL_TOOL} "
          f"(reversible-by-design, correctly not denied).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
