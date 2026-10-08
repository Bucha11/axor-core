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
is the provenance for the paper's ``origin 444 / consequence-only 60 / origin-only
0`` decomposition over the 510 dh base cases.

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

# Expected decomposition (fails the run if the core or taxonomy drifts from the paper).
_EXPECT_ORIGIN = 444
_EXPECT_CONS_ONLY = 60
_EXPECT_ORIGIN_ONLY = 0


def dh_refused(item: dict, consequence: bool) -> bool:
    """Replay the attacker egress with an injection-derived driving value; True iff
    the governor refuses it. consequence=False strips the consequence taxonomy, so
    only the provenance/taint axis can refuse (the origin-alone baseline)."""
    cfg = _CFG if consequence else _CFG_NOCONS
    g = ToolCallGovernor(**cfg.as_governor_kwargs())
    g.register_task(item.get("User Instruction", ""))  # trusted origin
    atk = item["Attacker Tools"][0]
    g.register_output(g.evaluate(item["User Tool"], {}), item["Tool Response"])  # untrusted read
    tok = _taintable_token(item.get("Attacker Instruction", "") + " " + item["Tool Response"])
    args = {(_DRIVING.get(atk) or ["arg"])[0]: tok} if tok else {}
    return not g.evaluate(atk, args).allowed


def main() -> int:
    path = os.path.join(INJECAGENT_DIR, "data", "test_cases_dh_base.json")
    data = json.load(open(path))
    n = len(data)
    origin, both = set(), set()
    for i, it in enumerate(data):
        if dh_refused(it, consequence=False):
            origin.add(i)
        if dh_refused(it, consequence=True):
            both.add(i)
    cons_only = both - origin   # consequence denies, origin alone would allow
    origin_only = origin - both  # (should be empty: consequence never un-denies)
    print(f"dh base n={n}")
    print(f"  denied origin alone        = {len(origin)}")
    print(f"  denied origin+consequence  = {len(both)}")
    print(f"  consequence-only           = {len(cons_only)}  (origin alone would ALLOW)")
    print(f"  origin-only                = {len(origin_only)}")
    ok = (len(origin) == _EXPECT_ORIGIN and len(cons_only) == _EXPECT_CONS_ONLY
          and len(origin_only) == _EXPECT_ORIGIN_ONLY)
    if not ok:
        print(f"MISMATCH vs paper: expected origin={_EXPECT_ORIGIN}, "
              f"consequence-only={_EXPECT_CONS_ONLY}, origin-only={_EXPECT_ORIGIN_ONLY}",
              file=sys.stderr)
        return 1
    print(f"OK: origin {len(origin)} / consequence-only {len(cons_only)} / "
          f"origin-only {len(origin_only)} — both axes load-bearing, consequence unmasked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
