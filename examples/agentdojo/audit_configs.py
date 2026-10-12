"""Every value in every config/*.yaml against every argument value of every
AgentDojo injection task's ground truth (the attacker's intended calls).
A shared value means a config names something the attacker targets.

Run: uv run --no-project --with agentdojo==0.1.35 --with pyyaml \
       python examples/agentdojo/audit_configs.py
"""
from __future__ import annotations

import glob
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def _leaves(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from _leaves(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _leaves(v)
    elif isinstance(x, str):
        yield x


def attacker_values():
    from agentdojo.task_suite.load_suites import get_suite
    out = {}
    for name in ("banking", "slack", "travel", "workspace"):
        suite = get_suite("v1", name)
        env = suite.load_and_inject_default_environment({})
        vals = set()
        for it in suite.injection_tasks.values():
            for call in it.ground_truth(env):
                vals |= {v for v in _leaves(call.args) if len(v) >= 4}
        out[name] = vals
    return out


def audit():
    att = attacker_values()
    every = set().union(*att.values())
    rows = {}
    for path in sorted(glob.glob(os.path.join(HERE, "config", "*.yaml"))):
        cfg = yaml.safe_load(open(path)) or {}
        names = set(_leaves({k: v for k, v in cfg.items() if k not in ("consequence_overrides",)}))
        rows[os.path.basename(path)] = sorted(names & every)
    return rows


if __name__ == "__main__":  # pragma: no cover
    for k, v in audit().items():
        print(f"{k:28} {v}")
