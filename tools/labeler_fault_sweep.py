#!/usr/bin/env python3
"""Replay a recorded trace under every integrity-labeler fault arm.

This is the instrument for the three fault-injection items in
``docs/floor-dependencies.md`` §3 that need captured trajectories:

* the C2 invariant over real runs (not just the synthetic trace in
  ``tests/kernel/test_floor_labeler_independence.py``);
* **benign** AgentDojo trajectories under ``ALL_TRUSTED`` — utility must climb
  toward undefended while consequence-gate refusals stay, which is what
  separates "the defence holds" from "everything is refused";
* attribution of the InjecAgent cases whose refusal rests on origin alone — the
  ``integrity-only`` column names them per trace.

It is pure replay: no model, no network, no cost. Every arm re-gates the same
recorded events, so a sweep is as reproducible as the trace it reads.

Usage
-----
    python tools/labeler_fault_sweep.py TRACE.jsonl [TRACE.jsonl ...] \\
        [--config governance.yaml] [--seed s] [--json]

``TRACE.jsonl`` is one kernel event per line (``kernel.events.event_to_json_line``
format, as the collector writes). ``--config`` is the deployment's governance
YAML; without it the sweep runs on an empty declaration, which means
``egress_sinks`` is empty and the only ground for ``is_exfil`` is the
normalizer's ``destination_kind`` — honest, but it understates egress for any
deployment that declares its sinks, so pass the config you actually ran.

Exit status is 1 if the C2 invariant fails on any trace: the set of calls whose
``confidentiality_risk`` predicate is True must be identical across all arms.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from axor_core.config import GovernanceConfig  # noqa: E402
from axor_core.kernel.events import Event, EventKind, Verdict, event_from_json_line  # noqa: E402
from axor_core.kernel.labeler_fault import LabelerFault  # noqa: E402
from axor_core.kernel.replay import KernelConfig, replay  # noqa: E402


def load_trace(path: pathlib.Path) -> list[Event]:
    events = [
        event_from_json_line(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    return sorted(events, key=lambda e: e.seq)


def base_config(config_path: pathlib.Path | None) -> GovernanceConfig:
    if config_path is None:
        return GovernanceConfig()
    return GovernanceConfig.from_yaml(str(config_path))


def sweep_one(
    events: list[Event], governance: GovernanceConfig, seed: str
) -> dict[str, dict[str, object]]:
    """Per-arm summary for one trace."""
    out: dict[str, dict[str, object]] = {}
    for arm in LabelerFault:
        config: KernelConfig = governance.as_kernel_config(
            labeler_fault=arm, labeler_fault_seed=seed
        )
        result = replay(events, config)
        calls = [s for s in result.steps if s.event.kind is EventKind.TOOL_CALL]
        denied = [s for s in calls if s.reevaluated_verdict is Verdict.DENY]
        by_category = Counter(s.deny.category for s in denied if s.deny is not None)
        risk = sorted(s.event.seq for s in calls if s.confidentiality_risk)
        # Refused while NOTHING but the integrity axis speaks for them: these are
        # the calls a broken integrity labeler actually releases, and the ones the
        # InjecAgent attribution needs named.
        integrity_only = sorted(
            s.event.seq for s in denied if not s.confidentiality_risk
        )
        out[arm.value] = {
            "calls": len(calls),
            "allowed": len(calls) - len(denied),
            "denied": len(denied),
            "denied_by_category": dict(sorted(by_category.items())),
            "confidentiality_risk_calls": risk,
            "integrity_only_denials": integrity_only,
            "first_divergence": result.first_divergence,
        }
    return out


def _fmt_table(name: str, arms: dict[str, dict[str, object]]) -> str:
    head = f"{'arm':<13}{'calls':>6}{'allow':>7}{'deny':>6}{'conf-risk':>11}{'int-only':>10}"
    lines = [f"\n{name}", "-" * len(head), head, "-" * len(head)]
    for arm, row in arms.items():
        lines.append(
            f"{arm:<13}{row['calls']:>6}{row['allowed']:>7}{row['denied']:>6}"
            f"{len(row['confidentiality_risk_calls']):>11}"  # type: ignore[arg-type]
            f"{len(row['integrity_only_denials']):>10}"  # type: ignore[arg-type]
        )
    for arm, row in arms.items():
        cats = row["denied_by_category"]
        if cats:
            lines.append(f"  {arm}: {cats}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=pathlib.Path)
    parser.add_argument("--config", type=pathlib.Path, default=None,
                        help="governance YAML the run was governed by")
    parser.add_argument("--seed", default="sweep",
                        help="arm seed for the per-ref fault mode")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    governance = base_config(args.config)
    report: dict[str, object] = {}
    failures: list[str] = []

    for path in args.traces:
        arms = sweep_one(load_trace(path), governance, args.seed)
        report[str(path)] = arms
        risk_sets = {arm: tuple(r["confidentiality_risk_calls"]) for arm, r in arms.items()}  # type: ignore[arg-type]
        if len(set(risk_sets.values())) != 1:
            failures.append(f"{path}: confidentiality_risk moved under fault: {risk_sets}")
        control = arms[LabelerFault.NONE.value]
        if control["first_divergence"] is not None:
            failures.append(
                f"{path}: control arm diverges from the recorded verdicts at step "
                f"{control['first_divergence']} — the config does not match the run, "
                f"so every other arm is measuring the wrong thing"
            )
        if not args.json:
            print(_fmt_table(str(path), arms))

    if args.json:
        print(json.dumps({"traces": report, "failures": failures}, indent=2))
    elif failures:
        print("\nC2 INVARIANT FAILED")
        for f in failures:
            print(f"  {f}")
    else:
        print("\nC2 invariant holds on every trace: the set of calls carrying "
              "confidentiality risk is identical in all arms.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
