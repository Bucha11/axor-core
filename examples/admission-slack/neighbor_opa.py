"""Step 5 — the same pairs through a real neighbouring policy engine (OPA).

The question is not "is OPA better or worse than axor". It is whether the excess
admission is a property of *this* monitor or of the whole class of per-call
declarative checks. So the same pairs are run through Open Policy Agent under two
explicitly stated configurations, both in ``neighbor/``:

``call_only.rego``   the engine sees the call and the operator's static facts
                     (declared sinks, approved destinations, bounds) — the usual
                     deployment, with no data-flow input.
``with_labels.rego`` the engine additionally sees axor's own per-value origin
                     labels and the floor boolean — the fair comparison: same
                     observations, different engine.

For each case the input JSON handed to OPA is written out beside the verdict, so
the configuration is auditable and the run reproducible.

Requires the ``opa`` binary (https://openpolicyagent.org); point ``--opa`` at it.

    python examples/admission-slack/neighbor_opa.py --opa ./opa --out neighbor-runs/
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from agentdojo.task_suite.load_suites import get_suites  # noqa: E402

from axor_core.policy.provenance import derive_driving_root, source_tokens  # noqa: E402

from cases import CASES  # noqa: E402
from enrichment import _cfg, install, replay_prefix  # noqa: E402
from observation import decide  # noqa: E402


def labels_for(governor: Any, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    """axor's per-value observation of each argument, as data for the engine.

    Derived through the kernel's own ``derive_driving_root`` per argument, with the
    integrity-sink flag the kernel would use for this tool — so a number in a
    driving argument is a checked value here exactly when it is there. Getting
    this detail wrong would make the engines disagree over a transcription bug
    rather than over the information they hold.
    """
    integrity_sink = tool in governor._integrity_sinks
    out: dict[str, Any] = {}
    taint = governor._taint
    for name, value in args.items():
        probe = value[0] if isinstance(value, list) and value else value
        root = derive_driving_root(taint, {name: probe}, [name],
                                   integrity_sink=integrity_sink)
        origin = taint.trusted_origin(probe)
        out[name] = {
            "sources": sorted(source_tokens(root.sources)),
            "tainted": bool(root.is_tainted),
            "sensitive": bool(root.sensitive),
            "trusted_origin": origin.value if origin is not None else None,
        }
    return out


def config_facts(cfg: Any) -> dict[str, Any]:
    return {
        "untrusted_sources": sorted(cfg.untrusted_sources),
        "sensitive_sources": sorted(cfg.sensitive_sources),
        "egress_sinks": sorted(cfg.egress_sinks),
        "integrity_sinks": sorted(cfg.integrity_sinks),
        "driving_args": {k: sorted(v) for k, v in cfg.driving_args.items()},
        "value_policies": {
            tool: [
                {
                    "arg": p.arg, "kind": p.kind,
                    # Raw values, not stringified: the kernel's enum check is
                    # type-exact (``value in set``), so the export must be too.
                    **({"allowed": sorted(p.allowed, key=str)} if p.kind == "enum"
                       else {"lo": p.lo, "hi": p.hi}),
                }
                for p in preds
            ]
            for tool, preds in cfg.value_policies.items()
        },
    }


def opa_eval(opa: str, policy: str, query: str, payload: dict[str, Any]) -> dict[str, Any]:
    proc = subprocess.run(
        [opa, "eval", "-d", os.path.join(_HERE, "neighbor", policy),
         "-I", "--format", "json", query],
        input=json.dumps(payload), capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"opa eval failed: {proc.stderr.strip()[:400]}")
    out = json.loads(proc.stdout)
    try:
        return out["result"][0]["expressions"][0]["value"]
    except (KeyError, IndexError):
        return {"allowed": None, "reasons": ["opa returned no value"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--opa", default="opa", help="path to the opa binary")
    ap.add_argument("--out", help="directory for the input/verdict records")
    ap.add_argument("--ground", default="as_deployed",
                    help="which step-4 ground to install before comparing")
    args = ap.parse_args()

    rows = []
    for case in CASES:
        suite_obj = get_suites("v1")[case.suite]
        cfg = install(
            _cfg(case.suite, case.config), args.ground, suite_obj=suite_obj, case=case,
            tools_to_gate={case.tool},
        )
        task = suite_obj.get_user_task_by_id(case.task)
        governor, _env, _rt, plan = replay_prefix(suite_obj, task, case.step, cfg)
        call = plan[case.step]
        orig_args = dict(call.args)
        value = orig_args.get(case.arg)
        cand = case.candidate
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                cand = type(value)(cand)
            except Exception:
                pass
        sub_args = dict(orig_args)
        sub_args[case.arg] = [cand] if isinstance(value, list) else cand

        facts = config_facts(cfg)
        row: dict[str, Any] = {"case": case.name, "tool": case.tool, "arg": case.arg,
                               "ground": args.ground}
        for label, call_args in (("needed", orig_args), ("alternative", sub_args)):
            axor = decide(governor, case.tool, call_args)
            payload = {
                "tool": case.tool,
                "args": {k: (v[0] if isinstance(v, list) and v else v)
                         for k, v in call_args.items()},
                "labels": labels_for(governor, case.tool, call_args),
                "floor_active": bool(governor._taint.confidentiality_floor_active()),
                "config": facts,
            }
            call_only = opa_eval(args.opa, "call_only.rego", "data.neighbor.decision", payload)
            with_labels = opa_eval(
                args.opa, "with_labels.rego", "data.neighbor_labels.decision", payload
            )
            row[label] = {
                "axor": axor.as_dict(),
                "opa_call_only": call_only,
                "opa_with_labels": with_labels,
            }
            if args.out:
                os.makedirs(args.out, exist_ok=True)
                with open(os.path.join(args.out, f"{case.name}-{label}-{args.ground}.json"),
                          "w") as fh:
                    json.dump(payload, fh, indent=1)
        rows.append(row)

    print(f"## The same pairs through OPA (ground: {args.ground})\n")
    print("| case | axor: needed / alternative | OPA call-only | OPA with axor's labels | separates? |")
    print("|---|---|---|---|---|")
    for r in rows:
        a = (f"{'allow' if r['needed']['axor']['allowed'] else 'deny'} / "
             f"{'allow' if r['alternative']['axor']['allowed'] else 'deny'}")
        c = (f"{'allow' if r['needed']['opa_call_only']['allowed'] else 'deny'} / "
             f"{'allow' if r['alternative']['opa_call_only']['allowed'] else 'deny'}")
        w = (f"{'allow' if r['needed']['opa_with_labels']['allowed'] else 'deny'} / "
             f"{'allow' if r['alternative']['opa_with_labels']['allowed'] else 'deny'}")
        sep = []
        for name, key in (("axor", "axor"), ("call-only", "opa_call_only"),
                          ("with-labels", "opa_with_labels")):
            if r["needed"][key]["allowed"] and not r["alternative"][key]["allowed"]:
                sep.append(name)
        print(f"| {r['case']} | {a} | {c} | {w} | {', '.join(sep) or 'none'} |")
    if args.out:
        with open(os.path.join(args.out, f"verdicts-{args.ground}.json"), "w") as fh:
            json.dump(rows, fh, indent=1)
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
