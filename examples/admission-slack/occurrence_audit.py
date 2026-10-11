"""E2 — how much of the measurement rests on an occurrence that could be accidental.

Under context-default integrity a model-emitted value is trusted when it *occurs*
in a trusted text: the kernel awards ``TrustedOrigin.TASK`` on a token-bounded span
match of the value against the user's request. Occurrence is not dependence. For a
short or numeric value — ``7``, ``Bob``, ``general`` — the match can happen for an
unrelated reason, and then the verdict rests on a coincidence rather than on the
user having named the object.

This audits the census for that: of the needed calls the monitor admits, how many
have a driving value whose trust is a span match shorter than a threshold, and how
many verdicts would change if such matches were withheld. It changes no verdict by
itself — it reports the sensitivity.

Reads the run records written by ``pairs.py``; no replay.

    python examples/admission-slack/occurrence_audit.py runs/
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any

THRESHOLDS = (4, 6, 8, 12)


def _fragile(value: Any, origin: str | None, min_len: int) -> bool:
    """Would withholding short/numeric span matches take this value's trust away?"""
    if origin != "task":
        return False
    text = str(value).strip()
    return len(text) < min_len or text.replace(".", "", 1).isdigit()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--json")
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    values: list[dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(args.run_dir, "*.json"))):
        with open(path) as fh:
            d = json.load(fh)
        if not isinstance(d, dict) or "summary" not in d:
            continue
        s = d["summary"]
        allowed = [c for c in d["calls"] if c["verdict"]["allowed"]]
        with_driving = [c for c in allowed if c.get("driving_values")]
        origins: dict[str, int] = {}
        for c in allowed:
            for arg, origin in (c.get("driving_origins") or {}).items():
                origins[str(origin)] = origins.get(str(origin), 0) + 1
                values.append({
                    "run": f"{s['suite']}·{s['config']}", "task": c["task"],
                    "tool": c["tool"], "arg": arg,
                    "value": c["driving_values"].get(arg), "origin": origin,
                })
        row = {
            "run": f"{s['suite']}·{s['config']}",
            "allowed_calls": len(allowed),
            "allowed_with_declared_driving_arg": len(with_driving),
            "driving_origins": origins,
        }
        for k in THRESHOLDS:
            row[f"fragile_at_{k}"] = sum(
                1 for c in allowed
                if any(_fragile(c["driving_values"].get(a), o, k)
                       for a, o in (c.get("driving_origins") or {}).items())
            )
        rows.append(row)

    print("## E2 — trust that rests on a span match, by minimum match length\n")
    print("`fragile@k` counts admitted needed calls with a driving value whose TASK")
    print("origin is a span match shorter than k characters (or a bare number).")
    print("`driving_origins` shows what the admitted calls' driving values were")
    print("trusted *as*: `task` is a span match in the request, `operator` is an")
    print("allowlist member, `null` means the value carried no trusted origin at all")
    print("(the call was admitted for another reason, e.g. supersession or because no")
    print("gate keys on that tool).\n")
    print("| run | admitted calls | with a declared driving arg | driving-value origins | "
          + " | ".join(f"fragile@{k}" for k in THRESHOLDS) + " |")
    print("|---|---|---|---|" + "---|" * len(THRESHOLDS))
    for r in rows:
        print(f"| {r['run']} | {r['allowed_calls']} | {r['allowed_with_declared_driving_arg']} "
              f"| {r['driving_origins']} | "
              + " | ".join(str(r[f'fragile_at_{k}']) for k in THRESHOLDS) + " |")

    task_vals = [v for v in values if v["origin"] == "task"]
    print(f"\n### Every admitted driving value trusted by a span match ({len(task_vals)})\n")
    if task_vals:
        print("| run | task | call | arg | value | length |")
        print("|---|---|---|---|---|---|")
        for v in sorted(task_vals, key=lambda x: (x["run"], x["task"])):
            text = str(v["value"])
            print(f"| {v['run']} | {v['task']} | `{v['tool']}` | `{v['arg']}` "
                  f"| `{text[:40]}` | {len(text)} |")
    else:
        print("None: no admitted needed call's driving value was trusted by a span")
        print("match in these runs — the admitted ones are operator-trusted allowlist")
        print("members or calls no gate keys on. The accidental-occurrence risk does")
        print("not touch the admissions measured here.")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"rows": rows, "values": values}, fh, indent=1)
        print(f"\n-> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
