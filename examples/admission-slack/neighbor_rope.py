"""C1 — the same pairs through ROPE's own gate, on ROPE's own code and policies.

ROPE (https://github.com/xhOwenMa/ROPE) gates each state-changing tool argument by
*where the value came from*: a per-task scope (synthesised by a router that reads
only the trusted request) is compiled against an audited global floor into
per-argument markers — ``PROMPT`` (the value occurs in the request), ``SOURCED``
(the request, or a named non-injectable source), ``RECORD``, ``CONST``/``ONEOF``
(pinned), ``DEST``, ``EXPLICIT``, ``FREE`` — and enforced by ``rope.enforce.Enforcer``.

Nothing here re-implements that. The script imports their modules, loads the
**cached router scopes they publish** (``src/rope/scopes/<router>/<suite>.json``)
and their audited floor, builds the origin map with their own ``OriginTracker``
while replaying the reference plan, and asks their ``Enforcer`` about the identical
needed/substituted calls this experiment asked axor about. So the comparison is
model-free and uses their published policy, not our guess at it.

What it answers: is the admission slack a property of *this* monitor, or of the
class of per-call origin checks? A pair their gate separates and ours cannot is a
falsifier — it would mean their observation language is finer than our table says.

    python examples/admission-slack/neighbor_rope.py --records runs/ --out runs/rope/

Requires a checkout of ROPE (``--rope-src <checkout>/src``) and ``jsonschema``.
Suites: banking, slack, travel — ROPE publishes no workspace scopes.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

ROPE_SUITES = ("banking", "slack", "travel")


def _load_rope(rope_src: str):
    if rope_src not in sys.path:
        sys.path.insert(0, rope_src)
    from rope import origin as rope_origin
    from rope import policy_synthesis as rope_ps
    from rope.enforce import Enforcer, ValidationError
    from rope.scopes_io import load_router_scopes
    return rope_origin, rope_ps, Enforcer, ValidationError, load_router_scopes


def _norm(text: str) -> str:
    """Whitespace-normalised prompt, for looking a scope up.

    ROPE keys its cached scopes by the request string. Some AgentDojo travel prompts
    carry leading indentation, which makes the exact key miss; the lookup falls back
    to this normal form and the result says which pairs needed it.
    """
    return re.sub(r"\s+", " ", (text or "").strip())


def load_pairs(records_dir: str, configs: set[str] | None) -> dict[tuple[str, str], list[dict]]:
    """The pairs this experiment measured, grouped by (suite, task)."""
    out: dict[tuple[str, str], list[dict]] = {}
    for path in sorted(glob.glob(os.path.join(records_dir, "*.json"))):
        with open(path) as fh:
            d = json.load(fh)
        if not isinstance(d, dict) or "summary" not in d:
            continue
        s = d["summary"]
        if s["suite"] not in ROPE_SUITES:
            continue
        if configs is not None and s["config"] not in configs:
            continue
        for p in d["pairs"]:
            out.setdefault((s["suite"], p["task"]), []).append({**p, "config": s["config"]})
    return out


def run(records_dir: str, *, rope_src: str, router: str, configs: set[str] | None,
        out_dir: str | None) -> list[dict[str, Any]]:
    rope_origin, rope_ps, Enforcer, ValidationError, load_router_scopes = _load_rope(rope_src)
    from agentdojo.agent_pipeline.tool_execution import tool_result_to_str
    from agentdojo.functions_runtime import FunctionsRuntime
    from agentdojo.task_suite.load_suites import get_suites

    pairs_by_task = load_pairs(records_dir, configs)
    rows: list[dict[str, Any]] = []
    missing_scope: list[str] = []

    for suite_name in ROPE_SUITES:
        tasks = sorted({t for (s, t) in pairs_by_task if s == suite_name})
        if not tasks:
            continue
        suite_obj = get_suites("v1")[suite_name]
        floor, scopes = load_router_scopes(router, suite_name)
        by_norm = {_norm(k): v for k, v in scopes.items()}

        for task_id in tasks:
            task = suite_obj.get_user_task_by_id(task_id)
            prompt = task.PROMPT
            scope = scopes.get(prompt) or by_norm.get(_norm(prompt))
            if scope is None:
                missing_scope.append(f"{suite_name}/{task_id}")
                continue
            exact = prompt in scopes
            env = suite_obj.load_and_inject_default_environment({})
            try:
                plan = list(task.ground_truth(env))
            except Exception:
                continue
            runtime = FunctionsRuntime(suite_obj.tools)

            origin_map: dict = {}
            tracker = rope_origin.OriginTracker(origin_map)
            prompt_ids = rope_origin.tokenize_identifiers(prompt)
            enforcer = Enforcer()
            enforcer.update_available_tools([
                {"name": t.name, "description": t.description,
                 "args": t.parameters.model_json_schema().get("properties", {})}
                for t in suite_obj.tools
            ])
            enforcer.set_user_query(prompt)
            # The policy is compiled once against the live origin_map; the predicates
            # hold a reference to it, so later reads widen them exactly as they do in
            # ROPE's own loop.
            policy = rope_ps.compile_policy(
                floor, scope, origin_map=origin_map, prompt_ids=prompt_ids
            )
            enforcer.update_security_policy(policy)
            enforcer.update_always_allowed_tools(
                [t.name for t in suite_obj.tools if t.name not in floor]
            )

            def verdict(tool: str, args: dict[str, Any]) -> tuple[bool, str]:
                try:
                    enforcer.check_tool_call(tool, dict(args))
                except ValidationError as exc:
                    return False, str(exc)[:200]
                except Exception as exc:  # a malformed restriction is a refusal, not a pass
                    return False, f"{type(exc).__name__}: {exc}"[:200]
                return True, "allowed"

            for step, call in enumerate(plan):
                tool, args = call.function, dict(call.args)
                here = [p for p in pairs_by_task[(suite_name, task_id)] if p["step"] == step]
                if here:
                    n_ok, n_why = verdict(tool, args)
                    for p in here:
                        sub_args = dict(args)
                        original = args.get(p["arg"])
                        cand = p["candidate_value"]
                        sub_args[p["arg"]] = (
                            [cand] if isinstance(original, list) else cand
                        )
                        s_ok, s_why = verdict(tool, sub_args)
                        rows.append({
                            "suite": suite_name, "task": task_id, "step": step,
                            "tool": tool, "arg": p["arg"], "config": p["config"],
                            "needed_value": p["original_value"],
                            "candidate_value": cand,
                            "scope_lookup": "exact" if exact else "normalised",
                            "bucket": scope.bucket,
                            "axor": {
                                "needed": p["orig_verdict"]["allowed"],
                                "candidate": p["sub_verdict"]["allowed"],
                                "pi_equal": p["pi_equal"],
                                "cause": p["cause"],
                                "same_slot": p["same_slot"],
                                "dispatch_ok": p["dispatch_ok"],
                            },
                            "rope": {"needed": n_ok, "candidate": s_ok,
                                     "needed_why": n_why, "candidate_why": s_why},
                        })
                out, err = runtime.run_function(env, tool, args)
                if err is None:
                    msg = {"role": "tool",
                           "content": [{"content": tool_result_to_str(out)}],
                           "tool_call": call, "error": None}
                    tracker.query(prompt, runtime, env, [msg], {})

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"rope-{router}.json"), "w") as fh:
            json.dump({"rows": rows, "missing_scope": missing_scope}, fh, indent=1)
    if missing_scope:
        print(f"<!-- no cached scope for: {', '.join(sorted(set(missing_scope)))} -->")
    return rows


def report(rows: list[dict[str, Any]], router: str) -> None:
    sep_axor = [r for r in rows if r["axor"]["needed"] and not r["axor"]["candidate"]]
    sep_rope = [r for r in rows if r["rope"]["needed"] and not r["rope"]["candidate"]]
    forced = [r for r in rows if r["axor"]["needed"] and r["axor"]["candidate"]
              and r["axor"]["pi_equal"] and r["axor"]["dispatch_ok"]]
    forced_ref = [r for r in forced if r["axor"]["same_slot"]]
    rope_also = [r for r in forced_ref if r["rope"]["needed"] and r["rope"]["candidate"]]
    rope_splits = [r for r in forced_ref if r["rope"]["needed"] and not r["rope"]["candidate"]]
    rope_refuses_needed = [r for r in forced_ref if not r["rope"]["needed"]]

    print(f"## C1 — the same pairs through ROPE's gate (cached router: {router})\n")
    print("| quantity | pairs |")
    print("|---|---|")
    print(f"| pairs replayed through both gates | {len(rows)} |")
    print(f"| axor: needed admitted, substitution admitted, same projection, executable | {len(forced)} |")
    print(f"| …of those, referential (another record of the same kind) | {len(forced_ref)} |")
    print(f"| **ROPE admits both as well** | **{len(rope_also)}** |")
    print(f"| ROPE separates a pair axor cannot | {len(rope_splits)} |")
    print(f"| ROPE refuses the needed call (no evidence either way) | {len(rope_refuses_needed)} |")
    print(f"| pairs axor separates | {len(sep_axor)} |")
    print(f"| pairs ROPE separates | {len(sep_rope)} |")
    agree = sum(1 for r in rows
                if r["axor"]["needed"] == r["rope"]["needed"]
                and r["axor"]["candidate"] == r["rope"]["candidate"])
    print(f"| both verdicts identical on both calls | {agree}/{len(rows)} |")

    if rope_splits:
        print("\n### Pairs ROPE separates and axor does not (the falsifier)\n")
        print("| suite · task | call · arg | needed | substitution | ROPE's reason |")
        print("|---|---|---|---|---|")
        for r in rope_splits[:40]:
            print(f"| {r['suite']} · {r['task']} | `{r['tool']}.{r['arg']}` "
                  f"| `{str(r['needed_value'])[:28]}` | `{str(r['candidate_value'])[:28]}` "
                  f"| {r['rope']['candidate_why'][:90]} |")

    print("\n### Referential pairs both gates admit, by call and argument\n")
    counts: dict[tuple[str, str, str], int] = {}
    ex: dict[tuple[str, str, str], dict] = {}
    for r in rope_also:
        k = (r["suite"], r["tool"], r["arg"])
        counts[k] = counts.get(k, 0) + 1
        ex.setdefault(k, r)
    print("| suite | call · arg | pairs | bucket | example |")
    print("|---|---|---|---|---|")
    for k, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        r = ex[k]
        print(f"| {k[0]} | `{k[1]}.{k[2]}` | {n} | {r['bucket']} | {r['task']}: "
              f"`{str(r['needed_value'])[:24]}` → `{str(r['candidate_value'])[:24]}` |")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="directory of pairs.py JSON records")
    ap.add_argument("--rope-src", default="/home/user/xhowenma/rope/src")
    ap.add_argument("--router", default="opus", help="which cached ROPE router to use")
    ap.add_argument("--config", action="append",
                    help="restrict to pairs from these deployment configs")
    ap.add_argument("--out", help="directory for the verdict records")
    args = ap.parse_args()
    rows = run(args.records, rope_src=args.rope_src, router=args.router,
               configs=set(args.config) if args.config else None, out_dir=args.out)
    report(rows, args.router)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
