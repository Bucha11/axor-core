"""Rule or feature? The parity run on the 18 pairs ROPE separates.

ROPE separates 18 of the 64 referential pairs (`neighbor_rope.py`). Two readings
compete, and they are different axes:

* a **feature** reading — its observation language reads something of the call that
  axor's does not, so no axor deployment could separate those pairs;
* a **rule** reading — the feature is one axor already has ("the value occurs in the
  trusted request"), and what differed is the rule compiled over it: ROPE's router
  reads the request per task and pins the record-selecting argument, while axor's
  deployed suite taxonomy never declared that argument request-bound.

This settles it by installing, in axor, the same per-task rule their scope carries,
and re-judging the identical pairs. Three deployments per pair:

``as_deployed``     the §6 suite taxonomy — the baseline that admits both calls.
``declare_origin``  the argument is declared the driving argument of an integrity
                    sink and **no** allowlist covers it, so axor's own origin
                    feature decides: a value that occurs in the user's request is
                    TASK-trusted, one that does not carries the context root. This
                    is the *feature* test — it adds no per-task constant.
``const_pin``       the argument additionally carries a singleton ``enum`` holding
                    the value the request names — ROPE's ``CONST`` pin transcribed.
                    This is the *rule* test.

Prediction (the rule reading): ``declare_origin`` separates the pairs whose needed
value occurs in the request, ``const_pin`` separates all 18, and the baseline
separates none. Falsifier: a pair that stays unseparated under ``const_pin`` — then
their gate reads something this language does not, and the reverted "finer
language" claim was right after all.

    python examples/admission-slack/rope_parity.py \
        --rope-record examples/admission-slack/records/experiments/rope-opus.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from agentdojo.task_suite.load_suites import get_suites  # noqa: E402

from axor_core.policy.value_policy import ValuePredicate, enum as enum_pred  # noqa: E402
from axor_core.taint.trusted import _token_bounded  # noqa: E402  (the kernel's matcher)

from enrichment import _cfg, replay_prefix  # noqa: E402
from observation import decide, pi_diff, projection  # noqa: E402

GROUNDS = ("as_deployed", "declare_origin", "const_pin")


def separating_pairs(record_path: str) -> list[dict[str, Any]]:
    """The referential pairs ROPE separates and the recorded axor deployment admits."""
    with open(record_path) as fh:
        rows = json.load(fh)["rows"]
    return [
        r for r in rows
        if r["axor"]["needed"] and r["axor"]["candidate"] and r["axor"]["pi_equal"]
        and r["axor"]["dispatch_ok"] and r["axor"]["same_slot"]
        and r["rope"]["needed"] and not r["rope"]["candidate"]
    ]


def install(cfg: Any, ground: str, *, tool: str, arg: str, needed_value: Any) -> Any:
    cfg = dataclasses.replace(cfg, integrity_default="context")
    if ground == "as_deployed":
        return cfg
    driving = {k: list(v) for k, v in cfg.driving_args.items()}
    driving[tool] = sorted(set(driving.get(tool, [])) | {arg})
    policies: dict[str, list[ValuePredicate]] = {
        k: list(v) for k, v in cfg.value_policies.items()
    }
    if ground == "declare_origin":
        # Strip any allowlist covering this argument: an enum over the whole trusted
        # class would supersede the integrity axis, and the point here is to let
        # axor's origin feature — and nothing else — decide.
        policies[tool] = [p for p in policies.get(tool, []) if p.arg != arg]
    elif ground == "const_pin":
        policies[tool] = [p for p in policies.get(tool, []) if p.arg != arg]
        policies[tool].append(enum_pred(arg, [needed_value]))
    return dataclasses.replace(
        cfg,
        integrity_sinks=frozenset(set(cfg.integrity_sinks) | {tool}),
        driving_args=driving,
        value_policies={k: v for k, v in policies.items() if v},
    )


def run(record_path: str) -> list[dict[str, Any]]:
    pairs = separating_pairs(record_path)
    out: list[dict[str, Any]] = []
    for ground in GROUNDS:
        # group by the call the pair sits on, so the prefix is replayed once per group
        groups: dict[tuple[str, str, str, int, str, str], list[Any]] = {}
        for r in pairs:
            key = (r["config"], r["suite"], r["task"], r["step"], r["tool"], r["arg"])
            groups.setdefault(key, []).append(r["candidate_value"])
        for (config, suite, task_id, step, tool, arg), candidates in sorted(groups.items()):
            suite_obj = get_suites("v1")[suite]
            task = suite_obj.get_user_task_by_id(task_id)
            plan_call_args: dict[str, Any] = {}
            cfg = _cfg(suite, config)
            # the needed value, read off the reference plan itself
            env = suite_obj.load_and_inject_default_environment({})
            needed_raw = dict(list(task.ground_truth(env))[step].args).get(arg)
            needed = needed_raw[0] if isinstance(needed_raw, list) and needed_raw else needed_raw
            cfg = install(cfg, ground, tool=tool, arg=arg, needed_value=needed)
            governor, _env, _rt, plan = replay_prefix(suite_obj, task, step, cfg)
            call = plan[step]
            plan_call_args = dict(call.args)
            n = decide(governor, tool, plan_call_args)
            pi_n = projection(governor, tool, plan_call_args)
            in_request = _token_bounded(
                str(needed).casefold(), (task.PROMPT or "").casefold()
            )
            for cand in candidates:
                sub = dict(plan_call_args)
                sub[arg] = [cand] if isinstance(needed_raw, list) else cand
                s = decide(governor, tool, sub)
                out.append({
                    "ground": ground, "config": config, "suite": suite, "task": task_id,
                    "tool": tool, "arg": arg, "needed": needed, "candidate": cand,
                    "needed_in_request": in_request,
                    "candidate_in_request": _token_bounded(
                        str(cand).casefold(), (task.PROMPT or "").casefold()),
                    "needed_allowed": n.allowed, "candidate_allowed": s.allowed,
                    "separates": bool(n.allowed and not s.allowed),
                    "candidate_category": s.category,
                    "pi_difference": pi_diff(pi_n, projection(governor, tool, sub)),
                })
    return out


def report(rows: list[dict[str, Any]]) -> None:
    print("## The parity run — can axor separate the 18 pairs ROPE separates?\n")
    print("| ground | pairs | separated | needed call refused | still admitted |")
    print("|---|---|---|---|---|")
    for ground in GROUNDS:
        g = [r for r in rows if r["ground"] == ground]
        print(f"| `{ground}` | {len(g)} | **{sum(r['separates'] for r in g)}** "
              f"| {sum(not r['needed_allowed'] for r in g)} "
              f"| {sum(r['needed_allowed'] and r['candidate_allowed'] for r in g)} |")

    print("\n### By call, per ground\n")
    print("| suite · task | call · arg | needed (in request?) | candidates "
          + " | ".join(f"`{g}`" for g in GROUNDS) + " |")
    print("|---|---|---|---|" + "---|" * len(GROUNDS))
    keys = sorted({(r["suite"], r["task"], r["tool"], r["arg"]) for r in rows})
    for suite, task, tool, arg in keys:
        cells = []
        n_cand = 0
        in_req = None
        for ground in GROUNDS:
            g = [r for r in rows if r["ground"] == ground and r["suite"] == suite
                 and r["task"] == task and r["tool"] == tool and r["arg"] == arg]
            n_cand = len(g)
            in_req = g[0]["needed_in_request"] if g else None
            sep = sum(r["separates"] for r in g)
            refused = sum(not r["needed_allowed"] for r in g)
            cells.append(f"{sep}/{len(g)}" + (" (needed refused)" if refused else ""))
        print(f"| {suite} · {task} | `{tool}.{arg}` | {'yes' if in_req else 'no'} "
              f"| {n_cand} | " + " | ".join(cells) + " |")

    print("\n### Why each candidate is refused (first of each kind)\n")
    seen: set[tuple] = set()
    print("| ground | call · arg | candidate | category | observation components that differ |")
    print("|---|---|---|---|---|")
    for r in rows:
        if not r["separates"]:
            continue
        k = (r["ground"], r["tool"], r["arg"], r["candidate_category"])
        if k in seen:
            continue
        seen.add(k)
        print(f"| `{r['ground']}` | `{r['tool']}.{r['arg']}` | `{str(r['candidate'])[:26]}` "
              f"| {r['candidate_category']} | {', '.join(r['pi_difference']) or '—'} |")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rope-record", required=True)
    ap.add_argument("--json")
    args = ap.parse_args()
    rows = run(args.rope_record)
    report(rows)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)
        print(f"\n-> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
