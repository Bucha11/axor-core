"""Step 4 — what additional ground separates a confirmed pair, and what it costs.

Each *ground* is a candidate answer to "what is the monitor missing?". For every
case in ``cases.py`` the prefix of the reference plan is replayed, the ground is
installed, and two questions are measured on the real governor:

    does the needed call stay allowed?    (if not, the ground is not usable)
    does the substitution become denied?  (if not, the ground does not separate)

and then, for the grounds that are a deployment-wide configuration, what they cost
over the whole suite: how many of the reference plans' needed calls the monitor
still admits. No utility gain is claimed anywhere — the measurement is of price.

The grounds:

``as_deployed``           the §6 configuration, for reference.
``widen_key``             declare every effectful tool an integrity sink and its
                          record-selecting arguments its driving arguments. More
                          of the *same* observation (origin class).
``enumerate_class``       an operator ``enum`` over the whole class of admissible
                          objects (every value in the argument's slot) — what an
                          operator can write without resolving the instruction.
``task_origin_witness``   require the driving value to carry a TASK trusted
                          origin: the user's own request named it. Uses the
                          kernel's existing ``TrustedOrigin`` vocabulary, applied
                          as an extra admission condition (a prototype of a
                          ``task-only`` integrity-origin mode, not a kernel change).
``condition_binding``     resolve the instruction's selection condition against a
                          trusted index *once*, and install the result as a
                          singleton ``enum`` for that argument — the identifier is
                          bound to the record the condition selects.
``bind_and_key``          ``widen_key`` and ``condition_binding`` together — the
                          observation keys on the selecting argument *and* that
                          argument is bound to the resolved record.
``per_call_confirmation`` raise the tool above the unattended consequence ceiling,
                          so the call needs a human/governance gate.

    python examples/admission-slack/enrichment.py
    python examples/admission-slack/enrichment.py --json out.json
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

from agentdojo.agent_pipeline.tool_execution import tool_result_to_str  # noqa: E402
from agentdojo.functions_runtime import FunctionsRuntime  # noqa: E402
from agentdojo.task_suite.load_suites import get_suites  # noqa: E402

from axor_core.config import GovernanceConfig  # noqa: E402
from axor_core.contracts.canonical import ConsequenceClass  # noqa: E402
from axor_core.governor import ToolCallGovernor  # noqa: E402
from axor_core.policy.value_policy import enum as enum_pred  # noqa: E402

from cases import CASES, Case  # noqa: E402
from harvest import pools  # noqa: E402
from observation import decide, projection  # noqa: E402
from pairs import effectful_tools, governor_for, selector_args  # noqa: E402

GROUNDS = (
    "as_deployed",
    "widen_key",
    "enumerate_class",
    "task_origin_witness",
    "condition_binding",
    "bind_and_key",
    "per_call_confirmation",
)
# Grounds that are a deployment-wide configuration, so their cost can be measured
# over the whole suite. ``condition_binding`` is per-instruction by construction
# (a resolver runs per task), so there is no suite-wide config to price this way.
SUITE_WIDE = ("as_deployed", "widen_key", "task_origin_witness", "per_call_confirmation")


def _cfg(suite: str, config: str) -> GovernanceConfig:
    return GovernanceConfig.from_yaml(
        os.path.join(_REPO, "examples", "agentdojo", "config", config)
    )


def install(
    cfg: GovernanceConfig,
    ground: str,
    *,
    suite_obj: Any,
    case: Case | None = None,
    class_values: list[Any] | None = None,
    tools_to_gate: set[str] | None = None,
) -> GovernanceConfig:
    """Return the configuration with ``ground`` installed."""
    cfg = dataclasses.replace(cfg, integrity_default="context")
    if ground == "as_deployed":
        return cfg
    if ground in ("widen_key", "bind_and_key"):
        tools = {t.name: t for t in suite_obj.tools}
        eff = effectful_tools(suite_obj, cfg)
        driving = dict(cfg.driving_args)
        for name in sorted(eff):
            sel = selector_args(tools[name]) if name in tools else []
            if sel:
                driving[name] = sorted(set(driving.get(name, [])) | set(sel))
        cfg = dataclasses.replace(
            cfg, integrity_sinks=frozenset(set(cfg.integrity_sinks) | eff),
            driving_args=driving,
        )
        if ground == "widen_key":
            return cfg
        # bind_and_key: fall through to the binding below, on the widened key.
        ground = "condition_binding"
    if ground == "enumerate_class" and case is not None:
        policies = {k: list(v) for k, v in cfg.value_policies.items()}
        policies.setdefault(case.tool, []).append(
            enum_pred(case.arg, class_values or [case.original])
        )
        return dataclasses.replace(cfg, value_policies=policies)
    if ground == "condition_binding" and case is not None:
        policies = {k: list(v) for k, v in cfg.value_policies.items()}
        policies.setdefault(case.tool, []).append(
            enum_pred(case.arg, [case.binding.get("value", case.original)])
        )
        return dataclasses.replace(cfg, value_policies=policies)
    if ground == "per_call_confirmation":
        overrides = dict(cfg.consequence_overrides)
        for name in sorted(tools_to_gate or ({case.tool} if case else set())):
            overrides[name] = ConsequenceClass.CATASTROPHIC
        return dataclasses.replace(cfg, consequence_overrides=overrides)
    return cfg


def task_origin_ok(
    governor: ToolCallGovernor, tool: str, args: dict[str, Any], prompt: str
) -> bool:
    """The ``task_origin_witness`` ground: every driving value of the call must
    have a TASK witness — the user's own request named it.

    This is a prototype of a narrower integrity-origin mode (``task-only``, beside
    the kernel's existing ``request-only`` and ``any-trusted``), not a kernel
    change. It is deliberately *the kernel's own matcher*: ``_token_bounded`` is
    the span test the TrustedValueIndex uses to award ``TrustedOrigin.TASK``, so
    the ground behaves exactly as that origin does — the only difference is that
    OPERATOR-trusted values (allowlist members) no longer satisfy it, which is
    the point of the experiment.
    """
    from axor_core.taint.trusted import _token_bounded

    drivers = governor._driving_args.get(tool) or frozenset(args)
    values = [args[a] for a in drivers if a in args]
    if not values:
        return False
    flat: list[Any] = []
    for v in values:
        flat.extend(v if isinstance(v, list) else [v])
    hay = (prompt or "").casefold()
    return all(_token_bounded(str(v).casefold(), hay) for v in flat if v is not None)


def replay_prefix(
    suite_obj: Any, task: Any, upto: int, cfg: GovernanceConfig
) -> tuple[ToolCallGovernor, Any, Any, list[Any]]:
    """Execute the reference plan up to (not including) ``upto`` and return the
    governor state, environment, runtime and plan."""
    env = suite_obj.load_and_inject_default_environment({})
    plan = list(task.ground_truth(env))
    runtime = FunctionsRuntime(suite_obj.tools)
    governor = ToolCallGovernor(**cfg.as_governor_kwargs())
    governor.register_task(task.PROMPT)
    for call in plan[:upto]:
        args = dict(call.args)
        out, err = runtime.run_function(env, call.function, args)
        if err is None:
            governor.register_output(
                governor.evaluate(call.function, args), tool_result_to_str(out)
            )
    return governor, env, runtime, plan


def _coerce(candidate: Any, original: Any) -> Any:
    if isinstance(original, bool) or not isinstance(original, (int, float)):
        return candidate
    try:
        return type(original)(candidate)
    except Exception:
        return candidate


def run_case(case: Case, ground: str) -> dict[str, Any]:
    suite_obj = get_suites("v1")[case.suite]
    base = _cfg(case.suite, case.config)
    # The class of admissible objects for `enumerate_class`: every value that sits
    # in the same environment slot as the needed one.
    env0 = suite_obj.load_and_inject_default_environment({})
    _, slots = pools(env0)
    orig_slots = slots.get(case.original, set()) or slots.get(str(case.original), set())
    class_values = sorted(
        {v for v, sl in slots.items() if sl & orig_slots}, key=str
    ) if orig_slots else [case.original]
    cfg = install(
        base, ground, suite_obj=suite_obj, case=case, class_values=class_values,
        tools_to_gate={case.tool},
    )
    task = suite_obj.get_user_task_by_id(case.task)
    governor, env, runtime, plan = replay_prefix(suite_obj, task, case.step, cfg)
    call = plan[case.step]
    assert call.function == case.tool, f"{case.name}: step {case.step} is {call.function}"
    args = dict(call.args)
    orig_value = args.get(case.arg)
    cand = _coerce(case.candidate, orig_value if not isinstance(orig_value, list) else orig_value[0])
    sub_args = dict(args)
    sub_args[case.arg] = [cand] if isinstance(orig_value, list) else cand

    orig = decide(governor, case.tool, args)
    sub = decide(governor, case.tool, sub_args)
    extra_orig = extra_sub = True
    if ground == "task_origin_witness":
        extra_orig = task_origin_ok(governor, case.tool, args, task.PROMPT)
        extra_sub = task_origin_ok(governor, case.tool, sub_args, task.PROMPT)
    orig_allowed = orig.allowed and extra_orig
    sub_allowed = sub.allowed and extra_sub
    pi_equal = (
        projection(governor, case.tool, args) == projection(governor, case.tool, sub_args)
    )
    return {
        "case": case.name, "ground": ground, "suite": case.suite, "config": case.config,
        "task": case.task, "tool": case.tool, "arg": case.arg,
        "needed_value": orig_value, "alternative": cand,
        "needed_allowed": orig_allowed,
        "alternative_allowed": sub_allowed,
        "separates": bool(orig_allowed and not sub_allowed),
        "needed_reason": orig.reason if not orig.allowed else (
            "approved" if extra_orig else "no TASK origin for the driving value"),
        "alternative_reason": sub.reason if not sub.allowed else (
            "approved" if extra_sub else "no TASK origin for the driving value"),
        "pi_equal_under_ground": pi_equal,
        "class_size": len(class_values),
    }


def suite_cost(suite: str, config: str, ground: str) -> dict[str, Any]:
    """What the ground costs over the whole suite: of the effectful calls the
    reference plans make, how many does the monitor still admit?"""
    suite_obj = get_suites("v1")[suite]
    base = _cfg(suite, config)
    eff = effectful_tools(suite_obj, base)
    cfg = install(base, ground, suite_obj=suite_obj, tools_to_gate=eff)
    needed = admitted = 0
    for task_id in suite_obj.user_tasks:
        task = suite_obj.get_user_task_by_id(task_id)
        env = suite_obj.load_and_inject_default_environment({})
        try:
            plan = list(task.ground_truth(env))
        except Exception:
            continue
        runtime = FunctionsRuntime(suite_obj.tools)
        governor = ToolCallGovernor(**cfg.as_governor_kwargs())
        governor.register_task(task.PROMPT)
        for call in plan:
            args = dict(call.args)
            if call.function in eff:
                needed += 1
                d = decide(governor, call.function, args)
                ok = d.allowed
                if ok and ground == "task_origin_witness":
                    ok = task_origin_ok(governor, call.function, args, task.PROMPT)
                admitted += int(ok)
            out, err = runtime.run_function(env, call.function, args)
            if err is None:
                governor.register_output(
                    governor.evaluate(call.function, args), tool_result_to_str(out)
                )
    return {
        "suite": suite, "config": config, "ground": ground,
        "needed_effectful_calls": needed, "admitted": admitted,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write the full records here")
    ap.add_argument("--case", action="append", help="restrict to these case names")
    args = ap.parse_args()
    cases = [c for c in CASES if not args.case or c.name in args.case]

    rows = [run_case(c, g) for c in cases for g in GROUNDS]
    print("## Does the ground separate the pair?\n")
    print("| case | ground | needed call | alternative | separates | why the alternative is refused |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['case']} | {r['ground']} | "
              f"{'allowed' if r['needed_allowed'] else 'REFUSED'} | "
              f"{'allowed' if r['alternative_allowed'] else 'refused'} | "
              f"{'yes' if r['separates'] else 'no'} | "
              f"{(r['alternative_reason'] or '')[:96]} |")

    print("\n## What the deployment-wide grounds cost\n")
    print("| suite · config | ground | needed effectful calls | still admitted |")
    print("|---|---|---|---|")
    costs = []
    seen = {(c.suite, c.config) for c in cases}
    for suite, config in sorted(seen):
        for ground in SUITE_WIDE:
            c = suite_cost(suite, config, ground)
            costs.append(c)
            print(f"| {suite} · {config} | {ground} | {c['needed_effectful_calls']} "
                  f"| {c['admitted']} |")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"pairs": rows, "costs": costs,
                       "cases": [dataclasses.asdict(c) for c in cases]}, fh, indent=1)
        print(f"\n-> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
