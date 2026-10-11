"""The decisive experiment: pairs on the real governor.

For every benign AgentDojo user task, the suite's own reference plan (the
benchmark's ``ground_truth``) is replayed call by call against the real
environment and a real :class:`~axor_core.governor.ToolCallGovernor` configured
with the deployment taxonomy from ``examples/agentdojo/config/<suite>.yaml`` —
the same configuration the §6 runs load. At each effectful call of the plan:

1. the prefix is already executed, so the governor's state *is* the history ``h``
   the reference plan reaches that call on;
2. the **original** call is judged (``evaluate`` is pure, so judging does not
   advance the ledger);
3. every **substitution** — the same tool, one argument replaced by another
   shape-compatible value that exists in the environment and that the rest of the
   plan does not need — is judged from that same state, and its *projection* is
   compared with the original's;
4. the substitution is **dispatched** on a fork of the environment, so the record
   says whether the tool would really have executed it, not merely whether the
   monitor would have let it through;
5. then the original call executes for real, its output is registered in the
   per-value ledger, and the walk moves on.

Each pair lands in exactly one of three buckets, which are the only three things
a pair can establish:

    orig_allowed_sub_allowed  the deployed monitor admits the alternative too
    orig_allowed_sub_denied   the monitor has a distinguishing feature — named
                              in ``pi_diff``/``category``
    orig_denied               no evidence of forced admission on this prefix

Nothing here involves a model: the plan is the benchmark's own reference plan and
every verdict is the kernel's. What the experiment therefore measures is the
monitor's **admission** behaviour on the reference plans, not what some agent
would have done, and not an upper bound on utility for all ways of solving a task.

    python examples/admission-slack/pairs.py --suite banking
    python examples/admission-slack/pairs.py --all --out runs/

Requires AgentDojo (``pip install agentdojo``); no API key, no network.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from dataclasses import asdict, dataclass, field
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

from harvest import plan_values, pools, prompt_values, shape  # noqa: E402
from observation import decide, pi_diff, pi_key, projection  # noqa: E402

SUITES = ("banking", "slack", "travel", "workspace")
_CONFIG_DIR = os.path.join(_REPO, "examples", "agentdojo", "config")

# Which calls of a plan are worth probing: a substitution is only interesting if
# the call *does* something. Declared egress/integrity sinks always count; beyond
# those, a state-changing tool is recognised by its verb, and the derived list is
# printed with every run so the classification is auditable rather than implicit.
_EFFECT_VERBS = (
    "send_", "schedule_", "update_", "delete_", "create_", "add_", "append_",
    "share_", "post_", "reserve_", "cancel_", "set_", "rename_", "invite_",
    "remove_", "transfer_", "pay_", "write_", "put_",
)

# How many substitutions to try per (call, argument). The pool of same-shape
# environment values can be large (workspace has hundreds of strings); the cap
# keeps a full four-suite run to minutes. Both numbers are reported: `pool` is the
# full number of same-shape alternatives found, `tested` the number judged.
MAX_PER_ARG = int(os.environ.get("AXOR_SLACK_MAX_PER_ARG", "12"))


@dataclass
class PairRecord:
    suite: str
    task: str
    step: int
    tool: str
    arg: str
    original_value: Any
    candidate_value: Any
    value_shape: str
    candidate_path: str
    bucket: str
    orig_verdict: dict[str, Any]
    sub_verdict: dict[str, Any]
    pi_equal: bool
    pi_difference: list[str]
    dispatch_ok: bool
    dispatch_error: str | None
    driving_arg: bool
    arg_role: str
    pool_size: int
    same_slot: bool
    original_slots: list[str]
    cause: str


@dataclass
class CallRecord:
    """The census row: one effectful call of a reference plan, as the monitor saw it."""

    suite: str
    task: str
    step: int
    tool: str
    args: dict[str, Any]
    driving_args: list[str]
    verdict: dict[str, Any]
    driving_sources: list[str]
    arg_origins: dict[str, list[str]]
    floor_active: bool
    superseded: bool
    driving_values: dict[str, Any]
    driving_origins: dict[str, str | None]
    fragile_occurrence: list[str]


@dataclass
class SuiteResult:
    suite: str
    config: str
    integrity: str
    max_per_arg: int = MAX_PER_ARG
    tasks_total: int = 0
    tasks_with_effectful_call: int = 0
    plan_replay_errors: list[str] = field(default_factory=list)
    calls: list[CallRecord] = field(default_factory=list)
    pairs: list[PairRecord] = field(default_factory=list)
    # Theorem-2 obligation check: a pair with equal projections and unequal
    # verdicts would falsify the projection stated in observation.py.
    pi_violations: list[dict[str, Any]] = field(default_factory=list)


def selector_args(tool_obj: Any) -> list[str]:
    """The arguments of a tool that name the record it acts on, by name.

    Used only by ``--declare-effectful``, to build the *widest reasonable*
    declaration a careful operator could write without inventing a new mechanism:
    every effectful tool declared, with the arguments that select its object as
    the arguments the integrity decision keys on.
    """
    try:
        fields = list(tool_obj.parameters.model_fields)
    except Exception:
        return []
    return [
        f for f in fields
        if f.lower() in _SELECTOR_HINTS or f.lower().endswith(("_id", "_ids", "_name", "_names"))
    ]


def governor_for(
    suite: str,
    config: str | None,
    integrity: str,
    *,
    declare_effectful: bool = False,
    suite_obj: Any = None,
    lift_consequence: "list[str] | None" = None,
    origins: str | None = None,
    neutral: bool = False,
) -> tuple[ToolCallGovernor, GovernanceConfig]:
    """Build the deployment under test.

    The knobs exist for the experiment matrix, and each isolates one variable:

    ``lift_consequence``  grant the escalation the consequence gate asks for on
                          these tools (classify them at the unattended ceiling), so
                          a needed call refused by the *action-class* gate — a knob
                          orthogonal to the observation language — stops masking
                          whatever the language does or does not separate.
    ``origins``           the integrity-origin mode (``request-only`` admits TASK and
                          OPERATOR only; ``any-trusted`` is the kernel default).
    ``neutral``           drop the operator's taxonomy entirely and keep only the
                          mode: the monitor then runs on the normalizer's name
                          heuristics, which is the "neutral table" baseline.
    """
    cfg = GovernanceConfig.from_yaml(os.path.join(_CONFIG_DIR, config or f"{suite}.yaml"))
    if neutral:
        cfg = dataclasses.replace(
            cfg, untrusted_sources=frozenset(), sensitive_sources=frozenset(),
            egress_sinks=frozenset(), integrity_sinks=frozenset(),
            positional_sinks=frozenset(), imperative_sinks=frozenset(),
            benign_tools=frozenset(), value_policies={}, driving_args={},
            consequence_overrides={},
        )
    if integrity:
        cfg = dataclasses.replace(cfg, integrity_default=integrity)
    if origins:
        cfg = dataclasses.replace(cfg, integrity_origins=origins)
    if lift_consequence:
        overrides = dict(cfg.consequence_overrides)
        for name in lift_consequence:
            overrides[name] = ConsequenceClass.CONSEQUENTIAL
        cfg = dataclasses.replace(cfg, consequence_overrides=overrides)
    if declare_effectful and suite_obj is not None:
        tools = {t.name: t for t in suite_obj.tools}
        eff = effectful_tools(suite_obj, cfg)
        driving = dict(cfg.driving_args)
        for name in sorted(eff):
            sel = selector_args(tools[name]) if name in tools else []
            if sel:
                # Union with whatever the deployment already declared: the point is
                # the widest reasonable key, not a replacement of the operator's.
                driving[name] = sorted(set(driving.get(name, [])) | set(sel))
        cfg = dataclasses.replace(
            cfg,
            integrity_sinks=frozenset(set(cfg.integrity_sinks) | eff),
            driving_args=driving,
        )
    return ToolCallGovernor(**cfg.as_governor_kwargs()), cfg


def effectful_tools(suite_obj: Any, cfg: GovernanceConfig) -> set[str]:
    declared = set(cfg.egress_sinks) | set(cfg.integrity_sinks) | set(cfg.positional_sinks)
    verbs = {t.name for t in suite_obj.tools if t.name.startswith(_EFFECT_VERBS)}
    return declared | verbs


def _candidates(
    value: Any,
    env_pools: dict[str, list[tuple[str, Any]]],
    forbidden: set[Any],
    extra: list[tuple[str, Any]] | None = None,
    slots: dict[Any, set[str]] | None = None,
    original: Any = None,
) -> list[tuple[str, Any]]:
    """Alternatives for one argument value: same shape, not the original, not
    needed by the rest of the plan.

    Three sources, all of them values the agent could really have emitted:
    the live environment, the user prompt (the trusted origin), and the
    operator's own ``enum`` allowlist for this argument (members the deployment
    declared trusted, so a substitution between them is the sharpest test of the
    supersession rule).
    """
    s = shape(value)
    pool = list(env_pools.get(s, []))
    for path, cand in extra or []:
        if shape(cand) == s:
            pool.append((path, cand))
    seen: set[Any] = set()
    out: list[tuple[str, Any]] = []
    for p, v in pool:
        if v == value or v in forbidden or v in seen:
            continue
        seen.add(v)
        out.append((p, v))
    # Referential alternatives first (another record of the same kind), then prompt
    # tokens and allowlist members, then the rest: the per-argument cap must not
    # spend itself on environment noise.
    orig_slots = (slots or {}).get(original if original is not None else value, set())

    def rank(pv: tuple[str, Any]) -> tuple[int, int, str]:
        same = bool(orig_slots & (slots or {}).get(pv[1], set()))
        return (0 if same else 1, 0 if pv[0].startswith("<") else 1, str(pv[1]))

    out.sort(key=rank)
    return out


# Reporting-only classification of what an argument *does* in a call. It is NOT
# part of the monitor's observation language (observation.py is), and no verdict
# depends on it: it exists so the tables can say whether an admitted substitution
# changed the observed field, the record the call selects, or its payload.
_SELECTOR_HINTS = (
    "id", "ids", "file_path", "path", "filename", "recipient", "recipients",
    "email", "channel", "user", "participants", "iban", "event_id", "address",
    "url", "to", "cc", "bcc", "name", "title", "restaurant", "hotel", "company",
)


def admission_cause(
    tool: str, arg: str, drivers: set[str], cfg: GovernanceConfig
) -> str:
    """*Why* the monitor cannot separate this pair — three different findings that
    must not be added up as one.

    ``undeclared_tool``        the deployment taxonomy declares no role for this
                              tool, so no gate keys on it at all. A coverage gap
                              in the declaration, not a limit of the observation.
    ``unobserved_argument``    the tool is declared, but this argument is not a
                              driving argument and carries no value predicate, so
                              its value is outside what any gate reads.
    ``same_observation_class`` the argument *is* observed, and the two values fall
                              in the same class of it (same origin class, or both
                              members of the operator's allowlist). This is the
                              sharp case: more of the same observation cannot
                              separate the pair.
    """
    declared = (
        set(cfg.egress_sinks) | set(cfg.integrity_sinks) | set(cfg.positional_sinks)
        | set(cfg.sensitive_sources) | set(cfg.untrusted_sources) | set(cfg.value_policies)
    )
    if tool not in declared:
        return "undeclared_tool"
    has_predicate = any(
        getattr(p, "arg", None) == arg for p in (cfg.value_policies.get(tool) or [])
    )
    if arg in drivers or has_predicate:
        return "same_observation_class"
    return "unobserved_argument"


def arg_role(arg: str, drivers: set[str]) -> str:
    if arg in drivers:
        return "observed_driving"
    a = arg.lower()
    if a in _SELECTOR_HINTS or a.endswith("_id") or a.endswith("_ids"):
        return "selector"
    return "payload"


def allowlist_members(governor: ToolCallGovernor, tool: str, arg: str) -> list[tuple[str, Any]]:
    """The operator's declared trusted set for this argument, if any."""
    out: list[tuple[str, Any]] = []
    for pred in (governor._value_policies or {}).get(tool, ()):
        if getattr(pred, "kind", None) == "enum" and getattr(pred, "arg", None) == arg:
            for member in getattr(pred, "allowed", ()) or ():
                out.append(("<config:enum>", member))
    return out


def _substituted(args: dict[str, Any], arg: str, candidate: Any) -> dict[str, Any]:
    """Replace one argument. A list argument (``recipients``) is replaced by the
    single-element list, which is the shape the reference plans use."""
    out = dict(args)
    original = args.get(arg)
    out[arg] = [candidate] if isinstance(original, list) else candidate
    return out


def run_suite(
    suite_name: str,
    *,
    config: str | None = None,
    integrity: str = "context",
    tasks: list[str] | None = None,
    max_per_arg: int = MAX_PER_ARG,
    declare_effectful: bool = False,
    lift_consequence: "list[str] | None" = None,
    origins: str | None = None,
    neutral: bool = False,
    occurrence_min_len: int = 5,
    verbose: bool = True,
) -> SuiteResult:
    suite_obj = get_suites("v1")[suite_name]
    knobs = dict(
        declare_effectful=declare_effectful, suite_obj=suite_obj,
        lift_consequence=lift_consequence, origins=origins, neutral=neutral,
    )
    _, cfg0 = governor_for(suite_name, config, integrity, **knobs)
    effectful = effectful_tools(suite_obj, cfg0)
    label = (config or f"{suite_name}.yaml") + "".join([
        "+declared" if declare_effectful else "",
        "+neutral" if neutral else "",
        f"+{origins}" if origins else "",
        "+lifted" if lift_consequence else "",
    ])
    result = SuiteResult(
        suite=suite_name, config=label, integrity=integrity, max_per_arg=max_per_arg,
    )
    task_ids = tasks or list(suite_obj.user_tasks.keys())
    result.tasks_total = len(task_ids)
    if verbose:
        print(f"\n=== {suite_name} · config={result.config} · integrity={integrity}")
        print(f"    effectful tools: {sorted(effectful)}")

    for task_id in task_ids:
        task = suite_obj.get_user_task_by_id(task_id)
        env = suite_obj.load_and_inject_default_environment({})
        try:
            plan = list(task.ground_truth(env))
        except Exception as exc:  # a few AgentDojo plans need a live env detail
            result.plan_replay_errors.append(f"{task_id}: ground_truth {type(exc).__name__}: {exc}")
            continue
        runtime = FunctionsRuntime(suite_obj.tools)
        governor, _ = governor_for(suite_name, config, integrity, **knobs)
        governor.register_task(task.PROMPT)
        needed = plan_values(plan)
        prompt_candidates = prompt_values(task.PROMPT)
        saw_effect = False

        for step, call in enumerate(plan):
            tool, args = call.function, dict(call.args)
            if tool in effectful:
                saw_effect = True
                orig = decide(governor, tool, args)
                pi_orig = projection(governor, tool, args)
                frag = fragile_occurrences(governor, tool, args, min_len=occurrence_min_len)
                result.calls.append(CallRecord(
                    suite=suite_name, task=task_id, step=step, tool=tool,
                    args={k: _jsonable(v) for k, v in args.items()},
                    driving_args=sorted(governor._driving_args.get(tool, ())),
                    verdict=orig.as_dict(),
                    driving_sources=pi_orig["driving_sources"],
                    arg_origins={
                        k: sorted(_sources_of(governor, v)) for k, v in args.items()
                    },
                    floor_active=pi_orig["floor_active"],
                    superseded=bool(pi_orig["integrity_superseded"]),
                    driving_values={
                        a: _jsonable(args[a])
                        for a in sorted(governor._driving_args.get(tool, ())) if a in args
                    },
                    driving_origins=_driving_origins(governor, tool, args),
                    fragile_occurrence=frag,
                ))
                env_pools, env_slots = pools(env)
                drivers = set(governor._driving_args.get(tool, ()))
                for arg, value in args.items():
                    probe = value[0] if isinstance(value, list) and value else value
                    if not isinstance(probe, (str, int, float)) or isinstance(probe, bool):
                        continue
                    cands = _candidates(
                        probe, env_pools, needed,
                        extra=prompt_candidates + allowlist_members(governor, tool, arg),
                        slots=env_slots, original=probe,
                    )
                    pool_size = len(cands)
                    orig_slots = sorted(env_slots.get(probe, ()))
                    for path, cand in cands[:max_per_arg]:
                        sub_args = _substituted(args, arg, cand)
                        sub = decide(governor, tool, sub_args)
                        pi_sub = projection(governor, tool, sub_args)
                        equal = pi_key(pi_orig) == pi_key(pi_sub)
                        if equal and (orig.allowed, orig.category) != (sub.allowed, sub.category):
                            result.pi_violations.append({
                                "task": task_id, "step": step, "tool": tool, "arg": arg,
                                "original": _jsonable(value), "candidate": _jsonable(cand),
                                "orig": orig.as_dict(), "sub": sub.as_dict(),
                            })
                        if not orig.allowed:
                            bucket = "orig_denied"
                        elif sub.allowed:
                            bucket = "orig_allowed_sub_allowed"
                        else:
                            bucket = "orig_allowed_sub_denied"
                        ok, err = _dispatch(runtime, env, tool, sub_args)
                        result.pairs.append(PairRecord(
                            suite=suite_name, task=task_id, step=step, tool=tool, arg=arg,
                            original_value=_jsonable(value), candidate_value=_jsonable(cand),
                            value_shape=shape(probe), candidate_path=path, bucket=bucket,
                            orig_verdict=orig.as_dict(), sub_verdict=sub.as_dict(),
                            pi_equal=equal, pi_difference=pi_diff(pi_orig, pi_sub),
                            dispatch_ok=ok, dispatch_error=err,
                            driving_arg=arg in drivers,
                            arg_role=arg_role(arg, drivers), pool_size=pool_size,
                            same_slot=bool(
                                set(orig_slots) & env_slots.get(cand, set())
                            ),
                            original_slots=orig_slots,
                            cause=admission_cause(tool, arg, drivers, cfg0),
                        ))
            # The original call executes regardless of the verdict: the experiment
            # asks what the monitor admits *on the reference plan's prefix*, so the
            # prefix must stay the reference plan's own, not a governed variant of
            # it. The verdict on each call is recorded above.
            out, err = runtime.run_function(env, tool, args)
            if err is None:
                try:
                    governor.register_output(
                        governor.evaluate(tool, args), tool_result_to_str(out)
                    )
                except Exception as exc:
                    result.plan_replay_errors.append(
                        f"{task_id}#{step} register_output {type(exc).__name__}: {exc}"
                    )
            else:
                result.plan_replay_errors.append(f"{task_id}#{step} {tool}: {err}")
        if saw_effect:
            result.tasks_with_effectful_call += 1
        if verbose:
            n_pairs = sum(1 for p in result.pairs if p.task == task_id)
            print(f"    {task_id:<15} plan={len(plan):<2} probed_pairs={n_pairs}")
    return result


def _driving_origins(
    governor: ToolCallGovernor, tool: str, args: dict[str, Any]
) -> dict[str, str | None]:
    """The trusted origin of each driving value, as the kernel resolved it."""
    out: dict[str, str | None] = {}
    for a in sorted(governor._driving_args.get(tool, ())):
        if a not in args:
            continue
        v = args[a]
        probe = v[0] if isinstance(v, list) and v else v
        origin = governor._taint.trusted_origin(probe)
        out[a] = origin.value if origin is not None else None
    return out


def fragile_occurrences(
    governor: ToolCallGovernor, tool: str, args: dict[str, Any], *, min_len: int = 5
) -> list[str]:
    """Driving arguments whose TASK origin rests on an occurrence that could be
    accidental.

    Trust under context-default integrity is awarded by a token-bounded *span
    match* of the value against the user's request. For a short or purely numeric
    value ("7", "Bob", "general") that match establishes no causal dependence: the
    token could appear in the request for an unrelated reason. This flags those
    cases so the census can be recomputed with them withheld (E2); it does not
    change any verdict by itself.
    """
    flagged: list[str] = []
    for a in sorted(governor._driving_args.get(tool, ())):
        if a not in args:
            continue
        v = args[a]
        probe = v[0] if isinstance(v, list) and v else v
        origin = governor._taint.trusted_origin(probe)
        if origin is None or origin.value != "task":
            continue
        text = str(probe).strip()
        if len(text) < min_len or text.replace(".", "", 1).isdigit():
            flagged.append(a)
    return flagged


def _sources_of(governor: ToolCallGovernor, value: Any) -> list[str]:
    from axor_core.policy.provenance import source_tokens

    try:
        return list(source_tokens(governor._taint.derive_value(value).sources))
    except Exception:
        return []


def _dispatch(runtime: Any, env: Any, tool: str, args: dict[str, Any]) -> tuple[bool, str | None]:
    """Would the tool really execute this call? Run it on a fork of the environment
    so the reference plan's own state is untouched."""
    try:
        fork = env.model_copy(deep=True)
    except Exception as exc:
        return False, f"fork failed: {exc}"
    try:
        _, err = runtime.run_function(fork, tool, args)
    except Exception as exc:  # pragma: no cover - a tool raising past the runtime
        return False, f"{type(exc).__name__}: {exc}"
    return err is None, err


def _jsonable(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return str(v)


def _refusals(res: SuiteResult) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in res.calls:
        if not c.verdict["allowed"]:
            out[c.verdict["category"]] = out.get(c.verdict["category"], 0) + 1
    return out


def summarise(res: SuiteResult) -> dict[str, Any]:
    buckets: dict[str, int] = {}
    for p in res.pairs:
        buckets[p.bucket] = buckets.get(p.bucket, 0) + 1
    admitted = [p for p in res.pairs if p.bucket == "orig_allowed_sub_allowed"]
    executable = [p for p in admitted if p.dispatch_ok]
    # Theorem 2 only *forces* admission where the projection is equal. An
    # alternative admitted under a different projection is extra permissiveness
    # of this deployment, not a consequence of the observation language, and is
    # counted apart so the two are never conflated.
    forced = [p for p in executable if p.pi_equal]
    roles: dict[str, int] = {}
    for p in forced:
        roles[p.arg_role] = roles.get(p.arg_role, 0) + 1
    # Referential: the alternative names another record of the same kind in the
    # environment (same structural slot), so the admitted call acts on a different
    # object rather than carrying different payload text. These are the pairs the
    # authorization analysis is run on.
    referential = [p for p in forced if p.same_slot]
    causes: dict[str, int] = {}
    for p in referential:
        causes[p.cause] = causes.get(p.cause, 0) + 1
    return {
        "suite": res.suite,
        "config": res.config,
        "integrity": res.integrity,
        "tasks_total": res.tasks_total,
        "tasks_with_effectful_call": res.tasks_with_effectful_call,
        "effectful_calls": len(res.calls),
        "calls_allowed": sum(1 for c in res.calls if c.verdict["allowed"]),
        "pairs": len(res.pairs),
        "buckets": buckets,
        # Counted separately on purpose — alternative values, full calls and tasks
        # are three different quantities and may not be added together.
        "admitted_alternative_values": len(admitted),
        "admitted_executable_substitutions": len(executable),
        "admitted_calls": len({(p.task, p.step) for p in executable}),
        "admitted_tasks": len({p.task for p in executable}),
        "forced_executable_substitutions": len(forced),
        "forced_calls": len({(p.task, p.step) for p in forced}),
        "forced_tasks": len({p.task for p in forced}),
        "forced_by_arg_role": roles,
        "referential_forced_substitutions": len(referential),
        "referential_calls": len({(p.task, p.step) for p in referential}),
        "referential_tasks": len({p.task for p in referential}),
        "referential_by_cause": causes,
        "max_per_arg": res.max_per_arg,
        # A4: how often the enum supersession actually carries the integrity axis.
        "calls_superseded": sum(1 for c in res.calls if c.superseded),
        "calls_allowed_superseded": sum(
            1 for c in res.calls if c.superseded and c.verdict["allowed"]),
        # E2: verdicts resting on an occurrence that could be accidental.
        "allowed_calls_on_fragile_occurrence": sum(
            1 for c in res.calls if c.verdict["allowed"] and c.fragile_occurrence),
        "refusal_categories": _refusals(res),
        "denied_substitutions_with_allowed_original": sum(
            1 for p in res.pairs if p.bucket == "orig_allowed_sub_denied"),
        "pi_violations": len(res.pi_violations),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", choices=SUITES)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--config", help="config file name under examples/agentdojo/config")
    ap.add_argument("--integrity", default="context", choices=("context", "clean"))
    ap.add_argument("--task", action="append", help="restrict to these user task ids")
    ap.add_argument("--max-per-arg", type=int, default=MAX_PER_ARG)
    ap.add_argument("--lift-consequence", help="comma-separated tools to classify at "
                    "the unattended ceiling (grant the escalation the consequence "
                    "gate asks for)")
    ap.add_argument("--origins", choices=("request-only", "any-trusted"),
                    help="integrity-origin mode")
    ap.add_argument("--neutral", action="store_true",
                    help="drop the operator taxonomy entirely (name heuristics only)")
    ap.add_argument("--occurrence-min-len", type=int, default=5,
                    help="below this length a TASK-origin span match is flagged as "
                         "possibly accidental (E2 audit; changes no verdict)")
    ap.add_argument(
        "--declare-effectful", action="store_true",
        help="widen the taxonomy: declare every effectful tool an integrity sink and "
             "its record-selecting arguments its driving arguments",
    )
    ap.add_argument("--out", help="directory for the per-suite JSON records")
    args = ap.parse_args(argv)
    if not args.suite and not args.all:
        ap.error("pass --suite <name> or --all")
    names = list(SUITES) if args.all else [args.suite]

    summaries = []
    for name in names:
        res = run_suite(
            name, config=args.config, integrity=args.integrity, tasks=args.task,
            max_per_arg=args.max_per_arg, declare_effectful=args.declare_effectful,
            lift_consequence=[t for t in (args.lift_consequence or "").split(",") if t],
            origins=args.origins, neutral=args.neutral,
            occurrence_min_len=args.occurrence_min_len,
        )
        s = summarise(res)
        summaries.append(s)
        print(json.dumps(s, indent=2))
        if res.pi_violations:
            print(f"!! projection violated on {len(res.pi_violations)} pairs")
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            path = os.path.join(args.out, f"{name}-{args.integrity}-{res.config}.json")
            with open(path, "w") as fh:
                json.dump(
                    {
                        "summary": s,
                        "calls": [asdict(c) for c in res.calls],
                        "pairs": [asdict(p) for p in res.pairs],
                        "pi_violations": res.pi_violations,
                        "plan_replay_errors": res.plan_replay_errors,
                    },
                    fh, indent=1,
                )
            print(f"   -> {path}")
    if len(summaries) > 1:
        print("\n== totals")
        print(json.dumps({
            "tasks_total": sum(s["tasks_total"] for s in summaries),
            "effectful_calls": sum(s["effectful_calls"] for s in summaries),
            "pairs": sum(s["pairs"] for s in summaries),
            "admitted_alternative_values": sum(s["admitted_alternative_values"] for s in summaries),
            "admitted_executable_substitutions": sum(
                s["admitted_executable_substitutions"] for s in summaries),
            "admitted_calls": sum(s["admitted_calls"] for s in summaries),
            "admitted_tasks": sum(s["admitted_tasks"] for s in summaries),
            "forced_executable_substitutions": sum(
                s["forced_executable_substitutions"] for s in summaries),
            "forced_calls": sum(s["forced_calls"] for s in summaries),
            "forced_tasks": sum(s["forced_tasks"] for s in summaries),
            "referential_forced_substitutions": sum(
                s["referential_forced_substitutions"] for s in summaries),
            "referential_calls": sum(s["referential_calls"] for s in summaries),
            "referential_tasks": sum(s["referential_tasks"] for s in summaries),
            "pi_violations": sum(s["pi_violations"] for s in summaries),
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
