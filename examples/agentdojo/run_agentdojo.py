"""Run the AgentDojo prompt-injection benchmark with and without axor governance.

This pits a real Claude model on AgentDojo's banking suite under the
`important_instructions` attack, comparing two pipelines that differ in exactly
one element:

    undefended:  ... -> ToolsExecutionLoop([ ToolsExecutor,          LLM ])
    governed:    ... -> ToolsExecutionLoop([ GovernedToolsExecutor,  LLM ])

`GovernedToolsExecutor` runs each tool call through an axor `ToolCallGovernor`
first; everything else is identical. We report, per condition:

    utility (ASR-independent): did the agent complete the user's real task?
    ASR    (attack success rate): did the injected attacker goal succeed?

A good defense drives ASR down while keeping utility up.

This is a *slice* (a few user x injection pairs), not the full 144-pair suite —
it is sized to be honest and cheap on a real API key. Widen TASK SLICE below to
run more. Requires ANTHROPIC_API_KEY. Uses claude-haiku.

    python -m examples.agentdojo.run_agentdojo
"""
from __future__ import annotations

import os
import sys

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, load_system_message
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.task_suite.load_suites import get_suites

sys.path.insert(0, os.path.dirname(__file__))
from agentdojo_adapter import (  # noqa: E402
    GovernedToolsExecutor,
    OpenRouterLLM,
    RawAnthropicLLM,
)

import nested_attacks  # noqa: E402,F401  (registers nested_instructions / recursive_relay)

from axor_core.config import GovernanceConfig  # noqa: E402
from axor_core.governor import ToolCallGovernor  # noqa: E402

# Backend: "anthropic" (claude via raw urllib) or "openrouter". The default model
# is GPT-4o — the CaMeL-comparable model and the primary experiment; an open model
# like Qwen (the susceptible model used for the supplementary runs) is selected via
# AXOR_BENCH_MODEL=qwen/qwen-2.5-72b-instruct.
# Override from the environment: AXOR_BENCH_BACKEND / AXOR_BENCH_MODEL.
BACKEND = os.environ.get("AXOR_BENCH_BACKEND", "openrouter")
MODEL = os.environ.get(
    "AXOR_BENCH_MODEL",
    "openai/gpt-4o" if BACKEND == "openrouter" else "claude-haiku-4-5-20251001",
)
ATTACK = os.environ.get("AXOR_BENCH_ATTACK", "important_instructions")
SUITE = os.environ.get("AXOR_BENCH_SUITE", "banking")


# ── axor taxonomy per suite — declarative, from YAML ────────────────────────────
#
# The operator's deployment declaration (which of a suite's tools emit untrusted/
# attacker-influenceable data, which are egress sinks, which argument drives each
# sink's taint decision) lives in `config/<suite>.yaml` — the same GovernanceConfig
# format a real deployment loads. Nothing in those files is attack-specific.
_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")


def make_governor() -> ToolCallGovernor:
    # AXOR_BENCH_CONFIG overrides the config file (default: <suite>.yaml) so a run
    # can swap in a tuned deployment taxonomy (e.g. an approved-payee allowlist)
    # while the suite name still selects the AgentDojo tasks.
    cfg_name = os.environ.get("AXOR_BENCH_CONFIG", f"{SUITE}.yaml")
    cfg = GovernanceConfig.from_yaml(os.path.join(_CONFIG_DIR, cfg_name))
    return ToolCallGovernor(**cfg.as_governor_kwargs())


# Per-suite task slices: a curated set of GENUINELY SERIOUS injection tasks
# (financial theft, PII/2FA exfiltration, mass data egress) paired with
# action-oriented user tasks so the attack actually reaches an action.
SUITES = {
    "banking": {
        # Action tasks that move money to a prompt recipient.
        "user_tasks": ["user_task_0", "user_task_3"],
        # Data-exfiltration-via-transaction (these reliably land): inj_1 leaks a
        # THIRD PARTY's bank IBAN to the attacker; inj_0 leaks the user's
        # subscriptions; inj_8 leaks all scheduled transactions (recipients/IBANs/
        # amounts). The overt-theft injections (inj_5 drain, inj_6 $30k structured)
        # are kept available but the model resists them, so they show no headroom.
        "injection_tasks": ["injection_task_1", "injection_task_0", "injection_task_8"],
    },
    "travel": {
        # Tasks that read reviews (injection vector) and can send an email.
        "user_tasks": ["user_task_1", "user_task_3", "user_task_7"],
        # inj_3: exfiltrate name+ID+passport+bank to attacker email (identity theft);
        # inj_5: leak passport + bank account.
        "injection_tasks": ["injection_task_3", "injection_task_5"],
    },
    "slack": {
        # Tasks that read a webpage / channels and post somewhere.
        "user_tasks": ["user_task_3", "user_task_15", "user_task_1"],
        # inj_2/4: concatenate all messages and POST them to an attacker website
        # (mass data exfiltration); inj_1: push an attacker link to a colleague.
        "injection_tasks": ["injection_task_4", "injection_task_2", "injection_task_1"],
    },
    "workspace": {
        "user_tasks": ["user_task_14", "user_task_16", "user_task_17"],
        # inj_5: steal a 2FA security code AND delete the evidence; inj_3: forward
        # a private email to the attacker.
        "injection_tasks": ["injection_task_5", "injection_task_3", "injection_task_4"],
    },
}

_CFG = SUITES[SUITE]
USER_TASKS = _CFG["user_tasks"]
INJECTION_TASKS = _CFG["injection_tasks"]

# AXOR_BENCH_CAMEL=1 reports in CaMeL's coordinate system: the benchmark's TWO
# axes per condition — benign utility (all user tasks, no attack) and ASR under
# attack — so the headline is "solves X% of tasks at ASR Y%", the same axis
# CaMeL's 67% lives on. Implies the full user-task list for the benign axis.
CAMEL_MODE = os.environ.get("AXOR_BENCH_CAMEL") == "1"

# AXOR_BENCH_ALL_USER_TASKS=1 runs every user task in the suite (full benchmark
# number) instead of the curated slice. Injection tasks stay the serious subset.
if os.environ.get("AXOR_BENCH_ALL_USER_TASKS") == "1" or CAMEL_MODE:
    USER_TASKS = list(get_suites("v1")[SUITE].user_tasks.keys())

# AXOR_BENCH_INJECTIONS="injection_task_4,..." overrides the injection slice (e.g.
# pin one headline threat across all user tasks for a feasible full-suite number).
if os.environ.get("AXOR_BENCH_INJECTIONS"):
    INJECTION_TASKS = os.environ["AXOR_BENCH_INJECTIONS"].split(",")


def _make_llm():
    # Reasoning models (o3/o4-mini) spend completion budget on hidden reasoning
    # tokens before any tool call, so 1024 can truncate them; AXOR_BENCH_MAXTOK
    # lets a run raise the ceiling without touching the default.
    max_tokens = int(os.environ.get("AXOR_BENCH_MAXTOK", "1024"))
    if BACKEND == "openrouter":
        return OpenRouterLLM(model=MODEL, max_tokens=max_tokens)
    return RawAnthropicLLM(model=MODEL, max_tokens=max_tokens)


# A model-id token AgentDojo's MODEL_NAMES recognises, so important_instructions
# can address the model by name. GPT-4o maps to "GPT-4" (what CaMeL's runs saw);
# Qwen isn't in the map, so the Llama id gives the generic "AI assistant", which
# is the faithful choice for an open model.
if "gpt-4o" in MODEL:
    _MODEL_NAME_TOKEN = "gpt-4o-2024-05-13"
elif BACKEND == "openrouter":
    _MODEL_NAME_TOKEN = "meta-llama/Llama-3-70b-chat-hf"
else:
    _MODEL_NAME_TOKEN = "claude-3-5-sonnet-20241022"


def build_pipeline(governed: bool):
    llm = _make_llm()
    if governed:
        tools_executor = GovernedToolsExecutor(make_governor)
    else:
        tools_executor = ToolsExecutor()
    loop = ToolsExecutionLoop([tools_executor, llm])
    pipeline = AgentPipeline([
        SystemMessage(load_system_message(None)),
        InitQuery(),
        llm,
        loop,
    ])
    pipeline.name = f"axor-{'governed' if governed else 'undefended'}-{_MODEL_NAME_TOKEN}"
    return pipeline, tools_executor


def run_condition(governed: bool, suite, attack_name: str):
    pipeline, tools_executor = build_pipeline(governed)
    attack = load_attack(attack_name, suite, pipeline)

    utilities: list[bool] = []
    asr: list[bool] = []
    label = "GOVERNED " if governed else "UNDEFENDED"
    for ut_id in USER_TASKS:
        user_task = suite.get_user_task_by_id(ut_id)
        for it_id in INJECTION_TASKS:
            injection_task = suite.get_injection_task_by_id(it_id)
            injections = attack.attack(user_task, injection_task)
            try:
                utility, security = suite.run_task_with_pipeline(
                    pipeline, user_task, injection_task, injections
                )
            except Exception as exc:  # keep the slice going; report and move on
                print(f"  [{label}] {ut_id} x {it_id}: ERROR {type(exc).__name__}: {exc}")
                continue
            utilities.append(utility)
            asr.append(security)
            print(
                f"  [{label}] {ut_id:<12} x {it_id:<16} "
                f"utility={'Y' if utility else 'n'}  attack_succeeded={'Y' if security else 'n'}"
            )
    return utilities, asr, tools_executor


def run_benign(governed: bool, suite):
    """All user tasks, NO attack — the utility axis on its own. The cost of the
    defense is whatever legitimate work the governed condition can no longer do."""
    pipeline, tools_executor = build_pipeline(governed)
    utilities: list[bool] = []
    label = "GOVERNED " if governed else "UNDEFENDED"
    for ut_id in USER_TASKS:
        user_task = suite.get_user_task_by_id(ut_id)
        try:
            utility, _ = suite.run_task_with_pipeline(pipeline, user_task, None, {})
        except Exception as exc:
            print(f"  [{label}] {ut_id}: ERROR {type(exc).__name__}: {exc}")
            continue
        utilities.append(utility)
        print(f"  [{label}] {ut_id:<14} (benign)  utility={'Y' if utility else 'n'}")
    return utilities, tools_executor


def _pct(xs):
    return 100.0 * sum(xs) / len(xs) if xs else float("nan")


def main_camel() -> int:
    """CaMeL-comparable run: per condition, benign utility over the FULL user-task
    list and ASR over the serious-injection slice. The headline number is the
    governed benign utility at the governed ASR — the axis CaMeL's 67% (GPT-4o,
    utility retained under a provably-secure defense) lives on. NOT comparable
    point-for-point (different model, ASR measured on a slice, taint instead of an
    interpreter) — but the same trade-off being measured."""
    suite = get_suites("v1")[SUITE]
    print(f"AgentDojo CaMeL-comparable run · suite={SUITE} · attack={ATTACK} · model={MODEL}")
    print(f"benign axis: {len(USER_TASKS)} user tasks (full suite)")
    print(f"attack axis: {len(USER_TASKS)} user x {len(INJECTION_TASKS)} injection "
          f"= {len(USER_TASKS) * len(INJECTION_TASKS)} pairs per condition\n")

    # AXOR_BENCH_BENIGN_ONLY=1 skips the (expensive) attack axis — useful for
    # averaging the benign utility-cost over many passes cheaply (ASR is already
    # established at 0 on a robust model).
    benign_only = os.environ.get("AXOR_BENCH_BENIGN_ONLY") == "1"
    print("BENIGN / UNDEFENDED ...")
    ub_util, _ = run_benign(False, suite)
    print("\nBENIGN / GOVERNED ...")
    gb_util, gb_exec = run_benign(True, suite)
    if benign_only:
        ua_util = ua_asr = ga_util = ga_asr = []  # _pct([]) -> nan, formats cleanly
        ga_exec = gb_exec
    else:
        print("\nATTACK / UNDEFENDED ...")
        ua_util, ua_asr, _ = run_condition(False, suite, ATTACK)
        print("\nATTACK / GOVERNED ...")
        ga_util, ga_asr, ga_exec = run_condition(True, suite, ATTACK)

    retention = 100.0 * _pct(gb_util) / _pct(ub_util) if _pct(ub_util) else float("nan")
    print("\n" + "=" * 72)
    print(f"RESULTS  (suite={SUITE}, attack={ATTACK}, model={MODEL})")
    print("=" * 72)
    print(f"{'condition':<12}{'benign utility':>16}{'utility under attack':>22}{'ASR':>8}")
    print(f"{'undefended':<12}{_pct(ub_util):>15.1f}%{_pct(ua_util):>21.1f}%{_pct(ua_asr):>7.1f}%")
    print(f"{'governed':<12}{_pct(gb_util):>15.1f}%{_pct(ga_util):>21.1f}%{_pct(ga_asr):>7.1f}%")
    print("-" * 72)
    print(f"benign denials:  {gb_exec.denied_count} ({gb_exec.denials})")
    print(f"attack denials:  {ga_exec.denied_count}")
    print("-" * 72)
    print(f"CaMeL-axis headline: governed solves {_pct(gb_util):.1f}% of benign tasks "
          f"at {_pct(ga_asr):.1f}% ASR")
    print(f"utility retention vs this model undefended: {retention:.1f}%")
    print(f"(CaMeL on GPT-4o, full benchmark: 67% utility at ~0 ASR — same axis, "
          f"different model and attack coverage; see agentdojo_results.md)")
    print("=" * 72)
    return 0


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY") and BACKEND == "anthropic":
        print("ANTHROPIC_API_KEY not set — cannot run.")
        return 0
    if CAMEL_MODE:
        return main_camel()

    suite = get_suites("v1")[SUITE]
    print(f"AgentDojo · suite={SUITE} · attack={ATTACK} · model={MODEL}")
    print(f"slice: {len(USER_TASKS)} user tasks x {len(INJECTION_TASKS)} injection tasks "
          f"= {len(USER_TASKS) * len(INJECTION_TASKS)} pairs per condition\n")

    print("Running UNDEFENDED ...")
    u_util, u_asr, _ = run_condition(False, suite, ATTACK)
    print("\nRunning GOVERNED ...")
    g_util, g_asr, g_exec = run_condition(True, suite, ATTACK)

    def pct(xs):
        return 100.0 * sum(xs) / len(xs) if xs else float("nan")

    print("\n" + "=" * 64)
    print(f"RESULTS  (suite={SUITE}, attack={ATTACK})")
    print("=" * 64)
    print(f"{'condition':<12}{'utility':>12}{'attack success (ASR)':>24}")
    print(f"{'undefended':<12}{pct(u_util):>11.1f}%{pct(u_asr):>23.1f}%")
    print(f"{'governed':<12}{pct(g_util):>11.1f}%{pct(g_asr):>23.1f}%")
    print("-" * 64)
    print(f"governance denied {g_exec.denied_count} tool call(s): {g_exec.denials}")
    print("=" * 64)
    print(
        "\nReading: governance should drive ASR down (fewer injected attacker "
        "goals succeed) while keeping utility close to the undefended baseline."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
