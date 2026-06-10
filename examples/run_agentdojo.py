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

    python -m examples.run_agentdojo
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
from agentdojo_adapter import GovernedToolsExecutor, RawAnthropicLLM  # noqa: E402

from axor_core.governor import ToolCallGovernor  # noqa: E402

MODEL = "claude-haiku-4-5-20251001"
ATTACK = "important_instructions"
SUITE = "banking"

# ── Slice to run (keep small — real API calls) ──────────────────────────────────
# user_task_0 (pay a bill → reads bill text) and user_task_3 (refund a dinner
# share → reads an incoming transaction) both have working baseline utility and
# traverse an injection vector, so the attack actually reaches the model.
USER_TASKS = ["user_task_0", "user_task_3"]
INJECTION_TASKS = ["injection_task_0", "injection_task_1", "injection_task_5"]


# ── axor taxonomy for the banking suite ─────────────────────────────────────────
#
# This is the operator's deployment declaration: which of the suite's tools emit
# untrusted/attacker-influenceable data, and which are value-egress sinks. axor's
# normalizer knows generic names (read/curl/...); a real tool set is named for its
# domain, so the operator declares the roles. Nothing here is attack-specific — it
# is the same taxonomy you would write to deploy axor over this banking agent.
def make_banking_governor() -> ToolCallGovernor:
    return ToolCallGovernor(
        # Tools whose output can carry injected instructions (the read side).
        untrusted_sources={
            "get_most_recent_transactions",
            "get_scheduled_transactions",
            "read_file",
            "get_user_info",
        },
        # Money/data leaves the boundary here — gate when driven by tainted content.
        egress_sinks={
            "send_money",
            "schedule_transaction",
            "update_scheduled_transaction",
        },
    )


def build_pipeline(governed: bool):
    llm = RawAnthropicLLM(model=MODEL)
    if governed:
        tools_executor = GovernedToolsExecutor(make_banking_governor)
    else:
        tools_executor = ToolsExecutor()
    loop = ToolsExecutionLoop([tools_executor, llm])
    pipeline = AgentPipeline([
        SystemMessage(load_system_message(None)),
        InitQuery(),
        llm,
        loop,
    ])
    # Name must contain a model id AgentDojo's MODEL_NAMES maps to "Claude" — the
    # important_instructions attack personalises its injection with the model name.
    pipeline.name = (
        f"axor-{'governed' if governed else 'undefended'}-claude-3-5-sonnet-20241022"
    )
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


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set — cannot run.")
        return 0

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
